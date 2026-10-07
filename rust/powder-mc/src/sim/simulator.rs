//! The discrete-event simulation engine.
//!
//! Port of `powder/simulation/simulator.py`.  Events are processed in time
//! order; each one may update cluster state, provoke a strategy reaction, and
//! schedule further events.  Between events the simulator advances the commit
//! index over the elapsed wall-clock interval and keeps syncing nodes moving.
//!
//! The ordering inside [`Simulator::process_event`] is semantic, not
//! cosmetic, and matches the Python source step for step.

// Several hot loops walk a scratch buffer by index rather than by
// iterator.  That is not an oversight: the buffer is moved out of `self`
// for the duration of the walk, and the loop body needs `&mut self`, so
// borrowing the buffer to iterate it would conflict.
#![allow(clippy::needless_range_loop)]

use super::cluster::{is_effectively_available, ClusterState};
use super::distributions::{make_rng, Rng, Seconds};
use super::events::{Event, EventMeta, EventQueue, EventType};
use super::ids::Sym;
use super::metrics::{MetricsCollector, MetricsSnapshot};
use super::network::NetworkConfig;
use super::node::{NodeState, SyncPhase, SyncState};
use super::protocol::Protocol;
use super::strategy::{Action, ClusterStrategy};
use super::util::snapshot_boundary;

/// Why a simulation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// The time limit was reached.
    TimeLimit,
    /// Data was definitely lost.
    DataLoss,
    /// A caller-supplied stop condition fired.
    ConditionMet,
    /// The event queue drained.
    NoEvents,
    /// The run never started or ended for no recorded reason.
    Unknown,
}

impl EndReason {
    /// The string Python reports, for JSON and comparisons.
    pub fn as_str(self) -> &'static str {
        match self {
            EndReason::TimeLimit => "time_limit",
            EndReason::DataLoss => "data_loss",
            EndReason::ConditionMet => "condition_met",
            EndReason::NoEvents => "no_events",
            EndReason::Unknown => "unknown",
        }
    }
}

/// Outcome of a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Final simulation time, in seconds.
    pub end_time: Seconds,
    /// Why the run stopped.
    pub end_reason: EndReason,
    /// Metrics at the point it stopped.
    pub metrics: MetricsSnapshot,
    /// Every processed event, when event logging is enabled.
    pub event_log: Vec<Event>,
}

/// Discrete-event simulation engine for RSM clusters.
pub struct Simulator {
    /// Cluster state, evolving over the run.
    pub cluster: ClusterState,
    /// Strategy reacting to events.
    pub strategy: Box<dyn ClusterStrategy>,
    /// Protocol supplying availability semantics.
    pub protocol: Box<dyn Protocol>,
    /// Optional region outage configuration.
    pub network_config: Option<NetworkConfig>,
    /// Whether to retain every processed event.
    pub log_events: bool,
    /// Random source for all sampling.
    pub rng: Rng,
    /// Pending events, ordered by time.
    pub event_queue: EventQueue,
    /// Accumulated metrics.
    pub metrics: MetricsCollector,
    /// Processed events, when logging is enabled.
    pub event_log: Vec<Event>,

    initialized: bool,
    /// Reused buffers, so a steady-state event costs no allocation.
    ///
    /// Each is taken out of `self` for the duration of its walk and put
    /// back afterwards, which is what lets the loop body hold `&mut self`.
    /// They are separate fields rather than one pool because some walks
    /// nest: an outage walks its region's nodes and, for each, walks that
    /// node's dependents.
    scratch_sync: Vec<(usize, f64)>,
    scratch_resched: Vec<usize>,
    scratch_retry: Vec<usize>,
    scratch_donor: Vec<usize>,
    scratch_region: Vec<usize>,
    scratch_region_syms: Vec<Sym>,
    scratch_actions: Vec<Action>,
    scratch_events: Vec<Event>,
}

impl Simulator {
    /// Build a simulator.
    pub fn new(
        initial_cluster: ClusterState,
        strategy: Box<dyn ClusterStrategy>,
        protocol: Box<dyn Protocol>,
        network_config: Option<NetworkConfig>,
        seed: Option<u64>,
        log_events: bool,
    ) -> Self {
        Simulator {
            cluster: initial_cluster,
            strategy,
            protocol,
            network_config,
            log_events,
            rng: make_rng(seed),
            event_queue: EventQueue::new(),
            metrics: MetricsCollector::new(),
            event_log: Vec::new(),
            initialized: false,
            scratch_sync: Vec::new(),
            scratch_resched: Vec::new(),
            scratch_retry: Vec::new(),
            scratch_donor: Vec::new(),
            scratch_region: Vec::new(),
            scratch_region_syms: Vec::new(),
            scratch_actions: Vec::new(),
            scratch_events: Vec::new(),
        }
    }

    /// Re-arm this simulator for another run, keeping every buffer.
    ///
    /// A Monte Carlo experiment runs the same scenario thousands of times.
    /// Building a fresh `Simulator` each time re-allocates the event heap,
    /// the cancellation tables, the node vector and six scratch buffers;
    /// reusing them takes the steady-state cost of a run to roughly zero
    /// allocations, which is what keeps the allocator off the critical
    /// path when many worker threads run experiments at once.
    ///
    /// `template` must describe the same initial state every run -- which
    /// it does, since every run of an experiment starts from the same
    /// cluster.
    pub fn reset(
        &mut self,
        template: &ClusterState,
        strategy: Box<dyn ClusterStrategy>,
        protocol: Box<dyn Protocol>,
        seed: Option<u64>,
    ) {
        self.cluster.reset_from(template);
        self.strategy = strategy;
        self.protocol = protocol;
        self.rng = make_rng(seed);
        self.event_queue.clear();
        self.metrics.reset();
        self.event_log.clear();
        self.initialized = false;
        // The scratch buffers keep their capacity on purpose.
    }

    // -- setup -----------------------------------------------------------

    /// Seed the queue with the first failure, data-loss and outage events,
    /// run the protocol and strategy start hooks, and kick off any syncs the
    /// initial state already requires.
    ///
    /// Idempotent, and public so tests can set the queue up and then adjust
    /// it before the first `run_*` call -- the same thing Python's tests do
    /// by calling `_initialize()` directly.
    pub fn initialize(&mut self) {
        if self.initialized {
            return;
        }

        for index in self.cluster.active_indices() {
            self.schedule_node_events(index);
        }

        if let Some(config) = &self.network_config {
            for region in config.regions.clone() {
                self.schedule_network_outage(region);
            }
        }

        let mut events = std::mem::take(&mut self.scratch_events);
        events.clear();
        {
            let Simulator {
                protocol,
                cluster,
                rng,
                ..
            } = self;
            protocol.on_simulation_start(cluster, rng, &mut events);
        }
        for event in events.drain(..) {
            self.event_queue.push(event);
        }
        self.scratch_events = events;

        let mut actions = std::mem::take(&mut self.scratch_actions);
        actions.clear();
        {
            let Simulator {
                strategy,
                cluster,
                rng,
                ..
            } = self;
            strategy.on_simulation_start(cluster, rng, &mut actions);
        }
        for i in 0..actions.len() {
            let action = actions[i].clone();
            self.execute_action(action);
        }
        actions.clear();
        self.scratch_actions = actions;

        self.retry_pending_syncs(None);
        self.initialized = true;
    }

    /// Schedule the first transient failure and data loss for a node.
    fn schedule_node_events(&mut self, node_index: usize) {
        let node = self.cluster.node_at(node_index);
        let node_id = node.node_id;
        let config = node.config.clone();
        let current_time = self.cluster.current_time;

        let failure_time = config.failure_dist.sample(&mut self.rng);
        self.event_queue.push(Event::new(
            current_time + failure_time,
            EventType::NodeFailure,
            node_id,
        ));

        let data_loss_time = config.data_loss_dist.sample(&mut self.rng);
        self.event_queue.push(Event::new(
            current_time + data_loss_time,
            EventType::NodeDataLoss,
            node_id,
        ));
    }

    /// Schedule the next outage for a region.
    ///
    /// Each region keeps its own independent outage chain, so several
    /// regions can be down at once.
    fn schedule_network_outage(&mut self, region: Sym) {
        let Some(config) = &self.network_config else {
            return;
        };
        if !config.regions.contains(&region) {
            return;
        }

        let outage_delay = config.outage_dist.sample(&mut self.rng);
        let current_time = self.cluster.current_time;
        self.event_queue.push(Event::with_meta(
            current_time + outage_delay,
            EventType::NetworkOutageStart,
            region,
            EventMeta::Region(region),
        ));
    }

    // -- time advance ----------------------------------------------------

    /// Advance the commit index and node positions over `elapsed` seconds.
    ///
    /// While the system can commit, the frontier moves at the protocol's
    /// commit rate and every available, non-syncing node that was already at
    /// the frontier moves with it.  Syncing nodes advance either way, because
    /// they are replaying data the donor already holds.
    fn advance_commit_index(&mut self, elapsed: f64, can_commit: Option<bool>) {
        if elapsed <= 0.0 {
            return;
        }

        let can_commit =
            can_commit.unwrap_or_else(|| self.protocol.can_commit(&self.cluster));

        if can_commit {
            let commit_amount = elapsed * self.protocol.commit_rate();
            let old_index = self.cluster.commit_index;
            let new_index = old_index + commit_amount;
            self.cluster.commit_index = new_index;

            let snapshot_interval = self.protocol.snapshot_interval();
            let (nodes, network) = self.cluster.nodes_and_network_mut();
            for node in nodes.iter_mut() {
                if node.group == super::node::Group::Provisioning {
                    continue;
                }
                if !is_effectively_available(network, node)
                    || node.sync.is_some()
                    || node.last_applied_index < old_index
                {
                    continue;
                }
                node.last_applied_index = new_index;
                if snapshot_interval > 0.0 {
                    let new_snap = snapshot_boundary(new_index, snapshot_interval);
                    if new_snap > node.last_snapshot_index {
                        node.last_snapshot_index = new_snap;
                    }
                }
            }
        }

        self.advance_syncing_nodes(elapsed);
    }

    /// Move every active sync forward by `elapsed` seconds.
    ///
    /// A syncing node replays at its sampled rate and can never pass its
    /// donor's `last_applied_index`.  When the cluster cannot commit, the
    /// donor is frozen and the gap closes at the full replay rate.
    fn advance_syncing_nodes(&mut self, elapsed: f64) {
        let snapshot_interval = self.protocol.snapshot_interval();

        // Phase one: resolve each syncing node's donor position while
        // everything is still immutably borrowed.
        let mut scratch = std::mem::take(&mut self.scratch_sync);
        scratch.clear();
        for (index, node) in self.cluster.all_entries().iter().enumerate() {
            let Some(sync) = &node.sync else { continue };
            if node.group == super::node::Group::Provisioning
                || !self.cluster.node_effectively_available(node)
            {
                continue;
            }
            let Some(donor_index) = self.cluster.index_of(sync.donor_id) else {
                continue;
            };
            let donor = self.cluster.node_at(donor_index);
            if !self.cluster.node_effectively_available(donor) {
                // Sync paused: the donor is unreachable.
                continue;
            }
            scratch.push((index, donor.last_applied_index));
        }

        // Phase two: apply the progress.
        for &(index, donor_applied) in &scratch {
            let node = self.cluster.node_at_mut(index);
            let Some(sync) = node.sync.as_mut() else {
                continue;
            };
            let mut remaining_time = elapsed;

            if sync.phase == SyncPhase::SnapshotDownload {
                if remaining_time >= sync.snapshot_remaining {
                    remaining_time -= sync.snapshot_remaining;
                    sync.snapshot_remaining = 0.0;
                    let target = sync.target_snapshot_index;
                    sync.phase = SyncPhase::LogReplay;
                    node.last_applied_index = target;
                    if snapshot_interval > 0.0 && target > node.last_snapshot_index {
                        node.last_snapshot_index = target;
                    }
                } else {
                    sync.snapshot_remaining -= remaining_time;
                    remaining_time = 0.0;
                }
            }

            let node = self.cluster.node_at_mut(index);
            let Some(sync) = node.sync.as_ref() else {
                continue;
            };
            if sync.phase == SyncPhase::LogReplay && remaining_time > 0.0 {
                let replayed = node.last_applied_index + sync.log_replay_rate * remaining_time;
                node.last_applied_index = replayed.min(donor_applied);
                if snapshot_interval > 0.0 {
                    let new_snap = snapshot_boundary(node.last_applied_index, snapshot_interval);
                    if new_snap > node.last_snapshot_index {
                        node.last_snapshot_index = new_snap;
                    }
                }
            }
        }

        self.scratch_sync = scratch;
    }

    // -- sync lifecycle --------------------------------------------------

    /// Start a sync for a lagging node.
    ///
    /// Picks a donor, decides between log-only replay and snapshot-plus-log
    /// using distribution means, samples the actual rates, installs the
    /// [`SyncState`], and schedules the completion event.
    fn start_sync(&mut self, node_index: usize, can_commit: Option<bool>) {
        let node = self.cluster.node_at(node_index);
        if node.sync.is_some() {
            return; // Already syncing.
        }

        let node_id = node.node_id;
        let node_applied = node.last_applied_index;
        let config = node.config.clone();

        let Some(donor_index) = self.cluster.find_sync_donor(node_id) else {
            // No donor; retry_pending_syncs picks this up once one appears.
            return;
        };
        let donor = self.cluster.node_at(donor_index);
        let donor_id = donor.node_id;
        let donor_applied = donor.last_applied_index;

        let donor_lag = donor_applied - node_applied;
        if donor_lag <= 0.0 {
            return; // Already caught up to the donor.
        }

        let log_replay_rate = config.log_replay_rate_dist.sample(&mut self.rng);

        let can_commit =
            can_commit.unwrap_or_else(|| self.protocol.can_commit(&self.cluster));
        let commit_rate_eff = if can_commit {
            self.protocol.commit_rate()
        } else {
            0.0
        };

        let snapshot_interval = self.protocol.snapshot_interval();
        let log_retention = self.protocol.log_retention_ops();

        let donor_earliest_log = if log_retention > 0.0 {
            (donor_applied - log_retention).max(0.0)
        } else {
            0.0
        };
        let must_snapshot = log_retention > 0.0 && node_applied < donor_earliest_log;

        let mut use_snapshot = false;
        let mut target_snap = 0.0;
        let mut snapshot_download_time = 0.0;

        if must_snapshot && snapshot_interval > 0.0 {
            // The donor has discarded the entries we need.
            use_snapshot = true;
            target_snap = snapshot_boundary(donor_applied, snapshot_interval);
            snapshot_download_time = config.snapshot_download_time_dist.sample(&mut self.rng);
        } else if snapshot_interval > 0.0 && !must_snapshot {
            // Both paths are open; estimate each from means and take the
            // faster one.  The decision has to be made before the sync
            // starts, so sampling would be the wrong tool here.
            let mean_replay_rate = config.log_replay_rate_dist.mean();
            let mut mean_net_rate = mean_replay_rate - commit_rate_eff;
            if mean_net_rate <= 0.0 {
                mean_net_rate = mean_replay_rate;
            }
            let log_only_est = donor_lag / mean_net_rate;

            let target_snap_candidate = snapshot_boundary(donor_applied, snapshot_interval);
            if target_snap_candidate > node_applied {
                let mean_snap_time = config.snapshot_download_time_dist.mean();
                let remaining_after_snap =
                    donor_applied - target_snap_candidate + commit_rate_eff * mean_snap_time;
                let snap_est = mean_snap_time + remaining_after_snap / mean_net_rate;

                if snap_est < log_only_est {
                    use_snapshot = true;
                    target_snap = target_snap_candidate;
                    snapshot_download_time =
                        config.snapshot_download_time_dist.sample(&mut self.rng);
                }
            }
        }

        self.cluster.node_at_mut(node_index).sync = Some(if use_snapshot {
            SyncState::snapshot_download(
                donor_id,
                log_replay_rate,
                snapshot_download_time,
                target_snap,
            )
        } else {
            SyncState::log_replay(donor_id, log_replay_rate)
        });

        // Schedule completion.  A `None` here means the node cannot catch up
        // under current conditions; the sync stays active without an event
        // and reschedule_active_syncs revisits it when conditions change.
        if let Some(remaining) = self.compute_remaining_sync_time(node_index, Some(can_commit)) {
            if remaining >= 0.0 {
                let time = self.cluster.current_time + remaining.max(0.0);
                self.event_queue
                    .push(Event::new(time, EventType::NodeSyncComplete, node_id));
            }
        }
    }

    /// Wall-clock time until a syncing node reaches its donor's position.
    ///
    /// `None` when the sync cannot complete under current conditions: no
    /// reachable donor, or a net catch-up rate at or below zero.
    fn compute_remaining_sync_time(
        &self,
        node_index: usize,
        can_commit: Option<bool>,
    ) -> Option<Seconds> {
        let node = self.cluster.node_at(node_index);
        let sync = node.sync.as_ref()?;

        let donor_index = self.cluster.index_of(sync.donor_id)?;
        let donor = self.cluster.node_at(donor_index);
        if !self.cluster.node_effectively_available(donor) {
            return None; // Donor unreachable, sync paused.
        }

        let can_commit =
            can_commit.unwrap_or_else(|| self.protocol.can_commit(&self.cluster));
        let commit_rate_eff = if can_commit {
            self.protocol.commit_rate()
        } else {
            0.0
        };
        let net_rate = sync.log_replay_rate - commit_rate_eff;

        match sync.phase {
            SyncPhase::SnapshotDownload => {
                let remaining_snapshot = sync.snapshot_remaining.max(0.0);
                // The donor keeps advancing while the snapshot downloads.
                let donor_at_download_end =
                    donor.last_applied_index + commit_rate_eff * remaining_snapshot;
                let remaining_log = donor_at_download_end - sync.target_snapshot_index;

                if remaining_log <= 0.0 {
                    return Some(remaining_snapshot);
                }
                if net_rate <= 0.0 {
                    return None;
                }
                Some(remaining_snapshot + remaining_log / net_rate)
            }
            SyncPhase::LogReplay => {
                let remaining_lag = donor.last_applied_index - node.last_applied_index;
                if remaining_lag <= 0.0 {
                    return Some(0.0);
                }
                if net_rate <= 0.0 {
                    return None;
                }
                Some(remaining_lag / net_rate)
            }
        }
    }

    /// Handle syncs whose donor has just gone away.
    ///
    /// Fails over to another donor that is actually ahead, keeping progress;
    /// otherwise clears the sync so `retry_pending_syncs` can restart it
    /// later.
    fn cancel_syncs_from_donor(&mut self, donor_id: Sym) {
        let mut dependents = std::mem::take(&mut self.scratch_donor);
        self.cluster.fill_nodes_syncing_from(donor_id, &mut dependents);

        for i in 0..dependents.len() {
            let node_index = dependents[i];
            let node_id = self.cluster.node_at(node_index).node_id;
            let node_applied = self.cluster.node_at(node_index).last_applied_index;

            let alt = self
                .cluster
                .find_sync_donor(node_id)
                .map(|i| self.cluster.node_at(i))
                .filter(|alt| alt.last_applied_index > node_applied)
                .map(|alt| alt.node_id);

            // Either way the pending completion is stale.
            self.event_queue
                .cancel_events_for(node_id, EventType::NodeSyncComplete);

            match alt {
                Some(alt_id) => {
                    // Seamless failover: swap donors and keep progress.
                    // reschedule_active_syncs issues a fresh completion.
                    if let Some(sync) = self.cluster.node_at_mut(node_index).sync.as_mut() {
                        sync.donor_id = alt_id;
                    }
                }
                None => self.cluster.node_at_mut(node_index).sync = None,
            }
        }

        self.scratch_donor = dependents;
    }

    /// Refresh completion times for every active sync.
    ///
    /// Run after each event, because commit ability and donor availability
    /// may both have changed.
    fn reschedule_active_syncs(&mut self, can_commit: Option<bool>) {
        let can_commit =
            can_commit.unwrap_or_else(|| self.protocol.can_commit(&self.cluster));

        let mut indices = std::mem::take(&mut self.scratch_resched);
        self.cluster.fill_active_and_standby_indices(&mut indices);

        for i in 0..indices.len() {
            let node_index = indices[i];
            let node = self.cluster.node_at(node_index);
            let Some(sync) = node.sync.as_ref() else {
                continue;
            };
            let node_id = node.node_id;
            let node_applied = node.last_applied_index;
            let donor_id = sync.donor_id;

            // Is the current donor still usable?
            let donor_ok = match self.cluster.index_of(donor_id) {
                Some(i) => {
                    let donor = self.cluster.node_at(i);
                    self.cluster.node_effectively_available(donor)
                        && donor.last_applied_index > node_applied
                }
                None => false,
            };

            if !donor_ok {
                let alt = self
                    .cluster
                    .find_sync_donor(node_id)
                    .map(|i| self.cluster.node_at(i))
                    .filter(|alt| alt.last_applied_index > node_applied)
                    .map(|alt| alt.node_id);

                match alt {
                    Some(alt_id) => {
                        if let Some(sync) = self.cluster.node_at_mut(node_index).sync.as_mut() {
                            sync.donor_id = alt_id;
                        }
                    }
                    None => {
                        self.event_queue
                            .cancel_events_for(node_id, EventType::NodeSyncComplete);
                        self.cluster.node_at_mut(node_index).sync = None;
                        continue;
                    }
                }
            }

            match self.compute_remaining_sync_time(node_index, Some(can_commit)) {
                Some(remaining) if remaining >= 0.0 => {
                    let time = self.cluster.current_time + remaining.max(0.0);
                    self.event_queue.reschedule(
                        node_id,
                        EventType::NodeSyncComplete,
                        time,
                        EventMeta::None,
                    );
                }
                _ => {
                    // Cannot catch up right now.  Drop the event but keep the
                    // sync state so it can resume when conditions improve.
                    self.event_queue
                        .cancel_events_for(node_id, EventType::NodeSyncComplete);
                }
            }
        }

        self.scratch_resched = indices;
    }

    /// Start syncs for lagging nodes that have none in flight.
    fn retry_pending_syncs(&mut self, can_commit: Option<bool>) {
        let mut pending = std::mem::take(&mut self.scratch_retry);
        self.cluster.fill_nodes_needing_sync(&mut pending);

        for i in 0..pending.len() {
            self.start_sync(pending[i], can_commit);
        }

        self.scratch_retry = pending;
    }

    // -- event processing ------------------------------------------------

    /// Apply one event and let the protocol and strategy respond.
    fn process_event(&mut self, event: Event) {
        let elapsed = event.time - self.cluster.current_time;

        // Record the interval against the pre-event cluster: it has not
        // changed since the previous event, so it is the state that actually
        // held during the interval.  Costs likewise, so that nodes added or
        // removed by this event are not billed for time before it.
        let pre_event_can_commit = self.protocol.can_commit(&self.cluster);
        self.metrics.record_elapsed(
            event.time,
            &self.cluster,
            self.protocol.as_ref(),
            Some(pre_event_can_commit),
        );
        self.advance_commit_index(elapsed, Some(pre_event_can_commit));
        self.cluster.current_time = event.time;

        if self.log_events {
            self.event_log.push(event.clone());
        }

        self.metrics.record_event(event.event_type);

        match event.event_type {
            EventType::NodeFailure => self.apply_node_failure(&event),
            EventType::NodeRecovery => self.apply_node_recovery(&event),
            EventType::NodeDataLoss => self.apply_node_data_loss(&event),
            EventType::NodeSyncComplete => self.apply_node_sync_complete(&event),
            EventType::NodeSpawnComplete => self.apply_node_spawn_complete(&event),
            EventType::NetworkOutageStart => self.apply_network_outage_start(&event),
            EventType::NetworkOutageEnd => self.apply_network_outage_end(&event),
            _ => {}
        }

        // A successful election is one where a LEADER_ELECTION_COMPLETE
        // event leaves us with a leader we did not have before -- i.e. not
        // stalled and not an epoch mismatch.
        let had_leader_before = self.protocol.leader_id().is_some();
        let mut new_events = std::mem::take(&mut self.scratch_events);
        new_events.clear();
        {
            let Simulator {
                protocol,
                cluster,
                rng,
                ..
            } = self;
            protocol.on_event(&event, cluster, rng, &mut new_events);
        }
        for new_event in new_events.drain(..) {
            self.event_queue.push(new_event);
        }
        self.scratch_events = new_events;
        if event.event_type == EventType::LeaderElectionComplete
            && !had_leader_before
            && self.protocol.leader_id().is_some()
        {
            self.metrics.record_leader_election();
        }

        let mut actions = std::mem::take(&mut self.scratch_actions);
        actions.clear();
        {
            let Simulator {
                strategy,
                cluster,
                rng,
                protocol,
                ..
            } = self;
            strategy.on_event(&event, cluster, rng, protocol.as_ref(), &mut actions);
        }
        for i in 0..actions.len() {
            let action = actions[i].clone();
            self.execute_action(action);
        }
        actions.clear();
        self.scratch_actions = actions;

        // Keep sync timing honest: conditions may have changed.
        let post_event_can_commit = self.protocol.can_commit(&self.cluster);
        self.reschedule_active_syncs(Some(post_event_can_commit));
        self.retry_pending_syncs(Some(post_event_can_commit));

        self.metrics
            .record_unavailability_transition(post_event_can_commit, event.time);
        self.metrics.update(
            &self.cluster,
            event.time,
            self.protocol.as_ref(),
            Some(post_event_can_commit),
        );
    }

    /// Apply a transient node failure.
    fn apply_node_failure(&mut self, event: &Event) {
        let Some(node_index) = self.cluster.index_of(event.target_id) else {
            return;
        };
        if !self.cluster.node_at(node_index).has_data {
            return;
        }

        let config = self.cluster.node_at(node_index).config.clone();
        {
            let node = self.cluster.node_at_mut(node_index);
            node.is_available = false;
            node.sync = None;
        }
        self.event_queue
            .cancel_events_for(event.target_id, EventType::NodeSyncComplete);

        // This node may have been someone's donor.
        self.cancel_syncs_from_donor(event.target_id);

        let recovery_time = config.recovery_dist.sample(&mut self.rng);
        let current_time = self.cluster.current_time;
        self.event_queue.push(Event::new(
            current_time + recovery_time,
            EventType::NodeRecovery,
            event.target_id,
        ));

        // The next failure is measured from the end of this outage.
        let next_failure_time = config.failure_dist.sample(&mut self.rng);
        self.event_queue.push(Event::new(
            current_time + recovery_time + next_failure_time,
            EventType::NodeFailure,
            event.target_id,
        ));
    }

    /// Apply recovery from a transient failure.
    ///
    /// The recovered node may need to catch up, and may itself unblock other
    /// nodes' syncs -- the latter is handled by the `retry_pending_syncs`
    /// call in [`process_event`](Self::process_event).
    fn apply_node_recovery(&mut self, event: &Event) {
        let Some(node_index) = self.cluster.index_of(event.target_id) else {
            return;
        };
        if !self.cluster.node_at(node_index).has_data {
            return;
        }

        self.cluster.node_at_mut(node_index).is_available = true;

        let commit_index = self.cluster.commit_index;
        if !self.cluster.node_at(node_index).is_up_to_date(commit_index) {
            self.start_sync(node_index, None);
        }
    }

    /// Apply permanent data loss to a node.
    fn apply_node_data_loss(&mut self, event: &Event) {
        let Some(node_index) = self.cluster.index_of(event.target_id) else {
            return;
        };
        {
            let node = self.cluster.node_at_mut(node_index);
            node.has_data = false;
            node.is_available = false;
            node.sync = None;
        }

        self.event_queue.cancel_all_for(event.target_id);
        self.cancel_syncs_from_donor(event.target_id);
    }

    /// Apply sync completion.
    ///
    /// On success the node snaps to the donor's position -- the data the
    /// donor actually holds -- rather than to `commit_index`.
    fn apply_node_sync_complete(&mut self, event: &Event) {
        let Some(node_index) = self.cluster.index_of(event.target_id) else {
            return;
        };
        let node = self.cluster.node_at(node_index);
        let Some(sync) = node.sync.as_ref() else {
            return;
        };
        let donor_id = sync.donor_id;
        let node_id = node.node_id;

        if !self.cluster.node_effectively_available(node) {
            self.cluster.node_at_mut(node_index).sync = None;
            return;
        }

        let donor_applied = self
            .cluster
            .index_of(donor_id)
            .map(|i| self.cluster.node_at(i))
            .filter(|d| self.cluster.node_effectively_available(d))
            .map(|d| d.last_applied_index);

        match donor_applied {
            Some(applied) => {
                let snapshot_interval = self.protocol.snapshot_interval();
                let node = self.cluster.node_at_mut(node_index);
                node.last_applied_index = applied;
                if snapshot_interval > 0.0 {
                    let snap = snapshot_boundary(applied, snapshot_interval);
                    if snap > node.last_snapshot_index {
                        node.last_snapshot_index = snap;
                    }
                }
            }
            None => {
                // The donor went down right at completion; try a failover and
                // let rescheduling take it from there.
                let alt = self
                    .cluster
                    .find_sync_donor(node_id)
                    .map(|i| self.cluster.node_at(i).node_id);
                if let Some(alt_id) = alt {
                    if let Some(sync) = self.cluster.node_at_mut(node_index).sync.as_mut() {
                        sync.donor_id = alt_id;
                    }
                    return;
                }
                // No donor at all: clear the sync and let the retry pass
                // pick it up later.
            }
        }

        self.cluster.node_at_mut(node_index).sync = None;
    }

    /// Apply completion of a node spawn.
    ///
    /// Moves the node out of the provisioning set -- where it was placed at
    /// request time so billing starts immediately -- and into the active set
    /// or standby block.
    fn apply_node_spawn_complete(&mut self, event: &Event) {
        let Some(node_config) = event.metadata.node_config().cloned() else {
            return;
        };
        let node_id = event.metadata.node_id().unwrap_or(event.target_id);
        let standby = event.metadata.standby();

        self.cluster.remove_provisioning_node(node_id);

        let region = self.cluster.intern(&node_config.region);
        let new_node = NodeState::new(node_id, region, node_config);
        if standby {
            self.cluster.add_standby_node(new_node);
        } else {
            self.cluster.add_node(new_node);
        }

        let Some(node_index) = self.cluster.index_of(node_id) else {
            return;
        };
        self.schedule_node_events(node_index);
        self.start_sync(node_index, None);
    }

    /// Apply the start of a region outage.
    fn apply_network_outage_start(&mut self, event: &Event) {
        let region = event.metadata.region().unwrap_or(event.target_id);
        self.cluster.network.add_outage(region);

        // Clear syncs for nodes in the affected region, and deal with
        // downstream nodes that were pulling from them.
        let mut affected = std::mem::take(&mut self.scratch_region_syms);
        affected.clear();
        affected.extend(
            self.cluster
                .active_and_standby()
                .filter(|n| n.region == region)
                .map(|n| n.node_id),
        );

        for i in 0..affected.len() {
            let node_id = affected[i];
            self.event_queue
                .cancel_events_for(node_id, EventType::NodeSyncComplete);
            if let Some(index) = self.cluster.index_of(node_id) {
                self.cluster.node_at_mut(index).sync = None;
            }
            self.cancel_syncs_from_donor(node_id);
        }
        self.scratch_region_syms = affected;

        if let Some(config) = &self.network_config {
            let duration = config.outage_duration_dist.sample(&mut self.rng);
            let current_time = self.cluster.current_time;
            self.event_queue.push(Event::with_meta(
                current_time + duration,
                EventType::NetworkOutageEnd,
                region,
                EventMeta::Region(region),
            ));
        }
    }

    /// Apply the end of a region outage.
    fn apply_network_outage_end(&mut self, event: &Event) {
        let region = event.metadata.region().unwrap_or(event.target_id);
        self.cluster.network.remove_outage(region);

        let commit_index = self.cluster.commit_index;
        let mut recovered = std::mem::take(&mut self.scratch_region);
        recovered.clear();
        recovered.extend(
            self.cluster
                .all_entries()
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    n.group != crate::sim::node::Group::Provisioning
                        && n.region == region
                        && n.is_available
                        && n.has_data
                        && !n.is_up_to_date(commit_index)
                        && n.sync.is_none()
                })
                .map(|(i, _)| i),
        );

        for i in 0..recovered.len() {
            self.start_sync(recovered[i], None);
        }
        self.scratch_region = recovered;

        self.schedule_network_outage(region);
    }

    // -- actions ---------------------------------------------------------

    /// Carry out one strategy action.
    fn execute_action(&mut self, action: Action) {
        match action {
            Action::SpawnNode {
                node_config,
                node_id,
                standby,
            } => {
                // Bill from the moment the node is requested, matching cloud
                // providers; it moves to its real group on spawn completion.
                let region = self.cluster.intern(&node_config.region);
                let mut provisioning_node =
                    NodeState::new(node_id, region, node_config.clone());
                provisioning_node.is_available = false;
                provisioning_node.has_data = false;
                self.cluster.add_provisioning_node(provisioning_node);

                let spawn_time = node_config.spawn_dist.sample(&mut self.rng);
                let current_time = self.cluster.current_time;
                self.event_queue.push(Event::with_meta(
                    current_time + spawn_time,
                    EventType::NodeSpawnComplete,
                    node_id,
                    EventMeta::Spawn {
                        node_config,
                        node_id,
                        standby,
                    },
                ));
            }

            Action::RemoveNode { node_id } => {
                self.cluster.remove_node(node_id);
                self.cluster.remove_standby_node(node_id);
                self.cluster.remove_provisioning_node(node_id);
                self.event_queue.cancel_all_for(node_id);
            }

            Action::PromoteNode { node_id } => {
                self.cluster.promote_standby(node_id);
            }

            Action::ScaleDown { new_size } | Action::ScaleUp { new_size } => {
                self.cluster.target_cluster_size = new_size;
            }

            Action::StartSync { node_id } => {
                if let Some(index) = self.cluster.index_of(node_id) {
                    self.start_sync(index, None);
                }
            }

            Action::ScheduleReplacementCheck { node_id, timeout } => {
                if timeout > 0.0 {
                    let time = self.cluster.current_time + timeout;
                    self.event_queue.push(Event::new(
                        time,
                        EventType::NodeReplacementTimeout,
                        node_id,
                    ));
                }
            }

            Action::CancelReplacementCheck { node_id } => {
                self.event_queue
                    .cancel_events_for(node_id, EventType::NodeReplacementTimeout);
            }

            Action::ScheduleReconfiguration { delay, target_size } => {
                let delay_val = delay.sample(&mut self.rng);
                if delay_val >= 0.0 && target_size > 0 {
                    let time = self.cluster.current_time + delay_val;
                    self.event_queue.push(Event::with_meta(
                        time,
                        EventType::ClusterReconfiguration,
                        super::ids::SYM_CLUSTER,
                        EventMeta::TargetSize(target_size),
                    ));
                }
            }

            Action::NoOp => {}
        }
    }

    // -- driving ---------------------------------------------------------

    /// Run until a stopping condition is met.
    ///
    /// Safe to call repeatedly: when the time limit is hit, the event that
    /// crossed it is pushed back so a later call sees it.
    pub fn run_until(
        &mut self,
        end_time: Option<Seconds>,
        stop_condition: Option<&dyn Fn(&ClusterState) -> bool>,
    ) -> SimulationResult {
        self.initialize();

        let end_reason = loop {
            let Some(event) = self.event_queue.pop() else {
                break EndReason::NoEvents;
            };

            if let Some(limit) = end_time {
                if event.time > limit {
                    self.event_queue.push(event);

                    let elapsed = limit - self.cluster.current_time;
                    self.advance_commit_index(elapsed, None);

                    // No event to apply, so state does not change between
                    // recording and updating.
                    self.metrics
                        .record_elapsed(limit, &self.cluster, self.protocol.as_ref(), None);
                    self.metrics
                        .update(&self.cluster, limit, self.protocol.as_ref(), None);
                    self.cluster.current_time = limit;
                    break EndReason::TimeLimit;
                }
            }

            self.process_event(event);

            // Data loss is the more specific outcome, so it is checked first.
            if self.protocol.has_actual_data_loss(&self.cluster) {
                break EndReason::DataLoss;
            }

            if let Some(condition) = stop_condition {
                if condition(&self.cluster) {
                    break EndReason::ConditionMet;
                }
            }
        };

        SimulationResult {
            end_time: self.cluster.current_time,
            end_reason,
            metrics: self.metrics.snapshot(),
            event_log: if self.log_events {
                self.event_log.clone()
            } else {
                Vec::new()
            },
        }
    }

    /// Run for a fixed duration from the current time.
    pub fn run_for(&mut self, duration: Seconds) -> SimulationResult {
        let end_time = self.cluster.current_time + duration;
        self.run_until(Some(end_time), None)
    }

    /// Run until data loss occurs or the time limit is reached.
    pub fn run_until_data_loss(&mut self, max_time: Option<Seconds>) -> SimulationResult {
        self.run_until(max_time, None)
    }
}
