//! Consensus protocol abstraction.
//!
//! Port of `powder/simulation/protocol.py`.
//!
//! A protocol computes algorithm-specific availability on top of the raw
//! cluster state, and may carry its own state (current leader,
//! election-in-progress).  [`ClusterState`] stays a pure description of
//! physical reality; the protocol decides what that means for the algorithm.
//!
//! Protocols also own recovery semantics: commit rate, snapshot interval, log
//! retention, and how a lagging node catches up.
//!
//! Python models this as an ABC with concrete defaults.  The Rust trait keeps
//! the same split: required methods for `can_commit` and `on_event`, defaults
//! for everything else, so a test can define a protocol by implementing two
//! methods exactly as `tests/test_strategy_refactor.py` does.

use super::cluster::ClusterState;
use super::distributions::{Distribution, Rng, Seconds};
use super::events::{Event, EventMeta, EventType};
use super::ids::{Sym, SYM_PROTOCOL};
use super::util::snapshot_boundary;

/// Upcast helper so a `&dyn Protocol` can be downcast to its concrete type.
///
/// Blanket-implemented, so protocol implementations -- including ones
/// defined inside tests -- never write this themselves.
pub trait AsAnyProtocol {
    /// The protocol as `&dyn Any`, for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: Protocol + 'static> AsAnyProtocol for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Algorithm-specific availability and recovery semantics.
pub trait Protocol: AsAnyProtocol {
    /// Committed data units produced per second of wall time while the
    /// system can commit.
    fn commit_rate(&self) -> f64 {
        1.0
    }

    /// Commit-index interval between snapshots.
    ///
    /// When a node's applied index crosses a multiple of this value it takes
    /// a snapshot and may truncate earlier log entries.  Zero disables
    /// snapshots.
    fn snapshot_interval(&self) -> f64 {
        0.0
    }

    /// Committed-data units of log a node retains.
    ///
    /// A node at `last_applied_index = D` keeps entries from
    /// `max(0, D - log_retention_ops)` to `D`; anything earlier has been
    /// garbage-collected and is no longer available for log-only replay by
    /// a syncing peer.  Zero means infinite retention.
    fn log_retention_ops(&self) -> f64 {
        0.0
    }

    /// Whether the system can accept writes right now.
    fn can_commit(&self, cluster: &ClusterState) -> bool;

    /// React to an event that has just been applied to the cluster,
    /// appending any events to schedule to `out`.
    ///
    /// Events go into a caller-owned buffer rather than a fresh `Vec`, so
    /// the simulator's per-event path does not allocate.  See
    /// [`collect_events`] for a returning form.
    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        rng: &mut Rng,
        out: &mut Vec<Event>,
    );

    /// Hook for protocol-specific setup, e.g. picking an initial leader.
    fn on_simulation_start(
        &mut self,
        _cluster: &ClusterState,
        _rng: &mut Rng,
        _out: &mut Vec<Event>,
    ) {
    }

    /// Nodes needed for a commit quorum.  Defaults to a simple majority.
    fn quorum_size(&self, cluster: &ClusterState) -> usize {
        cluster.num_active() / 2 + 1
    }

    /// Whether quorum is lost, meaning the surviving nodes may not hold the
    /// latest committed data.
    fn has_potential_data_loss(&self, cluster: &ClusterState) -> bool {
        let counts = cluster.availability_counts();
        counts.available < counts.quorum_size()
    }

    /// Whether every node holding the latest committed data is gone.
    ///
    /// Note this ignores region outages: a node that still has up-to-date
    /// data behind a partition has not lost it.
    fn has_actual_data_loss(&self, cluster: &ClusterState) -> bool {
        let commit_index = cluster.commit_index;
        !cluster
            .active()
            .any(|n| n.has_data && n.is_up_to_date(commit_index))
    }

    /// This protocol as a [`RaftLikeProtocol`], when it is one.
    ///
    /// Lets callers reach election state that the generic trait does not
    /// expose, without every protocol having to carry those concepts.
    fn as_raft(&self) -> Option<&RaftLikeProtocol> {
        self.as_any().downcast_ref::<RaftLikeProtocol>()
    }

    /// Current leader, for protocols that have one.
    ///
    /// Python reaches for this with `getattr(protocol, 'leader_id', None)`;
    /// the default here is the same "no leader concept" answer.
    fn leader_id(&self) -> Option<Sym> {
        None
    }

    /// Wall-clock time for a lagging node to catch up to its donor.
    ///
    /// The sync target is the donor's `last_applied_index` -- the data the
    /// donor actually has -- not `cluster.commit_index`.  The donor advances
    /// at `commit_rate` while the syncing node replays at `log_replay_rate`,
    /// giving a net catch-up rate of `log_replay_rate - commit_rate` while
    /// the cluster can commit, or the full replay rate while it cannot.
    ///
    /// The sync path depends on the donor's log garbage collection:
    ///
    /// * If the donor has discarded entries the node needs, the node *must*
    ///   download the donor's latest snapshot first, then replay the suffix.
    /// * Otherwise the node *may* take either path; the faster one is chosen
    ///   by comparing estimates built from distribution means.
    ///
    /// Returns `None` when the node cannot catch up (replay rate at or below
    /// commit rate) or no donor is available.
    fn compute_sync_time(
        &self,
        node_index: usize,
        cluster: &ClusterState,
        rng: &mut Rng,
    ) -> Option<Seconds> {
        let node = cluster.node_at(node_index);
        let donor_index = cluster.find_sync_donor(node.node_id)?;
        let donor = cluster.node_at(donor_index);

        let donor_lag = donor.last_applied_index - node.last_applied_index;
        if donor_lag <= 0.0 {
            return Some(0.0);
        }

        let can_commit = self.can_commit(cluster);
        let commit_rate_eff = if can_commit { self.commit_rate() } else { 0.0 };
        let log_replay_rate = node.config.log_replay_rate_dist.sample(rng);
        let net_rate = log_replay_rate - commit_rate_eff;

        if net_rate <= 0.0 {
            return None;
        }

        let snapshot_interval = self.snapshot_interval();
        let log_retention = self.log_retention_ops();

        let donor_earliest_log = if log_retention > 0.0 {
            (donor.last_applied_index - log_retention).max(0.0)
        } else {
            0.0
        };
        let must_snapshot = log_retention > 0.0 && node.last_applied_index < donor_earliest_log;

        if must_snapshot && snapshot_interval > 0.0 {
            // Forced snapshot: the donor no longer has the entries we need.
            let target_snap = snapshot_boundary(donor.last_applied_index, snapshot_interval);
            let snapshot_download_time = node.config.snapshot_download_time_dist.sample(rng);
            // The donor keeps advancing while we download.
            let remaining_log = donor.last_applied_index - target_snap
                + commit_rate_eff * snapshot_download_time;
            return Some(snapshot_download_time + remaining_log / net_rate);
        }

        if snapshot_interval > 0.0 && !must_snapshot {
            // Both paths are open; estimate each from means and take the
            // faster.  Sampling here would be wrong -- the choice has to be
            // made before the sync starts.
            let mean_replay_rate = node.config.log_replay_rate_dist.mean();
            let mut mean_net_rate = mean_replay_rate - commit_rate_eff;
            if mean_net_rate <= 0.0 {
                mean_net_rate = mean_replay_rate;
            }
            let log_only_est = donor_lag / mean_net_rate;

            let target_snap = snapshot_boundary(donor.last_applied_index, snapshot_interval);
            if target_snap > node.last_applied_index {
                let mean_snap_time = node.config.snapshot_download_time_dist.mean();
                let remaining_after_snap =
                    donor.last_applied_index - target_snap + commit_rate_eff * mean_snap_time;
                let snap_est = mean_snap_time + remaining_after_snap / mean_net_rate;

                if snap_est < log_only_est {
                    let snapshot_download_time =
                        node.config.snapshot_download_time_dist.sample(rng);
                    let remaining_log = donor.last_applied_index - target_snap
                        + commit_rate_eff * snapshot_download_time;
                    return Some(snapshot_download_time + remaining_log / net_rate);
                }
            }

            return Some(donor_lag / net_rate);
        }

        // No snapshots configured: log-only replay.
        Some(donor_lag / net_rate)
    }
}

/// Leaderless protocol with configurable quorum semantics.
///
/// With `up_to_date_quorum` set (the default), commits need a majority of
/// nodes that are available, hold data, **and** are current -- the model for
/// EPaxos or multi-decree Paxos.  With it clear, a majority of available
/// nodes suffices regardless of how far behind they are, modelling
/// eventually-consistent systems.
#[derive(Debug, Clone)]
pub struct LeaderlessProtocol {
    commit_rate: f64,
    snapshot_interval: f64,
    log_retention_ops: f64,
    up_to_date_quorum: bool,
}

impl Default for LeaderlessProtocol {
    fn default() -> Self {
        LeaderlessProtocol::new(1.0, 0.0, 0.0, true)
    }
}

impl LeaderlessProtocol {
    /// Construct with explicit parameters.
    pub fn new(
        commit_rate: f64,
        snapshot_interval: f64,
        log_retention_ops: f64,
        up_to_date_quorum: bool,
    ) -> Self {
        LeaderlessProtocol {
            commit_rate,
            snapshot_interval,
            log_retention_ops,
            up_to_date_quorum,
        }
    }

    /// Quorum of up-to-date nodes, the Python default.
    pub fn up_to_date_quorum_protocol(commit_rate: f64) -> Self {
        LeaderlessProtocol::new(commit_rate, 0.0, 0.0, true)
    }

    /// Quorum of available nodes, lagging or not.
    ///
    /// Equivalent to Python's `LeaderlessMajorityAvailableProtocol`.
    pub fn majority_available(commit_rate: f64) -> Self {
        LeaderlessProtocol::new(commit_rate, 0.0, 0.0, false)
    }

    /// Whether commits require an up-to-date quorum.
    pub fn up_to_date_quorum(&self) -> bool {
        self.up_to_date_quorum
    }
}

impl Protocol for LeaderlessProtocol {
    fn commit_rate(&self) -> f64 {
        self.commit_rate
    }

    fn snapshot_interval(&self) -> f64 {
        self.snapshot_interval
    }

    fn log_retention_ops(&self) -> f64 {
        self.log_retention_ops
    }

    fn can_commit(&self, cluster: &ClusterState) -> bool {
        let counts = cluster.availability_counts();
        let quorum = counts.quorum_size();
        if self.up_to_date_quorum {
            counts.up_to_date >= quorum
        } else {
            counts.available >= quorum
        }
    }

    fn on_event(
        &mut self,
        _event: &Event,
        _cluster: &ClusterState,
        _rng: &mut Rng,
        _out: &mut Vec<Event>,
    ) {
    }
}

/// Run one event through a protocol and collect the events it schedules.
///
/// A convenience over the buffered [`Protocol::on_event`], for tests and
/// other callers that are not on a hot path.
pub fn collect_events(
    protocol: &mut dyn Protocol,
    event: &Event,
    cluster: &ClusterState,
    rng: &mut Rng,
) -> Vec<Event> {
    let mut out = Vec::new();
    protocol.on_event(event, cluster, rng, &mut out);
    out
}

/// Run a protocol's start hook and collect the events it schedules.
pub fn collect_start_events(
    protocol: &mut dyn Protocol,
    cluster: &ClusterState,
    rng: &mut Rng,
) -> Vec<Event> {
    let mut out = Vec::new();
    protocol.on_simulation_start(cluster, rng, &mut out);
    out
}

/// Leader-based protocol with election downtime, modelling Raft or
/// multi-Paxos.
///
/// A single leader must be present to accept writes.  When the leader is
/// lost an election runs and the system is unavailable for its duration.
/// Commits additionally require a quorum of up-to-date nodes.
///
/// If no node is eligible when an election completes, the election *stalls*
/// and restarts on the next event that could make a node eligible.  Losing
/// quorum mid-election invalidates it the same way.
#[derive(Debug, Clone)]
pub struct RaftLikeProtocol {
    /// Time to complete a leader election after the leader is lost.
    pub election_time_dist: Distribution,
    commit_rate: f64,
    snapshot_interval: f64,
    log_retention_ops: f64,
    leader_id: Option<Sym>,
    election_in_progress: bool,
    election_stalled: bool,
    election_epoch: u64,
}

/// Events that may have made a node eligible, so a stalled election can
/// restart.
const AVAILABILITY_EVENTS: [EventType; 4] = [
    EventType::NodeRecovery,
    EventType::NetworkOutageEnd,
    EventType::NodeSyncComplete,
    EventType::NodeSpawnComplete,
];

/// Events that may have cost the cluster its quorum mid-election.
const FAILURE_EVENTS: [EventType; 3] = [
    EventType::NodeFailure,
    EventType::NodeDataLoss,
    EventType::NetworkOutageStart,
];

impl RaftLikeProtocol {
    /// Construct with an election-time distribution and default recovery
    /// parameters.
    pub fn new(election_time_dist: Distribution) -> Self {
        RaftLikeProtocol::with_params(election_time_dist, 1.0, 0.0, 0.0)
    }

    /// Construct with explicit parameters.
    pub fn with_params(
        election_time_dist: Distribution,
        commit_rate: f64,
        snapshot_interval: f64,
        log_retention_ops: f64,
    ) -> Self {
        RaftLikeProtocol {
            election_time_dist,
            commit_rate,
            snapshot_interval,
            log_retention_ops,
            leader_id: None,
            election_in_progress: false,
            election_stalled: false,
            election_epoch: 0,
        }
    }

    /// Rate at which elections complete (`1 / mean election time`).
    pub fn election_rate(&self) -> f64 {
        self.election_time_dist.approx_rate()
    }

    /// Whether an election is currently running.
    pub fn election_in_progress(&self) -> bool {
        self.election_in_progress
    }

    /// Whether an election has stalled waiting for an eligible node.
    pub fn election_stalled(&self) -> bool {
        self.election_stalled
    }

    /// Current election epoch, incremented on every start or restart.
    pub fn election_epoch(&self) -> u64 {
        self.election_epoch
    }

    /// Override the leader directly.  Used by tests to set up a scenario.
    pub fn set_leader(&mut self, leader_id: Option<Sym>) {
        self.leader_id = leader_id;
    }

    /// Force the election flags, for tests that need to start mid-election.
    ///
    /// Python's tests reach into `_election_in_progress` directly; this is
    /// the same escape hatch with a name that says so.
    pub fn force_election_state(&mut self, in_progress: bool, stalled: bool) {
        self.election_in_progress = in_progress;
        self.election_stalled = stalled;
    }

    /// Begin a leader election, appending its completion event.
    fn start_election(&mut self, cluster: &ClusterState, rng: &mut Rng, out: &mut Vec<Event>) {
        self.leader_id = None;
        self.election_in_progress = true;
        self.election_stalled = false;
        self.election_epoch += 1;
        self.schedule_election_completion(cluster, rng, out);
    }

    /// Restart a stalled election after a node became available.
    fn restart_election(&mut self, cluster: &ClusterState, rng: &mut Rng, out: &mut Vec<Event>) {
        self.election_stalled = false;
        self.election_epoch += 1;
        self.schedule_election_completion(cluster, rng, out);
    }

    fn schedule_election_completion(
        &mut self,
        cluster: &ClusterState,
        rng: &mut Rng,
        out: &mut Vec<Event>,
    ) {
        let election_duration = self.election_time_dist.sample(rng);
        out.push(Event::with_meta(
            cluster.current_time + election_duration,
            EventType::LeaderElectionComplete,
            SYM_PROTOCOL,
            EventMeta::Epoch(self.election_epoch),
        ));
    }

    /// Finalise an election: install a leader, or stall until a node becomes
    /// eligible.
    fn handle_election_complete(&mut self, event: &Event, cluster: &ClusterState) {
        // Drop stale completions from a cancelled epoch.  These arise when
        // quorum was lost mid-election and a new election already started.
        let event_epoch = event.metadata.epoch().unwrap_or(0);
        if event_epoch != self.election_epoch {
            return;
        }

        match self.pick_leader(cluster) {
            Some(new_leader) => {
                self.leader_id = Some(new_leader);
                self.election_in_progress = false;
                self.election_stalled = false;
            }
            None => {
                // No eligible node.  `on_event` restarts the election when a
                // recovery, outage-end, sync-complete or spawn-complete
                // event arrives.
                self.election_stalled = true;
            }
        }
    }

    /// Choose a leader from the eligible nodes.
    ///
    /// A node is eligible when it is effectively available and up-to-date,
    /// and a majority of nodes must be available for anyone to win the vote.
    /// Candidates are ordered by node ID *string*, matching Python's
    /// `eligible.sort(key=lambda n: n.node_id)`.
    fn pick_leader(&self, cluster: &ClusterState) -> Option<Sym> {
        let counts = cluster.availability_counts();
        if counts.available < counts.quorum_size() {
            return None;
        }

        let commit_index = cluster.commit_index;
        let mut best: Option<Sym> = None;
        for node in cluster.active() {
            if !cluster.node_effectively_available(node) || !node.is_up_to_date(commit_index) {
                continue;
            }
            best = match best {
                None => Some(node.node_id),
                Some(current) => {
                    if cluster.cmp_names(node.node_id, current) == std::cmp::Ordering::Less {
                        Some(node.node_id)
                    } else {
                        Some(current)
                    }
                }
            };
        }
        best
    }
}

impl Protocol for RaftLikeProtocol {
    fn commit_rate(&self) -> f64 {
        self.commit_rate
    }

    fn snapshot_interval(&self) -> f64 {
        self.snapshot_interval
    }

    fn log_retention_ops(&self) -> f64 {
        self.log_retention_ops
    }

    fn leader_id(&self) -> Option<Sym> {
        self.leader_id
    }

    fn can_commit(&self, cluster: &ClusterState) -> bool {
        if self.election_in_progress {
            return false;
        }
        let Some(leader_id) = self.leader_id else {
            return false;
        };
        let Some(leader) = cluster.get_node(leader_id) else {
            return false;
        };
        if !cluster.node_effectively_available(leader) {
            return false;
        }
        let counts = cluster.availability_counts();
        counts.up_to_date >= counts.quorum_size()
    }

    fn on_simulation_start(
        &mut self,
        cluster: &ClusterState,
        _rng: &mut Rng,
        _out: &mut Vec<Event>,
    ) {
        self.leader_id = self.pick_leader(cluster);
    }

    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        rng: &mut Rng,
        out: &mut Vec<Event>,
    ) {
        if event.event_type == EventType::LeaderElectionComplete {
            self.handle_election_complete(event, cluster);
            return;
        }

        if self.election_stalled && AVAILABILITY_EVENTS.contains(&event.event_type) {
            self.restart_election(cluster, rng, out);
            return;
        }

        if let Some(leader_id) = self.leader_id {
            if !self.election_in_progress {
                let leader_lost = match event.event_type {
                    EventType::NodeFailure | EventType::NodeDataLoss => {
                        event.target_id == leader_id
                    }
                    EventType::NetworkOutageStart => match cluster.get_node(leader_id) {
                        Some(leader) => {
                            let region = event.metadata.region().unwrap_or(event.target_id);
                            leader.region == region
                        }
                        None => false,
                    },
                    _ => false,
                };

                if leader_lost {
                    self.start_election(cluster, rng, out);
                    return;
                }
            }
        }

        // Losing quorum during a live election invalidates it; it has to
        // restart from scratch once quorum returns.
        if self.election_in_progress
            && !self.election_stalled
            && FAILURE_EVENTS.contains(&event.event_type)
            && cluster.num_available() < self.quorum_size(cluster)
        {
            self.election_stalled = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{collect_events, collect_start_events};
    use crate::sim::distributions::make_rng;
    use crate::sim::node::{NodeConfig, NodeConfigRef};
    use std::rc::Rc;

    fn config_in(region: &str) -> NodeConfigRef {
        Rc::new(NodeConfig {
            region: region.to_string(),
            cost_per_hour: 1.0,
            failure_dist: Distribution::constant(1000.0),
            recovery_dist: Distribution::constant(10.0),
            data_loss_dist: Distribution::constant(10_000.0),
            log_replay_rate_dist: Distribution::constant(100.0),
            snapshot_download_time_dist: Distribution::constant(5.0),
            spawn_dist: Distribution::constant(30.0),
        })
    }

    fn cluster_of(n: usize) -> ClusterState {
        let mut c = ClusterState::new(n);
        for i in 0..n {
            c.add_named_node(&format!("node{i}"), config_in("us-east"));
        }
        c
    }

    // -- quorum and data loss defaults -----------------------------------

    #[test]
    fn quorum_size_is_a_simple_majority() {
        let p = LeaderlessProtocol::default();
        for (n, expected) in [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (7, 4)] {
            assert_eq!(p.quorum_size(&cluster_of(n)), expected, "n = {n}");
        }
    }

    #[test]
    fn potential_data_loss_tracks_quorum() {
        let p = LeaderlessProtocol::default();
        let mut c = cluster_of(3);
        assert!(!p.has_potential_data_loss(&c));

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        assert!(!p.has_potential_data_loss(&c));

        let n1 = c.sym_of("node1").unwrap();
        c.get_node_mut(n1).unwrap().is_available = false;
        assert!(p.has_potential_data_loss(&c));
    }

    #[test]
    fn actual_data_loss_needs_every_current_node_gone() {
        let p = LeaderlessProtocol::default();
        let mut c = cluster_of(3);
        assert!(!p.has_actual_data_loss(&c));

        for i in c.active_indices() {
            c.node_at_mut(i).has_data = false;
        }
        assert!(p.has_actual_data_loss(&c));
    }

    #[test]
    fn actual_data_loss_ignores_region_outages() {
        let p = LeaderlessProtocol::default();
        let mut c = cluster_of(3);
        let region = c.sym_of("us-east").unwrap();
        c.network.add_outage(region);
        // Partitioned nodes still hold the data; it is not lost.
        assert!(!p.has_actual_data_loss(&c));
        assert!(p.has_potential_data_loss(&c));
    }

    // -- leaderless ------------------------------------------------------

    #[test]
    fn leaderless_up_to_date_quorum_requires_current_nodes() {
        let p = LeaderlessProtocol::default();
        let mut c = cluster_of(3);
        c.commit_index = 100.0;
        assert!(!p.can_commit(&c));

        let indices = c.active_indices();
        c.node_at_mut(indices[0]).last_applied_index = 100.0;
        assert!(!p.can_commit(&c));
        c.node_at_mut(indices[1]).last_applied_index = 100.0;
        assert!(p.can_commit(&c));
    }

    #[test]
    fn leaderless_majority_available_ignores_lag() {
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut c = cluster_of(3);
        c.commit_index = 100.0;
        // Everyone lags, but everyone is up.
        assert!(p.can_commit(&c));

        let n0 = c.sym_of("node0").unwrap();
        let n1 = c.sym_of("node1").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        assert!(p.can_commit(&c));
        c.get_node_mut(n1).unwrap().is_available = false;
        assert!(!p.can_commit(&c));
    }

    #[test]
    fn leaderless_ignores_events() {
        let mut p = LeaderlessProtocol::default();
        let c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        let event = Event::new(0.0, EventType::NodeFailure, 2);
        assert!(collect_events(&mut p, &event, &c, &mut rng).is_empty());
        assert!(collect_start_events(&mut p, &c, &mut rng).is_empty());
        assert_eq!(p.leader_id(), None);
    }

    // -- raft ------------------------------------------------------------

    #[test]
    fn raft_picks_an_initial_leader_by_name_order() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        assert_eq!(c.name_of(p.leader_id().unwrap()), "node0");
    }

    #[test]
    fn raft_leader_order_is_lexicographic_not_creation_order() {
        let mut c = ClusterState::new(3);
        // Created in an order where name ordering and creation order differ.
        c.add_named_node("node10", config_in("us-east"));
        c.add_named_node("node2", config_in("us-east"));
        c.add_named_node("node3", config_in("us-east"));

        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        assert_eq!(c.name_of(p.leader_id().unwrap()), "node10");
    }

    #[test]
    fn raft_cannot_commit_without_a_leader() {
        let p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let c = cluster_of(3);
        assert!(!p.can_commit(&c));
    }

    #[test]
    fn raft_commits_with_leader_and_quorum() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        assert!(p.can_commit(&c));
    }

    #[test]
    fn raft_leader_failure_starts_an_election() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        let leader = p.leader_id().unwrap();

        c.current_time = 10.0;
        c.get_node_mut(leader).unwrap().is_available = false;
        let events = collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, leader),
            &c,
            &mut rng,
        );

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EventType::LeaderElectionComplete);
        // Scheduled from the cluster clock, not the triggering event's time.
        assert_eq!(events[0].time, 15.0);
        assert_eq!(events[0].target_id, SYM_PROTOCOL);
        assert!(p.election_in_progress());
        assert_eq!(p.leader_id(), None);
        assert!(!p.can_commit(&c));
    }

    #[test]
    fn raft_ignores_failure_of_a_follower() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);

        let follower = c.sym_of("node2").unwrap();
        c.get_node_mut(follower).unwrap().is_available = false;
        let events = collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, follower),
            &c,
            &mut rng,
        );
        assert!(events.is_empty());
        assert!(!p.election_in_progress());
    }

    #[test]
    fn raft_region_outage_covering_the_leader_starts_an_election() {
        let mut c = ClusterState::new(3);
        c.add_named_node("node0", config_in("us-east"));
        c.add_named_node("node1", config_in("eu-west"));
        c.add_named_node("node2", config_in("eu-west"));

        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        assert_eq!(c.name_of(p.leader_id().unwrap()), "node0");

        let us_east = c.sym_of("us-east").unwrap();
        c.network.add_outage(us_east);
        let event = Event::with_meta(
            10.0,
            EventType::NetworkOutageStart,
            us_east,
            EventMeta::Region(us_east),
        );
        let events = collect_events(&mut p, &event, &c, &mut rng);
        assert_eq!(events.len(), 1);
        assert!(p.election_in_progress());
    }

    #[test]
    fn raft_election_completion_installs_a_new_leader() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        let old_leader = p.leader_id().unwrap();

        c.get_node_mut(old_leader).unwrap().is_available = false;
        let events = collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, old_leader),
            &c,
            &mut rng,
        );
        collect_events(&mut p, &events[0], &c, &mut rng);

        assert!(!p.election_in_progress());
        assert_eq!(c.name_of(p.leader_id().unwrap()), "node1");
        assert!(p.can_commit(&c));
    }

    #[test]
    fn raft_election_stalls_when_nobody_is_eligible() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        let leader = p.leader_id().unwrap();

        for i in c.active_indices() {
            c.node_at_mut(i).is_available = false;
        }
        let events = collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, leader),
            &c,
            &mut rng,
        );
        collect_events(&mut p, &events[0], &c, &mut rng);

        assert!(p.election_stalled());
        assert_eq!(p.leader_id(), None);
    }

    #[test]
    fn raft_stalled_election_restarts_on_recovery() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        let leader = p.leader_id().unwrap();

        for i in c.active_indices() {
            c.node_at_mut(i).is_available = false;
        }
        let events = collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, leader),
            &c,
            &mut rng,
        );
        collect_events(&mut p, &events[0], &c, &mut rng);
        assert!(p.election_stalled());

        for i in c.active_indices() {
            c.node_at_mut(i).is_available = true;
        }
        c.current_time = 30.0;
        let restart = collect_events(&mut p, 
            &Event::new(30.0, EventType::NodeRecovery, leader),
            &c,
            &mut rng,
        );
        assert_eq!(restart.len(), 1);
        assert!(!p.election_stalled());
        assert_eq!(restart[0].time, 35.0);
    }

    #[test]
    fn raft_stale_election_events_are_ignored() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        let leader = p.leader_id().unwrap();
        c.get_node_mut(leader).unwrap().is_available = false;
        collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, leader),
            &c,
            &mut rng,
        );

        // Epoch 0 belongs to an election that never happened.
        let stale = Event::with_meta(
            12.0,
            EventType::LeaderElectionComplete,
            SYM_PROTOCOL,
            EventMeta::Epoch(0),
        );
        collect_events(&mut p, &stale, &c, &mut rng);
        assert!(p.election_in_progress());
        assert_eq!(p.leader_id(), None);
    }

    #[test]
    fn raft_losing_quorum_mid_election_stalls_it() {
        let mut p = RaftLikeProtocol::new(Distribution::constant(5.0));
        let mut c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        collect_start_events(&mut p, &c, &mut rng);
        let leader = p.leader_id().unwrap();

        c.get_node_mut(leader).unwrap().is_available = false;
        collect_events(&mut p, 
            &Event::new(10.0, EventType::NodeFailure, leader),
            &c,
            &mut rng,
        );
        assert!(!p.election_stalled());

        let other = c.sym_of("node1").unwrap();
        c.get_node_mut(other).unwrap().is_available = false;
        collect_events(&mut p, 
            &Event::new(11.0, EventType::NodeFailure, other),
            &c,
            &mut rng,
        );
        assert!(p.election_stalled());
    }

    #[test]
    fn raft_election_rate_is_the_inverse_mean() {
        let p = RaftLikeProtocol::new(Distribution::constant(4.0));
        assert_eq!(p.election_rate(), 0.25);
    }

    // -- sync time -------------------------------------------------------

    #[test]
    fn sync_time_is_zero_when_already_caught_up() {
        let p = LeaderlessProtocol::default();
        let c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        assert_eq!(p.compute_sync_time(0, &c, &mut rng), Some(0.0));
    }

    #[test]
    fn sync_time_is_lag_over_net_rate() {
        let p = LeaderlessProtocol::new(1.0, 0.0, 0.0, true);
        let mut c = cluster_of(3);
        let indices = c.active_indices();
        // node0 lags 100 behind node1; the cluster can commit at 1.0/s and
        // node0 replays at 100/s, for a net 99/s.
        c.node_at_mut(indices[1]).last_applied_index = 100.0;
        c.node_at_mut(indices[2]).last_applied_index = 100.0;

        let mut rng = make_rng(Some(1));
        let t = p.compute_sync_time(indices[0], &c, &mut rng).unwrap();
        assert!((t - 100.0 / 99.0).abs() < 1e-12, "got {t}");
    }

    #[test]
    fn sync_time_uses_full_rate_when_the_cluster_cannot_commit() {
        let p = LeaderlessProtocol::new(1.0, 0.0, 0.0, true);
        let mut c = cluster_of(3);
        c.commit_index = 100.0;
        let indices = c.active_indices();
        c.node_at_mut(indices[1]).last_applied_index = 100.0;
        // Only one node is current, so there is no quorum and no commits.
        assert!(!p.can_commit(&c));

        let mut rng = make_rng(Some(1));
        let t = p.compute_sync_time(indices[0], &c, &mut rng).unwrap();
        assert!((t - 1.0).abs() < 1e-12, "got {t}");
    }

    #[test]
    fn sync_time_is_none_when_replay_cannot_outpace_commits() {
        let p = LeaderlessProtocol::new(500.0, 0.0, 0.0, false);
        let mut c = cluster_of(3);
        let indices = c.active_indices();
        c.node_at_mut(indices[1]).last_applied_index = 100.0;
        c.node_at_mut(indices[2]).last_applied_index = 100.0;

        let mut rng = make_rng(Some(1));
        // Replay runs at 100/s against a 500/s commit rate.
        assert_eq!(p.compute_sync_time(indices[0], &c, &mut rng), None);
    }

    #[test]
    fn sync_time_is_none_without_a_donor() {
        let p = LeaderlessProtocol::default();
        let mut c = cluster_of(2);
        let indices = c.active_indices();
        c.node_at_mut(indices[1]).is_available = false;
        let mut rng = make_rng(Some(1));
        assert_eq!(p.compute_sync_time(indices[0], &c, &mut rng), None);
    }

    #[test]
    fn forced_snapshot_path_when_the_donor_gced_the_log() {
        // Retention of 50 with the donor at 1000 means entries below 950 are
        // gone, so a node at 0 must start from the snapshot at 900.
        let p = LeaderlessProtocol::new(0.0, 300.0, 50.0, false);
        let mut c = cluster_of(3);
        let indices = c.active_indices();
        c.node_at_mut(indices[1]).last_applied_index = 1000.0;
        c.node_at_mut(indices[2]).last_applied_index = 1000.0;

        let mut rng = make_rng(Some(1));
        let t = p.compute_sync_time(indices[0], &c, &mut rng).unwrap();
        // 5s download, then 100 units of log suffix at 100/s.
        assert!((t - (5.0 + 1.0)).abs() < 1e-12, "got {t}");
    }
}
