//! Cluster management strategies.
//!
//! Port of `powder/simulation/strategy.py`.  A strategy observes events and
//! returns [`Action`]s for the simulator to execute; it never mutates the
//! cluster itself.
//!
//! Python expresses the adaptive strategy as a subclass of the replacement
//! strategy that overrides `_maintain_cluster_size`.  Rust has no inherited
//! virtual dispatch, so the shared behaviour lives in [`ReplacementCore`] and
//! the one varying input -- which cluster size to trim toward -- is passed in
//! explicitly.

use super::cluster::ClusterState;
use super::distributions::{Distribution, Rng, Seconds};
use super::events::{Event, EventType};
use super::ids::Sym;
use super::node::NodeConfigRef;
use super::protocol::Protocol;

/// A delay that is either a fixed number of seconds or drawn from a
/// distribution.
///
/// Python types `reconfiguration_dist` as `Seconds` but then checks
/// `hasattr(delay, "sample")`, so both forms are accepted at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Delay {
    /// A constant number of seconds.
    Fixed(Seconds),
    /// Sampled fresh each time the delay is used.
    Sampled(Distribution),
}

impl Delay {
    /// Draw the delay.
    pub fn sample(&self, rng: &mut Rng) -> Seconds {
        match self {
            Delay::Fixed(v) => *v,
            Delay::Sampled(d) => d.sample(rng),
        }
    }
}

/// Discriminant for [`Action`], mirroring Python's `ActionType` enum.
///
/// Useful in tests that want to assert on the kind of action without
/// destructuring the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /// Start spawning a new node.
    SpawnNode,
    /// Remove a node from the cluster.
    RemoveNode,
    /// Reduce the target cluster size.
    ScaleDown,
    /// Increase the target cluster size.
    ScaleUp,
    /// Trigger a node to start syncing.
    StartSync,
    /// Schedule a replacement timeout.
    ScheduleReplacementCheck,
    /// Cancel a pending replacement timeout.
    CancelReplacementCheck,
    /// Schedule a cluster reconfiguration.
    ScheduleReconfiguration,
    /// Promote a node from standby to active.
    PromoteNode,
    /// Do nothing.
    NoOp,
}

/// An instruction from a strategy to the simulator.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Begin provisioning a node.
    SpawnNode {
        /// Configuration for the new node.
        node_config: NodeConfigRef,
        /// Identifier to give it.
        node_id: Sym,
        /// Whether it joins standby rather than the active set.
        standby: bool,
    },
    /// Drop a node from every membership set and cancel its events.
    RemoveNode {
        /// Node to remove.
        node_id: Sym,
    },
    /// Lower the target cluster size.
    ScaleDown {
        /// New target.
        new_size: usize,
    },
    /// Raise the target cluster size.
    ScaleUp {
        /// New target.
        new_size: usize,
    },
    /// Ask a node to begin catching up.
    StartSync {
        /// Node to sync.
        node_id: Sym,
    },
    /// Arm a replacement timeout for an unavailable node.
    ScheduleReplacementCheck {
        /// Node being watched.
        node_id: Sym,
        /// How long to wait before replacing it.
        timeout: Seconds,
    },
    /// Disarm a node's replacement timeout.
    CancelReplacementCheck {
        /// Node to stop watching.
        node_id: Sym,
    },
    /// Schedule a future change of target cluster size.
    ScheduleReconfiguration {
        /// How long the reconfiguration takes to take effect.
        delay: Delay,
        /// Size to reconfigure to.
        target_size: usize,
    },
    /// Move a synced standby node into the active set.
    PromoteNode {
        /// Node to promote.
        node_id: Sym,
    },
    /// Explicit no-op.
    NoOp,
}

impl Action {
    /// The action's discriminant.
    pub fn action_type(&self) -> ActionType {
        match self {
            Action::SpawnNode { .. } => ActionType::SpawnNode,
            Action::RemoveNode { .. } => ActionType::RemoveNode,
            Action::ScaleDown { .. } => ActionType::ScaleDown,
            Action::ScaleUp { .. } => ActionType::ScaleUp,
            Action::StartSync { .. } => ActionType::StartSync,
            Action::ScheduleReplacementCheck { .. } => ActionType::ScheduleReplacementCheck,
            Action::CancelReplacementCheck { .. } => ActionType::CancelReplacementCheck,
            Action::ScheduleReconfiguration { .. } => ActionType::ScheduleReconfiguration,
            Action::PromoteNode { .. } => ActionType::PromoteNode,
            Action::NoOp => ActionType::NoOp,
        }
    }

    /// The node this action targets, where it has one.
    pub fn node_id(&self) -> Option<Sym> {
        match self {
            Action::SpawnNode { node_id, .. }
            | Action::RemoveNode { node_id }
            | Action::StartSync { node_id }
            | Action::ScheduleReplacementCheck { node_id, .. }
            | Action::CancelReplacementCheck { node_id }
            | Action::PromoteNode { node_id } => Some(*node_id),
            _ => None,
        }
    }
}

/// Reacts to simulation events by appending actions.
///
/// Actions are written into a caller-owned buffer rather than returned in a
/// fresh `Vec`.  The simulator calls this on every event and reuses one
/// buffer for the whole run, so a steady-state event allocates nothing.
/// Python returns a list here; that shape would cost an allocation per
/// event without changing any behaviour.  Use [`collect_actions`] where a
/// returned vector reads better, such as in a test.
pub trait ClusterStrategy {
    /// React to an event that has just been applied to the cluster,
    /// appending any actions to `out`.
    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        rng: &mut Rng,
        protocol: &dyn Protocol,
        out: &mut Vec<Action>,
    );

    /// Hook for initialisation actions, run once at simulation start.
    fn on_simulation_start(
        &mut self,
        _cluster: &ClusterState,
        _rng: &mut Rng,
        _out: &mut Vec<Action>,
    ) {
    }

    /// Rate at which the timeout-then-replace pipeline completes.
    ///
    /// Consumed by the Markov builders on the Python side; kept here for
    /// interface parity.
    fn replacement_rate(&self) -> f64 {
        0.0
    }
}

/// Strategy that takes no actions.
///
/// Useful as a baseline: it shows natural cluster degradation with no
/// intervention.
#[derive(Debug, Clone, Default)]
pub struct NoOpStrategy;

impl ClusterStrategy for NoOpStrategy {
    fn on_event(
        &mut self,
        _event: &Event,
        _cluster: &ClusterState,
        _rng: &mut Rng,
        _protocol: &dyn Protocol,
        _out: &mut Vec<Action>,
    ) {
    }
}

/// Run one event through a strategy and collect the actions it produces.
///
/// A convenience over the buffered [`ClusterStrategy::on_event`], for tests
/// and other callers that are not on a hot path.
pub fn collect_actions(
    strategy: &mut dyn ClusterStrategy,
    event: &Event,
    cluster: &ClusterState,
    rng: &mut Rng,
    protocol: &dyn Protocol,
) -> Vec<Action> {
    let mut out = Vec::new();
    strategy.on_event(event, cluster, rng, protocol, &mut out);
    out
}

/// Run a strategy's start hook and collect the actions it produces.
pub fn collect_start_actions(
    strategy: &mut dyn ClusterStrategy,
    cluster: &ClusterState,
    rng: &mut Rng,
) -> Vec<Action> {
    let mut out = Vec::new();
    strategy.on_simulation_start(cluster, rng, &mut out);
    out
}

/// Events that can change whether a node is available, and so may require
/// arming or disarming replacement timeouts.
const AVAILABILITY_EVENTS: [EventType; 6] = [
    EventType::NodeFailure,
    EventType::NodeRecovery,
    EventType::NodeDataLoss,
    EventType::NetworkOutageStart,
    EventType::NetworkOutageEnd,
    EventType::NodeSpawnComplete,
];

/// Events that prompt the adaptive strategy to reconsider cluster size.
const SCALING_EVENTS: [EventType; 4] = [
    EventType::NodeFailure,
    EventType::NodeDataLoss,
    EventType::NodeRecovery,
    EventType::NodeSpawnComplete,
];

/// Behaviour shared by the replacement strategies.
///
/// Holds the timeout bookkeeping and the spawn counter.  The only thing the
/// adaptive subclass changes is the size it trims the cluster toward, which
/// callers pass to [`maintain_cluster_size`](Self::maintain_cluster_size).
#[derive(Debug, Clone)]
struct ReplacementCore {
    failure_timeout: Seconds,
    default_node_config: Option<NodeConfigRef>,
    safe_mode: bool,
    spawn_counter: usize,
    /// Spawn requests that have not completed yet.
    pending_spawns: Vec<Sym>,
    /// Nodes with an armed replacement timeout.
    timeout_pending: Vec<Sym>,
    /// Reused buffer for the cluster-size ranking, so the trim does not
    /// allocate on every event.
    scratch_indices: Vec<usize>,
    /// Reused buffer for the replacement node name.
    scratch_name: String,
}

impl ReplacementCore {
    fn new(
        failure_timeout: Seconds,
        default_node_config: Option<NodeConfigRef>,
        safe_mode: bool,
    ) -> Self {
        ReplacementCore {
            failure_timeout,
            default_node_config,
            safe_mode,
            spawn_counter: 0,
            pending_spawns: Vec::new(),
            timeout_pending: Vec::new(),
            scratch_indices: Vec::new(),
            scratch_name: String::new(),
        }
    }

    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        protocol: &dyn Protocol,
        maintain_target: usize,
        out: &mut Vec<Action>,
    ) {
        match event.event_type {
            EventType::NodeRecovery => {
                if let Some(node) = cluster.get_node(event.target_id) {
                    if !node.is_up_to_date(cluster.commit_index) {
                        out.push(Action::StartSync {
                            node_id: node.node_id,
                        });
                    }
                }
            }
            EventType::NodeReplacementTimeout => {
                let node_id = event.target_id;
                // The node stays in `timeout_pending` so a second timeout
                // cannot spawn a duplicate replacement.
                if let Some(node) = cluster.get_node(node_id) {
                    if !cluster.node_effectively_available(node) {
                        self.replace_node(node_id, cluster, out);
                    }
                }
            }
            EventType::NodeSpawnComplete => {
                let node_id = event.metadata.node_id().unwrap_or(event.target_id);
                self.pending_spawns.retain(|&s| s != node_id);
            }
            _ => {}
        }

        if AVAILABILITY_EVENTS.contains(&event.event_type) {
            let num_available = cluster.num_available();
            self.reassess_timeouts(cluster, num_available, out);
        }

        self.promote_eligible_standbys(cluster, protocol, out);
        self.maintain_cluster_size(cluster, maintain_target, out);
    }

    /// Cancel timeouts when the cluster is entirely down, otherwise keep the
    /// armed set in step with which nodes are unavailable.
    fn reassess_timeouts(
        &mut self,
        cluster: &ClusterState,
        num_available: usize,
        out: &mut Vec<Action>,
    ) {
        if num_available == 0 {
            for &pending_id in &self.timeout_pending {
                out.push(Action::CancelReplacementCheck {
                    node_id: pending_id,
                });
            }
            self.timeout_pending.clear();
            return;
        }

        // Disarm nodes that recovered or no longer exist.  Removal is in
        // place and order-preserving; the set is small enough that the
        // shift costs nothing and keeping the order keeps the emitted
        // action sequence stable.
        let mut i = 0;
        while i < self.timeout_pending.len() {
            let pending_id = self.timeout_pending[i];
            let recovered = match cluster.get_node(pending_id) {
                None => true,
                Some(node) => cluster.node_effectively_available(node),
            };
            if recovered {
                out.push(Action::CancelReplacementCheck {
                    node_id: pending_id,
                });
                self.timeout_pending.remove(i);
            } else {
                i += 1;
            }
        }

        // Arm nodes that are unavailable and not already being watched.
        for node in cluster.active() {
            if !cluster.node_effectively_available(node)
                && !self.timeout_pending.contains(&node.node_id)
            {
                self.timeout_pending.push(node.node_id);
                out.push(Action::ScheduleReplacementCheck {
                    node_id: node.node_id,
                    timeout: self.failure_timeout,
                });
            }
        }
    }

    /// Promote standby nodes that have caught up, if the protocol permits.
    fn promote_eligible_standbys(
        &self,
        cluster: &ClusterState,
        protocol: &dyn Protocol,
        out: &mut Vec<Action>,
    ) {
        let commit_index = cluster.commit_index;
        // Hoisted out of the loop: the cluster does not change while this
        // runs, so one evaluation answers for every standby node.  Python
        // re-asks per node, which is the same answer at more cost.
        let mut can_commit: Option<bool> = None;
        for node in cluster.standby() {
            if !node.is_up_to_date(commit_index) {
                continue;
            }
            let permitted = if self.safe_mode {
                *can_commit.get_or_insert_with(|| protocol.can_commit(cluster))
            } else {
                true
            };
            if permitted {
                out.push(Action::PromoteNode {
                    node_id: node.node_id,
                });
            }
        }
    }

    /// Spawn a replacement for a failed node, without removing it yet.
    ///
    /// Replacement is purely additive: the new node lands in standby and is
    /// promoted once it has synced.
    fn replace_node(
        &mut self,
        failed_node_id: Sym,
        cluster: &ClusterState,
        out: &mut Vec<Action>,
    ) {
        let Some(failed_node) = cluster.get_node(failed_node_id) else {
            return;
        };

        let node_config = self
            .default_node_config
            .clone()
            .unwrap_or_else(|| failed_node.config.clone());

        self.spawn_counter += 1;
        let new_node_id = self.intern_replacement_name(cluster);
        self.pending_spawns.push(new_node_id);

        out.push(Action::SpawnNode {
            node_config,
            node_id: new_node_id,
            standby: true,
        });
    }

    /// Intern `replacement_{counter}` using a reused string buffer.
    fn intern_replacement_name(&mut self, cluster: &ClusterState) -> Sym {
        use std::fmt::Write;
        self.scratch_name.clear();
        let _ = write!(self.scratch_name, "replacement_{}", self.spawn_counter);
        cluster.intern(&self.scratch_name)
    }

    /// Trim the active set back to `target`, keeping the healthiest nodes.
    ///
    /// Nodes are ranked by availability, then currency, then applied index.
    /// Python relies on a stable reverse sort over dict-insertion order to
    /// break ties; the port adds an explicit node-ID tiebreaker so ordering
    /// never depends on container internals.
    fn maintain_cluster_size(
        &mut self,
        cluster: &ClusterState,
        target: usize,
        out: &mut Vec<Action>,
    ) {
        let mut indices = std::mem::take(&mut self.scratch_indices);
        cluster.fill_active_indices(&mut indices);
        if indices.len() <= target {
            self.scratch_indices = indices;
            return;
        }

        let commit_index = cluster.commit_index;
        let score = |i: &usize| {
            let n = cluster.node_at(*i);
            (
                cluster.node_effectively_available(n),
                n.is_up_to_date(commit_index),
                n.last_applied_index,
                n.node_id,
            )
        };

        indices.sort_by(|a, b| {
            let (aa, ab, ac, ad) = score(a);
            let (ba, bb, bc, bd) = score(b);
            // Best first: available, then current, then furthest ahead.
            rank_desc(aa, ba)
                .then_with(|| rank_desc(ab, bb))
                .then_with(|| bc.partial_cmp(&ac).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| ad.cmp(&bd))
        });

        out.extend(indices[target..].iter().map(|&i| Action::RemoveNode {
            node_id: cluster.node_at(i).node_id,
        }));
        self.scratch_indices = indices;
    }

    /// Top up the cluster to its target size at simulation start.
    fn on_simulation_start(&mut self, cluster: &ClusterState, out: &mut Vec<Action>) {
        while cluster.num_active() + self.pending_spawns.len() < cluster.target_cluster_size {
            let node_config = match self.default_node_config.clone() {
                Some(config) => config,
                // Fall back to an existing node's config; with no nodes at
                // all there is nothing to copy and we stop.
                None => match cluster.active().next() {
                    Some(existing) => existing.config.clone(),
                    None => break,
                },
            };

            self.spawn_counter += 1;
            let node_id = self.intern_replacement_name(cluster);
            self.pending_spawns.push(node_id);

            out.push(Action::SpawnNode {
                node_config,
                node_id,
                standby: false,
            });
        }
    }
}

/// Order two booleans with `true` first, so sorting puts the better node
/// ahead of the worse one.
#[inline]
fn rank_desc(a: bool, b: bool) -> std::cmp::Ordering {
    b.cmp(&a)
}

/// Replaces nodes that stay unavailable past a configurable timeout.
///
/// This models the standard production response to a node failure:
///
/// 1. A node becomes unavailable (transient failure, outage, or data loss).
/// 2. A failure-timeout countdown starts.
/// 3. Recovery before the timeout cancels the countdown.
/// 4. Otherwise a replacement is spawned into standby, promoted once synced,
///    and the cluster is trimmed back to its target size.
///
/// Data loss is treated as an ordinary failure and waits for the same
/// timeout rather than reacting immediately.
#[derive(Debug, Clone)]
pub struct NodeReplacementStrategy {
    core: ReplacementCore,
}

impl NodeReplacementStrategy {
    /// Construct with a failure timeout, an optional replacement-node config
    /// template, and whether removals require the protocol to be committable.
    pub fn new(
        failure_timeout: Seconds,
        default_node_config: Option<NodeConfigRef>,
        safe_mode: bool,
    ) -> Self {
        NodeReplacementStrategy {
            core: ReplacementCore::new(failure_timeout, default_node_config, safe_mode),
        }
    }

    /// How long a node must be unavailable before replacement is triggered.
    pub fn failure_timeout(&self) -> Seconds {
        self.core.failure_timeout
    }

    /// Nodes with an armed replacement timeout.
    pub fn timeout_pending(&self) -> &[Sym] {
        &self.core.timeout_pending
    }

    /// Spawn requests that have not completed yet.
    pub fn pending_spawns(&self) -> &[Sym] {
        &self.core.pending_spawns
    }
}

impl ClusterStrategy for NodeReplacementStrategy {
    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        _rng: &mut Rng,
        protocol: &dyn Protocol,
        out: &mut Vec<Action>,
    ) {
        let target = cluster.target_cluster_size;
        self.core.on_event(event, cluster, protocol, target, out);
    }

    fn on_simulation_start(
        &mut self,
        cluster: &ClusterState,
        _rng: &mut Rng,
        out: &mut Vec<Action>,
    ) {
        self.core.on_simulation_start(cluster, out);
    }

    fn replacement_rate(&self) -> f64 {
        1.0 / self.core.failure_timeout
    }
}

/// Replacement strategy that also shrinks the cluster during failures to
/// preserve a workable quorum, then grows it back as nodes return.
#[derive(Debug, Clone)]
pub struct AdaptiveReplacementStrategy {
    core: ReplacementCore,
    /// How long a reconfiguration takes to take effect.
    pub reconfiguration_dist: Delay,
    /// Unavailable-node count that triggers a scale down.
    pub scale_down_threshold: usize,
    /// Whether an external consensus service permits shrinking to 2 or 1.
    pub external_consensus: bool,
    /// Target sizes with a reconfiguration already in flight.
    pending_reconfigurations: Vec<usize>,
    /// Largest size the cluster should ever return to, captured at start.
    max_target_cluster_size: usize,
}

impl AdaptiveReplacementStrategy {
    /// Construct the adaptive strategy.
    pub fn new(
        failure_timeout: Seconds,
        reconfiguration_dist: Delay,
        scale_down_threshold: usize,
        external_consensus: bool,
        default_node_config: Option<NodeConfigRef>,
        safe_mode: bool,
    ) -> Self {
        AdaptiveReplacementStrategy {
            core: ReplacementCore::new(failure_timeout, default_node_config, safe_mode),
            reconfiguration_dist,
            scale_down_threshold,
            external_consensus,
            pending_reconfigurations: Vec::new(),
            max_target_cluster_size: 0,
        }
    }

    /// Largest size the cluster will scale back up to.
    pub fn max_target_cluster_size(&self) -> usize {
        self.max_target_cluster_size
    }

    /// Nodes with an armed replacement timeout.
    pub fn timeout_pending(&self) -> &[Sym] {
        &self.core.timeout_pending
    }

    /// Decide whether the cluster should change size, and if so schedule it.
    fn check_and_schedule_reconfiguration(
        &mut self,
        cluster: &ClusterState,
        out: &mut Vec<Action>,
    ) {
        let current_target = cluster.target_cluster_size;
        self.plan_resize(cluster, current_target, out);
    }

    /// Shared scale-up/scale-down decision, evaluated against `current_target`.
    ///
    /// Called both for the live target and, after a reconfiguration lands,
    /// for the target that is about to take effect.
    fn plan_resize(
        &mut self,
        cluster: &ClusterState,
        current_target: usize,
        out: &mut Vec<Action>,
    ) {
        let num_available = cluster.num_available();
        let mut new_target = current_target;

        // Scale down when too many nodes are missing.
        let deficit = current_target as isize - num_available as isize;
        if deficit >= self.scale_down_threshold as isize {
            if self.external_consensus {
                if current_target > 1 {
                    new_target = current_target - 1;
                }
            } else if current_target >= 5 {
                new_target = current_target - 2;
            }
        }

        // Scale back up when enough nodes are available to support it.
        if current_target < self.max_target_cluster_size {
            let potential_target = if self.external_consensus {
                current_target + 1
            } else {
                current_target + 2
            };
            if num_available >= potential_target {
                new_target = potential_target;
            }
        }

        if new_target != current_target && !self.pending_reconfigurations.contains(&new_target) {
            self.pending_reconfigurations.push(new_target);
            out.push(Action::ScheduleReconfiguration {
                delay: self.reconfiguration_dist,
                target_size: new_target,
            });
        }
    }

    /// Apply a reconfiguration whose delay has elapsed.
    fn handle_reconfiguration(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        protocol: &dyn Protocol,
        out: &mut Vec<Action>,
    ) {
        let target_size = event.metadata.target_size().unwrap_or(0);
        if target_size != 0 {
            self.pending_reconfigurations.retain(|&t| t != target_size);
        }

        if target_size == 0 || target_size == cluster.target_cluster_size {
            return;
        }

        // Reconfiguring needs the current configuration to be committable,
        // unless an external consensus service is doing the work.
        if !self.external_consensus && !protocol.can_commit(cluster) {
            return;
        }

        if target_size < cluster.target_cluster_size {
            // Everything recovered while the reconfiguration was in flight,
            // so there is no longer a reason to shrink.
            let deficit = cluster.target_cluster_size as isize - cluster.num_available() as isize;
            if deficit == 0 {
                return;
            }
            out.push(Action::ScaleDown {
                new_size: target_size,
            });
        } else {
            out.push(Action::ScaleUp {
                new_size: target_size,
            });
        }

        self.plan_resize(cluster, target_size, out);
    }
}

impl ClusterStrategy for AdaptiveReplacementStrategy {
    fn on_event(
        &mut self,
        event: &Event,
        cluster: &ClusterState,
        _rng: &mut Rng,
        protocol: &dyn Protocol,
        out: &mut Vec<Action>,
    ) {
        // Trim toward the provisioning target: keep the full physical fleet
        // even while the logical quorum target is temporarily lower.
        let maintain_target = self.max_target_cluster_size;
        self.core
            .on_event(event, cluster, protocol, maintain_target, out);

        if event.event_type == EventType::ClusterReconfiguration {
            self.handle_reconfiguration(event, cluster, protocol, out);
        }

        if SCALING_EVENTS.contains(&event.event_type) {
            self.check_and_schedule_reconfiguration(cluster, out);
        }
    }

    fn on_simulation_start(
        &mut self,
        cluster: &ClusterState,
        _rng: &mut Rng,
        out: &mut Vec<Action>,
    ) {
        self.max_target_cluster_size = cluster.target_cluster_size;
        self.core.on_simulation_start(cluster, out);
    }

    fn replacement_rate(&self) -> f64 {
        1.0 / self.core.failure_timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{collect_actions, collect_start_actions};
    use crate::sim::events::EventMeta;
    use crate::sim::distributions::make_rng;
    use crate::sim::node::{NodeConfig, NodeState};
    use crate::sim::protocol::LeaderlessProtocol;
    use std::rc::Rc;

    fn config() -> NodeConfigRef {
        Rc::new(NodeConfig {
            region: "us-east".to_string(),
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
            c.add_named_node(&format!("node{i}"), config());
        }
        c
    }

    fn of_type(actions: &[Action], t: ActionType) -> Vec<&Action> {
        actions.iter().filter(|a| a.action_type() == t).collect()
    }

    #[test]
    fn noop_strategy_does_nothing() {
        let mut s = NoOpStrategy;
        let c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));
        let event = Event::new(0.0, EventType::NodeFailure, 2);
        assert!(collect_actions(&mut s, &event, &c, &mut rng, &p).is_empty());
        assert!(collect_start_actions(&mut s, &c, &mut rng).is_empty());
        assert_eq!(s.replacement_rate(), 0.0);
    }

    #[test]
    fn replacement_rate_is_the_inverse_timeout() {
        let s = NodeReplacementStrategy::new(600.0, None, true);
        assert!((s.replacement_rate() - 1.0 / 600.0).abs() < 1e-15);
    }

    #[test]
    fn failure_arms_a_replacement_timeout() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeFailure, n0),
            &c,
            &mut rng,
            &p,
        );

        let scheduled = of_type(&actions, ActionType::ScheduleReplacementCheck);
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].node_id(), Some(n0));
        assert_eq!(s.timeout_pending(), &[n0]);
    }

    #[test]
    fn recovery_disarms_the_timeout() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeFailure, n0),
            &c,
            &mut rng,
            &p,
        );

        c.get_node_mut(n0).unwrap().is_available = true;
        let actions = collect_actions(&mut s, 
            &Event::new(20.0, EventType::NodeRecovery, n0),
            &c,
            &mut rng,
            &p,
        );

        assert_eq!(of_type(&actions, ActionType::CancelReplacementCheck).len(), 1);
        assert!(s.timeout_pending().is_empty());
    }

    #[test]
    fn total_outage_cancels_every_timeout() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeFailure, n0),
            &c,
            &mut rng,
            &p,
        );
        assert_eq!(s.timeout_pending().len(), 1);

        for i in c.active_indices() {
            c.node_at_mut(i).is_available = false;
        }
        let n1 = c.sym_of("node1").unwrap();
        let actions = collect_actions(&mut s, 
            &Event::new(20.0, EventType::NodeFailure, n1),
            &c,
            &mut rng,
            &p,
        );

        assert_eq!(of_type(&actions, ActionType::CancelReplacementCheck).len(), 1);
        assert!(s.timeout_pending().is_empty());
    }

    #[test]
    fn timeout_on_a_still_down_node_spawns_a_standby_replacement() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let n0 = c.sym_of("node0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        let actions = collect_actions(&mut s, 
            &Event::new(610.0, EventType::NodeReplacementTimeout, n0),
            &c,
            &mut rng,
            &p,
        );

        let spawns = of_type(&actions, ActionType::SpawnNode);
        assert_eq!(spawns.len(), 1);
        match spawns[0] {
            Action::SpawnNode {
                node_id, standby, ..
            } => {
                assert!(*standby, "replacement should land in standby");
                assert_eq!(c.name_of(*node_id), "replacement_1");
            }
            _ => unreachable!(),
        }
        assert_eq!(s.pending_spawns().len(), 1);
    }

    #[test]
    fn timeout_on_a_recovered_node_spawns_nothing() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let n0 = c.sym_of("node0").unwrap();
        let actions = collect_actions(&mut s, 
            &Event::new(610.0, EventType::NodeReplacementTimeout, n0),
            &c,
            &mut rng,
            &p,
        );
        assert!(of_type(&actions, ActionType::SpawnNode).is_empty());
    }

    #[test]
    fn synced_standby_nodes_are_promoted() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let spare = c.intern("spare");
        let region = c.intern("us-east");
        c.add_standby_node(NodeState::new(spare, region, config()));

        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeSpawnComplete, spare),
            &c,
            &mut rng,
            &p,
        );
        let promotions = of_type(&actions, ActionType::PromoteNode);
        assert_eq!(promotions.len(), 1);
        assert_eq!(promotions[0].node_id(), Some(spare));
    }

    #[test]
    fn lagging_standby_nodes_are_not_promoted() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        c.commit_index = 500.0;
        for i in c.active_indices() {
            c.node_at_mut(i).last_applied_index = 500.0;
        }
        let spare = c.intern("spare");
        let region = c.intern("us-east");
        c.add_standby_node(NodeState::new(spare, region, config()));

        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeSpawnComplete, spare),
            &c,
            &mut rng,
            &p,
        );
        assert!(of_type(&actions, ActionType::PromoteNode).is_empty());
    }

    #[test]
    fn safe_mode_blocks_promotion_while_uncommittable() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        // Two of three down: no quorum, so no promotion in safe mode.
        for i in c.active_indices().into_iter().take(2) {
            c.node_at_mut(i).is_available = false;
        }
        let spare = c.intern("spare");
        let region = c.intern("us-east");
        c.add_standby_node(NodeState::new(spare, region, config()));

        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeSpawnComplete, spare),
            &c,
            &mut rng,
            &p,
        );
        assert!(of_type(&actions, ActionType::PromoteNode).is_empty());

        // The same situation with safe mode off does promote.
        let mut unsafe_strategy = NodeReplacementStrategy::new(600.0, None, false);
        let actions = collect_actions(&mut unsafe_strategy, 
            &Event::new(10.0, EventType::NodeSpawnComplete, spare),
            &c,
            &mut rng,
            &p,
        );
        assert_eq!(of_type(&actions, ActionType::PromoteNode).len(), 1);
    }

    #[test]
    fn oversized_clusters_shed_their_worst_nodes() {
        let mut s = NodeReplacementStrategy::new(600.0, None, true);
        let mut c = cluster_of(5);
        c.target_cluster_size = 3;
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        // node0 and node1 are down, so they are the ones to drop.
        let n0 = c.sym_of("node0").unwrap();
        let n1 = c.sym_of("node1").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        c.get_node_mut(n1).unwrap().is_available = false;

        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeSpawnComplete, n0),
            &c,
            &mut rng,
            &p,
        );
        let removals: Vec<Sym> = of_type(&actions, ActionType::RemoveNode)
            .iter()
            .filter_map(|a| a.node_id())
            .collect();
        assert_eq!(removals.len(), 2);
        assert!(removals.contains(&n0));
        assert!(removals.contains(&n1));
    }

    #[test]
    fn simulation_start_tops_the_cluster_up_to_target() {
        let mut s = NodeReplacementStrategy::new(600.0, Some(config()), true);
        let mut c = cluster_of(2);
        c.target_cluster_size = 5;
        let mut rng = make_rng(Some(1));

        let actions = collect_start_actions(&mut s, &c, &mut rng);
        assert_eq!(actions.len(), 3);
        for action in &actions {
            match action {
                Action::SpawnNode { standby, .. } => {
                    assert!(!standby, "startup spawns join the active set");
                }
                _ => panic!("expected spawn actions"),
            }
        }
        assert_eq!(s.pending_spawns().len(), 3);
    }

    #[test]
    fn simulation_start_is_a_no_op_at_target_size() {
        let mut s = NodeReplacementStrategy::new(600.0, Some(config()), true);
        let c = cluster_of(3);
        let mut rng = make_rng(Some(1));
        assert!(collect_start_actions(&mut s, &c, &mut rng).is_empty());
    }

    #[test]
    fn spawn_completion_clears_the_pending_entry() {
        let mut s = NodeReplacementStrategy::new(600.0, Some(config()), true);
        let mut c = cluster_of(2);
        c.target_cluster_size = 3;
        let p = LeaderlessProtocol::default();
        let mut rng = make_rng(Some(1));

        let actions = collect_start_actions(&mut s, &c, &mut rng);
        let spawned = actions[0].node_id().unwrap();
        assert_eq!(s.pending_spawns(), &[spawned]);

        let mut event = Event::new(30.0, EventType::NodeSpawnComplete, spawned);
        event.metadata = EventMeta::Spawn {
            node_config: config(),
            node_id: spawned,
            standby: false,
        };
        collect_actions(&mut s, &event, &c, &mut rng, &p);
        assert!(s.pending_spawns().is_empty());
    }

    // -- adaptive --------------------------------------------------------

    fn adaptive(external_consensus: bool) -> AdaptiveReplacementStrategy {
        AdaptiveReplacementStrategy::new(
            600.0,
            Delay::Fixed(30.0),
            2,
            external_consensus,
            None,
            true,
        )
    }

    #[test]
    fn adaptive_captures_the_max_target_at_start() {
        let mut s = adaptive(false);
        let c = cluster_of(5);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);
        assert_eq!(s.max_target_cluster_size(), 5);
    }

    #[test]
    fn adaptive_schedules_a_scale_down_past_the_threshold() {
        let mut s = adaptive(false);
        let mut c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        // Two of five down meets the threshold; a 5-node cluster steps to 3.
        let indices = c.active_indices();
        c.node_at_mut(indices[0]).is_available = false;
        c.node_at_mut(indices[1]).is_available = false;
        let n0 = c.node_at(indices[0]).node_id;

        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeFailure, n0),
            &c,
            &mut rng,
            &p,
        );
        let reconfigs = of_type(&actions, ActionType::ScheduleReconfiguration);
        assert_eq!(reconfigs.len(), 1);
        match reconfigs[0] {
            Action::ScheduleReconfiguration { target_size, .. } => assert_eq!(*target_size, 3),
            _ => unreachable!(),
        }
    }

    #[test]
    fn adaptive_with_external_consensus_steps_down_by_one() {
        let mut s = adaptive(true);
        let mut c = cluster_of(3);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        let indices = c.active_indices();
        c.node_at_mut(indices[0]).is_available = false;
        c.node_at_mut(indices[1]).is_available = false;
        let n0 = c.node_at(indices[0]).node_id;

        let actions = collect_actions(&mut s, 
            &Event::new(10.0, EventType::NodeFailure, n0),
            &c,
            &mut rng,
            &p,
        );
        match of_type(&actions, ActionType::ScheduleReconfiguration)[0] {
            Action::ScheduleReconfiguration { target_size, .. } => assert_eq!(*target_size, 2),
            _ => unreachable!(),
        }
    }

    #[test]
    fn adaptive_does_not_schedule_the_same_target_twice() {
        let mut s = adaptive(false);
        let mut c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        let indices = c.active_indices();
        c.node_at_mut(indices[0]).is_available = false;
        c.node_at_mut(indices[1]).is_available = false;
        let n0 = c.node_at(indices[0]).node_id;
        let event = Event::new(10.0, EventType::NodeFailure, n0);

        let first = collect_actions(&mut s, &event, &c, &mut rng, &p);
        assert_eq!(of_type(&first, ActionType::ScheduleReconfiguration).len(), 1);
        let second = collect_actions(&mut s, &event, &c, &mut rng, &p);
        assert!(of_type(&second, ActionType::ScheduleReconfiguration).is_empty());
    }

    #[test]
    fn adaptive_reconfiguration_scales_down_then_plans_the_next_step() {
        let mut s = adaptive(false);
        let mut c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        let indices = c.active_indices();
        c.node_at_mut(indices[0]).is_available = false;
        c.node_at_mut(indices[1]).is_available = false;

        let mut event = Event::new(40.0, EventType::ClusterReconfiguration, 1);
        event.metadata = EventMeta::TargetSize(3);
        let actions = collect_actions(&mut s, &event, &c, &mut rng, &p);

        let downs = of_type(&actions, ActionType::ScaleDown);
        assert_eq!(downs.len(), 1);
        match downs[0] {
            Action::ScaleDown { new_size } => assert_eq!(*new_size, 3),
            _ => unreachable!(),
        }
    }

    #[test]
    fn adaptive_reconfiguration_is_skipped_once_everything_recovered() {
        let mut s = adaptive(false);
        let c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        // No deficit: every node is up, so shrinking makes no sense.
        let mut event = Event::new(40.0, EventType::ClusterReconfiguration, 1);
        event.metadata = EventMeta::TargetSize(3);
        let actions = collect_actions(&mut s, &event, &c, &mut rng, &p);
        assert!(of_type(&actions, ActionType::ScaleDown).is_empty());
    }

    #[test]
    fn adaptive_reconfiguration_requires_commit_without_external_consensus() {
        let mut s = adaptive(false);
        let mut c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        // Four of five down: the cluster cannot commit, so it cannot
        // reconfigure either.
        for i in c.active_indices().into_iter().take(4) {
            c.node_at_mut(i).is_available = false;
        }
        let mut event = Event::new(40.0, EventType::ClusterReconfiguration, 1);
        event.metadata = EventMeta::TargetSize(3);
        let actions = collect_actions(&mut s, &event, &c, &mut rng, &p);
        assert!(of_type(&actions, ActionType::ScaleDown).is_empty());
    }

    #[test]
    fn adaptive_scales_back_up_as_nodes_return() {
        let mut s = adaptive(false);
        let mut c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        // Already shrunk to 3 while all five nodes are healthy again.
        c.target_cluster_size = 3;
        let n0 = c.sym_of("node0").unwrap();
        let actions = collect_actions(&mut s, 
            &Event::new(100.0, EventType::NodeRecovery, n0),
            &c,
            &mut rng,
            &p,
        );
        match of_type(&actions, ActionType::ScheduleReconfiguration)[0] {
            Action::ScheduleReconfiguration { target_size, .. } => assert_eq!(*target_size, 5),
            _ => unreachable!(),
        }
    }

    #[test]
    fn adaptive_trims_toward_the_max_target_not_the_current_one() {
        let mut s = adaptive(false);
        let mut c = cluster_of(5);
        let p = LeaderlessProtocol::majority_available(1.0);
        let mut rng = make_rng(Some(1));
        collect_start_actions(&mut s, &c, &mut rng);

        // Logical target shrank to 3, but all five nodes should be kept.
        c.target_cluster_size = 3;
        let n0 = c.sym_of("node0").unwrap();
        let actions = collect_actions(&mut s, 
            &Event::new(100.0, EventType::NodeRecovery, n0),
            &c,
            &mut rng,
            &p,
        );
        assert!(of_type(&actions, ActionType::RemoveNode).is_empty());
    }

    #[test]
    fn delay_samples_both_forms() {
        let mut rng = make_rng(Some(1));
        assert_eq!(Delay::Fixed(12.0).sample(&mut rng), 12.0);
        assert_eq!(
            Delay::Sampled(Distribution::constant(7.0)).sample(&mut rng),
            7.0
        );
    }
}
