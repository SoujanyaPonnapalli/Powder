//! Port of `tests/test_strategy_refactor.py` and
//! `tests/test_simple_strategy_scaling.py`.
//!
//! Both Python files define a `DummyProtocol` inline to isolate the
//! strategies from real quorum semantics; the port does the same, which
//! also exercises the `Protocol` trait's default methods.

mod common;

use common::ConfigBuilder;

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::{make_rng, Distribution, Rng};
use powder_mc::sim::events::{Event, EventType};
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::Protocol;
use powder_mc::sim::simulator::Simulator;
use powder_mc::sim::strategy::{
    collect_actions, Action, ActionType, AdaptiveReplacementStrategy, Delay, NodeReplacementStrategy,
};

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

/// Always committable, never loses data.  Used by the replacement-timeout
/// tests so the strategy is the only thing under test.
struct AlwaysCommitProtocol;

impl Protocol for AlwaysCommitProtocol {
    fn can_commit(&self, _cluster: &ClusterState) -> bool {
        true
    }

    fn on_event(
        &mut self,
        _event: &Event,
        _cluster: &ClusterState,
        _rng: &mut Rng,
        _out: &mut Vec<Event>,
    ) {
    }

    fn has_potential_data_loss(&self, _cluster: &ClusterState) -> bool {
        false
    }

    fn has_actual_data_loss(&self, _cluster: &ClusterState) -> bool {
        false
    }
}

/// Commits on a simple majority of the *target* size, never loses data.
/// Used by the scaling tests, where the target is what moves.
struct TargetMajorityProtocol;

impl Protocol for TargetMajorityProtocol {
    fn can_commit(&self, cluster: &ClusterState) -> bool {
        let quorum = cluster.target_cluster_size / 2 + 1;
        cluster.num_available() >= quorum
    }

    fn on_event(
        &mut self,
        _event: &Event,
        _cluster: &ClusterState,
        _rng: &mut Rng,
        _out: &mut Vec<Event>,
    ) {
    }

    fn has_potential_data_loss(&self, _cluster: &ClusterState) -> bool {
        false
    }

    fn has_actual_data_loss(&self, _cluster: &ClusterState) -> bool {
        false
    }
}

fn make_test_node_config() -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east-1")
        .cost(1.0)
        .failure(Distribution::constant(1000.0))
        .recovery(Distribution::constant(10.0))
        .data_loss(Distribution::constant(10_000.0))
        .log_replay_rate(Distribution::constant(1000.0))
        .snapshot_download(Distribution::constant(10.0))
        .spawn(Distribution::constant(10.0))
        .build()
}

/// `n` healthy nodes named `node_0..node_{n-1}`, with the given target.
fn cluster_of(n: usize, target: usize) -> ClusterState {
    let mut cluster = ClusterState::new(target);
    let config = make_test_node_config();
    for i in 0..n {
        cluster.add_named_node(&format!("node_{i}"), config.clone());
    }
    cluster
}

fn push_failure(sim: &mut Simulator, node: &str, at: f64) {
    let sym = sim.cluster.sym_of(node).expect("node must exist");
    sim.event_queue
        .push(Event::new(at, EventType::NodeFailure, sym));
}

// ===========================================================================
// Replacement timing and cleanup
// ===========================================================================

#[test]
fn test_data_loss_timeout() {
    // Data loss should trigger a replacement only after the failure
    // timeout: 50 s to decide plus 10 s to spawn.
    let mut sim = Simulator::new(
        cluster_of(3, 3),
        Box::new(NodeReplacementStrategy::new(50.0, None, false)),
        Box::new(AlwaysCommitProtocol),
        None,
        None,
        true,
    );
    sim.initialize();

    let data_loss_time = 100.0;
    let node0 = sim.cluster.sym_of("node_0").unwrap();
    sim.event_queue.push(Event::new(
        data_loss_time,
        EventType::NodeDataLoss,
        node0,
    ));

    sim.run_until(Some(200.0), None);

    let events = &sim.event_log;
    let data_loss_event = events
        .iter()
        .find(|e| e.event_type == EventType::NodeDataLoss)
        .expect("data loss event not found");
    assert_eq!(data_loss_event.time, data_loss_time);

    let spawn_event = events
        .iter()
        .find(|e| e.event_type == EventType::NodeSpawnComplete)
        .expect("replacement was not spawned");

    let expected_spawn_time = data_loss_time + 50.0 + 10.0;
    assert!(
        (spawn_event.time - expected_spawn_time).abs() <= 1.0,
        "spawn at {} vs expected {expected_spawn_time}",
        spawn_event.time
    );
    // And definitely not before the timeout elapsed.
    assert!(spawn_event.time >= data_loss_time + 50.0);
}

#[test]
fn test_zombie_cleanup() {
    // Four healthy nodes against a target of three: the extra one goes.
    let mut cluster = cluster_of(4, 3);
    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 100.0;
    }

    let mut strategy = NodeReplacementStrategy::new(50.0, None, false);
    let protocol = AlwaysCommitProtocol;
    let mut rng = make_rng(Some(42));

    // A recovery is one of the events that prompts a size check.
    let node3 = cluster.sym_of("node_3").unwrap();
    let actions = collect_actions(&mut strategy, 
        &Event::new(10.0, EventType::NodeRecovery, node3),
        &cluster,
        &mut rng,
        &protocol,
    );

    let removals: Vec<&Action> = actions
        .iter()
        .filter(|a| a.action_type() == ActionType::RemoveNode)
        .collect();
    assert_eq!(removals.len(), 1, "exactly one node should be removed");
}

// ===========================================================================
// Adaptive scaling
// ===========================================================================

#[test]
fn test_scale_down_flow() {
    // Five nodes; two fail at t=10, meeting the threshold of two, so a
    // reconfiguration to three is scheduled 5 s later.  Both nodes recover
    // at t=20, and the cluster scales back to five.
    let reconfig_delay = 5.0;
    let mut sim = Simulator::new(
        cluster_of(5, 5),
        Box::new(AdaptiveReplacementStrategy::new(
            10.0,
            Delay::Fixed(reconfig_delay),
            2,
            false,
            Some(make_test_node_config()),
            true,
        )),
        Box::new(TargetMajorityProtocol),
        None,
        None,
        true,
    );
    sim.initialize();

    push_failure(&mut sim, "node_0", 10.0);
    push_failure(&mut sim, "node_1", 10.0);

    sim.run_until(Some(35.0), None);

    let reconfigs: Vec<&Event> = sim
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::ClusterReconfiguration)
        .collect();
    assert!(!reconfigs.is_empty(), "a reconfiguration was expected");
    // The first failure leaves one node down, below the threshold; the
    // second trips it, so the event lands at t = 10 + 5.
    assert_eq!(reconfigs[0].time, 15.0);

    let down = reconfigs
        .iter()
        .filter(|e| e.metadata.target_size() == Some(3))
        .count();
    let up = reconfigs
        .iter()
        .filter(|e| e.metadata.target_size() == Some(5))
        .count();
    assert!(down > 0, "should have scaled down");
    assert!(up > 0, "should have scaled back up");

    // Healed by the end of the run.
    assert_eq!(sim.cluster.target_cluster_size, 5);

    // The strategy keeps the physical fleet at its maximum, so the failed
    // nodes are retained and recover rather than being replaced.
    let spawns = sim
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeSpawnComplete)
        .count();
    assert_eq!(spawns, 0);
}

#[test]
fn test_scale_down_failure_no_quorum() {
    // Three of five fail, leaving two available against a majority of
    // three.  The cluster cannot commit, so the reconfiguration is
    // scheduled but never applied.
    let mut sim = Simulator::new(
        cluster_of(5, 5),
        Box::new(AdaptiveReplacementStrategy::new(
            10.0,
            Delay::Fixed(5.0),
            2,
            false,
            Some(make_test_node_config()),
            true,
        )),
        Box::new(TargetMajorityProtocol),
        None,
        None,
        true,
    );
    sim.initialize();

    for i in 0..3 {
        push_failure(&mut sim, &format!("node_{i}"), 10.0);
    }

    sim.run_until(Some(20.0), None);

    let reconfigs = sim
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::ClusterReconfiguration)
        .count();
    assert!(reconfigs > 0, "a reconfiguration should have been scheduled");

    assert_eq!(sim.cluster.target_cluster_size, 5);
}

#[test]
fn test_external_consensus_3_to_2() {
    // With an external consensus service the cluster may shrink below
    // three, so a single failure at a threshold of one takes it to two.
    let mut sim = Simulator::new(
        cluster_of(3, 3),
        Box::new(AdaptiveReplacementStrategy::new(
            10.0,
            Delay::Fixed(1.0),
            1,
            true,
            Some(make_test_node_config()),
            true,
        )),
        Box::new(TargetMajorityProtocol),
        None,
        None,
        true,
    );
    sim.initialize();

    push_failure(&mut sim, "node_0", 10.0);
    sim.run_until(Some(20.0), None);

    assert_eq!(sim.cluster.target_cluster_size, 2);
}

#[test]
fn test_scaling_cycle_3_2_1_2_3() {
    // The full round trip, one step at a time.
    let mut sim = Simulator::new(
        cluster_of(3, 3),
        Box::new(AdaptiveReplacementStrategy::new(
            10.0,
            Delay::Fixed(3.0),
            1,
            true,
            Some(make_test_node_config()),
            true,
        )),
        Box::new(TargetMajorityProtocol),
        None,
        None,
        true,
    );
    sim.initialize();

    // Phase 1: node_0 fails at t=10, so 3 -> 2 lands at t=13.
    push_failure(&mut sim, "node_0", 10.0);
    sim.run_until(Some(14.0), None);
    assert_eq!(sim.cluster.target_cluster_size, 2);

    // Phase 2: node_1 fails at t=15, so 2 -> 1 lands at t=18.
    push_failure(&mut sim, "node_1", 15.0);
    sim.run_until(Some(20.0), None);
    assert_eq!(sim.cluster.target_cluster_size, 1);

    // Phase 3: node_0 recovers at t=20, bringing availability to two, so
    // 1 -> 2 lands at t=23.
    sim.run_until(Some(24.0), None);
    assert_eq!(sim.cluster.target_cluster_size, 2);

    // Phase 4: node_1 recovers at t=25, bringing availability to three, so
    // 2 -> 3 lands at t=28.
    sim.run_until(Some(30.0), None);
    assert_eq!(sim.cluster.target_cluster_size, 3);
}

/// Not in the Python suite: a protocol defined with only the two required
/// methods must still work, which is what the trait's default methods
/// promise.
#[test]
fn a_minimal_protocol_relies_on_trait_defaults() {
    struct Minimal;
    impl Protocol for Minimal {
        fn can_commit(&self, cluster: &ClusterState) -> bool {
            cluster.num_available() >= self.quorum_size(cluster)
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

    let cluster = cluster_of(3, 3);
    let protocol = Minimal;

    // Defaults supply the quorum rule, data-loss checks, rates and the
    // absent leader.
    assert_eq!(protocol.quorum_size(&cluster), 2);
    assert!(protocol.can_commit(&cluster));
    assert!(!protocol.has_potential_data_loss(&cluster));
    assert!(!protocol.has_actual_data_loss(&cluster));
    assert_eq!(protocol.commit_rate(), 1.0);
    assert_eq!(protocol.snapshot_interval(), 0.0);
    assert_eq!(protocol.log_retention_ops(), 0.0);
    assert_eq!(protocol.leader_id(), None);
    assert!(protocol.as_raft().is_none());

    let mut sim = Simulator::new(
        cluster_of(3, 3),
        Box::new(NodeReplacementStrategy::new(50.0, None, false)),
        Box::new(Minimal),
        None,
        Some(1),
        false,
    );
    let result = sim.run_for(5000.0);
    assert_eq!(result.metrics.total_time(), 5000.0);
}
