//! Port of `tests/test_metrics_counters.py`.
//!
//! Checks that the collector tracks transient failures, data-loss failures,
//! spawns, unavailability incidents and leader elections, and that the
//! Monte Carlo runner aggregates them.

mod common;

use common::{cluster_with, days, hours, minutes, ConfigBuilder};

use powder_mc::monte_carlo::{
    run_monte_carlo, ConvergenceMetric, MonteCarloResults, ScenarioFactory,
};
use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::Distribution;
use powder_mc::sim::events::EventType;
use powder_mc::sim::metrics::MetricsCollector;
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::{LeaderlessProtocol, Protocol, RaftLikeProtocol};
use powder_mc::sim::simulator::{EndReason, Simulator};
use powder_mc::sim::strategy::{ClusterStrategy, NoOpStrategy, NodeReplacementStrategy};

// =============================================================================
// Helper factories
// =============================================================================

/// Never fails within a reasonable test window.
fn make_stable_config() -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build()
}

/// Fails at a fixed time and recovers after a fixed duration.
fn make_fragile_config(failure_time: f64, recovery_time: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(failure_time))
        .recovery(Distribution::constant(recovery_time))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build()
}

/// Loses data permanently at a fixed time.
fn make_data_loss_config(data_loss_time: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(data_loss_time))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

/// Two fragile nodes failing together plus one stable node.
fn two_fragile_one_stable(failure_time: f64, recovery_time: f64) -> ClusterState {
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", make_fragile_config(failure_time, recovery_time));
    cluster.add_named_node("node1", make_fragile_config(failure_time, recovery_time));
    cluster.add_named_node("node2", make_stable_config());
    cluster
}

/// One fragile node plus two stable ones.
fn one_fragile_two_stable(failure_time: f64, recovery_time: f64) -> ClusterState {
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", make_fragile_config(failure_time, recovery_time));
    cluster.add_named_node("node1", make_stable_config());
    cluster.add_named_node("node2", make_stable_config());
    cluster
}

// =============================================================================
// MetricsCollector unit tests
// =============================================================================

#[test]
fn test_record_event_transient_failure() {
    let mut mc = MetricsCollector::new();
    mc.record_event(EventType::NodeFailure);
    mc.record_event(EventType::NodeFailure);
    assert_eq!(mc.total_transient_failures, 2);
}

#[test]
fn test_record_event_data_loss() {
    let mut mc = MetricsCollector::new();
    mc.record_event(EventType::NodeDataLoss);
    assert_eq!(mc.total_dataloss_failures, 1);
}

#[test]
fn test_record_event_spawn_complete() {
    let mut mc = MetricsCollector::new();
    for _ in 0..3 {
        mc.record_event(EventType::NodeSpawnComplete);
    }
    assert_eq!(mc.total_nodes_spawned, 3);
}

#[test]
fn test_record_event_other_types_ignored() {
    let mut mc = MetricsCollector::new();
    mc.record_event(EventType::NodeRecovery);
    mc.record_event(EventType::NetworkOutageStart);
    mc.record_event(EventType::LeaderElectionComplete);
    assert_eq!(mc.total_transient_failures, 0);
    assert_eq!(mc.total_dataloss_failures, 0);
    assert_eq!(mc.total_nodes_spawned, 0);
}

#[test]
fn test_unavailability_transition_basic() {
    let mut mc = MetricsCollector::new();
    // The collector starts out presuming availability.
    mc.record_unavailability_transition(false, 100.0);
    assert_eq!(mc.total_unavailability_incidents, 1);
    assert_eq!(mc.time_to_first_unavailability, Some(100.0));
}

#[test]
fn test_unavailability_transition_multiple() {
    let mut mc = MetricsCollector::new();
    mc.record_unavailability_transition(false, 10.0); // available -> not: +1
    mc.record_unavailability_transition(false, 20.0); // still down: no change
    mc.record_unavailability_transition(true, 30.0); // back up: no change
    mc.record_unavailability_transition(false, 40.0); // down again: +1
    assert_eq!(mc.total_unavailability_incidents, 2);
    assert_eq!(mc.time_to_first_unavailability, Some(10.0));
}

#[test]
fn test_unavailability_transition_stays_available() {
    let mut mc = MetricsCollector::new();
    mc.record_unavailability_transition(true, 10.0);
    mc.record_unavailability_transition(true, 20.0);
    assert_eq!(mc.total_unavailability_incidents, 0);
    assert_eq!(mc.time_to_first_unavailability, None);
}

#[test]
fn test_leader_election_counter() {
    let mut mc = MetricsCollector::new();
    mc.record_leader_election();
    mc.record_leader_election();
    assert_eq!(mc.total_leader_elections, 2);
}

#[test]
fn test_snapshot_includes_all_counters() {
    let mut mc = MetricsCollector::new();
    mc.total_transient_failures = 5;
    mc.total_dataloss_failures = 2;
    mc.total_nodes_spawned = 3;
    mc.total_unavailability_incidents = 4;
    mc.total_leader_elections = 1;

    let snap = mc.snapshot();
    assert_eq!(snap.total_transient_failures, 5);
    assert_eq!(snap.total_dataloss_failures, 2);
    assert_eq!(snap.total_nodes_spawned, 3);
    assert_eq!(snap.total_unavailability_incidents, 4);
    assert_eq!(snap.total_leader_elections, 1);
}

// =============================================================================
// Simulator integration
// =============================================================================

#[test]
fn test_transient_failure_counter() {
    let mut sim = Simulator::new(
        two_fragile_one_stable(hours(2.0), hours(1.0)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        false,
    );
    let result = sim.run_for(hours(2.5));

    assert_eq!(result.metrics.total_transient_failures, 2);
    assert_eq!(result.metrics.total_dataloss_failures, 0);
}

#[test]
fn test_dataloss_failure_counter() {
    let mut cluster = ClusterState::new(3);
    for (i, hour) in [1.0, 2.0, 3.0].iter().enumerate() {
        cluster.add_named_node(&format!("node{i}"), make_data_loss_config(hours(*hour)));
    }

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        false,
    );
    let result = sim.run_for(hours(4.0));

    // The run stops at hour 3, when the last current node loses its data.
    assert_eq!(result.end_reason, EndReason::DataLoss);
    assert_eq!(result.metrics.total_dataloss_failures, 3);
    assert_eq!(result.metrics.total_transient_failures, 0);
}

#[test]
fn test_unavailability_incident_counter() {
    let mut sim = Simulator::new(
        two_fragile_one_stable(hours(2.0), hours(1.0)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        false,
    );
    let result = sim.run_for(hours(4.0));

    assert!(result.metrics.total_unavailability_incidents >= 1);
    assert!((result.metrics.availability_fraction() - 0.75).abs() < 1e-9);
}

#[test]
fn test_nodes_spawned_counter() {
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", make_data_loss_config(hours(1.0)));
    cluster.add_named_node("node1", make_stable_config());
    cluster.add_named_node("node2", make_stable_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NodeReplacementStrategy::new(
            60.0,
            Some(make_stable_config()),
            true,
        )),
        Box::new(LeaderlessProtocol::majority_available(1.0)),
        None,
        Some(42),
        false,
    );
    let result = sim.run_for(hours(3.0));

    // node0 loses data at hour 1, the timeout fires, and a replacement
    // finishes spawning.
    assert!(result.metrics.total_nodes_spawned >= 1);
    assert!(result.metrics.total_dataloss_failures >= 1);
}

#[test]
fn test_leader_election_counter_raft() {
    let mut sim = Simulator::new(
        one_fragile_two_stable(hours(1.0), hours(0.5)),
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(minutes(1.0)))),
        None,
        Some(0),
        false,
    );
    let result = sim.run_for(hours(2.0));

    // node0 leads from the start and fails at hour 1, forcing an election.
    assert!(result.metrics.total_leader_elections >= 1);
}

#[test]
fn test_leader_election_counter_leaderless_is_zero() {
    let mut sim = Simulator::new(
        one_fragile_two_stable(hours(1.0), hours(0.5)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        false,
    );
    let result = sim.run_for(hours(2.0));
    assert_eq!(result.metrics.total_leader_elections, 0);
}

#[test]
fn test_counters_in_snapshot_match_collector() {
    let mut sim = Simulator::new(
        one_fragile_two_stable(hours(1.0), hours(0.5)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        false,
    );
    let result = sim.run_for(hours(2.0));

    // The snapshot is a copy of the live collector, so the two agree.
    let live = sim.metrics.snapshot();
    assert_eq!(result.metrics, live);
}

// =============================================================================
// Monte Carlo aggregation
// =============================================================================

#[test]
fn test_counters_propagated_to_results() {
    let cluster = || one_fragile_two_stable(hours(2.0), hours(1.0));
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
    let scenario = ScenarioFactory {
        cluster: &cluster,
        strategy: &strategy,
        protocol: &protocol,
        network_config: None,
    };

    let results = run_monte_carlo(&scenario, 5, Some(hours(4.0)), false, Some(42)).unwrap();

    assert_eq!(results.transient_failure_samples.len(), 5);
    assert_eq!(results.dataloss_failure_samples.len(), 5);
    assert_eq!(results.nodes_spawned_samples.len(), 5);
    assert_eq!(results.unavailability_incident_samples.len(), 5);
    assert_eq!(results.leader_election_samples.len(), 5);

    // node0 always fails inside the window.
    assert!(results.transient_failure_samples.iter().all(|&s| s >= 1));
    assert!(results.dataloss_failure_samples.iter().all(|&s| s == 0));
    // A leaderless protocol never holds an election.
    assert!(results.leader_election_samples.iter().all(|&s| s == 0));
}

#[test]
fn test_summary_includes_counters() {
    let mut results = MonteCarloResults::new();
    results.availability_samples = vec![0.99, 0.98];
    results.cost_samples = vec![10.0, 11.0];
    results.time_to_potential_loss_samples = vec![None, None];
    results.time_to_actual_loss_samples = vec![None, None];
    results.end_reasons = vec![EndReason::TimeLimit, EndReason::TimeLimit];
    results.transient_failure_samples = vec![5, 3];
    results.dataloss_failure_samples = vec![0, 1];
    results.nodes_spawned_samples = vec![2, 1];
    results.unavailability_incident_samples = vec![1, 2];
    results.leader_election_samples = vec![3, 4];

    let summary = results.summary();
    assert!(summary.contains("Transient failures"));
    assert!(summary.contains("Dataloss failures"));
    assert!(summary.contains("Nodes spawned"));
    assert!(summary.contains("Unavailability incidents"));
    assert!(summary.contains("Leader elections"));
}

#[test]
fn test_convergence_metric_samples_extraction() {
    let mut results = MonteCarloResults::new();
    results.transient_failure_samples = vec![5, 3, 4];
    results.dataloss_failure_samples = vec![0, 1, 0];
    results.nodes_spawned_samples = vec![2, 1, 3];
    results.unavailability_incident_samples = vec![1, 2, 1];
    results.leader_election_samples = vec![3, 4, 2];

    assert_eq!(
        results.metric_samples(ConvergenceMetric::TransientFailures),
        vec![5.0, 3.0, 4.0]
    );
    assert_eq!(
        results.metric_samples(ConvergenceMetric::DatalossFailures),
        vec![0.0, 1.0, 0.0]
    );
    assert_eq!(
        results.metric_samples(ConvergenceMetric::NodesSpawned),
        vec![2.0, 1.0, 3.0]
    );
    assert_eq!(
        results.metric_samples(ConvergenceMetric::UnavailabilityIncidents),
        vec![1.0, 2.0, 1.0]
    );
    assert_eq!(
        results.metric_samples(ConvergenceMetric::LeaderElections),
        vec![3.0, 4.0, 2.0]
    );
}

/// Not in the Python suite: the counters a run reports should equal what a
/// direct event-log tally finds, which guards the dispatch wiring.
#[test]
fn counters_agree_with_the_event_log() {
    let mut sim = Simulator::new(
        cluster_with(3, make_fragile_config(hours(1.0), minutes(30.0))),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::majority_available(1.0)),
        None,
        Some(3),
        true,
    );
    let result = sim.run_for(hours(24.0));

    let count = |t: EventType| {
        result
            .event_log
            .iter()
            .filter(|e| e.event_type == t)
            .count() as u64
    };

    assert_eq!(
        result.metrics.total_transient_failures,
        count(EventType::NodeFailure)
    );
    assert_eq!(
        result.metrics.total_dataloss_failures,
        count(EventType::NodeDataLoss)
    );
    assert_eq!(
        result.metrics.total_nodes_spawned,
        count(EventType::NodeSpawnComplete)
    );
}
