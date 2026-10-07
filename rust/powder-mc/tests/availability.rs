//! Port of `tests/test_availability.py`.
//!
//! Each test builds a scenario whose availability can be worked out by hand,
//! runs the full simulator, and checks the result matches.

mod common;

use common::{days, hours, ConfigBuilder};

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::Distribution;
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::{LeaderlessProtocol, RaftLikeProtocol};
use powder_mc::sim::simulator::{EndReason, Simulator};
use powder_mc::sim::strategy::NoOpStrategy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Never fails or loses data inside any reasonable test window.
fn never_fail_config() -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

/// Fails once at `fail_after`, recovers `recovery_time` later, then stays up.
fn single_failure_config(fail_after: f64, recovery_time: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(fail_after))
        .recovery(Distribution::constant(recovery_time))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

/// Fails every `fail_interval` and takes `recovery_time` to come back,
/// cycling indefinitely.
fn cycling_failure_config(fail_interval: f64, recovery_time: f64) -> NodeConfigRef {
    single_failure_config(fail_interval, recovery_time)
}

/// `node0` with the given config, plus two nodes that never fail.
fn cluster_with_one_flaky(config: NodeConfigRef) -> ClusterState {
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", config);
    cluster.add_named_node("node1", never_fail_config());
    cluster.add_named_node("node2", never_fail_config());
    cluster
}

// ===========================================================================
// Leaderless availability
// ===========================================================================

#[test]
fn test_single_node_cycling_100_percent() {
    // One node cycling a day up and a day down never costs the 3-node
    // cluster its quorum of two, so availability is perfect.
    let mut sim = Simulator::new(
        cluster_with_one_flaky(cycling_failure_config(days(1.0), days(1.0))),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::majority_available(1.0)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(days(10.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert!((result.metrics.availability_fraction() - 1.0).abs() < 1e-12);
}

#[test]
fn test_overlapping_failures_80_percent() {
    // Days 0-1: all three up.
    // Days 1-2: node0 down, two up -> quorum holds.
    // Days 2-3: node0 and node1 both down, one up -> quorum lost.
    // Days 3-4: node0 back, node1 still down -> quorum holds.
    // Days 4-5: all three up.
    //
    // Four of five days available.
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", single_failure_config(days(1.0), days(2.0)));
    cluster.add_named_node("node1", single_failure_config(days(2.0), days(2.0)));
    cluster.add_named_node("node2", never_fail_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::majority_available(1.0)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(days(5.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert!((result.metrics.availability_fraction() - 0.8).abs() < 1e-12);
}

// ===========================================================================
// Raft availability
// ===========================================================================

#[test]
fn test_leader_failure_availability_hit_equals_election_time() {
    // Losing the leader costs one election, not the leader's whole recovery:
    // 10 s out of a 2 h run.
    let election_time = 10.0;
    let duration = hours(2.0);

    let mut sim = Simulator::new(
        cluster_with_one_flaky(single_failure_config(hours(1.0), hours(1.0))),
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(duration);

    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let expected = (duration - election_time) / duration;
    assert!((result.metrics.availability_fraction() - expected).abs() < 1e-12);
    assert!((result.metrics.time_unavailable - election_time).abs() < 1e-9);
}

#[test]
fn test_non_leader_failure_100_percent_availability() {
    // node0 leads and stays up; node1 fails but quorum holds, so no
    // election runs and nothing is lost.
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", never_fail_config());
    cluster.add_named_node("node1", single_failure_config(hours(1.0), hours(1.0)));
    cluster.add_named_node("node2", never_fail_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(10.0))),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(4.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert!((result.metrics.availability_fraction() - 1.0).abs() < 1e-12);
    assert_eq!(result.metrics.time_unavailable, 0.0);
}

#[test]
fn test_leader_failure_proportional_election_hit() {
    // The same shape with a far longer election: the hit tracks the
    // election, not the day-long recovery.
    let election_time = hours(1.0);
    let duration = days(10.0);

    let mut sim = Simulator::new(
        cluster_with_one_flaky(single_failure_config(days(1.0), days(1.0))),
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(duration);

    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let expected = (duration - election_time) / duration;
    assert!((result.metrics.availability_fraction() - expected).abs() < 1e-12);
    assert!((result.metrics.time_unavailable - election_time).abs() < 1e-9);
}
