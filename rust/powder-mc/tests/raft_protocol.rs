//! Port of `tests/test_raft_protocol.py`.
//!
//! Covers leader election (candidate failure, stall and restart, quorum
//! loss) and recovery semantics (snapshot versus log-only sync, syncing
//! during an election, leader failure mid-sync).

mod common;

use common::{cluster_with, days, hours, minutes, ConfigBuilder};

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::{make_rng, Distribution};
use powder_mc::sim::events::{Event, EventType};
use powder_mc::sim::ids::Sym;
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::{collect_events, collect_start_events, Protocol, RaftLikeProtocol};
use powder_mc::sim::simulator::{EndReason, SimulationResult, Simulator};
use powder_mc::sim::strategy::NoOpStrategy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Never fails inside any reasonable test window.
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

/// Fails once at `fail_after`, recovers `recovery_time` later, then never
/// fails again inside the window.
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

/// Never fails on its own, and takes `recovery_time` to come back when a
/// failure is injected by hand.
fn injected_failure_config(recovery_time: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(recovery_time))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

fn election_events(result: &SimulationResult) -> Vec<&Event> {
    result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::LeaderElectionComplete)
        .collect()
}

fn sync_events_for(result: &SimulationResult, target: Sym) -> Vec<&Event> {
    result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeSyncComplete && e.target_id == target)
        .collect()
}

/// Replace a node's scheduled failure with one at a chosen time, the way
/// Python's tests reach into the queue after `_initialize()`.
fn inject_failure(sim: &mut Simulator, node: &str, at: f64) {
    let sym = sim.cluster.sym_of(node).expect("node must exist");
    sim.event_queue
        .cancel_events_for(sym, EventType::NodeFailure);
    sim.event_queue
        .push(Event::new(at, EventType::NodeFailure, sym));
}

/// The protocol as a concrete Raft handle, for asserting on election state.
fn raft(sim: &Simulator) -> &RaftLikeProtocol {
    sim.protocol
        .as_raft()
        .expect("the simulator was built with a Raft protocol")
}

// ===========================================================================
// Raft Leader Election
// ===========================================================================

#[test]
fn test_basic_leader_election_on_failure() {
    // node0 leads and fails at 1 h; the other two never fail, so one of
    // them takes over after a 10 s election.
    let election_time = 10.0;
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node(
        "node0",
        single_failure_config(hours(1.0), hours(1.0)),
    );
    cluster.add_named_node("node1", never_fail_config());
    cluster.add_named_node("node2", never_fail_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(2.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let elections = election_events(&result);
    assert_eq!(elections.len(), 1, "exactly one election should complete");
    assert!((elections[0].time - (hours(1.0) + election_time)).abs() < 1e-9);

    // node1 is the first eligible node in name order.
    assert_eq!(
        sim.cluster.name_of(sim.protocol.leader_id().unwrap()),
        "node1"
    );
    assert!(!raft(&sim).election_in_progress());

    // The only downtime is the election itself.
    assert!((result.metrics.time_unavailable - election_time).abs() < 1e-9);
}

#[test]
fn test_candidate_fails_during_election() {
    // t=100 the leader fails and a 50 s election starts.
    // t=120 node1 fails, t=130 node2 fails, so at t=150 nobody is eligible
    // and the election stalls.
    // t=180 node2 returns and restarts it; t=230 it completes.
    let election_time = 50.0;
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", single_failure_config(100.0, 500.0));
    cluster.add_named_node("node1", single_failure_config(120.0, 70.0));
    cluster.add_named_node("node2", single_failure_config(130.0, 50.0));

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );
    let result = sim.run_until(Some(300.0), None);

    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let elections = election_events(&result);
    assert_eq!(
        elections.len(),
        2,
        "expected a stalled attempt plus a restart"
    );
    assert!((elections[0].time - 150.0).abs() < 1e-9);
    assert!((elections[1].time - 230.0).abs() < 1e-9);

    assert_eq!(
        sim.cluster.name_of(sim.protocol.leader_id().unwrap()),
        "node1"
    );
    assert!(!raft(&sim).election_in_progress());

    // Unavailable from the leader failing at t=100 to the election
    // completing at t=230.
    assert!((result.metrics.time_unavailable - 130.0).abs() < 1e-9);
}

#[test]
fn test_all_nodes_fail_election_stalls_then_restarts() {
    // Everything goes down, then comes back one node at a time.  A single
    // recovered node cannot win a 3-node election, so the restart only
    // succeeds once a majority is back.
    let election_time = 50.0;
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", single_failure_config(100.0, 900.0));
    cluster.add_named_node("node1", injected_failure_config(250.0));
    cluster.add_named_node("node2", injected_failure_config(200.0));

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );

    sim.initialize();
    inject_failure(&mut sim, "node1", 50.0);
    inject_failure(&mut sim, "node2", 80.0);

    let result = sim.run_until(Some(500.0), None);
    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let elections = election_events(&result);
    assert!(
        elections.len() >= 2,
        "expected at least two election events, got {}",
        elections.len()
    );

    let leader = sim
        .protocol
        .leader_id()
        .expect("a leader should be elected once quorum returns");
    assert!(!raft(&sim).election_in_progress());
    assert_eq!(sim.cluster.name_of(leader), "node1");
}

#[test]
fn test_unavailable_during_election_period() {
    // The system is unavailable for exactly the election window.
    let election_time = 100.0;
    let fail_time = 1000.0;
    let duration = 2000.0;

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", single_failure_config(fail_time, 500.0));
    cluster.add_named_node("node1", never_fail_config());
    cluster.add_named_node("node2", never_fail_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(duration);

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert!((result.metrics.time_unavailable - election_time).abs() < 1e-9);

    let expected = (duration - election_time) / duration;
    assert!((result.metrics.availability_fraction() - expected).abs() < 1e-12);

    let elections = election_events(&result);
    assert_eq!(elections.len(), 1);
    assert!((elections[0].time - (fail_time + election_time)).abs() < 1e-9);
}

#[test]
fn test_quorum_lost_during_election() {
    // Five nodes.  The leader fails at t=100 and a 100 s election starts.
    // node1 fails at t=120 (3/5, still quorum) and node2 at t=140 (2/5,
    // quorum lost), which invalidates the election immediately.  The
    // original completion fires at t=200 with a stale epoch and is ignored.
    // node1 returns at t=300, a fresh election starts, and it completes at
    // t=400.
    let election_time = 100.0;
    let mut cluster = ClusterState::new(5);
    cluster.add_named_node("node0", single_failure_config(100.0, days(9999.0)));
    cluster.add_named_node("node1", injected_failure_config(180.0));
    cluster.add_named_node("node2", injected_failure_config(days(9999.0)));
    cluster.add_named_node("node3", never_fail_config());
    cluster.add_named_node("node4", never_fail_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );

    sim.initialize();
    inject_failure(&mut sim, "node1", 120.0);
    inject_failure(&mut sim, "node2", 140.0);

    let result = sim.run_until(Some(500.0), None);
    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let elections = election_events(&result);
    assert_eq!(
        elections.len(),
        2,
        "expected a stale completion plus a fresh one, got {}",
        elections.len()
    );
    assert!((elections[0].time - 200.0).abs() < 1e-9);
    assert!((elections[1].time - 400.0).abs() < 1e-9);

    // The stale event carries the older epoch.
    assert!(elections[0].metadata.epoch().unwrap() < elections[1].metadata.epoch().unwrap());

    let leader = sim.protocol.leader_id().expect("a leader should be elected");
    assert!(!raft(&sim).election_in_progress());
    assert_eq!(sim.cluster.name_of(leader), "node1");

    // Unavailable from t=100 to t=400.
    assert!((result.metrics.time_unavailable - 300.0).abs() < 1e-9);
}

#[test]
fn test_no_election_without_majority() {
    // Only two of five are up when the election fires, so nobody can win.
    // It restarts once node2 returns and restores a majority.
    let election_time = 50.0;
    let mut cluster = ClusterState::new(5);
    cluster.add_named_node("node0", single_failure_config(100.0, days(9999.0)));
    cluster.add_named_node("node1", injected_failure_config(days(9999.0)));
    cluster.add_named_node("node2", injected_failure_config(200.0));
    cluster.add_named_node("node3", never_fail_config());
    cluster.add_named_node("node4", never_fail_config());

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );

    sim.initialize();
    inject_failure(&mut sim, "node1", 50.0);
    inject_failure(&mut sim, "node2", 60.0);

    let result = sim.run_until(Some(400.0), None);
    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let elections = election_events(&result);
    assert!(elections.len() >= 2);
    assert!(
        sim.protocol.leader_id().is_some(),
        "a leader should be elected once quorum returns"
    );
    assert!(!raft(&sim).election_in_progress());
}

#[test]
fn test_election_stall_generates_no_events() {
    // With nothing recovering, a stalled election must sit quiet rather
    // than polling.  The old polling design would have emitted hundreds of
    // events over this window.
    let election_time = 10.0;
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", single_failure_config(100.0, days(9999.0)));
    cluster.add_named_node("node1", injected_failure_config(days(9999.0)));
    cluster.add_named_node("node2", injected_failure_config(days(9999.0)));

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(election_time))),
        None,
        Some(0),
        true,
    );

    sim.initialize();
    inject_failure(&mut sim, "node1", 50.0);
    inject_failure(&mut sim, "node2", 60.0);

    let result = sim.run_until(Some(10_000.0), None);
    assert_eq!(result.end_reason, EndReason::TimeLimit);

    let elections = election_events(&result);
    assert_eq!(
        elections.len(),
        1,
        "expected exactly one (stalled) election event, got {}",
        elections.len()
    );

    assert!(sim.protocol.leader_id().is_none());
    assert!(raft(&sim).election_in_progress());
    assert!(raft(&sim).election_stalled());
}

// ===========================================================================
// Raft Snapshot and Log Recovery
// ===========================================================================

#[test]
fn test_recovery_without_snapshot() {
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(365.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(365.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(minutes(5.0)))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let mut protocol =
        RaftLikeProtocol::with_params(Distribution::constant(5.0), 1.0, 100.0, 0.0);

    let mut cluster = cluster_with(3, config);
    cluster.commit_index = 150.0;
    for i in cluster.active_indices() {
        let node = cluster.node_at_mut(i);
        node.last_applied_index = 150.0;
        node.last_snapshot_index = 100.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 110.0;

    let mut rng = make_rng(Some(42));
    collect_start_events(&mut protocol, &cluster, &mut rng);

    let index = cluster.index_of(n0).unwrap();
    let sync_time = protocol
        .compute_sync_time(index, &cluster, &mut rng)
        .expect("a donor is available");
    assert!(sync_time < minutes(5.0), "got {sync_time}");
}

#[test]
fn test_recovery_with_snapshot() {
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(365.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(365.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(60.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let mut protocol =
        RaftLikeProtocol::with_params(Distribution::constant(5.0), 1.0, 100.0, 100.0);

    let mut cluster = cluster_with(3, config);
    cluster.commit_index = 250.0;
    for i in cluster.active_indices() {
        let node = cluster.node_at_mut(i);
        node.last_applied_index = 250.0;
        node.last_snapshot_index = 200.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();
    {
        let node = cluster.get_node_mut(n0).unwrap();
        node.last_applied_index = 50.0;
        node.last_snapshot_index = 0.0;
    }

    let mut rng = make_rng(Some(42));
    collect_start_events(&mut protocol, &cluster, &mut rng);

    let index = cluster.index_of(n0).unwrap();
    let sync_time = protocol
        .compute_sync_time(index, &cluster, &mut rng)
        .expect("a donor is available");
    assert!(sync_time >= 60.0, "got {sync_time}");
}

#[test]
fn test_no_snapshot_interval_means_log_only() {
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(365.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(365.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(60.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let mut protocol =
        RaftLikeProtocol::with_params(Distribution::constant(5.0), 1.0, 0.0, 0.0);

    let mut cluster = cluster_with(3, config);
    cluster.commit_index = 1000.0;
    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 1000.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 0.0;

    let mut rng = make_rng(Some(42));
    collect_start_events(&mut protocol, &cluster, &mut rng);

    let index = cluster.index_of(n0).unwrap();
    let sync_time = protocol
        .compute_sync_time(index, &cluster, &mut rng)
        .expect("a donor is available");
    assert!(sync_time < 120.0, "got {sync_time}");
}

#[test]
fn test_quorum_loss_freezes_commit_index() {
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(10.0))
        .recovery(Distribution::constant(100.0))
        .data_loss(Distribution::constant(days(365.0)))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(days(365.0)))
        .build();

    let mut sim = Simulator::new(
        cluster_with(3, config),
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::with_params(
            Distribution::constant(5.0),
            1.0,
            0.0,
            0.0,
        )),
        None,
        Some(42),
        true,
    );
    sim.run_for(200.0);

    assert!(sim.cluster.commit_index < 200.0);
    assert!(sim.cluster.commit_index > 0.0);
}

#[test]
fn test_node_snapshot_state_advances() {
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::exponential(1.0 / hours(24.0)).unwrap())
        .recovery(Distribution::constant(minutes(5.0)))
        .data_loss(Distribution::exponential(1.0 / days(365.0)).unwrap())
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let mut sim = Simulator::new(
        cluster_with(3, config),
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::with_params(
            Distribution::constant(5.0),
            1.0,
            50.0,
            0.0,
        )),
        None,
        Some(42),
        false,
    );
    sim.run_for(200.0);

    for node in sim.cluster.active() {
        if node.is_available && node.has_data {
            assert!(node.last_snapshot_index > 0.0);
            assert!((node.last_snapshot_index % 50.0).abs() < 1e-9);
        }
    }
}

// ===========================================================================
// Raft Sync Model
// ===========================================================================

#[test]
fn test_sync_during_election() {
    // While an election runs the cluster cannot commit, so the donor's
    // position is frozen and the gap closes at the full replay rate:
    // 2.0 instead of 2.0 - 1.0.
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut protocol =
        RaftLikeProtocol::with_params(Distribution::constant(5.0), 1.0, 0.0, 0.0);

    let mut cluster = cluster_with(3, config);
    cluster.commit_index = 10.0;
    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 10.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 0.0;
    let index = cluster.index_of(n0).unwrap();

    let mut rng = make_rng(Some(42));

    // With a leader the net rate is 1.0, so the 10-unit gap takes 10 s.
    collect_start_events(&mut protocol, &cluster, &mut rng);
    assert!(protocol.leader_id().is_some());
    assert!(protocol.can_commit(&cluster));

    let sync_with_leader = protocol
        .compute_sync_time(index, &cluster, &mut rng)
        .expect("a donor is available");
    assert!((sync_with_leader - 10.0).abs() < 0.01, "got {sync_with_leader}");

    // Mid-election the net rate is 2.0, so the same gap takes 5 s.
    protocol.force_election_state(true, false);
    protocol.set_leader(None);
    assert!(!protocol.can_commit(&cluster));

    let sync_during_election = protocol
        .compute_sync_time(index, &cluster, &mut rng)
        .expect("a donor is available");
    assert!(
        (sync_during_election - 5.0).abs() < 0.01,
        "got {sync_during_election}"
    );
    assert!(sync_during_election < sync_with_leader);
}

#[test]
fn test_leader_failure_during_sync() {
    // node0 lags 20 with a net rate of 2.0, so 10 s of work.  The leader
    // fails at t=3, and during the election the net rate rises to 3.0, so
    // the sync finishes early -- under 10 s.
    let stable = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();
    let leader = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(3.0))
        .recovery(Distribution::constant(20.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", stable.clone());
    cluster.add_named_node("node1", leader);
    cluster.add_named_node("node2", stable);
    cluster.commit_index = 20.0;
    let indices = cluster.active_indices();
    cluster.node_at_mut(indices[0]).last_applied_index = 0.0;
    cluster.node_at_mut(indices[1]).last_applied_index = 20.0;
    cluster.node_at_mut(indices[2]).last_applied_index = 20.0;
    let n0 = cluster.sym_of("node0").unwrap();

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::with_params(
            Distribution::constant(5.0),
            1.0,
            0.0,
            0.0,
        )),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(15.0);

    let syncs = sync_events_for(&result, n0);
    assert!(!syncs.is_empty(), "node0 should have completed a sync");
    assert!(syncs[0].time < 10.0, "sync finished at {}", syncs[0].time);
    assert!(sim.cluster.get_node(n0).unwrap().sync.is_none());
}

#[test]
fn test_gc_forced_snapshot_with_leader() {
    // The same forced-snapshot scenario as the leaderless case, to confirm
    // the leader-based protocol shares the sync logic: 60 s download plus a
    // 110 unit suffix at a net 9/s, about 72.2 s.
    let cfg = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(60.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut cluster = cluster_with(3, cfg);
    cluster.commit_index = 250.0;
    let indices = cluster.active_indices();
    for &i in &indices {
        let node = cluster.node_at_mut(i);
        node.last_applied_index = 250.0;
        node.last_snapshot_index = 200.0;
    }
    {
        let node = cluster.node_at_mut(indices[0]);
        node.last_applied_index = 50.0;
        node.last_snapshot_index = 0.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::with_params(
            Distribution::constant(5.0),
            1.0,
            100.0,
            100.0,
        )),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(80.0);

    let syncs = sync_events_for(&result, n0);
    assert!(!syncs.is_empty());
    assert!(syncs[0].time >= 60.0, "sync finished at {}", syncs[0].time);
    assert!(
        (syncs[0].time - 72.2).abs() < 1.0,
        "sync finished at {}",
        syncs[0].time
    );

    let node0 = sim.cluster.get_node(n0).unwrap();
    assert!(node0.sync.is_none());
    assert!(node0.last_applied_index >= sim.cluster.commit_index - 0.1);
}

// ===========================================================================
// Raft protocol unit tests from test_simulation.py
// ===========================================================================

fn make_test_cluster(num_nodes: usize) -> ClusterState {
    let mut cluster = ClusterState::new(num_nodes);
    for i in 0..num_nodes {
        let config = ConfigBuilder::new()
            .region(&format!("region-{}", i % 3))
            .cost(1.0)
            .failure(Distribution::exponential(1.0 / hours(24.0)).unwrap())
            .recovery(Distribution::constant(minutes(5.0)))
            .data_loss(Distribution::exponential(1.0 / days(365.0)).unwrap())
            .log_replay_rate(Distribution::constant(2.0))
            .snapshot_download(Distribution::constant(0.0))
            .spawn(Distribution::constant(minutes(10.0)))
            .build();
        cluster.add_named_node(&format!("node{i}"), config);
    }
    cluster
}

#[test]
fn test_initial_leader_selection() {
    let cluster = make_test_cluster(3);
    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    let events = collect_start_events(&mut protocol, &cluster, &mut rng);

    let leader = protocol.leader_id().expect("a leader should be chosen");
    assert!(cluster.get_node(leader).is_some());
    assert!(!protocol.election_in_progress());
    assert!(events.is_empty());
}

#[test]
fn test_leader_failure_triggers_election() {
    let mut cluster = make_test_cluster(3);
    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    collect_start_events(&mut protocol, &cluster, &mut rng);
    let leader_id = protocol.leader_id().unwrap();

    cluster.get_node_mut(leader_id).unwrap().is_available = false;
    cluster.current_time = 100.0;

    let new_events = collect_events(&mut protocol, 
        &Event::new(100.0, EventType::NodeFailure, leader_id),
        &cluster,
        &mut rng,
    );

    assert!(protocol.election_in_progress());
    assert!(protocol.leader_id().is_none());
    assert_eq!(new_events.len(), 1);
    assert_eq!(
        new_events[0].event_type,
        EventType::LeaderElectionComplete
    );
    assert_eq!(new_events[0].time, 110.0);
}

#[test]
fn test_unavailable_during_election() {
    let mut cluster = make_test_cluster(3);
    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    collect_start_events(&mut protocol, &cluster, &mut rng);
    assert!(protocol.can_commit(&cluster));

    let leader_id = protocol.leader_id().unwrap();
    cluster.get_node_mut(leader_id).unwrap().is_available = false;
    cluster.current_time = 100.0;
    collect_events(&mut protocol, 
        &Event::new(100.0, EventType::NodeFailure, leader_id),
        &cluster,
        &mut rng,
    );

    assert!(!protocol.can_commit(&cluster));
}

#[test]
fn test_election_completes_with_new_leader() {
    let mut cluster = make_test_cluster(3);
    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    collect_start_events(&mut protocol, &cluster, &mut rng);
    let old_leader_id = protocol.leader_id().unwrap();

    cluster.current_time = 100.0;
    cluster.commit_index = 100.0;
    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 100.0;
    }
    cluster.get_node_mut(old_leader_id).unwrap().is_available = false;

    let new_events = collect_events(&mut protocol, 
        &Event::new(100.0, EventType::NodeFailure, old_leader_id),
        &cluster,
        &mut rng,
    );

    let election_event = new_events[0].clone();
    cluster.current_time = election_event.time;
    cluster.commit_index = 110.0;
    for i in cluster.active_indices() {
        if cluster.node_at(i).is_available {
            cluster.node_at_mut(i).last_applied_index = 110.0;
        }
    }

    let result_events = collect_events(&mut protocol, &election_event, &cluster, &mut rng);

    assert!(!protocol.election_in_progress());
    let new_leader = protocol.leader_id().expect("a new leader");
    assert_ne!(new_leader, old_leader_id);
    assert!(protocol.can_commit(&cluster));
    assert!(result_events.is_empty());
}

#[test]
fn test_election_retries_when_no_eligible_node() {
    let mut cluster = make_test_cluster(3);
    cluster.current_time = 100.0;
    cluster.commit_index = 100.0;
    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 0.0;
    }

    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    // Start mid-election, as Python does by setting the private flag.
    protocol.force_election_state(true, false);

    let election_event = Event::new(
        100.0,
        EventType::LeaderElectionComplete,
        powder_mc::sim::ids::SYM_PROTOCOL,
    );
    let new_events = collect_events(&mut protocol, &election_event, &cluster, &mut rng);

    // Nobody is current, so the election stalls rather than retrying.
    assert!(protocol.election_in_progress());
    assert!(protocol.leader_id().is_none());
    assert!(new_events.is_empty());
    assert!(protocol.election_stalled());

    // A recovery with an eligible node restarts it.
    let first = cluster.active_indices()[0];
    cluster.node_at_mut(first).last_applied_index = 100.0;
    let first_id = cluster.node_at(first).node_id;
    cluster.current_time = 200.0;

    let restart_events = collect_events(&mut protocol, 
        &Event::new(200.0, EventType::NodeRecovery, first_id),
        &cluster,
        &mut rng,
    );

    assert_eq!(restart_events.len(), 1);
    assert_eq!(
        restart_events[0].event_type,
        EventType::LeaderElectionComplete
    );
    assert!(!protocol.election_stalled());
}

#[test]
fn test_network_outage_on_leader_region_triggers_election() {
    let mut cluster = make_test_cluster(3);
    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    collect_start_events(&mut protocol, &cluster, &mut rng);
    let leader_id = protocol.leader_id().unwrap();
    let leader_region = cluster.get_node(leader_id).unwrap().region;

    cluster.network.add_outage(leader_region);
    cluster.current_time = 50.0;

    let outage_event = Event::with_meta(
        50.0,
        EventType::NetworkOutageStart,
        leader_region,
        powder_mc::sim::events::EventMeta::Region(leader_region),
    );
    let new_events = collect_events(&mut protocol, &outage_event, &cluster, &mut rng);

    assert!(protocol.election_in_progress());
    assert!(protocol.leader_id().is_none());
    assert_eq!(new_events.len(), 1);
    assert_eq!(
        new_events[0].event_type,
        EventType::LeaderElectionComplete
    );
}

#[test]
fn test_non_leader_failure_no_election() {
    let mut cluster = make_test_cluster(3);
    let mut protocol = RaftLikeProtocol::new(Distribution::constant(10.0));
    let mut rng = make_rng(Some(42));

    collect_start_events(&mut protocol, &cluster, &mut rng);
    let leader_id = protocol.leader_id().unwrap();

    let failed_id = cluster
        .active()
        .map(|n| n.node_id)
        .find(|&id| id != leader_id)
        .unwrap();
    cluster.get_node_mut(failed_id).unwrap().is_available = false;
    cluster.current_time = 100.0;

    let new_events = collect_events(&mut protocol, 
        &Event::new(100.0, EventType::NodeFailure, failed_id),
        &cluster,
        &mut rng,
    );

    assert!(!protocol.election_in_progress());
    assert_eq!(protocol.leader_id(), Some(leader_id));
    assert!(new_events.is_empty());
}

#[test]
fn test_full_simulation_with_raft_protocol() {
    let mut sim = Simulator::new(
        make_test_cluster(3),
        Box::new(NoOpStrategy),
        Box::new(RaftLikeProtocol::new(Distribution::constant(5.0))),
        None,
        Some(42),
        false,
    );
    let result = sim.run_for(1000.0);

    assert_eq!(result.end_time, 1000.0);
    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert_eq!(result.metrics.total_time(), 1000.0);
}

#[test]
fn test_raft_less_available_than_leaderless() {
    let config = || {
        ConfigBuilder::new()
            .cost(1.0)
            .failure(Distribution::exponential(1.0 / hours(12.0)).unwrap())
            .recovery(Distribution::constant(minutes(5.0)))
            .data_loss(Distribution::exponential(1.0 / days(3650.0)).unwrap())
            .log_replay_rate(Distribution::constant(2.0))
            .snapshot_download(Distribution::constant(0.0))
            .spawn(Distribution::constant(minutes(10.0)))
            .build()
    };

    let run = |protocol: Box<dyn Protocol>| {
        let mut sim = Simulator::new(
            cluster_with(5, config()),
            Box::new(NoOpStrategy),
            protocol,
            None,
            Some(42),
            false,
        );
        sim.run_for(days(30.0)).metrics.availability_fraction()
    };

    let leaderless = run(Box::<powder_mc::sim::protocol::LeaderlessProtocol>::default());
    let raft = run(Box::new(RaftLikeProtocol::new(Distribution::constant(
        minutes(2.0),
    ))));

    assert!(
        raft <= leaderless,
        "election downtime should cost availability: raft {raft} > leaderless {leaderless}"
    );
}

#[test]
fn test_raft_commit_rate_and_snapshot_interval() {
    let protocol =
        RaftLikeProtocol::with_params(Distribution::constant(5.0), 0.5, 1000.0, 0.0);
    assert_eq!(protocol.commit_rate(), 0.5);
    assert_eq!(protocol.snapshot_interval(), 1000.0);
}
