//! Port of `tests/test_simulation.py` (engine sections).
//!
//! Test names match the Python originals so the mapping stays auditable.
//! The convergence and adaptive Monte Carlo sections of that file live in
//! `tests/monte_carlo_statistics.rs`, which covers the runner.

mod common;

use common::{basic_cluster, cluster_with, days, hours, minutes, ConfigBuilder};

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::{self, make_rng, Distribution};
use powder_mc::sim::events::{Event, EventQueue, EventType};
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::{LeaderlessProtocol, Protocol};
use powder_mc::sim::simulator::{EndReason, Simulator};
use powder_mc::sim::strategy::{
    collect_actions, Action, ActionType, NoOpStrategy,
    NodeReplacementStrategy,
};

// =============================================================================
// Time Unit Tests
// =============================================================================

#[test]
fn test_hours_conversion() {
    assert_eq!(distributions::hours(1.0), 3600.0);
    assert_eq!(distributions::hours(2.5), 9000.0);
}

#[test]
fn test_days_conversion() {
    assert_eq!(distributions::days(1.0), 86400.0);
    assert_eq!(distributions::days(0.5), 43200.0);
}

#[test]
fn test_minutes_conversion() {
    assert_eq!(distributions::minutes(1.0), 60.0);
    assert_eq!(distributions::minutes(30.0), 1800.0);
}

// =============================================================================
// Distribution Tests
// =============================================================================

fn sample_many(d: Distribution, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = make_rng(Some(seed));
    (0..n).map(|_| d.sample(&mut rng)).collect()
}

fn mean_of(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

#[test]
fn test_exponential_sample() {
    let samples = sample_many(Distribution::exponential(1.0).unwrap(), 1000, 42);
    let m = mean_of(&samples);
    assert!(0.8 < m && m < 1.2, "mean was {m}");
    assert!(samples.iter().all(|&s| s > 0.0));
}

#[test]
fn test_exponential_invalid_rate() {
    assert!(Distribution::exponential(0.0).is_err());
    assert!(Distribution::exponential(-1.0).is_err());
}

#[test]
fn test_weibull_sample() {
    let samples = sample_many(Distribution::weibull(2.0, 1.0).unwrap(), 1000, 42);
    assert!(samples.iter().all(|&s| s > 0.0));
}

#[test]
fn test_normal_sample_with_min() {
    let samples = sample_many(Distribution::normal(5.0, 2.0, 0.0).unwrap(), 1000, 42);
    assert!(samples.iter().all(|&s| s >= 0.0));
    let m = mean_of(&samples);
    assert!(4.0 < m && m < 6.0, "mean was {m}");
}

#[test]
fn test_uniform_sample() {
    let samples = sample_many(Distribution::uniform(10.0, 20.0).unwrap(), 1000, 42);
    assert!(samples.iter().all(|&s| (10.0..20.0).contains(&s)));
    let m = mean_of(&samples);
    assert!(14.5 < m && m < 15.5, "mean was {m}");
}

#[test]
fn test_constant_sample() {
    let samples = sample_many(Distribution::constant(42.0), 10, 42);
    assert!(samples.iter().all(|&s| s == 42.0));
}

// =============================================================================
// Node Tests
// =============================================================================

/// Port of Python's `make_test_node_config`.
fn make_test_node_config(region: &str) -> NodeConfigRef {
    ConfigBuilder::new()
        .region(region)
        .cost(1.0)
        // About one failure per day.
        .failure(Distribution::exponential(1.0 / hours(24.0)).unwrap())
        .recovery(Distribution::constant(minutes(5.0)))
        // About one data loss per year.
        .data_loss(Distribution::exponential(1.0 / days(365.0)).unwrap())
        // Replays twice as fast as the cluster commits.
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build()
}

/// Port of Python's `make_test_cluster`: nodes spread over three regions.
fn make_test_cluster(num_nodes: usize) -> ClusterState {
    let mut cluster = ClusterState::new(num_nodes);
    for i in 0..num_nodes {
        let config = make_test_node_config(&format!("region-{}", i % 3));
        cluster.add_named_node(&format!("node{i}"), config);
    }
    cluster
}

#[test]
fn test_is_up_to_date() {
    let mut cluster = basic_cluster(1);
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 100.0;

    let node = cluster.get_node(n0).unwrap();
    assert!(node.is_up_to_date(100.0));
    assert!(node.is_up_to_date(50.0));
    assert!(!node.is_up_to_date(150.0));
}

#[test]
fn test_lag() {
    let mut cluster = basic_cluster(1);
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 100.0;

    let node = cluster.get_node(n0).unwrap();
    assert_eq!(node.lag(150.0), 50.0);
    assert_eq!(node.lag(100.0), 0.0);
    // Never negative, even when the node is ahead of the frontier.
    assert_eq!(node.lag(50.0), 0.0);
}

// =============================================================================
// Network Tests
// =============================================================================

#[test]
fn test_is_region_down() {
    let mut cluster = ClusterState::new(0);
    let us_east = cluster.intern("us-east");
    let us_west = cluster.intern("us-west");

    assert!(!cluster.network.is_region_down(us_east));
    cluster.network.add_outage(us_east);
    assert!(cluster.network.is_region_down(us_east));
    assert!(!cluster.network.is_region_down(us_west));
}

#[test]
fn test_is_partitioned() {
    let mut cluster = ClusterState::new(0);
    let us_east = cluster.intern("us-east");
    let us_west = cluster.intern("us-west");

    assert!(!cluster.network.is_partitioned(us_east, us_west));
    cluster.network.add_outage(us_east);
    assert!(cluster.network.is_partitioned(us_east, us_west));
    assert!(cluster.network.is_partitioned(us_west, us_east));
    cluster.network.remove_outage(us_east);
    assert!(!cluster.network.is_partitioned(us_east, us_west));
}

#[test]
fn test_regions_reachable_from() {
    let mut cluster = ClusterState::new(0);
    let us_east = cluster.intern("us-east");
    let us_west = cluster.intern("us-west");
    let eu_west = cluster.intern("eu-west");
    let all = [us_east, us_west, eu_west];

    assert_eq!(cluster.network.regions_reachable_from(us_east, &all), all);

    cluster.network.add_outage(us_east);
    assert!(cluster
        .network
        .regions_reachable_from(us_east, &all)
        .is_empty());
    assert_eq!(
        cluster.network.regions_reachable_from(us_west, &all),
        vec![us_west, eu_west]
    );
}

// =============================================================================
// Event Queue Tests
// =============================================================================

#[test]
fn test_basic_ordering() {
    let mut queue = EventQueue::new();
    queue.push(Event::new(100.0, EventType::NodeFailure, 2));
    queue.push(Event::new(50.0, EventType::NodeRecovery, 3));
    queue.push(Event::new(75.0, EventType::NodeDataLoss, 4));

    assert_eq!(queue.pop().unwrap().time, 50.0);
    assert_eq!(queue.pop().unwrap().time, 75.0);
    assert_eq!(queue.pop().unwrap().time, 100.0);
    assert!(queue.pop().is_none());
}

#[test]
fn test_cancel_events() {
    let mut queue = EventQueue::new();
    queue.push(Event::new(100.0, EventType::NodeFailure, 2));
    queue.push(Event::new(50.0, EventType::NodeRecovery, 2));

    queue.cancel_events_for(2, EventType::NodeFailure);

    let remaining = queue.pop().unwrap();
    assert_eq!(remaining.event_type, EventType::NodeRecovery);
    assert!(queue.pop().is_none());
}

#[test]
fn test_is_empty() {
    let mut queue = EventQueue::new();
    assert!(queue.is_empty());
    queue.push(Event::new(10.0, EventType::NodeFailure, 2));
    assert!(!queue.is_empty());
    queue.pop();
    assert!(queue.is_empty());
}

#[test]
fn test_cancel_all_events_for_target() {
    let mut queue = EventQueue::new();
    queue.push(Event::new(10.0, EventType::NodeFailure, 2));
    queue.push(Event::new(20.0, EventType::NodeRecovery, 2));
    queue.push(Event::new(30.0, EventType::NodeDataLoss, 2));
    queue.push(Event::new(40.0, EventType::NodeFailure, 3));

    queue.cancel_all_for(2);

    let remaining = queue.pop().unwrap();
    assert_eq!(remaining.target_id, 3);
    assert!(queue.pop().is_none());
}

// =============================================================================
// Cluster State Tests
// =============================================================================

#[test]
fn test_node_counts() {
    let cluster = make_test_cluster(5);
    assert_eq!(cluster.num_available(), 5);
    assert_eq!(cluster.num_up_to_date(), 5);
    assert_eq!(cluster.num_with_data(), 5);
}

#[test]
fn test_node_counts_with_failures() {
    let mut cluster = make_test_cluster(5);
    for name in ["node0", "node1"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().is_available = false;
    }
    assert_eq!(cluster.num_available(), 3);

    let n2 = cluster.sym_of("node2").unwrap();
    cluster.get_node_mut(n2).unwrap().is_available = false;
    assert_eq!(cluster.num_available(), 2);
}

// =============================================================================
// Protocol Quorum and Data Loss Tests
// =============================================================================

#[test]
fn test_quorum_calculations() {
    let cluster = make_test_cluster(5);
    let protocol = LeaderlessProtocol::default();

    assert_eq!(protocol.quorum_size(&cluster), 3);
    assert_eq!(cluster.num_available(), 5);
    assert!(protocol.can_commit(&cluster));
}

#[test]
fn test_can_commit_with_failures() {
    let mut cluster = make_test_cluster(5);
    let protocol = LeaderlessProtocol::default();

    for name in ["node0", "node1"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().is_available = false;
    }
    assert_eq!(cluster.num_available(), 3);
    assert!(protocol.can_commit(&cluster));

    let n2 = cluster.sym_of("node2").unwrap();
    cluster.get_node_mut(n2).unwrap().is_available = false;
    assert_eq!(cluster.num_available(), 2);
    assert!(!protocol.can_commit(&cluster));
}

#[test]
fn test_potential_data_loss() {
    let mut cluster = make_test_cluster(3);
    let protocol = LeaderlessProtocol::default();
    assert!(!protocol.has_potential_data_loss(&cluster));

    for name in ["node0", "node1"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().is_available = false;
    }
    assert!(protocol.has_potential_data_loss(&cluster));
}

#[test]
fn test_actual_data_loss() {
    let mut cluster = make_test_cluster(3);
    cluster.commit_index = 100.0;
    let protocol = LeaderlessProtocol::default();

    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 100.0;
    }
    assert!(!protocol.has_actual_data_loss(&cluster));

    // node0 lags; the two current nodes lose their data.
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 50.0;
    for name in ["node1", "node2"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().has_data = false;
    }

    // node0 has data but is behind, so the committed data is gone.
    assert!(protocol.has_actual_data_loss(&cluster));
}

// =============================================================================
// Deterministic Data Loss Metrics
// =============================================================================

/// A node whose only scheduled event inside the test window is data loss.
fn data_loss_only_config(data_loss_time: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(data_loss_time))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

#[test]
fn test_availability_is_two_thirds() {
    // Three nodes lose data at hours 1, 2 and 3.
    //
    // 0-1 h : 3 data-bearing nodes, quorum 2 -> can commit
    // 1-2 h : 2 data-bearing nodes, quorum 2 -> can commit
    // 2-3 h : 1 data-bearing node,  quorum 2 -> cannot commit
    // 3 h   : the last node loses data -> actual data loss, run stops
    //
    // Availability = 2 h / 3 h.
    let mut cluster = ClusterState::new(3);
    for (i, loss_hour) in [1.0, 2.0, 3.0].iter().enumerate() {
        cluster.add_named_node(&format!("node{i}"), data_loss_only_config(hours(*loss_hour)));
    }

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(4.0));

    assert_eq!(result.end_reason, EndReason::DataLoss);
    assert!((result.end_time - hours(3.0)).abs() < 1e-9);
    assert!((result.metrics.availability_fraction() - 2.0 / 3.0).abs() < 1e-9);
    assert!((result.metrics.time_to_potential_data_loss.unwrap() - hours(2.0)).abs() < 1e-9);
    assert!((result.metrics.time_to_actual_data_loss.unwrap() - hours(3.0)).abs() < 1e-9);

    // Only data-loss events should have fired: no syncs, no transient failures.
    let data_loss_events = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeDataLoss)
        .count();
    let sync_events = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeSyncComplete)
        .count();
    assert_eq!(data_loss_events, 3);
    assert_eq!(sync_events, 0);
}

// =============================================================================
// Deterministic Transient Failure Metrics
// =============================================================================

#[test]
fn test_availability_is_three_quarters() {
    // Two nodes fail at hour 2 and recover at hour 3; the run lasts 4 hours.
    //
    // 0-2 h : all three current      -> can commit
    // 2-3 h : only node2 current     -> cannot commit, commit index frozen
    // 3-4 h : all three current again (the freeze kept them current)
    //
    // Availability = 3 h / 4 h.
    let fragile = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(hours(2.0)))
        .recovery(Distribution::constant(hours(1.0)))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();
    let stable = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", fragile.clone());
    cluster.add_named_node("node1", fragile);
    cluster.add_named_node("node2", stable);

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(4.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert!((result.end_time - hours(4.0)).abs() < 1e-9);
    assert!((result.metrics.availability_fraction() - 0.75).abs() < 1e-9);
    // commit_rate is 1.0, so the index equals the available seconds.
    assert!((sim.cluster.commit_index - hours(3.0)).abs() < 1e-6);

    // Commits were frozen during the outage, so the recovered nodes were
    // already current and no sync was needed.
    let sync_events = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeSyncComplete)
        .count();
    assert_eq!(sync_events, 0);

    let failures = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeFailure)
        .count();
    let recoveries = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeRecovery)
        .count();
    assert_eq!(failures, 2);
    assert_eq!(recoveries, 2);
}

#[test]
fn test_all_nodes_transient_failure_is_not_data_loss() {
    // Every node fails at hour 1 and recovers at hour 2.  Transient failure
    // leaves the data intact, so this must not register as data loss even
    // though nothing is reachable for an hour.
    let fragile = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(hours(1.0)))
        .recovery(Distribution::constant(hours(1.0)))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut sim = Simulator::new(
        cluster_with(3, fragile),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(3.0));

    assert_eq!(
        result.end_reason,
        EndReason::TimeLimit,
        "simultaneous transient failures should not count as data loss"
    );

    let failures = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeFailure)
        .count();
    let recoveries = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeRecovery)
        .count();
    assert!(failures >= 3);
    assert!(recoveries >= 3);

    assert!((result.metrics.availability_fraction() - 2.0 / 3.0).abs() < 1e-9);

    let data_loss_events = result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeDataLoss)
        .count();
    assert_eq!(data_loss_events, 0);
}

// =============================================================================
// Simulator Tests
// =============================================================================

#[test]
fn test_basic_simulation_runs() {
    let mut sim = Simulator::new(
        make_test_cluster(3),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
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
fn test_simulation_stops_on_data_loss() {
    let config = ConfigBuilder::new()
        .cost(1.0)
        // A failure roughly every 10 s.
        .failure(Distribution::exponential(1.0 / 10.0).unwrap())
        .recovery(Distribution::constant(100.0))
        .data_loss(Distribution::constant(50.0))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(1000.0))
        .build();

    let mut sim = Simulator::new(
        cluster_with(3, config),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        false,
    );
    let result = sim.run_until_data_loss(Some(1000.0));

    assert_eq!(result.end_reason, EndReason::DataLoss);
    assert!(result.end_time < 1000.0);
}

#[test]
fn test_simulation_reproducibility() {
    let run = || {
        let mut sim = Simulator::new(
            make_test_cluster(3),
            Box::new(NoOpStrategy),
            Box::new(LeaderlessProtocol::default()),
            None,
            Some(12345),
            false,
        );
        sim.run_for(100.0)
    };

    let a = run();
    let b = run();
    assert_eq!(a.metrics.time_available, b.metrics.time_available);
    assert_eq!(a.metrics.time_unavailable, b.metrics.time_unavailable);
}

// =============================================================================
// Node Replacement Strategy Tests
// =============================================================================

fn replacement_node_config(region: &str) -> NodeConfigRef {
    ConfigBuilder::new()
        .region(region)
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

fn actions_of(actions: &[Action], t: ActionType) -> Vec<&Action> {
    actions.iter().filter(|a| a.action_type() == t).collect()
}

#[test]
fn test_spawns_replacement_on_data_loss() {
    let config = replacement_node_config("r1");
    let mut strategy = NodeReplacementStrategy::new(300.0, Some(config), true);

    let mut cluster = make_test_cluster(3);
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().has_data = false;

    let event = Event::new(100.0, EventType::NodeDataLoss, n0);
    let mut rng = make_rng(Some(42));
    let protocol = LeaderlessProtocol::default();

    let actions = collect_actions(&mut strategy, &event, &cluster, &mut rng, &protocol);

    let scheduled = actions_of(&actions, ActionType::ScheduleReplacementCheck);
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].node_id(), Some(n0));
}

#[test]
fn test_starts_sync_on_recovery() {
    let config = replacement_node_config("r1");
    let mut strategy = NodeReplacementStrategy::new(300.0, Some(config), true);

    let mut cluster = make_test_cluster(3);
    cluster.commit_index = 100.0;
    let n0 = cluster.sym_of("node0").unwrap();
    {
        let node = cluster.get_node_mut(n0).unwrap();
        node.is_available = true;
        node.last_applied_index = 50.0;
    }

    let event = Event::new(100.0, EventType::NodeRecovery, n0);
    let mut rng = make_rng(Some(42));
    let protocol = LeaderlessProtocol::default();

    let actions = collect_actions(&mut strategy, &event, &cluster, &mut rng, &protocol);

    let syncs = actions_of(&actions, ActionType::StartSync);
    assert_eq!(syncs.len(), 1);
    assert_eq!(syncs[0].node_id(), Some(n0));
}

// =============================================================================
// Commit Index Tests
// =============================================================================

#[test]
fn test_commit_index_advances_when_can_commit() {
    let mut sim = Simulator::new(
        make_test_cluster(3),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::up_to_date_quorum_protocol(1.0)),
        None,
        Some(42),
        false,
    );
    sim.run_for(100.0);
    assert!(sim.cluster.commit_index > 0.0);
}

#[test]
fn test_commit_index_frozen_when_unavailable() {
    // Every node sits at index 0 while the frontier is already at 100, so
    // there is no up-to-date quorum and nothing can commit.
    let mut cluster = cluster_with(3, make_test_node_config("us-east"));
    cluster.commit_index = 100.0;

    let protocol = LeaderlessProtocol::up_to_date_quorum_protocol(1.0);
    assert!(!protocol.can_commit(&cluster));

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(protocol),
        None,
        Some(42),
        false,
    );
    sim.run_for(10.0);
    assert_eq!(sim.cluster.commit_index, 100.0);
}

#[test]
fn test_variable_commit_rate() {
    let run = |rate: f64| {
        let mut sim = Simulator::new(
            make_test_cluster(3),
            Box::new(NoOpStrategy),
            Box::new(LeaderlessProtocol::up_to_date_quorum_protocol(rate)),
            None,
            Some(42),
            false,
        );
        sim.run_for(100.0);
        sim.cluster.commit_index
    };

    let slow = run(0.5);
    let medium = run(1.0);
    let fast = run(2.0);
    assert!(slow < medium, "{slow} !< {medium}");
    assert!(medium < fast, "{medium} !< {fast}");
}

// =============================================================================
// Snapshot Recovery Tests
// =============================================================================

#[test]
fn test_recovery_without_snapshot() {
    // The lagging node is behind but still ahead of the last snapshot, so
    // log-only replay wins over an expensive snapshot download.
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(365.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(365.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(minutes(5.0)))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let protocol = LeaderlessProtocol::new(1.0, 100.0, 0.0, true);

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
    let index = cluster.index_of(n0).unwrap();
    let sync_time = protocol.compute_sync_time(index, &cluster, &mut rng);

    // lag 40 at a net 9/s is about 4.4 s, far under the snapshot cost.
    let sync_time = sync_time.expect("a donor is available");
    assert!(sync_time < minutes(5.0), "got {sync_time}");
}

#[test]
fn test_recovery_with_snapshot() {
    // With retention of 100 and the donor at 250, the log only reaches back
    // to 150.  A node at 50 is outside that window and must snapshot.
    let config = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(365.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(365.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(60.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let protocol = LeaderlessProtocol::new(1.0, 100.0, 100.0, true);

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

    let protocol = LeaderlessProtocol::new(1.0, 0.0, 0.0, true);

    let mut cluster = cluster_with(3, config);
    cluster.commit_index = 1000.0;
    for i in cluster.active_indices() {
        cluster.node_at_mut(i).last_applied_index = 1000.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();
    cluster.get_node_mut(n0).unwrap().last_applied_index = 0.0;

    let mut rng = make_rng(Some(42));
    let index = cluster.index_of(n0).unwrap();
    let sync_time = protocol
        .compute_sync_time(index, &cluster, &mut rng)
        .expect("a donor is available");

    // lag 1000 at a net 9/s is about 111 s, with no 60 s download on top.
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
        Box::new(LeaderlessProtocol::up_to_date_quorum_protocol(1.0)),
        None,
        Some(42),
        true,
    );
    sim.run_for(200.0);

    // Some commits happened, but less than wall-clock time: the cluster was
    // unavailable for part of the run.
    assert!(sim.cluster.commit_index < 200.0);
    assert!(sim.cluster.commit_index > 0.0);
}

#[test]
fn test_node_snapshot_state_advances() {
    let mut sim = Simulator::new(
        cluster_with(3, make_test_node_config("us-east")),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::new(1.0, 50.0, 0.0, true)),
        None,
        Some(42),
        false,
    );
    sim.run_for(200.0);

    for node in sim.cluster.active() {
        if node.is_available && node.has_data {
            assert!(node.last_snapshot_index > 0.0);
            assert!(
                (node.last_snapshot_index % 50.0).abs() < 1e-9,
                "snapshot at {} is not a multiple of 50",
                node.last_snapshot_index
            );
        }
    }
}

// =============================================================================
// Deterministic Sync Model Tests
// =============================================================================

/// Never fails inside a test window; replays at 2 units/s.
fn stable_config() -> NodeConfigRef {
    ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

/// Never fails; replays at 3 units/s.
fn stable_config_fast() -> NodeConfigRef {
    ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

/// Fails at `failure_time` and recovers `recovery_time` later.
fn fragile_config(failure_time: f64, recovery_time: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(failure_time))
        .recovery(Distribution::constant(recovery_time))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

fn sync_events_for(
    result: &powder_mc::sim::simulator::SimulationResult,
    target: powder_mc::sim::ids::Sym,
) -> Vec<&Event> {
    result
        .event_log
        .iter()
        .filter(|e| e.event_type == EventType::NodeSyncComplete && e.target_id == target)
        .collect()
}

#[test]
fn test_basic_sync_timing() {
    // commit_rate 1.0 against a replay rate of 2.0 closes 1 unit of gap per
    // second.  node0 is 10 behind, so it should finish at t = 10.
    let cfg = stable_config();
    let mut cluster = cluster_with(3, cfg);
    cluster.commit_index = 10.0;
    let indices = cluster.active_indices();
    cluster.node_at_mut(indices[0]).last_applied_index = 0.0;
    cluster.node_at_mut(indices[1]).last_applied_index = 10.0;
    cluster.node_at_mut(indices[2]).last_applied_index = 10.0;
    let n0 = cluster.sym_of("node0").unwrap();

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::up_to_date_quorum_protocol(1.0)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(12.0);

    let syncs = sync_events_for(&result, n0);
    assert!(!syncs.is_empty(), "node0 should have completed a sync");
    assert!(
        (syncs[0].time - 10.0).abs() < 0.01,
        "sync finished at {}",
        syncs[0].time
    );

    let commit_index = sim.cluster.commit_index;
    for node in sim.cluster.active() {
        assert!(node.last_applied_index >= commit_index - 0.01);
    }
}

#[test]
fn test_sync_pauses_when_donors_unavailable() {
    // node0 lags 10 with a net catch-up of 2/s, so 5 s of productive work.
    // Both donors are down from t = 3 to t = 7, freezing progress, so the
    // sync lands around t = 9.
    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", stable_config_fast());
    cluster.add_named_node("node1", fragile_config(3.0, 4.0));
    cluster.add_named_node("node2", fragile_config(3.0, 4.0));
    cluster.commit_index = 10.0;
    let indices = cluster.active_indices();
    cluster.node_at_mut(indices[0]).last_applied_index = 0.0;
    cluster.node_at_mut(indices[1]).last_applied_index = 10.0;
    cluster.node_at_mut(indices[2]).last_applied_index = 10.0;
    let n0 = cluster.sym_of("node0").unwrap();

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::up_to_date_quorum_protocol(1.0)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(9.5);

    let syncs = sync_events_for(&result, n0);
    assert!(!syncs.is_empty(), "node0 should have completed a sync");
    assert!(
        (syncs[0].time - 9.0).abs() < 0.2,
        "sync finished at {}",
        syncs[0].time
    );

    let node0 = sim.cluster.get_node(n0).unwrap();
    assert!(node0.sync.is_none());
    assert!(node0.last_applied_index >= sim.cluster.commit_index - 0.1);
}

#[test]
fn test_multinode_sync_with_donor_outage() {
    // Two lagging nodes against three donors that cycle through outages.
    // Over a long enough run both must finish catching up.
    let mut cluster = ClusterState::new(5);
    cluster.add_named_node("node0", stable_config_fast());
    cluster.add_named_node("node1", stable_config_fast());
    for i in 2..5 {
        cluster.add_named_node(&format!("node{i}"), fragile_config(2.0, 6.0));
    }
    cluster.commit_index = 100.0;
    let indices = cluster.active_indices();
    cluster.node_at_mut(indices[0]).last_applied_index = 90.0;
    cluster.node_at_mut(indices[1]).last_applied_index = 80.0;
    for &i in &indices[2..] {
        cluster.node_at_mut(i).last_applied_index = 100.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();
    let n1 = cluster.sym_of("node1").unwrap();

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::up_to_date_quorum_protocol(1.0)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(40.0);

    let commit = sim.cluster.commit_index;
    let node0 = sim.cluster.get_node(n0).unwrap();
    let node1 = sim.cluster.get_node(n1).unwrap();
    assert!(node0.sync.is_none(), "node0 should have finished syncing");
    assert!(node1.sync.is_none(), "node1 should have finished syncing");
    assert!(node0.last_applied_index >= commit - 0.1);
    assert!(node1.last_applied_index >= commit - 0.1);

    assert!(!sync_events_for(&result, n0).is_empty());
    assert!(!sync_events_for(&result, n1).is_empty());
}

#[test]
fn test_gc_log_only_within_window() {
    // Retention of 300 covers the whole log, so the node at 150 stays inside
    // the window and log-only replay is chosen over a 60 s snapshot.
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
        node.last_applied_index = 150.0;
        node.last_snapshot_index = 100.0;
    }
    let n0 = cluster.sym_of("node0").unwrap();

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::new(1.0, 100.0, 300.0, true)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(15.0);

    let syncs = sync_events_for(&result, n0);
    assert!(!syncs.is_empty());
    // lag 100 at a net 9/s is about 11.1 s, with no snapshot overhead.
    assert!(syncs[0].time < 15.0, "sync finished at {}", syncs[0].time);
    assert!(sim.cluster.get_node(n0).unwrap().sync.is_none());
}

#[test]
fn test_gc_forced_snapshot_outside_window() {
    // Retention of 100 leaves the donor's log covering 150-250 only, so a
    // node at 50 must download the snapshot at 200 first.
    //
    // 60 s download, during which the donor advances to 310, then a 110 unit
    // suffix at a net 9/s: about 72.2 s in total.
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
        Box::new(LeaderlessProtocol::new(1.0, 100.0, 100.0, true)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(80.0);

    let syncs = sync_events_for(&result, n0);
    assert!(!syncs.is_empty());
    assert!(
        syncs[0].time >= 60.0,
        "the snapshot download is mandatory, but sync finished at {}",
        syncs[0].time
    );
    assert!(
        (syncs[0].time - 72.2).abs() < 1.0,
        "sync finished at {}",
        syncs[0].time
    );

    let node0 = sim.cluster.get_node(n0).unwrap();
    assert!(node0.sync.is_none());
    assert!(node0.last_applied_index >= sim.cluster.commit_index - 0.1);
}

// =============================================================================
// Leaderless Protocol Tests
// =============================================================================

#[test]
fn test_can_commit_with_up_to_date_quorum() {
    let mut cluster = make_test_cluster(5);
    let protocol = LeaderlessProtocol::default();
    assert!(protocol.can_commit(&cluster));

    // Two down still leaves three current, which is quorum.
    for name in ["node0", "node1"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().is_available = false;
    }
    assert!(protocol.can_commit(&cluster));

    // A third takes it below quorum.
    let n2 = cluster.sym_of("node2").unwrap();
    cluster.get_node_mut(n2).unwrap().is_available = false;
    assert!(!protocol.can_commit(&cluster));
}

#[test]
fn test_commit_rate_and_snapshot_interval() {
    let protocol = LeaderlessProtocol::new(2.0, 100.0, 0.0, true);
    assert_eq!(protocol.commit_rate(), 2.0);
    assert_eq!(protocol.snapshot_interval(), 100.0);

    let default = LeaderlessProtocol::default();
    assert_eq!(default.commit_rate(), 1.0);
    assert_eq!(default.snapshot_interval(), 0.0);
}

#[test]
fn test_available_but_not_up_to_date_can_commit() {
    let mut cluster = make_test_cluster(5);
    cluster.commit_index = 100.0;

    let majority_available = LeaderlessProtocol::majority_available(1.0);
    let up_to_date = LeaderlessProtocol::default();

    // Everyone is up but nobody is current.
    assert!(!up_to_date.can_commit(&cluster));
    assert!(majority_available.can_commit(&cluster));
}

#[test]
fn test_unavailable_nodes_dont_count() {
    let mut cluster = make_test_cluster(5);
    cluster.commit_index = 100.0;
    let protocol = LeaderlessProtocol::majority_available(1.0);

    for name in ["node0", "node1", "node2"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().is_available = false;
    }
    // Two available against a quorum of three.
    assert!(!protocol.can_commit(&cluster));
}

#[test]
fn test_simulation_more_available_than_up_to_date() {
    // Nodes recover quickly but replay slowly, so they spend time available
    // yet lagging -- exactly the gap between the two quorum rules.
    let config = || {
        ConfigBuilder::new()
            .cost(1.0)
            .failure(Distribution::exponential(1.0 / hours(2.0)).unwrap())
            .recovery(Distribution::constant(minutes(5.0)))
            .data_loss(Distribution::exponential(1.0 / days(365.0)).unwrap())
            .log_replay_rate(Distribution::constant(1.5))
            .snapshot_download(Distribution::constant(0.0))
            .spawn(Distribution::constant(minutes(10.0)))
            .build()
    };

    let run = |protocol: Box<dyn Protocol>| {
        let mut sim = Simulator::new(
            cluster_with(3, config()),
            Box::new(NoOpStrategy),
            protocol,
            None,
            Some(42),
            false,
        );
        sim.run_for(days(30.0))
            .metrics
            .availability_fraction()
    };

    let utd = run(Box::<LeaderlessProtocol>::default());
    let avail = run(Box::new(LeaderlessProtocol::majority_available(1.0)));
    assert!(avail >= utd, "majority-available {avail} < up-to-date {utd}");
}

#[test]
fn test_up_to_date_quorum_true_requires_up_to_date_nodes() {
    let mut cluster = make_test_cluster(5);
    cluster.commit_index = 100.0;
    let protocol = LeaderlessProtocol::new(1.0, 0.0, 0.0, true);

    assert!(!protocol.can_commit(&cluster));

    for name in ["node0", "node1", "node2"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().last_applied_index = 100.0;
    }
    assert!(protocol.can_commit(&cluster));
}

#[test]
fn test_up_to_date_quorum_false_allows_lagging_nodes() {
    let mut cluster = make_test_cluster(5);
    cluster.commit_index = 100.0;
    let protocol = LeaderlessProtocol::new(1.0, 0.0, 0.0, false);

    assert!(protocol.can_commit(&cluster));

    for name in ["node0", "node1", "node2"] {
        let sym = cluster.sym_of(name).unwrap();
        cluster.get_node_mut(sym).unwrap().is_available = false;
    }
    assert!(!protocol.can_commit(&cluster));
}

#[test]
fn test_default_is_up_to_date_quorum() {
    assert!(LeaderlessProtocol::default().up_to_date_quorum());
}

#[test]
fn test_majority_available_alias_defaults_false() {
    assert!(!LeaderlessProtocol::majority_available(1.0).up_to_date_quorum());
}

#[test]
fn test_configurable_properties() {
    let protocol = LeaderlessProtocol::new(2.0, 100.0, 50.0, false);
    assert_eq!(protocol.commit_rate(), 2.0);
    assert_eq!(protocol.snapshot_interval(), 100.0);
    assert_eq!(protocol.log_retention_ops(), 50.0);
    assert!(!protocol.up_to_date_quorum());
}

/// Builds the three-node staggered-failure cluster both staggered tests use.
///
/// node0 fails at 1 h and recovers an hour later; node1 fails at 2 h and
/// recovers an hour later; node2 never fails.
fn staggered_cluster(node0_replay_rate: f64) -> ClusterState {
    let staggered = |failure_hours: f64, replay: f64| {
        ConfigBuilder::new()
            .cost(1.0)
            .failure(Distribution::constant(hours(failure_hours)))
            .recovery(Distribution::constant(hours(1.0)))
            .data_loss(Distribution::constant(days(9999.0)))
            .log_replay_rate(Distribution::constant(replay))
            .snapshot_download(Distribution::constant(0.0))
            .spawn(Distribution::constant(0.0))
            .build()
    };
    let stable = ConfigBuilder::new()
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(3.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut cluster = ClusterState::new(3);
    cluster.add_named_node("node0", staggered(1.0, node0_replay_rate));
    cluster.add_named_node("node1", staggered(2.0, 3.0));
    cluster.add_named_node("node2", stable);
    cluster
}

#[test]
fn test_staggered_failure_causes_downtime_when_up_to_date_quorum() {
    // 0-1 h: all three current.
    // 1-2 h: node0 down; the other two keep committing, so node0 falls behind.
    // 2-3 h: node0 is back but lagging and node1 is down, leaving only node2
    //        current -- below quorum, so no commits.
    // 3-4 h: node1 returns and the cluster recovers.
    //
    // node0 replays at 1.001/s against a commit rate of 1.0, so it stays
    // behind for the whole window.
    let mut sim = Simulator::new(
        staggered_cluster(1.001),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::new(1.0, 0.0, 0.0, true)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(4.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    let availability = result.metrics.availability_fraction();
    assert!(availability < 1.0);
    assert!(
        (availability - 0.75).abs() < 0.01,
        "availability was {availability}"
    );
}

#[test]
fn test_staggered_failure_no_downtime_when_any_quorum() {
    // The same timeline, but lagging no longer disqualifies a node, so two
    // are always available and the cluster never stops committing.
    let mut sim = Simulator::new(
        staggered_cluster(3.0),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::new(1.0, 0.0, 0.0, false)),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(4.0));

    assert_eq!(result.end_reason, EndReason::TimeLimit);
    assert!((result.metrics.availability_fraction() - 1.0).abs() < 1e-12);
}
