//! Port of `tests/test_event_timing.py`.
//!
//! Runs long simulations with known input distributions and checks that the
//! inter-event times in the event log are consistent with those
//! distributions, using Kolmogorov-Smirnov tests.

mod common;

use std::collections::HashMap;

use common::{cluster_with, days, hours, minutes, ConfigBuilder};

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::Distribution;
use powder_mc::sim::events::{Event, EventType};
use powder_mc::sim::ids::Sym;
use powder_mc::sim::node::NodeConfigRef;
use powder_mc::sim::protocol::LeaderlessProtocol;
use powder_mc::sim::simulator::Simulator;
use powder_mc::sim::strategy::{NoOpStrategy, NodeReplacementStrategy};
use powder_mc::stats;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Significance level for the KS tests.  1% keeps flakiness very low while
/// retaining good power at these sample sizes.
const KS_ALPHA: f64 = 0.01;

fn make_config(failure_rate: f64, recovery_rate: f64) -> NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::exponential(failure_rate).unwrap())
        .recovery(Distribution::exponential(recovery_rate).unwrap())
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(10.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build()
}

/// Group event times of one type by target, preserving order.
fn per_node_times(event_log: &[Event], event_type: EventType) -> HashMap<Sym, Vec<f64>> {
    let mut by_node: HashMap<Sym, Vec<f64>> = HashMap::new();
    for event in event_log {
        if event.event_type == event_type {
            by_node.entry(event.target_id).or_default().push(event.time);
        }
    }
    by_node
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_failure_inter_arrival_times() {
    // `apply_node_failure` schedules the next failure at
    // `now + recovery_time + next_failure_time`, so the gap between a
    // recovery and the following failure is exactly one draw from the
    // failure distribution.
    let failure_rate = 1.0 / hours(10.0);
    let recovery_rate = 1.0 / hours(2.0);

    let mut sim = Simulator::new(
        cluster_with(3, make_config(failure_rate, recovery_rate)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(42),
        true,
    );
    let result = sim.run_for(hours(50_000.0));

    let failures = per_node_times(&result.event_log, EventType::NodeFailure);
    let recoveries = per_node_times(&result.event_log, EventType::NodeRecovery);

    let mut samples = Vec::new();
    for (node_id, fail_times) in &failures {
        let empty = Vec::new();
        let rec_times = recoveries.get(node_id).unwrap_or(&empty);

        // The first failure is scheduled from t = 0.
        let mut prev_available_time = 0.0;
        let mut rec_idx = 0;
        for &ft in fail_times {
            while rec_idx < rec_times.len() && rec_times[rec_idx] <= ft {
                prev_available_time = rec_times[rec_idx];
                rec_idx += 1;
            }
            let gap = ft - prev_available_time;
            if gap > 0.0 {
                samples.push(gap);
            }
        }
    }

    assert!(samples.len() > 100, "got only {} samples", samples.len());

    let scale = 1.0 / failure_rate;
    let (stat, p) = stats::ks_test(&samples, |x| stats::exponential_cdf(x, scale));
    assert!(
        p > KS_ALPHA,
        "failure inter-arrival times do not match Exponential(rate={failure_rate}): \
         KS stat={stat:.4}, p={p:.4}, n={}",
        samples.len()
    );
}

#[test]
fn test_recovery_durations() {
    // Recovery is scheduled at `now + recovery_time`, so the gap between a
    // failure and its recovery is one draw from the recovery distribution.
    let failure_rate = 1.0 / hours(10.0);
    let recovery_rate = 1.0 / hours(2.0);

    let mut sim = Simulator::new(
        cluster_with(3, make_config(failure_rate, recovery_rate)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(123),
        true,
    );
    let result = sim.run_for(hours(50_000.0));

    let failures = per_node_times(&result.event_log, EventType::NodeFailure);
    let recoveries = per_node_times(&result.event_log, EventType::NodeRecovery);

    let mut samples = Vec::new();
    for (node_id, fail_times) in &failures {
        let empty = Vec::new();
        let rec_times = recoveries.get(node_id).unwrap_or(&empty);

        let mut rec_idx = 0;
        for &ft in fail_times {
            while rec_idx < rec_times.len() && rec_times[rec_idx] <= ft {
                rec_idx += 1;
            }
            if rec_idx < rec_times.len() {
                let duration = rec_times[rec_idx] - ft;
                if duration > 0.0 {
                    samples.push(duration);
                }
                rec_idx += 1;
            }
        }
    }

    assert!(samples.len() > 100, "got only {} samples", samples.len());

    let scale = 1.0 / recovery_rate;
    let (stat, p) = stats::ks_test(&samples, |x| stats::exponential_cdf(x, scale));
    assert!(
        p > KS_ALPHA,
        "recovery durations do not match Exponential(rate={recovery_rate}): \
         KS stat={stat:.4}, p={p:.4}, n={}",
        samples.len()
    );
}

#[test]
fn test_spawn_durations() {
    // The spawn action runs when the replacement timeout fires, so the gap
    // from that timeout to the spawn completion is one draw from the spawn
    // distribution.  Aggressive data loss with safe_mode off makes
    // replacements cascade, producing plenty of spawns.
    let spawn_mean = minutes(5.0);
    let spawn_std = minutes(1.0);

    let fragile = ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(days(9999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::exponential(1.0 / hours(10.0)).unwrap())
        .log_replay_rate(Distribution::constant(100.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::normal(spawn_mean, spawn_std, 0.0).unwrap())
        .build();

    let mut sim = Simulator::new(
        cluster_with(5, fragile.clone()),
        Box::new(NodeReplacementStrategy::new(30.0, Some(fragile), false)),
        Box::new(LeaderlessProtocol::majority_available(1.0)),
        None,
        Some(7),
        true,
    );
    let result = sim.run_for(hours(50_000.0));

    let mut timeout_times = Vec::new();
    let mut spawn_complete_times = Vec::new();
    for event in &result.event_log {
        match event.event_type {
            EventType::NodeReplacementTimeout => timeout_times.push(event.time),
            EventType::NodeSpawnComplete => spawn_complete_times.push(event.time),
            _ => {}
        }
    }
    spawn_complete_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Pair each completion with the most recent preceding timeout.
    let mut samples = Vec::new();
    let mut t_idx = 0usize;
    for &sc_time in &spawn_complete_times {
        while t_idx + 1 < timeout_times.len() && timeout_times[t_idx + 1] <= sc_time {
            t_idx += 1;
        }
        if t_idx < timeout_times.len() && timeout_times[t_idx] <= sc_time {
            samples.push(sc_time - timeout_times[t_idx]);
        }
    }

    assert!(
        samples.len() > 20,
        "got only {} spawn samples",
        samples.len()
    );

    // The normal is clamped at zero, but a mean of 300 s with a 60 s spread
    // makes that boundary irrelevant.
    let (stat, p) = stats::ks_test(&samples, |x| stats::normal_cdf(x, spawn_mean, spawn_std));
    assert!(
        p > KS_ALPHA,
        "spawn durations do not match Normal(mean={spawn_mean}, std={spawn_std}): \
         KS stat={stat:.4}, p={p:.4}, n={}",
        samples.len()
    );
}

#[test]
fn test_snapshot_download_durations() {
    // With an enormous replay rate the log phase is effectively free, and
    // with retention of one unit every sync is forced through the snapshot
    // path.  The sync duration is then just the snapshot download.
    let failure_rate = 1.0 / hours(50.0);
    let recovery_rate = 1.0 / hours(0.5);
    let snapshot_download_rate = 1.0 / hours(0.25);

    let config = ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::exponential(failure_rate).unwrap())
        .recovery(Distribution::exponential(recovery_rate).unwrap())
        .data_loss(Distribution::constant(days(9999.0)))
        .log_replay_rate(Distribution::constant(1e12))
        .snapshot_download(Distribution::exponential(snapshot_download_rate).unwrap())
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let mut sim = Simulator::new(
        cluster_with(5, config),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::new(1.0, 1.0, 1.0, true)),
        None,
        Some(99),
        true,
    );
    let result = sim.run_for(hours(100_000.0));

    let recoveries = per_node_times(&result.event_log, EventType::NodeRecovery);
    let syncs = per_node_times(&result.event_log, EventType::NodeSyncComplete);

    let mut sync_durations = Vec::new();
    for (node_id, sync_times) in &syncs {
        let empty = Vec::new();
        let rec_times = recoveries.get(node_id).unwrap_or(&empty);

        let mut rec_idx = 0;
        for &st in sync_times {
            let mut best_rec = None;
            while rec_idx < rec_times.len() && rec_times[rec_idx] <= st {
                best_rec = Some(rec_times[rec_idx]);
                rec_idx += 1;
            }
            if let Some(rec) = best_rec {
                let duration = st - rec;
                if duration > 0.0 {
                    sync_durations.push(duration);
                }
            }
        }
    }

    assert!(
        sync_durations.len() > 100,
        "got only {} sync samples",
        sync_durations.len()
    );

    let scale = 1.0 / snapshot_download_rate;
    let (stat, p) = stats::ks_test(&sync_durations, |x| stats::exponential_cdf(x, scale));
    assert!(
        p > KS_ALPHA,
        "snapshot download durations do not match \
         Exponential(rate={snapshot_download_rate}): \
         KS stat={stat:.4}, p={p:.4}, n={}",
        sync_durations.len()
    );
}

#[test]
fn test_mean_failure_recovery_cycle_time() {
    // A simpler sanity check: one full available-failed-recovered cycle
    // should average MTTF + MTTR.
    let mttf = hours(10.0);
    let mttr = hours(2.0);

    let mut sim = Simulator::new(
        cluster_with(3, make_config(1.0 / mttf, 1.0 / mttr)),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(0),
        true,
    );
    let result = sim.run_for(hours(100_000.0));

    let failures = per_node_times(&result.event_log, EventType::NodeFailure);
    let mut cycle_times = Vec::new();
    for times in failures.values() {
        for pair in times.windows(2) {
            cycle_times.push(pair[1] - pair[0]);
        }
    }

    assert!(cycle_times.len() > 100);

    let expected_cycle = mttf + mttr;
    let observed_mean = stats::mean(&cycle_times);
    let relative = (observed_mean - expected_cycle).abs() / expected_cycle;
    assert!(
        relative < 0.10,
        "mean cycle time {observed_mean:.0}s vs expected {expected_cycle:.0}s \
         (MTTF={mttf:.0} + MTTR={mttr:.0})"
    );
}

/// Not in the Python suite: confirms the interned cluster keeps per-node
/// event grouping intact, which every test above relies on.
#[test]
fn event_log_targets_group_by_node() {
    let mut cluster = ClusterState::new(3);
    for i in 0..3 {
        cluster.add_named_node(
            &format!("node{i}"),
            make_config(1.0 / hours(1.0), 1.0 / hours(1.0)),
        );
    }

    let mut sim = Simulator::new(
        cluster,
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        Some(5),
        true,
    );
    let result = sim.run_for(hours(500.0));

    let failures = per_node_times(&result.event_log, EventType::NodeFailure);
    assert_eq!(failures.len(), 3, "all three nodes should have failed");
    for times in failures.values() {
        assert!(times.windows(2).all(|w| w[0] <= w[1]), "times are ordered");
    }
}
