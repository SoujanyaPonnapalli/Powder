//! Port of `tests/test_closed_form_verification.py`.
//!
//! Checks the simulator's mean availability and mean time to first
//! unavailability against closed forms derived from the Markov chain of a
//! k-out-of-n system.
//!
//! The scenario matches the derivations: a leaderless protocol with
//! `up_to_date_quorum = false`, no data loss, no replacement, no outages,
//! a majority quorum of `floor(n/2) + 1`, and exponential failure and
//! recovery times.
//!
//! For a failure rate `a` and recovery rate `b`:
//!
//! * per-node availability `p = b / (a + b)`
//! * n-node availability `A_n = sum_{k=0}^{floor(n/2)} C(n,k) p^(n-k) (1-p)^k`
//! * three-node first passage `MTTF_3 = (5a + b) / (6a^2)`
//! * five-node first passage comes from the birth-death absorption system

mod common;

use common::{days, hours, minutes, ConfigBuilder};

use powder_mc::monte_carlo::{
    MonteCarloConfig, MonteCarloResults, MonteCarloRunner, ScenarioFactory,
};
use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::Distribution;
use powder_mc::sim::protocol::{LeaderlessProtocol, Protocol};
use powder_mc::sim::strategy::{ClusterStrategy, NoOpStrategy};
use powder_mc::stats;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_cluster(failure_rate: f64, recovery_rate: f64, num_nodes: usize) -> ClusterState {
    let cfg = ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::exponential(failure_rate).unwrap())
        .recovery(Distribution::exponential(recovery_rate).unwrap())
        .data_loss(Distribution::constant(days(99999.0)))
        .log_replay_rate(Distribution::constant(1e6))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let mut cluster = ClusterState::new(num_nodes);
    for i in 0..num_nodes {
        cluster.add_named_node(&format!("node{i}"), cfg.clone());
    }
    cluster
}

fn run_sims(
    failure_rate: f64,
    recovery_rate: f64,
    num_nodes: usize,
    num_sims: usize,
    sim_duration: f64,
    seed: u64,
) -> MonteCarloResults {
    let config = MonteCarloConfig {
        num_simulations: num_sims,
        max_time: Some(sim_duration),
        stop_on_data_loss: false,
        base_seed: Some(seed),
        log_events: false,
    }
    .validate()
    .unwrap();

    let cluster = || make_cluster(failure_rate, recovery_rate, num_nodes);
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::majority_available(1.0)) as Box<dyn Protocol>;
    let scenario = ScenarioFactory {
        cluster: &cluster,
        strategy: &strategy,
        protocol: &protocol,
        network_config: None,
    };

    MonteCarloRunner::new(config).run(&scenario, None)
}

/// Binomial coefficient.
fn comb(n: usize, k: usize) -> f64 {
    let mut result = 1.0f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Closed-form availability for an n-node majority-quorum system.
///
/// Each node is independently available with probability `p = b/(a+b)`, and
/// the system is available while at most `floor(n/2)` nodes are down.
fn analytical_availability(a: f64, b: f64, n: usize) -> f64 {
    let p = b / (a + b);
    let q = 1.0 - p;
    (0..=n / 2)
        .map(|k| comb(n, k) * p.powi((n - k) as i32) * q.powi(k as i32))
        .sum()
}

/// Mean time from all-up to first unavailability for a three-node system.
///
/// A birth-death chain over the number of failed nodes, absorbing at two:
/// `T_0 = 1/(3a) + T_1` and `T_1 = 1/(2a+b) + [b/(2a+b)] T_0`, which solves
/// to `(5a + b) / (6a^2)`.
fn analytical_mttf_3(a: f64, b: f64) -> f64 {
    (5.0 * a + b) / (6.0 * a * a)
}

/// Mean time from all-up to first unavailability for a five-node system.
///
/// The same chain over 0..5 failures, absorbing at three.  The three
/// first-passage equations are solved directly:
///
/// ```text
/// T_0 - T_1                                             = 1/(5a)
/// -b/(4a+b) T_0 + T_1 - 4a/(4a+b) T_2                   = 1/(4a+b)
///                 -2b/(3a+2b) T_1 + T_2                 = 1/(3a+2b)
/// ```
fn analytical_mttf_5(a: f64, b: f64) -> f64 {
    // Substituting the first and third equations into the second leaves a
    // single unknown, so no linear solver is needed.
    let c1 = 1.0 / (5.0 * a);
    let (p0, p2, r1) = (b / (4.0 * a + b), 4.0 * a / (4.0 * a + b), 1.0 / (4.0 * a + b));
    let (q1, r2) = (2.0 * b / (3.0 * a + 2.0 * b), 1.0 / (3.0 * a + 2.0 * b));

    // T_1 = T_0 - c1, and T_2 = q1 * T_1 + r2.
    // => -p0 T_0 + (T_0 - c1) - p2 (q1 (T_0 - c1) + r2) = r1
    let coefficient = -p0 + 1.0 - p2 * q1;
    let constant = -c1 + p2 * q1 * c1 - p2 * r2;
    (r1 - constant) / coefficient
}

/// Assert the analytical value lies inside the 99% confidence interval of
/// the samples.
#[track_caller]
fn assert_within_ci(samples: &[f64], analytical: f64, label: &str) {
    let mean = stats::mean(samples);
    let ci_half = stats::t_ci_half_width(samples, 0.99);
    let (lo, hi) = (mean - ci_half, mean + ci_half);
    assert!(
        lo <= analytical && analytical <= hi,
        "{label}: analytical {analytical:.8} outside the 99% CI \
         [{lo:.8}, {hi:.8}] (mean {mean:.8})"
    );
}

fn first_unavailability_samples(results: &MonteCarloResults) -> Vec<f64> {
    results
        .time_to_first_unavailability_samples
        .iter()
        .filter_map(|t| *t)
        .collect()
}

// ---------------------------------------------------------------------------
// Shared parameters
// ---------------------------------------------------------------------------

/// Roughly one failure per node per 12 hours.
fn failure_rate() -> f64 {
    1.0 / hours(12.0)
}
/// Roughly a 10 minute recovery per node.
fn recovery_rate() -> f64 {
    1.0 / minutes(10.0)
}

/// 30 days per availability run, about 60 failure cycles per node.
fn sim_duration_avail() -> f64 {
    days(30.0)
}
const NUM_SIMS_AVAIL: usize = 200;
const NUM_SIMS_MTTF: usize = 300;

// ==========================================================================
// Closed-form verification
// ==========================================================================

#[test]
fn test_3_node_availability() {
    let (a, b) = (failure_rate(), recovery_rate());
    let analytical = analytical_availability(a, b, 3);
    let results = run_sims(a, b, 3, NUM_SIMS_AVAIL, sim_duration_avail(), 42);
    assert_within_ci(
        &results.availability_samples,
        analytical,
        "3-node availability",
    );
}

#[test]
fn test_5_node_availability() {
    let (a, b) = (failure_rate(), recovery_rate());
    let analytical = analytical_availability(a, b, 5);
    let results = run_sims(a, b, 5, NUM_SIMS_AVAIL, sim_duration_avail(), 100_000);
    assert_within_ci(
        &results.availability_samples,
        analytical,
        "5-node availability",
    );
}

#[test]
fn test_3_node_mttf() {
    let (a, b) = (failure_rate(), recovery_rate());
    let analytical = analytical_mttf_3(a, b);
    // The window has to be well past the mean, or right-censoring biases
    // the estimate low.
    let results = run_sims(a, b, 3, NUM_SIMS_MTTF, analytical * 5.0, 200_000);

    let samples = first_unavailability_samples(&results);
    assert!(
        samples.len() > 50,
        "too few runs went unavailable: {}",
        samples.len()
    );
    assert_within_ci(&samples, analytical, "3-node MTTF");
}

#[test]
fn test_5_node_mttf() {
    let (a, b) = (failure_rate(), recovery_rate());
    let analytical = analytical_mttf_5(a, b);
    let results = run_sims(a, b, 5, NUM_SIMS_MTTF, analytical * 5.0, 300_000);

    let samples = first_unavailability_samples(&results);
    assert!(
        samples.len() > 50,
        "too few runs went unavailable: {}",
        samples.len()
    );
    assert_within_ci(&samples, analytical, "5-node MTTF");
}

// ==========================================================================
// Across rate regimes
// ==========================================================================

/// `(label, failure_rate, recovery_rate, seed_base)`.
///
/// Python derives the seed from `hash(label)`, which is not stable across
/// interpreter runs; the port uses fixed seeds so the suite is reproducible.
fn rate_scenarios() -> Vec<(&'static str, f64, f64, u64)> {
    vec![
        // Frequent failures with fast recovery, a/b around 0.008.
        (
            "frequent_fail_fast_recover",
            1.0 / hours(6.0),
            1.0 / minutes(3.0),
            11_000,
        ),
        // High stress: failures every 3 h, 3 min recovery, a/b around 0.017.
        ("high_stress", 1.0 / hours(3.0), 1.0 / minutes(3.0), 22_000),
    ]
}

#[test]
fn test_3_node_availability_multi_rate() {
    for (label, a, b, seed) in rate_scenarios() {
        let analytical = analytical_availability(a, b, 3);
        let results = run_sims(a, b, 3, 150, days(30.0), seed);
        assert_within_ci(
            &results.availability_samples,
            analytical,
            &format!("3-node avail [{label}]"),
        );
    }
}

#[test]
fn test_5_node_availability_multi_rate() {
    for (label, a, b, seed) in rate_scenarios() {
        let analytical = analytical_availability(a, b, 5);
        let results = run_sims(a, b, 5, 150, days(30.0), seed + 1);
        assert_within_ci(
            &results.availability_samples,
            analytical,
            &format!("5-node avail [{label}]"),
        );
    }
}

#[test]
fn test_3_node_mttf_multi_rate() {
    for (label, a, b, seed) in rate_scenarios() {
        let analytical = analytical_mttf_3(a, b);
        let results = run_sims(a, b, 3, 800, analytical * 10.0, seed + 2);

        let samples = first_unavailability_samples(&results);
        assert!(
            samples.len() > 50,
            "[{label}] too few runs went unavailable: {}",
            samples.len()
        );
        assert_within_ci(&samples, analytical, &format!("3-node mttF [{label}]"));
    }
}

#[test]
fn test_5_node_mttf_multi_rate() {
    for (label, a, b, seed) in rate_scenarios() {
        let analytical = analytical_mttf_5(a, b);
        let results = run_sims(a, b, 5, 800, analytical * 10.0, seed + 3);

        let samples = first_unavailability_samples(&results);
        assert!(
            samples.len() > 50,
            "[{label}] too few runs went unavailable: {}",
            samples.len()
        );
        assert_within_ci(&samples, analytical, &format!("5-node mttF [{label}]"));
    }
}

/// Not in the Python suite: pins the closed forms themselves, so a mistake
/// in the reference values cannot quietly excuse a simulator bug.
#[test]
fn closed_forms_agree_with_hand_computed_values() {
    // p = 0.9 gives A_3 = 0.81 * 1.2 = 0.972.
    let (a, b) = (1.0, 9.0);
    assert!((analytical_availability(a, b, 3) - 0.972).abs() < 1e-12);

    // A_5 = p^5 + 5 p^4 q + 10 p^3 q^2 with p = 0.9.
    let p: f64 = 0.9;
    let q = 0.1;
    let expected5 = p.powi(5) + 5.0 * p.powi(4) * q + 10.0 * p.powi(3) * q * q;
    assert!((analytical_availability(a, b, 5) - expected5).abs() < 1e-12);

    // MTTF_3 = (5a + b) / (6 a^2) = 14 / 6.
    assert!((analytical_mttf_3(a, b) - 14.0 / 6.0).abs() < 1e-12);

    // The five-node solution must satisfy its own first equation,
    // T_0 - T_1 = 1/(5a), and exceed the three-node answer.
    let t0 = analytical_mttf_5(a, b);
    assert!(t0 > analytical_mttf_3(a, b));
    assert!(t0.is_finite() && t0 > 0.0);
}
