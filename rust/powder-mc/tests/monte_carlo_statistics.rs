//! Port of `tests/test_monte_carlo_statistics.py`, plus the convergence and
//! adaptive Monte Carlo sections of `tests/test_simulation.py`.
//!
//! The statistical scenario is deliberately simple so it has a closed form:
//! a leaderless protocol with `up_to_date_quorum = false` (only node
//! availability matters), no data loss, no replacement, no outages, instant
//! recovery of sync state, three nodes and a majority quorum of two.
//!
//! With exponential failures and constant recovery each node has steady
//! state availability `p = MTBF / (MTBF + MTTR)`, so the system availability
//! is `A = 3p^2(1-p) + p^3 = p^2(3 - 2p)`.

mod common;

use common::{days, hours, minutes, ConfigBuilder};

use powder_mc::monte_carlo::{
    check_convergence, estimate_required_runs, run_monte_carlo_converged, ConvergenceCriteria,
    ConvergenceMetric, MonteCarloConfig, MonteCarloResults, MonteCarloRunner, ScenarioFactory,
};
use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::Distribution;
use powder_mc::sim::protocol::{LeaderlessProtocol, Protocol};
use powder_mc::sim::strategy::{ClusterStrategy, NoOpStrategy};
use powder_mc::stats;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Tunable failure and recovery, with data loss effectively disabled.
fn simple_node_config(mtbf_seconds: f64, mttr_seconds: f64) -> powder_mc::sim::node::NodeConfigRef {
    ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::exponential(1.0 / mtbf_seconds).unwrap())
        .recovery(Distribution::constant(mttr_seconds))
        .data_loss(Distribution::constant(days(99999.0)))
        .log_replay_rate(Distribution::constant(1e6))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build()
}

fn make_simple_cluster(mtbf_seconds: f64, mttr_seconds: f64, num_nodes: usize) -> ClusterState {
    let cfg = simple_node_config(mtbf_seconds, mttr_seconds);
    let mut cluster = ClusterState::new(num_nodes);
    for i in 0..num_nodes {
        cluster.add_named_node(&format!("node{i}"), cfg.clone());
    }
    cluster
}

/// Closed-form availability for a three-node majority quorum.
fn analytical_availability(mtbf: f64, mttr: f64) -> f64 {
    let p = mtbf / (mtbf + mttr);
    p * p * (3.0 - 2.0 * p)
}

fn run_simple(
    mtbf_seconds: f64,
    mttr_seconds: f64,
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

    let cluster = || make_simple_cluster(mtbf_seconds, mttr_seconds, 3);
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

/// MTBF of 4 h against an MTTR of 10 min, giving a per-node p of about 0.96.
/// The high failure rate keeps the variance non-trivial over short windows.
fn mtbf() -> f64 {
    hours(4.0)
}
fn mttr() -> f64 {
    minutes(10.0)
}
/// Seven days per run: roughly 42 failures per node, enough for a stable
/// per-run estimate.
fn sim_duration() -> f64 {
    days(7.0)
}

// ==========================================================================
// Statistical guarantees
// ==========================================================================

#[test]
fn test_ci_coverage_rate() {
    // Across K independent trials, the 95% CI built from N runs should
    // contain the analytical availability about 95% of the time.  The
    // closed form is used as ground truth rather than a noisy sample mean.
    const N: usize = 100;
    const K: usize = 50;
    let true_avail = analytical_availability(mtbf(), mttr());

    let mut covered = 0;
    for trial in 0..K {
        let seed = (trial * N) as u64;
        let results = run_simple(mtbf(), mttr(), N, sim_duration(), seed);
        let samples = &results.availability_samples;
        let mean = stats::mean(samples);
        let ci_half = stats::t_ci_half_width(samples, 0.95);
        if (mean - ci_half) <= true_avail && true_avail <= (mean + ci_half) {
            covered += 1;
        }
    }

    let coverage = covered as f64 / K as f64;
    // For a true 95% interval, seeing fewer than 42 of 50 covered is
    // vanishingly unlikely.
    assert!(
        coverage >= 0.84,
        "CI coverage {:.2}% is too low; expected about 95% (analytical mean {true_avail:.8})",
        coverage * 100.0
    );
}

#[test]
fn test_ci_width_scales_with_sqrt_n() {
    // Standard error goes as 1/sqrt(n), so quadrupling the sample size
    // should roughly halve the interval.
    const N: usize = 100;
    let seed = 100_000;

    let results_n = run_simple(mtbf(), mttr(), N, sim_duration(), seed);
    let results_4n = run_simple(mtbf(), mttr(), 4 * N, sim_duration(), seed);

    let hw_n = stats::t_ci_half_width(&results_n.availability_samples, 0.95);
    let hw_4n = stats::t_ci_half_width(&results_4n.availability_samples, 0.95);

    let ratio = hw_4n / hw_n;
    assert!(
        (0.3..=0.7).contains(&ratio),
        "CI width ratio (4N/N) = {ratio:.3}; expected about 0.5 (hw_n={hw_n:.6}, hw_4n={hw_4n:.6})"
    );
}

#[test]
fn test_availability_converges_to_analytical_value() {
    let analytical = analytical_availability(mtbf(), mttr());
    let results = run_simple(mtbf(), mttr(), 500, sim_duration(), 42);

    let samples = &results.availability_samples;
    let mean = stats::mean(samples);
    // A 99% interval keeps the false-positive rate down.
    let ci_half = stats::t_ci_half_width(samples, 0.99);

    assert!(
        (mean - ci_half) <= analytical && analytical <= (mean + ci_half),
        "analytical availability {analytical:.8} is outside the 99% CI \
         [{:.8}, {:.8}] (mean {mean:.8})",
        mean - ci_half,
        mean + ci_half
    );
}

#[test]
fn test_higher_failure_rate_lowers_availability() {
    let mtbf_low_rate = hours(8.0);
    let mtbf_high_rate = hours(2.0);
    let mttr = minutes(10.0);
    let n = 200;
    let seed = 200_000;

    let results_low = run_simple(mtbf_low_rate, mttr, n, sim_duration(), seed);
    let results_high = run_simple(mtbf_high_rate, mttr, n, sim_duration(), seed + n as u64);

    let mean_low = results_low.availability_mean();
    let mean_high = results_high.availability_mean();

    assert!(
        mean_low > mean_high,
        "a lower failure rate should mean higher availability: \
         MTBF 8h -> {mean_low:.6}, MTBF 2h -> {mean_high:.6}"
    );

    let analytical_low = analytical_availability(mtbf_low_rate, mttr);
    let analytical_high = analytical_availability(mtbf_high_rate, mttr);
    assert!((mean_low - analytical_low).abs() < 0.005);
    assert!((mean_high - analytical_high).abs() < 0.005);
}

#[test]
fn test_zero_variance_perfect_availability() {
    // Nodes that never fail give exactly 1.0 availability with no spread.
    let cfg = ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::constant(days(99999.0)))
        .recovery(Distribution::constant(0.0))
        .data_loss(Distribution::constant(days(99999.0)))
        .log_replay_rate(Distribution::constant(1e6))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(0.0))
        .build();

    let config = MonteCarloConfig {
        num_simulations: 20,
        max_time: Some(days(30.0)),
        stop_on_data_loss: false,
        base_seed: Some(0),
        log_events: false,
    }
    .validate()
    .unwrap();

    let cluster = || {
        let mut c = ClusterState::new(3);
        for i in 0..3 {
            c.add_named_node(&format!("node{i}"), cfg.clone());
        }
        c
    };
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::majority_available(1.0)) as Box<dyn Protocol>;
    let scenario = ScenarioFactory {
        cluster: &cluster,
        strategy: &strategy,
        protocol: &protocol,
        network_config: None,
    };

    let results = MonteCarloRunner::new(config).run(&scenario, None);
    assert_eq!(results.availability_mean(), 1.0);
    assert_eq!(results.availability_std(), 0.0);
}

// ==========================================================================
// Adaptive Monte Carlo
// ==========================================================================

/// Port of Python's `_make_fast_cluster`: short runs for convergence tests.
fn make_fast_cluster() -> ClusterState {
    let cfg = ConfigBuilder::new()
        .region("us-east")
        .cost(1.0)
        .failure(Distribution::exponential(1.0 / hours(24.0)).unwrap())
        .recovery(Distribution::constant(minutes(5.0)))
        .data_loss(Distribution::exponential(1.0 / days(365.0)).unwrap())
        .log_replay_rate(Distribution::constant(2.0))
        .snapshot_download(Distribution::constant(0.0))
        .spawn(Distribution::constant(minutes(10.0)))
        .build();

    let mut cluster = ClusterState::new(3);
    for i in 0..3 {
        cluster.add_named_node(&format!("node{i}"), cfg.clone());
    }
    cluster
}

fn fast_scenario<'a>(
    cluster: &'a dyn Fn() -> ClusterState,
    strategy: &'a dyn Fn() -> Box<dyn ClusterStrategy>,
    protocol: &'a dyn Fn() -> Box<dyn Protocol>,
) -> ScenarioFactory<'a> {
    ScenarioFactory {
        cluster,
        strategy,
        protocol,
        network_config: None,
    }
}

fn converge_config() -> MonteCarloConfig {
    MonteCarloConfig {
        num_simulations: 10_000,
        max_time: Some(days(30.0)),
        stop_on_data_loss: true,
        base_seed: Some(42),
        log_events: false,
    }
}

#[test]
fn test_run_until_converged_basic() {
    let runner = MonteCarloRunner::new(converge_config());
    let convergence = ConvergenceCriteria {
        confidence_level: 0.95,
        // 10% relative error is easy to reach.
        relative_error: Some(0.10),
        absolute_error: None,
        metrics: vec![ConvergenceMetric::Availability],
        min_runs: 10,
        max_runs: 500,
        batch_size: 10,
    }
    .validate()
    .unwrap();

    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
    let result = runner.run_until_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        &convergence,
        None,
    );

    assert!(result.total_runs >= convergence.min_runs);
    assert!(result.total_runs <= convergence.max_runs);
    assert_eq!(result.results.availability_samples.len(), result.total_runs);
    assert_eq!(result.metric_statuses.len(), 1);
    assert_eq!(
        result.metric_statuses[0].metric,
        ConvergenceMetric::Availability
    );
}

#[test]
fn test_convergence_reduces_ci() {
    let runner = MonteCarloRunner::new(converge_config());
    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
    let scenario = fast_scenario(&cluster, &strategy, &protocol);

    let loose = ConvergenceCriteria {
        relative_error: Some(0.20),
        min_runs: 10,
        max_runs: 500,
        batch_size: 10,
        ..Default::default()
    }
    .validate()
    .unwrap();
    let tight = ConvergenceCriteria {
        relative_error: Some(0.03),
        min_runs: 10,
        max_runs: 5000,
        batch_size: 20,
        ..Default::default()
    }
    .validate()
    .unwrap();

    let result_loose = runner.run_until_converged(&scenario, &loose, None);
    let result_tight = runner.run_until_converged(&scenario, &tight, None);

    assert!(result_tight.total_runs >= result_loose.total_runs);
}

#[test]
fn test_max_runs_respected() {
    // The cap must hold even when convergence is unreachable.
    //
    // Python pairs a 0.001 relative error with its fast cluster, which only
    // fails to converge because of the particular spread NumPy's stream
    // produces in the first few batches.  A threshold that tight is within
    // reach of a lucky draw, so the port uses a scenario with real variance
    // (MTBF 4 h over a 7-day window) and a threshold no finite sample can
    // meet unless every run is bit-identical.
    let config = MonteCarloConfig {
        num_simulations: 10_000,
        max_time: Some(sim_duration()),
        stop_on_data_loss: false,
        base_seed: Some(42),
        log_events: false,
    }
    .validate()
    .unwrap();
    let runner = MonteCarloRunner::new(config);

    let convergence = ConvergenceCriteria {
        confidence_level: 0.99,
        relative_error: Some(1e-9),
        absolute_error: None,
        metrics: vec![ConvergenceMetric::Availability],
        min_runs: 5,
        max_runs: 20,
        batch_size: 5,
    }
    .validate()
    .unwrap();

    let cluster = || make_simple_cluster(mtbf(), mttr(), 3);
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::majority_available(1.0)) as Box<dyn Protocol>;
    let result = runner.run_until_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        &convergence,
        None,
    );

    assert!(!result.converged);
    assert_eq!(result.total_runs, convergence.max_runs);
}

#[test]
fn test_multiple_metrics() {
    let runner = MonteCarloRunner::new(converge_config());
    let convergence = ConvergenceCriteria {
        confidence_level: 0.95,
        relative_error: Some(0.10),
        absolute_error: None,
        metrics: vec![ConvergenceMetric::Availability, ConvergenceMetric::Cost],
        min_runs: 10,
        max_runs: 500,
        batch_size: 10,
    }
    .validate()
    .unwrap();

    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
    let result = runner.run_until_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        &convergence,
        None,
    );

    assert_eq!(result.metric_statuses.len(), 2);
    let metrics: Vec<ConvergenceMetric> =
        result.metric_statuses.iter().map(|s| s.metric).collect();
    assert!(metrics.contains(&ConvergenceMetric::Availability));
    assert!(metrics.contains(&ConvergenceMetric::Cost));
}

#[test]
fn test_progress_callback_called() {
    let runner = MonteCarloRunner::new(converge_config());
    let convergence = ConvergenceCriteria {
        relative_error: Some(0.20),
        min_runs: 10,
        max_runs: 100,
        batch_size: 10,
        ..Default::default()
    }
    .validate()
    .unwrap();

    let mut progress_calls: Vec<(usize, usize, bool)> = Vec::new();
    {
        let mut on_progress = |completed: usize, estimated_total: usize, converged: bool| {
            progress_calls.push((completed, estimated_total, converged));
        };

        let cluster = make_fast_cluster;
        let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
        let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
        runner.run_until_converged(
            &fast_scenario(&cluster, &strategy, &protocol),
            &convergence,
            Some(&mut on_progress),
        );
    }

    assert!(!progress_calls.is_empty());
    // The first report lands after the minimum batch.
    assert_eq!(progress_calls[0].0, convergence.min_runs);
    // Completed counts never go backwards.
    let completed: Vec<usize> = progress_calls.iter().map(|c| c.0).collect();
    let mut sorted = completed.clone();
    sorted.sort_unstable();
    assert_eq!(completed, sorted);
}

#[test]
fn test_estimate_required_runs() {
    let pilot = {
        let config = MonteCarloConfig {
            num_simulations: 30,
            max_time: Some(days(30.0)),
            stop_on_data_loss: true,
            base_seed: Some(42),
            log_events: false,
        };
        let cluster = make_fast_cluster;
        let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
        let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
        MonteCarloRunner::new(config).run(&fast_scenario(&cluster, &strategy, &protocol), None)
    };

    let convergence = ConvergenceCriteria {
        confidence_level: 0.95,
        relative_error: Some(0.05),
        ..Default::default()
    }
    .validate()
    .unwrap();

    let estimates = estimate_required_runs(&pilot, &convergence);
    let availability = estimates
        .iter()
        .find(|(m, _)| *m == ConvergenceMetric::Availability)
        .expect("availability should be estimated");
    assert!(availability.1 >= 30);
}

#[test]
fn test_absolute_error_convergence() {
    let runner = MonteCarloRunner::new(converge_config());
    let convergence = ConvergenceCriteria {
        confidence_level: 0.95,
        // Plus or minus five percentage points on availability.
        relative_error: None,
        absolute_error: Some(0.05),
        metrics: vec![ConvergenceMetric::Availability],
        min_runs: 10,
        max_runs: 2000,
        batch_size: 20,
    }
    .validate()
    .unwrap();

    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
    let result = runner.run_until_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        &convergence,
        None,
    );

    assert!(result.total_runs >= convergence.min_runs);
    if result.converged {
        for status in &result.metric_statuses {
            assert!(status.ci_half_width <= convergence.absolute_error.unwrap());
        }
    }
}

#[test]
fn test_absolute_error_tighter_needs_more_runs() {
    let runner = MonteCarloRunner::new(converge_config());
    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
    let scenario = fast_scenario(&cluster, &strategy, &protocol);

    let loose = ConvergenceCriteria {
        relative_error: None,
        absolute_error: Some(0.10),
        min_runs: 10,
        max_runs: 2000,
        batch_size: 10,
        ..Default::default()
    }
    .validate()
    .unwrap();
    let tight = ConvergenceCriteria {
        relative_error: None,
        absolute_error: Some(0.02),
        min_runs: 10,
        max_runs: 5000,
        batch_size: 20,
        ..Default::default()
    }
    .validate()
    .unwrap();

    let result_loose = runner.run_until_converged(&scenario, &loose, None);
    let result_tight = runner.run_until_converged(&scenario, &tight, None);
    assert!(result_tight.total_runs >= result_loose.total_runs);
}

#[test]
fn test_absolute_error_estimate_required_runs() {
    let pilot = {
        let config = MonteCarloConfig {
            num_simulations: 30,
            max_time: Some(days(30.0)),
            stop_on_data_loss: true,
            base_seed: Some(42),
            log_events: false,
        };
        let cluster = make_fast_cluster;
        let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
        let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;
        MonteCarloRunner::new(config).run(&fast_scenario(&cluster, &strategy, &protocol), None)
    };

    let criteria = ConvergenceCriteria {
        confidence_level: 0.95,
        relative_error: None,
        // Very tight.
        absolute_error: Some(0.01),
        ..Default::default()
    }
    .validate()
    .unwrap();

    let estimates = estimate_required_runs(&pilot, &criteria);
    let availability = estimates
        .iter()
        .find(|(m, _)| *m == ConvergenceMetric::Availability)
        .expect("availability should be estimated");
    assert!(availability.1 >= 30);
}

// ==========================================================================
// Convergence criteria validation
// ==========================================================================

#[test]
fn test_defaults() {
    let criteria = ConvergenceCriteria::default().validate().unwrap();
    assert_eq!(criteria.confidence_level, 0.95);
    assert_eq!(criteria.relative_error, Some(0.05));
    assert_eq!(criteria.absolute_error, None);
    assert!(!criteria.uses_absolute_error());
    assert_eq!(criteria.metrics, vec![ConvergenceMetric::Availability]);
    assert_eq!(criteria.min_runs, 30);
    assert_eq!(criteria.max_runs, 10_000);
    assert_eq!(criteria.batch_size, 10);
}

#[test]
fn test_absolute_error_mode() {
    let criteria = ConvergenceCriteria {
        relative_error: None,
        absolute_error: Some(0.01),
        ..Default::default()
    }
    .validate()
    .unwrap();
    assert_eq!(criteria.absolute_error, Some(0.01));
    assert_eq!(criteria.relative_error, None);
    assert!(criteria.uses_absolute_error());
    assert_eq!(criteria.error_threshold(), 0.01);
}

#[test]
fn test_relative_error_mode() {
    let criteria = ConvergenceCriteria {
        relative_error: Some(0.10),
        ..Default::default()
    }
    .validate()
    .unwrap();
    assert_eq!(criteria.relative_error, Some(0.10));
    assert_eq!(criteria.absolute_error, None);
    assert!(!criteria.uses_absolute_error());
    assert_eq!(criteria.error_threshold(), 0.10);
}

#[test]
fn test_cannot_specify_both_errors() {
    let err = ConvergenceCriteria {
        relative_error: Some(0.05),
        absolute_error: Some(0.01),
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert!(err.0.contains("exactly one"));
}

#[test]
fn test_invalid_confidence_level() {
    for level in [0.0, 1.0, 1.5] {
        let err = ConvergenceCriteria {
            confidence_level: level,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.0.contains("confidence_level"));
    }
}

#[test]
fn test_invalid_relative_error() {
    for rel in [0.0, -0.1] {
        let err = ConvergenceCriteria {
            relative_error: Some(rel),
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.0.contains("relative_error"));
    }
}

#[test]
fn test_invalid_absolute_error() {
    for abs in [0.0, -0.01] {
        let err = ConvergenceCriteria {
            relative_error: None,
            absolute_error: Some(abs),
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(err.0.contains("absolute_error"));
    }
}

#[test]
fn test_invalid_min_runs() {
    let err = ConvergenceCriteria {
        min_runs: 1,
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert!(err.0.contains("min_runs"));
}

#[test]
fn test_invalid_max_runs() {
    let err = ConvergenceCriteria {
        min_runs: 100,
        max_runs: 50,
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert!(err.0.contains("max_runs"));
}

#[test]
fn test_convergence_status_fields_are_populated() {
    let results = run_simple(mtbf(), mttr(), 40, days(1.0), 7);
    let criteria = ConvergenceCriteria {
        metrics: vec![ConvergenceMetric::Availability, ConvergenceMetric::Cost],
        ..Default::default()
    }
    .validate()
    .unwrap();

    for status in check_convergence(&results, &criteria) {
        assert_eq!(status.num_samples, 40);
        assert!(status.ci_half_width.is_finite());
        assert!(status.estimated_runs_needed >= 40);
    }
}


#[test]
fn test_convenience_function() {
    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;

    let result = run_monte_carlo_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        Some(days(30.0)),
        true,
        Some(42),
        ConvergenceCriteria {
            confidence_level: 0.95,
            relative_error: Some(0.15),
            min_runs: 10,
            max_runs: 200,
            batch_size: 10,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert!(result.total_runs >= 10);
    assert_eq!(result.results.availability_samples.len(), result.total_runs);
}

#[test]
fn test_convergence_summary() {
    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;

    let result = run_monte_carlo_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        Some(days(30.0)),
        true,
        Some(42),
        ConvergenceCriteria {
            confidence_level: 0.95,
            relative_error: Some(0.15),
            min_runs: 10,
            max_runs: 200,
            batch_size: 10,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let summary = result.summary();
    assert!(summary.contains("Convergence:"));
    assert!(summary.contains("availability"));
    assert!(
        summary.to_lowercase().contains("runs")
            || summary.contains(&result.total_runs.to_string())
    );
}

#[test]
fn test_absolute_error_convenience_function() {
    let cluster = make_fast_cluster;
    let strategy = || Box::new(NoOpStrategy) as Box<dyn ClusterStrategy>;
    let protocol = || Box::new(LeaderlessProtocol::default()) as Box<dyn Protocol>;

    let result = run_monte_carlo_converged(
        &fast_scenario(&cluster, &strategy, &protocol),
        Some(days(30.0)),
        true,
        Some(42),
        ConvergenceCriteria {
            relative_error: None,
            absolute_error: Some(0.05),
            min_runs: 10,
            max_runs: 500,
            batch_size: 10,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert!(result.total_runs >= 10);
}

/// Python asserts `LeaderlessUpToDateQuorumProtocol is LeaderlessProtocol`,
/// i.e. the alias is the same class.  The port's equivalent is that the
/// named constructor builds the same type with the flag set, rather than a
/// separate protocol.
#[test]
fn test_is_alias_of_up_to_date_quorum_protocol() {
    let aliased: LeaderlessProtocol = LeaderlessProtocol::up_to_date_quorum_protocol(1.0);
    let plain = LeaderlessProtocol::default();
    assert!(aliased.up_to_date_quorum());
    assert_eq!(aliased.up_to_date_quorum(), plain.up_to_date_quorum());
    assert_eq!(aliased.commit_rate(), plain.commit_rate());
}

// ==========================================================================
// Simulator reuse
// ==========================================================================

/// The runner re-arms one simulator across the runs of an experiment
/// rather than building a fresh one each time.  That is only sound if a
/// reset leaves nothing behind, so each run must match what an
/// independently constructed simulator produces for the same seed.
#[test]
fn reused_simulator_matches_independent_runs() {
    use powder_mc::sim::simulator::Simulator;

    // Each case names itself and says whether a replacement strategy is in
    // play; the cluster is built below from a shared config.
    for (label, replacement) in [("leaderless", false), ("replacement", true)] {
        let cfg = ConfigBuilder::new()
            .region("us-east")
            .cost(1.0)
            .failure(Distribution::exponential(1.0 / hours(6.0)).unwrap())
            .recovery(Distribution::exponential(1.0 / minutes(20.0)).unwrap())
            .data_loss(Distribution::exponential(1.0 / days(45.0)).unwrap())
            .log_replay_rate(Distribution::constant(500.0))
            .snapshot_download(Distribution::constant(0.0))
            .spawn(Distribution::constant(minutes(3.0)))
            .build();

        let cluster_for_runner = {
            let cfg = cfg.clone();
            move || {
                let mut c = ClusterState::new(5);
                for i in 0..5 {
                    c.add_named_node(&format!("node{i}"), cfg.clone());
                }
                c
            }
        };
        let strategy = {
            let cfg = cfg.clone();
            move || -> Box<dyn ClusterStrategy> {
                if replacement {
                    Box::new(powder_mc::sim::strategy::NodeReplacementStrategy::new(
                        minutes(30.0),
                        Some(cfg.clone()),
                        true,
                    ))
                } else {
                    Box::new(NoOpStrategy)
                }
            }
        };
        let protocol = || Box::new(LeaderlessProtocol::majority_available(1.0)) as Box<dyn Protocol>;

        let num_runs = 25;
        let base_seed = 4242u64;

        // Through the runner, which reuses one simulator.
        let config = MonteCarloConfig {
            num_simulations: num_runs,
            max_time: Some(days(14.0)),
            stop_on_data_loss: false,
            base_seed: Some(base_seed),
            log_events: false,
        }
        .validate()
        .unwrap();
        let pooled = MonteCarloRunner::new(config).run(
            &ScenarioFactory {
                cluster: &cluster_for_runner,
                strategy: &strategy,
                protocol: &protocol,
                network_config: None,
            },
            None,
        );

        // One fresh simulator per run, as a reference.
        for i in 0..num_runs {
            let mut sim = Simulator::new(
                cluster_for_runner(),
                strategy(),
                protocol(),
                None,
                Some(base_seed + i as u64),
                false,
            );
            let independent = sim.run_until(Some(days(14.0)), None);
            let m = independent.metrics;

            assert_eq!(
                pooled.availability_samples[i],
                m.availability_fraction(),
                "{label} run {i}: availability differs after reuse"
            );
            assert_eq!(
                pooled.cost_samples[i], m.total_cost,
                "{label} run {i}: cost differs after reuse"
            );
            assert_eq!(
                pooled.transient_failure_samples[i], m.total_transient_failures,
                "{label} run {i}: failure count differs after reuse"
            );
            assert_eq!(
                pooled.nodes_spawned_samples[i], m.total_nodes_spawned,
                "{label} run {i}: spawn count differs after reuse"
            );
            assert_eq!(
                pooled.time_to_actual_loss_samples[i], m.time_to_actual_data_loss,
                "{label} run {i}: data loss time differs after reuse"
            );
            assert_eq!(
                pooled.end_reasons[i], independent.end_reason,
                "{label} run {i}: end reason differs after reuse"
            );
        }
    }
}

/// A reset must not carry the previous run's event queue, metrics or
/// cluster mutations forward.  Running the *same* seed twice through one
/// simulator has to give the same answer both times.
#[test]
fn resetting_with_the_same_seed_repeats_the_run() {
    use powder_mc::sim::simulator::Simulator;

    let template = make_fast_cluster();
    let mut sim = Simulator::new(
        template.clone(),
        Box::new(NoOpStrategy),
        Box::new(LeaderlessProtocol::default()),
        None,
        None,
        false,
    );

    let mut previous = None;
    for _ in 0..3 {
        sim.reset(
            &template,
            Box::new(NoOpStrategy),
            Box::new(LeaderlessProtocol::default()),
            Some(31337),
        );
        let result = sim.run_until(Some(days(30.0)), None);
        let summary = (
            result.end_reason,
            result.end_time,
            result.metrics.time_available,
            result.metrics.total_cost,
            result.metrics.total_transient_failures,
        );
        if let Some(first) = &previous {
            assert_eq!(&summary, first, "a reset leaked state into the next run");
        }
        previous = Some(summary);
    }
}
