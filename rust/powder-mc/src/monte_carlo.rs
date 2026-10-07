//! Monte Carlo driver: repeated simulation runs and aggregate statistics.
//!
//! Port of `powder/monte_carlo.py`.  Supports both a fixed run count and
//! adaptive convergence, where runs are added in batches until the
//! confidence interval for every target metric is inside the requested
//! error tolerance.
//!
//! **Threading.** A single experiment runs entirely on the calling thread --
//! `parallel_workers` from the Python config has no analogue here.
//! Parallelism lives one level up, in the binary's worker pool, which runs
//! whole jobs concurrently.  That keeps each experiment's sample ordering
//! deterministic, which the Python parallel path does not manage.

use crate::sim::cluster::ClusterState;
use crate::sim::distributions::Seconds;
use crate::sim::metrics::MetricsSnapshot;
use crate::sim::network::NetworkConfig;
use crate::sim::protocol::Protocol;
use crate::sim::simulator::{EndReason, SimulationResult, Simulator};
use crate::sim::strategy::ClusterStrategy;
use crate::stats;
use crate::stats::{norm_ppf, t_ppf};

/// Metrics that convergence can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvergenceMetric {
    /// Mean availability fraction.
    Availability,
    /// Fraction of runs that lost data.
    DataLossProbability,
    /// Mean total cost.
    Cost,
    /// Mean time to data loss, over the runs that lost data.
    MeanTimeToDataLoss,
    /// Mean count of transient failures.
    TransientFailures,
    /// Mean count of data-loss failures.
    DatalossFailures,
    /// Mean count of nodes spawned.
    NodesSpawned,
    /// Mean count of unavailability incidents.
    UnavailabilityIncidents,
    /// Mean count of leader elections.
    LeaderElections,
}

impl ConvergenceMetric {
    /// The string Python uses for this metric.
    pub fn as_str(self) -> &'static str {
        match self {
            ConvergenceMetric::Availability => "availability",
            ConvergenceMetric::DataLossProbability => "data_loss_probability",
            ConvergenceMetric::Cost => "cost",
            ConvergenceMetric::MeanTimeToDataLoss => "mean_time_to_data_loss",
            ConvergenceMetric::TransientFailures => "transient_failures",
            ConvergenceMetric::DatalossFailures => "dataloss_failures",
            ConvergenceMetric::NodesSpawned => "nodes_spawned",
            ConvergenceMetric::UnavailabilityIncidents => "unavailability_incidents",
            ConvergenceMetric::LeaderElections => "leader_elections",
        }
    }

    /// Parse the Python metric name.
    ///
    /// Deliberately not `FromStr`: this maps the exact strings the Python
    /// config uses, not a general-purpose parse.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "availability" => ConvergenceMetric::Availability,
            "data_loss_probability" => ConvergenceMetric::DataLossProbability,
            "cost" => ConvergenceMetric::Cost,
            "mean_time_to_data_loss" => ConvergenceMetric::MeanTimeToDataLoss,
            "transient_failures" => ConvergenceMetric::TransientFailures,
            "dataloss_failures" => ConvergenceMetric::DatalossFailures,
            "nodes_spawned" => ConvergenceMetric::NodesSpawned,
            "unavailability_incidents" => ConvergenceMetric::UnavailabilityIncidents,
            "leader_elections" => ConvergenceMetric::LeaderElections,
            _ => return None,
        })
    }
}

/// A rejected convergence configuration.  Mirrors Python's `ValueError`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Stopping rule for adaptive Monte Carlo.
///
/// Exactly one of `relative_error` and `absolute_error` is active:
///
/// * **relative**: the CI half-width as a fraction of the mean,
///   `n ~ (z * sigma / (eps * mu))^2`.
/// * **absolute**: the CI half-width in the metric's own units,
///   `n ~ (z * sigma / E)^2`.  Use this to pin availability to a given
///   number of decimal places.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvergenceCriteria {
    /// Desired confidence level, e.g. 0.95.
    pub confidence_level: f64,
    /// Maximum relative CI half-width, if in relative mode.
    pub relative_error: Option<f64>,
    /// Maximum absolute CI half-width, if in absolute mode.
    pub absolute_error: Option<f64>,
    /// Metrics that must all converge before stopping.
    pub metrics: Vec<ConvergenceMetric>,
    /// Runs to complete before the first convergence check.  At least 2, so
    /// a variance can be estimated.
    pub min_runs: usize,
    /// Safety cap on total runs.
    pub max_runs: usize,
    /// Runs per batch between checks.
    pub batch_size: usize,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        ConvergenceCriteria {
            confidence_level: 0.95,
            relative_error: Some(0.05),
            absolute_error: None,
            metrics: vec![ConvergenceMetric::Availability],
            min_runs: 30,
            max_runs: 10_000,
            batch_size: 10,
        }
    }
}

impl ConvergenceCriteria {
    /// Validate the criteria, applying the same defaulting Python does in
    /// `__post_init__`: with neither error mode given, relative 0.05.
    pub fn validate(mut self) -> Result<Self, ConfigError> {
        if !(self.confidence_level > 0.0 && self.confidence_level < 1.0) {
            return Err(ConfigError(format!(
                "confidence_level must be in (0, 1), got {}",
                self.confidence_level
            )));
        }

        if self.relative_error.is_none() && self.absolute_error.is_none() {
            self.relative_error = Some(0.05);
        }

        if self.relative_error.is_some() && self.absolute_error.is_some() {
            return Err(ConfigError(
                "Specify exactly one of relative_error or absolute_error, not both".to_string(),
            ));
        }

        if let Some(rel) = self.relative_error {
            if rel <= 0.0 {
                return Err(ConfigError(format!(
                    "relative_error must be > 0, got {rel}"
                )));
            }
        }
        if let Some(abs) = self.absolute_error {
            if abs <= 0.0 {
                return Err(ConfigError(format!(
                    "absolute_error must be > 0, got {abs}"
                )));
            }
        }

        if self.min_runs < 2 {
            return Err(ConfigError(format!(
                "min_runs must be >= 2 for variance estimation, got {}",
                self.min_runs
            )));
        }
        if self.max_runs < self.min_runs {
            return Err(ConfigError(format!(
                "max_runs ({}) must be >= min_runs ({})",
                self.max_runs, self.min_runs
            )));
        }

        Ok(self)
    }

    /// Whether the absolute error mode is active.
    pub fn uses_absolute_error(&self) -> bool {
        self.absolute_error.is_some()
    }

    /// The active error threshold, whichever mode is in use.
    pub fn error_threshold(&self) -> f64 {
        self.absolute_error
            .or(self.relative_error)
            .expect("validate() guarantees one mode is set")
    }
}

/// Convergence state for one metric.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricConvergenceStatus {
    /// Which metric this describes.
    pub metric: ConvergenceMetric,
    /// Whether it has converged.
    pub converged: bool,
    /// Current sample mean.
    pub current_mean: f64,
    /// Current sample standard deviation.
    pub current_std: f64,
    /// Current CI half-width, in metric units.
    pub ci_half_width: f64,
    /// Current relative error, `ci_half_width / |mean|`.
    pub relative_error: f64,
    /// Estimated total runs needed to converge.
    pub estimated_runs_needed: usize,
    /// Samples collected so far.
    pub num_samples: usize,
}

impl MetricConvergenceStatus {
    fn unconverged(metric: ConvergenceMetric, num_samples: usize) -> Self {
        MetricConvergenceStatus {
            metric,
            converged: false,
            current_mean: 0.0,
            current_std: 0.0,
            ci_half_width: f64::INFINITY,
            relative_error: f64::INFINITY,
            estimated_runs_needed: 0,
            num_samples,
        }
    }
}

/// Outcome of an adaptive convergence run.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Aggregated results across every run.
    pub results: MonteCarloResults,
    /// Whether every target metric converged.
    pub converged: bool,
    /// Runs executed.
    pub total_runs: usize,
    /// Per-metric convergence state at the end.
    pub metric_statuses: Vec<MetricConvergenceStatus>,
}

impl ConvergenceResult {
    /// A human-readable summary, mirroring Python's `ConvergenceResult.summary()`.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            self.results.summary(),
            String::new(),
            format!(
                "Convergence: {} ({} runs)",
                if self.converged { "yes" } else { "NO" },
                self.total_runs
            ),
        ];

        for status in &self.metric_statuses {
            let symbol = if status.converged { "+" } else { "-" };
            let ci_lo = status.current_mean - status.ci_half_width;
            let ci_hi = status.current_mean + status.ci_half_width;

            if status.metric == ConvergenceMetric::MeanTimeToDataLoss {
                // MTTDL reads better in days.
                lines.push(format!(
                    "  [{symbol}] {}: mean={:.1} days, CI=[{:.1}, {:.1}] days, \
                     +/-{:.1} days (rel={:.4}), est_n={}",
                    status.metric.as_str(),
                    status.current_mean / 86400.0,
                    ci_lo / 86400.0,
                    ci_hi / 86400.0,
                    status.ci_half_width / 86400.0,
                    status.relative_error,
                    status.estimated_runs_needed
                ));
            } else {
                lines.push(format!(
                    "  [{symbol}] {}: mean={:.8}, CI=[{:.8}, {:.8}], \
                     +/-{:.8} (rel={:.8}), est_n={}",
                    status.metric.as_str(),
                    status.current_mean,
                    ci_lo,
                    ci_hi,
                    status.ci_half_width,
                    status.relative_error,
                    status.estimated_runs_needed
                ));
            }
        }

        lines.join("\n")
    }
}

/// Configuration for a Monte Carlo experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloConfig {
    /// Number of runs to execute.
    pub num_simulations: usize,
    /// Time limit per run.  `None` with `stop_on_data_loss` runs each
    /// simulation until data loss, which is what MTTDL estimation needs.
    pub max_time: Option<Seconds>,
    /// Whether to stop a run when data is lost.
    pub stop_on_data_loss: bool,
    /// Base seed; run `i` uses `base_seed + i`.
    pub base_seed: Option<u64>,
    /// Whether to retain each run's event log.
    pub log_events: bool,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        MonteCarloConfig {
            num_simulations: 100,
            max_time: None,
            stop_on_data_loss: true,
            base_seed: None,
            log_events: false,
        }
    }
}

impl MonteCarloConfig {
    /// Reject configurations that would run forever.
    pub fn validate(self) -> Result<Self, ConfigError> {
        if self.max_time.is_none() && !self.stop_on_data_loss {
            return Err(ConfigError(
                "max_time is required when stop_on_data_loss is False, \
                 otherwise simulations would run indefinitely"
                    .to_string(),
            ));
        }
        Ok(self)
    }
}

/// Per-run samples and the statistics derived from them.
#[derive(Debug, Clone, Default)]
pub struct MonteCarloResults {
    /// Availability fraction per run.
    pub availability_samples: Vec<f64>,
    /// Time to potential data loss per run, where it occurred.
    pub time_to_potential_loss_samples: Vec<Option<Seconds>>,
    /// Time to actual data loss per run, where it occurred.
    pub time_to_actual_loss_samples: Vec<Option<Seconds>>,
    /// Total cost per run.
    pub cost_samples: Vec<f64>,
    /// Why each run ended.
    pub end_reasons: Vec<EndReason>,
    /// Transient failure count per run.
    pub transient_failure_samples: Vec<u64>,
    /// Data-loss failure count per run.
    pub dataloss_failure_samples: Vec<u64>,
    /// Nodes spawned per run.
    pub nodes_spawned_samples: Vec<u64>,
    /// Unavailability incidents per run.
    pub unavailability_incident_samples: Vec<u64>,
    /// Leader elections per run.
    pub leader_election_samples: Vec<u64>,
    /// Time to first unavailability per run, where it occurred.
    pub time_to_first_unavailability_samples: Vec<Option<Seconds>>,
    /// Total simulated time per run.
    pub total_time_samples: Vec<Seconds>,
}

impl MonteCarloResults {
    /// An empty result set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of runs collected.
    pub fn len(&self) -> usize {
        self.availability_samples.len()
    }

    /// Whether no runs have been collected.
    pub fn is_empty(&self) -> bool {
        self.availability_samples.is_empty()
    }

    /// Mean availability across runs.
    pub fn availability_mean(&self) -> f64 {
        stats::mean(&self.availability_samples)
    }

    /// Sample standard deviation of availability.
    pub fn availability_std(&self) -> f64 {
        stats::std(&self.availability_samples, 1)
    }

    /// A percentile of the availability samples.
    pub fn availability_percentile(&self, p: f64) -> f64 {
        stats::percentile(&self.availability_samples, p)
    }

    /// Mean cost across runs.
    pub fn cost_mean(&self) -> f64 {
        stats::mean(&self.cost_samples)
    }

    /// Actual-data-loss times, dropping the runs that never lost data.
    pub fn time_to_actual_loss_samples_filtered(&self) -> Vec<f64> {
        self.time_to_actual_loss_samples
            .iter()
            .filter_map(|t| *t)
            .collect()
    }

    /// Potential-data-loss times, dropping the runs that never lost quorum.
    pub fn time_to_potential_loss_samples_filtered(&self) -> Vec<f64> {
        self.time_to_potential_loss_samples
            .iter()
            .filter_map(|t| *t)
            .collect()
    }

    /// Fraction of runs that experienced actual data loss.
    pub fn data_loss_probability(&self) -> f64 {
        if self.time_to_actual_loss_samples.is_empty() {
            return 0.0;
        }
        let losses = self
            .time_to_actual_loss_samples
            .iter()
            .filter(|t| t.is_some())
            .count();
        losses as f64 / self.time_to_actual_loss_samples.len() as f64
    }

    /// Mean time to actual data loss, or `None` if no run lost data.
    pub fn mean_time_to_actual_loss(&self) -> Option<f64> {
        let filtered = self.time_to_actual_loss_samples_filtered();
        if filtered.is_empty() {
            return None;
        }
        Some(stats::mean(&filtered))
    }

    /// Standard deviation of time to actual data loss, or `None` with fewer
    /// than two loss events.
    pub fn std_time_to_actual_loss(&self) -> Option<f64> {
        let filtered = self.time_to_actual_loss_samples_filtered();
        if filtered.len() < 2 {
            return None;
        }
        Some(stats::std(&filtered, 1))
    }

    /// Confidence interval for the mean time to actual data loss, using the
    /// t-distribution.  `None` with fewer than two loss events.
    pub fn ci_time_to_actual_loss(&self, confidence_level: f64) -> Option<(f64, f64)> {
        let filtered = self.time_to_actual_loss_samples_filtered();
        let n = filtered.len();
        if n < 2 {
            return None;
        }
        let sample_mean = stats::mean(&filtered);
        let sample_std = stats::std(&filtered, 1);
        let alpha = 1.0 - confidence_level;
        let t_crit = t_ppf(1.0 - alpha / 2.0, (n - 1) as f64);
        let margin = t_crit * sample_std / (n as f64).sqrt();
        Some((sample_mean - margin, sample_mean + margin))
    }

    /// Mean time to potential data loss, or `None` if quorum was never lost.
    pub fn mean_time_to_potential_loss(&self) -> Option<f64> {
        let filtered = self.time_to_potential_loss_samples_filtered();
        if filtered.is_empty() {
            return None;
        }
        Some(stats::mean(&filtered))
    }

    /// PDF histogram of time to data loss.
    ///
    /// With `actual` set, uses actual-loss times; otherwise potential-loss.
    pub fn time_to_loss_pdf(&self, bins: usize, actual: bool) -> (Vec<f64>, Vec<f64>) {
        let samples = if actual {
            self.time_to_actual_loss_samples_filtered()
        } else {
            self.time_to_potential_loss_samples_filtered()
        };
        stats::histogram_density(&samples, bins)
    }

    /// Empirical CDF of time to data loss.
    pub fn time_to_loss_cdf(&self, actual: bool) -> (Vec<f64>, Vec<f64>) {
        let samples = if actual {
            self.time_to_actual_loss_samples_filtered()
        } else {
            self.time_to_potential_loss_samples_filtered()
        };
        stats::ecdf(&samples)
    }

    /// A human-readable summary, mirroring Python's `summary()`.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("Monte Carlo Results ({} runs)", self.len()),
            format!(
                "  Availability: {:.2}% (std: {:.2}%)",
                self.availability_mean() * 100.0,
                self.availability_std() * 100.0
            ),
            format!(
                "  Data loss probability: {:.1}%",
                self.data_loss_probability() * 100.0
            ),
        ];

        if let Some(mttl) = self.mean_time_to_actual_loss() {
            match self.ci_time_to_actual_loss(0.95) {
                Some((lo, hi)) => lines.push(format!(
                    "  Mean time to data loss: {:.1} days (95% CI: [{:.1}, {:.1}] days)",
                    mttl / 86400.0,
                    lo / 86400.0,
                    hi / 86400.0
                )),
                None => lines.push(format!(
                    "  Mean time to data loss: {:.1} days",
                    mttl / 86400.0
                )),
            }
        }

        lines.push(format!("  Mean cost: ${:.2}", self.cost_mean()));

        let counter_summary = |name: &str, samples: &[u64]| -> String {
            if samples.is_empty() {
                return format!("  {name}: N/A");
            }
            let xs: Vec<f64> = samples.iter().map(|&v| v as f64).collect();
            if xs.len() >= 2 {
                format!(
                    "  {name}: mean={:.2} (std={:.2})",
                    stats::mean(&xs),
                    stats::std(&xs, 1)
                )
            } else {
                format!("  {name}: mean={:.2}", stats::mean(&xs))
            }
        };

        lines.push(counter_summary(
            "Transient failures",
            &self.transient_failure_samples,
        ));
        lines.push(counter_summary(
            "Dataloss failures",
            &self.dataloss_failure_samples,
        ));
        lines.push(counter_summary("Nodes spawned", &self.nodes_spawned_samples));
        lines.push(counter_summary(
            "Unavailability incidents",
            &self.unavailability_incident_samples,
        ));
        lines.push(counter_summary(
            "Leader elections",
            &self.leader_election_samples,
        ));

        lines.join("\n")
    }

    /// Append one run's metrics.
    pub fn collect(&mut self, result: &SimulationResult) {
        let m: MetricsSnapshot = result.metrics;
        self.availability_samples.push(m.availability_fraction());
        self.time_to_potential_loss_samples
            .push(m.time_to_potential_data_loss);
        self.time_to_actual_loss_samples
            .push(m.time_to_actual_data_loss);
        self.cost_samples.push(m.total_cost);
        self.end_reasons.push(result.end_reason);
        self.transient_failure_samples
            .push(m.total_transient_failures);
        self.dataloss_failure_samples
            .push(m.total_dataloss_failures);
        self.nodes_spawned_samples.push(m.total_nodes_spawned);
        self.unavailability_incident_samples
            .push(m.total_unavailability_incidents);
        self.leader_election_samples.push(m.total_leader_elections);
        self.time_to_first_unavailability_samples
            .push(m.time_to_first_unavailability);
        self.total_time_samples.push(m.total_time());
    }

    /// Samples for a convergence metric, as floats.
    pub fn metric_samples(&self, metric: ConvergenceMetric) -> Vec<f64> {
        let as_floats = |xs: &[u64]| xs.iter().map(|&v| v as f64).collect();
        match metric {
            ConvergenceMetric::Availability => self.availability_samples.clone(),
            ConvergenceMetric::Cost => self.cost_samples.clone(),
            ConvergenceMetric::DataLossProbability => self
                .time_to_actual_loss_samples
                .iter()
                .map(|t| if t.is_some() { 1.0 } else { 0.0 })
                .collect(),
            // Only the runs that actually lost data carry a time.
            ConvergenceMetric::MeanTimeToDataLoss => self.time_to_actual_loss_samples_filtered(),
            ConvergenceMetric::TransientFailures => as_floats(&self.transient_failure_samples),
            ConvergenceMetric::DatalossFailures => as_floats(&self.dataloss_failure_samples),
            ConvergenceMetric::NodesSpawned => as_floats(&self.nodes_spawned_samples),
            ConvergenceMetric::UnavailabilityIncidents => {
                as_floats(&self.unavailability_incident_samples)
            }
            ConvergenceMetric::LeaderElections => as_floats(&self.leader_election_samples),
        }
    }
}

/// Everything needed to build one simulation.
///
/// Python deep-copies the cluster, strategy and protocol for each run.  Rust
/// cannot clone a trait object, so the caller supplies factories instead --
/// which is also cheaper, since nothing is copied that would be overwritten.
pub struct ScenarioFactory<'a> {
    /// Builds a fresh cluster for each run.
    pub cluster: &'a dyn Fn() -> ClusterState,
    /// Builds a fresh strategy for each run.
    pub strategy: &'a dyn Fn() -> Box<dyn ClusterStrategy>,
    /// Builds a fresh protocol for each run.
    pub protocol: &'a dyn Fn() -> Box<dyn Protocol>,
    /// Network configuration, shared across runs.
    pub network_config: Option<NetworkConfig>,
}

/// Run one simulation under the given configuration and seed.
pub fn run_single_simulation(
    scenario: &ScenarioFactory<'_>,
    max_time: Option<Seconds>,
    stop_on_data_loss: bool,
    seed: Option<u64>,
    log_events: bool,
) -> SimulationResult {
    let mut simulator = Simulator::new(
        (scenario.cluster)(),
        (scenario.strategy)(),
        (scenario.protocol)(),
        scenario.network_config.clone(),
        seed,
        log_events,
    );

    if stop_on_data_loss {
        simulator.run_until_data_loss(max_time)
    } else {
        simulator.run_until(max_time, None)
    }
}

/// Runs simulations and aggregates their results.
#[derive(Debug, Clone)]
pub struct MonteCarloRunner {
    /// Experiment configuration.
    pub config: MonteCarloConfig,
}

impl MonteCarloRunner {
    /// Build a runner.
    pub fn new(config: MonteCarloConfig) -> Self {
        MonteCarloRunner { config }
    }

    /// Run `num_simulations` simulations and aggregate.
    ///
    /// `progress` is called with `(completed, total)` after each run.
    pub fn run(
        &self,
        scenario: &ScenarioFactory<'_>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> MonteCarloResults {
        let mut results = MonteCarloResults::new();
        self.run_batch(scenario, &mut results, self.config.num_simulations, 0);
        if let Some(callback) = progress.as_mut() {
            callback(self.config.num_simulations, self.config.num_simulations);
        }
        results
    }

    /// Run a batch of simulations, appending to `results`.
    ///
    /// Seeds continue from `start_index`, so successive batches never reuse
    /// a seed.
    fn run_batch(
        &self,
        scenario: &ScenarioFactory<'_>,
        results: &mut MonteCarloResults,
        num_runs: usize,
        start_index: usize,
    ) {
        if num_runs == 0 {
            return;
        }

        // One simulator for the whole batch, re-armed between runs.  Every
        // run starts from the same cluster, so the template is built once
        // and each run resets onto it -- that is what keeps the event
        // heap, cancellation tables and scratch buffers from being
        // reallocated thousands of times.
        let template = (scenario.cluster)();
        let mut simulator = Simulator::new(
            template.clone(),
            (scenario.strategy)(),
            (scenario.protocol)(),
            scenario.network_config.clone(),
            None,
            self.config.log_events,
        );

        for i in 0..num_runs {
            let seed = self.config.base_seed.map(|base| base + (start_index + i) as u64);
            simulator.reset(
                &template,
                (scenario.strategy)(),
                (scenario.protocol)(),
                seed,
            );

            let sim_result = if self.config.stop_on_data_loss {
                simulator.run_until_data_loss(self.config.max_time)
            } else {
                simulator.run_until(self.config.max_time, None)
            };
            results.collect(&sim_result);
        }
    }

    /// Run in batches until every target metric converges or `max_runs` is
    /// reached.
    ///
    /// `progress` receives `(completed, estimated_total, converged)` after
    /// each batch; the estimate shifts as the variance estimate improves.
    pub fn run_until_converged(
        &self,
        scenario: &ScenarioFactory<'_>,
        convergence: &ConvergenceCriteria,
        mut progress: Option<&mut dyn FnMut(usize, usize, bool)>,
    ) -> ConvergenceResult {
        let mut results = MonteCarloResults::new();
        let mut run_count = 0usize;

        // Phase one: the minimum batch, so there is a variance to work with.
        self.run_batch(scenario, &mut results, convergence.min_runs, run_count);
        run_count += convergence.min_runs;

        let mut statuses = check_convergence(&results, convergence);
        let mut all_converged = statuses.iter().all(|s| s.converged);

        if let Some(callback) = progress.as_mut() {
            let estimated_total = statuses
                .iter()
                .map(|s| s.estimated_runs_needed)
                .max()
                .unwrap_or(run_count);
            callback(run_count, estimated_total, all_converged);
        }

        // Phase two: keep adding batches until converged or capped.
        while !all_converged && run_count < convergence.max_runs {
            let batch = convergence.batch_size.min(convergence.max_runs - run_count);
            if batch == 0 {
                break;
            }

            self.run_batch(scenario, &mut results, batch, run_count);
            run_count += batch;

            statuses = check_convergence(&results, convergence);
            all_converged = statuses.iter().all(|s| s.converged);

            if let Some(callback) = progress.as_mut() {
                let estimated_total = statuses
                    .iter()
                    .map(|s| s.estimated_runs_needed)
                    .max()
                    .unwrap_or(run_count);
                callback(run_count, estimated_total, all_converged);
            }
        }

        ConvergenceResult {
            results,
            converged: all_converged,
            total_runs: run_count,
            metric_statuses: statuses,
        }
    }
}

/// Evaluate convergence for every target metric.
///
/// Continuous metrics use a t-interval.  Proportions (data loss probability)
/// use a Wald interval, falling back to the rule of three when the observed
/// proportion is exactly 0 or 1 and the interval would otherwise collapse.
pub fn check_convergence(
    results: &MonteCarloResults,
    criteria: &ConvergenceCriteria,
) -> Vec<MetricConvergenceStatus> {
    let alpha = 1.0 - criteria.confidence_level;
    let use_absolute = criteria.uses_absolute_error();
    let threshold = criteria.error_threshold();
    let mut statuses = Vec::with_capacity(criteria.metrics.len());

    for &metric in &criteria.metrics {
        let samples = results.metric_samples(metric);
        let n = samples.len();

        if n < 2 {
            statuses.push(MetricConvergenceStatus::unconverged(metric, n));
            continue;
        }

        let sample_mean = stats::mean(&samples);
        let sample_std = stats::std(&samples, 1);

        if metric == ConvergenceMetric::DataLossProbability {
            let p = sample_mean;
            let z = norm_ppf(1.0 - alpha / 2.0);

            if p == 0.0 || p == 1.0 {
                // No variance to work with: the rule of three bounds p at
                // roughly 3/n, so require enough runs for that bound to sit
                // under the threshold.
                let rule_of_three_n = (3.0 / threshold).ceil() as usize;
                statuses.push(MetricConvergenceStatus {
                    metric,
                    converged: n >= rule_of_three_n,
                    current_mean: p,
                    current_std: 0.0,
                    ci_half_width: 3.0 / n as f64,
                    relative_error: if p == 0.0 { 0.0 } else { f64::INFINITY },
                    estimated_runs_needed: rule_of_three_n.max(n),
                    num_samples: n,
                });
                continue;
            }

            let se = (p * (1.0 - p) / n as f64).sqrt();
            let ci_half_width = z * se;
            let rel_err = if p > 0.0 {
                ci_half_width / p
            } else {
                f64::INFINITY
            };

            let (converged, target_e) = if use_absolute {
                (ci_half_width <= threshold, threshold)
            } else {
                (rel_err <= threshold, threshold * p)
            };

            let estimated_n = (z * z * p * (1.0 - p) / (target_e * target_e)).ceil() as usize;

            statuses.push(MetricConvergenceStatus {
                metric,
                converged,
                current_mean: p,
                current_std: se * (n as f64).sqrt(),
                ci_half_width,
                relative_error: rel_err,
                estimated_runs_needed: estimated_n.max(n),
                num_samples: n,
            });
            continue;
        }

        // Continuous metric: t-interval.
        let t_crit = t_ppf(1.0 - alpha / 2.0, (n - 1) as f64);
        let se = sample_std / (n as f64).sqrt();
        let ci_half_width = t_crit * se;

        let rel_err = if sample_mean == 0.0 {
            if sample_std > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            ci_half_width / sample_mean.abs()
        };

        let (converged, target_margin) = if use_absolute {
            (ci_half_width <= threshold, threshold)
        } else {
            (
                rel_err <= threshold,
                if sample_mean.abs() > 0.0 {
                    threshold * sample_mean.abs()
                } else {
                    0.0
                },
            )
        };

        let z = norm_ppf(1.0 - alpha / 2.0);
        let mut estimated_n = if target_margin > 0.0 {
            ((z * sample_std / target_margin).powi(2)).ceil() as usize
        } else {
            criteria.max_runs
        };

        // For MTTDL the estimate counts *loss events*, not runs.  Scale by
        // the inverse observed loss rate to get a run count.
        if metric == ConvergenceMetric::MeanTimeToDataLoss {
            let total_runs = results.availability_samples.len();
            if n > 0 && n < total_runs {
                let loss_rate = n as f64 / total_runs as f64;
                estimated_n = (estimated_n as f64 / loss_rate).ceil() as usize;
            }
        }

        statuses.push(MetricConvergenceStatus {
            metric,
            converged,
            current_mean: sample_mean,
            current_std: sample_std,
            ci_half_width,
            relative_error: rel_err,
            estimated_runs_needed: estimated_n.max(n),
            num_samples: n,
        });
    }

    statuses
}

/// Estimate runs needed for convergence from a pilot run.
///
/// Run a small pilot, then call this to size the full experiment.
pub fn estimate_required_runs(
    pilot_results: &MonteCarloResults,
    convergence: &ConvergenceCriteria,
) -> Vec<(ConvergenceMetric, usize)> {
    check_convergence(pilot_results, convergence)
        .into_iter()
        .map(|s| (s.metric, s.estimated_runs_needed))
        .collect()
}

/// Convenience wrapper: run a fixed number of simulations.
///
/// Mirrors Python's `run_monte_carlo`.
#[allow(clippy::too_many_arguments)]
pub fn run_monte_carlo(
    scenario: &ScenarioFactory<'_>,
    num_simulations: usize,
    max_time: Option<Seconds>,
    stop_on_data_loss: bool,
    seed: Option<u64>,
) -> Result<MonteCarloResults, ConfigError> {
    let config = MonteCarloConfig {
        num_simulations,
        max_time,
        stop_on_data_loss,
        base_seed: seed,
        log_events: false,
    }
    .validate()?;
    Ok(MonteCarloRunner::new(config).run(scenario, None))
}

/// Convenience wrapper: run adaptively until convergence.
///
/// Mirrors Python's `run_monte_carlo_converged`.  `max_runs` doubles as the
/// runner's nominal simulation count, as it does on the Python side.
#[allow(clippy::too_many_arguments)]
pub fn run_monte_carlo_converged(
    scenario: &ScenarioFactory<'_>,
    max_time: Option<Seconds>,
    stop_on_data_loss: bool,
    seed: Option<u64>,
    criteria: ConvergenceCriteria,
    progress: Option<&mut dyn FnMut(usize, usize, bool)>,
) -> Result<ConvergenceResult, ConfigError> {
    let criteria = criteria.validate()?;
    let config = MonteCarloConfig {
        num_simulations: criteria.max_runs,
        max_time,
        stop_on_data_loss,
        base_seed: seed,
        log_events: false,
    }
    .validate()?;
    Ok(MonteCarloRunner::new(config).run_until_converged(scenario, &criteria, progress))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn criteria_defaults_match_python() {
        let c = ConvergenceCriteria::default().validate().unwrap();
        assert_eq!(c.confidence_level, 0.95);
        assert_eq!(c.relative_error, Some(0.05));
        assert_eq!(c.absolute_error, None);
        assert!(!c.uses_absolute_error());
        assert_eq!(c.metrics, vec![ConvergenceMetric::Availability]);
        assert_eq!(c.min_runs, 30);
        assert_eq!(c.max_runs, 10_000);
        assert_eq!(c.batch_size, 10);
    }

    #[test]
    fn criteria_reject_invalid_inputs() {
        let base = ConvergenceCriteria::default;

        let both = ConvergenceCriteria {
            relative_error: Some(0.05),
            absolute_error: Some(0.01),
            ..base()
        };
        assert!(both.validate().unwrap_err().0.contains("exactly one"));

        for level in [0.0, 1.0, 1.5] {
            let c = ConvergenceCriteria {
                confidence_level: level,
                ..base()
            };
            assert!(c.validate().unwrap_err().0.contains("confidence_level"));
        }

        for rel in [0.0, -0.1] {
            let c = ConvergenceCriteria {
                relative_error: Some(rel),
                ..base()
            };
            assert!(c.validate().unwrap_err().0.contains("relative_error"));
        }

        for abs in [0.0, -0.01] {
            let c = ConvergenceCriteria {
                relative_error: None,
                absolute_error: Some(abs),
                ..base()
            };
            assert!(c.validate().unwrap_err().0.contains("absolute_error"));
        }

        let c = ConvergenceCriteria {
            min_runs: 1,
            ..base()
        };
        assert!(c.validate().unwrap_err().0.contains("min_runs"));

        let c = ConvergenceCriteria {
            min_runs: 100,
            max_runs: 50,
            ..base()
        };
        assert!(c.validate().unwrap_err().0.contains("max_runs"));
    }

    #[test]
    fn config_requires_a_bound() {
        let c = MonteCarloConfig {
            max_time: None,
            stop_on_data_loss: false,
            ..Default::default()
        };
        assert!(c.validate().is_err());

        let c = MonteCarloConfig {
            max_time: Some(100.0),
            stop_on_data_loss: false,
            ..Default::default()
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inverse_cdfs_match_scipy() {
        // scipy.stats.norm.ppf(0.975)
        assert!((norm_ppf(0.975) - 1.959_963_984_540_054).abs() < 1e-9);
        // scipy.stats.t.ppf(0.975, df=29)
        assert!((t_ppf(0.975, 29.0) - 2.045_229_642_132_703).abs() < 1e-9);
        // scipy.stats.t.ppf(0.975, df=1)
        assert!((t_ppf(0.975, 1.0) - 12.706_204_736_432_09).abs() < 1e-6);
    }

    fn results_with_availability(values: &[f64]) -> MonteCarloResults {
        let mut r = MonteCarloResults::new();
        for &v in values {
            r.availability_samples.push(v);
            r.time_to_actual_loss_samples.push(None);
            r.cost_samples.push(0.0);
        }
        r
    }

    #[test]
    fn convergence_needs_at_least_two_samples() {
        let r = results_with_availability(&[0.99]);
        let c = ConvergenceCriteria::default().validate().unwrap();
        let statuses = check_convergence(&r, &c);
        assert!(!statuses[0].converged);
        assert_eq!(statuses[0].num_samples, 1);
    }

    #[test]
    fn identical_samples_converge_immediately() {
        let r = results_with_availability(&[0.99; 30]);
        let c = ConvergenceCriteria::default().validate().unwrap();
        let statuses = check_convergence(&r, &c);
        assert!(statuses[0].converged);
        // 0.99 is not exactly representable, so the spread is float noise
        // rather than a hard zero.
        assert!(statuses[0].ci_half_width < 1e-15);
    }

    #[test]
    fn absolute_mode_compares_the_half_width_directly() {
        // Alternating 0.98/1.00 gives a spread the default relative error
        // would accept but a tight absolute threshold rejects.
        let values: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { 0.98 } else { 1.0 })
            .collect();
        let r = results_with_availability(&values);

        let loose = ConvergenceCriteria {
            relative_error: None,
            absolute_error: Some(0.05),
            ..Default::default()
        }
        .validate()
        .unwrap();
        assert!(check_convergence(&r, &loose)[0].converged);

        let tight = ConvergenceCriteria {
            relative_error: None,
            absolute_error: Some(0.0001),
            ..Default::default()
        }
        .validate()
        .unwrap();
        let status = &check_convergence(&r, &tight)[0];
        assert!(!status.converged);
        assert!(status.estimated_runs_needed > 30);
    }

    #[test]
    fn zero_proportion_uses_the_rule_of_three() {
        let mut r = MonteCarloResults::new();
        for _ in 0..100 {
            r.availability_samples.push(1.0);
            r.time_to_actual_loss_samples.push(None);
            r.cost_samples.push(0.0);
        }
        let c = ConvergenceCriteria {
            metrics: vec![ConvergenceMetric::DataLossProbability],
            relative_error: None,
            absolute_error: Some(0.05),
            ..Default::default()
        }
        .validate()
        .unwrap();

        let status = &check_convergence(&r, &c)[0];
        assert_eq!(status.current_mean, 0.0);
        // 3/0.05 = 60 runs needed; 100 clears it.
        assert!(status.converged);
        assert_eq!(status.estimated_runs_needed, 100);
    }

    #[test]
    fn mttdl_estimate_is_scaled_by_the_loss_rate() {
        // Twenty runs, four of which lost data at spread-out times.
        let mut r = MonteCarloResults::new();
        for i in 0..20 {
            r.availability_samples.push(0.99);
            r.cost_samples.push(1.0);
            r.time_to_actual_loss_samples
                .push(if i < 4 { Some(1000.0 * (i + 1) as f64) } else { None });
        }

        let c = ConvergenceCriteria {
            metrics: vec![ConvergenceMetric::MeanTimeToDataLoss],
            ..Default::default()
        }
        .validate()
        .unwrap();

        let status = &check_convergence(&r, &c)[0];
        assert_eq!(status.num_samples, 4);
        // The loss rate is 4/20, so the run estimate is five times the
        // number of loss events needed.
        assert!(status.estimated_runs_needed > 20);
    }

    #[test]
    fn metric_names_round_trip() {
        for metric in [
            ConvergenceMetric::Availability,
            ConvergenceMetric::DataLossProbability,
            ConvergenceMetric::Cost,
            ConvergenceMetric::MeanTimeToDataLoss,
            ConvergenceMetric::TransientFailures,
            ConvergenceMetric::DatalossFailures,
            ConvergenceMetric::NodesSpawned,
            ConvergenceMetric::UnavailabilityIncidents,
            ConvergenceMetric::LeaderElections,
        ] {
            assert_eq!(ConvergenceMetric::from_name(metric.as_str()), Some(metric));
        }
        assert_eq!(ConvergenceMetric::from_name("nope"), None);
    }
}
