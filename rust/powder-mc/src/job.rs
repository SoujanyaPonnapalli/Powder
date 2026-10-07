//! Running a resolved job and shaping the JSON result.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::{ConfigError, Job, JobMode, ResolvedJob};
use crate::monte_carlo::{
    MetricConvergenceStatus, MonteCarloResults, MonteCarloRunner,
    ScenarioFactory,
};
use crate::sim::cluster::ClusterState;
use crate::sim::distributions::Seconds;
use crate::sim::protocol::Protocol;
use crate::sim::simulator::Simulator;
use crate::sim::strategy::ClusterStrategy;

/// One processed event, as emitted when event logging is on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventRecord {
    /// When the event fired, in seconds.
    pub time: Seconds,
    /// Event type name, matching Python's enum member names.
    pub event_type: String,
    /// Node or region the event applied to.
    pub target_id: String,
}

/// Metrics from one simulation run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    /// Final simulation time.
    pub end_time: Seconds,
    /// Why the run stopped.
    pub end_reason: String,
    /// Fraction of simulated time the system could commit.
    pub availability: f64,
    /// Time the system could commit.
    pub time_available: Seconds,
    /// Time the system could not commit.
    pub time_unavailable: Seconds,
    /// Accumulated node cost.
    pub total_cost: f64,
    /// When quorum was first lost.
    pub time_to_potential_data_loss: Option<Seconds>,
    /// When data was definitely lost.
    pub time_to_actual_data_loss: Option<Seconds>,
    /// When the system first became unavailable.
    pub time_to_first_unavailability: Option<Seconds>,
    /// Count of transient failures.
    pub total_transient_failures: u64,
    /// Count of data-loss failures.
    pub total_dataloss_failures: u64,
    /// Count of nodes spawned.
    pub total_nodes_spawned: u64,
    /// Count of unavailability incidents.
    pub total_unavailability_incidents: u64,
    /// Count of leader elections.
    pub total_leader_elections: u64,
    /// Every processed event, when event logging is on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_log: Option<Vec<EventRecord>>,
}

/// Aggregate statistics over a job's runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SummaryRecord {
    /// Number of runs.
    pub num_runs: usize,
    /// Mean availability.
    pub availability_mean: f64,
    /// Sample standard deviation of availability.
    pub availability_std: f64,
    /// 5th percentile of availability.
    pub availability_p5: f64,
    /// 50th percentile of availability.
    pub availability_p50: f64,
    /// 95th percentile of availability.
    pub availability_p95: f64,
    /// Mean cost.
    pub cost_mean: f64,
    /// Fraction of runs that lost data.
    pub data_loss_probability: f64,
    /// Mean time to data loss, over the runs that lost data.
    pub mean_time_to_actual_loss: Option<f64>,
    /// Standard deviation of time to data loss.
    pub std_time_to_actual_loss: Option<f64>,
    /// 95% confidence interval for the mean time to data loss.
    pub ci_time_to_actual_loss: Option<(f64, f64)>,
    /// Mean time to potential data loss.
    pub mean_time_to_potential_loss: Option<f64>,
    /// Mean transient failures per run.
    pub transient_failures_mean: f64,
    /// Mean data-loss failures per run.
    pub dataloss_failures_mean: f64,
    /// Mean nodes spawned per run.
    pub nodes_spawned_mean: f64,
    /// Mean unavailability incidents per run.
    pub unavailability_incidents_mean: f64,
    /// Mean leader elections per run.
    pub leader_elections_mean: f64,
}

/// Convergence state for one metric, in serialisable form.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConvergenceStatusRecord {
    /// Metric name.
    pub metric: String,
    /// Whether it converged.
    pub converged: bool,
    /// Sample mean.
    pub current_mean: f64,
    /// Sample standard deviation.
    pub current_std: f64,
    /// CI half-width.
    pub ci_half_width: f64,
    /// Relative error.
    pub relative_error: f64,
    /// Estimated runs needed.
    pub estimated_runs_needed: usize,
    /// Samples collected.
    pub num_samples: usize,
}

impl From<&MetricConvergenceStatus> for ConvergenceStatusRecord {
    fn from(s: &MetricConvergenceStatus) -> Self {
        ConvergenceStatusRecord {
            metric: s.metric.as_str().to_string(),
            converged: s.converged,
            current_mean: s.current_mean,
            current_std: s.current_std,
            ci_half_width: s.ci_half_width,
            relative_error: s.relative_error,
            estimated_runs_needed: s.estimated_runs_needed,
            num_samples: s.num_samples,
        }
    }
}

/// Convergence outcome for a job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConvergenceRecord {
    /// Whether every target metric converged.
    pub converged: bool,
    /// Runs executed.
    pub total_runs: usize,
    /// Per-metric state at the end.
    pub metric_statuses: Vec<ConvergenceStatusRecord>,
}

/// The complete result of one job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobResult {
    /// The job's identifier, echoed back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    /// Set when the job failed; every other field is then absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Per-run metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runs: Option<Vec<RunRecord>>,
    /// Aggregate statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SummaryRecord>,
    /// Convergence outcome, for `converged` jobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub convergence: Option<ConvergenceRecord>,
    /// Wall-clock seconds the job took.
    pub elapsed_seconds: f64,
}

impl JobResult {
    /// A failure result carrying only the message.
    pub fn failed(job_id: Option<String>, message: String, elapsed_seconds: f64) -> Self {
        JobResult {
            job_id,
            error: Some(message),
            runs: None,
            summary: None,
            convergence: None,
            elapsed_seconds,
        }
    }
}

fn summarize(results: &MonteCarloResults) -> SummaryRecord {
    let mean_u64 = |xs: &[u64]| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        xs.iter().map(|&v| v as f64).sum::<f64>() / xs.len() as f64
    };

    SummaryRecord {
        num_runs: results.len(),
        availability_mean: results.availability_mean(),
        availability_std: results.availability_std(),
        availability_p5: results.availability_percentile(5.0),
        availability_p50: results.availability_percentile(50.0),
        availability_p95: results.availability_percentile(95.0),
        cost_mean: results.cost_mean(),
        data_loss_probability: results.data_loss_probability(),
        mean_time_to_actual_loss: results.mean_time_to_actual_loss(),
        std_time_to_actual_loss: results.std_time_to_actual_loss(),
        ci_time_to_actual_loss: results.ci_time_to_actual_loss(0.95),
        mean_time_to_potential_loss: results.mean_time_to_potential_loss(),
        transient_failures_mean: mean_u64(&results.transient_failure_samples),
        dataloss_failures_mean: mean_u64(&results.dataloss_failure_samples),
        nodes_spawned_mean: mean_u64(&results.nodes_spawned_samples),
        unavailability_incidents_mean: mean_u64(&results.unavailability_incident_samples),
        leader_elections_mean: mean_u64(&results.leader_election_samples),
    }
}

/// Per-run records reconstructed from the aggregated samples.
fn run_records(results: &MonteCarloResults) -> Vec<RunRecord> {
    (0..results.len())
        .map(|i| {
            let available = results.availability_samples[i];
            let total = results.total_time_samples[i];
            RunRecord {
                end_time: total,
                end_reason: results.end_reasons[i].as_str().to_string(),
                availability: available,
                time_available: available * total,
                time_unavailable: (1.0 - available) * total,
                total_cost: results.cost_samples[i],
                time_to_potential_data_loss: results.time_to_potential_loss_samples[i],
                time_to_actual_data_loss: results.time_to_actual_loss_samples[i],
                time_to_first_unavailability: results.time_to_first_unavailability_samples[i],
                total_transient_failures: results.transient_failure_samples[i],
                total_dataloss_failures: results.dataloss_failure_samples[i],
                total_nodes_spawned: results.nodes_spawned_samples[i],
                total_unavailability_incidents: results.unavailability_incident_samples[i],
                total_leader_elections: results.leader_election_samples[i],
                event_log: None,
            }
        })
        .collect()
}

/// Run a single simulation and emit its exact metrics, including the event
/// log when requested.
///
/// `mode: "single"` takes this path rather than the aggregate one so that
/// per-run timings and traces survive intact.
fn run_single(resolved: &ResolvedJob) -> RunRecord {
    let mut simulator = Simulator::new(
        resolved.build_cluster(),
        resolved.build_strategy(),
        resolved.build_protocol(),
        resolved.network_config.clone(),
        resolved.run.base_seed,
        resolved.run.log_events,
    );

    let result = if resolved.run.stop_on_data_loss {
        simulator.run_until_data_loss(resolved.run.max_time)
    } else {
        simulator.run_until(resolved.run.max_time, None)
    };

    let m = result.metrics;
    let event_log = if resolved.run.log_events {
        Some(
            result
                .event_log
                .iter()
                .map(|e| EventRecord {
                    time: e.time,
                    event_type: e.event_type.as_str().to_string(),
                    target_id: simulator.cluster.name_of(e.target_id),
                })
                .collect(),
        )
    } else {
        None
    };

    RunRecord {
        end_time: result.end_time,
        end_reason: result.end_reason.as_str().to_string(),
        availability: m.availability_fraction(),
        time_available: m.time_available,
        time_unavailable: m.time_unavailable,
        total_cost: m.total_cost,
        time_to_potential_data_loss: m.time_to_potential_data_loss,
        time_to_actual_data_loss: m.time_to_actual_data_loss,
        time_to_first_unavailability: m.time_to_first_unavailability,
        total_transient_failures: m.total_transient_failures,
        total_dataloss_failures: m.total_dataloss_failures,
        total_nodes_spawned: m.total_nodes_spawned,
        total_unavailability_incidents: m.total_unavailability_incidents,
        total_leader_elections: m.total_leader_elections,
        event_log,
    }
}

/// Execute a resolved job.
///
/// Runs entirely on the calling thread: a job is the unit of parallelism,
/// not a simulation.
pub fn execute(resolved: &ResolvedJob) -> JobResult {
    let started = Instant::now();

    let cluster = || resolved.build_cluster();
    let strategy = || resolved.build_strategy();
    let protocol = || resolved.build_protocol();
    let scenario = ScenarioFactory {
        cluster: &cluster as &dyn Fn() -> ClusterState,
        strategy: &strategy as &dyn Fn() -> Box<dyn ClusterStrategy>,
        protocol: &protocol as &dyn Fn() -> Box<dyn Protocol>,
        network_config: resolved.network_config.clone(),
    };

    let (runs, summary, convergence) = match resolved.mode {
        JobMode::Single => {
            let record = run_single(resolved);
            (Some(vec![record]), None, None)
        }
        JobMode::MonteCarlo => {
            let results = MonteCarloRunner::new(resolved.run.clone()).run(&scenario, None);
            (
                Some(run_records(&results)),
                Some(summarize(&results)),
                None,
            )
        }
        JobMode::Converged => {
            let criteria = resolved
                .convergence
                .clone()
                .expect("resolve() rejects converged jobs without criteria");
            let outcome = MonteCarloRunner::new(resolved.run.clone())
                .run_until_converged(&scenario, &criteria, None);
            let record = ConvergenceRecord {
                converged: outcome.converged,
                total_runs: outcome.total_runs,
                metric_statuses: outcome
                    .metric_statuses
                    .iter()
                    .map(ConvergenceStatusRecord::from)
                    .collect(),
            };
            (
                Some(run_records(&outcome.results)),
                Some(summarize(&outcome.results)),
                Some(record),
            )
        }
    };

    JobResult {
        job_id: resolved.job_id.clone(),
        error: None,
        runs,
        summary,
        convergence,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    }
}

/// Parse, resolve and execute one job, turning any failure into an error
/// result rather than a panic.
pub fn run_job(job: &Job) -> JobResult {
    let started = Instant::now();
    match job.resolve() {
        Ok(resolved) => execute(&resolved),
        Err(ConfigError(message)) => JobResult::failed(
            job.job_id.clone(),
            message,
            started.elapsed().as_secs_f64(),
        ),
    }
}

/// Parse one JSON job and run it.
pub fn run_job_json(text: &str) -> JobResult {
    match serde_json::from_str::<Job>(text) {
        Ok(job) => run_job(&job),
        Err(e) => JobResult::failed(None, format!("invalid job JSON: {e}"), 0.0),
    }
}

/// Drop per-run detail from a result, keeping only the aggregates.
///
/// Sweeps over thousands of jobs rarely want every sample echoed back.
pub fn strip_runs(mut result: JobResult) -> JobResult {
    result.runs = None;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_job(extra: &str) -> String {
        format!(
            r#"{{
            "job_id": "t",
            {extra}
            "node_configs": {{
                "standard": {{
                    "region": "us-east",
                    "cost_per_hour": 1.0,
                    "failure_dist": {{"type": "constant", "value": 7200.0}},
                    "recovery_dist": {{"type": "constant", "value": 600.0}},
                    "data_loss_dist": {{"type": "constant", "value": 863999999.0}},
                    "log_replay_rate_dist": {{"type": "constant", "value": 100.0}},
                    "snapshot_download_time_dist": {{"type": "constant", "value": 0.0}},
                    "spawn_dist": {{"type": "constant", "value": 0.0}}
                }}
            }},
            "cluster": {{
                "target_cluster_size": 3,
                "nodes": [
                    {{"node_id": "node0", "config": "standard"}},
                    {{"node_id": "node1", "config": "standard"}},
                    {{"node_id": "node2", "config": "standard"}}
                ]
            }},
            "protocol": {{"type": "leaderless"}},
            "strategy": {{"type": "noop"}}
        }}"#
        )
    }

    #[test]
    fn single_mode_emits_one_run_with_no_summary() {
        let text = minimal_job(
            r#""mode": "single",
               "run": {"max_time": 86400.0, "stop_on_data_loss": false, "base_seed": 1},"#,
        );
        let result = run_job_json(&text);
        assert!(result.error.is_none(), "{:?}", result.error);
        assert_eq!(result.job_id.as_deref(), Some("t"));
        let runs = result.runs.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].end_reason, "time_limit");
        assert_eq!(runs[0].end_time, 86400.0);
        assert!(result.summary.is_none());
        assert!(result.convergence.is_none());
    }

    #[test]
    fn single_mode_emits_an_event_log_when_asked() {
        let text = minimal_job(
            r#""mode": "single",
               "run": {"max_time": 86400.0, "stop_on_data_loss": false, "base_seed": 1, "log_events": true},"#,
        );
        let result = run_job_json(&text);
        let runs = result.runs.unwrap();
        let log = runs[0].event_log.as_ref().expect("event log requested");
        assert!(!log.is_empty());
        // Node failures every 2 h with a 10 min recovery, so both show up.
        assert!(log.iter().any(|e| e.event_type == "NODE_FAILURE"));
        assert!(log.iter().any(|e| e.event_type == "NODE_RECOVERY"));
        // Targets are reported by name, not by interned symbol.
        assert!(log.iter().any(|e| e.target_id == "node0"));
    }

    #[test]
    fn monte_carlo_mode_emits_runs_and_a_summary() {
        let text = minimal_job(
            r#""mode": "monte_carlo",
               "run": {"max_time": 86400.0, "stop_on_data_loss": false, "num_simulations": 8, "base_seed": 1},"#,
        );
        let result = run_job_json(&text);
        assert!(result.error.is_none(), "{:?}", result.error);
        assert_eq!(result.runs.as_ref().unwrap().len(), 8);
        let summary = result.summary.unwrap();
        assert_eq!(summary.num_runs, 8);
        assert!(summary.availability_mean > 0.0);
        assert!(summary.cost_mean > 0.0);
    }

    #[test]
    fn converged_mode_emits_convergence_state() {
        let text = minimal_job(
            r#""mode": "converged",
               "run": {"max_time": 86400.0, "stop_on_data_loss": false, "base_seed": 1},
               "convergence": {"relative_error": 0.2, "min_runs": 5, "max_runs": 50, "batch_size": 5},"#,
        );
        let result = run_job_json(&text);
        assert!(result.error.is_none(), "{:?}", result.error);
        let convergence = result.convergence.unwrap();
        assert!(convergence.total_runs >= 5);
        assert_eq!(convergence.metric_statuses.len(), 1);
        assert_eq!(convergence.metric_statuses[0].metric, "availability");
        assert_eq!(
            result.runs.as_ref().unwrap().len(),
            convergence.total_runs
        );
    }

    #[test]
    fn bad_json_becomes_an_error_result() {
        let result = run_job_json("{not json");
        assert!(result.error.unwrap().contains("invalid job JSON"));
        assert!(result.runs.is_none());
    }

    #[test]
    fn a_bad_config_becomes_an_error_result() {
        let text = minimal_job(
            r#""mode": "monte_carlo",
               "run": {"stop_on_data_loss": false, "num_simulations": 2},"#,
        );
        let result = run_job_json(&text);
        assert!(result.error.unwrap().contains("max_time is required"));
    }

    #[test]
    fn results_round_trip_through_json() {
        let text = minimal_job(
            r#""mode": "monte_carlo",
               "run": {"max_time": 3600.0, "stop_on_data_loss": false, "num_simulations": 3, "base_seed": 1},"#,
        );
        let result = run_job_json(&text);
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: JobResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn the_same_seed_gives_the_same_result() {
        let text = minimal_job(
            r#""mode": "monte_carlo",
               "run": {"max_time": 86400.0, "stop_on_data_loss": false, "num_simulations": 5, "base_seed": 99},"#,
        );
        let a = run_job_json(&text);
        let b = run_job_json(&text);
        assert_eq!(a.summary, b.summary);
        assert_eq!(a.runs, b.runs);
    }

    #[test]
    fn strip_runs_keeps_only_the_aggregates() {
        let text = minimal_job(
            r#""mode": "monte_carlo",
               "run": {"max_time": 3600.0, "stop_on_data_loss": false, "num_simulations": 3, "base_seed": 1},"#,
        );
        let stripped = strip_runs(run_job_json(&text));
        assert!(stripped.runs.is_none());
        assert!(stripped.summary.is_some());
    }
}
