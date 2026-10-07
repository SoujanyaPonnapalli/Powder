//! JSON job schema and the translation into engine objects.
//!
//! Jobs are read from a file, stdin, or the environment, so changing an
//! input never requires a rebuild.  The schema covers every constructor
//! argument the Python API exposes.

use std::collections::HashMap;
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::monte_carlo::{ConvergenceCriteria, ConvergenceMetric, MonteCarloConfig};
use crate::sim::cluster::ClusterState;
use crate::sim::distributions::{Distribution, Seconds};
use crate::sim::network::NetworkConfig;
use crate::sim::node::{NodeConfig, NodeConfigRef, NodeState};
use crate::sim::protocol::{LeaderlessProtocol, Protocol, RaftLikeProtocol};
use crate::sim::strategy::{
    AdaptiveReplacementStrategy, ClusterStrategy, Delay, NoOpStrategy, NodeReplacementStrategy,
};

/// A malformed or inconsistent job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

impl From<crate::monte_carlo::ConfigError> for ConfigError {
    fn from(e: crate::monte_carlo::ConfigError) -> Self {
        ConfigError(e.0)
    }
}

impl From<crate::sim::distributions::DistributionError> for ConfigError {
    fn from(e: crate::sim::distributions::DistributionError) -> Self {
        ConfigError(e.0)
    }
}

type Result<T> = std::result::Result<T, ConfigError>;

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

/// Serialised form of a [`Distribution`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistributionSpec {
    /// `{"type": "exponential", "rate": 1e-6}`
    Exponential {
        /// Events per unit time.
        rate: f64,
    },
    /// `{"type": "weibull", "shape": 1.2, "scale": 3.1e7}`
    Weibull {
        /// Shape parameter.
        shape: f64,
        /// Scale parameter.
        scale: f64,
    },
    /// `{"type": "normal", "mean": 100, "std": 10, "min_val": 0}`
    Normal {
        /// Distribution mean.
        mean: f64,
        /// Standard deviation.
        std: f64,
        /// Lower clamp applied on sampling.
        #[serde(default)]
        min_val: f64,
    },
    /// `{"type": "uniform", "low": 10, "high": 20}`
    Uniform {
        /// Inclusive lower bound.
        low: f64,
        /// Exclusive upper bound.
        high: f64,
    },
    /// `{"type": "constant", "value": 60}`
    Constant {
        /// The fixed value.
        value: f64,
    },
}

impl DistributionSpec {
    /// Build the engine distribution, validating parameters.
    pub fn build(&self) -> Result<Distribution> {
        Ok(match *self {
            DistributionSpec::Exponential { rate } => Distribution::exponential(rate)?,
            DistributionSpec::Weibull { shape, scale } => Distribution::weibull(shape, scale)?,
            DistributionSpec::Normal {
                mean,
                std,
                min_val,
            } => Distribution::normal(mean, std, min_val)?,
            DistributionSpec::Uniform { low, high } => Distribution::uniform(low, high)?,
            DistributionSpec::Constant { value } => Distribution::constant(value),
        })
    }
}

/// A delay given either as a bare number of seconds or as a distribution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum DelaySpec {
    /// A fixed number of seconds.
    Fixed(f64),
    /// Sampled from a distribution.
    Sampled(DistributionSpec),
}

impl DelaySpec {
    /// Build the engine delay.
    pub fn build(&self) -> Result<Delay> {
        Ok(match self {
            DelaySpec::Fixed(v) => Delay::Fixed(*v),
            DelaySpec::Sampled(spec) => Delay::Sampled(spec.build()?),
        })
    }
}

// ---------------------------------------------------------------------------
// Nodes and cluster
// ---------------------------------------------------------------------------

/// Serialised [`NodeConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeConfigSpec {
    /// Region the node sits in.
    pub region: String,
    /// Dollar cost per hour.
    pub cost_per_hour: f64,
    /// Time until transient unavailability.
    pub failure_dist: DistributionSpec,
    /// Time to recover from a transient failure.
    pub recovery_dist: DistributionSpec,
    /// Time until permanent data loss.
    pub data_loss_dist: DistributionSpec,
    /// Log replay rate, in committed units per second.
    pub log_replay_rate_dist: DistributionSpec,
    /// Wall-clock time to download a snapshot.
    pub snapshot_download_time_dist: DistributionSpec,
    /// Time to spawn a replacement.
    pub spawn_dist: DistributionSpec,
}

impl NodeConfigSpec {
    /// Build the shared engine config.
    pub fn build(&self) -> Result<NodeConfigRef> {
        Ok(Rc::new(NodeConfig {
            region: self.region.clone(),
            cost_per_hour: self.cost_per_hour,
            failure_dist: self.failure_dist.build()?,
            recovery_dist: self.recovery_dist.build()?,
            data_loss_dist: self.data_loss_dist.build()?,
            log_replay_rate_dist: self.log_replay_rate_dist.build()?,
            snapshot_download_time_dist: self.snapshot_download_time_dist.build()?,
            spawn_dist: self.spawn_dist.build()?,
        }))
    }
}

/// One node's starting state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeSpec {
    /// Unique identifier.
    pub node_id: String,
    /// Key into the job's `node_configs` map.
    pub config: String,
    /// Whether the node starts up.
    #[serde(default = "default_true")]
    pub is_available: bool,
    /// Whether the node starts with its data.
    #[serde(default = "default_true")]
    pub has_data: bool,
    /// Starting applied index.
    #[serde(default)]
    pub last_applied_index: f64,
    /// Starting snapshot index.
    #[serde(default)]
    pub last_snapshot_index: f64,
}

fn default_true() -> bool {
    true
}

/// The cluster's starting state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClusterSpec {
    /// Desired number of active nodes.
    pub target_cluster_size: usize,
    /// Active nodes.
    pub nodes: Vec<NodeSpec>,
    /// Nodes that start in standby.
    #[serde(default)]
    pub standby_nodes: Vec<NodeSpec>,
    /// Nodes that start mid-provisioning.
    #[serde(default)]
    pub provisioning_nodes: Vec<NodeSpec>,
    /// Regions that start in an outage.
    #[serde(default)]
    pub active_outages: Vec<String>,
    /// Starting wall-clock time.
    #[serde(default)]
    pub current_time: Seconds,
    /// Starting commit index.
    #[serde(default)]
    pub commit_index: f64,
}

// ---------------------------------------------------------------------------
// Protocol and strategy
// ---------------------------------------------------------------------------

/// Serialised protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProtocolSpec {
    /// Leaderless, with configurable quorum semantics.
    Leaderless {
        /// Committed units per second of wall time.
        #[serde(default = "default_commit_rate")]
        commit_rate: f64,
        /// Commit-index interval between snapshots.
        #[serde(default)]
        snapshot_interval: f64,
        /// Committed units of log retained; 0 is infinite.
        #[serde(default)]
        log_retention_ops: f64,
        /// Whether commits need an up-to-date quorum.
        #[serde(default = "default_true")]
        up_to_date_quorum: bool,
    },
    /// Leader-based, with election downtime.
    Raft {
        /// Time to complete an election.
        election_time_dist: DistributionSpec,
        /// Committed units per second of wall time.
        #[serde(default = "default_commit_rate")]
        commit_rate: f64,
        /// Commit-index interval between snapshots.
        #[serde(default)]
        snapshot_interval: f64,
        /// Committed units of log retained; 0 is infinite.
        #[serde(default)]
        log_retention_ops: f64,
    },
}

fn default_commit_rate() -> f64 {
    1.0
}

impl ProtocolSpec {
    /// Build a fresh protocol instance.
    pub fn build(&self) -> Result<Box<dyn Protocol>> {
        Ok(match self {
            ProtocolSpec::Leaderless {
                commit_rate,
                snapshot_interval,
                log_retention_ops,
                up_to_date_quorum,
            } => Box::new(LeaderlessProtocol::new(
                *commit_rate,
                *snapshot_interval,
                *log_retention_ops,
                *up_to_date_quorum,
            )),
            ProtocolSpec::Raft {
                election_time_dist,
                commit_rate,
                snapshot_interval,
                log_retention_ops,
            } => Box::new(RaftLikeProtocol::with_params(
                election_time_dist.build()?,
                *commit_rate,
                *snapshot_interval,
                *log_retention_ops,
            )),
        })
    }
}

/// Serialised strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StrategySpec {
    /// Takes no action.
    Noop,
    /// Replaces nodes that stay down past a timeout.
    NodeReplacement {
        /// Seconds a node must be unavailable before replacement.
        failure_timeout: Seconds,
        /// Key into `node_configs` for replacement nodes; defaults to the
        /// failed node's own config.
        #[serde(default)]
        default_node_config: Option<String>,
        /// Whether promotion requires the protocol to be committable.
        #[serde(default = "default_true")]
        safe_mode: bool,
    },
    /// Replacement plus dynamic cluster resizing.
    AdaptiveReplacement {
        /// Seconds a node must be unavailable before replacement.
        failure_timeout: Seconds,
        /// How long a reconfiguration takes to land.
        reconfiguration_dist: DelaySpec,
        /// Unavailable-node count that triggers a scale down.
        #[serde(default = "default_scale_down_threshold")]
        scale_down_threshold: usize,
        /// Whether an external consensus service permits shrinking below 3.
        #[serde(default)]
        external_consensus: bool,
        /// Key into `node_configs` for replacement nodes.
        #[serde(default)]
        default_node_config: Option<String>,
        /// Whether promotion requires the protocol to be committable.
        #[serde(default = "default_true")]
        safe_mode: bool,
    },
}

fn default_scale_down_threshold() -> usize {
    2
}

/// Serialised network configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NetworkConfigSpec {
    /// Time until a region's next outage.
    pub outage_dist: DistributionSpec,
    /// Duration of each outage.
    pub outage_duration_dist: DistributionSpec,
    /// Regions that can experience outages.
    #[serde(default)]
    pub regions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Run and convergence settings
// ---------------------------------------------------------------------------

/// Run-level settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSpec {
    /// Time limit per simulation; `null` runs to data loss.
    #[serde(default)]
    pub max_time: Option<Seconds>,
    /// Whether to stop a simulation when data is lost.
    #[serde(default = "default_true")]
    pub stop_on_data_loss: bool,
    /// Number of simulations in the experiment.
    #[serde(default = "default_num_simulations")]
    pub num_simulations: usize,
    /// Base seed; run `i` uses `base_seed + i`.
    #[serde(default)]
    pub base_seed: Option<u64>,
    /// Whether to retain and emit event logs.
    #[serde(default)]
    pub log_events: bool,
    /// Accepted for schema parity with the Python config and ignored:
    /// parallelism here is per job, not per simulation.
    #[serde(default)]
    pub parallel_workers: Option<usize>,
}

fn default_num_simulations() -> usize {
    1
}

impl Default for RunSpec {
    fn default() -> Self {
        RunSpec {
            max_time: None,
            stop_on_data_loss: true,
            num_simulations: 1,
            base_seed: None,
            log_events: false,
            parallel_workers: None,
        }
    }
}

impl RunSpec {
    /// Build the engine configuration.
    pub fn build(&self) -> Result<MonteCarloConfig> {
        Ok(MonteCarloConfig {
            num_simulations: self.num_simulations,
            max_time: self.max_time,
            stop_on_data_loss: self.stop_on_data_loss,
            base_seed: self.base_seed,
            log_events: self.log_events,
        }
        .validate()?)
    }
}

/// Serialised convergence criteria.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConvergenceSpec {
    /// Desired confidence level.
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
    /// Maximum relative CI half-width.
    #[serde(default)]
    pub relative_error: Option<f64>,
    /// Maximum absolute CI half-width.
    #[serde(default)]
    pub absolute_error: Option<f64>,
    /// Metrics that must all converge.
    #[serde(default = "default_metrics")]
    pub metrics: Vec<String>,
    /// Runs before the first check.
    #[serde(default = "default_min_runs")]
    pub min_runs: usize,
    /// Safety cap on total runs.
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    /// Runs per batch between checks.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_confidence_level() -> f64 {
    0.95
}
fn default_metrics() -> Vec<String> {
    vec!["availability".to_string()]
}
fn default_min_runs() -> usize {
    30
}
fn default_max_runs() -> usize {
    10_000
}
fn default_batch_size() -> usize {
    10
}

impl ConvergenceSpec {
    /// Build and validate the engine criteria.
    pub fn build(&self) -> Result<ConvergenceCriteria> {
        let mut metrics = Vec::with_capacity(self.metrics.len());
        for name in &self.metrics {
            match ConvergenceMetric::from_name(name) {
                Some(m) => metrics.push(m),
                None => {
                    return Err(ConfigError(format!("Unknown convergence metric: {name}")))
                }
            }
        }
        Ok(ConvergenceCriteria {
            confidence_level: self.confidence_level,
            relative_error: self.relative_error,
            absolute_error: self.absolute_error,
            metrics,
            min_runs: self.min_runs,
            max_runs: self.max_runs,
            batch_size: self.batch_size,
        }
        .validate()?)
    }
}

// ---------------------------------------------------------------------------
// The job
// ---------------------------------------------------------------------------

/// What a job should produce.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum JobMode {
    /// A single simulation.
    Single,
    /// A fixed number of simulations.
    #[default]
    MonteCarlo,
    /// Adaptive runs until the convergence criteria are met.
    Converged,
}

/// One unit of work: a complete scenario plus what to do with it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Job {
    /// Caller-supplied identifier, echoed in the result.
    #[serde(default)]
    pub job_id: Option<String>,
    /// What to produce.
    #[serde(default)]
    pub mode: JobMode,
    /// Named node configurations, referenced by the node specs.
    pub node_configs: HashMap<String, NodeConfigSpec>,
    /// Starting cluster state.
    pub cluster: ClusterSpec,
    /// Consensus protocol.
    pub protocol: ProtocolSpec,
    /// Management strategy.
    pub strategy: StrategySpec,
    /// Optional region outage configuration.
    #[serde(default)]
    pub network_config: Option<NetworkConfigSpec>,
    /// Run-level settings.
    #[serde(default)]
    pub run: RunSpec,
    /// Convergence criteria; required when `mode` is `converged`.
    #[serde(default)]
    pub convergence: Option<ConvergenceSpec>,
}

/// A job with every spec resolved into engine objects.
///
/// Building this once per job and reusing it across the job's simulations
/// keeps parsing and validation out of the per-run path.
#[derive(Debug)]
pub struct ResolvedJob {
    /// Caller-supplied identifier.
    pub job_id: Option<String>,
    /// What to produce.
    pub mode: JobMode,
    /// Starting cluster state, cloned fresh for each simulation.
    pub cluster_template: ClusterState,
    /// Protocol specification, rebuilt fresh for each simulation.
    pub protocol: ProtocolSpec,
    /// Strategy, rebuilt fresh for each simulation.
    pub strategy: StrategySpec,
    /// Replacement-node config for the strategies that take one.
    pub strategy_default_config: Option<NodeConfigRef>,
    /// Network configuration, shared across simulations.
    pub network_config: Option<NetworkConfig>,
    /// Engine run configuration.
    pub run: MonteCarloConfig,
    /// Engine convergence criteria.
    pub convergence: Option<ConvergenceCriteria>,
}

impl ResolvedJob {
    /// A fresh protocol for one simulation.
    pub fn build_protocol(&self) -> Box<dyn Protocol> {
        self.protocol
            .build()
            .expect("the spec was validated when the job was resolved")
    }

    /// A fresh strategy for one simulation.
    pub fn build_strategy(&self) -> Box<dyn ClusterStrategy> {
        match &self.strategy {
            StrategySpec::Noop => Box::new(NoOpStrategy),
            StrategySpec::NodeReplacement {
                failure_timeout,
                safe_mode,
                ..
            } => Box::new(NodeReplacementStrategy::new(
                *failure_timeout,
                self.strategy_default_config.clone(),
                *safe_mode,
            )),
            StrategySpec::AdaptiveReplacement {
                failure_timeout,
                reconfiguration_dist,
                scale_down_threshold,
                external_consensus,
                safe_mode,
                ..
            } => Box::new(AdaptiveReplacementStrategy::new(
                *failure_timeout,
                reconfiguration_dist
                    .build()
                    .expect("the spec was validated when the job was resolved"),
                *scale_down_threshold,
                *external_consensus,
                self.strategy_default_config.clone(),
                *safe_mode,
            )),
        }
    }

    /// A fresh cluster for one simulation.
    pub fn build_cluster(&self) -> ClusterState {
        self.cluster_template.clone()
    }
}

impl Job {
    /// Resolve every spec into engine objects, validating as it goes.
    pub fn resolve(&self) -> Result<ResolvedJob> {
        // Node configs first: everything else refers to them by name.
        let mut configs: HashMap<&str, NodeConfigRef> =
            HashMap::with_capacity(self.node_configs.len());
        for (name, spec) in &self.node_configs {
            configs.insert(name.as_str(), spec.build()?);
        }

        let lookup = |name: &str| -> Result<NodeConfigRef> {
            configs
                .get(name)
                .cloned()
                .ok_or_else(|| ConfigError(format!("Unknown node config: {name}")))
        };

        let mut cluster = ClusterState::new(self.cluster.target_cluster_size);

        let add = |cluster: &mut ClusterState,
                       spec: &NodeSpec,
                       group: crate::sim::node::Group|
         -> Result<()> {
            let config = lookup(&spec.config)?;
            let sym = cluster.intern(&spec.node_id);
            let region = cluster.intern(&config.region);
            let mut node = NodeState::new(sym, region, config);
            node.is_available = spec.is_available;
            node.has_data = spec.has_data;
            node.last_applied_index = spec.last_applied_index;
            node.last_snapshot_index = spec.last_snapshot_index;
            match group {
                crate::sim::node::Group::Active => cluster.add_node(node),
                crate::sim::node::Group::Standby => cluster.add_standby_node(node),
                crate::sim::node::Group::Provisioning => cluster.add_provisioning_node(node),
            }
            Ok(())
        };

        for spec in &self.cluster.nodes {
            add(&mut cluster, spec, crate::sim::node::Group::Active)?;
        }
        for spec in &self.cluster.standby_nodes {
            add(&mut cluster, spec, crate::sim::node::Group::Standby)?;
        }
        for spec in &self.cluster.provisioning_nodes {
            add(&mut cluster, spec, crate::sim::node::Group::Provisioning)?;
        }

        for region in &self.cluster.active_outages {
            let sym = cluster.intern(region);
            cluster.network.add_outage(sym);
        }

        cluster.current_time = self.cluster.current_time;
        cluster.commit_index = self.cluster.commit_index;

        // Validate the protocol and strategy specs now so the per-run
        // builders can be infallible.
        self.protocol.build()?;
        let strategy_default_config = match &self.strategy {
            StrategySpec::Noop => None,
            StrategySpec::NodeReplacement {
                default_node_config,
                ..
            }
            | StrategySpec::AdaptiveReplacement {
                default_node_config,
                ..
            } => match default_node_config {
                Some(name) => Some(lookup(name)?),
                None => None,
            },
        };
        if let StrategySpec::AdaptiveReplacement {
            reconfiguration_dist,
            ..
        } = &self.strategy
        {
            reconfiguration_dist.build()?;
        }

        let network_config = match &self.network_config {
            Some(spec) => {
                let regions = spec
                    .regions
                    .iter()
                    .map(|r| cluster.intern(r))
                    .collect::<Vec<_>>();
                Some(NetworkConfig {
                    outage_dist: spec.outage_dist.build()?,
                    outage_duration_dist: spec.outage_duration_dist.build()?,
                    regions,
                })
            }
            None => None,
        };

        let mut run = self.run.clone();
        if self.mode == JobMode::Single {
            run.num_simulations = 1;
        }
        let run = run.build()?;

        let convergence = match (&self.convergence, self.mode) {
            (Some(spec), _) => Some(spec.build()?),
            (None, JobMode::Converged) => {
                return Err(ConfigError(
                    "mode \"converged\" requires a \"convergence\" block".to_string(),
                ))
            }
            (None, _) => None,
        };

        Ok(ResolvedJob {
            job_id: self.job_id.clone(),
            mode: self.mode,
            cluster_template: cluster,
            protocol: self.protocol.clone(),
            strategy: self.strategy.clone(),
            strategy_default_config,
            network_config,
            run,
            convergence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "node_configs": {
            "standard": {
                "region": "us-east",
                "cost_per_hour": 0.1,
                "failure_dist": {"type": "exponential", "rate": 1e-5},
                "recovery_dist": {"type": "constant", "value": 300},
                "data_loss_dist": {"type": "weibull", "shape": 1.2, "scale": 3.1e7},
                "log_replay_rate_dist": {"type": "normal", "mean": 100, "std": 10},
                "snapshot_download_time_dist": {"type": "uniform", "low": 10, "high": 20},
                "spawn_dist": {"type": "constant", "value": 60}
            }
        },
        "cluster": {
            "target_cluster_size": 3,
            "nodes": [
                {"node_id": "n0", "config": "standard"},
                {"node_id": "n1", "config": "standard"},
                {"node_id": "n2", "config": "standard"}
            ]
        },
        "protocol": {"type": "leaderless"},
        "strategy": {"type": "noop"},
        "run": {"max_time": 86400.0, "num_simulations": 5, "base_seed": 7}
    }"#;

    #[test]
    fn minimal_job_round_trips_and_resolves() {
        let job: Job = serde_json::from_str(MINIMAL).unwrap();
        assert_eq!(job.mode, JobMode::MonteCarlo);
        assert_eq!(job.cluster.nodes.len(), 3);

        let resolved = job.resolve().unwrap();
        assert_eq!(resolved.cluster_template.num_active(), 3);
        assert_eq!(resolved.run.num_simulations, 5);
        assert_eq!(resolved.run.base_seed, Some(7));
        assert_eq!(resolved.run.max_time, Some(86400.0));
        assert!(resolved.convergence.is_none());
    }

    #[test]
    fn defaults_fill_in_optional_fields() {
        let job: Job = serde_json::from_str(MINIMAL).unwrap();
        let spec = &job.node_configs["standard"];
        // min_val defaults to 0 for the normal distribution.
        assert_eq!(
            spec.log_replay_rate_dist,
            DistributionSpec::Normal {
                mean: 100.0,
                std: 10.0,
                min_val: 0.0
            }
        );
        // Nodes default to healthy and current.
        assert!(job.cluster.nodes[0].is_available);
        assert!(job.cluster.nodes[0].has_data);
        assert_eq!(job.cluster.nodes[0].last_applied_index, 0.0);
    }

    #[test]
    fn unknown_node_config_is_rejected() {
        let mut job: Job = serde_json::from_str(MINIMAL).unwrap();
        job.cluster.nodes[0].config = "nope".to_string();
        let err = job.resolve().unwrap_err();
        assert!(err.0.contains("Unknown node config"));
    }

    #[test]
    fn invalid_distribution_parameters_are_rejected() {
        let mut job: Job = serde_json::from_str(MINIMAL).unwrap();
        job.node_configs.get_mut("standard").unwrap().failure_dist =
            DistributionSpec::Exponential { rate: 0.0 };
        let err = job.resolve().unwrap_err();
        assert!(err.0.contains("Rate must be positive"));
    }

    #[test]
    fn converged_mode_requires_criteria() {
        let mut job: Job = serde_json::from_str(MINIMAL).unwrap();
        job.mode = JobMode::Converged;
        let err = job.resolve().unwrap_err();
        assert!(err.0.contains("requires a \"convergence\" block"));
    }

    #[test]
    fn unknown_convergence_metric_is_rejected() {
        let mut job: Job = serde_json::from_str(MINIMAL).unwrap();
        job.mode = JobMode::Converged;
        job.convergence = Some(ConvergenceSpec {
            confidence_level: 0.95,
            relative_error: Some(0.05),
            absolute_error: None,
            metrics: vec!["not_a_metric".to_string()],
            min_runs: 30,
            max_runs: 100,
            batch_size: 10,
        });
        let err = job.resolve().unwrap_err();
        assert!(err.0.contains("Unknown convergence metric"));
    }

    #[test]
    fn single_mode_forces_one_simulation() {
        let mut job: Job = serde_json::from_str(MINIMAL).unwrap();
        job.mode = JobMode::Single;
        let resolved = job.resolve().unwrap();
        assert_eq!(resolved.run.num_simulations, 1);
    }

    #[test]
    fn unbounded_runs_are_rejected() {
        let mut job: Job = serde_json::from_str(MINIMAL).unwrap();
        job.run.max_time = None;
        job.run.stop_on_data_loss = false;
        let err = job.resolve().unwrap_err();
        assert!(err.0.contains("max_time is required"));
    }

    #[test]
    fn raft_protocol_and_adaptive_strategy_resolve() {
        let text = MINIMAL
            .replace(
                r#""protocol": {"type": "leaderless"}"#,
                r#""protocol": {"type": "raft", "election_time_dist": {"type": "constant", "value": 5.0}, "snapshot_interval": 100.0}"#,
            )
            .replace(
                r#""strategy": {"type": "noop"}"#,
                r#""strategy": {"type": "adaptive_replacement", "failure_timeout": 600.0, "reconfiguration_dist": 30.0, "default_node_config": "standard"}"#,
            );

        let job: Job = serde_json::from_str(&text).unwrap();
        let resolved = job.resolve().unwrap();
        assert!(resolved.strategy_default_config.is_some());
        // Each build hands back an independent instance.
        let a = resolved.build_strategy();
        let b = resolved.build_strategy();
        assert_eq!(a.replacement_rate(), b.replacement_rate());
        assert!(resolved.build_protocol().snapshot_interval() == 100.0);
    }

    #[test]
    fn reconfiguration_delay_accepts_a_distribution() {
        let text = MINIMAL.replace(
            r#""strategy": {"type": "noop"}"#,
            r#""strategy": {"type": "adaptive_replacement", "failure_timeout": 600.0, "reconfiguration_dist": {"type": "exponential", "rate": 0.01}}"#,
        );
        let job: Job = serde_json::from_str(&text).unwrap();
        assert!(job.resolve().is_ok());
    }

    #[test]
    fn standby_provisioning_and_outages_are_applied() {
        let text = MINIMAL.replace(
            r#""nodes": ["#,
            r#""active_outages": ["us-east"],
               "standby_nodes": [{"node_id": "spare", "config": "standard"}],
               "provisioning_nodes": [{"node_id": "pending", "config": "standard", "is_available": false, "has_data": false}],
               "nodes": ["#,
        );
        let job: Job = serde_json::from_str(&text).unwrap();
        let resolved = job.resolve().unwrap();
        let cluster = &resolved.cluster_template;

        assert_eq!(cluster.num_active(), 3);
        assert_eq!(cluster.standby().count(), 1);
        assert_eq!(cluster.provisioning().count(), 1);
        assert!(cluster.network.has_active_outages());
        // Every node is in the region that is down.
        assert_eq!(cluster.num_available(), 0);
    }

    #[test]
    fn network_config_regions_are_interned() {
        let text = MINIMAL.replace(
            r#""run": {"#,
            r#""network_config": {"outage_dist": {"type": "constant", "value": 1000.0},
                                  "outage_duration_dist": {"type": "constant", "value": 60.0},
                                  "regions": ["us-east"]},
               "run": {"#,
        );
        let job: Job = serde_json::from_str(&text).unwrap();
        let resolved = job.resolve().unwrap();
        let net = resolved.network_config.unwrap();
        assert_eq!(net.regions.len(), 1);
        assert_eq!(
            net.regions[0],
            resolved.cluster_template.sym_of("us-east").unwrap()
        );
    }

    #[test]
    fn parallel_workers_is_accepted_and_ignored() {
        let text = MINIMAL.replace(
            r#""base_seed": 7"#,
            r#""base_seed": 7, "parallel_workers": 16"#,
        );
        let job: Job = serde_json::from_str(&text).unwrap();
        assert_eq!(job.run.parallel_workers, Some(16));
        // It has no effect on the resolved engine configuration.
        assert!(job.resolve().is_ok());
    }
}
