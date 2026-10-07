//! Shared fixtures for the ported test suite.
//!
//! Mirrors the helper constructors that the Python tests define inline
//! (`make_test_node_config`, `make_pricing_config`, and the various
//! `cluster_of` builders), so each ported test reads close to its original.

#![allow(dead_code)]

use std::rc::Rc;

use powder_mc::sim::cluster::ClusterState;
use powder_mc::sim::distributions::Distribution;
use powder_mc::sim::node::{NodeConfig, NodeConfigRef};

/// Builder for a node config, defaulting every distribution to something
/// deterministic and far away so tests only specify what they care about.
pub struct ConfigBuilder {
    region: String,
    cost_per_hour: f64,
    failure: Distribution,
    recovery: Distribution,
    data_loss: Distribution,
    log_replay_rate: Distribution,
    snapshot_download: Distribution,
    spawn: Distribution,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        ConfigBuilder {
            region: "us-east".to_string(),
            cost_per_hour: 1.0,
            // Ten years out: effectively "never" for a test horizon.
            failure: Distribution::constant(days(3650.0)),
            recovery: Distribution::constant(0.0),
            data_loss: Distribution::constant(days(3650.0)),
            log_replay_rate: Distribution::constant(100.0),
            snapshot_download: Distribution::constant(0.0),
            spawn: Distribution::constant(0.0),
        }
    }
}

impl ConfigBuilder {
    /// Start from the defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the region.
    pub fn region(mut self, region: &str) -> Self {
        self.region = region.to_string();
        self
    }

    /// Set the hourly cost.
    pub fn cost(mut self, cost_per_hour: f64) -> Self {
        self.cost_per_hour = cost_per_hour;
        self
    }

    /// Set the time-to-failure distribution.
    pub fn failure(mut self, d: Distribution) -> Self {
        self.failure = d;
        self
    }

    /// Set the recovery-time distribution.
    pub fn recovery(mut self, d: Distribution) -> Self {
        self.recovery = d;
        self
    }

    /// Set the time-to-data-loss distribution.
    pub fn data_loss(mut self, d: Distribution) -> Self {
        self.data_loss = d;
        self
    }

    /// Set the log replay rate distribution.
    pub fn log_replay_rate(mut self, d: Distribution) -> Self {
        self.log_replay_rate = d;
        self
    }

    /// Set the snapshot download time distribution.
    pub fn snapshot_download(mut self, d: Distribution) -> Self {
        self.snapshot_download = d;
        self
    }

    /// Set the spawn-time distribution.
    pub fn spawn(mut self, d: Distribution) -> Self {
        self.spawn = d;
        self
    }

    /// Finish the config.
    pub fn build(self) -> NodeConfigRef {
        Rc::new(NodeConfig {
            region: self.region,
            cost_per_hour: self.cost_per_hour,
            failure_dist: self.failure,
            recovery_dist: self.recovery,
            data_loss_dist: self.data_loss,
            log_replay_rate_dist: self.log_replay_rate,
            snapshot_download_time_dist: self.snapshot_download,
            spawn_dist: self.spawn,
        })
    }
}

/// Seconds in the given number of days.
pub fn days(d: f64) -> f64 {
    d * 86400.0
}

/// Seconds in the given number of hours.
pub fn hours(h: f64) -> f64 {
    h * 3600.0
}

/// Seconds in the given number of minutes.
pub fn minutes(m: f64) -> f64 {
    m * 60.0
}

/// A cluster of `n` nodes named `node0..node{n-1}`, all sharing `config`.
pub fn cluster_with(n: usize, config: NodeConfigRef) -> ClusterState {
    let mut cluster = ClusterState::new(n);
    for i in 0..n {
        cluster.add_named_node(&format!("node{i}"), config.clone());
    }
    cluster
}

/// A cluster of `n` healthy nodes using the default config.
pub fn basic_cluster(n: usize) -> ClusterState {
    cluster_with(n, ConfigBuilder::new().build())
}

/// Assert two floats agree to a relative tolerance.
#[track_caller]
pub fn assert_close(actual: f64, expected: f64, rel: f64) {
    let tolerance = rel * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tolerance {tolerance})"
    );
}
