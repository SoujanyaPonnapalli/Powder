//! High-performance Monte Carlo simulator for replicated state machine
//! deployments.
//!
//! A port of the `powder.simulation` package and `powder.monte_carlo` module.
//! A single Monte Carlo experiment runs entirely on one thread; parallelism
//! is applied across independent jobs by the binary's worker pool.

pub mod config;
pub mod job;
pub mod monte_carlo;
pub mod pool;
pub mod sim;
pub mod stats;
