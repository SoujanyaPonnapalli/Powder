//! Discrete-event simulation of replicated state machine deployments.
//!
//! Port of the `powder.simulation` package.  Module names and the split of
//! responsibilities mirror the Python source so the two can be read side by
//! side.

pub mod cluster;
pub mod distributions;
pub mod events;
pub mod ids;
pub mod metrics;
pub mod network;
pub mod node;
pub mod protocol;
pub mod simulator;
pub mod strategy;
pub mod util;
