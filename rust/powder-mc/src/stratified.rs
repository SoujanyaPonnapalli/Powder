//! Stratified Monte Carlo over failure counts.
//!
//! A run's outcome is driven mostly by how many failures land in the
//! horizon.  Stratifying on those counts lets the common strata (few or no
//! failures) be estimated cheaply and forces samples into the rare,
//! many-failure strata that plain Monte Carlo almost never visits.
//!
//! **Strata.**  Each [`StratumAxis`] is an independent count of candidate
//! failures of one kind and rate (see [`schedule`](crate::sim::schedule)).
//! A stratum is one count per axis; its weight is the product of the axes'
//! count probabilities.  Independence is what makes pruning cheap: strata
//! are visited best-first from the joint mode, so the walk can stop once
//! the unvisited mass is small enough, without enumerating the rest.
//!
//! **Pruning.**  The unvisited mass is never sampled.  Its contribution is
//! imputed at the metric's midpoint and half its range is added to the
//! confidence half-width as a worst-case bias, so the interval stays honest.
//!
//! **Exact stratum.**  With no candidate failures at all nothing random
//! happens: the cluster never fails, so recovery, spawn and election times
//! are never drawn.  That stratum is simulated once and treated as known.
//!
//! **Stopping rule.**  Every arm, naive included, stops when
//!
//! ```text
//! sqrt(2 V L) + (7/3) L max_s w_s / (n_s - 1) + bias  <=  tolerance
//! ```
//!
//! where `V = sum_s w_s^2 s_s^2 / n_s` is the estimator's variance and
//! `L = ln(4 / delta)`.  With a single stratum this is exactly the
//! Maurer-Pontil empirical Bernstein bound for a mean in `[0, 1]`
//! (arXiv:0907.3740, Theorem 4, two-sided).  Across strata it is the same
//! Bernstein shape with the per-observation range scaled by each stratum's
//! weight -- a heuristic extension rather than a proved bound, but one that
//! cannot collapse to zero width just because every sample agreed, which is
//! the failure mode a plain t-interval has on rare events.
//!
//! The `clt` stopping rule drops the range term, `z sqrt(V) + bias`: the
//! classic stratified interval.  It is cheaper but can declare convergence
//! on a sample that has simply not seen a rare event yet.  Both half-widths
//! are computed at every check whichever rule stops the run.
//!
//! The axis is a trait so that non-exponential failures can be added later:
//! they need a different count distribution and conditional sampler, but
//! the enumeration, allocation and stopping rule are unchanged.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::monte_carlo::ConvergenceMetric;
use crate::sim::cluster::ClusterState;
use crate::sim::distributions::{Distribution, Rng, Seconds};
use crate::sim::ids::Sym;
use crate::sim::metrics::MetricsSnapshot;
use crate::sim::network::NetworkConfig;
use crate::sim::node::{Group, NodeConfigRef};
use crate::sim::protocol::Protocol;
use crate::sim::schedule::{random_slot, Candidate, FailureKind, ForcedSchedule, ScheduleAxis};
use crate::sim::simulator::Simulator;
use crate::sim::strategy::ClusterStrategy;

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

/// One independent count of failures over the horizon.
pub trait StratumAxis {
    /// The candidate stream this axis drives.
    fn schedule_axis(&self) -> &ScheduleAxis;
    /// Probability of exactly `k` candidates in the horizon.
    fn pmf(&self, k: u32) -> f64;
    /// The most likely count.
    fn mode(&self) -> u32;
    /// Draw the times of `k` candidates in `[0, horizon)`, given there are
    /// exactly `k`.
    fn sample_times(&self, k: u32, horizon: Seconds, rng: &mut Rng, out: &mut Vec<Seconds>);
}

/// A homogeneous Poisson stream: exact for exponential clocks.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonAxis {
    /// The candidate stream.
    pub axis: ScheduleAxis,
    /// Expected candidates over the horizon, `rate * slots * horizon`.
    pub mean: f64,
}

impl StratumAxis for PoissonAxis {
    fn schedule_axis(&self) -> &ScheduleAxis {
        &self.axis
    }

    fn pmf(&self, k: u32) -> f64 {
        if self.mean == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        let k = k as f64;
        (k * self.mean.ln() - self.mean - statrs::function::gamma::ln_gamma(k + 1.0)).exp()
    }

    fn mode(&self) -> u32 {
        self.mean.floor() as u32
    }

    fn sample_times(&self, k: u32, horizon: Seconds, rng: &mut Rng, out: &mut Vec<Seconds>) {
        // Given the count, a Poisson process's points are i.i.d. uniform.
        for _ in 0..k {
            out.push(rng.random::<f64>() * horizon);
        }
    }
}

/// Group a cluster's nodes into Poisson axes.
///
/// With `leader` set, that node gets a reserved transient and permanent
/// axis of one slot each, ahead of the shared axes, and is left out of the
/// shared ones.
///
/// Nodes whose clocks share a kind and an exponential rate share an axis.
/// Each axis gets one slot per starting node, plus `headroom` more on any
/// axis that replacement nodes can join: `replacement` is the config the
/// strategy spawns, `None` meaning it copies the failed node's config (so
/// every axis can grow), and `spawns` is false for strategies that never
/// add nodes.
pub fn poisson_axes(
    template: &ClusterState,
    replacement: Option<&NodeConfigRef>,
    spawns: bool,
    headroom: usize,
    horizon: Seconds,
    leader: Option<Sym>,
) -> Result<Vec<PoissonAxis>, String> {
    if template
        .all_entries()
        .iter()
        .any(|n| n.group != Group::Active)
    {
        // The native simulator gives starting standby nodes no clocks; the
        // forced schedule cannot express that.
        return Err("stratified sampling needs a cluster with only active nodes".to_string());
    }

    let rate_of = |kind: FailureKind, dist: &Distribution| -> Result<f64, String> {
        match *dist {
            Distribution::Exponential { rate } => Ok(rate),
            other => Err(format!(
                "stratified sampling needs exponential {} failures, got {other}",
                kind.as_str()
            )),
        }
    };

    let mut axes = Vec::new();
    if let Some(id) = leader {
        let node = template
            .get_node(id)
            .ok_or_else(|| "leader is not in the cluster".to_string())?;
        for kind in [FailureKind::Transient, FailureKind::Permanent] {
            let rate = rate_of(kind, kind.dist(node))?;
            axes.push(PoissonAxis {
                axis: ScheduleAxis {
                    kind,
                    rate,
                    slots: 1,
                    reserved: Some(id),
                },
                mean: rate * horizon,
            });
        }
    }
    for kind in [FailureKind::Transient, FailureKind::Permanent] {
        // (rate bits, slots), in first-seen order so axis order is stable.
        let mut groups: Vec<(u64, usize)> = Vec::new();
        let mut bump = |rate: f64, by: usize| match groups
            .iter_mut()
            .find(|(bits, _)| *bits == rate.to_bits())
        {
            Some((_, slots)) => *slots += by,
            None => groups.push((rate.to_bits(), by)),
        };
        for node in template.all_entries() {
            if Some(node.node_id) != leader {
                bump(rate_of(kind, kind.dist(node))?, 1);
            }
        }
        if spawns {
            match replacement {
                Some(config) => {
                    let dist = match kind {
                        FailureKind::Transient => &config.failure_dist,
                        FailureKind::Permanent => &config.data_loss_dist,
                    };
                    bump(rate_of(kind, dist)?, headroom);
                }
                None => {
                    for (_, slots) in groups.iter_mut() {
                        *slots += headroom;
                    }
                }
            }
        }
        for (bits, slots) in groups {
            let rate = f64::from_bits(bits);
            axes.push(PoissonAxis {
                axis: ScheduleAxis {
                    kind,
                    rate,
                    slots,
                    reserved: None,
                },
                mean: rate * slots as f64 * horizon,
            });
        }
    }
    Ok(axes)
}

// ---------------------------------------------------------------------------
// Strata
// ---------------------------------------------------------------------------

/// One cell of the stratification: a count per axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Stratum {
    /// Candidate count on each axis.
    pub counts: Vec<u32>,
    /// Probability of this stratum.
    pub weight: f64,
}

struct Frontier(f64, Vec<u32>);

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// Visit strata from most to least likely until at most `prune_mass` of
/// probability is left unvisited.
///
/// Returns the kept strata, most likely first, and the pruned mass.  Every
/// axis pmf is unimodal, so each stratum other than the mode has a more
/// likely neighbour one step closer to the mode; expanding neighbours of
/// popped strata therefore discovers strata in decreasing weight, and the
/// walk never needs to look past the cut.
pub fn enumerate_strata(
    axes: &[&dyn StratumAxis],
    prune_mass: f64,
    max_strata: usize,
) -> Result<(Vec<Stratum>, f64), String> {
    let weight = |counts: &[u32]| -> f64 {
        axes.iter()
            .zip(counts)
            .map(|(axis, &k)| axis.pmf(k))
            .product()
    };

    let start: Vec<u32> = axes.iter().map(|a| a.mode()).collect();
    let mut heap = BinaryHeap::new();
    let mut seen = HashSet::new();
    heap.push(Frontier(weight(&start), start.clone()));
    seen.insert(start);

    let mut kept = Vec::new();
    let mut mass = 0.0;
    while let Some(Frontier(w, counts)) = heap.pop() {
        if 1.0 - mass <= prune_mass {
            break;
        }
        if kept.len() == max_strata {
            return Err(format!(
                "more than {max_strata} strata needed to leave {prune_mass:e} unvisited \
                 ({:e} still unvisited)",
                1.0 - mass
            ));
        }
        mass += w;
        for i in 0..counts.len() {
            for step in [-1i64, 1] {
                let k = counts[i] as i64 + step;
                if k < 0 {
                    continue;
                }
                let mut next = counts.clone();
                next[i] = k as u32;
                if seen.insert(next.clone()) {
                    heap.push(Frontier(weight(&next), next));
                }
            }
        }
        kept.push(Stratum { counts, weight: w });
    }
    Ok((kept, (1.0 - mass).max(0.0)))
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/// Running mean and variance (Welford).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Moments {
    /// Observations.
    pub n: u64,
    /// Mean of the observations.
    pub mean: f64,
    m2: f64,
}

impl Moments {
    /// Add one observation.
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Unbiased sample variance; zero below two observations.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }
}

/// A stratified estimate and its half-widths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    /// Point estimate.
    pub mean: f64,
    /// Bernstein half-width: variance term plus range term plus bias.
    pub half_width: f64,
    /// CLT half-width: `z sqrt(V)` plus bias.
    pub clt_half_width: f64,
    /// `sqrt(2 V L)`.
    pub variance_term: f64,
    /// `(7/3) L max_s w_s / (n_s - 1)`.
    pub range_term: f64,
    /// Worst-case bias from pruned strata, half the pruned mass.
    pub bias: f64,
}

/// Combine per-stratum moments into one estimate.
///
/// `sampled` pairs each sampled stratum's weight with its moments; `exact`
/// pairs each known stratum's weight with its value.  Metrics live in
/// `[0, 1]`, so the pruned mass is imputed at 0.5.
pub fn estimate(
    sampled: &[(f64, Moments)],
    exact: &[(f64, f64)],
    pruned_mass: f64,
    confidence_level: f64,
) -> Estimate {
    let log_term = (4.0 / (1.0 - confidence_level)).ln();
    let z = crate::stats::norm_ppf(1.0 - (1.0 - confidence_level) / 2.0);
    let mut mean = 0.5 * pruned_mass;
    let mut variance = 0.0;
    let mut range = 0.0f64;
    for &(w, m) in exact {
        mean += w * m;
    }
    for &(w, m) in sampled {
        mean += w * m.mean;
        if m.n < 2 {
            range = f64::INFINITY;
            continue;
        }
        variance += w * w * m.variance() / m.n as f64;
        range = range.max(w / (m.n - 1) as f64);
    }
    let variance_term = (2.0 * variance * log_term).sqrt();
    let range_term = 7.0 / 3.0 * log_term * range;
    let bias = 0.5 * pruned_mass;
    Estimate {
        mean,
        half_width: variance_term + range_term + bias,
        variance_term,
        range_term,
        bias,
        clt_half_width: if range.is_finite() {
            z * variance.sqrt() + bias
        } else {
            f64::INFINITY
        },
    }
}

/// Which half-width decides convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoppingRule {
    /// Empirical-Bernstein: cannot collapse on agreeing samples.
    #[default]
    Bernstein,
    /// Normal approximation: cheaper, can stop before a rare event shows.
    Clt,
}

impl StoppingRule {
    fn half_width(self, e: &Estimate) -> f64 {
        match self {
            StoppingRule::Bernstein => e.half_width,
            StoppingRule::Clt => e.clt_half_width,
        }
    }
}

// ---------------------------------------------------------------------------
// The experiment
// ---------------------------------------------------------------------------

/// How runs are spread over strata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Allocation {
    /// No stratification: plain Monte Carlo with native clocks.
    Naive,
    /// Runs in proportion to stratum weight, at least `min_stratum_runs`
    /// per stratum.  The default: with leader strata and 30 runs per
    /// stratum it was the best-calibrated interval in the coverage study.
    #[default]
    Proportional,
    /// Runs in proportion to weight times standard deviation, at least two
    /// per stratum.
    Neyman,
    /// Runs placed to minimise the Bernstein half-width itself: Neyman on
    /// top of per-stratum floors that cap the range term.
    Bernstein,
}

/// Settings for a stratified experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratifiedSpec {
    /// How runs are allocated.
    #[serde(default)]
    pub allocation: Allocation,
    /// Metric that must converge: `availability` or
    /// `data_loss_probability`.
    pub metric: String,
    /// Confidence level of the half-width.
    #[serde(default = "default_confidence")]
    pub confidence_level: f64,
    /// Half-width to reach.
    pub absolute_error: f64,
    /// Which half-width must reach it.
    #[serde(default)]
    pub stopping_rule: StoppingRule,
    /// Fraction of `absolute_error` the pruned strata may contribute, as
    /// pruned mass times the metric's range.
    #[serde(default = "default_prune_fraction")]
    pub prune_fraction: f64,
    /// No arm stops before this many runs.  The naive arm's first check
    /// comes here; stratified arms also start from `min_stratum_runs` per
    /// stratum, which may be more.
    #[serde(default = "default_min_runs")]
    pub min_runs: usize,
    /// Cap on total runs.
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    /// Factor by which the run budget grows between checks.
    #[serde(default = "default_growth")]
    pub growth: f64,
    /// Extra slots for axes that replacement nodes join; defaults to the
    /// target cluster size plus one.  Never a correctness bound: a node that
    /// finds no free slot keeps its native clocks, outside the strata.
    #[serde(default)]
    pub slot_headroom: Option<usize>,
    /// Cap on strata kept.
    #[serde(default = "default_max_strata")]
    pub max_strata: usize,
    /// Keep a (runs, mean, half-width) row per check.
    #[serde(default = "default_true")]
    pub trace: bool,
    /// Report every stratum's weight, runs and moments.
    #[serde(default)]
    pub emit_strata: bool,
    /// Give the initial leader its own axes, and treat every stratum where
    /// it never fails and the other nodes fail at most `f` times in total
    /// as exact: quorum and leader both survive, so nothing is random.
    /// Unset means on whenever the protocol has an initial leader.
    #[serde(default)]
    pub leader_strata: Option<bool>,
    /// Runs every sampled stratum gets before its variance is trusted,
    /// and never drops below.  At least two, so a variance exists.
    #[serde(default = "default_min_stratum_runs")]
    pub min_stratum_runs: u64,
    /// Extra runs of each exact stratum, which must all repeat the first
    /// run's outcome exactly; a mismatch is an error, not a bias.
    #[serde(default = "default_exact_checks")]
    pub exact_checks: u64,
}

fn default_min_stratum_runs() -> u64 {
    30
}

fn default_exact_checks() -> u64 {
    3
}

fn default_confidence() -> f64 {
    0.99
}
fn default_prune_fraction() -> f64 {
    0.1
}
fn default_min_runs() -> usize {
    50_000
}
fn default_max_runs() -> usize {
    10_000_000
}
fn default_growth() -> f64 {
    1.1
}
fn default_max_strata() -> usize {
    1_000_000
}
fn default_true() -> bool {
    true
}

impl StratifiedSpec {
    /// The convergence metric, if it is one this mode supports.
    pub fn target(&self) -> Result<ConvergenceMetric, String> {
        match ConvergenceMetric::from_name(&self.metric) {
            Some(
                m @ (ConvergenceMetric::Availability | ConvergenceMetric::DataLossProbability),
            ) => Ok(m),
            _ => Err(format!(
                "stratified mode supports availability and data_loss_probability, got {}",
                self.metric
            )),
        }
    }

    /// Reject settings that cannot work.
    pub fn validate(&self) -> Result<(), String> {
        self.target()?;
        if !(self.confidence_level > 0.0 && self.confidence_level < 1.0) {
            return Err("confidence_level must be in (0, 1)".to_string());
        }
        if self.absolute_error <= 0.0 {
            return Err("absolute_error must be > 0".to_string());
        }
        if !(self.prune_fraction >= 0.0 && self.prune_fraction < 1.0) {
            return Err("prune_fraction must be in [0, 1)".to_string());
        }
        if self.growth <= 1.0 {
            return Err("growth must be > 1".to_string());
        }
        if self.min_runs < 2 {
            return Err("min_runs must be >= 2".to_string());
        }
        if self.min_stratum_runs < 2 {
            return Err("min_stratum_runs must be >= 2".to_string());
        }
        Ok(())
    }
}

/// One convergence check.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TraceRow {
    /// Runs so far.
    pub runs: usize,
    /// Estimate of the target metric.
    pub mean: f64,
    /// Its Bernstein half-width.
    pub half_width: f64,
    /// Its CLT half-width.
    pub clt_half_width: f64,
}

/// Summary of one axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisRecord {
    /// `transient` or `permanent`.
    pub kind: String,
    /// Per-node rate.
    pub rate: f64,
    /// Slots, each an independent candidate stream.
    pub slots: usize,
    /// Expected candidates over the horizon.
    pub mean: f64,
    /// Most slots held at once in any run.
    pub peak_held: usize,
    /// Nodes, over all runs, that found every slot held and kept their
    /// native clocks.
    pub native_nodes: u64,
    /// Whether the axis is reserved for the initial leader.
    #[serde(default)]
    pub leader: bool,
}

/// Outcome of a stratified experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratifiedRecord {
    /// How runs were allocated.
    pub allocation: Allocation,
    /// Target metric.
    pub metric: String,
    /// Which half-width decided convergence.
    pub stopping_rule: StoppingRule,
    /// Whether the half-width reached the tolerance.
    pub converged: bool,
    /// Simulations run, including the exact stratum's one.
    pub total_runs: usize,
    /// Half-width the run aimed for.
    pub tolerance: f64,
    /// Target metric's estimate at the end.
    pub target: Estimate,
    /// Availability estimate at the end.
    pub availability: Estimate,
    /// Data-loss probability estimate at the end.
    pub data_loss_probability: Estimate,
    /// Strata kept, the exact one included.
    pub num_strata: usize,
    /// Probability mass of the kept strata.
    pub kept_mass: f64,
    /// Probability mass treated as exact.
    pub exact_mass: f64,
    /// Probability mass pruned.
    pub pruned_mass: f64,
    /// The axes, empty for the naive arm.
    pub axes: Vec<AxisRecord>,
    /// One row per check.
    pub trace: Vec<TraceRow>,
    /// Per-stratum detail, when `emit_strata` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strata: Option<Vec<StratumRecord>>,
}

/// One stratum's share of the result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratumRecord {
    /// Candidate count per axis.
    pub counts: Vec<u32>,
    /// Probability of the stratum.
    pub weight: f64,
    /// Whether it was treated as known after one run.
    pub exact: bool,
    /// Runs spent on it.
    pub runs: u64,
    /// Mean availability.
    pub availability_mean: f64,
    /// Sample variance of availability.
    pub availability_variance: f64,
    /// Fraction of its runs that lost data.
    pub data_loss_mean: f64,
    /// Sample variance of the loss indicator.
    pub data_loss_variance: f64,
}

/// SplitMix64 finaliser, for deriving independent seeds.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn seed_for(base: u64, stratum: usize, run: u64, stream: u64) -> u64 {
    mix(mix(mix(base ^ stream) ^ stratum as u64) ^ run)
}

/// Availability and loss indicator of one run.
fn observe(metrics: &MetricsSnapshot, horizon: Seconds) -> (f64, f64) {
    let lost = metrics
        .time_to_actual_data_loss
        .is_some_and(|t| t <= horizon);
    (
        metrics.availability_fraction(),
        if lost { 1.0 } else { 0.0 },
    )
}

/// What one experiment needs to build its simulations.
pub struct StratifiedScenario<'a> {
    /// Starting cluster, identical for every run.
    pub template: ClusterState,
    /// Builds a fresh strategy per run.
    pub strategy: &'a dyn Fn() -> Box<dyn ClusterStrategy>,
    /// Builds a fresh protocol per run.
    pub protocol: &'a dyn Fn() -> Box<dyn Protocol>,
    /// Region outages; must be `None`, since they are not stratified.
    pub network_config: Option<NetworkConfig>,
    /// Replacement config, as in [`poisson_axes`].
    pub replacement: Option<NodeConfigRef>,
    /// Whether the strategy ever spawns nodes.
    pub spawns: bool,
    /// Simulated horizon.
    pub horizon: Seconds,
    /// Base seed.
    pub base_seed: u64,
}

/// Run a stratified (or naive) experiment until the target metric's
/// half-width reaches the tolerance or the run cap is hit.
pub fn run_stratified(
    scenario: &StratifiedScenario<'_>,
    spec: &StratifiedSpec,
) -> Result<StratifiedRecord, String> {
    spec.validate()?;
    if scenario.network_config.is_some() {
        return Err("stratified mode does not model region outages yet".to_string());
    }
    let target = spec.target()?;
    let pick = |a: Estimate, d: Estimate| match target {
        ConvergenceMetric::Availability => a,
        _ => d,
    };
    let horizon = scenario.horizon;

    let mut simulator = Simulator::new(
        scenario.template.clone(),
        (scenario.strategy)(),
        (scenario.protocol)(),
        None,
        None,
        false,
    );
    let simulate = |simulator: &mut Simulator, seed: u64| -> (f64, f64) {
        simulator.reset(
            &scenario.template,
            (scenario.strategy)(),
            (scenario.protocol)(),
            Some(seed),
        );
        let result = simulator.run_through(horizon);
        observe(&result.metrics, horizon)
    };

    let mut trace = Vec::new();

    if spec.allocation == Allocation::Naive {
        let mut avail = Moments::default();
        let mut loss = Moments::default();
        let mut runs = 0usize;
        let mut budget = spec.min_runs.min(spec.max_runs);
        loop {
            while runs < budget {
                let (a, d) = simulate(
                    &mut simulator,
                    seed_for(scenario.base_seed, 0, runs as u64, 0),
                );
                avail.push(a);
                loss.push(d);
                runs += 1;
            }
            let ea = estimate(&[(1.0, avail)], &[], 0.0, spec.confidence_level);
            let ed = estimate(&[(1.0, loss)], &[], 0.0, spec.confidence_level);
            let et = pick(ea, ed);
            if spec.trace {
                trace.push(TraceRow {
                    runs,
                    mean: et.mean,
                    half_width: et.half_width,
                    clt_half_width: et.clt_half_width,
                });
            }
            let converged = spec.stopping_rule.half_width(&et) <= spec.absolute_error;
            if converged || runs >= spec.max_runs {
                return Ok(StratifiedRecord {
                    allocation: spec.allocation,
                    metric: spec.metric.clone(),
                    stopping_rule: spec.stopping_rule,
                    converged,
                    total_runs: runs,
                    tolerance: spec.absolute_error,
                    target: et,
                    availability: ea,
                    data_loss_probability: ed,
                    num_strata: 1,
                    kept_mass: 1.0,
                    exact_mass: 0.0,
                    pruned_mass: 0.0,
                    axes: Vec::new(),
                    trace,
                    strata: None,
                });
            }
            budget = next_budget(runs, spec);
        }
    }

    // -- stratified arms --------------------------------------------------

    let headroom = spec
        .slot_headroom
        .unwrap_or(scenario.template.target_cluster_size + 1);
    let leader = match spec.leader_strata {
        Some(true) => Some(initial_leader(scenario)?),
        Some(false) => None,
        None => initial_leader(scenario).ok(),
    };
    let axes = poisson_axes(
        &scenario.template,
        scenario.replacement.as_ref(),
        scenario.spawns,
        headroom,
        horizon,
        leader,
    )?;
    // Failures the cluster absorbs without losing quorum.
    let tolerated = (scenario.template.target_cluster_size as u32).saturating_sub(1) / 2;
    let is_exact = |counts: &[u32]| -> bool {
        let (own, shared): (Vec<_>, Vec<_>) = axes
            .iter()
            .zip(counts)
            .partition(|(a, _)| a.axis.reserved.is_some());
        if leader.is_none() {
            return counts.iter().all(|&k| k == 0);
        }
        own.iter().all(|(_, &k)| k == 0) && shared.iter().map(|(_, &k)| k).sum::<u32>() <= tolerated
    };
    let axis_refs: Vec<&dyn StratumAxis> = axes.iter().map(|a| a as &dyn StratumAxis).collect();
    // Metrics span [0, 1], so pruned mass times range is the pruned mass.
    let (strata, pruned_mass) = enumerate_strata(
        &axis_refs,
        spec.prune_fraction * spec.absolute_error,
        spec.max_strata,
    )?;

    simulator.forced = Some(ForcedSchedule::new(
        axes.iter().map(|a| a.axis.clone()).collect(),
    ));

    let mut times = Vec::new();
    let mut run_stratum = |simulator: &mut Simulator, index: usize, run: u64| -> (f64, f64) {
        let stratum = &strata[index];
        let mut rng = Rng::seed_from_u64(seed_for(scenario.base_seed, index, run, 1));
        let mut candidates = Vec::new();
        for (axis, (&k, spec_axis)) in stratum.counts.iter().zip(&axes).enumerate() {
            times.clear();
            spec_axis.sample_times(k, horizon, &mut rng, &mut times);
            for &time in &times {
                let slot = random_slot(&spec_axis.axis, &mut rng);
                candidates.push(Candidate { time, axis, slot });
            }
        }
        simulator
            .forced
            .as_mut()
            .expect("set above")
            .reset(candidates);
        simulate(simulator, seed_for(scenario.base_seed, index, run, 2))
    };

    // Exact strata are deterministic: simulate once, then check that a few
    // more runs agree to the last bit.
    let mut exact_avail = Vec::new();
    let mut exact_loss = Vec::new();
    let mut exact = Vec::new();
    let mut sampled = Vec::new();
    let mut total_runs = 0usize;
    for (index, stratum) in strata.iter().enumerate() {
        if is_exact(&stratum.counts) {
            let (a, d) = run_stratum(&mut simulator, index, 0);
            for run in 1..=spec.exact_checks {
                let again = run_stratum(&mut simulator, index, run);
                if again != (a, d) {
                    return Err(format!(
                        "stratum {:?} was assumed exact but run {run} gave {again:?}, not {:?}",
                        stratum.counts,
                        (a, d)
                    ));
                }
            }
            total_runs += spec.exact_checks as usize;
            exact_avail.push((stratum.weight, a));
            exact_loss.push((stratum.weight, d));
            exact.push(index);
            total_runs += 1;
        } else {
            sampled.push(index);
        }
    }
    let exact_mass: f64 = exact_avail.iter().map(|&(w, _)| w).sum();
    let sampled_mass: f64 = sampled.iter().map(|&i| strata[i].weight).sum();

    let mut avail = vec![Moments::default(); sampled.len()];
    let mut loss = vec![Moments::default(); sampled.len()];
    let floor = spec.min_stratum_runs;
    let mut targets = vec![floor; sampled.len()];
    let mut budget = floor as usize * sampled.len();

    let estimates = |avail: &[Moments], loss: &[Moments]| {
        let pairs = |m: &[Moments]| -> Vec<(f64, Moments)> {
            sampled
                .iter()
                .zip(m)
                .map(|(&i, &m)| (strata[i].weight, m))
                .collect()
        };
        (
            estimate(
                &pairs(avail),
                &exact_avail,
                pruned_mass,
                spec.confidence_level,
            ),
            estimate(
                &pairs(loss),
                &exact_loss,
                pruned_mass,
                spec.confidence_level,
            ),
        )
    };

    loop {
        for (slot, &index) in sampled.iter().enumerate() {
            while avail[slot].n < targets[slot] {
                let (a, d) = run_stratum(&mut simulator, index, avail[slot].n);
                avail[slot].push(a);
                loss[slot].push(d);
                total_runs += 1;
            }
        }

        let (ea, ed) = estimates(&avail, &loss);
        let et = pick(ea, ed);
        if spec.trace {
            trace.push(TraceRow {
                runs: total_runs,
                mean: et.mean,
                half_width: et.half_width,
                clt_half_width: et.clt_half_width,
            });
        }
        let converged = total_runs >= spec.min_runs
            && spec.stopping_rule.half_width(&et) <= spec.absolute_error;
        if converged || total_runs >= spec.max_runs {
            let forced = simulator.forced.as_ref().expect("set above");
            return Ok(StratifiedRecord {
                allocation: spec.allocation,
                metric: spec.metric.clone(),
                stopping_rule: spec.stopping_rule,
                converged,
                total_runs,
                tolerance: spec.absolute_error,
                target: et,
                availability: ea,
                data_loss_probability: ed,
                num_strata: strata.len(),
                kept_mass: 1.0 - pruned_mass,
                exact_mass,
                pruned_mass,
                axes: axes
                    .iter()
                    .zip(forced.peak_held().iter().zip(forced.native_count()))
                    .map(|(a, (&peak, &native))| AxisRecord {
                        kind: a.axis.kind.as_str().to_string(),
                        rate: a.axis.rate,
                        slots: a.axis.slots,
                        mean: a.mean,
                        peak_held: peak,
                        native_nodes: native,
                        leader: a.axis.reserved.is_some(),
                    })
                    .collect(),
                trace,
                strata: spec.emit_strata.then(|| {
                    let exact_rows = exact.iter().zip(exact_avail.iter().zip(&exact_loss)).map(
                        |(&i, (&(_, a), &(_, d)))| StratumRecord {
                            counts: strata[i].counts.clone(),
                            weight: strata[i].weight,
                            exact: true,
                            runs: 1,
                            availability_mean: a,
                            availability_variance: 0.0,
                            data_loss_mean: d,
                            data_loss_variance: 0.0,
                        },
                    );
                    let sampled_rows =
                        sampled
                            .iter()
                            .zip(avail.iter().zip(&loss))
                            .map(|(&i, (a, d))| StratumRecord {
                                counts: strata[i].counts.clone(),
                                weight: strata[i].weight,
                                exact: false,
                                runs: a.n,
                                availability_mean: a.mean,
                                availability_variance: a.variance(),
                                data_loss_mean: d.mean,
                                data_loss_variance: d.variance(),
                            });
                    exact_rows.chain(sampled_rows).collect()
                }),
            });
        }

        budget = next_budget(budget, spec).max(spec.min_runs.min(spec.max_runs));
        let target_moments = match target {
            ConvergenceMetric::Availability => &avail,
            _ => &loss,
        };
        let scores: Vec<f64> = match spec.allocation {
            Allocation::Proportional => sampled.iter().map(|&i| strata[i].weight).collect(),
            Allocation::Neyman => neyman_scores(&sampled, &strata, target_moments),
            Allocation::Bernstein => {
                let weights: Vec<f64> = sampled.iter().map(|&i| strata[i].weight).collect();
                let sigmas = shrunk_sigmas(&sampled, &strata, target_moments);
                let log_term = (4.0 / (1.0 - spec.confidence_level)).ln();
                let plan = bernstein_allocation(&weights, &sigmas, budget as f64, log_term);
                for (target, n) in targets.iter_mut().zip(plan) {
                    *target = (*target).max(n.ceil() as u64).max(floor);
                }
                continue;
            }
            Allocation::Naive => unreachable!("handled above"),
        };
        // Every stratum keeps its first run; the rest of the budget is
        // shared out.  Sharing all of it would leave strata on a handful of
        // runs paying up to double on the range term, `w_s / (n_s - 1)`.
        let total: f64 = scores.iter().sum();
        let extra = budget.saturating_sub(sampled.len()) as f64;
        for (slot, score) in scores.iter().enumerate() {
            let share = if total > 0.0 {
                score / total
            } else {
                strata[sampled[slot]].weight / sampled_mass
            };
            let want = 1 + (extra * share).ceil() as u64;
            targets[slot] = targets[slot].max(want).max(floor);
        }
    }
}

/// The node the protocol makes leader at the start of every run.
///
/// Every run starts from the same cluster, and leader choice at start-up is
/// deterministic, so this is the same node in all of them.
fn initial_leader(scenario: &StratifiedScenario<'_>) -> Result<Sym, String> {
    let mut protocol = (scenario.protocol)();
    let mut rng = Rng::seed_from_u64(0);
    let mut events = Vec::new();
    protocol.on_simulation_start(&scenario.template, &mut rng, &mut events);
    protocol
        .leader_id()
        .ok_or_else(|| "leader_strata needs a protocol with an initial leader".to_string())
}

/// The next run budget: `growth` times the current one, at least one more.
fn next_budget(current: usize, spec: &StratifiedSpec) -> usize {
    let grown = (current as f64 * spec.growth).ceil() as usize;
    grown.max(current + 1).min(spec.max_runs)
}

/// Neyman scores `w_s * sigma_s`, with each stratum's variance shrunk
/// toward the pooled within-stratum variance.
///
/// Two agreeing pilot samples give a zero sample variance, and a raw
/// Neyman allocation would then never revisit that stratum.  Shrinking
/// with one pseudo-observation at the pooled variance keeps every stratum
/// in play until its own samples say otherwise.
fn neyman_scores(sampled: &[usize], strata: &[Stratum], moments: &[Moments]) -> Vec<f64> {
    shrunk_sigmas(sampled, strata, moments)
        .into_iter()
        .zip(sampled)
        .map(|(sigma, &i)| strata[i].weight * sigma)
        .collect()
}

/// Per-stratum standard deviations, shrunk toward the pooled variance with
/// one pseudo-observation; see [`neyman_scores`].
fn shrunk_sigmas(sampled: &[usize], strata: &[Stratum], moments: &[Moments]) -> Vec<f64> {
    let mut pooled_num = 0.0;
    let mut pooled_den = 0.0;
    for (&i, m) in sampled.iter().zip(moments) {
        pooled_num += strata[i].weight * m.variance();
        pooled_den += strata[i].weight;
    }
    let pooled = if pooled_den > 0.0 {
        pooled_num / pooled_den
    } else {
        0.0
    };
    sampled
        .iter()
        .zip(moments)
        .map(|(_, m)| {
            let n = m.n.max(1) as f64;
            (((n - 1.0) * m.variance() + pooled) / n).sqrt()
        })
        .collect()
}

/// Split `budget` runs over strata to minimise the Bernstein half-width
///
/// ```text
/// sqrt(2 L sum_s w_s^2 sigma_s^2 / n_s) + (7/3) L max_s w_s / (n_s - 1)
/// ```
///
/// For a range level `r`, floors `n_s >= 1 + w_s / r` hold the range term
/// at `(7/3) L r`, and the variance term is smallest with the rest of the
/// budget water-filled Neyman-style: `n_s = max(floor_s, lambda w_s sigma_s)`.
/// The best `r` is found on a log grid between the tightest feasible level
/// (pure proportional: the whole budget on floors) and one where the floors
/// cost almost nothing (close to pure Neyman).
pub fn bernstein_allocation(
    weights: &[f64],
    sigmas: &[f64],
    budget: f64,
    log_term: f64,
) -> Vec<f64> {
    let strata = weights.len() as f64;
    let mass: f64 = weights.iter().sum();
    let floors_for = |r: f64| -> Vec<f64> { weights.iter().map(|w| 1.0 + w / r).collect() };
    if budget <= strata || mass <= 0.0 {
        return floors_for(f64::INFINITY);
    }
    let tightest = mass / (budget - strata);
    let neyman: Vec<f64> = weights.iter().zip(sigmas).map(|(w, s)| w * s).collect();
    let neyman_total: f64 = neyman.iter().sum();

    let plan_for = |r: f64| -> Vec<f64> {
        let floors = floors_for(r);
        if neyman_total <= 0.0 {
            return floors;
        }
        let spend = |lambda: f64| -> f64 {
            floors
                .iter()
                .zip(&neyman)
                .map(|(f, a)| f.max(lambda * a))
                .sum()
        };
        let (mut lo, mut hi) = (0.0, budget / neyman_total);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if spend(mid) > budget {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        floors
            .iter()
            .zip(&neyman)
            .map(|(f, a)| f.max(lo * a))
            .collect()
    };
    let half_width = |plan: &[f64]| -> f64 {
        let mut variance = 0.0;
        let mut range = 0.0f64;
        for ((w, s), n) in weights.iter().zip(sigmas).zip(plan) {
            variance += w * w * s * s / n;
            range = range.max(w / (n - 1.0));
        }
        (2.0 * variance * log_term).sqrt() + 7.0 / 3.0 * log_term * range
    };

    let mut best = plan_for(tightest);
    let mut best_width = half_width(&best);
    const STEPS: usize = 48;
    for step in 1..=STEPS {
        // Up to a factor 1e6 looser than the tightest level.
        let r = tightest * 10f64.powf(6.0 * step as f64 / STEPS as f64);
        let plan = plan_for(r);
        let width = half_width(&plan);
        if width < best_width {
            best = plan;
            best_width = width;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poisson(mean: f64) -> PoissonAxis {
        PoissonAxis {
            axis: ScheduleAxis {
                kind: FailureKind::Transient,
                rate: 1.0,
                slots: 1,
                reserved: None,
            },
            mean,
        }
    }

    #[test]
    fn poisson_pmf_sums_to_one() {
        let axis = poisson(3.7);
        let total: f64 = (0..100).map(|k| axis.pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert_eq!(axis.mode(), 3);
    }

    #[test]
    fn strata_come_out_most_likely_first_and_account_for_all_mass() {
        let a = poisson(0.4);
        let b = poisson(2.5);
        let axes: Vec<&dyn StratumAxis> = vec![&a, &b];
        let (strata, pruned) = enumerate_strata(&axes, 1e-9, 100_000).unwrap();
        for pair in strata.windows(2) {
            assert!(pair[0].weight >= pair[1].weight * (1.0 - 1e-12));
        }
        let kept: f64 = strata.iter().map(|s| s.weight).sum();
        assert!((kept + pruned - 1.0).abs() < 1e-12);
        assert!(pruned <= 1e-9);
        // Every stratum heavier than the cut was kept.
        let lightest = strata.last().unwrap().weight;
        for i in 0..30u32 {
            for j in 0..30u32 {
                let w = a.pmf(i) * b.pmf(j);
                if w > lightest * (1.0 + 1e-9) {
                    assert!(strata.iter().any(|s| s.counts == vec![i, j]), "{i},{j}");
                }
            }
        }
    }

    #[test]
    fn single_stratum_bound_is_maurer_pontil() {
        let mut m = Moments::default();
        for x in [0.9, 1.0, 0.95, 1.0, 0.99] {
            m.push(x);
        }
        let e = estimate(&[(1.0, m)], &[], 0.0, 0.99);
        let log_term = (4.0f64 / 0.01).ln();
        let expect = (2.0 * m.variance() * log_term / 5.0).sqrt() + 7.0 * log_term / (3.0 * 4.0);
        assert!((e.half_width - expect).abs() < 1e-15);
    }

    #[test]
    fn agreeing_samples_do_not_collapse_the_bound() {
        let mut m = Moments::default();
        for _ in 0..1000 {
            m.push(1.0);
        }
        let e = estimate(&[(1.0, m)], &[], 0.0, 0.99);
        assert_eq!(e.variance_term, 0.0);
        assert!(e.range_term > 0.0);
    }

    #[test]
    fn bernstein_allocation_beats_both_pure_rules_on_its_own_objective() {
        // One noisy common stratum and many quiet rare ones: Neyman starves
        // the rare ones, proportional overspends on them.
        let mut weights = vec![0.5];
        let mut sigmas = vec![0.05];
        for _ in 0..200 {
            weights.push(0.5 / 200.0);
            sigmas.push(1e-4);
        }
        let log_term = (4.0f64 / 0.01).ln();
        let budget = 100_000.0;
        let width = |plan: &[f64]| -> f64 {
            let mut variance = 0.0;
            let mut range = 0.0f64;
            for ((w, s), n) in weights.iter().zip(&sigmas).zip(plan) {
                variance += w * w * s * s / n;
                range = range.max(w / (n - 1.0));
            }
            (2.0 * variance * log_term).sqrt() + 7.0 / 3.0 * log_term * range
        };
        let extra = budget - weights.len() as f64;
        let proportional: Vec<f64> = weights.iter().map(|w| 1.0 + extra * w).collect();
        let total: f64 = weights.iter().zip(&sigmas).map(|(w, s)| w * s).sum();
        let neyman: Vec<f64> = weights
            .iter()
            .zip(&sigmas)
            .map(|(w, s)| (1.0 + extra * w * s / total).max(2.0))
            .collect();
        let plan = bernstein_allocation(&weights, &sigmas, budget, log_term);
        assert!((plan.iter().sum::<f64>() - budget).abs() < 1e-6 * budget);
        assert!(width(&plan) < width(&proportional));
        assert!(width(&plan) < width(&neyman));
    }

    #[test]
    fn pruned_mass_is_imputed_at_the_midpoint_and_bounded() {
        let e = estimate(&[], &[(0.9, 1.0)], 0.1, 0.99);
        assert!((e.mean - 0.95).abs() < 1e-15);
        assert!((e.bias - 0.05).abs() < 1e-15);
    }
}
