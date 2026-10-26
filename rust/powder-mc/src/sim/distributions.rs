//! Time units and probability distributions.
//!
//! Port of `powder/simulation/distributions.py`.  All time values use seconds
//! as the canonical unit.
//!
//! Python models distributions as an ABC with one subclass per family.  Here
//! they collapse into a single enum: the set of families is closed, and enum
//! dispatch keeps sampling in the hot path free of virtual calls.

use rand::{RngExt, SeedableRng};
use rand_distr::Distribution as _;

/// All simulation times are seconds.
pub type Seconds = f64;

/// Convert hours to seconds.
pub fn hours(h: f64) -> Seconds {
    h * 3600.0
}

/// Convert days to seconds.
pub fn days(d: f64) -> Seconds {
    d * 86400.0
}

/// Convert minutes to seconds.
pub fn minutes(m: f64) -> Seconds {
    m * 60.0
}

/// The simulator's random number generator.
///
/// PCG64, chosen for speed and a small state.  The stream differs from
/// NumPy's `default_rng` by design -- the port targets statistical, not
/// bitwise, agreement.
pub type Rng = rand_pcg::Pcg64;

/// Build a generator from an optional seed, mirroring
/// `np.random.default_rng(seed)`.  `None` draws entropy from the OS.
pub fn make_rng(seed: Option<u64>) -> Rng {
    match seed {
        Some(s) => Rng::seed_from_u64(s),
        None => Rng::from_rng(&mut rand::rng()),
    }
}

/// Rejected distribution parameters.  Mirrors the `ValueError`s raised by the
/// Python constructors, with the same messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributionError(pub String);

impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DistributionError {}

type Result<T> = std::result::Result<T, DistributionError>;

/// Digitized CDF table for [`Distribution::Empirical`]: `F` is non-decreasing
/// from 0 to 1 at the support points `t` (`t = ln x` when `log_x`).
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalTable {
    /// Support points (log-seconds when `log_x`), strictly increasing.
    pub t: Vec<f64>,
    /// CDF values at `t`, non-decreasing, `F[0] = 0`, `F[last] = 1`.
    pub f: Vec<f64>,
    /// Interpolate in log space.
    pub log_x: bool,
    /// Exact mean of the piecewise-linear inverse CDF.
    pub mean: f64,
}

impl EmpiricalTable {
    /// Inverse CDF at `u` (linear interpolation of t against F, as numpy.interp).
    #[inline]
    pub fn ppf(&self, u: f64) -> f64 {
        let f = &self.f;
        let n = f.len();
        // first index with f[i] >= u (binary search)
        let i = f.partition_point(|&v| v < u).clamp(1, n - 1);
        let (f0, f1) = (f[i - 1], f[i]);
        let (t0, t1) = (self.t[i - 1], self.t[i]);
        let t = if f1 > f0 { t0 + (t1 - t0) * (u - f0) / (f1 - f0) } else { t1 };
        if self.log_x { t.exp() } else { t }
    }
}

/// A probability distribution over non-negative times or rates.
///
/// Not `Copy`: the empirical and mixture families carry shared (`Arc`) data,
/// so cloning stays cheap.
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    /// Memoryless; parameterised by rate (events per unit time).
    Exponential { rate: f64 },
    /// Generalises the exponential; `shape < 1` is infant mortality,
    /// `shape > 1` is wear-out.
    Weibull { shape: f64, scale: f64 },
    /// Gaussian, clamped below at `min_val` on sampling.
    Normal { mean: f64, std: f64, min_val: f64 },
    /// Uniform over `[low, high)`.
    Uniform { low: f64, high: f64 },
    /// Deterministic.
    Constant { value: f64 },
    /// `ln X ~ Normal(mu, sigma)`.
    LogNormal { mu: f64, sigma: f64 },
    /// Digitized CDF (piecewise-linear inverse CDF, optionally in log-time).
    Empirical(std::sync::Arc<EmpiricalTable>),
    /// Finite mixture: `(cumulative weight, component)`, last cumulative weight = 1.
    Mixture(std::sync::Arc<Vec<(f64, Distribution)>>),
    /// Non-homogeneous Poisson clock: piecewise hazards + a constant rate.
    /// Sampled from the current simulation time with [`Distribution::sample_at`].
    TimeVarying(std::sync::Arc<TimeVaryingTable>),
}

/// One piecewise-constant hazard (per second) on its own clock.
///
/// The clock reads `offset` at simulation time 0.  Past the last edge the
/// last rate continues, or the series repeats when `periodic`.  `age_start`
/// (edges, cumulative weights) is the clock position of a *new* unit: a
/// replacement node redraws it (age clocks); calendar clocks have none.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewisePart {
    pub edges: Vec<f64>,
    pub rates: Vec<f64>,
    cum: Vec<f64>,
    full: f64,
    pub periodic: bool,
    pub offset: f64,
    pub age_start: Option<(Vec<f64>, Vec<f64>)>,
}

impl PiecewisePart {
    /// Validate and precompute cumulative hazards.
    pub fn new(edges: Vec<f64>, rates: Vec<f64>, periodic: bool, offset: f64,
               age_start: Option<(Vec<f64>, Vec<f64>)>) -> Result<Self> {
        if edges.len() != rates.len() + 1 || edges.first() != Some(&0.0) {
            return Err(DistributionError("piecewise hazard: edges must start at 0 and have len(rates)+1 entries".into()));
        }
        if edges.windows(2).any(|w| w[1] <= w[0]) || rates.iter().any(|r| !(*r >= 0.0)) {
            return Err(DistributionError("piecewise hazard: edges must increase and rates be >= 0".into()));
        }
        let mut cum = vec![0.0; edges.len()];
        for i in 0..rates.len() {
            cum[i + 1] = cum[i] + (edges[i + 1] - edges[i]) * rates[i];
        }
        let full = *cum.last().unwrap();
        let age_start = match age_start {
            Some((e, w)) => {
                if e.len() != w.len() + 1 || w.iter().any(|x| !(*x >= 0.0)) || w.iter().sum::<f64>() <= 0.0 {
                    return Err(DistributionError("piecewise hazard: bad age_start".into()));
                }
                let tot: f64 = w.iter().sum();
                let mut acc = 0.0;
                let cw = w.iter().map(|x| { acc += x / tot; acc }).collect();
                Some((e, cw))
            }
            None => None,
        };
        Ok(PiecewisePart { edges, rates, cum, full, periodic, offset, age_start })
    }

    fn span(&self) -> f64 { *self.edges.last().unwrap() }

    fn cum_clock(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let span = self.span();
        if x >= span {
            if self.periodic {
                let k = (x / span).floor();
                return k * self.full + self.cum_clock(x - k * span);
            }
            return self.full + (x - span) * self.rates[self.rates.len() - 1];
        }
        let i = self.edges.partition_point(|e| *e <= x) - 1;
        self.cum[i] + (x - self.edges[i]) * self.rates[i]
    }

    fn inv_in(&self, c: f64) -> f64 {
        let n = self.cum.len();
        let i = self.cum.partition_point(|v| *v < c).clamp(1, n - 1);
        let r = self.rates[i - 1];
        if r > 0.0 { self.edges[i - 1] + (c - self.cum[i - 1]) / r } else { self.edges[i] }
    }

    fn inv_clock(&self, c: f64) -> f64 {
        if c <= 0.0 { return 0.0; }
        let span = self.span();
        if self.periodic {
            if self.full <= 0.0 { return f64::INFINITY; }
            let k = (c / self.full).floor();
            return k * span + self.inv_in(c - k * self.full);
        }
        if c >= self.full {
            let r = self.rates[self.rates.len() - 1];
            return if r <= 0.0 { f64::INFINITY } else { span + (c - self.full) / r };
        }
        self.inv_in(c)
    }

    /// Integrated hazard over simulation time `[t0, t1]`.
    pub fn cumulative(&self, t0: f64, t1: f64) -> f64 {
        self.cum_clock(self.offset + t1) - self.cum_clock(self.offset + t0)
    }

    /// Simulation time of the next event after `now`.
    pub fn next_event(&self, now: f64, rng: &mut Rng) -> f64 {
        let e: f64 = rand_distr::Exp1.sample(rng);
        let x = self.inv_clock(self.cum_clock(self.offset + now) + e);
        if x.is_infinite() { f64::INFINITY } else { x - self.offset }
    }

    fn respawned(&self, now: f64, rng: &mut Rng) -> Option<PiecewisePart> {
        let (e, cw) = self.age_start.as_ref()?;
        let u: f64 = rng.random();
        let i = cw.partition_point(|c| *c < u).min(cw.len() - 1);
        let v: f64 = rng.random();
        let age = e[i] + (e[i + 1] - e[i]) * v;
        let mut p = self.clone();
        p.offset = age - now;
        Some(p)
    }
}

/// Superposition of piecewise hazards and a constant rate.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeVaryingTable {
    pub parts: Vec<PiecewisePart>,
    pub constant_rate: f64,
    pub mean_horizon: f64,
    mean: f64,
}

impl TimeVaryingTable {
    /// Build; `mean` = 1 / (average hazard over `mean_horizon` from time 0), for planning only.
    pub fn new(parts: Vec<PiecewisePart>, constant_rate: f64, mean_horizon: f64) -> Result<Self> {
        if !(constant_rate >= 0.0) || !(mean_horizon > 0.0) {
            return Err(DistributionError("time-varying hazard: constant_rate >= 0 and mean_horizon > 0".into()));
        }
        let avg = constant_rate + parts.iter().map(|p| p.cumulative(0.0, mean_horizon)).sum::<f64>() / mean_horizon;
        let mean = if avg > 0.0 { 1.0 / avg } else { f64::INFINITY };
        Ok(TimeVaryingTable { parts, constant_rate, mean_horizon, mean })
    }

    fn sample_at(&self, now: f64, rng: &mut Rng) -> f64 {
        let mut nxt = f64::INFINITY;
        for p in &self.parts {
            nxt = nxt.min(p.next_event(now, rng));
        }
        if self.constant_rate > 0.0 {
            let e: f64 = rand_distr::Exp1.sample(rng);
            nxt = nxt.min(now + e / self.constant_rate);
        }
        nxt - now
    }
}

impl Distribution {
    /// Exponential with the given rate.  Rate must be positive.
    pub fn exponential(rate: f64) -> Result<Self> {
        if rate <= 0.0 {
            return Err(DistributionError(format!(
                "Rate must be positive, got {rate}"
            )));
        }
        Ok(Distribution::Exponential { rate })
    }

    /// Weibull with the given shape and scale.  Both must be positive.
    pub fn weibull(shape: f64, scale: f64) -> Result<Self> {
        if shape <= 0.0 {
            return Err(DistributionError(format!(
                "Shape must be positive, got {shape}"
            )));
        }
        if scale <= 0.0 {
            return Err(DistributionError(format!(
                "Scale must be positive, got {scale}"
            )));
        }
        Ok(Distribution::Weibull { shape, scale })
    }

    /// Normal with the given mean and standard deviation, clamped at
    /// `min_val` (0.0 in the Python default).  Std must be positive.
    pub fn normal(mean: f64, std: f64, min_val: f64) -> Result<Self> {
        if std <= 0.0 {
            return Err(DistributionError(format!(
                "Standard deviation must be positive, got {std}"
            )));
        }
        Ok(Distribution::Normal {
            mean,
            std,
            min_val,
        })
    }

    /// Uniform over `[low, high)`.  `low` must be strictly less than `high`.
    pub fn uniform(low: f64, high: f64) -> Result<Self> {
        if low >= high {
            return Err(DistributionError(format!(
                "Low must be less than high, got low={low}, high={high}"
            )));
        }
        Ok(Distribution::Uniform { low, high })
    }

    /// Deterministic distribution.  Any value is accepted.
    pub fn constant(value: f64) -> Self {
        Distribution::Constant { value }
    }

    /// Log-normal with log-mean `mu` and log-std `sigma` (> 0).
    pub fn lognormal(mu: f64, sigma: f64) -> Result<Self> {
        if !(sigma > 0.0) || !mu.is_finite() {
            return Err(DistributionError(format!(
                "LogNormal needs finite mu and sigma > 0, got mu={mu}, sigma={sigma}"
            )));
        }
        Ok(Distribution::LogNormal { mu, sigma })
    }

    /// Empirical distribution from CDF points `(x, F)`; `F` must start at 0 and
    /// end at 1 (no parametric tails in the engine).
    pub fn empirical(x: &[f64], f: &[f64], log_x: bool) -> Result<Self> {
        let n = x.len();
        if n < 2 || f.len() != n {
            return Err(DistributionError("Empirical needs >= 2 points and equal-length x, F".into()));
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(DistributionError("Empirical x must be strictly increasing".into()));
        }
        if f.windows(2).any(|w| w[1] < w[0]) || f[0] != 0.0 || (f[n - 1] - 1.0).abs() > 1e-9 {
            return Err(DistributionError(
                "Empirical F must be non-decreasing from 0 to 1 (tails are not supported in the engine)".into(),
            ));
        }
        if log_x && x[0] <= 0.0 {
            return Err(DistributionError("Empirical log_x requires x > 0".into()));
        }
        let t: Vec<f64> = if log_x { x.iter().map(|v| v.ln()).collect() } else { x.to_vec() };
        let mut fv = f.to_vec();
        fv[n - 1] = 1.0;
        // Exact mean of the piecewise-linear inverse CDF: each segment contributes
        // dF * mean of x over the segment (log-uniform when log_x, uniform otherwise).
        let mut mean = 0.0;
        for i in 1..n {
            let df = fv[i] - fv[i - 1];
            if df <= 0.0 {
                continue;
            }
            let (a, b) = (t[i - 1], t[i]);
            let seg = if log_x {
                if (b - a).abs() < 1e-12 { a.exp() } else { (b.exp() - a.exp()) / (b - a) }
            } else {
                0.5 * (a + b)
            };
            mean += df * seg;
        }
        Ok(Distribution::Empirical(std::sync::Arc::new(EmpiricalTable { t, f: fv, log_x, mean })))
    }

    /// Mixture of `components` with non-negative `weights` (normalised).
    pub fn mixture(weights: &[f64], components: Vec<Distribution>) -> Result<Self> {
        if weights.is_empty() || weights.len() != components.len() {
            return Err(DistributionError("Mixture needs equal, non-zero numbers of weights and components".into()));
        }
        let total: f64 = weights.iter().sum();
        if weights.iter().any(|w| *w < 0.0) || !(total > 0.0) {
            return Err(DistributionError("Mixture weights must be non-negative with a positive sum".into()));
        }
        let mut acc = 0.0;
        let mut v = Vec::with_capacity(weights.len());
        for (w, c) in weights.iter().zip(components) {
            acc += w / total;
            v.push((acc, c));
        }
        if let Some(last) = v.last_mut() {
            last.0 = 1.0;
        }
        Ok(Distribution::Mixture(std::sync::Arc::new(v)))
    }

    /// Draw one sample.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Distribution::LogNormal { mu, sigma } => {
                let z: f64 = rand_distr::StandardNormal.sample(rng);
                (mu + sigma * z).exp()
            }
            Distribution::Empirical(table) => {
                let u: f64 = rng.random();
                table.ppf(u)
            }
            Distribution::Mixture(parts) => {
                let u: f64 = rng.random();
                let i = parts.partition_point(|(c, _)| *c <= u).min(parts.len() - 1);
                parts[i].1.sample(rng)
            }
            Distribution::TimeVarying(t) => t.sample_at(0.0, rng),
            // Exp1 is the ziggurat unit-exponential sampler; dividing by the
            // rate is equivalent to NumPy's `exponential(scale=1/rate)` and
            // avoids per-call distribution setup.
            &Distribution::Exponential { rate } => {
                let e: f64 = rand_distr::Exp1.sample(rng);
                e / rate
            }
            // NumPy's `weibull(a)` is `standard_exponential() ** (1/a)`.
            &Distribution::Weibull { shape, scale } => {
                let e: f64 = rand_distr::Exp1.sample(rng);
                scale * e.powf(1.0 / shape)
            }
            &Distribution::Normal {
                mean,
                std,
                min_val,
            } => {
                let z: f64 = rand_distr::StandardNormal.sample(rng);
                let value = mean + std * z;
                if value > min_val {
                    value
                } else {
                    min_val
                }
            }
            &Distribution::Uniform { low, high } => {
                let u: f64 = rng.random();
                low + (high - low) * u
            }
            &Distribution::Constant { value } => value,
        }
    }

    /// Theoretical mean.
    ///
    /// Used for upfront estimation when planning sync paths (log-only vs
    /// snapshot), where the decision must be made before the sync starts and
    /// sampling would be inappropriate.
    ///
    /// Note that `Normal::mean` reports the untruncated mean, matching
    /// Python -- the `min_val` clamp is applied on sampling only.
    pub fn mean(&self) -> f64 {
        match self {
            &Distribution::Exponential { rate } => 1.0 / rate,
            &Distribution::Weibull { shape, scale } => {
                scale * statrs::function::gamma::gamma(1.0 + 1.0 / shape)
            }
            &Distribution::Normal { mean, .. } => mean,
            &Distribution::Uniform { low, high } => (low + high) / 2.0,
            &Distribution::Constant { value } => value,
            &Distribution::LogNormal { mu, sigma } => (mu + 0.5 * sigma * sigma).exp(),
            Distribution::Empirical(table) => table.mean,
            Distribution::Mixture(parts) => {
                let mut prev = 0.0;
                parts.iter().map(|(c, d)| { let w = c - prev; prev = *c; w * d.mean() }).sum()
            }
            Distribution::TimeVarying(t) => t.mean,
        }
    }

    /// Delay until the next event from simulation time `now` (time-aware for
    /// [`Distribution::TimeVarying`]; every other family ignores `now`).
    #[inline]
    pub fn sample_at(&self, now: f64, rng: &mut Rng) -> f64 {
        match self {
            Distribution::TimeVarying(t) => t.sample_at(now, rng),
            _ => self.sample(rng),
        }
    }

    /// The clock of a replacement unit created at `now`: age clocks redraw
    /// their position; `None` when nothing changes.
    pub fn respawned(&self, now: f64, rng: &mut Rng) -> Option<Distribution> {
        let Distribution::TimeVarying(t) = self else { return None; };
        if t.parts.iter().all(|p| p.age_start.is_none()) {
            return None;
        }
        let parts = t.parts.iter().map(|p| p.respawned(now, rng).unwrap_or_else(|| p.clone())).collect();
        Some(Distribution::TimeVarying(std::sync::Arc::new(TimeVaryingTable {
            parts, constant_rate: t.constant_rate, mean_horizon: t.mean_horizon, mean: t.mean })))
    }

    /// Exponential rate approximation (`1 / mean`).
    ///
    /// Exact for `Exponential`; for the other families it is the rate of the
    /// exponential with matching mean, which is the approximation the Markov
    /// builders consume.  Returns infinity when the mean is non-positive.
    pub fn approx_rate(&self) -> f64 {
        let m = self.mean();
        if m <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / m
        }
    }
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            &Distribution::Exponential { rate } => write!(f, "Exponential(rate={rate})"),
            &Distribution::Weibull { shape, scale } => {
                write!(f, "Weibull(shape={shape}, scale={scale})")
            }
            &Distribution::Normal {
                mean,
                std,
                min_val,
            } => write!(f, "Normal(mean={mean}, std={std}, min_val={min_val})"),
            &Distribution::Uniform { low, high } => write!(f, "Uniform(low={low}, high={high})"),
            &Distribution::Constant { value } => write!(f, "Constant(value={value})"),
            &Distribution::LogNormal { mu, sigma } => write!(f, "LogNormal(mu={mu}, sigma={sigma})"),
            Distribution::Empirical(t) => write!(f, "Empirical(n={}, log_x={})", t.t.len(), t.log_x),
            Distribution::Mixture(p) => write!(f, "Mixture(n={})", p.len()),
            Distribution::TimeVarying(t) => write!(f, "TimeVarying(parts={}, constant_rate={})", t.parts.len(), t.constant_rate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_unit_helpers() {
        assert_eq!(hours(1.0), 3600.0);
        assert_eq!(days(1.0), 86400.0);
        assert_eq!(minutes(1.0), 60.0);
        assert_eq!(hours(0.5), 1800.0);
    }

    #[test]
    fn constructors_reject_invalid_parameters() {
        assert!(Distribution::exponential(0.0).is_err());
        assert!(Distribution::exponential(-1.0).is_err());
        assert!(Distribution::weibull(0.0, 1.0).is_err());
        assert!(Distribution::weibull(1.0, 0.0).is_err());
        assert!(Distribution::normal(1.0, 0.0, 0.0).is_err());
        assert!(Distribution::uniform(5.0, 5.0).is_err());
        assert!(Distribution::uniform(6.0, 5.0).is_err());
    }

    #[test]
    fn means_are_correct() {
        assert_eq!(Distribution::exponential(0.25).unwrap().mean(), 4.0);
        assert_eq!(Distribution::uniform(2.0, 8.0).unwrap().mean(), 5.0);
        assert_eq!(Distribution::constant(7.0).mean(), 7.0);
        assert_eq!(Distribution::normal(3.0, 1.0, 0.0).unwrap().mean(), 3.0);
        // shape=1 reduces Weibull to Exponential with mean == scale.
        let w = Distribution::weibull(1.0, 10.0).unwrap();
        assert!((w.mean() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn approx_rate_is_inverse_mean() {
        assert_eq!(Distribution::exponential(0.5).unwrap().approx_rate(), 0.5);
        assert_eq!(Distribution::constant(4.0).approx_rate(), 0.25);
        assert_eq!(Distribution::constant(0.0).approx_rate(), f64::INFINITY);
    }

    #[test]
    fn constant_always_returns_its_value() {
        let mut rng = make_rng(Some(1));
        let d = Distribution::constant(42.0);
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), 42.0);
        }
    }

    #[test]
    fn uniform_samples_stay_in_range() {
        let mut rng = make_rng(Some(2));
        let d = Distribution::uniform(3.0, 9.0).unwrap();
        for _ in 0..10_000 {
            let x = d.sample(&mut rng);
            assert!((3.0..9.0).contains(&x));
        }
    }

    #[test]
    fn normal_samples_respect_min_val() {
        let mut rng = make_rng(Some(3));
        let d = Distribution::normal(0.0, 1.0, 0.0).unwrap();
        for _ in 0..10_000 {
            assert!(d.sample(&mut rng) >= 0.0);
        }
    }

    #[test]
    fn sample_means_approach_theoretical_means() {
        let mut rng = make_rng(Some(4));
        for d in [
            Distribution::exponential(0.01).unwrap(),
            Distribution::weibull(1.5, 100.0).unwrap(),
            Distribution::uniform(10.0, 50.0).unwrap(),
            Distribution::normal(100.0, 5.0, 0.0).unwrap(),
        ] {
            let n = 200_000;
            let total: f64 = (0..n).map(|_| d.sample(&mut rng)).sum();
            let observed = total / n as f64;
            let expected = d.mean();
            let rel = (observed - expected).abs() / expected.abs();
            assert!(rel < 0.02, "{d}: observed {observed}, expected {expected}");
        }
    }

    #[test]
    fn lognormal_mean_and_samples() {
        let d = Distribution::lognormal(6.0, 0.8).unwrap();
        let expected = (6.0f64 + 0.32).exp();
        assert!((d.mean() - expected).abs() < 1e-9);
        let mut rng = make_rng(Some(11));
        let n = 400_000;
        let m: f64 = (0..n).map(|_| d.sample(&mut rng)).sum::<f64>() / n as f64;
        assert!((m / expected - 1.0).abs() < 0.02, "{m} vs {expected}");
        assert!(Distribution::lognormal(1.0, 0.0).is_err());
    }

    #[test]
    fn empirical_inverse_cdf_and_mean() {
        // log-uniform on [10, 1000]: F linear in ln x; mean = (1000 - 10) / ln(100)
        let d = Distribution::empirical(&[10.0, 1000.0], &[0.0, 1.0], true).unwrap();
        let exact = 990.0 / 100f64.ln();
        assert!((d.mean() - exact).abs() < 1e-9);
        if let Distribution::Empirical(t) = &d {
            assert!((t.ppf(0.5) - 100.0).abs() < 1e-9);
            assert!((t.ppf(0.0) - 10.0).abs() < 1e-9 && (t.ppf(1.0) - 1000.0).abs() < 1e-9);
        } else {
            panic!("not empirical");
        }
        let mut rng = make_rng(Some(12));
        let n = 400_000;
        let m: f64 = (0..n).map(|_| d.sample(&mut rng)).sum::<f64>() / n as f64;
        assert!((m / exact - 1.0).abs() < 0.01);
        // flat stretch (no mass between 20 and 50) and a jump-like rise
        let d2 = Distribution::empirical(&[10.0, 20.0, 50.0, 51.0], &[0.0, 0.5, 0.5, 1.0], true).unwrap();
        let mut below = 0;
        for _ in 0..100_000 {
            let x = d2.sample(&mut rng);
            assert!(!(20.0 + 1e-9 < x && x < 50.0 - 1e-9), "sample {x} in the flat stretch");
            below += (x <= 20.0) as usize;
        }
        assert!((below as f64 / 100_000.0 - 0.5).abs() < 0.01);
        assert!(Distribution::empirical(&[1.0, 2.0], &[0.1, 1.0], true).is_err()); // tails unsupported
        assert!(Distribution::empirical(&[2.0, 1.0], &[0.0, 1.0], true).is_err());
    }

    #[test]
    fn mixture_weights_and_mean() {
        let d = Distribution::mixture(
            &[1.0, 3.0],
            vec![Distribution::constant(100.0), Distribution::constant(20.0)],
        )
        .unwrap();
        assert!((d.mean() - 40.0).abs() < 1e-12);
        let mut rng = make_rng(Some(13));
        let n = 200_000;
        let hits = (0..n).filter(|_| d.sample(&mut rng) == 100.0).count();
        assert!((hits as f64 / n as f64 - 0.25).abs() < 0.01);
        assert!(Distribution::mixture(&[], vec![]).is_err());
        assert!(Distribution::mixture(&[-1.0, 2.0], vec![Distribution::constant(1.0), Distribution::constant(2.0)]).is_err());
    }

    #[test]
    fn seeded_generators_are_reproducible() {
        let mut a = make_rng(Some(7));
        let mut b = make_rng(Some(7));
        let d = Distribution::exponential(1.0).unwrap();
        for _ in 0..1000 {
            assert_eq!(d.sample(&mut a), d.sample(&mut b));
        }
    }

    #[test]
    fn time_varying_nhpp_counts_and_respawn() {
        // rate 0.1 on [0, 10), 1.0 on [10, 20), then 1.0 onward: E[N(0, 20)] = 11
        let p = PiecewisePart::new(vec![0.0, 10.0, 20.0], vec![0.1, 1.0], false, 0.0, None).unwrap();
        let d = Distribution::TimeVarying(std::sync::Arc::new(TimeVaryingTable::new(vec![p], 0.0, 20.0).unwrap()));
        assert!((d.mean() - 20.0 / 11.0).abs() < 1e-12);
        let mut rng = make_rng(Some(3));
        let reps = 20_000;
        let mut total = 0usize;
        for _ in 0..reps {
            let mut t = 0.0;
            loop {
                t += d.sample_at(t, &mut rng);
                if t > 20.0 { break; }
                total += 1;
            }
        }
        assert!((total as f64 / reps as f64 - 11.0).abs() < 0.1);
        // periodic calendar clock with offset; respawn leaves calendar clocks alone
        let q = PiecewisePart::new(vec![0.0, 10.0, 20.0], vec![0.1, 1.0], true, 15.0, None).unwrap();
        assert!((q.cumulative(0.0, 20.0) - 11.0).abs() < 1e-12);
        let dq = Distribution::TimeVarying(std::sync::Arc::new(TimeVaryingTable::new(vec![q], 0.0, 40.0).unwrap()));
        assert!(dq.respawned(5.0, &mut rng).is_none());
        // age clock: a replacement at t = 100 draws an age in [0, 1)
        let a = PiecewisePart::new(vec![0.0, 1.0, 2.0], vec![1.0, 2.0], false, 0.0,
                                   Some((vec![0.0, 1.0], vec![1.0]))).unwrap();
        let da = Distribution::TimeVarying(std::sync::Arc::new(TimeVaryingTable::new(vec![a], 0.0, 1.0).unwrap()));
        match da.respawned(100.0, &mut rng) {
            Some(Distribution::TimeVarying(t)) => assert!(t.parts[0].offset >= -100.0 && t.parts[0].offset < -99.0),
            _ => panic!("age clock must respawn"),
        }
        assert!(PiecewisePart::new(vec![1.0, 2.0], vec![1.0], false, 0.0, None).is_err());
    }
}
