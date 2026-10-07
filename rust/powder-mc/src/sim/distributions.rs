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

/// A probability distribution over non-negative times or rates.
#[derive(Debug, Clone, Copy, PartialEq)]
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

    /// Draw one sample.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match *self {
            // Exp1 is the ziggurat unit-exponential sampler; dividing by the
            // rate is equivalent to NumPy's `exponential(scale=1/rate)` and
            // avoids per-call distribution setup.
            Distribution::Exponential { rate } => {
                let e: f64 = rand_distr::Exp1.sample(rng);
                e / rate
            }
            // NumPy's `weibull(a)` is `standard_exponential() ** (1/a)`.
            Distribution::Weibull { shape, scale } => {
                let e: f64 = rand_distr::Exp1.sample(rng);
                scale * e.powf(1.0 / shape)
            }
            Distribution::Normal {
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
            Distribution::Uniform { low, high } => {
                let u: f64 = rng.random();
                low + (high - low) * u
            }
            Distribution::Constant { value } => value,
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
        match *self {
            Distribution::Exponential { rate } => 1.0 / rate,
            Distribution::Weibull { shape, scale } => {
                scale * statrs::function::gamma::gamma(1.0 + 1.0 / shape)
            }
            Distribution::Normal { mean, .. } => mean,
            Distribution::Uniform { low, high } => (low + high) / 2.0,
            Distribution::Constant { value } => value,
        }
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
        match *self {
            Distribution::Exponential { rate } => write!(f, "Exponential(rate={rate})"),
            Distribution::Weibull { shape, scale } => {
                write!(f, "Weibull(shape={shape}, scale={scale})")
            }
            Distribution::Normal {
                mean,
                std,
                min_val,
            } => write!(f, "Normal(mean={mean}, std={std}, min_val={min_val})"),
            Distribution::Uniform { low, high } => write!(f, "Uniform(low={low}, high={high})"),
            Distribution::Constant { value } => write!(f, "Constant(value={value})"),
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
    fn seeded_generators_are_reproducible() {
        let mut a = make_rng(Some(7));
        let mut b = make_rng(Some(7));
        let d = Distribution::exponential(1.0).unwrap();
        for _ in 0..1000 {
            assert_eq!(d.sample(&mut a), d.sample(&mut b));
        }
    }
}
