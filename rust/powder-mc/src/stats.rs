//! Summary statistics used by the Monte Carlo aggregator.
//!
//! These mirror the NumPy calls in `powder/monte_carlo.py` (`np.mean`,
//! `np.std(ddof=1)`, `np.percentile`, `np.histogram(density=True)`) closely
//! enough for statistical comparison.  They are plain left-to-right
//! reductions rather than NumPy's pairwise summation, which is a deliberate
//! deviation -- see `rust/README.md`.

/// Inverse CDF of the standard normal (`scipy.stats.norm.ppf`).
pub fn norm_ppf(p: f64) -> f64 {
    use statrs::distribution::ContinuousCDF;
    statrs::distribution::Normal::new(0.0, 1.0)
        .expect("the standard normal is always valid")
        .inverse_cdf(p)
}

/// Inverse CDF of Student's t with `df` degrees of freedom
/// (`scipy.stats.t.ppf`).
///
/// # Panics
///
/// Panics if `df` is not positive.
pub fn t_ppf(p: f64, df: f64) -> f64 {
    use statrs::distribution::ContinuousCDF;
    statrs::distribution::StudentsT::new(0.0, 1.0, df)
        .expect("df must be positive")
        .inverse_cdf(p)
}

/// Half-width of a two-sided t confidence interval for the mean of `xs`.
///
/// Returns 0.0 when there are fewer than two samples.
pub fn t_ci_half_width(xs: &[f64], confidence_level: f64) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let sd = std(xs, 1);
    let alpha = 1.0 - confidence_level;
    t_ppf(1.0 - alpha / 2.0, (n - 1) as f64) * sd / (n as f64).sqrt()
}

/// Arithmetic mean.  Returns 0.0 for an empty slice, matching the guards in
/// `MonteCarloResults.availability_mean()` and friends.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for &x in xs {
        acc += x;
    }
    acc / xs.len() as f64
}

/// Sample standard deviation with the given delta degrees of freedom.
///
/// `ddof = 1` reproduces `np.std(x, ddof=1)`.  Returns 0.0 when there are
/// not enough samples to estimate a variance.
pub fn std(xs: &[f64], ddof: usize) -> f64 {
    variance(xs, ddof).sqrt()
}

/// Sample variance with the given delta degrees of freedom.
pub fn variance(xs: &[f64], ddof: usize) -> f64 {
    let n = xs.len();
    if n <= ddof {
        return 0.0;
    }
    let m = mean(xs);
    let mut acc = 0.0f64;
    for &x in xs {
        let d = x - m;
        acc += d * d;
    }
    acc / (n - ddof) as f64
}

/// Percentile with linear interpolation, matching `np.percentile`'s default
/// `method="linear"`.
///
/// `p` is in `[0, 100]`.  Returns 0.0 for an empty slice.
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, p)
}

/// Percentile of an already-sorted slice.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let virtual_index = (p / 100.0) * (n - 1) as f64;
    let lo = virtual_index.floor();
    let hi = virtual_index.ceil();
    if lo == hi {
        return sorted[lo as usize];
    }
    let frac = virtual_index - lo;
    let a = sorted[lo as usize];
    let b = sorted[hi as usize];
    a + (b - a) * frac
}

/// Histogram bin centres and densities, matching
/// `np.histogram(samples, bins=bins, density=True)` followed by the bin-centre
/// computation in `MonteCarloResults.time_to_loss_pdf`.
///
/// Returns `(bin_centers, densities)`; both empty when `samples` is empty.
pub fn histogram_density(samples: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    if samples.is_empty() || bins == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in samples {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    // NumPy widens a degenerate range to [lo - 0.5, hi + 0.5].
    if lo == hi {
        lo -= 0.5;
        hi += 0.5;
    }

    let width = (hi - lo) / bins as f64;
    let mut counts = vec![0.0f64; bins];
    for &x in samples {
        // NumPy's right-closed final bin: the maximum lands in the last bin.
        let mut idx = ((x - lo) / width).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        let mut idx = idx as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1.0;
    }

    let total = samples.len() as f64;
    let densities: Vec<f64> = counts.iter().map(|c| c / (total * width)).collect();
    let centers: Vec<f64> = (0..bins)
        .map(|i| {
            let left = lo + width * i as f64;
            let right = lo + width * (i + 1) as f64;
            (left + right) / 2.0
        })
        .collect();

    (centers, densities)
}

/// One-sample Kolmogorov-Smirnov test against a reference CDF.
///
/// Returns `(statistic, p_value)`, matching `scipy.stats.kstest` closely
/// enough for the significance levels the tests use.  The p-value comes from
/// the asymptotic Kolmogorov distribution, which is accurate for the sample
/// sizes here (hundreds and up); scipy switches to an exact computation only
/// for small samples.
///
/// Returns `(0.0, 1.0)` for an empty sample.
pub fn ks_test<F: Fn(f64) -> f64>(samples: &[f64], cdf: F) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 1.0);
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_f = n as f64;
    let mut d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let theoretical = cdf(x);
        // The empirical CDF steps at each sample, so both sides of the step
        // have to be compared.
        let below = theoretical - (i as f64) / n_f;
        let above = ((i + 1) as f64) / n_f - theoretical;
        d = d.max(below).max(above);
    }

    (d, kolmogorov_sf(d * n_f.sqrt()))
}

/// Two-sample Kolmogorov-Smirnov test.
///
/// Returns `(statistic, p_value)`, matching `scipy.stats.ks_2samp` with
/// `method="asymp"`.  Used to compare two engines' output distributions.
pub fn ks_2samp(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.is_empty() || b.is_empty() {
        return (0.0, 1.0);
    }

    let mut xs = a.to_vec();
    let mut ys = b.to_vec();
    xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));

    let (n, m) = (xs.len() as f64, ys.len() as f64);
    let (mut i, mut j) = (0usize, 0usize);
    let mut d = 0.0f64;

    // Walk both sorted samples together, tracking the gap between the two
    // empirical CDFs.
    while i < xs.len() && j < ys.len() {
        let value = xs[i].min(ys[j]);
        while i < xs.len() && xs[i] <= value {
            i += 1;
        }
        while j < ys.len() && ys[j] <= value {
            j += 1;
        }
        d = d.max(((i as f64) / n - (j as f64) / m).abs());
    }

    let effective_n = (n * m / (n + m)).sqrt();
    (d, kolmogorov_sf(d * effective_n))
}

/// Survival function of the Kolmogorov distribution,
/// `Q(x) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 x^2)`.
///
/// This is the asymptotic p-value for a KS statistic scaled by `sqrt(n)`.
fn kolmogorov_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // The series converges fast; beyond this the result underflows anyway.
    if x > 7.0 {
        return 0.0;
    }

    let mut total = 0.0f64;
    for k in 1..=100 {
        let k_f = k as f64;
        let term = (-2.0 * k_f * k_f * x * x).exp();
        if k % 2 == 1 {
            total += term;
        } else {
            total -= term;
        }
        if term < 1e-18 {
            break;
        }
    }

    (2.0 * total).clamp(0.0, 1.0)
}

/// CDF of the exponential distribution with the given scale (`1 / rate`).
pub fn exponential_cdf(x: f64, scale: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - (-x / scale).exp()
    }
}

/// CDF of the normal distribution.
pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    use statrs::distribution::ContinuousCDF;
    statrs::distribution::Normal::new(mean, std_dev)
        .expect("std_dev must be positive")
        .cdf(x)
}

/// Welch's t-test for the equality of two means, assuming unequal
/// variances.  Returns `(statistic, two_sided_p_value)`.
///
/// Matches `scipy.stats.ttest_ind(a, b, equal_var=False)`.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.len() < 2 || b.len() < 2 {
        return (0.0, 1.0);
    }

    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let (m1, m2) = (mean(a), mean(b));
    let (v1, v2) = (variance(a, 1), variance(b, 1));

    let se_squared = v1 / n1 + v2 / n2;
    if se_squared <= 0.0 {
        // No variance in either sample: identical means agree perfectly,
        // different means disagree completely.
        return if m1 == m2 { (0.0, 1.0) } else { (f64::INFINITY, 0.0) };
    }

    let t = (m1 - m2) / se_squared.sqrt();

    // Welch-Satterthwaite degrees of freedom.
    let df = se_squared * se_squared
        / ((v1 / n1).powi(2) / (n1 - 1.0) + (v2 / n2).powi(2) / (n2 - 1.0));

    use statrs::distribution::ContinuousCDF;
    let dist = statrs::distribution::StudentsT::new(0.0, 1.0, df.max(1.0))
        .expect("degrees of freedom are at least 1");
    let p = 2.0 * (1.0 - dist.cdf(t.abs()));

    (t, p.clamp(0.0, 1.0))
}

/// Empirical CDF, matching `MonteCarloResults.time_to_loss_cdf`: sorted
/// samples paired with `(1..=n) / n`.
pub fn ecdf(samples: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if samples.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let probs = (1..=n).map(|i| i as f64 / n as f64).collect();
    (sorted, probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} != {b}");
    }

    #[test]
    fn inverse_cdfs_match_scipy() {
        approx(norm_ppf(0.975), 1.959_963_984_540_054);
        assert!((t_ppf(0.975, 29.0) - 2.045_229_642_132_703).abs() < 1e-9);
    }

    #[test]
    fn t_ci_half_width_needs_two_samples() {
        assert_eq!(t_ci_half_width(&[], 0.95), 0.0);
        assert_eq!(t_ci_half_width(&[1.0], 0.95), 0.0);
        // Known value: t(0.975, df=3) * std([1,2,3,4], ddof=1) / 2
        let hw = t_ci_half_width(&[1.0, 2.0, 3.0, 4.0], 0.95);
        assert!((hw - 2.054_260_256_760_519).abs() < 1e-9, "got {hw}");
    }

    #[test]
    fn mean_and_std_match_numpy_semantics() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        approx(mean(&xs), 2.5);
        // np.std([1,2,3,4], ddof=1) == 1.2909944487358056
        approx(std(&xs, 1), 1.290_994_448_735_805_6);
        // np.std([1,2,3,4], ddof=0) == 1.118033988749895
        approx(std(&xs, 0), 1.118_033_988_749_895);
    }

    #[test]
    fn mean_and_std_degenerate_cases() {
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(std(&[], 1), 0.0);
        assert_eq!(std(&[5.0], 1), 0.0);
    }

    #[test]
    fn percentile_interpolates_linearly() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        approx(percentile(&xs, 0.0), 1.0);
        approx(percentile(&xs, 100.0), 4.0);
        approx(percentile(&xs, 50.0), 2.5);
        // np.percentile([1,2,3,4], 25) == 1.75
        approx(percentile(&xs, 25.0), 1.75);
        // np.percentile([1,2,3,4], 90) == 3.7
        approx(percentile(&xs, 90.0), 3.7);
    }

    #[test]
    fn percentile_handles_unsorted_input() {
        let xs = [4.0, 1.0, 3.0, 2.0];
        approx(percentile(&xs, 50.0), 2.5);
    }

    #[test]
    fn histogram_density_integrates_to_one() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (centers, densities) = histogram_density(&xs, 10);
        assert_eq!(centers.len(), 10);
        let width = centers[1] - centers[0];
        let area: f64 = densities.iter().map(|d| d * width).sum();
        approx(area, 1.0);
    }

    #[test]
    fn histogram_handles_degenerate_range() {
        let (centers, densities) = histogram_density(&[3.0, 3.0, 3.0], 4);
        assert_eq!(centers.len(), 4);
        let width = centers[1] - centers[0];
        let area: f64 = densities.iter().map(|d| d * width).sum();
        approx(area, 1.0);
    }

    #[test]
    fn ecdf_is_sorted_and_reaches_one() {
        let (times, probs) = ecdf(&[3.0, 1.0, 2.0]);
        assert_eq!(times, vec![1.0, 2.0, 3.0]);
        approx(probs[2], 1.0);
        assert!(probs.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn ks_test_accepts_samples_from_the_reference_distribution() {
        use crate::sim::distributions::{make_rng, Distribution};
        let mut rng = make_rng(Some(11));
        let scale = 7.0;
        let d = Distribution::exponential(1.0 / scale).unwrap();
        let samples: Vec<f64> = (0..5000).map(|_| d.sample(&mut rng)).collect();

        let (stat, p) = ks_test(&samples, |x| exponential_cdf(x, scale));
        assert!(p > 0.01, "KS rejected a true match: stat={stat}, p={p}");
    }

    #[test]
    fn ks_test_rejects_a_wrong_distribution() {
        use crate::sim::distributions::{make_rng, Distribution};
        let mut rng = make_rng(Some(12));
        let d = Distribution::exponential(1.0 / 7.0).unwrap();
        let samples: Vec<f64> = (0..5000).map(|_| d.sample(&mut rng)).collect();

        // Compare against a scale that is off by a factor of two.
        let (_, p) = ks_test(&samples, |x| exponential_cdf(x, 14.0));
        assert!(p < 1e-6, "KS failed to reject a clear mismatch: p={p}");
    }

    #[test]
    fn ks_2samp_accepts_two_draws_from_the_same_distribution() {
        use crate::sim::distributions::{make_rng, Distribution};
        let d = Distribution::normal(10.0, 2.0, f64::NEG_INFINITY).unwrap();
        let mut rng_a = make_rng(Some(21));
        let mut rng_b = make_rng(Some(22));
        let a: Vec<f64> = (0..2000).map(|_| d.sample(&mut rng_a)).collect();
        let b: Vec<f64> = (0..2000).map(|_| d.sample(&mut rng_b)).collect();

        let (_, p) = ks_2samp(&a, &b);
        assert!(p > 0.01, "two-sample KS rejected a true match: p={p}");
    }

    #[test]
    fn ks_2samp_rejects_shifted_distributions() {
        use crate::sim::distributions::{make_rng, Distribution};
        let mut rng_a = make_rng(Some(31));
        let mut rng_b = make_rng(Some(32));
        let a_dist = Distribution::normal(10.0, 2.0, f64::NEG_INFINITY).unwrap();
        let b_dist = Distribution::normal(13.0, 2.0, f64::NEG_INFINITY).unwrap();
        let a: Vec<f64> = (0..2000).map(|_| a_dist.sample(&mut rng_a)).collect();
        let b: Vec<f64> = (0..2000).map(|_| b_dist.sample(&mut rng_b)).collect();

        let (_, p) = ks_2samp(&a, &b);
        assert!(p < 1e-6, "two-sample KS failed to reject a 1.5-sigma shift: p={p}");
    }

    #[test]
    fn welch_t_test_matches_scipy_on_a_known_case() {
        // scipy.stats.ttest_ind([1,2,3,4,5], [2,4,6,8,10], equal_var=False)
        // gives t = -1.8973665961010275, p = 0.10753119493062724
        let (t, p) = welch_t_test(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0]);
        assert!((t - -1.897_366_596_101_027_5).abs() < 1e-12, "t = {t}");
        assert!((p - 0.107_531_194_930_627_24).abs() < 1e-9, "p = {p}");
    }

    #[test]
    fn welch_t_test_handles_degenerate_samples() {
        assert_eq!(welch_t_test(&[1.0], &[2.0]), (0.0, 1.0));
        let (_, p) = welch_t_test(&[5.0; 10], &[5.0; 10]);
        assert_eq!(p, 1.0);
        let (_, p) = welch_t_test(&[5.0; 10], &[9.0; 10]);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn empty_inputs_return_empty_vectors() {
        assert_eq!(histogram_density(&[], 10).0.len(), 0);
        assert_eq!(ecdf(&[]).0.len(), 0);
    }
}
