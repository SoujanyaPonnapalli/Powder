//! Small shared helpers for the simulation engine.

/// The largest multiple of `interval` at or below `index`.
///
/// Python writes this as `int(index // interval) * interval`.  CPython's
/// float floor-division is an `fmod`-based routine that disagrees with
/// `floor(x / y)` in edge cases (`100.0 // 0.0001` is `999999.0`, while
/// `floor(100.0 / 0.0001)` is `1000000.0`).  The port uses the plain
/// division, which is faster and is the mathematically intended answer; see
/// `rust/README.md` for the list of deliberate deviations.
///
/// Returns 0.0 when `interval` is not positive, matching the guards that
/// surround every call site in the Python source.
#[inline]
pub fn snapshot_boundary(index: f64, interval: f64) -> f64 {
    if interval <= 0.0 {
        return 0.0;
    }
    (index / interval).floor() * interval
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_down_to_the_interval() {
        assert_eq!(snapshot_boundary(250.0, 100.0), 200.0);
        assert_eq!(snapshot_boundary(200.0, 100.0), 200.0);
        assert_eq!(snapshot_boundary(99.0, 100.0), 0.0);
    }

    #[test]
    fn non_positive_interval_yields_zero() {
        assert_eq!(snapshot_boundary(250.0, 0.0), 0.0);
        assert_eq!(snapshot_boundary(250.0, -5.0), 0.0);
    }
}
