//! Region-level network outage model.
//!
//! Port of `powder/simulation/network.py`.  While a region's network is down
//! every node in it counts as unavailable.
//!
//! Python stores active outages in a `set[str]`.  Regions are interned, so
//! the port uses a bitmap indexed by region symbol plus a running count, which
//! makes the "no outages at all" fast path in
//! [`ClusterState`](super::cluster::ClusterState) a single integer test.

use super::distributions::Distribution;
use super::ids::Sym;

/// Name of the pseudo-region that targets global (all-regions) outage events.
pub const GLOBAL_REGION: &str = "__global__";

/// Configuration for network outage behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkConfig {
    /// Time (seconds) until the next outage in a region (default).
    pub outage_dist: Distribution,
    /// Duration (seconds) of each outage (default).
    pub outage_duration_dist: Distribution,
    /// Interned regions that can experience full outages.
    pub regions: Vec<Sym>,
    /// Per-region overrides: `(region, time-to-outage, optional duration)`.
    pub per_region: Vec<(Sym, Distribution, Option<Distribution>)>,
    /// Global process: `(pseudo-region symbol, time-to-outage, optional duration)`;
    /// each event takes every region in `regions` down at once.
    pub global: Option<(Sym, Distribution, Option<Distribution>)>,
}

impl NetworkConfig {
    /// Whether `sym` is the global pseudo-region.
    pub fn is_global(&self, sym: Sym) -> bool {
        matches!(&self.global, Some((g, _, _)) if *g == sym)
    }

    /// Time-to-outage distribution for a region (or the global process).
    pub fn outage_for(&self, sym: Sym) -> Option<&Distribution> {
        if let Some((g, d, _)) = &self.global {
            if *g == sym {
                return Some(d);
            }
        }
        if let Some((_, d, _)) = self.per_region.iter().find(|(r, _, _)| *r == sym) {
            return Some(d);
        }
        if self.regions.contains(&sym) {
            Some(&self.outage_dist)
        } else {
            None
        }
    }

    /// Duration distribution for a region (or the global process).
    pub fn duration_for(&self, sym: Sym) -> &Distribution {
        if let Some((g, _, d)) = &self.global {
            if *g == sym {
                return d.as_ref().unwrap_or(&self.outage_duration_dist);
            }
        }
        match self.per_region.iter().find(|(r, _, _)| *r == sym) {
            Some((_, _, Some(d))) => d,
            _ => &self.outage_duration_dist,
        }
    }
}

/// Dynamic network state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NetworkState {
    /// `down[r]` counts the active (possibly overlapping) outages of region `r`;
    /// the region is down while it is positive.
    down: Vec<u32>,
    /// Number of regions currently down; lets callers skip the lookup
    /// entirely in the common no-outage case.
    active_count: usize,
}

impl NetworkState {
    /// A network with no active outages.
    pub fn new() -> Self {
        NetworkState::default()
    }

    /// Whether any region is currently experiencing an outage.
    #[inline]
    pub fn has_active_outages(&self) -> bool {
        self.active_count != 0
    }

    /// Number of regions currently down.
    #[inline]
    pub fn active_outage_count(&self) -> usize {
        self.active_count
    }

    /// Whether a region's network is currently down.
    #[inline]
    pub fn is_region_down(&self, region: Sym) -> bool {
        match self.down.get(region as usize) {
            Some(&d) => d > 0,
            None => false,
        }
    }

    /// Whether two regions cannot communicate, i.e. either end is down.
    pub fn is_partitioned(&self, region_a: Sym, region_b: Sym) -> bool {
        self.is_region_down(region_a) || self.is_region_down(region_b)
    }

    /// Record a new network outage for a region.  Outages may overlap (e.g. a
    /// global outage during a regional one); each must be removed once.
    pub fn add_outage(&mut self, region: Sym) {
        let idx = region as usize;
        if self.down.len() <= idx {
            self.down.resize(idx + 1, 0);
        }
        if self.down[idx] == 0 {
            self.active_count += 1;
        }
        self.down[idx] += 1;
    }

    /// Record the end of one network outage for a region.
    pub fn remove_outage(&mut self, region: Sym) {
        let idx = region as usize;
        if idx < self.down.len() && self.down[idx] > 0 {
            self.down[idx] -= 1;
            if self.down[idx] == 0 {
                self.active_count -= 1;
            }
        }
    }

    /// Reset to match `src`, reusing this state's allocation.
    ///
    /// Used between the runs of one experiment: a fresh `NetworkState` per
    /// run would allocate, and there can be millions of runs.
    pub fn reset_from(&mut self, src: &NetworkState) {
        self.down.clear();
        self.down.extend_from_slice(&src.down);
        self.active_count = src.active_count;
    }

    /// Regions currently down, in symbol order.
    pub fn active_outages(&self) -> Vec<Sym> {
        self.down
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > 0)
            .map(|(i, _)| i as Sym)
            .collect()
    }

    /// Regions reachable from `region`.
    ///
    /// Nothing is reachable from a downed region; otherwise every region that
    /// is itself up, including the starting one.
    pub fn regions_reachable_from(&self, region: Sym, all_regions: &[Sym]) -> Vec<Sym> {
        if self.is_region_down(region) {
            return Vec::new();
        }
        all_regions
            .iter()
            .copied()
            .filter(|&r| !self.is_region_down(r))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_no_outages() {
        let net = NetworkState::new();
        assert!(!net.has_active_outages());
        assert!(!net.is_region_down(0));
        assert!(!net.is_region_down(999));
        assert!(net.active_outages().is_empty());
    }

    #[test]
    fn add_and_remove_outages() {
        let mut net = NetworkState::new();
        net.add_outage(3);
        assert!(net.is_region_down(3));
        assert!(net.has_active_outages());
        assert_eq!(net.active_outage_count(), 1);

        net.remove_outage(3);
        assert!(!net.is_region_down(3));
        assert!(!net.has_active_outages());
    }

    #[test]
    fn add_and_remove_are_idempotent() {
        let mut net = NetworkState::new();
        net.add_outage(1);
        net.add_outage(1);
        assert_eq!(net.active_outage_count(), 1);

        net.remove_outage(1);
        net.remove_outage(1);
        assert_eq!(net.active_outage_count(), 0);
    }

    #[test]
    fn tracks_multiple_simultaneous_outages() {
        let mut net = NetworkState::new();
        net.add_outage(0);
        net.add_outage(5);
        assert_eq!(net.active_outage_count(), 2);
        assert_eq!(net.active_outages(), vec![0, 5]);
    }

    #[test]
    fn partitioned_when_either_side_is_down() {
        let mut net = NetworkState::new();
        assert!(!net.is_partitioned(0, 1));
        net.add_outage(1);
        assert!(net.is_partitioned(0, 1));
        assert!(net.is_partitioned(1, 0));
    }

    #[test]
    fn reachability_excludes_downed_regions() {
        let mut net = NetworkState::new();
        let all = [0, 1, 2];
        assert_eq!(net.regions_reachable_from(0, &all), vec![0, 1, 2]);

        net.add_outage(1);
        assert_eq!(net.regions_reachable_from(0, &all), vec![0, 2]);
        // Nothing is reachable from inside a downed region.
        assert!(net.regions_reachable_from(1, &all).is_empty());
    }
}
