//! Cluster state: the collection of nodes plus network state.
//!
//! Port of `powder/simulation/cluster.py`.
//!
//! Python keeps three dicts (`nodes`, `standby_nodes`, `provisioning_nodes`)
//! and caches their concatenation.  The port keeps one `Vec<NodeState>` where
//! each entry carries a [`Group`] tag: membership queries become filtered
//! iteration over contiguous memory, promotion is a single field write
//! instead of a remove-and-reinsert, and the cache disappears along with its
//! invalidation bookkeeping.
//!
//! Node counts are small (tens), so lookups are linear scans over that
//! vector.  That beats hashing a string key, which is what the Python version
//! pays on every `get_node`.

use std::rc::Rc;

use super::distributions::Seconds;
use super::ids::{Interner, Sym};
use super::network::NetworkState;
use super::node::{Group, NodeConfigRef, NodeState};

/// Whether a node is up, has data, and is not in a region outage.
///
/// Free function so callers holding a mutable borrow of the node vector can
/// still evaluate availability; see
/// [`ClusterState::nodes_and_network_mut`].
#[inline]
pub fn is_effectively_available(network: &NetworkState, node: &NodeState) -> bool {
    if !node.is_available || !node.has_data {
        return false;
    }
    // Fast path: no outages anywhere means no region lookup.
    if !network.has_active_outages() {
        return true;
    }
    !network.is_region_down(node.region)
}

/// Active, effectively-available and up-to-date node counts.
///
/// Produced in a single pass by
/// [`ClusterState::availability_counts`](ClusterState::availability_counts).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AvailabilityCounts {
    /// Nodes in the active set, whatever their health.
    pub active: usize,
    /// Active nodes that are up, hold data, and are reachable.
    pub available: usize,
    /// Available nodes that have applied everything up to `commit_index`.
    pub up_to_date: usize,
}

impl AvailabilityCounts {
    /// Majority quorum over the active set.
    #[inline]
    pub fn quorum_size(&self) -> usize {
        self.active / 2 + 1
    }
}

/// Complete state of an RSM cluster during simulation.
#[derive(Debug)]
pub struct ClusterState {
    /// Every node, in creation order, tagged with its membership group.
    nodes: Vec<NodeState>,
    /// Current network partition state.
    pub network: NetworkState,
    /// Desired number of active nodes.
    pub target_cluster_size: usize,
    /// Current wall-clock simulation time, in seconds.
    pub current_time: Seconds,
    /// Position in the committed data stream.  Advances only while the
    /// system can commit, at the protocol's commit rate.  Not a wall-clock
    /// time.
    pub commit_index: f64,
    /// Identifier table shared by nodes, regions and event targets.
    ///
    /// Shared rather than copied on clone.  A Monte Carlo experiment clones
    /// one template cluster per simulation, and every run names its
    /// replacements the same way, so a private table would re-intern the
    /// same strings thousands of times.  Symbols are append-only and
    /// stable, so sharing the table only means later runs find their names
    /// already present.
    ids: Rc<Interner>,
}

impl Clone for ClusterState {
    fn clone(&self) -> Self {
        ClusterState {
            nodes: self.nodes.clone(),
            network: self.network.clone(),
            target_cluster_size: self.target_cluster_size,
            current_time: self.current_time,
            commit_index: self.commit_index,
            ids: Rc::clone(&self.ids),
        }
    }
}

impl ClusterState {
    /// An empty cluster with the given target size.
    pub fn new(target_cluster_size: usize) -> Self {
        ClusterState {
            nodes: Vec::new(),
            network: NetworkState::new(),
            target_cluster_size,
            current_time: 0.0,
            commit_index: 0.0,
            ids: Rc::new(Interner::new()),
        }
    }

    /// Reset to match `template`, reusing this cluster's allocations.
    ///
    /// A Monte Carlo experiment starts every run from the same initial
    /// state.  Cloning the template afresh each time allocates a node
    /// vector per run; this reuses the one already here.
    pub fn reset_from(&mut self, template: &ClusterState) {
        // `Vec::clone_from` reuses the existing buffer when it is large
        // enough, which is the whole point of this method.
        self.nodes.clone_from(&template.nodes);
        self.network.reset_from(&template.network);
        self.target_cluster_size = template.target_cluster_size;
        self.current_time = template.current_time;
        self.commit_index = template.commit_index;
        self.ids = Rc::clone(&template.ids);
    }

    // -- identifiers -----------------------------------------------------

    /// Intern an identifier, assigning a fresh symbol if it is new.
    ///
    /// Takes `&self` so protocols and strategies, which only hold a shared
    /// reference, can mint identifiers for nodes they are about to request.
    pub fn intern(&self, name: &str) -> Sym {
        self.ids.intern(name)
    }

    /// Look up an already-interned identifier.
    pub fn sym_of(&self, name: &str) -> Option<Sym> {
        self.ids.get(name)
    }

    /// Resolve a symbol back to its string form, for reporting.
    pub fn name_of(&self, sym: Sym) -> String {
        self.ids.name(sym)
    }

    /// Number of symbols assigned so far.
    pub fn sym_count(&self) -> usize {
        self.ids.len()
    }

    /// Order two identifiers by their string form, without allocating.
    pub fn cmp_names(&self, a: Sym, b: Sym) -> std::cmp::Ordering {
        self.ids.cmp_names(a, b)
    }

    // -- membership ------------------------------------------------------

    /// All nodes in every group, in creation order.
    pub fn all_entries(&self) -> &[NodeState] {
        &self.nodes
    }

    /// Mutable access to every node in every group.
    pub fn all_entries_mut(&mut self) -> &mut [NodeState] {
        &mut self.nodes
    }

    /// Active nodes: those that count toward quorum.  Equivalent to Python's
    /// `cluster.nodes.values()`.
    pub fn active(&self) -> impl Iterator<Item = &NodeState> {
        self.nodes.iter().filter(|n| n.group == Group::Active)
    }

    /// Active and standby nodes.  Equivalent to Python's `_all_nodes`.
    pub fn active_and_standby(&self) -> impl Iterator<Item = &NodeState> {
        self.nodes.iter().filter(|n| n.group != Group::Provisioning)
    }

    /// Standby nodes only.
    pub fn standby(&self) -> impl Iterator<Item = &NodeState> {
        self.nodes.iter().filter(|n| n.group == Group::Standby)
    }

    /// Provisioning nodes only.
    pub fn provisioning(&self) -> impl Iterator<Item = &NodeState> {
        self.nodes.iter().filter(|n| n.group == Group::Provisioning)
    }

    /// Number of active nodes.  Equivalent to Python's `len(cluster.nodes)`.
    pub fn num_active(&self) -> usize {
        self.active().count()
    }

    /// Indices into [`all_entries`](Self::all_entries) for the active and
    /// standby nodes, in creation order.
    ///
    /// Useful when a caller needs to mutate nodes while iterating, which the
    /// borrow checker forbids through [`active_and_standby`](Self::active_and_standby).
    pub fn active_and_standby_indices(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.fill_active_and_standby_indices(&mut out);
        out
    }

    /// Write the active and standby indices into `out`, clearing it first.
    ///
    /// The simulator calls this on every event, so it hands in a reused
    /// buffer rather than allocating a fresh vector each time.
    pub fn fill_active_and_standby_indices(&self, out: &mut Vec<usize>) {
        out.clear();
        out.extend(
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.group != Group::Provisioning)
                .map(|(i, _)| i),
        );
    }

    /// Indices of the active nodes, in creation order.
    pub fn active_indices(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.fill_active_indices(&mut out);
        out
    }

    /// Write the active indices into `out`, clearing it first.
    pub fn fill_active_indices(&self, out: &mut Vec<usize>) {
        out.clear();
        out.extend(
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.group == Group::Active)
                .map(|(i, _)| i),
        );
    }

    /// Index of an active or standby node.  Mirrors Python's `get_node`,
    /// which deliberately does not search the provisioning set.
    pub fn index_of(&self, node_id: Sym) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.node_id == node_id && n.group != Group::Provisioning)
    }

    /// Index of a node in any group, including provisioning.
    pub fn index_of_any(&self, node_id: Sym) -> Option<usize> {
        self.nodes.iter().position(|n| n.node_id == node_id)
    }

    /// An active or standby node by identifier.
    pub fn get_node(&self, node_id: Sym) -> Option<&NodeState> {
        self.index_of(node_id).map(|i| &self.nodes[i])
    }

    /// Mutable access to an active or standby node by identifier.
    pub fn get_node_mut(&mut self, node_id: Sym) -> Option<&mut NodeState> {
        match self.index_of(node_id) {
            Some(i) => Some(&mut self.nodes[i]),
            None => None,
        }
    }

    /// Node at a raw index into [`all_entries`](Self::all_entries).
    #[inline]
    pub fn node_at(&self, index: usize) -> &NodeState {
        &self.nodes[index]
    }

    /// Mutable node at a raw index.
    #[inline]
    pub fn node_at_mut(&mut self, index: usize) -> &mut NodeState {
        &mut self.nodes[index]
    }

    /// Add an active node.
    pub fn add_node(&mut self, mut node: NodeState) {
        node.group = Group::Active;
        self.nodes.push(node);
    }

    /// Add a standby node: running and billable, but outside quorum.
    pub fn add_standby_node(&mut self, mut node: NodeState) {
        node.group = Group::Standby;
        self.nodes.push(node);
    }

    /// Add a provisioning node: billed from request time, otherwise inert.
    pub fn add_provisioning_node(&mut self, mut node: NodeState) {
        node.group = Group::Provisioning;
        self.nodes.push(node);
    }

    /// Remove an active node, returning it if it was present.
    pub fn remove_node(&mut self, node_id: Sym) -> Option<NodeState> {
        self.remove_from_group(node_id, Group::Active)
    }

    /// Remove a standby node, returning it if it was present.
    pub fn remove_standby_node(&mut self, node_id: Sym) -> Option<NodeState> {
        self.remove_from_group(node_id, Group::Standby)
    }

    /// Remove a provisioning node, returning it if it was present.
    pub fn remove_provisioning_node(&mut self, node_id: Sym) -> Option<NodeState> {
        self.remove_from_group(node_id, Group::Provisioning)
    }

    fn remove_from_group(&mut self, node_id: Sym, group: Group) -> Option<NodeState> {
        let pos = self
            .nodes
            .iter()
            .position(|n| n.node_id == node_id && n.group == group)?;
        Some(self.nodes.remove(pos))
    }

    /// Move a node from standby into the active set.
    ///
    /// Python removes from one dict and inserts into the other; here the
    /// group tag flips in place, which preserves creation order.  Ordering
    /// only ever breaks ties, and every tie-break in the port is explicit.
    pub fn promote_standby(&mut self, node_id: Sym) -> bool {
        for node in &mut self.nodes {
            if node.node_id == node_id && node.group == Group::Standby {
                node.group = Group::Active;
                return true;
            }
        }
        false
    }

    // -- availability ----------------------------------------------------

    /// Whether a node is up, has data, and is not in a region outage.
    #[inline]
    pub fn node_effectively_available(&self, node: &NodeState) -> bool {
        is_effectively_available(&self.network, node)
    }

    /// Mutable node storage alongside a shared view of the network.
    ///
    /// The simulator needs to walk the nodes mutably while still asking
    /// whether each one is reachable.  Going through
    /// [`node_effectively_available`](Self::node_effectively_available) would
    /// borrow all of `self`, so this hands out the two halves separately for
    /// use with [`is_effectively_available`].
    #[inline]
    pub fn nodes_and_network_mut(&mut self) -> (&mut [NodeState], &NetworkState) {
        (&mut self.nodes, &self.network)
    }

    /// Active, effectively-available and up-to-date counts, in one pass.
    ///
    /// A quorum decision needs all three, and taking them separately walks
    /// the node vector two or three times for the same answer.  `can_commit`
    /// is the hottest call in the simulator, so it takes this instead.
    #[inline]
    pub fn availability_counts(&self) -> AvailabilityCounts {
        let commit_index = self.commit_index;
        let has_outages = self.network.has_active_outages();

        let mut counts = AvailabilityCounts::default();
        for node in &self.nodes {
            if node.group != Group::Active {
                continue;
            }
            counts.active += 1;

            if !node.is_available || !node.has_data {
                continue;
            }
            if has_outages && self.network.is_region_down(node.region) {
                continue;
            }
            counts.available += 1;

            if node.last_applied_index >= commit_index {
                counts.up_to_date += 1;
            }
        }
        counts
    }

    /// Count active nodes that are fully synced to `commit_index` and
    /// effectively available.
    pub fn num_up_to_date(&self) -> usize {
        let commit_index = self.commit_index;
        self.active()
            .filter(|n| n.is_up_to_date(commit_index) && self.node_effectively_available(n))
            .count()
    }

    /// Count active nodes that are effectively available, lagging or not.
    pub fn num_available(&self) -> usize {
        self.active()
            .filter(|n| self.node_effectively_available(n))
            .count()
    }

    /// Count active nodes that still have their data, available or not.
    pub fn num_with_data(&self) -> usize {
        self.active().filter(|n| n.has_data).count()
    }

    // -- queries ---------------------------------------------------------

    /// Regions that currently host at least one active node, in symbol order.
    pub fn all_regions(&self) -> Vec<Sym> {
        let mut regions: Vec<Sym> = self.active().map(|n| n.region).collect();
        regions.sort_unstable();
        regions.dedup();
        regions
    }

    /// Active nodes grouped by region, as `(region, node indices)` pairs in
    /// region-symbol order.
    pub fn nodes_by_region(&self) -> Vec<(Sym, Vec<usize>)> {
        let mut grouped: Vec<(Sym, Vec<usize>)> = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.group != Group::Active {
                continue;
            }
            match grouped.iter_mut().find(|(r, _)| *r == node.region) {
                Some((_, members)) => members.push(i),
                None => grouped.push((node.region, vec![i])),
            }
        }
        grouped.sort_by_key(|(r, _)| *r);
        grouped
    }

    /// The effectively-available node furthest behind in sync.
    ///
    /// Ties break on the lower node symbol.  Python relies on `min` over
    /// dict-insertion order; the explicit tie-break keeps the port
    /// deterministic without depending on container ordering.
    pub fn most_lagging_node(&self) -> Option<usize> {
        self.extreme_available_node(|candidate, best| {
            candidate.last_applied_index < best.last_applied_index
        })
    }

    /// The effectively-available node with the most recent data.
    ///
    /// Ties break on the lower node symbol.
    pub fn most_up_to_date_node(&self) -> Option<usize> {
        self.extreme_available_node(|candidate, best| {
            candidate.last_applied_index > best.last_applied_index
        })
    }

    fn extreme_available_node(
        &self,
        better: impl Fn(&NodeState, &NodeState) -> bool,
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.group == Group::Provisioning || !self.node_effectively_available(node) {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    let current = &self.nodes[b];
                    if better(node, current)
                        || (node.last_applied_index == current.last_applied_index
                            && node.node_id < current.node_id)
                    {
                        best = Some(i);
                    }
                }
            }
        }
        best
    }

    /// Best donor for a lagging node: the effectively-available peer with the
    /// highest `last_applied_index`.
    ///
    /// The donor's position, not `commit_index`, bounds what the syncing node
    /// can obtain.  Ties break on the lower node symbol.
    pub fn find_sync_donor(&self, node_id: Sym) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.group == Group::Provisioning
                || node.node_id == node_id
                || !self.node_effectively_available(node)
            {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    let current = &self.nodes[b];
                    if node.last_applied_index > current.last_applied_index
                        || (node.last_applied_index == current.last_applied_index
                            && node.node_id < current.node_id)
                    {
                        best = Some(i);
                    }
                }
            }
        }
        best
    }

    /// Nodes that are lagging, effectively available, and have no active
    /// sync -- candidates for `retry_pending_syncs`.
    pub fn nodes_needing_sync(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.fill_nodes_needing_sync(&mut out);
        out
    }

    /// Write the nodes needing a sync into `out`, clearing it first.
    pub fn fill_nodes_needing_sync(&self, out: &mut Vec<usize>) {
        out.clear();
        let commit_index = self.commit_index;
        out.extend(
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    n.group != Group::Provisioning
                        && self.node_effectively_available(n)
                        && !n.is_up_to_date(commit_index)
                        && n.sync.is_none()
                })
                .map(|(i, _)| i),
        );
    }

    /// Nodes whose active sync references `donor_id`.
    pub fn nodes_syncing_from(&self, donor_id: Sym) -> Vec<usize> {
        let mut out = Vec::new();
        self.fill_nodes_syncing_from(donor_id, &mut out);
        out
    }

    /// Write the nodes syncing from `donor_id` into `out`, clearing it first.
    pub fn fill_nodes_syncing_from(&self, donor_id: Sym, out: &mut Vec<usize>) {
        out.clear();
        out.extend(
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    n.group != Group::Provisioning
                        && n.sync.as_ref().is_some_and(|s| s.donor_id == donor_id)
                })
                .map(|(i, _)| i),
        );
    }

    /// Nodes that should be billed.
    ///
    /// Cloud VMs are charged from launch -- including provisioning and
    /// through failures -- until termination.  Nodes that have lost data are
    /// not billed.
    pub fn all_nodes_for_billing(&self) -> impl Iterator<Item = &NodeState> {
        self.nodes.iter().filter(|n| n.has_data)
    }

    // -- convenience -----------------------------------------------------

    /// Intern `name` and its region, then add a healthy active node.
    /// Returns the node's symbol.
    pub fn add_named_node(&mut self, name: &str, config: NodeConfigRef) -> Sym {
        let sym = self.intern(name);
        let region = self.intern(&config.region);
        self.add_node(NodeState::new(sym, region, config));
        sym
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::distributions::Distribution;
    use crate::sim::node::NodeConfig;
    use std::rc::Rc;

    fn config_in(region: &str) -> NodeConfigRef {
        Rc::new(NodeConfig {
            region: region.to_string(),
            cost_per_hour: 1.0,
            failure_dist: Distribution::constant(1000.0),
            recovery_dist: Distribution::constant(10.0),
            data_loss_dist: Distribution::constant(10_000.0),
            log_replay_rate_dist: Distribution::constant(100.0),
            snapshot_download_time_dist: Distribution::constant(5.0),
            spawn_dist: Distribution::constant(30.0),
        })
    }

    fn cluster_of(n: usize) -> ClusterState {
        let mut c = ClusterState::new(n);
        for i in 0..n {
            c.add_named_node(&format!("node_{i}"), config_in("us-east"));
        }
        c
    }

    #[test]
    fn counts_reflect_node_health() {
        let mut c = cluster_of(3);
        assert_eq!(c.num_active(), 3);
        assert_eq!(c.num_available(), 3);
        assert_eq!(c.num_up_to_date(), 3);
        assert_eq!(c.num_with_data(), 3);

        let n0 = c.sym_of("node_0").unwrap();
        c.get_node_mut(n0).unwrap().is_available = false;
        assert_eq!(c.num_available(), 2);
        assert_eq!(c.num_with_data(), 3);

        let n1 = c.sym_of("node_1").unwrap();
        c.get_node_mut(n1).unwrap().has_data = false;
        assert_eq!(c.num_available(), 1);
        assert_eq!(c.num_with_data(), 2);
    }

    #[test]
    fn lagging_nodes_are_not_up_to_date() {
        let mut c = cluster_of(3);
        c.commit_index = 100.0;
        let n0 = c.sym_of("node_0").unwrap();
        c.get_node_mut(n0).unwrap().last_applied_index = 100.0;
        assert_eq!(c.num_up_to_date(), 1);
        assert_eq!(c.num_available(), 3);
    }

    #[test]
    fn region_outage_makes_nodes_unavailable() {
        let mut c = ClusterState::new(3);
        c.add_named_node("a", config_in("us-east"));
        c.add_named_node("b", config_in("us-east"));
        c.add_named_node("c", config_in("eu-west"));

        let us_east = c.sym_of("us-east").unwrap();
        c.network.add_outage(us_east);

        assert_eq!(c.num_available(), 1);
        assert_eq!(c.num_with_data(), 3);
    }

    #[test]
    fn get_node_ignores_provisioning_entries() {
        let mut c = ClusterState::new(1);
        let sym = c.intern("pending");
        let region = c.intern("us-east");
        let mut node = NodeState::new(sym, region, config_in("us-east"));
        node.is_available = false;
        node.has_data = false;
        c.add_provisioning_node(node);

        assert!(c.get_node(sym).is_none());
        assert!(c.index_of_any(sym).is_some());
    }

    #[test]
    fn promotion_moves_standby_into_the_active_set() {
        let mut c = cluster_of(2);
        let sym = c.intern("spare");
        let region = c.intern("us-east");
        c.add_standby_node(NodeState::new(sym, region, config_in("us-east")));

        assert_eq!(c.num_active(), 2);
        assert_eq!(c.standby().count(), 1);

        assert!(c.promote_standby(sym));
        assert_eq!(c.num_active(), 3);
        assert_eq!(c.standby().count(), 0);
        assert!(!c.promote_standby(sym));
    }

    #[test]
    fn removal_is_group_scoped() {
        let mut c = cluster_of(2);
        let n0 = c.sym_of("node_0").unwrap();
        assert!(c.remove_standby_node(n0).is_none());
        assert!(c.remove_node(n0).is_some());
        assert_eq!(c.num_active(), 1);
        assert!(c.get_node(n0).is_none());
    }

    #[test]
    fn donor_selection_picks_the_furthest_ahead_peer() {
        let mut c = cluster_of(3);
        let (a, b, d) = (
            c.sym_of("node_0").unwrap(),
            c.sym_of("node_1").unwrap(),
            c.sym_of("node_2").unwrap(),
        );
        c.get_node_mut(a).unwrap().last_applied_index = 10.0;
        c.get_node_mut(b).unwrap().last_applied_index = 50.0;
        c.get_node_mut(d).unwrap().last_applied_index = 30.0;

        let donor = c.find_sync_donor(a).unwrap();
        assert_eq!(c.node_at(donor).node_id, b);
        // A node is never its own donor.
        let donor = c.find_sync_donor(b).unwrap();
        assert_eq!(c.node_at(donor).node_id, d);
    }

    #[test]
    fn donor_selection_skips_unavailable_peers() {
        let mut c = cluster_of(3);
        let (a, b, d) = (
            c.sym_of("node_0").unwrap(),
            c.sym_of("node_1").unwrap(),
            c.sym_of("node_2").unwrap(),
        );
        c.get_node_mut(b).unwrap().last_applied_index = 50.0;
        c.get_node_mut(b).unwrap().is_available = false;
        c.get_node_mut(d).unwrap().last_applied_index = 30.0;

        let donor = c.find_sync_donor(a).unwrap();
        assert_eq!(c.node_at(donor).node_id, d);
    }

    #[test]
    fn donor_ties_break_deterministically() {
        let c = cluster_of(3);
        let a = c.sym_of("node_0").unwrap();
        // node_1 and node_2 are both at 0.0; the lower symbol wins.
        let donor = c.find_sync_donor(a).unwrap();
        assert_eq!(c.name_of(c.node_at(donor).node_id), "node_1");
    }

    #[test]
    fn most_lagging_and_most_up_to_date() {
        let mut c = cluster_of(3);
        let (a, b, d) = (
            c.sym_of("node_0").unwrap(),
            c.sym_of("node_1").unwrap(),
            c.sym_of("node_2").unwrap(),
        );
        c.get_node_mut(a).unwrap().last_applied_index = 10.0;
        c.get_node_mut(b).unwrap().last_applied_index = 50.0;
        c.get_node_mut(d).unwrap().last_applied_index = 30.0;

        assert_eq!(c.node_at(c.most_lagging_node().unwrap()).node_id, a);
        assert_eq!(c.node_at(c.most_up_to_date_node().unwrap()).node_id, b);
    }

    #[test]
    fn extremes_are_none_when_nothing_is_available() {
        let mut c = cluster_of(2);
        for i in c.active_indices() {
            c.node_at_mut(i).is_available = false;
        }
        assert!(c.most_lagging_node().is_none());
        assert!(c.most_up_to_date_node().is_none());
        assert!(c.find_sync_donor(c.sym_of("node_0").unwrap()).is_none());
    }

    #[test]
    fn nodes_needing_sync_excludes_current_and_syncing_nodes() {
        let mut c = cluster_of(3);
        c.commit_index = 100.0;
        let indices = c.active_indices();
        // node_0 is current, node_1 lags with a sync in flight, node_2 lags freely.
        c.node_at_mut(indices[0]).last_applied_index = 100.0;
        c.node_at_mut(indices[1]).sync = Some(crate::sim::node::SyncState::log_replay(0, 1.0));

        let needing = c.nodes_needing_sync();
        assert_eq!(needing.len(), 1);
        assert_eq!(c.name_of(c.node_at(needing[0]).node_id), "node_2");
    }

    #[test]
    fn nodes_syncing_from_finds_dependents() {
        let mut c = cluster_of(3);
        let donor = c.sym_of("node_0").unwrap();
        let indices = c.active_indices();
        c.node_at_mut(indices[1]).sync = Some(crate::sim::node::SyncState::log_replay(donor, 1.0));
        c.node_at_mut(indices[2]).sync = Some(crate::sim::node::SyncState::log_replay(donor, 1.0));

        assert_eq!(c.nodes_syncing_from(donor).len(), 2);
        assert_eq!(c.nodes_syncing_from(999).len(), 0);
    }

    #[test]
    fn billing_includes_provisioning_and_standby_but_not_lost_data() {
        let mut c = cluster_of(2);
        let spare = c.intern("spare");
        let pending = c.intern("pending");
        let region = c.intern("us-east");
        c.add_standby_node(NodeState::new(spare, region, config_in("us-east")));

        let mut prov = NodeState::new(pending, region, config_in("us-east"));
        prov.is_available = false;
        prov.has_data = true;
        c.add_provisioning_node(prov);

        assert_eq!(c.all_nodes_for_billing().count(), 4);

        let n0 = c.sym_of("node_0").unwrap();
        c.get_node_mut(n0).unwrap().has_data = false;
        assert_eq!(c.all_nodes_for_billing().count(), 3);
    }

    #[test]
    fn regions_and_grouping() {
        let mut c = ClusterState::new(3);
        c.add_named_node("a", config_in("us-east"));
        c.add_named_node("b", config_in("eu-west"));
        c.add_named_node("c", config_in("us-east"));

        assert_eq!(c.all_regions().len(), 2);
        let grouped = c.nodes_by_region();
        assert_eq!(grouped.len(), 2);
        let total: usize = grouped.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(total, 3);
    }
}
