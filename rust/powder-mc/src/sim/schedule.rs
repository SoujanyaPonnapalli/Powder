//! Pre-drawn failure schedules for stratified sampling.
//!
//! Normally every node owns two clocks -- time to the next transient failure
//! and time to permanent data loss -- and the simulator samples them as it
//! goes.  Stratified sampling instead fixes, up front, how many failures of
//! each kind land in the horizon, which needs the failure times drawn before
//! the run starts.
//!
//! The construction is a set of *slots*.  Nodes whose clocks share a kind
//! and an exponential rate form one *axis*, which has a fixed number of
//! slots.  Each slot carries an independent Poisson stream of candidate
//! failures at the per-node rate, and hits whichever node owns the slot at
//! that moment, if the node can fail right then.  A node takes a free slot
//! when its clocks would normally start, and keeps it until it is removed or
//! loses its data.  Because the clocks are memoryless, an owned slot is the
//! same process the native clock produces:
//!
//! * a transient clock restarts after recovery, which matches discarding the
//!   candidates that land while the node is down;
//! * a replacement node starts fresh clocks on spawn, which matches it
//!   picking up a slot whose past is independent of its future.
//!
//! A node that finds every slot taken keeps its native clocks instead.
//! That is equally exact -- the native clocks are independent of every slot
//! stream -- it just leaves that node's failures outside the stratification.
//! So the slot count is a tuning knob, never a correctness bound.

use super::cluster::ClusterState;
use super::distributions::{Distribution, Rng, Seconds};
use super::ids::Sym;
use super::node::NodeState;
use rand::RngExt;

/// Which clock an axis replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureKind {
    /// `failure_dist`: the node goes down and later recovers.
    Transient,
    /// `data_loss_dist`: the node loses its data for good.
    Permanent,
}

impl FailureKind {
    /// Lowercase name, for JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            FailureKind::Transient => "transient",
            FailureKind::Permanent => "permanent",
        }
    }

    /// The distribution this kind of failure draws from.
    pub fn dist(self, node: &NodeState) -> &Distribution {
        match self {
            FailureKind::Transient => &node.config.failure_dist,
            FailureKind::Permanent => &node.config.data_loss_dist,
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One family of candidate streams.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleAxis {
    /// Which clock it replaces.
    pub kind: FailureKind,
    /// Per-node exponential rate of the nodes it covers.
    pub rate: f64,
    /// Number of slots, each an independent stream at `rate`.
    pub slots: usize,
    /// When set, the axis belongs to this one node alone, and that node
    /// takes no slot on any shared axis.  Lets a stratification single out
    /// a node whose failures matter more than the rest -- a Raft leader.
    pub reserved: Option<Sym>,
}

impl ScheduleAxis {
    /// Whether `node` draws this axis's kind of failure at this rate.
    pub fn covers(&self, node: &NodeState) -> bool {
        matches!(*self.kind.dist(node), Distribution::Exponential { rate } if rate == self.rate)
    }
}

/// A candidate failure: when, and on which slot of which axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// When it fires.
    pub time: Seconds,
    /// Index into [`ForcedSchedule::axes`].
    pub axis: usize,
    /// Slot within the axis.
    pub slot: usize,
}

/// Every candidate failure for one run, in time order, plus who owns each
/// slot.
#[derive(Debug, Clone, Default)]
pub struct ForcedSchedule {
    /// The candidate streams.
    pub axes: Vec<ScheduleAxis>,
    /// Candidates sorted by time.
    candidates: Vec<Candidate>,
    /// Next candidate to fire.
    cursor: usize,
    /// `owners[axis][slot]`: the node holding the slot, if any.
    owners: Vec<Vec<Option<Sym>>>,
    /// `native[sym * 2 + kind]`: whether that node runs its own clock for
    /// that kind because no slot was free.
    native: Vec<bool>,
    /// Most slots held at once, per axis, since the last
    /// [`clear_stats`](Self::clear_stats).
    peak_held: Vec<usize>,
    /// Nodes that fell back to native clocks, per axis, since the last
    /// [`clear_stats`](Self::clear_stats).
    native_count: Vec<u64>,
}

/// Whether `owner` still holds its slot: it exists and has data.
fn holds(cluster: &ClusterState, owner: Sym) -> bool {
    cluster.get_node(owner).is_some_and(|n| n.has_data)
}

impl ForcedSchedule {
    /// An empty schedule over the given axes.
    pub fn new(axes: Vec<ScheduleAxis>) -> Self {
        let owners = axes.iter().map(|a| vec![None; a.slots]).collect();
        let n = axes.len();
        ForcedSchedule {
            axes,
            owners,
            peak_held: vec![0; n],
            native_count: vec![0; n],
            ..Default::default()
        }
    }

    /// Start a new run with these candidates, releasing every slot.
    /// `candidates` need not be sorted.
    pub fn reset(&mut self, candidates: impl IntoIterator<Item = Candidate>) {
        self.candidates.clear();
        self.candidates.extend(candidates);
        self.candidates.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.cursor = 0;
        for slots in &mut self.owners {
            slots.iter_mut().for_each(|s| *s = None);
        }
        self.native.iter_mut().for_each(|n| *n = false);
    }

    /// Give `node` a slot for `kind` failures.
    ///
    /// Returns false when the node must run its native clock instead: no
    /// axis covers it, or every slot is held.
    pub fn assign(&mut self, kind: FailureKind, node: &NodeState, cluster: &ClusterState) -> bool {
        let found = self
            .axes
            .iter()
            .position(|a| a.kind == kind && a.reserved == Some(node.node_id))
            .or_else(|| {
                self.axes
                    .iter()
                    .position(|a| a.kind == kind && a.reserved.is_none() && a.covers(node))
            });
        if let Some(axis) = found {
            let slots = &mut self.owners[axis];
            let free = slots
                .iter()
                .position(|s| s.is_none_or(|owner| !holds(cluster, owner)));
            if let Some(slot) = free {
                slots[slot] = Some(node.node_id);
                let held = slots
                    .iter()
                    .filter(|s| {
                        s.is_some_and(|owner| owner == node.node_id || holds(cluster, owner))
                    })
                    .count();
                self.peak_held[axis] = self.peak_held[axis].max(held);
                return true;
            }
            self.native_count[axis] += 1;
        }
        let index = node.node_id as usize * 2 + kind.index();
        if self.native.len() <= index {
            self.native.resize(index + 1, false);
        }
        self.native[index] = true;
        false
    }

    /// Whether `node_id` runs its own clock for `kind`.
    pub fn is_native(&self, kind: FailureKind, node_id: Sym) -> bool {
        self.native
            .get(node_id as usize * 2 + kind.index())
            .copied()
            .unwrap_or(false)
    }

    /// The next candidate's time, if any remain.
    pub fn peek_time(&self) -> Option<Seconds> {
        self.candidates.get(self.cursor).map(|c| c.time)
    }

    /// Consume the next candidate and decide which node, if any, it hits.
    pub fn fire(&mut self, cluster: &ClusterState) -> Option<(FailureKind, Sym)> {
        let candidate = *self.candidates.get(self.cursor)?;
        self.cursor += 1;
        let kind = self.axes[candidate.axis].kind;
        let owner = self.owners[candidate.axis][candidate.slot]?;
        let node = cluster.get_node(owner)?;
        // A node without data has no clocks; a node that is already down
        // cannot fail transiently until it recovers.
        let can_fail = node.has_data && (kind == FailureKind::Permanent || node.is_available);
        can_fail.then_some((kind, owner))
    }

    /// Most slots held at once on each axis.
    pub fn peak_held(&self) -> &[usize] {
        &self.peak_held
    }

    /// Nodes that ran native clocks on each axis because it was full.
    pub fn native_count(&self) -> &[u64] {
        &self.native_count
    }

    /// Forget the recorded statistics.
    pub fn clear_stats(&mut self) {
        self.peak_held.iter_mut().for_each(|p| *p = 0);
        self.native_count.iter_mut().for_each(|c| *c = 0);
    }

    /// Candidates in this run.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Whether the run has no candidates.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// Draw a slot uniformly, for callers building candidates.
pub fn random_slot(axis: &ScheduleAxis, rng: &mut Rng) -> usize {
    rng.random_range(0..axis.slots)
}
