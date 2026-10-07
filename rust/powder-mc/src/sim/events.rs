//! Discrete-event types and the simulator's priority queue.
//!
//! Port of `powder/simulation/events.py`, including its lazy O(1)
//! cancellation scheme: every pushed event is stamped with an incrementing
//! generation counter, cancellation records the current counter as a
//! threshold for a key, and any event whose generation falls below the
//! threshold for its key is discarded when it reaches the top of the heap.
//! Events pushed *after* a cancellation carry a higher generation and survive.
//!
//! Two differences from Python, both performance-motivated and
//! behaviour-neutral:
//!
//! * Targets are interned [`Sym`]s, so the cancellation tables are flat
//!   vectors indexed by symbol rather than string-keyed hash maps.
//! * Event metadata is a struct of optional fields rather than a `dict`.

use super::distributions::Seconds;
use super::ids::Sym;
use super::node::NodeConfigRef;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Types of events that can occur during simulation.
///
/// Discriminants match the Python `IntEnum` so logs and JSON line up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EventType {
    /// Transient unavailability.
    NodeFailure = 0,
    /// Recovery from transient failure.
    NodeRecovery = 1,
    /// Permanent data loss (e.g. disk failure).
    NodeDataLoss = 2,
    /// A node finished syncing.
    NodeSyncComplete = 3,
    /// A new node finished spawning.
    NodeSpawnComplete = 4,
    /// A region partition begins.
    NetworkOutageStart = 5,
    /// A region partition ends.
    NetworkOutageEnd = 6,
    /// A node's replacement timeout fired.
    NodeReplacementTimeout = 7,
    /// A leader election finished.
    LeaderElectionComplete = 8,
    /// An attempt to reconfigure cluster size.
    ClusterReconfiguration = 9,
}

/// Number of distinct event types, used to size the cancellation table.
pub const EVENT_TYPE_COUNT: usize = 10;

impl EventType {
    /// All event types, in discriminant order.
    pub const ALL: [EventType; EVENT_TYPE_COUNT] = [
        EventType::NodeFailure,
        EventType::NodeRecovery,
        EventType::NodeDataLoss,
        EventType::NodeSyncComplete,
        EventType::NodeSpawnComplete,
        EventType::NetworkOutageStart,
        EventType::NetworkOutageEnd,
        EventType::NodeReplacementTimeout,
        EventType::LeaderElectionComplete,
        EventType::ClusterReconfiguration,
    ];

    /// Discriminant, for indexing the cancellation table.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// The Python enum member name, for logs and JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            EventType::NodeFailure => "NODE_FAILURE",
            EventType::NodeRecovery => "NODE_RECOVERY",
            EventType::NodeDataLoss => "NODE_DATA_LOSS",
            EventType::NodeSyncComplete => "NODE_SYNC_COMPLETE",
            EventType::NodeSpawnComplete => "NODE_SPAWN_COMPLETE",
            EventType::NetworkOutageStart => "NETWORK_OUTAGE_START",
            EventType::NetworkOutageEnd => "NETWORK_OUTAGE_END",
            EventType::NodeReplacementTimeout => "NODE_REPLACEMENT_TIMEOUT",
            EventType::LeaderElectionComplete => "LEADER_ELECTION_COMPLETE",
            EventType::ClusterReconfiguration => "CLUSTER_RECONFIGURATION",
        }
    }
}

/// Event-specific payload.
///
/// Replaces Python's `metadata: dict[str, Any]`.  The payloads are mutually
/// exclusive -- an outage carries a region, an election carries an epoch,
/// and so on -- so this is an enum rather than a struct of six options.
/// That matters: events live in a binary heap, and the struct form made
/// every heap entry 40 bytes larger than it needed to be.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum EventMeta {
    /// No payload.
    #[default]
    None,
    /// `NETWORK_OUTAGE_*`: the affected region.
    Region(Sym),
    /// `LEADER_ELECTION_COMPLETE`: the election epoch, used to drop stale
    /// events from a cancelled election.
    Epoch(u64),
    /// `CLUSTER_RECONFIGURATION`: the requested new cluster size.
    TargetSize(usize),
    /// `NODE_SPAWN_COMPLETE`: what to create and where to put it.
    Spawn {
        /// Configuration for the new node.
        node_config: NodeConfigRef,
        /// Identifier to give it.
        node_id: Sym,
        /// Whether it joins standby rather than the active set.
        standby: bool,
    },
}

impl EventMeta {
    /// The affected region, for outage events.
    #[inline]
    pub fn region(&self) -> Option<Sym> {
        match self {
            EventMeta::Region(r) => Some(*r),
            _ => None,
        }
    }

    /// The election epoch, for election completions.
    #[inline]
    pub fn epoch(&self) -> Option<u64> {
        match self {
            EventMeta::Epoch(e) => Some(*e),
            _ => None,
        }
    }

    /// The requested cluster size, for reconfigurations.
    #[inline]
    pub fn target_size(&self) -> Option<usize> {
        match self {
            EventMeta::TargetSize(t) => Some(*t),
            _ => None,
        }
    }

    /// The config for the node being spawned.
    #[inline]
    pub fn node_config(&self) -> Option<&NodeConfigRef> {
        match self {
            EventMeta::Spawn { node_config, .. } => Some(node_config),
            _ => None,
        }
    }

    /// The identifier of the node being spawned.
    #[inline]
    pub fn node_id(&self) -> Option<Sym> {
        match self {
            EventMeta::Spawn { node_id, .. } => Some(*node_id),
            _ => None,
        }
    }

    /// Whether the node being spawned joins standby.
    #[inline]
    pub fn standby(&self) -> bool {
        match self {
            EventMeta::Spawn { standby, .. } => *standby,
            _ => false,
        }
    }
}

/// A simulation event scheduled to occur at a specific time.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// When the event occurs, in seconds.
    pub time: Seconds,
    /// What kind of event it is.
    pub event_type: EventType,
    /// Node or region the event applies to.
    pub target_id: Sym,
    /// Event-specific payload.
    pub metadata: EventMeta,
}

impl Event {
    /// An event with no metadata.
    pub fn new(time: Seconds, event_type: EventType, target_id: Sym) -> Self {
        Event {
            time,
            event_type,
            target_id,
            metadata: EventMeta::default(),
        }
    }

    /// An event with metadata.
    pub fn with_meta(
        time: Seconds,
        event_type: EventType,
        target_id: Sym,
        metadata: EventMeta,
    ) -> Self {
        Event {
            time,
            event_type,
            target_id,
            metadata,
        }
    }
}

/// Heap entry ordered by `(time, generation)`, reversed so `BinaryHeap`
/// behaves as a min-heap.
///
/// Generations are unique, so the ordering is total and the pop sequence does
/// not depend on the heap's internal layout -- which is what lets this differ
/// from Python's `heapq` without changing behaviour.
#[derive(Debug)]
struct Entry {
    time: Seconds,
    generation: u64,
    event: Event,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
    }
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.generation.cmp(&self.generation))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue for simulation events, ordered by time.
///
/// `is_empty` takes `&mut self` because answering it means discarding any
/// cancelled entries sitting at the top of the heap, so clippy's usual
/// `len`/`is_empty` pairing does not apply.
#[derive(Debug, Default)]
#[allow(clippy::len_without_is_empty)]
pub struct EventQueue {
    heap: BinaryHeap<Entry>,
    counter: u64,
    /// Per-target cancellation thresholds, indexed by symbol.
    cancel_target: Vec<u64>,
    /// Per-(target, type) thresholds, indexed by `sym * EVENT_TYPE_COUNT + type`.
    cancel_type: Vec<u64>,
}

impl EventQueue {
    /// An empty queue.
    pub fn new() -> Self {
        EventQueue::default()
    }

    #[inline]
    fn target_threshold(&self, target: Sym) -> u64 {
        match self.cancel_target.get(target as usize) {
            Some(&t) => t,
            None => 0,
        }
    }

    #[inline]
    fn type_threshold(&self, target: Sym, event_type: EventType) -> u64 {
        let idx = target as usize * EVENT_TYPE_COUNT + event_type.index();
        match self.cancel_type.get(idx) {
            Some(&t) => t,
            None => 0,
        }
    }

    #[inline]
    fn is_cancelled(&self, generation: u64, event: &Event) -> bool {
        generation < self.target_threshold(event.target_id)
            || generation < self.type_threshold(event.target_id, event.event_type)
    }

    /// Schedule an event.
    pub fn push(&mut self, event: Event) {
        let generation = self.counter;
        self.counter += 1;
        self.heap.push(Entry {
            time: event.time,
            generation,
            event,
        });
    }

    /// Remove and return the next event, skipping cancelled ones.
    pub fn pop(&mut self) -> Option<Event> {
        while let Some(entry) = self.heap.pop() {
            if self.is_cancelled(entry.generation, &entry.event) {
                continue;
            }
            return Some(entry.event);
        }
        None
    }

    /// Return the next event without removing it, discarding cancelled
    /// entries from the top of the heap along the way.
    pub fn peek(&mut self) -> Option<&Event> {
        while let Some(entry) = self.heap.peek() {
            if self.is_cancelled(entry.generation, &entry.event) {
                self.heap.pop();
                continue;
            }
            break;
        }
        self.heap.peek().map(|e| &e.event)
    }

    /// Cancel every pending event for `target_id`, regardless of type.
    ///
    /// O(1): records the current generation counter as the threshold.  Stale
    /// events are discarded lazily on `pop`/`peek`.
    pub fn cancel_all_for(&mut self, target_id: Sym) {
        let idx = target_id as usize;
        if self.cancel_target.len() <= idx {
            self.cancel_target.resize(idx + 1, 0);
        }
        let current = self.counter;
        if self.cancel_target[idx] < current {
            self.cancel_target[idx] = current;
        }
    }

    /// Cancel pending events of one type for `target_id`.
    pub fn cancel_events_for(&mut self, target_id: Sym, event_type: EventType) {
        let idx = target_id as usize * EVENT_TYPE_COUNT + event_type.index();
        if self.cancel_type.len() <= idx {
            self.cancel_type.resize(idx + 1, 0);
        }
        let current = self.counter;
        if self.cancel_type[idx] < current {
            self.cancel_type[idx] = current;
        }
    }

    /// Cancel any existing event of this type for the target and schedule a
    /// replacement at `new_time`.
    pub fn reschedule(
        &mut self,
        target_id: Sym,
        event_type: EventType,
        new_time: Seconds,
        metadata: EventMeta,
    ) {
        self.cancel_events_for(target_id, event_type);
        self.push(Event::with_meta(new_time, event_type, target_id, metadata));
    }

    /// Drop every event and cancellation threshold, keeping the
    /// allocations for the next run.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.counter = 0;
        // `clear` keeps capacity, so the regrow in `cancel_*` is free.
        self.cancel_target.clear();
        self.cancel_type.clear();
    }

    /// Whether the queue has no live events left.
    pub fn is_empty(&mut self) -> bool {
        self.peek().is_none()
    }

    /// Number of entries still in the heap, including cancelled ones that
    /// have not yet surfaced.  Matches Python's `__len__`.
    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(time: f64, event_type: EventType, target: Sym) -> Event {
        Event::new(time, event_type, target)
    }

    #[test]
    fn pops_in_time_order() {
        let mut q = EventQueue::new();
        q.push(ev(30.0, EventType::NodeFailure, 2));
        q.push(ev(10.0, EventType::NodeFailure, 3));
        q.push(ev(20.0, EventType::NodeFailure, 4));

        assert_eq!(q.pop().unwrap().time, 10.0);
        assert_eq!(q.pop().unwrap().time, 20.0);
        assert_eq!(q.pop().unwrap().time, 30.0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn ties_break_by_insertion_order() {
        let mut q = EventQueue::new();
        q.push(ev(5.0, EventType::NodeFailure, 2));
        q.push(ev(5.0, EventType::NodeRecovery, 3));
        q.push(ev(5.0, EventType::NodeDataLoss, 4));

        assert_eq!(q.pop().unwrap().target_id, 2);
        assert_eq!(q.pop().unwrap().target_id, 3);
        assert_eq!(q.pop().unwrap().target_id, 4);
    }

    #[test]
    fn peek_does_not_consume() {
        let mut q = EventQueue::new();
        q.push(ev(7.0, EventType::NodeFailure, 2));
        assert_eq!(q.peek().unwrap().time, 7.0);
        assert_eq!(q.peek().unwrap().time, 7.0);
        assert_eq!(q.pop().unwrap().time, 7.0);
        assert!(q.peek().is_none());
    }

    #[test]
    fn cancel_by_type_only_affects_that_type() {
        let mut q = EventQueue::new();
        q.push(ev(10.0, EventType::NodeFailure, 2));
        q.push(ev(20.0, EventType::NodeRecovery, 2));

        q.cancel_events_for(2, EventType::NodeFailure);

        let next = q.pop().unwrap();
        assert_eq!(next.event_type, EventType::NodeRecovery);
        assert!(q.pop().is_none());
    }

    #[test]
    fn cancel_all_affects_every_type_for_the_target() {
        let mut q = EventQueue::new();
        q.push(ev(10.0, EventType::NodeFailure, 2));
        q.push(ev(20.0, EventType::NodeRecovery, 2));
        q.push(ev(30.0, EventType::NodeFailure, 3));

        q.cancel_all_for(2);

        let next = q.pop().unwrap();
        assert_eq!(next.target_id, 3);
        assert!(q.pop().is_none());
    }

    #[test]
    fn events_pushed_after_cancellation_survive() {
        let mut q = EventQueue::new();
        q.push(ev(10.0, EventType::NodeFailure, 2));
        q.cancel_all_for(2);
        q.push(ev(15.0, EventType::NodeFailure, 2));

        assert_eq!(q.pop().unwrap().time, 15.0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn reschedule_replaces_the_pending_event() {
        let mut q = EventQueue::new();
        q.push(ev(10.0, EventType::NodeSyncComplete, 2));
        q.reschedule(2, EventType::NodeSyncComplete, 50.0, EventMeta::default());

        assert_eq!(q.pop().unwrap().time, 50.0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn repeated_cancellation_keeps_the_highest_threshold() {
        let mut q = EventQueue::new();
        q.cancel_all_for(2);
        q.push(ev(10.0, EventType::NodeFailure, 2));
        // An earlier threshold must not resurrect the newer event.
        q.cancel_events_for(2, EventType::NodeRecovery);
        assert_eq!(q.pop().unwrap().time, 10.0);
    }

    #[test]
    fn is_empty_skips_cancelled_entries() {
        let mut q = EventQueue::new();
        q.push(ev(10.0, EventType::NodeFailure, 2));
        assert!(!q.is_empty());
        q.cancel_all_for(2);
        assert!(q.is_empty());
    }

    #[test]
    fn cancelling_an_unseen_target_is_harmless() {
        let mut q = EventQueue::new();
        q.cancel_all_for(500);
        q.cancel_events_for(500, EventType::NodeFailure);
        assert!(q.is_empty());
    }

    #[test]
    fn event_type_names_match_python() {
        assert_eq!(EventType::NodeFailure.as_str(), "NODE_FAILURE");
        assert_eq!(
            EventType::LeaderElectionComplete.as_str(),
            "LEADER_ELECTION_COMPLETE"
        );
        for (i, t) in EventType::ALL.iter().enumerate() {
            assert_eq!(t.index(), i);
        }
    }
}
