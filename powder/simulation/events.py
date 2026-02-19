"""
Event system for the discrete-event Monte Carlo simulator.

Provides event types, event dataclass, and priority queue for managing
simulation events ordered by time.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import heapq

from .distributions import Seconds


class EventType(Enum):
    """Types of events that can occur during simulation."""

    # Node events
    NODE_FAILURE = "node_failure"  # Transient unavailability
    NODE_RECOVERY = "node_recovery"  # Recovery from transient failure
    NODE_DATA_LOSS = "node_data_loss"  # Permanent data loss (e.g., disk failure)
    NODE_SYNC_COMPLETE = "node_sync_complete"  # Node finished syncing
    NODE_SPAWN_COMPLETE = "node_spawn_complete"  # New node finished spawning

    # Network events
    NETWORK_OUTAGE_START = "network_outage_start"  # Partition begins
    NETWORK_OUTAGE_END = "network_outage_end"  # Partition ends

    # Replacement events
    NODE_REPLACEMENT_TIMEOUT = "node_replacement_timeout"  # Node replacement timeout reached

    # Protocol events
    LEADER_ELECTION_COMPLETE = "leader_election_complete"  # Leader election finished

    # Cluster events
    CLUSTER_RECONFIGURATION = "cluster_reconfiguration"  # Attempt to reconfigure cluster size


@dataclass(order=True)
class Event:
    """A simulation event scheduled to occur at a specific time.

    Events are ordered by time for priority queue operations.

    Attributes:
        time: When the event occurs (in seconds).
        event_type: Type of event (not used for ordering).
        target_id: Identifier for the event target (node_id or region name).
        metadata: Additional event-specific data (not used for ordering).

    Metadata conventions:
        - NODE_SYNC_COMPLETE: {"sync_to_time": Seconds} - time node will be synced to
        - NODE_SPAWN_COMPLETE: {"node_config": NodeConfig} - config for new node
        - NETWORK_OUTAGE_*: {"region": str} - affected region name
    """

    time: Seconds
    event_type: EventType = field(compare=False)
    target_id: str = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return f"Event({self.time:.2f}s, {self.event_type.value}, {self.target_id})"


class EventQueue:
    """Priority queue for simulation events ordered by time.

    Uses a min-heap to efficiently retrieve the next event.
    Supports lazy O(1) cancellation via generation thresholds.

    Each pushed event is stamped with an incrementing generation counter.
    Cancellation records the current counter as a threshold for the
    cancelled key; any event whose generation is below that threshold
    is silently discarded on ``pop``/``peek``.  Events pushed *after*
    the cancellation carry a generation >= the threshold and are
    unaffected.
    """

    def __init__(self) -> None:
        # Heap entries: (event.time, gen, event)
        self._heap: list[tuple[float, int, Event]] = []
        self._counter = 0
        # Lazy cancellation thresholds.  An event is stale when its gen
        # is below the recorded threshold for its key.
        self._cancel_type: dict[tuple[str, EventType], int] = {}
        self._cancel_target: dict[str, int] = {}

    # -- internal helpers ------------------------------------------------

    def _is_cancelled(self, gen: int, event: Event) -> bool:
        """Return True if *event* at generation *gen* has been cancelled."""
        target_thresh = self._cancel_target.get(event.target_id, 0)
        if gen < target_thresh:
            return True
        type_thresh = self._cancel_type.get(
            (event.target_id, event.event_type), 0
        )
        return gen < type_thresh

    # -- public API ------------------------------------------------------

    def push(self, event: Event) -> None:
        """Add an event to the queue.

        Args:
            event: Event to schedule.
        """
        gen = self._counter
        self._counter += 1
        heapq.heappush(self._heap, (event.time, gen, event))

    def pop(self) -> Event | None:
        """Remove and return the next event.

        Skips cancelled events automatically.

        Returns:
            Next event by time, or None if queue is empty.
        """
        while self._heap:
            _time, gen, event = heapq.heappop(self._heap)
            if self._is_cancelled(gen, event):
                continue
            return event
        return None

    def peek(self) -> Event | None:
        """Return the next event without removing it.

        Returns:
            Next event by time, or None if queue is empty.
        """
        while self._heap:
            _time, gen, event = self._heap[0]
            if self._is_cancelled(gen, event):
                heapq.heappop(self._heap)
                continue
            return event
        return None

    def cancel_events_for(
        self, target_id: str, event_type: EventType | None = None
    ) -> None:
        """Cancel all pending events for *target_id* (optionally of a
        specific *event_type*).

        O(1) — just records the current generation counter as the
        threshold.  Stale events are lazily discarded in ``pop``/``peek``.

        Args:
            target_id: Target whose events to cancel.
            event_type: If provided, only cancel events of this type.
                       If None, cancel all event types for the target.
        """
        if event_type is not None:
            key = (target_id, event_type)
            self._cancel_type[key] = max(
                self._cancel_type.get(key, 0), self._counter
            )
        else:
            self._cancel_target[target_id] = max(
                self._cancel_target.get(target_id, 0), self._counter
            )

    def reschedule(
        self,
        target_id: str,
        event_type: EventType,
        new_time: Seconds,
        metadata: dict | None = None,
    ) -> None:
        """Cancel existing event and schedule a new one at a different time.

        Args:
            target_id: Target to reschedule event for.
            event_type: Type of event to reschedule.
            new_time: New time for the event.
            metadata: Optional new metadata (if None, uses empty dict).
        """
        self.cancel_events_for(target_id, event_type)
        self.push(
            Event(
                time=new_time,
                event_type=event_type,
                target_id=target_id,
                metadata=metadata or {},
            )
        )

    def is_empty(self) -> bool:
        """Check if queue has no pending events."""
        return self.peek() is None

    def __len__(self) -> int:
        """Return number of events in queue (may include cancelled)."""
        return len(self._heap)

    def __repr__(self) -> str:
        return f"EventQueue({len(self._heap)} events)"

