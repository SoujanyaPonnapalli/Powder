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

    # Protocol events
    LEADER_ELECTION_COMPLETE = "leader_election_complete"  # Leader election finished


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
    Supports cancellation of events by target_id.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._cancelled: set[tuple[str, EventType]] = set()
        self._counter = 0  # For stable ordering of equal-time events

    def push(self, event: Event) -> None:
        """Add an event to the queue.

        Args:
            event: Event to schedule.
        """
        heapq.heappush(self._heap, event)

    def pop(self) -> Event | None:
        """Remove and return the next event.

        Skips cancelled events automatically.

        Returns:
            Next event by time, or None if queue is empty.
        """
        while self._heap:
            event = heapq.heappop(self._heap)
            key = (event.target_id, event.event_type)
            if key in self._cancelled:
                self._cancelled.discard(key)
                continue
            return event
        return None

    def peek(self) -> Event | None:
        """Return the next event without removing it.

        Returns:
            Next event by time, or None if queue is empty.
        """
        # Skip cancelled events
        while self._heap:
            event = self._heap[0]
            key = (event.target_id, event.event_type)
            if key in self._cancelled:
                heapq.heappop(self._heap)
                self._cancelled.discard(key)
                continue
            return event
        return None

    def cancel_events_for(self, target_id: str, event_type: EventType | None = None) -> None:
        """Cancel pending events for a target.

        Args:
            target_id: Target whose events to cancel.
            event_type: If provided, only cancel events of this type.
                       If None, cancel all event types for the target.
        """
        if event_type is not None:
            self._cancelled.add((target_id, event_type))
        else:
            # Cancel all event types for this target
            for et in EventType:
                self._cancelled.add((target_id, et))

    def reschedule(
        self, target_id: str, event_type: EventType, new_time: Seconds, metadata: dict | None = None
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
        # Peek handles cancelled events
        return self.peek() is None

    def __len__(self) -> int:
        """Return number of events in queue (may include cancelled)."""
        return len(self._heap)

    def __repr__(self) -> str:
        return f"EventQueue({len(self._heap)} events)"
