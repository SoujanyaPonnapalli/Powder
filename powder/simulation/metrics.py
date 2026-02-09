"""
Metrics collection for the Monte Carlo RSM simulator.

Tracks availability, cost, and data loss timing during simulation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .distributions import Seconds
from .cluster import ClusterState

if TYPE_CHECKING:
    from .protocol import Protocol


@dataclass
class MetricsCollector:
    """Collects metrics during a simulation run.

    Attributes:
        time_available: Total time (seconds) the system could commit.
        time_unavailable: Total time (seconds) the system could not commit.
        total_cost: Accumulated node costs in dollars.
        time_to_potential_data_loss: Time when quorum was first lost (or None).
        time_to_actual_data_loss: Time when data was definitely lost (or None).
    """

    time_available: Seconds = field(default_factory=lambda: Seconds(0))
    time_unavailable: Seconds = field(default_factory=lambda: Seconds(0))
    total_cost: float = 0.0

    time_to_potential_data_loss: Seconds | None = None
    time_to_actual_data_loss: Seconds | None = None

    # Internal state for tracking
    _last_update_time: Seconds = field(default_factory=lambda: Seconds(0))
    _was_available: bool = True
    _had_potential_loss: bool = False
    _had_actual_loss: bool = False

    def record_elapsed(
        self,
        current_time: Seconds,
        cluster: ClusterState | None = None,
    ) -> None:
        """Record elapsed time and costs using the pre-event cluster state.

        This should be called BEFORE an event is applied so that the interval
        since the last event is categorized using the post-previous-event state
        (which correctly represents the cluster during the interval).

        Costs are also computed here (rather than in ``update``) so that they
        reflect the nodes that actually existed during the interval, not
        nodes that were added or removed by the event being processed.

        Args:
            current_time: Current simulation time.
            cluster: Current cluster state (before event applied).
                When provided, node costs are accumulated for the elapsed
                interval. All nodes are billed regardless of availability
                state, matching real cloud provider billing.
        """
        time_delta = Seconds(current_time - self._last_update_time)
        if time_delta < 0:
            raise ValueError(f"Time went backwards: {self._last_update_time} -> {current_time}")

        if time_delta > 0:
            # Availability
            if self._was_available:
                self.time_available = Seconds(self.time_available + time_delta)
            else:
                self.time_unavailable = Seconds(self.time_unavailable + time_delta)

            # Cost: bill all nodes (active + provisioning) regardless of state.
            # In cloud environments you pay for VMs from launch through
            # failures until termination.
            if cluster is not None:
                hours_elapsed = time_delta / 3600.0
                for node in cluster.all_nodes_for_billing():
                    self.total_cost += node.config.cost_per_hour * hours_elapsed

        self._last_update_time = current_time

    def update(
        self,
        cluster: ClusterState,
        current_time: Seconds,
        protocol: Protocol | None = None,
    ) -> None:
        """Update availability state based on current cluster state.

        Should be called AFTER each event is fully applied so that
        ``_was_available`` reflects the post-event state for the next
        interval. Also checks for data-loss milestones.

        Note: cost accumulation happens in ``record_elapsed`` (before the
        event) so that costs reflect the nodes that existed during the
        interval rather than nodes added/removed by the event.

        Args:
            cluster: Current cluster state (after event applied).
            current_time: Current simulation time.
            protocol: Optional protocol for algorithm-specific availability.
                If provided, uses protocol.can_commit() instead of
                cluster.can_commit() for availability tracking.
        """
        # Set availability state for the *next* interval (post-event).
        if protocol is not None:
            self._was_available = protocol.can_commit(cluster)
        else:
            self._was_available = cluster.can_commit()

        # Track data loss events (only record first occurrence)
        if not self._had_potential_loss and cluster.has_potential_data_loss():
            self.time_to_potential_data_loss = current_time
            self._had_potential_loss = True

        if not self._had_actual_loss and cluster.has_actual_data_loss():
            self.time_to_actual_data_loss = current_time
            self._had_actual_loss = True

    def availability_fraction(self) -> float:
        """Calculate fraction of time the system was available.

        Returns:
            Availability as a fraction between 0 and 1.
        """
        total = self.total_time()
        if total <= 0:
            return 1.0  # No time has passed, consider available
        return self.time_available / total

    def total_time(self) -> Seconds:
        """Get total simulated time.

        Returns:
            Sum of available and unavailable time.
        """
        return Seconds(self.time_available + self.time_unavailable)

    def availability_percent(self) -> float:
        """Calculate availability as a percentage.

        Returns:
            Availability as a percentage between 0 and 100.
        """
        return self.availability_fraction() * 100.0

    def nines_of_availability(self) -> float | None:
        """Calculate availability in 'nines' notation.

        Returns:
            Number of nines (e.g., 3 for 99.9%), or None if 100% available.
        """
        fraction = self.availability_fraction()
        if fraction >= 1.0:
            return None  # Perfect availability
        if fraction <= 0.0:
            return 0.0

        import math

        return -math.log10(1.0 - fraction)

    def snapshot(self) -> "MetricsSnapshot":
        """Create an immutable snapshot of current metrics.

        Returns:
            MetricsSnapshot with current values.
        """
        return MetricsSnapshot(
            time_available=self.time_available,
            time_unavailable=self.time_unavailable,
            total_cost=self.total_cost,
            time_to_potential_data_loss=self.time_to_potential_data_loss,
            time_to_actual_data_loss=self.time_to_actual_data_loss,
        )

    def __repr__(self) -> str:
        avail = self.availability_percent()
        return (
            f"MetricsCollector(availability={avail:.2f}%, "
            f"cost=${self.total_cost:.2f}, "
            f"potential_loss={self.time_to_potential_data_loss}, "
            f"actual_loss={self.time_to_actual_data_loss})"
        )


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable snapshot of metrics at a point in time.

    Used as the return value from simulations to ensure metrics
    cannot be accidentally modified after simulation completes.
    """

    time_available: Seconds
    time_unavailable: Seconds
    total_cost: float
    time_to_potential_data_loss: Seconds | None
    time_to_actual_data_loss: Seconds | None

    def availability_fraction(self) -> float:
        """Calculate fraction of time the system was available."""
        total = self.time_available + self.time_unavailable
        if total <= 0:
            return 1.0
        return self.time_available / total

    def total_time(self) -> Seconds:
        """Get total simulated time."""
        return Seconds(self.time_available + self.time_unavailable)

    def __repr__(self) -> str:
        avail = self.availability_fraction() * 100
        return (
            f"MetricsSnapshot(availability={avail:.2f}%, "
            f"cost=${self.total_cost:.2f})"
        )
