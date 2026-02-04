"""
Node model for the Monte Carlo RSM simulator.

Defines node configuration (static properties and distributions) and
node state (dynamic properties that change during simulation).
"""

from dataclasses import dataclass, field

from .distributions import Distribution, Seconds


@dataclass
class NodeConfig:
    """Static configuration for a node.

    Attributes:
        region: Geographic region where the node is located.
        cost_per_hour: Dollar cost of running this node per hour.
        failure_dist: Distribution for time (seconds) until transient unavailability.
        recovery_dist: Distribution for time (seconds) to recover from transient failure.
        data_loss_dist: Distribution for time (seconds) until permanent data loss.
        sync_rate_dist: Distribution for sync rate (seconds of data per second of real time).
        spawn_dist: Distribution for time (seconds) to spawn a fresh replacement node.
    """

    region: str
    cost_per_hour: float
    failure_dist: Distribution
    recovery_dist: Distribution
    data_loss_dist: Distribution
    sync_rate_dist: Distribution
    spawn_dist: Distribution


@dataclass
class NodeState:
    """Dynamic state of a node during simulation.

    Attributes:
        node_id: Unique identifier for this node.
        config: Static configuration for the node.
        is_available: Whether the node is currently available (not transiently failed).
        has_data: Whether the node has data (False = permanent data loss).
        last_up_to_date_time: Simulation time when node was last fully synced.
    """

    node_id: str
    config: NodeConfig
    is_available: bool = True
    has_data: bool = True
    last_up_to_date_time: Seconds = field(default_factory=lambda: Seconds(0))

    def is_up_to_date(self, current_time: Seconds) -> bool:
        """Check if node is up-to-date at the given time.

        A node is up-to-date if its last sync time equals or exceeds current time.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            True if node has all committed data up to current_time.
        """
        return self.last_up_to_date_time >= current_time

    def lag_seconds(self, current_time: Seconds) -> Seconds:
        """Calculate how far behind the node is in committed data.

        Args:
            current_time: Current simulation time in seconds.

        Returns:
            Number of seconds of committed data the node is missing.
        """
        return Seconds(max(0.0, current_time - self.last_up_to_date_time))

    def time_to_sync(self, current_time: Seconds, sync_rate: float) -> Seconds | None:
        """Calculate time needed to catch up to current state.

        The sync_rate represents seconds of data synced per second of real time.
        For example:
        - sync_rate = 2.0: Node syncs 2 seconds of data per 1 second of real time
        - sync_rate = 1.0: Node syncs at exactly commit rate (stays same distance behind)
        - sync_rate = 0.5: Node syncs slower than commits arrive (falls further behind)

        Args:
            current_time: Current simulation time in seconds.
            sync_rate: Seconds of data synced per second of real time.

        Returns:
            Time in seconds to catch up, or None if sync_rate <= 1.0 (can never catch up).
        """
        if sync_rate <= 1.0:
            return None  # Cannot catch up if syncing slower than commits arrive

        lag = self.lag_seconds(current_time)
        if lag <= 0:
            return Seconds(0.0)  # Already up to date

        # Net catch-up rate = sync_rate - 1.0 (since 1 second of new commits per second)
        return Seconds(lag / (sync_rate - 1.0))

    def __repr__(self) -> str:
        status = []
        if not self.is_available:
            status.append("unavailable")
        if not self.has_data:
            status.append("data_lost")
        status_str = ", ".join(status) if status else "healthy"
        return f"NodeState({self.node_id}, {status_str}, synced_to={self.last_up_to_date_time:.1f}s)"
