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
        log_replay_rate_dist: Distribution for log replay rate (committed-data units
            replayed per second of wall time).
        snapshot_download_time_dist: Distribution for wall-clock time (seconds) to
            download a full snapshot from a donor node.
        spawn_dist: Distribution for time (seconds) to spawn a fresh replacement node.
    """

    region: str
    cost_per_hour: float
    failure_dist: Distribution
    recovery_dist: Distribution
    data_loss_dist: Distribution
    log_replay_rate_dist: Distribution
    snapshot_download_time_dist: Distribution
    spawn_dist: Distribution


@dataclass
class NodeState:
    """Dynamic state of a node during simulation.

    Attributes:
        node_id: Unique identifier for this node.
        config: Static configuration for the node.
        is_available: Whether the node is currently available (not transiently failed).
        has_data: Whether the node has data (False = permanent data loss).
        last_applied_index: Position in the committed data stream that this node
            has applied up to. Raw float, not a wall-clock time.
        last_snapshot_index: Position in the committed data stream at which this
            node last took a snapshot. Log entries before this can be truncated.
    """

    node_id: str
    config: NodeConfig
    is_available: bool = True
    has_data: bool = True
    last_applied_index: float = 0.0
    last_snapshot_index: float = 0.0

    def is_up_to_date(self, commit_index: float) -> bool:
        """Check if node is up-to-date at the given commit index.

        A node is up-to-date if it has applied all data up to commit_index.

        Args:
            commit_index: Current committed data position.

        Returns:
            True if node has all committed data up to commit_index.
        """
        return self.last_applied_index >= commit_index

    def lag(self, commit_index: float) -> float:
        """Calculate how far behind the node is in committed data.

        Args:
            commit_index: Current committed data position.

        Returns:
            Amount of committed data the node is missing.
        """
        return max(0.0, commit_index - self.last_applied_index)

    def __repr__(self) -> str:
        status = []
        if not self.is_available:
            status.append("unavailable")
        if not self.has_data:
            status.append("data_lost")
        status_str = ", ".join(status) if status else "healthy"
        return f"NodeState({self.node_id}, {status_str}, applied={self.last_applied_index:.1f})"
