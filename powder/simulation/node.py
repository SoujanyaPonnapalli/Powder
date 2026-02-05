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
        log_replay_rate_dist: Distribution for log replay rate (seconds of data per second).
        snapshot_download_time_dist: Distribution for time to download a snapshot.
        snapshot_interval_dist: Distribution for time between snapshot creation.
        spawn_dist: Distribution for time (seconds) to spawn a fresh replacement node.
    """

    region: str
    cost_per_hour: float
    failure_dist: Distribution
    recovery_dist: Distribution
    data_loss_dist: Distribution
    log_replay_rate_dist: Distribution  # Rate of replaying log (seconds of data / second)
    snapshot_download_time_dist: Distribution  # Time to download a snapshot
    snapshot_interval_dist: Distribution  # Time between creating snapshots
    spawn_dist: Distribution


@dataclass
class NodeState:
    """Dynamic state of a node during simulation.

    Attributes:
        node_id: Unique identifier for this node.
        config: Static configuration for the node.
        is_available: Whether the node is currently available (not transiently failed).
        has_data: Whether the node has data (False = permanent data loss).
        last_up_to_date_time: Simulation time of the latest committed data this node has.
        snapshot_time: Simulation time up to which this node has a snapshot (None if no snapshot).
    """

    node_id: str
    config: NodeConfig
    is_available: bool = True
    has_data: bool = True
    last_up_to_date_time: Seconds = field(default_factory=lambda: Seconds(0))
    snapshot_time: Seconds | None = None  # Time up to which snapshot contains data

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

    def time_to_sync_via_log(
        self, current_time: Seconds, log_replay_rate: float, commit_rate: float
    ) -> Seconds | None:
        """Calculate time to catch up by replaying log entries.

        Args:
            current_time: Current simulation time in seconds.
            log_replay_rate: Seconds of log data replayed per second of real time.
            commit_rate: Seconds of new commits arriving per second of real time.

        Returns:
            Time in seconds to catch up via log replay, or None if can't catch up.
        """
        if log_replay_rate <= commit_rate:
            return None  # Can't catch up if replaying slower than commits arrive

        lag = self.lag_seconds(current_time)
        if lag <= 0:
            return Seconds(0.0)  # Already up to date

        # Net catch-up rate = log_replay_rate - commit_rate
        net_rate = log_replay_rate - commit_rate
        return Seconds(lag / net_rate)

    def time_to_sync_via_snapshot(
        self,
        current_time: Seconds,
        snapshot_time: Seconds,
        snapshot_download_time: float,
        log_replay_rate: float,
        commit_rate: float,
    ) -> Seconds | None:
        """Calculate time to catch up by downloading a snapshot then replaying remaining log.

        Args:
            current_time: Current simulation time in seconds.
            snapshot_time: The time up to which the snapshot contains data.
            snapshot_download_time: Time to download the snapshot.
            log_replay_rate: Seconds of log data replayed per second.
            commit_rate: Seconds of new commits arriving per second.

        Returns:
            Total time to sync via snapshot + log, or None if can't catch up.
        """
        if log_replay_rate <= commit_rate:
            return None  # Can't catch up after snapshot

        # After downloading snapshot, we'll be at snapshot_time
        # But time will have advanced by snapshot_download_time
        # So we need to replay: (current_time + snapshot_download_time) - snapshot_time
        time_after_download = current_time + snapshot_download_time
        
        # During download, new commits arrive
        lag_after_snapshot = time_after_download - snapshot_time
        
        if lag_after_snapshot <= 0:
            # Snapshot is ahead of where we'd be - just download time
            return Seconds(snapshot_download_time)

        # Time to replay remaining log after snapshot
        net_rate = log_replay_rate - commit_rate
        log_replay_time = lag_after_snapshot / net_rate

        return Seconds(snapshot_download_time + log_replay_time)

    def has_snapshot(self) -> bool:
        """Check if this node has a snapshot available."""
        return self.snapshot_time is not None

    def create_snapshot(self, current_time: Seconds) -> None:
        """Create a snapshot at the current time (only if up-to-date)."""
        if self.is_up_to_date(current_time):
            self.snapshot_time = current_time

    def __repr__(self) -> str:
        status = []
        if not self.is_available:
            status.append("unavailable")
        if not self.has_data:
            status.append("data_lost")
        status_str = ", ".join(status) if status else "healthy"
        snapshot_info = f", snapshot={self.snapshot_time:.1f}s" if self.snapshot_time else ""
        return f"NodeState({self.node_id}, {status_str}, synced_to={self.last_up_to_date_time:.1f}s{snapshot_info})"
