"""
Protocol abstraction for the Monte Carlo RSM simulator.

Protocols define algorithm-specific availability semantics on top of the
raw cluster state. Different consensus algorithms have different requirements
for when the system can commit (quorum rules, leader presence, etc.).

The Protocol layer computes what the physical cluster state means for the
algorithm -- can we commit? is there a leader? is an election happening?
ClusterState remains a pure representation of physical reality.

Protocols also define recovery semantics: commit rate, snapshot intervals,
and how nodes sync when they fall behind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .distributions import Distribution, Seconds
from .events import Event, EventType

if TYPE_CHECKING:
    from .cluster import ClusterState
    from .node import NodeState


class Protocol(ABC):
    """Abstract base class for consensus protocol models.

    A protocol computes algorithm-specific availability on top of the raw
    cluster state. It may maintain internal state (e.g., current leader,
    election-in-progress) and react to simulation events.

    The simulator calls on_event() after each event is applied to the cluster,
    and uses can_commit() for metrics collection instead of the cluster's
    built-in can_commit().

    Protocols also control:
    - commit_rate: how fast data is committed (data units per second of wall time)
    - snapshot_interval: how often nodes take snapshots (in commit-index units)
    - compute_sync_time: how long it takes a lagging node to catch up
    """

    @property
    def commit_rate(self) -> float:
        """Committed data units produced per second of wall time when system can commit.

        Default is 1.0 (one unit of data committed per second of wall time).
        """
        return 1.0

    @property
    def snapshot_interval(self) -> float:
        """Commit-index interval between snapshots.

        When a node's applied index crosses a multiple of this value,
        it takes a snapshot and can truncate log entries before it.
        0 means no snapshots / no log truncation.
        """
        return 0.0

    @property
    def log_retention_ops(self) -> float:
        """Number of committed-data units of log a node retains.

        A node at ``last_applied_index = D`` keeps log entries from
        ``max(0, D - log_retention_ops)`` to ``D``.  Entries before
        that boundary have been garbage-collected and are no longer
        available for log-only replay by syncing peers.

        0 means infinite retention (no garbage collection).  Nodes
        may still keep transactions preceding their latest snapshot,
        so the log window and snapshot schedule are independent.
        """
        return 0.0

    @abstractmethod
    def can_commit(self, cluster: ClusterState) -> bool:
        """Determine if the system can accept writes right now.

        This is the algorithm-specific availability check. Different protocols
        have different requirements (quorum rules, leader presence, etc.).

        Args:
            cluster: Current cluster state (physical reality).

        Returns:
            True if the system can accept new commits.
        """

    @abstractmethod
    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """React to a simulation event.

        Called by the simulator after each event is applied to the cluster.
        May update internal protocol state and return new events to schedule
        (e.g., leader election completion).

        Args:
            event: The event that just occurred.
            cluster: Current cluster state (after event was applied).
            rng: Random number generator for reproducibility.

        Returns:
            List of new events for the simulator to schedule.
        """

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """Called once at simulation start for initialization.

        Override to perform protocol-specific setup (e.g., pick initial leader).

        Args:
            cluster: Initial cluster state.
            rng: Random number generator.

        Returns:
            List of events for the simulator to schedule.
        """
        return []

    def quorum_size(self, cluster: ClusterState) -> int:
        """Calculate the quorum size needed for commits.

        Default implementation uses majority quorum (n // 2 + 1).
        Override for protocols with different quorum requirements.

        Args:
            cluster: Current cluster state.

        Returns:
            Minimum number of nodes needed for quorum.
        """
        return len(cluster.nodes) // 2 + 1

    def has_potential_data_loss(self, cluster: ClusterState) -> bool:
        """Check if quorum is lost (potential data loss).

        When fewer than a quorum of nodes are available, we can't be
        certain the remaining nodes have the latest committed data.

        Args:
            cluster: Current cluster state.

        Returns:
            True if quorum is lost.
        """
        return cluster.num_available() < self.quorum_size(cluster)

    def has_actual_data_loss(self, cluster: ClusterState) -> bool:
        """Check if all up-to-date nodes have failed (definite data loss).

        Data is definitely lost when no nodes have the latest committed
        data, even if some nodes exist with older data.

        Args:
            cluster: Current cluster state.

        Returns:
            True if data is definitely lost.
        """
        for n in cluster.nodes.values():
            if n.has_data and n.is_up_to_date(cluster.commit_index):
                return False
        return True

    def compute_sync_time(
        self,
        node: NodeState,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> Seconds | None:
        """Compute wall-clock time for a lagging node to catch up to its donor.

        The sync target is the donor's ``last_applied_index`` (the data the
        donor actually has), not ``cluster.commit_index``.  The donor's
        position advances at ``commit_rate`` while the syncing node replays
        at ``log_replay_rate``, giving a net catch-up rate of
        ``log_replay_rate - commit_rate`` when the cluster can commit, or
        ``log_replay_rate`` when it cannot.

        The sync path (log-only vs snapshot + log suffix) depends on the
        donor's log garbage-collection state:

        * If the donor has GC'd log entries the node needs
          (``node.last_applied_index < donor.last_applied_index - log_retention_ops``),
          the node **must** download the donor's latest snapshot first, then
          replay the remaining log suffix.
        * If the log is still available, the node **may** choose log-only
          replay *or* snapshot + log.  The faster path is chosen by
          comparing estimated times using ``Distribution.mean``.

        Args:
            node: The lagging node that needs to sync.
            cluster: Current cluster state.
            rng: Random number generator for sampling distributions.

        Returns:
            Wall-clock seconds to complete sync, or None if the node cannot
            catch up (replay rate <= commit rate) or no donor is available.
        """
        donor = cluster.find_sync_donor(node)
        if donor is None:
            return None  # No donor available

        donor_lag = donor.last_applied_index - node.last_applied_index
        if donor_lag <= 0:
            return Seconds(0)

        can_commit = self.can_commit(cluster)
        commit_rate_eff = self.commit_rate if can_commit else 0.0
        log_replay_rate = node.config.log_replay_rate_dist.sample(rng)
        net_rate = log_replay_rate - commit_rate_eff

        if net_rate <= 0:
            return None  # Can't catch up

        snapshot_interval = self.snapshot_interval
        log_retention = self.log_retention_ops

        # Determine the donor's earliest available log entry
        if log_retention > 0:
            donor_earliest_log = max(0.0, donor.last_applied_index - log_retention)
        else:
            donor_earliest_log = 0.0  # Infinite retention

        # Determine if the donor has GC'd log entries the node needs
        must_snapshot = (
            log_retention > 0
            and node.last_applied_index < donor_earliest_log
        )

        if must_snapshot and snapshot_interval > 0:
            # --- Forced snapshot path ---
            # Pick donor's latest snapshot as the target
            target_snap = (
                int(donor.last_applied_index // snapshot_interval)
                * snapshot_interval
            )
            snapshot_download_time = node.config.snapshot_download_time_dist.sample(rng)

            # After downloading, node is at target_snap.  During the download,
            # the donor advances by commit_rate_eff * download_time.
            remaining_log = (
                donor.last_applied_index
                - target_snap
                + commit_rate_eff * snapshot_download_time
            )
            return Seconds(snapshot_download_time + remaining_log / net_rate)

        elif snapshot_interval > 0 and not must_snapshot:
            # --- Log is available; choose the faster path via mean estimates ---
            mean_replay_rate = node.config.log_replay_rate_dist.mean
            mean_net_rate = mean_replay_rate - commit_rate_eff
            if mean_net_rate <= 0:
                mean_net_rate = mean_replay_rate  # Fallback

            log_only_est = donor_lag / mean_net_rate

            target_snap = (
                int(donor.last_applied_index // snapshot_interval)
                * snapshot_interval
            )
            if target_snap > node.last_applied_index:
                mean_snap_time = node.config.snapshot_download_time_dist.mean
                remaining_after_snap = (
                    donor.last_applied_index
                    - target_snap
                    + commit_rate_eff * mean_snap_time
                )
                snap_est = mean_snap_time + remaining_after_snap / mean_net_rate

                if snap_est < log_only_est:
                    # Snapshot path is estimated faster -- use it
                    snapshot_download_time = node.config.snapshot_download_time_dist.sample(rng)
                    remaining_log = (
                        donor.last_applied_index
                        - target_snap
                        + commit_rate_eff * snapshot_download_time
                    )
                    return Seconds(snapshot_download_time + remaining_log / net_rate)

            # Log-only replay (default when no snapshot advantage)
            return Seconds(donor_lag / net_rate)

        else:
            # --- No snapshots configured: log-only replay ---
            return Seconds(donor_lag / net_rate)


class LeaderlessUpToDateQuorumProtocol(Protocol):
    """Leaderless protocol requiring a quorum of up-to-date nodes.

    This replicates the simulator's original hardcoded behavior and serves
    as the backward-compatible default. The system can commit when a majority
    of nodes are available, have data, and are up-to-date.

    Suitable for modeling protocols like EPaxos or multi-decree Paxos variants
    where any node can propose and a quorum of up-to-date replicas is needed.
    """

    def __init__(
        self,
        commit_rate: float = 1.0,
        snapshot_interval: float = 0.0,
        log_retention_ops: float = 0.0,
    ) -> None:
        self._commit_rate = commit_rate
        self._snapshot_interval = snapshot_interval
        self._log_retention_ops = log_retention_ops

    @property
    def commit_rate(self) -> float:
        return self._commit_rate

    @property
    def snapshot_interval(self) -> float:
        return self._snapshot_interval

    @property
    def log_retention_ops(self) -> float:
        return self._log_retention_ops

    def can_commit(self, cluster: ClusterState) -> bool:
        """Commit requires a majority of up-to-date nodes."""
        return cluster.num_up_to_date() >= self.quorum_size(cluster)

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """No-op: leaderless protocol has no leader tracking."""
        return []

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """No-op: no initialization needed."""
        return []


class LeaderlessMajorityAvailableProtocol(Protocol):
    """Leaderless protocol requiring a quorum of available nodes.

    The system can commit when a majority of nodes are available and have data,
    regardless of whether they are up-to-date. This models protocols where
    any available node can serve writes and sync happens asynchronously.

    Suitable for modeling eventually-consistent systems or protocols where
    lagging replicas can still participate in commits.
    """

    def __init__(
        self,
        commit_rate: float = 1.0,
        snapshot_interval: float = 0.0,
        log_retention_ops: float = 0.0,
    ) -> None:
        self._commit_rate = commit_rate
        self._snapshot_interval = snapshot_interval
        self._log_retention_ops = log_retention_ops

    @property
    def commit_rate(self) -> float:
        return self._commit_rate

    @property
    def snapshot_interval(self) -> float:
        return self._snapshot_interval

    @property
    def log_retention_ops(self) -> float:
        return self._log_retention_ops

    def can_commit(self, cluster: ClusterState) -> bool:
        """Commit requires a majority of available nodes (not necessarily up-to-date)."""
        return cluster.num_available() >= self.quorum_size(cluster)

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """No-op: leaderless protocol has no leader tracking."""
        return []

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """No-op: no initialization needed."""
        return []


class RaftLikeProtocol(Protocol):
    """Leader-based protocol with election downtime.

    Models protocols like Raft or multi-Paxos where:
    - A single leader must be present to accept writes.
    - If the leader fails, an election occurs and the system is unavailable
      for the duration of the election.
    - A quorum of up-to-date nodes is needed to commit.

    The election time is sampled from a configurable distribution each time
    a leader fails. If no eligible node exists when the election completes,
    another election attempt is scheduled.

    Attributes:
        election_time_dist: Distribution for leader election duration.
    """

    def __init__(
        self,
        election_time_dist: Distribution,
        commit_rate: float = 1.0,
        snapshot_interval: float = 0.0,
        log_retention_ops: float = 0.0,
    ) -> None:
        """Initialize the Raft-like protocol.

        Args:
            election_time_dist: Distribution for time (seconds) to complete
                a leader election after the leader fails.
            commit_rate: Committed data units per second of wall time.
            snapshot_interval: Commit-index interval between snapshots.
            log_retention_ops: Number of committed-data units of log retained.
                0 means infinite retention.
        """
        self.election_time_dist = election_time_dist
        self._commit_rate = commit_rate
        self._snapshot_interval = snapshot_interval
        self._log_retention_ops = log_retention_ops
        self._leader_id: str | None = None
        self._election_in_progress: bool = False
        self._election_stalled: bool = False
        self._election_epoch: int = 0

    @property
    def commit_rate(self) -> float:
        return self._commit_rate

    @property
    def snapshot_interval(self) -> float:
        return self._snapshot_interval

    @property
    def log_retention_ops(self) -> float:
        return self._log_retention_ops

    @property
    def leader_id(self) -> str | None:
        """Current leader node ID, or None if no leader."""
        return self._leader_id

    @property
    def election_in_progress(self) -> bool:
        """Whether a leader election is currently in progress."""
        return self._election_in_progress

    def can_commit(self, cluster: ClusterState) -> bool:
        """Commit requires a leader, no active election, and a quorum of up-to-date nodes.

        Returns False if:
        - An election is in progress, OR
        - There is no leader, OR
        - The leader node is not effectively available, OR
        - Fewer than a quorum of nodes are up-to-date.
        """
        if self._election_in_progress:
            return False

        if self._leader_id is None:
            return False

        # Verify leader is still effectively available
        leader = cluster.get_node(self._leader_id)
        if leader is None or not cluster._node_effectively_available(leader):
            return False

        return cluster.num_up_to_date() >= self.quorum_size(cluster)

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """Pick an initial leader from available up-to-date nodes."""
        self._leader_id = self._pick_leader(cluster)
        return []

    # Events that indicate a node may have become available/eligible,
    # which could unblock a stalled election.
    _AVAILABILITY_EVENTS = {
        EventType.NODE_RECOVERY,
        EventType.NETWORK_OUTAGE_END,
        EventType.NODE_SYNC_COMPLETE,
        EventType.NODE_SPAWN_COMPLETE,
    }

    # Events that indicate a node may have become unavailable,
    # which could invalidate an in-progress election.
    _FAILURE_EVENTS = {
        EventType.NODE_FAILURE,
        EventType.NODE_DATA_LOSS,
        EventType.NETWORK_OUTAGE_START,
    }

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """React to events that affect leader state.

        - NODE_FAILURE of the leader: starts an election.
        - NODE_DATA_LOSS of the leader: starts an election.
        - NETWORK_OUTAGE_START affecting the leader's region: starts an election.
        - LEADER_ELECTION_COMPLETE: finalizes the election (if epoch matches).
        - NODE_RECOVERY / NETWORK_OUTAGE_END / NODE_SYNC_COMPLETE /
          NODE_SPAWN_COMPLETE: restarts a stalled election.
        - NODE_FAILURE / NODE_DATA_LOSS / NETWORK_OUTAGE_START during election:
          invalidates election if quorum is lost.

        Returns:
            List of new events to schedule (e.g., election completion).
        """
        if event.event_type == EventType.LEADER_ELECTION_COMPLETE:
            return self._handle_election_complete(event, cluster, rng)

        # Check if a stalled election should be restarted
        if self._election_stalled and event.event_type in self._AVAILABILITY_EVENTS:
            return self._restart_election(cluster, rng)

        # Check if the leader was lost
        if self._leader_id is not None and not self._election_in_progress:
            leader_lost = False

            if event.event_type in (EventType.NODE_FAILURE, EventType.NODE_DATA_LOSS):
                if event.target_id == self._leader_id:
                    leader_lost = True

            elif event.event_type == EventType.NETWORK_OUTAGE_START:
                leader = cluster.get_node(self._leader_id)
                if leader is not None:
                    region = event.metadata.get("region", event.target_id)
                    if leader.config.region == region:
                        leader_lost = True

            if leader_lost:
                return self._start_election(cluster, rng)

        # Check if quorum was lost during an in-progress (non-stalled) election.
        # If so, invalidate the current election immediately — it must restart
        # from scratch when quorum is restored.
        if (
            self._election_in_progress
            and not self._election_stalled
            and event.event_type in self._FAILURE_EVENTS
            and cluster.num_available() < self.quorum_size(cluster)
        ):
            self._election_stalled = True

        return []

    def _start_election(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """Begin a leader election.

        Returns:
            List containing the election completion event.
        """
        self._leader_id = None
        self._election_in_progress = True
        self._election_stalled = False
        self._election_epoch += 1

        election_duration = self.election_time_dist.sample(rng)
        return [
            Event(
                time=Seconds(cluster.current_time + election_duration),
                event_type=EventType.LEADER_ELECTION_COMPLETE,
                target_id="protocol",
                metadata={"epoch": self._election_epoch},
            )
        ]

    def _restart_election(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """Restart a stalled election after a node became available.

        Returns:
            List containing a new election completion event.
        """
        self._election_stalled = False
        self._election_epoch += 1

        election_duration = self.election_time_dist.sample(rng)
        return [
            Event(
                time=Seconds(cluster.current_time + election_duration),
                event_type=EventType.LEADER_ELECTION_COMPLETE,
                target_id="protocol",
                metadata={"epoch": self._election_epoch},
            )
        ]

    def _handle_election_complete(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        """Handle election completion: pick a new leader or stall.

        Ignores stale events from cancelled elections (epoch mismatch).

        Returns:
            Empty list. If no leader was found, the election stalls
            until a node-availability event triggers a restart.
        """
        # Ignore stale election events from a previous (cancelled) epoch.
        # This happens when quorum was lost mid-election, invalidating
        # the pending event, and a new election was already restarted.
        event_epoch = event.metadata.get("epoch", 0)
        if event_epoch != self._election_epoch:
            return []

        new_leader = self._pick_leader(cluster)

        if new_leader is not None:
            self._leader_id = new_leader
            self._election_in_progress = False
            self._election_stalled = False
            return []

        # No eligible node available -- stall until a node comes online.
        # on_event will restart the election when a recovery/outage-end/
        # sync-complete/spawn-complete event arrives.
        self._election_stalled = True
        return []

    def _pick_leader(self, cluster: ClusterState) -> str | None:
        """Pick a leader from eligible nodes.

        A node is eligible if it is effectively available and up-to-date.
        Additionally, a majority of nodes must be effectively available
        (a leader needs votes from a quorum to win an election).

        Returns:
            Node ID of the chosen leader, or None if no eligible node exists
            or majority is not available.
        """
        # A leader election requires a majority of nodes to participate
        if cluster.num_available() < self.quorum_size(cluster):
            return None

        eligible = [
            node
            for node in cluster.nodes.values()
            if cluster._node_effectively_available(node)
            and node.is_up_to_date(cluster.commit_index)
        ]

        if not eligible:
            return None

        # Pick the first eligible node (deterministic for reproducibility)
        eligible.sort(key=lambda n: n.node_id)
        return eligible[0].node_id
