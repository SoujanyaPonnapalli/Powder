"""
Discrete-event simulator engine for the Monte Carlo RSM simulator.

The simulator processes events in time order, updating cluster state
and invoking the strategy to react to changes.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cluster import ClusterState
from .distributions import Seconds
from .events import Event, EventQueue, EventType
from .metrics import MetricsCollector, MetricsSnapshot
from .network import NetworkConfig
from .node import NodeConfig, NodeState, SyncState
from .protocol import Protocol
from .strategy import Action, ActionType, ClusterStrategy


@dataclass
class SimulationResult:
    """Result of a simulation run.

    Attributes:
        end_time: Final simulation time in seconds.
        end_reason: Why the simulation ended ("time_limit", "data_loss", "condition_met").
        metrics: Final metrics snapshot.
        event_log: List of all events processed (if logging enabled).
        final_cluster: Final cluster state.
    """

    end_time: Seconds
    end_reason: str
    metrics: MetricsSnapshot
    event_log: list[Event] = field(default_factory=list)
    final_cluster: ClusterState | None = None


class Simulator:
    """Discrete-event simulation engine for RSM clusters.

    The simulator maintains:
    - A cluster state that evolves over time
    - An event queue ordered by time
    - A strategy that reacts to events
    - A metrics collector tracking availability and cost

    Events are processed in time order. Each event may:
    1. Update cluster state (e.g., node fails)
    2. Trigger strategy reactions (e.g., spawn replacement)
    3. Schedule new events (e.g., recovery time)

    Between events, the simulator advances commit_index for the elapsed
    wall-clock interval (if the system can commit) and keeps available
    nodes' applied indices in sync.
    """

    def __init__(
        self,
        initial_cluster: ClusterState,
        strategy: ClusterStrategy,
        protocol: Protocol,
        network_config: NetworkConfig | None = None,
        seed: int | None = None,
        log_events: bool = False,
    ):
        """Initialize the simulator.

        Args:
            initial_cluster: Starting cluster state.
            strategy: Strategy for reacting to events.
            protocol: Protocol for algorithm-specific availability semantics.
            network_config: Optional network partition configuration.
            seed: Random seed for reproducibility.
            log_events: Whether to keep a log of all events.
        """
        self.cluster = initial_cluster
        self.strategy = strategy
        self.network_config = network_config
        self.protocol = protocol
        self.log_events = log_events

        self.rng = np.random.default_rng(seed)
        self.event_queue = EventQueue()
        self.metrics = MetricsCollector()
        self.event_log: list[Event] = []

        self._initialized = False

    def _initialize(self) -> None:
        """Set up initial events and state."""
        if self._initialized:
            return

        # Schedule initial events for each node
        for node in self.cluster.nodes.values():
            self._schedule_node_events(node)

        # Schedule network events if configured
        if self.network_config:
            for region in self.network_config.regions:
                self._schedule_network_outage(region)

        # Initialize protocol (e.g., pick initial leader)
        for event in self.protocol.on_simulation_start(self.cluster, self.rng):
            self.event_queue.push(event)

        # Let strategy take initial actions
        actions = self.strategy.on_simulation_start(self.cluster, self.rng)
        for action in actions:
            self._execute_action(action)

        # Start syncs for any initially lagging nodes
        self._retry_pending_syncs()

        self._initialized = True

    def _schedule_node_events(self, node: NodeState) -> None:
        """Schedule failure and data loss events for a node."""
        current_time = self.cluster.current_time

        # Schedule transient failure
        failure_time = node.config.failure_dist.sample(self.rng)
        self.event_queue.push(
            Event(
                time=Seconds(current_time + failure_time),
                event_type=EventType.NODE_FAILURE,
                target_id=node.node_id,
            )
        )

        # Schedule data loss
        data_loss_time = node.config.data_loss_dist.sample(self.rng)
        self.event_queue.push(
            Event(
                time=Seconds(current_time + data_loss_time),
                event_type=EventType.NODE_DATA_LOSS,
                target_id=node.node_id,
            )
        )

    def _schedule_network_outage(self, region: str) -> None:
        """Schedule the next network outage event for a region.

        Each affected region maintains its own independent outage chain,
        allowing multiple regions to experience outages simultaneously.

        Args:
            region: The region to schedule an outage for.
        """
        if not self.network_config or region not in self.network_config.regions:
            return

        current_time = self.cluster.current_time
        outage_delay = self.network_config.outage_dist.sample(self.rng)

        self.event_queue.push(
            Event(
                time=Seconds(current_time + outage_delay),
                event_type=EventType.NETWORK_OUTAGE_START,
                target_id=region,
                metadata={"region": region},
            )
        )

    def _advance_commit_index(self, elapsed: float) -> None:
        """Advance commit_index, up-to-date nodes, and syncing nodes.

        Called before processing each event (or at time-limit) for the
        wall-clock interval since the last event.

        If the system can commit, commit_index advances by
        elapsed * protocol.commit_rate.  Available, up-to-date, non-syncing
        nodes keep up with the frontier and take snapshots when crossing
        snapshot boundaries.

        Syncing nodes advance independently -- even when the cluster
        cannot commit -- because they are replaying data the donor
        already has.

        Args:
            elapsed: Wall-clock seconds since last event.
        """
        if elapsed <= 0:
            return

        can_commit = self.protocol.can_commit(self.cluster)

        if can_commit:
            commit_amount = elapsed * self.protocol.commit_rate
            old_index = self.cluster.commit_index
            new_index = old_index + commit_amount
            self.cluster.commit_index = new_index

            snapshot_interval = self.protocol.snapshot_interval
            for node in self.cluster.nodes.values():
                if (
                    self.cluster._node_effectively_available(node)
                    and node.sync is None  # Not actively syncing
                    and node.last_applied_index >= old_index
                ):
                    # Node keeps up with commit frontier
                    node.last_applied_index = new_index
                    # Advance snapshot if crossing a boundary
                    if snapshot_interval > 0:
                        new_snap = int(new_index // snapshot_interval) * snapshot_interval
                        if new_snap > node.last_snapshot_index:
                            node.last_snapshot_index = new_snap

        # Advance syncing nodes (even when cluster can't commit)
        self._advance_syncing_nodes(elapsed)

    def _advance_syncing_nodes(self, elapsed: float) -> None:
        """Advance sync progress for all actively syncing nodes.

        Syncing nodes advance regardless of whether the cluster can commit.
        A node replays data from its donor at ``log_replay_rate`` and cannot
        advance past the donor's ``last_applied_index`` -- a node can only
        obtain data the donor actually has.

        When the cluster cannot commit the donor's position is frozen, so
        the syncing node closes the gap at the full replay rate.

        Args:
            elapsed: Wall-clock seconds to advance.
        """
        snapshot_interval = self.protocol.snapshot_interval

        for node in self.cluster.nodes.values():
            if node.sync is None:
                continue
            if not self.cluster._node_effectively_available(node):
                continue

            donor = self.cluster.get_node(node.sync.donor_id)
            if donor is None or not self.cluster._node_effectively_available(donor):
                continue  # Sync paused -- donor unavailable

            remaining_time = elapsed

            # Phase 1: snapshot download (if applicable)
            if node.sync.phase == "snapshot_download":
                if remaining_time >= node.sync.snapshot_remaining:
                    # Snapshot finishes within this interval
                    remaining_time -= node.sync.snapshot_remaining
                    node.sync.snapshot_remaining = 0.0
                    node.last_applied_index = node.sync.target_snapshot_index
                    node.sync.phase = "log_replay"
                    # Record the snapshot
                    if (
                        snapshot_interval > 0
                        and node.sync.target_snapshot_index > node.last_snapshot_index
                    ):
                        node.last_snapshot_index = node.sync.target_snapshot_index
                else:
                    node.sync.snapshot_remaining -= remaining_time
                    remaining_time = 0.0

            # Phase 2: log replay (if time remains)
            if node.sync.phase == "log_replay" and remaining_time > 0:
                node.last_applied_index = min(
                    node.last_applied_index
                    + node.sync.log_replay_rate * remaining_time,
                    donor.last_applied_index,
                )
                # Advance snapshot if crossing boundaries
                if snapshot_interval > 0:
                    new_snap = (
                        int(node.last_applied_index // snapshot_interval)
                        * snapshot_interval
                    )
                    if new_snap > node.last_snapshot_index:
                        node.last_snapshot_index = new_snap

    def _start_sync(self, node: NodeState) -> None:
        """Start a sync for a lagging node.

        Finds the best donor, determines the sync path (log-only vs
        snapshot + log suffix) using ``Distribution.mean`` for estimation,
        samples actual rates, creates ``SyncState`` on the node, and
        schedules a ``NODE_SYNC_COMPLETE`` event.

        The sync target is ``donor.last_applied_index`` -- a node can only
        obtain data the donor actually has.
        """
        if node.sync is not None:
            return  # Already syncing

        donor = self.cluster.find_sync_donor(node)
        if donor is None:
            return  # No donor; _retry_pending_syncs will pick this up later

        donor_lag = donor.last_applied_index - node.last_applied_index
        if donor_lag <= 0:
            return  # Already caught up to donor

        # Sample the actual log replay rate for this sync session
        log_replay_rate = node.config.log_replay_rate_dist.sample(self.rng)

        can_commit = self.protocol.can_commit(self.cluster)
        commit_rate_eff = self.protocol.commit_rate if can_commit else 0.0

        snapshot_interval = self.protocol.snapshot_interval
        log_retention = self.protocol.log_retention_ops

        # Determine donor's earliest available log entry
        if log_retention > 0:
            donor_earliest_log = max(
                0.0, donor.last_applied_index - log_retention
            )
        else:
            donor_earliest_log = 0.0  # Infinite retention

        must_snapshot = (
            log_retention > 0 and node.last_applied_index < donor_earliest_log
        )

        use_snapshot = False
        target_snap = 0.0
        snapshot_download_time = 0.0

        if must_snapshot and snapshot_interval > 0:
            # Donor has GC'd needed log entries -- must download snapshot
            use_snapshot = True
            target_snap = (
                int(donor.last_applied_index // snapshot_interval)
                * snapshot_interval
            )
            snapshot_download_time = (
                node.config.snapshot_download_time_dist.sample(self.rng)
            )

        elif snapshot_interval > 0 and not must_snapshot:
            # Log is available -- estimate both paths using mean, pick faster
            mean_replay_rate = node.config.log_replay_rate_dist.mean
            mean_net_rate = mean_replay_rate - commit_rate_eff
            if mean_net_rate <= 0:
                mean_net_rate = mean_replay_rate  # Fallback

            log_only_est = donor_lag / mean_net_rate

            target_snap_candidate = (
                int(donor.last_applied_index // snapshot_interval)
                * snapshot_interval
            )
            if target_snap_candidate > node.last_applied_index:
                mean_snap_time = node.config.snapshot_download_time_dist.mean
                remaining_after_snap = (
                    donor.last_applied_index
                    - target_snap_candidate
                    + commit_rate_eff * mean_snap_time
                )
                snap_est = mean_snap_time + remaining_after_snap / mean_net_rate

                if snap_est < log_only_est:
                    use_snapshot = True
                    target_snap = target_snap_candidate
                    snapshot_download_time = (
                        node.config.snapshot_download_time_dist.sample(self.rng)
                    )

        # Create the SyncState
        if use_snapshot:
            node.sync = SyncState(
                donor_id=donor.node_id,
                phase="snapshot_download",
                log_replay_rate=log_replay_rate,
                snapshot_remaining=snapshot_download_time,
                target_snapshot_index=target_snap,
            )
        else:
            node.sync = SyncState(
                donor_id=donor.node_id,
                phase="log_replay",
                log_replay_rate=log_replay_rate,
            )

        # Schedule the initial SYNC_COMPLETE event
        remaining = self._compute_remaining_sync_time(node)
        if remaining is not None and remaining >= 0:
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + max(remaining, 0)),
                    event_type=EventType.NODE_SYNC_COMPLETE,
                    target_id=node.node_id,
                )
            )
        # If remaining is None (net_rate <= 0), sync is active but no event
        # scheduled. _reschedule_active_syncs will handle it when conditions
        # change.

    def _compute_remaining_sync_time(self, node: NodeState) -> Seconds | None:
        """Compute wall-clock time until a syncing node catches up to its donor.

        Uses ``donor.last_applied_index`` as the sync target -- the data the
        donor actually has -- not ``cluster.commit_index``.

        Returns:
            Estimated seconds to sync completion, or None if sync cannot
            complete under current conditions (no donor or net_rate <= 0).
        """
        sync = node.sync
        if sync is None:
            return None

        donor = self.cluster.get_node(sync.donor_id)
        if donor is None or not self.cluster._node_effectively_available(donor):
            return None  # Donor unavailable, sync paused

        can_commit = self.protocol.can_commit(self.cluster)
        commit_rate_eff = self.protocol.commit_rate if can_commit else 0.0
        net_rate = sync.log_replay_rate - commit_rate_eff

        if sync.phase == "snapshot_download":
            remaining_snapshot = max(0.0, sync.snapshot_remaining)

            # After snapshot, node will be at target_snapshot_index.
            # During download, donor advances by commit_rate * remaining_snapshot.
            donor_at_download_end = (
                donor.last_applied_index + commit_rate_eff * remaining_snapshot
            )
            remaining_log = donor_at_download_end - sync.target_snapshot_index

            if remaining_log <= 0:
                return Seconds(remaining_snapshot)
            if net_rate <= 0:
                return None  # Can't catch up the log suffix
            return Seconds(remaining_snapshot + remaining_log / net_rate)

        elif sync.phase == "log_replay":
            remaining_lag = donor.last_applied_index - node.last_applied_index
            if remaining_lag <= 0:
                return Seconds(0)
            if net_rate <= 0:
                return None  # Can't catch up
            return Seconds(remaining_lag / net_rate)

        return None

    def _cancel_syncs_from_donor(self, donor_id: str) -> None:
        """Cancel or failover syncs when a donor becomes unavailable.

        For each node syncing from the downed donor:
        - If another donor exists *and* is ahead of the syncing node,
          swap the donor reference and keep progress (seamless failover).
        - Otherwise, cancel the SYNC_COMPLETE event and clear the sync
          state.  ``_retry_pending_syncs`` will resume the sync when a
          suitable donor becomes available later.
        """
        for node in self.cluster.nodes_syncing_from(donor_id):
            alt_donor = self.cluster.find_sync_donor(node)
            if (
                alt_donor is not None
                and alt_donor.last_applied_index > node.last_applied_index
            ):
                # Seamless failover: swap donor, keep progress.
                # Cancel the old SYNC_COMPLETE event so that
                # _reschedule_active_syncs creates a fresh one based
                # on the new donor's position.
                self.event_queue.cancel_events_for(
                    node.node_id, EventType.NODE_SYNC_COMPLETE
                )
                node.sync.donor_id = alt_donor.node_id
            else:
                # No useful donor available: pause sync
                self.event_queue.cancel_events_for(
                    node.node_id, EventType.NODE_SYNC_COMPLETE
                )
                node.sync = None

    def _reschedule_active_syncs(self) -> None:
        """Reschedule SYNC_COMPLETE events for all actively syncing nodes.

        Called after every event to keep sync completion timing accurate
        when conditions change (commit ability, donor availability).
        """
        for node in self.cluster.nodes.values():
            if node.sync is None:
                continue

            # Check if donor is still available and ahead
            donor = self.cluster.get_node(node.sync.donor_id)
            if (
                donor is None
                or not self.cluster._node_effectively_available(donor)
                or donor.last_applied_index <= node.last_applied_index
            ):
                # Try failover to a donor that is ahead of us
                alt = self.cluster.find_sync_donor(node)
                if (
                    alt is not None
                    and alt.last_applied_index > node.last_applied_index
                ):
                    node.sync.donor_id = alt.node_id
                    donor = alt
                else:
                    # No useful donor: pause sync
                    self.event_queue.cancel_events_for(
                        node.node_id, EventType.NODE_SYNC_COMPLETE
                    )
                    node.sync = None
                    continue

            remaining = self._compute_remaining_sync_time(node)
            if remaining is not None and remaining >= 0:
                self.event_queue.reschedule(
                    node.node_id,
                    EventType.NODE_SYNC_COMPLETE,
                    Seconds(self.cluster.current_time + max(remaining, 0)),
                )
            else:
                # net_rate <= 0: can't catch up now, cancel event but keep
                # sync state so it can be rescheduled when conditions improve
                self.event_queue.cancel_events_for(
                    node.node_id, EventType.NODE_SYNC_COMPLETE
                )

    def _retry_pending_syncs(self) -> None:
        """Start syncs for lagging nodes that don't have an active sync.

        Called whenever a potential donor becomes available (node recovery,
        network outage end, sync completion).
        """
        for node in self.cluster.nodes_needing_sync():
            self._start_sync(node)

    def _process_event(self, event: Event) -> None:
        """Process a single event and update cluster state."""
        # Advance commit_index for the wall-clock interval since last event
        elapsed = event.time - self.cluster.current_time
        # Record elapsed time, availability, and costs BEFORE applying the
        # event.  The pre-event cluster state is unchanged since the previous
        # event, so querying cluster/protocol directly gives the correct
        # availability for the interval.  Costs use the pre-event cluster
        # state so that nodes added/removed by this event are not incorrectly
        # billed for the prior interval.
        self.metrics.record_elapsed(event.time, cluster=self.cluster, protocol=self.protocol)
        
        
        self._advance_commit_index(elapsed)
        self.cluster.current_time = event.time

        if self.log_events:
            self.event_log.append(event)

        # Apply event to cluster state
        if event.event_type == EventType.NODE_FAILURE:
            self._apply_node_failure(event)

        elif event.event_type == EventType.NODE_RECOVERY:
            self._apply_node_recovery(event)

        elif event.event_type == EventType.NODE_DATA_LOSS:
            self._apply_node_data_loss(event)

        elif event.event_type == EventType.NODE_SYNC_COMPLETE:
            self._apply_node_sync_complete(event)

        elif event.event_type == EventType.NODE_SPAWN_COMPLETE:
            self._apply_node_spawn_complete(event)

        elif event.event_type == EventType.NETWORK_OUTAGE_START:
            self._apply_network_outage_start(event)

        elif event.event_type == EventType.NETWORK_OUTAGE_END:
            self._apply_network_outage_end(event)

        elif event.event_type == EventType.NODE_REPLACEMENT_TIMEOUT:
            pass  # Handled by strategy below

        elif event.event_type == EventType.LEADER_ELECTION_COMPLETE:
            pass  # Handled by protocol below

        elif event.event_type == EventType.CLUSTER_RECONFIGURATION:
            pass  # Handled by strategy below

        # Let protocol react (e.g., detect leader failure, complete election)
        for new_event in self.protocol.on_event(event, self.cluster, self.rng):
            self.event_queue.push(new_event)

        # Let strategy react
        actions = self.strategy.on_event(event, self.cluster, self.rng, self.protocol)
        for action in actions:
            self._execute_action(action)

        # Keep sync completion timing accurate after every event.
        # Conditions may have changed (commit ability, donor availability).
        self._reschedule_active_syncs()
        self._retry_pending_syncs()

        # Check for data-loss milestones AFTER the event is fully applied.
        self.metrics.update(self.cluster, event.time, self.protocol)

    def _apply_node_failure(self, event: Event) -> None:
        """Apply a transient node failure."""
        node = self.cluster.get_node(event.target_id)
        if node and node.has_data:
            node.is_available = False

            # Cancel any pending sync for this node and clear sync state
            self.event_queue.cancel_events_for(
                event.target_id, EventType.NODE_SYNC_COMPLETE
            )
            node.sync = None

            # This node may have been a donor -- cancel/failover syncs from it
            self._cancel_syncs_from_donor(event.target_id)

            # Schedule recovery
            recovery_time = node.config.recovery_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + recovery_time),
                    event_type=EventType.NODE_RECOVERY,
                    target_id=node.node_id,
                )
            )

            # Schedule next failure (after recovery)
            next_failure_time = node.config.failure_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(
                        self.cluster.current_time + recovery_time + next_failure_time
                    ),
                    event_type=EventType.NODE_FAILURE,
                    target_id=node.node_id,
                )
            )

    def _apply_node_recovery(self, event: Event) -> None:
        """Apply node recovery from transient failure.

        The recovered node may need to sync if it fell behind.  It may
        also serve as a donor for other nodes, so pending syncs are
        retried (handled by the _retry_pending_syncs call in _process_event).
        """
        node = self.cluster.get_node(event.target_id)
        if node and node.has_data:
            node.is_available = True

            # Node may need to sync if it fell behind
            if not node.is_up_to_date(self.cluster.commit_index):
                self._start_sync(node)

    def _apply_node_data_loss(self, event: Event) -> None:
        """Apply permanent data loss to a node."""
        node = self.cluster.get_node(event.target_id)
        if node:
            node.has_data = False
            node.is_available = False
            node.sync = None

            # Cancel all pending events for this node
            self.event_queue.cancel_events_for(event.target_id)

            # This node may have been a donor -- cancel/failover syncs from it
            self._cancel_syncs_from_donor(event.target_id)

    def _apply_node_sync_complete(self, event: Event) -> None:
        """Apply sync completion for a node.

        Validates that the donor is still available.  If the donor went
        down and no alternate exists, the sync is cleared and the node
        will be picked up by ``_retry_pending_syncs``.

        On success the node snaps to ``donor.last_applied_index`` (the
        data the donor actually has) rather than ``cluster.commit_index``.
        """
        node = self.cluster.get_node(event.target_id)
        if node is None or node.sync is None:
            return

        if not self.cluster._node_effectively_available(node):
            node.sync = None
            return

        donor = self.cluster.get_node(node.sync.donor_id)
        if donor is not None and self.cluster._node_effectively_available(donor):
            # Sync complete: snap to donor's position
            node.last_applied_index = donor.last_applied_index
            # Take snapshot if crossing a boundary
            snapshot_interval = self.protocol.snapshot_interval
            if snapshot_interval > 0:
                snap = (
                    int(node.last_applied_index // snapshot_interval)
                    * snapshot_interval
                )
                if snap > node.last_snapshot_index:
                    node.last_snapshot_index = snap
        else:
            # Donor went down at completion -- try failover
            alt = self.cluster.find_sync_donor(node)
            if alt is not None:
                node.sync.donor_id = alt.node_id
                # Reschedule will handle the rest
                return
            # No donor: clear sync, will be retried
        node.sync = None

    def _apply_node_spawn_complete(self, event: Event) -> None:
        """Apply completion of node spawning.

        Moves the node from the provisioning set (where it was placed at
        spawn request time for billing) to the active node set.
        """
        node_config = event.metadata.get("node_config")
        node_id = event.metadata.get("node_id", event.target_id)

        if node_config:
            # Remove from provisioning (was added at spawn request time)
            self.cluster.remove_provisioning_node(node_id)

            # Create active node (starts with data, needs to sync)
            new_node = NodeState(
                node_id=node_id,
                config=node_config,
                is_available=True,
                has_data=True,
                last_applied_index=0.0,
                last_snapshot_index=0.0,
            )
            self.cluster.add_node(new_node)

            # Schedule events for new node
            self._schedule_node_events(new_node)

            # Start sync from a donor
            self._start_sync(new_node)

    def _apply_network_outage_start(self, event: Event) -> None:
        """Apply start of a region network outage (all nodes in region become unavailable).

        Cancels syncs for nodes in the affected region and handles donor
        failover for nodes that were syncing from them.
        """
        region = event.metadata.get("region", event.target_id)
        if region:
            self.cluster.network.add_outage(region)

            # Cancel syncs for nodes in the affected region and handle
            # downstream syncs that used these nodes as donors.
            for node in self.cluster.nodes.values():
                if node.config.region == region:
                    self.event_queue.cancel_events_for(
                        node.node_id, EventType.NODE_SYNC_COMPLETE
                    )
                    node.sync = None
                    # This node may have been a donor
                    self._cancel_syncs_from_donor(node.node_id)

            # Schedule outage end
            duration = self.network_config.outage_duration_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + duration),
                    event_type=EventType.NETWORK_OUTAGE_END,
                    target_id=region,
                    metadata={"region": region},
                )
            )

    def _apply_network_outage_end(self, event: Event) -> None:
        """Apply end of a region network outage.

        Recovered nodes may need to sync.  They may also serve as donors,
        so pending syncs are retried (handled by the _retry_pending_syncs
        call in _process_event).
        """
        region = event.metadata.get("region", event.target_id)
        if region:
            self.cluster.network.remove_outage(region)

            # Start syncs for nodes in the recovered region that fell behind.
            for node in self.cluster.nodes.values():
                if (
                    node.config.region == region
                    and node.is_available
                    and node.has_data
                    and not node.is_up_to_date(self.cluster.commit_index)
                    and node.sync is None
                ):
                    self._start_sync(node)

            # Schedule next outage for this region
            self._schedule_network_outage(region)

    def _execute_action(self, action: Action) -> None:
        """Execute a strategy action."""
        if action.action_type == ActionType.SPAWN_NODE:
            self._action_spawn_node(action)

        elif action.action_type == ActionType.REMOVE_NODE:
            node_id = action.params.get("node_id")
            if node_id:
                self.cluster.remove_node(node_id)
                self.cluster.remove_provisioning_node(node_id)
                self.event_queue.cancel_events_for(node_id)

        elif action.action_type == ActionType.SCALE_DOWN:
            new_size = action.params.get("new_size", self.cluster.target_cluster_size)
            self.cluster.target_cluster_size = new_size

        elif action.action_type == ActionType.SCALE_UP:
            new_size = action.params.get("new_size", self.cluster.target_cluster_size)
            self.cluster.target_cluster_size = new_size

        elif action.action_type == ActionType.START_SYNC:
            node_id = action.params.get("node_id")
            node = self.cluster.get_node(node_id) if node_id else None
            if node:
                self._start_sync(node)

        elif action.action_type == ActionType.SCHEDULE_REPLACEMENT_CHECK:
            self._action_schedule_replacement_check(action)

        elif action.action_type == ActionType.CANCEL_REPLACEMENT_CHECK:
            self._action_cancel_replacement_check(action)

        elif action.action_type == ActionType.SCHEDULE_RECONFIGURATION:
            self._action_schedule_reconfiguration(action)

    def _action_schedule_replacement_check(self, action: Action) -> None:
        """Schedule a replacement timeout event for a node."""
        node_id: str = action.params.get("node_id", "")
        timeout: float = action.params.get("timeout", 0)

        if node_id and timeout > 0:
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + timeout),
                    event_type=EventType.NODE_REPLACEMENT_TIMEOUT,
                    target_id=node_id,
                )
            )

    def _action_cancel_replacement_check(self, action: Action) -> None:
        """Cancel a pending replacement timeout event for a node."""
        node_id: str = action.params.get("node_id", "")
        if node_id:
            self.event_queue.cancel_events_for(
                node_id, EventType.NODE_REPLACEMENT_TIMEOUT
            )

    def _action_schedule_reconfiguration(self, action: Action) -> None:
        """Schedule a cluster reconfiguration event."""
        delay: float = action.params.get("delay", 0)
        target_size: int = action.params.get("target_size", 0)

        if delay > 0 and target_size > 0:
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + delay),
                    event_type=EventType.CLUSTER_RECONFIGURATION,
                    target_id="cluster",  # Global target
                    metadata={"target_size": target_size},
                )
            )

    def _action_spawn_node(self, action: Action) -> None:
        """Execute a spawn node action.

        Immediately adds the node to the cluster's provisioning set so
        that it is billed from the moment it is requested (matching
        real cloud provider billing). The node moves to the active set
        when NODE_SPAWN_COMPLETE fires.
        """
        node_config: NodeConfig = action.params.get("node_config")
        node_id: str = action.params.get("node_id", f"spawned_{self.rng.integers(10000)}")

        if node_config:
            # Add to provisioning immediately for cost tracking
            provisioning_node = NodeState(
                node_id=node_id,
                config=node_config,
                is_available=False,
                has_data=False,
            )
            self.cluster.add_provisioning_node(provisioning_node)

            spawn_time = node_config.spawn_dist.sample(self.rng)
            self.event_queue.push(
                Event(
                    time=Seconds(self.cluster.current_time + spawn_time),
                    event_type=EventType.NODE_SPAWN_COMPLETE,
                    target_id=node_id,
                    metadata={"node_config": node_config, "node_id": node_id},
                )
            )

    def run_until(
        self,
        end_time: Seconds | None = None,
        stop_condition: Callable[[ClusterState], bool] | None = None,
    ) -> SimulationResult:
        """Run the simulation until a stopping condition is met.

        Args:
            end_time: Maximum simulation time (or None for no limit).
            stop_condition: Callable that returns True when simulation should stop.

        Returns:
            SimulationResult with final metrics and state.
        """
        self._initialize()

        end_reason = "unknown"

        while True:
            # Check for empty queue
            if self.event_queue.is_empty():
                end_reason = "no_events"
                break

            # Peek at next event
            next_event = self.event_queue.peek()
            if next_event is None:
                end_reason = "no_events"
                break

            # Check time limit
            if end_time is not None and next_event.time > end_time:
                # Advance commit_index up to end_time
                elapsed = end_time - self.cluster.current_time
                self._advance_commit_index(elapsed)

                # Record elapsed time/cost and update metrics up to end_time.
                # No event to apply, so state doesn't change — record + update together.
                self.metrics.record_elapsed(end_time, cluster=self.cluster, protocol=self.protocol)
                self.metrics.update(self.cluster, end_time, self.protocol)
                self.cluster.current_time = end_time
                end_reason = "time_limit"
                break

            # Process event
            event = self.event_queue.pop()
            if event:
                self._process_event(event)

            # Check for data loss first (more specific)
            if self.protocol.has_actual_data_loss(self.cluster):
                end_reason = "data_loss"
                break

            # Check stop condition
            if stop_condition and stop_condition(self.cluster):
                end_reason = "condition_met"
                break

        return SimulationResult(
            end_time=self.cluster.current_time,
            end_reason=end_reason,
            metrics=self.metrics.snapshot(),
            event_log=self.event_log if self.log_events else [],
            final_cluster=self.cluster,
        )

    def run_for(self, duration: Seconds) -> SimulationResult:
        """Run the simulation for a specified duration.

        Args:
            duration: How long to run in seconds.

        Returns:
            SimulationResult with final metrics and state.
        """
        end_time = Seconds(self.cluster.current_time + duration)
        return self.run_until(end_time=end_time)

    def run_until_data_loss(
        self, max_time: Seconds | None = None
    ) -> SimulationResult:
        """Run until data loss occurs or time limit reached.

        Args:
            max_time: Maximum simulation time.

        Returns:
            SimulationResult with final metrics and state.
        """
        return self.run_until(
            end_time=max_time,
            stop_condition=lambda c: self.protocol.has_actual_data_loss(c),
        )
