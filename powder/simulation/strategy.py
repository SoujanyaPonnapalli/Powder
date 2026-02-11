"""
Strategy interface and implementations for the Monte Carlo RSM simulator.

Strategies define how the cluster management system reacts to events
like node failures, recoveries, and data loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from .cluster import ClusterState
from .distributions import Seconds
from .events import Event, EventType
from .node import NodeConfig, NodeState

if TYPE_CHECKING:
    from .protocol import Protocol


class ActionType(Enum):
    """Types of actions a strategy can take."""

    SPAWN_NODE = "spawn_node"  # Start spawning a new node
    REMOVE_NODE = "remove_node"  # Remove a node from cluster
    SCALE_DOWN = "scale_down"  # Reduce target cluster size
    SCALE_UP = "scale_up"  # Increase target cluster size
    START_SYNC = "start_sync"  # Trigger a node to start syncing
    SCHEDULE_REPLACEMENT_CHECK = "schedule_replacement_check"  # Schedule a replacement timeout
    CANCEL_REPLACEMENT_CHECK = "cancel_replacement_check"  # Cancel a pending replacement timeout
    NO_OP = "no_op"  # Do nothing


@dataclass
class Action:
    """An action to be taken by the simulator.

    Attributes:
        action_type: Type of action to take.
        params: Action-specific parameters.

    Param conventions by action type:
        SPAWN_NODE: {"node_config": NodeConfig, "node_id": str}
        REMOVE_NODE: {"node_id": str}
        SCALE_DOWN: {"new_size": int}
        SCALE_UP: {"new_size": int}
        START_SYNC: {"node_id": str}
        NO_OP: {}
    """

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)


class ClusterStrategy(ABC):
    """Abstract base class for cluster management strategies.

    Strategies react to simulation events and return actions for
    the simulator to execute.
    """

    @abstractmethod
    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        """React to a simulation event.

        Called by the simulator after each event is processed.
        The strategy can inspect the current cluster state and
        return actions to modify the cluster.

        Args:
            event: The event that just occurred.
            cluster: Current cluster state (after event was applied).
            rng: Random number generator for reproducibility.
            protocol: Protocol for algorithm-specific queries
                (e.g. can_commit, quorum_size).

        Returns:
            List of actions for the simulator to execute.
        """
        pass

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Called once at simulation start.

        Override to perform initialization actions.

        Args:
            cluster: Initial cluster state.
            rng: Random number generator.

        Returns:
            List of initial actions.
        """
        return []


class NoOpStrategy(ClusterStrategy):
    """Strategy that takes no actions.

    Useful for baseline comparisons or when you want to observe
    natural cluster degradation without intervention.
    """

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        return []


class SimpleReplacementStrategy(ClusterStrategy):
    """Simple strategy that spawns replacements for failed nodes.

    When a node fails (transient or data loss), this strategy:
    1. If multiple nodes fail simultaneously, may scale down cluster
    2. Otherwise, spawns a replacement node

    Attributes:
        scale_down_threshold: Number of simultaneous failures that trigger scale-down.
        default_node_config: Config template for spawned nodes.
    """

    def __init__(
        self,
        default_node_config: NodeConfig,
        scale_down_threshold: int = 2,
    ):
        """Initialize the strategy.

        Args:
            default_node_config: Template for spawning new nodes.
            scale_down_threshold: If this many or more nodes are unavailable,
                                  scale down instead of replacing.
        """
        self.default_node_config = default_node_config
        self.scale_down_threshold = scale_down_threshold
        self._spawn_counter = 0
        self._pending_spawns: set[str] = set()

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        actions: list[Action] = []

        if event.event_type == EventType.NODE_FAILURE:
            actions.extend(self._handle_failure(event, cluster, rng))

        elif event.event_type == EventType.NODE_DATA_LOSS:
            actions.extend(self._handle_data_loss(event, cluster, rng))

        elif event.event_type == EventType.NODE_RECOVERY:
            actions.extend(self._handle_recovery(event, cluster, rng))

        elif event.event_type == EventType.NODE_SPAWN_COMPLETE:
            # Remove from pending spawns
            node_id = event.metadata.get("node_id", event.target_id)
            self._pending_spawns.discard(node_id)

        return actions

    def _handle_failure(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Handle a transient node failure."""
        unavailable_count = len(cluster.nodes) - cluster.num_available()

        # If too many failures, consider scaling down
        if unavailable_count >= self.scale_down_threshold:
            new_size = max(3, cluster.target_cluster_size - 2)
            if new_size < cluster.target_cluster_size:
                return [Action(ActionType.SCALE_DOWN, {"new_size": new_size})]

        return []  # Wait for recovery

    def _handle_data_loss(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Handle permanent data loss on a node."""
        # Check if we should spawn a replacement
        nodes_with_data = cluster.num_with_data()
        pending_count = len(self._pending_spawns)

        # Only spawn if we're below target and not already spawning
        if nodes_with_data + pending_count < cluster.target_cluster_size:
            return self._spawn_replacement(cluster, rng)

        return []

    def _handle_recovery(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Handle a node recovering from transient failure."""
        node = cluster.get_node(event.target_id)
        if node and not node.is_up_to_date(cluster.commit_index):
            # Node recovered but is lagging, start sync
            return [Action(ActionType.START_SYNC, {"node_id": node.node_id})]
        return []

    def _spawn_replacement(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Create action to spawn a replacement node."""
        self._spawn_counter += 1
        node_id = f"node_{self._spawn_counter}"
        self._pending_spawns.add(node_id)

        return [
            Action(
                ActionType.SPAWN_NODE,
                {
                    "node_config": self.default_node_config,
                    "node_id": node_id,
                },
            )
        ]

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Ensure we have target number of nodes at start."""
        actions = []

        # Spawn nodes if below target
        while len(cluster.nodes) + len(self._pending_spawns) < cluster.target_cluster_size:
            actions.extend(self._spawn_replacement(cluster, rng))

        return actions


class NodeReplacementStrategy(ClusterStrategy):
    """Strategy that replaces nodes after a configurable failure timeout.

    This models the standard production protocol for handling node failures
    in replicated state machines:

    1. A node becomes unavailable (transient failure, network outage, or data loss).
    2. A **failure timeout** countdown begins.
    3. If the node recovers before the timeout, the countdown is cancelled.
    4. If the timeout fires and the node is still unavailable:
       - Check that the cluster can still commit (quorum exists).
       - Remove the failed node from the cluster.
       - Spawn a replacement node with similar characteristics.
       - The replacement joins, syncs from peers, and becomes operational.

    The failed node is removed *before* the replacement is added to keep the
    quorum denominator stable. Adding a 4th node to a 3-node cluster would
    raise the quorum to 3, but only 2 nodes are up-to-date, causing the
    cluster to lose commit ability during the entire sync period.

    For permanent data loss (disk failure), replacement is immediate — there
    is no point waiting for a timeout since the node cannot recover.

    Attributes:
        failure_timeout: How long a node must be unavailable before replacement
            is triggered (in seconds).
        default_node_config: Config template for spawned replacement nodes.
            If None, uses the failed node's own config.
    """

    def __init__(
        self,
        failure_timeout: Seconds,
        default_node_config: NodeConfig | None = None,
    ):
        """Initialize the node replacement strategy.

        Args:
            failure_timeout: Duration (seconds) a node must be unavailable
                before triggering replacement. For transient failures, this
                should be longer than the typical recovery time to avoid
                unnecessary replacements.
            default_node_config: Config template for replacement nodes.
                If None, the failed node's own config is used.
        """
        self.failure_timeout = failure_timeout
        self.default_node_config = default_node_config
        self._spawn_counter = 0
        self._pending_spawns: set[str] = set()
        # Nodes currently being tracked for potential replacement
        self._timeout_pending: set[str] = set()

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        actions: list[Action] = []

        if event.event_type == EventType.NODE_FAILURE:
            actions.extend(self._handle_failure(event, cluster))

        elif event.event_type == EventType.NODE_RECOVERY:
            actions.extend(self._handle_recovery(event, cluster))

        elif event.event_type == EventType.NODE_DATA_LOSS:
            actions.extend(self._handle_data_loss(event, cluster, rng, protocol))

        elif event.event_type == EventType.NODE_REPLACEMENT_TIMEOUT:
            actions.extend(self._handle_replacement_timeout(event, cluster, rng, protocol))

        elif event.event_type == EventType.NODE_SPAWN_COMPLETE:
            node_id = event.metadata.get("node_id", event.target_id)
            self._pending_spawns.discard(node_id)

        elif event.event_type == EventType.NETWORK_OUTAGE_START:
            actions.extend(self._handle_network_outage_start(event, cluster))

        elif event.event_type == EventType.NETWORK_OUTAGE_END:
            actions.extend(self._handle_network_outage_end(event, cluster))

        return actions

    def _handle_failure(
        self,
        event: Event,
        cluster: ClusterState,
    ) -> list[Action]:
        """Handle a transient node failure by scheduling a replacement timeout."""
        node = cluster.get_node(event.target_id)
        if node is None or not node.has_data:
            return []  # Already lost or removed

        self._timeout_pending.add(event.target_id)
        return [
            Action(
                ActionType.SCHEDULE_REPLACEMENT_CHECK,
                {"node_id": event.target_id, "timeout": self.failure_timeout},
            )
        ]

    def _handle_recovery(
        self,
        event: Event,
        cluster: ClusterState,
    ) -> list[Action]:
        """Handle node recovery: cancel replacement timeout, start sync if lagging."""
        actions: list[Action] = []

        # Cancel any pending replacement timeout for this node
        if event.target_id in self._timeout_pending:
            self._timeout_pending.discard(event.target_id)
            actions.append(
                Action(
                    ActionType.CANCEL_REPLACEMENT_CHECK,
                    {"node_id": event.target_id},
                )
            )

        # If node is lagging, start sync
        node = cluster.get_node(event.target_id)
        if node and not node.is_up_to_date(cluster.commit_index):
            actions.append(Action(ActionType.START_SYNC, {"node_id": node.node_id}))

        return actions

    def _handle_data_loss(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        """Handle permanent data loss: immediately replace if cluster can commit."""
        # Cancel any pending replacement timeout (data loss is handled immediately)
        self._timeout_pending.discard(event.target_id)

        # Only replace if the cluster can still commit
        if not protocol.can_commit(cluster):
            return []

        return self._replace_node(event.target_id, cluster)

    def _handle_replacement_timeout(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        """Handle replacement timeout: replace node if still unavailable and cluster can commit."""
        node_id = event.target_id
        self._timeout_pending.discard(node_id)

        node = cluster.get_node(node_id)
        if node is None:
            return []  # Node was already removed

        # Check if node is still unavailable
        if cluster._node_effectively_available(node):
            return []  # Node recovered, no replacement needed

        # Check if cluster can still commit (needed for membership change)
        if not protocol.can_commit(cluster):
            return []

        return self._replace_node(node_id, cluster)

    def _handle_network_outage_start(
        self,
        event: Event,
        cluster: ClusterState,
    ) -> list[Action]:
        """Handle network outage: schedule replacement timeouts for affected nodes."""
        region = event.metadata.get("region", event.target_id)
        actions: list[Action] = []

        for node in cluster.nodes.values():
            if node.config.region == region and node.node_id not in self._timeout_pending:
                self._timeout_pending.add(node.node_id)
                actions.append(
                    Action(
                        ActionType.SCHEDULE_REPLACEMENT_CHECK,
                        {"node_id": node.node_id, "timeout": self.failure_timeout},
                    )
                )

        return actions

    def _handle_network_outage_end(
        self,
        event: Event,
        cluster: ClusterState,
    ) -> list[Action]:
        """Handle network recovery: cancel replacement timeouts for recovered nodes."""
        region = event.metadata.get("region", event.target_id)
        actions: list[Action] = []

        for node in cluster.nodes.values():
            if node.config.region == region and node.node_id in self._timeout_pending:
                self._timeout_pending.discard(node.node_id)
                actions.append(
                    Action(
                        ActionType.CANCEL_REPLACEMENT_CHECK,
                        {"node_id": node.node_id},
                    )
                )

        return actions

    def _replace_node(
        self,
        failed_node_id: str,
        cluster: ClusterState,
    ) -> list[Action]:
        """Remove a failed node and spawn a replacement.

        The failed node is removed first to keep the quorum denominator
        stable during the transition period while the replacement syncs.
        """
        failed_node = cluster.get_node(failed_node_id)
        if failed_node is None:
            return []

        # Use provided default config, or copy the failed node's config
        node_config = self.default_node_config or failed_node.config

        self._spawn_counter += 1
        new_node_id = f"replacement_{self._spawn_counter}"
        self._pending_spawns.add(new_node_id)

        return [
            # Remove the failed node to keep quorum stable
            Action(ActionType.REMOVE_NODE, {"node_id": failed_node_id}),
            # Spawn the replacement
            Action(
                ActionType.SPAWN_NODE,
                {"node_config": node_config, "node_id": new_node_id},
            ),
        ]

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        """Ensure we have target number of nodes at start."""
        actions: list[Action] = []

        while len(cluster.nodes) + len(self._pending_spawns) < cluster.target_cluster_size:
            self._spawn_counter += 1
            node_id = f"replacement_{self._spawn_counter}"
            self._pending_spawns.add(node_id)
            node_config = self.default_node_config
            if node_config is None:
                # Use config from an existing node
                existing = next(iter(cluster.nodes.values()), None)
                if existing:
                    node_config = existing.config
                else:
                    break  # No config to use
            actions.append(
                Action(
                    ActionType.SPAWN_NODE,
                    {"node_config": node_config, "node_id": node_id},
                )
            )

        return actions
