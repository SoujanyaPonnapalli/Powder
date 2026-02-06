"""
Strategy interface and implementations for the Monte Carlo RSM simulator.

Strategies define how the cluster management system reacts to events
like node failures, recoveries, and data loss.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .cluster import ClusterState
from .distributions import Seconds
from .events import Event, EventType
from .node import NodeConfig, NodeState


class ActionType(Enum):
    """Types of actions a strategy can take."""

    SPAWN_NODE = "spawn_node"  # Start spawning a new node
    REMOVE_NODE = "remove_node"  # Remove a node from cluster
    SCALE_DOWN = "scale_down"  # Reduce target cluster size
    SCALE_UP = "scale_up"  # Increase target cluster size
    START_SYNC = "start_sync"  # Trigger a node to start syncing
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
    ) -> list[Action]:
        """React to a simulation event.

        Called by the simulator after each event is processed.
        The strategy can inspect the current cluster state and
        return actions to modify the cluster.

        Args:
            event: The event that just occurred.
            cluster: Current cluster state (after event was applied).
            rng: Random number generator for reproducibility.

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
        if node and not node.is_up_to_date(cluster.current_time):
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
