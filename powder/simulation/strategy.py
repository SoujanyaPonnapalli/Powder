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
    SCHEDULE_RECONFIGURATION = "schedule_reconfiguration"  # Schedule a cluster reconfiguration event
    PROMOTE_NODE = "promote_node"  # Promote a node from standby to active
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


class NodeReplacementStrategy(ClusterStrategy):
    """Strategy that replaces nodes after a configurable failure timeout.

    This models the standard production protocol for handling node failures
    in replicated state machines:

    1. A node becomes unavailable (transient failure, network outage, or data loss).
    2. A **failure timeout** countdown begins.
    3. If the node recovers before the timeout, the countdown is cancelled.
    4. If the timeout fires and the node is still unavailable:
       - Spawn a replacement node (additive replacement).
       - Maintain the cluster size by removing the least desirable node (usually the failed one),
         but ensuring the cluster allows the removal (e.g. quorum safety).

    For data loss, we treat it as a generic failure and wait for the timeout
    rather than reacting immediately.

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
        safe_mode: bool = True,
    ):
        """Initialize the node replacement strategy.

        Args:
            failure_timeout: Duration (seconds) a node must be unavailable
                before triggering replacement. For transient failures, this
                should be longer than the typical recovery time to avoid
                unnecessary replacements.
            default_node_config: Config template for replacement nodes.
                If None, the failed node's own config is used.
            safe_mode: Whether to use safe mode for replacement.
                If True, the cluster must be able to commit before replacing a node.
                If False, the cluster can replace a node even if it cannot commit.
        """
        self.failure_timeout = failure_timeout
        self.default_node_config = default_node_config
        self._spawn_counter = 0
        self._pending_spawns: set[str] = set()
        # Nodes currently being tracked for potential replacement
        self._timeout_pending: set[str] = set()
        self.safe_mode = safe_mode

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        actions: list[Action] = []

        if event.event_type == EventType.NODE_RECOVERY:
            node = cluster.get_node(event.target_id)
            if node and not node.is_up_to_date(cluster.commit_index):
                actions.append(Action(ActionType.START_SYNC, {"node_id": node.node_id}))

        elif event.event_type == EventType.NODE_REPLACEMENT_TIMEOUT:
            node_id = event.target_id
            # Leave node_id in _timeout_pending so we don't spawn duplicate replacements
            # self._timeout_pending.discard(node_id)
            node = cluster.get_node(node_id)
            if node is not None and not cluster._node_effectively_available(node):
                actions.extend(self._replace_node(node_id, cluster))

        elif event.event_type == EventType.NODE_SPAWN_COMPLETE:
            node_id = event.metadata.get("node_id", event.target_id)
            self._pending_spawns.discard(node_id)
            
        # Only reassess timeouts on events that can change node availability
        _AVAILABILITY_EVENTS = (
            EventType.NODE_FAILURE,
            EventType.NODE_RECOVERY,
            EventType.NODE_DATA_LOSS,
            EventType.NETWORK_OUTAGE_START,
            EventType.NETWORK_OUTAGE_END,
            EventType.NODE_SPAWN_COMPLETE,
        )
        if event.event_type in _AVAILABILITY_EVENTS:
            # Compute num_available once for reuse by multiple methods below
            nav = cluster.num_available()
            actions.extend(self._reassess_timeouts(cluster, num_available=nav))

        # Check if we can promote any synced standby nodes
        actions.extend(self._promote_eligible_standbys(cluster, protocol))

        # Always check cluster size boundaries
        actions.extend(self._maintain_cluster_size(cluster, protocol))

        return actions

    def _reassess_timeouts(self, cluster: ClusterState, num_available: int | None = None) -> list[Action]:
        """Cancel timeouts if totally unavailable, or schedule them appropriately."""
        actions: list[Action] = []
        if (num_available if num_available is not None else cluster.num_available()) == 0:
            for pending_id in list(self._timeout_pending):
                actions.append(Action(ActionType.CANCEL_REPLACEMENT_CHECK, {"node_id": pending_id}))
            self._timeout_pending.clear()
        else:
            # Cancel timeouts for nodes that have recovered or no longer exist
            for pending_id in list(self._timeout_pending):
                node = cluster.get_node(pending_id)
                if node is None or cluster._node_effectively_available(node):
                    self._timeout_pending.discard(pending_id)
                    actions.append(Action(ActionType.CANCEL_REPLACEMENT_CHECK, {"node_id": pending_id}))

            # Schedule timeouts for nodes that are unavailable
            for n in cluster.nodes.values():
                if not cluster._node_effectively_available(n) and n.node_id not in self._timeout_pending:
                    self._timeout_pending.add(n.node_id)
                    actions.append(
                        Action(
                            ActionType.SCHEDULE_REPLACEMENT_CHECK,
                            {"node_id": n.node_id, "timeout": self.failure_timeout},
                        )
                    )
        return actions

    def _promote_eligible_standbys(
        self, cluster: ClusterState, protocol: Protocol, can_commit: bool | None = None,
    ) -> list[Action]:
        """Promote synced standby nodes if protocol safely permits."""
        actions: list[Action] = []
        for node_id, node in cluster.standby_nodes.items():
            if node.is_up_to_date(cluster.commit_index):
                if not self.safe_mode or (can_commit if can_commit is not None else protocol.can_commit(cluster)):
                    actions.append(Action(ActionType.PROMOTE_NODE, {"node_id": node_id}))
        return actions

    def _replace_node(
        self,
        failed_node_id: str,
        cluster: ClusterState,
    ) -> list[Action]:
        """Spawn a replacement node without removing the failed node immediately."""
        failed_node = cluster.get_node(failed_node_id)
        if failed_node is None:
            return []

        # Use provided default config, or copy the failed node's config
        node_config = self.default_node_config or failed_node.config

        self._spawn_counter += 1
        new_node_id = f"replacement_{self._spawn_counter}"
        self._pending_spawns.add(new_node_id)

        # Purely additive replacement, deploy to standby first
        return [
            Action(
                ActionType.SPAWN_NODE,
                {"node_config": node_config, "node_id": new_node_id, "standby": True},
            ),
        ]

    def _maintain_cluster_size(
        self,
        cluster: ClusterState,
        protocol: Protocol,
    ) -> list[Action]:
        """Remove extra nodes if the cluster is over-provisioned."""
        target = cluster.target_cluster_size
        nodes = list(cluster.nodes.values())
        
        if len(nodes) <= target:
            return []

        # We have more nodes than target. Identify victims.
        # Sort key: (is_available, is_up_to_date, last_applied_index)
        # We want to KEEP the best nodes (highest sort key).
        
        def node_score(n: NodeState):
            is_avail = cluster._node_effectively_available(n)
            is_updated = n.is_up_to_date(cluster.commit_index)
            return (is_avail, is_updated, n.last_applied_index)

        sorted_nodes = sorted(nodes, key=node_score, reverse=True)
        
        # Keep the top `target` nodes. The rest are candidates for removal.
        candidates = sorted_nodes[target:]
        
        actions = []
        for node in candidates:
            # Use safe_mode check: ensure protocol allows removal
            if self.safe_mode:
                 pass # Approximation: if we keep `target` best nodes, we assume we are safe?
            
            actions.append(Action(ActionType.REMOVE_NODE, {"node_id": node.node_id}))

        return actions

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


class AdaptiveReplacementStrategy(NodeReplacementStrategy):
    """Adaptive strategy that scales down during failures to maintain quorum.

    This strategy extends NodeReplacementStrategy by dynamically reducing the
    target cluster size when multiple nodes fail, effectively maintaining
    availability for the remaining nodes. It automatically scales the target
    size back up as nodes recover or replacements arrive.
    
    Attributes:
        reconfiguration_dist: Time it takes to reconfigure the cluster.
        external_consensus: Allow scaling down to 2 or 1 nodes.
        scale_down_threshold: Number of unavailable nodes that triggers scale down.
    """

    def __init__(
        self,
        failure_timeout: Seconds,
        reconfiguration_dist: Seconds,
        scale_down_threshold: int = 2,
        external_consensus: bool = False,
        default_node_config: NodeConfig | None = None,
        safe_mode: bool = True,
    ):
        super().__init__(failure_timeout, default_node_config, safe_mode)
        self.reconfiguration_dist = reconfiguration_dist
        self.scale_down_threshold = scale_down_threshold
        self.external_consensus = external_consensus
        self._pending_reconfigurations: set[int] = set()
        self.max_target_cluster_size = 0  # To be set on simulation start

    def on_simulation_start(
        self,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Action]:
        self.max_target_cluster_size = cluster.target_cluster_size
        return super().on_simulation_start(cluster, rng)

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol: Protocol,
    ) -> list[Action]:
        # Delegate standard event handling to super class
        actions = super().on_event(event, cluster, rng, protocol)
        
        # Intercept events relevant to scaling checks
        should_check_scaling = False
        
        if event.event_type in [
            EventType.NODE_FAILURE, 
            EventType.NODE_DATA_LOSS, 
            EventType.NODE_RECOVERY,
            EventType.NODE_FAILURE, 
            EventType.NODE_DATA_LOSS, 
            EventType.NODE_RECOVERY,
            EventType.NODE_SPAWN_COMPLETE,
        ]:
            should_check_scaling = True
            
        # Specific handling for reconfiguration event
        if event.event_type == EventType.CLUSTER_RECONFIGURATION:
             actions.extend(self._handle_reconfiguration(event, cluster, protocol))

        if should_check_scaling:
            actions.extend(self._check_and_schedule_reconfiguration(cluster))
            
        return actions

    def _maintain_cluster_size(
        self,
        cluster: ClusterState,
        protocol: Protocol,
    ) -> list[Action]:
        """Override to cleanup based on max_target_cluster_size (provisioning target)."""
        # We want to keep up to max_target_cluster_size nodes physically,
        # even if target_cluster_size (logical/quorum target) is lower.
        target = self.max_target_cluster_size
        nodes = list(cluster.nodes.values())
        
        if len(nodes) <= target:
            return []

        # Copied logic from parent but using max_target
        def node_score(n: NodeState):
            is_avail = cluster._node_effectively_available(n)
            is_updated = n.is_up_to_date(cluster.commit_index)
            return (is_avail, is_updated, n.last_applied_index)

        sorted_nodes = sorted(nodes, key=node_score, reverse=True)
        candidates = sorted_nodes[target:]
        
        actions = []
        for node in candidates:
             actions.append(Action(ActionType.REMOVE_NODE, {"node_id": node.node_id}))

        return actions

    def _check_and_schedule_reconfiguration(self, cluster: ClusterState) -> list[Action]:
        """Check for scaling opportunities based on deficits."""
        current_target = cluster.target_cluster_size
        new_target = current_target
        
        # Calculate Deficit: Target - Available
        deficit = current_target - cluster.num_available()
        
        # 1. SCALE DOWN
        if deficit >= self.scale_down_threshold:
            if self.external_consensus:
                if current_target > 1:
                    new_target = current_target - 1
            else:
                 if current_target >= 5:
                     new_target = current_target - 2
        
        # 2. SCALE UP
        # If we are below max target, we try to scale up if we have enough available nodes
        if current_target < self.max_target_cluster_size:
             potential_target = current_target
             if self.external_consensus:
                 potential_target = current_target + 1
             else:
                 potential_target = current_target + 2
             
             # If we have enough available nodes to satisfy the NEW target
             if cluster.num_available() >= potential_target:
                 new_target = potential_target
        
        if new_target != current_target:
             if new_target not in self._pending_reconfigurations:
                 self._pending_reconfigurations.add(new_target)
                 return [
                     Action(
                         ActionType.SCHEDULE_RECONFIGURATION, 
                         {"delay": self.reconfiguration_dist, "target_size": new_target}
                     )
                 ]
        
        return []

    def _handle_reconfiguration(
        self,
        event: Event,
        cluster: ClusterState,
        protocol: Protocol,
    ) -> list[Action]:
        target_size = event.metadata.get("target_size")
        if target_size:
             self._pending_reconfigurations.discard(target_size)
        
        if not target_size or target_size == cluster.target_cluster_size:
            return []

        if not target_size or target_size == cluster.target_cluster_size:
            return []

        # Check if we can commit in the CURRENT configuration
        # If external_consensus is True, we bypass this check (allowing risky scale down)
        if not self.external_consensus and not protocol.can_commit(cluster):
            # Failed to reconfigure
            return []

        actions = []
        if target_size < cluster.target_cluster_size:
             # Verify we still need to scale down (deficit check)
             deficit = cluster.target_cluster_size - cluster.num_available()
             # Ideally deficit should match threshold logic, but simpler: 
             # if we are fully available, why scale down?
             # Exception: if we want to confirm the down-scale for consistency?
             # But if everything recovered, we shouldn't scale down.
             # Strict check: deficit >= 1 at least?
             if deficit == 0:
                  return []
             
             actions.append(Action(ActionType.SCALE_DOWN, {"new_size": target_size}))

        else:
             actions.append(Action(ActionType.SCALE_UP, {"new_size": target_size}))
             
        actions.extend(self._check_next_step(cluster, target_size))
        return actions

    def _check_next_step(self, cluster: ClusterState, current_target: int) -> list[Action]:
        """Check for next reconfiguration step assuming current_target is active."""
        new_target = current_target
        deficit = current_target - cluster.num_available()
        
        if deficit >= self.scale_down_threshold:
            if self.external_consensus:
                 if current_target > 1:
                     new_target = current_target - 1
            else:
                if current_target >= 5:
                     new_target = current_target - 2
        
        # Scale Up Logic
        if current_target < self.max_target_cluster_size:
             potential_target = current_target
             if self.external_consensus:
                 potential_target = current_target + 1
             else:
                 potential_target = current_target + 2
             
             if cluster.num_available() >= potential_target:
                 new_target = potential_target
        
        if new_target != current_target and new_target not in self._pending_reconfigurations:
             self._pending_reconfigurations.add(new_target)
             return [
                 Action(
                     ActionType.SCHEDULE_RECONFIGURATION, 
                     {"delay": self.reconfiguration_dist, "target_size": new_target}
                 )
             ]
        return []
