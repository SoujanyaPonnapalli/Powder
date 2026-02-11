"""
Cluster state management for the Monte Carlo RSM simulator.

Manages the collection of nodes and network state, providing methods
for quorum checks, data loss detection, and cluster queries.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from powder.simulation.node import NodeState

from .distributions import Seconds
from .network import NetworkState
from .node import NodeState


@dataclass
class ClusterState:
    """Complete state of an RSM cluster during simulation.

    Attributes:
        nodes: Dictionary mapping node_id to NodeState.
        network: Current network partition state.
        target_cluster_size: Desired number of nodes in the cluster.
        current_time: Current wall-clock simulation time in seconds.
        commit_index: Current position in the committed data stream.
            Advances only when the system can commit, at the protocol's
            commit_rate. Raw float, not a wall-clock time.
        provisioning_nodes: Nodes being provisioned (not yet active).
            These nodes are tracked for cost billing but do not participate
            in quorum calculations or availability checks.
    """

    nodes: dict[str, NodeState]
    network: NetworkState
    target_cluster_size: int
    current_time: Seconds = field(default_factory=lambda: Seconds(0))
    commit_index: float = 0.0
    provisioning_nodes: dict[str, NodeState] = field(default_factory=dict)

    def _node_effectively_available(self, node: NodeState) -> bool:
        """True if node is available and not in a region with an active network outage."""
        return (
            node.is_available
            and node.has_data
            and not self.network.is_region_down(node.config.region)
        )

    def num_up_to_date(self) -> int:
        """Count nodes that are fully synced to commit_index.

        Only counts nodes that are available, have data, are up-to-date,
        and not in a region with an active network outage.

        Returns:
            Number of up-to-date nodes.
        """
        return sum(
            1
            for n in self.nodes.values()
            if n.is_up_to_date(self.commit_index) and self._node_effectively_available(n)
        )

    def num_available(self) -> int:
        """Count nodes that are available (may be lagging).

        Excludes nodes in regions with an active network outage.

        Returns:
            Number of available nodes with data.
        """
        return sum(1 for n in self.nodes.values() if self._node_effectively_available(n))

    def num_with_data(self) -> int:
        """Count nodes that have data (available or not).

        Returns:
            Number of nodes that haven't lost data.
        """
        return sum(1 for n in self.nodes.values() if n.has_data)

    def nodes_by_region(self) -> dict[str, list[NodeState]]:
        """Group nodes by their region.

        Returns:
            Dictionary mapping region name to list of nodes in that region.
        """
        by_region: dict[str, list[NodeState]] = defaultdict(list)
        for node in self.nodes.values():
            by_region[node.config.region].append(node)
        return dict(by_region)

    def all_regions(self) -> set[str]:
        """Get all regions that have nodes.

        Returns:
            Set of region names.
        """
        return {n.config.region for n in self.nodes.values()}

    def most_lagging_node(self) -> NodeState | None:
        """Return the available node furthest behind in sync.

        Useful for prioritizing which nodes to sync first.
        Excludes nodes in regions with an active network outage.

        Returns:
            The most lagging available node, or None if no available nodes.
        """
        available = [n for n in self.nodes.values() if self._node_effectively_available(n)]
        if not available:
            return None
        return min(available, key=lambda n: n.last_applied_index)

    def most_up_to_date_node(self) -> NodeState | None:
        """Return the node with the most recent data.

        Useful for determining what data is available for recovery.
        Only considers effectively available nodes (not in a region outage).
        """
        available = [n for n in self.nodes.values() if self._node_effectively_available(n)]
        if not available:
            return None
        return max(available, key=lambda n: n.last_applied_index)

    def get_node(self, node_id: str) -> NodeState | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier.

        Returns:
            NodeState if found, None otherwise.
        """
        return self.nodes.get(node_id)

    def add_node(self, node: NodeState) -> None:
        """Add a node to the cluster.

        Args:
            node: Node to add.
        """
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> NodeState | None:
        """Remove a node from the cluster.

        Args:
            node_id: ID of node to remove.

        Returns:
            The removed node, or None if not found.
        """
        return self.nodes.pop(node_id, None)

    def add_provisioning_node(self, node: NodeState) -> None:
        """Add a node to the provisioning set (not yet active).

        Provisioning nodes are tracked for cost billing but do not
        participate in quorum calculations or availability checks.

        Args:
            node: Node being provisioned.
        """
        self.provisioning_nodes[node.node_id] = node

    def remove_provisioning_node(self, node_id: str) -> NodeState | None:
        """Remove a node from the provisioning set.

        Args:
            node_id: ID of node to remove.

        Returns:
            The removed node, or None if not found.
        """
        return self.provisioning_nodes.pop(node_id, None)

    def all_nodes_for_billing(self) -> list[NodeState]:
        """Return all nodes that should be billed, including provisioning.

        Only bills nodes that have not incurred data loss. In cloud environments,
        you pay for VMs from launch (including provisioning), through failures,
        until explicit termination or data loss. Nodes with data loss are not billed.

        Returns:
            List of all active and provisioning nodes that have not lost data.
        """
        active_billable = [n for n in self.nodes.values() if n.has_data]
        provisioning_billable = [n for n in self.provisioning_nodes.values() if n.has_data]
        return active_billable + provisioning_billable

    def __repr__(self) -> str:
        up_to_date = self.num_up_to_date()
        available = self.num_available()
        total = len(self.nodes)
        return (
            f"ClusterState(t={self.current_time:.1f}s, "
            f"commit_index={self.commit_index:.1f}, "
            f"{up_to_date}/{available}/{total} up-to-date/available/total)"
        )
