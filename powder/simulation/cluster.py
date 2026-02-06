"""
Cluster state management for the Monte Carlo RSM simulator.

Manages the collection of nodes and network state, providing methods
for quorum checks, data loss detection, and cluster queries.
"""

from collections import defaultdict
from dataclasses import dataclass, field

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
        current_time: Current simulation time in seconds.
    """

    nodes: dict[str, NodeState]
    network: NetworkState
    target_cluster_size: int
    current_time: Seconds = field(default_factory=lambda: Seconds(0))

    def _node_effectively_available(self, node: NodeState) -> bool:
        """True if node is available and not in a region with an active network outage."""
        return (
            node.is_available
            and node.has_data
            and not self.network.is_region_down(node.config.region)
        )

    def num_up_to_date(self) -> int:
        """Count nodes that are fully synced to current_time.

        Only counts nodes that are available, have data, are up-to-date,
        and not in a region with an active network outage.

        Returns:
            Number of up-to-date nodes.
        """
        return sum(
            1
            for n in self.nodes.values()
            if n.is_up_to_date(self.current_time) and self._node_effectively_available(n)
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

    def quorum_size(self) -> int:
        """Calculate the quorum size needed for commits.

        Returns:
            Minimum number of nodes needed for quorum (majority).
        """
        return len(self.nodes) // 2 + 1

    def can_commit(self) -> bool:
        """Check if quorum of up-to-date nodes exists.

        The system can commit new requests if a majority of nodes
        are up-to-date, available, and have data.

        Returns:
            True if the system can accept new commits.
        """
        return self.num_up_to_date() >= self.quorum_size()

    def has_potential_data_loss(self) -> bool:
        """Check if quorum is lost (potential data loss).

        When fewer than a quorum of nodes are available, we can't be
        certain the remaining nodes have the latest committed data.

        Returns:
            True if quorum is lost.
        """
        return self.num_available() < self.quorum_size()

    def has_actual_data_loss(self) -> bool:
        """Check if all up-to-date nodes have failed (definite data loss).

        Data is definitely lost when no nodes have the latest committed
        data, but some nodes still exist with older data.

        Returns:
            True if data is definitely lost.
        """
        if self.num_up_to_date() > 0:
            return False  # Still have up-to-date nodes

        # Check if any node has data but is not up-to-date
        for n in self.nodes.values():
            if n.has_data and not n.is_up_to_date(self.current_time):
                return True

        # If no nodes have data at all, that's also data loss
        return self.num_with_data() == 0 and len(self.nodes) > 0

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
        return min(available, key=lambda n: n.last_up_to_date_time)

    def most_up_to_date_node(self) -> NodeState | None:
        """Return the node with the most recent data.

        Useful for determining what data is available for recovery.
        Only considers effectively available nodes (not in a region outage).
        """
        available = [n for n in self.nodes.values() if self._node_effectively_available(n)]
        if not available:
            return None
        return max(available, key=lambda n: n.last_up_to_date_time)

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

    def __repr__(self) -> str:
        up_to_date = self.num_up_to_date()
        available = self.num_available()
        total = len(self.nodes)
        can_commit = "can_commit" if self.can_commit() else "cannot_commit"
        return (
            f"ClusterState(t={self.current_time:.1f}s, "
            f"{up_to_date}/{available}/{total} up-to-date/available/total, "
            f"{can_commit})"
        )
