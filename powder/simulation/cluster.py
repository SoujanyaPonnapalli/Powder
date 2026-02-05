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
        commit_rate: Seconds of new committed data per second of real time.
    """

    nodes: dict[str, NodeState]
    network: NetworkState
    target_cluster_size: int
    current_time: Seconds = field(default_factory=lambda: Seconds(0))
    commit_rate: float = 1.0  # 1 second of data committed per second of real time

    def num_up_to_date(self) -> int:
        """Count nodes that are fully synced to current_time.

        Only counts nodes that are available, have data, and are up-to-date.

        Returns:
            Number of up-to-date nodes.
        """
        return sum(
            1
            for n in self.nodes.values()
            if n.is_up_to_date(self.current_time) and n.is_available and n.has_data
        )

    def num_available(self) -> int:
        """Count nodes that are available (may be lagging).

        Returns:
            Number of available nodes with data.
        """
        return sum(1 for n in self.nodes.values() if n.is_available and n.has_data)

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
        """Check if a quorum of up-to-date nodes exists that can communicate.

        The system can commit new requests if there exists a connected
        component (accounting for network partitions) where a majority
        of nodes are up-to-date, available, and have data.

        Returns:
            True if the system can accept new commits.
        """
        # Get up-to-date nodes
        up_to_date_nodes = [
            n for n in self.nodes.values()
            if n.is_up_to_date(self.current_time) and n.is_available and n.has_data
        ]

        if len(up_to_date_nodes) < self.quorum_size():
            return False  # Not enough up-to-date nodes globally

        # If no network partitions, simple check suffices
        if not self.network.active_outages:
            return True

        # Check if any connected component has a quorum
        return self._has_quorum_in_connected_component(up_to_date_nodes)

    def _has_quorum_in_connected_component(self, nodes: list["NodeState"]) -> bool:
        """Check if any connected component of nodes forms a quorum.

        Args:
            nodes: List of candidate nodes (should be up-to-date and available).

        Returns:
            True if any connected component has >= quorum_size nodes.
        """
        if not nodes:
            return False

        all_regions = self.all_regions()
        quorum = self.quorum_size()

        # Group nodes by region
        nodes_by_region: dict[str, list["NodeState"]] = {}
        for node in nodes:
            region = node.config.region
            if region not in nodes_by_region:
                nodes_by_region[region] = []
            nodes_by_region[region].append(node)

        # Get connected components of regions
        components = self.network.get_connected_components(all_regions)

        # Check if any component has enough nodes
        for component in components:
            component_node_count = sum(
                len(nodes_by_region.get(region, []))
                for region in component
            )
            if component_node_count >= quorum:
                return True

        return False

    def has_potential_data_loss(self) -> bool:
        """Check if quorum is lost (potential data loss).

        When fewer than a quorum of available nodes can communicate,
        we can't be certain the remaining nodes have the latest committed data.
        This accounts for network partitions that may split available nodes.

        Returns:
            True if no connected component has a quorum of available nodes.
        """
        available_nodes = [
            n for n in self.nodes.values()
            if n.is_available and n.has_data
        ]

        if len(available_nodes) < self.quorum_size():
            return True  # Not enough available nodes globally

        # If no network partitions, simple check suffices
        if not self.network.active_outages:
            return False

        # Check if any connected component has a quorum of available nodes
        return not self._has_quorum_in_connected_component(available_nodes)

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

    def largest_reachable_quorum_size(self) -> int:
        """Get the size of the largest connected component of up-to-date nodes.

        Useful for understanding how close we are to losing quorum due to
        network partitions.

        Returns:
            Number of up-to-date nodes in the largest connected component.
        """
        up_to_date_nodes = [
            n for n in self.nodes.values()
            if n.is_up_to_date(self.current_time) and n.is_available and n.has_data
        ]

        if not up_to_date_nodes:
            return 0

        if not self.network.active_outages:
            return len(up_to_date_nodes)

        all_regions = self.all_regions()
        components = self.network.get_connected_components(all_regions)

        # Group nodes by region
        nodes_by_region: dict[str, list["NodeState"]] = {}
        for node in up_to_date_nodes:
            region = node.config.region
            if region not in nodes_by_region:
                nodes_by_region[region] = []
            nodes_by_region[region].append(node)

        # Find the largest component
        max_size = 0
        for component in components:
            component_size = sum(
                len(nodes_by_region.get(region, []))
                for region in component
            )
            max_size = max(max_size, component_size)

        return max_size

    def is_network_partitioned(self) -> bool:
        """Check if there are any active network partitions.

        Returns:
            True if at least one region pair is partitioned.
        """
        return len(self.network.active_outages) > 0

    def most_lagging_node(self) -> NodeState | None:
        """Return the available node furthest behind in sync.

        Useful for prioritizing which nodes to sync first.

        Returns:
            The most lagging available node, or None if no available nodes.
        """
        available = [n for n in self.nodes.values() if n.is_available and n.has_data]
        if not available:
            return None
        return min(available, key=lambda n: n.last_up_to_date_time)

    def most_up_to_date_node(self) -> NodeState | None:
        """Return the node with the most recent data.

        Useful for determining what data is available for recovery.

        Returns:
            The most up-to-date node with data, or None if no nodes have data.
        """
        with_data = [n for n in self.nodes.values() if n.has_data]
        if not with_data:
            return None
        return max(with_data, key=lambda n: n.last_up_to_date_time)

    def best_reachable_snapshot_for_node(self, node: NodeState) -> tuple[NodeState, Seconds] | None:
        """Find the most recent snapshot reachable from a node.

        Nodes can only download snapshots from other nodes that they are not
        partitioned from via network outages.

        Args:
            node: The node looking for a snapshot.

        Returns:
            Tuple of (source_node, snapshot_time) for the best snapshot, or None.
        """
        node_region = node.config.region
        all_regions = self.all_regions()

        best_snapshot: tuple[NodeState, Seconds] | None = None

        for other in self.nodes.values():
            if other.node_id == node.node_id:
                continue
            if not other.is_available or not other.has_data:
                continue
            if other.snapshot_time is None:
                continue

            # Check if the two nodes can communicate (not partitioned)
            if not self.network.can_communicate(node_region, other.config.region, all_regions):
                continue

            # Found a reachable snapshot
            if best_snapshot is None or other.snapshot_time > best_snapshot[1]:
                best_snapshot = (other, other.snapshot_time)

        return best_snapshot

    def nodes_with_log_reachable_from(self, node: NodeState) -> list[NodeState]:
        """Find nodes with log data reachable from a node.

        Log entries can be downloaded from any available node that has data
        and is not partitioned from the requesting node.

        Args:
            node: The node looking for log sources.

        Returns:
            List of nodes that can provide log entries.
        """
        node_region = node.config.region
        all_regions = self.all_regions()

        sources = []
        for other in self.nodes.values():
            if other.node_id == node.node_id:
                continue
            if not other.is_available or not other.has_data:
                continue

            # Check if they can communicate
            if not self.network.can_communicate(node_region, other.config.region, all_regions):
                continue

            sources.append(other)

        return sources

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
        partition_info = ""
        if self.is_network_partitioned():
            largest = self.largest_reachable_quorum_size()
            partition_info = f", partitioned (largest_component={largest})"
        return (
            f"ClusterState(t={self.current_time:.1f}s, "
            f"{up_to_date}/{available}/{total} up-to-date/available/total, "
            f"{can_commit}{partition_info})"
        )
