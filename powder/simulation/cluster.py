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
    standby_nodes: dict[str, NodeState] = field(default_factory=dict)

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
        all_nodes = list(self.nodes.values()) + list(self.standby_nodes.values())
        available = [n for n in all_nodes if self._node_effectively_available(n)]
        if not available:
            return None
        return min(available, key=lambda n: n.last_applied_index)

    def find_sync_donor(self, node: NodeState) -> NodeState | None:
        """Find the best available donor for a lagging node to sync from.

        Returns the effectively available node (excluding *node*) with the
        highest ``last_applied_index``.  The syncing node can only obtain
        data that the donor actually has, so the donor's position -- not
        ``commit_index`` -- determines what data is available.

        Args:
            node: The lagging node that needs a donor.

        Returns:
            The best donor NodeState, or None if no donor is available.
        """
        donor = None
        all_nodes = list(self.nodes.values()) + list(self.standby_nodes.values())
        for n in all_nodes:
            if n.node_id != node.node_id and self._node_effectively_available(n):
                if donor is None or n.last_applied_index > donor.last_applied_index:
                    donor = n
        return donor

    def nodes_needing_sync(self) -> list[NodeState]:
        """Return nodes that are lagging, available, and have no active sync.

        These are candidates for ``_retry_pending_syncs`` -- they fell behind
        and either never had a sync scheduled or had one cancelled (e.g. donor
        went down with no replacement).

        Returns:
            List of nodes that need a sync to be started.
        """
        all_nodes = list(self.nodes.values()) + list(self.standby_nodes.values())
        return [
            n
            for n in all_nodes
            if (
                self._node_effectively_available(n)
                and not n.is_up_to_date(self.commit_index)
                and n.sync is None
            )
        ]

    def nodes_syncing_from(self, donor_id: str) -> list[NodeState]:
        """Return nodes currently syncing from a specific donor.

        Used to cancel or failover syncs when a donor becomes unavailable.

        Args:
            donor_id: Node ID of the donor.

        Returns:
            List of nodes whose active sync references *donor_id*.
        """
        all_nodes = list(self.nodes.values()) + list(self.standby_nodes.values())
        return [
            n
            for n in all_nodes
            if n.sync is not None and n.sync.donor_id == donor_id
        ]

    def most_up_to_date_node(self) -> NodeState | None:
        """Return the node with the most recent data.

        Useful for determining what data is available for recovery.
        Only considers effectively available nodes (not in a region outage).
        """
        all_nodes = list(self.nodes.values()) + list(self.standby_nodes.values())
        available = [n for n in all_nodes if self._node_effectively_available(n)]
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
        return self.nodes.get(node_id) or self.standby_nodes.get(node_id)

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

    def add_standby_node(self, node: NodeState) -> None:
        """Add a node to the standby set (active but not part of quorum).

        Args:
            node: Node being added to standby.
        """
        self.standby_nodes[node.node_id] = node

    def remove_standby_node(self, node_id: str) -> NodeState | None:
        """Remove a node from the standby set.

        Args:
            node_id: ID of node to remove.

        Returns:
            The removed node, or None if not found.
        """
        return self.standby_nodes.pop(node_id, None)

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
        standby_billable = [n for n in self.standby_nodes.values() if n.has_data]
        return active_billable + provisioning_billable + standby_billable

    def __repr__(self) -> str:
        up_to_date = self.num_up_to_date()
        available = self.num_available()
        total = len(self.nodes)
        return (
            f"ClusterState(t={self.current_time:.1f}s, "
            f"commit_index={self.commit_index:.1f}, "
            f"{up_to_date}/{available}/{total} up-to-date/available/total)"
        )
