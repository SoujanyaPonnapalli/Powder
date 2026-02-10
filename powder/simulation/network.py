"""
Network model for the Monte Carlo RSM simulator.

Models region-level network outages: when a region's network is down, all
nodes in that region become unavailable until the network recovers.
"""

from dataclasses import dataclass, field

from .distributions import Distribution


@dataclass
class NetworkConfig:
    """Configuration for network outage behavior.

    Attributes:
        outage_dist: Distribution for time (seconds) until next outage occurs.
        outage_duration_dist: Distribution for duration (seconds) of each outage.
        affected_regions: List of region names that can experience full outages.
    """

    outage_dist: Distribution
    outage_duration_dist: Distribution
    regions: list[str] = field(default_factory=list)


@dataclass
class NetworkState:
    """Dynamic state of the network during simulation.

    When a region is in active_outages, all nodes in that region are
    considered unavailable (e.g. for quorum) until the outage ends.

    Attributes:
        active_outages: Set of region names currently experiencing network outage.
    """

    active_outages: set[str] = field(default_factory=set)

    def is_region_down(self, region: str) -> bool:
        """Check if a region's network is currently down.

        Args:
            region: Region name.

        Returns:
            True if the region is in an active outage.
        """
        return region in self.active_outages

    def is_partitioned(self, region_a: str, region_b: str) -> bool:
        """Check if two regions cannot communicate.

        Regions cannot communicate if either region's network is down.

        Args:
            region_a: First region name.
            region_b: Second region name.

        Returns:
            True if there is an active outage affecting either region.
        """
        return self.is_region_down(region_a) or self.is_region_down(region_b)

    def add_outage(self, region: str) -> None:
        """Record a new network outage for a region."""
        self.active_outages.add(region)

    def remove_outage(self, region: str) -> None:
        """Record end of a network outage for a region."""
        self.active_outages.discard(region)

    def regions_reachable_from(self, region: str, all_regions: set[str]) -> set[str]:
        """Get all regions reachable from the given region.

        If the starting region is down, nothing is reachable. Otherwise,
        all regions that are not currently down are reachable.

        Args:
            region: Starting region.
            all_regions: Set of all known regions.

        Returns:
            Set of regions reachable from the starting region (includes itself
            when not down).
        """
        if self.is_region_down(region):
            return set()
        return {r for r in all_regions if not self.is_region_down(r)}

    def __repr__(self) -> str:
        if not self.active_outages:
            return "NetworkState(no outages)"
        outages = ", ".join(sorted(self.active_outages))
        return f"NetworkState(outages: {outages})"
