"""
Network model for the Monte Carlo RSM simulator.

Models region-level network outages: when a region's network is down, all
nodes in that region become unavailable until the network recovers.
"""

from dataclasses import dataclass, field

from .distributions import Distribution


#: Pseudo-region used as the target of global (all-regions) outage events.
GLOBAL_REGION = "__global__"


@dataclass
class NetworkConfig:
    """Configuration for network outage behavior.

    Attributes:
        outage_dist: Distribution for time (seconds) until next outage occurs
            (default for every region).
        outage_duration_dist: Distribution for duration (seconds) of each outage.
        regions: Region names that can experience full outages.
        per_region: Optional per-region overrides, ``{region: (outage_dist,
            duration_dist or None)}``; ``None`` keeps the default duration.
        global_outage_dist: Optional time until the next *global* outage, which
            takes every region in ``regions`` down at once (correlated process).
        global_duration_dist: Duration of global outages (defaults to
            ``outage_duration_dist``).
    """

    outage_dist: Distribution
    outage_duration_dist: Distribution
    regions: list[str] = field(default_factory=list)
    per_region: dict = field(default_factory=dict)
    global_outage_dist: Distribution | None = None
    global_duration_dist: Distribution | None = None

    def outage_for(self, region: str) -> Distribution:
        return self.per_region.get(region, (self.outage_dist, None))[0]

    def duration_for(self, region: str) -> Distribution:
        if region == GLOBAL_REGION:
            return self.global_duration_dist or self.outage_duration_dist
        dur = self.per_region.get(region, (None, None))[1]
        return dur or self.outage_duration_dist


@dataclass
class NetworkState:
    """Dynamic state of the network during simulation.

    When a region is in active_outages, all nodes in that region are
    considered unavailable (e.g. for quorum) until the outage ends.

    Attributes:
        active_outages: Set of region names currently experiencing network outage.
    """

    active_outages: set[str] = field(default_factory=set)
    #: Overlapping outages per region (a global outage can overlap a regional
    #: one); a region is down while its count is positive.
    outage_counts: dict = field(default_factory=dict)

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
        """Record a new network outage for a region (outages may overlap)."""
        self.outage_counts[region] = self.outage_counts.get(region, 0) + 1
        self.active_outages.add(region)

    def remove_outage(self, region: str) -> None:
        """Record end of one network outage for a region."""
        n = self.outage_counts.get(region, 1) - 1
        if n <= 0:
            self.outage_counts.pop(region, None)
            self.active_outages.discard(region)
        else:
            self.outage_counts[region] = n

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
