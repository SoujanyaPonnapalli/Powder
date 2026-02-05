"""
Network model for the Monte Carlo RSM simulator.

Models network partitions between regions that can cause nodes to become
unreachable from each other.
"""

from dataclasses import dataclass, field

from .distributions import Distribution


@dataclass(frozen=True)
class RegionPair:
    """An unordered pair of regions that can experience network partitions.

    The pair is normalized so that region_a <= region_b alphabetically,
    ensuring consistent hashing and equality comparison.

    Attributes:
        region_a: First region (alphabetically smaller or equal).
        region_b: Second region (alphabetically larger or equal).
    """

    region_a: str
    region_b: str

    def __post_init__(self) -> None:
        # Normalize ordering for consistent comparison
        if self.region_a > self.region_b:
            # Use object.__setattr__ since dataclass is frozen
            # Must save values before swapping
            a, b = self.region_a, self.region_b
            object.__setattr__(self, "region_a", b)
            object.__setattr__(self, "region_b", a)

    def contains(self, region: str) -> bool:
        """Check if this pair includes the given region."""
        return region == self.region_a or region == self.region_b

    def other(self, region: str) -> str | None:
        """Get the other region in the pair, or None if region not in pair."""
        if region == self.region_a:
            return self.region_b
        if region == self.region_b:
            return self.region_a
        return None

    def __repr__(self) -> str:
        return f"RegionPair({self.region_a!r}, {self.region_b!r})"


def make_region_pair(a: str, b: str) -> RegionPair:
    """Create a RegionPair with automatic normalization."""
    return RegionPair(a, b)


@dataclass
class NetworkConfig:
    """Configuration for network partition behavior.

    Attributes:
        outage_dist: Distribution for time (seconds) until next outage occurs.
        outage_duration_dist: Distribution for duration (seconds) of each outage.
        affected_regions: List of region pairs that can experience outages.
    """

    outage_dist: Distribution
    outage_duration_dist: Distribution
    affected_regions: list[RegionPair] = field(default_factory=list)


@dataclass
class NetworkState:
    """Dynamic state of the network during simulation.

    Attributes:
        active_outages: Set of region pairs currently experiencing partitions.
    """

    active_outages: set[RegionPair] = field(default_factory=set)

    def is_partitioned(self, region_a: str, region_b: str) -> bool:
        """Check if two regions are currently partitioned from each other.

        Args:
            region_a: First region name.
            region_b: Second region name.

        Returns:
            True if there is an active partition between the regions.
        """
        pair = RegionPair(region_a, region_b)
        return pair in self.active_outages

    def add_outage(self, pair: RegionPair) -> None:
        """Record a new network partition."""
        self.active_outages.add(pair)

    def remove_outage(self, pair: RegionPair) -> None:
        """Record end of a network partition."""
        self.active_outages.discard(pair)

    def regions_reachable_from(self, region: str, all_regions: set[str]) -> set[str]:
        """Get all regions transitively reachable from the given region.

        Uses BFS to find all regions that can communicate with the starting
        region, considering that partitions block communication.

        Args:
            region: Starting region.
            all_regions: Set of all known regions.

        Returns:
            Set of regions reachable from the starting region (includes itself).
        """
        reachable = {region}
        frontier = [region]

        while frontier:
            current = frontier.pop()
            for other in all_regions:
                if other not in reachable and not self.is_partitioned(current, other):
                    reachable.add(other)
                    frontier.append(other)

        return reachable

    def get_connected_components(self, all_regions: set[str]) -> list[set[str]]:
        """Get all connected components of regions.

        Regions in the same component can all communicate with each other.

        Args:
            all_regions: Set of all known regions.

        Returns:
            List of sets, where each set is a connected component of regions.
        """
        remaining = set(all_regions)
        components = []

        while remaining:
            start = next(iter(remaining))
            component = self.regions_reachable_from(start, all_regions)
            components.append(component)
            remaining -= component

        return components

    def can_communicate(self, region_a: str, region_b: str, all_regions: set[str]) -> bool:
        """Check if two regions can communicate (directly or transitively).

        Args:
            region_a: First region.
            region_b: Second region.
            all_regions: Set of all known regions.

        Returns:
            True if the regions are in the same connected component.
        """
        if region_a == region_b:
            return True
        reachable = self.regions_reachable_from(region_a, all_regions)
        return region_b in reachable

    def __repr__(self) -> str:
        if not self.active_outages:
            return "NetworkState(no outages)"
        outages = ", ".join(f"{p.region_a}<->{p.region_b}" for p in self.active_outages)
        return f"NetworkState(outages: {outages})"
