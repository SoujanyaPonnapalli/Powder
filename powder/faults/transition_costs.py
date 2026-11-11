"""Pure, per-shard transfer and transition accounting for named RSM placements.

Write rates are decimal GB/s sent by a leader to *each* follower. Running
replication uses a uniform-leader approximation, not a measured leader mix.
Unlike the legacy cadence migration metric, same-region copies are free here
regardless of AZ; cross-region copies pay directed egress once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    write_gb_per_s: float
    partition_gb: float
    old_placement: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shard_id, str) or not self.shard_id:
            raise ValueError("shard_id must be a nonempty string")
        _nonnegative("write_gb_per_s", self.write_gb_per_s)
        _nonnegative("partition_gb", self.partition_gb)
        if not isinstance(self.old_placement, tuple):
            raise ValueError("old_placement must be a tuple of machine IDs")
        _machines(self.old_placement)


@dataclass(frozen=True)
class TransitionCost:
    replica_additions: int
    replica_removals: int
    gb_moved: float
    migration_usd: float


def _nonnegative(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _machines(placement: Sequence[str]) -> tuple[str, ...]:
    if isinstance(placement, (str, bytes)):
        raise ValueError("placement must be a sequence of machine IDs")
    ids = tuple(placement)
    if any(not isinstance(m, str) or not m for m in ids):
        raise ValueError("machine IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate machine IDs in placement")
    return ids


def _locations(machines: Sequence[str], location: Mapping[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    result = {}
    for machine in machines:
        try:
            loc = location[machine]
        except KeyError as exc:
            raise ValueError(f"missing location for machine {machine!r}") from exc
        if (not isinstance(loc, tuple) or len(loc) != 2
                or any(not isinstance(part, str) or not part for part in loc)):
            raise ValueError(f"machine {machine!r} needs a (region, AZ) location")
        result[machine] = loc
    return result


def edge_average_per_gb(
    placement: Sequence[str],
    location: Mapping[str, tuple[str, str]],
    traffic_price: Callable[[str, str, bool], float],
) -> float:
    """Average directed $/GB over all ordered replica pairs, zero for RF < 2."""
    machines = _machines(placement)
    places = _locations(machines, location)
    k = len(machines)
    total = 0.0
    for src in machines:
        src_region, src_az = places[src]
        for dst in machines:
            if src == dst:
                continue
            dst_region, dst_az = places[dst]
            if src_region == dst_region and src_az == dst_az:
                continue  # same AZ is free, independent of the supplied price resolver
            try:
                price = traffic_price(src_region, dst_region, False)
            except KeyError as exc:
                raise ValueError(f"missing traffic price for {src_region}->{dst_region}") from exc
            total += _nonnegative("traffic price", price)
    return _nonnegative("directed edge price total", total) / (k * (k - 1)) if k > 1 else 0.0


def replication_per_hour(
    placement: Sequence[str],
    write_gb_per_s: float,
    location: Mapping[str, tuple[str, str]],
    traffic_price: Callable[[str, str, bool], float],
) -> float:
    """Expected replication $/h with one copy per follower and a uniform leader."""
    rate = _nonnegative("write_gb_per_s", write_gb_per_s)
    machines = _machines(placement)
    q = edge_average_per_gb(machines, location, traffic_price)
    return _nonnegative("replication $/h", 3600.0 * rate * max(len(machines) - 1, 0) * q)


def migration_cost(
    old: Sequence[str],
    new: Sequence[str],
    partition_gb: float,
    location: Mapping[str, tuple[str, str]],
    region_price: Callable[[str, str], float],
) -> TransitionCost:
    """Copy each added replica from its cheapest *old* source, charging once.

    Retained IDs copy nothing. Same-region copies (including inter-AZ) cost $0,
    but still count towards GB moved and the number of additions.
    """
    size = _nonnegative("partition_gb", partition_gb)
    old_ids, new_ids = _machines(old), _machines(new)
    if not old_ids and new_ids:
        raise ValueError("cannot migrate to a nonempty placement without an old source")
    places = _locations(tuple(dict.fromkeys((*old_ids, *new_ids))), location)
    old_set, new_set = set(old_ids), set(new_ids)
    added = new_set - old_set
    cost = 0.0
    for machine in added:
        dst_region = places[machine][0]
        if any(places[source][0] == dst_region for source in old_ids):
            continue  # a free old source is already optimal; no other prices needed
        prices = []
        for source in old_ids:
            src_region = places[source][0]
            try:
                price = region_price(src_region, dst_region)
            except KeyError as exc:
                raise ValueError(f"missing migration price for {src_region}->{dst_region}") from exc
            prices.append(_nonnegative("migration price", price))
        cost += size * min(prices)
    return TransitionCost(
        replica_additions=len(added),
        replica_removals=len(old_set - new_set),
        gb_moved=_nonnegative("GB moved", size * len(added)),
        migration_usd=_nonnegative("migration USD", cost),
    )
