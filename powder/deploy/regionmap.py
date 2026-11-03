"""Normalized PD Region map (a snapshot of key ranges, replica sets and leaders) used for attribution.

Requests and probes are attributed to **partitions** (stable experiment-defined key buckets) and to **RSMs**
(the set of stores holding the Raft group of the Region that contains the key *at that moment*). Regions
split, merge and move during a run, so attribution uses the snapshot in force at each timestamp.

``from_pd_json`` (below) converts PD's ``/pd/api/v1/regions`` and ``/pd/api/v1/stores`` documents into this
form; everything else in the metric code depends only on these types.
"""

from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field


class RegionMapError(ValueError):
    pass


def _enc_int(v: int) -> bytes:
    """TiDB's memcomparable int64: big-endian with the sign bit flipped."""
    return struct.pack(">Q", (v + (1 << 63)) & 0xFFFFFFFFFFFFFFFF)


def row_key(table_id: int, row_id: int) -> bytes:
    """Raw (pre-MVCC) record key of a row of a TiDB table: ``t{table}_r{row}``."""
    return b"t" + _enc_int(table_id) + b"_r" + _enc_int(row_id)


def bucket_of(row: int, keys: int, buckets: int) -> int:
    """Stable partition index (0..buckets-1) of ``row`` when ``keys`` rows are split into equal buckets."""
    if not 0 <= row < keys:
        raise RegionMapError(f"row {row} outside 0..{keys - 1}")
    return min(buckets - 1, row * buckets // keys)


@dataclass(frozen=True)
class Region:
    id: int
    start: bytes
    end: bytes                      # b"" means +infinity
    voters: tuple[int, ...]         # store ids of voting peers, sorted
    learners: tuple[int, ...] = ()
    leader: int | None = None
    size_mb: float = 0.0
    down: tuple[int, ...] = ()      # store ids of peers PD reports as down
    pending: tuple[int, ...] = ()   # store ids of peers still catching up

    @property
    def rsm(self) -> tuple[int, ...]:
        return self.voters

    def contains(self, key: bytes) -> bool:
        return key >= self.start and (self.end == b"" or key < self.end)

    def under_replicated(self, replication_factor: int) -> bool:
        healthy = [s for s in self.voters if s not in self.down and s not in self.pending]
        return len(healthy) < replication_factor


@dataclass(frozen=True)
class StoreInfo:
    id: int
    address: str = ""
    state: str = "Up"
    region: str = ""                # location labels from the store's labels
    zone: str = ""
    host: str = ""
    machine_class: str = ""


@dataclass
class RegionMap:
    ts: float
    regions: list[Region]
    stores: dict[int, StoreInfo] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.regions = sorted(self.regions, key=lambda r: r.start)
        self._starts = [r.start for r in self.regions]

    def find(self, key: bytes) -> Region | None:
        i = bisect.bisect_right(self._starts, key) - 1
        if i < 0:
            return None
        r = self.regions[i]
        return r if r.contains(key) else None

    def boundaries(self) -> set[bytes]:
        return {r.start for r in self.regions if r.start != b""}

    def coverage_problems(self) -> list[str]:
        """Gaps or overlaps between consecutive Regions (a valid map tiles the whole key space)."""
        p = []
        if not self.regions:
            return ["empty map"]
        if self.regions[0].start != b"":
            p.append("first Region does not start at the beginning of the key space")
        for a, b in zip(self.regions, self.regions[1:]):
            if a.end == b"":
                p.append(f"Region {a.id} is unbounded but is followed by Region {b.id}")
            elif a.end < b.start:
                p.append(f"gap between Region {a.id} and {b.id}")
            elif a.end > b.start:
                p.append(f"overlap between Region {a.id} and {b.id}")
        if self.regions[-1].end != b"":
            p.append("last Region is not unbounded")
        return p

    def rsm_label(self, key: bytes) -> str:
        r = self.find(key)
        return ",".join(map(str, r.rsm)) if r else "unknown"


class RegionMapSeries:
    """Time-ordered snapshots; ``at(ts)`` is the latest snapshot at or before ``ts`` (or the first one)."""

    def __init__(self, maps: list[RegionMap]):
        if not maps:
            raise RegionMapError("need at least one Region map")
        self.maps = sorted(maps, key=lambda m: m.ts)
        self._ts = [m.ts for m in self.maps]

    def at(self, ts: float) -> RegionMap:
        i = bisect.bisect_right(self._ts, ts) - 1
        return self.maps[max(i, 0)]

    def __len__(self) -> int:
        return len(self.maps)
