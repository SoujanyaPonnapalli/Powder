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


def mc_encode(raw: bytes) -> bytes:
    """TiKV/TiDB memcomparable bytes encoding: 8-byte groups, each followed by ``0xFF - padding``."""
    out = bytearray()
    for i in range(0, len(raw) + 1, 8):
        chunk = raw[i:i + 8]
        pad = 8 - len(chunk)
        out += chunk + b"\x00" * pad + bytes([0xFF - pad])
        if pad:
            break
    return bytes(out)


def raw_row_key(table_id: int, row_id: int) -> bytes:
    """Raw (pre-encoding) record key of a TiDB row: ``t{table}_r{row}``."""
    return b"t" + _enc_int(table_id) + b"_r" + _enc_int(row_id)


def row_key(table_id: int, row_id: int) -> bytes:
    """Key of a row as PD reports Region boundaries (memcomparable-encoded), so it compares with Region keys."""
    return mc_encode(raw_row_key(table_id, row_id))


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
    down: tuple[int, ...] = ()      # voters on stores that are not Up (PD's own down_peers field is never populated)
    pending: tuple[int, ...] = ()   # store ids of peers still catching up
    conf_ver: int = 0               # epoch.conf_ver: +1 for every membership change (add learner, promote, demote, remove)
    version: int = 0                # epoch.version: +1 on split / merge

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


# ------------------------------------------------------------------ PD JSON loader
UP_STATES = {"Up"}
VOTER_ROLES = {"Voter", "IncomingVoter", "DemotingVoter"}     # joint-consensus roles still count toward membership


def parse_stores(doc: dict) -> dict[int, StoreInfo]:
    """``GET /pd/api/v1/stores`` -> {store id: StoreInfo}; location labels come from the store labels."""
    out = {}
    for item in doc.get("stores", []):
        st = item["store"]
        labels = {l["key"]: l["value"] for l in st.get("labels", [])}
        out[int(st["id"])] = StoreInfo(
            id=int(st["id"]), address=st.get("address", ""), state=st.get("state_name", "Up"),
            region=labels.get("region", ""), zone=labels.get("zone", ""), host=labels.get("host", ""),
            machine_class=labels.get("machine_class", ""))
    return out


def from_pd_json(regions_doc: dict, ts: float, stores_doc: dict | None = None) -> RegionMap:
    """``GET /pd/api/v1/regions`` (+ optionally ``/stores``) -> :class:`RegionMap`.

    Notes from real v8.5.8 output: the list is **unordered** (``RegionMap`` sorts it); keys are uppercase hex of
    memcomparable-encoded keys (``''`` = -inf / +inf); ``pending_peers`` is omitted when empty; ``down_peers`` never
    appears, so a peer is considered *down* when its store is not ``Up`` (needs ``stores_doc``).
    """
    stores = parse_stores(stores_doc) if stores_doc else {}
    regions = []
    for r in regions_doc.get("regions", []):
        voters = sorted(p["store_id"] for p in r.get("peers", []) if p.get("role_name", "Voter") in VOTER_ROLES)
        learners = sorted(p["store_id"] for p in r.get("peers", []) if p.get("role_name") == "Learner")
        down = tuple(s for s in voters if s in stores and stores[s].state not in UP_STATES)
        pending = tuple(sorted(p["store_id"] for p in r.get("pending_peers", [])))
        leader = r.get("leader") or {}
        regions.append(Region(
            id=int(r["id"]), start=bytes.fromhex(r.get("start_key", "")), end=bytes.fromhex(r.get("end_key", "")),
            voters=tuple(voters), learners=tuple(learners), leader=leader.get("store_id"),
            size_mb=float(r.get("approximate_size", 0)), down=down, pending=pending,
            conf_ver=int(r.get("epoch", {}).get("conf_ver", 0)), version=int(r.get("epoch", {}).get("version", 0))))
    return RegionMap(ts, regions, stores)
