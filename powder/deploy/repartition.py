"""Repartitioning analysis: how often and how long TiDB re-partitions (plan §7, measurement 3).

"Repartitioning" = splits and merges (key ranges change), replica moves (add / remove / move a peer: an RSM's
membership changes and data is copied) and leader transfers (no data moves, but the serving node changes).

Two sources, cross-checked in the summary:

* **Region-map snapshots** (:func:`diff_maps`): version-independent ground truth at snapshot resolution. An
  event is only known to have happened inside ``[t_start, t_end]`` (consecutive snapshots), so windows are
  upper bounds. Splits and merges are detected from *boundary sets* (a start key that appeared or vanished),
  not from Region ids, because ids are reassigned on split/merge.
* **PD operator lifecycle** (:class:`OperatorEvent`, produced by the PD log parser in ``pdlog.py``): exact
  create/start/finish times, the creating scheduler/checker, and outcome.

Derived metrics: counts and per-hour rates by kind, episodes (events separated by less than ``gap_s`` merge
into one), the share of run time with something in flight, GB moved split intra- vs inter-region, per-RSM
membership changes, and under-replication time (the restore-replication time exec #10 needs).
"""

from __future__ import annotations

import bisect
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .availability import percentile  # noqa: F401  (re-exported for callers)
from .regionmap import Region, RegionMap, RegionMapSeries

KINDS = ("split", "merge", "add_peer", "remove_peer", "move_peer", "peer_churn", "transfer_leader")
DATA_MOVING = ("add_peer", "move_peer")
MIB_PER_GIB = 1024.0
LEADER_WINDOW_S = 3.0


@dataclass(frozen=True)
class Event:
    """One repartitioning event inferred from consecutive Region-map snapshots."""
    kind: str
    t_start: float                  # earlier snapshot
    t_end: float                    # later snapshot
    key: bytes                      # start key of the affected range (the boundary for split/merge)
    range_start: bytes
    range_end: bytes
    region_id: int | None = None
    from_store: int | None = None
    to_store: int | None = None
    size_mb: float = 0.0
    inter_region: bool | None = None    # data crossed AWS regions; None when store locations are unknown
    rsm_before: tuple[int, ...] = ()
    source: str = "region_map"
    conf_delta: int = 0                 # peer_churn: how many membership changes (epoch.conf_ver) happened in the window

    def covers(self, key: bytes) -> bool:
        return key >= self.range_start and (self.range_end == b"" or key < self.range_end)


@dataclass(frozen=True)
class OperatorEvent:
    """One PD operator, from its log lifecycle."""
    kind: str                       # PD's operator description, mapped to KINDS where possible, else raw
    region_id: int | None
    t_create: float
    t_start: float | None = None
    t_finish: float | None = None
    outcome: str = "finished"       # finished | timeout | canceled | replaced | unknown
    trigger: str = ""               # scheduler / checker that created it (balance-region, merge-checker, ...)
    from_store: int | None = None
    to_store: int | None = None
    desc: str = ""
    size_mb: float = 0.0
    cancel_reason: str = ""
    name: str = ""                  # PD operator name, e.g. admin-transfer-leader, replace-rule-down-peer

    @property
    def t_end(self) -> float:
        return self.t_finish if self.t_finish is not None else (self.t_start or self.t_create)

    @property
    def duration(self) -> float | None:
        return None if self.t_finish is None else self.t_finish - self.t_create


# ------------------------------------------------------------------ snapshot diffs
def _store_region(maps: tuple[RegionMap, RegionMap], store: int | None) -> str | None:
    for m in (maps[1], maps[0]):
        s = m.stores.get(store) if store is not None else None
        if s and s.region:
            return s.region
    return None


def diff_maps(prev: RegionMap, cur: RegionMap) -> list[Event]:
    """Events that happened between two consecutive snapshots."""
    ev: list[Event] = []
    t0, t1 = prev.ts, cur.ts

    for b in sorted(cur.boundaries() - prev.boundaries()):
        parent = prev.find(b)
        child = cur.find(b)
        ev.append(Event("split", t0, t1, b, parent.start if parent else b, parent.end if parent else b"",
                        child.id if child else None, rsm_before=parent.rsm if parent else ()))
    for b in sorted(prev.boundaries() - cur.boundaries()):
        merged = cur.find(b)
        gone = prev.find(b)
        ev.append(Event("merge", t0, t1, b, merged.start if merged else b, merged.end if merged else b"",
                        gone.id if gone else None, rsm_before=gone.rsm if gone else ()))

    old = {(r.start, r.end): r for r in prev.regions}
    for r in cur.regions:
        p = old.get((r.start, r.end))
        if p is None:
            continue                                    # range changed shape: split/merge above covers it
        peers_p, peers_r = set(p.voters) | set(p.learners), set(r.voters) | set(r.learners)
        added, removed = sorted(peers_r - peers_p), sorted(peers_p - peers_r)
        base = dict(t_start=t0, t_end=t1, key=r.start, range_start=r.start, range_end=r.end, region_id=r.id,
                    size_mb=r.size_mb, rsm_before=p.rsm)
        src = r.leader if r.leader in peers_p else p.leader     # data is streamed from the leader
        paired = min(len(added), len(removed))
        for i in range(paired):
            ev.append(Event("move_peer", from_store=removed[i], to_store=added[i],
                            inter_region=_cross((prev, cur), src, added[i]), **base))
        for s in added[paired:]:
            ev.append(Event("add_peer", to_store=s, inter_region=_cross((prev, cur), src, s), **base))
        for s in removed[paired:]:
            ev.append(Event("remove_peer", from_store=s, **base))
        if peers_p == peers_r and r.conf_ver > p.conf_ver:
            # membership changed and changed back inside one snapshot interval (e.g. add + remove of the same
            # store): the peer sets look identical, but epoch.conf_ver proves it happened, and data may have moved
            ev.append(Event("peer_churn", conf_delta=r.conf_ver - p.conf_ver, **base))
        if p.leader is not None and r.leader is not None and p.leader != r.leader:
            ev.append(Event("transfer_leader", from_store=p.leader, to_store=r.leader, **base))
    return sorted(ev, key=lambda e: (e.t_end, e.key, e.kind))


def _cross(maps, src: int | None, dst: int | None) -> bool | None:
    a, b = _store_region(maps, src), _store_region(maps, dst)
    return None if a is None or b is None else a != b


def events_from_series(series: RegionMapSeries) -> list[Event]:
    out: list[Event] = []
    for a, b in zip(series.maps, series.maps[1:]):
        out += diff_maps(a, b)
    return out


# ------------------------------------------------------------------ intervals, episodes
def merge_intervals(intervals: list[tuple[float, float]], gap_s: float = 0.0) -> list[tuple[float, float, int]]:
    """Union of intervals; ones closer than ``gap_s`` merge. Returns (start, end, count)."""
    out: list[list] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + gap_s:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2] += 1
        else:
            out.append([a, b, 1])
    return [(a, b, n) for a, b, n in out]


def episode_summary(intervals: list[tuple[float, float]], run_seconds: float, gap_s: float = 60.0) -> dict:
    eps = merge_intervals(intervals, gap_s)
    durs = sorted(b - a for a, b, _ in eps)
    busy = sum(b - a for a, b, _ in merge_intervals(intervals, 0.0))
    return {
        "gap_s": gap_s,
        "count": len(eps),
        "episodes": [{"start": a, "end": b, "duration_s": b - a, "events": n} for a, b, n in eps],
        "duration_s": {"p50": _pct(durs, 0.5), "p90": _pct(durs, 0.9), "max": durs[-1] if durs else None},
        "busy_seconds": busy,
        "busy_share": busy / run_seconds if run_seconds > 0 else None,
    }


def _pct(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


# ------------------------------------------------------------------ under-replication
def under_replication(series: RegionMapSeries, replication_factor: int, window_end: float | None = None) -> dict:
    """Time Regions spent with fewer healthy voters than the replication factor (down or catching-up peers count
    as unhealthy). ``restore_times_s`` is, per Region (keyed by start key), first flagged snapshot to first clear
    snapshot; episodes still open at the end are reported as ``unresolved`` (censored), not as restored."""
    maps = series.maps
    region_seconds = 0.0
    any_intervals: list[tuple[float, float]] = []
    by_rsm: Counter = Counter()
    open_since: dict[bytes, float] = {}
    restore: list[float] = []
    for i, m in enumerate(maps):
        nxt = maps[i + 1].ts if i + 1 < len(maps) else (window_end if window_end is not None else m.ts)
        d = max(0.0, nxt - m.ts)
        flagged = {r.start: r for r in m.regions if r.under_replicated(replication_factor)}
        if flagged and d > 0:
            any_intervals.append((m.ts, nxt))
        for start, r in flagged.items():
            region_seconds += d
            by_rsm[",".join(map(str, r.rsm))] += d
            open_since.setdefault(start, m.ts)
        for start in [s for s in open_since if s not in flagged]:
            restore.append(m.ts - open_since.pop(start))
    last_ts = maps[-1].ts if window_end is None else window_end
    unresolved = len(open_since)
    total = last_ts - maps[0].ts
    anysec = sum(b - a for a, b, _ in merge_intervals(any_intervals))
    rs = sorted(restore)
    return {
        "region_seconds": region_seconds,
        "any_seconds": anysec,
        "any_share": anysec / total if total > 0 else None,
        "restore_times_s": rs,
        "restore_s": {"p50": _pct(rs, 0.5), "p90": _pct(rs, 0.9), "max": rs[-1] if rs else None},
        "unresolved": unresolved,
        "by_rsm_region_seconds": dict(by_rsm),
    }


# ------------------------------------------------------------------ operator summary
def summarize_operators(ops: list[OperatorEvent], run_seconds: float, gap_s: float = 60.0) -> dict:
    by_kind: Counter = Counter(o.kind for o in ops)
    finished_kind: Counter = Counter(o.kind for o in ops if o.outcome == "finished")
    by_trigger: Counter = Counter(o.trigger or "unknown" for o in ops)
    by_outcome: Counter = Counter(o.outcome for o in ops)
    durs: dict[str, list[float]] = {}
    for o in ops:
        if o.duration is not None and o.outcome == "finished":
            durs.setdefault(o.kind, []).append(o.duration)
    dist = {k: {"n": len(v), "p50": _pct(sorted(v), 0.5), "p90": _pct(sorted(v), 0.9), "p99": _pct(sorted(v), 0.99),
                "max": max(v)} for k, v in durs.items()}
    hours = run_seconds / 3600.0 if run_seconds > 0 else None
    return {
        "count": len(ops),
        "by_kind": dict(by_kind), "by_kind_finished": dict(finished_kind), "by_trigger": dict(by_trigger), "by_outcome": dict(by_outcome),
        "rate_per_hour_by_kind": {k: n / hours for k, n in by_kind.items()} if hours else None,
        "duration_s_by_kind": dist,
        "episodes": episode_summary([(o.t_create, o.t_end) for o in ops], run_seconds, gap_s),
    }


# ------------------------------------------------------------------ flag index
def region_for_key_fn(series: RegionMapSeries):
    """``(ts, key) -> region id`` using the snapshot in force at ``ts`` (for matching operators to keys)."""
    def f(ts: float, key: bytes) -> int | None:
        r = series.at(ts).find(key)
        return r.id if r else None
    return f


class RepartitionIndex:
    """Answers "was this key's Region being repartitioned at time ts?" for failed-request attribution.

    Uses map-diff events (their windows are upper bounds) and/or operators (exact). Leader transfers count for
    ``LEADER_WINDOW_S`` after they start, since an election gap outlives the transfer itself.
    """

    def __init__(self, events: list[Event] | None = None, operators: list[OperatorEvent] | None = None,
                 region_for_key=None, leader_window_s: float = LEADER_WINDOW_S):
        self.leader_window_s = leader_window_s
        self.region_for_key = region_for_key            # key -> region id (needed to match operators by region)
        items = []
        for e in events or []:
            pad = leader_window_s if e.kind == "transfer_leader" else 0.0
            items.append((e.t_start, e.t_end + pad, ("range", e.range_start, e.range_end)))
        for o in operators or []:
            pad = leader_window_s if o.kind == "transfer_leader" else 0.0
            items.append((o.t_create, o.t_end + pad, ("region", o.region_id)))
        items.sort(key=lambda x: x[0])
        self._items = items
        self._starts = [i[0] for i in items]
        run, mx = [], float("-inf")
        for _, end, _ in items:
            mx = max(mx, end)
            run.append(mx)
        self._max_end = run

    def during(self, ts: float, key: bytes) -> bool:
        i = bisect.bisect_right(self._starts, ts) - 1
        while i >= 0 and self._max_end[i] >= ts:
            _, end, sel = self._items[i]
            if end >= ts:
                if sel[0] == "range":
                    if key >= sel[1] and (sel[2] == b"" or key < sel[2]):
                        return True
                elif self.region_for_key is not None and self.region_for_key(ts, key) == sel[1]:
                    return True
            i -= 1
        return False


# ------------------------------------------------------------------ summary
def summarize(series: RegionMapSeries, replication_factor: int, window: tuple[float, float],
              operators: list[OperatorEvent] | None = None, gap_s: float = 60.0,
              inter_region_usd_per_gb: float | None = None, log_splits: int | None = None) -> dict:
    t0, t1 = window
    run_s = max(0.0, t1 - t0)
    events = events_from_series(series)
    counts = Counter(e.kind for e in events)
    hours = run_s / 3600.0 if run_s > 0 else None
    gb = {"intra_region": 0.0, "inter_region": 0.0, "unknown": 0.0}
    for e in events:
        if e.kind in DATA_MOVING:
            g = e.size_mb / MIB_PER_GIB
            gb["unknown" if e.inter_region is None else "inter_region" if e.inter_region else "intra_region"] += g
    gb["total"] = sum(gb.values())
    if inter_region_usd_per_gb is not None:
        gb["inter_region_usd"] = gb["inter_region"] * inter_region_usd_per_gb
    changes: Counter = Counter()
    for e in events:
        if e.kind in ("add_peer", "remove_peer", "move_peer"):
            changes[",".join(map(str, e.rsm_before))] += 1
    sizes = [len(m.regions) for m in series.maps]
    ivals = [b.ts - a.ts for a, b in zip(series.maps, series.maps[1:])]
    out = {
        "region_maps": {"snapshots": len(series), "median_interval_s": statistics.median(ivals) if ivals else None,
                        "coverage_problems": sum(bool(m.coverage_problems()) for m in series.maps)},
        "counts": {k: counts.get(k, 0) for k in KINDS},
        "rates_per_hour": {k: counts.get(k, 0) / hours for k in KINDS} if hours else None,
        "region_count": {"start": sizes[0], "end": sizes[-1], "min": min(sizes), "max": max(sizes)},
        "episodes_region_maps": episode_summary([(e.t_start, e.t_end) for e in events], run_s, gap_s),
        "gb_moved": gb,
        "per_rsm_membership_changes": dict(changes),
        "under_replication": under_replication(series, replication_factor, t1),
        "operators": summarize_operators(operators, run_s, gap_s) if operators is not None else None,
        "events": len(events),
    }
    if operators is not None:
        op_counts = Counter(o.kind for o in operators if o.outcome == "finished")   # a merge logs two operators, one canceled
        cc = {k: {"region_maps": counts.get(k, 0), "operators": op_counts.get(k, 0)}
              for k in ("split", "merge", "add_peer", "remove_peer", "move_peer", "transfer_leader")}
        if log_splits is not None:                      # TiKV-initiated splits are not operators: they are logged as batch splits
            cc["split"]["operators"] += log_splits
        out["cross_check"] = cc
    return out
