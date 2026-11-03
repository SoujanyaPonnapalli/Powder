"""Availability summaries from request and probe logs (plan §7).

Two independent measurements, each reported **per partition, per RSM, per client region and overall**:

* :class:`WorkloadAggregator` - what clients saw. ``unavailability = timeout / (ok + timeout)``; conflicts and
  ``other`` are excluded from that ratio and reported as their own rates. Every non-``ok`` request keeps its
  ``cause``. Each RSM's share of all requests (``weight``) is reported beside its number, and RSMs with fewer
  than ``LOW_SAMPLE`` requests are flagged instead of being presented as 0% unavailable.
* :class:`ProbeAggregator` - time-based partition availability from the fixed-rate prober. A partition is
  unavailable in a time bin when probes fail (``all``: every reporting client region failed, the RSM-level
  quantity the simulator predicts; ``any``: at least one client region failed, the client-experienced one).

*Partition* = a stable key bucket defined by the experiment (``regions_target`` equal slices of ``keys`` rows).
*RSM* = the voter store set of the Region containing the key at the request's timestamp (Regions split, merge
and move, so the Region-map snapshot in force at that time is used). Both aggregators stream records, so
memory is proportional to the number of groups, not requests.

Records are dicts: ``ts`` (epoch s), ``client_region``, ``row`` (key id), ``outcome``, ``cause``, ``code``,
``layer``, ``latency_ms``; ``outcome`` may be missing if ``code``/``err`` are present (it is then classified
with :mod:`outcomes`). Latency percentiles come from log-scale buckets with ~2% resolution.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from . import outcomes as O
from .regionmap import RegionMapSeries, bucket_of, row_key

LOW_SAMPLE = 100
BIN_EPS_S = 1e-3            # tolerance for timestamps that sit on a bin boundary
OTHER_FLAG = 0.001          # run is flagged when more than 0.1% of requests are unclassified ("other")
_LAT_BASE = math.log(1.02)


def lat_bucket(ms: float) -> int:
    return 0 if ms <= 0.01 else max(0, int(math.log(ms / 0.01) / _LAT_BASE) + 1)


def lat_value(idx: int) -> float:
    return 0.01 * math.exp(idx * _LAT_BASE)       # bucket upper bound


def percentile(hist: Counter, q: float) -> float | None:
    n = sum(hist.values())
    if not n:
        return None
    target, seen = q * n, 0
    for idx in sorted(hist):
        seen += hist[idx]
        if seen >= target:
            return lat_value(idx)
    return lat_value(max(hist))


def normalize(rec: dict) -> dict:
    """Fill in ``outcome``/``cause``/``layer`` from ``code``/``err`` when a record carries only raw fields."""
    if rec.get("outcome"):
        return rec
    out = O.classify(code=rec.get("code"), text=rec.get("err"), deadline_exceeded=bool(rec.get("deadline")),
                     failed=bool(rec.get("failed", True)))
    return {**rec, "outcome": out.outcome, "cause": out.cause, "layer": out.layer}


def merge_sorted(*streams: Iterable[dict]) -> Iterator[dict]:
    """Merge per-client record streams (each already time-ordered) into one time-ordered stream."""
    return heapq.merge(*streams, key=lambda r: r["ts"])


# ------------------------------------------------------------------ workload
@dataclass
class _Group:
    counts: O.Counts = field(default_factory=O.Counts)
    causes: Counter = field(default_factory=Counter)        # (cause, outcome, layer) -> n  for non-ok
    lat: Counter = field(default_factory=Counter)           # latency bucket -> n  for ok

    def add(self, outcome: str, cause, layer, latency_ms, w: int = 1) -> None:
        self.counts.add(outcome, w)
        if outcome == O.OK:
            if latency_ms is not None:
                self.lat[lat_bucket(float(latency_ms))] += w
        else:
            self.causes[(cause or "unknown", outcome, layer)] += w

    def to_dict(self, grand_total: int) -> dict:
        d = self.counts.to_dict()
        d["latency_ms"] = {"p50": percentile(self.lat, 0.5), "p99": percentile(self.lat, 0.99)}
        d["causes"] = _cause_table(self.causes, self.counts.total - self.counts.ok)
        d["weight"] = self.counts.total / grand_total if grand_total else None
        d["low_sample"] = self.counts.total < LOW_SAMPLE
        return d


def _cause_table(causes: Counter, non_ok_total: int) -> dict:
    out = {}
    for (cause, outcome, layer), n in sorted(causes.items(), key=lambda kv: -kv[1]):
        out[cause] = {"count": n, "share": n / non_ok_total if non_ok_total else None,
                      "outcome": outcome, "layer": layer}
    return out


class WorkloadAggregator:
    def __init__(self, maps: RegionMapSeries, table_id: int, keys: int, regions_target: int,
                 interval_s: float = 10.0, repartition_index=None):
        self.maps, self.table_id, self.keys, self.buckets = maps, table_id, keys, regions_target
        self.interval_s = interval_s
        self.repartition = repartition_index
        self.overall = _Group()
        self.by_partition: dict[int, _Group] = {}
        self.by_rsm: dict[str, _Group] = {}
        self.by_client: dict[str, _Group] = {}
        self.over_time: dict[int, Counter] = {}
        self._t_first = self._t_last = None
        self._fail_total = self._fail_during = self._to_total = self._to_during = 0

    def add(self, rec: dict) -> None:
        rec = normalize(rec)
        ts, outcome = rec["ts"], rec["outcome"]
        row = rec["row"]
        key = row_key(self.table_id, row)
        rsm = self.maps.at(ts).rsm_label(key)
        w = int(rec.get("w", 1))                     # >1 when the load generator down-sampled ok requests
        args = (outcome, rec.get("cause"), rec.get("layer"), rec.get("latency_ms"), w)
        self.overall.add(*args)
        self.by_partition.setdefault(bucket_of(row, self.keys, self.buckets), _Group()).add(*args)
        self.by_rsm.setdefault(rsm, _Group()).add(*args)
        self.by_client.setdefault(rec["client_region"], _Group()).add(*args)
        self._t_first = ts if self._t_first is None else min(self._t_first, ts)
        self._t_last = ts if self._t_last is None else max(self._t_last, ts)
        if outcome != O.OK:
            b = int(ts // self.interval_s)
            self.over_time.setdefault(b, Counter())[rec.get("cause") or "unknown"] += w
            if self.repartition is not None:
                during = self.repartition.during(ts, key)
                self._fail_total += w
                self._fail_during += w * during
                if outcome == O.TIMEOUT:
                    self._to_total += w
                    self._to_during += w * during

    def consume(self, records: Iterable[dict]) -> "WorkloadAggregator":
        for r in records:
            self.add(r)
        return self

    def result(self) -> dict:
        total = self.overall.counts.total
        o = self.overall.to_dict(total)
        low = sorted(k for k, g in self.by_rsm.items() if g.counts.total < LOW_SAMPLE)
        res = {
            "window": {"first_ts": self._t_first, "last_ts": self._t_last},
            "overall": o,
            "causes": o["causes"],
            "causes_by_interval": [{"t": b * self.interval_s, "interval_s": self.interval_s, "causes": dict(c)}
                                   for b, c in sorted(self.over_time.items())],
            "by_partition": {str(k): g.to_dict(total) for k, g in sorted(self.by_partition.items())},
            "by_rsm": {k: g.to_dict(total) for k, g in sorted(self.by_rsm.items())},
            "by_client_region": {k: g.to_dict(total) for k, g in sorted(self.by_client.items())},
            "flags": {
                "other_rate_above_threshold": bool(total) and self.overall.counts.other / total > OTHER_FLAG,
                "low_sample_rsms": low,
            },
            "during_repartition": None,
        }
        if self.repartition is not None:
            res["during_repartition"] = {
                "failed_requests": self._fail_total, "failed_during": self._fail_during,
                "timeouts": self._to_total, "timeouts_during": self._to_during,
                "timeout_share_during": self._to_during / self._to_total if self._to_total else None,
            }
        return res


# ------------------------------------------------------------------ probes
@dataclass
class _Bins:
    bins: int = 0
    failed_any: int = 0
    failed_all: int = 0

    def to_dict(self, interval_s: float, expected_bins: float | None = None) -> dict:
        d = {"bins": self.bins, "seconds": self.bins * interval_s,
             "unavailable_seconds_all": self.failed_all * interval_s,
             "unavailable_seconds_any": self.failed_any * interval_s,
             "unavailability_all": self.failed_all / self.bins if self.bins else None,
             "unavailability_any": self.failed_any / self.bins if self.bins else None}
        if expected_bins:
            d["coverage"] = self.bins / expected_bins
        return d


class ProbeAggregator:
    """Fixed-interval prober -> time-based partition availability.

    Records are stamped with their *scheduled* time but written at *completion*, so a probe that waits out a 5 s
    timeout lands in the file ~5 s after later, faster ones. Records may therefore arrive out of order by up to
    ``max_lag_s`` (set it to the probe timeout plus margin); a bin is finalised only once the stream is that far
    past it. A record later than that is counted in ``late_records`` and flagged, never double-counted. Bins with no
    probe at all are not counted (they lower ``coverage``, which is reported and flagged below 95%).
    """

    def __init__(self, maps: RegionMapSeries, table_id: int, keys: int, regions_target: int,
                 interval_s: float, window: tuple[float, float], max_lag_s: float = 30.0):
        self.maps, self.table_id, self.keys, self.buckets = maps, table_id, keys, regions_target
        self.interval_s, self.t0, self.t1 = interval_s, window[0], window[1]
        self.lag_bins = int(math.ceil(max_lag_s / interval_s)) + 1
        self._closed_below = 0               # every bin index below this is finalised
        self.late = 0
        self.expected_bins = max(1.0, (self.t1 - self.t0) / interval_s)
        self._open: dict[tuple[int, int], dict] = {}        # (bin, bucket) -> {"regions": {region: failed}, "row": row}
        self._max_bin = -1
        self.overall = _Bins()
        self.by_partition: dict[int, _Bins] = {}
        self.by_rsm: dict[str, _Bins] = {}
        self.causes: Counter = Counter()
        self.other = 0
        self.total = 0

    def add(self, rec: dict) -> None:
        rec = normalize(rec)
        # Probes are timestamped at their scheduled instants, i.e. exactly on bin boundaries. At epoch magnitudes a
        # float64 is only good to ~2e-7 s, so floor() alone could put one tick into two different bins (seen on a real
        # cluster: 71-73 bins for a 70-tick run). A 1 ms tolerance keeps every tick in its own bin.
        b = int(math.floor((rec["ts"] - self.t0) / self.interval_s + BIN_EPS_S / self.interval_s))
        if b < 0 or rec["ts"] > self.t1 + BIN_EPS_S:
            return                                           # outside the experiment window
        if b < self._closed_below:
            self.late += 1                   # arrived after its bin was finalised: reported, not re-counted
            return
        bucket = bucket_of(rec["row"], self.keys, self.buckets)
        slot = self._open.setdefault((b, bucket), {"regions": {}, "row": rec["row"]})
        failed = rec["outcome"] == O.TIMEOUT
        reg = rec["client_region"]
        slot["regions"][reg] = slot["regions"].get(reg, False) or failed
        self.total += 1
        if rec["outcome"] == O.OTHER:
            self.other += 1
        if rec["outcome"] != O.OK:
            self.causes[(rec.get("cause") or "unknown", rec["outcome"], rec.get("layer"))] += 1
        if b > self._max_bin:
            self._max_bin = b
            self._close_before(b - self.lag_bins)

    def consume(self, records: Iterable[dict]) -> "ProbeAggregator":
        for r in records:
            self.add(r)
        return self

    def _close_before(self, bin_idx: int) -> None:
        self._closed_below = max(self._closed_below, bin_idx)
        for key in [k for k in self._open if k[0] < bin_idx]:
            self._finalize(key, self._open.pop(key))

    def _finalize(self, key: tuple[int, int], slot: dict) -> None:
        b, bucket = key
        flags = list(slot["regions"].values())
        any_f, all_f = any(flags), all(flags)
        mid = self.t0 + (b + 0.5) * self.interval_s
        rsm = self.maps.at(mid).rsm_label(row_key(self.table_id, slot["row"]))
        for bins in (self.overall, self.by_partition.setdefault(bucket, _Bins()), self.by_rsm.setdefault(rsm, _Bins())):
            bins.bins += 1
            bins.failed_any += any_f
            bins.failed_all += all_f

    def result(self) -> dict:
        self._close_before(10 ** 12)
        per = {str(k): v.to_dict(self.interval_s, self.expected_bins) for k, v in sorted(self.by_partition.items())}
        ua = [v["unavailability_all"] for v in per.values() if v["unavailability_all"] is not None]
        uy = [v["unavailability_any"] for v in per.values() if v["unavailability_any"] is not None]
        cov = [v["coverage"] for v in per.values()]
        return {
            "interval_s": self.interval_s,
            "window": {"start": self.t0, "end": self.t1},
            "overall": self.overall.to_dict(self.interval_s, self.expected_bins * max(1, len(per))),
            "aggregate": {
                "partitions": len(per),
                "mean_unavailability_all": sum(ua) / len(ua) if ua else None,
                "worst_unavailability_all": max(ua) if ua else None,
                "mean_unavailability_any": sum(uy) / len(uy) if uy else None,
                "worst_unavailability_any": max(uy) if uy else None,
                "min_coverage": min(cov) if cov else None,
            },
            "by_partition": per,
            "by_rsm": {k: v.to_dict(self.interval_s) for k, v in sorted(self.by_rsm.items())},
            "causes": _cause_table(self.causes, sum(self.causes.values())),
            "late_records": self.late,
            "flags": {"low_coverage": bool(cov) and min(cov) < 0.95, "late_records": self.late > 0,
                      "other_rate_above_threshold": bool(self.total) and self.other / self.total > OTHER_FLAG},
        }
