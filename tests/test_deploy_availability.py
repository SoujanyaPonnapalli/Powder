"""Workload and probe availability summaries against hand-computed answers (exec #7 step 2).

Layout used throughout: table 100 with 1000 rows split into 4 partitions of 250 rows; two Regions,
R1 = rows < 500 on stores (1,2,3) and R2 = rows >= 500 on stores (1,2,4).
"""

import pytest

from powder.deploy import availability as A
from powder.deploy.regionmap import Region, RegionMap, RegionMapSeries, row_key

TABLE, KEYS, BUCKETS = 100, 1000, 4
SPLIT = row_key(TABLE, 500)


def two_regions(ts=0.0):
    return RegionMap(ts, [Region(1, b"", SPLIT, (1, 2, 3), leader=1),
                          Region(2, SPLIT, b"", (1, 2, 4), leader=1)])


def series(*maps):
    return RegionMapSeries(list(maps))


def req(ts, row, outcome="ok", region="us-east-1", cause=None, latency=5.0, layer=None):
    return {"ts": ts, "client_region": region, "row": row, "outcome": outcome, "cause": cause,
            "latency_ms": latency, "layer": layer}


def workload(maps=None, **kw):
    return A.WorkloadAggregator(maps or series(two_regions()), TABLE, KEYS, BUCKETS, **kw)


def outage_stream():
    """600 s, one request/s to row 100 (R1) and one to row 700 (R2); R2's RSM is down for t in [100,160)."""
    for t in range(600):
        yield req(t + 0.5, 100)
        if 100 <= t < 160:
            yield req(t + 0.5, 700, "timeout", cause="tidb_9005_region_unavailable", layer="tikv", latency=5000)
        else:
            yield req(t + 0.5, 700)


def test_per_rsm_and_per_partition_outage():
    r = workload().consume(outage_stream()).result()
    o = r["overall"]
    assert (o["ok"], o["timeout"], o["total"]) == (1140, 60, 1200)
    assert o["unavailability"] == pytest.approx(60 / 1200)
    assert r["by_rsm"]["1,2,3"]["unavailability"] == 0.0
    assert r["by_rsm"]["1,2,4"]["unavailability"] == pytest.approx(60 / 600)
    assert r["by_rsm"]["1,2,3"]["weight"] == pytest.approx(0.5) == r["by_rsm"]["1,2,4"]["weight"]
    assert r["by_partition"]["0"]["unavailability"] == 0.0          # row 100 -> partition 0
    assert r["by_partition"]["2"]["unavailability"] == pytest.approx(0.1)   # row 700 -> partition 2
    assert set(r["by_partition"]) == {"0", "2"}
    assert r["by_client_region"]["us-east-1"]["total"] == 1200


def test_cause_breakdown_overall_per_rsm_and_over_time():
    r = workload(interval_s=10).consume(outage_stream()).result()
    assert r["causes"] == {"tidb_9005_region_unavailable": {"count": 60, "share": 1.0, "outcome": "timeout",
                                                            "layer": "tikv"}}
    assert r["by_rsm"]["1,2,4"]["causes"]["tidb_9005_region_unavailable"]["count"] == 60
    assert r["by_rsm"]["1,2,3"]["causes"] == {}
    windows = {w["t"]: w["causes"] for w in r["causes_by_interval"]}
    assert sorted(windows) == [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]      # six 10 s windows, 10 each
    assert all(c == {"tidb_9005_region_unavailable": 10} for c in windows.values())


def test_conflicts_are_recorded_with_causes_but_excluded_from_unavailability():
    stream = list(outage_stream())
    stream += [req(t + 0.2, 300, "conflict", cause="tidb_9007_write_conflict", layer="tikv") for t in range(40)]
    r = workload().consume(stream).result()
    o = r["overall"]
    assert o["conflict"] == 40 and o["total"] == 1240
    assert o["unavailability"] == pytest.approx(60 / 1200)                # unchanged by conflicts
    assert o["conflict_rate"] == pytest.approx(40 / 1240)
    assert r["causes"]["tidb_9007_write_conflict"] == {"count": 40, "share": pytest.approx(0.4), "outcome": "conflict",
                                                       "layer": "tikv"}
    assert r["causes"]["tidb_9005_region_unavailable"]["share"] == pytest.approx(0.6)


def test_attribution_follows_the_region_map_in_force():
    whole = RegionMap(0.0, [Region(1, b"", b"", (1, 2, 3), leader=1)])
    split = RegionMap(300.0, [Region(1, b"", SPLIT, (1, 2, 3), leader=1), Region(5, SPLIT, b"", (1, 2, 4), leader=1)])
    agg = workload(series(whole, split))
    for t in (100.0, 200.0, 299.9):
        agg.add(req(t, 700))                       # before the split: row 700 lives on stores 1,2,3
    for t in (300.0, 400.0):
        agg.add(req(t, 700))                       # after: moved to 1,2,4
    r = agg.result()
    assert r["by_rsm"]["1,2,3"]["total"] == 3 and r["by_rsm"]["1,2,4"]["total"] == 2
    assert r["by_rsm"]["1,2,3"]["weight"] == pytest.approx(0.6)


def test_low_sample_rsm_is_flagged_not_reported_as_perfect():
    agg = workload()
    for t in range(500):
        agg.add(req(t, 100))
    for t in range(10):
        agg.add(req(t, 700))
    r = agg.result()
    assert r["by_rsm"]["1,2,4"]["low_sample"] is True and r["by_rsm"]["1,2,3"]["low_sample"] is False
    assert r["flags"]["low_sample_rsms"] == ["1,2,4"]
    assert r["by_rsm"]["1,2,4"]["weight"] == pytest.approx(10 / 510)


def test_unclassified_errors_above_threshold_flag_the_run():
    agg = workload()
    for t in range(1000):
        agg.add(req(t, 100))
    assert agg.result()["flags"]["other_rate_above_threshold"] is False
    for t in range(5):                                           # 5/1005 = 0.5% > 0.1%
        agg.add(req(t, 100, "other", cause="unclassified_1146"))
    assert agg.result()["flags"]["other_rate_above_threshold"] is True


def test_raw_records_are_classified_on_the_way_in():
    agg = workload()
    agg.add({"ts": 1.0, "client_region": "us-east-1", "row": 700, "code": 9005, "latency_ms": 10})
    agg.add({"ts": 2.0, "client_region": "us-east-1", "row": 700, "failed": False, "latency_ms": 10})
    r = agg.result()
    assert r["overall"]["timeout"] == 1 and r["overall"]["ok"] == 1
    assert "tidb_9005_region_unavailable" in r["causes"]


def test_latency_percentiles_have_two_percent_resolution():
    agg = workload()
    for i in range(1, 101):                                       # latencies 1..100 ms
        agg.add(req(i, 100, latency=float(i)))
    lat = agg.result()["overall"]["latency_ms"]
    assert lat["p50"] == pytest.approx(50, rel=0.03)
    assert lat["p99"] == pytest.approx(99, rel=0.03)


class _Index:
    def during(self, ts, key):
        return 100 <= ts < 130                                    # one repartitioning episode


def test_failures_during_repartitioning_are_flagged():
    r = workload(repartition_index=_Index()).consume(outage_stream()).result()
    d = r["during_repartition"]
    assert (d["timeouts"], d["timeouts_during"]) == (60, 30)      # outage seconds 100..159, episode 100..129
    assert d["timeout_share_during"] == pytest.approx(0.5)


def test_no_repartition_index_means_no_claim():
    assert workload().consume(outage_stream()).result()["during_repartition"] is None


# ------------------------------------------------------------------ probes
REGIONS = ("us-east-1", "us-east-2", "us-west-2")
PROBE_ROWS = {0: 100, 1: 350, 2: 600, 3: 850}                      # one fixed row per partition


def probe_stream(drop_bucket=None, drop_from=0, drop_to=0):
    for t in range(600):
        for bucket, row in PROBE_ROWS.items():
            if bucket == drop_bucket and drop_from <= t < drop_to:
                continue                                           # prober died for a while
            for reg in REGIONS:
                failed = False
                if bucket >= 2 and 100 <= t < 160:                 # R2's RSM down: every client region fails
                    failed = True
                if bucket == 0 and reg == "us-west-2" and 300 <= t < 330:   # only us-west-2 is cut off
                    failed = True
                yield req(t + 0.3, row, "timeout" if failed else "ok", region=reg,
                          cause="tidb_9005_region_unavailable" if failed else None)


def probes(stream):
    agg = A.ProbeAggregator(series(two_regions()), TABLE, KEYS, BUCKETS, 1.0, (0.0, 600.0))
    return agg.consume(stream).result()


def test_probe_unavailability_per_partition_and_rsm():
    r = probes(probe_stream())
    p = r["by_partition"]
    assert p["2"]["unavailability_all"] == pytest.approx(60 / 600) == p["3"]["unavailability_all"]
    assert p["2"]["unavailable_seconds_all"] == 60
    assert p["0"]["unavailability_all"] == 0.0                     # a single cut-off region is not "all"
    assert p["0"]["unavailability_any"] == pytest.approx(30 / 600)
    assert p["1"]["unavailability_all"] == p["1"]["unavailability_any"] == 0.0
    assert r["by_rsm"]["1,2,4"]["unavailability_all"] == pytest.approx(120 / 1200)
    assert r["by_rsm"]["1,2,3"]["unavailability_all"] == 0.0
    assert r["by_rsm"]["1,2,3"]["unavailability_any"] == pytest.approx(30 / 1200)


def test_probe_aggregate_mean_and_worst():
    a = probes(probe_stream())["aggregate"]
    assert a["partitions"] == 4
    assert a["mean_unavailability_all"] == pytest.approx((0 + 0 + 0.1 + 0.1) / 4)
    assert a["worst_unavailability_all"] == pytest.approx(0.1)
    assert a["mean_unavailability_any"] == pytest.approx((0.05 + 0 + 0.1 + 0.1) / 4)
    assert a["min_coverage"] == pytest.approx(1.0)


def test_probe_gaps_lower_coverage_and_are_flagged_not_counted_as_downtime():
    r = probes(probe_stream(drop_bucket=1, drop_from=200, drop_to=300))
    p1 = r["by_partition"]["1"]
    assert p1["bins"] == 500 and p1["coverage"] == pytest.approx(500 / 600)
    assert p1["unavailability_all"] == 0.0                         # missing data is not downtime
    assert r["flags"]["low_coverage"] is True
    assert probes(probe_stream())["flags"]["low_coverage"] is False


def test_probe_causes_and_window():
    r = probes(probe_stream())
    assert r["causes"]["tidb_9005_region_unavailable"]["count"] == 2 * 60 * 3 + 30   # 2 partitions x 60 s x 3 regions + 30
    agg = A.ProbeAggregator(series(two_regions()), TABLE, KEYS, BUCKETS, 1.0, (0.0, 10.0))
    agg.add(req(50.0, 100, "timeout"))                             # after the window: ignored
    agg.add(req(-5.0, 100, "timeout"))                             # before: ignored
    assert agg.result()["overall"]["bins"] == 0


def test_merge_sorted_interleaves_streams_by_time():
    a = [{"ts": 1}, {"ts": 4}]
    b = [{"ts": 2}, {"ts": 3}, {"ts": 5}]
    assert [r["ts"] for r in A.merge_sorted(a, b)] == [1, 2, 3, 4, 5]
