"""Self-test planning and evaluation (exec #7 gate G3 logic), on the real captured Region map."""

import gzip
import json
from pathlib import Path

import pytest

from powder.deploy import selftest as ST
from powder.deploy.regionmap import from_pd_json

FIX = Path(__file__).parent / "fixtures" / "tidb"


def real_map():
    doc = json.load(gzip.open(FIX / "regions" / "02-split-t31.json.gz"))
    stores = json.loads((FIX / "stores" / "00-baseline.json").read_text())
    return from_pd_json(doc, 0.0, stores)


STORE_NODE = {1: "tikv-1", 4: "tikv-2", 5: "tikv-3", 6: "tikv-4"}


def test_losing_quorum_means_a_majority_of_voters_stopped():
    m = real_map()
    rsms = {ST.rsm_label(r.voters) for r in m.regions}
    assert rsms == {"1,4,5", "1,4,6", "1,5,6", "4,5,6"} or len(rsms) >= 3          # 3 voters on 4 stores
    lost = ST.losing_quorum(m, {5, 6})
    assert lost and all(set(map(int, l.split(","))) >= {5, 6} for l in lost)       # only RSMs containing BOTH stopped stores
    assert ST.losing_quorum(m, {5}) == []                                           # one store never loses a majority of 3
    assert set(ST.losing_quorum(m, {1, 4, 5, 6})) == rsms                           # everything stopped: every RSM


def test_plan_rounds_use_distinct_pairs_and_stagger_in_time():
    rounds = ST.plan_rounds(real_map(), STORE_NODE, rounds=3, hold_s=60, gap_s=90, first_offset_s=60)
    assert len({r.stop_stores for r in rounds}) == 3
    assert [r.start_offset_s for r in rounds] == [60, 210, 360]
    for r in rounds:
        assert r.expected_rsms and r.stop_nodes == tuple(STORE_NODE[s] for s in r.stop_stores)
    assert ST.experiment_duration(rounds) == 360 + 60 + 90


def test_schedule_stops_then_starts_each_node_in_order():
    rounds = ST.plan_rounds(real_map(), STORE_NODE, rounds=1, hold_s=60, first_offset_s=30)
    acts = sorted(ST.schedule_for(rounds), key=lambda a: a.offset_s)
    assert [(a.offset_s, a.label.split("-")[1]) for a in acts] == [(30, "stop")] * 2 + [(90, "start")] * 2
    assert {a.node for a in acts} == set(rounds[0].stop_nodes)
    assert all(a.command.endswith("tikv-20160") for a in acts)
    # a crash (SIGKILL), not a graceful stop that spends up to 20 s evicting leaders (G3)
    assert all("--signal=SIGKILL" in a.command for a in acts if a.label.endswith("stop"))


def test_plan_needs_enough_distinct_pairs():
    with pytest.raises(ValueError, match="distinct store pairs"):
        ST.plan_rounds(real_map(), {1: "a", 4: "b"}, rounds=3, hold_s=60)


def summary(by_rsm):
    return {"probe": {"by_rsm": {k: {"unavailable_seconds_all": v} for k, v in by_rsm.items()}}}


def fake_rounds():
    return [ST.Round((5, 6), ("a", "b"), ["1,5,6", "4,5,6"], 60, 60.0), ST.Round((1, 6), ("c", "b"), ["1,5,6"], 210, 60.0)]


def test_evaluate_passes_when_measurements_match_the_plan():
    # 1,5,6 was hit by both rounds (120 s expected), 4,5,6 by one, 1,4,5 by none
    v = ST.evaluate(summary({"1,5,6": 128, "4,5,6": 66, "1,4,5": 3, "1,4,6": 0}), fake_rounds())
    assert v["passed"] and v["rounds"] == 2
    rows = {r["rsm"]: r for r in v["rsms"]}
    assert rows["1,5,6"]["expected_s"] == 120 and rows["4,5,6"]["expected_s"] == 60 and rows["1,4,5"]["expected_s"] == 0


@pytest.mark.parametrize("measured, why", [
    ({"1,5,6": 20, "4,5,6": 66}, "an RSM that lost quorum twice shows far less than 120 s"),
    ({"1,5,6": 128, "4,5,6": 0}, "an expected RSM shows no downtime at all"),
    ({"1,5,6": 128, "4,5,6": 66, "1,4,5": 80}, "an RSM that kept its quorum shows a long outage"),
    ({"1,5,6": 500, "4,5,6": 66}, "implausibly long outage"),
    ({"1,5,6": 128}, "an expected RSM is missing from the measurements"),
])
def test_evaluate_fails_on_mismatches(measured, why):
    assert not ST.evaluate(summary(measured), fake_rounds())["passed"], why


def test_evaluate_reports_unobserved_expected_rsms():
    v = ST.evaluate(summary({"1,5,6": 128}), fake_rounds())
    assert v["expected_rsms_not_observed"] == ["4,5,6"]


def test_evaluate_compares_the_mean_per_partition_not_the_sum():
    # G3: by_rsm sums partition-seconds; 26 partitions x 62 s must read as 62 s, not 1612 s
    s = {"probe": {"by_partition": {"0": {"bins": 510}},
                   "by_rsm": {"1,5,6": {"bins": 26 * 510, "unavailable_seconds_all": 26 * 125},
                              "4,5,6": {"bins": 24 * 510, "unavailable_seconds_all": 24 * 62}}}}
    v = ST.evaluate(s, fake_rounds())
    rows = {r["rsm"]: r for r in v["rsms"]}
    assert v["passed"] and rows["1,5,6"]["measured_s"] == 125 and rows["4,5,6"]["measured_s"] == 62


def test_plan_is_restricted_to_the_measured_regions():
    # G3: planning on every Region picked system-table RSMs the prober never touches
    m = real_map()
    one = ST.rsm_label(m.regions[0].voters)
    measured = [r for r in m.regions if ST.rsm_label(r.voters) == one]
    rounds = ST.plan_rounds(m, STORE_NODE, rounds=3, hold_s=60, regions=measured)
    assert {l for r in rounds for l in r.expected_rsms} == {one}
    assert ST.planned_duration(3, 60) == ST.experiment_duration(rounds)


def _fails(spans):
    """second -> all probes failed, for failure spans [(a, b), ...] within 0..600 s."""
    return {s: any(a <= s < b for a, b in spans) for s in range(600)}


def _round():
    return [ST.Round((5, 6), ("a", "b"), ["4,5,6"], 60, 60.0)]


def test_evaluate_rounds_passes_when_exactly_the_lost_partitions_go_down():
    fails = {0: _fails([(66, 134)]), 1: _fails([(66, 70)]), 2: _fails([])}     # 68 s lost; 4 s leader blip; untouched
    v = ST.evaluate_rounds(_round(), [(62.0, 122.0)], [{0}], fails, buckets=3)
    assert v["passed"] and v["rounds"][0]["lost_down_s"] == {"min": 68, "max": 68}
    assert v["rounds"][0]["kept_down_s_max"] == 4


@pytest.mark.parametrize("fails, why", [
    ({0: _fails([(90, 125)]), 1: _fails([])}, "lost partition down far shorter than the hold (graceful stop, G3)"),
    ({0: _fails([(66, 134)]), 1: _fails([(95, 135)])}, "a partition that kept quorum is down 40 s (interference, G3)"),
    ({0: _fails([]), 1: _fails([])}, "the lost partition never went down"),
])
def test_evaluate_rounds_fails_on_mismatches(fails, why):
    assert not ST.evaluate_rounds(_round(), [(62.0, 122.0)], [{0}], fails, buckets=2)["passed"], why


def test_evaluate_rounds_leaves_out_partitions_whose_region_moved():
    fails = {0: _fails([(66, 134)]), 1: _fails([(66, 134)])}                  # bucket 1 moved around the kill
    v = ST.evaluate_rounds(_round(), [(62.0, 122.0)], [{0}], fails, buckets=2, moving_by_round=[{1}])
    assert v["passed"] and v["rounds"][0]["moving_buckets"] == [1]


def test_probe_regions_finds_the_region_of_each_probe_row():
    m = real_map()
    table_id = 31
    regs = ST.probe_regions(m, table_id, keys=1000, buckets=4)
    assert regs and all(any(r.contains(ST.row_key(table_id, int((b + 0.5) * 250))) for r in regs) for b in range(4))
