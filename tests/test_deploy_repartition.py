"""Repartition detection and summaries against hand-built Region-map sequences (exec #7 step 2).

Stores: 1 -> us-east-1, 2 -> us-east-2, 3 -> us-west-2, 4 -> us-east-1, 5 -> us-west-2.
"""

import pytest

from powder.deploy import repartition as R
from powder.deploy.regionmap import Region, RegionMap, RegionMapSeries, StoreInfo, row_key

K500, K700 = row_key(100, 500), row_key(100, 700)
STORES = {1: StoreInfo(1, region="us-east-1"), 2: StoreInfo(2, region="us-east-2"),
          3: StoreInfo(3, region="us-west-2"), 4: StoreInfo(4, region="us-east-1"),
          5: StoreInfo(5, region="us-west-2")}


def rmap(ts, regions):
    return RegionMap(ts, regions, STORES)


def reg(i, start, end, voters, leader, learners=(), size=96.0, down=(), pending=()):
    return Region(i, start, end, tuple(voters), tuple(learners), leader, size, tuple(down), tuple(pending))


def story():
    r1 = lambda v=(1, 2, 3), l=1: reg(1, b"", K500, v, l)                 # noqa: E731
    return [
        rmap(0, [r1(), reg(2, K500, b"", (1, 2, 3), 2)]),
        rmap(10, [r1(), reg(2, K500, b"", (1, 2, 3), 2)]),                                          # nothing happens
        rmap(20, [r1(), reg(2, K500, K700, (1, 2, 3), 2), reg(3, K700, b"", (1, 2, 3), 2)]),         # split at K700
        rmap(30, [r1(), reg(2, K500, K700, (1, 2, 3), 2),
                  reg(3, K700, b"", (1, 2, 3), 2, learners=(4,), pending=(4,))]),                    # learner 4 added
        rmap(40, [r1(), reg(2, K500, K700, (1, 2, 3), 2), reg(3, K700, b"", (1, 2, 4), 2)]),         # 4 promoted, 3 removed
        rmap(50, [r1((1, 2, 5)), reg(2, K500, K700, (1, 2, 3), 2), reg(3, K700, b"", (1, 2, 4), 2)]),  # R1: 3 -> 5
        rmap(60, [r1((1, 2, 5), 2), reg(2, K500, K700, (1, 2, 3), 2), reg(3, K700, b"", (1, 2, 4), 2)]),  # leader 1 -> 2
        rmap(70, [r1((1, 2, 5), 2), reg(9, K500, b"", (1, 2, 4), 2)]),                              # merge back
    ]


def test_each_known_event_is_detected_once_with_its_window():
    ev = R.events_from_series(RegionMapSeries(story()))
    got = [(e.kind, e.t_start, e.t_end, e.from_store, e.to_store) for e in ev]
    assert got == [
        ("split", 10, 20, None, None),
        ("add_peer", 20, 30, None, 4),
        ("remove_peer", 30, 40, 3, None),
        ("move_peer", 40, 50, 3, 5),
        ("transfer_leader", 50, 60, 1, 2),
        ("merge", 60, 70, None, None),
    ]
    split = ev[0]
    assert split.key == K700 and (split.range_start, split.range_end) == (K500, b"")     # the parent Region
    assert ev[-1].key == K700 and ev[-1].range_start == K500                             # merged Region


def test_unchanged_snapshots_produce_no_events_and_ids_do_not_matter():
    a = rmap(0, [reg(1, b"", K500, (1, 2, 3), 1), reg(2, K500, b"", (1, 2, 3), 2)])
    b = rmap(10, [reg(77, b"", K500, (1, 2, 3), 1), reg(78, K500, b"", (1, 2, 3), 2)])   # ids reassigned
    assert R.diff_maps(a, b) == []


def test_data_movement_is_attributed_intra_vs_inter_region():
    ev = {e.kind: e for e in R.events_from_series(RegionMapSeries(story()))}
    assert ev["add_peer"].inter_region is True       # leader store 2 (us-east-2) -> store 4 (us-east-1)
    assert ev["move_peer"].inter_region is True      # leader 1 (us-east-1) -> store 5 (us-west-2)
    # same-region copy: leader store 1 (us-east-1) -> store 4 (us-east-1)
    a = rmap(0, [reg(1, b"", b"", (1, 2, 3), 1)])
    b = rmap(10, [reg(1, b"", b"", (1, 2, 3), 1, learners=(4,))])
    assert R.diff_maps(a, b)[0].inter_region is False
    unknown = RegionMap(10, [reg(1, b"", b"", (1, 2, 3), 1, learners=(9,))], {})
    assert R.diff_maps(RegionMap(0, [reg(1, b"", b"", (1, 2, 3), 1)], {}), unknown)[0].inter_region is None


def test_summary_counts_rates_gb_and_membership_changes():
    s = R.summarize(RegionMapSeries(story()), 3, (0, 70), inter_region_usd_per_gb=0.02)
    assert s["counts"] == {"split": 1, "merge": 1, "add_peer": 1, "remove_peer": 1, "move_peer": 1,
                           "peer_churn": 0, "transfer_leader": 1}
    assert s["rates_per_hour"]["split"] == pytest.approx(3600 / 70)
    assert s["region_count"] == {"start": 2, "end": 2, "min": 2, "max": 3}
    gb = s["gb_moved"]
    assert gb["inter_region"] == pytest.approx(192 / 1024) and gb["intra_region"] == 0 and gb["unknown"] == 0
    assert gb["inter_region_usd"] == pytest.approx(192 / 1024 * 0.02)
    # add_peer, remove_peer and move_peer all happened to Regions whose voters were (1,2,3) just before
    assert s["per_rsm_membership_changes"] == {"1,2,3": 3}
    assert s["region_maps"]["snapshots"] == 8 and s["region_maps"]["median_interval_s"] == 10
    assert s["operators"] is None and "cross_check" not in s


def test_back_to_back_events_form_one_episode_covering_the_window():
    s = R.summarize(RegionMapSeries(story()), 3, (0, 70))
    ep = s["episodes_region_maps"]
    assert ep["count"] == 1 and ep["episodes"][0]["start"] == 10 and ep["episodes"][0]["end"] == 70
    assert ep["episodes"][0]["events"] == 6
    assert ep["busy_seconds"] == 60 and ep["busy_share"] == pytest.approx(60 / 70)


def test_episode_gap_rule():
    ivals = [(0, 10), (100, 110), (150, 160)]
    s = R.episode_summary(ivals, 200, gap_s=60)
    assert s["count"] == 2                                    # 100..160 merge (40 s gap), 0..10 stays apart
    assert [(e["start"], e["end"], e["events"]) for e in s["episodes"]] == [(0, 10, 1), (100, 160, 2)]
    assert s["busy_seconds"] == 30 and s["busy_share"] == pytest.approx(0.15)   # idle gap not counted as busy
    assert R.episode_summary(ivals, 200, gap_s=10)["count"] == 3
    assert R.episode_summary([(0, 10), (5, 20)], 100)["busy_seconds"] == 20     # overlaps are not double counted
    empty = R.episode_summary([], 100)
    assert empty["count"] == 0 and empty["busy_share"] == 0 and empty["duration_s"]["max"] is None


def test_under_replication_time_restore_time_and_censoring():
    A_, B_ = (b"", K500), (K500, b"")
    series = RegionMapSeries([
        rmap(0, [reg(1, *A_, (1, 2, 3), 1), reg(2, *B_, (1, 2, 4), 1)]),
        rmap(10, [reg(1, *A_, (1, 2, 3), 1, down=(3,)), reg(2, *B_, (1, 2, 4), 1)]),
        rmap(20, [reg(1, *A_, (1, 2, 3), 1, down=(3,)), reg(2, *B_, (1, 2, 4), 1)]),
        rmap(30, [reg(1, *A_, (1, 2, 3), 1), reg(2, *B_, (1, 2, 4), 1, pending=(4,))]),
        rmap(40, [reg(1, *A_, (1, 2, 3), 1), reg(2, *B_, (1, 2, 4), 1, pending=(4,))]),
    ])
    u = R.under_replication(series, 3, window_end=50)
    assert u["region_seconds"] == 40                       # A: 10..30, B: 30..50
    assert u["any_seconds"] == 40 and u["any_share"] == pytest.approx(40 / 50)
    assert u["restore_times_s"] == [20]                    # A was flagged at 10, clear at 30
    assert u["unresolved"] == 1                            # B still catching up at the end: censored, not restored
    assert u["by_rsm_region_seconds"] == {"1,2,3": 20, "1,2,4": 20}


def test_healthy_cluster_has_zero_under_replication():
    u = R.under_replication(RegionMapSeries(story()), 3, window_end=70)
    assert u["region_seconds"] == 0 and u["restore_times_s"] == [] and u["unresolved"] == 0 and u["any_share"] == 0


def op(kind, create, finish=None, outcome="finished", trigger="balance-region", region=3):
    return R.OperatorEvent(kind, region, create, create, finish, outcome, trigger)


def test_operator_summary_counts_durations_outcomes_and_triggers():
    ops = [op("add_peer", 0, 12), op("add_peer", 100, 130), op("add_peer", 200, None, "timeout"),
           op("transfer_leader", 5, 5.2, trigger="balance-leader"), op("merge", 300, 310, trigger="merge-checker")]
    s = R.summarize_operators(ops, 3600)
    assert s["count"] == 5 and s["by_kind"] == {"add_peer": 3, "transfer_leader": 1, "merge": 1}
    assert s["by_outcome"] == {"finished": 4, "timeout": 1}
    assert s["by_trigger"] == {"balance-region": 3, "balance-leader": 1, "merge-checker": 1}
    assert s["duration_s_by_kind"]["add_peer"]["n"] == 2 and s["duration_s_by_kind"]["add_peer"]["max"] == 30
    assert "timeout" not in s["duration_s_by_kind"]                    # unfinished ops have no duration
    assert s["rate_per_hour_by_kind"]["add_peer"] == pytest.approx(3.0)
    assert op("add_peer", 0, 12).duration == 12 and op("add_peer", 7, None, "timeout").duration is None


def test_cross_check_compares_both_sources():
    ops = [op("split", 15, 15.1), op("move_peer", 42, 47)]
    s = R.summarize(RegionMapSeries(story()), 3, (0, 70), operators=ops)
    assert s["cross_check"]["split"] == {"region_maps": 1, "operators": 1}
    assert s["cross_check"]["move_peer"] == {"region_maps": 1, "operators": 1}
    assert s["cross_check"]["merge"] == {"region_maps": 1, "operators": 0}


def test_coverage_problems_are_counted():
    broken = rmap(10, [reg(1, b"", K500, (1, 2, 3), 1), reg(2, K700, b"", (1, 2, 3), 1)])   # gap K500..K700
    assert "gap" in broken.coverage_problems()[0]
    s = R.summarize(RegionMapSeries([story()[0], broken]), 3, (0, 10))
    assert s["region_maps"]["coverage_problems"] == 1


def test_index_flags_keys_inside_event_windows_only():
    ev = R.events_from_series(RegionMapSeries(story()))
    idx = R.RepartitionIndex(events=ev)
    r2, r3, r1 = row_key(100, 600), row_key(100, 800), row_key(100, 100)
    assert idx.during(15, r2) is True                         # split window [10,20] covers the parent K500..end
    assert idx.during(25, r2) is False                        # the add_peer at [20,30] hit the OTHER child (K700..)
    assert idx.during(25, r3) is True and idx.during(35, r3) is True      # add_peer [20,30], remove_peer [30,40]
    assert idx.during(5, r3) is False                         # before anything happened
    assert idx.during(25, r1) is False                        # R1 untouched at that time
    assert idx.during(45, r1) is True                         # move_peer of R1 in [40,50]
    assert idx.during(55, r1) is True                         # leader transfer of R1 in [50,60]
    assert idx.during(62, r1) is True                         # leader-transfer pad of 3 s extends past 60
    assert idx.during(64, r1) is False


def test_index_matches_operators_by_region_id():
    series = RegionMapSeries(story())
    idx = R.RepartitionIndex(operators=[op("add_peer", 22, 28, region=3)], region_for_key=R.region_for_key_fn(series))
    assert idx.during(25, row_key(100, 800)) is True          # key is in Region 3 at t=25
    assert idx.during(25, row_key(100, 100)) is False         # Region 1
    assert idx.during(40, row_key(100, 800)) is False         # operator finished


def test_membership_change_that_reverts_inside_one_interval_is_seen_as_peer_churn():
    # add + remove of the same store between two snapshots: identical peer sets, but conf_ver proves it happened
    a = rmap(0, [Region(1, b"", b"", (1, 2, 3), (), 1, 96.0, conf_ver=17)])
    b = rmap(10, [Region(1, b"", b"", (1, 2, 3), (), 1, 96.0, conf_ver=20)])
    ev = R.diff_maps(a, b)
    assert [(e.kind, e.conf_delta) for e in ev] == [("peer_churn", 3)]
    assert R.diff_maps(a, rmap(10, [Region(1, b"", b"", (1, 2, 3), (), 1, 96.0, conf_ver=17)])) == []   # no change, no event
    s = R.summarize(RegionMapSeries([a, b]), 3, (0, 10))
    assert s["counts"]["peer_churn"] == 1 and s["counts"]["add_peer"] == 0


def test_cross_check_counts_tikv_initiated_splits_from_the_log():
    # TiKV splits Regions on size; PD only logs "region batch split" (no operator), so summarize takes that count
    ops = [op("move_peer", 42, 47)]
    s = R.summarize(RegionMapSeries(story()), 3, (0, 70), operators=ops, log_splits=1)
    assert s["cross_check"]["split"] == {"region_maps": 1, "operators": 1}
    # a merge logs two operators (source canceled, target finished): only finished ones are counted
    merge_pair = [op("merge", 60, 70, region=9), op("merge", 60, 70, "canceled", region=3)]
    s2 = R.summarize(RegionMapSeries(story()), 3, (0, 70), operators=merge_pair)
    assert s2["cross_check"]["merge"] == {"region_maps": 1, "operators": 1}
    assert s2["operators"]["by_kind"]["merge"] == 2 and s2["operators"]["by_kind_finished"]["merge"] == 1
