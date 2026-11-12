"""Shared-event request availability: single shard versus every shard at once."""

import pytest

from powder.faults.composed_availability import CandidateTimeline, merge_timelines


def test_shared_pd_outage_counts_once_for_single_and_full():
    timeline = CandidateTimeline((10., 10.), (True, False),
                                 {f"s{i}": {(f"m{i}",): (True, True)} for i in range(300)})
    result = timeline.compose({f"s{i}": (f"m{i}",) for i in range(300)})
    assert all(v == 0.5 for v in result.per_shard.values())
    assert result.single_access == 0.5 and result.full_access == 0.5
    assert result.global_access == 0.5


def test_one_shard_failure_lowers_full_access_more_than_uniform_single():
    timeline = CandidateTimeline((10., 10.), (True, True),
                                 {"a": {("m1",): (True, False)},
                                  "b": {("m2",): (True, True)},
                                  "c": {("m3",): (True, True)}})
    dep = {"a": ("m1",), "b": ("m2",), "c": ("m3",)}
    result = timeline.compose(dep)
    assert result.per_shard == {"a": 0.5, "b": 1.0, "c": 1.0}
    assert result.single_access == pytest.approx(5 / 6)
    assert result.full_access == 0.5
    assert timeline.compose(dep, {"a": 0.5, "b": 0.25, "c": 0.25}).single_access == 0.75


def test_overlap_is_union_not_sum_or_product():
    timeline = CandidateTimeline((1., 2., 3.), (False, True, True),
                                 {"a": {("a",): (False, False, True)},
                                  "b": {("b",): (False, False, True)}})
    result = timeline.compose({"a": ("a",), "b": ("b",)})
    assert result.global_access == pytest.approx(5 / 6)
    assert result.single_access == result.full_access == 0.5


def test_merging_whole_deployment_runs_weights_by_elapsed_time():
    good = CandidateTimeline((2.,), (True,), {"a": {("m",): (True,)}})
    bad = CandidateTimeline((8.,), (False,), {"a": {("m",): (True,)}})
    result = merge_timelines((good, bad)).compose({"a": ("m",)})
    assert result.horizon_seconds == 10.0 and result.single_access == result.full_access == 0.2
    with pytest.raises(ValueError, match="whole-fleet"):
        merge_timelines(())


def test_incomplete_or_malformed_masks_fail_closed():
    with pytest.raises(ValueError, match="interval seconds"):
        CandidateTimeline((0.,), (True,), {"a": {("a",): (True,)}})
    with pytest.raises(ValueError, match="boolean"):
        CandidateTimeline((1.,), (1,), {"a": {("a",): (True,)}})
    t = CandidateTimeline((1.,), (True,), {"a": {("a",): (True,)}})
    with pytest.raises(ValueError, match="every and only"):
        t.compose({"wrong": ("a",)})
    with pytest.raises(ValueError, match="sum to one"):
        t.compose({"a": ("a",)}, {"a": 0.5})
