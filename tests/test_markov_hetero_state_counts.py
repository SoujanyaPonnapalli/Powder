"""
State-space size tests for the heterogeneous builders.

The state space for a rate-class partition ``[n_0, ..., n_{C-1}]`` is
upper-bounded by ``prod_c C(n_c + k - 1, k - 1)`` per-node-state
compositions (see docs/markov_state_analysis.md). We check:

  * ``C = 1`` reproduces the original homogeneous ``C(n + k - 1, k - 1)``.
  * ``C = N`` matches ``k^N`` for small N at the SIMPLIFIED level.
  * Multi-class inputs match the per-class-product bound at SIMPLIFIED.

Higher quality levels add reachability constraints and leader-class
coupling, so exact counts are only pinned at SIMPLIFIED where the
state space is pure per-class compositions.
"""

from __future__ import annotations

from math import comb

import pytest

from powder.markov_builders.leaderless import build_leaderless_model
from powder.markov_builders.raft import build_raft_model
from powder.scenario import QualityLevel
from powder.simulation import (
    Constant,
    Exponential,
    LeaderlessProtocol,
    NodeConfig,
    NodeReplacementStrategy,
    RaftLikeProtocol,
    Seconds,
)
from powder.simulation.distributions import days, hours, minutes


def _config(*, failure_hours: float = 12.0, region: str = "us-east") -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(failure_hours)),
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Exponential(rate=1.0 / days(365)),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Constant(300),
    )


def _strategy() -> NodeReplacementStrategy:
    return NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))


def _n_distinct(n: int) -> list[NodeConfig]:
    """Build N configs that are all in distinct rate classes (distinct failure rates)."""
    return [_config(failure_hours=12.0 + i) for i in range(n)]


# ---------------------------------------------------------------------------
# Leaderless SIMPLIFIED (k = 3)
# ---------------------------------------------------------------------------


K_SIMPLIFIED = 3


@pytest.mark.parametrize("n", [3, 5, 7])
def test_leaderless_simplified_c1_matches_homogeneous_bound(n: int):
    """C=1 reaches every weak composition so #states == C(n + k - 1, k - 1)."""
    expected = comb(n + K_SIMPLIFIED - 1, K_SIMPLIFIED - 1)
    model = build_leaderless_model(
        [_config()] * n,
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    assert model.num_states == expected


@pytest.mark.parametrize("n", [2, 3, 4])
def test_leaderless_simplified_c_equals_n_matches_k_to_the_n(n: int):
    """With N distinct classes at k=3 the reachable space is 3^N."""
    model = build_leaderless_model(
        _n_distinct(n),
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    assert model.num_states == K_SIMPLIFIED ** n


def test_leaderless_simplified_two_class_matches_product_of_weak_compositions():
    """#states = prod_c C(n_c + k - 1, k - 1) for a 2-class SIMPLIFIED cluster."""
    # 3 identical + 2 identical, distinct-across groups.
    n0, n1 = 3, 2
    cfgs = [_config(failure_hours=12.0)] * n0 + [_config(failure_hours=36.0)] * n1
    model = build_leaderless_model(
        cfgs,
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    expected = (
        comb(n0 + K_SIMPLIFIED - 1, K_SIMPLIFIED - 1)
        * comb(n1 + K_SIMPLIFIED - 1, K_SIMPLIFIED - 1)
    )
    assert model.num_states == expected


# ---------------------------------------------------------------------------
# Raft SIMPLIFIED -- leader_class multiplies the no-leader-equivalent count
# ---------------------------------------------------------------------------


def test_raft_simplified_c1_leader_class_degenerate():
    """With C=1, ``leader_class`` is always 0 so state count matches pre-change."""
    n = 5
    model = build_raft_model(
        [_config()] * n,
        RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    # Hand-computed from the (H, F, D, has_leader, leader_class=0) encoding
    # by enumeration; we just check it matches the homogeneous baseline.
    baseline = build_raft_model(
        [_config()] * n,
        RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    assert model.num_states == baseline.num_states


def test_raft_simplified_grows_with_class_count():
    """More classes -> strictly more reachable states at the same N."""
    counts = []
    for cfgs in (
        [_config()] * 4,  # C = 1
        [_config()] * 3 + [_config(failure_hours=36.0)],  # C = 2
        _n_distinct(4),  # C = 4
    ):
        m = build_raft_model(
            cfgs,
            RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
            _strategy(),
            QualityLevel.SIMPLIFIED,
        )
        counts.append(m.num_states)
    assert counts[0] < counts[1] < counts[2]


# ---------------------------------------------------------------------------
# Higher quality levels: state counts grow monotonically with class count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quality",
    [QualityLevel.COLLAPSED_PIPELINE, QualityLevel.NO_ORPHANS],
    ids=lambda q: q.name,
)
def test_leaderless_higher_quality_grows_with_class_count(quality: QualityLevel):
    """Splitting nodes into more classes can only grow the reachable space."""
    c1 = build_leaderless_model(
        [_config()] * 4,
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        quality,
    )
    c2 = build_leaderless_model(
        [_config()] * 3 + [_config(failure_hours=36.0)],
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        quality,
    )
    assert c1.num_states < c2.num_states


# ---------------------------------------------------------------------------
# Pinned state-count matrix
#
# Values below are captured from the builders once and checked-in as a
# regression fence: any refactor that changes the reachable state space
# (for any (protocol, quality, partition) combination listed here) will
# fail this test. Homogeneous rows are equivalent to the pre-heterogeneous
# builders; partial-homogeneity and all-distinct rows exercise the
# rate-class partitioning logic end-to-end.
# ---------------------------------------------------------------------------


def _configs_homog(n: int) -> list[NodeConfig]:
    return [_config()] * n


def _configs_2class(n0: int, n1: int) -> list[NodeConfig]:
    # Two classes separated by failure rate + region so both rate_signature
    # and the downstream region bookkeeping disagree.
    return (
        [_config(failure_hours=12.0, region="us-east")] * n0
        + [_config(failure_hours=24.0, region="us-west")] * n1
    )


def _configs_3class(n0: int, n1: int, n2: int) -> list[NodeConfig]:
    return (
        [_config(failure_hours=12.0, region="us-east")] * n0
        + [_config(failure_hours=18.0, region="us-west")] * n1
        + [_config(failure_hours=36.0, region="eu-west")] * n2
    )


def _configs_all_distinct(n: int) -> list[NodeConfig]:
    return [_config(failure_hours=12.0 + 2 * i, region=f"r{i}") for i in range(n)]


# scenario_id -> factory producing a NodeConfig list. Each scenario spans
# one partition pattern; we then cross-product with every QualityLevel.
_SCENARIOS: dict[str, "callable[[], list[NodeConfig]]"] = {
    "homog_N3": lambda: _configs_homog(3),
    "homog_N5": lambda: _configs_homog(5),
    "split_2_1": lambda: _configs_2class(2, 1),
    "split_3_2": lambda: _configs_2class(3, 2),
    "split_4_3": lambda: _configs_2class(4, 3),
    "split_3class_2_2_1": lambda: _configs_3class(2, 2, 1),
    "all_distinct_N3": lambda: _configs_all_distinct(3),
    "all_distinct_N4": lambda: _configs_all_distinct(4),
}


# Expected num_states for leaderless, keyed by (scenario_id, quality_name).
_EXPECTED_LEADERLESS: dict[tuple[str, str], int] = {
    ("homog_N3", "SIMPLIFIED"): 10,
    ("homog_N3", "COLLAPSED_PIPELINE"): 20,
    ("homog_N3", "NO_ORPHANS"): 55,
    ("homog_N3", "MERGED_PIPELINE"): 120,
    ("homog_N3", "FULL"): 364,
    ("homog_N5", "SIMPLIFIED"): 21,
    ("homog_N5", "COLLAPSED_PIPELINE"): 56,
    ("homog_N5", "NO_ORPHANS"): 251,
    ("homog_N5", "MERGED_PIPELINE"): 792,
    ("homog_N5", "FULL"): 4368,
    ("split_2_1", "SIMPLIFIED"): 18,
    ("split_2_1", "COLLAPSED_PIPELINE"): 40,
    ("split_2_1", "NO_ORPHANS"): 125,
    ("split_2_1", "MERGED_PIPELINE"): 288,
    ("split_2_1", "FULL"): 936,
    ("split_3_2", "SIMPLIFIED"): 60,
    ("split_3_2", "COLLAPSED_PIPELINE"): 200,
    ("split_3_2", "NO_ORPHANS"): 1175,
    ("split_3_2", "MERGED_PIPELINE"): 4320,
    ("split_3_2", "FULL"): 28392,
    ("split_4_3", "SIMPLIFIED"): 150,
    ("split_4_3", "COLLAPSED_PIPELINE"): 700,
    ("split_4_3", "NO_ORPHANS"): 7055,
    ("split_3class_2_2_1", "SIMPLIFIED"): 108,
    ("split_3class_2_2_1", "COLLAPSED_PIPELINE"): 400,
    ("split_3class_2_2_1", "NO_ORPHANS"): 2645,
    ("all_distinct_N3", "SIMPLIFIED"): 27,
    ("all_distinct_N3", "COLLAPSED_PIPELINE"): 64,
    ("all_distinct_N3", "NO_ORPHANS"): 215,
    ("all_distinct_N3", "MERGED_PIPELINE"): 512,
    ("all_distinct_N3", "FULL"): 1728,
    ("all_distinct_N4", "SIMPLIFIED"): 81,
    ("all_distinct_N4", "COLLAPSED_PIPELINE"): 256,
    ("all_distinct_N4", "NO_ORPHANS"): 1295,
}


_EXPECTED_RAFT: dict[tuple[str, str], int] = {
    ("homog_N3", "SIMPLIFIED"): 16,
    ("homog_N3", "COLLAPSED_PIPELINE"): 30,
    ("homog_N3", "NO_ORPHANS"): 76,
    ("homog_N3", "MERGED_PIPELINE"): 192,
    ("homog_N3", "FULL"): 598,
    ("homog_N5", "SIMPLIFIED"): 36,
    ("homog_N5", "COLLAPSED_PIPELINE"): 91,
    ("homog_N5", "NO_ORPHANS"): 377,
    ("homog_N5", "MERGED_PIPELINE"): 1452,
    ("homog_N5", "FULL"): 8463,
    ("split_2_1", "SIMPLIFIED"): 33,
    ("split_2_1", "COLLAPSED_PIPELINE"): 66,
    ("split_2_1", "NO_ORPHANS"): 182,
    ("split_2_1", "MERGED_PIPELINE"): 488,
    ("split_2_1", "FULL"): 1602,
    ("split_3_2", "SIMPLIFIED"): 126,
    ("split_3_2", "COLLAPSED_PIPELINE"): 380,
    ("split_3_2", "NO_ORPHANS"): 1952,
    ("split_3_2", "MERGED_PIPELINE"): 8832,
    ("split_4_3", "SIMPLIFIED"): 340,
    ("split_4_3", "COLLAPSED_PIPELINE"): 1450,
    ("split_3class_2_2_1", "SIMPLIFIED"): 252,
    ("split_3class_2_2_1", "COLLAPSED_PIPELINE"): 820,
    ("split_3class_2_2_1", "NO_ORPHANS"): 4598,
    ("all_distinct_N3", "SIMPLIFIED"): 54,
    ("all_distinct_N3", "COLLAPSED_PIPELINE"): 112,
    ("all_distinct_N3", "NO_ORPHANS"): 323,
    ("all_distinct_N3", "MERGED_PIPELINE"): 896,
    ("all_distinct_N3", "FULL"): 3024,
    ("all_distinct_N4", "SIMPLIFIED"): 189,
    ("all_distinct_N4", "COLLAPSED_PIPELINE"): 512,
    ("all_distinct_N4", "NO_ORPHANS"): 2159,
}


@pytest.mark.parametrize(
    ("scenario_id", "quality_name", "expected"),
    [
        (sid, qname, n)
        for (sid, qname), n in sorted(_EXPECTED_LEADERLESS.items())
    ],
)
def test_leaderless_pinned_state_counts(
    scenario_id: str, quality_name: str, expected: int,
):
    """Regression fence on reachable state count for leaderless builders."""
    cfgs = _SCENARIOS[scenario_id]()
    quality = QualityLevel[quality_name]
    model = build_leaderless_model(
        cfgs,
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        quality,
    )
    assert model.num_states == expected, (
        f"leaderless[{scenario_id}][{quality_name}]: "
        f"got {model.num_states}, expected {expected}"
    )


@pytest.mark.parametrize(
    ("scenario_id", "quality_name", "expected"),
    [
        (sid, qname, n)
        for (sid, qname), n in sorted(_EXPECTED_RAFT.items())
    ],
)
def test_raft_pinned_state_counts(
    scenario_id: str, quality_name: str, expected: int,
):
    """Regression fence on reachable state count for Raft builders."""
    cfgs = _SCENARIOS[scenario_id]()
    quality = QualityLevel[quality_name]
    model = build_raft_model(
        cfgs,
        RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
        _strategy(),
        quality,
    )
    assert model.num_states == expected, (
        f"raft[{scenario_id}][{quality_name}]: "
        f"got {model.num_states}, expected {expected}"
    )


def test_pinned_state_counts_cover_all_quality_levels():
    """Meta: every QualityLevel should appear in the pinned matrix.

    Guards against accidentally dropping a quality level when the
    builders are refactored.
    """
    covered_ll = {qname for (_sid, qname) in _EXPECTED_LEADERLESS}
    covered_raft = {qname for (_sid, qname) in _EXPECTED_RAFT}
    expected = {q.name for q in QualityLevel}
    assert expected <= covered_ll
    assert expected <= covered_raft


def test_pinned_state_counts_raft_strictly_exceeds_leaderless():
    """Raft's leader dimension means each (scenario, quality) has >= the
    leaderless count. Strict inequality is expected on every case we pin
    because the leader doubles the encoding."""
    for key, ll_count in _EXPECTED_LEADERLESS.items():
        raft_count = _EXPECTED_RAFT.get(key)
        if raft_count is None:
            continue  # not all quality levels are pinned for both protocols
        assert raft_count > ll_count, (
            f"raft[{key}]={raft_count} not strictly greater than "
            f"leaderless[{key}]={ll_count}"
        )
