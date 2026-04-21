"""
Per-class state-conservation tests for the heterogeneous builders.

Extends ``tests/test_markov_state_conservation.py`` to check that:

  * Total node count (``sum(per-class counts) + has_leader``) equals
    ``N`` in every reachable state of a multi-class cluster.
  * Per-class counts stay within their class size for every reachable
    state: ``sum(class_c counts) <= size_c`` (with equality when the
    leader is not in class ``c``, and ``size_c - 1`` when it is).

Replacement strategy is exercised because the MERGED_PIPELINE and FULL
builders emit the most intricate transitions there.
"""

from __future__ import annotations

import pytest

from powder.markov_builders import group_configs_into_classes
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


K_BY_QUALITY = {
    QualityLevel.SIMPLIFIED: 3,
    QualityLevel.COLLAPSED_PIPELINE: 4,
    QualityLevel.NO_ORPHANS: 6,
    QualityLevel.MERGED_PIPELINE: 8,
    QualityLevel.FULL: 12,
}

RAFT_TRAILING = {
    QualityLevel.SIMPLIFIED: 2,
    QualityLevel.COLLAPSED_PIPELINE: 2,
    QualityLevel.NO_ORPHANS: 2,
    QualityLevel.MERGED_PIPELINE: 3,
    QualityLevel.FULL: 3,
}


def _make_config(
    *,
    failure_hours: float = 12.0,
    region: str = "us-east",
    cost_per_hour: float = 1.0,
) -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=cost_per_hour,
        failure_dist=Exponential(rate=1.0 / hours(failure_hours)),
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Exponential(rate=1.0 / days(365)),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Constant(300),
    )


def _strategy() -> NodeReplacementStrategy:
    return NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))


def _hetero_configs_2class(n0: int, n1: int) -> list[NodeConfig]:
    return [_make_config(failure_hours=12.0)] * n0 + [
        _make_config(failure_hours=36.0, region="us-west")
    ] * n1


def _hetero_configs_3class(n0: int, n1: int, n2: int) -> list[NodeConfig]:
    return (
        [_make_config(failure_hours=12.0)] * n0
        + [_make_config(failure_hours=24.0, region="us-west")] * n1
        + [_make_config(failure_hours=48.0, region="eu-west")] * n2
    )


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
@pytest.mark.parametrize(
    "configs, label",
    [
        (_hetero_configs_2class(2, 1), "2class_2_1"),
        (_hetero_configs_2class(3, 2), "2class_3_2"),
        (_hetero_configs_3class(2, 2, 1), "3class_2_2_1"),
    ],
    ids=lambda arg: arg if isinstance(arg, str) else "cfgs",
)
def test_leaderless_hetero_preserves_total_n(
    configs: list[NodeConfig], label: str, quality: QualityLevel,
):
    n = len(configs)
    model = build_leaderless_model(
        configs,
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        quality,
    )
    bad = [
        (name, sum(int(x) for x in name.split(":")))
        for name in model.state_names
        if sum(int(x) for x in name.split(":")) != n
    ]
    assert not bad, (
        f"leaderless/{quality.name}/{label}: {len(bad)} states do not have "
        f"exactly {n} nodes. First offenders: {bad[:5]}"
    )


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
@pytest.mark.parametrize(
    "configs, label",
    [
        (_hetero_configs_2class(2, 1), "2class_2_1"),
        (_hetero_configs_2class(3, 2), "2class_3_2"),
        (_hetero_configs_3class(2, 2, 1), "3class_2_2_1"),
    ],
    ids=lambda arg: arg if isinstance(arg, str) else "cfgs",
)
def test_raft_hetero_preserves_total_n(
    configs: list[NodeConfig], label: str, quality: QualityLevel,
):
    n = len(configs)
    trailing = RAFT_TRAILING[quality]
    model = build_raft_model(
        configs,
        RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
        _strategy(),
        quality,
    )
    bad: list[tuple[str, int]] = []
    for name in model.state_names:
        parts = tuple(int(x) for x in name.split(":"))
        body = parts[:-trailing]
        has_leader = parts[-trailing]
        total = sum(body) + has_leader
        if total != n:
            bad.append((name, total))
    assert not bad, (
        f"raft/{quality.name}/{label}: {len(bad)} states do not have "
        f"exactly {n} nodes. First offenders: {bad[:5]}"
    )


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_leaderless_per_class_counts_respect_class_size(quality: QualityLevel):
    configs = _hetero_configs_2class(3, 2)
    classes = group_configs_into_classes(
        configs, _strategy(), quality,
    )
    k = K_BY_QUALITY[quality]
    model = build_leaderless_model(
        configs, LeaderlessProtocol(up_to_date_quorum=False), _strategy(), quality,
    )
    for name in model.state_names:
        parts = tuple(int(x) for x in name.split(":"))
        for rc in classes:
            cls = parts[rc.class_idx * k : (rc.class_idx + 1) * k]
            assert sum(cls) == rc.size, (
                f"{quality.name}: class {rc.class_idx} total {sum(cls)} "
                f"!= size {rc.size} in state {name}"
            )
            assert all(c >= 0 for c in cls), f"negative count in {name}"


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_raft_per_class_counts_plus_leader_match_class_size(quality: QualityLevel):
    """In Raft, class ``c`` has ``size_c`` slots; one of them may be the leader."""
    configs = _hetero_configs_2class(3, 2)
    classes = group_configs_into_classes(
        configs, _strategy(), quality,
    )
    k = K_BY_QUALITY[quality]
    trailing = RAFT_TRAILING[quality]
    model = build_raft_model(
        configs,
        RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
        _strategy(),
        quality,
    )
    for name in model.state_names:
        parts = tuple(int(x) for x in name.split(":"))
        has_leader = parts[-trailing]
        leader_class = parts[-trailing + 1]
        for rc in classes:
            cls = parts[rc.class_idx * k : (rc.class_idx + 1) * k]
            leader_here = 1 if has_leader == 1 and leader_class == rc.class_idx else 0
            assert sum(cls) + leader_here == rc.size, (
                f"{quality.name}: class {rc.class_idx} "
                f"sum={sum(cls)}+leader={leader_here} != size={rc.size} "
                f"in state {name}"
            )


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_raft_no_leader_state_has_canonical_leader_class(quality: QualityLevel):
    """``has_leader == 0`` implies ``leader_class == 0`` (canonical form)."""
    configs = _hetero_configs_2class(3, 2)
    trailing = RAFT_TRAILING[quality]
    model = build_raft_model(
        configs,
        RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
        _strategy(),
        quality,
    )
    for name in model.state_names:
        parts = tuple(int(x) for x in name.split(":"))
        has_leader = parts[-trailing]
        leader_class = parts[-trailing + 1]
        if has_leader == 0:
            assert leader_class == 0, (
                f"{quality.name}: no-leader state {name} has "
                f"non-canonical leader_class={leader_class}"
            )
