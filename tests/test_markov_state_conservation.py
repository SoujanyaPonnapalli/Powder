"""
State-space invariant tests for Markov builders.

Every builder models a fixed cluster of exactly N node-slots. The total
node count implied by a state tuple must stay equal to N across every
reachable state and every transition, regardless of protocol, quality
level, or whether a replacement strategy is active.

The Raft builders encode the leader via a ``has_leader`` flag (plus
``leader_orphan`` in MERGED_PIPELINE and ``leader_pipe`` in FULL) that
acts as an on/off indicator for one leader slot. Non-leader counts live
in the leading indices; every _R / _P / _S subscript is a pipeline
annotation on an existing slot and must not inflate the count.

These tests protect against regressions where a "pipeline complete"
transition incorrectly adds a fresh node without decrementing anything
else -- which historically caused the BFS state space to grow without
bound under replacement strategies.
"""

from __future__ import annotations

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
    NoOpStrategy,
    RaftLikeProtocol,
    Seconds,
)
from powder.simulation.distributions import hours, minutes


# ---------------------------------------------------------------------------
# State-tuple → implied node count
# ---------------------------------------------------------------------------


def _leaderless_count(state: tuple[int, ...]) -> int:
    """Every leaderless index is a per-node slot count."""
    return sum(state)


def _raft_count(state: tuple[int, ...], quality: QualityLevel) -> int:
    """Sum non-leader slot counts plus the leader flag.

    The trailing bits are boolean/phase annotations on the leader, never
    additional slots:

        SIMPLIFIED / COLLAPSED_PIPELINE / NO_ORPHANS:
            state = (..., has_leader)
        MERGED_PIPELINE:
            state = (..., has_leader, leader_orphan)
        FULL:
            state = (..., has_leader, leader_pipe)
    """
    trailing_flags = {
        QualityLevel.SIMPLIFIED: 1,
        QualityLevel.COLLAPSED_PIPELINE: 1,
        QualityLevel.NO_ORPHANS: 1,
        QualityLevel.MERGED_PIPELINE: 2,
        QualityLevel.FULL: 2,
    }[quality]
    body = state[:-trailing_flags]
    has_leader = state[-trailing_flags]
    return sum(body) + has_leader


# ---------------------------------------------------------------------------
# Common scenario builders
# ---------------------------------------------------------------------------


def _config() -> NodeConfig:
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(2)),
        recovery_dist=Exponential(rate=1.0 / minutes(5)),
        data_loss_dist=Constant(float("inf")),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Exponential(rate=1.0 / 120.0),
    )


def _strategies(cfg: NodeConfig):
    yield "noop", NoOpStrategy()
    yield "replacement", NodeReplacementStrategy(
        failure_timeout=Seconds(minutes(5)),
        default_node_config=cfg,
    )


# ---------------------------------------------------------------------------
# Conservation tests
# ---------------------------------------------------------------------------


RAFT_QUALITIES = list(QualityLevel)
LEADERLESS_QUALITIES = list(QualityLevel)


@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("quality", RAFT_QUALITIES, ids=lambda q: q.name)
def test_raft_every_reachable_state_preserves_n(n: int, quality: QualityLevel):
    """Every state enumerated by the Raft BFS must imply exactly n nodes."""
    cfg = _config()
    prot = RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 5.0))
    for label, strat in _strategies(cfg):
        model = build_raft_model([cfg] * n, prot, strat, quality)
        bad = [
            (name, _raft_count_from_name(name, quality))
            for name in model.state_names
            if _raft_count_from_name(name, quality) != n
        ]
        assert not bad, (
            f"raft/{quality.name}/{label}: {len(bad)} states do not have "
            f"exactly {n} nodes. First offenders (state -> count): {bad[:5]}"
        )


@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("quality", LEADERLESS_QUALITIES, ids=lambda q: q.name)
def test_leaderless_every_reachable_state_preserves_n(
    n: int, quality: QualityLevel,
):
    cfg = _config()
    prot = LeaderlessProtocol(up_to_date_quorum=False)
    for label, strat in _strategies(cfg):
        model = build_leaderless_model([cfg] * n, prot, strat, quality)
        bad = [
            (name, sum(int(x) for x in name.split(":")))
            for name in model.state_names
            if sum(int(x) for x in name.split(":")) != n
        ]
        assert not bad, (
            f"leaderless/{quality.name}/{label}: {len(bad)} states do not "
            f"have exactly {n} nodes. First offenders: {bad[:5]}"
        )


def _raft_count_from_name(name: str, quality: QualityLevel) -> int:
    parts = tuple(int(x) for x in name.split(":"))
    return _raft_count(parts, quality)


# ---------------------------------------------------------------------------
# Bounded-size sanity: replacement-strategy builds must terminate small
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality", RAFT_QUALITIES, ids=lambda q: q.name)
def test_raft_replacement_state_space_is_bounded(quality: QualityLevel):
    """Replacement strategy must not cause unbounded BFS growth.

    Before the node-conservation fix the MERGED_PIPELINE and FULL
    builders grew without bound under replacement because some
    "pipeline complete" transitions added a fresh H slot instead of
    flipping the leader's sub-state. This test pins a generous upper
    bound on the reachable state count so any future regression fails
    fast rather than hanging.
    """
    cfg = _config()
    prot = RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 5.0))
    strat = NodeReplacementStrategy(
        failure_timeout=Seconds(minutes(5)),
        default_node_config=cfg,
    )
    model = build_raft_model([cfg] * 3, prot, strat, quality)

    # Generous ceilings based on the per-node sub-state counts in the
    # docstring: k_nl=12 + 3 leader sub-states for FULL gives a dense
    # upper bound well under 10k for n=3.
    ceiling = {
        QualityLevel.SIMPLIFIED: 100,
        QualityLevel.COLLAPSED_PIPELINE: 400,
        QualityLevel.NO_ORPHANS: 600,
        QualityLevel.MERGED_PIPELINE: 2_000,
        QualityLevel.FULL: 10_000,
    }[quality]
    assert model.num_states <= ceiling, (
        f"raft/{quality.name} produced {model.num_states} states (limit "
        f"{ceiling}); the node-conservation invariant may be broken."
    )
