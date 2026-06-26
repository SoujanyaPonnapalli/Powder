"""Regression tests for Raft Markov election-rate normalization."""

from __future__ import annotations

import pytest

from powder.markov import MarkovModel
from powder.markov_builders import group_configs_into_classes
from powder.markov_builders.raft import build_raft_model
from powder.scenario import QualityLevel
from powder.simulation import (
    Constant,
    Exponential,
    NodeConfig,
    NodeReplacementStrategy,
    RaftLikeProtocol,
    Seconds,
)
from powder.simulation.distributions import days, hours, minutes


_K_BY_QUALITY = {
    QualityLevel.SIMPLIFIED: 3,
    QualityLevel.COLLAPSED_PIPELINE: 4,
    QualityLevel.NO_ORPHANS: 6,
    QualityLevel.MERGED_PIPELINE: 8,
    QualityLevel.FULL: 12,
}

_ELIGIBLE_OFFSETS_BY_QUALITY = {
    QualityLevel.SIMPLIFIED: (0,),
    QualityLevel.COLLAPSED_PIPELINE: (0,),
    QualityLevel.NO_ORPHANS: (0,),
    QualityLevel.MERGED_PIPELINE: (0, 1),
    QualityLevel.FULL: (0, 1, 2),
}


def _config(failure_hours: float, region: str) -> NodeConfig:
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


def _state_tuple(model: MarkovModel, state_id: int) -> tuple[int, ...]:
    return tuple(int(part) for part in model.state_names[state_id].split(":"))


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_raft_no_leader_election_exit_rate_is_single_cluster_clock(
    quality: QualityLevel,
):
    """Candidate count changes election outcome probabilities, not election speed."""
    configs = [
        _config(12.0, "us-east"),
        _config(18.0, "us-west"),
        _config(24.0, "eu-central"),
    ]
    strategy = _strategy()
    mu_election = 0.25
    protocol = RaftLikeProtocol(election_time_dist=Exponential(rate=mu_election))
    model = build_raft_model(configs, protocol, strategy, quality)

    classes = group_configs_into_classes(configs, strategy, quality)
    class_count = len(classes)
    k = _K_BY_QUALITY[quality]
    has_leader_idx = class_count * k
    quorum = len(configs) // 2 + 1
    live_state_ids = model.live_state_ids

    checked_states = 0
    for state_id in range(model.num_states):
        state = _state_tuple(model, state_id)
        has_leader = state[has_leader_idx]
        if has_leader:
            continue

        eligible = 0
        for class_idx in range(class_count):
            offset = class_idx * k
            eligible += sum(
                state[offset + eligible_offset]
                for eligible_offset in _ELIGIBLE_OFFSETS_BY_QUALITY[quality]
            )

        if eligible < quorum:
            continue

        checked_states += 1
        election_exit_rate = model.Q[state_id, live_state_ids].sum()
        assert election_exit_rate == pytest.approx(mu_election)

    assert checked_states > 0
