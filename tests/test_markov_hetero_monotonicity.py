"""
Monotonicity sanity checks for the heterogeneous builders.

Replacing one node with a strictly worse configuration (higher
failure rate or lower recovery rate) should not improve availability
and should not increase mean-time-to-unavailability. These tests
are probabilistic-in-spirit (they hold for the *true* Markov chain)
but deterministic in the solver: any improvement is evidence of a
modeling bug.
"""

from __future__ import annotations

import pytest

from powder.results import markov_analyze
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


def _cfg(
    *,
    failure_hours: float = 12.0,
    recovery_minutes: float = 10.0,
    region: str = "us-east",
) -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(failure_hours)),
        recovery_dist=Exponential(rate=1.0 / minutes(recovery_minutes)),
        data_loss_dist=Exponential(rate=1.0 / days(365)),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Constant(300),
    )


def _strategy() -> NodeReplacementStrategy:
    return NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))


@pytest.mark.parametrize(
    "protocol_factory",
    [
        lambda: LeaderlessProtocol(up_to_date_quorum=False),
        lambda: RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
    ],
    ids=["leaderless", "raft"],
)
@pytest.mark.parametrize(
    "quality", [QualityLevel.SIMPLIFIED, QualityLevel.COLLAPSED_PIPELINE],
    ids=lambda q: q.name,
)
def test_worse_failure_rate_never_improves_availability(
    protocol_factory, quality: QualityLevel,
):
    good = _cfg(failure_hours=12.0)
    worse = _cfg(failure_hours=6.0, region="us-west")  # 2x failure rate
    baseline = markov_analyze(
        [good] * 3, protocol_factory(), _strategy(), quality,
    )
    degraded = markov_analyze(
        [good, good, worse], protocol_factory(), _strategy(), quality,
    )
    assert degraded.availability <= baseline.availability + 1e-12
    assert (
        degraded.mean_time_to_unavailability
        <= baseline.mean_time_to_unavailability + 1e-6
    )


@pytest.mark.parametrize(
    "protocol_factory",
    [
        lambda: LeaderlessProtocol(up_to_date_quorum=False),
        lambda: RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
    ],
    ids=["leaderless", "raft"],
)
def test_slower_recovery_never_improves_availability(protocol_factory):
    good = _cfg(recovery_minutes=10.0)
    worse = _cfg(recovery_minutes=60.0, region="us-west")  # 6x slower
    baseline = markov_analyze(
        [good] * 3, protocol_factory(), _strategy(), QualityLevel.SIMPLIFIED,
    )
    degraded = markov_analyze(
        [good, good, worse],
        protocol_factory(),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    assert degraded.availability <= baseline.availability + 1e-12
    assert (
        degraded.mean_time_to_unavailability
        <= baseline.mean_time_to_unavailability + 1e-6
    )


def test_strictly_worse_node_strictly_degrades_availability():
    """With a meaningfully worse node the gap should be noticeable."""
    good = _cfg(failure_hours=12.0, recovery_minutes=10.0)
    much_worse = _cfg(
        failure_hours=1.0,  # 12x
        recovery_minutes=60.0,
        region="us-west",
    )
    baseline = markov_analyze(
        [good] * 3,
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    degraded = markov_analyze(
        [good, good, much_worse],
        LeaderlessProtocol(up_to_date_quorum=False),
        _strategy(),
        QualityLevel.SIMPLIFIED,
    )
    # Availability must drop; demand at least a meaningful difference.
    assert degraded.availability < baseline.availability - 1e-4
