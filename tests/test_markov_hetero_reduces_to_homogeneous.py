"""
Homogeneous-equivalence tests for the heterogeneous builders.

Every builder has been generalized to operate on per-class count
vectors. When the input ``list[NodeConfig]`` is uniform, the
partitioner returns a single rate class, the state tuple collapses
to the original homogeneous layout (with a trailing ``leader_class``
field that is always 0 for Raft), and every observable metric should
match the pre-change implementation bit-for-bit (modulo sub-ULP
floating-point noise).

Golden values below were captured from the pre-change git revision
using ``scripts``-less inline invocation of ``markov_analyze`` at
``N = 3`` for every ``(protocol, quality)`` combination.
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


def _config() -> NodeConfig:
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(12)),
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Exponential(rate=1.0 / days(365)),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Constant(300),
    )


def _strategy() -> NodeReplacementStrategy:
    return NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))


# Values captured from the pre-change implementation for N=3.
# Structure: (availability, mean_time_to_unavailability, expected_cost_per_hour, num_states)
_GOLDEN_LEADERLESS = {
    QualityLevel.SIMPLIFIED: (
        0.9994931940688163, 580341.2568635877, 2.9996233349693533, 10,
    ),
    QualityLevel.COLLAPSED_PIPELINE: (
        0.9995682338153856, 626594.4416595312, 2.999623334969353, 20,
    ),
    QualityLevel.NO_ORPHANS: (
        0.9995264452382667, 598848.6092977963, 2.999657828769315, 55,
    ),
    QualityLevel.MERGED_PIPELINE: (
        0.9995266894243281, 598992.1461947862, 2.999657979505005, 120,
    ),
    QualityLevel.FULL: (
        0.9995239798613347, 597301.5781190963, 2.9996579989870655, 364,
    ),
}

_GOLDEN_RAFT = {
    QualityLevel.SIMPLIFIED: (
        0.999261508963264, 42084.99478149899, 2.9996233349693533, 16,
    ),
    QualityLevel.COLLAPSED_PIPELINE: (
        0.9992474634560282, 42069.056579303164, 2.9996233349693524, 30,
    ),
    QualityLevel.NO_ORPHANS: (
        0.9991970031141639, 42020.12703586393, 2.9996578287693074, 76,
    ),
    QualityLevel.MERGED_PIPELINE: (
        0.9991972974205018, 42020.39033052767, 2.999657979505005, 192,
    ),
    QualityLevel.FULL: (
        0.9991940245021012, 42017.2653513525, 2.9996579989870646, 598,
    ),
}


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_leaderless_homogeneous_matches_golden(quality: QualityLevel):
    """Uniform input at C=1 must reproduce pre-change leaderless metrics."""
    protocol = LeaderlessProtocol(up_to_date_quorum=False)
    result = markov_analyze([_config()] * 3, protocol, _strategy(), quality)
    exp_avail, exp_mttu, exp_cost, exp_states = _GOLDEN_LEADERLESS[quality]
    assert result.num_states == exp_states
    assert result.availability == pytest.approx(exp_avail, rel=1e-9, abs=1e-12)
    assert result.mean_time_to_unavailability == pytest.approx(
        exp_mttu, rel=1e-9, abs=1e-9,
    )
    assert result.expected_cost_per_hour == pytest.approx(
        exp_cost, rel=1e-9, abs=1e-12,
    )


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_raft_homogeneous_matches_golden(quality: QualityLevel):
    """Uniform input at C=1 must reproduce pre-change Raft metrics.

    Mean-time-to-unavailability is held to a looser tolerance because
    the election transitions now decompose ``mu_election`` as
    ``mu_election * nH_c / total_H``. At C=1 ``nH_c == total_H`` so
    the sum still equals ``mu_election``, but the individual triples
    have different floating-point rounding.
    """
    protocol = RaftLikeProtocol(
        election_time_dist=Exponential(rate=1.0 / 10.0),
    )
    result = markov_analyze([_config()] * 3, protocol, _strategy(), quality)
    exp_avail, exp_mttu, exp_cost, exp_states = _GOLDEN_RAFT[quality]
    assert result.num_states == exp_states
    assert result.availability == pytest.approx(exp_avail, rel=1e-6, abs=1e-9)
    assert result.mean_time_to_unavailability == pytest.approx(
        exp_mttu, rel=1e-6, abs=1e-6,
    )
    assert result.expected_cost_per_hour == pytest.approx(
        exp_cost, rel=1e-9, abs=1e-12,
    )
