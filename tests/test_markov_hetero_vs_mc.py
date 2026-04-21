"""
Markov-vs-Monte Carlo validation for heterogeneous clusters.

Mirrors ``tests/test_markov_vs_mc.py`` but feeds the backends a
non-uniform ``list[NodeConfig]``. The Markov builder partitions
configs into rate classes; the MC simulator uses each node's own
config directly. Steady-state availability must agree within the
MC confidence interval.

All scenarios use exponential distributions so the Markov exponential
approximation is exact.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from scipy import stats as scipy_stats

from powder.monte_carlo import MonteCarloConfig, MonteCarloResults, MonteCarloRunner
from powder.results import markov_analyze
from powder.scenario import QualityLevel
from powder.simulation import (
    ClusterState,
    Constant,
    Exponential,
    LeaderlessProtocol,
    NetworkState,
    NodeConfig,
    NodeState,
    NoOpStrategy,
    RaftLikeProtocol,
    Seconds,
)
from powder.simulation.distributions import days, hours, minutes


_SIM_DURATION = days(30)
_NUM_SIMS = 200


def _reliable() -> NodeConfig:
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(24)),  # 24h MTBF
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Constant(float("inf")),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


def _unreliable() -> NodeConfig:
    return NodeConfig(
        region="us-west",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(6)),  # 6h MTBF
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Constant(float("inf")),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


def _make_cluster(configs: list[NodeConfig]) -> ClusterState:
    nodes = {
        f"node{i}": NodeState(node_id=f"node{i}", config=c)
        for i, c in enumerate(configs)
    }
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=len(configs),
    )


def _run_mc(
    configs: list[NodeConfig],
    protocol,
    seed: int,
    num_sims: int = _NUM_SIMS,
    sim_duration: float = _SIM_DURATION,
) -> MonteCarloResults:
    workers = os.cpu_count() or 1
    config = MonteCarloConfig(
        num_simulations=num_sims,
        max_time=Seconds(sim_duration),
        stop_on_data_loss=False,
        parallel_workers=workers,
        base_seed=seed,
    )
    runner = MonteCarloRunner(config)
    return runner.run(
        cluster=_make_cluster(configs),
        strategy=NoOpStrategy(),
        protocol=protocol,
    )


def _assert_within_ci(
    markov_avail: float,
    mc_samples: list[float],
    label: str,
    confidence: float = 0.99,
) -> None:
    n = len(mc_samples)
    arr = np.array(mc_samples)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_half = t_crit * std / math.sqrt(n)
    ci_lo = mean - ci_half
    ci_hi = mean + ci_half
    assert ci_lo <= markov_avail <= ci_hi, (
        f"{label}: Markov availability {markov_avail:.10f} outside "
        f"{confidence*100:.0f}% CI [{ci_lo:.10f}, {ci_hi:.10f}] "
        f"(MC mean={mean:.10f}, n={n})"
    )


def test_leaderless_2_class_availability_matches_mc():
    """2 reliable + 1 unreliable, leaderless no-leader-quorum."""
    protocol = LeaderlessProtocol(up_to_date_quorum=False)
    configs = [_reliable(), _reliable(), _unreliable()]
    markov = markov_analyze(
        configs, protocol, NoOpStrategy(), QualityLevel.FULL,
    )
    mc = _run_mc(configs, protocol, seed=82_000)
    _assert_within_ci(
        markov.availability, mc.availability_samples, "Leaderless 2r+1u",
    )


def test_raft_2_class_availability_matches_mc():
    """2 reliable + 1 unreliable, Raft with 10s election."""
    protocol = RaftLikeProtocol(
        election_time_dist=Exponential(rate=1.0 / 10.0),
    )
    configs = [_reliable(), _reliable(), _unreliable()]
    markov = markov_analyze(
        configs, protocol, NoOpStrategy(), QualityLevel.FULL,
    )
    mc = _run_mc(configs, protocol, seed=83_000)
    _assert_within_ci(
        markov.availability, mc.availability_samples, "Raft 2r+1u",
    )
