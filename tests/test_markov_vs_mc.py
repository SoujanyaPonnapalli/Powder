"""
Markov vs Monte Carlo validation tests.

Runs the same scenario through both the Markov solver (highest quality)
and the Monte Carlo simulator, then asserts the Markov steady-state
availability falls within the MC 99% confidence interval.

Test matrix:
  - Leaderless, NoOpStrategy, N=3 and N=5
  - Raft, NoOpStrategy, N=3 and N=5

All scenarios use exponential distributions so the Markov assumptions
hold exactly. Data loss is disabled to avoid absorbing states that
break steady-state analysis.
"""

import math
import os

import numpy as np
import pytest
from scipy import stats as scipy_stats

from powder.monte_carlo import MonteCarloConfig, MonteCarloResults, MonteCarloRunner
from powder.results import ClusterAnalysisResult, markov_analyze, monte_carlo_analyze
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


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

_FAILURE_RATE = 1.0 / hours(12)
_RECOVERY_RATE = 1.0 / minutes(10)
_SIM_DURATION = days(30)
_NUM_SIMS = 200


def _node_config() -> NodeConfig:
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=_FAILURE_RATE),
        recovery_dist=Exponential(rate=_RECOVERY_RATE),
        data_loss_dist=Constant(float("inf")),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


def _make_cluster(num_nodes: int) -> ClusterState:
    cfg = _node_config()
    nodes = {
        f"node{i}": NodeState(node_id=f"node{i}", config=cfg)
        for i in range(num_nodes)
    }
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=num_nodes,
    )


def _run_mc(
    num_nodes: int,
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
        cluster=_make_cluster(num_nodes),
        strategy=NoOpStrategy(),
        protocol=protocol,
    )


def _assert_markov_within_mc_ci(
    markov_avail: float,
    mc_samples: list[float],
    label: str,
    confidence: float = 0.99,
) -> None:
    """Assert the Markov availability falls within the MC confidence interval."""
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


# ---------------------------------------------------------------------------
# Leaderless tests
# ---------------------------------------------------------------------------


class TestLeaderlessMarkovVsMC:
    """Validate leaderless Markov model against Monte Carlo."""

    def test_3_node_availability(self):
        """Markov availability should fall within MC 99% CI for 3 nodes."""
        protocol = LeaderlessProtocol(up_to_date_quorum=False)
        cfg = _node_config()

        markov = markov_analyze(
            [cfg] * 3, protocol, NoOpStrategy(), QualityLevel.FULL,
        )
        mc = _run_mc(3, protocol, seed=42_000)

        _assert_markov_within_mc_ci(
            markov.availability,
            mc.availability_samples,
            "Leaderless N=3",
        )

    def test_5_node_availability(self):
        """Markov availability should fall within MC 99% CI for 5 nodes."""
        protocol = LeaderlessProtocol(up_to_date_quorum=False)
        cfg = _node_config()

        markov = markov_analyze(
            [cfg] * 5, protocol, NoOpStrategy(), QualityLevel.FULL,
        )
        mc = _run_mc(5, protocol, seed=43_000)

        _assert_markov_within_mc_ci(
            markov.availability,
            mc.availability_samples,
            "Leaderless N=5",
        )


# ---------------------------------------------------------------------------
# Raft tests
# ---------------------------------------------------------------------------


class TestRaftMarkovVsMC:
    """Validate Raft Markov model against Monte Carlo."""

    def test_3_node_availability(self):
        """Raft Markov availability should fall within MC 99% CI for 3 nodes."""
        protocol = RaftLikeProtocol(
            election_time_dist=Exponential(rate=1.0 / 10.0),
        )
        cfg = _node_config()

        markov = markov_analyze(
            [cfg] * 3, protocol, NoOpStrategy(), QualityLevel.FULL,
        )
        mc = _run_mc(3, protocol, seed=44_000)

        _assert_markov_within_mc_ci(
            markov.availability,
            mc.availability_samples,
            "Raft N=3",
        )

    def test_5_node_availability(self):
        """Raft Markov availability should fall within MC 99% CI for 5 nodes."""
        protocol = RaftLikeProtocol(
            election_time_dist=Exponential(rate=1.0 / 10.0),
        )
        cfg = _node_config()

        markov = markov_analyze(
            [cfg] * 5, protocol, NoOpStrategy(), QualityLevel.FULL,
        )
        mc = _run_mc(5, protocol, seed=45_000)

        _assert_markov_within_mc_ci(
            markov.availability,
            mc.availability_samples,
            "Raft N=5",
        )


# ---------------------------------------------------------------------------
# Unified result adapter
# ---------------------------------------------------------------------------


class TestMonteCarloAnalyzeAdapter:
    """Verify monte_carlo_analyze produces a ClusterAnalysisResult comparable to markov_analyze."""

    def test_adapter_fields_populated(self):
        protocol = LeaderlessProtocol(up_to_date_quorum=False)
        mc_results = _run_mc(3, protocol, seed=46_000, num_sims=50)

        unified = monte_carlo_analyze(mc_results)

        assert isinstance(unified, ClusterAnalysisResult)
        assert unified.method == "monte_carlo"
        assert unified.quality_level is None
        assert unified.num_states is None
        assert unified.steady_state_distribution is None
        assert 0.0 <= unified.availability <= 1.0
        assert unified.expected_cost_per_hour is not None
        assert unified.expected_cost_per_hour > 0.0

    def test_adapter_matches_markov_availability(self):
        """Unified results from both backends should agree on availability."""
        protocol = LeaderlessProtocol(up_to_date_quorum=False)
        cfg = _node_config()

        markov = markov_analyze(
            [cfg] * 3, protocol, NoOpStrategy(), QualityLevel.FULL,
        )
        mc_raw = _run_mc(3, protocol, seed=47_000)
        mc = monte_carlo_analyze(mc_raw)

        _assert_markov_within_mc_ci(
            markov.availability,
            mc_raw.availability_samples,
            "Adapter comparison N=3",
        )
        assert abs(markov.availability - mc.availability) < 0.01

    def test_adapter_cost_matches_expected_rate(self):
        """Per-node cost rate should match configuration."""
        protocol = LeaderlessProtocol(up_to_date_quorum=False)
        mc_raw = _run_mc(3, protocol, seed=48_000, num_sims=50)

        mc = monte_carlo_analyze(mc_raw)

        expected_rate = 3 * 1.0
        assert mc.expected_cost_per_hour == pytest.approx(expected_rate, rel=0.02)
