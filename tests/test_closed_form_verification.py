"""
Closed-form verification tests for the Monte Carlo simulator.

These tests verify that the simulator's mean availability and mean time
to first unavailability (mttF) match closed-form analytical values
derived from the Markov chain of a k-out-of-n system.

Assumptions matching the closed-form derivations:
  - Leaderless protocol with up_to_date_quorum=False (only node
    availability matters for commits, not sync state).
  - No data-loss failures (effectively infinite MTTDL).
  - NoOpStrategy (no replacement, no recovery management).
  - No network outages.
  - k = floor(n/2) + 1 (standard majority quorum).
  - Exponential failure and recovery times.

Closed-form formulas for failure rate ``a`` and recovery rate ``b``::

  Individual node availability: p = b / (a + b)

  n-node availability (majority quorum, independent nodes):
      A_n = Σ_{k=0}^{floor(n/2)} C(n,k) p^(n-k) (1-p)^k

  Equivalently for 3 nodes: A_3 = p²(3 - 2p) = b²(3a + b) / (a + b)³

  3-node mttF (mean time from all-up to first 2+ nodes down):
      MTTF_3 = (5a + b) / (6a²)

  5-node mttF (mean time from all-up to first 3+ nodes down):
      Solved from the birth-death chain absorption time equations.
"""

import math
import os

import numpy as np
import pytest
from scipy import stats as scipy_stats

from powder.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResults,
    MonteCarloRunner,
)
from powder.simulation import (
    ClusterState,
    Constant,
    Exponential,
    LeaderlessProtocol,
    NetworkState,
    NodeConfig,
    NodeState,
    NoOpStrategy,
    Seconds,
    days,
    hours,
    minutes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_node_config(
    failure_rate: float,
    recovery_rate: float,
) -> NodeConfig:
    """Create a node config with exponential failure/recovery and no data loss."""
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=failure_rate),
        recovery_dist=Exponential(rate=recovery_rate),
        data_loss_dist=Constant(days(99999)),   # effectively never
        log_replay_rate_dist=Constant(1e6),     # instant replay
        snapshot_download_time_dist=Constant(0), # instant snapshot
        spawn_dist=Constant(0),
    )


def _make_cluster(
    failure_rate: float,
    recovery_rate: float,
    num_nodes: int,
) -> ClusterState:
    """Build a cluster for the simplified scenario."""
    cfg = _simple_node_config(failure_rate, recovery_rate)
    nodes = {
        f"node{i}": NodeState(node_id=f"node{i}", config=cfg)
        for i in range(num_nodes)
    }
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=num_nodes,
    )


def _run_sims(
    failure_rate: float,
    recovery_rate: float,
    num_nodes: int,
    num_sims: int,
    sim_duration: float,
    seed: int,
) -> MonteCarloResults:
    """Run simulations and return results."""
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
        cluster=_make_cluster(failure_rate, recovery_rate, num_nodes),
        strategy=NoOpStrategy(),
        protocol=LeaderlessProtocol(up_to_date_quorum=False),
    )


# ---------------------------------------------------------------------------
# Closed-form formulas
# ---------------------------------------------------------------------------


def _analytical_availability(a: float, b: float, n: int) -> float:
    """Closed-form availability for n-node majority-quorum system.

    Each node has independent steady-state availability p = b/(a+b).
    System available iff <= floor(n/2) nodes are down.

    Args:
        a: Transient failure rate per node.
        b: Recovery rate per node.
        n: Number of nodes.
    """
    p = b / (a + b)
    q = 1 - p
    max_failures = n // 2  # floor(n/2) failures still have quorum
    return sum(
        math.comb(n, k) * p ** (n - k) * q ** k
        for k in range(max_failures + 1)
    )


def _analytical_mttf_3(a: float, b: float) -> float:
    """Mean time from all-up to first unavailability for 3-node system.

    Birth-death chain with states = number of failed nodes.
    Absorbing at state 2 (only 1 of 3 nodes up = no quorum).

    T_0 = 1/(3a) + T_1
    T_1 = 1/(2a+b) + [b/(2a+b)]·T_0    (prob 2a/(2a+b) absorbs)

    Solution: T_0 = (5a + b) / (6a²)
    """
    return (5 * a + b) / (6 * a**2)


def _analytical_mttf_5(a: float, b: float) -> float:
    """Mean time from all-up to first unavailability for 5-node system.

    Birth-death chain with states = number of failed nodes (0..5).
    System unavailable at state 3+ (only 2 of 5 nodes up = no quorum).
    Absorbing at state 3.

    Solve the linear system for first passage times T_0, T_1, T_2.
    """
    # T_0 = 1/(5a) + T_1                             => T_0 - T_1 = 1/(5a)
    # T_1 = 1/(4a+b) + [b/(4a+b)]*T_0 + [4a/(4a+b)]*T_2
    #     => -b/(4a+b)*T_0 + T_1 - 4a/(4a+b)*T_2 = 1/(4a+b)
    # T_2 = 1/(3a+2b) + [2b/(3a+2b)]*T_1 + [3a/(3a+2b)]*0
    #     => -2b/(3a+2b)*T_1 + T_2 = 1/(3a+2b)
    A_mat = np.array([
        [1, -1, 0],
        [-b / (4 * a + b), 1, -4 * a / (4 * a + b)],
        [0, -2 * b / (3 * a + 2 * b), 1],
    ])
    b_vec = np.array([
        1 / (5 * a),
        1 / (4 * a + b),
        1 / (3 * a + 2 * b),
    ])
    T = np.linalg.solve(A_mat, b_vec)
    return float(T[0])  # T_0


def _assert_within_ci(
    samples: np.ndarray,
    analytical: float,
    label: str,
) -> None:
    """Assert the analytical value falls within the 99% CI of the samples."""
    n = len(samples)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    t_crit = scipy_stats.t.ppf(0.995, df=n - 1)
    ci_half = t_crit * std / math.sqrt(n)
    ci_lo = mean - ci_half
    ci_hi = mean + ci_half
    assert ci_lo <= analytical <= ci_hi, (
        f"{label}: analytical {analytical:.8f} outside 99% CI "
        f"[{ci_lo:.8f}, {ci_hi:.8f}] (mean={mean:.8f})"
    )


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

# Rates chosen so that:
#  - Failures are frequent enough that mttF tests don't need absurdly many runs
#  - Recovery is fast enough that availability is high (realistic regime)
_FAILURE_RATE = 1.0 / hours(12)   # ~1 failure per 12 hours per node
_RECOVERY_RATE = 1.0 / minutes(10)  # ~10 min recovery per node

_SIM_DURATION_AVAIL = days(30)  # 30 days per availability sim (~60 cycles/node)
_NUM_SIMS_AVAIL = 200           # simulations for availability tests
_NUM_SIMS_MTTF = 300            # simulations for mttF tests


# ==========================================================================
# Test class
# ==========================================================================


class TestClosedFormVerification:
    """Verify simulator accuracy against closed-form analytical values."""

    # ------------------------------------------------------------------
    # 1. 3-node availability
    # ------------------------------------------------------------------

    def test_3_node_availability(self):
        """Monte Carlo mean availability should match the 3-node closed form.

        A_3 = p²(3 − 2p) where p = b/(a+b).
        """
        a, b = _FAILURE_RATE, _RECOVERY_RATE
        analytical = _analytical_availability(a, b, n=3)

        results = _run_sims(
            a, b, num_nodes=3,
            num_sims=_NUM_SIMS_AVAIL,
            sim_duration=_SIM_DURATION_AVAIL,
            seed=42,
        )
        _assert_within_ci(
            np.array(results.availability_samples), analytical,
            "3-node availability",
        )

    # ------------------------------------------------------------------
    # 2. 5-node availability
    # ------------------------------------------------------------------

    def test_5_node_availability(self):
        """Monte Carlo mean availability should match the 5-node closed form.

        A_5 = Σ_{k=0}^{2} C(5,k) p^(5-k) (1-p)^k
        """
        a, b = _FAILURE_RATE, _RECOVERY_RATE
        analytical = _analytical_availability(a, b, n=5)

        results = _run_sims(
            a, b, num_nodes=5,
            num_sims=_NUM_SIMS_AVAIL,
            sim_duration=_SIM_DURATION_AVAIL,
            seed=100_000,
        )
        _assert_within_ci(
            np.array(results.availability_samples), analytical,
            "5-node availability",
        )

    # ------------------------------------------------------------------
    # 3. 3-node mttF
    # ------------------------------------------------------------------

    def test_3_node_mttf(self):
        """Monte Carlo mean time to first unavailability should match 3-node closed form.

        MTTF_3 = (5a + b) / (6a²)
        """
        a, b = _FAILURE_RATE, _RECOVERY_RATE
        analytical = _analytical_mttf_3(a, b)

        # Duration must be >> mttF to avoid right-censoring bias
        results = _run_sims(
            a, b, num_nodes=3,
            num_sims=_NUM_SIMS_MTTF,
            sim_duration=analytical * 5,
            seed=200_000,
        )

        mttf_samples = [
            t for t in results.time_to_first_unavailability_samples
            if t is not None
        ]
        assert len(mttf_samples) > 50, (
            f"Too few runs experienced unavailability: {len(mttf_samples)}"
        )
        _assert_within_ci(
            np.array(mttf_samples), analytical,
            "3-node MTTF",
        )

    # ------------------------------------------------------------------
    # 4. 5-node mttF
    # ------------------------------------------------------------------

    def test_5_node_mttf(self):
        """Monte Carlo mean time to first unavailability should match 5-node closed form.

        Solved from the birth-death chain first-passage-time linear system.
        """
        a, b = _FAILURE_RATE, _RECOVERY_RATE
        analytical = _analytical_mttf_5(a, b)

        # Duration must be >> mttF to avoid right-censoring bias
        results = _run_sims(
            a, b, num_nodes=5,
            num_sims=_NUM_SIMS_MTTF,
            sim_duration=analytical * 5,
            seed=300_000,
        )

        mttf_samples = [
            t for t in results.time_to_first_unavailability_samples
            if t is not None
        ]
        assert len(mttf_samples) > 50, (
            f"Too few runs experienced unavailability: {len(mttf_samples)}"
        )
        _assert_within_ci(
            np.array(mttf_samples), analytical,
            "5-node MTTF",
        )


# ==========================================================================
# Parameterized tests across different rate regimes
# ==========================================================================

# Each tuple: (label, failure_rate, recovery_rate)
_RATE_SCENARIOS = [
    # Frequent failures, fast recovery  (a/b ratio ≈ 0.008)
    ("frequent_fail_fast_recover", 1.0 / hours(6), 1.0 / minutes(3)),
    # High-stress: failures every 3h, 3min recovery  (a/b ratio ≈ 0.017)
    ("high_stress", 1.0 / hours(3), 1.0 / minutes(3)),
]

_RATE_IDS = [s[0] for s in _RATE_SCENARIOS]


class TestClosedFormMultiRate:
    """Verify simulator across multiple failure/recovery rate regimes."""

    # ------------------------------------------------------------------
    # Availability (3-node and 5-node)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("label, a, b", _RATE_SCENARIOS, ids=_RATE_IDS)
    def test_3_node_availability_multi_rate(self, label, a, b):
        """3-node availability matches closed form across rate regimes."""
        analytical = _analytical_availability(a, b, n=3)
        results = _run_sims(
            a, b, num_nodes=3,
            num_sims=150,
            sim_duration=days(30),
            seed=hash(label) % 2**31,
        )
        _assert_within_ci(
            np.array(results.availability_samples), analytical,
            f"3-node avail [{label}]",
        )

    @pytest.mark.parametrize("label, a, b", _RATE_SCENARIOS, ids=_RATE_IDS)
    def test_5_node_availability_multi_rate(self, label, a, b):
        """5-node availability matches closed form across rate regimes."""
        analytical = _analytical_availability(a, b, n=5)
        results = _run_sims(
            a, b, num_nodes=5,
            num_sims=150,
            sim_duration=days(30),
            seed=hash(label) % 2**31 + 1,
        )
        _assert_within_ci(
            np.array(results.availability_samples), analytical,
            f"5-node avail [{label}]",
        )

    # ------------------------------------------------------------------
    # mttF (3-node and 5-node)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("label, a, b", _RATE_SCENARIOS, ids=_RATE_IDS)
    def test_3_node_mttf_multi_rate(self, label, a, b):
        """3-node mttF matches closed form across rate regimes."""
        analytical = _analytical_mttf_3(a, b)
        results = _run_sims(
            a, b, num_nodes=3,
            num_sims=800,
            sim_duration=analytical * 10,
            seed=hash(label) % 2**31 + 2,
        )
        mttf_samples = [
            t for t in results.time_to_first_unavailability_samples
            if t is not None
        ]
        assert len(mttf_samples) > 50, (
            f"Too few runs experienced unavailability: {len(mttf_samples)}"
        )
        _assert_within_ci(
            np.array(mttf_samples), analytical,
            f"3-node mttF [{label}]",
        )

    @pytest.mark.parametrize("label, a, b", _RATE_SCENARIOS, ids=_RATE_IDS)
    def test_5_node_mttf_multi_rate(self, label, a, b):
        """5-node mttF matches closed form across rate regimes."""
        analytical = _analytical_mttf_5(a, b)
        results = _run_sims(
            a, b, num_nodes=5,
            num_sims=800,
            sim_duration=analytical * 10,
            seed=hash(label) % 2**31 + 3,
        )
        mttf_samples = [
            t for t in results.time_to_first_unavailability_samples
            if t is not None
        ]
        assert len(mttf_samples) > 50, (
            f"Too few runs experienced unavailability: {len(mttf_samples)}"
        )
        _assert_within_ci(
            np.array(mttf_samples), analytical,
            f"5-node mttF [{label}]",
        )
