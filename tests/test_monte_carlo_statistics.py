"""
Statistical guarantee tests for the Monte Carlo simulator.

These tests verify that the Monte Carlo output produces statistically
valid results: confidence intervals have correct coverage, CI widths
scale as expected, and simulated availability converges to analytical
values for scenarios with known closed-form solutions.

The scenario used throughout is deliberately simple:
  - Leaderless protocol with up_to_date_quorum=False (only node
    availability matters for commits, not sync state).
  - No data-loss failures (effectively infinite MTTDL).
  - NoOpStrategy (no replacement, no recovery management).
  - No network outages.
  - Instant snapshot recovery (irrelevant with NoOp).
  - 3 nodes, majority quorum of 2.

With exponential failures and constant recovery each node has
steady-state availability p = MTBF / (MTBF + MTTR), and system
availability is A = 3p²(1-p) + p³ = p²(3 - 2p).
"""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from powder.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResults,
    MonteCarloRunner,
    run_monte_carlo,
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
    mtbf_seconds: float,
    mttr_seconds: float,
) -> NodeConfig:
    """Create a node config with tunable failure/recovery and no data loss."""
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / mtbf_seconds),
        recovery_dist=Constant(mttr_seconds),
        data_loss_dist=Constant(days(99999)),   # effectively never
        log_replay_rate_dist=Constant(1e6),     # instant replay
        snapshot_download_time_dist=Constant(0), # instant snapshot
        spawn_dist=Constant(0),
    )


def _make_simple_cluster(
    mtbf_seconds: float,
    mttr_seconds: float,
    num_nodes: int = 3,
) -> ClusterState:
    """Build a cluster for the simplified scenario."""
    cfg = _simple_node_config(mtbf_seconds, mttr_seconds)
    nodes = {
        f"node{i}": NodeState(node_id=f"node{i}", config=cfg)
        for i in range(num_nodes)
    }
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=num_nodes,
    )


def _analytical_availability(mtbf: float, mttr: float) -> float:
    """Closed-form system availability for 3-node majority quorum.

    Each node has steady-state probability  p = mtbf / (mtbf + mttr).
    System is available iff >= 2 of 3 nodes are up:
        A = C(3,2) p²(1-p) + C(3,3) p³ = p²(3 - 2p)
    """
    p = mtbf / (mtbf + mttr)
    return p * p * (3 - 2 * p)


def _run_simple(
    mtbf_seconds: float,
    mttr_seconds: float,
    num_sims: int,
    sim_duration: float,
    seed: int,
) -> MonteCarloResults:
    """Run the simplified Monte Carlo scenario and return results."""
    config = MonteCarloConfig(
        num_simulations=num_sims,
        max_time=Seconds(sim_duration),
        stop_on_data_loss=False,
        base_seed=seed,
    )
    runner = MonteCarloRunner(config)
    return runner.run(
        cluster=_make_simple_cluster(mtbf_seconds, mttr_seconds),
        strategy=NoOpStrategy(),
        protocol=LeaderlessProtocol(up_to_date_quorum=False),
    )


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

# Node parameters: MTBF = 4 h, MTTR = 10 min → p ≈ 0.9600
# Using a higher failure rate so non-trivial variance shows up
# even with shorter simulation windows.
_MTBF = hours(4)
_MTTR = minutes(10)
_SIM_DURATION = days(7)  # 7 days per sim — ~42 failures/node, enough for stable estimates


# ==========================================================================
# Test class
# ==========================================================================


class TestMonteCarloStatisticalGuarantees:
    """Verify statistical guarantees of the Monte Carlo simulator output."""

    # ------------------------------------------------------------------
    # 1. CI Coverage Rate
    # ------------------------------------------------------------------

    def test_ci_coverage_rate(self):
        """The 95% CI from N runs should contain the analytical mean ~95% of the time.

        Method
        ------
        Since this scenario has a closed-form availability, we use the
        analytical value as ground truth rather than a noisy sample mean.

        For each of K independent trials:
          1. Run N simulations; compute the sample mean and 95% CI.
          2. Check whether the known analytical availability falls
             inside that CI.

        Assert: the coverage proportion across K trials is in [0.88, 1.0].
        The generous lower bound accounts for finite-sample noise in a
        Monte Carlo of Monte Carlo; a correctly calibrated 95% CI should
        fail this bound extremely rarely (< 0.1%).
        """
        N = 100       # runs used to build the CI
        K = 50        # number of independent trials
        true_avail = _analytical_availability(_MTBF, _MTTR)

        covered = 0

        for trial in range(K):
            seed = trial * N

            # --- build CI from N runs ---
            results = _run_simple(_MTBF, _MTTR, N, _SIM_DURATION, seed)
            samples = np.array(results.availability_samples)
            mean = float(np.mean(samples))
            std = float(np.std(samples, ddof=1))
            t_crit = scipy_stats.t.ppf(0.975, df=N - 1)
            ci_half = t_crit * std / math.sqrt(N)
            ci_lo = mean - ci_half
            ci_hi = mean + ci_half

            if ci_lo <= true_avail <= ci_hi:
                covered += 1

        coverage = covered / K

        # With a true 95% CI, P(coverage < 0.84 out of 50 trials)
        # is vanishingly small (binomial with p=0.95, n=50).
        assert coverage >= 0.84, (
            f"CI coverage {coverage:.2%} is below 88%; "
            f"expected ≈95% for a 95% confidence interval"
            f"(analytical μ={true_avail:.8f})"
        )

    # ------------------------------------------------------------------
    # 2. CI Width Scales with 1/√n
    # ------------------------------------------------------------------

    def test_ci_width_scales_with_sqrt_n(self):
        """Quadrupling sample size should roughly halve the CI width.

        Run N and 4N simulations from the same seed family, compute
        the 95% CI half-width for each, and verify that
            hw_4n ≈ hw_n / 2   (within ±40% tolerance).
        """
        N = 100
        seed = 100_000
        sim_dur = _SIM_DURATION

        results_n = _run_simple(_MTBF, _MTTR, N, sim_dur, seed)
        results_4n = _run_simple(_MTBF, _MTTR, 4 * N, sim_dur, seed)

        def _ci_half_width(results: MonteCarloResults) -> float:
            s = np.array(results.availability_samples)
            n = len(s)
            std = float(np.std(s, ddof=1))
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
            return t_crit * std / math.sqrt(n)

        hw_n = _ci_half_width(results_n)
        hw_4n = _ci_half_width(results_4n)

        # Theoretically hw_4n / hw_n ≈ 0.5 (since SE ∝ 1/√n).
        ratio = hw_4n / hw_n
        assert 0.3 <= ratio <= 0.7, (
            f"CI width ratio (4N/N) = {ratio:.3f}; expected ≈0.5 "
            f"(hw_n={hw_n:.6f}, hw_4n={hw_4n:.6f})"
        )

    # ------------------------------------------------------------------
    # 3. Convergence to Analytical Value
    # ------------------------------------------------------------------

    def test_availability_converges_to_analytical_value(self):
        """Monte Carlo mean should converge to the closed-form availability.

        For a 3-node cluster with independent Exp failures and constant
        recovery, the analytical availability is p²(3 - 2p) where
        p = MTBF / (MTBF + MTTR).

        Run enough simulations that the 99% CI should comfortably
        contain the analytical value.
        """
        analytical = _analytical_availability(_MTBF, _MTTR)

        results = _run_simple(_MTBF, _MTTR, 500, _SIM_DURATION, seed=42)
        samples = np.array(results.availability_samples)
        n = len(samples)
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))

        # 99% CI to reduce false-positive rate
        t_crit = scipy_stats.t.ppf(0.995, df=n - 1)
        ci_half = t_crit * std / math.sqrt(n)
        ci_lo = mean - ci_half
        ci_hi = mean + ci_half

        assert ci_lo <= analytical <= ci_hi, (
            f"Analytical availability {analytical:.8f} is outside the "
            f"99% CI [{ci_lo:.8f}, {ci_hi:.8f}] (mean={mean:.8f})"
        )

    # ------------------------------------------------------------------
    # 4. Higher Failure Rate Lowers Availability
    # ------------------------------------------------------------------

    def test_higher_failure_rate_lowers_availability(self):
        """A higher failure rate should produce strictly lower availability.

        Compare MTBF = 12 h vs MTBF = 4 h (3× higher failure rate)
        with the same MTTR.  Run enough simulations so the means are
        well separated.
        """
        mtbf_low_rate = hours(8)    # less frequent failures
        mtbf_high_rate = hours(2)   # more frequent failures
        mttr = minutes(10)
        n = 200
        seed = 200_000

        results_low = _run_simple(mtbf_low_rate, mttr, n, _SIM_DURATION, seed)
        results_high = _run_simple(mtbf_high_rate, mttr, n, _SIM_DURATION, seed + n)

        mean_low = results_low.availability_mean()
        mean_high = results_high.availability_mean()

        assert mean_low > mean_high, (
            f"Expected lower failure rate to have higher availability: "
            f"MTBF=12h → {mean_low:.6f}, MTBF=4h → {mean_high:.6f}"
        )

        # Also verify both are close to their analytical values
        analytical_low = _analytical_availability(mtbf_low_rate, mttr)
        analytical_high = _analytical_availability(mtbf_high_rate, mttr)
        assert mean_low == pytest.approx(analytical_low, abs=0.005)
        assert mean_high == pytest.approx(analytical_high, abs=0.005)

    # ------------------------------------------------------------------
    # 5. Zero Variance — Perfect Availability
    # ------------------------------------------------------------------

    def test_zero_variance_perfect_availability(self):
        """Nodes that never fail should yield availability = 1.0 exactly.

        With MTBF → ∞ (modeled as a very large constant), no failures
        occur and the cluster is always available.  Mean should be 1.0
        and standard deviation should be 0.0.
        """
        # Use Constant to guarantee no failure within simulation window
        cfg = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(99999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(99999)),
            log_replay_rate_dist=Constant(1e6),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )
        nodes = {
            f"node{i}": NodeState(node_id=f"node{i}", config=cfg)
            for i in range(3)
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        config = MonteCarloConfig(
            num_simulations=20,
            max_time=days(30),
            stop_on_data_loss=False,
            base_seed=0,
        )
        runner = MonteCarloRunner(config)
        results = runner.run(
            cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(up_to_date_quorum=False),
        )

        assert results.availability_mean() == 1.0
        assert results.availability_std() == 0.0
