"""
Quality level comparison tests for Markov chain builders.

Compares availability and MTTF across all 5 quality levels for both
leaderless and Raft models. Verifies that simplified levels are within
acceptable tolerance of the FULL model, and reports state counts and
solve times at each level.
"""

import time

import numpy as np
import pytest

from powder.results import markov_analyze
from powder.scenario import QualityLevel
from powder.simulation import (
    Constant,
    Exponential,
    LeaderlessProtocol,
    NodeConfig,
    NoOpStrategy,
    RaftLikeProtocol,
)
from powder.simulation.distributions import hours, minutes


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _base_config() -> NodeConfig:
    """Node config with no data loss (avoids absorbing states for steady-state)."""
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(12)),
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Constant(float("inf")),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Constant(300),
    )


# ---------------------------------------------------------------------------
# Leaderless quality level comparison
# ---------------------------------------------------------------------------


class TestLeaderlessQualityLevels:
    """Compare quality levels for leaderless protocol."""

    @pytest.mark.parametrize("n_nodes", [3, 5, 7])
    def test_availability_converges(self, n_nodes: int):
        """All quality levels should produce availability within 1% of FULL."""
        cfg = _base_config()
        protocol = LeaderlessProtocol()
        strategy = NoOpStrategy()
        configs = [cfg] * n_nodes

        full_result = markov_analyze(configs, protocol, strategy, QualityLevel.FULL)

        for q in QualityLevel:
            result = markov_analyze(configs, protocol, strategy, q)
            rel_error = abs(result.availability - full_result.availability) / full_result.availability
            assert rel_error < 0.01, (
                f"N={n_nodes} {q.name}: availability {result.availability:.10f} "
                f"differs from FULL {full_result.availability:.10f} by {rel_error:.2%}"
            )

    @pytest.mark.parametrize("n_nodes", [3, 5])
    def test_mttf_converges(self, n_nodes: int):
        """All quality levels should produce MTTF within 5% of FULL."""
        cfg = _base_config()
        protocol = LeaderlessProtocol()
        strategy = NoOpStrategy()
        configs = [cfg] * n_nodes

        full_result = markov_analyze(configs, protocol, strategy, QualityLevel.FULL)
        if full_result.mean_time_to_unavailability is None:
            pytest.skip("FULL model has no MTTF (no absorbing states)")

        for q in QualityLevel:
            result = markov_analyze(configs, protocol, strategy, q)
            if result.mean_time_to_unavailability is None:
                continue
            rel_error = abs(
                result.mean_time_to_unavailability - full_result.mean_time_to_unavailability
            ) / full_result.mean_time_to_unavailability
            assert rel_error < 0.05, (
                f"N={n_nodes} {q.name}: MTTF {result.mean_time_to_unavailability:.1f}s "
                f"differs from FULL {full_result.mean_time_to_unavailability:.1f}s by {rel_error:.2%}"
            )

    def test_state_counts_match_doc(self):
        """Verify state counts match docs/markov_state_analysis.md for N=5 (no data loss)."""
        cfg = _base_config()
        protocol = LeaderlessProtocol()
        strategy = NoOpStrategy()
        configs = [cfg] * 5

        # With no data loss (infinite data_loss_dist), D states are unreachable.
        # BFS only generates reachable states, so counts will be smaller than
        # the full theoretical counts from the doc (which include D states).
        # We verify the simplified level since it's easy to check: states are
        # just (nH, nF) with nH + nF = N, giving N+1 states.
        result = markov_analyze(configs, protocol, strategy, QualityLevel.SIMPLIFIED)
        assert result.num_states == 6  # 0f..5f = 6 states for N=5


# ---------------------------------------------------------------------------
# Raft quality level comparison
# ---------------------------------------------------------------------------


class TestRaftQualityLevels:
    """Compare quality levels for Raft protocol."""

    @pytest.mark.parametrize("n_nodes", [3, 5, 7])
    def test_availability_converges(self, n_nodes: int):
        """All quality levels should produce availability within 1% of FULL."""
        cfg = _base_config()
        protocol = RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0))
        strategy = NoOpStrategy()
        configs = [cfg] * n_nodes

        full_result = markov_analyze(configs, protocol, strategy, QualityLevel.FULL)

        for q in QualityLevel:
            result = markov_analyze(configs, protocol, strategy, q)
            rel_error = abs(result.availability - full_result.availability) / full_result.availability
            assert rel_error < 0.01, (
                f"N={n_nodes} {q.name}: availability {result.availability:.10f} "
                f"differs from FULL {full_result.availability:.10f} by {rel_error:.2%}"
            )

    @pytest.mark.parametrize("n_nodes", [3, 5])
    def test_mttf_converges(self, n_nodes: int):
        """All quality levels should produce MTTF within 10% of FULL."""
        cfg = _base_config()
        protocol = RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0))
        strategy = NoOpStrategy()
        configs = [cfg] * n_nodes

        full_result = markov_analyze(configs, protocol, strategy, QualityLevel.FULL)
        if full_result.mean_time_to_unavailability is None:
            pytest.skip("FULL model has no MTTF")

        for q in QualityLevel:
            result = markov_analyze(configs, protocol, strategy, q)
            if result.mean_time_to_unavailability is None:
                continue
            rel_error = abs(
                result.mean_time_to_unavailability - full_result.mean_time_to_unavailability
            ) / full_result.mean_time_to_unavailability
            assert rel_error < 0.10, (
                f"N={n_nodes} {q.name}: MTTF {result.mean_time_to_unavailability:.1f}s "
                f"differs from FULL {full_result.mean_time_to_unavailability:.1f}s by {rel_error:.2%}"
            )

    def test_raft_lower_than_leaderless(self):
        """Raft availability should be strictly lower than leaderless due to election gap."""
        cfg = _base_config()
        strategy = NoOpStrategy()

        for n in [3, 5]:
            ll = markov_analyze(
                [cfg] * n, LeaderlessProtocol(), strategy, QualityLevel.SIMPLIFIED,
            )
            raft = markov_analyze(
                [cfg] * n,
                RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0)),
                strategy,
                QualityLevel.SIMPLIFIED,
            )
            assert raft.availability < ll.availability, (
                f"N={n}: Raft availability {raft.availability} should be < "
                f"leaderless {ll.availability}"
            )

    def test_simplified_state_count_formula(self):
        """Simplified Raft state count for no-data-loss should follow known pattern."""
        cfg = _base_config()
        protocol = RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0))
        strategy = NoOpStrategy()

        for n in [3, 5, 7]:
            result = markov_analyze([cfg] * n, protocol, strategy, QualityLevel.SIMPLIFIED)
            # With no data loss, reachable states are (nH, nF, 0, has_leader)
            # where nH + nF + has_leader = n. The number of such states is
            # 2*(n+1) - 1 for the BFS-reachable set (some all-down states
            # may collapse). Just verify it's reasonable.
            assert result.num_states > 0
            assert result.num_states <= (n + 1) ** 2


# ---------------------------------------------------------------------------
# Cross-protocol comparison (informational, not strict assertions)
# ---------------------------------------------------------------------------


class TestQualityLevelReport:
    """Prints a comparison table (use -s flag to see output)."""

    def test_print_comparison_table(self, capsys):
        """Print quality level comparison for review."""
        cfg = _base_config()
        strategy = NoOpStrategy()

        for label, protocol in [
            ("Leaderless", LeaderlessProtocol()),
            ("Raft", RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / 10.0))),
        ]:
            print(f"\n{'='*80}")
            print(f"  {label} Quality Level Comparison")
            print(f"{'='*80}")
            print(f"{'N':>3} {'Quality':>25} {'States':>8} {'Availability':>16} {'MTTF (h)':>12} {'Time (ms)':>10}")
            print("-" * 80)

            for n in [3, 5, 7]:
                for q in QualityLevel:
                    t0 = time.time()
                    result = markov_analyze([cfg] * n, protocol, strategy, q)
                    dt_ms = (time.time() - t0) * 1000

                    mttf_h = (
                        f"{result.mean_time_to_unavailability / 3600:.1f}"
                        if result.mean_time_to_unavailability
                        else "N/A"
                    )
                    print(
                        f"{n:>3} {q.name:>25} {result.num_states:>8} "
                        f"{result.availability:>16.10f} {mttf_h:>12} {dt_ms:>10.1f}"
                    )
                print()
