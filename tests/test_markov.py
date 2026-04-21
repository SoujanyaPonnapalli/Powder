"""Tests for the Markov model primitives and absorbing-state properties."""

import numpy as np
import pytest
from scipy import sparse

from powder.markov import MarkovModel
from powder.markov_solver import (
    SteadyStateResidual,
    build_q_from_triples,
    compute_steady_state_residual,
    steady_state,
)
from powder.results import markov_analyze
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    Constant,
    Exponential,
    LeaderlessProtocol,
    NodeConfig,
    NodeReplacementStrategy,
    NoOpStrategy,
    RaftLikeProtocol,
)
from powder.simulation.distributions import Seconds, days, hours, minutes


# ---------------------------------------------------------------------------
# New MarkovModel / solver primitives
# ---------------------------------------------------------------------------


def test_build_q_from_triples_rows_sum_to_zero():
    Q = build_q_from_triples(2, [(0, 1, 0.5), (1, 0, 1.0 / 3)])
    dense = Q.toarray()
    assert dense[0, 0] == pytest.approx(-0.5)
    assert dense[0, 1] == pytest.approx(0.5)
    assert np.allclose(dense.sum(axis=1), 0.0)


def test_steady_state_two_state_chain():
    Q = build_q_from_triples(2, [(0, 1, 1.0), (1, 0, 3.0)])
    model = MarkovModel(
        Q=Q,
        initial_distribution=np.array([1.0, 0.0]),
        live_mask=np.array([True, False]),
        state_costs=np.zeros(2),
        state_names=["A", "B"],
    )
    pi = steady_state(model)
    assert pi[0] == pytest.approx(0.75)
    assert pi[1] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Steady-state residual reporting
# ---------------------------------------------------------------------------


def _toy_two_state_model() -> MarkovModel:
    Q = build_q_from_triples(2, [(0, 1, 1.0), (1, 0, 3.0)])
    return MarkovModel(
        Q=Q,
        initial_distribution=np.array([1.0, 0.0]),
        live_mask=np.array([True, False]),
        state_costs=np.zeros(2),
        state_names=["A", "B"],
    )


def test_steady_state_residual_small_for_well_conditioned_chain():
    """Well-conditioned 2-state chain: residual should be at machine precision."""
    model = _toy_two_state_model()
    pi, residual = steady_state(model, return_residual=True)
    assert isinstance(residual, SteadyStateResidual)
    assert residual.balance < 1e-12
    assert residual.normalization < 1e-12
    assert residual.negativity < 1e-12
    assert residual.worst < 1e-12
    assert pi.sum() == pytest.approx(1.0)


def test_steady_state_no_warning_below_threshold(caplog):
    """Default threshold should not fire on well-conditioned chains."""
    model = _toy_two_state_model()
    with caplog.at_level("WARNING", logger="powder.markov_solver"):
        steady_state(model)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == []


def test_steady_state_warns_when_residual_exceeds_threshold(caplog):
    """Force the threshold below the achievable residual to exercise the warn path."""
    model = _toy_two_state_model()
    with caplog.at_level("WARNING", logger="powder.markov_solver"):
        steady_state(model, residual_threshold=-1.0)
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "Steady-state residual" in r.getMessage()
    ]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "balance=" in msg
    assert "normalization=" in msg
    assert "negativity=" in msg


def test_steady_state_residual_independent_of_returned_pi():
    """compute_steady_state_residual should reproduce the diagnostic struct."""
    model = _toy_two_state_model()
    pi = steady_state(model)
    residual = compute_steady_state_residual(model, pi)
    # After renormalization and clipping the balance may be *smaller* than the
    # pre-clip residual but must still be near zero.
    assert residual.balance < 1e-10
    assert residual.normalization < 1e-12


# ---------------------------------------------------------------------------
# Data-loss absorbing state property tests
# ---------------------------------------------------------------------------


def _dataloss_config() -> NodeConfig:
    """Config with nonzero data loss rate for absorbing-state tests."""
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


def _find_absorbing(model: MarkovModel) -> list[str]:
    """Return names of absorbing states (zero outgoing rate)."""
    Q: sparse.csr_matrix = model.Q
    dense = Q.toarray()
    absorbing = []
    for i, name in enumerate(model.state_names):
        row = dense[i].copy()
        row[i] = 0
        if np.allclose(row, 0):
            absorbing.append(name)
    return absorbing


class TestAbsorbingStateProperties:
    """Verify that the all-data-loss approximation works correctly."""

    @pytest.mark.parametrize(
        "protocol",
        [LeaderlessProtocol(), RaftLikeProtocol(election_time_dist=Exponential(rate=0.1))],
        ids=["leaderless", "raft"],
    )
    def test_no_absorbing_states_with_replacement(self, protocol):
        cfg = _dataloss_config()
        strategy = NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))
        for q in [QualityLevel.SIMPLIFIED, QualityLevel.COLLAPSED_PIPELINE]:
            model = build_markov_model([cfg] * 3, protocol, strategy, quality=q)
            absorbing = _find_absorbing(model)
            assert absorbing == [], (
                f"{type(protocol).__name__} {q.name}: unexpected absorbing states {absorbing}"
            )

    @pytest.mark.parametrize(
        "protocol",
        [LeaderlessProtocol(), RaftLikeProtocol(election_time_dist=Exponential(rate=0.1))],
        ids=["leaderless", "raft"],
    )
    def test_all_transient_failure_not_absorbing(self, protocol):
        cfg = _dataloss_config()
        strategy = NoOpStrategy()
        model = build_markov_model(
            [cfg] * 3, protocol, strategy, quality=QualityLevel.SIMPLIFIED,
        )
        dense = model.Q.toarray()
        for sid, name in enumerate(model.state_names):
            counts = tuple(int(x) for x in name.split(":"))
            is_all_f = (
                (isinstance(protocol, LeaderlessProtocol) and counts == (0, 3, 0))
                # Raft SIMPLIFIED state tuple: (H, F, D, has_leader, leader_class)
                or (
                    isinstance(protocol, RaftLikeProtocol)
                    and counts == (0, 3, 0, 0, 0)
                )
            )
            if is_all_f:
                row = dense[sid].copy()
                row[sid] = 0
                assert np.sum(row) > 0, f"All-F state {name} should not be absorbing"

    @pytest.mark.parametrize(
        "protocol",
        [LeaderlessProtocol(), RaftLikeProtocol(election_time_dist=Exponential(rate=0.1))],
        ids=["leaderless", "raft"],
    )
    def test_availability_nonzero_with_dataloss(self, protocol):
        cfg = _dataloss_config()
        strategy = NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))
        result = markov_analyze([cfg] * 3, protocol, strategy, QualityLevel.SIMPLIFIED)
        assert result.availability > 0.99, (
            f"Expected high availability, got {result.availability}"
        )


# ---------------------------------------------------------------------------
# Cost model tests
# ---------------------------------------------------------------------------


class TestCostModel:
    """Verify the homogeneous state-cost accounting."""

    def test_initial_state_cost_is_full_cluster(self):
        cfg = NodeConfig(
            region="us-east",
            cost_per_hour=2.0,
            failure_dist=Exponential(rate=1.0 / hours(100)),
            recovery_dist=Exponential(rate=1.0 / minutes(30)),
            data_loss_dist=Constant(float("inf")),
            log_replay_rate_dist=Constant(1e6),
            snapshot_download_time_dist=Constant(60),
            spawn_dist=Constant(300),
        )
        protocol = LeaderlessProtocol()
        strategy = NoOpStrategy()
        model = build_markov_model(
            [cfg] * 5, protocol, strategy, quality=QualityLevel.SIMPLIFIED,
        )
        initial = model.initial_state_id
        expected_per_second = 5 * (2.0 / 3600.0)
        assert model.state_costs[initial] == pytest.approx(expected_per_second)

    def test_expected_cost_matches_availability_for_stable_cluster(self):
        """With ~100% availability and no data loss, expected cost ~= N * rate."""
        cfg = NodeConfig(
            region="us-east",
            cost_per_hour=5.0,
            failure_dist=Exponential(rate=1.0 / days(365)),
            recovery_dist=Exponential(rate=1.0 / minutes(10)),
            data_loss_dist=Constant(float("inf")),
            log_replay_rate_dist=Constant(1e6),
            snapshot_download_time_dist=Constant(60),
            spawn_dist=Constant(300),
        )
        result = markov_analyze(
            [cfg] * 3,
            LeaderlessProtocol(),
            NodeReplacementStrategy(failure_timeout=Seconds(hours(1))),
            QualityLevel.SIMPLIFIED,
        )
        expected_hourly = 3 * 5.0
        assert result.expected_cost_per_hour == pytest.approx(expected_hourly, rel=1e-3)

    def test_cost_drops_when_dataloss_not_replaced(self):
        """NoOp strategy with data loss reaches the all-data-loss absorbing state.

        Steady-state expected cost is therefore 0 (all nodes are unbilled once
        data-lost and never replaced).
        """
        cfg = NodeConfig(
            region="us-east",
            cost_per_hour=5.0,
            failure_dist=Exponential(rate=1.0 / days(365)),
            recovery_dist=Exponential(rate=1.0 / minutes(10)),
            data_loss_dist=Exponential(rate=1.0 / days(30)),
            log_replay_rate_dist=Constant(1e6),
            snapshot_download_time_dist=Constant(60),
            spawn_dist=Constant(300),
        )
        result = markov_analyze(
            [cfg] * 3,
            LeaderlessProtocol(),
            NoOpStrategy(),
            QualityLevel.SIMPLIFIED,
        )
        assert result.expected_cost_per_hour is not None
        assert result.expected_cost_per_hour < 3 * 5.0
