"""Tests for the Markov model primitives and absorbing-state properties."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import powder.markov_solver as markov_solver_module
from powder.markov import MarkovModel
from powder.markov_solver import (
    SparseSolverUnavailable,
    SteadyStateResidual,
    build_q_from_triples,
    compute_steady_state_residual,
    mean_first_passage,
    mixing_time,
    resolve_sparse_solver_backend,
    steady_state,
    time_averaged_distribution,
    total_variation_distance,
    transient_distribution,
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


def test_steady_state_explicit_scipy_backend():
    model = _toy_two_state_model()
    pi = steady_state(model, backend="scipy")
    assert pi[0] == pytest.approx(0.75)
    assert pi[1] == pytest.approx(0.25)


def test_mean_first_passage_explicit_scipy_backend():
    model = _toy_two_state_model()
    mfp = mean_first_passage(model, [1], backend="scipy")
    assert mfp[0] == pytest.approx(1.0)
    assert mfp[1] == pytest.approx(0.0)


def test_transient_distribution_two_state_chain():
    model = _toy_two_state_model()
    p = transient_distribution(model, 0.5)
    expected_a = 0.75 + 0.25 * np.exp(-4.0 * 0.5)
    expected_b = 0.25 - 0.25 * np.exp(-4.0 * 0.5)
    assert p[0] == pytest.approx(expected_a)
    assert p[1] == pytest.approx(expected_b)
    assert p.sum() == pytest.approx(1.0)


def test_time_averaged_distribution_two_state_chain():
    model = _toy_two_state_model()
    horizon = 0.5
    p = time_averaged_distribution(model, horizon)
    transient_mass = (1.0 - np.exp(-4.0 * horizon)) / (4.0 * horizon)
    expected_a = 0.75 + 0.25 * transient_mass
    expected_b = 0.25 - 0.25 * transient_mass
    assert p[0] == pytest.approx(expected_a)
    assert p[1] == pytest.approx(expected_b)
    assert p.sum() == pytest.approx(1.0)


def test_time_averaged_distribution_zero_time_limit_is_initial_distribution():
    model = _toy_two_state_model()
    p = time_averaged_distribution(model, 0.0)
    assert np.allclose(p, model.initial_distribution)


def test_mixing_time_two_state_chain_closed_form():
    model = _toy_two_state_model()
    pi = steady_state(model)
    epsilon = 0.01
    expected = np.log(0.25 / epsilon) / 4.0
    actual = mixing_time(model, epsilon=epsilon, pi=pi)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)

    mixed = transient_distribution(model, actual)
    assert total_variation_distance(mixed, pi) == pytest.approx(epsilon, rel=1e-10)


def test_mixing_time_zero_when_initial_distribution_already_close():
    model = _toy_two_state_model()
    assert mixing_time(model, epsilon=0.25) == pytest.approx(0.0)


def test_auto_backend_falls_back_to_scipy_when_cupy_unavailable(monkeypatch):
    def unavailable():
        raise SparseSolverUnavailable("no CUDA in test")

    monkeypatch.delenv("POWDER_MARKOV_SOLVER", raising=False)
    monkeypatch.setattr(markov_solver_module, "_load_cupy_sparse_modules", unavailable)
    assert resolve_sparse_solver_backend("auto") == "scipy"

    model = _toy_two_state_model()
    pi = steady_state(model, backend="auto")
    assert pi[0] == pytest.approx(0.75)
    assert pi[1] == pytest.approx(0.25)


def test_explicit_cupy_backend_reports_unavailable(monkeypatch):
    def unavailable():
        raise SparseSolverUnavailable("no CUDA in test")

    monkeypatch.setattr(markov_solver_module, "_load_cupy_sparse_modules", unavailable)
    with pytest.raises(SparseSolverUnavailable):
        resolve_sparse_solver_backend("cupy")


def test_explicit_pardiso_backend_reports_unavailable(monkeypatch):
    def unavailable():
        raise SparseSolverUnavailable("no Pardiso in test")

    monkeypatch.setattr(markov_solver_module, "_load_pardiso_spsolve", unavailable)
    with pytest.raises(SparseSolverUnavailable):
        resolve_sparse_solver_backend("pardiso")


def test_pardiso_backend_uses_loaded_spsolve(monkeypatch):
    calls = []

    def fake_spsolve(A, b):
        calls.append(A.getformat())
        return np.linalg.solve(A.toarray(), b)

    monkeypatch.setattr(
        markov_solver_module,
        "_load_pardiso_spsolve",
        lambda: fake_spsolve,
    )
    A = sparse.csc_matrix(np.array([[4.0, 1.0], [2.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x = markov_solver_module._sparse_solve(A, b, backend="pardiso")
    assert np.allclose(A @ x, b)
    assert calls == ["csr"]


def test_explicit_cudss_backend_reports_unavailable(monkeypatch):
    def unavailable():
        raise SparseSolverUnavailable("no cuDSS in test")

    monkeypatch.setattr(markov_solver_module, "_load_cudss_direct_solver", unavailable)
    with pytest.raises(SparseSolverUnavailable):
        resolve_sparse_solver_backend("cudss")


def test_cudss_backend_uses_loaded_direct_solver(monkeypatch):
    calls = []

    class FakeDirectSolverOptions:
        def __init__(self, *, multithreading_lib):
            self.multithreading_lib = multithreading_lib

    fake_nvmath = SimpleNamespace(
        sparse=SimpleNamespace(
            advanced=SimpleNamespace(DirectSolverOptions=FakeDirectSolverOptions),
        ),
    )

    def fake_direct_solver(A, b, *, options=None):
        calls.append(A.getformat())
        if options is not None:
            calls.append(options.multithreading_lib)
        return np.linalg.solve(A.toarray(), b)

    monkeypatch.setattr(
        markov_solver_module,
        "_load_cudss_direct_solver",
        lambda: (fake_nvmath, fake_direct_solver),
    )
    monkeypatch.setenv("POWDER_MARKOV_CUDSS_MULTITHREADING_LIB", "/tmp/libiomp5.so")
    A = sparse.csc_matrix(np.array([[5.0, 1.0], [1.0, 4.0]]))
    b = np.array([2.0, 3.0])
    x = markov_solver_module._sparse_solve(A, b, backend="cudss")
    assert np.allclose(A @ x, b)
    assert calls == ["csr", "/tmp/libiomp5.so"]


def _fake_cupy_modules(*, info: int = 0):
    def gmres(A, b, **_kwargs):
        if info != 0:
            return np.zeros_like(b), info
        return np.linalg.solve(A.toarray(), b), 0

    cp = SimpleNamespace(
        asarray=lambda value, dtype=None: np.asarray(value, dtype=dtype),
        asnumpy=lambda value: np.asarray(value),
    )
    cpsparse = SimpleNamespace(csr_matrix=lambda value: value.tocsr())
    cpsplinalg = SimpleNamespace(gmres=gmres)
    return cp, cpsparse, cpsplinalg


def test_cupy_backend_uses_gmres(monkeypatch):
    monkeypatch.setattr(
        markov_solver_module,
        "_load_cupy_sparse_modules",
        lambda: _fake_cupy_modules(),
    )
    A = sparse.csr_matrix(np.array([[3.0, 1.0], [1.0, 2.0]]))
    b = np.array([1.0, 0.0])
    x = markov_solver_module._sparse_solve(A, b, backend="cupy")
    assert np.allclose(A @ x, b)


def test_cupy_backend_reports_gmres_nonconvergence(monkeypatch):
    monkeypatch.setattr(
        markov_solver_module,
        "_load_cupy_sparse_modules",
        lambda: _fake_cupy_modules(info=7),
    )
    A = sparse.eye(2, format="csr")
    b = np.ones(2)
    with pytest.raises(RuntimeError, match="GMRES sparse solve did not converge"):
        markov_solver_module._sparse_solve(A, b, backend="cupy")


def test_steady_state_cupy_backend_matches_scipy_when_available():
    try:
        resolve_sparse_solver_backend("cupy")
    except SparseSolverUnavailable as exc:
        pytest.skip(f"CuPy/CUDA unavailable: {exc}")

    model = _toy_two_state_model()
    pi_cpu = steady_state(model, backend="scipy")
    pi_gpu = steady_state(model, backend="cupy")
    assert np.allclose(pi_gpu, pi_cpu, rtol=1e-10, atol=1e-12)


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

    def test_markov_analyze_can_compute_mixing_time(self):
        cfg = _dataloss_config()
        strategy = NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))
        result = markov_analyze(
            [cfg] * 3,
            LeaderlessProtocol(),
            strategy,
            QualityLevel.SIMPLIFIED,
            mixing_time_epsilon=0.01,
        )
        assert result.mixing_time_to_steady_state is not None
        assert result.mixing_time_to_steady_state > 0.0

    def test_markov_analyze_can_compute_time_averaged_distribution(self):
        cfg = _dataloss_config()
        strategy = NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))
        result = markov_analyze(
            [cfg] * 3,
            LeaderlessProtocol(),
            strategy,
            QualityLevel.SIMPLIFIED,
            time_average_seconds=hours(1),
        )
        assert result.time_averaged_distribution is not None
        assert sum(result.time_averaged_distribution.values()) == pytest.approx(1.0)


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
