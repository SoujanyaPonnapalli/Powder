"""
Sparse-aware numeric solver for continuous-time Markov chains.

Operates directly on MarkovModel (powder.markov). All linear-algebra
calls use scipy.sparse so the solver scales to the tens-of-thousands
of states produced by the higher-quality builders, and the code path
is straightforward to port to GPU sparse kernels later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from .markov import MarkovModel

_logger = logging.getLogger(__name__)

# Default residual threshold above which `steady_state` logs a warning.
# The sparse LU used by scipy is usually accurate to ~1e-12 for well-
# conditioned generators; 1e-8 leaves a generous margin for ill-conditioned
# chains (e.g. very stiff failure/recovery rate ratios) while still flagging
# genuine numerical breakdowns.
DEFAULT_RESIDUAL_THRESHOLD = 1e-8


@dataclass(frozen=True)
class SteadyStateResidual:
    """Residual diagnostics for a steady-state solve.

    Attributes:
        balance: ``||pi @ Q||_inf`` — how close the solution is to the
            global-balance equation ``pi Q = 0``. Should be at machine
            precision for a well-conditioned chain.
        normalization: ``|sum(pi) - 1|`` — the solution is renormalized
            before being returned from :func:`steady_state`, so this
            measures the raw LU solve before renormalization.
        negativity: ``-min(pi, 0)`` — magnitude of the most-negative
            entry. Physical steady states are nonnegative; small
            negative values (<= balance) are round-off and are clipped
            to zero.
    """

    balance: float
    normalization: float
    negativity: float

    @property
    def worst(self) -> float:
        """Max of the three residual components, for single-threshold checks."""
        return max(self.balance, self.normalization, self.negativity)


def compute_steady_state_residual(
    model: MarkovModel, pi: np.ndarray,
) -> SteadyStateResidual:
    """Compute balance / normalization / negativity residuals for ``pi``."""
    pi = np.asarray(pi, dtype=np.float64)
    balance = float(np.abs(pi @ model.Q).max()) if pi.size else 0.0
    normalization = float(abs(pi.sum() - 1.0))
    negativity = float(-pi.min()) if pi.size else 0.0
    return SteadyStateResidual(
        balance=balance,
        normalization=normalization,
        negativity=max(negativity, 0.0),
    )


def steady_state(
    model: MarkovModel,
    *,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, SteadyStateResidual]:
    """Compute the steady-state distribution pi with pi @ Q = 0, sum(pi)=1.

    Replaces the last row of Q^T with the normalization constraint and
    solves the resulting linear system. Uses scipy's sparse direct solver.
    After the solve, the residual ``pi @ Q`` is measured and a warning is
    logged if any component exceeds ``residual_threshold``. The returned
    vector is renormalized to ``sum(pi) == 1`` and clipped to be
    nonnegative.

    Args:
        model: The :class:`MarkovModel` to solve.
        residual_threshold: If any residual component (balance,
            normalization, negativity) is strictly greater than this
            value, a warning is logged via the module logger. Pass
            ``float('inf')`` to disable.
        return_residual: If true, returns ``(pi, residual)`` instead of
            just ``pi`` so callers can assert on the diagnostics.

    Raises:
        RuntimeError: If the chain has absorbing states and no unique
            stationary distribution (matrix singular after modification).
    """
    n = model.num_states
    QT = model.Q.T.tolil(copy=True)
    QT.rows[-1] = list(range(n))
    QT.data[-1] = [1.0] * n
    b = np.zeros(n, dtype=np.float64)
    b[-1] = 1.0
    try:
        pi_raw = splinalg.spsolve(QT.tocsr(), b)
    except RuntimeError as exc:
        raise RuntimeError(
            "Steady-state solve failed (likely absorbing state or reducible chain)"
        ) from exc
    pi_raw = np.asarray(pi_raw, dtype=np.float64)

    residual = compute_steady_state_residual(model, pi_raw)
    if residual.worst > residual_threshold:
        _logger.warning(
            "Steady-state residual exceeded threshold %g: "
            "balance=%.3e, normalization=%.3e, negativity=%.3e (num_states=%d)",
            residual_threshold,
            residual.balance,
            residual.normalization,
            residual.negativity,
            n,
        )

    pi = np.clip(pi_raw, 0.0, None)
    total = pi.sum()
    if total > 0:
        pi = pi / total

    if return_residual:
        return pi, residual
    return pi


def mean_first_passage(
    model: MarkovModel,
    absorbing_state_ids: np.ndarray | list[int],
) -> np.ndarray:
    """Mean first passage times to any state in absorbing_state_ids.

    For each transient state i, returns E[T_i] = mean time to first hit
    any state in the target set, starting from state i. Solves the
    truncated system Q_trunc @ t = -1 for the transient sub-matrix.

    Entries for absorbing states are 0.
    """
    n = model.num_states
    absorbing = np.zeros(n, dtype=bool)
    absorbing[list(absorbing_state_ids)] = True
    transient = ~absorbing

    result = np.zeros(n, dtype=np.float64)
    if not transient.any():
        return result

    transient_ids = np.flatnonzero(transient)
    Q_trunc = model.Q[transient_ids][:, transient_ids].tocsc()
    rhs = -np.ones(transient_ids.size, dtype=np.float64)
    t_transient = splinalg.spsolve(Q_trunc, rhs)
    result[transient_ids] = t_transient
    return result


def availability(model: MarkovModel, pi: np.ndarray | None = None) -> float:
    """Steady-state fraction of time the system is live.

    Args:
        model: The MarkovModel to analyze.
        pi: Optional precomputed steady state to avoid recomputation.
    """
    if pi is None:
        pi = steady_state(model)
    return float(pi[model.live_mask].sum())


def expected_cost_per_second(
    model: MarkovModel, pi: np.ndarray | None = None,
) -> float:
    """Steady-state expected billing rate in dollars per second.

    Integrates state_costs against the steady-state distribution:
        E[cost_rate] = sum_i pi[i] * state_costs[i]
    """
    if pi is None:
        pi = steady_state(model)
    return float(np.dot(pi, model.state_costs))


def build_q_from_triples(
    num_states: int,
    triples: list[tuple[int, int, float]],
) -> sparse.csr_matrix:
    """Build a sparse Q matrix from (i, j, rate) transition triples.

    Aggregates duplicates, sets the diagonal to -row_sum, and returns CSR.
    """
    if not triples:
        Q = sparse.csr_matrix((num_states, num_states), dtype=np.float64)
        return Q

    rows = np.fromiter((t[0] for t in triples), dtype=np.int64, count=len(triples))
    cols = np.fromiter((t[1] for t in triples), dtype=np.int64, count=len(triples))
    data = np.fromiter((t[2] for t in triples), dtype=np.float64, count=len(triples))

    Q = sparse.coo_matrix(
        (data, (rows, cols)), shape=(num_states, num_states), dtype=np.float64,
    ).tocsr()
    Q.sum_duplicates()
    Q.setdiag(0.0)
    Q.eliminate_zeros()
    row_sums = np.asarray(Q.sum(axis=1)).ravel()
    Q.setdiag(-row_sums)
    return Q
