"""
Sparse CTMC representation optimized for the numeric solver.

MarkovModel replaces the legacy dict/list-based ContinuousMarkovModel with
integer state IDs and a scipy.sparse Q matrix. Builders construct this
directly from enumerated states; the solver consumes it without any
string-keyed lookups.

State space conventions:
    - States are identified by contiguous integers [0, num_states).
    - state_names[i] is a human-readable label for state i (debug only).
    - initial_distribution[i] gives the starting probability of state i.
    - live_mask[i] is True iff state i satisfies the availability predicate.
    - state_costs[i] is the billing rate ($/second) while in state i.
    - Q is the infinitesimal generator: off-diagonal Q[i,j] is the rate
      of transition i -> j; diagonal Q[i,i] = -sum(Q[i, :]) over j != i.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse


@dataclass
class MarkovModel:
    """Continuous-time Markov chain with sparse generator.

    Attributes:
        Q: Generator matrix as a sparse CSR matrix of shape (n, n).
        initial_distribution: Probability of starting in each state,
            shape (n,), sums to 1.
        live_mask: Boolean mask of shape (n,); True where the system is
            "available" in the liveness sense.
        state_costs: Cost rate in dollars per second for each state,
            shape (n,). Defaults to zeros if not provided.
        state_names: Human-readable labels (debug only), length n.
    """

    Q: sparse.csr_matrix
    initial_distribution: np.ndarray
    live_mask: np.ndarray
    state_costs: np.ndarray
    state_names: list[str] = field(default_factory=list)

    @property
    def num_states(self) -> int:
        return self.Q.shape[0]

    @property
    def live_state_ids(self) -> np.ndarray:
        return np.flatnonzero(self.live_mask)

    @property
    def initial_state_id(self) -> int:
        """Index of the state with nonzero initial probability.

        Returns the first such state; builders currently always put
        all mass on a single starting state (the all-healthy cluster).
        """
        idx = np.flatnonzero(self.initial_distribution > 0)
        if idx.size == 0:
            raise ValueError("No state has nonzero initial probability")
        return int(idx[0])
