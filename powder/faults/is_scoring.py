"""Importance-sampled Rust MC for one RSM: permanent failures tilted up, runs reweighted by likelihood ratio.

Why: near 5 nines, a 3-replica RSM's downtime is dominated by rare runs where a majority of replicas lose their
data (permanent failures) before replacements catch up; safe-mode replacement cannot promote without quorum, so
the RSM stays down for the rest of the horizon. Plain MC sees a handful of those per 100k runs, so per-placement
scores vary 3-10x and an optimizer that selects on them picks lucky draws (docs/plans/
figs-01-whatif-modeling-scale-sensitivity.md, "Why the 1x deployment misses the 5-nine SLA").

Tilt: every node's permanent-failure clock (``data_loss_dist``, exponential) runs at ``beta`` x its rate; the
replacement nodes inherit the tilted config (``default_node_config=None``). A run with K permanent failures has
likelihood ratio

    LR = beta**(-K) * exp((beta - 1) * Lambda * H),     Lambda = sum of the original permanent rates,

treating every replica slot as exposed to its permanent clock for the whole horizon H (the gaps between a
permanent failure and the replacement's spawn, ~10 min per failure, are ignored: relative error
~ (beta - 1) * lambda * gap, below 1e-3 here). The estimator mean((1 - A_r) * LR_r) is unbiased for the
unavailability; with equal beta on every node, LR depends only on K.

Only exponential permanent clocks are tilted; any other distribution raises (no likelihood ratio available).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ISStats:
    runs: int
    unavailability: float
    se: float
    beta: float
    mean_permanent_failures: float

    @property
    def availability(self) -> float:
        return 1.0 - self.unavailability


def tilted_configs(configs: Sequence, beta: float):
    from powder.simulation.distributions import Exponential

    out, lam = [], 0.0
    for c in configs:
        d = c.data_loss_dist
        if not isinstance(d, Exponential):
            if math.isinf(getattr(d, "mean", math.inf)):
                out.append(c)        # never fails permanently
                continue
            raise TypeError(f"importance sampling needs an exponential permanent clock, got {type(d).__name__}")
        lam += d.rate
        out.append(replace(c, data_loss_dist=Exponential(d.rate * beta)))
    return out, lam


def is_stats(configs: Sequence, protocol, strategy, *, horizon_s: float, runs: int, seed: int, beta: float = 4.0,
             network=None, chunk: int = 100_000) -> ISStats:
    """Importance-sampled unavailability of one RSM (beta = 1 is plain MC)."""
    from powder.mc_backend import run_monte_carlo
    from powder.simulation import ClusterState, NetworkState, NodeState

    tilted, lam = tilted_configs(configs, beta)
    nodes = {f"n{i}": NodeState(node_id=f"n{i}", config=c) for i, c in enumerate(tilted)}
    cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=len(nodes))
    log_c = (beta - 1.0) * lam * horizon_s
    n, s1, s2, k_sum = 0, 0.0, 0.0, 0.0
    while n < runs:
        k = min(chunk, runs - n)
        r = run_monte_carlo(cluster, strategy, protocol, num_simulations=k, max_time=horizon_s,
                            stop_on_data_loss=False, seed=seed + n, network_config=network)
        u = 1.0 - np.asarray(r.availability_samples, dtype=float)
        K = np.asarray(r.dataloss_failure_samples, dtype=float)
        if u.size != k or K.size != k:
            raise ValueError("Rust MC returned fewer runs than requested")
        w = np.exp(log_c - K * math.log(beta)) if beta != 1.0 else np.ones(k)
        x = u * w
        s1, s2, k_sum = s1 + x.sum(), s2 + (x * x).sum(), k_sum + K.sum()
        n += k
    mean = s1 / n
    var = max(0.0, (s2 - n * mean * mean) / (n - 1))
    return ISStats(n, mean, math.sqrt(var / n), beta, k_sum / n)
