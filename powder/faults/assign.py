"""Assign on-demand prices to machine classes (plan §3.4).

No source pairs these fault rates with a price, so prices are drawn from
the empirical distribution of current on-demand $/h of 8 vCPU x86 AWS
instance types and assigned two ways, kept side by side:

* rho = 0 (independent): a seeded random permutation, unrelated to reliability
* rho = 1 (reliability-ranked): the most reliable class gets the highest price

Both are instances of a Gaussian-copula rule:
``s_m = Phi^-1((rank_m - 0.5)/n)`` (rank 1 = least reliable),
``z_m = rho s_m + sqrt(1 - rho^2) eps_m``; the class with the k-th smallest z
gets the k-th smallest price. Prices are the stratified quantiles
``F^-1((k - 0.5)/n)`` of the distribution, so the price multiset is fixed and
spans the distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from powder.simulation.distributions import normal_ppf

from .convert import node_profile_rates
from .schema import PERMANENT_CAUSES, TRANSIENT_CAUSES

HOURS_PER_YEAR = 8766.0


def reliability_score(profile: dict[str, Any], rebuild_hours: float = 1.0, level: str = "central") -> float:
    """Expected unavailable hours per node-year from node-level causes.

    ``H * sum_transient(lambda_i * E[D_i]) + H * lambda_P * R``. Planned
    downtime and region outages are excluded: they are identical for every
    machine class in a scenario, so they cannot change the ranking.
    """
    rates = node_profile_rates(profile, level)
    down = 0.0
    for c, r in rates.items():
        if c in TRANSIENT_CAUSES and r.rate_per_second > 0:
            down += r.rate_per_second * r.duration.mean
        elif c in PERMANENT_CAUSES:
            down += r.rate_per_second * rebuild_hours * 3600.0
    return down * HOURS_PER_YEAR  # (per-second rate x seconds) x hours per year -> hours per year


def stratified_prices(points: Sequence[float], n: int) -> np.ndarray:
    """n prices at the (k - 0.5)/n quantiles of the empirical points (ascending)."""
    p = np.sort(np.asarray(points, float))
    return np.quantile(p, (np.arange(n) + 0.5) / n)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            conc += s > 0
            disc += s < 0
    pairs = n * (n - 1) / 2
    return float((conc - disc) / pairs) if pairs else 1.0


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if len(a) < 2:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class Assignment:
    rho: float
    seed: int
    prices: dict[str, float]
    realised_spearman: float  # Spearman(price, reliability) where reliability = -score


def assign_prices(scores: dict[str, float], price_points: Sequence[float], rho: float, seed: int = 0) -> Assignment:
    """Gaussian-copula assignment of stratified prices to machine classes."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    ids = sorted(scores)  # deterministic order; ties broken by id
    n = len(ids)
    prices = stratified_prices(price_points, n)
    # rank 1 = least reliable = highest score
    order = sorted(ids, key=lambda m: (-scores[m], m))
    rank = {m: i + 1 for i, m in enumerate(order)}
    s = np.array([normal_ppf((rank[m] - 0.5) / n) for m in ids])
    eps = np.random.default_rng(seed).standard_normal(n)
    z = rho * s + math.sqrt(max(0.0, 1.0 - rho**2)) * eps
    out = {}
    for k, idx in enumerate(np.argsort(z, kind="stable")):
        out[ids[idx]] = float(prices[k])
    rel = [-scores[m] for m in ids]
    return Assignment(rho, seed, out, spearman([out[m] for m in ids], rel))
