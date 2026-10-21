"""Fitting utilities (NumPy only).

* :func:`nelder_mead` -- small derivative-free minimiser
* CDF-curve fits for digitized data (no raw events, so least squares on
  the CDF rather than MLE): lognormal, Weibull, 2-component lognormal mixture
* :func:`ks_distance` -- sup-norm distance between a model CDF and points
* :func:`random_effects_pool` -- DerSimonian-Laird pooling on the log scale
* :func:`gamma_poisson_eb` -- empirical-Bayes Gamma-Poisson for count data
* :func:`gamma_ppf` -- Gamma quantile (Wilson-Hilferty start + Newton on the
  regularised incomplete gamma)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from powder.simulation.distributions import LogNormal, Mixture, Weibull, normal_cdf


def nelder_mead(f: Callable[[np.ndarray], float], x0: Sequence[float], step: float | Sequence[float] = 0.5,
                tol: float = 1e-10, max_iter: int = 20_000) -> tuple[np.ndarray, float]:
    """Minimise ``f`` from ``x0`` (standard Nelder-Mead with shrink)."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    steps = np.broadcast_to(np.asarray(step, dtype=float), (n,))
    simplex = [x0] + [x0 + np.eye(n)[i] * steps[i] for i in range(n)]
    vals = [f(s) for s in simplex]
    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + tol):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex[1:]]
                vals = [vals[0]] + [f(s) for s in simplex[1:]]
    i = int(np.argmin(vals))
    return simplex[i], float(vals[i])


# -- CDF fits ----------------------------------------------------------------

def _lognormal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (np.log(x) - mu) / sigma
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _weibull_cdf(x: np.ndarray, k: float, scale: float) -> np.ndarray:
    return 1.0 - np.exp(-((x / scale) ** k))


@dataclass
class CdfFit:
    family: str
    params: dict
    dist: object
    sse: float
    ks: float

    def spec(self) -> dict:
        from .dists import to_spec
        return to_spec(self.dist)


def ks_distance(cdf: Callable[[np.ndarray], np.ndarray], x: np.ndarray, F: np.ndarray,
                F_left: np.ndarray | None = None) -> float:
    """sup |model - data| over the points (both sides of jumps if given)."""
    m = cdf(x)
    d = np.abs(m - F)
    if F_left is not None:
        d = np.maximum(d, np.abs(m - F_left))
    return float(d.max())


def fit_lognormal_cdf(x, F, F_left=None) -> CdfFit:
    x, F = np.asarray(x, float), np.asarray(F, float)
    lx = np.log(x)
    mu0 = float(np.interp(0.5, F, lx))
    s0 = max(0.1, float(np.interp(0.84, F, lx) - mu0))

    def loss(p):
        mu, ls = p
        return float(np.sum((_lognormal_cdf(x, mu, math.exp(ls)) - F) ** 2))

    (mu, ls), sse = nelder_mead(loss, [mu0, math.log(s0)], step=[0.5, 0.3])
    sigma = math.exp(ls)
    ks = ks_distance(lambda v: _lognormal_cdf(v, mu, sigma), x, F, F_left)
    return CdfFit("lognormal", {"mu": mu, "sigma": sigma}, LogNormal(mu, sigma), sse, ks)


def fit_weibull_cdf(x, F, F_left=None) -> CdfFit:
    x, F = np.asarray(x, float), np.asarray(F, float)
    scale0 = float(np.exp(np.interp(0.632, F, np.log(x))))

    def loss(p):
        lk, lsc = p
        return float(np.sum((_weibull_cdf(x, math.exp(lk), math.exp(lsc)) - F) ** 2))

    (lk, lsc), sse = nelder_mead(loss, [math.log(0.7), math.log(scale0)], step=[0.3, 0.5])
    k, sc = math.exp(lk), math.exp(lsc)
    ks = ks_distance(lambda v: _weibull_cdf(v, k, sc), x, F, F_left)
    return CdfFit("weibull", {"shape": k, "scale": sc}, Weibull(k, sc), sse, ks)


def fit_lognormal_mixture_cdf(x, F, F_left=None) -> CdfFit:
    """Two-component lognormal mixture (5 parameters), several restarts."""
    x, F = np.asarray(x, float), np.asarray(F, float)
    lx = np.log(x)
    q = lambda p: float(np.interp(p, F, lx))  # noqa: E731

    def unpack(p):
        w = 1.0 / (1.0 + math.exp(-p[0]))
        return w, p[1], math.exp(p[2]), p[3], math.exp(p[4])

    def model(v, p):
        w, m1, s1, m2, s2 = unpack(p)
        return w * _lognormal_cdf(v, m1, s1) + (1 - w) * _lognormal_cdf(v, m2, s2)

    def loss(p):
        return float(np.sum((model(x, p) - F) ** 2))

    best = None
    for w0, a, b in ((0.5, 0.25, 0.75), (0.7, 0.35, 0.85), (0.3, 0.15, 0.6), (0.85, 0.4, 0.95)):
        p0 = [math.log(w0 / (1 - w0)), q(a), math.log(0.8), q(b), math.log(0.8)]
        p, sse = nelder_mead(loss, p0, step=[0.5, 0.5, 0.3, 0.5, 0.3])
        if best is None or sse < best[1]:
            best = (p, sse)
    p, sse = best
    w, m1, s1, m2, s2 = unpack(p)
    if m1 > m2:  # canonical order: short component first
        w, m1, s1, m2, s2 = 1 - w, m2, s2, m1, s1
    dist = Mixture([w, 1 - w], [LogNormal(m1, s1), LogNormal(m2, s2)])
    ks = ks_distance(lambda v: w * _lognormal_cdf(v, m1, s1) + (1 - w) * _lognormal_cdf(v, m2, s2), x, F, F_left)
    return CdfFit("lognormal_mixture2", {"weight1": w, "mu1": m1, "sigma1": s1, "mu2": m2, "sigma2": s2},
                  dist, sse, ks)


def best_cdf_fit(x, F, F_left=None) -> tuple[CdfFit, list[CdfFit]]:
    fits = [fit_lognormal_cdf(x, F, F_left), fit_weibull_cdf(x, F, F_left),
            fit_lognormal_mixture_cdf(x, F, F_left)]
    return min(fits, key=lambda f: f.ks), fits


# -- pooling -----------------------------------------------------------------

@dataclass
class Pooled:
    central: float
    low: float
    high: float
    tau2: float
    se: float
    k: int


#: Two-sided 90% Student-t quantiles (0.95 one-sided) by degrees of freedom.
_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
        10: 1.812, 12: 1.782, 15: 1.753, 20: 1.725, 30: 1.697, 60: 1.671}


def t95(dof: int) -> float:
    if dof <= 0:
        return math.inf
    keys = sorted(_T95)
    for k in keys:
        if dof <= k:
            return _T95[k]
    return 1.645


def random_effects_pool(rates: Sequence[float], variances: Sequence[float], z: float | None = None) -> Pooled:
    """DerSimonian-Laird on log rates; low/high = 90% prediction interval.

    The prediction interval uses Student-t with k - 2 degrees of freedom
    (Higgins, Thompson & Spiegelhalter 2009) unless ``z`` is given.
    """
    y = np.log(np.asarray(rates, float))
    v = np.asarray(variances, float)
    w = 1.0 / v
    ybar = float(np.sum(w * y) / np.sum(w))
    k = len(y)
    q = float(np.sum(w * (y - ybar) ** 2))
    denom = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / denom) if k > 1 and denom > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se = float(1.0 / math.sqrt(np.sum(ws)))
    crit = z if z is not None else (t95(k - 2) if k >= 3 else 1.645)
    half = crit * math.sqrt(se**2 + tau2)
    return Pooled(math.exp(mu), math.exp(mu - half), math.exp(mu + half), tau2, se, k)


# -- Gamma / Gamma-Poisson -------------------------------------------------

def gammainc_lower_reg(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) (series / continued fraction)."""
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(10_000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Lentz continued fraction for Q(a, x)
    b = x + 1 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_ppf(p: float, shape: float, rate: float) -> float:
    """Quantile of Gamma(shape, rate)."""
    from powder.simulation.distributions import normal_ppf
    z = normal_ppf(p)
    x = shape * (1 - 1 / (9 * shape) + z / (3 * math.sqrt(shape))) ** 3
    x = max(x, 1e-12 * shape)
    lo, hi = 0.0, max(10 * x, shape + 50 * math.sqrt(shape) + 50)
    for _ in range(300):  # bisection: robust for small shapes
        mid = 0.5 * (lo + hi)
        if gammainc_lower_reg(shape, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / rate


@dataclass
class GammaPoissonEB:
    alpha: float
    beta: float
    pooled_rate: float
    heterogeneous: bool
    loglik: float
    total_n: float
    total_exposure: float

    def posterior(self, n: float, exposure: float) -> tuple[float, float]:
        """(shape, rate) of the Gamma posterior for one unit.

        Without detectable heterogeneity every unit shares the pooled rate,
        whose posterior is Gamma(N + 1/2, T) (Jeffreys prior on the pool).
        """
        if not self.heterogeneous:
            return (self.total_n + 0.5, self.total_exposure)
        return (self.alpha + n, self.beta + exposure)

    def summary(self, n: float, exposure: float, lo_p: float = 0.05, hi_p: float = 0.95) -> dict:
        a, b = self.posterior(n, exposure)
        return {"mean": a / b, "low": gamma_ppf(lo_p, a, b), "high": gamma_ppf(hi_p, a, b)}


def gamma_poisson_eb(counts: Sequence[float], exposures: Sequence[float]) -> GammaPoissonEB:
    """Empirical-Bayes fit of lambda_r ~ Gamma(alpha, beta), n_r ~ Poisson(lambda_r T_r).

    Maximises the negative-binomial marginal likelihood
    prod_r Gamma(a+n)/(Gamma(a) n!) (b/(b+T))^a (T/(b+T))^n.
    If the optimum runs to alpha -> infinity (no detectable heterogeneity),
    returns ``heterogeneous=False`` and the pooled rate.
    """
    n = np.asarray(counts, float)
    T = np.asarray(exposures, float)
    pooled = float(n.sum() / T.sum())

    def nll(p):
        a, b = math.exp(p[0]), math.exp(p[1])
        ll = 0.0
        for ni, ti in zip(n, T):
            ll += (math.lgamma(a + ni) - math.lgamma(a) - math.lgamma(ni + 1)
                   + a * math.log(b / (b + ti)) + ni * math.log(ti / (b + ti)) if ni > 0 else
                   a * math.log(b / (b + ti)))
        return -ll

    best = None
    for a0 in (0.5, 2.0, 10.0, 50.0):
        b0 = a0 / max(pooled, 1e-12)
        p, v = nelder_mead(nll, [math.log(a0), math.log(b0)], step=[0.7, 0.7])
        if best is None or v < best[1]:
            best = (p, v)
    a, b = math.exp(best[0][0]), math.exp(best[0][1])
    hetero = a < 1e4
    return GammaPoissonEB(a, b, pooled, hetero, -best[1], float(n.sum()), float(T.sum()))


# -- small-sample duration MLE (for region-outage durations) -----------------

@dataclass
class MleFit:
    family: str
    params: dict
    loglik: float
    aic: float
    dist: object


def mle_duration_fits(x: Sequence[float]) -> list[MleFit]:
    """MLE fits of exponential, lognormal and Weibull to positive data."""
    from powder.simulation.distributions import Exponential
    x = np.asarray(x, float)
    n = len(x)
    lx = np.log(x)
    fits = []
    lam = 1.0 / x.mean()
    ll = n * math.log(lam) - lam * x.sum()
    fits.append(MleFit("exponential", {"rate": lam}, ll, 2 * 1 - 2 * ll, Exponential(lam)))
    mu, sig = float(lx.mean()), float(lx.std(ddof=0))
    ll = float(np.sum(-np.log(x * sig * math.sqrt(2 * math.pi)) - (lx - mu) ** 2 / (2 * sig**2)))
    fits.append(MleFit("lognormal", {"mu": mu, "sigma": sig}, ll, 2 * 2 - 2 * ll, LogNormal(mu, sig)))

    def nll(p):
        k, lam_ = math.exp(p[0]), math.exp(p[1])
        z = x / lam_
        return -float(np.sum(math.log(k / lam_) + (k - 1) * np.log(z) - z**k))

    (lk, ll_), v = nelder_mead(nll, [0.0, math.log(x.mean())], step=[0.3, 0.3])
    k, sc = math.exp(lk), math.exp(ll_)
    fits.append(MleFit("weibull", {"shape": k, "scale": sc}, -v, 2 * 2 + 2 * v, Weibull(k, sc)))
    return fits


def _cdf_of(dist) -> Callable[[np.ndarray], np.ndarray]:
    return lambda v: np.array([dist.cdf(float(t)) for t in np.atleast_1d(v)])


def ks_statistic(dist, x: Sequence[float]) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    F = _cdf_of(dist)(x)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))


def parametric_bootstrap_ks(fit: MleFit, x: Sequence[float], reps: int = 1000, seed: int = 0) -> float:
    """p-value of the KS statistic with parameters re-estimated per replicate."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    d0 = ks_statistic(fit.dist, x)
    exceed = 0
    for _ in range(reps):
        sim = np.array([fit.dist.sample(rng) for _ in range(len(x))])
        refit = next(f for f in mle_duration_fits(sim) if f.family == fit.family)
        exceed += ks_statistic(refit.dist, sim) >= d0
    return (exceed + 1) / (reps + 1)


# -- frailty (negative binomial) ----------------------------------------------

def nb_frailty_from_p_and_conditional_mean(p_any: float, mean_given_any: float) -> tuple[float, float]:
    """Gamma-Poisson (NB) parameters from P(N>=1) and E[N | N>=1].

    N ~ Poisson(Z mu), Z ~ Gamma(a, 1/a) (mean 1). Then
    P0 = (1 + mu/a)^-a and E[N | N>=1] = mu / (1 - P0).
    Returns (mu, a). ``a`` -> infinity means no heterogeneity.
    """
    mu = mean_given_any * p_any
    # Under pure Poisson, E[N | N>=1] = mu / (1 - exp(-mu)); larger observed
    # conditional means imply overdispersion (finite a).
    poisson_p = 1 - math.exp(-mu)
    if p_any >= poisson_p:
        return mu, math.inf
    lo, hi = 1e-6, 1e6
    for _ in range(300):  # P(N>=1) increases with a toward the Poisson value
        mid = math.sqrt(lo * hi)
        p_mid = 1 - (1 + mu / mid) ** (-mid)
        if p_mid < p_any:
            lo = mid
        else:
            hi = mid
    return mu, math.sqrt(lo * hi)
