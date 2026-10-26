"""
Time units and probability distributions for the Monte Carlo simulator.

All time values use seconds as the canonical unit. Helper functions provide
readable constructors for common time units.
"""

import math
from abc import ABC, abstractmethod
from typing import NewType

import numpy as np

# Explicit time unit - all times are in seconds
Seconds = NewType("Seconds", float)


def hours(h: float) -> Seconds:
    """Convert hours to seconds."""
    return Seconds(h * 3600)


def days(d: float) -> Seconds:
    """Convert days to seconds."""
    return Seconds(d * 86400)


def minutes(m: float) -> Seconds:
    """Convert minutes to seconds."""
    return Seconds(m * 60)


class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from the distribution.

        Args:
            rng: NumPy random number generator for reproducibility.

        Returns:
            A sampled value from the distribution.
        """
        pass

    @property
    @abstractmethod
    def mean(self) -> float:
        """Theoretical mean (expected value) of the distribution.

        Used for upfront estimation when planning sync paths
        (log-only vs snapshot), where sampling is not appropriate
        because the decision must be made before the sync starts.
        """
        pass

    @property
    def approx_rate(self) -> float:
        """Exponential rate approximation (1 / mean).

        For Exponential distributions this is exact. For others it gives
        the rate of the exponential distribution with matching mean,
        which is the approximation used by Markov chain builders.

        Returns inf when mean is 0 (instantaneous event).
        """
        m = self.mean
        if m <= 0:
            return float("inf")
        return 1.0 / m


class Exponential(Distribution):
    """Exponential distribution parameterized by rate.

    The exponential distribution is memoryless, making it suitable for
    modeling time until random events like failures.

    Args:
        rate: Events per unit time (1/mean). Must be positive.
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self.rate = rate

    @property
    def mean(self) -> float:
        return 1.0 / self.rate

    def sample(self, rng: np.random.Generator) -> float:
        """Sample time until next event."""
        # NumPy's exponential takes scale = 1/rate
        return rng.exponential(1.0 / self.rate)

    def cdf(self, x: float) -> float:
        return 0.0 if x <= 0 else 1.0 - math.exp(-self.rate * x)

    def ppf(self, p: float) -> float:
        return math.inf if p >= 1.0 else -math.log1p(-p) / self.rate

    def __repr__(self) -> str:
        return f"Exponential(rate={self.rate})"


class Weibull(Distribution):
    """Weibull distribution for modeling failure times.

    The Weibull distribution generalizes the exponential distribution and can
    model increasing (shape > 1) or decreasing (shape < 1) failure rates.

    Args:
        shape: Shape parameter (k). Must be positive.
               k < 1: decreasing failure rate (infant mortality)
               k = 1: constant failure rate (exponential)
               k > 1: increasing failure rate (wear-out)
        scale: Scale parameter (λ). Must be positive.
    """

    def __init__(self, shape: float, scale: float):
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        self.shape = shape
        self.scale = scale

    @property
    def mean(self) -> float:
        return self.scale * math.gamma(1.0 + 1.0 / self.shape)

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from the Weibull distribution."""
        return self.scale * rng.weibull(self.shape)

    def cdf(self, x: float) -> float:
        return weibull_cdf(self.shape, self.scale, x)

    def ppf(self, p: float) -> float:
        return weibull_ppf(self.shape, self.scale, p)

    def __repr__(self) -> str:
        return f"Weibull(shape={self.shape}, scale={self.scale})"


class Normal(Distribution):
    """Normal (Gaussian) distribution with optional lower bound.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation. Must be positive.
        min_val: Minimum value (samples below this are clamped). Defaults to 0.
    """

    def __init__(self, mean: float, std: float, min_val: float = 0.0):
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        self._mean = mean
        self.std = std
        self.min_val = min_val

    @property
    def mean(self) -> float:
        return self._mean

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value, clamped to min_val."""
        value = rng.normal(self._mean, self.std)
        return max(self.min_val, value)

    def __repr__(self) -> str:
        return f"Normal(mean={self._mean}, std={self.std}, min_val={self.min_val})"


class Uniform(Distribution):
    """Uniform distribution over [low, high].

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
    """

    def __init__(self, low: float, high: float):
        if low >= high:
            raise ValueError(f"Low must be less than high, got low={low}, high={high}")
        self.low = low
        self.high = high

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2.0

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value uniformly from [low, high)."""
        return rng.uniform(self.low, self.high)

    def cdf(self, x: float) -> float:
        return min(1.0, max(0.0, (x - self.low) / (self.high - self.low)))

    def ppf(self, p: float) -> float:
        return self.low + p * (self.high - self.low)

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


def normal_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF), Acklam's rational approximation.

    Relative error below 1.2e-9 over (0, 1); avoids a SciPy dependency so
    the fault library can run with NumPy alone.
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must be in [0, 1], got {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def normal_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class LogNormal(Distribution):
    """Log-normal distribution: ``ln X ~ Normal(mu, sigma)``.

    Args:
        mu: Mean of ``ln X`` (X in seconds, so ``exp(mu)`` is the median).
        sigma: Standard deviation of ``ln X``. Must be positive.
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        self.mu = mu
        self.sigma = sigma

    @property
    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma**2)

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return normal_cdf((math.log(x) - self.mu) / self.sigma)

    def ppf(self, p: float) -> float:
        return math.exp(self.mu + self.sigma * normal_ppf(p))

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))

    def __repr__(self) -> str:
        return f"LogNormal(mu={self.mu}, sigma={self.sigma})"


class Mixture(Distribution):
    """Finite mixture: pick component i with probability ``weights[i]``.

    For competing exponential risks this is exact for the recovery time of
    "a failure of any cause": cause i occurs with probability
    ``lambda_i / sum(lambda)``.

    Args:
        weights: Non-negative weights (normalised internally).
        components: One Distribution per weight.
    """

    def __init__(self, weights: list[float], components: list[Distribution]):
        if len(weights) != len(components) or not weights:
            raise ValueError("weights and components must be non-empty and equal length")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"weights must be non-negative with positive sum, got {weights}")
        total = float(sum(weights))
        self.weights = [w / total for w in weights]
        self.components = list(components)
        self._cum = np.cumsum(self.weights)

    @property
    def mean(self) -> float:
        return sum(w * c.mean for w, c in zip(self.weights, self.components))

    def sample(self, rng: np.random.Generator) -> float:
        i = int(np.searchsorted(self._cum, rng.random(), side="right"))
        i = min(i, len(self.components) - 1)
        return self.components[i].sample(rng)

    def cdf(self, x: float) -> float:
        """Weighted component CDFs (components must provide ``cdf``)."""
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))

    def ppf(self, p: float) -> float:
        """Quantile by bisection between the component quantiles."""
        lo = min(c.ppf(p) for c in self.components)
        hi = max(c.ppf(p) for c in self.components)
        if lo == hi:
            return lo
        for _ in range(200):
            mid = 0.5 * (lo + hi) if lo <= 0 else math.sqrt(lo * hi)
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-12 * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    def __repr__(self) -> str:
        return f"Mixture(weights={self.weights}, components={self.components})"


class Empirical(Distribution):
    """Distribution defined by digitized CDF points, with parametric tails.

    Between the points, F is interpolated linearly in ``log x`` (when
    ``log_x``) or in ``x``. Outside ``[x[0], x[-1]]`` the ``tail``
    distribution (any object with ``cdf``/``ppf``, e.g. LogNormal or
    Weibull fitted to the same points) supplies the shape, rescaled so the
    quantile function is continuous at both joins:

    * ``u < F[0]``:  ``x = tail.ppf(u / F[0] * tail.cdf(x[0]))``
    * ``u > F[-1]``: ``x = tail.ppf(tail.cdf(x[-1]) + (u - F[-1]) / (1 - F[-1])
      * (1 - tail.cdf(x[-1])))``

    If ``F[0] == 0`` / ``F[-1] == 1`` the corresponding tail is unused.

    Args:
        x: Strictly increasing support points (seconds), all > 0 if log_x.
        F: Non-decreasing CDF values in [0, 1] at those points.
        tail: Parametric distribution used beyond the digitized range.
        log_x: Interpolate in log-space (default True).
    """

    _MEAN_GRID = 200_000

    def __init__(self, x, F, tail=None, log_x: bool = True):
        x = np.asarray(x, dtype=float)
        F = np.asarray(F, dtype=float)
        if x.ndim != 1 or x.shape != F.shape or len(x) < 2:
            raise ValueError("x and F must be 1-D arrays of equal length >= 2")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        if np.any(np.diff(F) < 0) or F[0] < 0 or F[-1] > 1:
            raise ValueError("F must be non-decreasing within [0, 1]")
        if log_x and x[0] <= 0:
            raise ValueError("log_x requires x > 0")
        if (F[0] > 0 or F[-1] < 1) and tail is None:
            raise ValueError("a tail distribution is required when F does not span [0, 1]")
        self.x = x
        self.F = F
        self.tail = tail
        self.log_x = log_x
        self._t = np.log(x) if log_x else x
        # Flat stretches of F (no mass) are kept: np.interp brackets a tied
        # xp value from the right, so for u inside a rise it interpolates
        # between that rise's own endpoints, which is the correct inverse.
        self._Fk = F
        self._tk = self._t
        u = (np.arange(self._MEAN_GRID) + 0.5) / self._MEAN_GRID
        self._mean = float(np.mean(self._ppf_array(u)))

    def _ppf_array(self, u: np.ndarray) -> np.ndarray:
        out = np.empty_like(u, dtype=float)
        lo = u < self._Fk[0]
        hi = u > self._Fk[-1]
        mid = ~(lo | hi)
        t = np.interp(u[mid], self._Fk, self._tk)
        out[mid] = np.exp(t) if self.log_x else t
        if lo.any():
            c0 = self.tail.cdf(float(self.x[0]))
            out[lo] = [self.tail.ppf(v / self._Fk[0] * c0) for v in u[lo]]
        if hi.any():
            c1 = self.tail.cdf(float(self.x[-1]))
            out[hi] = [
                self.tail.ppf(c1 + (v - self._Fk[-1]) / (1 - self._Fk[-1]) * (1 - c1))
                for v in u[hi]
            ]
        return out

    def ppf(self, p: float) -> float:
        return float(self._ppf_array(np.array([p]))[0])

    def cdf(self, x: float) -> float:
        if x < self.x[0]:
            if self._Fk[0] == 0:
                return 0.0
            c0 = self.tail.cdf(float(self.x[0]))
            return 0.0 if c0 <= 0 else self.tail.cdf(x) / c0 * self._Fk[0]
        if x > self.x[-1]:
            if self._Fk[-1] == 1:
                return 1.0
            c1 = self.tail.cdf(float(self.x[-1]))
            return self._Fk[-1] + (self.tail.cdf(x) - c1) / (1 - c1) * (1 - self._Fk[-1])
        t = math.log(x) if self.log_x else x
        return float(np.interp(t, self._t, self.F))

    @property
    def mean(self) -> float:
        return self._mean

    def sample(self, rng: np.random.Generator) -> float:
        return float(self._ppf_array(np.array([rng.random()]))[0])

    def __repr__(self) -> str:
        return (f"Empirical(n={len(self.x)}, x=[{self.x[0]:.3g}..{self.x[-1]:.3g}], "
                f"F=[{self.F[0]:.3f}..{self.F[-1]:.3f}], tail={self.tail!r})")


def weibull_cdf(shape: float, scale: float, x: float) -> float:
    return 0.0 if x <= 0 else 1.0 - math.exp(-((x / scale) ** shape))


def weibull_ppf(shape: float, scale: float, p: float) -> float:
    if p >= 1.0:
        return math.inf
    return scale * (-math.log1p(-p)) ** (1.0 / shape)


class Constant(Distribution):
    """Constant (deterministic) distribution.

    Always returns the same value. Useful for testing or when a parameter
    is known exactly.

    Args:
        value: The constant value to return.
    """

    def __init__(self, value: float):
        self.value = value

    @property
    def mean(self) -> float:
        return self.value

    def sample(self, rng: np.random.Generator) -> float:
        """Return the constant value."""
        return self.value

    def cdf(self, x: float) -> float:
        return 1.0 if x >= self.value else 0.0

    def ppf(self, p: float) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Constant(value={self.value})"


class PiecewiseRate:
    """Piecewise-constant hazard (per second) on a clock that is ``offset`` at simulation time 0.

    ``edges`` start at 0 (seconds); past the last edge the last rate continues,
    or the series repeats when ``periodic``. ``age_start`` (optional) is a
    piecewise-uniform distribution ``(edges, weights)`` of the clock position
    for a *new* unit: a replacement node draws its age from it (age clocks);
    calendar clocks keep running for replacements.
    """

    __slots__ = ("edges", "rates", "periodic", "offset", "age_start", "_cum", "_full")

    def __init__(self, edges, rates, periodic: bool = False, offset: float = 0.0, age_start=None):
        self.edges = np.asarray(edges, float)
        self.rates = np.asarray(rates, float)
        if self.edges[0] != 0.0 or len(self.edges) != len(self.rates) + 1:
            raise ValueError("edges must start at 0 and have len(rates) + 1 entries")
        self.periodic = bool(periodic)
        self.offset = float(offset)
        self.age_start = age_start
        self._cum = np.concatenate([[0.0], np.cumsum(np.diff(self.edges) * self.rates)])
        self._full = float(self._cum[-1])

    def _cum_clock(self, x: float) -> float:
        if x <= 0:
            return 0.0
        span = self.edges[-1]
        if x >= span:
            if self.periodic:
                k, rem = divmod(x, span)
                return k * self._full + self._cum_clock(rem)
            return self._full + (x - span) * self.rates[-1]
        i = int(np.searchsorted(self.edges, x, side="right") - 1)
        return float(self._cum[i] + (x - self.edges[i]) * self.rates[i])

    def _inv_clock(self, c: float) -> float:
        """Smallest clock x with cum(x) >= c (inf if never)."""
        if c <= 0:
            return 0.0
        span = self.edges[-1]
        if self.periodic:
            if self._full <= 0:
                return math.inf
            k, rem = divmod(c, self._full)
            return k * span + self._inv_clock_in(rem)
        if c >= self._full:
            r = self.rates[-1]
            return math.inf if r <= 0 else span + (c - self._full) / r
        return self._inv_clock_in(c)

    def _inv_clock_in(self, c: float) -> float:
        i = int(np.searchsorted(self._cum, c, side="left"))
        i = max(1, min(i, len(self._cum) - 1))
        r = self.rates[i - 1]
        return float(self.edges[i - 1] + (c - self._cum[i - 1]) / r) if r > 0 else float(self.edges[i])

    def cumulative(self, t0: float, t1: float) -> float:
        return self._cum_clock(self.offset + t1) - self._cum_clock(self.offset + t0)

    def next_event(self, now: float, rng: np.random.Generator) -> float:
        """Simulation time of the next event after ``now`` (inverse cumulative hazard)."""
        x0 = self.offset + now
        x = self._inv_clock(self._cum_clock(x0) + rng.exponential(1.0))
        return math.inf if math.isinf(x) else x - self.offset

    def respawned(self, now: float, rng: np.random.Generator) -> "PiecewiseRate":
        """The same hazard for a replacement unit created at ``now`` (fresh age for age clocks)."""
        if self.age_start is None:
            return self
        e, w = np.asarray(self.age_start[0], float), np.asarray(self.age_start[1], float)
        i = int(rng.choice(len(w), p=w / w.sum()))
        age = float(rng.uniform(e[i], e[i + 1]))
        return PiecewiseRate(self.edges, self.rates, self.periodic, age - now, self.age_start)


class TimeVaryingHazard(Distribution):
    """Superposition of piecewise hazards and a constant rate (a non-homogeneous Poisson clock).

    The simulator calls :meth:`sample_at` with the current simulation time (a
    plain :meth:`sample` assumes time 0). ``mean`` is 1 / (average hazard over
    ``mean_horizon`` seconds from time 0), for planning heuristics only.
    """

    def __init__(self, parts: list, constant_rate: float = 0.0, mean_horizon: float = 30 * 86400.0):
        self.parts = list(parts)
        self.constant_rate = float(constant_rate)
        self.mean_horizon = float(mean_horizon)

    def sample_at(self, now: float, rng: np.random.Generator) -> float:
        nxt = math.inf
        for p in self.parts:
            nxt = min(nxt, p.next_event(now, rng))
        if self.constant_rate > 0:
            nxt = min(nxt, now + rng.exponential(1.0 / self.constant_rate))
        return nxt - now

    def sample(self, rng: np.random.Generator) -> float:
        return self.sample_at(0.0, rng)

    def average_rate(self, t0: float = 0.0, horizon: float | None = None) -> float:
        h = self.mean_horizon if horizon is None else horizon
        return self.constant_rate + sum(p.cumulative(t0, t0 + h) for p in self.parts) / h

    @property
    def mean(self) -> float:
        r = self.average_rate()
        return math.inf if r <= 0 else 1.0 / r

    def respawned(self, now: float, rng: np.random.Generator) -> "TimeVaryingHazard":
        return TimeVaryingHazard([p.respawned(now, rng) for p in self.parts], self.constant_rate, self.mean_horizon)


def sample_delay(dist: Distribution, now: float, rng: np.random.Generator) -> float:
    """Delay until the next event of ``dist`` from simulation time ``now`` (time-aware if supported)."""
    at = getattr(dist, "sample_at", None)
    return at(now, rng) if at is not None else dist.sample(rng)
