"""JSON distribution specs <-> Distribution objects.

The spec vocabulary is powder-mc's ``DistributionSpec`` (exponential,
weibull, normal, uniform, constant) plus three library-only families:

* ``{"type": "lognormal", "mu": .., "sigma": ..}``
* ``{"type": "empirical", "x": [...], "F": [...], "log_x": true,
     "tail": <spec>}``  -- digitized CDF with a parametric tail
* ``{"type": "mixture", "components": [{"weight": w, "dist": <spec>}, ...]}``

powder-mc samples all of these natively (since 2026-10-21), except empirical
distributions with parametric tails; ``engine_compatible`` returns supported
distributions unchanged and maps the rest to a Weibull with the same mean and
median.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from powder.simulation.distributions import (
    Constant,
    Distribution,
    Empirical,
    Exponential,
    LogNormal,
    Mixture,
    Normal,
    Uniform,
    Weibull,
)

ENGINE_TYPES = {"exponential", "weibull", "normal", "uniform", "constant"}
ALL_TYPES = ENGINE_TYPES | {"lognormal", "empirical", "mixture"}


class SpecError(ValueError):
    pass


_CACHE: dict[str, Distribution] = {}


def from_spec(spec: dict[str, Any]) -> Distribution:
    """Build a Distribution from its JSON spec (memoised: specs are immutable,
    and empirical distributions precompute their mean on construction)."""
    key = json.dumps(spec, sort_keys=True) if isinstance(spec, dict) else None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    dist = _build(spec)
    if key is not None and len(_CACHE) < 10_000:
        _CACHE[key] = dist
    return dist


def _build(spec: dict[str, Any]) -> Distribution:
    if not isinstance(spec, dict) or "type" not in spec:
        raise SpecError(f"distribution spec must be an object with 'type': {spec!r}")
    t = spec["type"]
    try:
        if t == "exponential":
            return Exponential(float(spec["rate"]))
        if t == "weibull":
            return Weibull(float(spec["shape"]), float(spec["scale"]))
        if t == "normal":
            return Normal(float(spec["mean"]), float(spec["std"]), float(spec.get("min_val", 0.0)))
        if t == "uniform":
            return Uniform(float(spec["low"]), float(spec["high"]))
        if t == "constant":
            return Constant(float(spec["value"]))
        if t == "lognormal":
            return LogNormal(float(spec["mu"]), float(spec["sigma"]))
        if t == "mixture":
            comps = spec["components"]
            return Mixture([float(c["weight"]) for c in comps], [from_spec(c["dist"]) for c in comps])
        if t == "empirical":
            tail = from_spec(spec["tail"]) if spec.get("tail") is not None else None
            return Empirical(spec["x"], spec["F"], tail=tail, log_x=bool(spec.get("log_x", True)))
    except KeyError as e:
        raise SpecError(f"{t} spec missing field {e}: {spec!r}") from e
    raise SpecError(f"unknown distribution type {t!r}")


def to_spec(dist: Distribution) -> dict[str, Any]:
    """Inverse of :func:`from_spec`."""
    if isinstance(dist, Exponential):
        return {"type": "exponential", "rate": dist.rate}
    if isinstance(dist, Weibull):
        return {"type": "weibull", "shape": dist.shape, "scale": dist.scale}
    if isinstance(dist, Normal):
        return {"type": "normal", "mean": dist.mean, "std": dist.std, "min_val": dist.min_val}
    if isinstance(dist, Uniform):
        return {"type": "uniform", "low": dist.low, "high": dist.high}
    if isinstance(dist, Constant):
        return {"type": "constant", "value": dist.value}
    if isinstance(dist, LogNormal):
        return {"type": "lognormal", "mu": dist.mu, "sigma": dist.sigma}
    if isinstance(dist, Mixture):
        return {"type": "mixture", "components": [
            {"weight": w, "dist": to_spec(c)} for w, c in zip(dist.weights, dist.components)]}
    if isinstance(dist, Empirical):
        return {"type": "empirical", "x": [float(v) for v in dist.x], "F": [float(v) for v in dist.F],
                "log_x": dist.log_x, "tail": to_spec(dist.tail) if dist.tail is not None else None}
    raise SpecError(f"no spec for {type(dist).__name__}")


def median(dist: Distribution, rng_seed: int = 0) -> float:
    """Median: exact where a ppf exists, else estimated from 200k samples."""
    if hasattr(dist, "ppf") and (not isinstance(dist, Mixture) or all(hasattr(c, "cdf") and hasattr(c, "ppf")
                                                                     for c in dist.components)):
        return float(dist.ppf(0.5))
    if isinstance(dist, Constant):
        return dist.value
    if isinstance(dist, Uniform):
        return dist.mean
    if isinstance(dist, Normal):
        return max(dist.min_val, dist.mean)
    rng = np.random.default_rng(rng_seed)
    return float(np.median([dist.sample(rng) for _ in range(200_000)]))


def weibull_from_mean_median(mean: float, med: float) -> Weibull | Exponential:
    """Weibull with the given mean and median (bisection on the shape).

    For Weibull, median/mean = ln(2)^(1/k) / Gamma(1 + 1/k), which increases
    monotonically with k (0.0 as k -> 0, about 1.0 near k = 3.4). If the
    requested ratio is outside [k=0.05, k=3], the shape is clamped and only
    the mean is matched.
    """
    if mean <= 0 or med <= 0:
        raise SpecError("mean and median must be positive")
    r = med / mean

    def ratio(k: float) -> float:
        return math.log(2.0) ** (1.0 / k) / math.gamma(1.0 + 1.0 / k)

    lo, hi = 0.05, 3.0
    if r <= ratio(lo):  # heavier-tailed than any Weibull in range: clamp the shape, keep the mean
        return Weibull(lo, mean / math.gamma(1.0 + 1.0 / lo))
    if r >= ratio(hi):
        return Weibull(hi, mean / math.gamma(1.0 + 1.0 / hi))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio(mid) < r:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    scale = mean / math.gamma(1.0 + 1.0 / k)
    return Weibull(k, scale)


_ENGINE_CACHE: dict[int, tuple] = {}


def engine_compatible(dist: Distribution) -> Distribution:
    """Return a distribution the Rust engine accepts (same mean, same median).

    Memoised per distribution object (from_spec memoises those)."""
    hit = _ENGINE_CACHE.get(id(dist))
    if hit is not None and hit[0] is dist:
        return hit[1]
    out = _engine_compatible(dist)
    _ENGINE_CACHE[id(dist)] = (dist, out)
    return out


def _engine_native(dist: Distribution) -> bool:
    """True if powder-mc can sample ``dist`` exactly (it supports lognormal,
    tail-free empirical and mixtures of supported families since 2026-10-21)."""
    if type(dist).__name__ in {"Exponential", "Weibull", "Normal", "Uniform", "Constant", "LogNormal"}:
        return True
    if isinstance(dist, Empirical):
        return dist.F[0] == 0.0 and dist.F[-1] >= 1.0 - 1e-9
    if isinstance(dist, Mixture):
        return all(_engine_native(c) for c in dist.components)
    return False


def _engine_compatible(dist: Distribution) -> Distribution:
    if _engine_native(dist):
        return dist
    if isinstance(dist, Mixture) and len(dist.components) == 1:
        return engine_compatible(dist.components[0])
    return weibull_from_mean_median(dist.mean, median(dist))


def validate_spec(spec: Any, where: str) -> list[str]:
    """Return a list of problems (empty if the spec builds)."""
    try:
        from_spec(spec)
    except Exception as e:  # noqa: BLE001 - report any construction failure
        return [f"{where}: {e}"]
    return []
