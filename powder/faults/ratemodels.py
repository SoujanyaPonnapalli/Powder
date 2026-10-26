"""Rate models: constant, population, curve, curve_population (exec #1b §1).

A cause (or component) may carry a ``rate_model`` next to its ``rate_per_hour``
triple. The triple stays the summary used by profile-level code (scenario
levels, catalogue tables); the rate model is what a *machine* draws from when
it is instantiated (:func:`instantiate`).

Specs (all rates per hour, times in hours):

* ``{"type": "constant"}`` -- the triple's central value (implicit default).
* ``{"type": "population", "members": [{"id", "rate_per_hour", "weight"}, ...]}``
  -- a distribution over units (drive models, systems, cells, pools); a machine
  draws one member with probability ∝ weight.
  ``{"type": "population", "lognormal": {"mean": m, "sigma": s}}`` -- a fitted
  population when only summary statistics exist (e.g. Ford Table 1).
* ``{"type": "curve", "clock": "age" | "calendar", "edges_h": [0, e1, ..., en],
  "rate_per_hour": [r1, ..., rn], "periodic": bool, "start_h": {...}}`` -- a
  piecewise-constant hazard. ``age`` curves run on the unit's age (a machine
  draws its age at t = 0 from ``start_h``); ``calendar`` curves run on wall
  time (a machine, or a whole pool, draws an offset into the series).
  Past the last edge the last rate continues, unless ``periodic`` (the series
  repeats). ``scale_with`` lets a curve hold only a *relative* shape
  (mean 1) that multiplies a level drawn from a population.
* ``{"type": "curve_population", "members": [{"id", "weight", <curve fields>}]}``
  -- a machine draws one member curve.

``start_h`` is ``{"type": "uniform"}`` (offset uniform over the curve's span),
``{"type": "fixed", "value": h}``, or ``{"type": "weights", "edges_h": [...],
"weights": [...]}`` (piecewise-uniform, e.g. a drive age distribution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODEL_TYPES = {"constant", "population", "curve", "curve_population"}
CLOCKS = {"age", "calendar"}


# --------------------------------------------------------------------------- hazards

@dataclass(frozen=True)
class PiecewiseHazard:
    """Piecewise-constant hazard h(t) on [edges[i], edges[i+1]) (hours).

    Past the last edge: the last rate continues, or the curve repeats when
    ``periodic``. ``offset`` is the clock position at machine time 0.
    """

    edges: tuple[float, ...]
    rates: tuple[float, ...]
    periodic: bool = False
    offset: float = 0.0

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.rates) + 1 or self.edges[0] != 0.0:
            raise ValueError("edges must start at 0 and have len(rates) + 1 entries")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError("edges must increase")
        if any(r < 0 for r in self.rates):
            raise ValueError("negative rate")

    @property
    def span(self) -> float:
        return self.edges[-1]

    def _cum_clock(self, x: float) -> float:
        """Cumulative hazard from clock 0 to clock x."""
        e, r = np.asarray(self.edges), np.asarray(self.rates)
        full = float(np.sum(np.diff(e) * r))
        if x <= 0:
            return 0.0
        if x >= self.span:
            if self.periodic:
                k, rem = divmod(x, self.span)
                return k * full + self._cum_clock(rem)
            return full + (x - self.span) * r[-1]
        i = int(np.searchsorted(e, x, side="right") - 1)
        return float(np.sum(np.diff(e[: i + 1]) * r[:i]) + (x - e[i]) * r[i])

    def cumulative(self, t0: float, t1: float) -> float:
        """∫ h over machine time [t0, t1] (hours)."""
        return self._cum_clock(self.offset + t1) - self._cum_clock(self.offset + t0)

    def mean_rate(self, t0: float, horizon_h: float) -> float:
        """Average hazard over machine time [t0, t0 + horizon] (the Markov approximation)."""
        return self.cumulative(t0, t0 + horizon_h) / horizon_h

    def rate_at(self, t: float) -> float:
        x = self.offset + t
        if x >= self.span:
            x = (x % self.span) if self.periodic else self.span - 1e-12
        i = int(np.searchsorted(self.edges, x, side="right") - 1)
        return self.rates[min(i, len(self.rates) - 1)]

    def next_event(self, t0: float, rng: np.random.Generator) -> float:
        """Machine time of the next event after t0 (NHPP by cumulative-hazard inversion); inf if none."""
        target = rng.exponential(1.0)
        e = self.edges
        x = self.offset + t0
        acc = 0.0
        for _ in range(1_000_000):
            pos = (x % self.span) if (self.periodic and x >= self.span) else x
            if not self.periodic and pos >= self.span:
                r = self.rates[-1]
                return math.inf if r <= 0 else (x + (target - acc) / r) - self.offset
            i = int(np.searchsorted(e, pos, side="right") - 1)
            seg_end = e[i + 1] - pos
            r = self.rates[i]
            if r > 0 and acc + r * seg_end >= target:
                return (x + (target - acc) / r) - self.offset
            acc += r * seg_end
            x += seg_end
            if self.periodic and x - (self.offset + t0) > 1e4 * self.span and acc == 0.0:
                return math.inf
        return math.inf

    def scaled(self, factor: float) -> "PiecewiseHazard":
        return PiecewiseHazard(self.edges, tuple(r * factor for r in self.rates), self.periodic, self.offset)

    def with_offset(self, offset: float) -> "PiecewiseHazard":
        return PiecewiseHazard(self.edges, self.rates, self.periodic, float(offset))

    def to_spec(self) -> dict[str, Any]:
        return {"edges_h": list(self.edges), "rate_per_hour": list(self.rates), "periodic": self.periodic,
                "offset_h": self.offset}


def hazard_from_spec(spec: dict[str, Any]) -> PiecewiseHazard:
    return PiecewiseHazard(tuple(float(x) for x in spec["edges_h"]), tuple(float(x) for x in spec["rate_per_hour"]),
                           bool(spec.get("periodic", False)), float(spec.get("offset_h", 0.0)))


# --------------------------------------------------------------------------- instantiation

@dataclass(frozen=True)
class MachineRate:
    """One cause's rate for one concrete machine."""

    kind: str                         # "constant" | "curve"
    rate_per_hour: float = 0.0        # constant rate (kind == constant)
    hazard: PiecewiseHazard | None = None
    clock: str | None = None          # age | calendar
    member: str | None = None         # population member id, if any
    draws: dict[str, Any] = field(default_factory=dict)

    def mean_rate(self, horizon_h: float, t0: float = 0.0) -> float:
        if self.kind == "constant":
            return self.rate_per_hour
        assert self.hazard is not None
        return self.hazard.mean_rate(t0, horizon_h)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind}
        if self.kind == "constant":
            d["rate_per_hour"] = self.rate_per_hour
        else:
            d["clock"] = self.clock
            d["hazard"] = self.hazard.to_spec()  # type: ignore[union-attr]
        if self.member is not None:
            d["member"] = self.member
        if self.draws:
            d["draws"] = self.draws
        return d


def machine_rate_from_dict(d: dict[str, Any]) -> MachineRate:
    if d["kind"] == "constant":
        return MachineRate("constant", float(d["rate_per_hour"]), member=d.get("member"), draws=d.get("draws", {}))
    return MachineRate("curve", hazard=hazard_from_spec(d["hazard"]), clock=d["clock"], member=d.get("member"),
                       draws=d.get("draws", {}))


def _choose(members: list[dict[str, Any]], rng: np.random.Generator, member_id: str | None) -> dict[str, Any]:
    if member_id is not None:
        for m in members:
            if m["id"] == member_id:
                return m
        raise KeyError(f"no member {member_id!r}")
    w = np.array([float(m.get("weight", 1.0)) for m in members])
    return members[int(rng.choice(len(members), p=w / w.sum()))]


def _start(spec: dict[str, Any] | None, span: float, rng: np.random.Generator) -> float:
    spec = spec or {"type": "uniform"}
    t = spec["type"]
    if t == "fixed":
        return float(spec["value"])
    if t == "uniform":
        return float(rng.uniform(0.0, span))
    if t == "weights":
        e, w = np.asarray(spec["edges_h"], float), np.asarray(spec["weights"], float)
        i = int(rng.choice(len(w), p=w / w.sum()))
        return float(rng.uniform(e[i], e[i + 1]))
    raise ValueError(f"unknown start_h type {t!r}")


def _curve(spec: dict[str, Any], rng: np.random.Generator, level: float | None, offset: float | None):
    h = hazard_from_spec(spec)
    if spec.get("scale_with") == "level" and level is not None:
        h = h.scaled(level)  # relative shape (mean 1) x a drawn level
    off = _start(spec.get("start_h"), h.span, rng) if offset is None else offset
    return h.with_offset(off), off


def instantiate(model: dict[str, Any] | None, triple: dict[str, float], rng: np.random.Generator, *,
                member_id: str | None = None, offset_h: float | None = None) -> MachineRate:
    """Draw one machine's rate for a cause.

    ``member_id`` pins a population member (catalogue recipes choose members
    explicitly); ``offset_h`` pins a calendar offset (shared by a spot pool).
    """
    model = model or {"type": "constant"}
    t = model["type"]
    if t == "constant":
        return MachineRate("constant", float(triple["central"]))
    if t == "population":
        if "members" in model:
            m = _choose(model["members"], rng, member_id)
            level, mid = float(m["rate_per_hour"]), m["id"]
        else:
            ln = model["lognormal"]
            mu = math.log(ln["mean"]) - ln["sigma"] ** 2 / 2
            z = float(rng.standard_normal())
            level, mid = float(math.exp(mu + ln["sigma"] * z)), None
        if "max_per_hour" in model:
            level = min(level, float(model["max_per_hour"]))
        if "curve" in model:  # population of levels x one relative calendar/age shape
            h, off = _curve(model["curve"], rng, level, offset_h)
            return MachineRate("curve", hazard=h, clock=model["curve"]["clock"], member=mid,
                               draws={"level_per_hour": level, "offset_h": off})
        return MachineRate("constant", level, member=mid, draws={} if mid else {"level_per_hour": level})
    if t == "curve":
        h, off = _curve(model, rng, float(triple["central"]), offset_h)
        draws = {"offset_h": off}
        if model["clock"] == "age" and model.get("start_h"):
            draws["start_h"] = model["start_h"]
        return MachineRate("curve", hazard=h, clock=model["clock"], draws=draws)
    if t == "curve_population":
        m = _choose(model["members"], rng, member_id)
        spec = {**{k: v for k, v in model.items() if k not in ("type", "members")}, **m}
        h, off = _curve(spec, rng, None, offset_h)
        draws = {"offset_h": off}
        if spec["clock"] == "age" and spec.get("start_h"):
            draws["start_h"] = spec["start_h"]  # a replacement unit draws a fresh age from this
        return MachineRate("curve", hazard=h, clock=spec["clock"], member=m["id"], draws=draws)
    raise ValueError(f"unknown rate model {t!r}")


def member_ids(model: dict[str, Any] | None) -> list[str | None]:
    """Member ids a recipe can pin (``[None]`` when the model has no discrete members)."""
    if model and model.get("type") in ("population", "curve_population") and "members" in model:
        return [m["id"] for m in model["members"]]
    return [None]


# --------------------------------------------------------------------------- validation

def _check_curve(spec: dict[str, Any], where: str) -> list[str]:
    errs = []
    if spec.get("clock") not in CLOCKS:
        errs.append(f"{where}: clock must be one of {sorted(CLOCKS)}")
    try:
        hazard_from_spec(spec)
    except (KeyError, ValueError, TypeError) as exc:
        errs.append(f"{where}: bad curve ({exc})")
    return errs


def validate_model(model: Any, where: str) -> list[str]:
    if model is None:
        return []
    if not isinstance(model, dict) or model.get("type") not in MODEL_TYPES:
        return [f"{where}: rate_model.type must be one of {sorted(MODEL_TYPES)}"]
    t = model["type"]
    errs: list[str] = []
    if t == "population":
        if "members" in model:
            if not model["members"]:
                errs.append(f"{where}: empty population")
            ids = [m.get("id") for m in model["members"]]
            if len(set(ids)) != len(ids):
                errs.append(f"{where}: duplicate member ids")
            for m in model["members"]:
                if not (float(m.get("rate_per_hour", -1)) >= 0 and float(m.get("weight", 1)) > 0):
                    errs.append(f"{where}: member {m.get('id')} needs rate_per_hour >= 0 and weight > 0")
        elif not ("lognormal" in model and model["lognormal"].get("mean", 0) > 0 and model["lognormal"].get("sigma", -1) >= 0):
            errs.append(f"{where}: population needs members or a lognormal {{mean, sigma}}")
        if "curve" in model:
            errs += _check_curve(model["curve"], f"{where}.curve")
    elif t == "curve":
        errs += _check_curve(model, where)
    elif t == "curve_population":
        if not model.get("members"):
            errs.append(f"{where}: empty curve population")
        for m in model.get("members", []):
            errs += _check_curve({**{k: v for k, v in model.items() if k not in ("type", "members")}, **m},
                                 f"{where}.{m.get('id')}")
    return errs
