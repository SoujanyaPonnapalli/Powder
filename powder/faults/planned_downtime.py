"""Planned downtime schedules (subtask 1.1).

Each node runs an independent renewal process of planned-downtime starts:

* gap between starts ``W ~ period`` (default Normal(7 d, 12 h), min 1 d)
* each event takes the node down for ``D ~ duration`` (default: Ford et al.
  planned-reboot CDF), data kept
* phase ``equilibrium``: the first start is drawn from the stationary residual
  life of the renewal process, so the planned-downtime rate is constant from
  t = 0. Sampling: draw a length-biased gap W~ (accept W with probability
  W / w_max), then put the first start at U * W~. ``uniform_first_period``
  instead uses U * W_0 with an ordinary first gap.
* if a downtime would still be running at the next scheduled start, that
  start is deferred to the end of the current downtime (no overlap).

The same schedule feeds the trace generator (emulation) and Goal 5's
software-update scheduling, which can later replace the random phase.
For Markov/ILP use, :func:`as_transient` folds the process into an
effective exponential transient rate ``1/E[W]`` with mean duration ``E[D]``
(an approximation: the real process is periodic, not memoryless).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .dists import from_spec
from .schema import pick


@dataclass(frozen=True)
class PlannedEvent:
    node: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _gap_bound(period) -> float:
    """Upper bound used for length-biased rejection sampling of gaps."""
    m = period.mean
    sd = getattr(period, "std", 0.0) or 0.0
    return m + 8.0 * sd if sd > 0 else 4.0 * m


def _first_start(period, phase: str, rng: np.random.Generator) -> float:
    if phase == "uniform_first_period":
        return float(rng.random() * period.sample(rng))
    w_max = _gap_bound(period)
    for _ in range(10_000):
        w = min(float(period.sample(rng)), w_max)
        if rng.random() * w_max <= w:
            return float(rng.random() * w)
    raise RuntimeError("length-biased sampling did not converge")


def schedule(
    node_ids: Iterable[str],
    horizon: float,
    profile: dict[str, Any],
    rng: np.random.Generator,
    level: str = "central",
    start_time: float = 0.0,
) -> list[PlannedEvent]:
    """All planned-downtime events that start in ``[start_time, start_time + horizon)``."""
    period = from_spec(profile["period"])
    duration = from_spec(pick(profile["duration"], level, "duration"))
    phase = profile.get("phase", "equilibrium")
    end_time = start_time + horizon
    out: list[PlannedEvent] = []
    for node in node_ids:
        t = start_time + _first_start(period, phase, rng)
        while t < end_time:
            d = float(duration.sample(rng))
            out.append(PlannedEvent(node, t, t + d))
            t = max(t + float(period.sample(rng)), t + d)
    out.sort(key=lambda e: (e.start, e.node))
    return out


def as_transient(profile: dict[str, Any], level: str = "central") -> tuple[float, float]:
    """(rate per second, mean duration in seconds) for Markov/ILP folding."""
    period = from_spec(profile["period"])
    duration = from_spec(pick(profile["duration"], level, "duration"))
    return 1.0 / period.mean, float(duration.mean)


def long_run_fraction(profile: dict[str, Any], level: str = "central") -> float:
    """Expected fraction of time a node is in planned downtime, E[D]/E[W]."""
    rate, mean_d = as_transient(profile, level)
    return rate * mean_d
