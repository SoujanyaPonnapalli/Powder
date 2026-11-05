"""Reconfiguration accounting for the cadence study (scripts/reconfig_cadence.py, MVP Fig 8).

Pure functions, no MC or ILP:

* :func:`schedule` -- reconfiguration times for a cadence, and the evaluation blocks.
* :func:`match_rsms` -- pairs each new RSM with an old one so the fewest replicas move (RSMs are
  interchangeable shards of the same database, so only the multiset of placements matters).
* :func:`migration` -- replicas added/removed between consecutive deployments, GB copied and the transfer bill
  (each new replica is copied from the cheapest surviving member of its RSM).
* :func:`sla_violation_minutes` -- downtime above the SLA's error budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Placement = tuple[str, ...]


def schedule(horizon_h: float, cadence_h: float, block_h: float) -> tuple[list[float], list[tuple[float, float]]]:
    """(reconfiguration start times, evaluation blocks [t0, t1)). Blocks never straddle a reconfiguration."""
    if cadence_h <= 0 or block_h <= 0 or horizon_h <= 0:
        raise ValueError("horizon, cadence and block must be positive")
    starts = [float(t) for t in np.arange(0.0, horizon_h, cadence_h)]
    cuts = sorted(set(starts) | {float(t) for t in np.arange(0.0, horizon_h, block_h)} | {float(horizon_h)})
    return starts, [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def match_rsms(old: Sequence[Placement], new: Sequence[Placement]) -> list[tuple[int | None, int]]:
    """Assignment (old index or None, new index) maximizing shared machines (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment

    if not old:
        return [(None, j) for j in range(len(new))]
    overlap = np.array([[len(set(o) & set(n)) for n in new] for o in old], dtype=float)
    rows, cols = linear_sum_assignment(-overlap)
    pairs = {int(c): int(r) for r, c in zip(rows, cols)}
    return [(pairs.get(j), j) for j in range(len(new))]


@dataclass
class Migration:
    replicas_added: int
    replicas_removed: int
    rsms_changed: int
    gb_moved: float
    transfer_cost: float
    leader_moves: int          # RSMs whose whole old membership is gone (leader must change)


def migration(old: Sequence[Placement], new: Sequence[Placement], *, gb_per_replica: float,
              location: Callable[[str], tuple[str, str]],
              price_per_gb: Callable[[tuple[str, str], tuple[str, str]], float]) -> Migration:
    """Replicas copied when switching deployment ``old`` -> ``new``.

    ``location(machine) -> (region, az)``; ``price_per_gb(src, dst)`` is the $/GB between two locations. A new
    replica is copied from the member of its RSM's old placement with the cheapest transfer (the RSM keeps
    serving: add learner, catch up, remove the old member). A fresh RSM (no old match) is not counted.
    """
    added = removed = changed = leaders = 0
    gb = cost = 0.0
    for i, j in match_rsms(old, new):
        if i is None:
            continue
        o, n = set(old[i]), set(new[j])
        add, rem = n - o, o - n
        if not add and not rem:
            continue
        changed += 1
        added += len(add)
        removed += len(rem)
        leaders += int(not (o & n))
        for m in add:
            best = min(price_per_gb(location(s), location(m)) for s in o)
            gb += gb_per_replica
            cost += gb_per_replica * best
    return Migration(added, removed, changed, gb, cost, leaders)


def sla_violation_minutes(downtime_minutes: float, period_minutes: float, sla_nines: float) -> float:
    """Downtime above the error budget ``period x 10**-nines`` (0 if within budget)."""
    budget = period_minutes * 10.0 ** (-sla_nines)
    return max(0.0, downtime_minutes - budget)


def cadence_label(hours: float) -> str:
    for h, s in ((720, "1mo"), (336, "2w"), (168, "1w"), (48, "2d"), (24, "1d")):
        if math.isclose(hours, h):
            return s
    return f"{hours:g}h"
