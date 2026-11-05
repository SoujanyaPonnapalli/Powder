"""Budget-targeted ILP candidates: knapsack fleets, packing passes, dominance (docs/plans/ilp-fleet-sampling.md).

A budget bounds the *fleet* (machines are billed once and host ``capacity`` replicas), so candidates are
built per budget: choose several fleets that use the budget, then pack 3/5/7-replica placements inside
each fleet. Placements are ranked by a centroid MC table (:mod:`centroid_table`) and only exact MC
scores reach the ILP.

* :func:`features` -- per-machine rates and price from the (scaled) optimizer machines.
* :func:`kmeans_log` / :func:`medoids` -- 2-D clusters on (log transient, log permanent rate).
* :func:`fleets_for_budget` -- fixed-size knapsack ILPs swept over fleet sizes, overlap-limited repeats.
* :func:`balanced_mix` -- the most balanced RF 3/5/7 RSM mix a fleet's slots allow (splits sampling effort).
* :func:`pack_pass` -- one randomized, slot-consuming pass of placements of one RF inside a fleet.
* :func:`dominate` -- swap members for fleet machines that are no dearer and no less reliable.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np

#: log10 floor for zero rates (per hour) when clustering
LOG_FLOOR = -9.0


@dataclass(frozen=True)
class MachineFeatures:
    machine_id: str
    region: str
    az: str
    price: float
    lam_t: float          # transient rate per hour (incl. planned downtime)
    lam_p: float          # permanent rate per hour (disk, host loss, spot)
    recovery_s: float     # mean transient recovery

    def dominates(self, other: "MachineFeatures") -> bool:
        """No dearer and no less reliable on every axis, strictly better on one."""
        a = (self.price, self.lam_t, self.lam_p, self.recovery_s)
        b = (other.price, other.lam_t, other.lam_p, other.recovery_s)
        return all(x <= y for x, y in zip(a, b)) and a != b


def _rate_per_hour(dist) -> float:
    mean = float(dist.mean)
    return 3600.0 / mean if 0 < mean < math.inf else 0.0


def features(machines: Sequence) -> list[MachineFeatures]:
    """From optimizer ``Machine`` objects (``ilp.catalogue_machines``, rate scale already applied)."""
    out = []
    for m in machines:
        c = m.node_config
        out.append(MachineFeatures(m.machine_id, c.region, c.az, float(c.cost_per_hour),
                                   _rate_per_hour(c.failure_dist), _rate_per_hour(c.data_loss_dist),
                                   float(c.recovery_dist.mean)))
    return out


def log_points(feats: Sequence[MachineFeatures]) -> np.ndarray:
    lt = np.log10(np.maximum([f.lam_t for f in feats], 10 ** LOG_FLOOR))
    lp = np.log10(np.maximum([f.lam_p for f in feats], 10 ** LOG_FLOOR))
    return np.column_stack([lt, lp])


def kmeans_log(feats: Sequence[MachineFeatures], k: int = 16, seed: int = 0,
               iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """k-means++ on (log10 transient, log10 permanent rate). Returns (labels, centroids)."""
    x = log_points(feats)
    rng = np.random.default_rng(seed)
    k = min(k, len(np.unique(x, axis=0)))
    cent = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        cent.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    cent = np.array(cent)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[labels == j].mean(0) if np.any(labels == j) else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return labels, cent


def medoids(feats: Sequence[MachineFeatures], labels: np.ndarray, centroids: np.ndarray) -> list[int]:
    """Index of the machine closest to each centroid (the cluster's MC representative)."""
    x = log_points(feats)
    out = []
    for j, c in enumerate(centroids):
        idx = np.flatnonzero(labels == j)
        out.append(int(idx[np.argmin(((x[idx] - c) ** 2).sum(-1))]))
    return out


# --------------------------------------------------------------------------- fleets (knapsack)

def _knapsack(feats: Sequence[MachineFeatures], u: np.ndarray, budget: float, *, alpha: float, min_k: int,
              k: int | None = None, objective: str = "u", excluded: Iterable[frozenset] = (),
              az_slack: float = 2.0, time_limit: float = 20.0) -> frozenset | None:
    """Binary fleet selection. ``objective``: ``u`` (min total downtime score, fixed size ``k``),
    ``min_count`` or ``max_count``. Constraints: alpha*B <= price <= B, size >= min_k, per-AZ count
    <= size/12 + az_slack, and at most 70% overlap with each ``excluded`` fleet."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(feats)
    price = np.array([f.price for f in feats])
    rows, lo, hi = [price], [alpha * budget], [budget]
    ones = np.ones(n)
    if k is None:
        rows.append(ones), lo.append(min_k), hi.append(n)
    else:
        rows.append(ones), lo.append(k), hi.append(k)
    azs = sorted({f.az for f in feats})
    n_az = len(azs)
    for az in azs:  # count(az) - size / n_az <= slack
        rows.append(np.array([1.0 if f.az == az else 0.0 for f in feats]) - 1.0 / n_az)
        lo.append(-np.inf), hi.append(az_slack)
    ids = [f.machine_id for f in feats]
    for fleet in excluded:
        rows.append(np.array([1.0 if mid in fleet else 0.0 for mid in ids]))
        lo.append(-np.inf), hi.append(math.floor(0.7 * len(fleet)))
    c = {"u": u, "min_count": ones, "max_count": -ones}[objective]
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=ones,
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None
    return frozenset(mid for mid, v in zip(ids, res.x) if v > 0.5)


def fleets_for_budget(feats: Sequence[MachineFeatures], u: np.ndarray, budget: float, *, num_rsms: int,
                      capacity: int, sizes: int = 4, repeats: int = 2, alpha: float = 0.9,
                      thresholds: Sequence[int] = (3, 5, 7)) -> list[frozenset]:
    """Fleets that use the budget: fixed-size min-downtime knapsacks over sizes from the fewest- to the
    most-machines fleet, always including the all-RF3/5/7 capacity sizes in range; ``repeats`` fleets per
    size, each sharing at most 70% of the machines of the earlier ones."""
    min_k = math.ceil(min(thresholds) * num_rsms / capacity)
    fmin = _knapsack(feats, u, budget, alpha=alpha, min_k=min_k, objective="min_count")
    fmax = _knapsack(feats, u, budget, alpha=alpha, min_k=min_k, objective="max_count")
    if fmin is None or fmax is None:
        return []
    kmin, kmax = len(fmin), len(fmax)
    ks = {kmin, kmax} | {round(x) for x in np.linspace(kmin, kmax, sizes)}
    ks |= {math.ceil(r * num_rsms / capacity) for r in thresholds} & set(range(kmin, kmax + 1))
    out: list[frozenset] = []
    for k in sorted(ks):
        done: list[frozenset] = []
        for _ in range(repeats):
            f = _knapsack(feats, u, budget, alpha=alpha, min_k=min_k, k=k, excluded=done)
            if f is None:
                break
            done.append(f)
        out += done
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- RF mix and packing

def region_patterns(r: int, n_regions: int = 4) -> list[tuple[int, ...]]:
    """Replicas per region (descending) with no regional majority (max <= r // 2)."""
    cap = r // 2
    pats = set()
    for combo in product(range(cap + 1), repeat=n_regions):
        if sum(combo) == r:
            pats.add(tuple(sorted((c for c in combo if c), reverse=True)))
    return sorted(pats, reverse=True)


def balanced_mix(num_rsms: int, slots: int, rfs: Sequence[int]) -> dict[int, int]:
    """RSM counts per RF: sum = N, sum rf*count <= slots, minimizing pairwise differences."""
    rfs = sorted(rfs)
    best, best_key = None, None
    for counts in product(range(num_rsms + 1), repeat=len(rfs) - 1):
        last = num_rsms - sum(counts)
        if last < 0:
            continue
        mix = dict(zip(rfs, (*counts, last)))
        if sum(r * c for r, c in mix.items()) > slots:
            continue
        vals = list(mix.values())
        key = (sum(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]), -max(mix))
        if best_key is None or key < best_key:
            best, best_key = mix, key
    if best is None:
        raise ValueError("fleet cannot host N RSMs at any mix")
    return best


def split_passes(mix: dict[int, int], total: int, floor_share: float = 0.1) -> dict[int, int]:
    """Passes per RF in proportion to the mix, at least one per RF with a non-zero floor share."""
    n = sum(mix.values())
    w = {r: max(c / n, floor_share) for r, c in mix.items()}
    s = sum(w.values())
    return {r: max(1, round(total * x / s)) for r, x in w.items()}


def pack_pass(fleet: Sequence[MachineFeatures], r: int, count: int, capacity: int, u: dict[str, float],
              rng: random.Random, *, score: Callable[[tuple[str, ...]], float] | None = None,
              noise: float = 0.0, near: float = 0.0) -> list[tuple[str, ...]]:
    """Up to ``count`` placements of ``r`` replicas, consuming each machine's slots as it goes.

    Each placement takes a random no-majority region pattern; regions and AZs with the most free slots go
    first (``noise`` jitters that order); in each AZ the machine is drawn among those within ``near`` of the
    lowest downtime score ``u``. ``score`` (complete placement -> availability) then tries one swap per
    member to another free machine of the same AZ. ``noise = near = 0`` is a deterministic covering."""
    slots = {f.machine_id: capacity for f in fleet}
    by_az: dict[str, list[MachineFeatures]] = defaultdict(list)
    region_of = {}
    for f in fleet:
        by_az[f.az].append(f)
        region_of[f.az] = f.region
    pats = region_patterns(r)
    out = []
    for _ in range(count):
        free_az = {az: sum(slots[f.machine_id] for f in ms) for az, ms in by_az.items()}
        free_region: dict[str, int] = Counter()
        for az, s in free_az.items():
            if s:
                free_region[region_of[az]] += 1  # regions ranked by AZs with free slots
        order = sorted(free_region, key=lambda g: -(free_region[g] + noise * rng.random()))
        pat = pats[rng.randrange(len(pats))] if noise else pats[0]
        if len(order) < len(pat) or any(free_region[g] < n for g, n in zip(order, pat)):
            ok = [p for p in pats if len(order) >= len(p) and all(free_region[g] >= n for g, n in zip(order, p))]
            if not ok:
                break
            pat = ok[rng.randrange(len(ok))] if noise else ok[0]
        chosen = []
        for g, n in zip(order, pat):
            azs = sorted((az for az in by_az if region_of[az] == g and free_az[az]),
                         key=lambda az: -(free_az[az] * (1 + noise * rng.random())))
            for az in azs[:n]:
                ms = [f for f in by_az[az] if slots[f.machine_id]]
                lo = min(u[f.machine_id] for f in ms)
                pool = [f for f in ms if u[f.machine_id] <= lo * (1 + near) + 1e-30]
                chosen.append(pool[rng.randrange(len(pool))] if near else min(pool, key=lambda f: u[f.machine_id]))
        ids = tuple(sorted(f.machine_id for f in chosen))
        if score is not None:
            ids = _improve(ids, by_az, slots, {f.machine_id: f for f in fleet}, score)
        for mid in ids:
            slots[mid] -= 1
        out.append(ids)
    return out


def _improve(ids, by_az, slots, feat, score):
    best, best_s = ids, score(ids)
    for mid in ids:
        az = feat[mid].az
        for alt in by_az[az]:
            a = alt.machine_id
            if a in best or not slots[a]:
                continue
            trial = tuple(sorted(a if x == mid else x for x in best))
            s = score(trial)
            if s > best_s:
                best, best_s = trial, s
    return best


def swap_neighbors(ids: tuple[str, ...], fleet: Sequence[MachineFeatures]) -> list[tuple[str, ...]]:
    """Placements one same-AZ swap away (AZs and regions unchanged, so still valid). Machine capacity is
    left to the ILP, so swaps ignore slots; :func:`improve` takes the best of these."""
    member = set(ids)
    by_az: dict[str, list[str]] = defaultdict(list)
    for f in fleet:
        if f.machine_id not in member:
            by_az[f.az].append(f.machine_id)
    az_of = {f.machine_id: f.az for f in fleet}
    return [tuple(sorted(a if x == mid else x for x in ids)) for mid in ids for a in by_az[az_of[mid]]]


def improve(ids: tuple[str, ...], fleet: Sequence[MachineFeatures],
            score: Callable[[tuple[str, ...]], float], steps: int = 2) -> tuple[str, ...]:
    """Steepest-ascent same-AZ swaps on ``score``; batch the neighbours' scores first to keep this cheap."""
    best, best_s = ids, score(ids)
    for _ in range(steps):
        cand = max(swap_neighbors(best, fleet), key=score, default=None)
        if cand is None or score(cand) <= best_s:
            break
        best, best_s = cand, score(cand)
    return best


def dominate(ids: tuple[str, ...], fleet: Sequence[MachineFeatures], u: dict[str, float]) -> tuple[str, ...]:
    """Replace each member by the best fleet machine that dominates it (no dearer, no less reliable) and keeps
    the placement's AZs distinct and regions without a majority. The bill and availability cannot worsen."""
    feat = {f.machine_id: f for f in fleet}
    cur = list(ids)
    for i, mid in enumerate(cur):
        others = [feat[x] for j, x in enumerate(cur) if j != i]
        used_az = {f.az for f in others}
        regions = Counter(f.region for f in others)
        alts = [f for f in fleet if f.machine_id not in cur and f.dominates(feat[mid]) and f.az not in used_az
                and regions[f.region] + 1 <= len(cur) // 2]
        if alts:
            cur[i] = min(alts, key=lambda f: (u[f.machine_id], f.price)).machine_id
    return tuple(sorted(cur))
