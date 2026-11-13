"""Budget-targeted ILP frontier (docs/plans/ilp-fleet-sampling.md) vs the random-menu ILP and baselines.

    python scripts/pilot_fleet_frontier.py [--scales 1 10 20 30] [--prices rho0 rho1] [--workers 8]

Per (fault scale, price model), each in its own process:

1. Scaled catalogue machines; 16 k-means clusters on (log transient, log permanent rate); a lazy centroid
   MC table (medoid machines, region/AZ outages on, common random numbers).
2. Budgets = fractions of the random-menu ILP's unrestricted cost (as in pilot_ilp_rate_scale.py).
3. Per budget: knapsack fleets (fixed size, min downtime score, sizes swept fewest..most machines, 2 per
   size); per fleet a deterministic covering per RF plus randomized packing passes split by the balanced
   3/5/7 mix, scored by the table, then dominance swaps inside the fleet.
4. ILP per budget over placements from fleets of budgets <= B plus the fixed-RF baselines, on table
   scores; selected placements get exact MC on their real machines and the ILP re-solves until every
   selected placement is exact. Reported availability is exact MC only.
5. The old random-menu ILP (Markov-scored candidates) at the same budgets and the RF 3/5/7 price-blind /
   cheapest-first baselines are rescored with the same exact MC, so all points are comparable.

Exact MC uses the scaled horizon-average (constant-rate) machine configs and region/AZ outages of the
scenario; region/AZ outage rates are not scaled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary  # noqa: E402
from powder.faults import catalogue as C  # noqa: E402
from powder.faults import fleets as F  # noqa: E402
from powder.faults.centroid_table import CentroidTable, load_cache, save_cache  # noqa: E402
from powder.faults.composed_availability import CandidateTimeline  # noqa: E402
from powder.faults.gurobi_ilp import TransitionProblem, solve_transition_gurobi, solve_unique_cost_gurobi  # noqa: E402
from powder.faults.convert import scenario_network_config  # noqa: E402
from powder.faults.ilp import solve_catalogue_placement  # noqa: E402
from powder.faults.mc_scoring import node_config_batch_sampler  # noqa: E402
from powder.faults.is_scoring import is_stats  # noqa: E402
from powder.placement_optimizer import PlacementCandidate, PlacementSolverConfig, solve_candidate_ilp  # noqa: E402
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402
from scripts.experiment_sizing import baseline  # noqa: E402

# Every new_ilp / main_ilp row carries ``placements``: one sorted machine-id list per RSM, so later studies
# (scripts/figs/) can re-score the chosen deployment under other strategies or horizons.

CATALOGUE = "data/fault_library/catalogue/machines-od1000-spot1000-seed0-az.json.gz"
SCENARIO = "scenario/canonical-central-az-aware@v3"
REGIONS = ("us-east-1", "us-east-2", "us-west-1", "us-west-2")


def nines(a: float) -> float:
    return -math.log10(1 - a) if a < 1 else math.inf


def exact_runs(scale: float) -> int:
    """~3% relative s.e. per placement: more fault events per run need fewer runs."""
    return max(20_000, int(100_000 / math.sqrt(scale)))


class Exact:
    """Exact MC of real (scaled) machines, memoized by sorted machine ids. Each call is its own Rust process,
    so :meth:`batch` scores many placements at once on ``threads`` threads (the GIL is released while waiting).

    ``min_batches`` > 1: a placement counts as exactly scored only after that many independent batches of
    ``runs`` (batch b uses seed + b x runs; batch 0 is the original single score). Each time the ILP selects an
    under-sampled placement it gets one more batch, so a lucky first draw regresses toward its mean instead of
    being kept (selection on noise). ``is_beta`` > 1: importance-sampled batches (powder.faults.is_scoring).
    Defaults (1, 1.0) reproduce the original single fixed-seed score exactly."""

    def __init__(self, by_id, protocol, strategy, horizon_s, network, runs, threads, cache_path: Path | None = None,
                 min_batches: int = 1, is_beta: float = 1.0):
        self.by_id, self.protocol, self.strategy = by_id, protocol, strategy
        self.horizon_s, self.network, self.runs, self.threads = horizon_s, network, runs, threads
        self.cache_path = cache_path
        self.min_batches, self.is_beta = int(min_batches), float(is_beta)
        self.seconds = 0.0
        self.sums: dict[tuple[str, ...], float] = {}
        self.counts: dict[tuple[str, ...], int] = {}
        self.memo: dict[tuple[str, ...], float] = {}
        for k, v in (load_cache(cache_path).items() if cache_path else ()):
            ids = tuple(k.split("|"))
            if isinstance(v, dict):
                self.sums[ids], self.counts[ids] = float(v["sum"]), int(v["n"])
                self.memo[ids] = self.sums[ids] / self.counts[ids]
            else:
                self.memo[ids], self.sums[ids], self.counts[ids] = float(v), float(v), 1

    def save(self) -> None:
        if self.cache_path:
            plain = self.min_batches == 1 and self.is_beta == 1.0
            save_cache(self.cache_path, {"|".join(k): (v if plain else {"sum": self.sums[k], "n": self.counts[k]})
                                         for k, v in self.memo.items()})

    def needs(self, ids) -> bool:
        """Not yet exactly scored (fewer than ``min_batches`` batches)."""
        return self.counts.get(tuple(sorted(ids)), 0) < self.min_batches

    def batch(self, placements) -> None:
        """One more batch for every listed placement that still needs one (at least one batch for new ones).

        Each batch is split into run chunks so there are at least ``threads`` Rust jobs even when only a few
        placements need scoring (the engine seeds run i with base_seed + i, so chunking is bit-identical)."""
        t = time.time()
        todo = list(dict.fromkeys(tuple(sorted(p)) for p in placements if self.needs(p)))
        if todo:
            chunks = max(1, min(32, -(-self.threads // len(todo)), self.runs // 5_000))
            jobs = [(ids, j, chunks) for ids in todo for j in range(chunks)]
            with ThreadPoolExecutor(self.threads) as pool:
                parts = list(pool.map(lambda job: (job[0], self._chunk(*job)), jobs))
            for ids in todo:
                got = [x for i, x in parts if i == ids]
                self._record(ids, sum(n * m for n, m in got) / sum(n for n, _ in got))
        self.seconds += time.time() - t

    def _seed(self, ids, batch: int) -> int:
        return int.from_bytes(hashlib.blake2b("|".join(ids).encode(), digest_size=6).digest(), "big") + batch * self.runs

    def _chunk(self, ids, j: int, chunks: int) -> tuple[int, float]:
        """(runs, mean availability) of chunk j of the placement's next batch."""
        size = -(-self.runs // chunks)
        lo, hi = j * size, min(self.runs, (j + 1) * size)
        if hi <= lo:
            return 0, 0.0
        seed = self._seed(ids, self.counts.get(ids, 0)) + lo
        configs = [self.by_id[m].node_config for m in ids]
        if self.is_beta == 1.0:
            s = node_config_batch_sampler(configs, self.protocol, self.strategy, horizon_s=self.horizon_s,
                                          network=self.network)
            return hi - lo, s(hi - lo, seed).mean
        return hi - lo, is_stats(configs, self.protocol, self.strategy, horizon_s=self.horizon_s, runs=hi - lo,
                                 seed=seed, beta=self.is_beta, network=self.network).availability

    def _record(self, ids, mean: float) -> None:
        self.sums[ids] = self.sums.get(ids, 0.0) + mean
        self.counts[ids] = self.counts.get(ids, 0) + 1
        self.memo[ids] = self.sums[ids] / self.counts[ids]

    def _add_batch(self, ids) -> None:
        self._record(ids, self._chunk(ids, 0, 1)[1])

    def __call__(self, ids) -> float:
        ids = tuple(sorted(ids))
        if ids not in self.memo:
            self._add_batch(ids)
        return self.memo[ids]


def validate(placements, by_id, protocol, strategy, horizon_s, network, runs: int, threads: int,
             seed_offset: int = 10**9) -> dict:
    """Independent re-score of a chosen deployment: plain MC on seeds disjoint from every optimizer batch."""
    t = time.time()
    counts = Counter(tuple(sorted(p)) for p in placements)
    chunks = max(1, min(64, -(-threads // len(counts)), runs // 10_000))
    size = -(-runs // chunks)

    def one(job):
        ids, j = job
        lo, hi = j * size, min(runs, (j + 1) * size)
        seed = int.from_bytes(hashlib.blake2b("|".join(ids).encode(), digest_size=6).digest(), "big") + seed_offset + lo
        s = node_config_batch_sampler([by_id[m].node_config for m in ids], protocol, strategy, horizon_s=horizon_s,
                                      network=network)
        return ids, hi - lo, s(hi - lo, seed).mean

    with ThreadPoolExecutor(threads) as pool:
        parts = list(pool.map(one, [(ids, j) for ids in counts for j in range(chunks)]))
    got = {ids: sum(n * m for i, n, m in parts if i == ids) / sum(n for i, n, _ in parts if i == ids)
           for ids in counts}
    mean = sum(c * got[p] for p, c in counts.items()) / sum(counts.values())
    return {"mean_availability": mean, "nines": nines(mean), "runs_per_placement": runs, "seconds": time.time() - t}


def summarize(sol, price: dict, exact: Exact, num_rsms: int) -> dict:
    if not sol.selected:
        return {"feasible": False, "status": str(sol.status)}
    used, rf, total = set(), Counter(), 0.0
    for item in sol.selected:
        ids = item.candidate.machine_ids
        if sol.composed_objective is None:
            total += exact(ids) * item.count
        rf[len(ids)] += item.count
        used |= set(ids)
    mean = (sol.single_access_availability if sol.composed_objective is not None else total / num_rsms)
    row = {"feasible": True, "cost_per_hour": sol.total_cost_per_hour if sol.shard_placements else sum(price[m] for m in used),
           "mean_availability": mean, "nines": nines(mean), "rf_counts": dict(sorted(rf.items())),
           "machines_used": len(used), "spot": sum(int(m[1:]) >= 1000 for m in used),
           "placements": ([list(ids) for _, ids in sol.shard_placements] if sol.shard_placements else
                          [list(item.candidate.machine_ids) for item in sol.selected for _ in range(item.count)])}
    if sol.composed_objective is not None:
        row.update(composed_objective=sol.composed_objective,
                   full_availability=sol.full_access_availability,
                   full_nines=nines(sol.full_access_availability),
                   global_availability=sol.global_access_availability,
                   per_shard_access=sol.per_shard_access)
    if sol.shard_placements:
        row.update(shard_ids=[sid for sid, _ in sol.shard_placements],
                   machine_cost_per_hour=sol.machine_cost_per_hour,
                   replication_cost_per_hour=sol.replication_cost_per_hour,
                   migration_usd=sol.migration_usd, gb_moved=sol.gb_moved,
                   replica_moves=sol.replica_moves, replica_removals=sol.replica_removals,
                   horizon_h=sol.horizon_h)
    return row


def summarize_placements(placements, price: dict, exact: Exact) -> dict:
    """A deployment given as one placement per RSM (e.g. a fleet covering)."""
    used = {m for p in placements for m in p}
    mean = sum(exact(p) for p in placements) / len(placements)
    return {"feasible": True, "cost_per_hour": sum(price[m] for m in used), "mean_availability": mean,
            "nines": nines(mean), "rf_counts": dict(sorted(Counter(len(p) for p in placements).items())),
            "machines_used": len(used), "spot": sum(int(m[1:]) >= 1000 for m in used),
            "placements": [sorted(p) for p in placements]}


def solve_exact(pool, price, exact: Exact, estimate, num_rsms, capacity, budget, *, warm=(), rounds=10,
                time_limit=120.0, threads=1, feat_of=None, seed=0, u_of=None, prefetch=None, max_added=400,
                transition: TransitionProblem | None = None, composed: CandidateTimeline | None = None,
                composed_objective: str = "single", shard_weights=None,
                min_single_access=None, min_full_access=None):
    """Gurobi ILP on estimated scores (exact where known); selected placements get exact MC; re-solve until
    the selection is all exact. With ``feat_of``, each solve also adds *consolidation columns*: valid
    placements built only from chosen machines with free slots (fleets.spare_placements), which cost nothing
    extra and let the next solve regroup RSMs onto under-used machines and drop others. At most ``max_added``
    per round (lowest summed downtime score ``u_of``); ``prefetch`` batch-computes their estimates.
    Returns (solution, rounds, ILP seconds, MC seconds, consolidation columns added)."""
    if composed is not None:
        if transition is None:
            raise ValueError("composed placement scoring needs named shards in transition")
        # The shared-trace masks are already scored. Legacy per-RSM exact MC and
        # its lattice estimator are not valid rescores of the composed objective.
        feat_of = None
    t_ilp = t_mc = 0.0
    pool = list(dict.fromkeys(pool))
    seen = set(pool)
    rng = random.Random(seed)
    added_total = 0
    last_added: set = set()
    sol = None
    for r in range(rounds):
        cands = [PlacementCandidate(f"c{i}", ids,
                                    0.0 if composed is not None else exact.memo.get(ids, estimate(ids)),
                                    sum(price[m] for m in ids))
                 for i, ids in enumerate(pool)]
        if transition is None:
            sol = solve_unique_cost_gurobi(cands, price, capacity=capacity, num_rsms=num_rsms, budget=budget,
                                           warm_start=warm, time_limit=time_limit, threads=threads)
        else:
            if len(transition.shards) != num_rsms:
                raise ValueError("transition shard count must equal num_rsms")
            sol = solve_transition_gurobi(cands, price, capacity=capacity, budget=budget,
                                          transition=transition, warm_start=warm,
                                          time_limit=time_limit, threads=threads,
                                          composed=composed, composed_objective=composed_objective,
                                          shard_weights=shard_weights, min_single_access=min_single_access,
                                          min_full_access=min_full_access)
        t_ilp += sol.solver_runtime_seconds
        missing = ([] if composed is not None else
                   [it.candidate.machine_ids for it in sol.selected if exact.needs(it.candidate.machine_ids)])
        added = []
        chosen = {it.candidate.machine_ids for it in sol.selected}
        if last_added and not (last_added & chosen):
            feat_of = None  # the last consolidation columns were not used: stop generating more
        if feat_of is not None and sol.selected:
            used = Counter(m for it in sol.selected for m in it.candidate.machine_ids for _ in range(it.count))
            added = [p for p in F.spare_placements({m: capacity - n for m, n in used.items()}, feat_of, rng)
                     if p not in seen]
            if u_of is not None and len(added) > max_added:
                added = sorted(added, key=lambda p: sum(u_of[m] for m in p))[:max_added]
            if prefetch is not None and added:
                prefetch(added)
            pool += added
            seen.update(added)
            added_total += len(added)
            last_added = set(added)
        if not missing and not added:
            return sol, r + 1, t_ilp, t_mc, added_total
        if missing:
            t = time.time()
            exact.batch(missing)
            t_mc += time.time() - t
        warm = ([ids for _, ids in sol.shard_placements] if transition is not None else
                [it.candidate.machine_ids for it in sol.selected for _ in range(it.count)])
    return sol, rounds, t_ilp, t_mc, added_total


def cost_tiebreak(pool, final_placements, price, exact: Exact, estimate, num_rsms, capacity, budget, *,
                  tolerance_nines: float, rounds=15, time_limit=120.0, threads=1, pessimism=1.5,
                  transition: TransitionProblem | None = None, composed: CandidateTimeline | None = None,
                  composed_objective: str = "single", shard_weights=None):
    """Cheapest deployment within ``tolerance_nines`` of the best one found (``final_placements``, all exact):
    minimize the price of the active machines subject to summed unavailability <= 10**tolerance x the best
    deployment's. Unscored placements enter with ``pessimism`` x their estimated unavailability (table
    estimates of cheap placements run optimistic); picks get exact MC and the model re-solves. Only a round
    whose picks are all exact *and* within the cap counts; the cheapest such round is returned, or None.
    Returns (solution or None, rounds, ILP seconds, MC seconds)."""
    final = [tuple(sorted(p)) for p in final_placements]
    if composed is not None:
        if transition is None or len(final) != len(transition.shards):
            raise ValueError("composed cost tie-break needs one named placement per shard")
        dep = {shard.shard_id: p for shard, p in zip(transition.shards, final)}
        base = composed.compose(dep, shard_weights)
        target = base.single_access if composed_objective == "single" else base.full_access
        floor = max(0.0, 1 - (1 - target) * 10 ** tolerance_nines)
        menu = list(dict.fromkeys([*pool, *final]))
        cands = [PlacementCandidate(f"c{i}", ids, 0.0, sum(price[m] for m in ids))
                 for i, ids in enumerate(menu)]
        sol = solve_transition_gurobi(
            cands, price, capacity=capacity, budget=budget, transition=transition,
            warm_start=final, time_limit=time_limit, threads=threads, minimize_cost=True,
            composed=composed, composed_objective=composed_objective, shard_weights=shard_weights,
            **({"min_single_access": floor} if composed_objective == "single" else
               {"min_full_access": floor}))
        return (sol if sol.selected else None), 1, sol.solver_runtime_seconds, 0.0
    cap = sum(1.0 - exact(p) for p in final) * 1e6 * 10 ** tolerance_nines
    pool = list(dict.fromkeys([*pool, *final]))
    warm, t_ilp, t_mc = final, 0.0, 0.0
    best = None

    def score(ids):
        if ids in exact.memo:
            return exact.memo[ids]
        return 1.0 - pessimism * (1.0 - estimate(ids))

    def verified(sol):
        picks = [it.candidate.machine_ids for it in sol.selected for _ in range(it.count)]
        return (all(not exact.needs(p) for p in picks)
                and sum(1.0 - exact.memo[p] for p in picks) * 1e6 <= cap * (1 + 1e-9))

    r = 0
    for r in range(rounds):
        cands = [PlacementCandidate(f"c{i}", ids, score(ids), sum(price[m] for m in ids))
                 for i, ids in enumerate(pool)]
        if transition is None:
            sol = solve_unique_cost_gurobi(cands, price, capacity=capacity, num_rsms=num_rsms, budget=budget,
                                           warm_start=warm, time_limit=time_limit, threads=threads,
                                           minimize_cost=True, max_unavailability_ppm=cap)
        else:
            if len(transition.shards) != num_rsms:
                raise ValueError("transition shard count must equal num_rsms")
            sol = solve_transition_gurobi(cands, price, capacity=capacity, budget=budget, transition=transition,
                                          warm_start=warm, time_limit=time_limit, threads=threads,
                                          minimize_cost=True, max_unavailability_ppm=cap)
        t_ilp += sol.solver_runtime_seconds
        if not sol.selected:
            break
        missing = [it.candidate.machine_ids for it in sol.selected if exact.needs(it.candidate.machine_ids)]
        if missing:
            t = time.time()
            exact.batch(missing)
            t_mc += time.time() - t
        if verified(sol) and (best is None or sol.total_cost_per_hour < best.total_cost_per_hour):
            best = sol
        if not missing:
            break
        warm = ([ids for _, ids in sol.shard_placements] if transition is not None else
                [it.candidate.machine_ids for it in sol.selected for _ in range(it.count)])
    return best, r + 1, t_ilp, t_mc


def lattice_ascent(sol, pool_feats, u_of, exact: Exact, price, num_rsms, capacity, budget, *, threads, seed,
                   transition: TransitionProblem | None = None, composed: CandidateTimeline | None = None,
                   rounds: int = 10):
    """Move the chosen deployment up its dominance lattices: each selected placement's lattice slice
    (fleets.lattice_slice) is estimated from its exact base and exact top (fleets.lattice_estimate), a small
    Gurobi ILP over these columns picks one deployment under capacity and budget, and every pick gets exact MC
    until the picks are all exact."""
    if composed is not None:
        # An unscored lattice neighbor has no shared-trajectory success mask.
        # Leave the exact candidate-menu solution intact until those masks exist.
        return sol, 0, 0.0, 0.0, 0
    rng = random.Random(seed)
    feat = {f.machine_id: f for f in pool_feats}
    lattices = {}
    for it in sol.selected:
        base = it.candidate.machine_ids
        if base not in lattices:
            lattices[base] = F.lattice_slice(base, pool_feats, u_of, rng)
    exact.batch([*lattices, *(tuple(sorted(top)) for _, top in lattices.values())])
    owner = {}
    for base, (elems, top) in lattices.items():
        for pl in elems:
            owner.setdefault(pl, base)

    def estimate(pl):
        best = 0.0
        for base, (elems, top) in lattices.items():
            if pl in elems:
                known = [(sl, exact.memo[p]) for p, sl in elems.items() if p in exact.memo]
                best = max(best, F.lattice_estimate(elems[pl], base, top, exact.memo[base],
                                                    exact.memo[tuple(sorted(top))], u_of, feat, known))
        return best

    pool = list(dict.fromkeys(pl for elems, _ in lattices.values() for pl in elems))
    warm = ([ids for _, ids in sol.shard_placements] if transition is not None else
            [it.candidate.machine_ids for it in sol.selected for _ in range(it.count)])
    sol, n_rounds, t_ilp, t_mc, _ = solve_exact(pool, price, exact, estimate, num_rsms, capacity, budget, warm=warm,
                                                threads=threads, transition=transition, rounds=rounds)
    return sol, n_rounds, t_ilp, t_mc, len(pool)


def carry_forward(new_rows: dict) -> None:
    """A deployment that fits a smaller budget fits every larger one: report the best at or below each budget
    (keys are budget fractions)."""
    best = None
    for f in sorted(new_rows, key=float):
        r = new_rows[f]
        if best is not None and best["feasible"] and (not r["feasible"]
                                                     or best["mean_availability"] > r["mean_availability"]):
            new_rows[f] = {**{k: v for k, v in best.items() if k != "from"},
                           "source": f"carried from the {float(best['from']):.0%} budget",
                           "candidates": r.get("candidates"), "rounds": r.get("rounds")}
        elif r["feasible"]:
            best = {**r, "from": f}


def run_one(a: argparse.Namespace, k: float, price_model: str) -> dict:
    t0 = time.time()
    lib = FaultLibrary.load()
    cat = C.load(ROOT / CATALOGUE)
    protocol = RaftLikeProtocol(election_time_dist=Exponential(0.2))
    strategy = NodeReplacementStrategy(failure_timeout=600.0, default_node_config=None)
    network = scenario_network_config(lib, SCENARIO).as_network_config(engine=True)
    horizon_s = cat["horizon_h"] * 3600.0
    log = lambda msg: print(f"[x{k:g} {price_model} {time.time() - t0:6.0f}s] {msg}", flush=True)  # noqa: E731

    # old random-menu ILP: candidates, unrestricted cost (sets the budget grid)
    old = solve_catalogue_placement(lib, cat, protocol, strategy, num_rsms=a.num_rsms, replica_counts=(3, 5),
                                    capacity=a.capacity, price=price_model, random_samples_per_size=400,
                                    scenario_id=SCENARIO, scoring="markov", rescore_exact=False, rate_scale=k)
    machines = old["machines"]
    by_id = {m.machine_id: m for m in machines}
    price = {m.machine_id: m.node_config.cost_per_hour for m in machines}
    tag = f"x{k:g}-{price_model}"
    mode = "" if (a.min_batches == 1 and a.is_beta == 1.0) else f"-b{a.min_batches}-is{a.is_beta:g}"
    exact = Exact(by_id, protocol, strategy, horizon_s, network, exact_runs(k), a.mc_threads,
                  cache_path=Path(a.cache_dir) / f"exact-x{k:g}{mode}.json",
                  min_batches=a.min_batches, is_beta=a.is_beta)
    free_cost = old["solution"].total_cost_per_hour
    fracs = list(a.budget_fractions)
    budgets = [f * free_cost for f in fracs]

    # clusters + centroid table + downtime score u
    feats = F.features(machines)
    labels, cent = F.kmeans_log(feats, k=a.clusters, seed=0)
    med = F.medoids(feats, labels, cent)
    table = CentroidTable([machines[i].node_config for i in med], {f.machine_id: int(l) for f, l in zip(feats, labels)},
                          REGIONS, {f.machine_id: f.region for f in feats}, protocol, strategy,
                          horizon_s=horizon_s, network=network, runs=max(2_000, exact_runs(k) // 5),
                          cache_path=Path(a.cache_dir) / f"table-x{k:g}.json")
    with ThreadPoolExecutor(a.mc_threads) as tp:
        u_c = [1 - x for x in tp.map(lambda j: table.availability([j, j, j], (1, 1, 1)), range(len(cent)))]
    lam = np.array([f.lam_t + f.lam_p for f in feats])
    u = np.array([u_c[l] * lam[i] / lam[med[l]] for i, l in enumerate(labels)])
    u_of = {f.machine_id: float(x) for f, x in zip(feats, u)}
    log(f"clusters + table seeds done; budgets {[round(b, 2) for b in budgets]}")

    # baselines (exact MC) -- also offered to the new ILP as columns
    base_pl, baselines, base_raw = [], {}, {}
    dummy = lambda ms: type("S", (), {"availability": 0.0})()  # noqa: E731
    for rf in (3, 5, 7):
        for policy, cheap in (("blind", False), ("cheap", True)):
            b = baseline(machines, a.num_rsms, rf, a.capacity, cheap=cheap, seed=0, scorer=dummy)
            base_raw[f"rf{rf}-{policy}"] = (b["cost_per_hour"], [tuple(sorted(p)) for p in b["placements"]])
            base_pl += base_raw[f"rf{rf}-{policy}"][1]
    exact.batch(base_pl)
    for name, (cost, pls) in base_raw.items():
        mean = sum(exact(p) for p in pls) / a.num_rsms
        baselines[name] = {"cost_per_hour": cost, "mean_availability": mean, "nines": nines(mean),
                           "placements": [list(p) for p in pls]}
    exact.save()
    matched = {}
    if a.cheapest_budgets:  # extra budgets at the cheapest-first baselines' costs
        for rf in (3, 5, 7):
            f = baselines[f"rf{rf}-cheap"]["cost_per_hour"] / free_cost
            matched[f"{f:g}"] = f"rf{rf}-cheap"
            fracs.append(f)
        budgets = [f * free_cost for f in fracs]
    log("baselines scored")

    # old ILP at each budget, exact-rescored
    old_sols = {f"{f:g}": solve_candidate_ilp(old["candidates"], machines, PlacementSolverConfig(
        num_rsms=a.num_rsms, objective="sum_availability", budget_per_hour=None if f >= 1 else b,
        unique_machine_cost=True)) for f, b in zip(fracs, budgets)}
    exact.batch(it.candidate.machine_ids for sol in old_sols.values() for it in sol.selected)
    old_rows = {f: summarize(sol, price, exact, a.num_rsms) for f, sol in old_sols.items()}
    exact.save()
    log("old ILP rescored")

    # new candidates: fleets per budget, packing passes, dominance
    feat_of = {f.machine_id: f for f in feats}

    def fleet_raw(job):
        """Covering + randomized passes on the per-machine score (own RNG, so results are reproducible)."""
        seed, fl = job
        rng = random.Random(seed)
        ff = [feat_of[m] for m in sorted(fl)]
        cover, rand, full = set(), set(), []
        rfs = [r for r in (3, 5, 7) if F.pack_pass(ff, r, 1, a.capacity, u_of, rng)]
        mix = F.balanced_mix(a.num_rsms, a.capacity * len(ff), rfs)
        for r, n_pass in F.split_passes(mix, a.passes).items():
            c = F.pack_pass(ff, r, a.num_rsms, a.capacity, u_of, rng)
            cover.update(c)
            if len(c) == a.num_rsms:
                full.append(c)          # a complete deployment inside the fleet: the ILP fallback
            for _ in range(n_pass):
                rand.update(F.pack_pass(ff, r, a.num_rsms, a.capacity, u_of, rng, noise=0.5, near=0.1))
        return ff, cover, rand, full

    def table_batch(placements) -> None:
        """Every centroid key the placements (and their one-swap neighbours) need, computed in parallel."""
        keys = {}
        for ids in placements:
            cids, pat = [table.labels[m] for m in ids], table.pattern_of(ids)
            keys.setdefault(CentroidTable.key(cids, pat), (cids, pat))
        todo = [v for k, v in keys.items() if k not in table.memo]
        with ThreadPoolExecutor(a.mc_threads) as tp:
            list(tp.map(lambda kv: table.availability(*kv), todo))

    new_rows, fleet_counts = {}, []
    for bi, (f, b) in enumerate(zip(fracs, budgets)):
        t_budget = time.time()
        fleets = F.fleets_for_budget(feats, u, b, num_rsms=a.num_rsms, capacity=a.capacity,
                                     sizes=a.fleet_sizes, repeats=a.fleet_repeats, alpha=a.alpha)
        fleet_counts.append(len(fleets))
        raw = [fleet_raw((1000 * bi + i, fl)) for i, fl in enumerate(fleets)]
        # improvement swaps step by step over all placements, each step's table entries in one parallel batch
        items = [(p, ff) for ff, _, rand, _ in raw for p in sorted(rand)]
        placements: set = set(F.improve_many(items, table.score, prefetch=table_batch))
        for ff, cover, rand, _ in raw:
            placements |= cover
        log(f"budget ${b:.2f}: {len(fleets)} fleets, {len(placements)} placements, table {len(table.memo)}")
        # this budget's fleets + the baselines; Gurobi starts from the best full covering
        pool = list(dict.fromkeys([*placements, *base_pl]))
        table_batch(pool)  # every candidate's estimate up front, in parallel, before the ILP reads them
        table.save()
        row_f, rounds, times = {"feasible": False}, 0, {}
        covers = [c for *_, full in raw for c in full]
        if fleets:
            best_cover = max(covers, key=lambda c: sum(map(table.score, c))) if covers else []
            if a.compare_highs:
                cands = [PlacementCandidate(f"c{i}", ids, exact.memo.get(ids, table.score(ids)),
                                            sum(price[m] for m in ids)) for i, ids in enumerate(pool)]
                t = time.time()
                hs = solve_candidate_ilp(cands, machines, PlacementSolverConfig(
                    num_rsms=a.num_rsms, objective="sum_availability", budget_per_hour=b, unique_machine_cost=True,
                    time_limit_seconds=300))
                gs = solve_unique_cost_gurobi(cands, price, capacity=a.capacity, num_rsms=a.num_rsms, budget=b,
                                              warm_start=best_cover, threads=a.threads)
                times["highs_first_solve_s"] = time.time() - t - gs.solver_runtime_seconds
                times["gurobi_first_solve_s"] = gs.solver_runtime_seconds
                log(f"first solve on {len(pool)} candidates: HiGHS {times['highs_first_solve_s']:.1f}s "
                    f"({'incumbent' if hs.selected else 'no incumbent'}), Gurobi {gs.solver_runtime_seconds:.1f}s "
                    f"({gs.status})")
            sol, rounds, t_ilp, t_mc, consolidated = solve_exact(
                pool, price, exact, table.score, a.num_rsms, a.capacity, b, warm=best_cover, threads=a.threads,
                feat_of=feat_of if a.consolidate else None, seed=bi, u_of=u_of, prefetch=table_batch,
                rounds=a.rounds)
            main = summarize(sol, price, exact, a.num_rsms)
            asc, a_rounds, a_ilp, a_mc, a_pool = lattice_ascent(
                sol, [feat_of[m] for m in {m for ids in pool for m in ids}], u_of, exact, price, a.num_rsms,
                a.capacity, b, threads=a.threads, seed=bi, rounds=a.rounds)
            row_f = summarize(asc, price, exact, a.num_rsms)
            if not row_f["feasible"] or (main["feasible"] and main["mean_availability"] > row_f["mean_availability"]):
                row_f = {**main, "source": "main ILP (lattice ascent found nothing better)"}
            row_f["main_ilp"] = main
            if a.tiebreak_nines > 0 and row_f["feasible"]:
                tb, tb_rounds, tb_ilp, tb_mc = cost_tiebreak(
                    pool, row_f["placements"], price, exact, table.score, a.num_rsms, a.capacity, b,
                    tolerance_nines=a.tiebreak_nines, threads=a.threads)
                tbs = summarize(tb, price, exact, a.num_rsms) if tb is not None else {"feasible": False}
                if (tbs["feasible"] and tbs["cost_per_hour"] < row_f["cost_per_hour"] - 1e-9
                        and tbs["nines"] >= row_f["nines"] - a.tiebreak_nines - 1e-9):
                    best_avail = {k: v for k, v in row_f.items() if k != "main_ilp"}
                    row_f = {**tbs, "main_ilp": main, "max_availability": best_avail,
                             "source": f"cost tie-break within {a.tiebreak_nines:g} nines of the best"}
                times.update({"tiebreak_rounds": tb_rounds, "tiebreak_ilp_s": tb_ilp, "tiebreak_mc_s": tb_mc})
                log(f"cost tie-break: ${row_f.get('max_availability', row_f)['cost_per_hour']:.2f} -> "
                    f"${row_f['cost_per_hour']:.2f}/h at {row_f['nines']:.3f} nines ({tb_rounds} rounds)")
            times.update({"main_rounds": rounds, "main_ilp_s": t_ilp, "main_mc_s": t_mc,
                          "consolidation_columns": consolidated, "ascent_rounds": a_rounds,
                          "ascent_ilp_s": a_ilp, "ascent_mc_s": a_mc, "ascent_columns": a_pool})
            log(f"main ILP {main['nines']:.3f} ({rounds} rounds, ILP {t_ilp:.0f}s, MC {t_mc:.0f}s, "
                f"+{consolidated} consolidation columns) -> lattice "
                f"{row_f.get('max_availability', row_f)['nines']:.3f} ({a_rounds} rounds on {a_pool} columns, ILP {a_ilp:.0f}s, MC {a_mc:.0f}s)")
        times["budget_wall_s"] = time.time() - t_budget          # optimization only (excludes validation)
        if a.validate_runs and row_f.get("feasible"):
            row_f["validated"] = validate(row_f["placements"], by_id, protocol, strategy, horizon_s, network,
                                          a.validate_runs, a.mc_threads)
            times["validate_s"] = row_f["validated"]["seconds"]
            log(f"validated {f:.0%}: believed {row_f['nines']:.3f} -> independent {row_f['validated']['nines']:.3f} nines")
        exact.save()
        new_rows[f"{f:g}"] = {**row_f, "candidates": len(pool), "rounds": rounds, "times": times}
        log(f"new ILP {f:.0%}: " + (f"{row_f['nines']:.2f} nines ${row_f['cost_per_hour']:.2f}/h, {len(pool)} cands, "
                                   f"{rounds} rounds{' (' + row_f['source'] + ')' if 'source' in row_f else ''}"
                                   if row_f["feasible"] else "infeasible"))
    row = {"rate_scale": k, "price": price_model, "budgets": dict(zip([f"{f:g}" for f in fracs], budgets)),
           "matched_budgets": matched,
           "free_cost_old": free_cost, "exact_runs": exact.runs, "table_runs": table.runs,
           "table_entries": len(table.memo), "exact_scored": len(exact.memo), "fleets_per_budget": fleet_counts,
           "new_ilp": new_rows, "old_ilp": old_rows, "baselines": baselines, "seconds": time.time() - t0}
    carry_forward(new_rows)
    rows_dir = ROOT / a.output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    (rows_dir / f"{tag}.json").write_text(json.dumps(row, indent=1, default=str) + "\n")
    log("done")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 10, 20, 30])
    ap.add_argument("--prices", nargs="+", default=["rho0", "rho1"])
    ap.add_argument("--num-rsms", type=int, default=50)
    ap.add_argument("--capacity", type=int, default=5)
    ap.add_argument("--budget-fractions", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    ap.add_argument("--cheapest-budgets", action="store_true",
                    help="add budgets equal to the RF 3/5/7 cheapest-first baselines' costs")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--fleet-sizes", type=int, default=4)
    ap.add_argument("--fleet-repeats", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.9, help="fleets must spend at least alpha x budget")
    ap.add_argument("--passes", type=int, default=4, help="randomized packing passes per fleet (split by RF mix)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="configurations in parallel")
    ap.add_argument("--threads", type=int, default=None,
                    help="MC threads per configuration (default: cores / configurations, at least 2)")
    ap.add_argument("--output", default="outputs/pilot-ilp-rate-scale")
    ap.add_argument("--resume", action="store_true", help="reuse finished configurations' rows/*.json")
    ap.add_argument("--cache-dir", default=None, help="MC caches (default: <output>/cache); shared across prices")
    ap.add_argument("--compare-highs", action="store_true", help="also time one HiGHS solve per budget")
    ap.add_argument("--min-batches", type=int, default=1,
                    help="independent exact-MC batches a placement needs before it counts as scored; each ILP "
                         "selection of an under-sampled placement adds one (1 = original single fixed-seed score)")
    ap.add_argument("--is-beta", type=float, default=1.0,
                    help="importance-sampling tilt on permanent failures for exact MC (1 = plain MC)")
    ap.add_argument("--validate-runs", type=int, default=0,
                    help="re-score each budget's final deployment on independent seeds with this many runs")
    ap.add_argument("--rounds", type=int, default=10, help="ILP <-> exact-MC rounds per solve")
    ap.add_argument("--mc-threads", type=int, default=os.cpu_count() or 1,
                    help="Rust MC jobs in flight per configuration (default: all cores; batches are chunked so "
                         "small batches still fill them)")
    ap.add_argument("--tiebreak-nines", type=float, default=0.01,
                    help="report the cheapest deployment within this many nines of the best (0 = off)")
    ap.add_argument("--no-consolidate", dest="consolidate", action="store_false",
                    help="skip consolidation columns from machines with free slots")
    a = ap.parse_args()
    configs = [(k, p) for k in a.scales for p in a.prices]
    a.cache_dir = str(ROOT / (a.cache_dir or f"{a.output}/cache"))
    if a.threads is None:
        a.threads = max(2, (os.cpu_count() or 1) // len(configs))
    done = {c: ROOT / a.output / "rows" / f"x{c[0]:g}-{c[1]}.json" for c in configs}
    todo = [c for c in configs if not (a.resume and done[c].exists())]
    if todo:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(todo))) as pool:
            list(pool.map(run_one, [a] * len(todo), *zip(*todo)))
    rows = [json.loads(done[c].read_text()) for c in configs]
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(a).items()
            if k not in ("output", "workers", "threads", "resume", "cache_dir", "compare_highs")}
    (out / "results-fleet.json").write_text(json.dumps(
        {"catalogue": CATALOGUE, "scenario": SCENARIO, "params": meta, "protocol": "raft, election Exp(0.2/s)",
         "strategy": "node replacement after 600 s", "scoring": "exact Rust MC, region/AZ outages on",
         "rows": rows}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
