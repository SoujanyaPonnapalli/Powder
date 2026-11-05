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
from powder.faults.centroid_table import CentroidTable  # noqa: E402
from powder.faults.convert import scenario_network_config  # noqa: E402
from powder.faults.ilp import solve_catalogue_placement  # noqa: E402
from powder.faults.mc_scoring import node_config_batch_sampler  # noqa: E402
from powder.placement_optimizer import PlacementCandidate, PlacementSolverConfig, solve_candidate_ilp  # noqa: E402
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402
from scripts.experiment_sizing import baseline  # noqa: E402

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
    so :meth:`batch` scores many placements at once on ``threads`` threads (the GIL is released while waiting)."""

    def __init__(self, by_id, protocol, strategy, horizon_s, network, runs, threads, cache_path: Path | None = None):
        self.by_id, self.protocol, self.strategy = by_id, protocol, strategy
        self.horizon_s, self.network, self.runs, self.threads = horizon_s, network, runs, threads
        self.cache_path = cache_path
        self.memo: dict[tuple[str, ...], float] = {}
        if cache_path and cache_path.exists():
            self.memo = {tuple(k.split("|")): v for k, v in json.loads(cache_path.read_text()).items()}

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({"|".join(k): v for k, v in self.memo.items()}))

    def batch(self, placements) -> None:
        todo = list(dict.fromkeys(tuple(sorted(p)) for p in placements if tuple(sorted(p)) not in self.memo))
        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(self, todo))

    def __call__(self, ids) -> float:
        ids = tuple(sorted(ids))
        if ids not in self.memo:
            seed = int.from_bytes(hashlib.blake2b("|".join(ids).encode(), digest_size=6).digest(), "big")
            s = node_config_batch_sampler([self.by_id[m].node_config for m in ids], self.protocol, self.strategy,
                                          horizon_s=self.horizon_s, network=self.network)
            self.memo[ids] = s(self.runs, seed).mean
        return self.memo[ids]


def summarize(sol, price: dict, exact: Exact, num_rsms: int) -> dict:
    if not sol.selected:
        return {"feasible": False, "status": str(sol.status)}
    used, rf, total = set(), Counter(), 0.0
    for item in sol.selected:
        ids = item.candidate.machine_ids
        total += exact(ids) * item.count
        rf[len(ids)] += item.count
        used |= set(ids)
    mean = total / num_rsms
    return {"feasible": True, "cost_per_hour": sum(price[m] for m in used), "mean_availability": mean,
            "nines": nines(mean), "rf_counts": dict(sorted(rf.items())), "machines_used": len(used),
            "spot": sum(int(m[1:]) >= 1000 for m in used)}


def summarize_placements(placements, price: dict, exact: Exact) -> dict:
    """A deployment given as one placement per RSM (e.g. a fleet covering)."""
    used = {m for p in placements for m in p}
    mean = sum(exact(p) for p in placements) / len(placements)
    return {"feasible": True, "cost_per_hour": sum(price[m] for m in used), "mean_availability": mean,
            "nines": nines(mean), "rf_counts": dict(sorted(Counter(len(p) for p in placements).items())),
            "machines_used": len(used), "spot": sum(int(m[1:]) >= 1000 for m in used)}


def solve_exact(pool, machines, budget, exact: Exact, table_score, num_rsms, price, *, rounds=10,
                time_limit=300.0):
    """ILP on table scores; selected placements get exact MC; re-solve until the selection is all exact.
    The model holds only the machines the pool uses, which keeps HiGHS fast."""
    used = {m for ids in pool for m in ids}
    ms = [m for m in machines if m.machine_id in used]
    sol = None
    for r in range(rounds):
        cands = [PlacementCandidate(f"c{i}", ids, exact.memo.get(ids, table_score(ids)), sum(price[m] for m in ids))
                 for i, ids in enumerate(pool)]
        sol = solve_candidate_ilp(cands, ms, PlacementSolverConfig(
            num_rsms=num_rsms, objective="sum_availability", budget_per_hour=budget, unique_machine_cost=True,
            time_limit_seconds=time_limit))
        missing = [it.candidate.machine_ids for it in sol.selected if it.candidate.machine_ids not in exact.memo]
        if not missing:
            return sol, r + 1
        exact.batch(missing)
    return sol, rounds


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
    exact = Exact(by_id, protocol, strategy, horizon_s, network, exact_runs(k), a.threads,
                  cache_path=ROOT / a.output / "cache" / f"exact-{tag}.json")
    free_cost = old["solution"].total_cost_per_hour
    budgets = [f * free_cost for f in a.budget_fractions]

    # clusters + centroid table + downtime score u
    feats = F.features(machines)
    labels, cent = F.kmeans_log(feats, k=a.clusters, seed=0)
    med = F.medoids(feats, labels, cent)
    table = CentroidTable([machines[i].node_config for i in med], {f.machine_id: int(l) for f, l in zip(feats, labels)},
                          REGIONS, {f.machine_id: f.region for f in feats}, protocol, strategy,
                          horizon_s=horizon_s, network=network, runs=max(2_000, exact_runs(k) // 5),
                          cache_path=ROOT / a.output / "cache" / f"table-{tag}.json")
    with ThreadPoolExecutor(a.threads) as tp:
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
        baselines[name] = {"cost_per_hour": cost, "mean_availability": mean, "nines": nines(mean)}
    exact.save()
    log("baselines scored")

    # old ILP at each budget, exact-rescored
    old_sols = {f"{f:g}": solve_candidate_ilp(old["candidates"], machines, PlacementSolverConfig(
        num_rsms=a.num_rsms, objective="sum_availability", budget_per_hour=None if f >= 1 else b,
        unique_machine_cost=True)) for f, b in zip(a.budget_fractions, budgets)}
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
        with ThreadPoolExecutor(a.threads) as tp:
            list(tp.map(lambda kv: table.availability(*kv), todo))

    new_rows, fleet_counts = {}, []
    for bi, (f, b) in enumerate(zip(a.budget_fractions, budgets)):
        fleets = F.fleets_for_budget(feats, u, b, num_rsms=a.num_rsms, capacity=a.capacity,
                                     sizes=a.fleet_sizes, repeats=a.fleet_repeats)
        fleet_counts.append(len(fleets))
        raw = [fleet_raw((1000 * bi + i, fl)) for i, fl in enumerate(fleets)]
        table_batch({n for ff, _, rand, _ in raw for p in rand for n in (p, *F.swap_neighbors(p, ff))})
        table.save()
        placements: set = set()
        for ff, cover, rand, _ in raw:
            placements |= cover
            placements |= {F.dominate(F.improve(p, ff, table.score), ff, u_of) for p in rand}
        log(f"budget ${b:.2f}: {len(fleets)} fleets, {len(placements)} placements, table {len(table.memo)}")
        # this budget's fleets + the baselines; the best full covering is a guaranteed feasible fallback
        pool = list(dict.fromkeys([*placements, *base_pl]))
        row_f = {"feasible": False}
        rounds = 0
        if fleets:
            # 100% = the random-menu ILP's unrestricted cost, as a cap: unbounded, the ILP buys dear machines
            # for no measurable gain (no cost tie-break)
            sol, rounds = solve_exact(pool, machines, b, exact, table.score, a.num_rsms, price)
            row_f = summarize(sol, price, exact, a.num_rsms)
            covers = [c for *_, full in raw for c in full]
            if covers:
                best = max(covers, key=lambda c: sum(map(table.score, c)))
                exact.batch(best)
                fb = summarize_placements(best, price, exact)
                if not row_f["feasible"] or fb["mean_availability"] > row_f["mean_availability"]:
                    row_f = {**fb, "source": "fleet covering (ILP had no better incumbent)"}
        exact.save()
        new_rows[f"{f:g}"] = {**row_f, "candidates": len(pool), "rounds": rounds}
        log(f"new ILP {f:.0%}: " + (f"{row_f['nines']:.2f} nines ${row_f['cost_per_hour']:.2f}/h, {len(pool)} cands, "
                                   f"{rounds} rounds{' (covering)' if 'source' in row_f else ''}"
                                   if row_f["feasible"] else "infeasible"))
    row = {"rate_scale": k, "price": price_model, "budgets": dict(zip([f"{f:g}" for f in a.budget_fractions], budgets)),
           "free_cost_old": free_cost, "exact_runs": exact.runs, "table_runs": table.runs,
           "table_entries": len(table.memo), "exact_scored": len(exact.memo), "fleets_per_budget": fleet_counts,
           "new_ilp": new_rows, "old_ilp": old_rows, "baselines": baselines, "seconds": time.time() - t0}
    # a deployment that fits a smaller budget fits every larger one: report the best at or below each budget
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
    ap.add_argument("--budget-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--fleet-sizes", type=int, default=4)
    ap.add_argument("--fleet-repeats", type=int, default=2)
    ap.add_argument("--passes", type=int, default=4, help="randomized packing passes per fleet (split by RF mix)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="configurations in parallel")
    ap.add_argument("--threads", type=int, default=None,
                    help="MC threads per configuration (default: cores / configurations, at least 2)")
    ap.add_argument("--output", default="outputs/pilot-ilp-rate-scale")
    ap.add_argument("--resume", action="store_true", help="reuse finished configurations' rows/*.json")
    a = ap.parse_args()
    configs = [(k, p) for k in a.scales for p in a.prices]
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
    meta = {k: v for k, v in vars(a).items() if k not in ("output", "workers", "threads", "resume")}
    (out / "results-fleet.json").write_text(json.dumps(
        {"catalogue": CATALOGUE, "scenario": SCENARIO, "params": meta, "protocol": "raft, election Exp(0.2/s)",
         "strategy": "node replacement after 600 s", "scoring": "exact Rust MC, region/AZ outages on",
         "rows": rows}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
