"""Pilot: catalogue ILP expected availability, unrestricted vs 50% budget, under scaled fault rates.

    python scripts/pilot_ilp_rate_scale.py [--scales 1 10 20 30] [--workers 8] [--output outputs/pilot-ilp-rate-scale]

For each fault-rate scale (every node fault rate x k: transient incl. planned, and permanent) and each
price model: generate Markov-scored candidates once, solve the max-availability ILP with no budget,
then again at budgets of 20/30/40/50/75% of the unrestricted solution's cost (``--budget-fractions``).
A budget the candidate menu cannot meet is recorded as infeasible. Baselines: the exec-02 fixed-RF
placements (``experiment_sizing.baseline``: one replica per region, AZ anti-affinity, fault-rate blind),
price-blind (random) and cheapest-first, at RF 3/5/7 on the same machines. Availability is the ILP's
*expected* per-RSM mean under the Markov model (exact rescoring for <= 5 nodes, binned otherwise):
no MC, no region/AZ outages (Markov does not model them). A screening pilot, not a certified result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary  # noqa: E402
from powder.faults import catalogue as C  # noqa: E402
from powder.faults.ilp import binned_markov_scorer, solve_catalogue_placement  # noqa: E402
from powder.placement_optimizer import PlacementSolverConfig, markov_candidate_scorer, solve_candidate_ilp  # noqa: E402
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402
from scripts.experiment_sizing import baseline  # noqa: E402

CATALOGUE = "data/fault_library/catalogue/machines-od1000-spot1000-seed0-az.json.gz"
SCENARIO = "scenario/canonical-central-az-aware@v3"


def nines(a: float) -> float:
    return -math.log10(1 - a) if a < 1 else math.inf


def summarize(sol, machines, protocol, strategy, num_rsms: int, exact_cache: dict) -> dict:
    if not sol.selected:
        return {"status": str(sol.status), "feasible": False, "budget_per_hour": sol.budget_per_hour}
    by_id = {m.machine_id: m for m in machines}
    scorer = markov_candidate_scorer(protocol, strategy)
    total, rf, tiers, used = 0.0, Counter(), Counter(), set()
    for item in sol.selected:
        ids = item.candidate.machine_ids
        cid = item.candidate.candidate_id
        if cid not in exact_cache:  # 7 distinct nodes take ~10 s each; keep the binned score there
            exact_cache[cid] = scorer([by_id[i] for i in ids]).availability if len(ids) <= 5 else item.candidate.availability
        total += exact_cache[cid] * item.count
        rf[len(ids)] += item.count
        used |= set(ids)
    for mid in used:
        tiers["spot" if int(mid[1:]) >= 1000 else "ondemand"] += 1
    mean = total / num_rsms
    return {"status": str(sol.status), "feasible": True, "cost_per_hour": sol.total_cost_per_hour, "budget_per_hour": sol.budget_per_hour,
            "mean_availability": mean, "nines": nines(mean), "rf_counts": dict(sorted(rf.items())),
            "machines_used": len(used), "tiers": dict(tiers)}


def baselines(machines, protocol, strategy, num_rsms: int, capacity: int, rfs) -> dict:
    exact, binned = markov_candidate_scorer(protocol, strategy), binned_markov_scorer(protocol, strategy)
    memo: dict = {}

    def scorer(ms):  # exact Markov for <= 5 nodes, binned for 7 (as for the ILP's selections)
        key = tuple(m.machine_id for m in ms)
        if key not in memo:
            memo[key] = exact(ms) if len(ms) <= 5 else binned(ms)
        return memo[key]

    out = {}
    for rf in rfs:
        for policy, cheap in (("blind", False), ("cheap", True)):
            b = baseline(machines, num_rsms, rf, capacity, cheap=cheap, seed=0, scorer=scorer)
            used = b["active_ids"]
            out[f"rf{rf}-{policy}"] = {"cost_per_hour": b["cost_per_hour"], "mean_availability": b["mean_availability"],
                                       "nines": nines(b["mean_availability"]), "machines_used": len(used),
                                       "spot": sum(int(m[1:]) >= 1000 for m in used)}
    return out


def run_one(a: argparse.Namespace, k: float, price: str) -> dict:
    """One (fault scale, price model) configuration; independent of the others, so runs in its own process."""
    t = time.time()
    lib = FaultLibrary.load()
    cat = C.load(ROOT / CATALOGUE)
    protocol = RaftLikeProtocol(election_time_dist=Exponential(0.2))
    strategy = NodeReplacementStrategy(failure_timeout=600.0, default_node_config=None)
    r = solve_catalogue_placement(lib, cat, protocol, strategy, num_rsms=a.num_rsms,
                                  replica_counts=tuple(a.replicas), capacity=a.capacity, price=price,
                                  random_samples_per_size=a.samples, scenario_id=SCENARIO,
                                  scoring="markov", rescore_exact=False, rate_scale=k)
    exact: dict = {}
    free = summarize(r["solution"], r["machines"], protocol, strategy, a.num_rsms, exact)
    capped = {}
    for f in a.budget_fractions:
        sol = solve_candidate_ilp(r["candidates"], r["machines"],
                                  PlacementSolverConfig(num_rsms=a.num_rsms, objective="sum_availability",
                                                        budget_per_hour=f * free["cost_per_hour"],
                                                        unique_machine_cost=True))
        capped[f"budget_{f:g}"] = summarize(sol, r["machines"], protocol, strategy, a.num_rsms, exact)
    row = {"rate_scale": k, "price": price, "candidates": len(r["candidates"]),
           "binned_classes": r["binned_classes"], "budget_fractions": a.budget_fractions,
           "unrestricted": free, **capped,
           "baselines": baselines(r["machines"], protocol, strategy, a.num_rsms, a.capacity, a.baseline_rfs)}
    row["seconds"] = time.time() - t
    print(f"x{k:g} {price}: free {free['nines']:.2f} nines ${free['cost_per_hour']:.2f}/h | " +
          " ".join(f"{key[7:]}: " + (f"{c['nines']:.2f}/${c['cost_per_hour']:.2f}" if c["feasible"] else "infeasible")
                   for key, c in capped.items()) + " | " +
          " ".join(f"{n} {b['nines']:.2f}/${b['cost_per_hour']:.2f}" for n, b in row["baselines"].items()) +
          f" ({row['seconds']:.0f}s)", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 10, 20, 30])
    ap.add_argument("--prices", nargs="+", default=["rho0", "rho1"])
    ap.add_argument("--num-rsms", type=int, default=50)
    ap.add_argument("--replicas", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--capacity", type=int, default=5)
    ap.add_argument("--samples", type=int, default=400, help="random candidates per replica count")
    ap.add_argument("--budget-fractions", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.75])
    ap.add_argument("--baseline-rfs", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="configurations run in parallel")
    ap.add_argument("--output", default="outputs/pilot-ilp-rate-scale")
    a = ap.parse_args()
    configs = [(k, price) for k in a.scales for price in a.prices]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(configs))) as pool:
        rows = list(pool.map(run_one, [a] * len(configs), *zip(*configs)))
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(a).items() if k not in ("output", "workers")}
    (out / "results.json").write_text(json.dumps({"catalogue": CATALOGUE, "scenario": SCENARIO,
                                                  "catalogue_digest": C.digest(C.load(ROOT / CATALOGUE)),
                                                  "params": meta, "protocol": "raft, election Exp(0.2/s)",
                                                  "strategy": "node replacement after 600 s", "rows": rows},
                                                 indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
