"""Pilot: catalogue ILP expected availability, unrestricted vs 50% budget, under scaled fault rates.

    python scripts/pilot_ilp_rate_scale.py [--scales 1 10 20 30] [--output outputs/pilot-ilp-rate-scale]

For each fault-rate scale (every node fault rate x k: transient incl. planned, and permanent) and each
price model: generate Markov-scored candidates once, solve the max-availability ILP with no budget,
then again with a budget of 50% of the unrestricted solution's cost. Availability is the ILP's
*expected* per-RSM mean under the Markov model (exact rescoring for <= 5 nodes, binned otherwise):
no MC, no region/AZ outages (Markov does not model them). A screening pilot, not a certified result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary  # noqa: E402
from powder.faults import catalogue as C  # noqa: E402
from powder.faults.ilp import solve_catalogue_placement  # noqa: E402
from powder.placement_optimizer import PlacementSolverConfig, markov_candidate_scorer, solve_candidate_ilp  # noqa: E402
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402

CATALOGUE = "data/fault_library/catalogue/machines-od1000-spot1000-seed0-az.json.gz"
SCENARIO = "scenario/canonical-central-az-aware@v3"


def nines(a: float) -> float:
    return -math.log10(1 - a) if a < 1 else math.inf


def summarize(sol, machines, protocol, strategy, num_rsms: int, exact_cache: dict) -> dict:
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
    return {"status": str(sol.status), "cost_per_hour": sol.total_cost_per_hour, "budget_per_hour": sol.budget_per_hour,
            "mean_availability": mean, "nines": nines(mean), "rf_counts": dict(sorted(rf.items())),
            "machines_used": len(used), "tiers": dict(tiers)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 10, 20, 30])
    ap.add_argument("--prices", nargs="+", default=["rho0", "rho1"])
    ap.add_argument("--num-rsms", type=int, default=50)
    ap.add_argument("--replicas", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--capacity", type=int, default=5)
    ap.add_argument("--samples", type=int, default=400, help="random candidates per replica count")
    ap.add_argument("--budget-fraction", type=float, default=0.5)
    ap.add_argument("--output", default="outputs/pilot-ilp-rate-scale")
    a = ap.parse_args()
    lib = FaultLibrary.load()
    cat = C.load(ROOT / CATALOGUE)
    protocol = RaftLikeProtocol(election_time_dist=Exponential(0.2))
    strategy = NodeReplacementStrategy(failure_timeout=600.0, default_node_config=None)
    rows = []
    for k in a.scales:
        for price in a.prices:
            t = time.time()
            r = solve_catalogue_placement(lib, cat, protocol, strategy, num_rsms=a.num_rsms,
                                          replica_counts=tuple(a.replicas), capacity=a.capacity, price=price,
                                          random_samples_per_size=a.samples, scenario_id=SCENARIO,
                                          scoring="markov", rescore_exact=False, rate_scale=k)
            exact: dict = {}
            free = summarize(r["solution"], r["machines"], protocol, strategy, a.num_rsms, exact)
            budget = a.budget_fraction * free["cost_per_hour"]
            sol = solve_candidate_ilp(r["candidates"], r["machines"],
                                      PlacementSolverConfig(num_rsms=a.num_rsms, objective="sum_availability",
                                                            budget_per_hour=budget, unique_machine_cost=True))
            capped = summarize(sol, r["machines"], protocol, strategy, a.num_rsms, exact)
            row = {"rate_scale": k, "price": price, "candidates": len(r["candidates"]),
                   "binned_classes": r["binned_classes"], "seconds": time.time() - t,
                   "unrestricted": free, f"budget_{a.budget_fraction:g}": capped}
            rows.append(row)
            print(f"x{k:g} {price}: free {free['nines']:.2f} nines ${free['cost_per_hour']:.2f}/h | "
                  f"{a.budget_fraction:.0%} {capped['nines']:.2f} nines ${capped['cost_per_hour']:.2f}/h "
                  f"({row['seconds']:.0f}s)", flush=True)
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(a).items() if k != "output"}
    (out / "results.json").write_text(json.dumps({"catalogue": CATALOGUE, "scenario": SCENARIO,
                                                  "catalogue_digest": C.digest(cat), "params": meta,
                                                  "protocol": "raft, election Exp(0.2/s)",
                                                  "strategy": "node replacement after 600 s", "rows": rows},
                                                 indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
