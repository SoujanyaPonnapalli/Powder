"""Equal-budget availability and modeled price/availability frontier for exec #2.

Replays the frozen inventory and candidate menu used by experiment_sizing.py.
The frontier is a candidate-menu result under the simplified Markov model,
not a global oracle or a prediction of the bill for identical AWS hosts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary, catalogue  # noqa: E402
from powder.faults.ilp import catalogue_machines  # noqa: E402
from powder.placement_optimizer import (  # noqa: E402
    PlacementCandidate, PlacementSolverConfig, markov_candidate_scorer, solve_candidate_ilp,
)
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402
from scripts.experiment_sizing import candidates_for, evaluate_solution, nested_order  # noqa: E402


def candidate_menu(machines, scorer, row):
    candidates, _ = candidates_for(machines, scorer, row["replicas"], row["price_seed"], 120, 600)
    by_id = {m.machine_id: m for m in machines}
    by_ids = {c.machine_ids: c for c in candidates}
    for ids in sorted({tuple(ids) for base in (row["cheap"], row["blind"]) for ids in base["placements"]}):
        if ids not in by_ids:
            score = scorer([by_id[mid] for mid in ids])
            by_ids[ids] = PlacementCandidate("baseline:" + ":".join(ids), ids,
                                              score.availability, score.cost_per_hour)
    return list(by_ids.values()), by_id


def solve_budget(candidates, machines, by_id, protocol, strategy, row, budget):
    cfg = PlacementSolverConfig(num_rsms=row["num_rsms"], unique_machine_cost=True,
                                max_machines=row["blind"]["deployed"],
                                min_availability=max(0.9999, row["blind"]["min_availability"]),
                                budget_per_hour=budget * (1 + 1e-9),
                                time_limit_seconds=30)
    sol = solve_candidate_ilp(candidates, machines, cfg)
    return evaluate_solution(sol, by_id, protocol, strategy, row["num_rsms"])


def run(sweep: Path, output: Path, pools: list[int], price_seeds: int):
    saved = [json.loads(line) for line in sweep.read_text().splitlines()]
    lookup = {(r["pool"], r["price_seed"], r["price"]): r for r in saved}
    source = saved[0]
    if source["spot_fraction"] != 0:
        raise ValueError("on-demand price frontier requires an on-demand-only inventory")
    lib = FaultLibrary.load()
    # The frozen sizing run used the integer 720-hour default; its JSON digest
    # distinguishes 720 from 720.0 even though their hazard calculations agree.
    master = catalogue.generate(lib, max(r["pool"] for r in saved), source["catalogue_seed"],
                                source["spot_fraction"], 720)
    ordered = nested_order(master["machines"], source["spot_fraction"])
    protocol = RaftLikeProtocol(election_time_dist=Exponential(0.2))
    strategy = NodeReplacementStrategy(failure_timeout=600.0, default_node_config=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for seed in range(price_seeds):
            master["price_assignments"] = catalogue.assign_catalogue_prices(
                lib, master["machines"], seed=seed, horizon_h=master["horizon_h"]
            )
            for pool in pools:
                for price in ("rho0", "rho1") if seed == 0 else ("rho0",):
                    row = lookup[(pool, seed, price)]
                    if row["master_digest"] != catalogue.digest(master):
                        raise ValueError("sweep and frontier catalogue or prices differ")
                    subset = {**master, "n": pool, "machines": ordered[:pool]}
                    machines = catalogue_machines(lib, subset, price=price,
                                                   capacity=row["capacity"])
                    scorer = markov_candidate_scorer(protocol, strategy)
                    candidates, by_id = candidate_menu(machines, scorer, row)
                    blind = row["blind"]
                    equal_cost = solve_budget(candidates, machines, by_id, protocol, strategy,
                                              row, blind["cost_per_hour"])
                    if equal_cost["status"] != "0":
                        raise RuntimeError(f"equal-cost solver failed: {equal_cost}")
                    if (equal_cost["mean_availability_exact"] + 1e-10 < blind["mean_availability"] or
                            equal_cost["min_availability_exact"] + 1e-10 < blind["min_availability"]):
                        raise RuntimeError("equal-cost solution violates baseline availability")
                    result = {
                        "pool": pool, "price_seed": seed, "price": price,
                        "master_digest": row["master_digest"], "candidates": len(candidates),
                        "blind": blind, "cheapest_first": row["cheap"],
                        "min_cost_at_blind_target": row["iso_blind_quality"],
                        "max_availability_at_blind_cost": equal_cost,
                    }
                    if seed == 0 and pool == max(pools):
                        lo = row["iso_blind_quality"]["cost_per_hour"]
                        hi = blind["cost_per_hour"]
                        budgets = [lo + (hi - lo) * i / 8 for i in range(9)]
                        result["frontier"] = [
                            {"budget_per_hour": b, **solve_budget(candidates, machines, by_id,
                                protocol, strategy, row, b)} for b in budgets
                        ]
                    fh.write(json.dumps(result, sort_keys=True) + "\n")
                    fh.flush()
                    base_u = 1 - blind["mean_availability"]
                    gain = 100 * (1 - (1 - equal_cost["mean_availability_exact"]) / base_u)
                    print(f"M={pool} {price} seed={seed} equal-$ unavailability reduction {gain:.2f}%",
                          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, default=Path("outputs/exec-02/blind-quality-deterministic.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("outputs/exec-02/frontier-deterministic.jsonl"))
    ap.add_argument("--pools", type=int, nargs="+", default=[250, 1000])
    ap.add_argument("--price-seeds", type=int, default=20)
    args = ap.parse_args()
    if not args.pools or args.price_seeds < 1:
        ap.error("provide at least one pool and one price seed")
    run(args.sweep, args.output, args.pools, args.price_seeds)


if __name__ == "__main__":
    main()
