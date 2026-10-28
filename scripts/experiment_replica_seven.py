"""Screening-only RF=7 cost frontier using rate-binned Markov scores.

Seven distinct-machine Markov solves are too slow for a broad candidate menu.
The plotted values here are *not exact*, and a few exact rescoring checks are
reported to expose approximation error. The RF=3/5 figures use exact scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary, catalogue  # noqa: E402
from powder.faults.ilp import binned_markov_scorer, catalogue_machines  # noqa: E402
from powder.placement_optimizer import (PlacementSolverConfig, markov_candidate_scorer,  # noqa: E402
                                        solve_candidate_ilp)
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402
from scripts.experiment_frontier import candidate_menu  # noqa: E402
from scripts.experiment_sizing import baseline, nested_order  # noqa: E402


def summarize(solution, count):
    if not solution.selected:
        return {"status": solution.status, "message": solution.message}
    active = {mid for entry in solution.selected for mid in entry.candidate.machine_ids}
    return {
        "status": solution.status, "mip_gap": solution.mip_gap,
        "cost_per_hour": solution.total_cost_per_hour,
        "mean_availability_model": sum(entry.candidate.availability * entry.count
                                       for entry in solution.selected) / count,
        "min_availability_model": min(entry.candidate.availability for entry in solution.selected),
        "deployed": len(active),
        "placements": [{"nodes": list(entry.candidate.machine_ids), "count": entry.count}
                       for entry in solution.selected],
    }


def selected_group(result):
    return max(result["placements"], key=lambda p: p["count"])["nodes"]


def run(output: Path, *, samples: int, candidate_cap: int, solver_seconds: float,
        frontier_points: int, exact_checks: int):
    lib = FaultLibrary.load()
    master = catalogue.generate(lib, 2000, 0, 0.0, 720)
    subset = {**master, "n": 1000, "machines": nested_order(master["machines"], 0.0)[:1000]}
    protocol = RaftLikeProtocol(election_time_dist=Exponential(.2))
    strategy = NodeReplacementStrategy(failure_timeout=600.0, default_node_config=None)
    scorer = binned_markov_scorer(protocol, strategy, bins_per_decade=4)
    exact = markov_candidate_scorer(protocol, strategy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for price in ("rho0", "rho1"):
            machines = catalogue_machines(lib, subset, price=price, capacity=500)
            by_id = {m.machine_id: m for m in machines}
            cheap = baseline(machines, 10000, 7, 500, cheap=True, seed=0, scorer=scorer)
            blind = baseline(machines, 10000, 7, 500, cheap=False, seed=0, scorer=scorer)
            row = {"replicas": 7, "price_seed": 0, "cheap": cheap, "blind": blind}
            candidates, _ = candidate_menu(machines, scorer, row,
                                           samples=samples, candidate_cap=candidate_cap)
            common = dict(num_rsms=10000, unique_machine_cost=True,
                          time_limit_seconds=solver_seconds)
            target = summarize(solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
                **common, objective="min_cost", max_machines=160,
                min_availability=max(.9999, blind["min_availability"]),
                min_mean_availability=blind["mean_availability"])), 10000)
            equal = summarize(solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
                **common, max_machines=blind["deployed"],
                min_availability=max(.9999, blind["min_availability"]),
                budget_per_hour=blind["cost_per_hour"] * (1 + 1e-9))), 10000)
            budgets = [cheap["cost_per_hour"] +
                       (blind["cost_per_hour"] - cheap["cost_per_hour"]) * i / (frontier_points - 1)
                       for i in range(frontier_points)]
            frontier = [
                {"budget_per_hour": b, **summarize(solve_candidate_ilp(
                    candidates, machines, PlacementSolverConfig(
                        **common, max_machines=blind["deployed"],
                        budget_per_hour=b * (1 + 1e-9))), 10000)}
                for b in budgets
            ]
            if any(x["status"] != "0" for x in [target, equal, *frontier]):
                raise RuntimeError(f"RF7 solve did not finish at {price}")
            checks = []
            groups = [blind["placements"][0], selected_group(target), selected_group(equal),
                      selected_group(frontier[0]), selected_group(frontier[-1])]
            seen = set()
            for ids in groups:
                ids = tuple(ids)
                if ids in seen:
                    continue
                seen.add(ids)
                nodes = [by_id[mid] for mid in ids]
                approx = scorer(nodes).availability
                observed = exact(nodes).availability
                checks.append({"nodes": list(ids), "binned": approx, "exact": observed,
                               "relative_unavailability_error": abs((1-approx)/(1-observed)-1)})
                if len(checks) >= exact_checks:
                    break
            result = {"price": price, "pool": 1000, "num_rsms": 10000, "replicas": 7,
                      "capacity": 500, "master_digest": catalogue.digest(master),
                      "score_method": "binned Markov, four rate bins per decade",
                      "candidate_count": len(candidates), "scorer_calls": scorer.calls,
                      "blind": blind, "cheapest_first": cheap,
                      "min_cost_at_blind_target": target,
                      "max_availability_at_blind_cost": equal,
                      "frontier_has_baseline_floor": False,
                      "frontier": frontier, "exact_spot_checks": checks}
            fh.write(json.dumps(result, sort_keys=True) + "\n")
            fh.flush()
            print(price, "candidate count", len(candidates), "exact checks", len(checks), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("outputs/exec-02/rsm10000-rf7-binned.jsonl"))
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--candidate-cap", type=int, default=120)
    ap.add_argument("--solver-seconds", type=float, default=120)
    ap.add_argument("--frontier-points", type=int, default=7)
    ap.add_argument("--exact-checks", type=int, default=2)
    args = ap.parse_args()
    if args.samples < 1 or args.candidate_cap < 1 or args.frontier_points < 2 or args.exact_checks < 1:
        ap.error("positive sampling and validation settings required")
    run(args.output, samples=args.samples, candidate_cap=args.candidate_cap,
        solver_seconds=args.solver_seconds, frontier_points=args.frontier_points,
        exact_checks=args.exact_checks)


if __name__ == "__main__":
    main()
