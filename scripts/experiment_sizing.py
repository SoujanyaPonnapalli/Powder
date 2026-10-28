"""Local-only execution-order-2 catalogue-size and pricing study.

Run from the repository root with ``python3 scripts/experiment_sizing.py --output outputs/exec-02/sweep.jsonl``.
The catalogue is a *choice set*, not an AWS fleet. Nothing here calls a cloud API.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary, catalogue  # noqa: E402
from powder.faults.ilp import catalogue_machines  # noqa: E402
from powder.placement_optimizer import (  # noqa: E402
    CandidateGenerationConfig, PlacementCandidate, PlacementSolverConfig, generate_candidates,
    markov_candidate_scorer, solve_candidate_ilp,
)
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402


def nested_order(machines: list[dict], spot_fraction: float) -> list[dict]:
    """Interleave the catalogue's tier-blocked rows, preserving all original draws/prices."""
    by_tier = {tier: iter([m for m in machines if m["tier"] == tier]) for tier in ("ondemand", "spot")}
    out = []
    for i in range(len(machines)):
        spot = round((i + 1) * spot_fraction) > round(i * spot_fraction)
        out.append(next(by_tier["spot" if spot else "ondemand"]))
    return out


def baseline(machines, num_rsms: int, replicas: int, capacity: int, *, cheap: bool, seed: int, scorer):
    """Select one node per region without seeing a node's fault rates."""
    rng = random.Random(seed)
    regions = sorted({m.node_config.region for m in machines})
    if len(regions) < min(replicas, 4):
        raise ValueError("not enough regions for the requested region-balanced baseline")
    by_region = {r: [m for m in machines if m.node_config.region == r] for r in regions}
    for pool in by_region.values():
        rng.shuffle(pool)
        if cheap:
            pool.sort(key=lambda m: m.node_config.cost_per_hour)
    remaining = {m.machine_id: capacity for m in machines}
    selected = []
    by_id = {m.machine_id: m for m in machines}
    spread = [tuple(regions[(i + j) % len(regions)] for j in range(replicas))
              for i in range(num_rsms)]
    # For 3-of-4 placement, balance region usage *in machine-sized blocks*.
    # A round-robin baseline wastes two mostly-empty machines when 150 replicas
    # are distributed 38/38/37/37 over four regions with capacity five.
    total = num_rsms * replicas
    if (len(regions) == replicas + 1 and total % capacity == 0):
        low = (total // (len(regions) * capacity)) * capacity
        extra = (total - low * len(regions)) // capacity
        if 0 <= low <= low + capacity <= num_rsms and extra <= len(regions):
            increments = sorted(regions, key=lambda r: by_region[r][low // capacity].node_config.cost_per_hour)
            if not cheap:
                rng.shuffle(increments)
            counts = {r: low + (capacity if r in increments[:extra] else 0) for r in regions}
            omitted = [r for r in regions for _ in range(num_rsms - counts[r])]
            rng.shuffle(omitted)
            spread = [tuple(r for r in regions if r != omitted[i]) for i in range(num_rsms)]
    balanced_pools = None
    if replicas > len(regions):
        demand = Counter(r for group in spread for r in group)
        balanced_pools = {r: by_region[r][: (demand[r] + capacity - 1) // capacity] for r in regions}
    for selected_regions in spread:
        chosen = []
        for r in selected_regions:
            if balanced_pools is None:
                available = next((m for m in by_region[r]
                                  if remaining[m.machine_id] and m.machine_id not in chosen), None)
            else:
                available = max((m for m in balanced_pools[r]
                                 if remaining[m.machine_id] and m.machine_id not in chosen),
                                key=lambda m: remaining[m.machine_id], default=None)
            if available is None:
                raise ValueError(f"baseline exhausted capacity in {r}")
            remaining[available.machine_id] -= 1
            chosen.append(available.machine_id)
        selected.append(tuple(sorted(chosen)))
    active = {mid for ids in selected for mid in ids}
    scores = {ids: scorer([by_id[mid] for mid in ids]).availability for ids in set(selected)}
    return {
        "active_ids": sorted(active),
        "cost_per_hour": sum(by_id[mid].node_config.cost_per_hour for mid in active),
        "mean_availability": sum(scores[ids] for ids in selected) / num_rsms,
        "min_availability": min(scores.values()),
        "placements": [list(ids) for ids in selected],
        "deployed": len(active),
    }


def candidates_for(machines, scorer, replicas: int, seed: int, samples: int, cap: int):
    """Search both reliability and price ends; preserve their candidate coverage."""
    by_ids = {}
    for weight in (0.0, 0.01):
        cfg = CandidateGenerationConfig(
            replica_counts=(replicas,), random_samples_per_size=samples,
            greedy_starts_per_size=3, local_search_steps=1,
            max_candidates=max(2 * (samples + 10), cap), rng_seed=seed,
            cost_weight=weight,
        )
        for candidate in generate_candidates(machines, scorer, cfg):
            by_ids[candidate.machine_ids] = candidate
    regions = {m.machine_id: m.node_config.region for m in machines}
    required_regions = min(replicas, len(set(regions.values())))
    ranked = [c for c in by_ids.values()
              if len({regions[mid] for mid in c.machine_ids}) == required_regions]
    if len(ranked) > cap:
        half = cap // 2
        cheapest = sorted(ranked, key=lambda c: (c.cost_per_hour, -c.availability))[:half]
        best = sorted(ranked, key=lambda c: (-c.availability, c.cost_per_hour))[:cap - half]
        chosen = {c.machine_ids: c for c in cheapest + best}
        for c in sorted(ranked, key=lambda c: (-c.availability, c.cost_per_hour)):
            if len(chosen) >= cap:
                break
            chosen[c.machine_ids] = c
        ranked = list(chosen.values())
    return ranked, len(by_ids)


def evaluate_solution(solution, by_id, protocol, strategy, num_rsms):
    if not solution.selected:
        return {"status": solution.status, "message": solution.message,
                "mip_gap": solution.mip_gap}
    active = {mid for item in solution.selected for mid in item.candidate.machine_ids}
    exact = markov_candidate_scorer(protocol, strategy)
    exact_scores = {item.candidate.machine_ids: exact([by_id[mid] for mid in item.candidate.machine_ids]).availability
                    for item in solution.selected}
    return {
        "status": solution.status, "message": solution.message, "mip_gap": solution.mip_gap,
        "cost_per_hour": solution.total_cost_per_hour,
        "mean_availability_scored": sum(item.candidate.availability * item.count for item in solution.selected) / num_rsms,
        "mean_availability_exact": sum(exact_scores[item.candidate.machine_ids] * item.count
                                       for item in solution.selected) / num_rsms,
        "min_availability_exact": min(exact_scores.values()),
        "deployed": len(active), "active_ids": sorted(active),
        "placements": [{"nodes": list(item.candidate.machine_ids), "count": item.count}
                       for item in solution.selected],
    }


def run(args):
    lib = FaultLibrary.load()
    master = catalogue.generate(lib, args.master_size or max(args.pools), args.catalogue_seed,
                                args.spot_fraction, args.horizon_days * 24)
    ordered = nested_order(master["machines"], args.spot_fraction)
    protocol = RaftLikeProtocol(election_time_dist=Exponential(0.2))
    strategy = NodeReplacementStrategy(failure_timeout=600.0, default_node_config=None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for seed in range(args.price_seeds):
            # Same machine faults and same 8-vCPU price quantiles, new independent permutation.
            master["price_assignments"] = catalogue.assign_catalogue_prices(
                lib, master["machines"], seed=seed, horizon_h=master["horizon_h"]
            )
            for size in args.pools:
                subset = {**master, "n": size, "machines": ordered[:size]}
                counts = Counter(m["tier"] for m in subset["machines"])
                recipes = len({m["recipe_key"] for m in subset["machines"]})
                for price in ("rho0", "rho1") if seed == 0 else ("rho0",):
                    started = time.monotonic()
                    machines = catalogue_machines(lib, subset, price=price, capacity=args.capacity)
                    scorer = markov_candidate_scorer(protocol, strategy)
                    by_id = {m.machine_id: m for m in machines}
                    cheap = baseline(machines, args.num_rsms, args.replicas, args.capacity,
                                     cheap=True, seed=seed, scorer=scorer)
                    blind = baseline(machines, args.num_rsms, args.replicas, args.capacity,
                                     cheap=False, seed=seed, scorer=scorer)
                    candidates, generated = candidates_for(machines, scorer, args.replicas,
                                                           seed, args.samples, args.candidate_cap)
                    candidate_by_ids = {c.machine_ids: c for c in candidates}
                    for ids in sorted({tuple(ids) for base in (cheap, blind) for ids in base["placements"]}):
                        if ids not in candidate_by_ids:
                            score = scorer([by_id[mid] for mid in ids])
                            candidate_by_ids[ids] = PlacementCandidate("baseline:" + ":".join(ids), ids,
                                                                        score.availability, score.cost_per_hour)
                    candidates = list(candidate_by_ids.values())
                    effective_floor = max(args.availability_floor, cheap["min_availability"])
                    cfg = PlacementSolverConfig(num_rsms=args.num_rsms, unique_machine_cost=True,
                                                max_machines=cheap["deployed"],
                                                budget_per_hour=cheap["cost_per_hour"] * (1 + 1e-9),
                                                time_limit_seconds=args.solver_seconds)
                    iso_cost = solve_candidate_ilp(candidates, machines, cfg)
                    floor_cfg = PlacementSolverConfig(num_rsms=args.num_rsms, objective="min_cost",
                                                     unique_machine_cost=True,
                                                     min_availability=effective_floor,
                                                     min_mean_availability=cheap["mean_availability"],
                                                     max_machines=args.max_deployed,
                                                     time_limit_seconds=args.solver_seconds)
                    iso_availability = solve_candidate_ilp(candidates, machines, floor_cfg)
                    blind_cfg = PlacementSolverConfig(num_rsms=args.num_rsms, objective="min_cost",
                                                      unique_machine_cost=True,
                                                      min_availability=max(args.availability_floor,
                                                                           blind["min_availability"]),
                                                      min_mean_availability=blind["mean_availability"],
                                                      max_machines=args.max_deployed,
                                                      time_limit_seconds=args.solver_seconds)
                    iso_blind_quality = solve_candidate_ilp(candidates, machines, blind_cfg)
                    row = {
                        "catalogue_seed": args.catalogue_seed, "price_seed": seed,
                        "master_digest": catalogue.digest(master), "master_size": len(master["machines"]),
                        "pool": size, "price": price,
                        "price_correlation": master["price_assignments"][price]["realised_spearman"],
                        "num_rsms": args.num_rsms, "replicas": args.replicas,
                        "max_deployed": args.max_deployed, "capacity": args.capacity,
                        "availability_floor": effective_floor,
                        "spot_fraction": args.spot_fraction,
                        "tier_counts": dict(counts), "distinct_recipes": recipes,
                        "distinct_on_demand_recipes": len({m["recipe_key"] for m in subset["machines"]
                                                            if m["tier"] == "ondemand"}),
                        "catalogue_capacity": master["capacity"],
                        "distinct_prices": len({round(m.node_config.cost_per_hour, 6) for m in machines}),
                        "candidates_generated": generated, "candidates_kept": len(candidates),
                        "candidate_machine_coverage": len({mid for c in candidates for mid in c.machine_ids}),
                        "cheap": cheap, "blind": blind,
                        "iso_cost": evaluate_solution(iso_cost, by_id, protocol, strategy, args.num_rsms),
                        "iso_availability": evaluate_solution(iso_availability, by_id, protocol, strategy,
                                                              args.num_rsms),
                        "iso_blind_quality": evaluate_solution(iso_blind_quality, by_id, protocol, strategy,
                                                               args.num_rsms),
                        "seconds": time.monotonic() - started,
                    }
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                    print(f"M={size} {price} seed={seed} {row['seconds']:.1f}s "
                          f"cost={row['iso_availability'].get('cost_per_hour')} "
                          f"status={iso_availability.status}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="outputs/exec-02/sweep.jsonl")
    ap.add_argument("--pools", type=int, nargs="+", default=[250, 500, 1000, 2000])
    ap.add_argument("--master-size", type=int, default=None,
                    help="freeze a larger inventory before taking the requested pool prefixes")
    ap.add_argument("--catalogue-seed", type=int, default=0)
    ap.add_argument("--price-seeds", type=int, default=3)
    ap.add_argument("--spot-fraction", type=float, default=0.0)
    ap.add_argument("--horizon-days", type=float, default=30)
    ap.add_argument("--num-rsms", type=int, default=50)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--capacity", type=int, default=5)
    ap.add_argument("--max-deployed", type=int, default=64)
    ap.add_argument("--availability-floor", type=float, default=0.9999)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--candidate-cap", type=int, default=600)
    ap.add_argument("--solver-seconds", type=float, default=30)
    args = ap.parse_args()
    if (not args.pools or any(m < args.max_deployed for m in args.pools) or
            (args.master_size is not None and args.master_size < max(args.pools)) or
            args.price_seeds < 1 or args.replicas < 1 or args.max_deployed * args.capacity < args.num_rsms * args.replicas):
        ap.error("master pool and capacity must accommodate the deployed fleet and replicas")
    run(args)


if __name__ == "__main__":
    main()
