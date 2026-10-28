"""Local AZ-aware Markov→MC placement frontiers for 10,000 RSMs.

Every scored candidate passes the 4.5-nines Markov *screen*, then receives at
least 100,000 full-horizon Rust MC runs. The full curves may include MC-failing
or undecided points; only certified candidates can form an SLA-qualified ILP
placement. The candidate menu is explicitly bounded, not a global oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary, catalogue  # noqa: E402
from powder.faults.ilp import (binned_markov_scorer, catalogue_machines,  # noqa: E402
                               mc_shortlist, _mc_seed)
from powder.faults.mc_scoring import (MCScoringConfig, catalogue_batch_sampler,  # noqa: E402
                                      certify_availability)
from powder.placement_optimizer import (CandidateGenerationConfig, PlacementCandidate,  # noqa: E402
                                        PlacementSolverConfig, generate_candidates,
                                        no_region_majority, no_shared_az, solve_candidate_ilp)
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402

SCENARIO = "scenario/canonical-central-az-aware@v2"
RF_COUNTS = (3, 5, 7)
NUM_RSMS = 10_000
CAPACITY = 500
GROUPS = NUM_RSMS // CAPACITY
# Bump when an engine change invalidates checkpointed MC decisions.
# trim-v2: replacement trims keep the leader and fault domains (5affbb1).
ENGINE = "trim-v2"


def block_baseline(machines, replicas: int, *, cheap: bool, seed: int) -> list[tuple[str, ...]]:
    """Twenty disjoint, 500-RSM blocks with regional and AZ anti-affinity."""
    rng = random.Random(seed)
    regions = sorted({m.node_config.region for m in machines})
    if len(regions) != 4 or replicas not in RF_COUNTS:
        raise ValueError("expected four regions and RF 3/5/7")
    by_region = {r: [m for m in machines if m.node_config.region == r] for r in regions}
    quota = GROUPS * replicas // len(regions)
    for pool in by_region.values():
        rng.shuffle(pool)
        if cheap:
            pool.sort(key=lambda m: m.node_config.cost_per_hour)
        del pool[quota:]
        if len(pool) < quota:
            raise ValueError("insufficient machines in region")
    by_id = {m.machine_id: m for m in machines}
    groups = []
    for i in range(GROUPS):
        domains = [regions[(i + j) % len(regions)] for j in range(replicas)]
        chosen = []
        for region in domains:
            pool = by_region[region]
            used_azs = {by_id[mid].node_config.az for mid in chosen}
            eligible = [m for m in pool if m.node_config.az not in used_azs]
            if not eligible:
                raise ValueError(f"AZ assignment cannot complete block {i} in {region}")
            counts = Counter(m.node_config.az for m in eligible)
            az = max(sorted(counts), key=counts.get)
            machine = next(m for m in pool if m.node_config.az == az)
            pool.remove(machine)
            chosen.append(machine.machine_id)
        ids = tuple(sorted(chosen))
        if not no_region_majority(ids, by_id) or not no_shared_az(ids, by_id):
            raise ValueError(f"invalid baseline block {ids}")
        groups.append(ids)
    if any(by_region.values()):
        raise ValueError("baseline did not use exactly the selected machines")
    return groups


def candidate_menu(machines, scorer, *, extras: int) -> dict[int, dict[str, dict[tuple[str, ...], PlacementCandidate]]]:
    """Per-RF baseline blocks plus small Markov-seeded alternative menus."""
    result = {}
    for rf in RF_COUNTS:
        pools = {}
        for policy, cheap in (("blind", False), ("cheap", True)):
            ids = block_baseline(machines, rf, cheap=cheap, seed=0)
            pools[policy] = {x: None for x in ids}
        if extras:
            generated = generate_candidates(machines, scorer, CandidateGenerationConfig(
                replica_counts=(rf,), random_samples_per_size=max(8, extras * 2),
                greedy_starts_per_size=2, local_search_steps=1, max_candidates=80,
                rng_seed=rf, reject_region_majority=True, reject_shared_az=True,
            ))
            selected = mc_shortlist(generated, extras)
            pools["extra"] = {c.machine_ids: c for c in selected}
        else:
            pools["extra"] = {}
        result[rf] = pools
    return result


def score_menu(machines, scorer, blocks):
    by_id = {m.machine_id: m for m in machines}
    scored = {}
    for rf in RF_COUNTS:
        for ids in (ids for pool in blocks[rf].values() for ids in pool):
            if ids not in scored:
                score = scorer([by_id[mid] for mid in ids])
                scored[ids] = PlacementCandidate(",".join(ids), ids,
                                                 score.availability, score.cost_per_hour,
                                                 dict(score.metadata))
    return scored


def mc_rows(lib, cat, candidates, protocol, strategy, config, checkpoint: Path,
            *, workers: int = 1):
    fingerprint = {"catalogue": catalogue.digest(cat), "scenario": SCENARIO, "engine": ENGINE,
                   "target_nines": config.target_nines, "min_runs": config.min_runs,
                   "max_runs": config.max_runs, "batch_runs": config.batch_runs,
                   "confidence": config.confidence, "comparisons": len(candidates),
                   "candidate_digest": hashlib.sha256("\n".join(
                       ",".join(ids) for ids in sorted(candidates)).encode()).hexdigest()[:16]}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            item = json.loads(line)
            if item["fingerprint"] != fingerprint:
                raise ValueError("checkpoint belongs to a different MC experiment")
            seen[tuple(item["nodes"])] = item["decision"]
    def score(ids):
        sampler = catalogue_batch_sampler(lib, cat, ids, SCENARIO, "rho0",
                                           protocol, strategy, include_region_outages=True)
        decision = certify_availability(sampler, config,
                                        base_seed=_mc_seed(config.base_seed, ",".join(ids), config.max_runs),
                                        comparisons=len(candidates))
        return ids, decision

    pending = [ids for ids in sorted(candidates) if ids not in seen]
    with checkpoint.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(score, ids) for ids in pending]):
            ids, decision = future.result()
            seen[ids] = asdict(decision)
            fh.write(json.dumps({"fingerprint": fingerprint, "nodes": ids,
                                 "decision": seen[ids]}, sort_keys=True) + "\n")
            fh.flush()
            print(f"RF{len(ids)} MC {len(seen)}/{len(candidates)}: {decision.status} "
                  f"A={decision.availability:.9f} n={decision.runs}", flush=True)
    return seen


def summarize(solution, count=NUM_RSMS):
    if not solution.selected:
        return {"status": solution.status, "message": solution.message}
    active = {mid for item in solution.selected for mid in item.candidate.machine_ids}
    counts = Counter()
    for item in solution.selected:
        counts[len(item.candidate.machine_ids)] += item.count
    return {"status": solution.status, "cost_per_hour": solution.total_cost_per_hour,
            "mean_availability": sum(item.count * item.candidate.availability for item in solution.selected) / count,
            "mean_ci_low": sum(item.count * item.candidate.metadata.get("ci_low", item.candidate.availability)
                               for item in solution.selected) / count,
            "mean_ci_high": sum(item.count * item.candidate.metadata.get("ci_high", item.candidate.availability)
                                for item in solution.selected) / count,
            "worst_availability": min(item.candidate.availability for item in solution.selected),
            "active_machines": len(active), "rf_counts": dict(counts),
            "certified_rsms": sum(item.count for item in solution.selected
                                  if item.candidate.metadata.get("certified")),
            "placements": [{"nodes": item.candidate.machine_ids, "count": item.count}
                           for item in solution.selected]}


def optimize(candidates, machines, *, budget=None, certified_only=False, target=0.0, max_active=None):
    eligible = [c for c in candidates if not certified_only or c.metadata.get("certified")]
    if not eligible:
        return {"status": "no_certified_candidates"}
    objective = "min_cost" if certified_only else "sum_availability"
    cfg = PlacementSolverConfig(num_rsms=NUM_RSMS, objective=objective,
                                unique_machine_cost=True, budget_per_hour=budget,
                                max_machines=max_active, time_limit_seconds=180,
                                min_availability=target if certified_only else None)
    return summarize(solve_candidate_ilp(eligible, machines, cfg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/fault_library/catalogue/machines-n2000-seed0-az-v2.json.gz"))
    ap.add_argument("--checkpoint", type=Path, default=Path("outputs/exec-02/az-mc-scores.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("outputs/exec-02/az-frontiers.json"))
    ap.add_argument("--extras", type=int, default=4)
    ap.add_argument("--max-runs", type=int, default=1_000_000)
    ap.add_argument("--frontier-points", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.extras < 0 or args.frontier_points < 2 or args.workers < 1:
        ap.error("extras must be nonnegative; frontier-points >= 2; workers >= 1")
    lib = FaultLibrary.load()
    master = catalogue.load(args.catalogue)
    cat = {**master, "n": 1000, "machines": master["machines"][:1000]}
    if not master.get("az_aware") or catalogue.validate(lib, master):
        raise ValueError("AZ-aware catalogue missing or invalid")
    if len(cat["machines"]) != 1000 or any(m.get("az") is None for m in cat["machines"]):
        raise ValueError("AZ-aware candidate prefix must contain 1000 bound machines")
    protocol = RaftLikeProtocol(election_time_dist=Exponential(.2))
    strategy = NodeReplacementStrategy(failure_timeout=600, default_node_config=None)
    markov = binned_markov_scorer(protocol, strategy, bins_per_decade=4)
    arms = {}
    all_ids = set()
    for price in ("rho0", "rho1"):
        machines = catalogue_machines(lib, cat, scenario_id=SCENARIO,
                                       price=price, capacity=CAPACITY)
        blocks = candidate_menu(machines, markov, extras=args.extras)
        scored = score_menu(machines, markov, blocks)
        arms[price] = {"machines": machines, "blocks": blocks, "scored": scored}
        all_ids.update(ids for ids, c in scored.items() if c.availability >= 1 - 10**-4.5)
    config = MCScoringConfig(target_nines=4.5, max_runs=args.max_runs,
                             batch_runs=100_000, max_candidates=max(1, len(all_ids)))
    mc = mc_rows(lib, cat, all_ids, protocol, strategy, config, args.checkpoint,
                 workers=args.workers)
    result = {"catalogue_digest": catalogue.digest(master), "scenario": SCENARIO,
              "score_method": "Markov screen then full-horizon Rust MC",
              "target_nines": config.target_nines, "mc_candidate_count": len(all_ids),
              "arms": {}}
    for price, arm in arms.items():
        machines, blocks = arm["machines"], arm["blocks"]
        selected = {}
        for ids, c in arm["scored"].items():
            if ids not in mc:
                continue
            d = mc[ids]
            selected[ids] = replace(c, availability=d["availability"],
                                    metadata={**c.metadata, "certified": d["status"] == "pass",
                                              "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                                              "runs": d["runs"]})
        summaries = {}
        for rf in RF_COUNTS:
            baselines = {}
            for policy in ("blind", "cheap"):
                ids = list(blocks[rf][policy])
                if any(x not in selected for x in ids):
                    baselines[policy] = {"status": "below_markov_screen"}
                    continue
                chosen = [selected[x] for x in ids]
                active = {mid for x in ids for mid in x}
                by_id = {m.machine_id: m for m in machines}
                baselines[policy] = {
                    "status": "scored", "cost_per_hour": sum(by_id[mid].node_config.cost_per_hour for mid in active),
                    "mean_availability": sum(c.availability for c in chosen) / GROUPS,
                    "mean_ci_low": sum(c.metadata["ci_low"] for c in chosen) / GROUPS,
                    "mean_ci_high": sum(c.metadata["ci_high"] for c in chosen) / GROUPS,
                    "worst_availability": min(c.availability for c in chosen),
                    "active_machines": len(active),
                    "certified_groups": sum(c.metadata["certified"] for c in chosen),
                }
            summaries[rf] = baselines
        result["arms"][price] = {"baselines": summaries, "frontiers": {}}
        for label, sizes in (("rf3", (3,)), ("rf5", (5,)), ("rf7", (7,)),
                             ("choice", RF_COUNTS)):
            menu = [c for c in selected.values() if len(c.machine_ids) in sizes]
            blind = [summaries[rf]["blind"] for rf in sizes if summaries[rf]["blind"]["status"] == "scored"]
            cheap = [summaries[rf]["cheap"] for rf in sizes if summaries[rf]["cheap"]["status"] == "scored"]
            if not blind or not cheap:
                result["arms"][price]["frontiers"][label] = {"status": "baseline_missing"}
                continue
            lower = min(b["cost_per_hour"] for b in cheap)
            upper = max(b["cost_per_hour"] for b in blind)
            budgets = [lower + (upper - lower) * i / (args.frontier_points - 1)
                       for i in range(args.frontier_points)]
            points = [{"budget_per_hour": b, **optimize(menu, machines, budget=b * (1 + 1e-9))}
                      for b in budgets]
            eligible = optimize(menu, machines, certified_only=True,
                                target=config.target_availability)
            result["arms"][price]["frontiers"][label] = {
                "status": "scored", "points": points, "target_qualified": eligible,
                "mc_qualified_candidates": sum(c.metadata["certified"] for c in menu),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
