"""Candidate ILP with unique machine billing, solved by Gurobi (docs/plans/ilp-fleet-sampling.md).

The original count-based model remains the default. ``solve_transition_gurobi``
adds a named-shard choice, recurring replication transfer and one-time migration
amortized over an explicit decision horizon. Both models charge a physical machine
once when used, regardless of whether it was part of the previous deployment.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

from powder.placement_optimizer import PlacementCandidate, PlacementSolution, SelectedCandidate
from .composed_availability import CandidateTimeline
from .transition_costs import ShardSpec, edge_average_per_gb, migration_cost


@dataclass(frozen=True)
class TransitionProblem:
    """Named-shard workload and old placement for one optimization decision.

    ``traffic_price(src_region, dst_region, same_az)`` uses real traffic pricing;
    ``region_price(src_region, dst_region)`` is directed inter-region transfer.
    Same-region migration is always free, independently of traffic pricing.
    """

    shards: tuple[ShardSpec, ...]
    locations: Mapping[str, tuple[str, str]]
    traffic_price: Callable[[str, str, bool], float]
    region_price: Callable[[str, str], float]
    horizon_h: float
    max_replica_moves: int | None = None


def solve_unique_cost_gurobi(candidates: Sequence[PlacementCandidate], price: dict[str, float], *,
                             capacity: int, num_rsms: int, budget: float | None,
                             warm_start: Sequence[tuple[str, ...]] = (), time_limit: float = 120.0,
                             mip_gap: float = 1e-5, threads: int = 1, seed: int = 0, minimize_cost: bool = False,
                             max_unavailability_ppm: float | None = None) -> PlacementSolution:
    import gurobipy as gp

    t0 = time.time()
    used = sorted({m for c in candidates for m in c.machine_ids})
    model = gp.Model("fleet_ilp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)
    x = model.addVars(len(candidates), vtype=gp.GRB.INTEGER, lb=0, ub=num_rsms)
    y = model.addVars(used, vtype=gp.GRB.BINARY)
    model.addConstr(x.sum() == num_rsms)
    by_machine: dict[str, list[int]] = {m: [] for m in used}
    for j, c in enumerate(candidates):
        for m in c.machine_ids:
            by_machine[m].append(j)
            model.addConstr(x[j] <= capacity * y[m])
    for m, js in by_machine.items():
        model.addConstr(gp.quicksum(x[j] for j in js) <= capacity * y[m])
    if budget is not None:
        model.addConstr(gp.quicksum(price[m] * y[m] for m in used) <= budget)
    unavailability = gp.quicksum((1.0 - c.availability) * 1e6 * x[j] for j, c in enumerate(candidates))
    if minimize_cost:
        if max_unavailability_ppm is None:
            raise ValueError("minimize_cost needs max_unavailability_ppm")
        model.addConstr(unavailability <= max_unavailability_ppm)
        model.setObjective(gp.quicksum(price[m] * y[m] for m in used), gp.GRB.MINIMIZE)
    else:
        model.setObjective(unavailability, gp.GRB.MINIMIZE)
    if warm_start:
        index = {c.machine_ids: j for j, c in enumerate(candidates)}
        counts = Counter(tuple(sorted(p)) for p in warm_start)
        if all(p in index for p in counts):
            for j in range(len(candidates)):
                x[j].Start = 0
            for p, n in counts.items():
                x[index[p]].Start = n
            on = {m for p in counts for m in p}
            for m in used:
                y[m].Start = 1 if m in on else 0
    model.optimize()
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit", gp.GRB.INFEASIBLE: "infeasible"}.get(
        model.Status, str(model.Status))
    if model.SolCount == 0:
        return PlacementSolution(status, "no incumbent", "sum_availability", float("nan"), float("nan"), budget, (),
                                 solver_runtime_seconds=time.time() - t0)
    selected = tuple(SelectedCandidate(c, int(round(x[j].X))) for j, c in enumerate(candidates) if x[j].X > 0.5)
    active = {m for it in selected for m in it.candidate.machine_ids}
    return PlacementSolution(status, "", "sum_availability",
                             sum(it.candidate.availability * it.count for it in selected),
                             sum(price[m] for m in active), budget, selected,
                             solver_runtime_seconds=time.time() - t0, mip_gap=model.MIPGap,
                             best_bound=model.ObjBound)


def solve_transition_gurobi(candidates: Sequence[PlacementCandidate], price: Mapping[str, float], *,
                            capacity: int, budget: float | None, transition: TransitionProblem,
                            warm_start: Sequence[tuple[str, ...]] = (), time_limit: float = 120.0,
                            mip_gap: float = 1e-5, threads: int = 1, seed: int = 0,
                            minimize_cost: bool = False,
                            max_unavailability_ppm: float | None = None,
                            composed: CandidateTimeline | None = None,
                            composed_objective: Literal["single", "full"] = "single",
                            shard_weights: Mapping[str, float] | None = None,
                            min_single_access: float | None = None,
                            min_full_access: float | None = None) -> PlacementSolution:
    """Choose one candidate per named shard under an all-in hourly budget.

    With ``composed``, candidate outcomes come from the SAME shared-fault
    trajectory; ``single`` maximizes weighted one-shard access, ``full`` uses
    scenario AND variables for simultaneous access to every shard. These are
    candidate-menu optima, not a claim of a globally optimal fault model.
    Legacy calls without ``composed`` retain the original availability goal.
    """
    import gurobipy as gp

    t0 = time.time()
    shards = transition.shards
    if not shards or len({s.shard_id for s in shards}) != len(shards):
        raise ValueError("shards must be nonempty and have unique IDs")
    if not math.isfinite(transition.horizon_h) or transition.horizon_h <= 0:
        raise ValueError("horizon_h must be positive and finite")
    if not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if budget is not None and (not math.isfinite(budget) or budget < 0):
        raise ValueError("budget must be finite and nonnegative")
    if transition.max_replica_moves is not None and (not isinstance(transition.max_replica_moves, int)
                                                      or transition.max_replica_moves < 0):
        raise ValueError("max_replica_moves must be a nonnegative integer")
    has_old = [bool(s.old_placement) for s in shards]
    if any(has_old) and not all(has_old):
        raise ValueError("supply old placements for every shard or none")
    if transition.max_replica_moves is not None and not all(has_old):
        raise ValueError("movement cap requires old placements")
    if composed is None:
        if (composed_objective != "single" or shard_weights is not None
                or min_single_access is not None or min_full_access is not None):
            raise ValueError("composed objectives and floors require a shared CandidateTimeline")
        weights = None
    else:
        if composed_objective not in ("single", "full"):
            raise ValueError("composed_objective must be single or full")
        if set(composed.local_ok) != {s.shard_id for s in shards}:
            raise ValueError("candidate timeline must contain every and only modeled shard")
        weights = ({s.shard_id: 1.0 / len(shards) for s in shards}
                   if shard_weights is None else dict(shard_weights))
        if (set(weights) != set(composed.local_ok)
                or any(not math.isfinite(v) or v < 0 for v in weights.values())
                or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9, rel_tol=0)):
            raise ValueError("shard_weights must be nonnegative and sum to one")
        for floor in (min_single_access, min_full_access):
            if floor is not None and (not math.isfinite(floor) or not 0 <= floor <= 1):
                raise ValueError("composed access floors must lie in [0,1]")
        if minimize_cost and min_single_access is None and min_full_access is None:
            raise ValueError("minimize_cost with composed scores requires an access floor")
        if max_unavailability_ppm is not None:
            raise ValueError("use min_single_access or min_full_access with composed scores")
        if len(shards) * len(composed.seconds) > 2_000_000:
            raise ValueError("full scenario model exceeds 2,000,000 shard-interval rows; reduce the menu")
    used = sorted({m for c in candidates for m in c.machine_ids})
    if any(m not in price or not math.isfinite(price[m]) or price[m] < 0 for m in used):
        raise ValueError("all candidate machines need finite nonnegative prices")
    if len({c.machine_ids for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate placements")
    for c in candidates:
        if not c.machine_ids or len(set(c.machine_ids)) != len(c.machine_ids):
            raise ValueError("candidate placements must have distinct machines")
        if not math.isfinite(c.availability) or not 0 <= c.availability <= 1:
            raise ValueError("candidate availability must lie in [0,1]")
    # Precompute only once per placement. Region-pair prices are directional;
    # same-AZ and inter-AZ traffic use the caller's original transfer policy.
    edge = [edge_average_per_gb(c.machine_ids, transition.locations, transition.traffic_price)
            for c in candidates]
    arcs: dict[tuple[int, int], tuple[float, int, int, float, float]] = {}
    for i, shard in enumerate(shards):
        if not math.isfinite(shard.write_gb_per_s) or shard.write_gb_per_s < 0 or not math.isfinite(
                shard.partition_gb) or shard.partition_gb < 0:
            raise ValueError("shard sizes and write rates must be finite and nonnegative")
        for j, c in enumerate(candidates):
            if composed is not None and c.machine_ids not in composed.local_ok[shard.shard_id]:
                continue  # candidate not evaluated for this shard on the shared trace
            moved = migration_cost(shard.old_placement, c.machine_ids, shard.partition_gb,
                                   transition.locations, transition.region_price) if has_old[i] else None
            additions = moved.replica_additions if moved else 0
            if transition.max_replica_moves is not None and additions > transition.max_replica_moves:
                continue
            traffic = 3600 * shard.write_gb_per_s * (len(c.machine_ids) - 1) * edge[j]
            if not math.isfinite(traffic):
                raise ValueError("replication hourly cost must be finite")
            arcs[i, j] = (traffic, additions, moved.replica_removals if moved else 0,
                          moved.gb_moved if moved else 0.0, moved.migration_usd if moved else 0.0)
    model = gp.Model("transition_fleet_ilp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)
    x = model.addVars(arcs, vtype=gp.GRB.BINARY)
    y = model.addVars(used, vtype=gp.GRB.BINARY)
    for i in range(len(shards)):
        model.addConstr(gp.quicksum(x[i, j] for j in range(len(candidates)) if (i, j) in arcs) == 1)
    by_machine: dict[str, list[tuple[int, int]]] = {m: [] for m in used}
    for i, j in arcs:
        for m in candidates[j].machine_ids:
            by_machine[m].append((i, j))
    for m, members in by_machine.items():
        use = gp.quicksum(x[i, j] for i, j in members)
        model.addConstr(use <= capacity * y[m])
        model.addConstr(y[m] <= use)  # no phantom billed activations in the reported solution
    machine_rate = gp.quicksum(price[m] * y[m] for m in used)
    traffic_rate = gp.quicksum(vals[0] * x[i, j] for (i, j), vals in arcs.items())
    migration = gp.quicksum(vals[4] * x[i, j] for (i, j), vals in arcs.items())
    effective_rate = machine_rate + traffic_rate + migration / transition.horizon_h
    if budget is not None:
        model.addConstr(effective_rate <= budget)
    if transition.max_replica_moves is not None:
        model.addConstr(gp.quicksum(vals[1] * x[i, j] for (i, j), vals in arcs.items())
                        <= transition.max_replica_moves)
    if composed is None:
        unavailability = gp.quicksum((1 - candidates[j].availability) * 1e6 * x[i, j] for i, j in arcs)
        if minimize_cost:
            if max_unavailability_ppm is None:
                raise ValueError("minimize_cost needs max_unavailability_ppm")
            model.addConstr(unavailability <= max_unavailability_ppm)
            model.setObjective(effective_rate, gp.GRB.MINIMIZE)
        else:
            model.setObjective(unavailability, gp.GRB.MINIMIZE)
    else:
        single_success = gp.quicksum(
            weights[shards[i].shard_id] * composed.candidate_access(shards[i].shard_id, candidates[j].machine_ids)
            * x[i, j] for i, j in arcs)
        full_success = None
        if composed_objective == "full" or min_full_access is not None:
            z = model.addVars(len(composed.seconds), vtype=gp.GRB.BINARY, name="all_shards_work")
            for k, global_ok in enumerate(composed.global_ok):
                model.addConstr(z[k] <= int(global_ok))
                if global_ok:
                    for i, shard in enumerate(shards):
                        model.addConstr(z[k] <= gp.quicksum(
                            int(composed.local_ok[shard.shard_id][candidates[j].machine_ids][k]) * x[i, j]
                            for j in range(len(candidates)) if (i, j) in arcs))
            full_success = gp.quicksum(
                composed.seconds[k] / composed.horizon_seconds * z[k]
                for k in range(len(composed.seconds)))
        if min_single_access is not None:
            model.addConstr(single_success >= min_single_access)
        if min_full_access is not None:
            model.addConstr(full_success >= min_full_access)
        if minimize_cost:
            model.setObjective(effective_rate, gp.GRB.MINIMIZE)
        elif composed_objective == "single":
            model.setObjective((1 - single_success) * 1e6, gp.GRB.MINIMIZE)
        else:
            model.setObjective((1 - full_success) * 1e6, gp.GRB.MINIMIZE)
    if len(warm_start) == len(shards):
        index = {c.machine_ids: j for j, c in enumerate(candidates)}
        pairs = [(i, index.get(tuple(sorted(p)))) for i, p in enumerate(warm_start)]
        if all(j is not None and (i, j) in arcs for i, j in pairs):
            chosen = {(i, j) for i, j in pairs}
            on = {m for _, j in chosen for m in candidates[j].machine_ids}
            for ij in arcs:
                x[ij].Start = int(ij in chosen)
            for m in used:
                y[m].Start = int(m in on)
    model.optimize()
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit", gp.GRB.INFEASIBLE: "infeasible"}.get(
        model.Status, str(model.Status))
    if not model.SolCount:
        return PlacementSolution(status, "no incumbent", "sum_availability", float("nan"), float("nan"), budget, (),
                                 solver_runtime_seconds=time.time() - t0, horizon_h=transition.horizon_h,
                                 composed_objective=composed_objective if composed is not None else None)
    picks = [(i, j) for i, j in arcs if x[i, j].X > 0.5]
    picks.sort()
    counts = Counter(j for _, j in picks)
    selected = tuple(SelectedCandidate(c, counts[j]) for j, c in enumerate(candidates) if counts[j])
    active = {m for _, j in picks for m in candidates[j].machine_ids}
    machine_usd_h = sum(price[m] for m in active)
    traffic_usd_h = sum(arcs[i, j][0] for i, j in picks)
    migration_usd = sum(arcs[i, j][4] for i, j in picks)
    composed_result = (composed.compose({shards[i].shard_id: candidates[j].machine_ids for i, j in picks},
                                        weights) if composed is not None else None)
    quality = (sum(candidates[j].availability for _, j in picks) if composed_result is None else
               (composed_result.single_access if composed_objective == "single" else composed_result.full_access))
    return PlacementSolution(status, "", "sum_availability",
                             quality,
                             machine_usd_h + traffic_usd_h + migration_usd / transition.horizon_h,
                             budget, selected, solver_runtime_seconds=time.time() - t0,
                             mip_gap=model.MIPGap, best_bound=model.ObjBound,
                             shard_placements=tuple((shards[i].shard_id, candidates[j].machine_ids) for i, j in picks),
                             machine_cost_per_hour=machine_usd_h,
                             replication_cost_per_hour=traffic_usd_h,
                             migration_usd=migration_usd,
                             gb_moved=sum(arcs[i, j][3] for i, j in picks),
                             replica_moves=sum(arcs[i, j][1] for i, j in picks),
                             replica_removals=sum(arcs[i, j][2] for i, j in picks),
                             horizon_h=transition.horizon_h,
                             composed_objective=composed_objective if composed_result is not None else None,
                             single_access_availability=(composed_result.single_access if composed_result else None),
                             full_access_availability=(composed_result.full_access if composed_result else None),
                             global_access_availability=(composed_result.global_access if composed_result else None),
                             per_shard_access=(composed_result.per_shard if composed_result else None))
