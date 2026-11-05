"""Candidate ILP with unique machine billing, solved by Gurobi (docs/plans/ilp-fleet-sampling.md).

Same model as ``placement_optimizer._solve_unique_machine_cost`` (integer count per candidate, binary
activation per machine, each active machine billed once, capacity per machine, hourly budget, minimize
summed unavailability), plus what HiGHS lacked at 3-11k candidates: a warm start, a MIP gap, Gurobi's
search, and the disaggregated link ``x_c <= cap * y_m`` for every member m of c, which stops the LP
relaxation from billing a fraction of a machine for a candidate that uses it.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Sequence

from powder.placement_optimizer import PlacementCandidate, PlacementSolution, SelectedCandidate


def solve_unique_cost_gurobi(candidates: Sequence[PlacementCandidate], price: dict[str, float], *,
                             capacity: int, num_rsms: int, budget: float | None,
                             warm_start: Sequence[tuple[str, ...]] = (), time_limit: float = 120.0,
                             mip_gap: float = 1e-5, threads: int = 1, seed: int = 0) -> PlacementSolution:
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
    model.setObjective(gp.quicksum((1.0 - c.availability) * 1e6 * x[j] for j, c in enumerate(candidates)),
                       gp.GRB.MINIMIZE)
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
