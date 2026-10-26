"""Library scenarios and machine catalogues -> placement optimizer inputs.

    from powder.faults import FaultLibrary
    from powder.faults.ilp import scenario_machine_types
    types = scenario_machine_types(FaultLibrary.load(), "scenario/canonical-central-rho1", copies=10)

Each (machine class, region) pair of a scenario becomes one
``placement_optimizer.MachineType`` with ``type_id = "<machine>@<region>"``.

Catalogues (exec #1b §5): every catalogue machine is its own
``placement_optimizer.Machine`` (no shared types), scored by the Markov model
on its horizon-average exponential approximation. With ~1000 distinct machines
two things blow up -- type-count enumeration, and the heterogeneous Markov
state space (7 distinct nodes: ~7k states, ~10 s per candidate) -- so
:func:`binned_markov_scorer` scores candidates on *rate-binned*
representatives (log-spaced bins of failure rate, recovery time and data-loss
rate; machines in the same bin share a Markov class) with a memo keyed on the
multiset of bins, while cost stays exact. :func:`solve_catalogue_placement`
generates candidates over the real machines, solves the ILP, and rescores the
chosen placements with the exact machines.

The optimizer module is imported lazily because it depends on SciPy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .convert import planned_as_cause, scenario_node_configs
from .library import FaultLibrary


def scenario_machine_types(lib: FaultLibrary, scenario_id: str, *, copies: int = 1, capacity: int = 1,
                           engine: bool = False, level: str | None = None,
                           price_assignment: str | None = None):
    from powder.placement_optimizer import MachineType

    cfgs = scenario_node_configs(lib, scenario_id, engine=engine, level=level, price_assignment=price_assignment)
    return [MachineType(type_id=f"{m}@{r}", node_config=cfg, copies=copies, capacity=capacity)
            for (m, r), cfg in sorted(cfgs.items())]


# --------------------------------------------------------------------------- catalogues

def catalogue_machines(lib: FaultLibrary, cat: dict[str, Any], *, scenario_id: str = "scenario/canonical-central",
                       price: str = "rho0", capacity: int = 10, horizon_h: float | None = None):
    """One optimizer ``Machine`` per catalogue machine (exponential approximation, planned downtime folded in).

    ``price``: which on-demand assignment to use (``rho0`` / ``rho1``); spot machines use their pool price.
    ``capacity``: RSM replicas a machine can host.
    """
    from powder.placement_optimizer import Machine
    from . import catalogue as C

    s = lib.get(scenario_id, "scenario")
    planned = planned_as_cause(lib, s, s.get("level", "central"))
    h = horizon_h or cat["horizon_h"]
    out = []
    for m in cat["machines"]:
        p = m["price"].get("per_hour") if m["tier"] == "spot" else m["price"].get(price)
        out.append(Machine(m["id"], C.node_config(m, horizon_h=h, planned=planned, cost_per_hour=p or 0.0),
                           capacity=capacity))
    return out


def _qlog(x: float, per_decade: int) -> float:
    if x <= 0 or math.isinf(x):
        return x
    return 10 ** (round(math.log10(x) * per_decade) / per_decade)


@dataclass
class BinnedScorer:
    """Markov scorer on rate-binned representatives, memoised on the multiset of bins."""

    protocol: Any
    strategy: Any
    bins_per_decade: int = 4
    quality: Any = None
    memo: dict = field(default_factory=dict)
    bin_of: dict = field(default_factory=dict)
    rep: dict = field(default_factory=dict)
    calls: int = 0
    seconds: float = 0.0

    def key(self, machine) -> tuple:
        k = self.bin_of.get(machine.machine_id)
        if k is None:
            from powder.simulation.distributions import Constant, Exponential
            c = machine.node_config
            lam_t = 1.0 / c.failure_dist.mean if c.failure_dist.mean > 0 else math.inf
            lam_p = 1.0 / c.data_loss_dist.mean if c.data_loss_dist.mean > 0 else math.inf
            rec = c.recovery_dist.mean
            k = (_qlog(lam_t, self.bins_per_decade), _qlog(rec, self.bins_per_decade),
                 _qlog(lam_p, self.bins_per_decade), c.region)
            self.bin_of[machine.machine_id] = k
            if k not in self.rep:
                never = Constant(math.inf)
                self.rep[k] = replace(c, cost_per_hour=0.0,
                                      failure_dist=Exponential(k[0]) if 0 < k[0] < math.inf else never,
                                      recovery_dist=Constant(k[1]),
                                      data_loss_dist=Exponential(k[2]) if 0 < k[2] < math.inf else never)
        return k

    def __call__(self, machines: Sequence):
        from powder.placement_optimizer import CandidateScore
        from powder.results import markov_analyze
        from powder.scenario import QualityLevel

        keys = tuple(sorted((self.key(m) for m in machines), key=repr))
        cost = sum(m.node_config.cost_per_hour for m in machines)
        avail = self.memo.get(keys)
        if avail is None:
            t = time.time()
            res = markov_analyze([self.rep[k] for k in keys], self.protocol, self.strategy,
                                 self.quality if self.quality is not None else QualityLevel.SIMPLIFIED)
            self.seconds += time.time() - t
            self.calls += 1
            avail = float(res.availability)
            self.memo[keys] = avail
        return CandidateScore(availability=avail, cost_per_hour=cost,
                              metadata={"method": "markov-binned", "bins_per_decade": self.bins_per_decade})


def binned_markov_scorer(protocol, strategy, *, bins_per_decade: int = 4, quality=None) -> BinnedScorer:
    return BinnedScorer(protocol, strategy, bins_per_decade, quality)


def solve_catalogue_placement(lib: FaultLibrary, cat: dict[str, Any], protocol, strategy, *, num_rsms: int = 100,
                              replica_counts: tuple[int, ...] = (3, 5), capacity: int = 10, price: str = "rho0",
                              budget_per_hour: float | None = None, objective: str = "sum_availability",
                              bins_per_decade: int = 4, random_samples_per_size: int = 400,
                              greedy_starts_per_size: int = 10, max_candidates: int = 3000, rng_seed: int = 0,
                              rescore_exact: bool = True, scenario_id: str = "scenario/canonical-central") -> dict:
    """Candidate generation + ILP over a catalogue; returns the solution and timing/diagnostics."""
    from powder.placement_optimizer import (CandidateGenerationConfig, PlacementSolverConfig, generate_candidates,
                                            markov_candidate_scorer, solve_candidate_ilp)

    t0 = time.time()
    machines = catalogue_machines(lib, cat, scenario_id=scenario_id, price=price, capacity=capacity)
    scorer = binned_markov_scorer(protocol, strategy, bins_per_decade=bins_per_decade)
    cfg = CandidateGenerationConfig(replica_counts=replica_counts, random_samples_per_size=random_samples_per_size,
                                    greedy_starts_per_size=greedy_starts_per_size, local_search_steps=1,
                                    max_candidates=max_candidates, rng_seed=rng_seed)
    cands = generate_candidates(machines, scorer, cfg)
    t1 = time.time()
    sol = solve_candidate_ilp(cands, machines, PlacementSolverConfig(num_rsms=num_rsms, objective=objective,
                                                                    budget_per_hour=budget_per_hour))
    t2 = time.time()
    exact = {}
    if rescore_exact and sol.selected:
        by_id = {m.machine_id: m for m in machines}
        ex = markov_candidate_scorer(protocol, strategy)
        for item in sol.selected:
            ids = item.candidate.machine_ids
            if len(ids) <= 5:  # 7 distinct nodes take ~10 s each; keep the binned score there
                exact[item.candidate.candidate_id] = ex([by_id[i] for i in ids]).availability
    return {"solution": sol, "machines": machines, "candidates": cands, "exact_availability": exact,
            "binned_classes": len(scorer.rep), "markov_calls": scorer.calls, "markov_seconds": scorer.seconds,
            "seconds": {"candidates": t1 - t0, "ilp": t2 - t1, "rescore": time.time() - t2}}
