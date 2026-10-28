"""Library scenarios and machine catalogues -> placement optimizer inputs.

    from powder.faults import FaultLibrary
    from powder.faults.ilp import scenario_machine_types
    types = scenario_machine_types(FaultLibrary.load(), "scenario/canonical-central-rho1", copies=10)

Each (machine class, region) pair of a scenario becomes one
``placement_optimizer.MachineType`` with ``type_id = "<machine>@<region>"``.

Catalogues (exec #1b §5): every catalogue machine is its own
``placement_optimizer.Machine`` (no shared types). Candidate generation uses
horizon-average exponential rates and a memoised rate-binned Markov scorer;
seven distinct nodes otherwise take seconds per solve. By default,
:func:`solve_catalogue_placement` screens below 4.5 Markov nines, then admits
only placements whose full-horizon Rust Monte Carlo t interval certifies the
configured availability target. The ILP receives the MC mean and exact price.
The older Markov-only path remains available for fast exploratory work.

The optimizer module is imported lazily because it depends on SciPy.
"""

from __future__ import annotations

import hashlib
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


def mc_shortlist(candidates: Sequence, count: int) -> list:
    """Keep both cheap and reliable Markov survivors before expensive MC scoring."""
    if len(candidates) <= count:
        return list(candidates)
    cheap = sorted(candidates, key=lambda c: (c.cost_per_hour, -c.availability, c.candidate_id))
    reliable = sorted(candidates, key=lambda c: (-c.availability, c.cost_per_hour, c.candidate_id))
    chosen = {}
    for a, b in zip(cheap, reliable):
        for candidate in (a, b):
            chosen[candidate.candidate_id] = candidate
            if len(chosen) >= count:
                return [chosen[k] for k in sorted(chosen)]
    return [chosen[k] for k in sorted(chosen)]


def _mc_seed(base_seed: int, candidate_id: str, max_runs: int) -> int:
    digest = hashlib.blake2b(candidate_id.encode(), digest_size=8, person=b"powdermc").digest()
    return (base_seed + int.from_bytes(digest, "big")) % (2**63 - max_runs)


def solve_catalogue_placement(lib: FaultLibrary, cat: dict[str, Any], protocol, strategy, *, num_rsms: int = 100,
                              replica_counts: tuple[int, ...] = (3, 5, 7), capacity: int = 10, price: str = "rho0",
                              budget_per_hour: float | None = None, objective: str = "sum_availability",
                              bins_per_decade: int = 4, random_samples_per_size: int = 400,
                              greedy_starts_per_size: int = 10, max_candidates: int = 3000, rng_seed: int = 0,
                              rescore_exact: bool = True, scenario_id: str = "scenario/canonical-central",
                              unique_machine_cost: bool = True, scoring: str = "hybrid",
                              mc_config: Any = None) -> dict:
    """Markov screening, optional MC certification, then the candidate ILP.

    ``scoring='hybrid'`` first rejects Markov scores below 4.5 nines, then uses
    fixed-horizon Rust MC to certify candidate availability against the configured
    k-nines target. Only MC-certified candidates enter the ILP. ``'markov'`` keeps
    the older fast path, explicitly for screening/tests. A capped candidate
    shortlist is not a proof of global placement optimality.
    """
    from powder.placement_optimizer import (CandidateGenerationConfig, PlacementSolverConfig, PlacementSolution,
                                            generate_candidates, markov_candidate_scorer, solve_candidate_ilp)

    if scoring not in ("hybrid", "markov"):
        raise ValueError("scoring must be 'hybrid' or 'markov'")
    t0 = time.time()
    machines = catalogue_machines(lib, cat, scenario_id=scenario_id, price=price, capacity=capacity)
    scorer = binned_markov_scorer(protocol, strategy, bins_per_decade=bins_per_decade)
    cfg = CandidateGenerationConfig(replica_counts=replica_counts, random_samples_per_size=random_samples_per_size,
                                    greedy_starts_per_size=greedy_starts_per_size, local_search_steps=1,
                                    max_candidates=max_candidates, rng_seed=rng_seed,
                                    reject_region_majority=True,
                                    reject_shared_az=bool(cat.get("az_aware")))
    markov_candidates = generate_candidates(machines, scorer, cfg)
    t1 = time.time()
    cands = markov_candidates
    mc_diagnostics = {}
    if scoring == "hybrid":
        from .mc_scoring import MCScoringConfig, catalogue_batch_sampler, certify_availability

        mc = mc_config if mc_config is not None else MCScoringConfig()
        if not isinstance(mc, MCScoringConfig):
            raise TypeError("mc_config must be an MCScoringConfig")
        survivors = [c for c in cands if c.availability >= mc.markov_floor]
        shortlist = mc_shortlist(survivors, mc.max_candidates)
        mc_diagnostics = {"markov_rejected": len(cands) - len(survivors),
                          "not_shortlisted": len(survivors) - len(shortlist),
                          "target_nines": mc.target_nines, "confidence": mc.confidence,
                          "decisions": {}}
        cands = []
        for candidate in shortlist:
            sampler = catalogue_batch_sampler(lib, cat, candidate.machine_ids, scenario_id,
                                               price, protocol, strategy,
                                               include_region_outages=mc.include_region_outages)
            decision = certify_availability(
                sampler, mc, base_seed=_mc_seed(mc.base_seed, candidate.candidate_id, mc.max_runs),
                comparisons=len(shortlist)
            )
            mc_diagnostics["decisions"][candidate.candidate_id] = {
                "status": decision.status, "runs": decision.runs,
                "availability": decision.availability,
                "ci_low": decision.ci_low, "ci_high": decision.ci_high,
                "markov_availability": candidate.availability,
            }
            if decision.status == "pass":
                cands.append(replace(candidate, availability=decision.availability,
                                     metadata={**candidate.metadata, "method": "rust-mc-certified",
                                               "runs": decision.runs, "ci_low": decision.ci_low,
                                               "ci_high": decision.ci_high}))
    t_mc = time.time()
    if cands:
        sol = solve_candidate_ilp(cands, machines, PlacementSolverConfig(num_rsms=num_rsms, objective=objective,
                                                                        budget_per_hour=budget_per_hour,
                                                                        unique_machine_cost=unique_machine_cost))
    else:
        sol = PlacementSolution("no_mc_qualified_candidates", "No candidate certified the MC threshold",
                                objective, float("nan"), float("nan"), budget_per_hour, ())
    t2 = time.time()
    exact = {}
    if scoring == "markov" and rescore_exact and sol.selected:
        by_id = {m.machine_id: m for m in machines}
        ex = markov_candidate_scorer(protocol, strategy)
        for item in sol.selected:
            ids = item.candidate.machine_ids
            if len(ids) <= 5:  # 7 distinct nodes take ~10 s each; keep the binned score there
                exact[item.candidate.candidate_id] = ex([by_id[i] for i in ids]).availability
    return {"solution": sol, "machines": machines, "candidates": cands,
            "markov_candidates": markov_candidates, "mc_diagnostics": mc_diagnostics,
            "exact_availability": exact, "binned_classes": len(scorer.rep),
            "markov_calls": scorer.calls, "markov_seconds": scorer.seconds,
            "seconds": {"candidates": t1 - t0, "mc": t_mc - t1,
                        "ilp": t2 - t_mc, "rescore": time.time() - t2}}
