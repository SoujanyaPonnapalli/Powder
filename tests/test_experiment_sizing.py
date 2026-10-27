import os
import subprocess
import sys

from scripts.experiment_horizon import wilson
from scripts.experiment_sizing import baseline, nested_order

from powder.placement_optimizer import (
    Machine,
    PlacementCandidate,
    PlacementSolverConfig,
    solve_candidate_ilp,
)
from powder.simulation import Constant, NodeConfig, days


def machine(mid, price, capacity=2):
    return Machine(mid, NodeConfig(
        region="us-east-1", cost_per_hour=price,
        failure_dist=Constant(days(365)), recovery_dist=Constant(10),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(100),
        snapshot_download_time_dist=Constant(0), spawn_dist=Constant(0),
    ), capacity=capacity)


def test_unique_machine_price_not_multiplied_by_replica_groups():
    machines = [machine("a", 2), machine("b", 3), machine("c", 5)]
    candidates = [PlacementCandidate("abc", ("a", "b", "c"), .999, 10)]
    result = solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
        num_rsms=2, objective="min_cost", min_availability=.99,
        unique_machine_cost=True, max_machines=3, budget_per_hour=10,
    ))
    assert result.total_rsms == 2
    assert result.total_cost_per_hour == 10
    assert result.objective_value == 10


def test_unique_machine_solver_matches_brute_force():
    from itertools import combinations_with_replacement
    from collections import Counter

    machines = [machine(f"m{i}", i + 1, 2) for i in range(5)]
    candidates = [PlacementCandidate(f"c{i}", ids, .999 + i * .0001, 0)
                  for i, ids in enumerate((("m0", "m1", "m2"), ("m1", "m2", "m3"),
                                           ("m0", "m3", "m4"), ("m2", "m3", "m4")))]
    target = .99915
    brute = []
    for selected in combinations_with_replacement(candidates, 2):
        uses = Counter(mid for c in selected for mid in c.machine_ids)
        if (max(uses.values()) <= 2 and len(uses) <= 4 and
                sum(c.availability for c in selected) / 2 >= target):
            brute.append(sum(m.node_config.cost_per_hour for m in machines if m.machine_id in uses))
    result = solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
        num_rsms=2, objective="min_cost", unique_machine_cost=True,
        min_mean_availability=target, max_machines=4,
    ))
    assert result.total_cost_per_hour == min(brute)


def test_availability_objective_resolves_sixth_decimal():
    machines = [machine(f"m{i}", 1, 1) for i in range(6)]
    candidates = [PlacementCandidate("worse", ("m0", "m1", "m2"), .999990, 3),
                  PlacementCandidate("better", ("m3", "m4", "m5"), .999991, 3)]
    result = solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
        num_rsms=1, unique_machine_cost=True, budget_per_hour=3,
    ))
    assert result.selected[0].candidate.candidate_id == "better"


def test_cost_minimization_enforces_mean_availability():
    machines = [machine("a", 1, 1), machine("b", 1, 1), machine("c", 1, 1),
                machine("d", 2, 1), machine("e", 2, 1), machine("f", 2, 1)]
    candidates = [PlacementCandidate("cheap", ("a", "b", "c"), .9, 3),
                  PlacementCandidate("reliable", ("d", "e", "f"), .999, 6)]
    result = solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
        num_rsms=2, objective="min_cost", min_mean_availability=.9495,
        unique_machine_cost=True, max_machines=6,
    ))
    assert result.total_rsms == 2
    assert {item.candidate.candidate_id for item in result.selected} == {"cheap", "reliable"}
    assert result.total_cost_per_hour == 9


def test_unique_machine_price_floor_and_limit():
    machines = [machine("a", 1, 1), machine("b", 1, 1), machine("c", 1, 1),
                machine("d", 4, 1), machine("e", 4, 1), machine("f", 4, 1)]
    candidates = [PlacementCandidate("cheap", ("a", "b", "c"), .9, 3),
                  PlacementCandidate("reliable", ("d", "e", "f"), .999, 12)]
    cfg = PlacementSolverConfig(num_rsms=1, objective="min_cost", min_availability=.99,
                                unique_machine_cost=True, max_machines=3)
    result = solve_candidate_ilp(candidates, machines, cfg)
    assert result.selected[0].candidate.candidate_id == "reliable"
    assert result.total_cost_per_hour == 12
    limited = solve_candidate_ilp(candidates, machines, PlacementSolverConfig(
        num_rsms=2, objective="sum_availability", unique_machine_cost=True, max_machines=3,
    ))
    assert limited.selected == ()


def test_nested_inventory_keeps_spot_fraction_and_identity():
    machines = ([{"id": f"od{i}", "tier": "ondemand"} for i in range(7)] +
                [{"id": f"sp{i}", "tier": "spot"} for i in range(3)])
    ordered = nested_order(machines, .3)
    assert {m["id"] for m in ordered} == {m["id"] for m in machines}
    assert sum(m["tier"] == "spot" for m in ordered[:5]) in (1, 2)
    assert sum(m["tier"] == "spot" for m in ordered) == 3


def test_baseline_counts_each_node_once_and_spreads_regions():
    from powder.placement_optimizer import CandidateScore
    from dataclasses import replace

    machines = [Machine(f"{r}-{i}", replace(machine(f"{r}-{i}", 1).node_config, region=r), capacity=3)
                for r in ("a", "b", "c") for i in range(3)]
    scored = baseline(machines, 4, 3, 3, cheap=True, seed=0,
                      scorer=lambda ms: CandidateScore(.99, sum(m.node_config.cost_per_hour for m in ms)))
    assert scored["cost_per_hour"] == scored["deployed"]
    assert all(len({mid.split("-")[0] for mid in placement}) == 3 for placement in scored["placements"])
    assert wilson(0, 100)[1] > 0


def test_price_blind_baseline_packs_region_capacity():
    from dataclasses import replace
    from powder.placement_optimizer import CandidateScore

    machines = [Machine(f"{r}-{i}", replace(machine(f"{r}-{i}", 1).node_config, region=r), capacity=5)
                for r in ("a", "b", "c", "d") for i in range(12)]
    result = baseline(machines, 50, 3, 5, cheap=True, seed=0,
                      scorer=lambda ms: CandidateScore(.999, 3))
    assert result["deployed"] == 30
    assert len(result["placements"]) == 50
    assert all(len({mid.split("-")[0] for mid in ids}) == 3 for ids in result["placements"])


def test_candidate_search_is_independent_of_python_hash_seed():
    code = """
from dataclasses import replace
from powder.placement_optimizer import Machine, CandidateScore, CandidateGenerationConfig, generate_candidates
from powder.simulation import Constant, NodeConfig
cfg = NodeConfig(region='a', cost_per_hour=1, failure_dist=Constant(100000),
                 recovery_dist=Constant(10), data_loss_dist=Constant(10000000),
                 log_replay_rate_dist=Constant(100), snapshot_download_time_dist=Constant(0),
                 spawn_dist=Constant(0))
ms = [Machine(f'm{i}', replace(cfg, region='abcd'[i % 4], cost_per_hour=1 + i / 20))
      for i in range(20)]
scorer = lambda nodes: CandidateScore(.999, sum(m.node_config.cost_per_hour for m in nodes))
candidates = generate_candidates(ms, scorer, CandidateGenerationConfig(
    replica_counts=(3,), random_samples_per_size=30, greedy_starts_per_size=2,
    local_search_steps=2, rng_seed=7))
print([c.machine_ids for c in candidates])
"""
    def result(seed):
        return subprocess.check_output([sys.executable, "-c", code],
                                       env={**os.environ, "PYTHONHASHSEED": str(seed)})
    assert result(1) == result(9)


def test_unique_machine_budget_charges_once():
    machines = [machine("a", 2), machine("b", 3), machine("c", 5)]
    result = solve_candidate_ilp(
        [PlacementCandidate("abc", ("a", "b", "c"), .999, 10)], machines,
        PlacementSolverConfig(num_rsms=2, unique_machine_cost=True, budget_per_hour=10),
    )
    assert result.total_rsms == 2
    assert result.total_cost_per_hour == 10
