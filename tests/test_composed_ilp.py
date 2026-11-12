"""Scenario-coupled full access differs from additive single-shard access."""

import pytest

pytest.importorskip("gurobipy")

from powder.faults.composed_availability import CandidateTimeline
from powder.faults.gurobi_ilp import TransitionProblem, solve_transition_gurobi
from powder.faults.transition_costs import ShardSpec
from powder.placement_optimizer import PlacementCandidate


A, B = ("a",), ("b",)
CANDIDATES = [PlacementCandidate("a", A, 0.99, 0.1), PlacementCandidate("b", B, 0.99, 0.1)]
SHARDS = (ShardSpec("s0", 0.0, 0.0, ()), ShardSpec("s1", 0.0, 0.0, ()))
PROBLEM = TransitionProblem(SHARDS, {"a": ("r1", "z1"), "b": ("r2", "z2")},
                            lambda src, dst, same: 0.0, lambda src, dst: 0.0, 10)


def timeline():
    return CandidateTimeline((1., 1., 1., 1., 1.), (True, True, True, True, False),
                             {"s0": {A: (True, True, True, False, True),
                                     B: (True, True, False, False, True)},
                              "s1": {A: (True, True, False, False, True),
                                     B: (False, False, True, True, True)}})


def solve(goal, **kwargs):
    return solve_transition_gurobi(CANDIDATES, {"a": 0.1, "b": 0.1},
                                   capacity=1, budget=0.2, transition=PROBLEM,
                                   composed=timeline(), composed_objective=goal,
                                   **kwargs)


def test_single_and_full_objectives_choose_different_correlated_placements():
    single = solve("single")
    full = solve("full")
    assert dict(single.shard_placements) == {"s0": A, "s1": B}
    assert single.single_access_availability == pytest.approx(0.5)
    assert single.full_access_availability == pytest.approx(0.2)
    assert single.objective_value == pytest.approx(0.5)
    assert dict(full.shard_placements) == {"s0": B, "s1": A}
    assert full.single_access_availability == pytest.approx(0.4)
    assert full.full_access_availability == pytest.approx(0.4)
    assert full.objective_value == pytest.approx(0.4)
    assert full.global_access_availability == pytest.approx(0.8)
    assert full.total_cost_per_hour == pytest.approx(0.2)
    assert full.status == "optimal"


def test_quality_floor_cost_minimization_uses_composed_metric():
    full = solve("full", minimize_cost=True, min_full_access=0.4)
    assert dict(full.shard_placements) == {"s0": B, "s1": A}
    single = solve("single", minimize_cost=True, min_single_access=0.5)
    assert dict(single.shard_placements) == {"s0": A, "s1": B}
    infeasible = solve("full", min_full_access=0.5)
    assert infeasible.status == "infeasible" and not infeasible.selected


def test_full_objective_respects_existing_replica_move_cap():
    old = TransitionProblem((ShardSpec("s0", 0, 10, A), ShardSpec("s1", 0, 10, B)),
                            PROBLEM.locations, PROBLEM.traffic_price, PROBLEM.region_price, 10,
                            max_replica_moves=1)
    stayed = solve_transition_gurobi(CANDIDATES, {"a": 0.1, "b": 0.1},
                                     capacity=1, budget=0.2, transition=old,
                                     composed=timeline(), composed_objective="full")
    assert dict(stayed.shard_placements) == {"s0": A, "s1": B}
    assert stayed.replica_moves == 0 and stayed.full_access_availability == pytest.approx(0.2)
    old = TransitionProblem(old.shards, old.locations, old.traffic_price, old.region_price,
                            old.horizon_h, max_replica_moves=2)
    swapped = solve_transition_gurobi(CANDIDATES, {"a": 0.1, "b": 0.1},
                                      capacity=1, budget=0.2, transition=old,
                                      composed=timeline(), composed_objective="full")
    assert dict(swapped.shard_placements) == {"s0": B, "s1": A}
    assert swapped.replica_moves == 2 and swapped.full_access_availability == pytest.approx(0.4)


def test_pd_outage_shared_once_and_bad_masks_rejected():
    all_ok = CandidateTimeline((5., 5.), (True, False),
                               {"s0": {A: (True, True), B: (True, True)},
                                "s1": {A: (True, True), B: (True, True)}})
    result = solve_transition_gurobi(CANDIDATES, {"a": 0.1, "b": 0.1}, capacity=1, budget=0.2,
                                     transition=PROBLEM, composed=all_ok, composed_objective="full")
    assert result.single_access_availability == result.full_access_availability == 0.5
    with pytest.raises(ValueError, match="every and only"):
        solve_transition_gurobi(CANDIDATES, {"a": 0.1, "b": 0.1}, capacity=1, budget=0.2,
                                transition=PROBLEM,
                                composed=CandidateTimeline((1.,), (True,), {"s0": {A: (True,)}}))
    with pytest.raises(ValueError, match="require a shared"):
        solve_transition_gurobi(CANDIDATES, {"a": 0.1, "b": 0.1}, capacity=1, budget=0.2,
                                transition=PROBLEM, composed_objective="full")
