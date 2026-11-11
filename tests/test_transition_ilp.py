"""Named-shard unique-machine MILP, including running and moving costs."""

import pytest

gp = pytest.importorskip("gurobipy")

from powder.faults.gurobi_ilp import TransitionProblem, solve_transition_gurobi
from powder.faults.transition_costs import ShardSpec
from powder.placement_optimizer import PlacementCandidate


def problem(shards, loc, *, horizon=10, cap=None, cross=0.02):
    def traffic(src, dst, same_az):
        return 0 if same_az else (0.01 if src == dst else cross)

    def migration(src, dst):
        return cross if src != dst else 0

    return TransitionProblem(tuple(shards), loc, traffic, migration, horizon, cap)


LOC = {"a": ("r1", "z1"), "b": ("r1", "z1"), "c": ("r1", "z1"),
       "d": ("r2", "z2")}
OLD = PlacementCandidate("old", ("a", "b", "c"), 0.9, 3)
NEW = PlacementCandidate("new", ("a", "b", "d"), 0.99, 3)


def test_cap_and_hourly_budget_include_migration_without_new_machine_premium():
    shard = ShardSpec("s", 0.0, 100, OLD.machine_ids)
    prices = {m: 0.1 for m in LOC}
    stay = solve_transition_gurobi([OLD, NEW], prices, capacity=1, budget=None,
                                   transition=problem([shard], LOC, cap=0))
    assert stay.shard_placements == (("s", OLD.machine_ids),)
    assert stay.replica_moves == 0 and stay.migration_usd == 0
    assert stay.total_cost_per_hour == pytest.approx(0.3)

    # The new fleet costs the same $/h, but its $2 copy is $0.20/h over 10h.
    capped = solve_transition_gurobi([OLD, NEW], prices, capacity=1, budget=0.4,
                                     transition=problem([shard], LOC, cap=1))
    assert capped.shard_placements == (("s", OLD.machine_ids),)
    switched = solve_transition_gurobi([OLD, NEW], prices, capacity=1, budget=0.5,
                                       transition=problem([shard], LOC, cap=1))
    assert switched.shard_placements == (("s", NEW.machine_ids),)
    assert switched.machine_cost_per_hour == pytest.approx(0.3)
    assert switched.migration_usd == pytest.approx(2.0)
    assert switched.gb_moved == 100 and switched.replica_moves == 1
    assert switched.total_cost_per_hour == pytest.approx(0.5)


def test_minimum_cost_switches_only_when_horizon_pays_for_copy():
    shard = ShardSpec("s", 0, 100, OLD.machine_ids)
    equally_reliable = [PlacementCandidate("old", OLD.machine_ids, 0.99, 3),
                        PlacementCandidate("new", NEW.machine_ids, 0.99, 2.1)]
    prices = {"a": 1, "b": 1, "c": 1, "d": 0.1}
    def solve(h):
        return solve_transition_gurobi(equally_reliable, prices, capacity=1, budget=None,
                                       transition=problem([shard], LOC, horizon=h, cap=1),
                                       minimize_cost=True, max_unavailability_ppm=10_000)
    assert solve(1).shard_placements[0][1] == OLD.machine_ids
    assert solve(10).shard_placements[0][1] == NEW.machine_ids


def test_shard_specific_traffic_assignment_respects_machine_capacity():
    loc = {"a": ("r1", "z1"), "b": ("r1", "z1"), "c": ("r2", "z2"),
           "d": ("r1", "z1")}
    candidates = [PlacementCandidate("expensive-traffic", ("a", "b", "c"), 0.99, 1.5),
                  PlacementCandidate("free-traffic", ("a", "b", "d"), 0.99, 1.5)]
    shards = [ShardSpec("hot", 0.01, 100, candidates[0].machine_ids),
              ShardSpec("cold", 0, 100, candidates[0].machine_ids)]
    result = solve_transition_gurobi(candidates, {m: 0.5 for m in loc}, capacity=2, budget=None,
                                     transition=problem(shards, loc, cap=1), minimize_cost=True,
                                     max_unavailability_ppm=20_000)
    assert dict(result.shard_placements) == {"hot": candidates[1].machine_ids,
                                             "cold": candidates[0].machine_ids}
    assert result.machine_cost_per_hour == pytest.approx(2.0)
    assert result.replication_cost_per_hour == 0


def test_change_cap_counts_assignments_not_newly_activated_fleet_machines():
    shards = [ShardSpec("s0", 0, 100, OLD.machine_ids), ShardSpec("s1", 0, 100, OLD.machine_ids)]
    # d is already used by one shard in the new deployment; switching the other
    # still counts as a second new shard-to-machine assignment.
    result = solve_transition_gurobi([OLD, NEW], {m: 0.1 for m in LOC}, capacity=2,
                                     budget=None, transition=problem(shards, LOC, cap=1))
    assert result.replica_moves == 1
    assert sum(ids == NEW.machine_ids for _, ids in result.shard_placements) == 1
    both = solve_transition_gurobi([OLD, NEW], {m: 0.1 for m in LOC}, capacity=2,
                                   budget=None, transition=problem(shards, LOC, cap=2))
    assert both.replica_moves == 2
    assert sum(ids == NEW.machine_ids for _, ids in both.shard_placements) == 2


def test_invalid_transition_inputs_and_infeasible_budget():
    shard = ShardSpec("s", 0, 100, OLD.machine_ids)
    prices = {m: 0.1 for m in LOC}
    with pytest.raises(ValueError, match="movement cap requires old"):
        solve_transition_gurobi([OLD], prices, capacity=1, budget=None,
                                transition=problem([ShardSpec("s", 0, 100, ())], LOC, cap=0))
    with pytest.raises(ValueError, match="horizon_h"):
        solve_transition_gurobi([OLD], prices, capacity=1, budget=None,
                                transition=problem([shard], LOC, horizon=0))
    result = solve_transition_gurobi([OLD, NEW], prices, capacity=1, budget=0.1,
                                     transition=problem([shard], LOC, cap=0))
    assert result.status == "infeasible" and not result.selected
