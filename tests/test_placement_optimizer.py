import pytest

from powder.placement_optimizer import (
    CandidateGenerationConfig,
    CandidateScore,
    Machine,
    MachineType,
    PlacementCandidate,
    PlacementSolverConfig,
    TypeCountCandidate,
    blind_balanced_type_count_placement,
    blind_cheap_type_count_placement,
    blind_large_type_count_placement,
    blind_random_type_count_placement,
    blind_reliable_type_count_placement,
    blind_value_type_count_placement,
    BlindPlacementConfig,
    generate_type_count_candidates,
    generate_candidates,
    rescore_type_count_solution,
    solve_blind_type_count_placement,
    solve_candidate_ilp,
    solve_candidate_ilp_gurobi,
    solve_type_count_ilp,
)
from powder.simulation import Constant, NodeConfig, days


def _node_config(region: str, cost_per_hour: float = 1.0) -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=cost_per_hour,
        failure_dist=Constant(days(365)),
        recovery_dist=Constant(days(1)),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(100.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


def _node_config_with_reliability(
    region: str,
    cost_per_hour: float,
    mtbf_days: float,
    mttr_days: float,
) -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=cost_per_hour,
        failure_dist=Constant(days(mtbf_days)),
        recovery_dist=Constant(days(mttr_days)),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(100.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


def _machines(count: int) -> list[Machine]:
    regions = ["us-east", "us-west", "eu", "ap"]
    return [
        Machine(
            machine_id=f"m{i}",
            node_config=_node_config(regions[i % len(regions)]),
        )
        for i in range(count)
    ]


def test_generate_candidates_scores_unique_valid_placements():
    machines = _machines(12)

    def scorer(placement):
        return CandidateScore(
            availability=0.9 + len(placement) / 100.0,
            cost_per_hour=sum(m.node_config.cost_per_hour for m in placement),
        )

    candidates = generate_candidates(
        machines,
        scorer,
        CandidateGenerationConfig(
            replica_counts=(3, 5),
            random_samples_per_size=8,
            greedy_starts_per_size=2,
            local_search_steps=1,
            max_candidates=20,
            rng_seed=4,
        ),
    )

    assert candidates
    assert len({c.candidate_id for c in candidates}) == len(candidates)
    assert {c.replica_count for c in candidates} <= {3, 5}
    assert all(len(set(c.machine_ids)) == c.replica_count for c in candidates)
    assert all(0.0 <= c.availability <= 1.0 for c in candidates)


def test_candidate_ilp_respects_machine_capacity():
    machines = [
        Machine(f"m{i}", _node_config("region"), capacity=1)
        for i in range(6)
    ]
    candidates = [
        PlacementCandidate("a", ("m0", "m1", "m2"), 0.99, 3.0),
        PlacementCandidate("b", ("m3", "m4", "m5"), 0.98, 3.0),
        PlacementCandidate("overlap", ("m0", "m1", "m3"), 0.999, 3.0),
    ]

    solution = solve_candidate_ilp(
        candidates,
        machines,
        PlacementSolverConfig(num_rsms=2, objective="sum_availability"),
    )

    assert solution.total_rsms == 2
    assert {item.candidate.candidate_id for item in solution.selected} == {"a", "b"}


def test_candidate_ilp_budget_constraint_changes_choice():
    machines = [
        Machine(f"m{i}", _node_config("region"), capacity=1)
        for i in range(6)
    ]
    candidates = [
        PlacementCandidate("expensive", ("m0", "m1", "m2"), 0.999, 30.0),
        PlacementCandidate("cheap", ("m3", "m4", "m5"), 0.95, 3.0),
    ]

    solution = solve_candidate_ilp(
        candidates,
        machines,
        PlacementSolverConfig(
            num_rsms=1,
            objective="sum_availability",
            budget_per_hour=3.0,
        ),
    )

    assert solution.total_rsms == 1
    assert solution.budget_per_hour == 3.0
    assert solution.selected[0].candidate.candidate_id == "cheap"


def test_candidate_ilp_reports_infeasible_problem():
    machines = [
        Machine(f"m{i}", _node_config("region"), capacity=1)
        for i in range(3)
    ]
    candidates = [
        PlacementCandidate("only", ("m0", "m1", "m2"), 0.99, 3.0),
    ]

    solution = solve_candidate_ilp(
        candidates,
        machines,
        PlacementSolverConfig(num_rsms=2, objective="sum_availability"),
    )

    assert solution.selected == ()
    assert solution.status != "0"


def test_gurobi_candidate_ilp_maximizes_product_availability():
    pytest.importorskip("gurobipy")
    machines = [
        Machine(f"m{i}", _node_config("region"), capacity=1)
        for i in range(9)
    ]
    candidates = [
        PlacementCandidate("best", ("m0", "m1", "m2"), 0.999, 8.0),
        PlacementCandidate("second", ("m3", "m4", "m5"), 0.99, 7.0),
        PlacementCandidate("cheap", ("m6", "m7", "m8"), 0.90, 2.0),
    ]

    solution = solve_candidate_ilp_gurobi(
        candidates,
        machines,
        PlacementSolverConfig(
            num_rsms=2,
            objective="product_availability",
            budget_per_hour=15.0,
        ),
        threads=1,
    )

    assert solution.status == "optimal"
    assert solution.total_rsms == 2
    assert {item.candidate.candidate_id for item in solution.selected} == {
        "best",
        "second",
    }
    assert solution.total_cost_per_hour == pytest.approx(15.0)
    assert solution.solver_runtime_seconds is not None
    assert solution.mip_gap == pytest.approx(0.0)


def test_gurobi_candidate_ilp_reports_budget_infeasibility():
    pytest.importorskip("gurobipy")
    machines = [
        Machine(f"m{i}", _node_config("region"), capacity=1)
        for i in range(6)
    ]
    candidates = [
        PlacementCandidate("a", ("m0", "m1", "m2"), 0.99, 5.0),
        PlacementCandidate("b", ("m3", "m4", "m5"), 0.98, 5.0),
    ]

    solution = solve_candidate_ilp_gurobi(
        candidates,
        machines,
        PlacementSolverConfig(
            num_rsms=2,
            objective="product_availability",
            budget_per_hour=9.0,
        ),
        threads=1,
    )

    assert solution.status == "infeasible"
    assert solution.selected == ()
    assert solution.total_rsms == 0
    assert "solutions=0" in solution.message


def _machine_types() -> list[MachineType]:
    return [
        MachineType("fast", _node_config("us-east", cost_per_hour=3.0), copies=2),
        MachineType("steady", _node_config("us-west", cost_per_hour=1.0), copies=3),
        MachineType("cheap", _node_config("eu", cost_per_hour=0.5), copies=4),
    ]


def test_generate_type_count_candidates_enumerates_bounded_multisets():
    machine_types = _machine_types()

    def scorer(types, counts):
        return CandidateScore(
            availability=0.9 + counts[0] * 0.01,
            cost_per_hour=sum(t.node_config.cost_per_hour * c for t, c in zip(types, counts)),
        )

    candidates = generate_type_count_candidates(
        machine_types,
        scorer,
        replica_counts=(3,),
    )

    assert candidates
    assert all(candidate.replica_count == 3 for candidate in candidates)
    assert all(candidate.type_ids == ("fast", "steady", "cheap") for candidate in candidates)
    assert all(candidate.counts[0] <= 2 for candidate in candidates)
    assert len({candidate.counts for candidate in candidates}) == len(candidates)


def test_type_count_ilp_respects_total_type_capacity():
    machine_types = [
        MachineType("fast", _node_config("us-east"), copies=1, capacity=1),
        MachineType("slow", _node_config("us-west"), copies=5, capacity=1),
    ]
    type_ids = tuple(t.type_id for t in machine_types)
    candidates = [
        TypeCountCandidate("fast", type_ids, (1, 2), 0.999, 3.0),
        TypeCountCandidate("slow", type_ids, (0, 3), 0.95, 1.5),
    ]

    solution = solve_type_count_ilp(
        candidates,
        machine_types,
        PlacementSolverConfig(num_rsms=2, objective="sum_availability"),
    )

    selected_counts = {
        item.candidate.candidate_id: item.count
        for item in solution.selected
    }
    assert solution.total_rsms == 2
    assert selected_counts == {"fast": 1, "slow": 1}


def test_type_count_ilp_budget_constraint_changes_choice():
    machine_types = [
        MachineType("fast", _node_config("us-east"), copies=3, capacity=1),
        MachineType("slow", _node_config("us-west"), copies=3, capacity=1),
    ]
    type_ids = tuple(t.type_id for t in machine_types)
    candidates = [
        TypeCountCandidate("expensive", type_ids, (3, 0), 0.999, 30.0),
        TypeCountCandidate("cheap", type_ids, (0, 3), 0.95, 3.0),
    ]

    solution = solve_type_count_ilp(
        candidates,
        machine_types,
        PlacementSolverConfig(
            num_rsms=1,
            objective="sum_availability",
            budget_per_hour=3.0,
        ),
    )

    assert solution.total_rsms == 1
    assert solution.budget_per_hour == 3.0
    assert solution.selected[0].candidate.candidate_id == "cheap"


def test_type_count_ilp_rejects_candidate_exceeding_type_copies():
    machine_types = [
        MachineType("fast", _node_config("us-east"), copies=1, capacity=10),
        MachineType("slow", _node_config("us-west"), copies=3, capacity=1),
    ]
    type_ids = tuple(t.type_id for t in machine_types)
    candidates = [
        TypeCountCandidate("duplicate-fast", type_ids, (2, 1), 0.999, 3.0),
    ]

    with pytest.raises(ValueError, match="only 1 physical copies"):
        solve_type_count_ilp(
            candidates,
            machine_types,
            PlacementSolverConfig(num_rsms=1),
        )


def _blind_machine_types() -> list[MachineType]:
    return [
        MachineType(
            "cheap",
            _node_config_with_reliability("us-east", cost_per_hour=1.0, mtbf_days=80, mttr_days=4),
            copies=30,
            capacity=1,
        ),
        MachineType(
            "balanced",
            _node_config_with_reliability("us-west", cost_per_hour=2.0, mtbf_days=250, mttr_days=2),
            copies=30,
            capacity=1,
        ),
        MachineType(
            "reliable",
            _node_config_with_reliability("eu", cost_per_hour=5.0, mtbf_days=1000, mttr_days=1),
            copies=30,
            capacity=1,
        ),
    ]


def _used_type_counts(solution):
    used = [0] * len(_blind_machine_types())
    for item in solution.selected:
        for idx, count in enumerate(item.candidate.counts):
            used[idx] += count * item.count
    return used


def _assert_blind_solution_feasible(solution, machine_types, num_rsms, budget=None):
    assert solution.status == "success"
    assert solution.total_rsms == num_rsms
    used = [0] * len(machine_types)
    for item in solution.selected:
        assert item.candidate.replica_count in {3, 5, 7}
        assert all(
            count <= machine_type.copies
            for count, machine_type in zip(item.candidate.counts, machine_types)
        )
        for idx, count in enumerate(item.candidate.counts):
            used[idx] += count * item.count
    assert all(
        count <= machine_type.total_capacity
        for count, machine_type in zip(used, machine_types)
    )
    if budget is not None:
        assert solution.total_simple_cost_per_hour <= budget


def test_blind_random_type_count_placement_finds_feasible_solution():
    machine_types = _blind_machine_types()
    solution = blind_random_type_count_placement(
        machine_types,
        BlindPlacementConfig(num_rsms=5, rng_seed=7),
    )

    _assert_blind_solution_feasible(solution, machine_types, num_rsms=5)


def test_blind_cheap_type_count_placement_prefers_cheapest_type():
    machine_types = _blind_machine_types()
    solution = blind_cheap_type_count_placement(
        machine_types,
        BlindPlacementConfig(num_rsms=4),
    )

    _assert_blind_solution_feasible(solution, machine_types, num_rsms=4)
    used = _used_type_counts(solution)
    assert used[0] == 12
    assert used[1:] == [0, 0]


def test_blind_reliable_type_count_placement_repairs_to_budget():
    machine_types = _blind_machine_types()
    solution = blind_reliable_type_count_placement(
        machine_types,
        BlindPlacementConfig(num_rsms=4, budget_per_hour=24.0),
    )

    _assert_blind_solution_feasible(
        solution,
        machine_types,
        num_rsms=4,
        budget=24.0,
    )
    assert solution.total_simple_cost_per_hour < 4 * 3 * 5.0


def test_blind_large_type_count_placement_uses_large_rsms_without_budget():
    machine_types = _blind_machine_types()
    solution = blind_large_type_count_placement(
        machine_types,
        BlindPlacementConfig(num_rsms=3),
    )

    _assert_blind_solution_feasible(solution, machine_types, num_rsms=3)
    assert all(item.candidate.replica_count == 7 for item in solution.selected)


def test_blind_large_type_count_placement_reserves_minimum_capacity():
    machine_types = [
        MachineType("fast", _node_config("us-east", cost_per_hour=3.0), copies=4),
        MachineType("steady", _node_config("us-west", cost_per_hour=2.0), copies=4),
        MachineType("cheap", _node_config("eu", cost_per_hour=1.0), copies=4),
    ]

    solution = blind_large_type_count_placement(
        machine_types,
        BlindPlacementConfig(num_rsms=3, replica_counts=(3, 5, 7)),
    )

    _assert_blind_solution_feasible(solution, machine_types, num_rsms=3)
    assert sum(item.candidate.replica_count * item.count for item in solution.selected) > 9


@pytest.mark.parametrize(
    "placement_fn,strategy",
    [
        (blind_balanced_type_count_placement, "balanced"),
        (blind_value_type_count_placement, "value"),
    ],
)
def test_extra_blind_type_count_placements_are_feasible(placement_fn, strategy):
    machine_types = _blind_machine_types()
    solution = placement_fn(
        machine_types,
        BlindPlacementConfig(num_rsms=4, budget_per_hour=36.0),
    )

    assert solution.strategy == strategy
    _assert_blind_solution_feasible(
        solution,
        machine_types,
        num_rsms=4,
        budget=36.0,
    )


def test_rescore_type_count_solution_uses_oracle_scores():
    machine_types = _blind_machine_types()
    blind = blind_cheap_type_count_placement(
        machine_types,
        BlindPlacementConfig(num_rsms=2),
    )

    def scorer(types, counts):
        del types
        return CandidateScore(
            availability=0.8 + 0.01 * counts[0],
            cost_per_hour=42.0,
        )

    rescored = rescore_type_count_solution(
        blind,
        machine_types,
        scorer,
        objective="sum_availability",
    )

    assert rescored.status == "success"
    assert rescored.total_rsms == 2
    assert rescored.budget_per_hour == blind.budget_per_hour
    assert rescored.total_cost_per_hour == 84.0
    assert all(item.candidate.cost_per_hour == 42.0 for item in rescored.selected)


def test_solve_blind_type_count_placement_rescores_after_blind_search():
    machine_types = _blind_machine_types()

    def scorer(types, counts):
        simple_cost = sum(
            machine_type.node_config.cost_per_hour * count
            for machine_type, count in zip(types, counts)
        )
        return CandidateScore(
            availability=0.7 + 0.01 * sum(counts),
            cost_per_hour=simple_cost + 20.0,
            metadata={"oracle": "test"},
        )

    solution = solve_blind_type_count_placement(
        machine_types,
        "cheap",
        scorer,
        BlindPlacementConfig(num_rsms=2, budget_per_hour=6.0),
        objective="sum_availability",
    )

    assert solution.status == "success"
    assert solution.budget_per_hour == 6.0
    assert solution.total_cost_per_hour == 46.0
    assert solution.total_cost_per_hour > solution.budget_per_hour
    assert solution.selected[0].candidate.availability == pytest.approx(0.73)
    assert solution.selected[0].candidate.metadata["blind_strategy"] == "cheap"
    assert solution.selected[0].candidate.metadata["prescore_cost_per_hour"] == 3.0
    assert solution.selected[0].candidate.metadata["oracle"] == "test"
