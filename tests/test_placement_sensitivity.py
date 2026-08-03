import math

import pytest

from powder.placement_optimizer import (
    MachineType,
    PlacementSolverConfig,
    SelectedTypeCountCandidate,
    TypeCountCandidate,
    TypeCountPlacementSolution,
    solve_type_count_min_cost_with_availability_floors,
)
from powder.placement_sensitivity import (
    RateParameter,
    TypeCountMarkovScoreCache,
    perturb_machine_types,
    placement_derivatives,
)
from powder.scenario import QualityLevel
from powder.simulation import (
    Constant,
    Exponential,
    LeaderlessProtocol,
    NodeConfig,
    NodeReplacementStrategy,
    NoOpStrategy,
    days,
    hours,
    minutes,
)


def _config(*, failure_rate=1 / days(30), recovery_rate=1 / hours(1), data_loss_rate=1 / days(3650)):
    return NodeConfig(
        region="test",
        cost_per_hour=1.0,
        failure_dist=Exponential(failure_rate),
        recovery_dist=Exponential(recovery_rate),
        data_loss_dist=Exponential(data_loss_rate),
        log_replay_rate_dist=Constant(1_000_000.0),
        snapshot_download_time_dist=Constant(minutes(1)),
        spawn_dist=Exponential(1 / minutes(1)),
    )


def _types():
    return (
        MachineType("low", _config(), copies=10),
        MachineType("high", _config(failure_rate=1 / days(60)), copies=10),
    )


def test_perturbation_changes_exactly_one_requested_rate():
    original = _types()
    changed = perturb_machine_types(
        original, RateParameter("low", "transient_failure"), 0.15
    )

    assert changed[0].node_config.failure_dist.approx_rate == pytest.approx(
        original[0].node_config.failure_dist.approx_rate * 1.15
    )
    assert changed[0].node_config.recovery_dist.approx_rate == pytest.approx(
        original[0].node_config.recovery_dist.approx_rate
    )
    assert changed[0].node_config.data_loss_dist.approx_rate == pytest.approx(
        original[0].node_config.data_loss_dist.approx_rate
    )
    assert changed[1] is original[1]


def test_nominal_perturbation_is_identity():
    original = _types()
    assert perturb_machine_types(original, None) == original


def test_min_cost_solver_satisfies_both_availability_floors():
    types = _types()
    candidates = [
        TypeCountCandidate("cheap", ("low", "high"), (3, 0), 0.95, 3.0),
        TypeCountCandidate("reliable", ("low", "high"), (0, 3), 0.99, 9.0),
    ]
    solution = solve_type_count_min_cost_with_availability_floors(
        candidates,
        types,
        PlacementSolverConfig(num_rsms=2, budget_per_hour=20.0),
        minimum_sum_availability=1.98,
        minimum_product_availability=0.9801,
    )

    assert solution.status == "success"
    assert solution.total_cost_per_hour == pytest.approx(18.0)
    assert solution.selected[0].candidate.candidate_id == "reliable"
    assert solution.selected[0].count == 2


def test_markov_derivative_signs_and_cache_behavior():
    machine_type = MachineType(
        "only",
        _config(failure_rate=1 / days(10), recovery_rate=1 / hours(2)),
        copies=3,
    )
    cache = TypeCountMarkovScoreCache(
        LeaderlessProtocol(),
        NodeReplacementStrategy(hours(1), safe_mode=False),
        quality=QualityLevel.SIMPLIFIED,
    )
    counts = (3,)
    score = cache.score((machine_type,), counts)
    cache.score((machine_type,), counts)
    assert cache.hits == 1

    candidate = TypeCountCandidate(
        "only:3", ("only",), counts, score.availability, score.cost_per_hour
    )
    solution = TypeCountPlacementSolution(
        status="success",
        message="success",
        objective="sum_availability",
        objective_value=score.availability,
        total_cost_per_hour=score.cost_per_hour,
        budget_per_hour=None,
        selected=(SelectedTypeCountCandidate(candidate, 1),),
    )
    derivatives = {
        derivative.parameter.kind: derivative
        for derivative in placement_derivatives(
            solution,
            (machine_type,),
            (
                RateParameter("only", "transient_failure"),
                RateParameter("only", "data_loss"),
                RateParameter("only", "recovery"),
            ),
            cache,
        )
    }

    assert derivatives["transient_failure"].sum_raw < 0
    assert derivatives["data_loss"].sum_raw < 0
    assert derivatives["recovery"].sum_raw > 0
    product = score.availability
    assert derivatives["recovery"].product_raw == pytest.approx(
        product * derivatives["recovery"].sum_raw / score.availability
    )
    assert math.isfinite(derivatives["recovery"].product_elasticity)
