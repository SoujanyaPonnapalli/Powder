"""Reusable one-at-a-time rate sensitivity analysis for placement experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

from .placement_optimizer import (
    CandidateScore,
    MachineType,
    PlacementSolverConfig,
    TypeCountCandidate,
    TypeCountPlacementSolution,
    generate_type_count_candidates,
    rescore_type_count_solution,
    solve_type_count_min_cost_with_availability_floors,
)
from .results import markov_analyze
from .scenario import QualityLevel
from .simulation.distributions import Exponential
from .simulation.protocol import Protocol
from .simulation.strategy import ClusterStrategy


RateKind = Literal["transient_failure", "data_loss", "recovery"]


@dataclass(frozen=True)
class RateParameter:
    """One machine-rate input that can be varied independently."""

    machine_type_id: str
    kind: RateKind

    @property
    def key(self) -> str:
        return f"{self.machine_type_id}.{self.kind}_rate"


@dataclass(frozen=True)
class AvailabilityMetrics:
    """Aggregate availability metrics for a selected homogeneous RSM fleet."""

    sum_availability: float
    product_availability: float


@dataclass(frozen=True)
class CandidateDerivative:
    """Availability derivative of one selected candidate shape."""

    candidate_id: str
    count: int
    availability: float
    raw: float
    elasticity: float


@dataclass(frozen=True)
class AggregateDerivative:
    """Derivative of the placement's sum and product availability."""

    parameter: RateParameter
    rate: float
    sum_raw: float
    sum_elasticity: float
    product_raw: float
    product_elasticity: float
    candidates: tuple[CandidateDerivative, ...]


@dataclass(frozen=True)
class SensitivityRun:
    """One nominal or one-at-a-time erroneous optimizer run."""

    parameter: RateParameter | None
    relative_error: float
    solution: TypeCountPlacementSolution
    predicted: AvailabilityMetrics
    nominal_rescore: AvailabilityMetrics
    derivatives: tuple[AggregateDerivative, ...]

    @property
    def label(self) -> str:
        if self.parameter is None:
            return "nominal"
        return f"{self.parameter.key}:{self.relative_error:+.0%}"


class TypeCountMarkovScoreCache:
    """Cache Markov candidate scores by exact rate vector and type counts."""

    def __init__(
        self,
        protocol: Protocol,
        strategy: ClusterStrategy,
        *,
        quality: QualityLevel = QualityLevel.SIMPLIFIED,
    ) -> None:
        self.protocol = protocol
        self.strategy = strategy
        self.quality = quality
        self._scores: dict[tuple[tuple[float, ...], tuple[int, ...]], CandidateScore] = {}
        self.hits = 0
        self.misses = 0

    def score(
        self,
        machine_types: Sequence[MachineType],
        counts: tuple[int, ...],
    ) -> CandidateScore:
        key = (_machine_type_signature(machine_types), counts)
        cached = self._scores.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        node_configs = [
            machine_type.node_config
            for machine_type, count in zip(machine_types, counts)
            for _ in range(count)
        ]
        result = markov_analyze(node_configs, self.protocol, self.strategy, self.quality)
        cost = result.expected_cost_per_hour
        if cost is None:
            cost = sum(
                machine_type.node_config.cost_per_hour * count
                for machine_type, count in zip(machine_types, counts)
            )
        score = CandidateScore(
            availability=result.availability,
            cost_per_hour=cost,
            metadata={
                "method": result.method,
                "quality_level": self.quality.name,
                "num_states": result.num_states,
            },
        )
        self._scores[key] = score
        self.misses += 1
        return score


def rate_parameters(machine_types: Sequence[MachineType]) -> tuple[RateParameter, ...]:
    """Return transient-failure, data-loss, and recovery rates for every type."""
    return tuple(
        RateParameter(machine_type.type_id, kind)
        for machine_type in machine_types
        for kind in ("transient_failure", "data_loss", "recovery")
    )


def perturb_machine_types(
    machine_types: Sequence[MachineType],
    parameter: RateParameter | None,
    relative_error: float = 0.0,
) -> tuple[MachineType, ...]:
    """Return immutable type copies with exactly one selected rate shifted."""
    if parameter is None:
        if relative_error != 0.0:
            raise ValueError("a nonzero error requires a rate parameter")
        return tuple(machine_types)
    if relative_error <= -1.0:
        raise ValueError("relative_error must be greater than -100%")

    found = False
    updated: list[MachineType] = []
    for machine_type in machine_types:
        if machine_type.type_id != parameter.machine_type_id:
            updated.append(machine_type)
            continue
        found = True
        config = machine_type.node_config
        rate = _config_rate(config, parameter.kind) * (1.0 + relative_error)
        if parameter.kind == "transient_failure":
            config = replace(config, failure_dist=Exponential(rate))
        elif parameter.kind == "data_loss":
            config = replace(config, data_loss_dist=Exponential(rate))
        else:
            config = replace(config, recovery_dist=Exponential(rate))
        updated.append(replace(machine_type, node_config=config))
    if not found:
        raise ValueError(f"unknown machine type {parameter.machine_type_id!r}")
    return tuple(updated)


def placement_metrics(solution: TypeCountPlacementSolution) -> AvailabilityMetrics:
    """Calculate sum and product availability from selected candidate shapes."""
    if not solution.selected:
        return AvailabilityMetrics(float("nan"), float("nan"))
    total_sum = sum(item.count * item.candidate.availability for item in solution.selected)
    log_product = sum(
        item.count * math.log(max(item.candidate.availability, 1e-300))
        for item in solution.selected
    )
    return AvailabilityMetrics(float(total_sum), float(math.exp(log_product)))


def placement_derivatives(
    solution: TypeCountPlacementSolution,
    machine_types: Sequence[MachineType],
    parameters: Sequence[RateParameter],
    scorer: TypeCountMarkovScoreCache,
    *,
    relative_step: float = 1e-4,
) -> tuple[AggregateDerivative, ...]:
    """Differentiate selected candidate and aggregate availability by rates."""
    if not 0 < relative_step < 1:
        raise ValueError("relative_step must be between zero and one")
    if not solution.selected:
        return ()

    base = placement_metrics(solution)
    results: list[AggregateDerivative] = []
    for parameter in parameters:
        current_rate = _parameter_rate(machine_types, parameter)
        plus_types = perturb_machine_types(machine_types, parameter, relative_step)
        minus_types = perturb_machine_types(machine_types, parameter, -relative_step)
        candidate_derivatives: list[CandidateDerivative] = []
        sum_raw = 0.0
        product_log_raw = 0.0
        denominator = 2.0 * relative_step * current_rate
        for item in solution.selected:
            plus = scorer.score(plus_types, item.candidate.counts).availability
            minus = scorer.score(minus_types, item.candidate.counts).availability
            raw = (plus - minus) / denominator
            elasticity = current_rate * raw / max(item.candidate.availability, 1e-300)
            candidate_derivatives.append(
                CandidateDerivative(
                    candidate_id=item.candidate.candidate_id,
                    count=item.count,
                    availability=item.candidate.availability,
                    raw=raw,
                    elasticity=elasticity,
                )
            )
            sum_raw += item.count * raw
            product_log_raw += item.count * raw / max(item.candidate.availability, 1e-300)
        product_raw = base.product_availability * product_log_raw
        results.append(
            AggregateDerivative(
                parameter=parameter,
                rate=current_rate,
                sum_raw=sum_raw,
                sum_elasticity=current_rate * sum_raw / max(base.sum_availability, 1e-300),
                product_raw=product_raw,
                product_elasticity=current_rate * product_log_raw,
                candidates=tuple(candidate_derivatives),
            )
        )
    return tuple(results)


def run_one_at_a_time_sensitivity(
    machine_types: Sequence[MachineType],
    protocol: Protocol,
    strategy: ClusterStrategy,
    solver_config: PlacementSolverConfig,
    *,
    minimum_sum_availability: float,
    minimum_product_availability: float,
    error_levels: Sequence[float],
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
    derivative_step: float = 1e-4,
    score_cache: TypeCountMarkovScoreCache | None = None,
) -> tuple[SensitivityRun, ...]:
    """Run the nominal baseline plus every independent rate-error scenario."""
    cache = score_cache or TypeCountMarkovScoreCache(protocol, strategy, quality=quality)
    parameters = rate_parameters(machine_types)
    perturbations = [(None, 0.0)] + [
        (parameter, error)
        for parameter in parameters
        for error in error_levels
    ]
    runs: list[SensitivityRun] = []
    for parameter, error in perturbations:
        active_types = perturb_machine_types(machine_types, parameter, error)
        candidates = generate_type_count_candidates(
            active_types,
            cache.score,
            replica_counts=(3, 5, 7),
        )
        solution = solve_type_count_min_cost_with_availability_floors(
            candidates,
            active_types,
            solver_config,
            minimum_sum_availability=minimum_sum_availability,
            minimum_product_availability=minimum_product_availability,
        )
        predicted = placement_metrics(solution)
        nominal = rescore_type_count_solution(
            solution,
            machine_types,
            cache.score,
            objective="sum_availability",
        )
        derivatives = placement_derivatives(
            solution,
            active_types,
            parameters,
            cache,
            relative_step=derivative_step,
        )
        runs.append(
            SensitivityRun(
                parameter=parameter,
                relative_error=error,
                solution=solution,
                predicted=predicted,
                nominal_rescore=placement_metrics(nominal),
                derivatives=derivatives,
            )
        )
    return tuple(runs)


def _config_rate(config: object, kind: RateKind) -> float:
    distribution = {
        "transient_failure": config.failure_dist,
        "data_loss": config.data_loss_dist,
        "recovery": config.recovery_dist,
    }[kind]
    rate = distribution.approx_rate
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{kind} rate must be finite and positive")
    return float(rate)


def _parameter_rate(
    machine_types: Sequence[MachineType],
    parameter: RateParameter,
) -> float:
    for machine_type in machine_types:
        if machine_type.type_id == parameter.machine_type_id:
            return _config_rate(machine_type.node_config, parameter.kind)
    raise ValueError(f"unknown machine type {parameter.machine_type_id!r}")


def _machine_type_signature(machine_types: Sequence[MachineType]) -> tuple[float, ...]:
    values: list[float] = []
    for machine_type in machine_types:
        config = machine_type.node_config
        values.extend(
            (
                _config_rate(config, "transient_failure"),
                _config_rate(config, "data_loss"),
                _config_rate(config, "recovery"),
                config.spawn_dist.approx_rate,
                1.0 / config.snapshot_download_time_dist.mean,
                config.cost_per_hour,
                float(machine_type.copies),
                float(machine_type.capacity),
            )
        )
    return tuple(values)
