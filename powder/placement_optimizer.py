"""Candidate generation and ILP solving for RSM placement.

The optimizer is intentionally candidate-based: callers generate a tractable
menu of possible placements, score each placement with an oracle, then solve a
small integer program over the scored candidates.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Iterable, Literal, Mapping, Sequence

from .results import markov_analyze
from .scenario import QualityLevel
from .simulation.node import NodeConfig
from .simulation.protocol import Protocol
from .simulation.strategy import ClusterStrategy


Objective = Literal["sum_availability", "product_availability", "min_cost"]
BlindPlacementStrategy = Literal[
    "random",
    "cheap",
    "reliable",
    "large",
    "balanced",
    "value",
]


@dataclass(frozen=True)
class Machine:
    """A physical machine that can host RSM replicas."""

    machine_id: str
    node_config: NodeConfig
    capacity: int = 1


@dataclass(frozen=True)
class MachineType:
    """A group of interchangeable physical machines."""

    type_id: str
    node_config: NodeConfig
    copies: int
    capacity: int = 1

    @cached_property
    def total_capacity(self) -> int:
        return self.copies * self.capacity


@dataclass(frozen=True)
class CandidateScore:
    """Oracle score for a candidate before it is attached to a placement.

    Scorers only need to report availability, price, and optional diagnostic
    metadata. The optimizer wraps that score into a PlacementCandidate or
    TypeCountCandidate once it knows the candidate identifier and shape.
    """

    availability: float
    cost_per_hour: float
    # Optional oracle diagnostics such as method, quality level, or state count.
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementCandidate:
    """A scored candidate placement for one RSM."""

    candidate_id: str
    machine_ids: tuple[str, ...]
    availability: float
    cost_per_hour: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def replica_count(self) -> int:
        return len(self.machine_ids)

    @cached_property
    def log_availability(self) -> float:
        return math.log(max(self.availability, 1e-300))


@dataclass(frozen=True)
class TypeCountCandidate:
    """A scored RSM placement represented by counts per machine type."""

    candidate_id: str
    type_ids: tuple[str, ...]
    counts: tuple[int, ...]
    availability: float
    cost_per_hour: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def replica_count(self) -> int:
        return sum(self.counts)

    @cached_property
    def log_availability(self) -> float:
        return math.log(max(self.availability, 1e-300))


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Tuning knobs for first-shot candidate generation.

    ``spread_weight`` is a cheap region-diversity heuristic used only to seed
    candidate generation; it is not a correlated-failure Markov model.
    ``diversity_jaccard_threshold`` limits how similar selected machine sets
    can be when trimming a large generated candidate pool.
    """

    replica_counts: tuple[int, ...] = (3, 5, 7)
    random_samples_per_size: int = 250
    greedy_starts_per_size: int = 50
    local_search_steps: int = 2
    max_candidates: int = 5000
    rng_seed: int = 0
    spread_weight: float = 0.05
    cost_weight: float = 0.0
    diversity_jaccard_threshold: float = 0.7


@dataclass(frozen=True)
class PlacementSolverConfig:
    """Configuration for the candidate ILP."""

    num_rsms: int = 100
    objective: Objective = "sum_availability"
    budget_per_hour: float | None = None


@dataclass(frozen=True)
class BlindPlacementConfig:
    """Configuration for blind type-count placement baselines."""

    num_rsms: int = 100
    budget_per_hour: float | None = None
    replica_counts: tuple[int, ...] = (3, 5, 7)
    rng_seed: int = 0
    max_random_restarts: int = 100


@dataclass(frozen=True)
class SelectedCandidate:
    """How many RSMs use a candidate in the optimized solution.

    ``count`` can be greater than one when the aggregated ILP assigns the same
    placement shape to multiple RSMs. Machine capacity constraints still bound
    how often any shared physical machine can be reused across those RSMs.
    """

    candidate: PlacementCandidate
    count: int


@dataclass(frozen=True)
class SelectedTypeCountCandidate:
    """How many RSMs use a type-count candidate."""

    candidate: TypeCountCandidate
    count: int


@dataclass(frozen=True)
class PlacementSolution:
    """Result of solving the candidate selection problem."""

    status: str
    message: str
    objective: Objective
    objective_value: float
    total_cost_per_hour: float
    budget_per_hour: float | None
    selected: tuple[SelectedCandidate, ...]

    @cached_property
    def total_rsms(self) -> int:
        return sum(item.count for item in self.selected)


@dataclass(frozen=True)
class TypeCountPlacementSolution:
    """Result of solving the type-count candidate selection problem."""

    status: str
    message: str
    objective: Objective
    objective_value: float
    total_cost_per_hour: float
    budget_per_hour: float | None
    selected: tuple[SelectedTypeCountCandidate, ...]

    @cached_property
    def total_rsms(self) -> int:
        return sum(item.count for item in self.selected)


@dataclass(frozen=True)
class BlindTypeCountPlacementSolution:
    """Result of a blind placement baseline.

    Blind baselines do not use the Markov oracle during placement. Candidate
    availability is a simple independent-node quorum estimate and candidate
    cost is the sum of per-node hourly prices.
    """

    status: str
    message: str
    strategy: BlindPlacementStrategy
    total_simple_cost_per_hour: float
    budget_per_hour: float | None
    selected: tuple[SelectedTypeCountCandidate, ...]

    @cached_property
    def total_rsms(self) -> int:
        return sum(item.count for item in self.selected)


CandidateScorer = Callable[[Sequence[Machine]], CandidateScore]
TypeCountScorer = Callable[[Sequence[MachineType], tuple[int, ...]], CandidateScore]
BlindTypeCountPlacementFn = Callable[
    [Sequence[MachineType], BlindPlacementConfig],
    BlindTypeCountPlacementSolution,
]


def markov_candidate_scorer(
    protocol: Protocol,
    strategy: ClusterStrategy,
    *,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> CandidateScorer:
    """Build a placement scorer backed by the existing Markov oracle."""

    def _score(machines: Sequence[Machine]) -> CandidateScore:
        result = markov_analyze(
            [m.node_config for m in machines],
            protocol,
            strategy,
            quality,
        )
        cost = result.expected_cost_per_hour
        if cost is None:
            cost = sum(m.node_config.cost_per_hour for m in machines)
        return CandidateScore(
            availability=result.availability,
            cost_per_hour=cost,
            metadata={
                "method": result.method,
                "quality_level": result.quality_level,
                "num_states": result.num_states,
            },
        )

    return _score


def markov_type_count_scorer(
    protocol: Protocol,
    strategy: ClusterStrategy,
    *,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> TypeCountScorer:
    """Build a type-count scorer backed by the existing Markov oracle."""

    def _score(
        machine_types: Sequence[MachineType],
        counts: tuple[int, ...],
    ) -> CandidateScore:
        node_configs = [
            machine_type.node_config
            for machine_type, count in zip(machine_types, counts)
            for _ in range(count)
        ]
        result = markov_analyze(
            node_configs,
            protocol,
            strategy,
            quality,
        )
        cost = result.expected_cost_per_hour
        if cost is None:
            cost = sum(
                machine_type.node_config.cost_per_hour * count
                for machine_type, count in zip(machine_types, counts)
            )
        return CandidateScore(
            availability=result.availability,
            cost_per_hour=cost,
            metadata={
                "method": result.method,
                "quality_level": result.quality_level,
                "num_states": result.num_states,
            },
        )

    return _score


def generate_candidates(
    machines: Sequence[Machine],
    scorer: CandidateScorer,
    config: CandidateGenerationConfig = CandidateGenerationConfig(),
) -> list[PlacementCandidate]:
    """Generate, score, filter, and diversify placement candidates."""

    if not machines:
        raise ValueError("machines must not be empty")

    machine_by_id = _machine_by_id(machines)
    rng = random.Random(config.rng_seed)
    raw: set[tuple[str, ...]] = set()

    for k in config.replica_counts:
        _validate_replica_count(k, machines)
        for _ in range(config.random_samples_per_size):
            raw.add(_stratified_sample(machines, k, rng))

        for _ in range(config.greedy_starts_per_size):
            raw.add(_greedy_candidate(machines, k, rng, config))

    for placement in list(raw):
        current = placement
        current_score = _approximate_candidate_score(current, machine_by_id, config)
        for _ in range(config.local_search_steps):
            mutated = _mutate_one_machine(current, machines, rng)
            mutated_score = _approximate_candidate_score(mutated, machine_by_id, config)
            if mutated_score >= current_score:
                raw.add(mutated)
                current = mutated
                current_score = mutated_score

    scored = _score_raw_candidates(raw, machine_by_id, scorer)
    scored = _bucketed_pareto_filter(scored)
    return _diversity_select(scored, config.max_candidates, config.diversity_jaccard_threshold)


def generate_type_count_candidates(
    machine_types: Sequence[MachineType],
    scorer: TypeCountScorer,
    replica_counts: Sequence[int] = (3, 5, 7),
) -> list[TypeCountCandidate]:
    """Enumerate and score all valid type-count placement candidates."""

    _validate_machine_types(machine_types)
    type_ids = tuple(machine_type.type_id for machine_type in machine_types)
    bounds = tuple(machine_type.copies for machine_type in machine_types)

    candidates: list[TypeCountCandidate] = []
    for k in replica_counts:
        if k <= 0:
            raise ValueError(f"replica count must be positive, got {k}")
        for counts in _bounded_integer_vectors(k, bounds):
            score = scorer(machine_types, counts)
            _validate_score(score, _type_count_candidate_id(type_ids, counts))
            candidates.append(
                TypeCountCandidate(
                    candidate_id=_type_count_candidate_id(type_ids, counts),
                    type_ids=type_ids,
                    counts=counts,
                    availability=float(score.availability),
                    cost_per_hour=float(score.cost_per_hour),
                    metadata=dict(score.metadata),
                )
            )
    return candidates


def solve_candidate_ilp(
    candidates: Sequence[PlacementCandidate],
    machines: Sequence[Machine],
    config: PlacementSolverConfig = PlacementSolverConfig(),
) -> PlacementSolution:
    """Solve the aggregated candidate ILP with SciPy/HiGHS.

    The aggregated formulation chooses an integer count for each candidate.
    It is equivalent to explicit RSM-to-candidate assignment when all RSMs have
    the same candidate menu and objective weight.
    """

    if not candidates:
        raise ValueError("candidates must not be empty")
    if config.num_rsms <= 0:
        raise ValueError("num_rsms must be positive")

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover - exercised without scipy installed
        raise ImportError(
            "solve_candidate_ilp requires scipy.optimize.milp. "
            "Install Powder's project dependencies before solving the ILP."
        ) from exc

    machine_ids = [m.machine_id for m in machines]
    capacities = {m.machine_id: m.capacity for m in machines}
    machine_index = {machine_id: i for i, machine_id in enumerate(machine_ids)}
    unknown = sorted(
        {mid for c in candidates for mid in c.machine_ids if mid not in machine_index}
    )
    if unknown:
        raise ValueError(f"candidates reference unknown machines: {unknown[:5]}")

    num_budget_rows = 1 if config.budget_per_hour is not None else 0
    row_count = 1 + len(machine_ids) + num_budget_rows
    col_count = len(candidates)
    constraints = lil_matrix((row_count, col_count), dtype=float)

    # Exactly num_rsms selected placements in aggregate.
    constraints[0, :] = 1.0
    lower = [float(config.num_rsms)]
    upper = [float(config.num_rsms)]

    # Machine capacity rows.
    for row_offset, machine_id in enumerate(machine_ids, start=1):
        lower.append(0.0)
        upper.append(float(capacities[machine_id]))
        for col, candidate in enumerate(candidates):
            if machine_id in candidate.machine_ids:
                constraints[row_offset, col] = 1.0

    if config.budget_per_hour is not None:
        budget_row = row_count - 1
        for col, candidate in enumerate(candidates):
            constraints[budget_row, col] = candidate.cost_per_hour
        lower.append(0.0)
        upper.append(float(config.budget_per_hour))

    objective_coeffs = np.array(
        [_objective_value(candidate, config.objective) for candidate in candidates],
        dtype=float,
    )
    result = milp(
        c=-objective_coeffs,
        integrality=np.ones(col_count, dtype=int),
        bounds=Bounds(lb=np.zeros(col_count), ub=np.full(col_count, config.num_rsms)),
        constraints=LinearConstraint(constraints.tocsr(), np.array(lower), np.array(upper)),
    )

    if not result.success:
        return PlacementSolution(
            status=str(result.status),
            message=str(result.message),
            objective=config.objective,
            objective_value=float("nan"),
            total_cost_per_hour=float("nan"),
            budget_per_hour=config.budget_per_hour,
            selected=(),
        )

    counts = np.rint(result.x).astype(int)
    selected = tuple(
        SelectedCandidate(candidate, int(count))
        for candidate, count in zip(candidates, counts)
        if count > 0
    )
    objective_value = sum(
        _objective_value(item.candidate, config.objective) * item.count
        for item in selected
    )
    total_cost = sum(item.candidate.cost_per_hour * item.count for item in selected)

    return PlacementSolution(
        status=str(result.status),
        message=str(result.message),
        objective=config.objective,
        objective_value=float(objective_value),
        total_cost_per_hour=float(total_cost),
        budget_per_hour=config.budget_per_hour,
        selected=selected,
    )


def solve_type_count_ilp(
    candidates: Sequence[TypeCountCandidate],
    machine_types: Sequence[MachineType],
    config: PlacementSolverConfig = PlacementSolverConfig(),
) -> TypeCountPlacementSolution:
    """Solve the aggregated ILP over type-count placement candidates."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    if config.num_rsms <= 0:
        raise ValueError("num_rsms must be positive")
    _validate_machine_types(machine_types)

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover - exercised without scipy installed
        raise ImportError(
            "solve_type_count_ilp requires scipy.optimize.milp. "
            "Install Powder's project dependencies before solving the ILP."
        ) from exc

    type_ids = tuple(machine_type.type_id for machine_type in machine_types)
    for candidate in candidates:
        _validate_type_count_candidate(candidate, machine_types, type_ids)

    num_budget_rows = 1 if config.budget_per_hour is not None else 0
    row_count = 1 + len(machine_types) + num_budget_rows
    col_count = len(candidates)
    constraints = lil_matrix((row_count, col_count), dtype=float)

    constraints[0, :] = 1.0
    lower = [float(config.num_rsms)]
    upper = [float(config.num_rsms)]

    for row_offset, machine_type in enumerate(machine_types, start=1):
        lower.append(0.0)
        upper.append(float(machine_type.total_capacity))
        for col, candidate in enumerate(candidates):
            constraints[row_offset, col] = candidate.counts[row_offset - 1]

    if config.budget_per_hour is not None:
        budget_row = row_count - 1
        for col, candidate in enumerate(candidates):
            constraints[budget_row, col] = candidate.cost_per_hour
        lower.append(0.0)
        upper.append(float(config.budget_per_hour))

    objective_coeffs = np.array(
        [_type_count_objective_value(candidate, config.objective) for candidate in candidates],
        dtype=float,
    )
    result = milp(
        c=-objective_coeffs,
        integrality=np.ones(col_count, dtype=int),
        bounds=Bounds(lb=np.zeros(col_count), ub=np.full(col_count, config.num_rsms)),
        constraints=LinearConstraint(constraints.tocsr(), np.array(lower), np.array(upper)),
    )

    if not result.success:
        return TypeCountPlacementSolution(
            status=str(result.status),
            message=str(result.message),
            objective=config.objective,
            objective_value=float("nan"),
            total_cost_per_hour=float("nan"),
            budget_per_hour=config.budget_per_hour,
            selected=(),
        )

    counts = np.rint(result.x).astype(int)
    selected = tuple(
        SelectedTypeCountCandidate(candidate, int(count))
        for candidate, count in zip(candidates, counts)
        if count > 0
    )
    objective_value = sum(
        _type_count_objective_value(item.candidate, config.objective) * item.count
        for item in selected
    )
    total_cost = sum(item.candidate.cost_per_hour * item.count for item in selected)

    return TypeCountPlacementSolution(
        status=str(result.status),
        message=str(result.message),
        objective=config.objective,
        objective_value=float(objective_value),
        total_cost_per_hour=float(total_cost),
        budget_per_hour=config.budget_per_hour,
        selected=selected,
    )


def solve_type_count_min_cost_with_availability_floors(
    candidates: Sequence[TypeCountCandidate],
    machine_types: Sequence[MachineType],
    config: PlacementSolverConfig,
    *,
    minimum_sum_availability: float,
    minimum_product_availability: float,
) -> TypeCountPlacementSolution:
    """Minimize cost while satisfying aggregate availability guarantees.

    The product floor is linearized as a lower bound on the sum of candidate
    log availabilities, exactly as in the product-availability maximization
    objective.  This is intended for homogeneous RSM fleets where selecting a
    candidate ``count`` times represents that many independent RSMs.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if config.num_rsms <= 0:
        raise ValueError("num_rsms must be positive")
    if minimum_sum_availability <= 0:
        raise ValueError("minimum_sum_availability must be positive")
    if not 0 < minimum_product_availability <= 1:
        raise ValueError("minimum_product_availability must be in (0, 1]")
    _validate_machine_types(machine_types)

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "solve_type_count_min_cost_with_availability_floors requires "
            "scipy.optimize.milp. Install Powder's project dependencies."
        ) from exc

    type_ids = tuple(machine_type.type_id for machine_type in machine_types)
    for candidate in candidates:
        _validate_type_count_candidate(candidate, machine_types, type_ids)

    has_budget = config.budget_per_hour is not None
    # RSM count + type capacities + sum floor + product floor + optional budget.
    row_count = 1 + len(machine_types) + 2 + int(has_budget)
    constraints = lil_matrix((row_count, len(candidates)), dtype=float)
    lower: list[float] = [float(config.num_rsms)]
    upper: list[float] = [float(config.num_rsms)]
    constraints[0, :] = 1.0

    for row, machine_type in enumerate(machine_types, start=1):
        lower.append(0.0)
        upper.append(float(machine_type.total_capacity))
        for col, candidate in enumerate(candidates):
            constraints[row, col] = candidate.counts[row - 1]

    sum_row = 1 + len(machine_types)
    log_row = sum_row + 1
    for col, candidate in enumerate(candidates):
        constraints[sum_row, col] = candidate.availability
        constraints[log_row, col] = candidate.log_availability
    lower.extend([float(minimum_sum_availability), math.log(minimum_product_availability)])
    upper.extend([float("inf"), float("inf")])

    if has_budget:
        budget_row = row_count - 1
        for col, candidate in enumerate(candidates):
            constraints[budget_row, col] = candidate.cost_per_hour
        lower.append(0.0)
        upper.append(float(config.budget_per_hour))

    result = milp(
        c=np.asarray([candidate.cost_per_hour for candidate in candidates], dtype=float),
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(lb=np.zeros(len(candidates)), ub=np.full(len(candidates), config.num_rsms)),
        constraints=LinearConstraint(constraints.tocsr(), np.asarray(lower), np.asarray(upper)),
    )
    if not result.success:
        return TypeCountPlacementSolution(
            status=str(result.status), message=str(result.message), objective="min_cost",
            objective_value=float("nan"), total_cost_per_hour=float("nan"),
            budget_per_hour=config.budget_per_hour, selected=(),
        )

    counts = np.rint(result.x).astype(int)
    selected = tuple(
        SelectedTypeCountCandidate(candidate, int(count))
        for candidate, count in zip(candidates, counts)
        if count > 0
    )
    total_cost = sum(item.candidate.cost_per_hour * item.count for item in selected)
    return TypeCountPlacementSolution(
        status="success", message="success", objective="min_cost",
        objective_value=float(total_cost), total_cost_per_hour=float(total_cost),
        budget_per_hour=config.budget_per_hour, selected=selected,
    )


def rescore_type_count_solution(
    solution: BlindTypeCountPlacementSolution | TypeCountPlacementSolution,
    machine_types: Sequence[MachineType],
    scorer: TypeCountScorer,
    *,
    objective: Objective = "sum_availability",
) -> TypeCountPlacementSolution:
    """Rescore a type-count solution with an oracle such as the Markov model."""

    if solution.status != "success" and not solution.selected:
        return TypeCountPlacementSolution(
            status=solution.status,
            message=solution.message,
            objective=objective,
            objective_value=float("nan"),
            total_cost_per_hour=float("nan"),
            budget_per_hour=solution.budget_per_hour,
            selected=(),
        )

    _validate_machine_types(machine_types)
    type_ids = tuple(machine_type.type_id for machine_type in machine_types)
    selected = []
    for item in solution.selected:
        counts = item.candidate.counts
        score = scorer(machine_types, counts)
        _validate_score(score, item.candidate.candidate_id)
        metadata = {
            **item.candidate.metadata,
            "prescore_availability": item.candidate.availability,
            "prescore_cost_per_hour": item.candidate.cost_per_hour,
            **dict(score.metadata),
        }
        candidate = TypeCountCandidate(
            candidate_id=item.candidate.candidate_id,
            type_ids=type_ids,
            counts=counts,
            availability=float(score.availability),
            cost_per_hour=float(score.cost_per_hour),
            metadata=metadata,
        )
        selected.append(SelectedTypeCountCandidate(candidate, item.count))

    objective_value = sum(
        _type_count_objective_value(item.candidate, objective) * item.count
        for item in selected
    )
    total_cost = sum(item.candidate.cost_per_hour * item.count for item in selected)
    return TypeCountPlacementSolution(
        status="success",
        message="success",
        objective=objective,
        objective_value=float(objective_value),
        total_cost_per_hour=float(total_cost),
        budget_per_hour=solution.budget_per_hour,
        selected=tuple(selected),
    )


def solve_blind_type_count_placement(
    machine_types: Sequence[MachineType],
    strategy: BlindPlacementStrategy,
    scorer: TypeCountScorer,
    config: BlindPlacementConfig = BlindPlacementConfig(),
    *,
    objective: Objective = "sum_availability",
) -> TypeCountPlacementSolution:
    """Place with a blind baseline, then rescore with the supplied oracle.

    The blind strategy still uses simple per-node prices while searching and
    repairing to the requested budget. The returned solution replaces those
    lower-bound estimates with the oracle's availability and cost per hour.
    """

    placement_fn = _blind_type_count_placement_fn(strategy)
    blind_solution = placement_fn(machine_types, config)
    return rescore_type_count_solution(
        blind_solution,
        machine_types,
        scorer,
        objective=objective,
    )


def blind_random_type_count_placement(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig = BlindPlacementConfig(),
) -> BlindTypeCountPlacementSolution:
    """Randomly place RSMs subject to type capacities and simple budget."""

    _validate_blind_inputs(machine_types, config)
    rng = random.Random(config.rng_seed)
    replica_counts = tuple(config.replica_counts)

    for _ in range(config.max_random_restarts):
        remaining = _initial_type_capacity(machine_types)
        placements: list[list[int]] = []
        total_cost = 0.0
        failed = False

        for _rsm in range(config.num_rsms):
            shuffled_k = list(replica_counts)
            rng.shuffle(shuffled_k)
            candidate: list[int] | None = None
            for k in shuffled_k:
                for _attempt in range(50):
                    trial = _random_counts(machine_types, remaining, k, rng)
                    if trial is None:
                        continue
                    trial_cost = _simple_candidate_cost(machine_types, trial)
                    if (
                        config.budget_per_hour is None
                        or total_cost + trial_cost <= config.budget_per_hour
                    ):
                        candidate = trial
                        break
                if candidate is not None:
                    break

            if candidate is None:
                failed = True
                break

            placements.append(candidate)
            _consume_capacity(remaining, candidate)
            total_cost += _simple_candidate_cost(machine_types, candidate)

        if not failed:
            return _make_blind_solution("random", machine_types, placements, config.budget_per_hour)

    return _blind_infeasible(
        "random",
        "could not find a random feasible placement",
        config.budget_per_hour,
    )


def blind_cheap_type_count_placement(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig = BlindPlacementConfig(),
) -> BlindTypeCountPlacementSolution:
    """Greedily build minimum-size RSMs from the cheapest available types."""

    _validate_blind_inputs(machine_types, config)
    placements = _greedy_blind_placements(
        machine_types,
        config,
        strategy="cheap",
        order_key=lambda i: (
            machine_types[i].node_config.cost_per_hour,
            -_type_reliability(machine_types[i]),
            machine_types[i].type_id,
        ),
    )
    if placements is None:
        return _blind_infeasible(
            "cheap",
            "could not build enough cheap RSMs",
            config.budget_per_hour,
        )

    if not _within_simple_budget(machine_types, placements, config.budget_per_hour):
        return _blind_infeasible(
            "cheap",
            "cheapest feasible placement exceeds budget",
            config.budget_per_hour,
        )
    return _make_blind_solution("cheap", machine_types, placements, config.budget_per_hour)


def blind_reliable_type_count_placement(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig = BlindPlacementConfig(),
) -> BlindTypeCountPlacementSolution:
    """Greedily choose reliable minimum-size RSMs, then repair cost."""

    _validate_blind_inputs(machine_types, config)
    placements = _greedy_blind_placements(
        machine_types,
        config,
        strategy="reliable",
        order_key=lambda i: (
            -_type_reliability(machine_types[i]),
            machine_types[i].node_config.cost_per_hour,
            machine_types[i].type_id,
        ),
    )
    if placements is None:
        return _blind_infeasible(
            "reliable",
            "could not build enough reliable RSMs",
            config.budget_per_hour,
        )

    if not _repair_to_budget(
        placements,
        machine_types,
        config,
        replacement_first=True,
        allow_shrink=True,
    ):
        return _blind_infeasible(
            "reliable",
            "could not downgrade under budget",
            config.budget_per_hour,
        )
    return _make_blind_solution("reliable", machine_types, placements, config.budget_per_hour)


def blind_large_type_count_placement(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig = BlindPlacementConfig(),
) -> BlindTypeCountPlacementSolution:
    """Build large RSMs first, then shrink/downgrade to fit the budget."""

    _validate_blind_inputs(machine_types, config)
    placements = _greedy_blind_placements(
        machine_types,
        config,
        strategy="large",
        replica_order=tuple(sorted(config.replica_counts, reverse=True)),
        reserve_min_remaining=True,
        order_key=lambda i: (
            -_type_reliability(machine_types[i]),
            machine_types[i].node_config.cost_per_hour,
            machine_types[i].type_id,
        ),
    )
    if placements is None:
        return _blind_infeasible(
            "large",
            "could not build enough large RSMs",
            config.budget_per_hour,
        )

    if not _repair_to_budget(
        placements,
        machine_types,
        config,
        replacement_first=False,
        allow_shrink=True,
    ):
        return _blind_infeasible(
            "large",
            "could not shrink or downgrade under budget",
            config.budget_per_hour,
        )
    return _make_blind_solution("large", machine_types, placements, config.budget_per_hour)


def blind_balanced_type_count_placement(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig = BlindPlacementConfig(),
) -> BlindTypeCountPlacementSolution:
    """Spread replicas over the least-utilized feasible type capacities."""

    _validate_blind_inputs(machine_types, config)
    remaining = _initial_type_capacity(machine_types)
    placements: list[list[int]] = []
    replica_order = tuple(sorted(config.replica_counts))

    for _rsm in range(config.num_rsms):
        candidate = None
        for k in replica_order:
            candidate = _balanced_counts(machine_types, remaining, k)
            if candidate is not None:
                break
        if candidate is None:
            return _blind_infeasible(
                "balanced",
                "could not build enough balanced RSMs",
                config.budget_per_hour,
            )
        placements.append(candidate)
        _consume_capacity(remaining, candidate)

    if not _repair_to_budget(
        placements,
        machine_types,
        config,
        replacement_first=True,
        allow_shrink=True,
    ):
        return _blind_infeasible(
            "balanced",
            "could not repair balanced placement under budget",
            config.budget_per_hour,
        )
    return _make_blind_solution("balanced", machine_types, placements, config.budget_per_hour)


def blind_value_type_count_placement(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig = BlindPlacementConfig(),
) -> BlindTypeCountPlacementSolution:
    """Greedily choose types by simple reliability value per dollar."""

    _validate_blind_inputs(machine_types, config)
    placements = _greedy_blind_placements(
        machine_types,
        config,
        strategy="value",
        order_key=lambda i: (
            -_type_value(machine_types[i]),
            -_type_reliability(machine_types[i]),
            machine_types[i].node_config.cost_per_hour,
            machine_types[i].type_id,
        ),
    )
    if placements is None:
        return _blind_infeasible(
            "value",
            "could not build enough value-based RSMs",
            config.budget_per_hour,
        )

    if not _repair_to_budget(
        placements,
        machine_types,
        config,
        replacement_first=True,
        allow_shrink=True,
    ):
        return _blind_infeasible(
            "value",
            "could not repair value placement under budget",
            config.budget_per_hour,
        )
    return _make_blind_solution("value", machine_types, placements, config.budget_per_hour)


def _validate_blind_inputs(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig,
) -> None:
    _validate_machine_types(machine_types)
    if config.num_rsms <= 0:
        raise ValueError("num_rsms must be positive")
    if not config.replica_counts:
        raise ValueError("replica_counts must not be empty")
    for replica_count in config.replica_counts:
        if replica_count <= 0:
            raise ValueError(f"replica count must be positive, got {replica_count}")


def _blind_type_count_placement_fn(
    strategy: BlindPlacementStrategy,
) -> BlindTypeCountPlacementFn:
    if strategy == "random":
        return blind_random_type_count_placement
    if strategy == "cheap":
        return blind_cheap_type_count_placement
    if strategy == "reliable":
        return blind_reliable_type_count_placement
    if strategy == "large":
        return blind_large_type_count_placement
    if strategy == "balanced":
        return blind_balanced_type_count_placement
    if strategy == "value":
        return blind_value_type_count_placement
    raise ValueError(f"unknown blind placement strategy {strategy!r}")


def _blind_infeasible(
    strategy: BlindPlacementStrategy,
    message: str,
    budget_per_hour: float | None,
) -> BlindTypeCountPlacementSolution:
    return BlindTypeCountPlacementSolution(
        status="infeasible",
        message=message,
        strategy=strategy,
        total_simple_cost_per_hour=float("nan"),
        budget_per_hour=budget_per_hour,
        selected=(),
    )


def _make_blind_solution(
    strategy: BlindPlacementStrategy,
    machine_types: Sequence[MachineType],
    placements: Sequence[Sequence[int]],
    budget_per_hour: float | None,
) -> BlindTypeCountPlacementSolution:
    type_ids = tuple(machine_type.type_id for machine_type in machine_types)
    grouped: dict[tuple[int, ...], int] = defaultdict(int)
    for placement in placements:
        grouped[tuple(placement)] += 1

    selected = []
    for counts, count in sorted(grouped.items()):
        candidate = TypeCountCandidate(
            candidate_id=_type_count_candidate_id(type_ids, counts),
            type_ids=type_ids,
            counts=counts,
            availability=_blind_candidate_availability(machine_types, counts),
            cost_per_hour=_simple_candidate_cost(machine_types, counts),
            metadata={"blind_strategy": strategy},
        )
        selected.append(SelectedTypeCountCandidate(candidate, count))

    total_cost = sum(
        item.candidate.cost_per_hour * item.count
        for item in selected
    )
    return BlindTypeCountPlacementSolution(
        status="success",
        message="success",
        strategy=strategy,
        total_simple_cost_per_hour=float(total_cost),
        budget_per_hour=budget_per_hour,
        selected=tuple(selected),
    )


def _greedy_blind_placements(
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig,
    *,
    strategy: BlindPlacementStrategy,
    order_key: Callable[[int], tuple],
    replica_order: Sequence[int] | None = None,
    reserve_min_remaining: bool = False,
) -> list[list[int]] | None:
    del strategy
    remaining = _initial_type_capacity(machine_types)
    placements: list[list[int]] = []
    ordered_replica_counts = tuple(replica_order or sorted(config.replica_counts))
    order = sorted(range(len(machine_types)), key=order_key)
    min_replica_count = min(config.replica_counts)

    for rsm_idx in range(config.num_rsms):
        candidate = None
        remaining_rsms_after_candidate = config.num_rsms - rsm_idx - 1
        for k in ordered_replica_counts:
            trial = _ordered_counts(machine_types, remaining, k, order)
            if trial is None:
                continue
            if reserve_min_remaining and (
                sum(remaining) - sum(trial)
                < remaining_rsms_after_candidate * min_replica_count
            ):
                continue
            candidate = trial
            if candidate is not None:
                break
        if candidate is None:
            return None
        placements.append(candidate)
        _consume_capacity(remaining, candidate)
    return placements


def _ordered_counts(
    machine_types: Sequence[MachineType],
    remaining_capacity: Sequence[int],
    replica_count: int,
    order: Sequence[int],
) -> list[int] | None:
    counts = [0] * len(machine_types)
    for _ in range(replica_count):
        chosen = None
        for idx in order:
            if (
                remaining_capacity[idx] - counts[idx] > 0
                and counts[idx] < machine_types[idx].copies
            ):
                chosen = idx
                break
        if chosen is None:
            return None
        counts[chosen] += 1
    return counts


def _random_counts(
    machine_types: Sequence[MachineType],
    remaining_capacity: Sequence[int],
    replica_count: int,
    rng: random.Random,
) -> list[int] | None:
    counts = [0] * len(machine_types)
    for _ in range(replica_count):
        feasible = [
            idx
            for idx, machine_type in enumerate(machine_types)
            if remaining_capacity[idx] - counts[idx] > 0
            and counts[idx] < machine_type.copies
        ]
        if not feasible:
            return None
        counts[rng.choice(feasible)] += 1
    return counts


def _balanced_counts(
    machine_types: Sequence[MachineType],
    remaining_capacity: Sequence[int],
    replica_count: int,
) -> list[int] | None:
    counts = [0] * len(machine_types)
    total_capacity = _initial_type_capacity(machine_types)
    for _ in range(replica_count):
        feasible = []
        for idx, machine_type in enumerate(machine_types):
            if (
                remaining_capacity[idx] - counts[idx] > 0
                and counts[idx] < machine_type.copies
            ):
                used = total_capacity[idx] - remaining_capacity[idx] + counts[idx]
                utilization = used / total_capacity[idx] if total_capacity[idx] else 1.0
                feasible.append(
                    (
                        utilization,
                        -_type_value(machine_type),
                        machine_type.node_config.cost_per_hour,
                        machine_type.type_id,
                        idx,
                    )
                )
        if not feasible:
            return None
        counts[min(feasible)[-1]] += 1
    return counts


def _repair_to_budget(
    placements: list[list[int]],
    machine_types: Sequence[MachineType],
    config: BlindPlacementConfig,
    *,
    replacement_first: bool,
    allow_shrink: bool,
) -> bool:
    if config.budget_per_hour is None:
        return True

    min_size = min(config.replica_counts)
    allowed_sizes = tuple(sorted(config.replica_counts))
    while _total_simple_cost(machine_types, placements) > config.budget_per_hour:
        if replacement_first:
            changed = _apply_best_replacement(placements, machine_types)
            if not changed and allow_shrink:
                changed = _apply_best_shrink(placements, machine_types, allowed_sizes, min_size)
        else:
            changed = False
            if allow_shrink:
                changed = _apply_best_shrink(placements, machine_types, allowed_sizes, min_size)
            if not changed:
                changed = _apply_best_replacement(placements, machine_types)

        if not changed:
            return False
    return True


def _apply_best_replacement(
    placements: list[list[int]],
    machine_types: Sequence[MachineType],
) -> bool:
    used = _used_capacity(placements, len(machine_types))
    best: tuple[float, float, int, int, int] | None = None

    for placement_idx, placement in enumerate(placements):
        for hi, hi_count in enumerate(placement):
            if hi_count <= 0:
                continue
            hi_price = machine_types[hi].node_config.cost_per_hour
            hi_reliability = _type_reliability(machine_types[hi])
            for lo, machine_type in enumerate(machine_types):
                lo_price = machine_type.node_config.cost_per_hour
                if lo_price >= hi_price:
                    continue
                if placement[lo] + 1 > machine_type.copies:
                    continue
                if used[lo] + 1 > machine_type.total_capacity:
                    continue
                savings = hi_price - lo_price
                loss = max(0.0, hi_reliability - _type_reliability(machine_type))
                score = loss / savings if savings > 0 else float("inf")
                contender = (score, -savings, placement_idx, hi, lo)
                if best is None or contender < best:
                    best = contender

    if best is None:
        return False

    _score, _negative_savings, placement_idx, hi, lo = best
    placements[placement_idx][hi] -= 1
    placements[placement_idx][lo] += 1
    return True


def _apply_best_shrink(
    placements: list[list[int]],
    machine_types: Sequence[MachineType],
    allowed_sizes: Sequence[int],
    min_size: int,
) -> bool:
    best: tuple[float, float, int, tuple[int, ...]] | None = None
    for placement_idx, placement in enumerate(placements):
        current_size = sum(placement)
        if current_size <= min_size:
            continue
        smaller_sizes = [size for size in allowed_sizes if size < current_size]
        if not smaller_sizes:
            continue
        target_size = max(smaller_sizes)
        remove_count = current_size - target_size
        for removal in _bounded_integer_vectors(remove_count, placement):
            savings = _simple_candidate_cost(machine_types, removal)
            if savings <= 0:
                continue
            loss = sum(
                _type_reliability(machine_type) * count
                for machine_type, count in zip(machine_types, removal)
            )
            score = loss / savings
            contender = (score, -savings, placement_idx, removal)
            if best is None or contender < best:
                best = contender

    if best is None:
        return False

    _score, _negative_savings, placement_idx, removal = best
    for idx, count in enumerate(removal):
        placements[placement_idx][idx] -= count
    return True


def _initial_type_capacity(machine_types: Sequence[MachineType]) -> list[int]:
    return [machine_type.total_capacity for machine_type in machine_types]


def _consume_capacity(
    remaining_capacity: list[int],
    counts: Sequence[int],
) -> None:
    for idx, count in enumerate(counts):
        remaining_capacity[idx] -= count


def _used_capacity(
    placements: Sequence[Sequence[int]],
    num_types: int,
) -> list[int]:
    used = [0] * num_types
    for placement in placements:
        for idx, count in enumerate(placement):
            used[idx] += count
    return used


def _within_simple_budget(
    machine_types: Sequence[MachineType],
    placements: Sequence[Sequence[int]],
    budget: float | None,
) -> bool:
    return budget is None or _total_simple_cost(machine_types, placements) <= budget


def _total_simple_cost(
    machine_types: Sequence[MachineType],
    placements: Sequence[Sequence[int]],
) -> float:
    return sum(_simple_candidate_cost(machine_types, placement) for placement in placements)


def _simple_candidate_cost(
    machine_types: Sequence[MachineType],
    counts: Sequence[int],
) -> float:
    return sum(
        machine_type.node_config.cost_per_hour * count
        for machine_type, count in zip(machine_types, counts)
    )


def _blind_candidate_availability(
    machine_types: Sequence[MachineType],
    counts: Sequence[int],
) -> float:
    probabilities = [
        _type_reliability(machine_type)
        for machine_type, count in zip(machine_types, counts)
        for _ in range(count)
    ]
    quorum = len(probabilities) // 2 + 1
    return _poisson_binomial_at_least(probabilities, quorum)


def _type_reliability(machine_type: MachineType) -> float:
    return _approximate_node_availability(machine_type.node_config)


def _type_value(machine_type: MachineType) -> float:
    price = machine_type.node_config.cost_per_hour
    reliability = _type_reliability(machine_type)
    if price <= 0:
        return float("inf")
    return -math.log(max(1.0 - reliability, 1e-300)) / price


def _machine_by_id(machines: Sequence[Machine]) -> dict[str, Machine]:
    machine_by_id: dict[str, Machine] = {}
    for machine in machines:
        if machine.capacity < 0:
            raise ValueError(f"machine {machine.machine_id!r} has negative capacity")
        if machine.machine_id in machine_by_id:
            raise ValueError(f"duplicate machine_id {machine.machine_id!r}")
        machine_by_id[machine.machine_id] = machine
    return machine_by_id


def _validate_machine_types(machine_types: Sequence[MachineType]) -> None:
    if not machine_types:
        raise ValueError("machine_types must not be empty")
    seen: set[str] = set()
    for machine_type in machine_types:
        if machine_type.type_id in seen:
            raise ValueError(f"duplicate type_id {machine_type.type_id!r}")
        if machine_type.copies < 0:
            raise ValueError(f"machine type {machine_type.type_id!r} has negative copies")
        if machine_type.capacity < 0:
            raise ValueError(
                f"machine type {machine_type.type_id!r} has negative capacity"
            )
        seen.add(machine_type.type_id)


def _validate_type_count_candidate(
    candidate: TypeCountCandidate,
    machine_types: Sequence[MachineType],
    type_ids: tuple[str, ...],
) -> None:
    if candidate.type_ids != type_ids:
        raise ValueError(
            f"candidate {candidate.candidate_id!r} type_ids do not match machine_types"
        )
    if len(candidate.counts) != len(machine_types):
        raise ValueError(
            f"candidate {candidate.candidate_id!r} has wrong count vector length"
        )
    for count, machine_type in zip(candidate.counts, machine_types):
        if count < 0:
            raise ValueError(f"candidate {candidate.candidate_id!r} has negative count")
        if count > machine_type.copies:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} uses {count} replicas of "
                f"type {machine_type.type_id!r}, but only {machine_type.copies} "
                "physical copies exist"
            )


def _validate_replica_count(k: int, machines: Sequence[Machine]) -> None:
    if k <= 0:
        raise ValueError(f"replica count must be positive, got {k}")
    if k > len(machines):
        raise ValueError(
            f"replica count {k} exceeds number of machines {len(machines)}"
        )


def _candidate_id(machine_ids: Iterable[str]) -> str:
    return ",".join(sorted(machine_ids))


def _type_count_candidate_id(type_ids: Sequence[str], counts: Sequence[int]) -> str:
    parts = [
        f"{type_id}:{count}"
        for type_id, count in zip(type_ids, counts)
        if count
    ]
    return ",".join(parts)


def _bounded_integer_vectors(
    total: int,
    bounds: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    if total < 0:
        return
    if not bounds:
        if total == 0:
            yield ()
        return

    first_bound = min(bounds[0], total)
    for value in range(first_bound + 1):
        for suffix in _bounded_integer_vectors(total - value, bounds[1:]):
            yield (value, *suffix)


def _stratified_sample(
    machines: Sequence[Machine],
    k: int,
    rng: random.Random,
) -> tuple[str, ...]:
    by_domain: dict[str, list[Machine]] = defaultdict(list)
    for machine in machines:
        by_domain[_primary_domain(machine)].append(machine)

    chosen: list[Machine] = []
    domains = list(by_domain)
    rng.shuffle(domains)
    for domain in domains:
        if len(chosen) >= k:
            break
        chosen.append(rng.choice(by_domain[domain]))

    remaining = [m for m in machines if m.machine_id not in {c.machine_id for c in chosen}]
    while len(chosen) < k:
        pick = rng.choice(remaining)
        chosen.append(pick)
        remaining = [m for m in remaining if m.machine_id != pick.machine_id]

    return tuple(sorted(m.machine_id for m in chosen))


def _greedy_candidate(
    machines: Sequence[Machine],
    k: int,
    rng: random.Random,
    config: CandidateGenerationConfig,
) -> tuple[str, ...]:
    machine_by_id = _machine_by_id(machines)
    selected = [rng.choice(machines).machine_id]
    while len(selected) < k:
        best: tuple[float, str] | None = None
        for machine in machines:
            if machine.machine_id in selected:
                continue
            trial = tuple(sorted([*selected, machine.machine_id]))
            score = _approximate_candidate_score(trial, machine_by_id, config)
            contender = (score, machine.machine_id)
            if best is None or contender > best:
                best = contender
        if best is None:
            raise ValueError("could not complete greedy candidate")
        selected.append(best[1])
    return tuple(sorted(selected))


def _mutate_one_machine(
    placement: tuple[str, ...],
    machines: Sequence[Machine],
    rng: random.Random,
) -> tuple[str, ...]:
    placement_set = set(placement)
    removable = rng.choice(placement)
    replacement_pool = [m.machine_id for m in machines if m.machine_id not in placement_set]
    if not replacement_pool:
        return placement
    replacement = rng.choice(replacement_pool)
    mutated = (placement_set - {removable}) | {replacement}
    return tuple(sorted(mutated))


def _score_raw_candidates(
    raw: Iterable[tuple[str, ...]],
    machine_by_id: Mapping[str, Machine],
    scorer: CandidateScorer,
) -> list[PlacementCandidate]:
    candidates: list[PlacementCandidate] = []
    for machine_ids in sorted(raw):
        machines = [machine_by_id[machine_id] for machine_id in machine_ids]
        score = scorer(machines)
        _validate_score(score, _candidate_id(machine_ids))
        candidates.append(
            PlacementCandidate(
                candidate_id=_candidate_id(machine_ids),
                machine_ids=tuple(machine_ids),
                availability=float(score.availability),
                cost_per_hour=float(score.cost_per_hour),
                metadata=dict(score.metadata),
            )
        )
    return candidates


def _validate_score(score: CandidateScore, candidate_id: str) -> None:
    if not 0.0 <= score.availability <= 1.0:
        raise ValueError(f"candidate {candidate_id} availability must be in [0, 1]")
    if score.cost_per_hour < 0:
        raise ValueError(f"candidate {candidate_id} has negative cost")


def _bucketed_pareto_filter(
    candidates: Sequence[PlacementCandidate],
) -> list[PlacementCandidate]:
    # Keep this deliberately conservative. A candidate that is dominated as a
    # standalone RSM placement may still be globally useful because it avoids
    # machines consumed by better-looking candidates.
    return list(candidates)


def _diversity_select(
    candidates: Sequence[PlacementCandidate],
    max_candidates: int,
    jaccard_threshold: float,
) -> list[PlacementCandidate]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if len(candidates) <= max_candidates:
        return list(candidates)

    ordered = sorted(
        candidates,
        key=lambda c: (c.availability, -c.cost_per_hour, -c.replica_count),
        reverse=True,
    )
    selected: list[PlacementCandidate] = []
    deferred: list[PlacementCandidate] = []
    for candidate in ordered:
        max_similarity = max(
            (_jaccard(candidate.machine_ids, picked.machine_ids) for picked in selected),
            default=0.0,
        )
        if max_similarity <= jaccard_threshold:
            selected.append(candidate)
            if len(selected) >= max_candidates:
                return selected
        else:
            deferred.append(candidate)

    for candidate in deferred:
        if len(selected) >= max_candidates:
            break
        selected.append(candidate)
    return selected


def _objective_value(candidate: PlacementCandidate, objective: Objective) -> float:
    if objective == "sum_availability":
        return candidate.availability
    if objective == "product_availability":
        return candidate.log_availability
    raise ValueError(f"unknown objective {objective!r}")


def _type_count_objective_value(
    candidate: TypeCountCandidate,
    objective: Objective,
) -> float:
    if objective == "sum_availability":
        return candidate.availability
    if objective == "product_availability":
        return candidate.log_availability
    raise ValueError(f"unknown objective {objective!r}")


def _approximate_candidate_score(
    machine_ids: tuple[str, ...],
    machine_by_id: Mapping[str, Machine],
    config: CandidateGenerationConfig,
) -> float:
    machines = [machine_by_id[machine_id] for machine_id in machine_ids]
    quorum = len(machines) // 2 + 1
    probabilities = [_approximate_node_availability(m.node_config) for m in machines]
    availability = _poisson_binomial_at_least(probabilities, quorum)
    spread = _domain_spread(machines)
    cost = sum(m.node_config.cost_per_hour for m in machines)
    return (
        math.log(max(availability, 1e-300))
        + config.spread_weight * spread
        - config.cost_weight * cost
    )


def _approximate_node_availability(config: NodeConfig) -> float:
    mtbf = float(config.failure_dist.mean)
    mttr = float(config.recovery_dist.mean)
    if mtbf <= 0:
        return 0.0
    if mttr <= 0:
        return 1.0
    return max(0.0, min(1.0, mtbf / (mtbf + mttr)))


def _poisson_binomial_at_least(probabilities: Sequence[float], threshold: int) -> float:
    dist = [1.0]
    for p in probabilities:
        next_dist = [0.0] * (len(dist) + 1)
        for successes, probability in enumerate(dist):
            next_dist[successes] += probability * (1.0 - p)
            next_dist[successes + 1] += probability * p
        dist = next_dist
    return float(sum(dist[threshold:]))


def _domain_spread(machines: Sequence[Machine]) -> float:
    if not machines:
        return 0.0
    primary_domains = {_primary_domain(machine) for machine in machines}
    return len(primary_domains) / len(machines)


def _primary_domain(machine: Machine) -> str:
    return f"region:{machine.node_config.region}"


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    lset = set(left)
    rset = set(right)
    union = lset | rset
    if not union:
        return 1.0
    return len(lset & rset) / len(union)
