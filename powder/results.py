"""
Unified result types for Markov and Monte Carlo analyses.

Provides ClusterAnalysisResult as a common interface and markov_analyze()
as a one-call convenience for Markov chain analysis using MC objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .markov import MarkovModel
from .markov_solver import (
    availability,
    expected_cost_per_second,
    mean_first_passage,
    steady_state,
)
from .scenario import QualityLevel, build_markov_model
from .simulation.node import NodeConfig
from .simulation.protocol import Protocol
from .simulation.strategy import ClusterStrategy

if TYPE_CHECKING:
    from .monte_carlo import MonteCarloResults


@dataclass
class ClusterAnalysisResult:
    """Common result type for both Markov and Monte Carlo analyses.

    Covers availability, reliability (MTTF/MTTDL), and expected cost.
    Fields that are not computed or not applicable for a given backend
    are left as None.

    Attributes:
        availability: Steady-state (Markov) or mean (MC) availability fraction.
        mean_time_to_unavailability: Mean time from the initial state to
            first unavailability, in seconds.
        mean_time_to_data_loss: Mean time from the initial state to first
            data loss (absorbing state), in seconds.
        expected_cost_per_hour: Steady-state expected $ spent per hour.
        steady_state_distribution: Mapping from state name to steady-state
            probability (Markov only).
        method: Which backend produced this result.
        quality_level: QualityLevel used (Markov only).
        num_states: Number of states in the Markov chain (Markov only).
    """

    availability: float
    mean_time_to_unavailability: float | None = None
    mean_time_to_data_loss: float | None = None
    expected_cost_per_hour: float | None = None
    steady_state_distribution: dict[str, float] | None = None
    method: Literal["markov", "monte_carlo"] = "markov"
    quality_level: QualityLevel | None = None
    num_states: int | None = None


def markov_analyze(
    node_configs: list[NodeConfig],
    protocol: Protocol,
    strategy: ClusterStrategy,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> ClusterAnalysisResult:
    """One-call Markov analysis using MC's own configuration objects.

    Builds the CTMC and solves for steady-state availability, expected
    cost, and (when possible) mean first passage to unavailability.

    Args:
        node_configs: One NodeConfig per node in the cluster.
        protocol: The consensus protocol.
        strategy: The cluster management strategy.
        quality: State-space fidelity level.

    Returns:
        A ClusterAnalysisResult with Markov-derived metrics.
    """
    model = build_markov_model(node_configs, protocol, strategy, quality)
    return analyze_model(model, quality=quality)


def analyze_model(
    model: MarkovModel,
    *,
    quality: QualityLevel | None = None,
) -> ClusterAnalysisResult:
    """Compute availability, MTTF, and expected cost for a prebuilt model.

    Useful when the caller wants to reuse the same MarkovModel across
    multiple analyses (e.g. cost-vs-availability studies) without
    re-running BFS state enumeration.
    """
    try:
        pi = steady_state(model)
    except RuntimeError:
        pi = None

    if pi is not None:
        avail = availability(model, pi)
        cost_per_second = expected_cost_per_second(model, pi)
        cost_per_hour = cost_per_second * 3600.0
        ss_dist = {
            name: float(prob) for name, prob in zip(model.state_names, pi)
        }
    else:
        avail = float("nan")
        cost_per_hour = None
        ss_dist = None

    mttf: float | None = None
    dead_ids = np.flatnonzero(~model.live_mask)
    if dead_ids.size > 0 and dead_ids.size < model.num_states:
        try:
            mfp = mean_first_passage(model, dead_ids.tolist())
            initial = model.initial_state_id
            value = float(mfp[initial])
            mttf = value if value > 0 else None
        except RuntimeError:
            pass

    return ClusterAnalysisResult(
        availability=avail,
        mean_time_to_unavailability=mttf,
        expected_cost_per_hour=cost_per_hour,
        steady_state_distribution=ss_dist,
        method="markov",
        quality_level=quality,
        num_states=model.num_states,
    )


def monte_carlo_analyze(results: "MonteCarloResults") -> ClusterAnalysisResult:
    """Adapt MonteCarloResults into the unified ClusterAnalysisResult shape.

    Maps MC aggregate statistics onto the shared schema so Markov and
    Monte Carlo outputs can be compared field-by-field:

    - availability = mean of per-run availability fractions
    - mean_time_to_unavailability = mean of per-run time_to_first_unavailability
      across runs that observed an unavailability event (None if no runs did)
    - mean_time_to_data_loss = mean time to actual data loss among runs that
      observed it (None if no runs did)
    - expected_cost_per_hour = sum(cost_samples) / sum(total_time_samples),
      i.e. time-weighted average cost rate. Falls back to None if total time
      was not recorded.

    Markov-specific fields (steady_state_distribution, quality_level,
    num_states) are left as None. The `method` field is set to "monte_carlo".
    """
    availability = results.availability_mean()

    unavail_samples = [
        float(t) for t in results.time_to_first_unavailability_samples if t is not None
    ]
    mttf = float(np.mean(unavail_samples)) if unavail_samples else None

    mttdl = results.mean_time_to_actual_loss()

    cost_per_hour: float | None = None
    if results.cost_samples and results.total_time_samples:
        total_cost = float(np.sum(results.cost_samples))
        total_seconds = float(np.sum(results.total_time_samples))
        if total_seconds > 0:
            cost_per_hour = total_cost / (total_seconds / 3600.0)

    return ClusterAnalysisResult(
        availability=availability,
        mean_time_to_unavailability=mttf,
        mean_time_to_data_loss=mttdl,
        expected_cost_per_hour=cost_per_hour,
        steady_state_distribution=None,
        method="monte_carlo",
        quality_level=None,
        num_states=None,
    )
