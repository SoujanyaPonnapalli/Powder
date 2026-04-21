"""
Markov chain state-space builders at multiple quality levels.

Provides leaderless and Raft model builders that accept the same
NodeConfig/Protocol/Strategy objects used by the Monte Carlo simulator.

Inputs are a ``list[NodeConfig]`` that may contain heterogeneous
rates; builders partition configs into rate classes via
``group_configs_into_classes`` and emit per-class transitions. When
every config is identical the resulting state space reduces to the
original homogeneous layout.
"""

from .leaderless import build_leaderless_model
from .raft import build_raft_model
from .common import (
    ExtractedRates,
    RateClass,
    build_model_bfs,
    extract_rates,
    group_configs_into_classes,
    heterogeneous_cost_fn,
    weak_compositions,
)

__all__ = [
    "build_leaderless_model",
    "build_raft_model",
    "ExtractedRates",
    "RateClass",
    "extract_rates",
    "group_configs_into_classes",
    "heterogeneous_cost_fn",
    "weak_compositions",
]
