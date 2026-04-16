"""
Shared scenario configuration: build a Markov model from MC simulation objects.

This module bridges the Monte Carlo and Markov backends by accepting the
same NodeConfig, Protocol, and ClusterStrategy objects used by the simulator
and converting them into a MarkovModel at a requested quality level.
"""

from __future__ import annotations

from enum import IntEnum

from .markov import MarkovModel
from .simulation.node import NodeConfig
from .simulation.protocol import Protocol, RaftLikeProtocol
from .simulation.strategy import ClusterStrategy


class QualityLevel(IntEnum):
    """State-space fidelity for the Markov model.

    Higher levels track more per-node detail (replacement pipeline stages,
    lagging nodes, orphaned replacements) at the cost of a larger state space.
    See docs/markov_state_analysis.md for the full breakdown.

    Levels (leaderless per-node states / Raft adds leader dimension):
        SIMPLIFIED:         k=3  (H, F, D)
        COLLAPSED_PIPELINE: k=4  (H, F, L, D)
        NO_ORPHANS:         k=6  (H, Fw, Fe_R, L, Dw, De_R)
        MERGED_PIPELINE:    k=8  (H, H_R, Fw, Fe_R, L, L_R, Dw, De_R)
        FULL:               k=12 (all pipeline + timer + availability states)
    """

    SIMPLIFIED = 0
    COLLAPSED_PIPELINE = 1
    NO_ORPHANS = 2
    MERGED_PIPELINE = 3
    FULL = 4


def build_markov_model(
    node_configs: list[NodeConfig],
    protocol: Protocol,
    strategy: ClusterStrategy,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> MarkovModel:
    """Generate a CTMC from the same objects used by MonteCarloRunner.run().

    Extracts exponential rates from Distribution.mean on each NodeConfig,
    protocol parameters (election time for Raft), and strategy parameters
    (replacement timeout) to build the appropriate state space.

    Args:
        node_configs: One NodeConfig per node in the cluster.
        protocol: The consensus protocol (LeaderlessProtocol or RaftLikeProtocol).
        strategy: The cluster management strategy (NoOpStrategy or NodeReplacementStrategy).
        quality: State-space fidelity level.

    Returns:
        A MarkovModel with a sparse Q matrix, ready for the solver.
    """
    if isinstance(protocol, RaftLikeProtocol):
        from .markov_builders.raft import build_raft_model

        return build_raft_model(node_configs, protocol, strategy, quality)
    else:
        from .markov_builders.leaderless import build_leaderless_model

        return build_leaderless_model(node_configs, protocol, strategy, quality)
