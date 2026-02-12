"""
Monte Carlo simulation package for RSM deployments.

This package provides discrete-event simulation capabilities for analyzing
replicated state machine availability and data loss scenarios.
"""

from .distributions import (
    Seconds,
    hours,
    days,
    minutes,
    Distribution,
    Exponential,
    Weibull,
    Normal,
    Uniform,
    Constant,
)
from .node import NodeConfig, NodeState, SyncState
from .network import NetworkConfig, NetworkState
from .events import EventType, Event, EventQueue
from .cluster import ClusterState
from .strategy import (
    ClusterStrategy,
    Action,
    ActionType,
    SimpleReplacementStrategy,
    NodeReplacementStrategy,
    NoOpStrategy,
)
from .protocol import (
    Protocol,
    LeaderlessUpToDateQuorumProtocol,
    LeaderlessMajorityAvailableProtocol,
    RaftLikeProtocol,
)
from .simulator import Simulator, SimulationResult
from .metrics import MetricsCollector, MetricsSnapshot

__all__ = [
    # Time units
    "Seconds",
    "hours",
    "days",
    "minutes",
    # Distributions
    "Distribution",
    "Exponential",
    "Weibull",
    "Normal",
    "Uniform",
    "Constant",
    # Node
    "NodeConfig",
    "NodeState",
    "SyncState",
    # Network
    "NetworkConfig",
    "NetworkState",
    # Events
    "EventType",
    "Event",
    "EventQueue",
    # Cluster
    "ClusterState",
    # Strategy
    "ClusterStrategy",
    "Action",
    "ActionType",
    "SimpleReplacementStrategy",
    "NodeReplacementStrategy",
    "NoOpStrategy",
    # Protocol
    "Protocol",
    "LeaderlessUpToDateQuorumProtocol",
    "LeaderlessMajorityAvailableProtocol",
    "RaftLikeProtocol",
    # Simulator
    "Simulator",
    "SimulationResult",
    # Metrics
    "MetricsCollector",
    "MetricsSnapshot",
]
