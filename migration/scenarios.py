"""Scenario definitions shared by both engines.

Each scenario is declared once here and knows how to produce *both* the
Python engine objects and the JSON job the Rust binary consumes.  That is
the point of this module: the two sides cannot drift apart, because there
is only one description of each scenario.

Part of the temporary migration harness -- see ``migration/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from powder.simulation import (
    AdaptiveReplacementStrategy,
    ClusterState,
    ClusterStrategy,
    Constant,
    Exponential,
    LeaderlessProtocol,
    NetworkConfig,
    NetworkState,
    NodeConfig,
    NodeReplacementStrategy,
    NodeState,
    NoOpStrategy,
    Normal,
    Protocol,
    RaftLikeProtocol,
    Uniform,
    Weibull,
    days,
    hours,
    minutes,
)

# ---------------------------------------------------------------------------
# Distribution specs
#
# A spec is a plain dict in the Rust JSON schema's shape.  `build_dist`
# turns it into the Python object, so neither side is written twice.
# ---------------------------------------------------------------------------


def exponential(rate: float) -> dict:
    return {"type": "exponential", "rate": rate}


def weibull(shape: float, scale: float) -> dict:
    return {"type": "weibull", "shape": shape, "scale": scale}


def normal(mean: float, std: float, min_val: float = 0.0) -> dict:
    return {"type": "normal", "mean": mean, "std": std, "min_val": min_val}


def uniform(low: float, high: float) -> dict:
    return {"type": "uniform", "low": low, "high": high}


def constant(value: float) -> dict:
    return {"type": "constant", "value": value}


def build_dist(spec: dict):
    """Instantiate the Python distribution described by *spec*."""
    kind = spec["type"]
    if kind == "exponential":
        return Exponential(rate=spec["rate"])
    if kind == "weibull":
        return Weibull(shape=spec["shape"], scale=spec["scale"])
    if kind == "normal":
        return Normal(
            mean=spec["mean"], std=spec["std"], min_val=spec.get("min_val", 0.0)
        )
    if kind == "uniform":
        return Uniform(low=spec["low"], high=spec["high"])
    if kind == "constant":
        return Constant(value=spec["value"])
    raise ValueError(f"unknown distribution type: {kind}")


def build_node_config(spec: dict) -> NodeConfig:
    """Instantiate the Python NodeConfig described by *spec*."""
    return NodeConfig(
        region=spec["region"],
        cost_per_hour=spec["cost_per_hour"],
        failure_dist=build_dist(spec["failure_dist"]),
        recovery_dist=build_dist(spec["recovery_dist"]),
        data_loss_dist=build_dist(spec["data_loss_dist"]),
        log_replay_rate_dist=build_dist(spec["log_replay_rate_dist"]),
        snapshot_download_time_dist=build_dist(spec["snapshot_download_time_dist"]),
        spawn_dist=build_dist(spec["spawn_dist"]),
    )


def node_config_spec(
    region: str = "us-east",
    cost_per_hour: float = 1.0,
    failure: dict | None = None,
    recovery: dict | None = None,
    data_loss: dict | None = None,
    log_replay_rate: dict | None = None,
    snapshot_download: dict | None = None,
    spawn: dict | None = None,
) -> dict:
    """A node config spec with far-future defaults for anything unset."""
    return {
        "region": region,
        "cost_per_hour": cost_per_hour,
        "failure_dist": failure or constant(days(3650)),
        "recovery_dist": recovery or constant(0.0),
        "data_loss_dist": data_loss or constant(days(3650)),
        "log_replay_rate_dist": log_replay_rate or constant(100.0),
        "snapshot_download_time_dist": snapshot_download or constant(0.0),
        "spawn_dist": spawn or constant(0.0),
    }


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One comparable workload.

    Attributes:
        name: Identifier used in test IDs and as the Rust job id.
        node_configs: Named node config specs.
        nodes: Active nodes as ``(node_id, config_name)`` pairs.
        standby_nodes: Standby nodes, same shape.
        provisioning_nodes: Provisioning nodes, same shape.
        target_cluster_size: Desired active node count.
        protocol: Protocol spec in the Rust JSON shape.
        strategy: Strategy spec in the Rust JSON shape.
        network: Optional network config spec.
        max_time: Time limit per simulation.
        stop_on_data_loss: Whether a run ends at data loss.
        num_simulations: Runs per experiment.
        deterministic: True when every distribution is Constant, so both
            engines must agree to a tight tolerance rather than only
            statistically.
    """

    name: str
    node_configs: dict[str, dict]
    nodes: list[tuple[str, str]]
    target_cluster_size: int
    protocol: dict
    strategy: dict
    max_time: float | None
    stop_on_data_loss: bool
    num_simulations: int
    deterministic: bool = False
    standby_nodes: list[tuple[str, str]] = field(default_factory=list)
    provisioning_nodes: list[tuple[str, str]] = field(default_factory=list)
    network: dict | None = None
    active_outages: list[str] = field(default_factory=list)

    # -- Rust side ------------------------------------------------------

    def to_job(self, base_seed: int, mode: str = "monte_carlo") -> dict:
        """The JSON job for the Rust binary."""
        def node_entries(pairs, available=True, has_data=True):
            return [
                {
                    "node_id": node_id,
                    "config": config_name,
                    "is_available": available,
                    "has_data": has_data,
                }
                for node_id, config_name in pairs
            ]

        job: dict[str, Any] = {
            "job_id": self.name,
            "mode": mode,
            "node_configs": self.node_configs,
            "cluster": {
                "target_cluster_size": self.target_cluster_size,
                "nodes": node_entries(self.nodes),
                "standby_nodes": node_entries(self.standby_nodes),
                "provisioning_nodes": node_entries(
                    self.provisioning_nodes, available=False, has_data=False
                ),
                "active_outages": self.active_outages,
            },
            "protocol": self.protocol,
            "strategy": self.strategy,
            "run": {
                "max_time": self.max_time,
                "stop_on_data_loss": self.stop_on_data_loss,
                "num_simulations": self.num_simulations,
                "base_seed": base_seed,
            },
        }
        if self.network is not None:
            job["network_config"] = self.network
        return job

    # -- Python side ----------------------------------------------------

    def build_cluster(self) -> ClusterState:
        """A fresh Python ClusterState for this scenario."""
        configs = {
            name: build_node_config(spec) for name, spec in self.node_configs.items()
        }

        nodes = {
            node_id: NodeState(node_id=node_id, config=configs[config_name])
            for node_id, config_name in self.nodes
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=self.target_cluster_size,
        )

        for node_id, config_name in self.standby_nodes:
            cluster.add_standby_node(
                NodeState(node_id=node_id, config=configs[config_name])
            )
        for node_id, config_name in self.provisioning_nodes:
            cluster.add_provisioning_node(
                NodeState(
                    node_id=node_id,
                    config=configs[config_name],
                    is_available=False,
                    has_data=False,
                )
            )
        for region in self.active_outages:
            cluster.network.add_outage(region)

        return cluster

    def build_protocol(self) -> Protocol:
        """A fresh Python protocol for this scenario."""
        spec = self.protocol
        if spec["type"] == "leaderless":
            return LeaderlessProtocol(
                commit_rate=spec.get("commit_rate", 1.0),
                snapshot_interval=spec.get("snapshot_interval", 0.0),
                log_retention_ops=spec.get("log_retention_ops", 0.0),
                up_to_date_quorum=spec.get("up_to_date_quorum", True),
            )
        if spec["type"] == "raft":
            return RaftLikeProtocol(
                election_time_dist=build_dist(spec["election_time_dist"]),
                commit_rate=spec.get("commit_rate", 1.0),
                snapshot_interval=spec.get("snapshot_interval", 0.0),
                log_retention_ops=spec.get("log_retention_ops", 0.0),
            )
        raise ValueError(f"unknown protocol type: {spec['type']}")

    def build_strategy(self) -> ClusterStrategy:
        """A fresh Python strategy for this scenario."""
        spec = self.strategy
        kind = spec["type"]
        if kind == "noop":
            return NoOpStrategy()

        default_name = spec.get("default_node_config")
        default_config = (
            build_node_config(self.node_configs[default_name]) if default_name else None
        )

        if kind == "node_replacement":
            return NodeReplacementStrategy(
                failure_timeout=spec["failure_timeout"],
                default_node_config=default_config,
                safe_mode=spec.get("safe_mode", True),
            )
        if kind == "adaptive_replacement":
            reconfig = spec["reconfiguration_dist"]
            return AdaptiveReplacementStrategy(
                failure_timeout=spec["failure_timeout"],
                reconfiguration_dist=(
                    reconfig
                    if isinstance(reconfig, (int, float))
                    else build_dist(reconfig)
                ),
                scale_down_threshold=spec.get("scale_down_threshold", 2),
                external_consensus=spec.get("external_consensus", False),
                default_node_config=default_config,
                safe_mode=spec.get("safe_mode", True),
            )
        raise ValueError(f"unknown strategy type: {kind}")

    def build_network_config(self) -> NetworkConfig | None:
        """A fresh Python NetworkConfig, or None when unset."""
        if self.network is None:
            return None
        return NetworkConfig(
            outage_dist=build_dist(self.network["outage_dist"]),
            outage_duration_dist=build_dist(self.network["outage_duration_dist"]),
            regions=list(self.network.get("regions", [])),
        )


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

LEADERLESS = {"type": "leaderless"}
LEADERLESS_ANY_QUORUM = {"type": "leaderless", "up_to_date_quorum": False}
RAFT = {"type": "raft", "election_time_dist": constant(minutes(1))}
NOOP = {"type": "noop"}


def _nodes(count: int, config_name: str = "standard") -> list[tuple[str, str]]:
    return [(f"node{i}", config_name) for i in range(count)]


# ---------------------------------------------------------------------------
# Deterministic scenarios
#
# Every distribution is Constant, so the engines run identical event
# sequences and are compared at a tight tolerance rather than statistically.
# ---------------------------------------------------------------------------

DETERMINISTIC_SCENARIOS: list[Scenario] = [
    Scenario(
        name="det_staggered_failures",
        node_configs={
            "flaky": node_config_spec(
                failure=constant(hours(2)),
                recovery=constant(hours(1)),
                log_replay_rate=constant(3.0),
            ),
            "stable": node_config_spec(log_replay_rate=constant(3.0)),
        },
        nodes=[("node0", "flaky"), ("node1", "flaky"), ("node2", "stable")],
        target_cluster_size=3,
        protocol=LEADERLESS,
        strategy=NOOP,
        max_time=days(2),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_cascading_data_loss",
        node_configs={
            "loss1": node_config_spec(data_loss=constant(hours(1))),
            "loss2": node_config_spec(data_loss=constant(hours(2))),
            "loss3": node_config_spec(data_loss=constant(hours(3))),
        },
        nodes=[("node0", "loss1"), ("node1", "loss2"), ("node2", "loss3")],
        target_cluster_size=3,
        protocol=LEADERLESS,
        strategy=NOOP,
        max_time=hours(4),
        stop_on_data_loss=True,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_raft_leader_election",
        node_configs={
            "flaky": node_config_spec(
                failure=constant(hours(1)), recovery=constant(hours(1))
            ),
            "stable": node_config_spec(),
        },
        nodes=[("node0", "flaky"), ("node1", "stable"), ("node2", "stable")],
        target_cluster_size=3,
        protocol={"type": "raft", "election_time_dist": constant(10.0)},
        strategy=NOOP,
        max_time=hours(4),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_snapshot_forced_sync",
        node_configs={
            "slow_snapshot": node_config_spec(
                failure=constant(hours(1)),
                recovery=constant(minutes(30)),
                log_replay_rate=constant(10.0),
                snapshot_download=constant(60.0),
            ),
        },
        nodes=_nodes(3, "slow_snapshot"),
        target_cluster_size=3,
        protocol={
            "type": "leaderless",
            "commit_rate": 1.0,
            "snapshot_interval": 100.0,
            "log_retention_ops": 100.0,
        },
        strategy=NOOP,
        max_time=hours(12),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_node_replacement",
        node_configs={
            "lossy": node_config_spec(
                data_loss=constant(hours(2)), spawn=constant(minutes(5))
            ),
            "stable": node_config_spec(spawn=constant(minutes(5))),
        },
        nodes=[("node0", "lossy"), ("node1", "stable"), ("node2", "stable")],
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy={
            "type": "node_replacement",
            "failure_timeout": minutes(10),
            "default_node_config": "stable",
        },
        max_time=hours(8),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_region_outage",
        node_configs={
            "east": node_config_spec(region="us-east"),
            "west": node_config_spec(region="us-west"),
        },
        nodes=[("node0", "east"), ("node1", "east"), ("node2", "west")],
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy=NOOP,
        network={
            "outage_dist": constant(hours(2)),
            "outage_duration_dist": constant(minutes(30)),
            "regions": ["us-east"],
        },
        max_time=hours(12),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_adaptive_scaling",
        node_configs={
            "flaky": node_config_spec(
                failure=constant(hours(1)),
                recovery=constant(hours(2)),
                spawn=constant(minutes(5)),
            ),
            "stable": node_config_spec(spawn=constant(minutes(5))),
        },
        nodes=[
            ("node0", "flaky"),
            ("node1", "flaky"),
            ("node2", "stable"),
            ("node3", "stable"),
            ("node4", "stable"),
        ],
        target_cluster_size=5,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy={
            "type": "adaptive_replacement",
            "failure_timeout": minutes(20),
            "reconfiguration_dist": minutes(2),
            "scale_down_threshold": 2,
            "external_consensus": False,
            "default_node_config": "stable",
        },
        max_time=hours(12),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_standby_and_provisioning",
        node_configs={
            "standard": node_config_spec(
                failure=constant(hours(3)),
                recovery=constant(minutes(20)),
                spawn=constant(minutes(5)),
            ),
        },
        nodes=_nodes(3),
        standby_nodes=[("spare0", "standard")],
        provisioning_nodes=[("pending0", "standard")],
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy={
            "type": "node_replacement",
            "failure_timeout": minutes(10),
            "default_node_config": "standard",
        },
        max_time=hours(12),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_starts_in_outage",
        node_configs={"east": node_config_spec(region="us-east")},
        nodes=_nodes(3, "east"),
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy=NOOP,
        active_outages=["us-east"],
        network={
            "outage_dist": constant(hours(10)),
            "outage_duration_dist": constant(hours(1)),
            "regions": ["us-east"],
        },
        max_time=hours(6),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        # Ends on data loss after only *two* loss events: the third node is
        # transiently down and lagging when the second one dies, so no node
        # holds the committed data.  The stochastic layer cannot resolve
        # this path (it fires in well under 1% of runs), so it is pinned
        # deterministically instead.
        name="det_data_loss_with_lagging_survivor",
        node_configs={
            "loses_first": node_config_spec(
                data_loss=constant(200.0), log_replay_rate=constant(1e6)
            ),
            "loses_second": node_config_spec(
                data_loss=constant(300.0), log_replay_rate=constant(1e6)
            ),
            "down_early": node_config_spec(
                failure=constant(100.0),
                recovery=constant(100_000.0),
                log_replay_rate=constant(1e6),
            ),
        },
        nodes=[
            ("nodeA", "loses_first"),
            ("nodeB", "loses_second"),
            ("nodeC", "down_early"),
        ],
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy=NOOP,
        max_time=100_000.0,
        stop_on_data_loss=True,
        num_simulations=1,
        deterministic=True,
    ),
    Scenario(
        name="det_no_events_runs_dry",
        node_configs={"inert": node_config_spec()},
        nodes=_nodes(3, "inert"),
        target_cluster_size=3,
        protocol=LEADERLESS,
        strategy=NOOP,
        max_time=hours(1),
        stop_on_data_loss=False,
        num_simulations=1,
        deterministic=True,
    ),
]


# ---------------------------------------------------------------------------
# Stochastic scenarios
#
# Compared statistically: the two engines draw from different RNG streams,
# so only their distributions should agree.
# ---------------------------------------------------------------------------

STOCHASTIC_SCENARIOS: list[Scenario] = [
    Scenario(
        name="sto_leaderless_3node",
        node_configs={
            "standard": node_config_spec(
                failure=exponential(1.0 / hours(12)),
                recovery=constant(minutes(10)),
                log_replay_rate=constant(1e6),
            ),
        },
        nodes=_nodes(3),
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy=NOOP,
        max_time=days(7),
        stop_on_data_loss=False,
        num_simulations=400,
    ),
    Scenario(
        name="sto_leaderless_5node_uptodate",
        node_configs={
            "standard": node_config_spec(
                failure=exponential(1.0 / hours(8)),
                recovery=exponential(1.0 / minutes(15)),
                log_replay_rate=constant(50.0),
            ),
        },
        nodes=_nodes(5),
        target_cluster_size=5,
        protocol=LEADERLESS,
        strategy=NOOP,
        max_time=days(7),
        stop_on_data_loss=False,
        num_simulations=400,
    ),
    Scenario(
        name="sto_raft_5node",
        node_configs={
            "standard": node_config_spec(
                failure=exponential(1.0 / hours(10)),
                recovery=exponential(1.0 / minutes(10)),
                log_replay_rate=constant(100.0),
            ),
        },
        nodes=_nodes(5),
        target_cluster_size=5,
        protocol=RAFT,
        strategy=NOOP,
        max_time=days(7),
        stop_on_data_loss=False,
        num_simulations=400,
    ),
    Scenario(
        name="sto_all_distribution_families",
        node_configs={
            "mixed": node_config_spec(
                cost_per_hour=0.192,
                failure=weibull(1.3, hours(10)),
                recovery=normal(minutes(12), minutes(3)),
                data_loss=exponential(1.0 / days(120)),
                log_replay_rate=normal(80.0, 10.0),
                snapshot_download=uniform(30.0, 90.0),
                spawn=constant(minutes(2)),
            ),
        },
        nodes=_nodes(5, "mixed"),
        target_cluster_size=5,
        protocol={
            "type": "leaderless",
            "snapshot_interval": 500.0,
            "log_retention_ops": 2000.0,
        },
        strategy=NOOP,
        max_time=days(14),
        stop_on_data_loss=False,
        num_simulations=400,
    ),
    Scenario(
        name="sto_replacement_strategy",
        node_configs={
            "standard": node_config_spec(
                cost_per_hour=0.5,
                failure=exponential(1.0 / hours(6)),
                recovery=exponential(1.0 / minutes(20)),
                data_loss=exponential(1.0 / days(30)),
                log_replay_rate=constant(500.0),
                spawn=constant(minutes(3)),
            ),
        },
        nodes=_nodes(5),
        target_cluster_size=5,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy={
            "type": "node_replacement",
            "failure_timeout": minutes(30),
            "default_node_config": "standard",
        },
        max_time=days(14),
        stop_on_data_loss=False,
        num_simulations=300,
    ),
    Scenario(
        name="sto_adaptive_strategy",
        node_configs={
            "standard": node_config_spec(
                cost_per_hour=0.5,
                failure=exponential(1.0 / hours(6)),
                recovery=exponential(1.0 / minutes(20)),
                log_replay_rate=constant(500.0),
                spawn=constant(minutes(3)),
            ),
        },
        nodes=_nodes(5),
        target_cluster_size=5,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy={
            "type": "adaptive_replacement",
            "failure_timeout": minutes(30),
            "reconfiguration_dist": exponential(1.0 / minutes(2)),
            "scale_down_threshold": 2,
            "external_consensus": True,
            "default_node_config": "standard",
        },
        max_time=days(7),
        stop_on_data_loss=False,
        num_simulations=300,
    ),
    Scenario(
        name="sto_region_outages",
        node_configs={
            "east": node_config_spec(
                region="us-east",
                failure=exponential(1.0 / hours(24)),
                recovery=exponential(1.0 / minutes(15)),
            ),
            "west": node_config_spec(
                region="us-west",
                failure=exponential(1.0 / hours(24)),
                recovery=exponential(1.0 / minutes(15)),
            ),
        },
        nodes=[
            ("node0", "east"),
            ("node1", "east"),
            ("node2", "east"),
            ("node3", "west"),
            ("node4", "west"),
        ],
        target_cluster_size=5,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy=NOOP,
        network={
            "outage_dist": exponential(1.0 / days(3)),
            "outage_duration_dist": exponential(1.0 / minutes(45)),
            "regions": ["us-east", "us-west"],
        },
        max_time=days(14),
        stop_on_data_loss=False,
        num_simulations=300,
    ),
    Scenario(
        name="sto_mttdl_until_data_loss",
        node_configs={
            "fragile": node_config_spec(
                failure=exponential(1.0 / hours(4)),
                recovery=exponential(1.0 / hours(1)),
                data_loss=exponential(1.0 / days(5)),
                log_replay_rate=constant(1e6),
            ),
        },
        nodes=_nodes(3, "fragile"),
        target_cluster_size=3,
        protocol=LEADERLESS_ANY_QUORUM,
        strategy=NOOP,
        # No time limit: every run goes to data loss, which is what MTTDL
        # estimation needs.
        max_time=None,
        stop_on_data_loss=True,
        num_simulations=400,
    ),
]


ALL_SCENARIOS = DETERMINISTIC_SCENARIOS + STOCHASTIC_SCENARIOS
