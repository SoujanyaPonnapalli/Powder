"""Translate simulation objects into jobs for the Rust Monte Carlo engine.

The Monte Carlo engine lives in `rust/` (see `rust/README.md`). Python
still owns the scenario vocabulary -- `NodeConfig`, `Distribution`,
`Protocol`, `ClusterStrategy` -- because the Markov backend and the
placement optimizer are built on the same objects. This module converts
those objects into the JSON the binary reads, and finds the binary.

Nothing here simulates anything; it is purely a translation layer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .simulation.cluster import ClusterState
from .simulation.distributions import (
    Constant,
    Distribution,
    Exponential,
    Normal,
    Uniform,
    Weibull,
)
from .simulation.network import NetworkConfig
from .simulation.node import NodeConfig
from .simulation.protocol import LeaderlessProtocol, Protocol, RaftLikeProtocol
from .simulation.strategy import (
    AdaptiveReplacementStrategy,
    ClusterStrategy,
    NodeReplacementStrategy,
    NoOpStrategy,
)

#: JSON has no infinity, and the schema takes finite seconds.  Scenarios
#: express "never happens" as ``Constant(inf)``; this stands in for it at
#: roughly 3e10 years, which outlasts any simulation horizon.
_NEVER = 1e18

_BINARY_NAME = "powder-mc"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class EngineNotFound(RuntimeError):
    """The Rust Monte Carlo binary could not be located."""


def binary_path() -> Path:
    """Locate the `powder-mc` binary.

    Checks ``POWDER_MC_BINARY``, then the release build under ``rust/``,
    then ``PATH``.

    Raises:
        EngineNotFound: with build instructions, if none of those hit.
    """
    override = os.environ.get("POWDER_MC_BINARY")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise EngineNotFound(
            f"POWDER_MC_BINARY points at {override}, which is not a file"
        )

    built = _REPO_ROOT / "rust" / "target" / "release" / _BINARY_NAME
    if built.is_file():
        return built

    on_path = shutil.which(_BINARY_NAME)
    if on_path:
        return Path(on_path)

    raise EngineNotFound(
        "the Monte Carlo engine binary was not found.\n"
        "Build it with:\n"
        "    cargo build --release --manifest-path rust/Cargo.toml\n"
        "or point POWDER_MC_BINARY at an existing build."
    )


def ensure_binary() -> Path:
    """Return the binary path, building it once if cargo is available."""
    try:
        return binary_path()
    except EngineNotFound:
        if shutil.which("cargo") is None:
            raise
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(_REPO_ROOT / "rust" / "Cargo.toml"),
                "--bin",
                _BINARY_NAME,
            ],
            check=True,
            capture_output=True,
        )
        return binary_path()


# ---------------------------------------------------------------------------
# Object -> JSON
# ---------------------------------------------------------------------------


def distribution_spec(dist: Distribution) -> dict[str, Any]:
    """Convert a `Distribution` into its JSON form."""
    if isinstance(dist, Exponential):
        return {"type": "exponential", "rate": dist.rate}
    if isinstance(dist, Weibull):
        return {"type": "weibull", "shape": dist.shape, "scale": dist.scale}
    if isinstance(dist, Normal):
        return {
            "type": "normal",
            "mean": dist.mean,
            "std": dist.std,
            "min_val": dist.min_val,
        }
    if isinstance(dist, Uniform):
        return {"type": "uniform", "low": dist.low, "high": dist.high}
    if isinstance(dist, Constant):
        value = dist.value
        return {"type": "constant", "value": _NEVER if value == float("inf") else value}
    raise TypeError(f"no JSON form for distribution {type(dist).__name__}")


def node_config_spec(config: NodeConfig) -> dict[str, Any]:
    """Convert a `NodeConfig` into its JSON form."""
    return {
        "region": config.region,
        "cost_per_hour": config.cost_per_hour,
        "failure_dist": distribution_spec(config.failure_dist),
        "recovery_dist": distribution_spec(config.recovery_dist),
        "data_loss_dist": distribution_spec(config.data_loss_dist),
        "log_replay_rate_dist": distribution_spec(config.log_replay_rate_dist),
        "snapshot_download_time_dist": distribution_spec(
            config.snapshot_download_time_dist
        ),
        "spawn_dist": distribution_spec(config.spawn_dist),
    }


def protocol_spec(protocol: Protocol) -> dict[str, Any]:
    """Convert a `Protocol` into its JSON form."""
    if isinstance(protocol, RaftLikeProtocol):
        return {
            "type": "raft",
            "election_time_dist": distribution_spec(protocol.election_time_dist),
            "commit_rate": protocol.commit_rate,
            "snapshot_interval": protocol.snapshot_interval,
            "log_retention_ops": protocol.log_retention_ops,
        }
    if isinstance(protocol, LeaderlessProtocol):
        return {
            "type": "leaderless",
            "commit_rate": protocol.commit_rate,
            "snapshot_interval": protocol.snapshot_interval,
            "log_retention_ops": protocol.log_retention_ops,
            "up_to_date_quorum": protocol.up_to_date_quorum,
        }
    raise TypeError(
        f"{type(protocol).__name__} has no JSON form. The Rust engine "
        "implements the leaderless and Raft protocols; a custom protocol "
        "would have to be added there."
    )


def _delay_spec(delay: Any) -> Any:
    """A reconfiguration delay: a bare number, or a distribution."""
    if isinstance(delay, Distribution):
        return distribution_spec(delay)
    return float(delay)


def strategy_spec(
    strategy: ClusterStrategy, config_names: dict[int, str]
) -> dict[str, Any]:
    """Convert a `ClusterStrategy` into its JSON form.

    `config_names` maps `id(NodeConfig)` to the name it was registered
    under, so a replacement template can be referenced rather than
    repeated.
    """
    if isinstance(strategy, NoOpStrategy):
        return {"type": "noop"}

    default = getattr(strategy, "default_node_config", None)
    default_name = config_names.get(id(default)) if default is not None else None

    if isinstance(strategy, AdaptiveReplacementStrategy):
        return {
            "type": "adaptive_replacement",
            "failure_timeout": strategy.failure_timeout,
            "reconfiguration_dist": _delay_spec(strategy.reconfiguration_dist),
            "scale_down_threshold": strategy.scale_down_threshold,
            "external_consensus": strategy.external_consensus,
            "default_node_config": default_name,
            "safe_mode": strategy.safe_mode,
        }
    if isinstance(strategy, NodeReplacementStrategy):
        return {
            "type": "node_replacement",
            "failure_timeout": strategy.failure_timeout,
            "default_node_config": default_name,
            "safe_mode": strategy.safe_mode,
        }
    raise TypeError(
        f"{type(strategy).__name__} has no JSON form. The Rust engine "
        "implements the no-op, replacement and adaptive strategies; a "
        "custom strategy would have to be added there."
    )


def network_config_spec(network: NetworkConfig) -> dict[str, Any]:
    """Convert a `NetworkConfig` into its JSON form."""
    return {
        "outage_dist": distribution_spec(network.outage_dist),
        "outage_duration_dist": distribution_spec(network.outage_duration_dist),
        "regions": list(network.regions),
    }


def build_job(
    cluster: ClusterState,
    strategy: ClusterStrategy,
    protocol: Protocol,
    *,
    num_simulations: int,
    max_time: float | None,
    stop_on_data_loss: bool,
    base_seed: int | None,
    network_config: NetworkConfig | None = None,
    mode: str = "monte_carlo",
    job_id: str | None = None,
) -> dict[str, Any]:
    """Assemble a complete job for the Rust engine.

    Node configurations are registered once and referenced by name, so a
    heterogeneous cluster stays compact and a homogeneous one carries a
    single config.
    """
    config_names: dict[int, str] = {}
    node_configs: dict[str, Any] = {}

    def register(config: NodeConfig) -> str:
        key = id(config)
        if key not in config_names:
            name = f"cfg{len(config_names)}"
            config_names[key] = name
            node_configs[name] = node_config_spec(config)
        return config_names[key]

    def entries(nodes) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node.node_id,
                "config": register(node.config),
                "is_available": node.is_available,
                "has_data": node.has_data,
                "last_applied_index": node.last_applied_index,
                "last_snapshot_index": node.last_snapshot_index,
            }
            for node in nodes
        ]

    active = entries(cluster.nodes.values())
    standby = entries(cluster.standby_nodes.values())
    provisioning = entries(cluster.provisioning_nodes.values())

    # The strategy may reference a replacement template that no current
    # node uses, so register it before emitting the strategy block.
    default = getattr(strategy, "default_node_config", None)
    if isinstance(default, NodeConfig):
        register(default)

    job: dict[str, Any] = {
        "mode": mode,
        "node_configs": node_configs,
        "cluster": {
            "target_cluster_size": cluster.target_cluster_size,
            "nodes": active,
            "standby_nodes": standby,
            "provisioning_nodes": provisioning,
            "active_outages": sorted(cluster.network.active_outages),
            "current_time": cluster.current_time,
            "commit_index": cluster.commit_index,
        },
        "protocol": protocol_spec(protocol),
        "strategy": strategy_spec(strategy, config_names),
        "run": {
            "max_time": max_time,
            "stop_on_data_loss": stop_on_data_loss,
            "num_simulations": num_simulations,
            "base_seed": base_seed,
        },
    }
    if job_id is not None:
        job["job_id"] = job_id
    if network_config is not None:
        job["network_config"] = network_config_spec(network_config)
    return job
