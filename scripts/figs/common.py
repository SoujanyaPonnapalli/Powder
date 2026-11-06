"""Shared setup for the paper-figure scripts (docs/plans/figs-01-whatif-modeling-scale-sensitivity.md).

The setup matches scripts/pilot_fleet_frontier.py: the 2000-machine catalogue, the AZ-aware scenario, Raft with
Exp(0.2/s) elections, and node replacement (timeout configurable here). Machine configs are the scaled,
horizon-average (constant-rate) catalogue configs the optimizer scored.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CATALOGUE = "data/fault_library/catalogue/machines-od1000-spot1000-seed0-az.json.gz"
SCENARIO = "scenario/canonical-central-az-aware@v3"
DAY_S = 86_400.0


def nines(a: float) -> float:
    return -math.log10(1 - a) if a < 1 else math.inf


@lru_cache(maxsize=None)
def library():
    from powder.faults import FaultLibrary
    return FaultLibrary.load()


@lru_cache(maxsize=None)
def catalogue() -> dict:
    from powder.faults import catalogue as C
    return C.load(ROOT / CATALOGUE)


@lru_cache(maxsize=None)
def machines(rate_scale: float, price: str, capacity: int = 5) -> dict:
    """machine id -> optimizer Machine (scaled node config, cost_per_hour under ``price``)."""
    from powder.faults.ilp import catalogue_machines
    ms = catalogue_machines(library(), catalogue(), scenario_id=SCENARIO, price=price, capacity=capacity,
                            rate_scale=rate_scale)
    return {m.machine_id: m for m in ms}


def protocol(election_timeout_s: float | None = None, vote_round_s: float = 0.1):
    """Raft. Default: the pipeline's Exp(0.2/s) election time (mean 5 s). With ``election_timeout_s`` = T:
    a follower times out after a randomized Uniform(T, 2T) (Raft's randomized timeout), plus one vote round."""
    from powder.simulation import Exponential, RaftLikeProtocol, Uniform
    if election_timeout_s is None:
        return RaftLikeProtocol(election_time_dist=Exponential(0.2))
    T = float(election_timeout_s)
    return RaftLikeProtocol(election_time_dist=Uniform(T + vote_round_s, 2 * T + vote_round_s))


def strategy(timeout_s: float = 600.0):
    from powder.simulation import NodeReplacementStrategy
    return NodeReplacementStrategy(failure_timeout=float(timeout_s), default_node_config=None)


@lru_cache(maxsize=None)
def network():
    from powder.faults.convert import scenario_network_config
    return scenario_network_config(library(), SCENARIO).as_network_config(engine=True)


def placement_seed(ids) -> int:
    """Stable per-placement seed (same as pilot_fleet_frontier.Exact), so every variant of a study that
    re-scores the same placement uses common random numbers."""
    return int.from_bytes(hashlib.blake2b("|".join(sorted(ids)).encode(), digest_size=6).digest(), "big")


@dataclass
class MCStats:
    runs: int
    mean: float              # availability
    std: float               # per-run sample std
    cost_mean: float         # $ per run over the horizon (replacements billed by the engine)
    spawned_mean: float      # replacement nodes per run
    seconds: float
    elections_mean: float = 0.0  # leader elections per run

    @property
    def se(self) -> float:
        return self.std / math.sqrt(self.runs)


def mc_stats(configs, *, timeout_s: float, horizon_s: float, runs: int, seed: int, chunk: int = 100_000,
             with_network: bool = True, proto=None) -> MCStats:
    """Rust MC of one RSM, in chunks of ``chunk`` runs on disjoint seed ranges (bounded memory)."""
    import time

    import numpy as np

    from powder.mc_backend import run_monte_carlo
    from powder.simulation import ClusterState, NetworkState, NodeState

    nodes = {f"n{i}": NodeState(node_id=f"n{i}", config=c) for i, c in enumerate(configs)}
    cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=len(nodes))
    t0, n, s1, s2, c1, sp, el = time.time(), 0, 0.0, 0.0, 0.0, 0.0, 0.0
    proto = proto or protocol()
    while n < runs:
        k = min(chunk, runs - n)
        r = run_monte_carlo(cluster, strategy(timeout_s), proto, num_simulations=k, max_time=horizon_s,
                            stop_on_data_loss=False, seed=seed + n, network_config=network() if with_network else None)
        a = np.asarray(r.availability_samples, dtype=float)
        if a.size != k:
            raise ValueError("Rust MC returned fewer runs than requested")
        s1, s2 = s1 + a.sum(), s2 + (a * a).sum()
        c1 += float(np.sum(r.cost_samples))
        sp += float(np.sum(r.nodes_spawned_samples))
        el += float(np.sum(r.leader_election_samples))
        n += k
    mean = s1 / n
    var = max(0.0, (s2 - n * mean * mean) / (n - 1))
    return MCStats(n, mean, math.sqrt(var), c1 / n, sp / n, time.time() - t0, el / n)


def load_row(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def deployment(row: dict, budget: str) -> list[tuple[str, ...]]:
    """The saved deployment (one sorted machine-id tuple per RSM) of a fleet-pilot row at a budget key."""
    r = row["new_ilp"][budget]
    if "placements" not in r:
        raise KeyError(f"row {row['rate_scale']}x {row['price']} budget {budget} has no saved placements "
                       "(rerun scripts/pilot_fleet_frontier.py, which now saves them)")
    return [tuple(sorted(p)) for p in r["placements"]]
