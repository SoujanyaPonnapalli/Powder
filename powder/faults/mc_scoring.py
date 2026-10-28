"""Fixed-horizon Monte Carlo certification for catalogue placement candidates.

A Student-t interval decides whether availability clears a configurable k-nines
threshold. Bonferroni adjustment covers repeated inspection and the number of
shortlisted candidates. The minimum is 100,000 runs; an undecided candidate at
the run cap must not enter the ILP. A t interval still has no rare-event coverage
guarantee, so zero-variance perfect samples cannot certify a threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class MCScoringConfig:
    target_nines: float = 5.0
    markov_floor_nines: float = 4.5
    confidence: float = 0.99
    min_runs: int = 100_000
    max_runs: int = 10_000_000
    batch_runs: int = 10_000
    max_candidates: int = 100
    base_seed: int = 0
    include_region_outages: bool = True

    def __post_init__(self) -> None:
        if (not math.isfinite(self.target_nines) or self.target_nines < self.markov_floor_nines or
                self.target_nines > 15):
            raise ValueError("target_nines must be between the Markov floor and 15")
        if not math.isfinite(self.markov_floor_nines) or self.markov_floor_nines <= 0:
            raise ValueError("markov_floor_nines must be positive and finite")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if self.min_runs < 100_000 or self.max_runs < self.min_runs or self.batch_runs < 2:
            raise ValueError("MC scoring requires >=100000 runs and valid batch/max limits")
        if self.max_candidates < 1 or self.base_seed < 0:
            raise ValueError("max_candidates and base_seed must be nonnegative/positive")

    @property
    def target_availability(self) -> float:
        return 1.0 - 10.0 ** (-self.target_nines)

    @property
    def markov_floor(self) -> float:
        return 1.0 - 10.0 ** (-self.markov_floor_nines)


@dataclass(frozen=True)
class BatchSummary:
    count: int
    mean: float
    sample_std: float

    @classmethod
    def from_samples(cls, values: Sequence[float]) -> BatchSummary:
        samples = np.asarray(values, dtype=float)
        if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
            raise ValueError("MC batch needs at least two finite availability samples")
        if np.any((samples < 0) | (samples > 1)):
            raise ValueError("availability samples must lie in [0, 1]")
        return cls(int(samples.size), float(np.mean(samples)), float(np.std(samples, ddof=1)))

    def __post_init__(self) -> None:
        if (self.count < 2 or not math.isfinite(self.mean) or not 0 <= self.mean <= 1 or
                not math.isfinite(self.sample_std) or self.sample_std < 0):
            raise ValueError("invalid MC batch summary")


@dataclass
class OnlineTInterval:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, batch: BatchSummary) -> None:
        delta = batch.mean - self.mean
        total = self.count + batch.count
        self.m2 += (batch.count - 1) * batch.sample_std ** 2
        self.m2 += delta * delta * self.count * batch.count / total
        self.mean += delta * batch.count / total
        self.count = total

    @property
    def sample_std(self) -> float:
        return math.sqrt(max(0.0, self.m2 / (self.count - 1))) if self.count > 1 else float("nan")

    def interval(self, confidence: float) -> tuple[float, float]:
        if self.count < 2:
            raise ValueError("at least two samples required for a t interval")
        from scipy.stats import t

        half = float(t.ppf((1 + confidence) / 2, self.count - 1)) * self.sample_std / math.sqrt(self.count)
        return max(0.0, self.mean - half), min(1.0, self.mean + half)


@dataclass(frozen=True)
class MCDecision:
    status: Literal["pass", "fail", "undecided"]
    runs: int
    availability: float
    ci_low: float
    ci_high: float
    sample_std: float


def certify_availability(sample_batch: Callable[[int, int], BatchSummary], config: MCScoringConfig,
                         *, base_seed: int, comparisons: int = 1) -> MCDecision:
    """Sample disjoint seeds; Bonferroni-adjust t intervals for repeated looks/candidates."""
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    max_looks = math.ceil(config.max_runs / config.batch_runs)
    look_confidence = 1.0 - (1.0 - config.confidence) / (max_looks * comparisons)
    stats = OnlineTInterval()
    status: Literal["pass", "fail", "undecided"] = "undecided"
    while stats.count < config.max_runs:
        n = min(config.batch_runs, config.max_runs - stats.count)
        if n < 2:
            break
        batch = sample_batch(n, base_seed + stats.count)
        if batch.count != n:
            raise ValueError("MC sampler returned the wrong number of runs")
        stats.add(batch)
        if stats.count < config.min_runs:
            continue
        low, high = stats.interval(look_confidence)
        if high < config.target_availability:
            status = "fail"
            break
        if low >= config.target_availability and stats.sample_std > 0:
            status = "pass"
            break
    low, high = stats.interval(look_confidence)
    return MCDecision(status, stats.count, stats.mean, low, high, stats.sample_std)


def catalogue_batch_sampler(lib, cat: dict, machine_ids: Sequence[str], scenario_id: str,
                            price: str, protocol, strategy, *, include_region_outages: bool):
    """Bind catalogue hazard curves to one fixed-horizon Rust MC RSM job."""
    from powder.mc_backend import run_monte_carlo
    from powder.simulation import ClusterState, NetworkState, NodeState

    from . import catalogue as C
    from .convert import planned_as_cause, scenario_network_config

    scenario = lib.get(scenario_id, "scenario")
    planned = planned_as_cause(lib, scenario, scenario.get("level", "central"))
    by_id = {m["id"]: m for m in cat["machines"]}
    nodes = {}
    for i, mid in enumerate(machine_ids):
        m = by_id[mid]
        cost = m["price"].get("per_hour") if m["tier"] == "spot" else m["price"].get(price)
        if cost is None:
            raise ValueError(f"no {price} price for {mid}")
        cfg = C.node_config(m, horizon_h=cat["horizon_h"], planned=planned,
                            system=scenario.get("system"), time_varying=True,
                            cost_per_hour=cost)
        nodes[f"n{i}"] = NodeState(node_id=f"n{i}", config=cfg)
    cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=len(nodes))
    model = scenario_network_config(lib, scenario_id) if include_region_outages else None
    network = model.as_network_config(engine=True) if model is not None else None
    horizon_s = cat["horizon_h"] * 3600.0

    def sample(count: int, seed: int) -> BatchSummary:
        runs = run_monte_carlo(cluster, strategy, protocol, num_simulations=count,
                               max_time=horizon_s, stop_on_data_loss=False, seed=seed,
                               network_config=network)
        if len(runs.availability_samples) != count:
            raise ValueError("Rust MC returned fewer runs than requested")
        return BatchSummary.from_samples(runs.availability_samples)

    return sample
