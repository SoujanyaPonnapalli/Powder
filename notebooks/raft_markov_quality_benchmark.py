"""
Benchmark: accuracy and runtime of Raft Markov models at each quality level.

Sweeps cluster sizes N in {3, 5, 7}, all 5 QualityLevel settings, and three
VM profiles that share the same transient-failure behaviour (MTBF 30d,
20min recovery) but differ in how often they lose data and in price:

  - Standard:   MTTDL 3 years,  cost $0.10/hr.
  - Spot:       MTTDL 1 day,    cost $0.03/hr.
  - Unreliable: MTTDL 1 year,   cost $0.05/hr.

For each (N, profile, quality):
  1. Build the Raft CTMC.
  2. Solve for steady-state availability and expected cost/hour.
  3. Time the build and solve phases (median of several repeats).

Accuracy is reported two ways:
  - Deviation from QualityLevel.FULL for the same (N, profile).
    FULL is analytically exact for these exponential-rate scenarios, so it
    serves as the reference within the Markov family.
  - Deviation from a Monte Carlo run, which acts as a ground-truth sanity
    check that our Markov math matches simulation.

The cluster strategy is NodeReplacementStrategy with a 5-minute failure
timeout: once a node stays unreachable past the timeout, a fresh VM is
provisioned and rejoins the quorum. This is the regime where the five
quality levels actually diverge -- SIMPLIFIED collapses the entire
spawn + sync pipeline to an instantaneous transition, while FULL models
each phase separately and even tracks orphaned pipelines after a
replacement is elected leader mid-flight.

Output: writes `notebooks/raft_markov_quality_benchmark.json`, which the
accompanying canvas embeds inline for display.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from powder.monte_carlo import MonteCarloConfig, MonteCarloRunner
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    ClusterState,
    Constant,
    Exponential,
    NetworkState,
    NodeConfig,
    NodeReplacementStrategy,
    NodeState,
    RaftLikeProtocol,
    Seconds,
)
from powder.simulation.distributions import days, minutes
from powder.markov_solver import (
    availability,
    expected_cost_per_second,
    steady_state,
)


# ---------------------------------------------------------------------------
# VM profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VMProfile:
    name: str
    description: str
    mean_time_between_failures_s: float
    mean_recovery_time_s: float
    mean_time_between_data_loss_s: float
    cost_per_hour: float


VM_PROFILES: list[VMProfile] = [
    VMProfile(
        name="Standard",
        description="Reliable on-demand VM. MTBF 30d, 20 min recovery, MTTDL 3y.",
        mean_time_between_failures_s=days(30),
        mean_recovery_time_s=minutes(20),
        mean_time_between_data_loss_s=days(365 * 3),
        cost_per_hour=0.10,
    ),
    VMProfile(
        name="Spot",
        description="Preemptible VM. MTBF 30d, 20 min recovery, MTTDL 1d.",
        mean_time_between_failures_s=days(30),
        mean_recovery_time_s=minutes(20),
        mean_time_between_data_loss_s=days(1),
        cost_per_hour=0.03,
    ),
    VMProfile(
        name="Unreliable",
        description="Flaky / budget VM. MTBF 30d, 20 min recovery, MTTDL 1y.",
        mean_time_between_failures_s=days(30),
        mean_recovery_time_s=minutes(20),
        mean_time_between_data_loss_s=days(365),
        cost_per_hour=0.05,
    ),
]


CLUSTER_SIZES = (3, 5, 7)
ELECTION_MEAN_S = 5.0
FAILURE_TIMEOUT_S = minutes(5)
MC_NUM_SIMS = 80
MC_DURATION_S = days(365)
RUNTIME_REPEATS = 5
# Cap the state count we'll actually solve. At higher N and higher quality
# levels the Raft CTMC can produce tens of thousands of states; the direct
# sparse LU behind steady_state() is roughly O(n^3) for these high-dimensional
# state graphs (see bench_markov_scaling.py), so anything past ~30k states
# takes hours to finish. We emit a skip record instead of stalling.
MAX_STATES_FOR_SOLVE = 30_000


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def node_config_for(profile: VMProfile) -> NodeConfig:
    return NodeConfig(
        region="us-east",
        cost_per_hour=profile.cost_per_hour,
        failure_dist=Exponential(rate=1.0 / profile.mean_time_between_failures_s),
        recovery_dist=Exponential(rate=1.0 / profile.mean_recovery_time_s),
        data_loss_dist=Exponential(rate=1.0 / profile.mean_time_between_data_loss_s),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(1.0),
        spawn_dist=Exponential(rate=1.0 / 60.0),
    )


def make_cluster(num_nodes: int, cfg: NodeConfig) -> ClusterState:
    nodes = {
        f"node{i}": NodeState(node_id=f"node{i}", config=cfg)
        for i in range(num_nodes)
    }
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=num_nodes,
    )


def raft_protocol() -> RaftLikeProtocol:
    return RaftLikeProtocol(
        election_time_dist=Exponential(rate=1.0 / ELECTION_MEAN_S),
    )


def replacement_strategy(cfg: NodeConfig) -> NodeReplacementStrategy:
    return NodeReplacementStrategy(
        failure_timeout=Seconds(FAILURE_TIMEOUT_S),
        default_node_config=cfg,
    )


# ---------------------------------------------------------------------------
# Timed Markov run
# ---------------------------------------------------------------------------


@dataclass
class MarkovRun:
    num_nodes: int
    profile: str
    quality: str
    num_states: int
    availability: float | None
    expected_cost_per_hour: float | None
    median_build_ms: float
    median_solve_ms: float | None
    median_total_ms: float | None
    skipped: bool = False
    skip_reason: str | None = None


def time_markov(
    num_nodes: int,
    profile: VMProfile,
    quality: QualityLevel,
    repeats: int = RUNTIME_REPEATS,
) -> MarkovRun:
    cfg = node_config_for(profile)
    prot = raft_protocol()
    strat = replacement_strategy(cfg)
    configs = [cfg] * num_nodes

    t0 = time.perf_counter()
    model = build_markov_model(configs, prot, strat, quality)
    t1 = time.perf_counter()
    first_build_ms = (t1 - t0) * 1000.0

    if model.num_states > MAX_STATES_FOR_SOLVE:
        return MarkovRun(
            num_nodes=num_nodes,
            profile=profile.name,
            quality=quality.name,
            num_states=model.num_states,
            availability=None,
            expected_cost_per_hour=None,
            median_build_ms=first_build_ms,
            median_solve_ms=None,
            median_total_ms=None,
            skipped=True,
            skip_reason=(
                f"{model.num_states} states exceeds MAX_STATES_FOR_SOLVE="
                f"{MAX_STATES_FOR_SOLVE}"
            ),
        )

    build_times: list[float] = [first_build_ms]
    solve_times: list[float] = []

    t_s0 = time.perf_counter()
    pi = steady_state(model)
    t_s1 = time.perf_counter()
    solve_times.append((t_s1 - t_s0) * 1000.0)

    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        model = build_markov_model(configs, prot, strat, quality)
        t1 = time.perf_counter()
        pi = steady_state(model)
        t2 = time.perf_counter()
        build_times.append((t1 - t0) * 1000.0)
        solve_times.append((t2 - t1) * 1000.0)

    avail = availability(model, pi)
    cost_per_hour = expected_cost_per_second(model, pi) * 3600.0
    return MarkovRun(
        num_nodes=num_nodes,
        profile=profile.name,
        quality=quality.name,
        num_states=model.num_states,
        availability=avail,
        expected_cost_per_hour=cost_per_hour,
        median_build_ms=statistics.median(build_times),
        median_solve_ms=statistics.median(solve_times),
        median_total_ms=statistics.median(
            b + s for b, s in zip(build_times, solve_times)
        ),
    )


# ---------------------------------------------------------------------------
# Monte Carlo cross-check
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloRun:
    num_nodes: int
    profile: str
    availability_mean: float
    availability_std: float
    num_sims: int
    wall_ms: float


def time_monte_carlo(num_nodes: int, profile: VMProfile) -> MonteCarloRun:
    cfg = node_config_for(profile)
    prot = raft_protocol()
    strat = replacement_strategy(cfg)

    runner = MonteCarloRunner(
        MonteCarloConfig(
            num_simulations=MC_NUM_SIMS,
            max_time=Seconds(MC_DURATION_S),
            stop_on_data_loss=False,
            parallel_workers=os.cpu_count() or 1,
            base_seed=42_000 + num_nodes * 1000 + (hash(profile.name) % 997),
        ),
    )
    t0 = time.perf_counter()
    results = runner.run(
        cluster=make_cluster(num_nodes, cfg),
        strategy=strat,
        protocol=prot,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    samples = results.availability_samples
    mean = float(sum(samples) / len(samples))
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return MonteCarloRun(
        num_nodes=num_nodes,
        profile=profile.name,
        availability_mean=mean,
        availability_std=std,
        num_sims=len(samples),
        wall_ms=wall_ms,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> dict:
    markov_runs: list[MarkovRun] = []
    for N in CLUSTER_SIZES:
        for profile in VM_PROFILES:
            for quality in QualityLevel:
                print(
                    f"[markov] N={N}  profile={profile.name:<10}  quality={quality.name}",
                    flush=True,
                )
                run = time_markov(N, profile, quality)
                if run.skipped:
                    print(
                        f"          SKIPPED: {run.skip_reason}",
                        flush=True,
                    )
                else:
                    print(
                        f"          states={run.num_states:>6}  "
                        f"avail={run.availability:.6f}  "
                        f"$/hr={run.expected_cost_per_hour:.4f}  "
                        f"build={run.median_build_ms:.1f}ms  "
                        f"solve={run.median_solve_ms:.1f}ms",
                        flush=True,
                    )
                markov_runs.append(run)

    mc_runs: list[MonteCarloRun] = []
    for N in CLUSTER_SIZES:
        for profile in VM_PROFILES:
            print(f"[mc] N={N}  profile={profile.name}", flush=True)
            run = time_monte_carlo(N, profile)
            print(
                f"     avail={run.availability_mean:.6f} "
                f"+/-{run.availability_std:.6f}  "
                f"sims={run.num_sims}  wall={run.wall_ms/1000.0:.1f}s",
                flush=True,
            )
            mc_runs.append(run)

    doc = {
        "config": {
            "cluster_sizes": list(CLUSTER_SIZES),
            "election_mean_s": ELECTION_MEAN_S,
            "mc_num_sims": MC_NUM_SIMS,
            "mc_duration_s": MC_DURATION_S,
            "runtime_repeats": RUNTIME_REPEATS,
            "strategy": "NodeReplacementStrategy",
            "failure_timeout_s": FAILURE_TIMEOUT_S,
            "max_states_for_solve": MAX_STATES_FOR_SOLVE,
        },
        "profiles": [asdict(p) for p in VM_PROFILES],
        "markov_runs": [asdict(r) for r in markov_runs],
        "monte_carlo_runs": [asdict(r) for r in mc_runs],
    }

    out_path = Path(__file__).resolve().parent / "raft_markov_quality_benchmark.json"
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {out_path}")
    return doc


if __name__ == "__main__":
    main()
