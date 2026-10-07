#!/usr/bin/env python
"""Benchmark the Python engine against the Rust port.

Runs identical scenarios through both and reports wall-clock time,
simulations per second and simulated events per second, for the Python
engine (single process and multiprocess) and the Rust binary (one worker
and full width).

Both sides are driven from ``scenarios.py``, so they are doing the same
work, and both produce full per-run output.  Each scenario is split into
several jobs on the Rust side, because a job is its unit of parallelism;
one job per scenario would measure job imbalance rather than throughput.

Usage::

    .venv/bin/python migration/bench.py            # full workload
    .venv/bin/python migration/bench.py --quick    # smoke test
    .venv/bin/python migration/bench.py --sims 4000

Part of the temporary migration harness -- see ``migration/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from powder.monte_carlo import MonteCarloConfig, MonteCarloRunner  # noqa: E402

from scenarios import STOCHASTIC_SCENARIOS, Scenario  # noqa: E402

BINARY = REPO_ROOT / "rust" / "target" / "release" / "powder-mc"
BASE_SEED = 777


def build_binary() -> None:
    print("Building the release binary ...", flush=True)
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "powder-mc"],
        cwd=REPO_ROOT / "rust",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"cargo build failed:\n{result.stdout}\n{result.stderr}")


def run_python(
    scenarios: list[Scenario], sims: int, workers: int
) -> tuple[float, int]:
    """Wall-clock seconds and simulated events for the Python engine."""
    events = 0
    start = time.perf_counter()
    for scenario in scenarios:
        config = MonteCarloConfig(
            num_simulations=sims,
            max_time=scenario.max_time,
            stop_on_data_loss=scenario.stop_on_data_loss,
            parallel_workers=workers,
            base_seed=BASE_SEED,
        )
        results = MonteCarloRunner(config).run(
            cluster=scenario.build_cluster(),
            strategy=scenario.build_strategy(),
            protocol=scenario.build_protocol(),
            network_config=scenario.build_network_config(),
        )
        events += sum(results.transient_failure_samples)
        events += sum(results.dataloss_failure_samples)
        events += sum(results.nodes_spawned_samples)
    return time.perf_counter() - start, events


def run_rust(
    scenarios: list[Scenario], sims: int, workers: int, shards: int
) -> tuple[float, int]:
    """Wall-clock seconds and simulated events for the Rust binary.

    Each scenario is split into `shards` jobs so the pool has more jobs
    than workers -- a job is the unit of parallelism, so one job per
    scenario would leave most workers idle and measure job imbalance
    rather than throughput.  The shards cover disjoint seed ranges, so
    together they are the same work the Python side does in one call.
    Process startup is included, since that is what a caller pays; it
    measures under 10 ms.
    """
    per_shard = sims // shards
    jobs = []
    for scenario in scenarios:
        for shard in range(shards):
            job = scenario.to_job(BASE_SEED + shard * per_shard)
            job["run"]["num_simulations"] = per_shard
            job["job_id"] = f"{scenario.name}#{shard}"
            jobs.append(json.dumps(job))
    payload = "\n".join(jobs) + "\n"

    start = time.perf_counter()
    result = subprocess.run(
        [str(BINARY), "--stream", "-j", str(workers), "--batch-size", "16"],
        input=payload,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise SystemExit(f"powder-mc failed:\n{result.stderr}")

    events = 0
    produced = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        produced += 1
        payload_out = json.loads(line)
        for run in payload_out["runs"]:
            events += (
                run["total_transient_failures"]
                + run["total_dataloss_failures"]
                + run["total_nodes_spawned"]
            )
    if produced != len(jobs):
        raise SystemExit(f"expected {len(jobs)} results, got {produced}")
    return elapsed, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller workload")
    parser.add_argument("--sims", type=int, default=None, help="simulations per scenario")
    args = parser.parse_args()

    sims = args.sims or (40 if args.quick else 1000)
    scenarios = STOCHASTIC_SCENARIOS[:3] if args.quick else STOCHASTIC_SCENARIOS
    cpu_count = os.cpu_count() or 1
    # Enough shards that the pool always has more jobs than workers.
    shards = max(4, 4 * cpu_count // len(scenarios))
    sims = (sims // shards) * shards  # keep the split exact

    build_binary()

    total_sims = sims * len(scenarios)
    print()
    print(
        f"Workload: {len(scenarios)} scenarios x {sims} simulations "
        f"= {total_sims} simulations"
    )
    print(f"Rust splits each scenario into {shards} jobs ({len(scenarios) * shards} total)")
    print(f"Host: {cpu_count} logical CPUs")
    print()

    measurements = []

    print("Python, 1 process ...", flush=True)
    t, ev = run_python(scenarios, sims, workers=1)
    measurements.append(("Python, 1 process", t, ev))

    print(f"Python, {cpu_count} processes ...", flush=True)
    t, ev = run_python(scenarios, sims, workers=cpu_count)
    measurements.append((f"Python, {cpu_count} processes", t, ev))

    print("Rust, -j 1 ...", flush=True)
    t, ev = run_rust(scenarios, sims, workers=1, shards=shards)
    measurements.append(("Rust, -j 1", t, ev))

    print(f"Rust, -j {cpu_count} ...", flush=True)
    t, ev = run_rust(scenarios, sims, workers=cpu_count, shards=shards)
    measurements.append((f"Rust, -j {cpu_count}", t, ev))

    py_serial = measurements[0][1]

    print()
    header = (
        f"{'engine':<26}{'wall (s)':>10}{'sims/s':>11}"
        f"{'events/s':>13}{'vs Python 1proc':>18}"
    )
    print(header)
    print("-" * len(header))
    for label, elapsed, events in measurements:
        print(
            f"{label:<26}{elapsed:>10.2f}{total_sims / elapsed:>11.0f}"
            f"{events / elapsed:>13.0f}{py_serial / elapsed:>17.1f}x"
        )

    py_best = min(m[1] for m in measurements[:2])
    rs_best = min(m[1] for m in measurements[2:])
    rs_serial = measurements[2][1]
    rs_parallel = measurements[3][1]

    print()
    print(f"Single-threaded:  {py_serial / rs_serial:>6.1f}x")
    print(f"Best vs best:     {py_best / rs_best:>6.1f}x")
    print(f"Rust 1 -> {cpu_count}:      {rs_serial / rs_parallel:>6.1f}x")
    print()
    print(
        "Both engines produce full per-run output, so the comparison "
        "includes\naggregating every sample, not just the simulation loop."
    )


if __name__ == "__main__":
    main()
