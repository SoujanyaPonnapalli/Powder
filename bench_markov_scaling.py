"""Empirical scaling benchmark for the Markov solver.

Runs the steady-state solve across quality levels and cluster sizes,
reports state count, nnz, build/solve time, peak memory, and fit
exponent. Includes a heterogeneous sweep that varies the number of
rate classes C in {1, 2, N//2, N} for each quality level.
"""

import gc
import time
import tracemalloc

import numpy as np

from powder.markov_solver import steady_state
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    Constant,
    Exponential,
    LeaderlessProtocol,
    NodeConfig,
    NodeReplacementStrategy,
)
from powder.simulation.distributions import Seconds, days, hours, minutes


def _cfg(*, failure_hours: float = 12.0, region: str = "us-east") -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1.0 / hours(failure_hours)),
        recovery_dist=Exponential(rate=1.0 / minutes(10)),
        data_loss_dist=Exponential(rate=1.0 / days(365)),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(60),
        spawn_dist=Constant(300),
    )


CFG = _cfg()
PROTO = LeaderlessProtocol()
STRATEGY = NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))
PER_CONFIG_BUDGET_S = 30.0


def _configs_with_classes(n: int, c: int) -> list[NodeConfig]:
    """Return N configs partitioned into C rate classes of as-equal-as-possible size.

    Classes are made distinct by tweaking the failure rate; tests in
    ``tests/test_rate_signature.py`` ensure that a small rate delta
    places configs in different classes at every quality level.
    """
    if c < 1 or c > n:
        raise ValueError(f"c must be in [1, n]; got n={n} c={c}")
    base = 12.0
    step = 2.0  # hours per class step
    sizes = [n // c + (1 if i < n % c else 0) for i in range(c)]
    out: list[NodeConfig] = []
    for cls_idx, size in enumerate(sizes):
        cfg = _cfg(failure_hours=base + cls_idx * step, region=f"r{cls_idx}")
        out.extend([cfg] * size)
    return out


def bench(configs: list[NodeConfig], q: QualityLevel, reps: int = 2):
    t_builds, t_solves = [], []
    states = nnz = 0
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        model = build_markov_model(configs, PROTO, STRATEGY, q)
        t1 = time.perf_counter()
        steady_state(model)
        t2 = time.perf_counter()
        t_builds.append(t1 - t0)
        t_solves.append(t2 - t1)
        states, nnz = model.num_states, model.Q.nnz

    gc.collect()
    tracemalloc.start()
    model = build_markov_model(configs, PROTO, STRATEGY, q)
    steady_state(model)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return states, nnz, min(t_builds), min(t_solves), peak


def _print_row(
    quality_name: str, n: int, classes: int, states: int, nnz: int,
    tb: float, ts: float, mem: float, scale: str,
) -> None:
    print(
        f"{quality_name:<22} {n:>3} {classes:>3} {states:>8} {nnz:>10} "
        f"{tb*1000:>10.2f} {ts*1000:>10.2f} {mem/1e6:>9.2f} {scale:>12}",
        flush=True,
    )


def _print_header() -> None:
    print(
        f"{'Quality':<22} {'N':>3} {'C':>3} {'states':>8} {'nnz':>10} "
        f"{'build ms':>10} {'solve ms':>10} {'peak MB':>9} {'scale':>12}",
        flush=True,
    )


def run_homogeneous() -> None:
    print("=== Homogeneous sweep (C = 1) ===", flush=True)
    _print_header()
    for q in QualityLevel:
        prev_s = prev_t = None
        for n in range(3, 20):
            try:
                s, z, tb, ts, mem = bench([CFG] * n, q)
            except MemoryError:
                print(f"{q.name:<22} {n:>3}   1 OOM", flush=True)
                break
            if prev_s and ts > 1e-5 and prev_t > 1e-5:
                exp = np.log(ts / prev_t) / np.log(s / prev_s)
                scale = f"O(n^{exp:.2f})"
            else:
                scale = "-"
            _print_row(q.name, n, 1, s, z, tb, ts, mem, scale)
            prev_s, prev_t = s, ts
            if ts + tb > PER_CONFIG_BUDGET_S:
                break
        print(flush=True)


def _class_schedule(n: int) -> list[int]:
    """Pick C values in {1, 2, N//2, N} deduplicated for this N."""
    raw = [1, 2, n // 2, n]
    out: list[int] = []
    for c in raw:
        if c >= 1 and c <= n and c not in out:
            out.append(c)
    return out


def run_heterogeneous() -> None:
    print("=== Heterogeneous sweep (C in {1, 2, N/2, N}) ===", flush=True)
    _print_header()
    for q in QualityLevel:
        for n in (4, 6, 8, 10):
            for c in _class_schedule(n):
                cfgs = _configs_with_classes(n, c)
                try:
                    s, z, tb, ts, mem = bench(cfgs, q)
                except MemoryError:
                    print(f"{q.name:<22} {n:>3} {c:>3} OOM", flush=True)
                    continue
                _print_row(q.name, n, c, s, z, tb, ts, mem, "-")
                if ts + tb > PER_CONFIG_BUDGET_S:
                    break
            print(flush=True)
        print(flush=True)


def main() -> None:
    run_homogeneous()
    print(flush=True)
    run_heterogeneous()


if __name__ == "__main__":
    main()
