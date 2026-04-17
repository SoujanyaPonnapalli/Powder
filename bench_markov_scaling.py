"""Empirical scaling benchmark for the Markov solver.

Runs the steady-state solve across quality levels and cluster sizes,
reports state count, nnz, build/solve time, peak memory, and fit exponent.
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

CFG = NodeConfig(
    region="us-east",
    cost_per_hour=1.0,
    failure_dist=Exponential(rate=1.0 / hours(12)),
    recovery_dist=Exponential(rate=1.0 / minutes(10)),
    data_loss_dist=Exponential(rate=1.0 / days(365)),
    log_replay_rate_dist=Constant(1e6),
    snapshot_download_time_dist=Constant(60),
    spawn_dist=Constant(300),
)
PROTO = LeaderlessProtocol()
STRATEGY = NodeReplacementStrategy(failure_timeout=Seconds(hours(1)))
PER_CONFIG_BUDGET_S = 30.0


def bench(n: int, q: QualityLevel, reps: int = 2):
    t_builds, t_solves = [], []
    states = nnz = 0
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        model = build_markov_model([CFG] * n, PROTO, STRATEGY, q)
        t1 = time.perf_counter()
        steady_state(model)
        t2 = time.perf_counter()
        t_builds.append(t1 - t0)
        t_solves.append(t2 - t1)
        states, nnz = model.num_states, model.Q.nnz

    gc.collect()
    tracemalloc.start()
    model = build_markov_model([CFG] * n, PROTO, STRATEGY, q)
    steady_state(model)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return states, nnz, min(t_builds), min(t_solves), peak


def main() -> None:
    print(
        f"{'Quality':<22} {'N':>3} {'states':>8} {'nnz':>10} "
        f"{'build ms':>10} {'solve ms':>10} {'peak MB':>9} {'scale':>12}",
        flush=True,
    )
    for q in QualityLevel:
        prev_s = prev_t = None
        for n in range(3, 20):
            try:
                s, z, tb, ts, mem = bench(n, q)
            except MemoryError:
                print(f"{q.name:<22} {n:>3} OOM", flush=True)
                break
            if prev_s and ts > 1e-5 and prev_t > 1e-5:
                exp = np.log(ts / prev_t) / np.log(s / prev_s)
                scale = f"O(n^{exp:.2f})"
            else:
                scale = "-"
            print(
                f"{q.name:<22} {n:>3} {s:>8} {z:>10} "
                f"{tb*1000:>10.2f} {ts*1000:>10.2f} {mem/1e6:>9.2f} {scale:>12}",
                flush=True,
            )
            prev_s, prev_t = s, ts
            if ts + tb > PER_CONFIG_BUDGET_S:
                break
        print(flush=True)


if __name__ == "__main__":
    main()
