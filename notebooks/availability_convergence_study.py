"""Reproduce and extend the availability convergence study on one worker.

Run from the repository root:
    .venv/bin/python -m notebooks.availability_convergence_study --suite original
    .venv/bin/python -m notebooks.availability_convergence_study --suite validation
    .venv/bin/python -m notebooks.availability_convergence_study --suite markov

Original inline source and historical outputs are preserved in the adjacent
availability_convergence_original.json. Replay a named experiment using
--replay NAME (the child process is bounded by --timeout seconds).
The narrative REPORT.md is manually edited; generated evidence is JSON/Markdown.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before importing NumPy/SciPy; each MC batch is also sequential.
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.stats import norm, t

from notebooks.raft_markov_quality_benchmark import (
    VM_PROFILES, make_cluster, node_config_for, raft_protocol, replacement_strategy,
)
from powder.markov_solver import availability, steady_state
from powder.monte_carlo import MonteCarloConfig, MonteCarloRunner
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import Seconds
from powder.simulation.distributions import days

ROOT = Path(__file__).resolve().parents[1]
TARGET = (0.999990, 0.999999)
CONFIDENCE = 0.99
EPSILON = 0.000005


def projected_runs(sd: float, margin: float, minimum: int = 30) -> int | None:
    """Smallest fixed-variance Student-t plan; not a convergence guarantee."""
    if margin <= 0:
        return None
    def fits(n: int) -> bool:
        return float(t.ppf(0.995, n - 1)) * sd / math.sqrt(n) <= margin
    low, high = minimum, minimum
    while not fits(high):
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if fits(middle):
            high = middle
        else:
            low = middle + 1
    return low


def summarize(samples: np.ndarray, elapsed: float) -> dict:
    count = len(samples)
    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1))
    half = float(t.ppf(0.995, count - 1)) * sd / math.sqrt(count)
    ci = [mean - half, mean + half]
    needed = math.ceil((float(norm.ppf(0.995)) * sd / EPSILON) ** 2)
    total_seconds = needed * elapsed / count
    return {
        "runs": count, "mean": mean, "std": sd, "ci99": ci,
        "half_width": half,
        "target_contained": bool(TARGET[0] <= ci[0] and ci[1] <= TARGET[1]),
        "half_width_satisfied": half <= EPSILON,
        "rounded_5dp_agrees": f"{max(0, ci[0]):.5f}" == f"{min(1, ci[1]):.5f}",
        "below_99pct": int(np.count_nonzero(samples < 0.99)),
        "below_90pct": int(np.count_nonzero(samples < 0.9)),
        "perfect_runs": int(np.count_nonzero(samples == 1)),
        "quantiles": dict(zip(
            ("0", "0.001", "0.01", "0.05", "0.5", "0.95", "0.99", "0.999", "1"),
            np.quantile(samples, [0, .001, .01, .05, .5, .95, .99, .999, 1]).tolist(),
        )),
        "wall_seconds": elapsed, "seconds_per_run": elapsed / count,
        "planned_target_runs": projected_runs(sd, min(mean - TARGET[0], TARGET[1] - mean)),
        "planned_precision_runs_normal": needed,
        "projected_python_seconds": total_seconds,
        "projected_rust_seconds_20_to_50x": [total_seconds / 50, total_seconds / 20],
    }


def simulate(profile: str, nodes: int, count: int, horizon: int, seed: int, label: str) -> dict:
    cfg = node_config_for(next(p for p in VM_PROFILES if p.name == profile))
    runner = MonteCarloRunner(MonteCarloConfig(
        num_simulations=count, max_time=Seconds(days(horizon)),
        stop_on_data_loss=False, parallel_workers=1, base_seed=seed,
    ))
    start = time.perf_counter()
    results = runner.run(make_cluster(nodes, cfg), replacement_strategy(cfg), raft_protocol())
    elapsed = time.perf_counter() - start
    samples = np.asarray(results.availability_samples, dtype=np.float64)
    row = {"label": label, "profile": profile, "nodes": nodes,
           "horizon_days": horizon, "base_seed": seed, **summarize(samples, elapsed)}
    row["sample_sha256"] = hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest()
    # Preserve the observed lower tail so new claims can be audited by seed.
    row["tail_samples_below_99pct"] = [
        {"seed": seed + int(i), "availability": float(samples[i])}
        for i in np.flatnonzero(samples < .99)
    ]
    if label == "validation_batches":
        row["batches"] = [
            {"base_seed": seed + i, **summarize(samples[i:i + 1000], elapsed * 1000 / count)}
            for i in range(0, count, 1000)
        ]
        row["passing_batches"] = sum(b["target_contained"] for b in row["batches"])
    return row


def original_cases():
    for nodes in (3, 5, 7):
        yield "Standard", nodes, 1000, 365, 73000 + nodes * 1000, "annual_pilot"
        yield "Standard", nodes, 10000, 7, 173000 + nodes * 10000, "weekly_pilot"
        for batch in range(10):
            yield "Standard", nodes, 1000, 7, 904000 + nodes * 10000 + batch * 1000, "weekly_batch"
    for index, profile in enumerate(VM_PROFILES[1:], start=1):
        for nodes in (3, 5, 7):
            yield profile.name, nodes, 1000, 7, 1204000 + index * 100000 + nodes * 10000, "less_reliable_batch"
    for nodes, count in ((3, 5000), (5, 5000), (7, 10000)):
        yield "Spot", nodes, count, 7, 2200000 + nodes * 100000, "spot_pilot"


def validation_cases(batches: int, spot_runs: int):
    # Disjoint from original seeds and from all other validation cases.
    for index, profile in enumerate(("Standard", "Unreliable")):
        for nodes in (3, 5, 7):
            yield profile, nodes, batches * 1000, 7, 10000000 + index * 10000000 + nodes * 1000000, "validation_batches"
    yield "Spot", 7, spot_runs, 7, 40000000, "spot_validation"


def markov_cases():
    cfg = node_config_for(VM_PROFILES[0])
    for nodes in (3, 5, 7):
        for quality in QualityLevel:
            # Do not run the hours-long N=7 FULL solve in the normal suite.
            if nodes == 7 and quality == QualityLevel.FULL:
                start = time.perf_counter()
                model = build_markov_model([cfg] * nodes, raft_protocol(), replacement_strategy(cfg), quality)
                yield {"nodes": nodes, "quality": quality.name, "states": model.num_states,
                       "build_seconds": time.perf_counter() - start, "solve_skipped": True}
                continue
            builds, solves, totals = [], [], []
            for _ in range(1 if quality == QualityLevel.FULL else 4):
                gc.collect()
                start = time.perf_counter()
                model = build_markov_model([cfg] * nodes, raft_protocol(), replacement_strategy(cfg), quality)
                built = time.perf_counter()
                pi = steady_state(model, backend="scipy")
                finished = time.perf_counter()
                builds.append(built - start)
                solves.append(finished - built)
                totals.append(finished - start)
            yield {"nodes": nodes, "quality": quality.name, "states": model.num_states,
                   "availability": availability(model, pi), "repeats": len(totals),
                   "median_build_seconds": statistics.median(builds),
                   "median_solve_seconds": statistics.median(solves),
                   "median_total_seconds": statistics.median(totals)}


def evidence_markdown(rows: list[dict]) -> str:
    lines = ["# Generated Monte Carlo evidence", "",
             "Nominal Student-t intervals; pilot sample counts are planning estimates.", "",
             "| Experiment | Profile | Nodes | Runs | Mean | 99% interval | Below 99% | Passing 1,000-run batches |",
             "|---|---|---:|---:|---:|---|---:|---|"]
    for row in rows:
        lo, hi = row["ci99"]
        batches = f'{row["passing_batches"]}/{len(row["batches"])}' if "batches" in row else "—"
        lines.append(f'| {row["label"]} | {row["profile"]} | {row["nodes"]} | {row["runs"]:,} '
                     f'| {row["mean"]:.9f} | [{lo:.9f}, {hi:.9f}] | {row["below_99pct"]} | {batches} |')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("original", "validation", "markov"), default="validation")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--spot-runs", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/availability-convergence-study")
    parser.add_argument("--replay", help="Replay a named historical experiment from the source archive")
    parser.add_argument("--timeout", type=float, default=330)
    args = parser.parse_args()
    if args.replay:
        archive = json.loads(Path(__file__).with_name("availability_convergence_original.json").read_text())
        experiment = next((e for e in archive["experiments"] if e["name"] == args.replay), None)
        if experiment is None:
            parser.error("Unknown experiment; names: " + ", ".join(e["name"] for e in archive["experiments"]))
        try:
            result = subprocess.run([sys.executable, "-u", "-"], input=experiment["source"],
                                    text=True, cwd=ROOT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            parser.exit(124, "Historical experiment reached its time limit.\n")
        raise SystemExit(result.returncode)
    if not 1 <= args.batches <= 1000 or not 2 <= args.spot_runs <= 1000000:
        parser.error("Use 1–1000 batches and 2–1000000 Spot runs to keep seed ranges disjoint.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data = {"metadata": {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_revision_before_run": revision,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "thread_limits": {key: os.environ[key] for key in THREAD_VARIABLES},
        "parallel_workers": 1, "confidence": CONFIDENCE, "target_band": TARGET,
        "precision_half_width": EPSILON, "suite": args.suite,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, "results": []}
    if args.suite == "markov":
        results = markov_cases()
    else:
        cases = original_cases() if args.suite == "original" else validation_cases(args.batches, args.spot_runs)
        results = (simulate(*case) for case in cases)
    for row in results:
        data["results"].append(row)
        # Checkpoint after each scenario, including during lengthy suites.
        (args.output_dir / f"{args.suite}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        if args.suite != "markov":
            (args.output_dir / f"{args.suite}.md").write_text(evidence_markdown(data["results"]))
        print(json.dumps({k: v for k, v in row.items() if k not in (
            "batches", "tail_samples_below_99pct", "quantiles",
        )}), flush=True)


if __name__ == "__main__":
    main()
