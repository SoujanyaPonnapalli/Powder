#!/usr/bin/env python3
"""Benchmark heterogeneous RSM Markov scoring and Gurobi placement scaling.

The default run creates 1,300 low, medium, and high physical machines, samples
50,000 unique RSM candidates of uniformly random size 3, 5, or 7, scores them
with the SIMPLIFIED Raft Markov model, and solves nested Gurobi ILPs for the
first 5,000, 10,000, ..., 50,000 candidates.

The scoring CSV is append-only and resumable. Run phases independently with
``--phase inventory``, ``--phase score``, or ``--phase ilp`` when desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import linalg as sparse_linalg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from powder.markov_solver import (
    availability,
    compute_steady_state_residual,
    expected_cost_per_second,
)
from powder.placement_optimizer import (
    Machine,
    PlacementCandidate,
    PlacementSolverConfig,
    solve_candidate_ilp_gurobi,
)
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    Constant,
    Exponential,
    NodeConfig,
    NodeReplacementStrategy,
    RaftLikeProtocol,
    Seconds,
    days,
    hours,
    minutes,
)


DEFAULT_SEED = 20_260_803
DEFAULT_SAMPLES = 50_000
MACHINES_PER_CLASS = 1_300
NUM_RSMS = 100
BUDGET_PER_HOUR = 2_000.0
REPLICA_COUNTS = (3, 5, 7)
QUALITY = QualityLevel.SIMPLIFIED
GMRES_RTOL = 1e-14
GMRES_ATOL = 1e-16
GMRES_RESTART = 200
GMRES_MAXITER = 2_000

MACHINE_SPECS = (
    ("low", 1.0, float(hours(48)), float(days(365))),
    ("medium", 3.0, float(days(7)), float(days(365))),
    ("high", 9.0, float(days(30)), float(days(3 * 365))),
)

INVENTORY_FIELDS = (
    "machine_id",
    "machine_class",
    "normal_z",
    "base_transient_failure_rate_per_second",
    "transient_failure_rate_per_second",
    "transient_mtbf_hours",
    "base_price_per_hour",
    "price_per_hour",
    "data_loss_mttdl_seconds",
)
SAMPLE_FIELDS = (
    "sample_index",
    "candidate_id",
    "replica_count",
    "machine_ids",
    "machine_1",
    "machine_2",
    "machine_3",
    "machine_4",
    "machine_5",
    "machine_6",
    "machine_7",
)
SCORE_FIELDS = SAMPLE_FIELDS + (
    "availability",
    "expected_cost_per_hour",
    "list_price_per_hour",
    "num_states",
    "markov_build_seconds",
    "steady_state_seconds",
    "metric_seconds",
    "markov_total_seconds",
    "steady_balance_residual",
    "steady_normalization_residual",
    "steady_negativity_residual",
    "quality_level",
    "markov_solver",
)


@dataclass(frozen=True)
class BenchmarkPaths:
    output_dir: Path

    @property
    def inventory(self) -> Path:
        return self.output_dir / "machine_inventory.csv"

    @property
    def samples(self) -> Path:
        return self.output_dir / "rsm_samples.csv"

    @property
    def candidates(self) -> Path:
        return self.output_dir / "rsm_candidates.csv"

    @property
    def ilp_runs(self) -> Path:
        return self.output_dir / "ilp_benchmark.csv"

    @property
    def selected(self) -> Path:
        return self.output_dir / "ilp_selected_candidates.csv"

    @property
    def plots(self) -> Path:
        return self.output_dir / "rsm_ilp_benchmark_plots.html"

    @property
    def metadata(self) -> Path:
        return self.output_dir / "benchmark_metadata.json"

def _protocol_and_strategy() -> tuple[RaftLikeProtocol, NodeReplacementStrategy]:
    return (
        RaftLikeProtocol(election_time_dist=Exponential(1.0 / Seconds(5))),
        NodeReplacementStrategy(failure_timeout=hours(1), safe_mode=False),
    )


def _node_config(
    machine_class: str,
    price_per_hour: float,
    transient_failure_rate: float,
    data_loss_mttdl_seconds: float,
) -> NodeConfig:
    return NodeConfig(
        region=machine_class,
        cost_per_hour=price_per_hour,
        failure_dist=Exponential(transient_failure_rate),
        recovery_dist=Exponential(1.0 / minutes(45)),
        data_loss_dist=Exponential(1.0 / data_loss_mttdl_seconds),
        log_replay_rate_dist=Constant(1_000_000.0),
        snapshot_download_time_dist=Constant(minutes(30)),
        spawn_dist=Exponential(1.0 / Seconds(60)),
    )


def generate_inventory(path: Path, seed: int) -> None:
    """Generate the deterministic 3,900-machine population."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for machine_class, base_price, transient_mtbf, data_loss_mttdl in MACHINE_SPECS:
        base_rate = 1.0 / transient_mtbf
        for class_index in range(MACHINES_PER_CLASS):
            z = float(rng.normal())
            rate = max(0.0, 1.05 * base_rate + 0.05 * base_rate * z)
            price = max(0.0, 1.05 * base_price - 0.05 * base_price * z)
            if rate <= 0.0:
                # Exponential requires a positive rate. This branch is
                # astronomically unlikely for N(1.05, 0.05), but keep the
                # persisted value consistent with a representable model.
                rate = float(np.nextafter(0.0, 1.0))
            rows.append(
                {
                    "machine_id": f"{machine_class}-{class_index:04d}",
                    "machine_class": machine_class,
                    "normal_z": z,
                    "base_transient_failure_rate_per_second": base_rate,
                    "transient_failure_rate_per_second": rate,
                    "transient_mtbf_hours": 1.0 / rate / 3600.0,
                    "base_price_per_hour": base_price,
                    "price_per_hour": price,
                    "data_loss_mttdl_seconds": data_loss_mttdl,
                }
            )
    _write_csv(path, INVENTORY_FIELDS, rows)


def generate_samples(path: Path, inventory_path: Path, count: int, seed: int) -> None:
    """Sample unique RSM sizes and machine sets uniformly at random."""
    machine_ids = [row["machine_id"] for row in _read_csv(inventory_path)]
    rng = np.random.default_rng(seed + 1)
    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, object]] = []
    while len(rows) < count:
        replica_count = int(rng.choice(REPLICA_COUNTS))
        chosen_indexes = rng.choice(len(machine_ids), size=replica_count, replace=False)
        chosen = tuple(sorted(machine_ids[int(index)] for index in chosen_indexes))
        if chosen in seen:
            continue
        seen.add(chosen)
        index = len(rows) + 1
        padded = list(chosen) + [""] * (7 - replica_count)
        rows.append(
            {
                "sample_index": index,
                "candidate_id": f"rsm-{index:05d}",
                "replica_count": replica_count,
                "machine_ids": ";".join(chosen),
                **{f"machine_{slot + 1}": value for slot, value in enumerate(padded)},
            }
        )
    _write_csv(path, SAMPLE_FIELDS, rows)


_WORKER_CONFIGS: dict[str, NodeConfig] = {}
_WORKER_PROTOCOL: RaftLikeProtocol | None = None
_WORKER_STRATEGY: NodeReplacementStrategy | None = None


def _initialize_score_worker(inventory_path: str) -> None:
    global _WORKER_CONFIGS, _WORKER_PROTOCOL, _WORKER_STRATEGY
    _WORKER_CONFIGS = _load_node_configs(Path(inventory_path))
    _WORKER_PROTOCOL, _WORKER_STRATEGY = _protocol_and_strategy()


def _steady_state_gmres(model: object) -> tuple[np.ndarray, object]:
    state_count = model.num_states
    transpose = model.Q.T.tolil(copy=True)
    transpose.rows[-1] = list(range(state_count))
    transpose.data[-1] = [1.0] * state_count
    right_hand_side = np.zeros(state_count, dtype=np.float64)
    right_hand_side[-1] = 1.0
    raw, info = sparse_linalg.gmres(
        transpose.tocsr(),
        right_hand_side,
        rtol=GMRES_RTOL,
        atol=GMRES_ATOL,
        restart=GMRES_RESTART,
        maxiter=GMRES_MAXITER,
    )
    if info != 0:
        raise RuntimeError(f"SciPy GMRES did not converge: info={info}")
    residual = compute_steady_state_residual(model, raw)
    probabilities = np.clip(raw, 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise RuntimeError("SciPy GMRES returned no positive stationary mass")
    return probabilities / total, residual


def _score_sample(row: dict[str, str]) -> dict[str, object]:
    if _WORKER_PROTOCOL is None or _WORKER_STRATEGY is None:
        raise RuntimeError("score worker was not initialized")
    machine_ids = tuple(value for value in row["machine_ids"].split(";") if value)
    configs = [_WORKER_CONFIGS[machine_id] for machine_id in machine_ids]
    total_start = time.perf_counter()
    start = time.perf_counter()
    model = build_markov_model(configs, _WORKER_PROTOCOL, _WORKER_STRATEGY, QUALITY)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities, residual = _steady_state_gmres(model)
    solve_seconds = time.perf_counter() - start
    start = time.perf_counter()
    candidate_availability = availability(model, probabilities)
    expected_cost = expected_cost_per_second(model, probabilities) * 3600.0
    metric_seconds = time.perf_counter() - start
    total_seconds = time.perf_counter() - total_start
    return {
        **row,
        "availability": candidate_availability,
        "expected_cost_per_hour": expected_cost,
        "list_price_per_hour": sum(config.cost_per_hour for config in configs),
        "num_states": model.num_states,
        "markov_build_seconds": build_seconds,
        "steady_state_seconds": solve_seconds,
        "metric_seconds": metric_seconds,
        "markov_total_seconds": total_seconds,
        "steady_balance_residual": residual.balance,
        "steady_normalization_residual": residual.normalization,
        "steady_negativity_residual": residual.negativity,
        "quality_level": QUALITY.name,
        "markov_solver": "scipy_gmres",
    }


def score_samples(
    paths: BenchmarkPaths,
    *,
    workers: int,
    progress_every: int,
) -> dict[str, float | int]:
    """Score pending sample rows, appending results in deterministic order."""
    completed = _csv_row_count(paths.candidates)
    sample_rows = list(_read_csv(paths.samples))
    if completed > len(sample_rows):
        raise ValueError("candidate CSV contains more rows than the sample manifest")
    pending = sample_rows[completed:]
    if not pending:
        return {"previously_completed": completed, "newly_completed": 0, "wall_seconds": 0.0}

    mode = "a" if completed else "w"
    wall_start = time.perf_counter()
    with paths.candidates.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        if not completed:
            writer.writeheader()
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_score_worker,
            initargs=(str(paths.inventory),),
        ) as executor:
            for offset, result in enumerate(
                executor.map(_score_sample, pending, chunksize=1), start=1
            ):
                writer.writerow(result)
                if offset % 100 == 0:
                    handle.flush()
                if offset % progress_every == 0 or offset == len(pending):
                    elapsed = time.perf_counter() - wall_start
                    total_done = completed + offset
                    rate = offset / elapsed if elapsed else float("inf")
                    remaining = (len(pending) - offset) / rate if rate else float("inf")
                    print(
                        f"[score] {total_done}/{len(sample_rows)} "
                        f"({rate:.2f} RSM/s, ETA {remaining / 60:.1f} min)",
                        flush=True,
                    )
    return {
        "previously_completed": completed,
        "newly_completed": len(pending),
        "wall_seconds": time.perf_counter() - wall_start,
    }


def summarize_scored_candidates(path: Path) -> dict[str, object]:
    rows = list(_read_csv(path))
    durations = [float(row["markov_total_seconds"]) for row in rows]
    size_counts = {
        str(replica_count): sum(
            int(row["replica_count"]) == replica_count for row in rows
        )
        for replica_count in REPLICA_COUNTS
    }
    return {
        "candidate_count": len(rows),
        "replica_count_distribution": size_counts,
        "total_markov_compute_seconds": float(sum(durations)),
        "mean_markov_compute_seconds_per_rsm": float(np.mean(durations)),
        "median_markov_compute_seconds_per_rsm": float(np.median(durations)),
        "max_balance_residual": max(
            float(row["steady_balance_residual"]) for row in rows
        ),
        "max_normalization_residual": max(
            float(row["steady_normalization_residual"]) for row in rows
        ),
        "max_negativity_residual": max(
            float(row["steady_negativity_residual"]) for row in rows
        ),
    }


def run_ilp_benchmarks(
    paths: BenchmarkPaths,
    *,
    k_min: int,
    k_step: int,
    gurobi_threads: int,
    seed: int,
) -> dict[str, float | int]:
    machines = _load_machines(paths.inventory)
    candidate_rows = list(_read_csv(paths.candidates))
    if len(candidate_rows) < k_min:
        raise ValueError(
            f"need at least {k_min} scored candidates; found {len(candidate_rows)}"
        )
    candidates = [_candidate_from_row(row) for row in candidate_rows]
    cumulative_markov = np.cumsum(
        [float(row["markov_total_seconds"]) for row in candidate_rows]
    )
    ks = list(range(k_min, len(candidates) + 1, k_step))
    if ks[-1] != len(candidates):
        ks.append(len(candidates))
    observed_cumulative = np.asarray([cumulative_markov[k - 1] for k in ks])
    k_values = np.asarray(ks, dtype=float)
    linear_seconds_per_rsm = float(
        np.dot(k_values, observed_cumulative) / np.dot(k_values, k_values)
    )

    run_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for run_index, k in enumerate(ks, start=1):
        start = time.perf_counter()
        solution = solve_candidate_ilp_gurobi(
            candidates[:k],
            machines,
            PlacementSolverConfig(
                num_rsms=NUM_RSMS,
                objective="product_availability",
                budget_per_hour=BUDGET_PER_HOUR,
            ),
            output_flag=False,
            threads=gurobi_threads,
            seed=seed,
        )
        ilp_end_to_end_seconds = time.perf_counter() - start
        selected_count = solution.total_rsms
        feasible_solution = selected_count == NUM_RSMS and math.isfinite(
            solution.total_cost_per_hour
        ) and solution.total_cost_per_hour <= BUDGET_PER_HOUR + 1e-7
        optimal = solution.status == "optimal"
        goal_achieved = feasible_solution and optimal
        if solution.selected:
            sum_availability = sum(
                item.count * item.candidate.availability for item in solution.selected
            )
            mean_availability = sum_availability / selected_count
            log_product = sum(
                item.count * item.candidate.log_availability for item in solution.selected
            )
            product_availability = math.exp(log_product)
        else:
            sum_availability = float("nan")
            mean_availability = float("nan")
            product_availability = float("nan")
        estimated_markov_seconds = linear_seconds_per_rsm * k
        combined_seconds = estimated_markov_seconds + ilp_end_to_end_seconds
        failure_reason = "" if goal_achieved else solution.message
        run_rows.append(
            {
                "candidate_count": k,
                "quality_mean_availability": mean_availability,
                "sum_availability": sum_availability,
                "product_availability": product_availability,
                "total_cost_per_hour": solution.total_cost_per_hour,
                "budget_per_hour": BUDGET_PER_HOUR,
                "selected_rsms": selected_count,
                "solver_status": solution.status,
                "feasible_solution": feasible_solution,
                "optimal": optimal,
                "goal_achieved": goal_achieved,
                "failure_reason": failure_reason,
                "gurobi_runtime_seconds": solution.solver_runtime_seconds,
                "ilp_end_to_end_seconds": ilp_end_to_end_seconds,
                "ilp_runtime_per_candidate_seconds": ilp_end_to_end_seconds / k,
                "gurobi_mip_gap": solution.mip_gap,
                "gurobi_best_bound_log_product": solution.best_bound,
                "objective_log_product": solution.objective_value,
                "observed_cumulative_markov_seconds": cumulative_markov[k - 1],
                "markov_linear_seconds_per_rsm": linear_seconds_per_rsm,
                "estimated_markov_runtime_seconds": estimated_markov_seconds,
                "combined_runtime_seconds": combined_seconds,
            }
        )
        for item in solution.selected:
            selected_rows.append(
                {
                    "candidate_count": k,
                    "candidate_id": item.candidate.candidate_id,
                    "selection_count": item.count,
                    "replica_count": item.candidate.replica_count,
                    "machine_ids": ";".join(item.candidate.machine_ids),
                    "availability": item.candidate.availability,
                    "expected_cost_per_hour": item.candidate.cost_per_hour,
                }
            )
        print(
            f"[ilp] {run_index}/{len(ks)} K={k} status={solution.status} "
            f"quality={mean_availability:.12f} time={ilp_end_to_end_seconds:.3f}s",
            flush=True,
        )

    _write_csv(paths.ilp_runs, tuple(run_rows[0]), run_rows)
    selected_fields = (
        "candidate_count",
        "candidate_id",
        "selection_count",
        "replica_count",
        "machine_ids",
        "availability",
        "expected_cost_per_hour",
    )
    _write_csv(paths.selected, selected_fields, selected_rows)
    create_plots(paths)
    return {
        "runs": len(run_rows),
        "linear_markov_seconds_per_rsm": linear_seconds_per_rsm,
    }


def create_plots(paths: BenchmarkPaths) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = list(_read_csv(paths.ilp_runs))
    x_combined = [float(row["combined_runtime_seconds"]) for row in rows]
    quality = [float(row["quality_mean_availability"]) for row in rows]
    product_availability = [float(row["product_availability"]) for row in rows]
    ks = [int(row["candidate_count"]) for row in rows]
    markov_per_rsm = [float(row["markov_linear_seconds_per_rsm"]) for row in rows]
    ilp_per_candidate = [float(row["ilp_runtime_per_candidate_seconds"]) for row in rows]
    modeling_runtime = [float(row["estimated_markov_runtime_seconds"]) for row in rows]
    ilp_runtime = [float(row["ilp_end_to_end_seconds"]) for row in rows]
    total_runtime = [float(row["combined_runtime_seconds"]) for row in rows]

    figure = make_subplots(
        rows=3,
        cols=1,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        vertical_spacing=0.11,
        subplot_titles=(
            "Mean and product availability vs. combined runtime",
            "Runtime per sampled RSM/candidate",
            "Total runtime vs. candidate RSMs",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=x_combined,
            y=quality,
            mode="lines+markers+text",
            text=[f"{k // 1000}k" for k in ks],
            textposition="top center",
            name="Mean availability",
            hovertemplate="K=%{text}<br>Combined=%{x:.2f}s<br>Mean availability=%{y:.12f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=x_combined,
            y=product_availability,
            mode="lines+markers",
            name="Product availability",
            line=dict(dash="dash"),
            marker=dict(symbol="diamond"),
            hovertemplate="K=%{customdata:,}<br>Combined=%{x:.2f}s<br>Product availability=%{y:.12f}<extra></extra>",
            customdata=ks,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=ks,
            y=markov_per_rsm,
            mode="lines+markers",
            name="Markov linear extrapolation",
            hovertemplate="K=%{x:,}<br>Seconds/RSM=%{y:.6f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=ks,
            y=total_runtime,
            mode="lines+markers",
            name="Total runtime",
            line=dict(width=3),
            marker=dict(symbol="circle"),
            hovertemplate="K=%{x:,}<br>Total=%{y:.3f}s<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=ks,
            y=modeling_runtime,
            mode="lines+markers",
            name="Modeling runtime",
            line=dict(dash="dash"),
            marker=dict(symbol="square"),
            hovertemplate="K=%{x:,}<br>Modeling=%{y:.3f}s<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=ks,
            y=ilp_runtime,
            mode="lines+markers",
            name="ILP runtime",
            line=dict(dash="dot"),
            marker=dict(symbol="diamond"),
            hovertemplate="K=%{x:,}<br>ILP=%{y:.6f}s<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=ks,
            y=ilp_per_candidate,
            mode="lines+markers",
            name="Gurobi ILP",
            hovertemplate="K=%{x:,}<br>Seconds/candidate=%{y:.8f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Combined estimated runtime (seconds)", row=1, col=1)
    figure.update_yaxes(title_text="Mean availability", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Product availability", row=1, col=1, secondary_y=True)
    figure.update_xaxes(title_text="Number of candidate RSMs (K)", row=2, col=1)
    figure.update_yaxes(title_text="Runtime per RSM/candidate (seconds)", type="log", row=2, col=1)
    figure.update_xaxes(title_text="Number of candidate RSMs (K)", row=3, col=1)
    figure.update_yaxes(title_text="Total runtime (seconds, log scale)", type="log", row=3, col=1)
    figure.update_layout(
        template="plotly_white",
        height=1250,
        legend_title_text="Pipeline",
        margin=dict(l=80, r=40, t=80, b=70),
    )
    figure.write_html(paths.plots, include_plotlyjs="cdn", full_html=True)


def _load_node_configs(path: Path) -> dict[str, NodeConfig]:
    return {
        row["machine_id"]: _node_config(
            row["machine_class"],
            float(row["price_per_hour"]),
            float(row["transient_failure_rate_per_second"]),
            float(row["data_loss_mttdl_seconds"]),
        )
        for row in _read_csv(path)
    }


def _load_machines(path: Path) -> list[Machine]:
    configs = _load_node_configs(path)
    return [Machine(machine_id, config, capacity=1) for machine_id, config in configs.items()]


def _candidate_from_row(row: dict[str, str]) -> PlacementCandidate:
    return PlacementCandidate(
        candidate_id=row["candidate_id"],
        machine_ids=tuple(row["machine_ids"].split(";")),
        availability=float(row["availability"]),
        cost_per_hour=float(row["expected_cost_per_hour"]),
        metadata={
            "replica_count": int(row["replica_count"]),
            "num_states": int(row["num_states"]),
        },
    )


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _versions() -> dict[str, object]:
    import gurobipy
    import scipy

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gurobi": ".".join(map(str, gurobipy.gurobi.version())),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _write_metadata(paths: BenchmarkPaths, updates: dict[str, object]) -> None:
    existing: dict[str, object] = {}
    if paths.metadata.exists():
        with paths.metadata.open() as handle:
            existing = json.load(handle)
    existing.update(updates)
    with paths.metadata.open("w") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "rsm-ilp-scaling-benchmark",
    )
    parser.add_argument("--phase", choices=("all", "inventory", "score", "ilp"), default="all")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--k-min", type=int, default=5_000)
    parser.add_argument("--k-step", type=int, default=5_000)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()
    if args.samples <= 0 or args.workers <= 0:
        raise ValueError("samples and workers must be positive")
    paths = BenchmarkPaths(args.output_dir.resolve())
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        paths,
        {
            "configuration": {
                "seed": args.seed,
                "samples": args.samples,
                "machines_per_class": MACHINES_PER_CLASS,
                "replica_counts": REPLICA_COUNTS,
                "num_rsms": NUM_RSMS,
                "budget_per_hour": BUDGET_PER_HOUR,
                "quality_level": QUALITY.name,
                "objective": "product_availability",
                "reported_quality": "mean_availability",
                "sample_size_distribution": "uniform over 3, 5, 7",
                "replica_sampling": "uniform without replacement",
                "price_rate_correlation": "inverse normal quantile using shared z",
                "gurobi_threads": args.gurobi_threads,
                "markov_workers": args.workers,
                "gmres": {
                    "rtol": GMRES_RTOL,
                    "atol": GMRES_ATOL,
                    "restart": GMRES_RESTART,
                    "maxiter": GMRES_MAXITER,
                },
            },
            "versions": _versions(),
        },
    )

    if args.phase in {"all", "inventory"}:
        start = time.perf_counter()
        generate_inventory(paths.inventory, args.seed)
        generate_samples(paths.samples, paths.inventory, args.samples, args.seed)
        _write_metadata(
            paths,
            {"inventory_and_sampling_wall_seconds": time.perf_counter() - start},
        )
        print(f"[inventory] wrote {paths.inventory} and {paths.samples}", flush=True)

    if args.phase in {"all", "score"}:
        if not paths.inventory.exists() or not paths.samples.exists():
            raise FileNotFoundError("run the inventory phase before scoring")
        scoring = score_samples(
            paths,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        _write_metadata(
            paths,
            {
                "scoring": {**scoring, **summarize_scored_candidates(paths.candidates)},
            },
        )

    if args.phase in {"all", "ilp"}:
        if not paths.candidates.exists():
            raise FileNotFoundError("run the score phase before the ILP phase")
        ilp = run_ilp_benchmarks(
            paths,
            k_min=args.k_min,
            k_step=args.k_step,
            gurobi_threads=args.gurobi_threads,
            seed=args.seed,
        )
        _write_metadata(paths, {"ilp": ilp})
    _write_metadata(paths, {"last_command_wall_seconds": time.perf_counter() - overall_start})


if __name__ == "__main__":
    main()
