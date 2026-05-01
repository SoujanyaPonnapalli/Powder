"""Analyze SIMPLIFIED Markov accuracy on bathtub machine classes.

Sweeps Raft clusters with N in {3, 5, 7} and C in 1..N distinct machine
classes. Each class is generated from a deterministic point in [0, 1] and
mapped through these annualized fault-rate / price curves:

    T(x) = 15(1-x)^2 + 4 + 20x^6
    D(x) = 0.5 + 0.5x^15
    P(x) = 1000 - (10T(x) + 500D(x))

SIMPLIFIED is run for every scenario. FULL is used as the accuracy reference
when its estimated state space is small enough to build safely; otherwise the
row reports a skipped reference. Every matrix solve runs in a subprocess with
a 60-second wall-clock budget so one large sparse solve cannot hang the run.

Output:
    notebooks/markov_simplified_bathtub_analysis.csv
"""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import queue
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from powder.markov import MarkovModel
from powder.markov_solver import (
    availability,
    expected_cost_per_second,
    mean_first_passage,
    steady_state,
)
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    Constant,
    Exponential,
    NodeConfig,
    NodeReplacementStrategy,
    RaftLikeProtocol,
    Seconds,
)


CLUSTER_SIZES = (3, 5, 7)
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
HOURS_PER_YEAR = 365.0 * 24.0

RECOVERY_MEAN_S = 20.0 * 60.0
SNAPSHOT_DOWNLOAD_S = 60.0
SPAWN_MEAN_S = 60.0
FAILURE_TIMEOUT_S = 5.0 * 60.0
ELECTION_MEAN_S = 5.0

SOLVE_TIMEOUT_S = 60.0
MAX_FULL_STATE_ESTIMATE = 1_000_000
OUTPUT_CSV = Path(__file__).resolve().with_suffix(".csv")

_K_BY_QUALITY = {
    QualityLevel.SIMPLIFIED: 3,
    QualityLevel.COLLAPSED_PIPELINE: 4,
    QualityLevel.NO_ORPHANS: 6,
    QualityLevel.MERGED_PIPELINE: 8,
    QualityLevel.FULL: 12,
}
_DATA_LOSS_INDICES = {
    QualityLevel.SIMPLIFIED: (2,),
    QualityLevel.COLLAPSED_PIPELINE: (3,),
    QualityLevel.NO_ORPHANS: (4, 5),
    QualityLevel.MERGED_PIPELINE: (6, 7),
    QualityLevel.FULL: (9, 10, 11),
}


@dataclass(frozen=True)
class MachineClass:
    class_idx: int
    x: float
    transient_failures_per_year: float
    dataloss_failures_per_year: float
    cost_per_year: float


@dataclass(frozen=True)
class SolveOutcome:
    status: str
    wall_ms: float | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class QualityRun:
    quality: QualityLevel
    state_estimate: int
    build_ms: float | None
    num_states: int | None
    steady: SolveOutcome | None
    mttdl: SolveOutcome | None
    skip_reason: str | None = None

    @property
    def solved(self) -> bool:
        return bool(self.steady and self.steady.ok and self.mttdl and self.mttdl.ok)


@dataclass(frozen=True)
class BenchmarkRow:
    n: int
    num_classes: int
    class_sizes: tuple[int, ...]
    machines: tuple[MachineClass, ...]
    simplified: QualityRun
    full: QualityRun


def _transient_failures_per_year(x: float) -> float:
    return 15.0 * (1.0 - x) ** 2 + 4.0 + 20.0 * x**6


def _dataloss_failures_per_year(x: float) -> float:
    return 0.5 + 0.5 * x**15


def _cost_per_year(x: float) -> float:
    t = _transient_failures_per_year(x)
    d = _dataloss_failures_per_year(x)
    return 1000.0 - (10.0 * t + 500.0 * d)


def _machine_classes(num_classes: int) -> tuple[MachineClass, ...]:
    points = [(i + 0.5) / num_classes for i in range(num_classes)]
    return tuple(
        MachineClass(
            class_idx=i,
            x=x,
            transient_failures_per_year=_transient_failures_per_year(x),
            dataloss_failures_per_year=_dataloss_failures_per_year(x),
            cost_per_year=_cost_per_year(x),
        )
        for i, x in enumerate(points)
    )


def _node_config(machine: MachineClass) -> NodeConfig:
    return NodeConfig(
        region=f"machine-class-{machine.class_idx}",
        cost_per_hour=machine.cost_per_year / HOURS_PER_YEAR,
        failure_dist=Exponential(
            rate=machine.transient_failures_per_year / SECONDS_PER_YEAR,
        ),
        recovery_dist=Exponential(rate=1.0 / RECOVERY_MEAN_S),
        data_loss_dist=Exponential(
            rate=machine.dataloss_failures_per_year / SECONDS_PER_YEAR,
        ),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(SNAPSHOT_DOWNLOAD_S),
        spawn_dist=Exponential(rate=1.0 / SPAWN_MEAN_S),
    )


def _class_sizes(n: int, num_classes: int) -> tuple[int, ...]:
    return tuple(
        n // num_classes + (1 if i < n % num_classes else 0)
        for i in range(num_classes)
    )


def _configs_for(n: int, machines: tuple[MachineClass, ...]) -> list[NodeConfig]:
    configs: list[NodeConfig] = []
    for machine, size in zip(machines, _class_sizes(n, len(machines))):
        configs.extend([_node_config(machine)] * size)
    return configs


def _weak_compositions_count(n: int, k: int) -> int:
    return math.comb(n + k - 1, k - 1)


def _raft_state_estimate(n: int, num_classes: int, quality: QualityLevel) -> int:
    k = _K_BY_QUALITY[quality]
    estimate = 1
    for size in _class_sizes(n, num_classes):
        estimate *= _weak_compositions_count(size, k)
    trailing_multiplier = num_classes + 1
    if quality in (QualityLevel.MERGED_PIPELINE, QualityLevel.FULL):
        trailing_multiplier *= 3
    return estimate * trailing_multiplier


def _all_replica_dataloss_state_ids(
    model: MarkovModel,
    *,
    n: int,
    num_classes: int,
    quality: QualityLevel,
) -> list[int]:
    """States where every replica slot is in a data-loss state.

    The Markov builders allow replacement out of all-data-loss states to keep
    steady-state solves meaningful. For MTTDL we use first passage time to the
    all-replica data-loss set as the loss event.
    """
    k = _K_BY_QUALITY[quality]
    dataloss_indices = _DATA_LOSS_INDICES[quality]
    has_leader_idx = num_classes * k
    target_ids: list[int] = []

    for state_id, name in enumerate(model.state_names):
        counts = tuple(int(part) for part in name.split(":"))
        dataloss_count = 0
        for class_idx in range(num_classes):
            off = class_idx * k
            dataloss_count += sum(counts[off + idx] for idx in dataloss_indices)
        has_leader = counts[has_leader_idx]
        if dataloss_count == n and has_leader == 0:
            target_ids.append(state_id)

    return target_ids


def _solve_worker(
    out: mp.Queue,
    kind: str,
    model: MarkovModel,
    target_ids: list[int] | None,
) -> None:
    try:
        t0 = time.perf_counter()
        if kind == "steady":
            pi = steady_state(model)
            payload = {
                "availability": availability(model, pi),
                "cost_per_year": expected_cost_per_second(model, pi) * SECONDS_PER_YEAR,
            }
        elif kind == "mttdl":
            if not target_ids:
                raise ValueError("MTTDL target set is empty")
            mfp = mean_first_passage(model, target_ids)
            mttdl_seconds = float(mfp[model.initial_state_id])
            min_seconds = float(mfp.min()) if mfp.size else 0.0
            payload = {
                "mttdl_years": mttdl_seconds / SECONDS_PER_YEAR,
                "min_first_passage_years": min_seconds / SECONDS_PER_YEAR,
            }
            if (
                not np.isfinite(mttdl_seconds)
                or mttdl_seconds <= 0.0
                or min_seconds < -max(1.0, abs(mttdl_seconds) * 1e-9)
            ):
                payload["solve_ms"] = (time.perf_counter() - t0) * 1000.0
                out.put(
                    (
                        "invalid",
                        payload,
                        "first-passage solve produced non-physical MTTDL; "
                        "the rare-event system is numerically ill-conditioned",
                    ),
                )
                return
        else:
            raise ValueError(f"unknown solve kind: {kind}")
        payload["solve_ms"] = (time.perf_counter() - t0) * 1000.0
        out.put(("ok", payload))
    except BaseException as exc:  # pragma: no cover - exercised via subprocess
        out.put(("error", f"{type(exc).__name__}: {exc}"))


def _mp_context() -> mp.context.BaseContext:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def _run_solve_with_timeout(
    kind: str,
    model: MarkovModel,
    target_ids: list[int] | None = None,
) -> SolveOutcome:
    ctx = _mp_context()
    out: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_solve_worker, args=(out, kind, model, target_ids))
    proc.daemon = True

    t0 = time.perf_counter()
    proc.start()
    deadline = t0 + SOLVE_TIMEOUT_S
    while proc.is_alive():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        proc.join(timeout=min(0.05, remaining))
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return SolveOutcome(
            status="timeout",
            wall_ms=wall_ms,
            error=f"exceeded {SOLVE_TIMEOUT_S:.0f}s solve budget",
        )

    try:
        result = out.get_nowait()
    except queue.Empty:
        return SolveOutcome(
            status="error",
            wall_ms=wall_ms,
            error=f"solver subprocess exited with code {proc.exitcode}",
        )

    status, payload, *rest = result
    if status == "ok":
        solve_ms = float(payload.get("solve_ms", 0.0))
        if solve_ms > SOLVE_TIMEOUT_S * 1000.0:
            return SolveOutcome(
                status="timeout",
                wall_ms=wall_ms,
                error=(
                    f"reported solve time {solve_ms / 1000.0:.2f}s "
                    f"exceeded {SOLVE_TIMEOUT_S:.0f}s solve budget"
                ),
            )
        return SolveOutcome(status="ok", wall_ms=wall_ms, metrics=payload)
    if status == "invalid":
        return SolveOutcome(
            status="invalid",
            wall_ms=wall_ms,
            metrics=payload,
            error=str(rest[0] if rest else "invalid solve result"),
        )
    return SolveOutcome(status="error", wall_ms=wall_ms, error=str(payload))


def _run_quality(
    configs: list[NodeConfig],
    *,
    n: int,
    num_classes: int,
    quality: QualityLevel,
    max_state_estimate: int | None = None,
) -> QualityRun:
    estimate = _raft_state_estimate(n, num_classes, quality)
    if max_state_estimate is not None and estimate > max_state_estimate:
        return QualityRun(
            quality=quality,
            state_estimate=estimate,
            build_ms=None,
            num_states=None,
            steady=None,
            mttdl=None,
            skip_reason=(
                f"estimated states {estimate:,} exceeds cap "
                f"{max_state_estimate:,}"
            ),
        )

    protocol = RaftLikeProtocol(election_time_dist=Exponential(rate=1.0 / ELECTION_MEAN_S))
    strategy = NodeReplacementStrategy(failure_timeout=Seconds(FAILURE_TIMEOUT_S))

    t0 = time.perf_counter()
    model = build_markov_model(configs, protocol, strategy, quality)
    build_ms = (time.perf_counter() - t0) * 1000.0

    steady = _run_solve_with_timeout("steady", model)
    if not steady.ok:
        return QualityRun(
            quality=quality,
            state_estimate=estimate,
            build_ms=build_ms,
            num_states=model.num_states,
            steady=steady,
            mttdl=SolveOutcome(status="skipped", error="steady solve did not finish"),
        )

    target_ids = _all_replica_dataloss_state_ids(
        model,
        n=n,
        num_classes=num_classes,
        quality=quality,
    )
    mttdl = _run_solve_with_timeout("mttdl", model, target_ids)
    return QualityRun(
        quality=quality,
        state_estimate=estimate,
        build_ms=build_ms,
        num_states=model.num_states,
        steady=steady,
        mttdl=mttdl,
    )


def _relative_error(approx: float | None, reference: float | None) -> float | None:
    if approx is None or reference is None or reference == 0:
        return None
    return abs(approx - reference) / abs(reference)


def _metric(run: QualityRun, solve: str, name: str) -> float | None:
    outcome = run.steady if solve == "steady" else run.mttdl
    if not outcome or not outcome.ok or not outcome.metrics:
        return None
    return outcome.metrics.get(name)


def _wall_ms(run: QualityRun, solve: str) -> float | None:
    outcome = run.steady if solve == "steady" else run.mttdl
    if not outcome:
        return None
    if outcome.metrics and "solve_ms" in outcome.metrics:
        return outcome.metrics["solve_ms"]
    return outcome.wall_ms


def _total_solve_wall_ms(run: QualityRun) -> float | None:
    values = [v for v in (_wall_ms(run, "steady"), _wall_ms(run, "mttdl")) if v is not None]
    if not values:
        return None
    return sum(values)


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:7.1f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:8.3f}%"


def _fmt_abs(value: float | None) -> str:
    return "-" if value is None else f"{value:.3e}"


def _row_status(run: QualityRun) -> str:
    if run.skip_reason:
        return "skipped"
    if not run.steady or not run.steady.ok:
        return f"steady-{run.steady.status if run.steady else 'missing'}"
    if not run.mttdl or not run.mttdl.ok:
        return f"mttdl-{run.mttdl.status if run.mttdl else 'missing'}"
    return "ok"


def _join(values: list[float] | tuple[float, ...], digits: int = 8) -> str:
    return "|".join(f"{value:.{digits}g}" for value in values)


def _csv_value(value: float | int | str | None) -> float | int | str:
    return "" if value is None else value


def _csv_row(row: BenchmarkRow) -> dict[str, float | int | str]:
    simp = row.simplified
    full = row.full
    simp_avail = _metric(simp, "steady", "availability")
    full_avail = _metric(full, "steady", "availability")
    simp_cost = _metric(simp, "steady", "cost_per_year")
    full_cost = _metric(full, "steady", "cost_per_year")
    simp_mttdl = _metric(simp, "mttdl", "mttdl_years")
    full_mttdl = _metric(full, "mttdl", "mttdl_years")
    avail_abs_error = (
        abs(simp_avail - full_avail)
        if simp_avail is not None and full_avail is not None
        else None
    )

    return {
        "n": row.n,
        "num_classes": row.num_classes,
        "class_sizes": "|".join(str(size) for size in row.class_sizes),
        "class_x": _join(tuple(machine.x for machine in row.machines)),
        "class_transient_failures_per_year": _join(
            tuple(machine.transient_failures_per_year for machine in row.machines),
        ),
        "class_dataloss_failures_per_year": _join(
            tuple(machine.dataloss_failures_per_year for machine in row.machines),
        ),
        "class_cost_per_year": _join(
            tuple(machine.cost_per_year for machine in row.machines),
        ),
        "simplified_status": _row_status(simp),
        "full_status": _row_status(full),
        "simplified_state_estimate": simp.state_estimate,
        "full_state_estimate": full.state_estimate,
        "simplified_num_states": _csv_value(simp.num_states),
        "full_num_states": _csv_value(full.num_states),
        "simplified_build_ms": _csv_value(simp.build_ms),
        "full_build_ms": _csv_value(full.build_ms),
        "simplified_steady_solve_ms": _csv_value(_wall_ms(simp, "steady")),
        "simplified_mttdl_solve_ms": _csv_value(_wall_ms(simp, "mttdl")),
        "simplified_total_solve_ms": _csv_value(_total_solve_wall_ms(simp)),
        "full_steady_solve_ms": _csv_value(_wall_ms(full, "steady")),
        "full_mttdl_solve_ms": _csv_value(_wall_ms(full, "mttdl")),
        "full_total_solve_ms": _csv_value(_total_solve_wall_ms(full)),
        "simplified_availability": _csv_value(simp_avail),
        "full_availability": _csv_value(full_avail),
        "availability_abs_error": _csv_value(avail_abs_error),
        "simplified_mttdl_years": _csv_value(simp_mttdl),
        "full_mttdl_years": _csv_value(full_mttdl),
        "simplified_raw_mttdl_years": _csv_value(
            simp.mttdl.metrics.get("mttdl_years") if simp.mttdl and simp.mttdl.metrics else None,
        ),
        "full_raw_mttdl_years": _csv_value(
            full.mttdl.metrics.get("mttdl_years") if full.mttdl and full.mttdl.metrics else None,
        ),
        "simplified_min_first_passage_years": _csv_value(
            simp.mttdl.metrics.get("min_first_passage_years")
            if simp.mttdl and simp.mttdl.metrics
            else None,
        ),
        "full_min_first_passage_years": _csv_value(
            full.mttdl.metrics.get("min_first_passage_years")
            if full.mttdl and full.mttdl.metrics
            else None,
        ),
        "mttdl_rel_error": _csv_value(_relative_error(simp_mttdl, full_mttdl)),
        "simplified_cost_per_year": _csv_value(simp_cost),
        "full_cost_per_year": _csv_value(full_cost),
        "cost_per_year_rel_error": _csv_value(_relative_error(simp_cost, full_cost)),
        "simplified_skip_reason": simp.skip_reason or "",
        "full_skip_reason": full.skip_reason or "",
        "simplified_steady_error": simp.steady.error if simp.steady else "",
        "simplified_mttdl_error": simp.mttdl.error if simp.mttdl else "",
        "full_steady_error": full.steady.error if full.steady else "",
        "full_mttdl_error": full.mttdl.error if full.mttdl else "",
    }


def write_csv(rows: list[BenchmarkRow], path: Path = OUTPUT_CSV) -> Path:
    records = [_csv_row(row) for row in rows]
    if not records:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return path


def _print_summary(rows: list[BenchmarkRow]) -> None:
    print("\n=== Simplified Markov Bathtub Analysis ===")
    print(f"Matrix solve timeout: {SOLVE_TIMEOUT_S:.0f}s per solve")
    print(f"FULL pre-build state estimate cap: {MAX_FULL_STATE_ESTIMATE:,}")
    print("Reference: QualityLevel.FULL when tractable")
    print("Per-row solve time sums the steady-state and MTTDL matrix solves.")
    print(
        "\n"
        "  N   C  class sizes  simp states  full states  "
        "simp solve ms  full solve ms  avail abs err  MTTDL rel err  "
        "cost rel err  simp      full"
    )
    print("-" * 139)

    availability_errors: list[float] = []
    mttdl_errors: list[float] = []
    cost_errors: list[float] = []
    simplified_solve_ms: list[float] = []
    full_solve_ms: list[float] = []

    for row in rows:
        simp = row.simplified
        full = row.full
        simp_avail = _metric(simp, "steady", "availability")
        full_avail = _metric(full, "steady", "availability")
        avail_abs_err = (
            abs(simp_avail - full_avail)
            if simp_avail is not None and full_avail is not None
            else None
        )
        mttdl_rel_err = _relative_error(
            _metric(simp, "mttdl", "mttdl_years"),
            _metric(full, "mttdl", "mttdl_years"),
        )
        cost_rel_err = _relative_error(
            _metric(simp, "steady", "cost_per_year"),
            _metric(full, "steady", "cost_per_year"),
        )

        if avail_abs_err is not None:
            availability_errors.append(avail_abs_err)
        if mttdl_rel_err is not None:
            mttdl_errors.append(mttdl_rel_err)
        if cost_rel_err is not None:
            cost_errors.append(cost_rel_err)

        simp_total_ms = _total_solve_wall_ms(simp)
        full_total_ms = _total_solve_wall_ms(full)
        if simp_total_ms is not None:
            simplified_solve_ms.append(simp_total_ms)
        if full_total_ms is not None:
            full_solve_ms.append(full_total_ms)

        print(
            f"{row.n:>3} {row.num_classes:>3}  {str(row.class_sizes):>11}  "
            f"{simp.num_states if simp.num_states is not None else '-':>11}  "
            f"{full.num_states if full.num_states is not None else '-':>11}  "
            f"{_fmt_ms(simp_total_ms)}      {_fmt_ms(full_total_ms)}    "
            f"{_fmt_abs(avail_abs_err):>13}  {_fmt_pct(mttdl_rel_err):>13}  "
            f"{_fmt_pct(cost_rel_err):>12}  {_row_status(simp):<9} {_row_status(full)}"
        )

    print("\nRuntime summary:")
    if simplified_solve_ms:
        print(
            "  SIMPLIFIED total solve wall ms "
            f"median={statistics.median(simplified_solve_ms):.1f}, "
            f"max={max(simplified_solve_ms):.1f}"
        )
    if full_solve_ms:
        print(
            "  FULL attempted-reference total solve wall ms "
            f"median={statistics.median(full_solve_ms):.1f}, "
            f"max={max(full_solve_ms):.1f}"
        )

    print("\nAccuracy loss vs FULL, on rows with a solved reference:")
    if availability_errors:
        print(f"  Availability max absolute error: {max(availability_errors):.3e}")
    if mttdl_errors:
        print(f"  MTTDL max relative error: {100.0 * max(mttdl_errors):.3f}%")
    if cost_errors:
        print(f"  Cost/year max relative error: {100.0 * max(cost_errors):.3f}%")

    skipped = [row for row in rows if row.full.skip_reason]
    if skipped:
        print(f"\nFULL reference skipped for {len(skipped)} rows due to state cap.")


def run_analysis() -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for n in CLUSTER_SIZES:
        for num_classes in range(1, n + 1):
            machines = _machine_classes(num_classes)
            configs = _configs_for(n, machines)
            print(f"[analysis] N={n} C={num_classes}", flush=True)
            simplified = _run_quality(
                configs,
                n=n,
                num_classes=num_classes,
                quality=QualityLevel.SIMPLIFIED,
            )
            full = _run_quality(
                configs,
                n=n,
                num_classes=num_classes,
                quality=QualityLevel.FULL,
                max_state_estimate=MAX_FULL_STATE_ESTIMATE,
            )
            rows.append(
                BenchmarkRow(
                    n=n,
                    num_classes=num_classes,
                    class_sizes=_class_sizes(n, num_classes),
                    machines=machines,
                    simplified=simplified,
                    full=full,
                )
            )
    return rows


def main() -> Path:
    rows = run_analysis()
    _print_summary(rows)
    out_path = write_csv(rows)
    print(f"\nwrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
