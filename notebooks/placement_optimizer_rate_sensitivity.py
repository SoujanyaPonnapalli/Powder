#!/usr/bin/env python3
"""Run deterministic rate-error sensitivity experiments for the placement ILP.

The default run evaluates the nominal configuration plus the 72 requested
one-at-a-time rate errors. The --quick option runs the nominal case and +/-1%
for each parameter, which is useful as an end-to-end smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from powder.placement_optimizer import MachineType, PlacementSolverConfig
from powder.placement_sensitivity import (
    SensitivityRun,
    TypeCountMarkovScoreCache,
    run_one_at_a_time_sensitivity,
)
from powder.scenario import QualityLevel
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


NUM_RSMS = 100
BUDGET = 2_000.0
MINIMUM_SUM_AVAILABILITY = 99.996244
MINIMUM_PRODUCT_AVAILABILITY = 0.996251109
ERROR_LEVELS = (-0.15, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, 0.15)


def scenario_machine_types() -> tuple[MachineType, ...]:
    """Build the final low/medium/high machine scenario from the prior study."""
    specs = (
        ("low", 1.0, hours(48), days(365)),
        ("medium", 3.0, days(7), days(365)),
        ("high", 9.0, days(30), days(3 * 365)),
    )
    return tuple(
        MachineType(
            type_id=type_id,
            copies=200,
            capacity=1,
            node_config=NodeConfig(
                region=type_id,
                cost_per_hour=cost,
                failure_dist=Exponential(1.0 / transient_mtbf),
                recovery_dist=Exponential(1.0 / minutes(45)),
                data_loss_dist=Exponential(1.0 / data_loss_mttdl),
                log_replay_rate_dist=Constant(1_000_000.0),
                snapshot_download_time_dist=Constant(minutes(30)),
                spawn_dist=Exponential(1.0 / Seconds(60)),
            ),
        )
        for type_id, cost, transient_mtbf, data_loss_mttdl in specs
    )


def scenario_protocol_and_strategy() -> tuple[RaftLikeProtocol, NodeReplacementStrategy]:
    return (
        RaftLikeProtocol(election_time_dist=Exponential(1.0 / Seconds(5))),
        NodeReplacementStrategy(failure_timeout=hours(1), safe_mode=False),
    )


def _placement_mix(run: SensitivityRun) -> dict[str, int]:
    totals = {"low": 0, "medium": 0, "high": 0}
    for item in run.solution.selected:
        for type_id, replicas in zip(item.candidate.type_ids, item.candidate.counts):
            totals[type_id] += item.count * replicas
    return totals


def _solution_rows(runs: tuple[SensitivityRun, ...]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        mix = _placement_mix(run)
        rows.append(
            {
                "run": run.label,
                "parameter": run.parameter.key if run.parameter else "",
                "relative_error": run.relative_error,
                "status": run.solution.status,
                "cost_per_hour": run.solution.total_cost_per_hour,
                "predicted_sum_availability": run.predicted.sum_availability,
                "predicted_product_availability": run.predicted.product_availability,
                "nominal_sum_availability": run.nominal_rescore.sum_availability,
                "nominal_product_availability": run.nominal_rescore.product_availability,
                "nominal_sum_slo_met": run.nominal_rescore.sum_availability
                >= MINIMUM_SUM_AVAILABILITY,
                "nominal_product_slo_met": run.nominal_rescore.product_availability
                >= MINIMUM_PRODUCT_AVAILABILITY,
                "used_low": mix["low"],
                "used_medium": mix["medium"],
                "used_high": mix["high"],
                "placement": json.dumps(
                    [
                        {
                            "count": item.count,
                            "type_counts": dict(
                                zip(item.candidate.type_ids, item.candidate.counts)
                            ),
                        }
                        for item in run.solution.selected
                    ],
                    sort_keys=True,
                ),
            }
        )
    return rows


def _derivative_rows(runs: tuple[SensitivityRun, ...]) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        for derivative in run.derivatives:
            rows.append(
                {
                    "run": run.label,
                    "input_parameter": run.parameter.key if run.parameter else "",
                    "input_relative_error": run.relative_error,
                    "derivative_parameter": derivative.parameter.key,
                    "rate_per_second": derivative.rate,
                    "sum_raw": derivative.sum_raw,
                    "sum_elasticity": derivative.sum_elasticity,
                    "product_raw": derivative.product_raw,
                    "product_elasticity": derivative.product_elasticity,
                    "candidate_derivatives": json.dumps(
                        [asdict(candidate) for candidate in derivative.candidates],
                        sort_keys=True,
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "placement-rate-sensitivity",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    machine_types = scenario_machine_types()
    protocol, strategy = scenario_protocol_and_strategy()
    errors = (-0.01, 0.01) if args.quick else ERROR_LEVELS
    cache = TypeCountMarkovScoreCache(protocol, strategy, quality=QualityLevel.SIMPLIFIED)
    runs = run_one_at_a_time_sensitivity(
        machine_types,
        protocol,
        strategy,
        PlacementSolverConfig(
            num_rsms=NUM_RSMS,
            budget_per_hour=BUDGET,
            objective="min_cost",
        ),
        minimum_sum_availability=MINIMUM_SUM_AVAILABILITY,
        minimum_product_availability=MINIMUM_PRODUCT_AVAILABILITY,
        error_levels=errors,
        quality=QualityLevel.SIMPLIFIED,
        score_cache=cache,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solution_rows = _solution_rows(runs)
    derivative_rows = _derivative_rows(runs)
    _write_csv(args.output_dir / "placement_runs.csv", solution_rows)
    _write_csv(args.output_dir / "availability_derivatives.csv", derivative_rows)
    with (args.output_dir / "placement_sensitivity.json").open("w") as handle:
        json.dump(
            {
                "scenario": {
                    "quality": QualityLevel.SIMPLIFIED.name,
                    "num_rsms": NUM_RSMS,
                    "budget": BUDGET,
                    "minimum_sum_availability": MINIMUM_SUM_AVAILABILITY,
                    "minimum_product_availability": MINIMUM_PRODUCT_AVAILABILITY,
                    "error_levels": errors,
                },
                "runs": solution_rows,
                "derivatives": derivative_rows,
                "cache": {"hits": cache.hits, "misses": cache.misses},
            },
            handle,
            indent=2,
            allow_nan=False,
        )

    print(
        "run                              cost      predicted product  nominal product  low/med/high"
    )
    for row in solution_rows:
        print(
            f"{row['run']:<32} {row['cost_per_hour']:>8.2f}  "
            f"{row['predicted_product_availability']:.9f}  "
            f"{row['nominal_product_availability']:.9f}  "
            f"{row['used_low']}/{row['used_medium']}/{row['used_high']}"
        )
    print(f"cache: {cache.hits} hits, {cache.misses} Markov solves")


if __name__ == "__main__":
    main()
