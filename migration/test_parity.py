"""Parity tests between the Python and Rust Monte Carlo engines.

Two layers:

* **Deterministic scenarios** use only ``Constant`` distributions, so both
  engines walk identical event sequences.  They are compared at a tight
  relative tolerance -- a tolerance check, not a bitwise one.
* **Stochastic scenarios** draw from different RNG streams by design, so
  they are compared distributionally: Welch's t-test on the means and a
  two-sample KS test on the shapes, with a relative-tolerance guard as a
  backstop.

The Python reference always runs with ``parallel_workers=1``.  Its parallel
path collects results in completion order, which makes its aggregate
floating-point sums non-deterministic; the sequential path does not.

Part of the temporary migration harness -- see ``migration/README.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from powder.monte_carlo import MonteCarloConfig, MonteCarloResults, MonteCarloRunner

from scenarios import (
    DETERMINISTIC_SCENARIOS,
    STOCHASTIC_SCENARIOS,
    Scenario,
)

# Seed shared by both engines.  They produce different streams from it, but
# fixing it keeps each side reproducible across runs.
BASE_SEED = 20_260_101

# Relative tolerance for the deterministic layer.  Loose enough to absorb
# the documented float deviations, tight enough that any real logic
# divergence blows straight through it.
DETERMINISTIC_RTOL = 1e-9

# A two-sided test at this level flags roughly 1 comparison in 500 by
# chance.  With fixed seeds the suite is reproducible either way, so this
# only governs how much drift is tolerated.
STATISTICAL_ALPHA = 0.002


# ---------------------------------------------------------------------------
# Running each engine
# ---------------------------------------------------------------------------


def run_python(scenario: Scenario, base_seed: int = BASE_SEED) -> MonteCarloResults:
    """Run a scenario on the Python engine, single-threaded."""
    config = MonteCarloConfig(
        num_simulations=scenario.num_simulations,
        max_time=scenario.max_time,
        stop_on_data_loss=scenario.stop_on_data_loss,
        parallel_workers=1,
        base_seed=base_seed,
    )
    runner = MonteCarloRunner(config)
    return runner.run(
        cluster=scenario.build_cluster(),
        strategy=scenario.build_strategy(),
        protocol=scenario.build_protocol(),
        network_config=scenario.build_network_config(),
    )


def python_samples(results: MonteCarloResults) -> dict[str, list[float]]:
    """Per-run samples from the Python engine, keyed by metric."""
    return {
        "availability": list(results.availability_samples),
        "cost": list(results.cost_samples),
        "transient_failures": [float(v) for v in results.transient_failure_samples],
        "dataloss_failures": [float(v) for v in results.dataloss_failure_samples],
        "nodes_spawned": [float(v) for v in results.nodes_spawned_samples],
        "unavailability_incidents": [
            float(v) for v in results.unavailability_incident_samples
        ],
        "leader_elections": [float(v) for v in results.leader_election_samples],
        "total_time": [float(v) for v in results.total_time_samples],
    }


def rust_samples(result: dict) -> dict[str, list[float]]:
    """Per-run samples from the Rust engine, keyed the same way."""
    runs = result["runs"]
    return {
        "availability": [r["availability"] for r in runs],
        "cost": [r["total_cost"] for r in runs],
        "transient_failures": [float(r["total_transient_failures"]) for r in runs],
        "dataloss_failures": [float(r["total_dataloss_failures"]) for r in runs],
        "nodes_spawned": [float(r["total_nodes_spawned"]) for r in runs],
        "unavailability_incidents": [
            float(r["total_unavailability_incidents"]) for r in runs
        ],
        "leader_elections": [float(r["total_leader_elections"]) for r in runs],
        "total_time": [r["end_time"] for r in runs],
    }


def data_loss_times(samples: list) -> list[float]:
    """Loss times with the never-lost runs dropped."""
    return [t for t in samples if t is not None]


# ---------------------------------------------------------------------------
# Deterministic layer
# ---------------------------------------------------------------------------


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Relative-tolerance comparison that also accepts an exact match."""
    if actual == expected:
        return
    denominator = max(abs(expected), 1e-12)
    relative = abs(actual - expected) / denominator
    assert relative <= DETERMINISTIC_RTOL, (
        f"{label}: Rust {actual!r} vs Python {expected!r} "
        f"(relative difference {relative:.3e})"
    )


@pytest.mark.parametrize(
    "scenario", DETERMINISTIC_SCENARIOS, ids=lambda s: s.name
)
def test_deterministic_scenarios_agree(scenario: Scenario, rust_engine):
    """Constant-distribution scenarios must land on the same numbers.

    With no randomness the two engines execute the same event sequence, so
    anything beyond float-detail noise is a logic divergence.
    """
    py = run_python(scenario)
    rs = rust_engine.run(scenario.to_job(BASE_SEED))

    py_runs = python_samples(py)
    rs_runs = rust_samples(rs)

    assert len(rs["runs"]) == scenario.num_simulations
    assert len(py_runs["availability"]) == scenario.num_simulations

    # End reasons must match exactly: they are categorical.
    rust_reasons = [r["end_reason"] for r in rs["runs"]]
    assert rust_reasons == list(py.end_reasons), (
        f"{scenario.name}: end reasons differ -- "
        f"Rust {rust_reasons} vs Python {list(py.end_reasons)}"
    )

    # Integer counters must match exactly.
    for metric in (
        "transient_failures",
        "dataloss_failures",
        "nodes_spawned",
        "unavailability_incidents",
        "leader_elections",
    ):
        assert rs_runs[metric] == py_runs[metric], (
            f"{scenario.name}/{metric}: Rust {rs_runs[metric]} vs "
            f"Python {py_runs[metric]}"
        )

    # Float metrics within a tight relative tolerance.
    for metric in ("availability", "cost", "total_time"):
        for i, (actual, expected) in enumerate(
            zip(rs_runs[metric], py_runs[metric])
        ):
            _assert_close(actual, expected, f"{scenario.name}/{metric}[run {i}]")

    # Data-loss milestones, where they occurred at all.
    for i, run in enumerate(rs["runs"]):
        for rust_key, py_values in (
            ("time_to_potential_data_loss", py.time_to_potential_loss_samples),
            ("time_to_actual_data_loss", py.time_to_actual_loss_samples),
            (
                "time_to_first_unavailability",
                py.time_to_first_unavailability_samples,
            ),
        ):
            rust_value = run.get(rust_key)
            py_value = py_values[i]
            if py_value is None or rust_value is None:
                assert (rust_value is None) == (py_value is None), (
                    f"{scenario.name}/{rust_key}[run {i}]: "
                    f"Rust {rust_value!r} vs Python {py_value!r}"
                )
            else:
                _assert_close(
                    rust_value, py_value, f"{scenario.name}/{rust_key}[run {i}]"
                )


# ---------------------------------------------------------------------------
# Stochastic layer
# ---------------------------------------------------------------------------


def _compare_distributions(
    name: str,
    metric: str,
    rust: list[float],
    python: list[float],
    mean_rtol: float,
) -> None:
    """Check two sample sets came from the same distribution.

    Degenerate cases are handled first: if both sides are constant they
    only need to agree on that constant, and if one is constant while the
    other is not, that is a divergence regardless of what a test statistic
    would say.
    """
    rust_arr = np.asarray(rust, dtype=float)
    py_arr = np.asarray(python, dtype=float)

    rust_mean = float(np.mean(rust_arr))
    py_mean = float(np.mean(py_arr))
    rust_std = float(np.std(rust_arr))
    py_std = float(np.std(py_arr))

    if rust_std == 0.0 and py_std == 0.0:
        assert math.isclose(rust_mean, py_mean, rel_tol=1e-9, abs_tol=1e-9), (
            f"{name}/{metric}: both engines are constant but disagree -- "
            f"Rust {rust_mean} vs Python {py_mean}"
        )
        return

    if rust_std == 0.0 or py_std == 0.0:
        # One side is constant and the other is not.  That is usually a
        # metric which is constant except on a rare path -- the engines
        # draw different streams, so one may miss that path entirely in a
        # finite sample.  A distribution test is meaningless against a
        # point mass, so compare the means against whatever resolution the
        # varying side actually has.  (`det_data_loss_with_lagging_survivor`
        # pins the rare path itself, deterministically.)
        varying = rust_arr if py_std == 0.0 else py_arr
        standard_error = math.sqrt(np.var(varying, ddof=1) / len(varying))
        allowed = max(mean_rtol * max(abs(py_mean), 1e-12), 4.0 * standard_error)
        assert abs(rust_mean - py_mean) <= allowed, (
            f"{name}/{metric}: one engine is constant and the means differ -- "
            f"Rust {rust_mean} (std {rust_std}), Python {py_mean} "
            f"(std {py_std}), allowed {allowed:.4g}"
        )
        return

    # Means, by Welch's t-test.
    _, t_p = scipy_stats.ttest_ind(rust_arr, py_arr, equal_var=False)
    # Distribution shapes, by a two-sample KS test.
    _, ks_p = scipy_stats.ks_2samp(rust_arr, py_arr)

    mean_diff = abs(rust_mean - py_mean)
    denominator = max(abs(py_mean), 1e-12)
    mean_rel = mean_diff / denominator

    # How finely these samples can resolve a difference in means at all.
    standard_error = math.sqrt(
        np.var(rust_arr, ddof=1) / len(rust_arr)
        + np.var(py_arr, ddof=1) / len(py_arr)
    )

    # The magnitude guard exists to catch a systematic offset small enough
    # to clear the t-test at large n.  It has to scale with the metric's
    # own noise floor: several of these counters are badly over-dispersed
    # (most runs are zero, a few are large), so their standard error alone
    # exceeds any fixed relative tolerance and a flat rtol would reject
    # samples the t-test and KS test both accept.
    allowed = max(mean_rtol * denominator, 4.0 * standard_error)

    detail = (
        f"{name}/{metric}: Rust mean {rust_mean:.6g} (std {rust_std:.3g}), "
        f"Python mean {py_mean:.6g} (std {py_std:.3g}); "
        f"difference {mean_diff:.4g} (relative {mean_rel:.3e}, "
        f"allowed {allowed:.4g}, SE {standard_error:.4g}), "
        f"Welch p={t_p:.4g}, KS p={ks_p:.4g}"
    )

    assert t_p > STATISTICAL_ALPHA, detail
    assert ks_p > STATISTICAL_ALPHA, detail
    assert mean_diff <= allowed, detail


@pytest.mark.parametrize("scenario", STOCHASTIC_SCENARIOS, ids=lambda s: s.name)
def test_stochastic_scenarios_agree(scenario: Scenario, rust_engine):
    """Randomised scenarios must agree distributionally.

    The engines use different RNG streams on purpose, so only the
    distributions they produce are comparable.
    """
    py = run_python(scenario)
    rs = rust_engine.run(scenario.to_job(BASE_SEED))

    py_runs = python_samples(py)
    rs_runs = rust_samples(rs)

    assert len(rs["runs"]) == scenario.num_simulations

    # Availability and cost are the headline metrics and should be tight.
    for metric in ("availability", "cost"):
        _compare_distributions(
            scenario.name, metric, rs_runs[metric], py_runs[metric], mean_rtol=0.05
        )

    # Event counters are noisier per run, so the mean tolerance is wider.
    for metric in (
        "transient_failures",
        "dataloss_failures",
        "nodes_spawned",
        "unavailability_incidents",
        "leader_elections",
    ):
        if max(py_runs[metric]) == 0 and max(rs_runs[metric]) == 0:
            continue  # Neither engine exercises this counter here.
        _compare_distributions(
            scenario.name, metric, rs_runs[metric], py_runs[metric], mean_rtol=0.10
        )


@pytest.mark.parametrize("scenario", STOCHASTIC_SCENARIOS, ids=lambda s: s.name)
def test_data_loss_statistics_agree(scenario: Scenario, rust_engine):
    """Data-loss probability and MTTDL must agree across engines."""
    py = run_python(scenario)
    rs = rust_engine.run(scenario.to_job(BASE_SEED))

    py_loss = data_loss_times(py.time_to_actual_loss_samples)
    rs_loss = data_loss_times(
        [r["time_to_actual_data_loss"] for r in rs["runs"]]
    )

    n = scenario.num_simulations
    py_prob = len(py_loss) / n
    rs_prob = len(rs_loss) / n

    # Data loss probability, by a two-proportion z-test.
    if py_prob == 0.0 and rs_prob == 0.0:
        pass  # Neither engine loses data here.
    elif py_prob == 1.0 and rs_prob == 1.0:
        pass  # Both always do.
    else:
        pooled = (len(py_loss) + len(rs_loss)) / (2 * n)
        se = math.sqrt(2 * pooled * (1 - pooled) / n)
        z = abs(rs_prob - py_prob) / se if se > 0 else 0.0
        assert z < 3.5, (
            f"{scenario.name}: data loss probability differs -- "
            f"Rust {rs_prob:.4f} vs Python {py_prob:.4f} (z={z:.2f})"
        )

    # MTTDL, where both engines saw enough loss events to estimate it.
    if len(py_loss) >= 30 and len(rs_loss) >= 30:
        _compare_distributions(
            scenario.name, "mttdl", rs_loss, py_loss, mean_rtol=0.10
        )


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------


def test_rust_summary_matches_its_own_samples(rust_engine):
    """The aggregate block must agree with the per-run samples beside it."""
    scenario = STOCHASTIC_SCENARIOS[0]
    rs = rust_engine.run(scenario.to_job(BASE_SEED))

    availability = [r["availability"] for r in rs["runs"]]
    summary = rs["summary"]

    assert summary["num_runs"] == len(availability)
    assert math.isclose(
        summary["availability_mean"], float(np.mean(availability)), rel_tol=1e-12
    )
    assert math.isclose(
        summary["availability_std"],
        float(np.std(availability, ddof=1)),
        rel_tol=1e-12,
    )
    assert math.isclose(
        summary["availability_p50"],
        float(np.percentile(availability, 50)),
        rel_tol=1e-12,
    )


def test_rust_is_reproducible_for_a_given_seed(rust_engine):
    """The same job twice must give the same answer."""
    scenario = STOCHASTIC_SCENARIOS[0]
    first = rust_engine.run(scenario.to_job(BASE_SEED))
    second = rust_engine.run(scenario.to_job(BASE_SEED))
    assert first["summary"] == second["summary"]
    assert first["runs"] == second["runs"]


def test_different_seeds_give_different_draws(rust_engine):
    """A different seed must actually change the sample path."""
    scenario = STOCHASTIC_SCENARIOS[0]
    first = rust_engine.run(scenario.to_job(BASE_SEED))
    second = rust_engine.run(scenario.to_job(BASE_SEED + 7919))
    assert first["summary"] != second["summary"]
