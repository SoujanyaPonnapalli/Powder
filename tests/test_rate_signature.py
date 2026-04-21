"""
Tests for rate-class grouping at each QualityLevel.

``group_configs_into_classes`` partitions a ``list[NodeConfig]`` by
the rates the chosen quality level actually consumes. SIMPLIFIED and
COLLAPSED_PIPELINE use already-combined composite rates, so two
configs whose raw rates differ can still land in the same class when
their composites match; FULL separates every raw rate so no such
collapsing is possible.

This file exercises those invariants directly and checks that float
comparison uses a tolerance.
"""

from __future__ import annotations

import math

import pytest

from powder.markov_builders import (
    RateClass,
    extract_rates,
    group_configs_into_classes,
)
from powder.scenario import QualityLevel
from powder.simulation import (
    Constant,
    Exponential,
    NodeConfig,
    NodeReplacementStrategy,
    NoOpStrategy,
    Seconds,
)
from powder.simulation.distributions import days, hours, minutes


def _strategy(timeout_s: float = 300.0) -> NodeReplacementStrategy:
    return NodeReplacementStrategy(failure_timeout=Seconds(timeout_s))


def _make_config(
    *,
    cost_per_hour: float = 1.0,
    failure_hours: float = 12.0,
    recovery_s: float = 400.0,
    snapshot_s: float = 60.0,
    spawn_s: float = 120.0,
    data_loss_days: float = 365.0,
    region: str = "us-east",
) -> NodeConfig:
    return NodeConfig(
        region=region,
        cost_per_hour=cost_per_hour,
        failure_dist=Exponential(rate=1.0 / hours(failure_hours)),
        recovery_dist=Exponential(rate=1.0 / recovery_s),
        data_loss_dist=Exponential(rate=1.0 / days(data_loss_days)),
        log_replay_rate_dist=Constant(1e6),
        snapshot_download_time_dist=Constant(snapshot_s),
        spawn_dist=Constant(spawn_s),
    )


# ---------------------------------------------------------------------------
# Composite-rate collapsing
# ---------------------------------------------------------------------------


def test_simplified_groups_configs_with_matching_composites():
    """Different raw (recovery, snapshot, spawn) but same composites share a SIMPLIFIED class.

    Construction: two configs where recovery_mean + snapshot_mean and
    timeout_mean + spawn_mean + snapshot_mean are each identical.
    At SIMPLIFIED both composites appear in the signature; raw
    recovery/snapshot/spawn do not, so the configs should be merged.
    """
    # Config A: recovery=400s, snapshot=60s, spawn=120s (mean sum = 460 / 480)
    # Config B: recovery=340s, snapshot=120s, spawn=60s (mean sum = 460 / 480)
    a = _make_config(recovery_s=400.0, snapshot_s=60.0, spawn_s=120.0)
    b = _make_config(recovery_s=340.0, snapshot_s=120.0, spawn_s=60.0)

    strat = _strategy(timeout_s=300.0)

    rates_a = extract_rates(a, strat)
    rates_b = extract_rates(b, strat)
    # Sanity check our construction is meaningful.
    assert rates_a.recovery_rate != pytest.approx(rates_b.recovery_rate)
    assert rates_a.sync_rate != pytest.approx(rates_b.sync_rate)
    assert rates_a.spawn_rate != pytest.approx(rates_b.spawn_rate)
    assert rates_a.recovery_with_sync_rate == pytest.approx(
        rates_b.recovery_with_sync_rate,
    )
    assert rates_a.collapsed_replace_rate == pytest.approx(
        rates_b.collapsed_replace_rate,
    )

    classes = group_configs_into_classes([a, b], strat, QualityLevel.SIMPLIFIED)
    assert len(classes) == 1
    only = classes[0]
    assert only.size == 2
    assert only.member_indices == (0, 1)


def test_full_splits_configs_with_matching_composites():
    """Same SIMPLIFIED composites but different raw rates -> different FULL classes."""
    a = _make_config(recovery_s=400.0, snapshot_s=60.0, spawn_s=120.0)
    b = _make_config(recovery_s=340.0, snapshot_s=120.0, spawn_s=60.0)
    strat = _strategy(timeout_s=300.0)

    classes = group_configs_into_classes([a, b], strat, QualityLevel.FULL)
    assert len(classes) == 2
    assert {c.size for c in classes} == {1}
    assert [c.class_idx for c in classes] == [0, 1]


def test_collapsed_pipeline_requires_matching_recovery_and_sync():
    """COLLAPSED_PIPELINE keeps ``recovery_rate`` and ``sync_rate`` separate."""
    a = _make_config(recovery_s=400.0, snapshot_s=60.0, spawn_s=120.0)
    b = _make_config(recovery_s=340.0, snapshot_s=120.0, spawn_s=60.0)
    strat = _strategy(timeout_s=300.0)

    classes = group_configs_into_classes(
        [a, b], strat, QualityLevel.COLLAPSED_PIPELINE,
    )
    assert len(classes) == 2


# ---------------------------------------------------------------------------
# Tolerant comparison
# ---------------------------------------------------------------------------


def test_rate_signature_tolerant_to_tiny_float_noise():
    """np.isclose absorbs sub-ULP perturbations introduced by arithmetic."""
    base = _make_config()
    # Perturb failure rate by a tiny relative amount well within rtol=1e-9.
    perturb = NodeConfig(
        region=base.region,
        cost_per_hour=base.cost_per_hour,
        failure_dist=Exponential(
            rate=base.failure_dist.rate * (1.0 + 1e-12),
        ),
        recovery_dist=base.recovery_dist,
        data_loss_dist=base.data_loss_dist,
        log_replay_rate_dist=base.log_replay_rate_dist,
        snapshot_download_time_dist=base.snapshot_download_time_dist,
        spawn_dist=base.spawn_dist,
    )
    strat = _strategy()
    classes = group_configs_into_classes(
        [base, perturb], strat, QualityLevel.FULL,
    )
    assert len(classes) == 1


def test_rate_signature_splits_configs_above_tolerance():
    """Configs that differ by a clearly-above-tolerance amount split into two classes."""
    a = _make_config()
    b = _make_config(failure_hours=12.000_01)  # ~1e-6 relative drift
    strat = _strategy()
    classes = group_configs_into_classes([a, b], strat, QualityLevel.FULL)
    assert len(classes) == 2


# ---------------------------------------------------------------------------
# Homogeneous collapse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality", list(QualityLevel), ids=lambda q: q.name)
def test_uniform_configs_always_collapse_to_one_class(quality):
    cfg = _make_config()
    classes = group_configs_into_classes([cfg] * 5, _strategy(), quality)
    assert len(classes) == 1
    assert classes[0].size == 5
    assert classes[0].class_idx == 0


def test_cost_difference_splits_classes_even_with_identical_rates():
    """Cost is part of the signature so two configs that differ only in price split."""
    a = _make_config(cost_per_hour=1.0)
    b = _make_config(cost_per_hour=2.0)
    strat = _strategy()
    classes = group_configs_into_classes([a, b], strat, QualityLevel.SIMPLIFIED)
    assert len(classes) == 2


# ---------------------------------------------------------------------------
# Strategy propagation
# ---------------------------------------------------------------------------


def test_noop_strategy_zeroes_timeout_rate_and_still_groups():
    """Under NoOpStrategy, ``timeout_rate`` is 0 for every config."""
    cfg = _make_config()
    classes = group_configs_into_classes(
        [cfg, cfg], NoOpStrategy(), QualityLevel.SIMPLIFIED,
    )
    assert len(classes) == 1
    # collapsed_replace_rate is 0 when timeout_rate is 0; the signature
    # still matches so the two identical configs collapse.
    rc = classes[0]
    assert rc.rates.timeout_rate == 0.0
    assert rc.rates.collapsed_replace_rate == 0.0


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------


def test_class_order_is_stable_by_first_occurrence():
    a = _make_config(region="us-east")
    b = _make_config(region="us-west", failure_hours=24.0)
    classes = group_configs_into_classes(
        [a, b, a, b, a], _strategy(), QualityLevel.FULL,
    )
    assert len(classes) == 2
    assert classes[0].class_idx == 0 and classes[0].size == 3
    assert classes[0].member_indices == (0, 2, 4)
    assert classes[1].class_idx == 1 and classes[1].size == 2
    assert classes[1].member_indices == (1, 3)


def test_empty_input_returns_empty_list():
    assert group_configs_into_classes([], _strategy(), QualityLevel.FULL) == []
