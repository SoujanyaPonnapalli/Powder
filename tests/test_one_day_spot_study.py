"""Checks for full-day accounting and the sequential threshold decision."""
from dataclasses import replace
import math

import pytest

from notebooks.availability_one_day_spot import (
    ALPHA, HORIZON, VM_PROFILES, ConvergenceCriteria, Simulator,
    look_alpha, summary, threshold_verdict, node_config_for,
    make_cluster, replacement_strategy, raft_protocol, run_fixed_week,
)
from powder.simulation.distributions import Constant


def test_full_day_remains_in_denominator_after_data_loss():
    cfg = node_config_for(next(p for p in VM_PROFILES if p.name == 'Spot'))
    cfg = replace(cfg, failure_dist=Constant(float('inf')), data_loss_dist=Constant(100.))
    simulator = Simulator(make_cluster(3, cfg), replacement_strategy(cfg), raft_protocol(), seed=1)
    metrics, _, reason, ended = run_fixed_week(simulator, horizon=HORIZON)
    assert reason == 'data_loss' and ended == pytest.approx(100.)
    assert metrics.total_time() == pytest.approx(HORIZON)
    assert metrics.availability_fraction() == pytest.approx(100./HORIZON)


def test_sequential_error_budget_telescopes():
    n = 10000
    assert math.fsum(look_alpha(j) for j in range(1, n+1)) == pytest.approx(ALPHA*(1-1/(n+1)))
    with pytest.raises(ValueError):
        look_alpha(0)


def test_threshold_requires_interval_on_one_side():
    assert threshold_verdict([.998, .9989], .999) == 'disproved'
    assert threshold_verdict([.999, 1.], .999) == 'verified'
    assert threshold_verdict([.998, 1.], .999) == 'unresolved'


def make_summary(values):
    n = len(values)
    samples = {'availability': values, 'first_data_loss_seconds': [float('nan')]*n,
               'first_physical_quorum_loss_seconds': [float('nan')]*n}
    criteria = ConvergenceCriteria(confidence_level=.99, absolute_error=.0005)
    return summary(samples, 0., criteria, .999, look=1)


def test_zero_sample_variance_cannot_prematurely_verify_nines():
    row = make_summary([1.]*1000)
    assert row['native_precision_converged']
    assert row['threshold_verdict'] == 'unresolved'
    assert not row['adaptive_stop']


def test_decisive_threshold_still_requires_native_precision():
    row = make_summary([0., 1.]*5000)
    assert row['threshold_verdict'] == 'disproved'
    assert not row['native_precision_converged']
    assert not row['adaptive_stop']
    assert make_summary([.9]*10000)['adaptive_stop']
