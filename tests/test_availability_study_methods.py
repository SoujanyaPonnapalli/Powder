"""Independent checks for the study's new numerical/statistical methods."""

import math

import numpy as np
import pytest
from scipy import sparse

from notebooks.availability_finite_horizon import dense_time_average
from notebooks.availability_skew_diagnostics import empirical_bernstein
from powder.markov import MarkovModel


def two_state(failure, recovery):
    return MarkovModel(
        Q=sparse.csr_matrix([[-failure, failure], [recovery, -recovery]]),
        initial_distribution=np.array([1., 0.]), live_mask=np.array([True, False]),
        state_costs=np.zeros(2),
    )


@pytest.mark.parametrize("horizon", [0., .5, 604800.])
def test_dense_integral_matches_analytic_two_state(horizon):
    model = two_state(1., 3.)
    average, diagnostics = dense_time_average(model, horizon)
    extra = -np.expm1(-4 * horizon) / (4 * horizon) if horizon else 1.
    assert average == pytest.approx([.75 + .25 * extra, .25 * (1 - extra)], abs=1e-11)
    assert abs(diagnostics["raw_average_mass_error"]) < 1e-8


def test_absorbing_outage_retains_finite_horizon_information():
    # A stationary-tail substitution would produce the wrong answer here.
    model = two_state(1e-6, 0.)
    horizon = 604800.
    average, _ = dense_time_average(model, horizon)
    expected_available = -math.expm1(-1e-6 * horizon) / (1e-6 * horizon)
    assert average[0] == pytest.approx(expected_available, abs=1e-12)


def test_zero_events_do_not_create_zero_uncertainty():
    bound = empirical_bernstein(1., 0., 1000)
    assert bound["ci"][0] < .99999
    assert bound["ci"][1] == 1
    assert not bound["precision_5e_6_certified"]


def test_bounded_interval_reflects_tail_and_sample_size():
    small = empirical_bernstein(.9999, .0028, 100000)
    large = empirical_bernstein(.9999, .0028, 1000000)
    assert small["radius"] > large["radius"] > 0
    assert 0 <= small["ci"][0] <= .9999 <= small["ci"][1] <= 1
    assert empirical_bernstein(.9999, 0., 100000)["radius"] < small["radius"]


@pytest.mark.parametrize("count,confidence", [(1,.99),(100,1.),(100,0.)])
def test_bound_rejects_invalid_inputs(count, confidence):
    with pytest.raises(ValueError):
        empirical_bernstein(.99, .01, count, confidence)
