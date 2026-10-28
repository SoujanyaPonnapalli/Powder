import math

import numpy as np
import pytest

from powder.faults.mc_scoring import (BatchSummary, MCScoringConfig, OnlineTInterval,
                                     certify_availability)


def config(**overrides):
    return MCScoringConfig(**{"min_runs": 100_000, "max_runs": 200_000,
                              "batch_runs": 10_000, **overrides})


def test_t_interval_merges_batches_without_losing_variance():
    first = np.array([.2, .4, .6, .8])
    second = np.array([.3, .5, .7])
    stats = OnlineTInterval()
    stats.add(BatchSummary.from_samples(first))
    stats.add(BatchSummary.from_samples(second))
    assert stats.count == 7
    assert stats.mean == pytest.approx(np.mean(np.r_[first, second]))
    assert stats.sample_std == pytest.approx(np.std(np.r_[first, second], ddof=1))


@pytest.mark.parametrize("mean,status", [(.999995, "pass"), (.999985, "fail")])
def test_certification_requires_at_least_one_hundred_thousand_runs(mean, status):
    seeds = []

    def sample(n, seed):
        seeds.append(seed)
        return BatchSummary(n, mean, .00001)

    result = certify_availability(sample, config(), base_seed=42)
    assert result.status == status
    assert result.runs == 100_000
    assert seeds == [42 + n for n in range(0, 100_000, 10_000)]
    if status == "pass":
        assert result.ci_low >= config().target_availability
    else:
        assert result.ci_high < config().target_availability


def test_straddling_interval_remains_undecided_at_run_cap():
    target = config().target_availability
    result = certify_availability(lambda n, seed: BatchSummary(n, target, .001),
                                  config(), base_seed=0)
    assert result.status == "undecided" and result.runs == 200_000
    assert result.ci_low < target < result.ci_high


def test_repeated_looks_and_multiple_candidates_widen_t_interval():
    settings = config(max_runs=100_000, batch_runs=50_000)
    mean = settings.target_availability + 2.1e-7
    sample = lambda n, seed: BatchSummary(n, mean, .00002)
    one = certify_availability(sample, settings, base_seed=0, comparisons=1)
    many = certify_availability(sample, settings, base_seed=0, comparisons=100)
    assert one.status == "pass"
    assert many.status == "undecided"
    assert many.ci_low < one.ci_low


def test_zero_variance_perfect_runs_do_not_falsely_certify_rare_events():
    result = certify_availability(lambda n, seed: BatchSummary(n, 1.0, 0),
                                  config(), base_seed=0)
    assert result.status == "undecided" and result.runs == 200_000
    assert result.sample_std == 0


def test_configuration_rejects_short_mc_and_invalid_targets():
    with pytest.raises(ValueError, match="100000"):
        config(min_runs=99_999)
    with pytest.raises(ValueError, match="Markov floor"):
        config(target_nines=4.0)
    assert math.isclose(config().markov_floor, 1 - 10 ** -4.5)
