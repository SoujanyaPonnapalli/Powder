"""Check periodic hazard inversion, phase independence, and full-week accounting."""

from dataclasses import replace

import pytest

from notebooks.availability_weekly_windows import (
    WeeklyWindowClock, WeeklyWindowSimulator, run_fixed_week, VM_PROFILES,
    node_config_for, make_cluster, raft_protocol, replacement_strategy, days,
)
from powder.simulation.distributions import Constant
from powder.simulation.events import EventQueue
from powder.simulation.simulator import Simulator


def integrated_hazard(clock, start, end, phase):
    # Independent closed form for cumulative time spent in high windows.
    def high_time(timestamp):
        cycles, position = divmod(timestamp-phase,clock.period)
        return cycles*clock.window + min(position,clock.window)
    high = high_time(end)-high_time(start)
    return clock.low_rate*(end-start)+(clock.high_rate-clock.low_rate)*high


@pytest.mark.parametrize("start_days,phase_days",[(0,0),(0,6.5),(1,0),(6.9,2.3),(70,6)])
@pytest.mark.parametrize("hazard",[0., .01, .7, 10.])
def test_delay_inverts_calendar_hazard(start_days,phase_days,hazard):
    clock = WeeklyWindowClock()
    start, phase = days(start_days), days(phase_days)
    delay = clock.delay_for_hazard(start,phase,hazard)
    assert delay >= 0
    assert integrated_hazard(clock,start,start+delay,phase) == pytest.approx(hazard,abs=1e-12)


def test_exact_rate_and_concentration():
    clock = WeeklyWindowClock()
    total = clock.high_rate*clock.window + clock.low_rate*(clock.period-clock.window)
    assert total/clock.period == pytest.approx(1/days(30))
    assert clock.high_rate*clock.window/total == pytest.approx(.7)
    assert clock.low_rate > 0
    assert clock.high_rate/clock.low_rate == pytest.approx(14)


def test_machine_phases_are_distinct_and_retained():
    cfg = node_config_for(VM_PROFILES[0])
    sim = WeeklyWindowSimulator(initial_cluster=make_cluster(7,cfg),strategy=replacement_strategy(cfg),protocol=raft_protocol(),seed=32)
    sim._initialize()
    phases = dict(sim.phases)
    assert len(set(phases.values())) == 7
    assert max(phases.values())-min(phases.values()) > days(1)
    sim.run_until(end_time=days(7))
    assert all(sim.phases[k] == phase for k,phase in phases.items())


def test_full_horizon_does_not_count_early_loss_as_perfect_availability():
    cfg = replace(node_config_for(VM_PROFILES[0]),failure_dist=Constant(days(1000)),data_loss_dist=Constant(1))
    sim = WeeklyWindowSimulator(initial_cluster=make_cluster(1,cfg),strategy=replacement_strategy(cfg),protocol=raft_protocol(),seed=1,windowed=False)
    metrics, legacy, reason, ended = run_fixed_week(sim,horizon=100.)
    assert legacy == 1
    assert reason == "data_loss" and ended == 1
    assert metrics.total_time() == 100
    assert metrics.availability_fraction() == pytest.approx(.01)


def test_control_preserves_original_engine_before_early_return():
    cfg = node_config_for(VM_PROFILES[1])
    kwargs = lambda: dict(initial_cluster=make_cluster(3,cfg),strategy=replacement_strategy(cfg),protocol=raft_protocol(),seed=93000000)
    old = Simulator(**kwargs()).run_until(end_time=days(7))
    _, legacy, reason, ended = run_fixed_week(WeeklyWindowSimulator(**kwargs(),windowed=False))
    assert legacy == old.metrics.availability_fraction()
    assert reason == old.end_reason
    assert ended == old.end_time


def test_empty_queue_still_accounts_for_complete_horizon():
    cfg = node_config_for(VM_PROFILES[0])
    sim = WeeklyWindowSimulator(initial_cluster=make_cluster(3,cfg),strategy=replacement_strategy(cfg),protocol=raft_protocol(),seed=1)
    sim._initialize()
    sim.event_queue = EventQueue()
    metrics, _, reason, ended = run_fixed_week(sim,horizon=100.)
    assert reason == "no_events" and ended == 0
    assert metrics.total_time() == 100
    assert metrics.availability_fraction() == 1
