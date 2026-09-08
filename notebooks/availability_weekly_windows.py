"""Periodic transient-failure hazard with independent per-machine phases.

Run from the repository root:
  .venv/bin/python -m notebooks.availability_weekly_windows --suite mc
  .venv/bin/python -m notebooks.availability_weekly_windows --suite markov

Preserve the old integrated failure intensity, one event per 30 calendar days
for an always-operational node. Put 70% of intensity in a one-day weekly window
and 30% in the other six days. Each machine independently draws its own fixed
weekly phase. Hazard is positive at every time. Failure clocks start again at
recovery and retain the same machine phase; new VMs receive independent phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from notebooks.availability_convergence_study import (
    ROOT, VM_PROFILES, THREAD_VARIABLES, np, scipy, os, days, make_cluster,
    node_config_for, raft_protocol, replacement_strategy, summarize,
)
from notebooks.availability_finite_horizon import dense_time_average
from notebooks.availability_skew_diagnostics import empirical_bernstein
from powder.markov_solver import steady_state
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation.distributions import Seconds
from powder.simulation.events import Event, EventType
from powder.simulation.simulator import Simulator

FOLDER = ROOT / "outputs/availability-weekly-windows-study"
HORIZON = days(7)


@dataclass(frozen=True)
class WeeklyWindowClock:
    concentrated_fraction: float = .7
    period: float = days(7)
    window: float = days(1)
    mean_interval: float = days(30)

    def __post_init__(self):
        if not 0 < self.concentrated_fraction < 1 or not 0 < self.window < self.period or self.mean_interval <= 0:
            raise ValueError("Invalid periodic hazard parameters")

    @property
    def weekly_hazard(self):
        return self.period / self.mean_interval

    @property
    def high_rate(self):
        return self.concentrated_fraction * self.weekly_hazard / self.window

    @property
    def low_rate(self):
        return (1-self.concentrated_fraction) * self.weekly_hazard / (self.period-self.window)

    @property
    def average_rate(self):
        return 1 / self.mean_interval

    def in_window(self, timestamp, phase):
        return (timestamp-phase) % self.period < self.window

    def delay_for_hazard(self, start, phase, hazard):
        """Invert the piecewise-linear integrated hazard exactly.

        Whole weeks have identical hazard regardless of starting phase. After
        skipping them, at most three partial constant-rate segments remain.
        """
        if hazard < 0 or not math.isfinite(hazard):
            raise ValueError("Hazard must be finite and nonnegative")
        cycles = math.floor(hazard / self.weekly_hazard)
        elapsed = cycles * self.period
        remaining = hazard - cycles * self.weekly_hazard
        position = (start-phase) % self.period
        while remaining > 0:
            high = position < self.window
            span = (self.window-position) if high else (self.period-position)
            rate = self.high_rate if high else self.low_rate
            if remaining <= rate*span:
                return elapsed + remaining/rate
            remaining -= rate*span
            elapsed += span
            position = self.window if high else 0.
        return elapsed

    def sample_delay(self, start, phase, rng):
        # An exponential draw in cumulative hazard does not imply exponential
        # calendar waiting times under this time-varying intensity.
        return self.delay_for_hazard(start, phase, float(rng.exponential()))


class WeeklyWindowSimulator(Simulator):
    def __init__(self, *args, windowed=True, **kwargs):
        self.windowed = windowed
        self.clock = WeeklyWindowClock()
        self.phases = {}
        self.failures_in_window = 0
        self.applied_transient_failures = 0
        super().__init__(*args, **kwargs)

    def _schedule_node_events(self, node):
        if self.windowed:
            phase = float(self.rng.uniform(0,self.clock.period))
            self.phases[node.node_id] = phase
            self.event_queue.push(Event(
                time=Seconds(self.cluster.current_time + self.clock.sample_delay(self.cluster.current_time,phase,self.rng)),
                event_type=EventType.NODE_FAILURE, target_id=node.node_id,
            ))
            self.event_queue.push(Event(
                time=Seconds(self.cluster.current_time + node.config.data_loss_dist.sample(self.rng)),
                event_type=EventType.NODE_DATA_LOSS, target_id=node.node_id,
            ))
        else:
            super()._schedule_node_events(node)

    def _apply_node_failure(self, event):
        if not self.windowed:
            return super()._apply_node_failure(event)
        node = self.cluster.get_node(event.target_id)
        if node and node.has_data:
            self.applied_transient_failures += 1
            self.failures_in_window += int(self.clock.in_window(event.time,self.phases[node.node_id]))
            node.is_available = False
            self.event_queue.cancel_events_for(event.target_id,EventType.NODE_SYNC_COMPLETE)
            node.sync = None
            self._cancel_syncs_from_donor(event.target_id)
            recovery_at = self.cluster.current_time + node.config.recovery_dist.sample(self.rng)
            self.event_queue.push(Event(time=Seconds(recovery_at),event_type=EventType.NODE_RECOVERY,target_id=node.node_id))
            delay = self.clock.sample_delay(recovery_at,self.phases[node.node_id],self.rng)
            self.event_queue.push(Event(time=Seconds(recovery_at+delay),event_type=EventType.NODE_FAILURE,target_id=node.node_id))


def run_fixed_week(simulator, horizon=HORIZON):
    """Continue through the horizon, preserving the first return for comparison.

    Simulator.run_until stops at actual data loss even when the MC config says
    stop_on_data_loss=False. Resume its existing event queue after such returns.
    If the queue empties, account for the unchanged final state through T.
    """
    result = simulator.run_until(end_time=horizon)
    legacy_availability = result.metrics.availability_fraction()
    first_end_reason, first_end_time = result.end_reason, float(result.end_time)
    while simulator.cluster.current_time < horizon:
        if result.end_reason == "no_events":
            elapsed = horizon - simulator.cluster.current_time
            simulator._advance_commit_index(elapsed)
            simulator.metrics.record_elapsed(horizon, simulator.cluster, simulator.protocol)
            simulator.metrics.update(simulator.cluster, horizon, simulator.protocol)
            simulator.cluster.current_time = horizon
            break
        result = simulator.run_until(end_time=horizon)
    metrics = simulator.metrics.snapshot()
    if abs(float(metrics.total_time())-horizon) > 1e-6:
        raise RuntimeError("Simulation did not account for the complete horizon")
    return metrics, legacy_availability, first_end_reason, first_end_time


def config_for(profile, mode):
    # Calendar-aware failure scheduling lives in the study simulator. The
    # unchanged 30-day exponential config is the mean-rate Markov proxy.
    return node_config_for(profile)


def simulate_case(profile, nodes, mode, runs, seed):
    cfg = config_for(profile, mode)
    values, legacy, failures = [], [], []
    in_window, applied_failures = 0, 0
    early_ends = []
    start = time.perf_counter()
    for i in range(runs):
        simulator = WeeklyWindowSimulator(
            initial_cluster=make_cluster(nodes, cfg), strategy=replacement_strategy(cfg),
            protocol=raft_protocol(), seed=seed+i, windowed=(mode=="weekly_windows"),
        )
        metrics, old_value, reason, ended = run_fixed_week(simulator)
        values.append(metrics.availability_fraction())
        legacy.append(old_value)
        failures.append(metrics.total_transient_failures)
        in_window += simulator.failures_in_window
        applied_failures += simulator.applied_transient_failures
        if ended < HORIZON:
            early_ends.append({"seed":seed+i, "reason":reason, "time":ended,
                               "legacy_availability":old_value, "weekly_availability":values[-1]})
        if (i+1) % 10000 == 0:
            print(f"{mode} {profile.name} N={nodes}: {i+1}/{runs}", flush=True)
    elapsed = time.perf_counter()-start
    samples = np.asarray(values)
    stats = summarize(samples, elapsed)
    batches = [summarize(samples[i:i+1000], elapsed*1000/runs) for i in range(0,runs,1000)]
    row = {"mode":mode, "profile":profile.name, "nodes":nodes, "seed":seed,
           "horizon_days":7, **stats, "mean_transient_failures":float(np.mean(failures)),
           "legacy_mean":float(np.mean(legacy)), "early_stop_count":len(early_ends),
           "early_stops":early_ends, "batches":batches,
           "passing_batches":sum(b["target_contained"] for b in batches),
           "bounded_ci99":empirical_bernstein(stats["mean"],stats["std"],runs),
           "tail_samples_below_99pct":[{"seed":seed+int(i),"availability":float(samples[i])}
                                       for i in np.flatnonzero(samples<.99)],
           "sample_sha256":hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest()}
    row["observed_failure_fraction_in_window"] = in_window/applied_failures if applied_failures else None
    row["applied_transient_failures"] = applied_failures
    return row


def distribution_audit():
    clock = WeeklyWindowClock()
    rng = np.random.default_rng(91000001)
    phases = rng.uniform(0,7,size=(10000,7))
    waits, positions, high_events = [], [], 0
    phase = phases[0,0]*days(1)
    timestamp = 0.
    for _ in range(200000):
        delay = clock.sample_delay(timestamp,phase,rng)
        timestamp += delay
        waits.append(delay/days(1))
        positions.append(((timestamp-phase)%clock.period)/days(1))
        high_events += int(clock.in_window(timestamp,phase))
    histogram_edges = np.linspace(0,7,57)
    result = {
        "seed":91000001, "events":len(waits),
        "description":"Always-operational single-machine clock; no recovery or replacement censoring",
        "observed_fraction_in_window":high_events/len(waits),
        "theoretical_fraction_in_window":.7,
        "mean_interevent_days":float(np.mean(waits)), "theoretical_mean_interevent_days":30,
        "high_rate_per_day":clock.high_rate*days(1),"low_rate_per_day":clock.low_rate*days(1),
        "phase_pairwise_correlations":np.corrcoef(phases.T).tolist(),
        "example_machine_window_starts_days":phases[:5].tolist(),
        "histogram_edges_days":histogram_edges.tolist(),
        "relative_event_phase_histogram":np.histogram(positions,bins=histogram_edges)[0].tolist(),
        "machine_phase_histogram":np.histogram(phases.ravel(),bins=histogram_edges)[0].tolist(),
    }
    (FOLDER/"distribution_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def markov_cases():
    for profile in VM_PROFILES:
        cfg = config_for(profile,"weekly_windows")
        control = config_for(profile,"exponential_30d")
        for nodes in (3,5,7):
            for quality in QualityLevel:
                start = time.perf_counter()
                model = build_markov_model([cfg]*nodes,raft_protocol(),replacement_strategy(cfg),quality)
                build_seconds = time.perf_counter()-start
                # Exact equality confirms these builders discard distribution shape.
                proxy = build_markov_model([control]*nodes,raft_protocol(),replacement_strategy(control),quality)
                if (model.Q != proxy.Q).nnz:
                    raise RuntimeError("Mean-matched generator mismatch")
                del proxy
                row = {"profile":profile.name,"nodes":nodes,"quality":quality.name,
                       "states":model.num_states,"build_seconds":build_seconds,
                       "interpretation":"Exponential mean-matched proxy; not shape-aware Markov",
                       "identical_generator_to_exponential_30d":True}
                if model.num_states > 2000:
                    yield {**row,"skipped":True,"reason":"Dense study cap of 2000 states"}
                    continue
                start = time.perf_counter()
                average, diagnostics = dense_time_average(model,HORIZON)
                solve_seconds = time.perf_counter()-start
                start = time.perf_counter()
                stationary = steady_state(model,backend="scipy")
                row.update({"skipped":False,"weekly_availability":1-float(average[~model.live_mask].sum()),
                            "steady_availability":1-float(stationary[~model.live_mask].sum()),
                            "weekly_solve_seconds":solve_seconds,"weekly_total_seconds":build_seconds+solve_seconds,
                            "steady_solve_seconds":time.perf_counter()-start,"diagnostics":diagnostics})
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",choices=("mc","markov","audit"),default="mc")
    parser.add_argument("--runs",type=int,default=100000,help="Main runs for Standard/Unreliable and Spot N=7")
    parser.add_argument("--spot-small-runs",type=int,default=10000)
    parser.add_argument("--control-runs",type=int,default=10000)
    args = parser.parse_args()
    if any(n<1000 or n%1000 for n in (args.runs,args.spot_small_runs,args.control_runs)):
        parser.error("Run counts must be positive multiples of 1000")
    if max(args.runs,args.spot_small_runs,args.control_runs)>1000000:
        parser.error("Run counts must not exceed disjoint seed-range width 1000000")
    FOLDER.mkdir(parents=True,exist_ok=True)
    if args.suite == "audit":
        print(json.dumps(distribution_audit()))
        return
    data = {"metadata":{
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "git_revision_before_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform(),
        "thread_limits":{k:os.environ[k] for k in THREAD_VARIABLES},"arguments":vars(args),
        "failure_distribution":"Periodic hazard: 70% intensity in one day/week; 30% in other six days; integral rate 1/30 days",
        "initial_clocks":"Each initially healthy machine draws an independent Uniform(0,7d) fixed window phase",
        "new_node_clocks":"Independent new phase for each new VM; recoveries retain the existing machine phase",
        "metric":"Full-week time-average, including continuation after early data-loss returns",
    },"results":[]}
    if args.suite == "markov":
        results = markov_cases()
    else:
        distribution_audit()
        cases=[]
        for mode_id,mode in enumerate(("weekly_windows","exponential_30d")):
            for profile_id,profile in enumerate(VM_PROFILES):
                for nodes in (3,5,7):
                    runs = args.control_runs if mode_id else (args.spot_small_runs if profile.name=="Spot" and nodes!=7 else args.runs)
                    seed=100000000+mode_id*100000000+profile_id*10000000+nodes*1000000
                    cases.append((profile,nodes,mode,runs,seed))
        results=(simulate_case(*case) for case in cases)
    for row in results:
        data["results"].append(row)
        (FOLDER/f"{args.suite}.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:v for k,v in row.items() if k not in ("batches","early_stops","tail_samples_below_99pct")}),flush=True)


if __name__ == "__main__":
    main()
