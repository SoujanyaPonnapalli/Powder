"""
Statistical validation of event timing in the simulator event log.

Runs long simulations with known input distributions and verifies that
the observed inter-event times in the event log are consistent with
those distributions using Kolmogorov–Smirnov tests.

Tests cover:
- Transient failure inter-arrival times (time between recovery and next failure)
- Recovery durations (time between failure and recovery)
- Spawn durations (time between spawn request and spawn completion)
- Snapshot download durations (sync durations that include snapshot downloads)
"""

from collections import defaultdict

import numpy as np
import pytest
from scipy.stats import kstest

from powder.simulation import (
    Seconds,
    hours,
    days,
    minutes,
    Constant,
    Exponential,
    Normal,
    NodeConfig,
    NodeState,
    NetworkState,
    EventType,
    ClusterState,
    NoOpStrategy,
    NodeReplacementStrategy,
    LeaderlessProtocol,
    Simulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Significance level for KS tests.  We use 1% to keep flakiness very low
# while retaining good power given the large sample sizes (hundreds–thousands).
KS_ALPHA = 0.01


def _make_config(
    failure_rate: float,
    recovery_rate: float,
    region: str = "us-east",
    snapshot_download_rate: float | None = None,
    spawn_time: float = minutes(10),
) -> NodeConfig:
    """Create a NodeConfig with Exponential failure/recovery distributions."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=failure_rate),
        recovery_dist=Exponential(rate=recovery_rate),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(10.0),
        snapshot_download_time_dist=(
            Exponential(rate=snapshot_download_rate)
            if snapshot_download_rate is not None
            else Constant(0)
        ),
        spawn_dist=Constant(spawn_time),
    )


def _extract_per_node_events(event_log, event_type):
    """Group events of a given type by target_id, preserving order."""
    by_node: dict[str, list] = defaultdict(list)
    for ev in event_log:
        if ev.event_type == event_type:
            by_node[ev.target_id].append(ev)
    return by_node


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventTimingDistributions:
    """Validate that inter-event intervals match expected distributions."""

    # -- Failure inter-arrival times -----------------------------------------

    def test_failure_inter_arrival_times(self):
        """Time between recovery and subsequent failure should follow failure_dist.

        The simulator schedules the *next* failure at
        ``current_time + recovery_time + next_failure_time`` inside
        ``_apply_node_failure``.  So the gap between a NODE_RECOVERY and the
        next NODE_FAILURE for the same node is exactly one sample from
        ``failure_dist``.
        """
        failure_rate = 1.0 / hours(10)  # MTTF = 10 hours
        recovery_rate = 1.0 / hours(2)  # MTTR = 2 hours

        config = _make_config(failure_rate, recovery_rate)
        nodes = {
            f"node{i}": NodeState(node_id=f"node{i}", config=config)
            for i in range(3)
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=42,
            log_events=True,
        )
        result = sim.run_for(hours(50_000))

        # Collect failure inter-arrival times: recovery → next failure
        failures = _extract_per_node_events(result.event_log, EventType.NODE_FAILURE)
        recoveries = _extract_per_node_events(result.event_log, EventType.NODE_RECOVERY)

        samples = []
        for nid in failures:
            # Build a sorted timeline of recoveries for this node
            rec_times = [e.time for e in recoveries.get(nid, [])]
            fail_times = [e.time for e in failures[nid]]

            # The very first failure is scheduled from t=0 (sim start)
            # Include it: gap = first_failure_time - 0
            prev_available_time = 0.0
            rec_idx = 0
            for ft in fail_times:
                # Find the most recent recovery before this failure
                while rec_idx < len(rec_times) and rec_times[rec_idx] <= ft:
                    prev_available_time = rec_times[rec_idx]
                    rec_idx += 1
                gap = ft - prev_available_time
                if gap > 0:
                    samples.append(gap)

        assert len(samples) > 100, f"Expected many samples, got {len(samples)}"

        # KS test against Exponential(rate=failure_rate)
        # scipy's expon has CDF = 1 - exp(-x / scale), scale = 1/rate
        stat, p_value = kstest(samples, "expon", args=(0, 1.0 / failure_rate))
        assert p_value > KS_ALPHA, (
            f"Failure inter-arrival times do not match Exponential(rate={failure_rate}): "
            f"KS stat={stat:.4f}, p={p_value:.4f}, n={len(samples)}"
        )

    # -- Recovery durations --------------------------------------------------

    def test_recovery_durations(self):
        """Time between failure and recovery should follow recovery_dist.

        The simulator schedules recovery at
        ``current_time + recovery_time`` inside ``_apply_node_failure``.
        """
        failure_rate = 1.0 / hours(10)
        recovery_rate = 1.0 / hours(2)

        config = _make_config(failure_rate, recovery_rate)
        nodes = {
            f"node{i}": NodeState(node_id=f"node{i}", config=config)
            for i in range(3)
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=123,
            log_events=True,
        )
        result = sim.run_for(hours(50_000))

        failures = _extract_per_node_events(result.event_log, EventType.NODE_FAILURE)
        recoveries = _extract_per_node_events(result.event_log, EventType.NODE_RECOVERY)

        samples = []
        for nid in failures:
            fail_times = [e.time for e in failures[nid]]
            rec_times = [e.time for e in recoveries.get(nid, [])]

            # Pair each failure with the next recovery
            rec_idx = 0
            for ft in fail_times:
                # Find the first recovery after this failure
                while rec_idx < len(rec_times) and rec_times[rec_idx] <= ft:
                    rec_idx += 1
                if rec_idx < len(rec_times):
                    duration = rec_times[rec_idx] - ft
                    if duration > 0:
                        samples.append(duration)
                    rec_idx += 1

        assert len(samples) > 100, f"Expected many samples, got {len(samples)}"

        stat, p_value = kstest(samples, "expon", args=(0, 1.0 / recovery_rate))
        assert p_value > KS_ALPHA, (
            f"Recovery durations do not match Exponential(rate={recovery_rate}): "
            f"KS stat={stat:.4f}, p={p_value:.4f}, n={len(samples)}"
        )

    # -- Spawn durations (Normal) ---------------------------------------------

    def test_spawn_durations(self):
        """Spawn duration should follow the Normal spawn_dist.

        Uses NodeReplacementStrategy to trigger spawns after data loss.
        The spawn action is created when the replacement timeout fires;
        NODE_SPAWN_COMPLETE fires spawn_dist.sample() seconds later.
        So spawn_complete_time − timeout_time = one sample from spawn_dist.

        Uses a 5-node cluster with aggressive data loss and safe_mode=False
        so that replacements cascade, producing many spawn events.
        """
        spawn_mean = minutes(5)
        spawn_std = minutes(1)

        fragile_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Exponential(rate=1.0 / hours(10)),
            log_replay_rate_dist=Constant(100.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Normal(mean=spawn_mean, std=spawn_std),
        )

        nodes = {
            f"node{i}": NodeState(node_id=f"node{i}", config=fragile_config)
            for i in range(5)
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=5,
        )
        strategy = NodeReplacementStrategy(
            failure_timeout=Seconds(30),
            default_node_config=fragile_config,
            safe_mode=False,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=strategy,
            protocol=LeaderlessProtocol(up_to_date_quorum=False),
            seed=7,
            log_events=True,
        )
        result = sim.run_for(hours(50_000))

        # Build a timeline: each NODE_REPLACEMENT_TIMEOUT triggers a spawn,
        # and the matching NODE_SPAWN_COMPLETE fires spawn_dist later.
        # The spawn action runs at cluster.current_time == timeout_time,
        # so spawn_complete_time − timeout_time = spawn_dist.sample().
        timeout_times: list[float] = []
        spawn_complete_times: list[float] = []

        for ev in result.event_log:
            if ev.event_type == EventType.NODE_REPLACEMENT_TIMEOUT:
                timeout_times.append(ev.time)
            elif ev.event_type == EventType.NODE_SPAWN_COMPLETE:
                spawn_complete_times.append(ev.time)

        # Pair each spawn_complete with the most recent preceding timeout
        samples = []
        t_idx = 0
        for sc_time in sorted(spawn_complete_times):
            while t_idx < len(timeout_times) - 1 and timeout_times[t_idx + 1] <= sc_time:
                t_idx += 1
            if t_idx < len(timeout_times) and timeout_times[t_idx] <= sc_time:
                samples.append(sc_time - timeout_times[t_idx])

        assert len(samples) > 20, f"Expected many spawn samples, got {len(samples)}"

        # KS test against Normal(mean=spawn_mean, std=spawn_std)
        # Normal distribution is clamped at 0, but with mean=300s and std=60s
        # the probability of sampling below 0 is negligible.
        stat, p_value = kstest(samples, "norm", args=(spawn_mean, spawn_std))
        assert p_value > KS_ALPHA, (
            f"Spawn durations do not match Normal(mean={spawn_mean}, std={spawn_std}): "
            f"KS stat={stat:.4f}, p={p_value:.4f}, n={len(samples)}"
        )

    # -- Snapshot download durations -----------------------------------------

    def test_snapshot_download_durations(self):
        """Sync duration ≈ snapshot download time when log replay is instant.

        By setting an extremely high log_replay_rate, the log-replay phase
        finishes in negligible time, so the total sync duration (recovery →
        sync_complete) is dominated entirely by the snapshot download.
        With aggressive log GC every sync *must* go through the snapshot
        path.

        Uses a 5-node cluster with infrequent failures relative to sync
        time so that donors rarely fail mid-sync, avoiding interrupted
        syncs that would bias the duration distribution.
        """
        failure_rate = 1.0 / hours(50)    # MTTF = 50 hours (infrequent)
        recovery_rate = 1.0 / hours(0.5)  # MTTR = 30 min (fast)
        snapshot_download_rate = 1.0 / hours(0.25)  # mean = 15 min (fast)

        # Extremely high replay rate so log replay is essentially instant
        log_replay_rate = 1e12

        # Very aggressive log retention → every sync requires a snapshot
        log_retention_ops = 1.0       # Practically zero retention
        snapshot_interval = 1.0       # Snapshot every 1 unit
        commit_rate = 1.0             # 1 unit/s

        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Exponential(rate=failure_rate),
            recovery_dist=Exponential(rate=recovery_rate),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(log_replay_rate),
            snapshot_download_time_dist=Exponential(rate=snapshot_download_rate),
            spawn_dist=Constant(minutes(10)),
        )
        nodes = {
            f"node{i}": NodeState(node_id=f"node{i}", config=config)
            for i in range(5)
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=5,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(
                commit_rate=commit_rate,
                snapshot_interval=snapshot_interval,
                log_retention_ops=log_retention_ops,
            ),
            seed=99,
            log_events=True,
        )
        result = sim.run_for(hours(100_000))

        # Sync duration = recovery → sync_complete for each node.
        # With log_retention_ops=1 and MTTR=30min, virtually every
        # recovery requires a snapshot sync (probability of recovery
        # in < 1s is negligible), so we include all durations.
        recoveries = _extract_per_node_events(result.event_log, EventType.NODE_RECOVERY)
        syncs = _extract_per_node_events(result.event_log, EventType.NODE_SYNC_COMPLETE)

        sync_durations = []
        for nid in syncs:
            rec_times = [e.time for e in recoveries.get(nid, [])]
            sync_times = [e.time for e in syncs[nid]]

            rec_idx = 0
            for st in sync_times:
                best_rec = None
                while rec_idx < len(rec_times) and rec_times[rec_idx] <= st:
                    best_rec = rec_times[rec_idx]
                    rec_idx += 1
                if best_rec is not None:
                    duration = st - best_rec
                    if duration > 0:
                        sync_durations.append(duration)

        assert len(sync_durations) > 100, (
            f"Expected many sync samples, got {len(sync_durations)}"
        )

        # With instant log replay, sync duration ≈ snapshot download time.
        # KS test against Exponential(rate=snapshot_download_rate).
        stat, p_value = kstest(
            sync_durations, "expon", args=(0, 1.0 / snapshot_download_rate)
        )
        assert p_value > KS_ALPHA, (
            f"Snapshot download durations do not match "
            f"Exponential(rate={snapshot_download_rate}): "
            f"KS stat={stat:.4f}, p={p_value:.4f}, n={len(sync_durations)}"
        )

    # -- Combined: mean inter-event time sanity check ------------------------

    def test_mean_failure_recovery_cycle_time(self):
        """The mean failure-recovery cycle time should approximate MTTF + MTTR.

        This is a simpler sanity check that doesn't require scipy: the
        average time for one full cycle (available → failed → recovered)
        should be close to MTTF + MTTR.
        """
        mttf = hours(10)
        mttr = hours(2)
        failure_rate = 1.0 / mttf
        recovery_rate = 1.0 / mttr

        config = _make_config(failure_rate, recovery_rate)
        nodes = {
            f"node{i}": NodeState(node_id=f"node{i}", config=config)
            for i in range(3)
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=0,
            log_events=True,
        )
        result = sim.run_for(hours(100_000))

        failures = _extract_per_node_events(result.event_log, EventType.NODE_FAILURE)

        # For each node, compute the average time between consecutive failures.
        # This should approximate MTTF + MTTR (one full cycle).
        cycle_times = []
        for nid in failures:
            times = [e.time for e in failures[nid]]
            for i in range(1, len(times)):
                cycle_times.append(times[i] - times[i - 1])

        assert len(cycle_times) > 100

        expected_cycle = mttf + mttr
        observed_mean = np.mean(cycle_times)

        # Allow 10% tolerance — with thousands of samples this should be tight
        assert abs(observed_mean - expected_cycle) / expected_cycle < 0.10, (
            f"Mean cycle time {observed_mean:.0f}s vs expected {expected_cycle:.0f}s "
            f"(MTTF={mttf:.0f} + MTTR={mttr:.0f})"
        )
