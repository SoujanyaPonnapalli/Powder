"""
Tests for the new simulation event counters.

Verifies that MetricsCollector and MetricsSnapshot correctly track:
- Total transient failures (NODE_FAILURE events)
- Total dataloss failures (NODE_DATA_LOSS events)
- Total nodes spawned (NODE_SPAWN_COMPLETE events)
- Total unavailability incidents (can-commit → cannot-commit transitions)
- Total leader elections (successful elections where a leader was picked)

Also verifies Monte Carlo aggregation of these counters.
"""

import numpy as np
import pytest

from powder.monte_carlo import (
    ConvergenceMetric,
    MonteCarloConfig,
    MonteCarloRunner,
    MonteCarloResults,
    run_monte_carlo,
    _get_metric_samples,
)
from powder.simulation import (
    Seconds,
    hours,
    days,
    minutes,
    Constant,
    Exponential,
    NodeConfig,
    NodeState,
    NetworkState,
    EventType,
    Event,
    ClusterState,
    NoOpStrategy,
    NodeReplacementStrategy,
    LeaderlessProtocol,
    RaftLikeProtocol,
    Simulator,
    MetricsCollector,
    MetricsSnapshot,
)


# =============================================================================
# Helper factories
# =============================================================================


def _make_stable_config(region: str = "us-east") -> NodeConfig:
    """Node that never fails within reasonable test windows."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Constant(days(9999)),
        recovery_dist=Constant(0),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(3.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(minutes(10)),
    )


def _make_fragile_config(
    failure_time: float = hours(2),
    recovery_time: float = hours(1),
    region: str = "us-east",
) -> NodeConfig:
    """Node that fails at a fixed time and recovers after a fixed duration."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Constant(failure_time),
        recovery_dist=Constant(recovery_time),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(3.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(minutes(10)),
    )


def _make_data_loss_config(data_loss_time: float, region: str = "us-east") -> NodeConfig:
    """Node that experiences permanent data loss at a fixed time."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Constant(days(9999)),
        recovery_dist=Constant(0),
        data_loss_dist=Constant(data_loss_time),
        log_replay_rate_dist=Constant(3.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


# =============================================================================
# MetricsCollector unit tests
# =============================================================================


class TestMetricsCollectorCounters:
    """Unit tests for the event counter methods on MetricsCollector."""

    def test_record_event_transient_failure(self):
        mc = MetricsCollector()
        mc.record_event(EventType.NODE_FAILURE)
        mc.record_event(EventType.NODE_FAILURE)
        assert mc.total_transient_failures == 2

    def test_record_event_data_loss(self):
        mc = MetricsCollector()
        mc.record_event(EventType.NODE_DATA_LOSS)
        assert mc.total_dataloss_failures == 1

    def test_record_event_spawn_complete(self):
        mc = MetricsCollector()
        mc.record_event(EventType.NODE_SPAWN_COMPLETE)
        mc.record_event(EventType.NODE_SPAWN_COMPLETE)
        mc.record_event(EventType.NODE_SPAWN_COMPLETE)
        assert mc.total_nodes_spawned == 3

    def test_record_event_other_types_ignored(self):
        mc = MetricsCollector()
        mc.record_event(EventType.NODE_RECOVERY)
        mc.record_event(EventType.NETWORK_OUTAGE_START)
        mc.record_event(EventType.LEADER_ELECTION_COMPLETE)
        assert mc.total_transient_failures == 0
        assert mc.total_dataloss_failures == 0
        assert mc.total_nodes_spawned == 0

    def test_unavailability_transition_basic(self):
        mc = MetricsCollector()
        # Starts available (_was_available=True)
        mc.record_unavailability_transition(can_commit=False, current_time=Seconds(100))  # True → False
        assert mc.total_unavailability_incidents == 1
        assert mc.time_to_first_unavailability == Seconds(100)

    def test_unavailability_transition_multiple(self):
        mc = MetricsCollector()
        mc.record_unavailability_transition(can_commit=False, current_time=Seconds(10))   # True → False: +1
        mc.record_unavailability_transition(can_commit=False, current_time=Seconds(20))   # False → False: no change
        mc.record_unavailability_transition(can_commit=True, current_time=Seconds(30))    # False → True: no change
        mc.record_unavailability_transition(can_commit=False, current_time=Seconds(40))   # True → False: +1
        assert mc.total_unavailability_incidents == 2
        # First unavailability was at t=10
        assert mc.time_to_first_unavailability == Seconds(10)

    def test_unavailability_transition_stays_available(self):
        mc = MetricsCollector()
        mc.record_unavailability_transition(can_commit=True, current_time=Seconds(10))
        mc.record_unavailability_transition(can_commit=True, current_time=Seconds(20))
        assert mc.total_unavailability_incidents == 0
        assert mc.time_to_first_unavailability is None

    def test_leader_election_counter(self):
        mc = MetricsCollector()
        mc.record_leader_election()
        mc.record_leader_election()
        assert mc.total_leader_elections == 2

    def test_snapshot_includes_all_counters(self):
        mc = MetricsCollector()
        mc.total_transient_failures = 5
        mc.total_dataloss_failures = 2
        mc.total_nodes_spawned = 3
        mc.total_unavailability_incidents = 4
        mc.total_leader_elections = 1

        snap = mc.snapshot()
        assert isinstance(snap, MetricsSnapshot)
        assert snap.total_transient_failures == 5
        assert snap.total_dataloss_failures == 2
        assert snap.total_nodes_spawned == 3
        assert snap.total_unavailability_incidents == 4
        assert snap.total_leader_elections == 1


# =============================================================================
# Simulator integration tests
# =============================================================================


class TestSimulatorCounters:
    """Integration tests verifying counters through actual simulation runs."""

    def test_transient_failure_counter(self):
        """Two nodes fail at hour 2, so we expect 2 transient failure events."""
        nodes = {
            "node0": NodeState(node_id="node0", config=_make_fragile_config(hours(2), hours(1))),
            "node1": NodeState(node_id="node1", config=_make_fragile_config(hours(2), hours(1))),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=0,
        )
        result = sim.run_for(hours(2.5))

        # Exactly 2 failures should have fired by hour 2.5
        assert result.metrics.total_transient_failures == 2
        assert result.metrics.total_dataloss_failures == 0

    def test_dataloss_failure_counter(self):
        """Three nodes lose data at hours 1, 2, 3."""
        nodes = {
            "node0": NodeState(node_id="node0", config=_make_data_loss_config(hours(1))),
            "node1": NodeState(node_id="node1", config=_make_data_loss_config(hours(2))),
            "node2": NodeState(node_id="node2", config=_make_data_loss_config(hours(3))),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=0,
        )
        result = sim.run_for(hours(4))

        # Sim stops at data loss (hour 3 when last up-to-date node loses data)
        assert result.end_reason == "data_loss"
        assert result.metrics.total_dataloss_failures == 3
        assert result.metrics.total_transient_failures == 0

    def test_unavailability_incident_counter(self):
        """Two nodes fail at hour 2, recover at hour 3 → 1 unavailability incident."""
        nodes = {
            "node0": NodeState(node_id="node0", config=_make_fragile_config(hours(2), hours(1))),
            "node1": NodeState(node_id="node1", config=_make_fragile_config(hours(2), hours(1))),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=0,
        )
        result = sim.run_for(hours(4))

        # One transition to unavailable when quorum is lost at hour 2
        assert result.metrics.total_unavailability_incidents >= 1
        # Verify availability reflects the downtime
        assert result.metrics.availability_fraction() == pytest.approx(3 / 4)

    def test_nodes_spawned_counter(self):
        """Node replacement strategy spawns a new node after data loss timeout."""
        default_config = _make_stable_config()
        strategy = NodeReplacementStrategy(
            failure_timeout=Seconds(60),
            default_node_config=default_config,
        )

        nodes = {
            "node0": NodeState(node_id="node0", config=_make_data_loss_config(hours(1))),
            "node1": NodeState(node_id="node1", config=_make_stable_config()),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=strategy,
            protocol=LeaderlessProtocol(up_to_date_quorum=False),
            seed=42,
        )
        result = sim.run_for(hours(3))

        # node0 loses data at hour 1, replacement timeout fires,
        # new node is spawned and completes
        assert result.metrics.total_nodes_spawned >= 1
        assert result.metrics.total_dataloss_failures >= 1

    def test_leader_election_counter_raft(self):
        """RaftLikeProtocol: leader fails, election should be counted."""
        election_dist = Constant(minutes(1))

        nodes = {
            "node0": NodeState(node_id="node0", config=_make_fragile_config(hours(1), hours(0.5))),
            "node1": NodeState(node_id="node1", config=_make_stable_config()),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=RaftLikeProtocol(election_time_dist=election_dist),
            seed=0,
        )
        result = sim.run_for(hours(2))

        # node0 becomes leader at start, fails at hour 1 → election
        assert result.metrics.total_leader_elections >= 1

    def test_leader_election_counter_leaderless_is_zero(self):
        """LeaderlessProtocol should never trigger leader elections."""
        nodes = {
            "node0": NodeState(node_id="node0", config=_make_fragile_config(hours(1), hours(0.5))),
            "node1": NodeState(node_id="node1", config=_make_stable_config()),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=0,
        )
        result = sim.run_for(hours(2))

        assert result.metrics.total_leader_elections == 0

    def test_counters_in_snapshot_match_collector(self):
        """Verify MetricsSnapshot fields match what the simulator produces."""
        nodes = {
            "node0": NodeState(node_id="node0", config=_make_fragile_config(hours(1), hours(0.5))),
            "node1": NodeState(node_id="node1", config=_make_stable_config()),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            seed=0,
        )
        result = sim.run_for(hours(2))

        snap = result.metrics
        assert isinstance(snap, MetricsSnapshot)
        # All counter fields should exist and be non-negative
        assert snap.total_transient_failures >= 0
        assert snap.total_dataloss_failures >= 0
        assert snap.total_nodes_spawned >= 0
        assert snap.total_unavailability_incidents >= 0
        assert snap.total_leader_elections >= 0


# =============================================================================
# Monte Carlo aggregation tests
# =============================================================================


class TestMonteCarloCounterAggregation:
    """Verify that event counters are correctly aggregated across MC runs."""

    def test_counters_propagated_to_results(self):
        """Run a small Monte Carlo and verify counter sample lists are populated."""
        nodes = {
            "node0": NodeState(node_id="node0", config=_make_fragile_config(hours(2), hours(1))),
            "node1": NodeState(node_id="node1", config=_make_stable_config()),
            "node2": NodeState(node_id="node2", config=_make_stable_config()),
        }
        cluster = ClusterState(
            nodes=nodes, network=NetworkState(), target_cluster_size=3,
        )

        results = run_monte_carlo(
            cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessProtocol(),
            num_simulations=5,
            max_time=hours(4),
            stop_on_data_loss=False,
            parallel_workers=1,
            seed=42,
        )

        # Should have 5 samples for each counter
        assert len(results.transient_failure_samples) == 5
        assert len(results.dataloss_failure_samples) == 5
        assert len(results.nodes_spawned_samples) == 5
        assert len(results.unavailability_incident_samples) == 5
        assert len(results.leader_election_samples) == 5

        # All transient failure counts should be > 0 (node0 always fails)
        assert all(s >= 1 for s in results.transient_failure_samples)
        # No data loss expected
        assert all(s == 0 for s in results.dataloss_failure_samples)
        # No leader elections with leaderless protocol
        assert all(s == 0 for s in results.leader_election_samples)

    def test_summary_includes_counters(self):
        """Verify summary() output mentions the new counter metrics."""
        results = MonteCarloResults()
        results.availability_samples = [0.99, 0.98]
        results.cost_samples = [10.0, 11.0]
        results.time_to_potential_loss_samples = [None, None]
        results.time_to_actual_loss_samples = [None, None]
        results.end_reasons = ["time_limit", "time_limit"]
        results.transient_failure_samples = [5, 3]
        results.dataloss_failure_samples = [0, 1]
        results.nodes_spawned_samples = [2, 1]
        results.unavailability_incident_samples = [1, 2]
        results.leader_election_samples = [3, 4]

        summary = results.summary()
        assert "Transient failures" in summary
        assert "Dataloss failures" in summary
        assert "Nodes spawned" in summary
        assert "Unavailability incidents" in summary
        assert "Leader elections" in summary

    def test_convergence_metric_samples_extraction(self):
        """Verify _get_metric_samples works for new convergence metrics."""
        results = MonteCarloResults()
        results.transient_failure_samples = [5, 3, 4]
        results.dataloss_failure_samples = [0, 1, 0]
        results.nodes_spawned_samples = [2, 1, 3]
        results.unavailability_incident_samples = [1, 2, 1]
        results.leader_election_samples = [3, 4, 2]

        tf = _get_metric_samples(results, ConvergenceMetric.TRANSIENT_FAILURES)
        assert list(tf) == [5.0, 3.0, 4.0]

        dl = _get_metric_samples(results, ConvergenceMetric.DATALOSS_FAILURES)
        assert list(dl) == [0.0, 1.0, 0.0]

        ns = _get_metric_samples(results, ConvergenceMetric.NODES_SPAWNED)
        assert list(ns) == [2.0, 1.0, 3.0]

        ui = _get_metric_samples(results, ConvergenceMetric.UNAVAILABILITY_INCIDENTS)
        assert list(ui) == [1.0, 2.0, 1.0]

        le = _get_metric_samples(results, ConvergenceMetric.LEADER_ELECTIONS)
        assert list(le) == [3.0, 4.0, 2.0]
"""Tests for the new simulation event counters."""
