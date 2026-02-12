"""
Tests for the Monte Carlo RSM simulator.

Tests cover distributions, node/cluster state, events, strategies,
protocols, the simulator engine, and adaptive Monte Carlo convergence.
"""

import numpy as np
import pytest

from powder.monte_carlo import (
    ConvergenceCriteria,
    ConvergenceMetric,
    ConvergenceResult,
    MonteCarloConfig,
    MonteCarloRunner,
    MonteCarloResults,
    estimate_required_runs,
    run_monte_carlo,
    run_monte_carlo_converged,
)
from powder.simulation import (
    # Distributions
    Seconds,
    hours,
    days,
    minutes,
    Exponential,
    Weibull,
    Normal,
    Uniform,
    Constant,
    # Node
    NodeConfig,
    NodeState,
    SyncState,
    # Network
    NetworkConfig,
    NetworkState,
    # Events
    EventType,
    Event,
    EventQueue,
    # Cluster
    ClusterState,
    # Strategy
    NoOpStrategy,
    SimpleReplacementStrategy,
    ActionType,
    # Protocol
    LeaderlessUpToDateQuorumProtocol,
    LeaderlessMajorityAvailableProtocol,
    RaftLikeProtocol,
    # Simulator
    Simulator,
)


# =============================================================================
# Time Unit Tests
# =============================================================================


class TestTimeUnits:
    def test_hours_conversion(self):
        assert hours(1) == 3600
        assert hours(2.5) == 9000

    def test_days_conversion(self):
        assert days(1) == 86400
        assert days(0.5) == 43200

    def test_minutes_conversion(self):
        assert minutes(1) == 60
        assert minutes(30) == 1800


# =============================================================================
# Distribution Tests
# =============================================================================


class TestDistributions:
    def test_exponential_sample(self):
        dist = Exponential(rate=1.0)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        # Mean should be approximately 1/rate = 1.0
        assert 0.8 < np.mean(samples) < 1.2
        # All samples should be positive
        assert all(s > 0 for s in samples)

    def test_exponential_invalid_rate(self):
        with pytest.raises(ValueError):
            Exponential(rate=0)
        with pytest.raises(ValueError):
            Exponential(rate=-1)

    def test_weibull_sample(self):
        dist = Weibull(shape=2.0, scale=1.0)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]
        assert all(s > 0 for s in samples)

    def test_normal_sample_with_min(self):
        dist = Normal(mean=5.0, std=2.0, min_val=0.0)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        # All samples should be >= min_val
        assert all(s >= 0 for s in samples)
        # Mean should be close to 5.0 (but affected by clamping)
        assert 4.0 < np.mean(samples) < 6.0

    def test_uniform_sample(self):
        dist = Uniform(low=10.0, high=20.0)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        assert all(10.0 <= s < 20.0 for s in samples)
        # Mean should be approximately 15
        assert 14.5 < np.mean(samples) < 15.5

    def test_constant_sample(self):
        dist = Constant(value=42.0)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(10)]
        assert all(s == 42.0 for s in samples)


# =============================================================================
# Node Tests
# =============================================================================


def make_test_node_config(region: str = "us-east") -> NodeConfig:
    """Create a test node configuration."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1 / hours(24)),  # ~1 failure per day
        recovery_dist=Constant(minutes(5)),
        data_loss_dist=Exponential(rate=1 / days(365)),  # ~1 per year
        log_replay_rate_dist=Constant(2.0),  # Replays 2x faster than commits
        snapshot_download_time_dist=Constant(0),  # No snapshot overhead
        spawn_dist=Constant(minutes(10)),
    )


class TestNodeState:
    def test_is_up_to_date(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_applied_index=100.0,
        )

        assert node.is_up_to_date(100.0)
        assert node.is_up_to_date(50.0)
        assert not node.is_up_to_date(150.0)

    def test_lag(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_applied_index=100.0,
        )

        assert node.lag(100.0) == 0
        assert node.lag(150.0) == 50
        assert node.lag(50.0) == 0  # Can't have negative lag


# =============================================================================
# Network Tests
# =============================================================================


class TestNetworkState:
    def test_is_region_down(self):
        state = NetworkState()

        assert not state.is_region_down("us-east")

        state.add_outage("us-east")
        assert state.is_region_down("us-east")

        state.remove_outage("us-east")
        assert not state.is_region_down("us-east")

    def test_is_partitioned(self):
        state = NetworkState()

        assert not state.is_partitioned("us-east", "us-west")

        state.add_outage("us-east")
        assert state.is_partitioned("us-east", "us-west")
        assert state.is_partitioned("us-west", "us-east")

        state.remove_outage("us-east")
        assert not state.is_partitioned("us-east", "us-west")

    def test_regions_reachable_from(self):
        state = NetworkState()
        all_regions = {"us-east", "us-west", "eu-west"}

        assert state.regions_reachable_from("us-east", all_regions) == all_regions

        state.add_outage("us-east")
        assert state.regions_reachable_from("us-east", all_regions) == set()
        assert state.regions_reachable_from("us-west", all_regions) == {"us-west", "eu-west"}


# =============================================================================
# Event Queue Tests
# =============================================================================


class TestEventQueue:
    def test_basic_ordering(self):
        queue = EventQueue()

        queue.push(Event(Seconds(10), EventType.NODE_FAILURE, "node1"))
        queue.push(Event(Seconds(5), EventType.NODE_RECOVERY, "node2"))
        queue.push(Event(Seconds(15), EventType.NODE_DATA_LOSS, "node3"))

        event1 = queue.pop()
        event2 = queue.pop()
        event3 = queue.pop()

        assert event1.time == 5
        assert event2.time == 10
        assert event3.time == 15

    def test_cancel_events(self):
        queue = EventQueue()

        queue.push(Event(Seconds(10), EventType.NODE_FAILURE, "node1"))
        queue.push(Event(Seconds(20), EventType.NODE_RECOVERY, "node1"))
        queue.push(Event(Seconds(15), EventType.NODE_FAILURE, "node2"))

        queue.cancel_events_for("node1", EventType.NODE_FAILURE)

        event1 = queue.pop()
        assert event1.target_id == "node2"
        assert event1.time == 15

        event2 = queue.pop()
        assert event2.target_id == "node1"
        assert event2.event_type == EventType.NODE_RECOVERY

    def test_is_empty(self):
        queue = EventQueue()
        assert queue.is_empty()

        queue.push(Event(Seconds(10), EventType.NODE_FAILURE, "node1"))
        assert not queue.is_empty()

        queue.pop()
        assert queue.is_empty()

    def test_cancel_all_events_for_target(self):
        """cancel_events_for without event_type cancels ALL event types."""
        queue = EventQueue()

        queue.push(Event(Seconds(10), EventType.NODE_FAILURE, "node1"))
        queue.push(Event(Seconds(20), EventType.NODE_RECOVERY, "node1"))
        queue.push(Event(Seconds(15), EventType.NODE_FAILURE, "node2"))

        queue.cancel_events_for("node1")  # no event_type → cancel all

        event1 = queue.pop()
        assert event1.target_id == "node2"
        assert event1.time == 15

        # node1 events are all gone
        assert queue.pop() is None


# =============================================================================
# Cluster State Tests
# =============================================================================


def make_test_cluster(num_nodes: int = 3) -> ClusterState:
    """Create a test cluster with the given number of nodes."""
    nodes = {}
    for i in range(num_nodes):
        config = make_test_node_config(region=f"region-{i % 3}")
        nodes[f"node{i}"] = NodeState(
            node_id=f"node{i}",
            config=config,
            last_applied_index=0.0,
        )

    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=num_nodes,
        current_time=Seconds(0),
    )


class TestClusterState:
    def test_node_counts(self):
        cluster = make_test_cluster(5)

        assert cluster.num_available() == 5
        assert cluster.num_up_to_date() == 5
        assert cluster.num_with_data() == 5

    def test_node_counts_with_failures(self):
        cluster = make_test_cluster(5)

        # Fail 2 nodes
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False

        assert cluster.num_available() == 3

        # Fail one more
        cluster.nodes["node2"].is_available = False
        assert cluster.num_available() == 2


class TestProtocolQuorumAndDataLoss:
    """Tests for protocol-level quorum, commit, and data loss methods."""

    def test_quorum_calculations(self):
        cluster = make_test_cluster(5)
        protocol = LeaderlessUpToDateQuorumProtocol()

        assert protocol.quorum_size(cluster) == 3
        assert cluster.num_available() == 5
        assert protocol.can_commit(cluster)

    def test_can_commit_with_failures(self):
        cluster = make_test_cluster(5)
        protocol = LeaderlessUpToDateQuorumProtocol()

        # Fail 2 nodes - should still have quorum
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False

        assert cluster.num_available() == 3
        assert protocol.can_commit(cluster)

        # Fail one more - no quorum
        cluster.nodes["node2"].is_available = False
        assert cluster.num_available() == 2
        assert not protocol.can_commit(cluster)

    def test_potential_data_loss(self):
        cluster = make_test_cluster(3)
        protocol = LeaderlessUpToDateQuorumProtocol()

        assert not protocol.has_potential_data_loss(cluster)

        # Lose quorum
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False

        assert protocol.has_potential_data_loss(cluster)

    def test_actual_data_loss(self):
        cluster = make_test_cluster(3)
        cluster.commit_index = 100.0
        protocol = LeaderlessUpToDateQuorumProtocol()

        # Mark all nodes as up-to-date initially
        for node in cluster.nodes.values():
            node.last_applied_index = 100.0

        assert not protocol.has_actual_data_loss(cluster)

        # Make one node lag and fail the up-to-date nodes
        cluster.nodes["node0"].last_applied_index = 50.0  # Lagging
        cluster.nodes["node1"].has_data = False  # Data lost
        cluster.nodes["node2"].has_data = False  # Data lost

        # node0 has data but is not up-to-date -> actual data loss
        assert protocol.has_actual_data_loss(cluster)


# =============================================================================
# Deterministic Metrics Tests
# =============================================================================


class TestDeterministicDataLossMetrics:
    """Exact-fraction tests with no randomness: only deterministic data-loss events."""

    @staticmethod
    def _make_data_loss_only_config(data_loss_time: float) -> NodeConfig:
        """Node that only experiences permanent data loss at a fixed time.

        No transient failures or network issues — the only event that fires
        within the test window is NODE_DATA_LOSS.
        """
        return NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),  # effectively never
            recovery_dist=Constant(0),
            data_loss_dist=Constant(data_loss_time),
            log_replay_rate_dist=Constant(3.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

    def test_availability_is_two_thirds(self):
        """3 nodes lose data at hours 1, 2, 3.

        Timeline
        --------
        0–1 h : 3 data-bearing nodes, quorum = 2  →  can_commit = True
        1–2 h : 2 data-bearing nodes, quorum = 2  →  can_commit = True
        2–3 h : 1 data-bearing node,  quorum = 2  →  can_commit = False
        3 h   : last node loses data                →  actual data loss, sim stops

        Availability = 2 h / 3 h = 2/3
        """
        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=self._make_data_loss_only_config(hours(1)),
            ),
            "node1": NodeState(
                node_id="node1",
                config=self._make_data_loss_only_config(hours(2)),
            ),
            "node2": NodeState(
                node_id="node2",
                config=self._make_data_loss_only_config(hours(3)),
            ),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(hours(4))

        # Simulation should stop on data loss at exactly 3 hours
        assert result.end_reason == "data_loss"
        assert result.end_time == pytest.approx(hours(3))

        # Availability must be exactly 2/3
        assert result.metrics.availability_fraction() == pytest.approx(2 / 3)

        # Potential data loss recorded at hour 2 (quorum lost)
        assert result.metrics.time_to_potential_data_loss == pytest.approx(hours(2))

        # Actual data loss recorded at hour 3 (all data gone)
        assert result.metrics.time_to_actual_data_loss == pytest.approx(hours(3))

        # Only data-loss events should have fired (no syncs, no transient failures)
        data_loss_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_DATA_LOSS
        ]
        sync_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
        ]
        assert len(data_loss_events) == 3
        assert len(sync_events) == 0


class TestDeterministicTransientFailureMetrics:
    """Exact-fraction test with deterministic transient failures only."""

    def test_availability_is_three_quarters(self):
        """Two nodes fail at hour 2, recover at hour 3. Run for 4 hours.

        Timeline  (3 nodes, quorum = 2)
        --------
        0–2 h : 3 nodes up-to-date      →  can_commit = True
        2–3 h : only node2 up-to-date    →  can_commit = False  (commit frozen)
        3–4 h : all 3 nodes up-to-date   →  can_commit = True
                (commit was frozen so recovered nodes are still up-to-date)

        Availability = 3 h / 4 h = 3/4
        """
        # node0 and node1: fail at 2 h, recover 1 h later
        fragile_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(hours(2)),
            recovery_dist=Constant(hours(1)),
            data_loss_dist=Constant(days(9999)),  # effectively never
            log_replay_rate_dist=Constant(3.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        # node2: never fails within the test window
        stable_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(3.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        nodes = {
            "node0": NodeState(node_id="node0", config=fragile_config),
            "node1": NodeState(node_id="node1", config=fragile_config),
            "node2": NodeState(node_id="node2", config=stable_config),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(hours(4))

        # No data loss — simulation runs to time limit
        assert result.end_reason == "time_limit"
        assert result.end_time == pytest.approx(hours(4))

        # Availability must be exactly 3/4
        assert result.metrics.availability_fraction() == pytest.approx(3 / 4)

        # Commit index should reflect the 3 hours of availability
        # (commit_rate = 1.0, so commit_index ≈ 3 hours in seconds)
        assert result.final_cluster.commit_index == pytest.approx(hours(3))

        # No sync events: commit was frozen during downtime so
        # recovered nodes are already up-to-date
        sync_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
        ]
        assert len(sync_events) == 0

        # Exactly 2 failure events and 2 recovery events within 4 h
        failure_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_FAILURE
        ]
        recovery_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_RECOVERY
        ]
        assert len(failure_events) == 2
        assert len(recovery_events) == 2


# =============================================================================
# Simulator Tests
# =============================================================================


class TestSimulator:
    def test_basic_simulation_runs(self):
        cluster = make_test_cluster(3)
        strategy = NoOpStrategy()

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=strategy,
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )

        result = simulator.run_for(Seconds(1000))

        assert result.end_time == 1000
        assert result.end_reason == "time_limit"
        assert result.metrics.total_time() == 1000

    def test_simulation_stops_on_data_loss(self):
        # Create a cluster with very high failure rates
        nodes = {}
        for i in range(3):
            config = NodeConfig(
                region="us-east",
                cost_per_hour=1.0,
                failure_dist=Exponential(rate=1 / 10),  # Failure every ~10s
                recovery_dist=Constant(100),  # Long recovery
                data_loss_dist=Constant(50),  # Data loss at 50s
                log_replay_rate_dist=Constant(2.0),
                snapshot_download_time_dist=Constant(0),
                spawn_dist=Constant(1000),
            )
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
            )

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )

        result = simulator.run_until_data_loss(max_time=Seconds(1000))

        # Should stop due to data loss before time limit
        assert result.end_reason == "data_loss"
        assert result.end_time < 1000

    def test_simulation_reproducibility(self):
        """Same seed should produce same results."""
        results = []

        for _ in range(2):
            cluster = make_test_cluster(3)
            simulator = Simulator(
                initial_cluster=cluster,
                strategy=NoOpStrategy(),
                protocol=LeaderlessUpToDateQuorumProtocol(),
                seed=12345,
            )
            result = simulator.run_for(Seconds(100))
            results.append(result)

        assert results[0].metrics.time_available == results[1].metrics.time_available
        assert results[0].metrics.time_unavailable == results[1].metrics.time_unavailable


# =============================================================================
# Strategy Tests
# =============================================================================


class TestSimpleReplacementStrategy:
    def test_spawns_replacement_on_data_loss(self):
        config = make_test_node_config()
        strategy = SimpleReplacementStrategy(default_node_config=config)

        # Create cluster and process a data loss event
        cluster = make_test_cluster(3)
        cluster.nodes["node0"].has_data = False  # Simulate data loss

        event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_DATA_LOSS,
            target_id="node0",
        )

        rng = np.random.default_rng(42)
        protocol = LeaderlessUpToDateQuorumProtocol()

        actions = strategy.on_event(event, cluster, rng, protocol)

        # Should spawn a replacement
        spawn_actions = [a for a in actions if a.action_type == ActionType.SPAWN_NODE]
        assert len(spawn_actions) == 1

    def test_starts_sync_on_recovery(self):
        config = make_test_node_config()
        strategy = SimpleReplacementStrategy(default_node_config=config)

        cluster = make_test_cluster(3)
        cluster.commit_index = 100.0

        # Node recovered but is lagging
        cluster.nodes["node0"].is_available = True
        cluster.nodes["node0"].last_applied_index = 50.0

        event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_RECOVERY,
            target_id="node0",
        )

        rng = np.random.default_rng(42)
        protocol = LeaderlessUpToDateQuorumProtocol()

        actions = strategy.on_event(event, cluster, rng, protocol)

        # Should start sync for lagging node
        sync_actions = [a for a in actions if a.action_type == ActionType.START_SYNC]
        assert len(sync_actions) == 1
        assert sync_actions[0].params["node_id"] == "node0"


# =============================================================================
# Protocol Tests
# =============================================================================


class TestLeaderlessUpToDateQuorumProtocol:
    """Tests for the default backward-compatible protocol."""

    def test_can_commit_with_up_to_date_quorum(self):
        """Default protocol should commit when a majority of up-to-date nodes exist."""
        cluster = make_test_cluster(5)
        protocol = LeaderlessUpToDateQuorumProtocol()

        # All up-to-date: should be able to commit
        assert protocol.can_commit(cluster) is True

        # Fail 2 nodes - still have 3 up-to-date (quorum)
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False
        assert protocol.can_commit(cluster) is True

        # Fail one more: lose quorum (only 2 of 5 available)
        cluster.nodes["node2"].is_available = False
        assert protocol.can_commit(cluster) is False

    def test_commit_rate_and_snapshot_interval(self):
        """Protocol should expose configurable commit_rate and snapshot_interval."""
        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=2.0,
            snapshot_interval=100.0,
        )
        assert protocol.commit_rate == 2.0
        assert protocol.snapshot_interval == 100.0

        # Defaults
        default = LeaderlessUpToDateQuorumProtocol()
        assert default.commit_rate == 1.0
        assert default.snapshot_interval == 0.0


class TestLeaderlessMajorityAvailableProtocol:
    """Tests for the majority-available protocol."""

    def test_available_but_not_up_to_date_can_commit(self):
        """Nodes that are available but lagging should still count for commit."""
        cluster = make_test_cluster(5)
        cluster.commit_index = 100.0

        protocol = LeaderlessMajorityAvailableProtocol()
        utd_protocol = LeaderlessUpToDateQuorumProtocol()

        # All nodes are available but none are up-to-date (last_applied_index=0)
        # Up-to-date quorum protocol would return False
        assert not utd_protocol.can_commit(cluster)

        # But majority-available protocol should return True (5 available >= quorum of 3)
        assert protocol.can_commit(cluster) is True

    def test_unavailable_nodes_dont_count(self):
        """Failed nodes should not count toward quorum."""
        cluster = make_test_cluster(5)
        cluster.commit_index = 100.0

        protocol = LeaderlessMajorityAvailableProtocol()

        # Fail 3 nodes
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False
        cluster.nodes["node2"].is_available = False

        # Only 2 available, need 3 for quorum
        assert protocol.can_commit(cluster) is False

    def test_simulation_more_available_than_up_to_date(self):
        """Protocol should give higher availability than up-to-date quorum when nodes lag."""
        # Create cluster where nodes recover slowly (will be available but lagging)
        nodes = {}
        for i in range(3):
            config = NodeConfig(
                region="us-east",
                cost_per_hour=1.0,
                failure_dist=Exponential(rate=1 / hours(2)),  # Fail every ~2h
                recovery_dist=Constant(minutes(5)),  # Quick recovery
                data_loss_dist=Exponential(rate=1 / days(365)),
                log_replay_rate_dist=Constant(1.5),  # Slow replay
                snapshot_download_time_dist=Constant(0),
                spawn_dist=Constant(minutes(10)),
            )
            nodes[f"node{i}"] = NodeState(node_id=f"node{i}", config=config)

        cluster_utd = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        # Run with up-to-date quorum
        sim_utd = Simulator(
            initial_cluster=cluster_utd,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )
        result_utd = sim_utd.run_for(days(30))

        # Run same scenario with majority-available protocol
        nodes2 = {}
        for i in range(3):
            config = NodeConfig(
                region="us-east",
                cost_per_hour=1.0,
                failure_dist=Exponential(rate=1 / hours(2)),
                recovery_dist=Constant(minutes(5)),
                data_loss_dist=Exponential(rate=1 / days(365)),
                log_replay_rate_dist=Constant(1.5),
                snapshot_download_time_dist=Constant(0),
                spawn_dist=Constant(minutes(10)),
            )
            nodes2[f"node{i}"] = NodeState(node_id=f"node{i}", config=config)

        cluster_avail = ClusterState(
            nodes=nodes2,
            network=NetworkState(),
            target_cluster_size=3,
        )

        sim_avail = Simulator(
            initial_cluster=cluster_avail,
            strategy=NoOpStrategy(),
            protocol=LeaderlessMajorityAvailableProtocol(),
            seed=42,
        )
        result_avail = sim_avail.run_for(days(30))

        # Majority-available should be at least as available (likely more)
        assert (
            result_avail.metrics.availability_fraction()
            >= result_utd.metrics.availability_fraction()
        )


class TestRaftLikeProtocol:
    """Tests for the leader-based Raft-like protocol."""

    def test_initial_leader_selection(self):
        """Protocol should pick an initial leader on simulation start."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        events = protocol.on_simulation_start(cluster, rng)

        assert protocol.leader_id is not None
        assert protocol.leader_id in cluster.nodes
        assert not protocol.election_in_progress
        assert events == []

    def test_leader_failure_triggers_election(self):
        """When the leader fails, an election should start."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        protocol.on_simulation_start(cluster, rng)
        leader_id = protocol.leader_id

        # Simulate leader failure
        cluster.nodes[leader_id].is_available = False
        cluster.current_time = Seconds(100)

        failure_event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_FAILURE,
            target_id=leader_id,
        )
        new_events = protocol.on_event(failure_event, cluster, rng)

        assert protocol.election_in_progress is True
        assert protocol.leader_id is None

        # Should have returned an election complete event
        assert len(new_events) == 1
        election_event = new_events[0]
        assert election_event.event_type == EventType.LEADER_ELECTION_COMPLETE
        assert election_event.time == Seconds(110)  # 100 + 10 (constant dist)

    def test_unavailable_during_election(self):
        """System should be unavailable during leader election."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        protocol.on_simulation_start(cluster, rng)

        # Before failure: should be able to commit
        assert protocol.can_commit(cluster) is True

        # Fail the leader
        leader_id = protocol.leader_id
        cluster.nodes[leader_id].is_available = False
        cluster.current_time = Seconds(100)

        failure_event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_FAILURE,
            target_id=leader_id,
        )
        protocol.on_event(failure_event, cluster, rng)

        # During election: cannot commit
        assert protocol.can_commit(cluster) is False

    def test_election_completes_with_new_leader(self):
        """Election should complete and pick a new leader."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        protocol.on_simulation_start(cluster, rng)
        old_leader_id = protocol.leader_id

        # Advance time and mark all nodes up-to-date
        cluster.current_time = Seconds(100)
        cluster.commit_index = 100.0
        for node in cluster.nodes.values():
            node.last_applied_index = 100.0

        # Fail the leader
        cluster.nodes[old_leader_id].is_available = False

        failure_event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_FAILURE,
            target_id=old_leader_id,
        )
        new_events = protocol.on_event(failure_event, cluster, rng)

        # Process the election complete event
        election_event = new_events[0]
        cluster.current_time = election_event.time

        # Keep non-failed nodes up-to-date at election completion time
        cluster.commit_index = 110.0
        for node in cluster.nodes.values():
            if node.is_available:
                node.last_applied_index = 110.0

        result_events = protocol.on_event(election_event, cluster, rng)

        # Election should be over with a new leader
        assert not protocol.election_in_progress
        assert protocol.leader_id is not None
        assert protocol.leader_id != old_leader_id
        assert protocol.can_commit(cluster) is True
        assert result_events == []

    def test_election_retries_when_no_eligible_node(self):
        """If no eligible node exists, election should retry."""
        cluster = make_test_cluster(3)
        cluster.current_time = Seconds(100)
        cluster.commit_index = 100.0

        # Make all nodes lag so none are up-to-date
        for node in cluster.nodes.values():
            node.last_applied_index = 0.0

        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        # Force election state
        protocol._election_in_progress = True

        election_event = Event(
            time=Seconds(100),
            event_type=EventType.LEADER_ELECTION_COMPLETE,
            target_id="protocol",
        )
        new_events = protocol.on_event(election_event, cluster, rng)

        # Should still be in election (no eligible node)
        assert protocol.election_in_progress is True
        assert protocol.leader_id is None

        # Should have returned a retry event
        assert len(new_events) == 1
        retry_event = new_events[0]
        assert retry_event.event_type == EventType.LEADER_ELECTION_COMPLETE
        assert retry_event.time == Seconds(110)

    def test_network_outage_on_leader_region_triggers_election(self):
        """Network outage in the leader's region should trigger election."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        protocol.on_simulation_start(cluster, rng)
        leader_id = protocol.leader_id
        leader_region = cluster.nodes[leader_id].config.region

        # Network outage in leader's region
        cluster.network.add_outage(leader_region)
        cluster.current_time = Seconds(50)

        outage_event = Event(
            time=Seconds(50),
            event_type=EventType.NETWORK_OUTAGE_START,
            target_id=leader_region,
            metadata={"region": leader_region},
        )
        new_events = protocol.on_event(outage_event, cluster, rng)

        assert protocol.election_in_progress is True
        assert protocol.leader_id is None
        assert len(new_events) == 1
        assert new_events[0].event_type == EventType.LEADER_ELECTION_COMPLETE

    def test_non_leader_failure_no_election(self):
        """Failure of a non-leader node should NOT trigger election."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))
        rng = np.random.default_rng(42)

        protocol.on_simulation_start(cluster, rng)
        leader_id = protocol.leader_id

        # Fail a non-leader node
        non_leader_ids = [nid for nid in cluster.nodes if nid != leader_id]
        failed_id = non_leader_ids[0]
        cluster.nodes[failed_id].is_available = False
        cluster.current_time = Seconds(100)

        failure_event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_FAILURE,
            target_id=failed_id,
        )
        new_events = protocol.on_event(failure_event, cluster, rng)

        # No election should be triggered
        assert not protocol.election_in_progress
        assert protocol.leader_id == leader_id
        assert new_events == []

    def test_full_simulation_with_raft_protocol(self):
        """End-to-end simulation with RaftLikeProtocol should run and produce metrics."""
        cluster = make_test_cluster(3)
        protocol = RaftLikeProtocol(election_time_dist=Constant(5.0))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
        )

        result = simulator.run_for(Seconds(1000))

        assert result.end_time == 1000
        assert result.end_reason == "time_limit"
        # Should have some unavailability due to elections
        assert result.metrics.total_time() == 1000

    def test_raft_less_available_than_leaderless(self):
        """Raft-like protocol should generally be less available than leaderless."""
        nodes_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Exponential(rate=1 / hours(12)),  # Moderate failures
            recovery_dist=Constant(minutes(5)),
            data_loss_dist=Exponential(rate=1 / days(3650)),  # Very rare data loss
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(minutes(10)),
        )

        # Run with leaderless protocol
        nodes1 = {
            f"node{i}": NodeState(node_id=f"node{i}", config=nodes_config)
            for i in range(5)
        }
        cluster1 = ClusterState(
            nodes=nodes1, network=NetworkState(), target_cluster_size=5
        )
        sim1 = Simulator(
            initial_cluster=cluster1,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )
        result_leaderless = sim1.run_for(days(30))

        # Run with Raft-like protocol (adds significant election downtime)
        nodes2 = {
            f"node{i}": NodeState(node_id=f"node{i}", config=nodes_config)
            for i in range(5)
        }
        cluster2 = ClusterState(
            nodes=nodes2, network=NetworkState(), target_cluster_size=5
        )
        sim2 = Simulator(
            initial_cluster=cluster2,
            strategy=NoOpStrategy(),
            protocol=RaftLikeProtocol(election_time_dist=Constant(minutes(2))),
            seed=42,
        )
        result_raft = sim2.run_for(days(30))

        # Raft should be less available due to election downtime
        assert (
            result_raft.metrics.availability_fraction()
            <= result_leaderless.metrics.availability_fraction()
        )

    def test_commit_rate_and_snapshot_interval(self):
        """RaftLikeProtocol should accept commit_rate and snapshot_interval."""
        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
            commit_rate=0.5,
            snapshot_interval=1000.0,
        )
        assert protocol.commit_rate == 0.5
        assert protocol.snapshot_interval == 1000.0


# =============================================================================
# Snapshot and Commit Index Tests
# =============================================================================


class TestCommitIndex:
    """Tests for the decoupled commit_index tracking."""

    def test_commit_index_advances_when_can_commit(self):
        """commit_index should advance only when the system can commit."""
        cluster = make_test_cluster(3)
        protocol = LeaderlessUpToDateQuorumProtocol(commit_rate=1.0)

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
        )

        result = simulator.run_for(Seconds(100))

        # commit_index should have advanced (system is available most of the time)
        assert result.final_cluster.commit_index > 0

    def test_commit_index_frozen_when_unavailable(self):
        """commit_index should NOT advance when the system cannot commit."""
        # Create a cluster that can't commit: all nodes lagging
        nodes = {}
        for i in range(3):
            config = make_test_node_config()
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_applied_index=0.0,
            )

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=100.0,  # Set commit_index ahead of nodes
        )

        # All nodes are at index 0, commit_index is 100 -> can't commit
        protocol = LeaderlessUpToDateQuorumProtocol(commit_rate=1.0)
        assert not protocol.can_commit(cluster)
        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
        )

        # Run briefly -- commit_index shouldn't advance since no quorum
        result = simulator.run_for(Seconds(10))
        # commit_index should still be 100 (system can't commit)
        assert result.final_cluster.commit_index == 100.0

    def test_variable_commit_rate(self):
        """Different commit_rate values should produce different commit_index advancement."""
        results = {}
        for rate in [0.5, 1.0, 2.0]:
            cluster = make_test_cluster(3)
            protocol = LeaderlessUpToDateQuorumProtocol(commit_rate=rate)
            simulator = Simulator(
                initial_cluster=cluster,
                strategy=NoOpStrategy(),
                protocol=protocol,
                seed=42,
            )
            result = simulator.run_for(Seconds(100))
            results[rate] = result.final_cluster.commit_index

        # Higher commit rate should produce larger commit_index
        assert results[0.5] < results[1.0] < results[2.0]


class TestSnapshotRecovery:
    """Tests for snapshot-aware node recovery."""

    def test_recovery_without_snapshot(self):
        """Node slightly behind should do log-only replay (no snapshot needed)."""
        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(365)),  # No failures during test
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(365)),
            log_replay_rate_dist=Constant(10.0),  # Fast replay
            snapshot_download_time_dist=Constant(minutes(5)),  # Slow snapshot
            spawn_dist=Constant(minutes(10)),
        )

        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=1.0,
            snapshot_interval=100.0,  # Snapshot every 100 units
        )

        # Create cluster with one lagging node (but AHEAD of the last snapshot)
        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_applied_index=150.0,
                last_snapshot_index=100.0,
            )

        # Make node0 lag, but still ahead of snapshot boundary
        nodes["node0"].last_applied_index = 110.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=150.0,
        )

        rng = np.random.default_rng(42)
        sync_time = protocol.compute_sync_time(nodes["node0"], cluster, rng)

        # Should be log-only: lag=40, net_rate=10-1=9 -> ~4.4s
        assert sync_time is not None
        assert sync_time < minutes(5)  # Much less than snapshot download time

    def test_recovery_with_snapshot(self):
        """Node far behind (outside log GC window) should download snapshot first."""
        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(365)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(365)),
            log_replay_rate_dist=Constant(10.0),
            snapshot_download_time_dist=Constant(60.0),  # 60s snapshot download
            spawn_dist=Constant(minutes(10)),
        )

        # log_retention_ops=100 means donor at 250 keeps log from 150–250.
        # Node at 50 is outside this window and must download a snapshot.
        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=1.0,
            snapshot_interval=100.0,
            log_retention_ops=100.0,
        )

        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_applied_index=250.0,
                last_snapshot_index=200.0,
            )

        # node0 is outside the donor's log GC window (at index 50, donor keeps 150-250)
        nodes["node0"].last_applied_index = 50.0
        nodes["node0"].last_snapshot_index = 0.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=250.0,
        )

        rng = np.random.default_rng(42)
        sync_time = protocol.compute_sync_time(nodes["node0"], cluster, rng)

        assert sync_time is not None
        # Should include snapshot download time (60s) + some log replay
        assert sync_time >= 60.0

    def test_no_snapshot_interval_means_log_only(self):
        """With snapshot_interval=0, recovery should always be log-only."""
        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(365)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(365)),
            log_replay_rate_dist=Constant(10.0),
            snapshot_download_time_dist=Constant(60.0),
            spawn_dist=Constant(minutes(10)),
        )

        # No snapshots
        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=1.0,
            snapshot_interval=0.0,
        )

        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_applied_index=1000.0,
                last_snapshot_index=0.0,
            )

        # node0 very far behind, but no snapshot truncation
        nodes["node0"].last_applied_index = 0.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=1000.0,
        )

        rng = np.random.default_rng(42)
        sync_time = protocol.compute_sync_time(nodes["node0"], cluster, rng)

        assert sync_time is not None
        # Should be log-only: lag=1000, net_rate=10-1=9 -> ~111s
        # Definitely should NOT include the 60s snapshot download
        assert sync_time < 120  # Well under 60 + 111

    def test_quorum_loss_freezes_commit_index(self):
        """When quorum is lost, commit_index should freeze, so recovery is instant."""
        # All nodes start up-to-date
        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(10.0),  # Fail at t=10
            recovery_dist=Constant(100.0),  # Recover at t=110
            data_loss_dist=Constant(days(365)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(days(365)),
        )

        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
            )

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        protocol = LeaderlessUpToDateQuorumProtocol(commit_rate=1.0)

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
            log_events=True,
        )

        result = simulator.run_for(Seconds(200))

        # The simulation ran; commit_index should reflect only time when quorum existed
        final = result.final_cluster
        assert final.commit_index < 200.0  # Less than wall-clock time (some unavailability)
        assert final.commit_index > 0.0  # But some commits happened

    def test_node_snapshot_state_advances(self):
        """Available nodes should take snapshots when commit_index crosses boundaries."""
        config = make_test_node_config()

        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
            )

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=1.0,
            snapshot_interval=50.0,  # Snapshot every 50 units
        )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
        )

        result = simulator.run_for(Seconds(200))

        # Nodes that remained available should have taken snapshots
        for node in result.final_cluster.nodes.values():
            if node.is_available and node.has_data:
                # Should have snapshot at some multiple of 50
                assert node.last_snapshot_index > 0
                # Snapshot should be at a multiple of the interval
                assert node.last_snapshot_index % 50.0 == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# Deterministic Sync Model Tests
# =============================================================================


class TestDeterministicSyncModel:
    """Deterministic tests for the new sync model with donor tracking,
    dynamic sync speed, and log GC-aware path selection.

    All distributions are Constant so that outcomes are fully predictable.
    """

    # -- helpers --

    @staticmethod
    def _stable_config(region: str = "us-east") -> NodeConfig:
        """Node that never fails or loses data within any reasonable test window."""
        return NodeConfig(
            region=region,
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

    @staticmethod
    def _stable_config_fast(region: str = "us-east") -> NodeConfig:
        """Node with log_replay_rate=3.0 that never fails."""
        return NodeConfig(
            region=region,
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(3.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

    @staticmethod
    def _fragile_config(
        failure_time: float, recovery_time: float, region: str = "us-east"
    ) -> NodeConfig:
        """Node that fails at *failure_time* and recovers *recovery_time* later."""
        return NodeConfig(
            region=region,
            cost_per_hour=1.0,
            failure_dist=Constant(failure_time),
            recovery_dist=Constant(recovery_time),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(3.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

    # ------------------------------------------------------------------ #
    # Test 1 – Basic sync timing                                          #
    # ------------------------------------------------------------------ #

    def test_basic_sync_timing(self):
        """3-node cluster, one node 10 units behind, syncs in exactly 10 s.

        Setup
        -----
        commit_rate   = 1.0   (1 unit / s)
        replay_rate   = 2.0   (2 units / s)
        net_rate      = 1.0   (closes 1 unit of gap / s)

        node0 starts at last_applied_index = 0, donors at 10.
        commit_index  = 10.

        Expected: gap = 10 → sync time = 10 / 1.0 = 10 s.
        """
        cfg = self._stable_config()

        nodes = {
            "node0": NodeState(node_id="node0", config=cfg,
                               last_applied_index=0.0),
            "node1": NodeState(node_id="node1", config=cfg,
                               last_applied_index=10.0),
            "node2": NodeState(node_id="node2", config=cfg,
                               last_applied_index=10.0),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=10.0,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(commit_rate=1.0),
            seed=0,
            log_events=True,
        )

        result = sim.run_for(Seconds(12))

        # Sync should have fired at t ≈ 10
        sync_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        assert len(sync_events) >= 1
        assert sync_events[0].time == pytest.approx(10.0, abs=0.01)

        # All nodes up-to-date at the end
        for node in result.final_cluster.nodes.values():
            assert node.last_applied_index >= result.final_cluster.commit_index - 0.01

    # ------------------------------------------------------------------ #
    # Test 2 – Sync pauses when donors are unavailable                    #
    # ------------------------------------------------------------------ #

    def test_sync_pauses_when_donors_unavailable(self):
        """Donor outage pauses sync; total wall-clock time increases.

        Setup
        -----
        commit_rate = 1.0, replay_rate = 3.0  →  net_rate = 2.0.
        node0 lag = 10 units  →  base sync time = 10 / 2 = 5 s.

        Donors (node1, node2) fail at t = 3, recover at t = 7 (4 s outage).

        Timeline
        --------
        [0, 3)  : net_rate 2.0, close 6 units.  gap: 10 → 4.
        [3, 7)  : outage.  Commit frozen, sync paused.  gap stays 4.
        t = 7   : node1 recovers first → only 1 up-to-date → can't commit.
                  node0 resumes sync at replay_rate = 3.0 (net_rate = 3.0
                  because commit is frozen with only 1 up-to-date node).
                  When node2 also recovers moments later → can_commit,
                  reschedule adjusts net_rate back to 2.0.
        gap = 4, net_rate = 2.0 → 2 s more.

        Sync completes at t ≈ 9.  (5 s productive + 4 s frozen)
        """
        lagging_cfg = self._stable_config_fast()
        donor_cfg = self._fragile_config(
            failure_time=3.0, recovery_time=4.0
        )

        nodes = {
            "node0": NodeState(node_id="node0", config=lagging_cfg,
                               last_applied_index=0.0),
            "node1": NodeState(node_id="node1", config=donor_cfg,
                               last_applied_index=10.0),
            "node2": NodeState(node_id="node2", config=donor_cfg,
                               last_applied_index=10.0),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=10.0,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(commit_rate=1.0),
            seed=0,
            log_events=True,
        )

        result = sim.run_for(Seconds(9.5))

        sync_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        assert len(sync_events) >= 1
        # Sync completes around t = 9 (5 s productive + 4 s outage)
        assert sync_events[0].time == pytest.approx(9.0, abs=0.2)

        # node0 should be caught up
        node0 = result.final_cluster.get_node("node0")
        assert node0.sync is None
        assert node0.last_applied_index >= result.final_cluster.commit_index - 0.1

    # ------------------------------------------------------------------ #
    # Test 3 – Multi-node sync with donor outage                          #
    # ------------------------------------------------------------------ #

    def test_multinode_sync_with_donor_outage(self):
        """Two lagging nodes pause during donor outage and resume correctly.

        Setup
        -----
        5 nodes.  commit_rate = 1.0, replay_rate = 3.0, net_rate = 2.0.
        node0 lag = 10, node1 lag = 20.  Donors: node2-4 at index 100.
        Donors fail at t = 2, recover at t = 8 (6 s outage).

        Timeline
        --------
        [0, 2)  : net_rate 2.0.  node0 gap 10→6.  node1 gap 20→16.
        [2, 8)  : outage.  Frozen.
        [8, ..  : resume.  node0 needs 6/2 = 3 s → t ≈ 11.
                           node1 needs 16/2 = 8 s → t ≈ 16.
        Next donor failure at t = 2+6+2 = 10 → before node1 finishes,
        but after node0 finishes (11 > 10). Let's adjust:

        Use failure_time = 2.0, recovery_time = 6.0 → next failure at
        2+6+2 = 10.  node0 sync done at t ≈ 11 → collision zone.

        To avoid: use recovery_time = 5.0 → recover at t=7, next fail
        at 7+2=9.  After t=7: node0 gap=6 → 6/2=3 → done t=10 → after
        next fail at t=9.

        Better: failure_time=2, recovery_time=6 → recover 8, next fail 10.
        node0 done ~11 but fail at 10 pauses it.  That's fine, the test
        just verifies both nodes eventually complete.

        We'll verify: after running long enough, both nodes are up-to-date.
        """
        lagging_cfg = self._stable_config_fast()
        donor_cfg = self._fragile_config(
            failure_time=2.0, recovery_time=6.0
        )

        nodes = {
            "node0": NodeState(node_id="node0", config=lagging_cfg,
                               last_applied_index=90.0),
            "node1": NodeState(node_id="node1", config=lagging_cfg,
                               last_applied_index=80.0),
            "node2": NodeState(node_id="node2", config=donor_cfg,
                               last_applied_index=100.0),
            "node3": NodeState(node_id="node3", config=donor_cfg,
                               last_applied_index=100.0),
            "node4": NodeState(node_id="node4", config=donor_cfg,
                               last_applied_index=100.0),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=5,
            commit_index=100.0,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(commit_rate=1.0),
            seed=0,
            log_events=True,
        )

        # Run long enough for multiple outage cycles
        result = sim.run_for(Seconds(40))

        # Both lagging nodes should be fully caught up
        node0 = result.final_cluster.get_node("node0")
        node1 = result.final_cluster.get_node("node1")
        commit = result.final_cluster.commit_index

        assert node0.sync is None, "node0 should have finished syncing"
        assert node1.sync is None, "node1 should have finished syncing"
        assert node0.last_applied_index >= commit - 0.1
        assert node1.last_applied_index >= commit - 0.1

        # Verify sync events fired for both nodes
        sync_node0 = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        sync_node1 = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node1"
        ]
        assert len(sync_node0) >= 1, "node0 should have completed sync"
        assert len(sync_node1) >= 1, "node1 should have completed sync"

    # ------------------------------------------------------------------ #
    # Test 4 – GC: log-only path (within retention window)                #
    # ------------------------------------------------------------------ #

    def test_gc_log_only_within_window(self):
        """When donor's log covers the lagging node's position, log-only is used.

        Setup
        -----
        commit_rate = 1.0, replay_rate = 10.0, net_rate = 9.0.
        snapshot_interval = 100, log_retention_ops = 300 (large window).
        Donor at 250, node0 at 150.  Donor keeps log from 0 to 250.
        Node0 is within GC window → log-only chosen.

        Sync time ≈ 100 / 9.0 ≈ 11.1 s  (no snapshot download overhead).
        """
        cfg = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(10.0),
            snapshot_download_time_dist=Constant(60.0),  # expensive snapshot
            spawn_dist=Constant(0),
        )

        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=1.0,
            snapshot_interval=100.0,
            log_retention_ops=300.0,  # keeps all log → node is inside window
        )

        nodes = {
            "node0": NodeState(node_id="node0", config=cfg,
                               last_applied_index=150.0,
                               last_snapshot_index=100.0),
            "node1": NodeState(node_id="node1", config=cfg,
                               last_applied_index=250.0,
                               last_snapshot_index=200.0),
            "node2": NodeState(node_id="node2", config=cfg,
                               last_applied_index=250.0,
                               last_snapshot_index=200.0),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=250.0,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        result = sim.run_for(Seconds(15))

        # Sync should complete well before 60 s (no snapshot downloaded)
        sync_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        assert len(sync_events) >= 1

        # lag=100, net_rate=9.0 → ~11.1 s.  Should be < 15 s (no snapshot).
        assert sync_events[0].time < 15.0
        # Must NOT include 60 s snapshot download overhead
        assert sync_events[0].time < 60.0

        # Verify node0 used log-only (no snapshot_download phase)
        node0 = result.final_cluster.get_node("node0")
        assert node0.sync is None

    # ------------------------------------------------------------------ #
    # Test 5 – GC: forced snapshot (outside retention window)             #
    # ------------------------------------------------------------------ #

    def test_gc_forced_snapshot_outside_window(self):
        """When donor's log is GC'd past the node's position, snapshot is forced.

        Setup
        -----
        commit_rate = 1.0, replay_rate = 10.0, net_rate = 9.0.
        snapshot_interval = 100, log_retention_ops = 100.
        Donor at 250 keeps log from 150–250.
        Node0 at 50: outside window (50 < 150).  Must download snapshot.

        Sync time ≈ 60 s (snapshot) + remaining_log / net_rate.
        Donor's latest snapshot index = floor(250/100)*100 = 200.
        After snapshot download (60 s), node jumps to 200.
        During download, donor advances 60 * 1.0 = 60 → donor at 310.
        Remaining log = 310 - 200 = 110.  Time = 110/9 ≈ 12.2 s.
        Total ≈ 72.2 s.
        """
        cfg = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(10.0),
            snapshot_download_time_dist=Constant(60.0),
            spawn_dist=Constant(0),
        )

        protocol = LeaderlessUpToDateQuorumProtocol(
            commit_rate=1.0,
            snapshot_interval=100.0,
            log_retention_ops=100.0,  # donor keeps log 150-250 only
        )

        nodes = {
            "node0": NodeState(node_id="node0", config=cfg,
                               last_applied_index=50.0,
                               last_snapshot_index=0.0),
            "node1": NodeState(node_id="node1", config=cfg,
                               last_applied_index=250.0,
                               last_snapshot_index=200.0),
            "node2": NodeState(node_id="node2", config=cfg,
                               last_applied_index=250.0,
                               last_snapshot_index=200.0),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=250.0,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        result = sim.run_for(Seconds(80))

        sync_events = [
            e for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        assert len(sync_events) >= 1

        # Must have taken >= 60 s (snapshot download is mandatory)
        assert sync_events[0].time >= 60.0

        # Total is snapshot (60) + log_suffix/net_rate ≈ 72.2 s
        assert sync_events[0].time == pytest.approx(72.2, abs=1.0)

        # node0 caught up
        node0 = result.final_cluster.get_node("node0")
        assert node0.sync is None
        assert node0.last_applied_index >= result.final_cluster.commit_index - 0.1


# =============================================================================
# Convergence Criteria Tests
# =============================================================================


class TestConvergenceCriteria:
    def test_defaults(self):
        criteria = ConvergenceCriteria()
        assert criteria.confidence_level == 0.95
        assert criteria.relative_error == 0.05
        assert criteria.absolute_error is None
        assert not criteria.uses_absolute_error
        assert criteria.metrics == [ConvergenceMetric.AVAILABILITY]
        assert criteria.min_runs == 30
        assert criteria.max_runs == 10_000
        assert criteria.batch_size == 10

    def test_absolute_error_mode(self):
        criteria = ConvergenceCriteria(absolute_error=0.01)
        assert criteria.absolute_error == 0.01
        assert criteria.relative_error is None
        assert criteria.uses_absolute_error
        assert criteria.error_threshold == 0.01

    def test_relative_error_mode(self):
        criteria = ConvergenceCriteria(relative_error=0.10)
        assert criteria.relative_error == 0.10
        assert criteria.absolute_error is None
        assert not criteria.uses_absolute_error
        assert criteria.error_threshold == 0.10

    def test_cannot_specify_both_errors(self):
        with pytest.raises(ValueError, match="exactly one"):
            ConvergenceCriteria(relative_error=0.05, absolute_error=0.01)

    def test_invalid_confidence_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            ConvergenceCriteria(confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            ConvergenceCriteria(confidence_level=1.0)
        with pytest.raises(ValueError, match="confidence_level"):
            ConvergenceCriteria(confidence_level=1.5)

    def test_invalid_relative_error(self):
        with pytest.raises(ValueError, match="relative_error"):
            ConvergenceCriteria(relative_error=0.0)
        with pytest.raises(ValueError, match="relative_error"):
            ConvergenceCriteria(relative_error=-0.1)

    def test_invalid_absolute_error(self):
        with pytest.raises(ValueError, match="absolute_error"):
            ConvergenceCriteria(absolute_error=0.0)
        with pytest.raises(ValueError, match="absolute_error"):
            ConvergenceCriteria(absolute_error=-0.01)

    def test_invalid_min_runs(self):
        with pytest.raises(ValueError, match="min_runs"):
            ConvergenceCriteria(min_runs=1)

    def test_invalid_max_runs(self):
        with pytest.raises(ValueError, match="max_runs"):
            ConvergenceCriteria(min_runs=100, max_runs=50)


# =============================================================================
# Adaptive Monte Carlo Tests
# =============================================================================


def _make_fast_cluster() -> ClusterState:
    """Create a fast cluster for convergence tests (short simulations)."""
    config = NodeConfig(
        region="us-east",
        cost_per_hour=1.0,
        failure_dist=Exponential(rate=1 / hours(24)),
        recovery_dist=Constant(minutes(5)),
        data_loss_dist=Exponential(rate=1 / days(365)),
        log_replay_rate_dist=Constant(2.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(minutes(10)),
    )
    nodes = {
        f"node{i}": NodeState(node_id=f"node{i}", config=config)
        for i in range(3)
    }
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=3,
    )


class TestAdaptiveMonteCarlo:
    def test_run_until_converged_basic(self):
        """Adaptive runner should produce results and report convergence status."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            stop_on_data_loss=True,
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        convergence = ConvergenceCriteria(
            confidence_level=0.95,
            relative_error=0.10,  # 10% relative error (easy to converge)
            metrics=[ConvergenceMetric.AVAILABILITY],
            min_runs=10,
            max_runs=500,
            batch_size=10,
        )

        result = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=convergence,
        )

        assert isinstance(result, ConvergenceResult)
        assert result.total_runs >= convergence.min_runs
        assert result.total_runs <= convergence.max_runs
        assert len(result.results.availability_samples) == result.total_runs
        assert len(result.metric_statuses) == 1
        assert result.metric_statuses[0].metric == ConvergenceMetric.AVAILABILITY

    def test_convergence_reduces_ci(self):
        """Running more simulations should reduce the confidence interval."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        # Run with tight convergence to force many runs
        loose = ConvergenceCriteria(
            confidence_level=0.95,
            relative_error=0.20,
            min_runs=10,
            max_runs=500,
            batch_size=10,
        )
        tight = ConvergenceCriteria(
            confidence_level=0.95,
            relative_error=0.03,
            min_runs=10,
            max_runs=5000,
            batch_size=20,
        )

        result_loose = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=loose,
        )
        result_tight = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=tight,
        )

        # Tighter convergence should require more runs
        assert result_tight.total_runs >= result_loose.total_runs

    def test_max_runs_respected(self):
        """Runner should stop at max_runs even if not converged."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        convergence = ConvergenceCriteria(
            confidence_level=0.99,
            relative_error=0.001,  # Extremely tight - unlikely to converge
            min_runs=5,
            max_runs=20,
            batch_size=5,
        )

        result = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=convergence,
        )

        assert result.total_runs == convergence.max_runs

    def test_multiple_metrics(self):
        """Should check convergence for all specified metrics."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        convergence = ConvergenceCriteria(
            confidence_level=0.95,
            relative_error=0.10,
            metrics=[
                ConvergenceMetric.AVAILABILITY,
                ConvergenceMetric.COST,
            ],
            min_runs=10,
            max_runs=500,
            batch_size=10,
        )

        result = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=convergence,
        )

        assert len(result.metric_statuses) == 2
        metric_names = {s.metric for s in result.metric_statuses}
        assert ConvergenceMetric.AVAILABILITY in metric_names
        assert ConvergenceMetric.COST in metric_names

    def test_progress_callback_called(self):
        """Progress callback should be invoked during convergence run."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        convergence = ConvergenceCriteria(
            confidence_level=0.95,
            relative_error=0.20,
            min_runs=10,
            max_runs=100,
            batch_size=10,
        )

        progress_calls = []

        def on_progress(completed: int, estimated_total: int, converged: bool):
            progress_calls.append((completed, estimated_total, converged))

        runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=convergence,
            progress_callback=on_progress,
        )

        assert len(progress_calls) >= 1
        # First call should be after min_runs
        assert progress_calls[0][0] == convergence.min_runs
        # Completed count should be non-decreasing
        completed_counts = [c[0] for c in progress_calls]
        assert completed_counts == sorted(completed_counts)

    def test_convenience_function(self):
        """run_monte_carlo_converged convenience function should work."""
        result = run_monte_carlo_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            max_time=days(30),
            confidence_level=0.95,
            relative_error=0.15,
            seed=42,
            min_runs=10,
            max_runs=200,
            batch_size=10,
        )

        assert isinstance(result, ConvergenceResult)
        assert result.total_runs >= 10
        assert len(result.results.availability_samples) == result.total_runs

    def test_estimate_required_runs(self):
        """estimate_required_runs should give a reasonable estimate from pilot data."""
        # Run a small pilot
        pilot = run_monte_carlo(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            num_simulations=30,
            max_time=days(30),
            seed=42,
        )

        convergence = ConvergenceCriteria(
            confidence_level=0.95,
            relative_error=0.05,
        )

        estimates = estimate_required_runs(pilot, convergence)

        assert ConvergenceMetric.AVAILABILITY in estimates
        # Should estimate more than the pilot size for tight convergence
        assert estimates[ConvergenceMetric.AVAILABILITY] >= 30

    def test_convergence_summary(self):
        """ConvergenceResult.summary() should produce readable output."""
        result = run_monte_carlo_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            max_time=days(30),
            confidence_level=0.95,
            relative_error=0.15,
            seed=42,
            min_runs=10,
            max_runs=200,
            batch_size=10,
        )

        summary = result.summary()
        assert "Convergence:" in summary
        assert "availability" in summary
        assert "runs" in summary.lower() or str(result.total_runs) in summary

    def test_absolute_error_convergence(self):
        """Absolute error mode should converge when CI half-width <= threshold."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        convergence = ConvergenceCriteria(
            confidence_level=0.95,
            absolute_error=0.05,  # ±0.05 on availability (±5 percentage points)
            metrics=[ConvergenceMetric.AVAILABILITY],
            min_runs=10,
            max_runs=2000,
            batch_size=20,
        )

        result = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=convergence,
        )

        assert isinstance(result, ConvergenceResult)
        assert result.total_runs >= convergence.min_runs
        # If converged, the CI half-width should be within the threshold
        if result.converged:
            for status in result.metric_statuses:
                assert status.ci_half_width <= convergence.absolute_error

    def test_absolute_error_tighter_needs_more_runs(self):
        """Tighter absolute error should require more runs."""
        config = MonteCarloConfig(
            num_simulations=10_000,
            max_time=days(30),
            base_seed=42,
        )
        runner = MonteCarloRunner(config)

        loose = ConvergenceCriteria(
            absolute_error=0.10,
            min_runs=10,
            max_runs=2000,
            batch_size=10,
        )
        tight = ConvergenceCriteria(
            absolute_error=0.02,
            min_runs=10,
            max_runs=5000,
            batch_size=20,
        )

        result_loose = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=loose,
        )
        result_tight = runner.run_until_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            convergence=tight,
        )

        assert result_tight.total_runs >= result_loose.total_runs

    def test_absolute_error_convenience_function(self):
        """run_monte_carlo_converged should accept absolute_error kwarg."""
        result = run_monte_carlo_converged(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            max_time=days(30),
            absolute_error=0.05,
            seed=42,
            min_runs=10,
            max_runs=500,
            batch_size=10,
        )

        assert isinstance(result, ConvergenceResult)
        assert result.total_runs >= 10

    def test_absolute_error_estimate_required_runs(self):
        """estimate_required_runs should work with absolute error criteria."""
        pilot = run_monte_carlo(
            cluster=_make_fast_cluster(),
            strategy=NoOpStrategy(),
            protocol=LeaderlessUpToDateQuorumProtocol(),
            num_simulations=30,
            max_time=days(30),
            seed=42,
        )

        criteria = ConvergenceCriteria(
            confidence_level=0.95,
            absolute_error=0.01,  # Very tight
        )

        estimates = estimate_required_runs(pilot, criteria)
        assert ConvergenceMetric.AVAILABILITY in estimates
        # Should need many runs for ±0.01 accuracy
        assert estimates[ConvergenceMetric.AVAILABILITY] >= 30
