"""
Tests for the Monte Carlo RSM simulator.

Tests cover distributions, node/cluster state, events, strategies,
and the simulator engine.
"""

import numpy as np
import pytest

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
        sync_rate_dist=Constant(2.0),  # Syncs 2x faster than commits
        spawn_dist=Constant(minutes(10)),
    )


class TestNodeState:
    def test_is_up_to_date(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_up_to_date_time=Seconds(100),
        )

        assert node.is_up_to_date(Seconds(100))
        assert node.is_up_to_date(Seconds(50))
        assert not node.is_up_to_date(Seconds(150))

    def test_lag_seconds(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_up_to_date_time=Seconds(100),
        )

        assert node.lag_seconds(Seconds(100)) == 0
        assert node.lag_seconds(Seconds(150)) == 50
        assert node.lag_seconds(Seconds(50)) == 0  # Can't have negative lag

    def test_time_to_sync(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_up_to_date_time=Seconds(80),
        )

        # 20 seconds behind, sync rate 2.0 -> net rate 1.0 -> 20 seconds to sync
        assert node.time_to_sync(Seconds(100), sync_rate=2.0) == 20

        # Sync rate 1.0 can never catch up
        assert node.time_to_sync(Seconds(100), sync_rate=1.0) is None

        # Sync rate 0.5 can never catch up
        assert node.time_to_sync(Seconds(100), sync_rate=0.5) is None


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
            last_up_to_date_time=Seconds(0),
        )

    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=num_nodes,
        current_time=Seconds(0),
    )


class TestClusterState:
    def test_quorum_calculations(self):
        cluster = make_test_cluster(5)

        assert cluster.quorum_size() == 3
        assert cluster.num_available() == 5
        assert cluster.can_commit()

    def test_can_commit_with_failures(self):
        cluster = make_test_cluster(5)

        # Fail 2 nodes - should still have quorum
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False

        assert cluster.num_available() == 3
        assert cluster.can_commit()

        # Fail one more - no quorum
        cluster.nodes["node2"].is_available = False
        assert cluster.num_available() == 2
        assert not cluster.can_commit()

    def test_potential_data_loss(self):
        cluster = make_test_cluster(3)

        assert not cluster.has_potential_data_loss()

        # Lose quorum
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False

        assert cluster.has_potential_data_loss()

    def test_actual_data_loss(self):
        cluster = make_test_cluster(3)
        cluster.current_time = Seconds(100)

        # Mark all nodes as up-to-date initially
        for node in cluster.nodes.values():
            node.last_up_to_date_time = Seconds(100)

        assert not cluster.has_actual_data_loss()

        # Make one node lag and fail the up-to-date nodes
        cluster.nodes["node0"].last_up_to_date_time = Seconds(50)  # Lagging
        cluster.nodes["node1"].has_data = False  # Data lost
        cluster.nodes["node2"].has_data = False  # Data lost

        # node0 has data but is not up-to-date -> actual data loss
        assert cluster.has_actual_data_loss()


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
                sync_rate_dist=Constant(2.0),
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
        queue = EventQueue()

        actions = strategy.on_event(event, cluster, queue, rng)

        # Should spawn a replacement
        spawn_actions = [a for a in actions if a.action_type == ActionType.SPAWN_NODE]
        assert len(spawn_actions) == 1

    def test_starts_sync_on_recovery(self):
        config = make_test_node_config()
        strategy = SimpleReplacementStrategy(default_node_config=config)

        cluster = make_test_cluster(3)
        cluster.current_time = Seconds(100)

        # Node recovered but is lagging
        cluster.nodes["node0"].is_available = True
        cluster.nodes["node0"].last_up_to_date_time = Seconds(50)

        event = Event(
            time=Seconds(100),
            event_type=EventType.NODE_RECOVERY,
            target_id="node0",
        )

        rng = np.random.default_rng(42)
        queue = EventQueue()

        actions = strategy.on_event(event, cluster, queue, rng)

        # Should start sync for lagging node
        sync_actions = [a for a in actions if a.action_type == ActionType.START_SYNC]
        assert len(sync_actions) == 1
        assert sync_actions[0].params["node_id"] == "node0"
