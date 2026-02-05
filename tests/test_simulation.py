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
    RegionPair,
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
        log_replay_rate_dist=Constant(2.0),  # Replays 2x faster than commits
        snapshot_download_time_dist=Constant(minutes(5)),  # 5 minutes to download snapshot
        snapshot_interval_dist=Constant(hours(1)),  # Create snapshot every hour
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

    def test_time_to_sync_via_log(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_up_to_date_time=Seconds(80),
        )

        # 20 seconds behind, log_replay_rate=2.0, commit_rate=1.0 -> net rate 1.0 -> 20 seconds
        assert node.time_to_sync_via_log(Seconds(100), log_replay_rate=2.0, commit_rate=1.0) == 20

        # log_replay_rate equal to commit_rate can never catch up
        assert node.time_to_sync_via_log(Seconds(100), log_replay_rate=1.0, commit_rate=1.0) is None

        # log_replay_rate slower than commit_rate can never catch up
        assert node.time_to_sync_via_log(Seconds(100), log_replay_rate=0.5, commit_rate=1.0) is None

    def test_time_to_sync_via_snapshot(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_up_to_date_time=Seconds(0),  # Very far behind
        )

        # Node is 100 seconds behind. Snapshot is at time 90.
        # After 10s download, time will be 110, node will be at 90, need to catch 20s of log.
        # With log_replay_rate=2.0 and commit_rate=1.0, net rate=1.0, need 20s to catch up.
        # Total: 10 + 20 = 30 seconds
        result = node.time_to_sync_via_snapshot(
            current_time=Seconds(100),
            snapshot_time=Seconds(90),
            snapshot_download_time=10.0,
            log_replay_rate=2.0,
            commit_rate=1.0,
        )
        assert result == 30

        # If log replay rate can't keep up, returns None
        result = node.time_to_sync_via_snapshot(
            current_time=Seconds(100),
            snapshot_time=Seconds(90),
            snapshot_download_time=10.0,
            log_replay_rate=0.5,
            commit_rate=1.0,
        )
        assert result is None

    def test_snapshot_creation(self):
        config = make_test_node_config()
        node = NodeState(
            node_id="node1",
            config=config,
            last_up_to_date_time=Seconds(100),
        )

        assert not node.has_snapshot()
        node.create_snapshot(Seconds(100))
        assert node.has_snapshot()
        assert node.snapshot_time == Seconds(100)

        # Snapshot creation updates time
        node.last_up_to_date_time = Seconds(200)
        node.create_snapshot(Seconds(200))
        assert node.snapshot_time == Seconds(200)


# =============================================================================
# Network Tests
# =============================================================================


class TestRegionPair:
    def test_normalized_ordering(self):
        pair1 = RegionPair("us-west", "us-east")
        pair2 = RegionPair("us-east", "us-west")

        # Both should normalize to same ordering
        assert pair1 == pair2
        assert pair1.region_a == "us-east"
        assert pair1.region_b == "us-west"

    def test_contains(self):
        pair = RegionPair("us-east", "us-west")
        assert pair.contains("us-east")
        assert pair.contains("us-west")
        assert not pair.contains("eu-west")

    def test_other(self):
        pair = RegionPair("us-east", "us-west")
        assert pair.other("us-east") == "us-west"
        assert pair.other("us-west") == "us-east"
        assert pair.other("eu-west") is None


class TestNetworkState:
    def test_is_partitioned(self):
        state = NetworkState()
        pair = RegionPair("us-east", "us-west")

        assert not state.is_partitioned("us-east", "us-west")

        state.add_outage(pair)
        assert state.is_partitioned("us-east", "us-west")
        assert state.is_partitioned("us-west", "us-east")  # Order doesn't matter

        state.remove_outage(pair)
        assert not state.is_partitioned("us-east", "us-west")


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


class TestNetworkPartitions:
    """Tests for network partition effects on availability."""

    def _make_multi_region_cluster(self) -> ClusterState:
        """Create a 5-node cluster across 3 regions."""
        nodes = {}
        regions = ["us-east", "us-west", "eu-west"]
        for i in range(5):
            region = regions[i % 3]
            config = make_test_node_config(region=region)
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_up_to_date_time=Seconds(0),
            )
        # Nodes: node0=us-east, node1=us-west, node2=eu-west, node3=us-east, node4=us-west

        return ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=5,
            current_time=Seconds(0),
        )

    def test_no_partition_can_commit(self):
        """Without partitions, cluster can commit if quorum exists."""
        cluster = self._make_multi_region_cluster()
        assert cluster.can_commit()
        assert not cluster.is_network_partitioned()

    def test_partition_splits_quorum(self):
        """Network partition can prevent quorum even with enough nodes."""
        cluster = self._make_multi_region_cluster()
        # 5 nodes, quorum = 3
        # Nodes: node0=us-east, node1=us-west, node2=eu-west, node3=us-east, node4=us-west
        # us-east has 2 nodes, us-west has 2 nodes, eu-west has 1 node

        # Partition eu-west from both us-east and us-west
        cluster.network.add_outage(RegionPair("eu-west", "us-east"))
        cluster.network.add_outage(RegionPair("eu-west", "us-west"))

        # Now we have: us-east+us-west (4 nodes) and eu-west (1 node)
        # The larger component has 4 nodes >= quorum of 3, so can still commit
        assert cluster.can_commit()

    def test_partition_isolates_all_regions(self):
        """Complete partition prevents quorum."""
        cluster = self._make_multi_region_cluster()
        # Partition all regions from each other
        cluster.network.add_outage(RegionPair("us-east", "us-west"))
        cluster.network.add_outage(RegionPair("us-east", "eu-west"))
        cluster.network.add_outage(RegionPair("us-west", "eu-west"))

        # Now each region is isolated:
        # us-east: 2 nodes, us-west: 2 nodes, eu-west: 1 node
        # No component has 3 nodes (quorum), so cannot commit
        assert not cluster.can_commit()
        assert cluster.is_network_partitioned()

    def test_partition_with_failures(self):
        """Combined failures and partitions."""
        cluster = self._make_multi_region_cluster()
        # Partition us-west from others
        cluster.network.add_outage(RegionPair("us-west", "us-east"))
        cluster.network.add_outage(RegionPair("us-west", "eu-west"))

        # us-east + eu-west = 3 nodes (quorum), us-west = 2 nodes
        assert cluster.can_commit()

        # Now fail one node in the larger partition
        cluster.nodes["node0"].is_available = False  # us-east node

        # us-east + eu-west = 2 nodes (not quorum)
        assert not cluster.can_commit()
        assert cluster.has_potential_data_loss()

    def test_largest_reachable_quorum_size(self):
        """Test the largest connected component calculation."""
        cluster = self._make_multi_region_cluster()

        # No partitions - all 5 nodes reachable
        assert cluster.largest_reachable_quorum_size() == 5

        # Partition eu-west
        cluster.network.add_outage(RegionPair("eu-west", "us-east"))
        cluster.network.add_outage(RegionPair("eu-west", "us-west"))

        # Largest component is us-east + us-west = 4 nodes
        assert cluster.largest_reachable_quorum_size() == 4

        # Full partition
        cluster.network.add_outage(RegionPair("us-east", "us-west"))

        # Largest component is us-east or us-west = 2 nodes
        assert cluster.largest_reachable_quorum_size() == 2

    def test_transitive_reachability(self):
        """Test that reachability is transitive through connected regions."""
        cluster = self._make_multi_region_cluster()

        # Partition us-east from us-west, but both can reach eu-west
        cluster.network.add_outage(RegionPair("us-east", "us-west"))

        # us-east can reach eu-west, eu-west can reach us-west
        # So all regions are in one connected component
        all_regions = cluster.all_regions()
        component = cluster.network.regions_reachable_from("us-east", all_regions)
        assert component == {"us-east", "us-west", "eu-west"}

        # All 5 nodes can form quorum through eu-west
        assert cluster.can_commit()
        assert cluster.largest_reachable_quorum_size() == 5

    def test_partitioned_nodes_fall_behind(self):
        """Test that nodes in minority partitions fall behind during simulation."""
        # Create cluster with nodes in different regions
        nodes = {}
        regions = ["us-east", "us-east", "us-east", "eu-west", "eu-west"]
        for i, region in enumerate(regions):
            config = make_test_node_config(region=region)
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_up_to_date_time=Seconds(0),
            )
        # us-east has 3 nodes (node0, node1, node2), eu-west has 2 nodes (node3, node4)

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=5,
            current_time=Seconds(0),
        )

        # Partition eu-west from us-east
        cluster.network.add_outage(RegionPair("us-east", "eu-west"))

        # Simulate time passing - us-east (3 nodes) can commit, eu-west (2 nodes) cannot
        # Create a simulator with very low failure rates so we can test the partition behavior
        from powder.simulation import Simulator, NoOpStrategy, Exponential, Constant, days

        # Override configs to have very rare failures
        for node in cluster.nodes.values():
            node.config = NodeConfig(
                region=node.config.region,
                cost_per_hour=1.0,
                failure_dist=Exponential(rate=1 / days(3650)),  # 1 failure per 10 years
                recovery_dist=Constant(60),
                data_loss_dist=Exponential(rate=1 / days(3650)),
                log_replay_rate_dist=Constant(2.0),
                snapshot_download_time_dist=Constant(300),
                snapshot_interval_dist=Constant(hours(1)),
                spawn_dist=Constant(600),
            )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            seed=42,
        )

        # Run for 1000 seconds
        result = simulator.run_for(Seconds(1000))

        # Check that us-east nodes are up-to-date
        for node_id in ["node0", "node1", "node2"]:
            node = result.final_cluster.nodes[node_id]
            assert node.last_up_to_date_time >= Seconds(1000), \
                f"{node_id} in majority partition should be up-to-date"

        # Check that eu-west nodes fell behind (still at time 0)
        for node_id in ["node3", "node4"]:
            node = result.final_cluster.nodes[node_id]
            assert node.last_up_to_date_time == Seconds(0), \
                f"{node_id} in minority partition should have fallen behind"


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
                log_replay_rate_dist=Constant(2.0),
                snapshot_download_time_dist=Constant(300),
                snapshot_interval_dist=Constant(3600),
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
