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

        actions = strategy.on_event(event, cluster, rng)

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

        actions = strategy.on_event(event, cluster, rng)

        # Should start sync for lagging node
        sync_actions = [a for a in actions if a.action_type == ActionType.START_SYNC]
        assert len(sync_actions) == 1
        assert sync_actions[0].params["node_id"] == "node0"


# =============================================================================
# Protocol Tests
# =============================================================================


class TestLeaderlessUpToDateQuorumProtocol:
    """Tests for the default backward-compatible protocol."""

    def test_matches_cluster_can_commit(self):
        """Default protocol should match ClusterState.can_commit() exactly."""
        cluster = make_test_cluster(5)
        protocol = LeaderlessUpToDateQuorumProtocol()

        # All up-to-date: both should agree
        assert protocol.can_commit(cluster) == cluster.can_commit()
        assert protocol.can_commit(cluster) is True

        # Fail 2 nodes
        cluster.nodes["node0"].is_available = False
        cluster.nodes["node1"].is_available = False
        assert protocol.can_commit(cluster) == cluster.can_commit()
        assert protocol.can_commit(cluster) is True

        # Fail one more: lose quorum
        cluster.nodes["node2"].is_available = False
        assert protocol.can_commit(cluster) == cluster.can_commit()
        assert protocol.can_commit(cluster) is False

    def test_simulation_with_default_protocol_matches_no_protocol(self):
        """Simulator with explicit default protocol should produce same results."""
        results = []

        for protocol in [None, LeaderlessUpToDateQuorumProtocol()]:
            cluster = make_test_cluster(3)
            simulator = Simulator(
                initial_cluster=cluster,
                strategy=NoOpStrategy(),
                protocol=protocol,
                seed=42,
            )
            result = simulator.run_for(Seconds(1000))
            results.append(result)

        assert results[0].metrics.time_available == results[1].metrics.time_available
        assert results[0].metrics.time_unavailable == results[1].metrics.time_unavailable


class TestLeaderlessMajorityAvailableProtocol:
    """Tests for the majority-available protocol."""

    def test_available_but_not_up_to_date_can_commit(self):
        """Nodes that are available but lagging should still count for commit."""
        cluster = make_test_cluster(5)
        cluster.current_time = Seconds(100)

        protocol = LeaderlessMajorityAvailableProtocol()

        # All nodes are available but none are up-to-date (last_up_to_date_time=0)
        # ClusterState.can_commit() would return False (needs up-to-date quorum)
        assert not cluster.can_commit()

        # But majority-available protocol should return True (5 available >= quorum of 3)
        assert protocol.can_commit(cluster) is True

    def test_unavailable_nodes_dont_count(self):
        """Failed nodes should not count toward quorum."""
        cluster = make_test_cluster(5)
        cluster.current_time = Seconds(100)

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
                sync_rate_dist=Constant(1.5),  # Slow sync
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
                sync_rate_dist=Constant(1.5),
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
        for node in cluster.nodes.values():
            node.last_up_to_date_time = Seconds(100)

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
        for node in cluster.nodes.values():
            if node.is_available:
                node.last_up_to_date_time = election_event.time

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

        # Make all nodes lag so none are up-to-date
        for node in cluster.nodes.values():
            node.last_up_to_date_time = Seconds(0)

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
            failure_dist=Exponential(rate=1 / hours(4)),  # Frequent failures
            recovery_dist=Constant(minutes(5)),
            data_loss_dist=Exponential(rate=1 / days(365)),
            sync_rate_dist=Constant(2.0),
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

        # Run with Raft-like protocol (adds election downtime)
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
            protocol=RaftLikeProtocol(election_time_dist=Constant(30.0)),
            seed=42,
        )
        result_raft = sim2.run_for(days(30))

        # Raft should be less available due to election downtime
        assert (
            result_raft.metrics.availability_fraction()
            <= result_leaderless.metrics.availability_fraction()
        )
