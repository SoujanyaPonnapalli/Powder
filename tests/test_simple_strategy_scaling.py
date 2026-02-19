
import pytest
import numpy as np
from powder.simulation import (
    Simulator,
    ClusterState,
    NetworkState,
    NodeConfig,
    NodeState,
    AdaptiveReplacementStrategy,
    EventType,
    Event,
    Constant,
    Seconds,
    Protocol,
    ActionType,
)
from powder.simulation.strategy import AdaptiveReplacementStrategy, Action


class DummyProtocol(Protocol):
    def can_commit(self, cluster: ClusterState) -> bool:
        # Simple majority rule (N // 2 + 1)
        return cluster.num_available() >= (cluster.target_cluster_size // 2 + 1)
    
    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
    ) -> list[Event]:
        return []

    def has_potential_data_loss(self, cluster: ClusterState) -> bool:
        return False

    def has_actual_data_loss(self, cluster: ClusterState) -> bool:
        return False

class TestAdaptiveReplacementStrategy:
    """Tests for AdaptiveReplacementStrategy smart scaling."""

    @staticmethod
    def make_test_node_config() -> NodeConfig:
        return NodeConfig(
            region="us-east-1",
            cost_per_hour=1.0,
            failure_dist=Constant(1000.0),
            recovery_dist=Constant(10.0),
            data_loss_dist=Constant(10000.0),
            log_replay_rate_dist=Constant(1000.0),
            snapshot_download_time_dist=Constant(10.0),
            spawn_dist=Constant(10.0),
        )

    def test_scale_down_flow(self):
        """Verify successful scale down with delay."""
        rng = np.random.default_rng(42)
        node_config = self.make_test_node_config()
        
        # 5 nodes
        nodes = {
            f"node_{i}": NodeState(
                node_id=f"node_{i}",
                config=node_config,
                is_available=True,
                has_data=True
            )
            for i in range(5)
        }
        
        cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=5)
        
        # Strategy with delay
        reconfig_delay = 5.0
        # Initialize strategy with new signature
        strategy = AdaptiveReplacementStrategy(
            failure_timeout=Seconds(10.0), # Added failure_timeout
            default_node_config=node_config,
            scale_down_threshold=2,
            reconfiguration_dist=Seconds(reconfig_delay),
            external_consensus=False
        )
        protocol = DummyProtocol()
        
        simulator = Simulator(cluster, strategy, protocol, log_events=True)
        simulator._initialize()
        
        # Fail 2 nodes at t=10
        # We need to ensure logic triggers. 
        # Strategy expects unavailable_count >= scale_down_threshold
        # If we fail 2 nodes, unavailable=2. Threshold=2. Should trigger.
        
        # Manually trigger failure events
        # Note: simulator._process_event calls _apply_node_failure first, then strategy.on_event
        # In _apply_node_failure, node.is_available becomes False.
        # So strategy sees the updated state.
        
        event1 = Event(Seconds(10.0), EventType.NODE_FAILURE, "node_0")
        event2 = Event(Seconds(10.0), EventType.NODE_FAILURE, "node_1")
        
        simulator.event_queue.push(event1)
        simulator.event_queue.push(event2)
        
        # Run until t=35 (10 failure + 10 timeout + 10 spawn + margin)
        simulator.run_until(Seconds(35.0))
        
        # Analyze events
        events = simulator.event_log
        
        # Check for reconfiguration event
        reconfig_events = [e for e in events if e.event_type == EventType.CLUSTER_RECONFIGURATION]
        assert len(reconfig_events) > 0, "Reconfiguration event expected"
        # The first failure event at t=10 might trigger it, or the second one.
        # Actually: 
        # Event 1: 1 failure. Unavail=1. Threshold=2. No reconfig.
        # Event 2: 2 failures. Unavail=2. Threshold=2. Trigger reconfig!
        
        # The reconfig event should be at t=10 + 5 = 15.0
        assert reconfig_events[0].time == 15.0
        
        # Check intermediate cluster size at t=25 (before scale up)
        # At t=25, reconfig 5->3 should be done. Replacements spawned (t=20) but scale up (t=35) not yet.
        # Wait, if replacements spawn at t=20, check logic might trigger scale up at t=20?
        # At t=20, Avail=3 (fail) + 2 (replace?) No, replacement complete at t=20??
        # Spawn dist 10s. Timeout 10s. Total 20s.
        # So at t=25, replacements NOT arrived yet (arrive t=30).
        # So Target should be 3.
        
        # We need to peek at state. simulator runs until 35.
        # Let's check event log for SCALE_UP and SCALE_DOWN actions.
        
        # Verify we scaled down
        # Current target might be 5 if it scaled back up.
        # But let's check events.
        scale_down_events = [e for e in simulator.event_log if e.event_type == EventType.CLUSTER_RECONFIGURATION and e.metadata.get("target_size") == 3]
        scale_up_events = [e for e in simulator.event_log if e.event_type == EventType.CLUSTER_RECONFIGURATION and e.metadata.get("target_size") == 5]
        
        assert len(scale_down_events) > 0, "Should have scaled down"
        assert len(scale_up_events) > 0, "Should have scaled back up"
        
        # Final state should be 5 (healed)
        assert cluster.target_cluster_size == 5
        
        # Verify replacements spawned
        # The strategy always spawns replacements on failure, regardless of scale down
        spawn_events = [e for e in events if e.event_type == EventType.NODE_SPAWN_COMPLETE]
        # 2 failures -> 2 replacements spawned immediately (t=10 + spawn_time 10 = 20)
        # They complete at t=20.
        # Verify replacements spawned? 
        # Strategy PRESERVES physical nodes (max_target=5).
        # So it does NOT remove the 2 failed nodes.
        # They recover at t=20 (10+10).
        # Timeout at t=20.
        # Since they recover, no replacements spawned.
        assert len(spawn_events) == 0


    def test_scale_down_failure_no_quorum(self):
        """Verify scale down works only if quorum exists."""
        # This test ensures we don't scale down if we can't commit.
        # But wait, my implementation of `can_commit` in DummyProtocol
        # uses `num_available >= majority`.
        # If we have 5 nodes, majority is 3.
        # If we fail 3 nodes, available=2. 2 < 3. No commit.
        
        rng = np.random.default_rng(42)
        node_config = self.make_test_node_config()
        
        nodes = {
            f"node_{i}": NodeState(
                node_id=f"node_{i}",
                config=node_config,
                is_available=True,
                has_data=True
            )
            for i in range(5)
        }
        
        cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=5)
        
        strategy = AdaptiveReplacementStrategy( # Changed to AdaptiveReplacementStrategy
            failure_timeout=Seconds(10.0), # Added failure_timeout
            default_node_config=node_config,
            scale_down_threshold=2,
            reconfiguration_dist=Seconds(5.0),
            external_consensus=False
        )
        protocol = DummyProtocol()
        
        simulator = Simulator(cluster, strategy, protocol, log_events=True)
        simulator._initialize()
        
        # Fail 3 nodes
        for i in range(3):
            simulator.event_queue.push(Event(Seconds(10.0), EventType.NODE_FAILURE, f"node_{i}"))
            
        simulator.run_until(Seconds(20.0))
        
        # Check reconfig triggered
        reconfig_events = [e for e in simulator.event_log if e.event_type == EventType.CLUSTER_RECONFIGURATION]
        assert len(reconfig_events) > 0
        
        # BUT target size should ideally remain 5 (failed to scale down)
        # Strategy checks `protocol.can_commit(cluster)` before emitting SCALE_DOWN action.
        # With 2 available, can_commit returns False.
        # So action list should be empty (or no scale down action).
        
        assert cluster.target_cluster_size == 5

    def test_external_consensus_3_to_2(self):
        """Verify 3->2 scaling with external consensus."""
        node_config = self.make_test_node_config()
        
        nodes = {
            f"node_{i}": NodeState(
                node_id=f"node_{i}",
                config=node_config,
                is_available=True,
                has_data=True
            )
            for i in range(3)
        }
        
        cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=3)
        
        # Set threshold to 1 for this test to force 3->2 on single failure
        strategy = AdaptiveReplacementStrategy( # Changed to AdaptiveReplacementStrategy
            failure_timeout=Seconds(10.0), # Added failure_timeout
            default_node_config=node_config,
            scale_down_threshold=1, 
            reconfiguration_dist=Seconds(1.0),
            external_consensus=True
        )
        protocol = DummyProtocol()
        
        simulator = Simulator(cluster, strategy, protocol, log_events=True)
        simulator._initialize()
        
        simulator.event_queue.push(Event(Seconds(10.0), EventType.NODE_FAILURE, "node_0"))
        
        simulator.run_until(Seconds(20.0))
        
        # With external consensus, should scale to 2
        assert cluster.target_cluster_size == 2

    def test_scaling_cycle_3_2_1_2_3(self):
        """Verify full cycle: 3 -> 2 -> 1 -> 2 -> 3."""
        node_config = self.make_test_node_config()
        
        # 3 nodes initially
        nodes = {
            f"node_{i}": NodeState(
                node_id=f"node_{i}",
                config=node_config,
                is_available=True,
                has_data=True
            )
            for i in range(3)
        }
        
        cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=3)
        
        strategy = AdaptiveReplacementStrategy(
            failure_timeout=Seconds(10.0),
            default_node_config=node_config,
            scale_down_threshold=1, 
            reconfiguration_dist=Seconds(3.0),
            external_consensus=True
        )
        protocol = DummyProtocol()
        
        simulator = Simulator(cluster, strategy, protocol, log_events=True)
        simulator._initialize()
        
        # --- Phase 1: Scale Down 3 -> 2 ---
        # Fail node_0 at t=10
        simulator.event_queue.push(Event(Seconds(10.0), EventType.NODE_FAILURE, "node_0"))
        
        # Run to t=14 (10 + 3 reconfig delay = 13 done)
        simulator.run_until(Seconds(14.0))
        assert cluster.target_cluster_size == 2
        
        # --- Phase 2: Scale Down 2 -> 1 ---
        # Fail node_1 at t=15
        simulator.event_queue.push(Event(Seconds(15.0), EventType.NODE_FAILURE, "node_1"))
        
        # Run to t=20 (15 + 3 = 18 reconfig)
        simulator.run_until(Seconds(20.0))
        assert cluster.target_cluster_size == 1
        
        # --- Phase 3: Scale Up 1 -> 2 ---
        # In this strategy, we keep failed nodes. 
        # Node 0 failed t=10. Recover t=20.
        # Node 1 failed t=15. Recover t=25.
        
        # At t=20: Node 0 recovers. Avail becomes 2 (Node 2 + Node 0).
        # Target 1. Max 3.
        # Deficit -1. 
        # Check Scale Up: Avail 2. Potential 2 (1+1 external).
        # Trigger Scale Up 1->2. Delay 3s. t=23.
        
        simulator.run_until(Seconds(24.0))
        assert cluster.target_cluster_size == 2
        
        # --- Phase 4: Scale Up 2 -> 3 ---
        # At t=25: Node 1 recovers. Avail becomes 3.
        # Target 2. 
        # Check Scale Up: Avail 3. Potential 3.
        # Trigger Scale Up 2->3. Delay 3s. t=28.
        
        simulator.run_until(Seconds(30.0))
        assert cluster.target_cluster_size == 3


