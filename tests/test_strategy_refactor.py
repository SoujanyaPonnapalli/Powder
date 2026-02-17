
import pytest
import numpy as np
from powder.simulation import (
    Simulator,
    ClusterState,
    NetworkState,
    NodeConfig,
    NodeState,
    NodeReplacementStrategy,
    EventType,
    Event,
    Constant,
    Seconds,
    Protocol,
    ActionType,
)

class DummyProtocol(Protocol):
    def can_commit(self, cluster: ClusterState) -> bool:
        return True
    
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

class TestNodeReplacementStrategy:
    """Tests for NodeReplacementStrategy including data loss timeouts and cleanup."""

    @staticmethod
    def make_test_node_config() -> NodeConfig:
        return NodeConfig(
            region="us-east-1",
            cost_per_hour=1.0,
            failure_dist=Constant(1000.0), # Long failure time
            recovery_dist=Constant(10.0),
            data_loss_dist=Constant(10000.0), # Data loss far in future
            log_replay_rate_dist=Constant(1000.0),
            snapshot_download_time_dist=Constant(10.0),
            spawn_dist=Constant(10.0),
        )

    def test_data_loss_timeout(self):
        """Verify that data loss triggers a replacement only after failure_timeout."""
        # 1. Setup
        rng = np.random.default_rng(42)
        node_config = self.make_test_node_config()
        
        # create 3 nodes
        nodes = {
            f"node_{i}": NodeState(
                node_id=f"node_{i}",
                config=node_config,
                last_applied_index=0.0
            )
            for i in range(3)
        }
        
        network = NetworkState()
        cluster = ClusterState(nodes=nodes, network=network, target_cluster_size=3)
        
        # Strategy with 50s timeout
        strategy = NodeReplacementStrategy(failure_timeout=Seconds(50.0), safe_mode=False)
        protocol = DummyProtocol()
        
        simulator = Simulator(cluster, strategy, protocol, log_events=True)
        simulator._initialize()
        
        # Manually trigger data loss on node_0 at t=100
        data_loss_time = 100.0
        simulator.event_queue.push(
            Event(
                 time=Seconds(data_loss_time),
                 event_type=EventType.NODE_DATA_LOSS,
                 target_id="node_0"
            )
        )
        
        # Run until t=200 (should be enough for timeout=50 + spawn=10 => t=160)
        simulator.run_until(Seconds(200.0))
        
        # 3. Analyze events
        events = simulator.event_log
        
        # Find data loss event
        data_loss_event = next((e for e in events if e.event_type == EventType.NODE_DATA_LOSS), None)
        assert data_loss_event is not None, "Data loss event not found"
        assert data_loss_event.time == data_loss_time
        
        # Find replacement spawn event
        spawn_event = next((e for e in events if e.event_type == EventType.NODE_SPAWN_COMPLETE), None)
        assert spawn_event is not None, "Replacement not spawned"
        
        # Check timing: timeout (50s) + spawn (10s) = 60s delay
        # spawn_event.time should be approx 160.0
        expected_spawn_time = data_loss_time + 50.0 + 10.0
        assert spawn_event.time == pytest.approx(expected_spawn_time, abs=1.0)
        
        # Ensure we didn't replace immediately
        assert spawn_event.time >= data_loss_time + 50.0

    def test_zombie_cleanup(self):
        """Verify that extra nodes are removed."""
        # 1. Setup cluster with 4 nodes but target 3
        rng = np.random.default_rng(42)
        node_config = self.make_test_node_config()
        
        # 4 healthy nodes, target 3
        nodes = {
            f"node_{i}": NodeState(
                node_id=f"node_{i}",
                config=node_config,
                last_applied_index=100.0,
                is_available=True
            )
            for i in range(4)
        }
        
        network = NetworkState()
        cluster = ClusterState(nodes=nodes, network=network, target_cluster_size=3)
        strategy = NodeReplacementStrategy(failure_timeout=Seconds(50.0), safe_mode=False)
        protocol = DummyProtocol()
        
        # Manually trigger a check by sending a "fake" spawn complete or similar event
        # The strategy checks for cleanup on NODE_SPAWN_COMPLETE, NODE_RECOVERY, etc.
        # We can simulate a recovery of the 4th node to trigger cleanup.
        event = Event(
            time=Seconds(10.0),
            event_type=EventType.NODE_RECOVERY, # This triggers _maintain_cluster_size
            target_id="node_3"
        )
        
        actions = strategy.on_event(event, cluster, rng, protocol)
        
        # Expect a REMOVE_NODE action
        remove_actions = [a for a in actions if a.action_type == ActionType.REMOVE_NODE]
        
        assert len(remove_actions) > 0, "Should trigger REMOVE_NODE"
        assert len(remove_actions) == 1
