
"""
Unit tests for pricing and billing logic in the Monte Carlo simulator.

Verifies that costs are calculated correctly under various failure and
replacement scenarios.
"""

import math
import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from powder.simulation import (
    # Distributions
    Seconds,
    hours,
    days,
    Constant,
    # Node
    NodeConfig,
    NodeState,
    # Network
    NetworkState,
    # Events
    EventType,
    Event,
    # Cluster
    ClusterState,
    # Strategy
    ClusterStrategy,
    Action,
    ActionType,
    NoOpStrategy,
    # Protocol
    LeaderlessUpToDateQuorumProtocol,
    # Simulator
    Simulator,
)


def make_pricing_config(
    failure_dist=None,
    recovery_dist=None,
    data_loss_dist=None,
    spawn_dist=None,
) -> NodeConfig:
    """Create a node config with deterministic distributions for pricing tests."""
    return NodeConfig(
        region="us-east",
        cost_per_hour=1.0,  # $1/hour for easy calculation
        failure_dist=failure_dist or Constant(days(3650)),  # Never fail
        recovery_dist=recovery_dist or Constant(0),
        data_loss_dist=data_loss_dist or Constant(days(3650)),  # Never lose data
        log_replay_rate_dist=Constant(100.0),  # Fast replay
        snapshot_download_time_dist=Constant(0),
        spawn_dist=spawn_dist or Constant(0),  # Instant spawn
    )



class GapReplacementStrategy(ClusterStrategy):
    """
    Strategy for Test 3:
    Replaces a failed node after a fixed 'downtime' gap to save costs.

    On Data Loss:
      1. Do NOT remove the node (billing stops automatically).
      2. Schedule a timer for 'downtime' duration.
    On Timer:
      1. Spawn a replacement node (starts billing).
    """
    def __init__(self, downtime: float, node_config: NodeConfig):
        self.downtime = downtime
        self.node_config = node_config
        self.spawn_counter = 0

    def on_event(
        self,
        event: Event,
        cluster: ClusterState,
        rng: np.random.Generator,
        protocol,
    ) -> list[Action]:
        actions = []

        if event.event_type == EventType.NODE_DATA_LOSS:
            # Billing stops automatically for nodes with data loss.
            # We explicitly do NOT remove it, per user request.
            
            # Schedule wake-up using a dummy replacement check
            # We use a unique ID derived from the failed node or just a counter
            # to avoid ID collisions if multiple fail.
            timer_id = f"timer_{event.target_id}"
            actions.append(
                Action(
                    ActionType.SCHEDULE_REPLACEMENT_CHECK,
                    {"node_id": timer_id, "timeout": self.downtime}
                )
            )

        elif event.event_type == EventType.NODE_REPLACEMENT_TIMEOUT:
            # Timer expired - spawn replacement
            timer_id = event.target_id
            if timer_id.startswith("timer_"):
                self.spawn_counter += 1
                new_node_id = f"node_repl_{self.spawn_counter}"
                
                actions.append(
                    Action(
                        ActionType.SPAWN_NODE,
                        {
                            "node_config": self.node_config,
                            "node_id": new_node_id
                        }
                    )
                )

        return actions


class TestPricing:
    
    def test_transient_failure_billing(self):
        """
        Scenario 1:
        Cluster with 3 machines. Two never fail.
        Third temp fails every week (6 days up, 1 day down).
        Run for 1 year (52 weeks).
        Expected: Billed for 3 machines * 1 year.
        Cost should not be reduced by transient failures.
        """
        # Config for stable nodes
        stable_config = make_pricing_config()
        
        # Config for transient failure node:
        # Fails after 6 days, recovers after 1 day. Cycle = 7 days.
        # Repeating pattern: Up 6d -> Fail -> Down 1d -> Recover -> Up 6d...
        transient_config = make_pricing_config(
            failure_dist=Constant(days(6)),
            recovery_dist=Constant(days(1)),
        )

        nodes = {
            "node1": NodeState(node_id="node1", config=stable_config),
            "node2": NodeState(node_id="node2", config=stable_config),
            "node3": NodeState(node_id="node3", config=transient_config),
        }
        
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        
        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(), # No replacement, just let it recover
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )

        # Run for exactly 52 weeks (364 days)
        duration_days = 364
        result = simulator.run_for(days(duration_days))

        # Expected cost: 3 nodes * 24 hours/day * 364 days * $1/hour
        expected_hours = 3 * 24 * duration_days
        expected_cost = expected_hours * 1.0

        # Allow small floating point margin
        assert result.metrics.total_cost == pytest.approx(expected_cost, rel=1e-9)

    def test_data_loss_billing(self):
        """
        Scenario 2:
        Cluster with 3 machines. Two never fail.
        Third encounters data loss after 1 week.
        Run for 1 year.
        Expected: Billed for 2 machines * 1 year + 1 machine * 1 week.
        Billing for node 3 should stop at 1 week (upon data loss).
        """
        stable_config = make_pricing_config()
        
        # Node 3 loses data at day 7.
        data_loss_config = make_pricing_config(
            data_loss_dist=Constant(days(7))
        )

        nodes = {
            "node1": NodeState(node_id="node1", config=stable_config),
            "node2": NodeState(node_id="node2", config=stable_config),
            "node3": NodeState(node_id="node3", config=data_loss_config),
        }

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(), # Do NOT replace
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )

        duration_days = 365
        result = simulator.run_for(days(duration_days))

        # Expected cost:
        # Node 1 & 2: 365 days
        # Node 3: 7 days
        total_hours = (2 * 365 * 24) + (7 * 24)
        expected_cost = total_hours * 1.0

        assert result.metrics.total_cost == pytest.approx(expected_cost, rel=1e-9)

    def test_replacement_strategy_billing_gap(self):
        """
        Scenario 3:
        Same failure pattern as Scenario 1 (fails every week), but...
        "replace the failed machine after 1 day of downtime".
        
        Interpretation:
        When it fails (after 6 days), we treat as Data Loss (billing stops).
        We wait 1 day (downtime gap).
        Then we spawn a new one (resume billing).
        
        Over 52 weeks (364 days):
        Each week has 6 days Up (billed), 1 day Down (gap, not billed).
        
        Expected:
        Node 1 & 2: Full 364 days.
        Node 3 (and its chain of replacements): 364 days - 52 days = 312 days.
        Total billed = 2 * 364 + 312 = 1040 days-worth.
        """
        stable_config = make_pricing_config()
        
        # Use data loss to stop billing automatically.
        # Repeating pattern effectively simulated by:
        # 1. Node runs 6 days -> Data Loss.
        # 2. Strategy waits 1 day.
        # 3. Spawns new node (same config).
        # 4. New node runs 6 days -> Data Loss...
        cycle_config = make_pricing_config(
            data_loss_dist=Constant(days(6)),
            # No transient failures needed for this test
        )

        nodes = {
            "node1": NodeState(node_id="node1", config=stable_config),
            "node2": NodeState(node_id="node2", config=stable_config),
            "node3": NodeState(node_id="node3", config=cycle_config),
        }

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )

        # Gap is 1 day
        strategy = GapReplacementStrategy(
            downtime=days(1),
            node_config=cycle_config
        )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=strategy,
            protocol=LeaderlessUpToDateQuorumProtocol(),
            seed=42,
        )

        # Run for 52 weeks = 364 days.
        # Week 1:
        #   Day 0-6: Node 3 active (6 days billed)
        #   Day 6: Data Loss. Billing stops. Timer set for 1 day.
        #   Day 6-7: Nothing (Gap).
        #   Day 7: Timer fires. Spawn new node. (Billing starts).
        #   Day 7-13: New node active (6 days billed)...
        # Pattern repeats 52 times.
        duration_days = 364
        result = simulator.run_for(days(duration_days))

        # Expected cost:
        # Node 1 & 2: 364 days * 24h
        # Dynamic Node: 52 cycles * 6 days * 24h = 312 days * 24h.
        
        billed_days_dynamic = 312
        total_days = (2 * 364) + billed_days_dynamic
        expected_cost = total_days * 24 * 1.0

        assert result.metrics.total_cost == pytest.approx(expected_cost, rel=1e-9)
