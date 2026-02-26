"""
Tests for the Raft-like protocol.

Covers leader election (candidate failure, stall/restart, quorum loss)
and recovery semantics (snapshot vs log-only sync, sync during elections,
leader failure during sync).
"""

import numpy as np
import pytest

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
    ClusterState,
    NoOpStrategy,
    EventType,
    Event,
    Simulator,
    RaftLikeProtocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _never_fail_config(region: str = "us-east") -> NodeConfig:
    """Node that never fails within any reasonable test window."""
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


def _single_failure_config(
    fail_after: float,
    recovery_time: float,
    region: str = "us-east",
) -> NodeConfig:
    """Node that fails once at `fail_after`, recovers in `recovery_time`,
    then never fails again (next failure at 9999 days)."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Constant(fail_after),
        recovery_dist=Constant(recovery_time),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(2.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


# ===========================================================================
# Raft Leader Election Failure Tests
# ===========================================================================


class TestRaftLeaderElection:
    """Tests for leader election when candidate nodes fail during election."""

    def test_basic_leader_election_on_failure(self):
        """Leader fails → election completes → new leader elected.

        Setup:
        - 3-node cluster: node0 (leader), node1, node2
        - node0 fails at 1 hour, recovers after 1 hour
        - Election takes 10 seconds
        - node1, node2 never fail → one becomes leader

        Validates:
        - Election occurs after leader failure
        - New leader is elected from remaining available nodes
        - System becomes available again after election
        """
        election_time = 10.0

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=hours(1),
                    recovery_time=hours(1),
                ),
            ),
            "node1": NodeState(
                node_id="node1",
                config=_never_fail_config(),
            ),
            "node2": NodeState(
                node_id="node2",
                config=_never_fail_config(),
            ),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(hours(2))

        assert result.end_reason == "time_limit"

        # Verify election events occurred
        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]
        assert len(election_events) == 1, "Exactly one election should complete"

        # Election started at leader failure (t=3600s), completed at t=3610s
        assert election_events[0].time == pytest.approx(hours(1) + election_time)

        # New leader should be node1 (first alphabetical eligible node)
        assert protocol.leader_id == "node1"
        assert not protocol.election_in_progress

        # Unavailability = election time only
        assert result.metrics.time_unavailable == pytest.approx(election_time)

    def test_candidate_fails_during_election(self):
        """All remaining candidates fail during election → stalls → restarts on recovery.

        Setup:
        - 3-node cluster: node0 (leader), node1, node2
        - node0 (leader) fails at t=100s
        - Election takes 50s (completes at t=150s)
        - node1 fails at t=120s (during election), recovers at t=190s
        - node2 fails at t=130s (during election), recovers at t=180s

        Timeline:
        - t=0:    All up, node0 is leader
        - t=100:  node0 fails → election starts (50s election time)
        - t=120:  node1 fails (during election)
        - t=130:  node2 fails (during election)
        - t=150:  Election completes → no eligible nodes → stalls
        - t=180:  node2 recovers → election restarts (50s)
        - t=190:  node1 recovers
        - t=230:  Election completes → both eligible → node1 elected

        Validates:
        - First election attempt finds no eligible candidate → stalls
        - Recovery event triggers restart (no polling)
        - After restart completes, leader elected
        """
        election_time = 50.0

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=100.0,      # fails at t=100
                    recovery_time=500.0,   # recovers at t=600 (after test)
                ),
            ),
            "node1": NodeState(
                node_id="node1",
                config=_single_failure_config(
                    fail_after=120.0,      # fails at t=120 (during election)
                    recovery_time=70.0,    # recovers at t=190
                ),
            ),
            "node2": NodeState(
                node_id="node2",
                config=_single_failure_config(
                    fail_after=130.0,      # fails at t=130 (during election)
                    recovery_time=50.0,    # recovers at t=180
                ),
            ),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(Seconds(300.0))

        assert result.end_reason == "time_limit"

        # Exactly 2 election events: first attempt (stalled) + restart (succeeded)
        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]
        assert len(election_events) == 2, (
            f"Expected 2 election events (attempt + restart), got {len(election_events)}"
        )

        # First election at t=150 (100+50), stalls.
        # node2 recovers at t=180 → restart, completes at t=230 (180+50)
        assert election_events[0].time == pytest.approx(150.0)
        assert election_events[1].time == pytest.approx(230.0)

        # After restart, both node1 and node2 are available and up-to-date.
        # _pick_leader selects alphabetically first → node1
        assert protocol.leader_id == "node1"
        assert not protocol.election_in_progress

        # Total unavailability = 130s (election starts at t=100, completes at t=230)
        assert result.metrics.time_unavailable == pytest.approx(130.0)

    def test_all_nodes_fail_election_stalls_then_restarts(self):
        """All nodes fail → election stalls → restarts when quorum recovers.

        With the majority requirement, a single recovered node can't win
        the election in a 3-node cluster (needs 2 for quorum). The election
        restarts only when enough nodes recover to form a majority.

        Setup:
        - 3-node cluster: node0 (leader), node1, node2
        - node1 fails at t=50 (quorum still met: node0 + node2)
        - node2 fails at t=80 (quorum lost: only node0)
        - node0 fails at t=100 → election starts
        - Election fires at t=150 → stalls (no one available)
        - node2 recovers at t=280 → restart, but only 1/3 = no quorum → stalls
        - node1 recovers at t=300 → restart: 2/3 = quorum!
        - Election completes at t=350 (300 + 50) → node1 elected

        Validates:
        - No polling retries while all nodes are down
        - Recovery triggers election restart
        - Majority requirement enforced (single node not enough)
        """
        election_time = 50.0

        # Custom configs prevent re-failure cycling (Constant(days(9999)):
        node1_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(250.0),      # recovers 250s after failure (t=300)
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        node2_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(200.0),      # recovers 200s after failure (t=280)
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=100.0,
                    recovery_time=900.0,   # recovers far in future
                ),
            ),
            "node1": NodeState(
                node_id="node1",
                config=node1_config,
            ),
            "node2": NodeState(
                node_id="node2",
                config=node2_config,
            ),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        # Manually inject failures for node1 and node2
        simulator._initialize()
        simulator.event_queue.cancel_events_for("node1", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(50.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node1",
        ))
        simulator.event_queue.cancel_events_for("node2", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(80.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node2",
        ))

        result = simulator.run_until(Seconds(500.0))

        assert result.end_reason == "time_limit"

        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]

        # Should have: t=150 (stall), t=330 (node2 recovery restart, stall due to
        # only 1 available), t=350 (node1 recovery restart, quorum met, succeeds)
        assert len(election_events) >= 2, (
            f"Expected at least 2 election events, got {len(election_events)}: "
            f"{[(e.time, e.target_id) for e in election_events]}"
        )

        # A leader should be elected after both nodes recover
        assert protocol.leader_id is not None, (
            f"A leader should be elected. "
            f"Election events: {[(e.time, e.target_id) for e in election_events]}"
        )
        assert not protocol.election_in_progress
        # node1 is alphabetically first among eligible (node1, node2)
        assert protocol.leader_id == "node1"

    def test_unavailable_during_election_period(self):
        """Verify can_commit is False for the entire election period.

        Setup:
        - 3-node cluster, node0 is leader
        - node0 fails at t=1000, election takes 100s
        - node1, node2 never fail

        Validates by checking event log timing:
        - Before t=1000: system available (leader present)
        - t=1000 to t=1100: system unavailable (election in progress)
        - After t=1100: system available (new leader elected)
        """
        election_time = 100.0
        fail_time = 1000.0

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=fail_time,
                    recovery_time=500.0,
                ),
            ),
            "node1": NodeState(
                node_id="node1",
                config=_never_fail_config(),
            ),
            "node2": NodeState(
                node_id="node2",
                config=_never_fail_config(),
            ),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        duration = 2000.0
        result = simulator.run_for(Seconds(duration))

        assert result.end_reason == "time_limit"

        # Check that unavailable time is exactly the election duration
        assert result.metrics.time_unavailable == pytest.approx(election_time)

        # Verify availability fraction
        expected = (duration - election_time) / duration
        assert result.metrics.availability_fraction() == pytest.approx(expected)

        # Verify the election event is at the right time
        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]
        assert len(election_events) == 1
        assert election_events[0].time == pytest.approx(fail_time + election_time)

    def test_quorum_lost_during_election(self):
        """Quorum lost mid-election → election immediately invalidated → restart.

        When quorum drops below majority during an in-progress election,
        the election is immediately cancelled (stalled). The pending
        LEADER_ELECTION_COMPLETE event becomes stale (epoch mismatch)
        and is ignored when it fires. A fresh election starts only when
        quorum is restored.

        Setup:
        - 5-node cluster: node0 (leader), node1..node4
        - node0 (leader) fails at t=100 → election starts (epoch=1, 100s)
        - At t=100: node1..node4 available (4/5 = quorum)
        - node1 fails at t=120 → still 3/5 = quorum, election continues
        - node2 fails at t=140 → only 2/5 available < 3 = quorum
          → election IMMEDIATELY invalidated (stalled, epoch=1 becomes stale)
        - t=200: stale LEADER_ELECTION_COMPLETE (epoch=1) fires → ignored
        - node1 recovers at t=300 → 3/5 available = quorum
          → fresh election starts (epoch=2, 100s)
        - t=400: LEADER_ELECTION_COMPLETE (epoch=2) → leader elected

        Validates:
        - Election cancelled immediately on quorum loss (not at completion)
        - Stale election events ignored via epoch
        - Full election timer restarts from scratch after quorum recovery
        """
        election_time = 100.0

        # node1: fails at t=120 during election, recovers at t=300
        node1_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(180.0),      # recovers 180s after failure (t=300)
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        # node2: fails at t=140 during election, never recovers in test
        node2_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(days(9999)),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=100.0,
                    recovery_time=days(9999),
                ),
            ),
            "node1": NodeState(node_id="node1", config=node1_config),
            "node2": NodeState(node_id="node2", config=node2_config),
            "node3": NodeState(node_id="node3", config=_never_fail_config()),
            "node4": NodeState(node_id="node4", config=_never_fail_config()),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=5,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        # Manually inject failures during election period
        simulator._initialize()
        simulator.event_queue.cancel_events_for("node1", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(120.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node1",
        ))
        simulator.event_queue.cancel_events_for("node2", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(140.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node2",
        ))

        result = simulator.run_until(Seconds(500.0))

        assert result.end_reason == "time_limit"

        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]

        # Two LEADER_ELECTION_COMPLETE events fire:
        # 1. t=200: stale event (epoch=1, cancelled at t=140) → ignored
        # 2. t=400: fresh event (epoch=2, started at t=300) → succeeds
        assert len(election_events) == 2, (
            f"Expected 2 election events (stale + fresh), got {len(election_events)}: "
            f"{[(e.time, e.metadata) for e in election_events]}"
        )

        # Stale event at t=200 (from original election at t=100+100)
        assert election_events[0].time == pytest.approx(200.0)
        # Fresh event at t=400 (from restart at t=300+100)
        assert election_events[1].time == pytest.approx(400.0)

        # Verify epochs: stale event has epoch 1, fresh has epoch 2
        assert election_events[0].metadata["epoch"] < election_events[1].metadata["epoch"]

        # Leader elected on the fresh election
        assert protocol.leader_id is not None
        assert not protocol.election_in_progress
        # node1 is alphabetically first among eligible (node1, node3, node4)
        assert protocol.leader_id == "node1"

        # Unavailability: from t=100 (leader fails) to t=400 (election succeeds)
        assert result.metrics.time_unavailable == pytest.approx(300.0)

    def test_no_election_without_majority(self):
        """5-node cluster: leader needs majority (3/5) to be elected.

        Setup:
        - 5 nodes: node0 (leader), node1..node4
        - node0 (leader) fails at t=100 → election starts
        - node1 fails at t=50, never recovers in test
        - node2 fails at t=60, never recovers in test
        - node3, node4 never fail

        At election time (t=200): only node3, node4 available (2/5 < 3 = quorum)
        → election stalls because majority not available.

        Then node2 recovers at t=260 → 3/5 available = quorum!
        → election restarts, completes at t=310 (260+50) → node2 elected

        Validates:
        - Leader election requires majority of cluster available
        - Single node or minority can't win election
        """
        election_time = 50.0

        # Nodes that never recover within test
        never_recover_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(days(9999)),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        # Node that recovers
        recoverable_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(200.0),   # recovers 200s after failure
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=100.0,
                    recovery_time=days(9999),
                ),
            ),
            "node1": NodeState(node_id="node1", config=never_recover_config),
            "node2": NodeState(node_id="node2", config=recoverable_config),
            "node3": NodeState(node_id="node3", config=_never_fail_config()),
            "node4": NodeState(node_id="node4", config=_never_fail_config()),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=5,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        # Manually inject failures for node1 and node2
        simulator._initialize()
        simulator.event_queue.cancel_events_for("node1", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(50.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node1",
        ))
        simulator.event_queue.cancel_events_for("node2", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(60.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node2",
        ))

        result = simulator.run_until(Seconds(400.0))

        assert result.end_reason == "time_limit"

        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]

        # First election at t=150 (stalls: only 2/5 available)
        # node2 recovers at t=260 → quorum met (3/5)
        # Election restarts, completes at t=310
        assert len(election_events) >= 2

        # Leader should be elected
        assert protocol.leader_id is not None, (
            f"Leader should be elected after quorum recovered. "
            f"Events: {[(e.time, e.target_id) for e in election_events]}"
        )
        assert not protocol.election_in_progress

    def test_election_stall_generates_no_events(self):
        """Verify that a stalled election produces no events until recovery.

        This is the key efficiency improvement: when all nodes are down,
        no election retry events are generated (unlike the old polling approach).

        Setup:
        - 3-node cluster, all fail quickly, no one recovers within test
        - Election fires once, stalls, and no more election events appear

        Validates:
        - Exactly 1 election event (the initial stalled attempt)
        - No polling retries
        """
        election_time = 10.0

        never_recover_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(days(9999)),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=100.0,
                    recovery_time=days(9999),
                ),
            ),
            "node1": NodeState(node_id="node1", config=never_recover_config),
            "node2": NodeState(node_id="node2", config=never_recover_config),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
        )
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        # All non-leader nodes fail before leader
        simulator._initialize()
        simulator.event_queue.cancel_events_for("node1", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(50.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node1",
        ))
        simulator.event_queue.cancel_events_for("node2", EventType.NODE_FAILURE)
        simulator.event_queue.push(Event(
            time=Seconds(60.0),
            event_type=EventType.NODE_FAILURE,
            target_id="node2",
        ))

        result = simulator.run_until(Seconds(10000.0))

        assert result.end_reason == "time_limit"

        election_events = [
            e for e in result.event_log
            if e.event_type == EventType.LEADER_ELECTION_COMPLETE
        ]

        # Only 1 election event (stalls, no retries since no recovery)
        assert len(election_events) == 1, (
            f"Expected exactly 1 election event (stalled), got {len(election_events)}. "
            f"The old polling approach would have generated ~990 events."
        )

        assert protocol.leader_id is None
        assert protocol.election_in_progress
        assert protocol._election_stalled


# ===========================================================================
# Raft Snapshot / Log Recovery Tests
# ===========================================================================


class TestRaftSnapshotRecovery:
    """Tests for snapshot-aware node recovery under RaftLikeProtocol.

    Mirrors TestSnapshotRecovery in test_simulation.py but uses
    RaftLikeProtocol to confirm the shared compute_sync_time logic
    works correctly in a leader-based context.
    """

    def test_recovery_without_snapshot(self):
        """Node slightly behind should do log-only replay (no snapshot needed)."""
        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(365)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(365)),
            log_replay_rate_dist=Constant(10.0),
            snapshot_download_time_dist=Constant(minutes(5)),  # Slow snapshot
            spawn_dist=Constant(minutes(10)),
        )

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
            commit_rate=1.0,
            snapshot_interval=100.0,
        )

        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_applied_index=150.0,
                last_snapshot_index=100.0,
            )
        nodes["node0"].last_applied_index = 110.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=150.0,
        )

        # Initialize protocol so there's a leader
        rng = np.random.default_rng(42)
        protocol.on_simulation_start(cluster, rng)
        assert protocol.leader_id is not None

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

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
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
        nodes["node0"].last_applied_index = 50.0
        nodes["node0"].last_snapshot_index = 0.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=250.0,
        )

        rng = np.random.default_rng(42)
        protocol.on_simulation_start(cluster, rng)

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

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
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
        nodes["node0"].last_applied_index = 0.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=1000.0,
        )

        rng = np.random.default_rng(42)
        protocol.on_simulation_start(cluster, rng)

        sync_time = protocol.compute_sync_time(nodes["node0"], cluster, rng)

        assert sync_time is not None
        # log-only: lag=1000, net_rate=10-1=9 -> ~111s
        assert sync_time < 120  # Well under 60 + 111

    def test_quorum_loss_freezes_commit_index(self):
        """When quorum is lost, commit_index should freeze, so recovery is instant."""
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

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
            commit_rate=1.0,
        )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
            log_events=True,
        )

        result = simulator.run_for(Seconds(200))

        final = result.final_cluster
        assert final.commit_index < 200.0  # Less than wall-clock (some unavailability)
        assert final.commit_index > 0.0  # But some commits happened

    def test_node_snapshot_state_advances(self):
        """Available nodes should take snapshots when commit_index crosses boundaries."""
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

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
            commit_rate=1.0,
            snapshot_interval=50.0,
        )

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=42,
        )

        result = simulator.run_for(Seconds(200))

        for node in result.final_cluster.nodes.values():
            if node.is_available and node.has_data:
                assert node.last_snapshot_index > 0
                assert node.last_snapshot_index % 50.0 == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# Raft-Specific Sync Model Tests
# ===========================================================================


class TestRaftSyncModel:
    """Tests for Raft-specific sync behavior not covered by leaderless tests.

    These test scenarios where the leader/election state affects sync timing.
    """

    def test_sync_during_election(self):
        """Node syncing during election should use full replay rate (no commit drag).

        When an election is in progress, can_commit() returns False, so the
        effective commit_rate is 0.  The net sync rate is replay_rate instead
        of replay_rate - commit_rate, making sync faster.

        Setup
        -----
        commit_rate = 1.0, replay_rate = 2.0.
        With leader:    net_rate = 2 - 1 = 1.0  →  gap 10 → 10 s.
        Without leader: net_rate = 2 - 0 = 2.0  →  gap 10 →  5 s.
        """
        config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(days(9999)),
            recovery_dist=Constant(0),
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(2.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
            commit_rate=1.0,
        )

        nodes = {}
        for i in range(3):
            nodes[f"node{i}"] = NodeState(
                node_id=f"node{i}",
                config=config,
                last_applied_index=10.0,
            )
        nodes["node0"].last_applied_index = 0.0

        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=10.0,
        )

        rng = np.random.default_rng(42)

        # --- With leader: net_rate = 1.0 ---
        protocol.on_simulation_start(cluster, rng)
        assert protocol.leader_id is not None
        assert protocol.can_commit(cluster)

        sync_with_leader = protocol.compute_sync_time(nodes["node0"], cluster, rng)
        assert sync_with_leader is not None
        assert sync_with_leader == pytest.approx(10.0, abs=0.01)

        # --- During election: net_rate = 2.0 (no commit drag) ---
        protocol._election_in_progress = True
        protocol._leader_id = None
        assert not protocol.can_commit(cluster)

        sync_during_election = protocol.compute_sync_time(nodes["node0"], cluster, rng)
        assert sync_during_election is not None
        assert sync_during_election == pytest.approx(5.0, abs=0.01)

        # Election sync should be faster
        assert sync_during_election < sync_with_leader

    def test_leader_failure_during_sync(self):
        """Leader fails mid-sync; sync continues and completes after election resolves.

        Setup
        -----
        3 nodes.  node0 lags 20 units.  commit_rate=1.0, replay_rate=3.0.
        net_rate = 2.0 → base sync time = 10 s.
        Leader (node1) fails at t=3, election takes 5 s (completes at t=8).

        Timeline
        --------
        [0, 3)  : net_rate = 2.0, close 6 units → gap from 20 to 14.
        [3, 8)  : election, can_commit=False, net_rate = 3.0.
                  Close 15 units (14 gap) → sync completes around t ≈ 7.7.
        """
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
        # Leader node that fails at t=3
        leader_config = NodeConfig(
            region="us-east",
            cost_per_hour=1.0,
            failure_dist=Constant(3.0),
            recovery_dist=Constant(20.0),  # Recovers long after test ends
            data_loss_dist=Constant(days(9999)),
            log_replay_rate_dist=Constant(3.0),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Constant(0),
        )

        nodes = {
            "node0": NodeState(
                node_id="node0", config=stable_config, last_applied_index=0.0
            ),
            "node1": NodeState(
                node_id="node1", config=leader_config, last_applied_index=20.0
            ),
            "node2": NodeState(
                node_id="node2", config=stable_config, last_applied_index=20.0
            ),
        }
        cluster = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=3,
            commit_index=20.0,
        )
        sim = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=RaftLikeProtocol(
                election_time_dist=Constant(5.0),
                commit_rate=1.0,
            ),
            seed=0,
            log_events=True,
        )

        result = sim.run_for(Seconds(15))

        # node0 should have completed sync despite leader failure
        sync_events = [
            e
            for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        assert len(sync_events) >= 1, "node0 should have completed sync"

        # Sync should complete during or soon after the election
        # (faster than the 10s it would take with a stable leader,
        #  since election period has higher net_rate)
        assert sync_events[0].time < 10.0

        node0 = result.final_cluster.get_node("node0")
        assert node0.sync is None

    def test_gc_forced_snapshot_with_leader(self):
        """Outside GC window with active leader → snapshot forced, same as leaderless.

        Mirrors TestDeterministicSyncModel.test_gc_forced_snapshot_outside_window
        but uses RaftLikeProtocol to confirm identical behavior.

        Setup
        -----
        commit_rate = 1.0, replay_rate = 10.0, net_rate = 9.0.
        snapshot_interval = 100, log_retention_ops = 100.
        Donor at 250 keeps log from 150–250.
        Node0 at 50: outside window → must download snapshot.

        Expected: snapshot (60s) + remaining_log/net_rate ≈ 72.2 s.
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

        protocol = RaftLikeProtocol(
            election_time_dist=Constant(5.0),
            commit_rate=1.0,
            snapshot_interval=100.0,
            log_retention_ops=100.0,
        )

        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=cfg,
                last_applied_index=50.0,
                last_snapshot_index=0.0,
            ),
            "node1": NodeState(
                node_id="node1",
                config=cfg,
                last_applied_index=250.0,
                last_snapshot_index=200.0,
            ),
            "node2": NodeState(
                node_id="node2",
                config=cfg,
                last_applied_index=250.0,
                last_snapshot_index=200.0,
            ),
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
            e
            for e in result.event_log
            if e.event_type == EventType.NODE_SYNC_COMPLETE
            and e.target_id == "node0"
        ]
        assert len(sync_events) >= 1

        # Must have taken >= 60 s (snapshot download is mandatory)
        assert sync_events[0].time >= 60.0

        # Total is snapshot (60) + log_suffix/net_rate ≈ 72.2 s
        assert sync_events[0].time == pytest.approx(72.2, abs=1.0)

        node0 = result.final_cluster.get_node("node0")
        assert node0.sync is None
        assert node0.last_applied_index >= result.final_cluster.commit_index - 0.1

