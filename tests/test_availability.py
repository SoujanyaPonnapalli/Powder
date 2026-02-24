"""
Deterministic availability tests for leaderless and Raft strategies.

Each test constructs a scenario where availability can be computed
analytically, then runs the full simulator and asserts the result
matches the expected fraction.
"""

import pytest

from powder.simulation import (
    Seconds,
    hours,
    days,
    minutes,
    Constant,
    NodeConfig,
    NodeState,
    NetworkState,
    ClusterState,
    NoOpStrategy,
    EventType,
    Simulator,
    LeaderlessMajorityAvailableProtocol,
    RaftLikeProtocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _never_fail_config(region: str = "us-east") -> NodeConfig:
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


def _single_failure_config(
    fail_after: float,
    recovery_time: float,
    region: str = "us-east",
) -> NodeConfig:
    """Node that fails once at `fail_after` seconds, recovers in
    `recovery_time` seconds, then never fails again."""
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


def _cycling_failure_config(
    fail_interval: float,
    recovery_time: float,
    region: str = "us-east",
) -> NodeConfig:
    """Node that fails every `fail_interval` seconds and takes
    `recovery_time` seconds to recover, cycling indefinitely."""
    return NodeConfig(
        region=region,
        cost_per_hour=1.0,
        failure_dist=Constant(fail_interval),
        recovery_dist=Constant(recovery_time),
        data_loss_dist=Constant(days(9999)),
        log_replay_rate_dist=Constant(2.0),
        snapshot_download_time_dist=Constant(0),
        spawn_dist=Constant(0),
    )


# ===========================================================================
# Leaderless Availability Tests
# ===========================================================================


class TestLeaderlessAvailability:
    """Deterministic availability tests using LeaderlessMajorityAvailableProtocol."""

    def test_single_node_cycling_100_percent(self):
        """In a 3-node cluster, one node cycling up/down never breaks quorum.

        Timeline (10 days, node0 cycles with 1 day up / 1 day down):
        ---------------------------------------------------------------
        Days 0–1: 3 available  → quorum 2 met ✓
        Days 1–2: 2 available  → quorum 2 met ✓  (node0 down)
        Days 2–3: 3 available  → quorum 2 met ✓  (node0 recovered)
        Days 3–4: 2 available  → quorum 2 met ✓  (node0 down again)
        ... pattern repeats

        Expected: 100% availability.
        """
        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_cycling_failure_config(
                    fail_interval=days(1),
                    recovery_time=days(1),
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
        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessMajorityAvailableProtocol(),
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(days(10))

        assert result.end_reason == "time_limit"
        assert result.metrics.availability_fraction() == pytest.approx(1.0)

    def test_overlapping_failures_80_percent(self):
        """Two nodes with overlapping downtime break quorum for 1 day.

        Timeline (5 days):
        ------------------
        Days 0–1: 3 available  → quorum 2 met ✓
        Days 1–2: node0 down, 2 available  → quorum 2 met ✓
        Days 2–3: node0 + node1 both down, 1 available  → quorum 2 NOT met ✗
        Days 3–4: node0 recovered, node1 still down, 2 available  → ✓
        Days 4–5: 3 available  → ✓

        Available time = 4 days,  Total = 5 days  →  80%
        """
        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=days(1),
                    recovery_time=days(2),
                ),
            ),
            "node1": NodeState(
                node_id="node1",
                config=_single_failure_config(
                    fail_after=days(2),
                    recovery_time=days(2),
                ),
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
        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=LeaderlessMajorityAvailableProtocol(),
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(days(5))

        assert result.end_reason == "time_limit"
        assert result.metrics.availability_fraction() == pytest.approx(4.0 / 5.0)


# ===========================================================================
# Raft Availability Tests
# ===========================================================================


class TestRaftAvailability:
    """Deterministic availability tests using RaftLikeProtocol.

    Key insight: when the *leader* fails, unavailability is proportional
    to election time, NOT the failed node's recovery time.  When a
    *non-leader* fails, there is no availability impact as long as
    quorum is maintained.
    """

    def test_leader_failure_availability_hit_equals_election_time(self):
        """Leader fails → unavailability = election time, not recovery time.

        Setup:
        - 3-node cluster, node0 is leader (lowest alphabetical ID)
        - node0 fails at 1 hour, recovers after 1 hour
        - Election takes exactly 10 seconds
        - node1, node2 never fail → new leader elected after 10s

        Expected: unavailability = 10 seconds.
        Availability = (7200 - 10) / 7200
        """
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

        election_time = 10.0  # seconds
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        duration = hours(2)  # 7200 seconds
        result = simulator.run_for(duration)

        assert result.end_reason == "time_limit"

        expected_availability = (duration - election_time) / duration
        assert result.metrics.availability_fraction() == pytest.approx(
            expected_availability
        )

        # Verify the unavailable time is exactly the election duration
        assert result.metrics.time_unavailable == pytest.approx(election_time)

    def test_non_leader_failure_100_percent_availability(self):
        """Non-leader failure should NOT trigger an election → 100% available.

        Setup:
        - 3-node cluster, node0 is leader
        - node1 fails at 1 hour, recovers after 1 hour
        - node0 and node2 never fail
        - Quorum: node0 (leader, up) + node2 (up) = 2 ≥ quorum of 2

        Expected: 100% availability.
        """
        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_never_fail_config(),
            ),
            "node1": NodeState(
                node_id="node1",
                config=_single_failure_config(
                    fail_after=hours(1),
                    recovery_time=hours(1),
                ),
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

        protocol = RaftLikeProtocol(election_time_dist=Constant(10.0))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        result = simulator.run_for(hours(4))

        assert result.end_reason == "time_limit"
        assert result.metrics.availability_fraction() == pytest.approx(1.0)
        assert result.metrics.time_unavailable == pytest.approx(0.0)

    def test_leader_failure_proportional_election_hit(self):
        """Long election time → availability hit proportional to election, not recovery.

        Setup:
        - 3-node cluster, node0 is leader
        - node0 fails at day 1, recovers after 1 day
        - Election takes exactly 1 hour
        - Run for 10 days

        Expected: unavailability = 1 hour.
        Availability = (10 days - 1 hour) / 10 days
        """
        nodes = {
            "node0": NodeState(
                node_id="node0",
                config=_single_failure_config(
                    fail_after=days(1),
                    recovery_time=days(1),
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

        election_time = hours(1)  # 3600 seconds
        protocol = RaftLikeProtocol(election_time_dist=Constant(election_time))

        simulator = Simulator(
            initial_cluster=cluster,
            strategy=NoOpStrategy(),
            protocol=protocol,
            seed=0,
            log_events=True,
        )

        duration = days(10)
        result = simulator.run_for(duration)

        assert result.end_reason == "time_limit"

        expected_availability = (duration - election_time) / duration
        assert result.metrics.availability_fraction() == pytest.approx(
            expected_availability
        )
        assert result.metrics.time_unavailable == pytest.approx(election_time)
