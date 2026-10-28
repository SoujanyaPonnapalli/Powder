"""Cluster trims keep the leader and the placement's fault domains.

Regressions for runs that stayed unavailable for hundreds of hours after an
AZ/region outage: a trim removed the Raft leader (no election followed), and
a trim deferred past a promotion later removed a sole-domain member, leaving
two replicas in one AZ.
"""

from powder.simulation import (ClusterState, Constant, EventType, Event, NetworkState,
                               NodeConfig, NodeReplacementStrategy, NodeState, RaftLikeProtocol)
from powder.simulation.strategy import ActionType

import numpy as np


def config(region):
    c = Constant(1000.0)
    return NodeConfig(region=region, az=f"{region}:az1", cost_per_hour=1.0, failure_dist=c,
                      recovery_dist=c, data_loss_dist=c, log_replay_rate_dist=Constant(100.0),
                      snapshot_download_time_dist=c, spawn_dist=c)


def cluster(regions, target):
    nodes = {f"node{i}": NodeState(node_id=f"node{i}", config=config(r)) for i, r in enumerate(regions)}
    return ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=target)


def removed(actions):
    return [a.params["node_id"] for a in actions if a.action_type == ActionType.REMOVE_NODE]


def test_trim_keeps_the_leader_among_equally_healthy_nodes():
    c = cluster(["r"] * 4, target=3)
    p = RaftLikeProtocol(election_time_dist=Constant(1.0))
    p._leader_id = "node3"  # ties otherwise remove the last-inserted node
    s = NodeReplacementStrategy(failure_timeout=600)
    actions = s.on_event(Event(10.0, EventType.NODE_RECOVERY, "node0"), c, np.random.default_rng(0), p)
    assert removed(actions) == ["node2"]


def test_promotion_is_trimmed_in_the_same_pass_within_the_shared_domain():
    c = cluster(["us-west-1", "us-east-2", "us-east-1"], target=3)
    c.add_standby_node(NodeState(node_id="replacement_1", config=config("us-west-1")))
    p = RaftLikeProtocol(election_time_dist=Constant(1.0))
    rng = np.random.default_rng(0)
    p.on_simulation_start(c, rng)  # a leader, so safe mode allows promotion
    s = NodeReplacementStrategy(failure_timeout=600)
    actions = s.on_event(Event(10.0, EventType.NODE_SYNC_COMPLETE, "replacement_1"), c, rng, p)
    assert any(a.action_type == ActionType.PROMOTE_NODE for a in actions)
    assert removed(actions) in (["node0"], ["replacement_1"])


def test_removing_the_leader_starts_an_election():
    c = cluster(["r"] * 3, target=3)
    p = RaftLikeProtocol(election_time_dist=Constant(1.0))
    rng = np.random.default_rng(0)
    p.on_simulation_start(c, rng)
    leader = p.leader_id
    assert p.on_node_removed("node1" if leader != "node1" else "node2", c, rng) == []
    c.remove_node(leader)
    events = p.on_node_removed(leader, c, rng)
    assert [e.event_type for e in events] == [EventType.LEADER_ELECTION_COMPLETE]
    assert p.leader_id is None
