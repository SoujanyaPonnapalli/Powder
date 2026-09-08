"""Replay the rare Unreliable N=3 week found by the convergence study.

Run: .venv/bin/python -m notebooks.availability_convergence_tail_trace
Outputs the event log and final state for the recorded seed, without changing
the simulator or its recovery policy.
"""

import json
from dataclasses import asdict

from notebooks.availability_convergence_study import (
    ROOT, VM_PROFILES, make_cluster, node_config_for, raft_protocol,
    replacement_strategy,
)
from powder.simulation import Seconds
from powder.simulation.distributions import days
from powder.simulation.simulator import Simulator


def main() -> None:
    seed = 23070693
    config = node_config_for(next(p for p in VM_PROFILES if p.name == "Unreliable"))
    protocol = raft_protocol()
    simulator = Simulator(
        initial_cluster=make_cluster(3, config),
        strategy=replacement_strategy(config), protocol=protocol,
        seed=seed, log_events=True,
    )
    result = simulator.run_until(end_time=Seconds(days(7)))
    observed = result.metrics.availability_fraction()
    if abs(observed - 0.044323613179782305) > 1e-12:
        raise RuntimeError(f"Recorded tail changed: availability={observed}")
    cluster = result.final_cluster
    record = {
        "profile": "Unreliable", "nodes": 3, "seed": seed, "horizon_days": 7,
        "availability": observed, "metrics": asdict(result.metrics),
        "end_reason": result.end_reason, "end_time": result.end_time,
        "can_commit_at_end": protocol.can_commit(cluster),
        "final_commit_index": cluster.commit_index,
        "synced_standbys_at_end": [
            node.node_id for node in cluster.standby_nodes.values()
            if node.is_up_to_date(cluster.commit_index)
        ],
        "final_nodes": [
            {"node_id": node.node_id, "available": node.is_available,
             "has_data": node.has_data, "applied_index": node.last_applied_index,
             "sync": asdict(node.sync) if node.sync else None}
            for node in cluster.nodes.values()
        ],
        "events": [
            {"time": event.time, "type": event.event_type.name,
             "target": event.target_id,
             "metadata": {key: value for key, value in event.metadata.items()
                          if key != "node_config"}}
            for event in result.event_log
        ],
    }
    output = ROOT / "outputs/availability-convergence-study/unreliable-3-tail-trace.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
