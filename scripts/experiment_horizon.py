"""Independent Rust MC check of selected RSMs at prospective live horizons.

These are *individual RSM* predictions, not joint-cluster availability or TiDB validation.
The engine uses catalogue NHPP hazards and an exponential approximation of planned maintenance
(Ford's cadence; saved results before 2026-11-17 used the removed weekly cadence).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary, catalogue  # noqa: E402
from powder.faults.convert import planned_as_cause, scenario_network_config  # noqa: E402
from powder.mc_jobs import build_job  # noqa: E402
from powder.simulation import (ClusterState, Exponential, NetworkState,  # noqa: E402
                               NodeReplacementStrategy, NodeState, RaftLikeProtocol)


def wilson(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = events / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0, center - half), min(1, center + half)


def representative(row: dict, key: str) -> list[str]:
    if key in ("cheap", "blind"):
        placements = row[key]["placements"]
        return placements[0]
    chosen = row[key]["placements"]
    return max(chosen, key=lambda item: item["count"])["nodes"]


def run(args):
    all_rows = [json.loads(line) for line in Path(args.sweep).read_text().splitlines()]
    rows = [r for r in all_rows if r["pool"] == args.pool and r["price_seed"] == args.price_seed]
    if not rows or len({r["price"] for r in rows}) != len(rows):
        raise ValueError("expected distinct price assignments for requested pool and seed")
    lib = FaultLibrary.load()
    master = catalogue.generate(lib, max(r["pool"] for r in all_rows), rows[0]["catalogue_seed"],
                                rows[0].get("spot_fraction", .3), 720)
    master["price_assignments"] = catalogue.assign_catalogue_prices(
        lib, master["machines"], seed=args.price_seed
    )
    if any(r["master_digest"] != catalogue.digest(master) for r in rows):
        raise ValueError("sweep and horizon catalogue draws or price assignments differ")
    machines = {m["id"]: m for m in master["machines"]}
    scenario = lib.get("scenario/canonical-central@v2", "scenario")
    planned = planned_as_cause(lib, scenario, "central")
    network = (None if args.no_network else
               scenario_network_config(lib, "scenario/canonical-central@v2").as_network_config(engine=True))
    protocol = RaftLikeProtocol(election_time_dist=Exponential(.2))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for row in rows:
            for policy in args.policies:
                if "placements" not in row[policy]:
                    continue
                ids = representative(row, policy)
                configs = [catalogue.node_config(machines[mid], planned=planned, time_varying=True,
                                                 cost_per_hour=machines[mid]["price"].get(row["price"],
                                                                                          machines[mid]["price"].get("per_hour")))
                           for mid in ids]
                cluster = ClusterState(nodes={f"n{i}": NodeState(node_id=f"n{i}", config=cfg)
                                              for i, cfg in enumerate(configs)},
                                       network=NetworkState(), target_cluster_size=len(configs))
                for days in args.days:
                    job = build_job(cluster, NodeReplacementStrategy(failure_timeout=600.0,
                                                                     default_node_config=None), protocol,
                                    num_simulations=args.runs, max_time=days * 86400.0,
                                    stop_on_data_loss=False, base_seed=args.seed,
                                    network_config=network)
                    proc = subprocess.run([str(args.binary), "--config", "-"], input=json.dumps(job),
                                          text=True, capture_output=True, check=True, timeout=args.timeout)
                    result = json.loads(proc.stdout)
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    runs = result["runs"]
                    if len(runs) != args.runs or any(abs(r["end_time"] - days * 86400) > 1 for r in runs):
                        raise RuntimeError("Monte Carlo runs did not cover their complete horizon")
                    events = sum(r["total_unavailability_incidents"] > 0 for r in runs)
                    faulty = sum((r["total_transient_failures"] + r["total_dataloss_failures"]) > 0 for r in runs)
                    record = {
                        "price": row["price"], "policy": policy, "pool": args.pool,
                        "price_seed": args.price_seed, "nodes": ids, "horizon_days": days,
                        "network_outages": not args.no_network,
                        "runs": args.runs, "fault_probability": faulty / args.runs,
                        "fault_interval_95": wilson(faulty, args.runs),
                        "incident_probability": events / args.runs,
                        "incident_interval_95": wilson(events, args.runs),
                        "availability_mean": result["summary"]["availability_mean"],
                        "availability_std": result["summary"]["availability_std"],
                        "unavailability_incidents_mean": result["summary"]["unavailability_incidents_mean"],
                    }
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                    fh.flush()
                    print(f"{row['price']} {policy} {days}d: {events}/{args.runs} runs with downtime", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="outputs/exec-02/sweep.jsonl")
    ap.add_argument("--output", default="outputs/exec-02/horizons.jsonl")
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--price-seed", type=int, default=0)
    ap.add_argument("--days", type=float, nargs="+", default=[1, 3, 7, 30])
    ap.add_argument("--policies", nargs="+", default=["cheap", "iso_cost"],
                    choices=["cheap", "blind", "iso_cost", "iso_availability", "iso_blind_quality"])
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--no-network", action="store_true", help="isolate machine-level faults as a sensitivity check")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--binary", type=Path, default=ROOT / "rust/target/release/powder-mc")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
