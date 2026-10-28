"""One-shot 24-hour Rust MC replays of every AZ-aware free-choice ILP deployment.

Each trial is one simulated day of a whole 10,000-RSM deployment. Machine sets
are disjoint across a deployment's placement groups, so per-group runs are
independent apart from the shared AZ/region processes. Those are sampled once
per trial in Python and injected identically into every group (and every
deployment), so a correlated outage hits the whole fleet on the same day. A
trial's value is the RSM-weighted mean of the groups' 24-hour availability.

The common seed for a machine set on a given trial makes the same group behave
identically wherever it appears. Injected outages use constant delays; if a
domain's event ends early enough (2t + d < 24 h) the engine repeats it, which is
conservative and is flagged in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary, catalogue  # noqa: E402
from powder.faults import catalogue as C  # noqa: E402
from powder.faults.convert import NEVER, engine_compatible, planned_as_cause, scenario_network_config  # noqa: E402
from powder.mc_backend import run_monte_carlo  # noqa: E402
from powder.simulation import (ClusterState, Constant, Exponential, NetworkConfig,  # noqa: E402
                               NetworkState, NodeReplacementStrategy, NodeState, RaftLikeProtocol)

SCENARIO = "scenario/canonical-central-az-aware@v2"
NUM_RSMS = 10_000
HORIZON_S = 24 * 3600.0


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") >> 1


def shared_events(model, rng: np.random.Generator) -> list[dict]:
    """First outage (if any) of each region/AZ process within the horizon."""
    domains = [(r, "region", rate, model.duration) for r, rate in sorted(model.rate_per_second.items())]
    domains += [(f"{r}:az{i}", "az", model.az_rate_per_second, model.az_duration)
                for r in sorted(model.rate_per_second) for i in range(1, model.zones_per_region + 1)]
    events = []
    for name, kind, rate, duration in domains:
        if rate <= 0:
            continue
        start = rng.exponential(1.0 / rate)
        if start < HORIZON_S:
            d = float(duration.sample(rng))
            events.append({"domain": name, "kind": kind, "start_s": float(start), "duration_s": d,
                           "may_repeat": 2 * start + d < HORIZON_S})
    return events


def network_for(events: list[dict], model) -> NetworkConfig | None:
    if not events:
        return None
    regions = sorted(model.rate_per_second)
    azs = [f"{r}:az{i}" for r in regions for i in range(1, model.zones_per_region + 1)]
    per_region = {r: (NEVER, None) for r in regions}
    per_az = {az: (NEVER, None) for az in azs}
    for e in events:
        table = per_region if e["kind"] == "region" else per_az
        table[e["domain"]] = (Constant(e["start_s"]), Constant(e["duration_s"]))
    return NetworkConfig(outage_dist=NEVER, outage_duration_dist=engine_compatible(model.duration),
                         regions=regions, per_region=per_region, azs=azs, per_az=per_az)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontiers", type=Path, default=Path("outputs/exec-02/az-frontiers.json"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/fault_library/catalogue/machines-n2000-seed0-az-v2.json.gz"))
    ap.add_argument("--output", type=Path, default=Path("outputs/exec-02/az-24h-oneshot.json"))
    ap.add_argument("--frontier", default="choice")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.trials < 1:
        ap.error("trials must be positive")
    frontiers = json.loads(args.frontiers.read_text())
    if frontiers["scenario"] != SCENARIO:
        raise ValueError("frontier file is not the AZ-aware scenario")
    lib = FaultLibrary.load()
    cat = catalogue.load(args.catalogue)
    if catalogue.digest(cat) != frontiers["catalogue_digest"]:
        raise ValueError("catalogue does not match the frontier study")
    scenario = lib.get(SCENARIO, "scenario")
    planned = planned_as_cause(lib, scenario, scenario.get("level", "central"))
    model = scenario_network_config(lib, SCENARIO)
    by_id = {m["id"]: m for m in cat["machines"]}
    protocol = RaftLikeProtocol(election_time_dist=Exponential(.2))
    strategy = NodeReplacementStrategy(failure_timeout=600, default_node_config=None)

    clusters = {}

    def cluster(ids):
        if ids not in clusters:
            nodes = {}
            for i, mid in enumerate(ids):
                cfg = C.node_config(by_id[mid], horizon_h=cat["horizon_h"], planned=planned,
                                    system=scenario.get("system"), time_varying=True, cost_per_hour=0.0)
                nodes[f"n{i}"] = NodeState(node_id=f"n{i}", config=cfg)
            clusters[ids] = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=len(nodes))
        return clusters[ids]

    rng = np.random.default_rng(_seed("shared", args.seed))
    trials = [shared_events(model, rng) for _ in range(args.trials)]
    group_cache = {}

    def day(ids, trial):
        key = (ids, trial)
        if key not in group_cache:
            runs = run_monte_carlo(cluster(ids), strategy, protocol, num_simulations=1, max_time=HORIZON_S,
                                   stop_on_data_loss=False, seed=_seed(args.seed, trial, ",".join(ids)),
                                   network_config=network_for(trials[trial], model))
            group_cache[key] = float(runs.availability_samples[0])
        return group_cache[key]

    out = {"scenario": SCENARIO, "catalogue_digest": frontiers["catalogue_digest"],
           "frontier": args.frontier, "horizon_h": HORIZON_S / 3600, "trials": args.trials,
           "seed": args.seed, "shared_events": trials, "arms": {}}
    for price, arm in frontiers["arms"].items():
        data = arm["frontiers"][args.frontier]
        deployments = [("point", i, p) for i, p in enumerate(data["points"])]
        deployments.append(("target_qualified", None, data.get("target_qualified", {})))
        rows = []
        for kind, index, point in deployments:
            if point.get("status") != "0":
                continue
            groups = [(tuple(sorted(g["nodes"])), g["count"]) for g in point["placements"]]
            machines = [m for ids, _ in groups for m in ids]
            if len(machines) != len(set(machines)):
                raise ValueError("placement groups share machines; per-group runs would not be independent")
            if sum(c for _, c in groups) != NUM_RSMS:
                raise ValueError("deployment does not place 10,000 RSMs")
            samples = []
            for t in range(args.trials):
                a = sum(c * day(ids, t) for ids, c in groups) / NUM_RSMS
                samples.append(a)
            rows.append({"kind": kind, "index": index, "cost_per_hour": point["cost_per_hour"],
                         "mc_mean_availability": point["mean_availability"],
                         "rf_counts": point["rf_counts"], "groups": len(groups),
                         "oneshot_availability": samples})
            shown = ", ".join("inf" if a >= 1 else f"{-math.log10(1 - a):.2f}" for a in samples)
            print(f"{price} {kind}{'' if index is None else index} ${point['cost_per_hour']:.2f}/h "
                  f"{len(groups)} groups: nines [{shown}]", flush=True)
        out["arms"][price] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
