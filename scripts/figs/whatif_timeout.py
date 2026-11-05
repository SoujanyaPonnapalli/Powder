"""What-if estimator: node replacement timeout (MVP Fig 4, the "expected" bars).

    python scripts/figs/whatif_timeout.py --row outputs/figs/deployments/rows/x30-rho1.json --budget 0.5 \
        [--timeouts-min 5 10 15 20 30 60] [--runs 500000] [--horizon-h 24]

Takes a saved fleet-ILP deployment (50 RSMs) and re-runs exact Rust MC on every unique placement under each
replacement timeout. Machines, protocol, region/AZ outages and seeds stay fixed, so the bars differ only by the
timeout (common random numbers: each placement uses the same seed under every timeout).

Deployment availability = mean over RSMs. Its standard error combines the per-placement MC standard errors
(placements repeated in the deployment share one estimate). Cost = the fleet's $/h (unique machines) plus the
replacement overhead the engine bills (replacement nodes at the failed node's price, until removed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, help="fleet-pilot row JSON with saved placements")
    ap.add_argument("--budget", default="0.5", help="budget key in the row's new_ilp")
    ap.add_argument("--timeouts-min", type=float, nargs="+", default=[5, 10, 15, 20, 30, 60])
    ap.add_argument("--runs", type=int, default=500_000)
    ap.add_argument("--horizon-h", type=float, default=24.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--output", default=None, help="JSON path (default outputs/figs/whatif/<tag>.json)")
    a = ap.parse_args()

    row = K.load_row(Path(a.row))
    scale, price = float(row["rate_scale"]), row["price"]
    dep = K.deployment(row, a.budget)
    counts = Counter(dep)
    by_id = K.machines(scale, price)
    fleet = sorted({m for p in dep for m in p})
    fleet_cost = sum(by_id[m].node_config.cost_per_hour for m in fleet)
    horizon_s = a.horizon_h * 3600.0
    tag = f"x{scale:g}-{price}-b{a.budget}-h{a.horizon_h:g}"
    out = Path(a.output) if a.output else K.ROOT / "outputs/figs/whatif" / f"{tag}.json"
    print(f"[{tag}] {len(dep)} RSMs, {len(counts)} unique placements, {len(fleet)} machines, ${fleet_cost:.2f}/h",
          flush=True)

    jobs = [(t, p) for t in a.timeouts_min for p in counts]
    t0 = time.time()

    def run(job):
        t, p = job
        st = K.mc_stats([by_id[m].node_config for m in p], timeout_s=t * 60, horizon_s=horizon_s, runs=a.runs,
                        seed=K.placement_seed(p))
        return job, st

    stats = {}
    with ThreadPoolExecutor(a.threads) as pool:
        for i, (job, st) in enumerate(pool.map(run, jobs), 1):
            stats[job] = st
            if i % 25 == 0 or i == len(jobs):
                print(f"[{tag}] {i}/{len(jobs)} placement-timeouts, {time.time() - t0:.0f}s", flush=True)

    n = len(dep)
    results = []
    for t in a.timeouts_min:
        mean = sum(c * stats[(t, p)].mean for p, c in counts.items()) / n
        se = math.sqrt(sum((c * stats[(t, p)].se) ** 2 for p, c in counts.items())) / n
        base = sum(c * sum(by_id[m].node_config.cost_per_hour for m in p) for p, c in counts.items())
        run_cost = sum(c * stats[(t, p)].cost_mean for p, c in counts.items()) / a.horizon_h
        spawned = sum(c * stats[(t, p)].spawned_mean for p, c in counts.items())
        lo, hi = max(mean - 1.96 * se, 0.0), min(mean + 1.96 * se, 1 - 1e-15)
        results.append({
            "timeout_min": t, "mean_availability": mean, "se": se, "nines": K.nines(mean),
            "nines_ci95": [K.nines(lo), K.nines(hi)],
            "fleet_cost_per_hour": fleet_cost,
            "replacement_overhead_per_hour": max(0.0, run_cost - base),
            "replacements_per_day": spawned * 24.0 / a.horizon_h,
            "mc_seconds": sum(stats[(t, p)].seconds for p in counts),
        })
        r = results[-1]
        print(f"[{tag}] timeout {t:g} min: {r['nines']:.3f} nines (95% {r['nines_ci95'][0]:.3f}-"
              f"{r['nines_ci95'][1]:.3f}), +${r['replacement_overhead_per_hour']:.3f}/h, "
              f"{r['replacements_per_day']:.2f} replacements/day", flush=True)

    committed = row["new_ilp"][a.budget]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "row": str(a.row), "rate_scale": scale, "price": price, "budget": a.budget,
        "budget_per_hour": row["budgets"][a.budget], "rf_counts": committed.get("rf_counts"),
        "optimizer_nines_720h_600s": committed.get("nines"), "horizon_h": a.horizon_h, "runs": a.runs,
        "protocol": "raft, election Exp(0.2/s)", "network": "region/AZ outages on",
        "unique_placements": len(counts), "machines": len(fleet), "results": results,
        "per_placement": [{"timeout_min": t, "machines": list(p), "count": counts[p], "mean": s.mean, "se": s.se,
                           "cost_mean": s.cost_mean, "spawned_mean": s.spawned_mean}
                          for (t, p), s in stats.items()],
        "seconds": time.time() - t0}, indent=1) + "\n")
    print(f"[{tag}] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
