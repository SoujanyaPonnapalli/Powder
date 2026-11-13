"""Benchmark the old vs new fleet-ILP scoring: timing, and error against an independent truth.

    python scripts/figs/bench_truth.py --strategies old=DIR new=DIR new_is=DIR [--runs 2000000]

For every strategy's chosen deployment (rows/x*-rho*.json, each budget), the truth is plain Rust MC with
``--runs`` per unique placement on seeds disjoint from the optimizer's batches (old: seed + 0; new: seed + b x
runs for b < 10) and from --validate-runs (seed + 1e9): seed + 7e9. Error = believed (or validated) - truth, in
nines. Timing = per-budget optimization wall time (``times.budget_wall_s``) and validation time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

TRUTH_OFFSET = 7 * 10**9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", required=True, help="name=output_dir")
    ap.add_argument("--runs", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--output", default=str(K.ROOT / "outputs/figs/bench/truth.json"))
    a = ap.parse_args()
    strategies = dict(s.split("=", 1) for s in a.strategies)

    rows = []   # (strategy, scale, price, budget, row dict)
    for name, d in strategies.items():
        for path in sorted(Path(d, "rows").glob("x*-*.json")):
            row = json.loads(path.read_text())
            for f, r in row["new_ilp"].items():
                if r.get("feasible") and "placements" in r:
                    rows.append((name, float(row["rate_scale"]), row["price"], f, r, row))
    need = {}
    for name, scale, price, f, r, _ in rows:
        for p in r["placements"]:
            need[(scale, price, tuple(sorted(p)))] = None
    print(f"{len(rows)} deployments, {len(need)} unique placements x {a.runs} runs", flush=True)

    def truth(key):
        scale, price, p = key
        by = K.machines(scale, price)
        st = K.mc_stats([by[m].node_config for m in p], timeout_s=600, horizon_s=720 * 3600, runs=a.runs,
                        seed=K.placement_seed(p) + TRUTH_OFFSET)
        return key, st

    done = 0
    with ThreadPoolExecutor(a.threads) as pool:
        for key, st in pool.map(truth, list(need)):
            need[key] = st
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(need)}", flush=True)

    out = []
    for name, scale, price, f, r, row in rows:
        counts = Counter(tuple(sorted(p)) for p in r["placements"])
        n = sum(counts.values())
        mean = sum(c * need[(scale, price, p)].mean for p, c in counts.items()) / n
        se = math.sqrt(sum((c * need[(scale, price, p)].se) ** 2 for p, c in counts.items())) / n
        t = r.get("times", {})
        rec = {"strategy": name, "rate_scale": scale, "price": price, "budget": f, "source": r.get("source", ""),
               "cost_per_hour": r["cost_per_hour"], "rf_counts": r.get("rf_counts"),
               "believed_nines": r["nines"], "truth_nines": K.nines(mean), "truth_se": se,
               "error_nines": r["nines"] - K.nines(mean),
               "validated_nines": r.get("validated", {}).get("nines"),
               "validated_error_nines": (r["validated"]["nines"] - K.nines(mean)) if "validated" in r else None,
               "optimize_s": t.get("budget_wall_s"), "validate_s": t.get("validate_s", 0.0),
               "carried": "carried" in r.get("source", ""), "config_seconds": row["seconds"]}
        out.append(rec)
        print(f"{name:7s} x{scale:g} {f:>9s}: believed {rec['believed_nines']:.2f} truth {rec['truth_nines']:.2f} "
              f"(err {rec['error_nines']:+.2f})" + (f" validated {rec['validated_nines']:.2f}" if rec["validated_nines"] else "")
              + f" ${rec['cost_per_hour']:.2f}/h, opt {rec['optimize_s'] or 0:.0f}s", flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps({"runs": a.runs, "truth_seed_offset": TRUTH_OFFSET, "strategies": strategies,
                                          "deployments": out}, indent=1) + "\n")
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
