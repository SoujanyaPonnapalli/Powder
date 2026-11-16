"""Sensitivity to reliability input error at 1x (MVP Fig 11).

1. Frontiers under assumed rates m x truth (truth = 1x), rho1 prices, saved deployments:

    python scripts/pilot_fleet_frontier.py --scales 0.25 0.35 0.5 0.7 1 1.4 2 2.8 4 --prices rho1 \
        --budget-fractions 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4 0.5 --cheapest-budgets \
        --output outputs/figs/sensitivity --cache-dir outputs/figs/cache --workers 3 --threads 3 --resume

2. This script: per m, the cheapest saved deployment whose *believed* availability (exact MC at scale m)
   meets the SLA; then its true availability (exact MC at 1x, same horizon, error bars from per-placement s.e.).
   The true estimate uses seeds disjoint from the optimizer's (``SEED_OFFSET``) and 1M runs: the optimizer's
   100k-run scores of 3-replica placements near 5 nines are optimistic (rare long outages are often missed, and the
   ILP picks the luckiest estimates), so reusing its seeds would repeat the bias.

    python scripts/figs/sensitivity.py [--sla-nines 5] [--runs 200000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

SEED_OFFSET = 7 * 10**9  # disjoint from the optimizer's batches (seed + b x runs) and its --validate-runs (seed + 1e9)


def score(r: dict, select_by: str) -> float:
    """Nines the operator decides on: the optimizer's own score, or its independent validation (new scoring)."""
    if select_by == "validated":
        if "validated" not in r:
            raise ValueError("--select-by validated needs rows from --validate-runs")
        return r["validated"]["nines"]
    return r["nines"]


def choose(row: dict, sla_nines: float, select_by: str = "believed") -> tuple[str, dict, bool]:
    """(budget key, row, meets SLA as scored): cheapest deployment scoring >= the SLA, else the most available."""
    ok = [(r["cost_per_hour"], f, r) for f, r in row["new_ilp"].items() if r.get("feasible") and "placements" in r]
    if not ok:
        raise ValueError(f"x{row['rate_scale']}: no saved deployments")
    meets = [x for x in ok if score(x[2], select_by) >= sla_nines]
    if meets:
        _, f, r = min(meets, key=lambda x: (x[0], -score(x[2], select_by)))
        return f, r, True
    _, f, r = max(ok, key=lambda x: score(x[2], select_by))
    return f, r, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(K.ROOT / "outputs/figs/sensitivity/rows"))
    ap.add_argument("--price", default="rho1")
    ap.add_argument("--sla-nines", type=float, default=5.0)
    ap.add_argument("--runs", type=int, default=1_000_000)
    ap.add_argument("--horizon-h", type=float, default=720.0, help="as the optimizer scores")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--select-by", choices=["believed", "validated"], default="believed")
    ap.add_argument("--output", default=str(K.ROOT / "outputs/figs/sensitivity/sensitivity.json"))
    a = ap.parse_args()

    truth = K.machines(1.0, a.price)
    out = []
    for path in sorted(Path(a.rows).glob(f"x*-{a.price}.json"), key=lambda p: float(p.stem[1:].split("-")[0])):
        row = K.load_row(path)
        m = float(row["rate_scale"])
        f, r, meets = choose(row, a.sla_nines, a.select_by)
        counts = Counter(tuple(sorted(p)) for p in r["placements"])

        def run(p):
            return p, K.mc_stats([truth[x].node_config for x in p], timeout_s=600, horizon_s=a.horizon_h * 3600,
                                 runs=a.runs, seed=K.placement_seed(p) + SEED_OFFSET)

        with ThreadPoolExecutor(a.threads) as pool:
            stats = dict(pool.map(run, counts))
        n = sum(counts.values())
        mean = sum(c * stats[p].mean for p, c in counts.items()) / n
        se = math.sqrt(sum((c * stats[p].se) ** 2 for p, c in counts.items())) / n
        lo, hi = mean - 1.96 * se, min(mean + 1.96 * se, 1 - 1e-15)
        out.append({"assumed_multiplier": m, "budget": f, "cost_per_hour": r["cost_per_hour"],
                    "believed_nines": r["nines"], "believed_meets_sla": meets, "rf_counts": r.get("rf_counts"),
                    "validated_nines": r.get("validated", {}).get("nines"), "select_by": a.select_by,
                    "true_availability": mean, "true_se": se, "true_nines": K.nines(mean),
                    "true_nines_ci95": [K.nines(lo), K.nines(hi)], "true_meets_sla": K.nines(mean) >= a.sla_nines})
        print(f"m={m:g}: budget {f} ${r['cost_per_hour']:.2f}/h believed {r['nines']:.2f} -> true "
              f"{K.nines(mean):.2f} nines", flush=True)
    Path(a.output).write_text(json.dumps({"sla_nines": a.sla_nines, "price": a.price, "horizon_h": a.horizon_h,
                                          "runs": a.runs, "points": out}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
