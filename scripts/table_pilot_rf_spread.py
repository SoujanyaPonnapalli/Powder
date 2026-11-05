"""RSM replica-count spread of every point of every strategy in a fleet-pilot results file.

    python scripts/table_pilot_rf_spread.py [--input outputs/pilot-ilp-rate-scale/lattice/results-fleet.json]

Writes ``rf_spread.csv`` (one row per point) and ``rf_spread.md`` (per configuration: budgets x strategies,
each cell ``nines, $/h, RF3/RF5/RF7 RSM counts``) next to the input. Strategies: random-menu ILP, fleet ILP
(main ILP, before lattice ascent), lattice (after ascent), and the fixed-RF baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICES = {"rho0": "independent prices", "rho1": "reliability-ranked prices"}


def spread(p: dict, num_rsms: int, rf: int | None = None) -> tuple[int, int, int]:
    counts = {int(k): v for k, v in p.get("rf_counts", {}).items()} if rf is None else {rf: num_rsms}
    return tuple(counts.get(r, 0) for r in (3, 5, 7))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/pilot-ilp-rate-scale/lattice/results-fleet.json")
    a = ap.parse_args()
    path = ROOT / a.input
    data = json.loads(path.read_text())
    n = data["params"]["num_rsms"]
    rows, md = [], ["# RSM replica-count spread per point", "",
                    "Each cell: availability (nines), cost ($/h), RSMs at RF 3/5/7. Exact Monte Carlo.", ""]
    for r in data["rows"]:
        k, price = r["rate_scale"], r["price"]
        md += [f"## {k:g}x fault rates, {PRICES[price]}", "",
               "| Budget | Random-menu ILP | Fleet ILP | Lattice |", "|---|---|---|---|"]
        for f, b in r["budgets"].items():
            cells = []
            for name, p in (("random-menu ILP", r["old_ilp"][f]), ("fleet ILP", r["new_ilp"][f].get("main_ilp", {})),
                            ("lattice", r["new_ilp"][f])):
                if not p.get("feasible"):
                    cells.append("infeasible")
                    rows.append([k, price, f, round(b, 2), name, "", "", "", "", ""])
                    continue
                s3, s5, s7 = spread(p, n)
                cells.append(f"{p['nines']:.2f}, ${p['cost_per_hour']:.1f}, {s3}/{s5}/{s7}")
                rows.append([k, price, f, round(b, 2), name, round(p["nines"], 3), round(p["cost_per_hour"], 2),
                             s3, s5, s7])
            md.append(f"| {float(f):.0%} (${b:.2f}) | " + " | ".join(cells) + " |")
        md += ["", "Baselines: " + "; ".join(
            f"{name} {p['nines']:.2f} at ${p['cost_per_hour']:.1f}" for name, p in r["baselines"].items()), ""]
        for name, p in r["baselines"].items():
            rf = int(name.split("-")[0][2:])
            rows.append([k, price, "", "", f"baseline {name}", round(p["nines"], 3), round(p["cost_per_hour"], 2),
                         *spread(p, n, rf)])
    with open(path.parent / "rf_spread.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fault_scale", "price", "budget_fraction", "budget_per_hour", "strategy", "nines",
                    "cost_per_hour", "rsms_rf3", "rsms_rf5", "rsms_rf7"])
        w.writerows(rows)
    (path.parent / "rf_spread.md").write_text("\n".join(md) + "\n")
    print(path.parent / "rf_spread.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
