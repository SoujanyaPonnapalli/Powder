"""Merge extra budgets from one fleet-pilot run into another (same configurations, same baselines).

    python scripts/merge_fleet_results.py --base A/results-fleet.json --extra B/results-fleet.json --output C

Per (fault scale, price model): the union of both runs' budgets, fleet-ILP and random-menu points (the
extra run wins on a shared budget key), the base run's baselines; the carry-forward (best result at or below
each budget) is re-applied over the merged budgets. Writes ``C/results-fleet.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_fleet_frontier import carry_forward  # noqa: E402


def uncarried(rows: dict) -> dict:
    """Drop carried copies so the merged carry-forward starts from each budget's own solve."""
    return {f: v for f, v in rows.items() if "carried" not in v.get("source", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    base = json.loads((ROOT / a.base).read_text())
    extra = {(r["rate_scale"], r["price"]): r for r in json.loads((ROOT / a.extra).read_text())["rows"]}
    rows = []
    for r in base["rows"]:
        e = extra.get((r["rate_scale"], r["price"]))
        if e is None:
            rows.append(r)
            continue
        budgets = {**r["budgets"], **e["budgets"]}
        order = sorted(budgets, key=float)
        new = {**uncarried(r["new_ilp"]), **uncarried(e["new_ilp"])}
        # a budget whose own solve was replaced by a carried copy has no point of its own: keep it infeasible
        new = {f: new.get(f, {"feasible": False}) for f in order}
        carry_forward(new)
        rows.append({**r, "budgets": {f: budgets[f] for f in order},
                     "matched_budgets": {**r.get("matched_budgets", {}), **e.get("matched_budgets", {})},
                     "new_ilp": new, "old_ilp": {f: {**r["old_ilp"], **e["old_ilp"]}[f] for f in order}})
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "results-fleet.json").write_text(json.dumps({**base, "rows": rows, "merged_from": [a.base, a.extra]},
                                                       indent=1, default=str) + "\n")
    print(out / "results-fleet.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
