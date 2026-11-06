"""Scalability from past runs (MVP Fig 10, table form): every optimizer solve that logged its wall time.

    python scripts/figs/harvest_timings.py     # -> outputs/figs/scalability/past_runs.{csv,json,md}

Sources:
* fleet-ILP pilot rows (``outputs/pilot-ilp-rate-scale/**/rows``, ``outputs/figs/**/rows``): one record per
  budget solve (main ILP + its MC + lattice ascent), 50 RSMs over the 2000-machine catalogue; cluster size =
  machines the deployment uses. Savings/month = (cheapest fixed-RF baseline at least as available - ILP cost)
  x 730 h; if no baseline is as available, against the most available baseline (a lower bound).
* exec-02 jsonl (``outputs/exec-02/*.jsonl``): type-count ILP runs with ``seconds``, ``num_rsms`` (50-10,000),
  ``pool``; cluster size = machines deployed by the ILP.

Only wall time was logged, not CPU time. Optimizer $ assumes the solve held one whole i4i.2xlarge
(library price, us-east-1 on-demand), and daily reconfiguration (x 30 per month).
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

PRICE_KEY = "aws:i4i.2xlarge:us-east-1:ondemand:linux"


def vm_price() -> float:
    for f in glob.glob(str(K.ROOT / "data/fault_library/prices/aws-ondemand.json")):
        d = json.loads(Path(f).read_text())
        for e in (d if isinstance(d, list) else [d]):
            if e.get("id") == PRICE_KEY:
                return float(e["price_per_hour"])
    raise KeyError(PRICE_KEY)


def fleet_records(price_vm: float) -> list[dict]:
    out, seen = [], set()
    paths = sorted(glob.glob(str(K.ROOT / "outputs/pilot-ilp-rate-scale/**/rows/*.json"), recursive=True)
                   + glob.glob(str(K.ROOT / "outputs/figs/**/rows/*.json"), recursive=True))
    for p in paths:
        row = json.loads(Path(p).read_text())
        base = row.get("baselines", {})
        for f, r in row.get("new_ilp", {}).items():
            t = r.get("times") or {}
            if not r.get("feasible") or "carried" in r.get("source", "") or "main_ilp_s" not in t:
                continue
            wall = sum(t.get(k, 0.0) for k in ("main_ilp_s", "main_mc_s", "ascent_ilp_s", "ascent_mc_s"))
            key = (row["rate_scale"], row["price"], round(r["cost_per_hour"], 4), round(wall, 1))
            if key in seen:
                continue
            seen.add(key)
            better = [b["cost_per_hour"] for b in base.values() if b["mean_availability"] >= r["mean_availability"]]
            ref = min(better) if better else max(base.values(), key=lambda b: b["mean_availability"])["cost_per_hour"]
            out.append({"source": str(Path(p).relative_to(K.ROOT)), "pipeline": "fleet ILP", "num_rsms": 50,
                        "catalogue_machines": 2000, "rate_scale": row["rate_scale"], "price": row["price"],
                        "budget": f, "cluster_nodes": r["machines_used"], "rsm_replicas": sum(
                            int(k) * v for k, v in r["rf_counts"].items()),
                        "nines": r["nines"], "cost_per_hour": r["cost_per_hour"], "wall_s": wall,
                        "optimizer_usd_per_month_daily": wall / 3600 * price_vm * 30,
                        "savings_usd_per_month": (ref - r["cost_per_hour"]) * 730,
                        "savings_is_lower_bound": not better})
    return out


def exec02_records(price_vm: float) -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(K.ROOT / "outputs/exec-02/*.jsonl"))):
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or "seconds" not in r or "num_rsms" not in r:
                continue
            ilp = r.get("iso_availability") or r.get("iso_cost") or {}
            blind = r.get("blind") or {}
            out.append({"source": str(Path(p).relative_to(K.ROOT)), "pipeline": "type-count ILP (exec-02)",
                        "num_rsms": r["num_rsms"], "catalogue_machines": r.get("pool"),
                        "rate_scale": 1.0, "price": r.get("price"), "budget": None,
                        "cluster_nodes": ilp.get("deployed"), "rsm_replicas": r["num_rsms"] * r.get("replicas", 3),
                        "nines": K.nines(ilp["mean_availability_exact"]) if "mean_availability_exact" in ilp else None,
                        "cost_per_hour": ilp.get("cost_per_hour"), "wall_s": float(r["seconds"]),
                        "optimizer_usd_per_month_daily": float(r["seconds"]) / 3600 * price_vm * 30,
                        "savings_usd_per_month": ((blind["cost_per_hour"] - ilp["cost_per_hour"]) * 730
                                                  if blind and "cost_per_hour" in ilp else None),
                        "savings_is_lower_bound": False})
    return out


def main() -> int:
    price_vm = vm_price()
    recs = fleet_records(price_vm) + exec02_records(price_vm)
    out = K.ROOT / "outputs/figs/scalability"
    out.mkdir(parents=True, exist_ok=True)
    (out / "past_runs.json").write_text(json.dumps({"vm_price_per_hour": price_vm, "records": recs}, indent=1) + "\n")
    with open(out / "past_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    lines = ["# Optimizer runtime from past runs", "",
             f"Wall time per solve; optimizer $/month assumes one i4i.2xlarge (${price_vm}/h) and daily reconfiguration.",
             "", "| pipeline | RSMs | replicas | nodes used | solves | wall s (median / max) | optimizer $/mo | savings $/mo (median) |",
             "|---|---|---|---|---|---|---|---|"]
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["pipeline"], r["num_rsms"]), []).append(r)
    for (pipe, n), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        walls = sorted(r["wall_s"] for r in rs)
        nodes = [r["cluster_nodes"] for r in rs if r["cluster_nodes"]]
        sav = sorted(r["savings_usd_per_month"] for r in rs if r["savings_usd_per_month"] is not None)
        reps = sorted({r["rsm_replicas"] for r in rs})
        lines.append(f"| {pipe} | {n} | {reps[0]}-{reps[-1]} | {min(nodes)}-{max(nodes)} | {len(rs)} | "
                     f"{walls[len(walls) // 2]:.0f} / {walls[-1]:.0f} | "
                     f"{walls[len(walls) // 2] / 3600 * price_vm * 30:.2f} | "
                     f"{sav[len(sav) // 2]:.0f} |" if sav else "| - |")
    (out / "past_runs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
