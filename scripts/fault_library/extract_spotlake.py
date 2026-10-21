"""SpotLake AWS archive (Mar 2025 - Mar 2026) -> daily US spot series + yearly statistics.

    python scripts/fault_library/extract_spotlake.py [ZIP]

Input: the SpotLake AWS archive the user downloaded (default: the newest aws-*.zip in the
repo root, or data/fault_library/raw/spot/spotlake/*.zip). It nests one zip per month,
each holding one Parquet file per day with 10-minute rows:
    Time, InstanceType, Region, AZ, SPS, T3, T2, IF, OndemandPrice, SpotPrice, Savings
IF is the Spot Advisor interruption band as 3.0 (<5%) ... 1.0 (>20%) in steps of 0.5
(-1 = unknown); band index = (3 - IF) * 2, verified against a Spot Advisor snapshot
(all 3962 US pools agree).

Outputs (raw data stays out of git; the derived files are small):
  data/fault_library/derived/spotlake_daily_us.csv.gz
      one row per (date, region, instance type): spot $/h (median over AZs per timestamp,
      then daily mean/min/max), on-demand $/h, modal band, mean SPS, AZ count
  data/fault_library/derived/spotlake_summary.json
      per pool: days, time share per band, band changes, spot price mean/p5/p50/p95/CV,
      mean spot/on-demand ratio; plus the exec #4.2 cross-pool correlation between the
      price ratio and the interruption band
Only US regions and instance types in the latest 8 vCPU x86 on-demand price list (plus the
deployment types) are kept.
"""

from __future__ import annotations

import csv
import glob
import gzip
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "data/fault_library"
US = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
DEPLOYMENT_TYPES = ["i4i.2xlarge", "m6id.2xlarge"]


def find_zip(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    cands = sorted(glob.glob(str(ROOT / "aws-*.zip"))) + sorted(glob.glob(str(LIB / "raw/spot/spotlake/*.zip")))
    if not cands:
        raise SystemExit("no SpotLake archive found (aws-*.zip in the repo root)")
    return Path(cands[-1])


def keep_types() -> set[str]:
    rows_path = sorted(glob.glob(str(LIB / "raw/aws_prices/*/ondemand_8vcpu_x86.json")))
    types = set(DEPLOYMENT_TYPES)
    if rows_path:
        types |= {r["instance_type"] for r in json.loads(Path(rows_path[-1]).read_text())}
    return types


def day_rows(table: pa.Table, types: set[str]):
    t = table.filter(pc.and_(pc.is_in(table["Region"], value_set=pa.array(US)),
                             pc.is_in(table["InstanceType"], value_set=pa.array(sorted(types)))))
    if t.num_rows == 0:
        return []
    df = t.select(["Time", "InstanceType", "Region", "AZ", "SPS", "IF", "OndemandPrice", "SpotPrice"]).to_pandas()
    df = df[df["SpotPrice"] > 0]
    # per timestamp: median over AZs; then daily statistics per pool
    per_ts = df.groupby(["Region", "InstanceType", "Time"]).agg(
        spot=("SpotPrice", "median"), od=("OndemandPrice", "median"), IF=("IF", "max"), sps=("SPS", "mean"),
        azs=("AZ", "nunique")).reset_index()
    out = []
    for (reg, it), g in per_ts.groupby(["Region", "InstanceType"]):
        ifs = g["IF"][g["IF"] > 0]
        band = int(round((3.0 - float(ifs.mode().iloc[0])) * 2)) if len(ifs) else -1
        out.append({"region": reg, "instance_type": it, "spot_mean": float(g["spot"].mean()),
                    "spot_min": float(g["spot"].min()), "spot_max": float(g["spot"].max()),
                    "ondemand": float(g["od"].median()), "band": band, "sps": float(g["sps"].mean()),
                    "azs": int(g["azs"].max())})
    return out


def main() -> int:
    zpath = find_zip(sys.argv[1] if len(sys.argv) > 1 else None)
    types = keep_types()
    outer = zipfile.ZipFile(zpath)
    months = sorted(n for n in outer.namelist() if n.endswith(".zip"))
    daily = []
    for m in months:
        inner = zipfile.ZipFile(io.BytesIO(outer.read(m)))
        for name in sorted(n for n in inner.namelist() if n.endswith(".parquet")):
            date = Path(name).stem.split("-", 1)[1]  # aws-YYYY-MM-DD -> YYYY-MM-DD
            for r in day_rows(pq.read_table(io.BytesIO(inner.read(name))), types):
                daily.append({"date": date, **r})
        print(f"{m}: {len(daily)} pool-days so far", flush=True)
    (LIB / "derived").mkdir(exist_ok=True)
    cols = ["date", "region", "instance_type", "spot_mean", "spot_min", "spot_max", "ondemand", "band", "sps", "azs"]
    with gzip.open(LIB / "derived/spotlake_daily_us.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(daily, key=lambda r: (r["region"], r["instance_type"], r["date"])):
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in cols})

    pools = defaultdict(list)
    for r in daily:
        pools[(r["region"], r["instance_type"])].append(r)
    summary = {}
    for (reg, it), rs in sorted(pools.items()):
        rs.sort(key=lambda r: r["date"])
        bands = [r["band"] for r in rs if r["band"] >= 0]
        prices = np.array([r["spot_mean"] for r in rs])
        ratio = np.array([r["spot_mean"] / r["ondemand"] for r in rs if r["ondemand"] > 0])
        summary[f"{reg}:{it}"] = {
            "region": reg, "instance_type": it, "days": len(rs), "first": rs[0]["date"], "last": rs[-1]["date"],
            "band_share": {str(b): bands.count(b) / len(bands) for b in sorted(set(bands))} if bands else {},
            "band_changes": int(sum(1 for a, b in zip(bands, bands[1:]) if a != b)),
            "band_min": min(bands) if bands else None, "band_max": max(bands) if bands else None,
            "spot_mean": float(prices.mean()), "spot_p5": float(np.quantile(prices, 0.05)),
            "spot_p50": float(np.median(prices)), "spot_p95": float(np.quantile(prices, 0.95)),
            "spot_cv": float(prices.std() / prices.mean()) if prices.mean() > 0 else None,
            "spot_over_ondemand_mean": float(ratio.mean()) if len(ratio) else None,
            "ondemand_last": rs[-1]["ondemand"], "sps_mean": float(np.mean([r["sps"] for r in rs])),
        }
    # exec #4.2: across pools, is a cheaper spot price associated with a worse interruption band?
    xs = [(v["spot_over_ondemand_mean"], sum(int(b) * s for b, s in v["band_share"].items()))
          for v in summary.values() if v["spot_over_ondemand_mean"] and v["band_share"]]
    a, b = np.array(xs).T
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    corr = {"n_pools": len(xs), "pearson_ratio_vs_mean_band": float(np.corrcoef(a, b)[0, 1]),
            "spearman_ratio_vs_mean_band": float(np.corrcoef(rank(a), rank(b))[0, 1]),
            "note": "ratio = spot/on-demand price (lower = cheaper); band 0 = <5% ... 4 = >20% interrupted per month"}
    out = {"source_zip": zpath.name, "months": [Path(m).stem for m in months], "regions": US,
           "n_types": len({k.split(':', 1)[1] for k in summary}), "correlation_exec_4_2": corr, "pools": summary}
    (LIB / "derived/spotlake_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pools": len(summary), "pool_days": len(daily), "corr": corr,
                      "deployment": {k: {kk: v[kk] for kk in ("days", "band_share", "band_changes", "spot_mean",
                                                                "spot_p5", "spot_p95", "spot_over_ondemand_mean")}
                                     for k, v in summary.items() if k.split(":")[1] in DEPLOYMENT_TYPES}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
