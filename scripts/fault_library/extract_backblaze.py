"""Backblaze drive stats 2023-2025 -> per-model failure rates and age hazard curves.

    python scripts/fault_library/extract_backblaze.py            # all quarters not yet aggregated
    python scripts/fault_library/extract_backblaze.py --budget 150   # stop after ~150 s (resumable)
    python scripts/fault_library/extract_backblaze.py --summarize    # parts -> derived files

Input: data/fault_library/raw/backblaze/data_Q<q>_<yyyy>.zip (public Backblaze
"Hard Drive Data and Stats": one CSV per day, one row per drive with
``date, serial_number, model, capacity_bytes, failure, smart_*``). A drive
fails on the last day it appears, with ``failure = 1``; Backblaze counts a
failure when the drive is removed for failing, so a failure is a permanent
loss of the drive's data (recoverability known).

Age: SMART 9 (power-on hours) / 24 days, binned in quarters of a year; drives
without SMART 9 count toward the model's rate but not its age curve.

Step 1 (per quarter, resumable): drive-days and failures per
(model, capacity, age quarter) -> raw/backblaze/parts/<quarter>.csv (not in git).
Step 2 (--summarize): derived/backblaze_model_age.csv.gz (the aggregate, in git)
and derived/backblaze_summary.json (per-model rate with a Poisson interval,
yearly age hazard, age distribution of drive-days).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import math
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "data/fault_library"
RAW = LIB / "raw/backblaze"
PARTS = RAW / "parts"
QUARTER_D = 365.25 / 4
MAX_BIN = 48  # 12 years
SSD_RE = re.compile(r"SSD|^CT\d|^MZ|SAMSUNG MZ|^Micron|^SSDSC|^WDS|^DELLBOSS|^MTFD|^SA510|^KINGSTON", re.I)
COLS = ["serial_number", "model", "capacity_bytes", "failure", "smart_9_raw"]


def is_ssd(model: str) -> bool:
    return bool(SSD_RE.search(model))


def aggregate_zip(path: Path) -> list[dict]:
    z = zipfile.ZipFile(path)
    names = sorted(n for n in z.namelist() if n.endswith(".csv") and "MACOSX" not in n)
    acc: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    opts = pcsv.ConvertOptions(include_columns=COLS, column_types={"smart_9_raw": pa.float64(),
                                                                  "capacity_bytes": pa.int64(), "failure": pa.int64()})
    for n in names:
        t = pcsv.read_csv(io.BytesIO(z.read(n)), convert_options=opts)
        age_d = pc.divide(t["smart_9_raw"], 24.0)
        b = pc.cast(pc.floor(pc.divide(age_d, QUARTER_D)), pa.int64())
        b = pc.if_else(pc.greater(b, MAX_BIN), MAX_BIN, b)
        b = pc.if_else(pc.less(b, 0), -1, b)
        b = pc.fill_null(b, -1)
        t2 = pa.table({"model": pc.utf8_trim_whitespace(t["model"]), "cap": pc.fill_null(t["capacity_bytes"], -1),
                       "bin": b, "failure": pc.fill_null(t["failure"], 0)})
        g = t2.group_by(["model", "cap", "bin"]).aggregate([("failure", "count"), ("failure", "sum")])
        for m, cap, bn, cnt, fail in zip(*(g[c].to_pylist() for c in ("model", "cap", "bin", "failure_count", "failure_sum"))):
            a = acc[(m, cap, bn)]
            a[0] += cnt
            a[1] += fail
    return [{"model": m, "capacity_bytes": cap, "age_q": bn, "drive_days": a[0], "failures": a[1]}
            for (m, cap, bn), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))]


def poisson_ci(k: int, exposure: float, conf: float = 0.90) -> tuple[float, float]:
    """Exact-ish Poisson interval via the Wilson-Hilferty chi-square approximation (NumPy only)."""
    z = 1.6448536269514722 if abs(conf - 0.90) < 1e-9 else 1.959963984540054

    def chi2_q(p_z: float, dof: float) -> float:
        return dof * (1 - 2 / (9 * dof) + p_z * math.sqrt(2 / (9 * dof))) ** 3

    lo = 0.0 if k == 0 else chi2_q(-z, 2 * k) / 2
    hi = chi2_q(z, 2 * (k + 1)) / 2
    return lo / exposure, hi / exposure


def summarize() -> dict:
    rows = []
    for f in sorted(PARTS.glob("*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append((r["model"], int(r["capacity_bytes"]), int(r["age_q"]), int(r["drive_days"]), int(r["failures"]),
                             f.stem))
    quarters = sorted({r[5] for r in rows})
    agg: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for m, cap, bn, dd, fl, _ in rows:
        a = agg[(m, bn)]
        a[0] += dd
        a[1] += fl
    with open(LIB / "derived/backblaze_model_age.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, io.TextIOWrapper(gz, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "ssd", "age_quarter", "drive_days", "failures"])
        for (m, bn), (dd, fl) in sorted(agg.items()):
            w.writerow([m, int(is_ssd(m)), bn, dd, fl])
    models: dict[str, dict] = {}
    by_model: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    caps: dict[str, set] = defaultdict(set)
    for m, cap, bn, dd, fl, _ in rows:
        by_model[m][bn][0] += dd
        by_model[m][bn][1] += fl
        if cap > 0:
            caps[m].add(cap)
    for m, bins in sorted(by_model.items()):
        dd = sum(v[0] for v in bins.values())
        fl = sum(v[1] for v in bins.values())
        lo, hi = poisson_ci(fl, dd)
        yearly = defaultdict(lambda: [0, 0])  # age in whole years
        for bn, (d, f) in bins.items():
            if bn >= 0:
                y = bn // 4
                yearly[y][0] += d
                yearly[y][1] += f
        models[m] = {
            "ssd": is_ssd(m), "capacity_bytes": max(caps[m]) if caps[m] else None,
            "drive_days": dd, "drive_years": dd / 365.25, "failures": fl,
            "afr": fl / dd * 365.25 if dd else None, "afr_90ci": [lo * 365.25, hi * 365.25],
            "rate_per_hour": fl / (dd * 24.0) if dd else None,
            "age_known_share": 1 - bins.get(-1, [0, 0])[0] / dd if dd else 0.0,
            "age_years": {str(y): {"drive_days": d, "failures": f} for y, (d, f) in sorted(yearly.items())},
        }
    out = {"quarters": quarters, "n_models": len(models),
           "fleet": {k: {"drive_days": sum(v["drive_days"] for v in models.values() if v["ssd"] == (k == "ssd")),
                         "failures": sum(v["failures"] for v in models.values() if v["ssd"] == (k == "ssd"))}
                     for k in ("hdd", "ssd")},
           "models": models}
    for v in out["fleet"].values():
        v["afr"] = v["failures"] / v["drive_days"] * 365.25 if v["drive_days"] else None
    (LIB / "derived/backblaze_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9, help="seconds; stop before starting a quarter past it")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        s = summarize()
        print(json.dumps({"quarters": s["quarters"], "n_models": s["n_models"], "fleet": s["fleet"]}, indent=1))
        return 0
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for z in sorted(glob.glob(str(RAW / "data_Q*_*.zip"))):
        zp = Path(z)
        if not Path(z + ".done").exists():
            continue  # still downloading
        q = zp.stem.replace("data_", "")
        out = PARTS / f"{q}.csv"
        if out.exists():
            continue
        if time.time() - t0 > args.budget:
            print("budget reached", flush=True)
            return 0
        rows = aggregate_zip(zp)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "capacity_bytes", "age_q", "drive_days", "failures"])
            w.writeheader()
            w.writerows(rows)
        tmp.rename(out)
        print(f"{q}: {len(rows)} rows, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
