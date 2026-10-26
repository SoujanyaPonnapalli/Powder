"""Schroeder & Gibson FAST'07 -> disk replacement-rate population and age curves (exec #1b step 4).

    python scripts/fault_library/build_sg07.py              # from the committed digitization
    python scripts/fault_library/build_sg07.py --digitize   # re-read the PDF (needs pdfplumber)

Source: docs/reading/Schroeder_Gibson_FAST07.pdf ("Disk failures in the real
world", FAST'07). A replacement means the drive was replaced as faulty, so the
event loses the drive's data (recoverability known).

* Table 1 (text): ARR per data-set row with its disk count -> component
  ``disk/sg07-systems`` (population; weight = disk count). The 1998 COM3 row
  (24.1%, past the 5-year nominal lifetime; excluded from the paper's Fig 1)
  is left out.
* Fig 3 (vector bars): yearly ARR by age for HPC1 compute nodes, HPC1
  file-system nodes (years 1-5) and the first HPC4 drive type (ages 1-3) ->
  component ``disk/sg07-age`` (curve population, age clock). Axis from the
  tick marks (0..8 %); where two bars overlap at one x, the visible (taller)
  one is used. The file-system nodes had no replacements in year 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "data/fault_library"
PDF = ROOT / "docs/reading/Schroeder_Gibson_FAST07.pdf"
DIG = LIB / "digitized/sg07"
H_YEAR = 8766.0

# Table 1 (text, p.2): data set, drive parameters, disk count, ARR (%)
TABLE1 = [
    ("HPC1-a", "18GB 10K SCSI", 2318, 4.0), ("HPC1-b", "36GB 10K SCSI", 1088, 2.2),
    ("HPC2", "36GB 10K SCSI", 520, 1.1), ("HPC3-a", "146GB 15K SCSI", 3064, 3.7),
    ("HPC3-b", "73GB 15K SCSI", 144, 3.0), ("HPC3-c", "250GB 7.2K SATA", 11000, 3.3),
    ("HPC4-a", "250GB SATA", 8430, 2.2), ("HPC4-b", "500GB SATA", 2030, 0.5),
    ("HPC4-c", "400GB SATA", 3158, 0.8), ("COM1", "10K SCSI", 26734, 2.8),
    ("COM2", "15K SCSI", 39039, 3.1), ("COM3-a", "10K FC", 56, 3.6),
    ("COM3-b", "10K FC", 2450, 5.4), ("COM3-c", "10K FC", 796, 13.6),
]


def digitize() -> list[dict]:
    import pdfplumber
    page = pdfplumber.open(PDF).pages[8]
    words = page.extract_words()
    cap = next(w for w in words if w["text"].startswith("Figure3"))["top"]
    ticks = sorted({round(l["top"], 2) for l in page.lines
                    if l["top"] < cap and abs(l["top"] - l["bottom"]) < 0.1 and (l["x1"] - l["x0"]) < 4})
    y0, y8 = max(ticks), min(ticks)  # 0 % and 8 %
    per_pct = (y0 - y8) / 8.0
    panels = {"hpc1-compute": (90, 215, [105.2, 128.2, 151.1, 174.0, 196.9]),
              "hpc1-filesystem": (250, 380, [266.2, 289.1, 312.0, 334.9, 357.9]),
              "hpc4-type1": (410, 545, [434.7, 472.9, 511.1])}
    rows = []
    for name, (xa, xb, labels) in panels.items():
        bars = [r for r in page.rects if r["fill"] and r["top"] < cap and xa < r["x0"] < xb
                and r.get("non_stroking_color") == (1.0, 1.0, 0.0)]
        for i, lx in enumerate(labels):
            near = [r for r in bars if abs((r["x0"] + r["x1"]) / 2 - (lx + 2.5)) < 8]
            arr = max(((y0 - r["top"]) / per_pct for r in near), default=0.0)
            rows.append({"series": name, "age_year": i + 1, "arr_pct": round(arr, 3), "n_rects": len(near)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digitize", action="store_true")
    args = ap.parse_args()
    DIG.mkdir(parents=True, exist_ok=True)
    path = DIG / "fig3_yearly.csv"
    if args.digitize or not path.exists():
        rows = digitize()
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["series", "age_year", "arr_pct", "n_rects"])
            w.writeheader()
            w.writerows(rows)
    rows = list(csv.DictReader(open(path)))
    rate = lambda arr_pct: -__import__("math").log1p(-arr_pct / 100.0) / H_YEAR  # ARR (fraction/yr) -> hazard/h

    src = {"id": "sg07", "kind": "source", "title": "Disk failures in the real world: What does an MTTF of 1,000,000 hours mean to you? (FAST'07)",
           "authors": "B. Schroeder, G. A. Gibson", "url": "https://www.usenix.org/legacy/events/fast07/tech/schroeder/schroeder.pdf",
           "accessed": "2026-10-26", "population": "~100k drives in 7 data sets (HPC sites and internet services), 2001-2006",
           "measurements": [
               {"id": "table1-arr", "metric_verbatim": "Table 1: ARR (%) and disk count per data-set row", "value": [
                   {"row": r, "drive": d, "disks": n, "arr_pct": a} for r, d, n, a in TABLE1],
                "unit": "% per year", "status": "V", "location": "p.2 Table 1", "covers": ["disk"],
                "conversion": "hazard = -ln(1 - ARR) / 8766 h"},
               {"id": "fig3-yearly", "metric_verbatim": "Fig 3: ARR by year of operation (HPC1 compute, HPC1 file system) and by drive age (HPC4 type 1)",
                "value": "digitized/sg07/fig3_yearly.csv", "unit": "% per year", "status": "V", "location": "p.9 Fig 3",
                "covers": ["disk"], "conversion": "vector bar heights, axis from tick marks (0-8 %)"}]}
    (LIB / "sources/sg07.json").write_text(json.dumps(src, indent=2) + "\n")

    members = [{"id": f"sg07:{r}", "rate_per_hour": rate(a), "weight": float(n), "drive": d, "arr_pct": a}
               for r, d, n, a in TABLE1]
    tot = sum(m["weight"] for m in members)
    mean = sum(m["rate_per_hour"] * m["weight"] for m in members) / tot
    rs = sorted(m["rate_per_hour"] for m in members)
    common = {"kind": "component", "component": "disk", "version": 1, "built_by": "scripts/fault_library/build_sg07.py"}
    systems = {**common, "id": "disk/sg07-systems", "derived_from": ["sg07#table1-arr"],
               "rate_per_hour": {"low": rs[0], "central": mean, "high": rs[-1]},
               "rate_model": {"type": "population", "members": members},
               "notes": ("Schroeder & Gibson FAST'07 Table 1: annual replacement rate per data-set row (HPC and internet-"
                         "service drives, 2001-2006), weight = disk count; the 1998 COM3 drives past their nominal "
                         "lifetime are excluded. Triple: min / disk-weighted mean / max.")}
    curves = []
    for series in ("hpc1-compute", "hpc1-filesystem", "hpc4-type1"):
        pts = sorted((int(r["age_year"]), float(r["arr_pct"])) for r in rows if r["series"] == series)
        n = len(pts)
        curves.append({"id": f"sg07:{series}", "weight": 1.0, "edges_h": [i * H_YEAR for i in range(n + 1)],
                       "rate_per_hour": [rate(a) for _, a in pts],
                       "start_h": {"type": "weights", "edges_h": [i * H_YEAR for i in range(n + 1)], "weights": [1.0] * n}})
    allr = [r for c in curves for r in c["rate_per_hour"]]
    age = {**common, "id": "disk/sg07-age", "derived_from": ["sg07#fig3-yearly"],
           "rate_per_hour": {"low": min(allr), "central": sum(allr) / len(allr), "high": max(allr)},
           "rate_model": {"type": "curve_population", "clock": "age", "periodic": False, "members": curves},
           "notes": ("Schroeder & Gibson FAST'07 Fig 3: yearly replacement rate by age (rising from year 1: early "
                     "wear-out, no infant-mortality dip). Age at t = 0 uniform over the observed years; past the last "
                     "year the last rate continues.")}
    (LIB / "profiles/component/sg07.json").write_text(json.dumps([systems, age], indent=1) + "\n")
    print(json.dumps({"rows": len(members), "curves": {c["id"]: [round(r * H_YEAR * 100, 2) for r in c["rate_per_hour"]]
                                                       for c in curves}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
