"""Attach rate models (population / curve / curve_population) to library entries (exec #1b §1-§2).

    python scripts/fault_library/build_rate_models.py

Run after build_ford10.py, build_literature.py and build_prices.py (it edits
their outputs in place, idempotently) and before build_machines.py.

* Ford et al. (hw_reboot = unplanned reboots, sw = >=15-min node restarts):
  population of cell-level rates (lognormal: mean = Table 1 average, sigma from
  the Table 1 max taken as the 1 - 1/(n+1) quantile of n = 30 cells, "tens of
  cells") x a relative calendar shape (Fig 3 daily series, one cell, mean 1,
  repeating every 91 days, uniform offset per machine). Applied to the Ford
  profiles and to cloud/aws-ondemand-nvme (draws capped at the profile's
  SLA-clipped high end).
* Backblaze drive stats 2023-2025: component ``disk/backblaze-models``
  (population of per-model constant rates) and ``disk/backblaze-models-age``
  (one age hazard curve per model; machine age drawn from that model's
  drive-day age distribution).
* SpotLake 2025-26: component ``spot/spotlake-pools`` (one calendar curve per
  US 8 vCPU x86 pool: daily band -> band-midpoint rate; offset shared by every
  machine of the pool), and a calendar curve on each ``spot/aws-*`` profile.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.aws_prices import band_rates  # noqa: E402
from powder.simulation.distributions import normal_ppf  # noqa: E402

LIB = ROOT / "data/fault_library"
H_YEAR = 8766.0
N_CELLS = 30            # Ford: "data is gathered from tens of Google storage cells"
MIN_DRIVE_YEARS = 500   # Backblaze model kept as a population member
MIN_BIN_DRIVE_YEARS = 200
MIN_AGE_KNOWN = 0.9


def r8(x):
    if isinstance(x, float):
        return float(f"{x:.8g}")
    if isinstance(x, dict):
        return {k: r8(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r8(v) for v in x]
    return x


def lognormal_sigma_from_mean_max(mean: float, mx: float, n: int) -> tuple[float, str]:
    """sigma such that the 1 - 1/(n+1) quantile equals mx for a lognormal with this mean.

    max/mean = exp(z s - s^2/2) peaks at s = z; if the ratio is larger than
    exp(z^2/2) no lognormal fits and s = z (the widest such population) is used.
    """
    z = normal_ppf(1 - 1 / (n + 1))
    ratio = math.log(mx / mean)
    disc = z * z - 2 * ratio
    if disc < 0:
        return z, f"max/mean {mx / mean:.3g} exceeds what n={n} lognormal draws reach; sigma = z = {z:.3g} (lower bound on spread)"
    return z - math.sqrt(disc), f"Table 1 max as the {1 - 1 / (n + 1):.3f} quantile of n={n} cells"


def ford_models() -> dict[str, dict]:
    ford = {p["id"]: p for p in json.loads((LIB / "profiles/node/ford10.json").read_text())}
    ge15 = ford["cluster/ford10-google-storage-ge15min"]["causes"]
    days = list(csv.DictReader(open(LIB / "digitized/ford10/fig3_daily.csv")))
    out = {}
    for cause, col in (("hw_reboot", "unplanned_reboot"), ("sw", "node_restart")):
        r = ge15[cause]["rate_per_hour"]
        sigma, how = lognormal_sigma_from_mean_max(r["central"], r["high"], N_CELLS)
        series = np.array([float(d[col]) for d in days])
        shape = series / series.mean()
        out[cause] = {"sigma": sigma, "sigma_basis": how,
                      "curve": {"clock": "calendar", "edges_h": [24.0 * i for i in range(len(shape) + 1)],
                                "rate_per_hour": [float(x) for x in shape], "periodic": True,
                                "scale_with": "level", "start_h": {"type": "uniform"},
                                "source": "ford10#fig3-daily",
                                "notes": "relative daily shape of one cell (mean 1); bursts are cell-wide in the data"}}
    return out


def attach_ford(profile: dict, fm: dict, cap_high: bool) -> None:
    for cause, m in fm.items():
        c = profile["causes"].get(cause)
        if not c:
            continue
        mean = c["rate_per_hour"]["central"]
        if mean <= 0:
            continue
        model = {"type": "population", "lognormal": {"mean": mean, "sigma": m["sigma"]},
                 "sigma_basis": m["sigma_basis"], "curve": m["curve"], "source": "ford10#table1 + ford10#fig3-daily"}
        if cap_high:
            model["max_per_hour"] = c["rate_per_hour"]["high"]
        c["rate_model"] = model


def merge_bins(years: dict[int, tuple[float, int]]) -> list[tuple[int, int, float, int]]:
    """Merge consecutive age years until each bin has >= MIN_BIN_DRIVE_YEARS: (y0, y1, drive_days, failures)."""
    out, cur = [], None
    for y in sorted(years):
        d, f = years[y]
        if cur is None:
            cur = [y, y + 1, d, f]
        else:
            cur[1], cur[2], cur[3] = y + 1, cur[2] + d, cur[3] + f
        if cur[2] / 365.25 >= MIN_BIN_DRIVE_YEARS:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        if out:
            y0, _, d, f = out.pop()
            out.append((y0, cur[1], d + cur[2], f + cur[3]))
        else:
            out.append(tuple(cur))
    return out


def backblaze_components() -> list[dict]:
    path = LIB / "derived/backblaze_summary.json"
    if not path.exists():
        return []
    s = json.loads(path.read_text())
    members, curves = [], []
    for model, v in sorted(s["models"].items()):
        if v["drive_years"] < MIN_DRIVE_YEARS:
            continue
        k = v["failures"]
        rate = (k if k > 0 else 0.5) / (v["drive_days"] * 24.0)
        members.append({"id": model, "rate_per_hour": rate, "weight": v["drive_years"], "ssd": v["ssd"],
                        "failures": k, "drive_years": v["drive_years"], "afr_90ci": v["afr_90ci"]})
        if v["age_known_share"] < MIN_AGE_KNOWN:
            continue
        years = {int(y): (a["drive_days"], a["failures"]) for y, a in v["age_years"].items()}
        bins = merge_bins(years)
        edges = [0.0] + [b[1] * H_YEAR for b in bins]
        rates = [(b[3] if b[3] > 0 else 0.5) / (b[2] * 24.0) for b in bins]
        ys = sorted(years)
        curves.append({"id": model, "weight": v["drive_years"], "ssd": v["ssd"], "edges_h": edges, "rate_per_hour": rates,
                       "start_h": {"type": "weights", "edges_h": [y * H_YEAR for y in ys] + [(ys[-1] + 1) * H_YEAR],
                                   "weights": [years[y][0] for y in ys]},
                       "observed_age_years": [ys[0], ys[-1] + 1]})
    rates = np.array([m["rate_per_hour"] for m in members])
    w = np.array([m["weight"] for m in members])
    order = np.argsort(rates)
    cw = np.cumsum(w[order]) / w.sum()
    q = lambda p: float(rates[order][np.searchsorted(cw, p)])  # noqa: E731
    fleet = sum(m["failures"] for m in members) / sum(m["drive_years"] * 365.25 * 24 for m in members)
    triple = {"low": min(q(0.05), fleet), "central": fleet, "high": max(q(0.95), fleet)}
    common = {"kind": "component", "component": "disk", "version": 1, "built_by": "scripts/fault_library/build_rate_models.py",
              "derived_from": ["backblaze-raw-2023-2025#models"], "rate_per_hour": triple}
    note = (f"Backblaze drive stats {s['quarters'][0]}..{s['quarters'][-1]} (raw daily records): models with >= "
            f"{MIN_DRIVE_YEARS} drive-years. Data-centre HDDs and boot SSDs, a proxy for instance-store NVMe. "
            "A failure = the drive was removed as failed (data lost). Zero-failure counts use 0.5 events.")
    pop = {**common, "id": "disk/backblaze-models", "rate_model": {"type": "population", "members": members},
           "notes": note + " Triple: drive-year-weighted 5th percentile / fleet rate / 95th percentile."}
    age = {**common, "id": "disk/backblaze-models-age",
           "rate_model": {"type": "curve_population", "clock": "age", "periodic": False, "members": curves},
           "notes": note + (f" Age hazard per model from SMART 9 power-on hours (models with >= {MIN_AGE_KNOWN:.0%} "
                            f"of drive-days with an age), yearly bins merged to >= {MIN_BIN_DRIVE_YEARS} drive-years; "
                            "the first bin extends back to age 0 and the last one onward. A machine's age at t = 0 "
                            "is drawn from the model's drive-day age distribution.")}
    return [pop, age]


def spot_curves() -> tuple[dict | None, dict[str, dict]]:
    path = LIB / "derived/spotlake_daily_us.csv.gz"
    summ_path = LIB / "derived/spotlake_summary.json"
    if not path.exists():
        return None, {}
    summ = json.loads(summ_path.read_text())["pools"]
    series = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            series[f"{r['region']}:{r['instance_type']}"].append((r["date"], int(r["band"]), float(r["spot_mean"])))
    curves = {}
    for key, rows in sorted(series.items()):
        rows.sort()
        known = [band_rates(b)["central"] for _, b, _ in rows if b >= 0]
        if not known:
            continue
        fill = float(np.mean(known))
        rates = [band_rates(b)["central"] if b >= 0 else fill for _, b, _ in rows]
        curves[key] = {"id": key, "weight": 1.0, "edges_h": [24.0 * i for i in range(len(rates) + 1)],
                       "rate_per_hour": rates, "first": rows[0][0], "last": rows[-1][0],
                       "spot_price_p50": summ.get(key, {}).get("spot_p50"),
                       "ondemand_price": summ.get(key, {}).get("ondemand_last")}
    means = np.array([np.mean(c["rate_per_hour"]) for c in curves.values()])
    comp = {"id": "spot/spotlake-pools", "kind": "component", "component": "spot_interrupt", "version": 1,
            "built_by": "scripts/fault_library/build_rate_models.py", "derived_from": ["spotlake-aws#pools"],
            "rate_per_hour": {"low": float(np.quantile(means, 0.05)), "central": float(means.mean()),
                              "high": float(np.quantile(means, 0.95))},
            "rate_model": {"type": "curve_population", "clock": "calendar", "periodic": True,
                           "start_h": {"type": "uniform"}, "shared_by": "pool", "members": list(curves.values())},
            "notes": ("SpotLake daily Spot Advisor band per US 8 vCPU x86 pool -> band-midpoint interruption rate "
                      "(plan §7.3 reading); unknown days take the pool's mean. The series repeats yearly; every "
                      "machine of a pool shares one offset (pool-wide reclaim waves). Spot reclaim loses instance "
                      "store (permanent).")}
    return comp, curves


def main() -> int:
    fm = ford_models()
    ford_path = LIB / "profiles/node/ford10.json"
    ford = json.loads(ford_path.read_text())
    for p in ford:
        attach_ford(p, fm, cap_high=False)
    ford_path.write_text(json.dumps(r8(ford), indent=2) + "\n")

    aws_path = LIB / "profiles/node/aws-ondemand.json"
    aws = json.loads(aws_path.read_text())
    for p in aws:
        attach_ford(p, fm, cap_high=True)
    aws_path.write_text(json.dumps(r8(aws), indent=2) + "\n")

    comps = backblaze_components()
    spot_comp, curves = spot_curves()
    spot_path = LIB / "profiles/node/aws-spot.json"
    if spot_path.exists():
        spot = json.loads(spot_path.read_text())
        for p in spot:
            attach_ford(p, fm, cap_high=True)
            m = re.match(r"spot/aws-(.+?)-(us-[a-z]+-\d)$", p["id"])
            key = f"{m.group(2)}:{m.group(1)}" if m else None
            if key and key in curves:
                c = curves[key]
                p["causes"]["spot_interrupt"]["rate_model"] = {
                    "type": "curve", "clock": "calendar", "periodic": True, "start_h": {"type": "uniform"},
                    "shared_by": "pool", "edges_h": c["edges_h"], "rate_per_hour": c["rate_per_hour"],
                    "source": "spotlake-aws#pools"}
        spot_path.write_text(json.dumps(r8(spot), indent=2) + "\n")
    if spot_comp:
        comps.append(spot_comp)
    (LIB / "profiles/component/rate-models.json").write_text(json.dumps(r8(comps), indent=1) + "\n")

    src = {"id": "backblaze-raw-2023-2025", "kind": "source",
           "title": "Backblaze Hard Drive Data and Stats, raw daily drive records 2023-Q1 .. 2025-Q4",
           "url": "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data", "accessed": "2026-10-26",
           "population": "Backblaze data-centre drives (HDD data drives and SSD boot drives), one row per drive per day",
           "notes": ("Downloaded from f001.backblazeb2.com (12 quarterly zips, ~12 GB, not in git); aggregated by "
                     "scripts/fault_library/extract_backblaze.py into derived/backblaze_model_age.csv.gz and "
                     "derived/backblaze_summary.json. failure = 1 on the drive's last day when it was removed as failed."),
           "measurements": [{"id": "models", "metric_verbatim": "per-model drive-days and failures by age quarter (SMART 9)",
                             "value": "derived/backblaze_summary.json", "unit": "drive-days, failures", "status": "V",
                             "location": "derived/backblaze_model_age.csv.gz", "covers": ["disk"]}]}
    if (LIB / "derived/backblaze_summary.json").exists():
        (LIB / "sources/backblaze-raw.json").write_text(json.dumps(src, indent=2) + "\n")
    print(json.dumps({"ford_sigma": {c: m["sigma"] for c, m in fm.items()},
                      "backblaze_members": [len(c["rate_model"]["members"]) for c in comps if c["component"] == "disk"],
                      "spot_pools": len(curves)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
