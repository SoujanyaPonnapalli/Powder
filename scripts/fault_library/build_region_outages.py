"""Build AWS-region outage profiles from Azure Post Incident Reviews (subtask 1.2).

    python scripts/fault_library/build_region_outages.py

Decision 2026-10-20: region outages come from Azure only; each AWS US region
inherits the outage process of one Azure region (powder.faults.regions).

Inputs:  data/fault_library/region_incidents_azure.csv
         (compiled 2026-10-21 from public PIR text and press via web search)
Outputs: data/fault_library/derived/region_map.json
         data/fault_library/sources/azure-pir-incidents.json
         data/fault_library/profiles/region_outage/aws-us-from-azure.json

Method (plan §7.8, with two recorded deviations):
* Window 2023-01-01 .. 2026-10-20. Earlier years are excluded: PIRs are kept
  about 5 years and, since mid-2022, published only for broad incidents, so
  the 2021-22 search found no confirmable VM outages.
* Count an incident for an Azure region if VMs there lost connectivity
  region-wide or across >= 2 AZs (scope region_wide / multi_az, compute
  affected, confidence high/medium). "global" incidents count for every
  region and are tagged multi_region. Single-AZ datacenter events go into the
  `-incl-single-az` variant (and exec #3).
* Rates: empirical-Bayes Gamma-Poisson over the nine Azure **US** regions
  (deviation: the plan said all regions worldwide, but the incident search was
  thorough only for US regions; a worldwide hyperprior would be biased low).
  central = posterior mean, low = 5th pct, high = 2 x 95th pct (explicit
  under-reporting allowance: public PIRs miss smaller incidents).
* Durations: pooled worldwide region-level VM outages in the window, using the
  time until the vast majority of VMs recovered where the PIR gives it. MLE
  exponential / lognormal / Weibull, AIC choice, parametric-bootstrap KS.
  low/high rescale the chosen fit to the 5th/95th percentile of the bootstrap
  mean.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.dists import to_spec  # noqa: E402
from powder.faults.fit import gamma_poisson_eb, mle_duration_fits, parametric_bootstrap_ks  # noqa: E402
from powder.faults.regions import AZURE_US_REGIONS, region_map  # noqa: E402
from powder.simulation.distributions import Exponential, LogNormal, Weibull  # noqa: E402

LIB = ROOT / "data/fault_library"
CSV = LIB / "region_incidents_azure.csv"
WINDOW = (datetime(2023, 1, 1), datetime(2026, 10, 20))
YEARS = (WINDOW[1] - WINDOW[0]).days / 365.25
H_PER_YEAR = 8766.0
STATUS = {"high": "V", "medium": "S", "low": "U"}
US = list(AZURE_US_REGIONS)


def _date(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "")[:16]) if "T" in s else datetime.fromisoformat(s)


def load():
    rows = list(csv.DictReader(CSV.open()))
    for r in rows:
        r["regions"] = [x.strip() for x in r["azure_regions"].split(";")]
        r["start"] = _date(r["start_utc"])
        r["in_window"] = WINDOW[0] <= r["start"] < WINDOW[1]
        r["compute"] = r["compute_affected"] == "true"
        r["usable"] = r["confidence"] in ("high", "medium")
        d = r["compute_duration_minutes"] or r["duration_minutes"]
        r["dur_min"] = float(d) if d else None
    return rows


def counts_for(rows, scopes):
    sel = [r for r in rows if r["in_window"] and r["compute"] and r["usable"] and r["scope"] in scopes]
    n = {reg: 0 for reg in US}
    used = {reg: [] for reg in US}
    for r in sel:
        targets = US if "global" in r["regions"] else [x for x in r["regions"] if x in n]
        for reg in targets:
            n[reg] += 1
            used[reg].append(r["incident_id"])
    return n, used, sel


def duration_model(events):
    x = np.array([r["dur_min"] * 60.0 for r in events if r["dur_min"]])
    fits = mle_duration_fits(x)
    best = min(fits, key=lambda f: f.aic)
    pval = parametric_bootstrap_ks(best, x, reps=1000, seed=1)
    rng = np.random.default_rng(2)
    boot_means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(4000)])
    lo_m, hi_m = np.quantile(boot_means, [0.05, 0.95])

    def rescaled(target_mean):
        d = best.dist
        if isinstance(d, LogNormal):
            return LogNormal(d.mu + math.log(target_mean / d.mean), d.sigma)
        if isinstance(d, Weibull):
            return Weibull(d.shape, d.scale * target_mean / d.mean)
        return Exponential(1.0 / target_mean)

    triple = {"low": to_spec(rescaled(lo_m)), "central": to_spec(best.dist), "high": to_spec(rescaled(hi_m))}
    info = {"n": int(len(x)), "values_s": sorted(float(v) for v in x), "best": best.family,
            "aic": {f.family: f.aic for f in fits}, "ks_bootstrap_p": pval,
            "mean_s": float(x.mean()), "median_s": float(np.median(x)),
            "bootstrap_mean_5_95_s": [float(lo_m), float(hi_m)]}
    return triple, info


def rates(n):
    eb = gamma_poisson_eb([n[r] for r in US], [YEARS] * len(US))
    out = {}
    for reg in US:
        s = eb.summary(n[reg], YEARS)
        out[reg] = {"low": s["low"] / H_PER_YEAR, "central": s["mean"] / H_PER_YEAR,
                    "high": 2.0 * s["high"] / H_PER_YEAR, "per_year": s}
    return out, eb


def r6(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r6(v) for v in x]
    return x


def main():
    rows = load()
    rmap = region_map()
    (LIB / "derived").mkdir(exist_ok=True)
    (LIB / "derived/region_map.json").write_text(json.dumps(
        {"note": "AWS metro -> Azure region; Azure coordinates from published region metadata "
                 "(refresh: scripts/fault_library/azure_locations.py). One Azure region per AWS region.",
         "mapping": rmap}, indent=2) + "\n")

    meas = []
    for r in rows:
        meas.append({"id": r["incident_id"], "metric_verbatim": r["cause"], "value": {
            "azure_regions": r["regions"], "start_utc": r["start_utc"], "end_utc": r["end_utc"] or None,
            "duration_minutes": float(r["duration_minutes"]) if r["duration_minutes"] else None,
            "compute_duration_minutes": float(r["compute_duration_minutes"]) if r["compute_duration_minutes"] else None,
            "scope": r["scope"], "compute_affected": r["compute_affected"] or None,
            "tracking_id": r["tracking_id"] or None},
            "unit": "incident", "status": STATUS[r["confidence"]], "location": r["source_url"],
            "covers": ["region"], "notes": r["notes"] or None})
    src = {"id": "azure-pir-incidents", "kind": "source",
           "title": "Azure Post Incident Reviews and press reports, 2021-2026 (compiled)",
           "url": "https://azure.status.microsoft/en-us/status/history/",
           "population": "Azure public regions; broad incidents with public PIRs",
           "accessed": "2026-10-21",
           "notes": ("Compiled by web search (search snippets of PIR pages and press; the PIR pages themselves "
                     "were not fetchable from the sandbox). Status V = PIR text seen in snippet, S = reliable "
                     "secondary report, U = weak single source (never used). Coverage is good for broad "
                     "2023-2026 incidents and US regions, poor for 2021-2022 and for single-region events "
                     "outside the US. Refine with scripts on a machine with network access."),
           "measurements": meas}
    (LIB / "sources/azure-pir-incidents.json").write_text(json.dumps(src, indent=2) + "\n")

    variants = {"": ({"region_wide", "multi_az"}, "default"),
                "-incl-single-az": ({"region_wide", "multi_az", "single_az", "unknown"},
                                    "pessimistic: single-datacenter / single-AZ VM outages and unknown-scope "
                                    "events also count as region outages")}
    profiles = []
    report = {}
    region_events = [r for r in rows if r["in_window"] and r["compute"] and r["usable"]
                     and r["scope"] in ("region_wide", "multi_az")]
    dur_triple, dur_info = duration_model(region_events)
    for suffix, (scopes, label) in variants.items():
        n, used, sel = counts_for(rows, scopes)
        rr, eb = rates(n)
        regions = {}
        for aws, m in rmap.items():
            az = m["azure_region"]
            regions[aws] = {"azure_region": az, "distance_km": m["distance_km"], "rule": m["rule"],
                            "rate_per_hour": {k: rr[az][k] for k in ("low", "central", "high")},
                            "n_incidents": n[az], "exposure_years": YEARS,
                            "incidents": used[az]}
        cited = sorted({iid for reg in US for iid in used[reg]} |
                       {r["incident_id"] for r in region_events if r["dur_min"]})
        profiles.append({
            "id": f"region_outage/aws-us-from-azure{suffix}", "kind": "region_outage", "version": 1,
            "tags": ["azure-pir", label.split(":")[0]],
            "derived_from": [f"azure-pir-incidents#{i}" for i in cited],
            "built_by": "scripts/fault_library/build_region_outages.py",
            "window": [WINDOW[0].date().isoformat(), WINDOW[1].date().isoformat()],
            "counting_rule": label + "; scopes counted: " + ", ".join(sorted(scopes)),
            "rate_model": {"method": "empirical-Bayes Gamma-Poisson over Azure US regions",
                           "alpha": eb.alpha, "beta_years": eb.beta, "heterogeneous": eb.heterogeneous,
                           "pooled_per_year": eb.pooled_rate,
                           "azure_counts": n, "high_rule": "2 x posterior 95th percentile (under-reporting)"},
            "regions": regions,
            "duration": dur_triple, "duration_fit": dur_info,
            "multi_region_incidents": [r["incident_id"] for r in sel if "global" in r["regions"]
                                       or sum(x in US for x in r["regions"]) > 1],
            "notes": ("Modelling assumption: an AWS region fails like its mapped Azure region. Rates are per "
                      "region; multi_region_incidents lists events that hit every region at once (input for "
                      "exec #3 correlated failures).")})
        report[suffix or "default"] = {"azure_counts": n, "eb": eb.__dict__,
                                       "aws_rates_per_year": {a: {k: v * H_PER_YEAR for k, v in d["rate_per_hour"].items()}
                                                              for a, d in regions.items()}}
    (LIB / "profiles/region_outage/aws-us-from-azure.json").write_text(json.dumps(r6(profiles), indent=2) + "\n")
    print(json.dumps(r6({"window_years": YEARS, "duration": dur_info, "variants": report, "map": rmap}), indent=1))


if __name__ == "__main__":
    main()
