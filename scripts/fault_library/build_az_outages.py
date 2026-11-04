"""Build a versioned AZ-aware region scenario from verified Azure PIR records.

The two recorded single-AZ compute outages set a pooled *proxy* AZ rate for
12 logical AWS US AZs. Broad WAN events are retained as a separately labelled
stress scenario, not treated as full all-region loss in the primary variant.
This does not revise any existing library entry used by prior results.
``az-aware@v3`` is the same scenario with planned downtime at Ford's cadence (the default from 2026-11-04).
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.fit import gamma_ppf  # noqa: E402
from powder.faults.regions import region_map  # noqa: E402

LIB = ROOT / "data/fault_library"
H_PER_YEAR = 8766.0
YEARS = 1602 / 365.25  # 2022-06-01 to 2026-10-20; same window as build_region_outages.py
AZ_IDS = ("2022-06-07-eus2-cooling", "2023-09-16-eus-power")


def main():
    source = json.loads((LIB / "sources/azure-pir-incidents.json").read_text())
    measurements = {m["id"]: m for m in source["measurements"]}
    incidents = []
    for event_id in AZ_IDS:
        entry = measurements[event_id]
        if entry["status"] != "V" or entry["value"]["scope"] != "single_az":
            raise ValueError(f"AZ outage source {event_id} not verified")
        incidents.append(entry)
    exposure = len(region_map()) * 3 * YEARS
    shape = len(incidents) + 0.5
    rates = {
        "low": gamma_ppf(.05, shape, exposure) / H_PER_YEAR,
        "central": shape / exposure / H_PER_YEAR,
        "high": 2 * gamma_ppf(.95, shape, exposure) / H_PER_YEAR,
    }
    minutes = sorted(float(m["value"]["compute_duration_minutes"]) for m in incidents)
    durations = {"low": {"type": "constant", "value": minutes[0] * 60},
                 "central": {"type": "constant", "value": sum(minutes) / len(minutes) * 60},
                 "high": {"type": "constant", "value": minutes[-1] * 60}}
    profiles = json.loads((LIB / "profiles/region_outage/aws-us-from-azure.json").read_text())
    original = next(p for p in profiles if p["id"] == "region_outage/aws-us-from-azure")
    az = deepcopy(original)
    az["id"] = "region_outage/aws-us-az-aware@v2"
    az["version"] = 2
    az["tags"] = ["azure-pir", "az-aware", "sensitivity"]
    az["derived_from"] = sorted(set(az["derived_from"]) | {
        f"azure-pir-incidents#{event_id}" for event_id in AZ_IDS})
    az["built_by"] = "scripts/fault_library/build_az_outages.py"
    az["az_outages"] = {"zones_per_region": 3, "rate_per_hour": rates,
                        "duration": durations, "n_incidents": len(incidents),
                        "exposure_az_years": exposure,
                        "incidents": list(AZ_IDS),
                        "rule": "Jeffreys Gamma(n+0.5, 4 AWS regions x 3 AZ x exposure years); pooled AZ proxy"}
    az["wan_stress"] = az.pop("global")
    az["notes"] = ("Primary AZ-aware sensitivity: per-region events retain the original conservative "
                   "scope; single-AZ rates pool two verified Azure events over 12 logical AWS AZs. "
                   "WAN incidents are recorded in wan_stress, not injected as full all-region "
                   "outages. This is a cross-cloud modelling assumption, not measured AWS AZ rates.")
    out = LIB / "profiles/region_outage/aws-us-az-aware@v2.json"
    out.write_text(json.dumps(az, indent=2) + "\n")

    central = json.loads((LIB / "scenarios/canonical.json").read_text())
    scenario = deepcopy(next(s for s in central if s["id"] == "scenario/canonical-central"))
    scenario["id"] = "scenario/canonical-central-az-aware@v2"
    scenario["version"] = 2
    scenario["region_outage"] = az["id"]
    scenario["notes"] = "AZ-aware primary sensitivity; partial WAN retained as separate stress metadata"
    (LIB / "scenarios/az-aware@v2.json").write_text(json.dumps(scenario, indent=2) + "\n")
    # @v3 (decision 2026-11-04): planned downtime at Ford's planned-reboot cadence, not weekly.
    v3 = {**scenario, "id": "scenario/canonical-central-az-aware@v3", "version": 3,
          "planned_downtime": "planned_downtime/ford10-rate",
          "notes": scenario["notes"] + "; planned downtime at Ford's cadence (ford10-rate), not weekly"}
    (LIB / "scenarios/az-aware@v3.json").write_text(json.dumps(v3, indent=2) + "\n")
    print(json.dumps({"profile": az["id"], "scenario": scenario["id"],
                      "az_rate_per_year": rates["central"] * H_PER_YEAR,
                      "az_mean_duration_minutes": durations["central"]["value"] / 60}, indent=2))


if __name__ == "__main__":
    main()
