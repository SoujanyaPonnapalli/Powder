"""Machine classes, price assignments and scenarios (Step G).

    python scripts/fault_library/build_machines.py

Run after build_ford10.py, build_literature.py, build_region_outages.py and
(once prices are fetched) build_prices.py.

* Machine classes: one per non-legacy, non-spot node profile (cloud and
  reference tiers), offered in all four US regions, with ``price.route =
  assigned``. Spot machine classes come from build_prices.py (native prices).
* Synthetic stress profiles: cluster/ford10-google-storage-ge15min with all rates x10 and
  x0.1 (heterogeneity for the ILP; stress tests).
* Price assignments (only when a price distribution exists): rho = 0 for
  seeds 0..19, rho = 0.5 (seed 0), rho = 1, over the default distribution
  variant, with region factors of the deployment type.
* Scenarios: canonical-{optimistic,central,pessimistic}@v2, planned downtime at Ford's Table 1
  planned-reboot cadence (planned_downtime/ford10-rate). Without prices they are availability-only
  (price_policy none); with prices there are priced variants for rho = 1, rho = 0 (seed 0) and rho = 0.5.
  Other seeds are selected at run time with ``scenario_node_configs(..., price_assignment=...)``.
  Reference scenario: reference-ford10 (only the ford10 machine classes).
* The v1 canonical scenarios and reference-ford10-weekly used an assumed weekly planned cadence; they
  were removed on 2026-11-17 (last commit that has them: 1b52ccf).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.assign import assign_prices, kendall_tau, reliability_score  # noqa: E402
from powder.faults.library import FaultLibrary  # noqa: E402

LIB = ROOT / "data/fault_library"
US_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
DEPLOYMENT = {"instance_type": "i4i.2xlarge", "fallback_instance_type": "m6id.2xlarge",
              "regions": US_REGIONS, "azs_per_region": 3,
              # from the AWS price list (2026-10-21): offered in all four US regions
              "vcpu": 8, "mem_gb": 64, "storage_gb": 1875, "storage": "1 x 1875 NVMe SSD (instance store)"}
#: bare-metal transient base of the stress profiles (the catalogue and spot classes use it too)
BARE_METAL_BASE = "cluster/ford10-google-storage-ge15min"
LEVELS = ("optimistic", "central", "pessimistic")
from powder.faults.system import REBUILD_H, SYSTEM  # noqa: E402  (exec #10 placeholders)
RHO0_SEEDS = range(20)
PLANNED_V2 = "planned_downtime/ford10-rate"


def synthetic_profiles(base: dict) -> list[dict]:
    out = []
    for factor, tag in ((10.0, "x10"), (0.1, "x0.1")):
        p = copy.deepcopy(base)
        p.update({"id": f"synthetic/ford10-ge15min-{tag}", "tier": "reference", "tags": ["synthetic", "stress"],
                  "derived_from": [base["id"]], "built_by": "scripts/fault_library/build_machines.py",
                  "notes": f"{base['id']} with every cause rate multiplied by {factor} (stress / heterogeneity)."})
        for c in p["causes"].values():
            c["rate_per_hour"] = {k: v * factor for k, v in c["rate_per_hour"].items()}
            c.pop("rate_model", None)  # a scaled profile keeps constant rates only
        out.append(p)
    return out


def main() -> int:
    base = next(p for p in json.loads((LIB / "profiles/node/ford10.json").read_text())
                if p["id"] == BARE_METAL_BASE)
    (LIB / "profiles/node/synthetic.json").write_text(json.dumps(synthetic_profiles(base), indent=1) + "\n")
    lib = FaultLibrary.load(LIB, validate=False)

    # not instantiable: legacy, spot (built by build_prices.py), and profiles whose transient vs permanent
    # split is unknown (e.g. Birke crash tickets: recoverability not reported)
    profiles = [p for p in lib.query(kind="node")
                if not {"legacy", "recoverability-unknown"} & set(p.get("tags", [])) and not p["id"].startswith("spot/")]
    machines = [{"id": f"machine/{p['id'].split('/', 1)[1]}", "kind": "machine", "fault_profile": p["id"],
                 "tier": "ondemand", "storage_mode": "local", "regions": US_REGIONS, "price": {"route": "assigned"}}
                for p in sorted(profiles, key=lambda p: p["id"])]
    (LIB / "machines/ondemand.json").write_text(json.dumps(machines, indent=1) + "\n")
    spot_machines = [m["id"] for m in lib.query(kind="machine") if m["id"].startswith("machine/spot-")]

    scores = {m["id"]: reliability_score(lib.get(m["fault_profile"]), REBUILD_H) for m in machines}
    tau = kendall_tau([reliability_score(lib.get(m["fault_profile"]), REBUILD_H / 4) for m in machines],
                      [reliability_score(lib.get(m["fault_profile"]), REBUILD_H * 4) for m in machines])

    dist = next((d for d in lib.query(kind="price") if d.get("price_kind") == "price_distribution"
                 and d.get("default")), None)
    assignments = []
    factors = {}
    if dist is not None:
        rf_path = LIB / "derived/region_factors.json"
        factors = json.loads(rf_path.read_text())["factors"].get(DEPLOYMENT["instance_type"], {}) if rf_path.exists() else {}
        specs = [(0.0, s, REBUILD_H) for s in RHO0_SEEDS] + [(0.5, 0, REBUILD_H), (1.0, 0, REBUILD_H)]
        if tau < 0.9:  # ranking depends on the rebuild time: also rank with R/4 and 4R (plan §3.4)
            specs += [(1.0, 0, REBUILD_H / 4), (1.0, 0, REBUILD_H * 4)]
        for rho, seed, R in specs:
            sc = scores if R == REBUILD_H else {m["id"]: reliability_score(lib.get(m["fault_profile"]), R) for m in machines}
            a = assign_prices(sc, dist["points"], rho, seed)
            suffix = "" if R == REBUILD_H else ("-Rx0.25" if R < REBUILD_H else "-Rx4")
            assignments.append({
                "id": f"price-assign/rho{rho:g}-seed{seed}{suffix}@{dist['id'].split('@')[-1]}", "kind": "price",
                "price_kind": "price_assignment", "rho": rho, "seed": seed, "distribution": dist["id"],
                "derived_from": [dist["id"]], "assignments": a.prices, "region_factors": factors,
                "reliability_scores_h_per_year": sc, "realised_spearman": a.realised_spearman,
                "rebuild_hours": R, "kendall_tau_R_quarter_vs_4x": tau,
                "notes": "base prices are us-east-1 equivalents; x region factor of the deployment type per region"})
    (LIB / "prices/assignments.json").write_text(json.dumps(assignments, indent=1) + "\n")

    all_machines = [m["id"] for m in machines] + spot_machines
    # The v1 canonical scenarios (weekly planned cadence) were removed (decision 2026-11-17). They are still
    # built in memory as the base of @v2, so @v2's content and key order are unchanged; only @v2 is written.
    common = {"kind": "scenario", "region_outage": "region_outage/aws-us-from-azure",
              "planned_downtime": PLANNED_V2, "deployment": DEPLOYMENT,
              "system": SYSTEM}
    scenarios = []
    for lvl in LEVELS:
        scenarios.append({**common, "id": f"scenario/canonical-{lvl}", "machines": all_machines, "level": lvl,
                          "price_assignment": None, "price_policy": "none",
                          "notes": "availability-only: every library machine class, no prices"})
        for a in assignments:
            if (a["rho"], a["seed"]) in ((1.0, 0), (0.0, 0), (0.5, 0)) and a["rebuild_hours"] == REBUILD_H:
                scenarios.append({**common, "id": f"scenario/canonical-{lvl}-rho{a['rho']:g}", "machines": all_machines,
                                  "level": lvl, "price_assignment": a["id"], "price_policy": "mean_over_window",
                                  "notes": f"priced: rho={a['rho']:g} assignment for on-demand classes; native spot prices"})
    ford_mids = [m["id"] for m in machines if "ford10" in m["id"]]
    scenarios.append({**common, "id": "scenario/reference-ford10", "machines": ford_mids, "level": "central",
                      "planned_downtime": "planned_downtime/ford10-rate", "price_assignment": None,
                      "price_policy": "none",
                      "notes": "contrasting environment: only ford10 machine classes, with Google's own planned-reboot "
                               "cadence (ford10-rate) so planned downtime is not stacked on Ford's restarts"})
    (LIB / "scenarios/reference-ford10.json").write_text(
        json.dumps([s for s in scenarios if s["id"] == "scenario/reference-ford10"], indent=1) + "\n")
    # @v2 (decision 2026-11-04): planned downtime at Ford's Table 1 planned-reboot cadence, not weekly.
    v2 = [{**s, "id": s["id"] + "@v2", "version": 2, "planned_downtime": PLANNED_V2,
           "notes": s["notes"] + "; planned downtime at Ford's cadence (ford10-rate), not weekly"}
          for s in scenarios if s["id"].startswith("scenario/canonical-")]
    (LIB / "scenarios/canonical@v2.json").write_text(json.dumps(v2, indent=1) + "\n")

    problems = FaultLibrary.load(LIB, validate=False).validate()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(json.dumps({"machines": len(machines), "spot_machines": len(spot_machines),
                      "assignments": len(assignments), "scenarios": len(scenarios),
                      "reliability_h_per_year": scores, "kendall_tau_R": tau}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
