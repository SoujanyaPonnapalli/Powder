"""Preview a deployment without touching AWS (becomes ``deploy_infra.py --dry-run`` in exec #7 step 3).

    python -m powder.deploy.plan infra/tidb/specs/smoke.yaml [--json] [--operator-cidr 203.0.113.7/32]

Prints the machines per region and role, the network plan, the estimated AWS bill (and the spend cap
implied by ``limits.max_hours``), vCPU needs per region for the quota check, inter-region transfer prices
and the modeled cost of the emulated machine classes. Exit status: 0 ok, 1 the plan breaks a limit or a price
is unknown, 2 the spec itself is invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from powder.faults import FaultLibrary

from . import netplan, pricing
from .spec import ClusterSpec, SpecError, parse_spec, plan_nodes, spec_warnings

PLACEHOLDER_AZS = ["a", "b", "c"]       # real AZs are chosen from what each region offers, at deploy time


def build_plan(spec: ClusterSpec, lib: FaultLibrary | None = None, operator_cidr: str = "203.0.113.1/32") -> dict:
    lib = lib or FaultLibrary.load()
    nodes = plan_nodes(spec)
    problems: list[str] = []
    try:
        est = pricing.estimate(spec, lib, nodes=nodes)
    except pricing.UnknownPrice as e:
        est = None
        problems.append(str(e))
    net = netplan.plan_network(spec.regions, {r: PLACEHOLDER_AZS for r in spec.regions}, operator_cidr)
    problems += netplan.validate_plan(net)
    plan = {
        "handle": spec.handle, "label": spec.label, "spec_hash": spec.spec_hash(), "tidb_version": spec.tidb_version,
        "scenario": spec.scenario, "regions": spec.regions,
        "nodes": [{"name": n.name, "role": n.role, "region": n.region, "type": n.instance_type, "az": n.az,
                   "machine": n.machine} for n in nodes],
        "network": {"vpcs": net.vpcs, "peerings": net.peerings, "peering_count": len(net.peerings)},
        "warnings": spec_warnings(spec, lib),
        "problems": problems,
        "cost": None,
    }
    if est is not None:
        problems += pricing.check_limits(est, spec.limits)
        plan["cost"] = {
            "hourly_usd": est.hourly_total,
            "cap_usd": est.total_for(spec.limits.max_hours),
            "max_hours": spec.limits.max_hours,
            "by_role": est.hourly_by_role, "by_region": est.hourly_by_region,
            "vcpu_by_region": est.vcpu_by_region,
            "inter_region_usd_per_gb": {"min": est.inter_region_per_gb[0], "max": est.inter_region_per_gb[1]},
            "modeled_hourly_usd": est.modeled_hourly, "modeled_note": est.modeled_note,
            "excluded": est.excluded,
        }
    return plan


def format_plan(p: dict) -> str:
    L = []
    h = p["handle"] if p["handle"].lower() != "auto" else "(random AdjectiveNoun, generated at deploy)"
    L.append(f"Deployment {h}   spec {p['spec_hash']}   TiDB {p['tidb_version']}   scenario {p['scenario']}")
    roles = Counter(n["role"] for n in p["nodes"])
    L.append(f"{len(p['nodes'])} machines in {len(p['regions'])} regions: " + ", ".join(f"{v} {k}" for k, v in roles.items()))
    L.append("")
    L.append(f"{'region':<10} " + " ".join(f"{r:>10}" for r in ("controller", "pd", "tikv", "systikv", "tidb", "loadgen")))
    for reg in p["regions"]:
        c = Counter(n["role"] for n in p["nodes"] if n["region"] == reg)
        L.append(f"{reg:<10} " + " ".join(f"{c.get(r, 0):>10}" for r in ("controller", "pd", "tikv", "systikv", "tidb", "loadgen")))
    net = p["network"]
    L.append("")
    L.append("Network: " + ", ".join(f"{r} {c}" for r, c in net["vpcs"].items()) + f"; {net['peering_count']} peerings")
    c = p["cost"]
    if c:
        L.append("")
        L.append(f"AWS bill: ~${c['hourly_usd']:.2f}/h   (cap for {c['max_hours']:g} h: ~${c['cap_usd']:.0f})")
        L.append("  by role:   " + "  ".join(f"{k} ${v:.2f}" for k, v in c["by_role"].items()))
        L.append("  by region: " + "  ".join(f"{k} ${v:.2f}" for k, v in c["by_region"].items()))
        L.append("  vCPUs to quota-check: " + "  ".join(f"{k} {v}" for k, v in c["vcpu_by_region"].items()))
        t = c["inter_region_usd_per_gb"]
        L.append(f"  inter-region transfer: ${t['min']:.3f}-${t['max']:.3f}/GB (volume depends on the workload)")
        if c["modeled_hourly_usd"] is not None:
            L.append(f"Modeled cost of the emulated machine classes (what the paper reports): "
                     f"${c['modeled_hourly_usd']:.2f}/h")
        else:
            L.append(f"Modeled cost: n/a ({c['modeled_note']})")
        L.append("  not included: " + ", ".join(c["excluded"]))
    for w in p["warnings"]:
        L.append(f"WARNING: {w}")
    for e in p["problems"]:
        L.append(f"PROBLEM: {e}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--operator-cidr", default="203.0.113.1/32")
    a = ap.parse_args(argv)
    try:
        spec = parse_spec(a.spec)
    except SpecError as e:
        print(e, file=sys.stderr)
        return 2
    plan = build_plan(spec, operator_cidr=a.operator_cidr)
    print(json.dumps(plan, indent=1, default=str) if a.json else format_plan(plan))
    return 1 if plan["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
