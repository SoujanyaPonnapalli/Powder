"""Cost and quota arithmetic for a run (plan §5, §9). No AWS calls.

Two different numbers are produced and must not be confused:

* **AWS bill estimate** - what the live fleet costs: every TiKV node is an ``i4i.2xlarge`` on-demand
  instance, plus the control plane, whatever machine class a slot emulates.
* **Modeled cost** - what the *emulated* machine classes would cost under the scenario's price assignment
  (the number the paper reports). ``None`` when the scenario has no prices (``price_policy: none``).

8-vCPU prices come from the fault library (``aws-ondemand``); the smaller control-plane types come from
``data/tidb/control_plane_prices.json`` (``infra/tidb/fetch_control_plane_prices.py``, same AWS Price List
source). A type with no price is an error, never a silent zero.

Not included in the bill estimate: root EBS volumes (~$0.006/h per 50 GB gp3 node), public IPv4 addresses
($0.005/h each), Prometheus/Grafana storage, and inter-region data transfer (reported separately in $/GB
because volume depends on the workload).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

from powder.faults.library import FaultLibrary
from powder.faults.prices import price_per_hour, transfer_per_gb

from .spec import ClusterSpec, Limits, NodePlan, plan_nodes

ROOT = Path(__file__).resolve().parents[2]
CP_PRICES_FILE = ROOT / "data" / "tidb" / "control_plane_prices.json"
LIBRARY_VCPU = 8     # the fault library only carries 8-vCPU x86 on-demand prices (exec #1)


class UnknownPrice(LookupError):
    pass


def load_control_plane_prices(path: Path = CP_PRICES_FILE) -> dict:
    if not Path(path).exists():
        return {"prices": {}, "vcpu": {}}
    return json.loads(Path(path).read_text())


def _library_key(itype: str, region: str) -> str:
    return f"aws:{itype}:{region}:ondemand:linux"


def ondemand_price(lib: FaultLibrary, cpp: dict, itype: str, region: str) -> float:
    key = _library_key(itype, region)
    if key in lib.entries:
        return float(lib.get(key, "price")["price_per_hour"])
    p = cpp.get("prices", {}).get(itype, {}).get(region)
    if p is None:
        raise UnknownPrice(f"no on-demand price for {itype} in {region}: not in the fault library (8 vCPU only) "
                           f"nor in {CP_PRICES_FILE.name}; rerun infra/tidb/fetch_control_plane_prices.py "
                           f"(adding the type to CANDIDATES)")
    return float(p)


def instance_vcpu(lib: FaultLibrary, cpp: dict, itype: str, region: str) -> int:
    if _library_key(itype, region) in lib.entries:
        return LIBRARY_VCPU
    v = cpp.get("vcpu", {}).get(itype)
    if v is None:
        raise UnknownPrice(f"vCPU count of {itype} unknown")
    return int(v)


@dataclass
class NodeCost:
    name: str
    role: str
    region: str
    instance_type: str
    vcpu: int
    hourly: float


@dataclass
class CostEstimate:
    nodes: list[NodeCost]
    hourly_total: float
    hourly_by_role: dict[str, float]
    hourly_by_region: dict[str, float]
    vcpu_by_region: dict[str, int]
    inter_region_per_gb: tuple[float, float]        # (min, max) over directed region pairs used
    modeled_hourly: float | None                     # emulated machine classes, scenario prices
    modeled_note: str = ""
    excluded: list[str] = field(default_factory=lambda: [
        "root EBS volumes", "public IPv4 addresses", "inter-region data transfer volume"])

    def total_for(self, hours: float) -> float:
        return self.hourly_total * hours


def transfer_entry_id(lib: FaultLibrary) -> str:
    ids = sorted(e["id"] for e in lib.query("price") if "per_gb" in e)
    if not ids:
        raise UnknownPrice("no transfer price entry in the library")
    return ids[-1]


def estimate(spec: ClusterSpec, lib: FaultLibrary | None = None, cpp: dict | None = None,
             nodes: list[NodePlan] | None = None) -> CostEstimate:
    lib = lib or FaultLibrary.load()
    cpp = cpp if cpp is not None else load_control_plane_prices()
    nodes = nodes or plan_nodes(spec)
    costs, by_role, by_region, vcpus = [], {}, {}, {}
    for n in nodes:
        hourly = ondemand_price(lib, cpp, n.instance_type, n.region)
        v = instance_vcpu(lib, cpp, n.instance_type, n.region)
        costs.append(NodeCost(n.name, n.role, n.region, n.instance_type, v, hourly))
        by_role[n.role] = by_role.get(n.role, 0.0) + hourly
        by_region[n.region] = by_region.get(n.region, 0.0) + hourly
        vcpus[n.region] = vcpus.get(n.region, 0) + v
    tid = transfer_entry_id(lib)
    pairs = [transfer_per_gb(lib, tid, a, b) for a, b in permutations(spec.regions, 2)]
    inter = (min(pairs), max(pairs)) if pairs else (0.0, 0.0)

    scenario = lib.get(spec.scenario, "scenario")
    modeled, note = None, ""
    if scenario.get("price_policy", "none") == "none":
        note = f"scenario {spec.scenario} has no prices (price_policy none); use a priced scenario for modeled cost"
    else:
        modeled = sum(price_per_hour(lib, scenario, s.machine, s.region) for s in spec.slots)
    return CostEstimate(costs, sum(c.hourly for c in costs), by_role, by_region, vcpus, inter, modeled, note)


def check_limits(est: CostEstimate, limits: Limits) -> list[str]:
    """Problems if the estimate exceeds the spec's own spend limits (deploy refuses to proceed on any)."""
    p = []
    if est.hourly_total > limits.max_hourly_usd:
        p.append(f"estimated ${est.hourly_total:.2f}/h exceeds limits.max_hourly_usd {limits.max_hourly_usd}")
    return p


def check_quotas(est: CostEstimate, quotas: dict[str, int]) -> list[str]:
    """Compare per-region vCPU needs with the on-demand Standard vCPU quota ``quotas[region]``."""
    p = []
    for region, need in sorted(est.vcpu_by_region.items()):
        have = quotas.get(region)
        if have is None:
            p.append(f"{region}: vCPU quota unknown")
        elif need > have:
            p.append(f"{region}: needs {need} vCPUs but the quota is {have}; request an increase")
    return p
