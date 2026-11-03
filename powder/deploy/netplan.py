"""Pure network planning for a run: VPC CIDRs, subnets, peering mesh, routes, security-group rules.

``deploy_infra.py`` (exec #7 step 3) executes this plan; nothing here calls AWS. Addressing is derived
from a fixed region ordering so that, for example, us-east-1 is always ``10.0.0.0/16`` and a private
IP reveals its region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from powder.faults.regions import AWS_US_REGIONS

#: Stable region index -> second octet of the VPC CIDR.
REGION_INDEX = {r: i for i, r in enumerate(AWS_US_REGIONS)}
SUBNET_PREFIX = 20          # one /20 per AZ inside the /16
MAX_AZS = 16                # 16 x /20 fill a /16
ORACLE_PORT = 8650          # reserved for the PD <-> ILP oracle (plan §8)


class NetPlanError(ValueError):
    pass


def vpc_cidr(region: str) -> str:
    if region not in REGION_INDEX:
        raise NetPlanError(f"unknown region {region!r}; expected one of {sorted(REGION_INDEX)}")
    return f"10.{REGION_INDEX[region]}.0.0/16"


def subnet_cidr(region: str, az_index: int) -> str:
    if not 0 <= az_index < MAX_AZS:
        raise NetPlanError(f"az_index {az_index} out of range")
    return f"10.{REGION_INDEX[region]}.{az_index * 16}.0/{SUBNET_PREFIX}"


@dataclass(frozen=True)
class SgRule:
    protocol: str            # "tcp" | "udp" | "-1"
    from_port: int
    to_port: int
    cidr: str
    description: str


@dataclass(frozen=True)
class RouteEntry:
    region: str              # route table owner
    destination_cidr: str
    peering: tuple[str, str]  # (requester region, accepter region)


@dataclass
class NetworkPlan:
    regions: list[str]
    vpcs: dict[str, str]
    subnets: dict[str, dict[str, str]]          # region -> az name -> cidr
    peerings: list[tuple[str, str]]             # (requester, accepter), requester has the lower index
    routes: list[RouteEntry]
    sg_rules: dict[str, list[SgRule]]
    operator_cidr: str = field(default="")


def plan_network(regions: list[str], azs: dict[str, list[str]], operator_cidr: str) -> NetworkPlan:
    """Plan the mesh for ``regions``.

    ``azs`` maps region -> AZ names to use (decided at deploy time from what is offered);
    ``operator_cidr`` is the /32 allowed SSH/Grafana/Prometheus access.
    """
    regions = sorted(set(regions), key=lambda r: REGION_INDEX[r] if r in REGION_INDEX else 99)
    if not regions:
        raise NetPlanError("no regions")
    if "/" not in operator_cidr or operator_cidr.startswith("0.0.0.0/0"):
        raise NetPlanError("operator_cidr must be a specific CIDR (never 0.0.0.0/0)")
    vpcs = {r: vpc_cidr(r) for r in regions}
    subnets: dict[str, dict[str, str]] = {}
    for r in regions:
        names = sorted(azs.get(r) or [])
        if not names:
            raise NetPlanError(f"no AZs given for {r}")
        subnets[r] = {az: subnet_cidr(r, i) for i, az in enumerate(names)}
    peerings = [(a, b) for a, b in combinations(regions, 2)]   # regions sorted => requester has lower index
    routes = []
    for a, b in peerings:
        routes.append(RouteEntry(a, vpcs[b], (a, b)))
        routes.append(RouteEntry(b, vpcs[a], (a, b)))
    mesh = [vpcs[r] for r in regions]
    sg: dict[str, list[SgRule]] = {}
    for r in regions:
        rules = [SgRule("-1", 0, 65535, cidr, f"cluster traffic from {reg} VPC") for reg, cidr in zip(regions, mesh)]
        rules += [
            SgRule("tcp", 22, 22, operator_cidr, "SSH from operator"),
            SgRule("tcp", 3000, 3000, operator_cidr, "Grafana from operator"),
            SgRule("tcp", 9090, 9090, operator_cidr, "Prometheus from operator"),
        ]
        sg[r] = rules
    return NetworkPlan(regions, vpcs, subnets, peerings, routes, sg, operator_cidr)


def validate_plan(plan: NetworkPlan) -> list[str]:
    """Self-consistency checks (used by tests and by deploy preflight)."""
    problems = []
    n = len(plan.regions)
    if len(plan.peerings) != n * (n - 1) // 2:
        problems.append(f"expected {n * (n - 1) // 2} peerings, got {len(plan.peerings)}")
    cidrs = list(plan.vpcs.values())
    if len(set(cidrs)) != len(cidrs):
        problems.append("overlapping VPC CIDRs")
    for r, subs in plan.subnets.items():
        if len(set(subs.values())) != len(subs):
            problems.append(f"duplicate subnet CIDRs in {r}")
        for cidr in subs.values():
            if not cidr.startswith(plan.vpcs[r][:-7]):     # same 10.<i>. prefix
                problems.append(f"subnet {cidr} outside {plan.vpcs[r]}")
    expected_routes = {(r.region, r.destination_cidr) for r in plan.routes}
    for a, b in plan.peerings:
        if (a, plan.vpcs[b]) not in expected_routes or (b, plan.vpcs[a]) not in expected_routes:
            problems.append(f"missing route on one side of {a}<->{b}")
    for r, rules in plan.sg_rules.items():
        if any(x.cidr == "0.0.0.0/0" for x in rules):
            problems.append(f"{r}: world-open rule")
    return problems
