"""Find and delete runs by tag across all regions (plan §6.0, §6.1).

Discovery never depends on ``state.json``: everything tagged ``Project=powder`` is grouped by its ``Run`` tag, so a
lost state file, a half-failed deploy or a deployment made from another machine are all visible and removable.
Deletion is in dependency order (instances -> peering -> route tables -> subnets -> IGW -> SG -> VPC -> key pair),
tolerant of resources that are already gone, and retries ``DependencyViolation`` while AWS releases interfaces.
``verify`` rescans every region for anything still tagged.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from botocore.exceptions import ClientError

from . import aws
from .aws import Clients, tags_to_dict

UNTAGGED = "(no Run tag)"
GONE_CODES = ("NotFound", "InvalidParameterValue")        # already deleted / never existed
RETRY_CODES = ("DependencyViolation", "InvalidGroup.InUse", "IncorrectState", "InvalidNetworkInterface.InUse")


@dataclass
class RegionRes:
    instances: list[dict] = field(default_factory=list)       # {id, name, type, state}
    peerings: list[str] = field(default_factory=list)
    route_tables: list[str] = field(default_factory=list)
    subnets: list[str] = field(default_factory=list)
    igws: list[tuple[str, list[str]]] = field(default_factory=list)   # (igw id, attached vpc ids)
    security_groups: list[str] = field(default_factory=list)
    vpcs: list[str] = field(default_factory=list)
    key_pairs: list[str] = field(default_factory=list)        # key names

    def empty(self) -> bool:
        return not any((self.instances, self.peerings, self.route_tables, self.subnets, self.igws,
                        self.security_groups, self.vpcs, self.key_pairs))

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in vars(self).items() if v}


@dataclass
class RunResources:
    handle: str
    owner: str = ""
    created_at: str = ""
    regions: dict[str, RegionRes] = field(default_factory=dict)

    def region(self, r: str) -> RegionRes:
        return self.regions.setdefault(r, RegionRes())

    def instance_count(self) -> int:
        return sum(len(r.instances) for r in self.regions.values())

    def age_seconds(self, now: datetime | None = None) -> float | None:
        if not self.created_at:
            return None
        return ((now or datetime.now(timezone.utc)) - aws.parse_ts(self.created_at)).total_seconds()


def discover(clients: Clients, regions: list[str] | None = None, handle: str | None = None) -> dict[str, RunResources]:
    flt = aws.run_filter(handle)
    runs: dict[str, RunResources] = {}

    def run_of(tags: list[dict] | None, region: str) -> RegionRes:
        t = tags_to_dict(tags)
        h = t.get("Run", UNTAGGED)
        r = runs.setdefault(h, RunResources(h))
        r.owner = r.owner or t.get("Owner", "")
        if t.get("CreatedAt") and (not r.created_at or t["CreatedAt"] < r.created_at):
            r.created_at = t["CreatedAt"]
        return r.region(region)

    for region in regions or aws.ALL_REGIONS:
        ec2 = clients.ec2(region)
        for res in ec2.describe_instances(Filters=flt + [{"Name": "instance-state-name", "Values": [
                "pending", "running", "shutting-down", "stopping", "stopped"]}])["Reservations"]:
            for i in res["Instances"]:
                t = tags_to_dict(i.get("Tags"))
                run_of(i.get("Tags"), region).instances.append(
                    {"id": i["InstanceId"], "name": t.get("Node", t.get("Name", "")), "type": i["InstanceType"],
                     "state": i["State"]["Name"]})
        for p in ec2.describe_vpc_peering_connections(Filters=flt)["VpcPeeringConnections"]:
            if p["Status"]["Code"] not in ("deleted", "rejected", "failed", "expired"):
                run_of(p.get("Tags"), region).peerings.append(p["VpcPeeringConnectionId"])
        for rt in ec2.describe_route_tables(Filters=flt)["RouteTables"]:
            run_of(rt.get("Tags"), region).route_tables.append(rt["RouteTableId"])
        for s in ec2.describe_subnets(Filters=flt)["Subnets"]:
            run_of(s.get("Tags"), region).subnets.append(s["SubnetId"])
        for g in ec2.describe_internet_gateways(Filters=flt)["InternetGateways"]:
            run_of(g.get("Tags"), region).igws.append((g["InternetGatewayId"], [a["VpcId"] for a in g.get("Attachments", [])
                                                                             if a.get("VpcId")]))
        for sg in ec2.describe_security_groups(Filters=flt)["SecurityGroups"]:
            run_of(sg.get("Tags"), region).security_groups.append(sg["GroupId"])
        for v in ec2.describe_vpcs(Filters=flt)["Vpcs"]:
            run_of(v.get("Tags"), region).vpcs.append(v["VpcId"])
        for k in ec2.describe_key_pairs(Filters=flt)["KeyPairs"]:
            run_of(k.get("Tags"), region).key_pairs.append(k["KeyName"])
    return runs


def hourly_usd(run: RunResources, lib=None, cpp: dict | None = None) -> float | None:
    """Estimated $/h of the run's live instances; None when any type has no known price."""
    from . import pricing
    from powder.faults import FaultLibrary
    lib = lib or FaultLibrary.load()
    cpp = cpp if cpp is not None else pricing.load_control_plane_prices()
    total = 0.0
    for region, rr in run.regions.items():
        for i in rr.instances:
            try:
                total += pricing.ondemand_price(lib, cpp, i["type"], region)
            except pricing.UnknownPrice:
                return None
    return total


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    return f"{d}d{h}h" if d else f"{h}h{m:02d}m" if h else f"{m}m"


def _abbr(region: str) -> str:
    parts = region.split("-")                       # us-east-1 -> use1
    return parts[0] + parts[1][0] + parts[2]


def format_list(runs: dict[str, RunResources], outputs_root: Path, lib=None, cpp: dict | None = None,
                now: datetime | None = None) -> str:
    if not runs:
        return "no Project=powder resources found in any region"
    lines = [f"{'HANDLE':<16}{'AGE':<9}{'REGIONS (nodes)':<44}{'$/h':>7}  LOCAL DIR"]
    for h, run in sorted(runs.items(), key=lambda kv: kv[1].created_at):
        per = {r: len(rr.instances) for r, rr in run.regions.items() if rr.instances}
        desc = (" ".join(f"{_abbr(r)}({n})" for r, n in sorted(per.items())) + f" ={sum(per.values())}") if per \
            else "no instances (" + ", ".join(sorted({k for rr in run.regions.values() for k in rr.counts()})) + " left)"
        price = hourly_usd(run, lib, cpp)
        local = "yes" if (outputs_root / h).is_dir() else "no"
        lines.append(f"{h:<16}{fmt_age(run.age_seconds(now)):<9}{desc:<44}{('%.1f' % price) if price is not None else '?':>7}  {local}")
    return "\n".join(lines)


class SelectionError(ValueError):
    pass


def select(runs: dict[str, RunResources], handles: list[str] | None = None, all_runs: bool = False,
           older_than_s: float | None = None, now: datetime | None = None) -> list[str]:
    """Pick handles to tear down. Named handles must exist (an unknown name is an error, not a no-op)."""
    if sum(bool(x) for x in (handles, all_runs, older_than_s is not None)) != 1:
        raise SelectionError("choose exactly one of: handle(s), --all, --older-than")
    if handles:
        unknown = [h for h in handles if h not in runs]
        if unknown:
            raise SelectionError(f"no resources found for handle(s): {', '.join(unknown)}")
        return list(dict.fromkeys(handles))
    if all_runs:
        return sorted(runs)
    out = []
    for h, run in runs.items():
        age = run.age_seconds(now)
        if age is not None and age > older_than_s:
            out.append(h)
    return sorted(out)


# ------------------------------------------------------------------ deletion
def _code(e: ClientError) -> str:
    return e.response.get("Error", {}).get("Code", "")


def _gone(e: ClientError) -> bool:
    return any(g in _code(e) for g in GONE_CODES)


def _retry(fn: Callable, what: str, sleep, tries: int = 12, delay: float = 5.0, log=lambda m: None):
    for n in range(tries):
        try:
            return fn()
        except ClientError as e:
            if _gone(e):
                return None
            if _code(e) in RETRY_CODES and n < tries - 1:
                log(f"  {what}: {_code(e)}, retrying")
                sleep(delay)
                continue
            raise


def teardown_run(clients: Clients, run: RunResources, sleep=time.sleep, log: Callable[[str], None] = print) -> dict:
    """Delete everything of one run. Returns {"deleted": {...}, "errors": [...]}; never raises on a single failure."""
    deleted: Counter = Counter()
    errors: list[str] = []

    def step(desc, fn):
        try:
            return fn()
        except ClientError as e:
            if _gone(e):                    # already deleted (stale snapshot, earlier partial teardown): success
                return None
            errors.append(f"{desc}: {e}")
            log(f"  ERROR {desc}: {e}")
        except Exception as e:  # keep going: report at the end, verify() is the safety net
            errors.append(f"{desc}: {e}")
            log(f"  ERROR {desc}: {e}")

    # 1. instances, in every region, then wait
    for region, rr in run.regions.items():
        ids = [i["id"] for i in rr.instances if i["state"] != "terminated"]
        if ids:
            ec2 = clients.ec2(region)
            step(f"{region} terminate {len(ids)} instances", lambda: ec2.terminate_instances(InstanceIds=ids))
            deleted["instances"] += len(ids)
    for region, rr in run.regions.items():
        ids = [i["id"] for i in rr.instances]
        if ids:
            ec2 = clients.ec2(region)
            step(f"{region} wait for termination", lambda: ec2.get_waiter("instance_terminated").wait(
                InstanceIds=ids, WaiterConfig={"Delay": 5, "MaxAttempts": 60}))

    # 2. peering connections (each appears in both regions; the second delete is a tolerated no-op)
    for region, rr in run.regions.items():
        for pcx in rr.peerings:
            if step(f"{region} delete peering {pcx}", lambda: _retry(
                    lambda: clients.ec2(region).delete_vpc_peering_connection(VpcPeeringConnectionId=pcx),
                    pcx, sleep, log=log)) is not None:
                deleted["peerings"] += 1

    # 3. per-region network, dependencies first
    for region, rr in run.regions.items():
        ec2 = clients.ec2(region)
        for rt in rr.route_tables:
            def drop_rt(rt=rt):
                assoc = ec2.describe_route_tables(RouteTableIds=[rt])["RouteTables"]
                for a in (assoc[0]["Associations"] if assoc else []):
                    if not a.get("Main"):
                        _retry(lambda a=a: ec2.disassociate_route_table(AssociationId=a["RouteTableAssociationId"]),
                               "disassociate", sleep, log=log)
                return _retry(lambda: ec2.delete_route_table(RouteTableId=rt), rt, sleep, log=log)
            step(f"{region} delete route table {rt}", drop_rt)
            deleted["route_tables"] += 1
        for s in rr.subnets:
            step(f"{region} delete subnet {s}", lambda s=s: _retry(lambda: ec2.delete_subnet(SubnetId=s), s, sleep, log=log))
            deleted["subnets"] += 1
        for igw, vpcs in rr.igws:
            for v in vpcs:
                step(f"{region} detach {igw}", lambda v=v, igw=igw: _retry(
                    lambda: ec2.detach_internet_gateway(InternetGatewayId=igw, VpcId=v), igw, sleep, log=log))
            step(f"{region} delete {igw}", lambda igw=igw: _retry(
                lambda: ec2.delete_internet_gateway(InternetGatewayId=igw), igw, sleep, log=log))
            deleted["igws"] += 1
        for sg in rr.security_groups:
            step(f"{region} delete security group {sg}", lambda sg=sg: _retry(
                lambda: ec2.delete_security_group(GroupId=sg), sg, sleep, log=log))
            deleted["security_groups"] += 1
        for v in rr.vpcs:
            step(f"{region} delete VPC {v}", lambda v=v: _retry(lambda: ec2.delete_vpc(VpcId=v), v, sleep, log=log))
            deleted["vpcs"] += 1
        for k in rr.key_pairs:
            step(f"{region} delete key pair {k}", lambda k=k: ec2.delete_key_pair(KeyName=k))
            deleted["key_pairs"] += 1
    return {"deleted": dict(deleted), "errors": errors}


def verify(clients: Clients, regions: list[str] | None = None) -> list[str]:
    """Anything still tagged Project=powder (terminated instances excluded). Empty list means clean."""
    left = []
    for h, run in discover(clients, regions).items():
        for region, rr in run.regions.items():
            if not rr.empty():
                left.append(f"{h} in {region}: {rr.counts()}")
    return left
