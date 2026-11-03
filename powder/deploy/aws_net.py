"""Execute a :mod:`netplan` plan on AWS: VPC, subnets, IGW, routes, security group, peering mesh (plan §5, §6.1).

Idempotent: each resource is looked up by its ``Run`` + ``Name`` tags before it is created, so re-running after
a partial failure continues instead of duplicating. Each id is written to ``RunState`` (and saved via the
``save`` callback) the moment it exists.
"""

from __future__ import annotations

import time
from typing import Callable

from . import aws
from .netplan import NetworkPlan, SgRule
from .state import RunState


def _tags(ctx: "Ctx", what: str):
    return aws.tag_list(ctx.handle, ctx.owner, ctx.created_at, what)


class Ctx:
    def __init__(self, clients: aws.Clients, handle: str, owner: str, created_at: str, state: RunState,
                 save: Callable[[], None] = lambda: None, sleep=time.sleep, log: Callable[[str], None] = lambda m: None):
        self.clients, self.handle, self.owner, self.created_at = clients, handle, owner, created_at
        self.state, self.save, self.sleep, self.log = state, save, sleep, log


def _find(describe, key: str, handle: str, name: str, **kw):
    """The resource of ``key`` in ``describe`` output tagged Run=handle and Name=<handle>-<name>, or None.

    The server-side filter narrows the query, but the tags of every result are re-checked here: reusing the
    wrong resource (for instance if a filter were ignored) would silently corrupt a deployment.
    """
    flt = [{"Name": "tag:Project", "Values": [aws.PROJECT]}, {"Name": "tag:Run", "Values": [handle]},
           {"Name": "tag:Name", "Values": [f"{handle}-{name}"]}]
    for r in describe(Filters=flt, **kw)[key]:
        t = aws.tags_to_dict(r.get("Tags"))
        if t.get("Project") == aws.PROJECT and t.get("Run") == handle and t.get("Name") == f"{handle}-{name}":
            return r
    return None


def ensure_region_network(ctx: Ctx, region: str, plan: NetworkPlan) -> None:
    ec2 = ctx.clients.ec2(region)
    net = ctx.state.region(region)
    h = ctx.handle

    if not net.vpc_id:
        v = _find(ec2.describe_vpcs, "Vpcs", h, "vpc")
        if v is None:
            v = ec2.create_vpc(CidrBlock=plan.vpcs[region],
                               TagSpecifications=aws.tag_spec("vpc", _tags(ctx, "vpc")))["Vpc"]
            ec2.modify_vpc_attribute(VpcId=v["VpcId"], EnableDnsSupport={"Value": True})
            ec2.modify_vpc_attribute(VpcId=v["VpcId"], EnableDnsHostnames={"Value": True})
            ctx.log(f"{region}: created VPC {v['VpcId']} {plan.vpcs[region]}")
        net.vpc_id = v["VpcId"]
        ctx.save()

    if not net.igw_id:
        g = _find(ec2.describe_internet_gateways, "InternetGateways", h, "igw")
        if g is None:
            g = ec2.create_internet_gateway(TagSpecifications=aws.tag_spec("internet-gateway", _tags(ctx, "igw")))[
                "InternetGateway"]
        net.igw_id = g["InternetGatewayId"]
        ctx.save()
        attached = {a.get("VpcId") for a in g.get("Attachments", [])}
        if net.vpc_id not in attached:
            ec2.attach_internet_gateway(InternetGatewayId=net.igw_id, VpcId=net.vpc_id)

    if not net.route_table_id:
        rt = _find(ec2.describe_route_tables, "RouteTables", h, "rt")
        if rt is None:
            rt = ec2.create_route_table(VpcId=net.vpc_id, TagSpecifications=aws.tag_spec("route-table", _tags(ctx, "rt")))[
                "RouteTable"]
            ec2.create_route(RouteTableId=rt["RouteTableId"], DestinationCidrBlock="0.0.0.0/0", GatewayId=net.igw_id)
        net.route_table_id = rt["RouteTableId"]
        ctx.save()

    for az, cidr in plan.subnets[region].items():
        if az in net.subnets:
            continue
        s = _find(ec2.describe_subnets, "Subnets", h, f"subnet-{az}")
        if s is None:
            s = ec2.create_subnet(VpcId=net.vpc_id, CidrBlock=cidr, AvailabilityZone=az,
                                  TagSpecifications=aws.tag_spec("subnet", _tags(ctx, f"subnet-{az}")))["Subnet"]
            ec2.modify_subnet_attribute(SubnetId=s["SubnetId"], MapPublicIpOnLaunch={"Value": True})
            ec2.associate_route_table(RouteTableId=net.route_table_id, SubnetId=s["SubnetId"])
        net.subnets[az] = s["SubnetId"]
        ctx.save()

    if not net.security_group_id:
        sg = _find(ec2.describe_security_groups, "SecurityGroups", h, "sg")
        if sg is None:
            sg = ec2.create_security_group(
                GroupName=f"{h}-sg", Description=f"Powder run {h}", VpcId=net.vpc_id,
                TagSpecifications=aws.tag_spec("security-group", _tags(ctx, "sg")))
            sg = {"GroupId": sg["GroupId"]}
            ec2.authorize_security_group_ingress(GroupId=sg["GroupId"], IpPermissions=[_permission(r) for r in plan.sg_rules[region]])
        net.security_group_id = sg["GroupId"]
        ctx.save()


def _permission(r: SgRule) -> dict:
    p: dict = {"IpProtocol": r.protocol, "IpRanges": [{"CidrIp": r.cidr, "Description": r.description[:250]}]}
    if r.protocol != "-1":
        p["FromPort"], p["ToPort"] = r.from_port, r.to_port
    return p


def ensure_peering(ctx: Ctx, a: str, b: str, plan: NetworkPlan, timeout: float = 120.0) -> None:
    """Peering requested from region ``a`` (lower index), accepted in ``b``; routes added on both sides."""
    sa, sb = ctx.state.region(a), ctx.state.region(b)
    ea, eb = ctx.clients.ec2(a), ctx.clients.ec2(b)
    if b not in sa.peerings:
        existing = _find(ea.describe_vpc_peering_connections, "VpcPeeringConnections", ctx.handle, f"pcx-{a}-{b}")
        if existing is not None and existing["Status"]["Code"] not in ("deleted", "deleting", "failed", "rejected", "expired"):
            pcx = existing
        else:
            pcx = ea.create_vpc_peering_connection(
                VpcId=sa.vpc_id, PeerVpcId=sb.vpc_id, PeerRegion=b,
                TagSpecifications=aws.tag_spec("vpc-peering-connection", _tags(ctx, f"pcx-{a}-{b}")))[
                "VpcPeeringConnection"]
            ctx.log(f"peering {a} <-> {b}: requested {pcx['VpcPeeringConnectionId']}")
        sa.peerings[b] = pcx["VpcPeeringConnectionId"]
        ctx.save()
    pcx_id = sa.peerings[b]
    sb.peerings[a] = pcx_id

    # accept in the peer region once the request is visible there (it can lag the requester's view)
    attempts = max(1, int(timeout / 2))
    code = "missing"
    for _ in range(attempts):
        got = eb.describe_vpc_peering_connections(VpcPeeringConnectionIds=[pcx_id])["VpcPeeringConnections"]
        code = got[0]["Status"]["Code"] if got else "missing"
        if code == "active":
            break
        if code == "pending-acceptance":
            eb.accept_vpc_peering_connection(VpcPeeringConnectionId=pcx_id)
        elif code in ("failed", "rejected", "deleted", "expired"):
            raise aws.AwsError(f"peering {pcx_id} ({a}<->{b}) ended in state {code}")
        ctx.sleep(2)
    else:
        raise aws.AwsError(f"peering {pcx_id} ({a}<->{b}) not active after {timeout:.0f}s (state {code})")
    ctx.save()

    for region, rt_id, dest in ((a, sa.route_table_id, plan.vpcs[b]), (b, sb.route_table_id, plan.vpcs[a])):
        ec2 = ctx.clients.ec2(region)
        routes = ec2.describe_route_tables(RouteTableIds=[rt_id])["RouteTables"][0]["Routes"]
        if not any(r.get("DestinationCidrBlock") == dest and r.get("VpcPeeringConnectionId") == pcx_id
                   and r.get("State", "active") == "active" for r in routes):
            ec2.create_route(RouteTableId=rt_id, DestinationCidrBlock=dest, VpcPeeringConnectionId=pcx_id)


def ensure_network(ctx: Ctx, plan: NetworkPlan) -> None:
    for region in plan.regions:
        ensure_region_network(ctx, region, plan)
    for a, b in plan.peerings:
        ensure_peering(ctx, a, b, plan)
