"""Launch and wait for the run's instances (plan §6.1 steps 4-7).

One ``RunInstances`` call per node, because each node carries its own ``Slot``/``Role`` tags and its own
idempotency token (``ClientToken`` from handle + node name [+ AZ attempt]). Tags are applied to the instance,
volume and network interface at launch. Existing nodes are found by ``Run`` + ``Node`` tags first, so a re-run
launches only what is missing; that is also how a permanently lost slot is replaced.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from botocore.exceptions import ClientError

from . import aws
from .aws_net import Ctx
from .spec import NodePlan
from .state import InstanceRecord

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SCRIPT = ROOT / "infra" / "tidb" / "node_bootstrap.sh"
ROOT_VOLUME_GB = 50
CAPACITY_ERRORS = {"InsufficientInstanceCapacity", "InsufficientHostCapacity", "Unsupported", "SpotMaxPriceTooLow"}
LIVE_STATES = ["pending", "running", "stopping", "stopped"]


def generate_ssh_key(path: Path) -> str:
    """ed25519 key pair for the run (private key 0600, never printed); returns the public key text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"powder-{path.parent.parent.name}",
                        "-f", str(path)], check=True)
    path.chmod(0o600)
    return Path(str(path) + ".pub").read_text().strip()


def key_name(handle: str) -> str:
    return f"powder-{handle}"


def ensure_key_pair(ctx: Ctx, region: str, public_key: str) -> None:
    net = ctx.state.region(region)
    if net.key_name:
        return
    ec2 = ctx.clients.ec2(region)
    name = key_name(ctx.handle)
    have = ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [name]}])["KeyPairs"]
    if not have:
        ec2.import_key_pair(KeyName=name, PublicKeyMaterial=public_key.encode(),
                            TagSpecifications=aws.tag_spec("key-pair", aws.tag_list(
                                ctx.handle, ctx.owner, ctx.created_at, "key")))
    net.key_name = name
    ctx.save()


def resolve_ami(ctx: Ctx, region: str) -> str:
    try:
        return ctx.clients.ssm(region).get_parameter(Name=aws.UBUNTU_AMI_PARAM)["Parameter"]["Value"]
    except ClientError as e:
        raise aws.AwsError(f"{region}: could not resolve the Ubuntu 24.04 AMI from SSM: {e}") from e


def az_offerings(ctx: Ctx, region: str, instance_type: str) -> list[str]:
    """AZ names of ``region`` that offer ``instance_type``."""
    ec2 = ctx.clients.ec2(region)
    out: set[str] = set()
    for page in ec2.get_paginator("describe_instance_type_offerings").paginate(
            LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": [instance_type]}]):
        out |= {o["Location"] for o in page["InstanceTypeOfferings"]}
    return sorted(out)


def choose_azs(ctx: Ctx, region: str, nodes: list[NodePlan], limit: int = 3) -> list[str]:
    """AZs of ``region`` where every instance type the run uses there is offered (at most ``limit``)."""
    types = {n.instance_type for n in nodes if n.region == region}
    common: set[str] | None = None
    for t in sorted(types):
        offered = set(az_offerings(ctx, region, t))
        common = offered if common is None else common & offered
    pinned = {n.az for n in nodes if n.region == region and n.az != "auto"}
    azs = sorted(common or set())
    if not azs:
        raise aws.AwsError(f"{region}: no availability zone offers all of {sorted(types)}")
    missing = pinned - set(azs)
    if missing:
        raise aws.AwsError(f"{region}: pinned AZ(s) {sorted(missing)} do not offer {sorted(types)}")
    keep = sorted(pinned) + [a for a in azs if a not in pinned]
    return sorted(keep[: max(limit, len(pinned))])


def run_params(ctx: Ctx, node: NodePlan, az: str, subnet_id: str, sg_id: str, ami: str, key: str, user_data: str,
               attempt: str = "") -> dict:
    """Arguments for one ``RunInstances`` call (pure, so tests can inspect them)."""
    extra = {"Node": node.name, "Role": node.role}
    if node.slot:
        extra["Slot"] = node.slot
    if node.machine:
        extra["Machine"] = node.machine
    tags = aws.tag_list(ctx.handle, ctx.owner, ctx.created_at, node.name, **extra)
    return {
        "ImageId": ami, "InstanceType": node.instance_type, "MinCount": 1, "MaxCount": 1, "KeyName": key,
        "UserData": user_data,
        "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet_id, "Groups": [sg_id],
                               "AssociatePublicIpAddress": True, "DeleteOnTermination": True}],
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
            "VolumeSize": ROOT_VOLUME_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled", "HttpPutResponseHopLimit": 2},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [{"ResourceType": rt, "Tags": tags} for rt in ("instance", "volume", "network-interface")],
        "ClientToken": aws.client_token(ctx.handle, node.name, attempt),
        "Placement": {"AvailabilityZone": az},
    }


def find_existing(ctx: Ctx, region: str, name: str) -> dict | None:
    flt = aws.run_filter(ctx.handle) + [{"Name": "tag:Node", "Values": [name]},
                                        {"Name": "instance-state-name", "Values": LIVE_STATES}]
    for res in ctx.clients.ec2(region).describe_instances(Filters=flt)["Reservations"]:
        for i in res["Instances"]:
            return i
    return None


def launch_node(ctx: Ctx, node: NodePlan, azs: list[str], first_az_index: int, ami: str, user_data: str) -> InstanceRecord:
    net = ctx.state.region(node.region)
    if node.name in ctx.state.instances and ctx.state.instances[node.name].instance_id:
        return ctx.state.instances[node.name]
    ec2 = ctx.clients.ec2(node.region)
    inst = find_existing(ctx, node.region, node.name)
    if inst is None:
        order = [node.az] if node.az != "auto" else [azs[(first_az_index + k) % len(azs)] for k in range(len(azs))]
        last: Exception | None = None
        for attempt, az in enumerate(order):
            params = run_params(ctx, node, az, net.subnets[az], net.security_group_id, ami, net.key_name, user_data,
                                attempt=az)
            try:
                inst = ec2.run_instances(**params)["Instances"][0]
                break
            except ClientError as e:
                if e.response["Error"]["Code"] not in CAPACITY_ERRORS or node.az != "auto":
                    raise
                ctx.log(f"{node.name}: no {node.instance_type} capacity in {az}; trying the next AZ")
                last = e
        if inst is None:
            raise aws.AwsError(f"{node.name}: no capacity for {node.instance_type} in any AZ of {node.region}: {last}")
    rec = InstanceRecord(
        name=node.name, role=node.role, region=node.region, instance_type=node.instance_type,
        az=inst["Placement"]["AvailabilityZone"], instance_id=inst["InstanceId"], slot=node.slot,
        machine=node.machine, state="launched")
    ctx.state.instances[node.name] = rec
    ctx.save()
    return rec


def launch_all(ctx: Ctx, nodes: list[NodePlan], azs_by_region: dict[str, list[str]]) -> None:
    user_data = BOOTSTRAP_SCRIPT.read_text()
    amis = {r: resolve_ami(ctx, r) for r in {n.region for n in nodes}}
    counters: dict[str, int] = {}
    for n in nodes:
        i = counters.get(n.region, 0)
        counters[n.region] = i + 1
        launch_node(ctx, n, azs_by_region[n.region], i, amis[n.region], user_data)


def wait_running(ctx: Ctx, status_checks: bool = True) -> None:
    """Block until every recorded instance is running (and passes status checks), then record its IPs."""
    by_region: dict[str, list[InstanceRecord]] = {}
    for rec in ctx.state.instances.values():
        if rec.state != "running":
            by_region.setdefault(rec.region, []).append(rec)
    for region, recs in by_region.items():
        ec2 = ctx.clients.ec2(region)
        ids = [r.instance_id for r in recs]
        ec2.get_waiter("instance_running").wait(InstanceIds=ids, WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        if status_checks:
            ec2.get_waiter("instance_status_ok").wait(InstanceIds=ids, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
        desc = {i["InstanceId"]: i for res in ec2.describe_instances(InstanceIds=ids)["Reservations"]
                for i in res["Instances"]}
        for r in recs:
            d = desc[r.instance_id]
            r.private_ip = d.get("PrivateIpAddress", "")
            r.public_ip = d.get("PublicIpAddress", "")
            r.state = "running"
        ctx.save()
