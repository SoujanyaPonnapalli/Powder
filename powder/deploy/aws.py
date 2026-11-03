"""Shared AWS plumbing: client factory, tagging, discovery filters (plan §6.1).

Every resource a run creates carries ``Project=powder``, ``Run=<handle>``, ``Owner``, ``CreatedAt`` and a
``Name`` of ``<handle>-<what>``, applied **at creation** (TagSpecifications), so there is never an untagged
window in which a crash could leak a resource. Teardown and listing work from these tags alone.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any

from powder.faults.regions import AWS_US_REGIONS

PROJECT = "powder"
ALL_REGIONS = list(AWS_US_REGIONS)
UBUNTU_AMI_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
EC2_STANDARD_VCPU_QUOTA = ("ec2", "L-1216C47A")      # Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances


class AwsError(RuntimeError):
    pass


class Clients:
    """Lazy per-region boto3 clients; pass a custom ``session`` in tests."""

    def __init__(self, session=None):
        if session is None:
            import boto3
            session = boto3.session.Session()
        self.session = session
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, service: str, region: str):
        key = (service, region)
        if key not in self._cache:
            self._cache[key] = self.session.client(service, region_name=region)
        return self._cache[key]

    def ec2(self, region: str):
        return self.get("ec2", region)

    def ssm(self, region: str):
        return self.get("ssm", region)

    def sts(self):
        return self.get("sts", ALL_REGIONS[0])


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def sanitize_tag(value: str, limit: int = 120) -> str:
    return re.sub(r"[^\w\s.:/=+\-@]", "_", value)[:limit]


def caller_owner(clients: Clients) -> str:
    """Short name of the calling identity (last path element of the ARN)."""
    arn = clients.sts().get_caller_identity()["Arn"]
    return sanitize_tag(arn.rsplit("/", 1)[-1] or arn)


def tag_list(handle: str, owner: str, created_at: str, name: str, **extra: str) -> list[dict[str, str]]:
    tags = {"Project": PROJECT, "Run": handle, "Owner": owner, "CreatedAt": created_at,
            "Name": f"{handle}-{name}", **extra}
    return [{"Key": k, "Value": sanitize_tag(str(v))} for k, v in tags.items()]


def tag_spec(resource_type: str, tags: list[dict[str, str]]) -> list[dict]:
    return [{"ResourceType": resource_type, "Tags": tags}]


def run_filter(handle: str | None = None) -> list[dict]:
    f = [{"Name": "tag:Project", "Values": [PROJECT]}]
    if handle:
        f.append({"Name": "tag:Run", "Values": [handle]})
    return f


def tags_to_dict(tags: list[dict] | None) -> dict[str, str]:
    return {t["Key"]: t["Value"] for t in tags or []}


def client_token(handle: str, name: str, attempt: str = "") -> str:
    """Idempotency token for RunInstances: a retried identical call cannot double-launch."""
    return hashlib.sha256(f"{handle}:{name}:{attempt}".encode()).hexdigest()[:32]


def detect_operator_cidr(timeout: float = 10.0) -> str:
    """The caller's public IPv4 as a /32 (for SSH/Grafana/Prometheus ingress)."""
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=timeout) as r:  # noqa: S310
        ip = r.read().decode().strip()
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip):
        raise AwsError(f"could not determine the public IP (got {ip!r}); pass --operator-cidr")
    return f"{ip}/32"


def live_handles(clients: Clients, regions: list[str] | None = None) -> set[str]:
    """Handles that currently have any tagged instance or VPC in any region."""
    out: set[str] = set()
    for region in regions or ALL_REGIONS:
        ec2 = clients.ec2(region)
        for v in ec2.describe_vpcs(Filters=run_filter())["Vpcs"]:
            h = tags_to_dict(v.get("Tags")).get("Run")
            if h:
                out.add(h)
        for res in ec2.describe_instances(Filters=run_filter())["Reservations"]:
            for i in res["Instances"]:
                h = tags_to_dict(i.get("Tags")).get("Run")
                if h:
                    out.add(h)
    return out
