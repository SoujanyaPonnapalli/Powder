"""``state.json``: what exists for a run (plan §4).

Written atomically after every AWS call that creates something, so a crash never loses an id, and re-read
by every later stage. The file is the only channel between the five scripts.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STAGES = ("infra", "install")
STAGE_STATUSES = ("pending", "partial", "done", "failed")


class StateError(ValueError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class InstanceRecord:
    name: str
    role: str
    region: str
    instance_type: str
    az: str = ""
    instance_id: str = ""
    private_ip: str = ""
    public_ip: str = ""
    slot: str | None = None
    machine: str | None = None
    state: str = "planned"      # planned | launched | running | terminated


@dataclass
class RegionNet:
    vpc_id: str = ""
    igw_id: str = ""
    route_table_id: str = ""
    security_group_id: str = ""
    key_name: str = ""
    subnets: dict[str, str] = field(default_factory=dict)       # az -> subnet id
    peerings: dict[str, str] = field(default_factory=dict)      # peer region -> pcx id


@dataclass
class RunState:
    handle: str
    spec_hash: str
    created_at: str = field(default_factory=utcnow)
    schema: int = SCHEMA_VERSION
    key_path: str = ""
    est_hourly_usd: float = 0.0
    stages: dict[str, str] = field(default_factory=lambda: {s: "pending" for s in STAGES})
    regions: dict[str, RegionNet] = field(default_factory=dict)
    instances: dict[str, InstanceRecord] = field(default_factory=dict)
    experiments: list[str] = field(default_factory=list)        # result dirs, in run order

    # -- mutation helpers --------------------------------------------------
    def region(self, name: str) -> RegionNet:
        return self.regions.setdefault(name, RegionNet())

    def set_stage(self, stage: str, status: str) -> None:
        if stage not in STAGES or status not in STAGE_STATUSES:
            raise StateError(f"bad stage/status {stage!r}/{status!r}")
        self.stages[stage] = status

    def add_instance(self, rec: InstanceRecord) -> None:
        if rec.name in self.instances:
            raise StateError(f"instance {rec.name!r} already recorded")
        self.instances[rec.name] = rec

    # -- queries -----------------------------------------------------------
    def by_role(self, role: str) -> list[InstanceRecord]:
        return [i for i in self.instances.values() if i.role == role]

    def by_slot(self, slot: str) -> InstanceRecord:
        for i in self.instances.values():
            if i.slot == slot:
                return i
        raise StateError(f"no instance for slot {slot!r}")

    @property
    def controller(self) -> InstanceRecord:
        c = self.by_role("controller")
        if len(c) != 1:
            raise StateError(f"expected one controller, found {len(c)}")
        return c[0]

    def require_installed(self) -> None:
        """Experiments need a finished install on an existing deployment (plan §4.2)."""
        if self.stages.get("install") != "done":
            raise StateError(f"deployment {self.handle} is not installed (install stage: {self.stages.get('install')})")

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunState":
        if d.get("schema") != SCHEMA_VERSION:
            raise StateError(f"unsupported state schema {d.get('schema')!r} (expected {SCHEMA_VERSION})")
        d = dict(d)
        regions = {k: RegionNet(**v) for k, v in d.pop("regions", {}).items()}
        instances = {k: InstanceRecord(**v) for k, v in d.pop("instances", {}).items()}
        return cls(**d, regions=regions, instances=instances)

    def save(self, path: Path | str) -> None:
        """Atomic write: temp file in the same directory, fsync, rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=1, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: Path | str) -> "RunState":
        path = Path(path)
        if not path.exists():
            raise StateError(f"no state file at {path}: has this handle been deployed?")
        return cls.from_dict(json.loads(path.read_text()))
