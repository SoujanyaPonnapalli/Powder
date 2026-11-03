"""Deployment spec and experiment config (plan §4, §4.2).

``ClusterSpec`` says what to *build*: slots (library machine class @ AWS region, the exact grammar of
``scripts/fault_library/make_trace.py --slot``), the control plane, TiDB settings and spend limits.
``ExperimentConfig`` says what to *do* on a built deployment.

Loading is strict: unknown keys, duplicate YAML keys, and references that the fault library cannot resolve
are errors, because a silently ignored typo in a 24-hour experiment is very expensive.
:func:`plan_nodes` expands a spec into the concrete list of machines every other module (pricing, quotas,
topology, deploy) works from.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from powder.faults.library import FaultLibrary, canonical_json
from powder.faults.regions import AWS_US_REGIONS

from . import names

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_FILE = ROOT / "third_party" / "tidb" / "VERSIONS"

ROLES = ("controller", "pd", "tikv", "tidb", "loadgen")
KEY_DISTS = ("uniform", "zipf", "hot-range", "shifting-hot-range")
ARRIVALS = ("constant", "diurnal")
WORKLOAD_TYPES = ("smoke",)
_SLOT_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
_SEMVER_RE = re.compile(r"^v\d+\.\d+\.\d+$")


class SpecError(ValueError):
    """Invalid spec or experiment config; the message lists every problem found."""


# ---------------------------------------------------------------- yaml helpers
class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (PyYAML keeps the last one silently)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise SpecError(f"duplicate key {key!r} (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_yaml(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    if isinstance(source, Path):
        text = source.read_text()
    elif "\n" not in source and Path(source).exists():
        text = Path(source).read_text()
    else:
        text = source
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader subclass)
    except yaml.YAMLError as e:
        raise SpecError(f"YAML syntax error: {e}") from e
    if not isinstance(data, dict):
        raise SpecError("top level of a spec must be a mapping")
    return data


def parse_duration(value: Any) -> float:
    """``'10m'`` -> 600.0; bare numbers are seconds. Units: s, m, h, d."""
    if isinstance(value, bool):
        raise SpecError(f"bad duration {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", str(value))
    if not m:
        raise SpecError(f"bad duration {value!r} (use e.g. 30s, 10m, 24h)")
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _check_keys(d: dict, allowed: set[str], where: str, problems: list[str]) -> None:
    for k in d:
        if k not in allowed:
            problems.append(f"{where}: unknown key {k!r} (allowed: {', '.join(sorted(allowed))})")


def pinned_tidb_version(path: Path = VERSIONS_FILE) -> str:
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "tidb":
            return parts[1]
    raise SpecError(f"no tidb row in {path}")


# ---------------------------------------------------------------- cluster spec
@dataclass
class Slot:
    name: str
    machine: str          # library machine id, e.g. "machine/ford10-google-storage"
    region: str
    az: str = "auto"      # "auto" or an explicit AZ name such as "us-east-1c"


@dataclass
class ControlPlane:
    region: str = "us-east-1"
    instance_type: str = "m6i.xlarge"
    pd: int = 3
    tidb_per_region: int = 1
    loadgen_per_region: int = 1


@dataclass
class Limits:
    max_hourly_usd: float = 25.0
    max_hours: float = 3.0


@dataclass
class ClusterSpec:
    handle: str = "auto"
    label: str = ""
    tidb_version: str = ""
    scenario: str = "scenario/canonical-central-rho0"
    seed: int = 1
    replication_factor: int = 3
    tikv_instance_type: str = "i4i.2xlarge"
    control_plane: ControlPlane = field(default_factory=ControlPlane)
    slots: list[Slot] = field(default_factory=list)
    pd_config: dict[str, Any] = field(default_factory=dict)
    tikv_config: dict[str, Any] = field(default_factory=dict)
    tidb_config: dict[str, Any] = field(default_factory=dict)
    oracle_enabled: bool = False
    limits: Limits = field(default_factory=Limits)

    @property
    def regions(self) -> list[str]:
        """Regions the run uses, in the fixed library order."""
        used = {s.region for s in self.slots} | {self.control_plane.region}
        return [r for r in AWS_US_REGIONS if r in used]

    def to_dict(self) -> dict:
        return asdict(self)

    def spec_hash(self) -> str:
        """Hash of everything that determines the deployment (not the handle or the free-text label)."""
        d = self.to_dict()
        d.pop("handle"), d.pop("label")
        return hashlib.sha256(canonical_json(d).encode()).hexdigest()[:16]


_TOP_KEYS = {"handle", "label", "tidb_version", "scenario", "seed", "replication_factor", "tikv_instance_type",
             "control_plane", "slots", "pd_config", "tikv_config", "tidb_config", "oracle", "limits"}


def _parse_slot(name: str, body: Any, problems: list[str]) -> Slot | None:
    where = f"slots.{name}"
    if not _SLOT_RE.match(str(name)):
        problems.append(f"{where}: slot names are lowercase letters, digits, '-' (max 31), start with a letter")
    if not isinstance(body, dict):
        problems.append(f"{where}: expected a mapping with 'machine: ID@REGION'")
        return None
    _check_keys(body, {"machine", "az"}, where, problems)
    m = body.get("machine")
    if not isinstance(m, str) or "@" not in m:
        problems.append(f"{where}: 'machine' must look like machine/ID@us-east-1")
        return None
    machine, region = m.rsplit("@", 1)
    return Slot(str(name), machine, region, str(body.get("az", "auto")))


def parse_spec(source: str | Path | dict, lib: FaultLibrary | None = None, check_library: bool = True) -> ClusterSpec:
    """Parse and validate a deployment spec (YAML text, file path or dict). Raises SpecError listing all problems."""
    data = load_yaml(source)
    problems: list[str] = []
    _check_keys(data, _TOP_KEYS, "spec", problems)

    spec = ClusterSpec()
    spec.handle = str(data.get("handle", "auto"))
    spec.label = str(data.get("label", ""))
    spec.tidb_version = str(data.get("tidb_version") or pinned_tidb_version())
    spec.scenario = str(data.get("scenario", spec.scenario))
    spec.seed = data.get("seed", 1)
    spec.replication_factor = data.get("replication_factor", 3)
    spec.tikv_instance_type = str(data.get("tikv_instance_type", spec.tikv_instance_type))
    for key in ("pd_config", "tikv_config", "tidb_config"):
        v = data.get(key) or {}
        if not isinstance(v, dict):
            problems.append(f"{key}: expected a mapping")
            v = {}
        setattr(spec, key, v)

    cp = data.get("control_plane") or {}
    _check_keys(cp, set(asdict(ControlPlane())), "control_plane", problems)
    spec.control_plane = ControlPlane(**{k: v for k, v in cp.items() if k in asdict(ControlPlane())})

    lim = data.get("limits") or {}
    _check_keys(lim, set(asdict(Limits())), "limits", problems)
    spec.limits = Limits(**{k: v for k, v in lim.items() if k in asdict(Limits())})

    oracle = data.get("oracle") or {}
    _check_keys(oracle, {"enabled"}, "oracle", problems)
    spec.oracle_enabled = bool(oracle.get("enabled", False))

    slots = data.get("slots")
    if not isinstance(slots, dict) or not slots:
        problems.append("slots: at least one slot is required (mapping of name -> {machine: ID@REGION})")
        slots = {}
    for name, body in slots.items():
        s = _parse_slot(name, body, problems)
        if s:
            spec.slots.append(s)

    problems += validate_spec(spec, lib, check_library)
    if problems:
        raise SpecError("invalid spec:\n  " + "\n  ".join(problems))
    return spec


def validate_spec(spec: ClusterSpec, lib: FaultLibrary | None = None, check_library: bool = True) -> list[str]:
    """Errors for an already-built spec (empty list means valid). See :func:`spec_warnings` for soft issues."""
    p: list[str] = []
    if spec.handle.lower() != "auto":
        try:
            names.validate(spec.handle)
        except names.HandleError as e:
            p.append(f"handle: {e}")
    if not _SEMVER_RE.match(spec.tidb_version):
        p.append(f"tidb_version {spec.tidb_version!r} must look like v8.5.8")
    if not isinstance(spec.seed, int) or isinstance(spec.seed, bool):
        p.append("seed must be an integer")
    rf = spec.replication_factor
    if not isinstance(rf, int) or rf < 1 or rf % 2 == 0:
        p.append(f"replication_factor must be an odd positive integer, got {rf!r}")
    cp = spec.control_plane
    if cp.region not in AWS_US_REGIONS:
        p.append(f"control_plane.region {cp.region!r} is not one of {list(AWS_US_REGIONS)}")
    for k in ("pd", "tidb_per_region", "loadgen_per_region"):
        v = getattr(cp, k)
        if not isinstance(v, int) or v < 1:
            p.append(f"control_plane.{k} must be an integer >= 1, got {v!r}")
    if isinstance(cp.pd, int) and cp.pd % 2 == 0:
        p.append(f"control_plane.pd must be odd (Raft quorum), got {cp.pd}")
    for k in ("max_hourly_usd", "max_hours"):
        v = getattr(spec.limits, k)
        if not isinstance(v, (int, float)) or v <= 0:
            p.append(f"limits.{k} must be positive, got {v!r}")

    names_seen = set()
    for s in spec.slots:
        if s.name in names_seen:
            p.append(f"duplicate slot {s.name!r}")
        names_seen.add(s.name)
        if s.region not in AWS_US_REGIONS:
            p.append(f"slots.{s.name}: region {s.region!r} is not one of {list(AWS_US_REGIONS)}")
        elif s.az != "auto" and not s.az.startswith(s.region):
            p.append(f"slots.{s.name}: az {s.az!r} is not in region {s.region}")
    if isinstance(rf, int) and len(spec.slots) < rf:
        p.append(f"{len(spec.slots)} TiKV slots cannot hold {rf} replicas; need at least {rf}")

    if check_library:
        lib = lib or FaultLibrary.load()
        try:
            scenario = lib.get(spec.scenario, "scenario")
        except Exception as e:  # LibraryError
            p.append(f"scenario: {e}")
            scenario = None
        for s in spec.slots:
            try:
                m = lib.get(s.machine, "machine")
            except Exception as e:
                p.append(f"slots.{s.name}: {e}")
                continue
            if s.region not in m.get("regions", []):
                p.append(f"slots.{s.name}: machine {s.machine} is not available in {s.region} "
                         f"(offered in {m.get('regions')})")
            if scenario is not None and s.machine not in scenario.get("machines", []):
                p.append(f"slots.{s.name}: machine {s.machine} is not in scenario {spec.scenario}")
    return p


def spec_warnings(spec: ClusterSpec, lib: FaultLibrary | None = None) -> list[str]:
    """Design smells that are legal but probably unintended."""
    w = []
    slot_regions = {s.region for s in spec.slots}
    if len(slot_regions) < spec.replication_factor:
        w.append(f"TiKV slots span {len(slot_regions)} region(s) but replication_factor is "
                 f"{spec.replication_factor}: some RSMs will keep several replicas in one region")
    if spec.control_plane.pd == 1:
        w.append("a single PD is a single point of failure")
    if spec.control_plane.pd > len(spec.regions):
        w.append(f"{spec.control_plane.pd} PD members across {len(spec.regions)} region(s): PD members share regions")
    try:
        lib = lib or FaultLibrary.load()
        dep = lib.get(spec.scenario, "scenario").get("deployment", {})
        if dep.get("instance_type") and dep["instance_type"] != spec.tikv_instance_type:
            w.append(f"tikv_instance_type {spec.tikv_instance_type} differs from the scenario's deployment "
                     f"type {dep['instance_type']} (exec #1 §3.5)")
    except Exception:
        pass
    return w


def resolve_handle(spec: ClusterSpec, taken: list[str] | set[str], rng=None) -> ClusterSpec:
    """Return a copy of ``spec`` with a concrete handle (generated if 'auto', checked if explicit)."""
    import copy
    out = copy.deepcopy(spec)
    out.handle = names.generate(taken, rng) if spec.handle.lower() == "auto" else names.ensure_new(spec.handle, taken)
    return out


# ---------------------------------------------------------------- node expansion
@dataclass(frozen=True)
class NodePlan:
    name: str                 # "controller", "pd-1", "tikv-3" (= slot), "tidb-us-east-1-1", "loadgen-us-east-1-1"
    role: str
    region: str
    instance_type: str
    slot: str | None = None
    machine: str | None = None
    az: str = "auto"


def plan_nodes(spec: ClusterSpec) -> list[NodePlan]:
    """Every machine the run needs, in a stable order (controller, PD, TiKV, TiDB, load generators)."""
    cp = spec.control_plane
    slot_regions = [r for r in AWS_US_REGIONS if r in {s.region for s in spec.slots}]
    # PD members round-robin over the regions, starting at the control-plane region
    order = [cp.region] + [r for r in slot_regions if r != cp.region]
    nodes = [NodePlan("controller", "controller", cp.region, cp.instance_type)]
    for i in range(cp.pd):
        nodes.append(NodePlan(f"pd-{i + 1}", "pd", order[i % len(order)], cp.instance_type))
    for s in spec.slots:
        nodes.append(NodePlan(s.name, "tikv", s.region, spec.tikv_instance_type, s.name, s.machine, s.az))
    for role, count in (("tidb", cp.tidb_per_region), ("loadgen", cp.loadgen_per_region)):
        for r in slot_regions:
            for j in range(count):
                nodes.append(NodePlan(f"{role}-{r}-{j + 1}", role, r, cp.instance_type))
    return nodes


# ---------------------------------------------------------------- experiment config
@dataclass
class Workload:
    type: str = "smoke"
    key_dist: str = "uniform"
    arrival: str = "constant"
    rate_per_region: float = 500.0          # requests/s issued from each load-generator region
    duration_s: float = 600.0
    keys: int = 1_000_000
    regions_target: int = 50                # pre-split key ranges (one Raft region each) the probes/requests target


@dataclass
class ExperimentConfig:
    name: str = "smoke"
    workload: Workload = field(default_factory=Workload)
    request_timeout_s: float = 5.0
    prober_interval_s: float = 1.0
    region_map_interval_s: float = 10.0
    pd_runtime_config: dict[str, Any] = field(default_factory=dict)
    injector: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


_EXP_KEYS = {"name", "workload", "request_timeout", "prober", "region_map_interval", "pd_runtime_config", "injector"}


def parse_experiment(source: str | Path | dict) -> ExperimentConfig:
    data = load_yaml(source)
    p: list[str] = []
    _check_keys(data, _EXP_KEYS, "experiment", p)
    cfg = ExperimentConfig()
    cfg.name = str(data.get("name", cfg.name))
    wl = data.get("workload") or {}
    _check_keys(wl, {"type", "key_dist", "arrival", "rate_per_region", "duration", "keys", "regions_target"},
                "workload", p)
    w = Workload()
    for k in ("type", "key_dist", "arrival"):
        setattr(w, k, str(wl.get(k, getattr(w, k))))
    for k in ("rate_per_region", "keys", "regions_target"):
        setattr(w, k, wl.get(k, getattr(w, k)))
    try:
        w.duration_s = parse_duration(wl.get("duration", w.duration_s))
        cfg.request_timeout_s = parse_duration(data.get("request_timeout", cfg.request_timeout_s))
        pr = data.get("prober") or {}
        _check_keys(pr, {"interval"}, "prober", p)
        cfg.prober_interval_s = parse_duration(pr.get("interval", cfg.prober_interval_s))
        cfg.region_map_interval_s = parse_duration(data.get("region_map_interval", cfg.region_map_interval_s))
    except SpecError as e:
        p.append(str(e))
    cfg.workload = w
    cfg.pd_runtime_config = data.get("pd_runtime_config") or {}
    cfg.injector = str(data.get("injector", "none"))

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$", cfg.name):
        p.append(f"name {cfg.name!r} must be 1-41 chars of letters, digits, '.', '_', '-'")
    if w.type not in WORKLOAD_TYPES:
        p.append(f"workload.type {w.type!r} not implemented (have: {', '.join(WORKLOAD_TYPES)})")
    if w.key_dist not in KEY_DISTS:
        p.append(f"workload.key_dist must be one of {KEY_DISTS}")
    if w.arrival not in ARRIVALS:
        p.append(f"workload.arrival must be one of {ARRIVALS}")
    for k, v in (("rate_per_region", w.rate_per_region), ("duration", w.duration_s), ("keys", w.keys),
                 ("regions_target", w.regions_target), ("request_timeout", cfg.request_timeout_s),
                 ("prober.interval", cfg.prober_interval_s)):
        if not isinstance(v, (int, float)) or v <= 0:
            p.append(f"{k} must be positive, got {v!r}")
    if cfg.region_map_interval_s < 1:
        p.append("region_map_interval must be >= 1s")
    if cfg.injector != "none" and not cfg.injector.startswith("trace:"):
        p.append("injector must be 'none' or 'trace:<file>' (the latter arrives with exec #8)")
    if p:
        raise SpecError("invalid experiment config:\n  " + "\n  ".join(p))
    return cfg
