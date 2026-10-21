"""Schema for fault-library entries, with dependency-free validation.

Every entry is a JSON object with at least ``id`` and ``kind``. Units are
fixed: durations in seconds, failure rates per hour (``rate_per_hour``),
prices in USD ($/h compute, $/GB transfer).

Triples ``{"low": a, "central": b, "high": c}`` require ``a <= b <= c``.
The scenario ``level`` picks one end; :data:`PESSIMISTIC_END` says which end
is pessimistic for each parameter type.
"""

from __future__ import annotations

from typing import Any

from .dists import validate_spec

KINDS = {
    "source",
    "node",
    "component",
    "planned_downtime",
    "region_outage",
    "correlation",
    "price",
    "machine",
    "scenario",
}

#: Causes that leave data intact (count toward the transient hazard).
TRANSIENT_CAUSES = ("hw_reboot", "sw", "pause")
#: Causes that lose the node's local NVMe data (count toward data loss).
PERMANENT_CAUSES = ("disk", "hw_host_loss", "spot_interrupt")
CAUSES = TRANSIENT_CAUSES + PERMANENT_CAUSES

LEVELS = ("low", "central", "high")
#: For each parameter type, which end of a triple is pessimistic.
PESSIMISTIC_END = {"rate": "high", "duration": "high", "weibull_k": "low"}

STATUS = {"V", "S", "U"}
TIERS = {"cloud", "reference"}
PRICE_KINDS = {
    "compute_ondemand",
    "compute_spot_series",
    "transfer",
    "price_distribution",
    "price_assignment",
}


def is_triple(value: Any) -> bool:
    return isinstance(value, dict) and set(value) >= set(LEVELS)


def check_triple(value: Any, where: str) -> list[str]:
    if not is_triple(value):
        return [f"{where}: expected a {{low, central, high}} triple, got {value!r}"]
    try:
        lo, c, hi = (float(value[k]) for k in LEVELS)
    except (TypeError, ValueError):
        return [f"{where}: triple values must be numbers"]
    if not (lo <= c <= hi):
        return [f"{where}: triple must satisfy low <= central <= high, got {lo}, {c}, {hi}"]
    if lo < 0:
        return [f"{where}: negative value {lo}"]
    return []


def pick(value: Any, level: str, param_type: str = "rate") -> Any:
    """Select a level from a triple (non-triples are returned unchanged).

    ``level`` is ``optimistic`` / ``central`` / ``pessimistic`` (scenario
    vocabulary) or directly ``low`` / ``central`` / ``high``.
    """
    if not is_triple(value):
        return value
    if level in LEVELS:
        return value[level]
    pess = PESSIMISTIC_END[param_type]
    opt = "low" if pess == "high" else "high"
    return value[{"optimistic": opt, "central": "central", "pessimistic": pess}[level]]


def _dist_or_triple(value: Any, where: str) -> list[str]:
    if is_triple(value):
        out: list[str] = []
        for k in LEVELS:
            out += validate_spec(value[k], f"{where}.{k}")
        return out
    return validate_spec(value, where)


def validate_entry(e: dict[str, Any]) -> list[str]:
    """Structural validation of one entry; returns a list of problems."""
    errs: list[str] = []
    eid = e.get("id", "<no id>")
    kind = e.get("kind")
    if not isinstance(e.get("id"), str) or not e["id"]:
        errs.append(f"{eid}: missing string id")
    if kind not in KINDS:
        return errs + [f"{eid}: unknown kind {kind!r}"]

    if kind == "source":
        for m in e.get("measurements", []):
            mid = f"{eid}#{m.get('id')}"
            if m.get("status") not in STATUS:
                errs.append(f"{mid}: status must be one of {sorted(STATUS)}")
            if "value" not in m:
                errs.append(f"{mid}: missing value")
        if not e.get("measurements"):
            errs.append(f"{eid}: source has no measurements")

    elif kind == "node":
        if e.get("tier") not in TIERS:
            errs.append(f"{eid}: tier must be one of {sorted(TIERS)}")
        causes = e.get("causes", {})
        if not causes:
            errs.append(f"{eid}: node profile has no causes")
        for c, spec in causes.items():
            if c not in CAUSES:
                errs.append(f"{eid}: unknown cause {c!r}")
                continue
            errs += check_triple(spec.get("rate_per_hour"), f"{eid}.causes.{c}.rate_per_hour")
            if c in TRANSIENT_CAUSES:
                if "duration" not in spec:
                    errs.append(f"{eid}.causes.{c}: transient cause needs a duration")
                else:
                    errs += _dist_or_triple(spec["duration"], f"{eid}.causes.{c}.duration")
        shape = e.get("shape", {"type": "exponential"})
        if shape.get("type") == "weibull":
            errs += check_triple(shape.get("k"), f"{eid}.shape.k")
        elif shape.get("type") != "exponential":
            errs.append(f"{eid}: shape.type must be exponential or weibull")
        if "spawn" in e:
            errs += validate_spec(e["spawn"], f"{eid}.spawn")

    elif kind == "component":
        errs += check_triple(e.get("rate_per_hour"), f"{eid}.rate_per_hour")

    elif kind == "planned_downtime":
        errs += validate_spec(e.get("period"), f"{eid}.period")
        errs += _dist_or_triple(e.get("duration"), f"{eid}.duration")
        if e.get("phase") not in {"equilibrium", "uniform_first_period"}:
            errs.append(f"{eid}: phase must be equilibrium or uniform_first_period")

    elif kind == "region_outage":
        regions = e.get("regions", {})
        if not regions:
            errs.append(f"{eid}: no regions")
        for r, spec in regions.items():
            errs += check_triple(spec.get("rate_per_hour"), f"{eid}.regions.{r}.rate_per_hour")
        errs += _dist_or_triple(e.get("duration"), f"{eid}.duration")

    elif kind == "price":
        if e.get("price_kind") not in PRICE_KINDS:
            errs.append(f"{eid}: price_kind must be one of {sorted(PRICE_KINDS)}")
        if e.get("price_kind") in {"compute_ondemand", "compute_spot_series", "transfer"}:
            if not e.get("fetched_at"):
                errs.append(f"{eid}: prices must carry fetched_at")

    elif kind == "machine":
        if not e.get("fault_profile"):
            errs.append(f"{eid}: machine needs fault_profile")
        route = e.get("price", {}).get("route")
        if route not in {"native", "assigned", "none"}:
            errs.append(f"{eid}: price.route must be native, assigned or none")

    elif kind == "scenario":
        if not e.get("machines"):
            errs.append(f"{eid}: scenario lists no machines")
        if e.get("level", "central") not in {"optimistic", "central", "pessimistic"}:
            errs.append(f"{eid}: level must be optimistic, central or pessimistic")
        if not e.get("deployment", {}).get("regions"):
            errs.append(f"{eid}: deployment.regions required")

    return errs
