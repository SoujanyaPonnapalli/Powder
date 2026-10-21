"""Price resolution for machine classes and transfer costs.

Price entries (``kind: price``) come in five ``price_kind`` flavours:

* ``compute_ondemand``     -- ``price_per_hour`` for one pool_key
* ``compute_spot_series``  -- ``latest`` and ``summary`` ($/h) for one pool_key,
                              plus a pointer to the full series in ``raw/``
* ``transfer``             -- ``per_gb`` table: intra_az, inter_az,
                              inter_region[src][dst], internet_egress
* ``price_distribution``   -- empirical $/h points of a set of instance types
* ``price_assignment``     -- frozen base $/h per machine class + region factors

``pool_key = aws:<instance_type>:<region>[:<az>]:<tier>:linux``.
"""

from __future__ import annotations

from typing import Any

from .library import FaultLibrary, LibraryError


def pool_key(instance_type: str, region: str, tier: str = "ondemand", az: str | None = None,
             provider: str = "aws", os: str = "linux") -> str:
    parts = [provider, instance_type, region] + ([az] if az else []) + [tier, os]
    return ":".join(parts)


def _native_price(entry: dict[str, Any], policy: str) -> float:
    kind = entry["price_kind"]
    if kind == "compute_ondemand":
        return float(entry["price_per_hour"])
    if kind == "compute_spot_series":
        if policy in ("latest", None):
            return float(entry["latest"])
        if policy == "mean_over_window":
            return float(entry["summary"]["mean"])
        raise LibraryError(f"price_policy {policy!r} needs a timestamp lookup; use spot_price_at()")
    raise LibraryError(f"{entry['id']} is a {kind}, not a compute price")


def price_per_hour(lib: FaultLibrary, scenario: dict[str, Any], machine_id: str, region: str) -> float:
    """$/h of a machine class in a region, under the scenario's policy."""
    policy = scenario.get("price_policy", "latest")
    if policy == "none":
        return 0.0
    m = lib.get(machine_id, "machine")
    route = m["price"]["route"]
    if route == "native":
        return _native_price(lib.get(m["price"]["pool_keys"][region], "price"), policy)
    if route == "assigned":
        a = lib.get(scenario["price_assignment"], "price")
        base = float(a["assignments"][machine_id])
        return base * float(a.get("region_factors", {}).get(region, 1.0))
    raise LibraryError(f"machine {machine_id} has no price but scenario {scenario['id']} needs one")


def transfer_per_gb(lib: FaultLibrary, transfer_id: str, src: str, dst: str,
                    same_az: bool = False) -> float:
    """$/GB to move data from region ``src`` to region ``dst``."""
    t = lib.get(transfer_id, "price")["per_gb"]
    if src == dst:
        return float(t["intra_az"] if same_az else t["inter_az"])
    return float(t["inter_region"][src][dst])
