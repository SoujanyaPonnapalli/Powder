"""AWS region -> nearest Azure region mapping (great-circle distance).

AWS does not publish data-centre coordinates, so each AWS region uses its
metro area. Azure coordinates are Azure's published region metadata
(``az account list-locations``; refresh with
``scripts/fault_library/azure_locations.py``).

Decision 2026-10-20: one Azure region per AWS region, never pooled. us-east-1
(Ashburn, VA) is equidistant from eastus and eastus2 (both Virginia) and is
fixed to **eastus2**; the other regions take the nearest Azure region.
"""

from __future__ import annotations

import math

#: AWS commercial US regions (GovCloud excluded: separate partition).
AWS_US_REGIONS = {
    "us-east-1": {"metro": "Ashburn, VA", "lat": 39.04, "lon": -77.49},
    "us-east-2": {"metro": "Columbus, OH", "lat": 39.96, "lon": -83.00},
    "us-west-1": {"metro": "San Jose, CA", "lat": 37.34, "lon": -121.89},
    "us-west-2": {"metro": "Boardman, OR", "lat": 45.84, "lon": -119.70},
}

#: Azure US public regions (published coordinates).
AZURE_US_REGIONS = {
    "eastus": (37.3719, -79.8164),
    "eastus2": (36.6681, -78.3889),
    "centralus": (41.5908, -93.6208),
    "northcentralus": (41.8819, -87.6278),
    "southcentralus": (29.4167, -98.5000),
    "westcentralus": (41.1330, -104.8200),
    "westus": (37.7830, -122.4170),
    "westus2": (47.2330, -119.8520),
    "westus3": (33.4480, -112.0740),
}

#: Explicit choices that override pure nearest-distance (with reasons).
OVERRIDES = {"us-east-1": ("eastus2", "tie with eastus (both Virginia); fixed to eastus2 by decision 2026-10-20")}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def region_map(azure: dict[str, tuple[float, float]] | None = None) -> dict[str, dict]:
    """Mapping with distances, the runner-up, and any override reason."""
    azure = azure or AZURE_US_REGIONS
    out = {}
    for aws, info in AWS_US_REGIONS.items():
        here = (info["lat"], info["lon"])
        ranked = sorted((haversine_km(here, xy), name) for name, xy in azure.items())
        chosen, reason = OVERRIDES.get(aws, (ranked[0][1], "nearest"))
        dist = dict((n, d) for d, n in ranked)
        runner = next(n for _, n in ranked if n != chosen)
        out[aws] = {
            "azure_region": chosen,
            "distance_km": round(dist[chosen], 1),
            "runner_up": runner,
            "runner_up_km": round(dist[runner], 1),
            "metro": info["metro"],
            "rule": reason,
        }
    return out
