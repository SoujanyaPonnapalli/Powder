"""Refresh Azure region coordinates from `az account list-locations`.

    az login   # once
    python scripts/fault_library/azure_locations.py

Writes data/fault_library/raw/azure_locations.json and prints any US region
whose coordinates differ from powder.faults.regions.AZURE_US_REGIONS by more
than 0.01 degrees, plus the AWS->Azure mapping recomputed with the fresh
coordinates (the us-east-1 -> eastus2 override is a fixed decision).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.regions import AZURE_US_REGIONS, region_map  # noqa: E402


def main() -> int:
    raw = json.loads(subprocess.run(["az", "account", "list-locations", "-o", "json"],
                                    check=True, capture_output=True, text=True).stdout)
    out = ROOT / "data/fault_library/raw/azure_locations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "locations": raw}, indent=1))
    fresh = {}
    for loc in raw:
        meta = loc.get("metadata") or {}
        if loc.get("name") in AZURE_US_REGIONS and meta.get("latitude"):
            fresh[loc["name"]] = (float(meta["latitude"]), float(meta["longitude"]))
    for name, (lat, lon) in fresh.items():
        old = AZURE_US_REGIONS[name]
        if abs(old[0] - lat) > 0.01 or abs(old[1] - lon) > 0.01:
            print(f"CHANGED {name}: {old} -> {(lat, lon)}")
    print(json.dumps(region_map({**AZURE_US_REGIONS, **fresh}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
