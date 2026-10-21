"""Fetch AWS spot data: Spot Advisor bands, SpotLake 2025 history, recent spot prices.

    python scripts/fault_library/fetch_spot.py advisor            # daily (cron): band snapshot
    python scripts/fault_library/fetch_spot.py spotlake           # once: SpotLake AWS 2025 (public S3)
    python scripts/fault_library/fetch_spot.py history [--days 90] # describe-spot-price-history (AWS creds)

Outputs under data/fault_library/raw/spot/. Spot Advisor history cannot be
fetched retroactively, so the daily snapshot should start as early as
possible (see infra/aws/cron).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/fault_library/raw/spot"
US_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
ADVISOR_URL = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"
SPOTLAKE_S3 = "s3://spotlake-open-data/data/aws/2025/"
#: Instance types whose spot history we keep (deployment type + fallback);
#: extend as needed. All 8 vCPU x86 types are kept from the Spot Advisor.
HISTORY_TYPES = ["i4i.2xlarge", "m6id.2xlarge"]


def advisor() -> None:
    now = datetime.now(timezone.utc)
    out = RAW / "advisor" / f"{now.date().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ADVISOR_URL, timeout=120) as r:
        doc = json.loads(r.read())
    keep = {"fetched_at": now.isoformat(), "ranges": doc["ranges"],
            "spot_advisor": {reg: doc["spot_advisor"].get(reg, {}) for reg in US_REGIONS},
            "instance_types": doc.get("instance_types", {})}
    out.write_text(json.dumps(keep))
    print(f"spot advisor snapshot -> {out}")


def spotlake() -> None:
    dest = RAW / "spotlake" / "aws-2025"
    dest.mkdir(parents=True, exist_ok=True)
    # The bucket holds 10-minute snapshots for all of 2025; sync is resumable.
    subprocess.run(["aws", "s3", "sync", "--no-sign-request", SPOTLAKE_S3, str(dest)], check=True)
    print(f"spotlake -> {dest}")


def history(days: int) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    out = RAW / "history" / f"{end.date().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    for reg in US_REGIONS:
        cmd = ["aws", "ec2", "describe-spot-price-history", "--region", reg,
               "--instance-types", *HISTORY_TYPES, "--product-descriptions", "Linux/UNIX",
               "--start-time", start.isoformat(), "--end-time", end.isoformat(), "--output", "json"]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        (out / f"{reg}.json").write_text(res.stdout)
        print(f"spot history {reg} -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["advisor", "spotlake", "history"])
    ap.add_argument("--days", type=int, default=90)
    a = ap.parse_args()
    {"advisor": advisor, "spotlake": spotlake}.get(a.what, lambda: history(a.days))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
