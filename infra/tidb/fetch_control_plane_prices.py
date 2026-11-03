#!/usr/bin/env python3
"""Fetch on-demand $/h of the small instance types used for the TiDB control plane.

    python3 infra/tidb/fetch_control_plane_prices.py [--out data/tidb/control_plane_prices.json]

The fault library only carries 8-vCPU x86 prices (exec #1), but the control plane (controller, PD, TiDB,
monitoring, load generators) runs on smaller instances, so deploy-time cost previews need their prices too.
Same public AWS Price List Bulk API and the same parser as ``scripts/fault_library/fetch_aws_prices.py``
(no credentials). Output cites its sources (offer URL + publication date per region), matching the library's
rule that every number has a source.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "fault_library"))

from fetch_aws_prices import BASE, INDEX, US_REGIONS, get_csv_offer, get_json  # noqa: E402
from powder.faults.aws_prices import parse_ondemand_offer  # noqa: E402

#: Instance types we consider for control-plane roles (vCPU counts are checked against the offer).
CANDIDATES = {
    "m6i.large": 2, "m6i.xlarge": 4, "c6i.large": 2, "c6i.xlarge": 4, "m5.large": 2, "m5.xlarge": 4,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tidb" / "control_plane_prices.json")
    args = ap.parse_args()
    index = get_json(INDEX)
    prices: dict[str, dict[str, float]] = {}
    sources: dict[str, dict] = {}
    for reg in US_REGIONS:
        rel = index["regions"][reg]["currentVersionUrl"].replace("index.json", "index.csv")
        print(f"fetching {reg}: {rel}", flush=True)
        offer = get_csv_offer(BASE + rel)
        sources[reg] = {"url": BASE + rel, "publicationDate": offer.get("publicationDate"),
                        "version": offer.get("version")}
        for vcpu in sorted(set(CANDIDATES.values())):
            for row in parse_ondemand_offer(offer, vcpu=vcpu):
                if row["instance_type"] in CANDIDATES and CANDIDATES[row["instance_type"]] == vcpu:
                    prices.setdefault(row["instance_type"], {})[reg] = row["price_per_hour"]
        del offer
    missing = [(t, r) for t in CANDIDATES for r in US_REGIONS if r not in prices.get(t, {})]
    if missing:
        print(f"WARNING: no price found for {missing}", file=sys.stderr)
    doc = {
        "kind": "control_plane_prices", "unit": "USD per hour, Linux, shared tenancy, on-demand",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vcpu": CANDIDATES, "prices": prices, "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {sum(len(v) for v in prices.values())} prices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
