"""Fetch AWS on-demand and data-transfer prices for US regions (Bulk API, no credentials).

    python scripts/fault_library/fetch_aws_prices.py [--date YYYY-MM-DD]

Downloads the per-region EC2 offer files listed in the public region index
and saves the parsed rows (not the multi-GB raw files) to
data/fault_library/raw/aws_prices/<date>/{ondemand_8vcpu_x86.json,transfer.json,meta.json}.
Then run build_prices.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.aws_prices import merge_transfer, parse_ondemand_offer, parse_transfer_offer  # noqa: E402

BASE = "https://pricing.us-east-1.amazonaws.com"
INDEX = BASE + "/offers/v1.0/aws/AmazonEC2/current/region_index.json"
US_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "powder-fault-library"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return json.loads(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = ap.parse_args()
    out = ROOT / "data/fault_library/raw/aws_prices" / args.date
    out.mkdir(parents=True, exist_ok=True)
    index = get_json(INDEX)
    rows, transfer_parts, versions = [], [], {}
    for reg in US_REGIONS:
        rel = index["regions"][reg]["currentVersionUrl"]
        print(f"fetching {reg}: {rel}", flush=True)
        offer = get_json(BASE + rel)
        versions[reg] = {"url": BASE + rel, "publicationDate": offer.get("publicationDate"),
                         "version": offer.get("version")}
        rows += parse_ondemand_offer(offer)
        transfer_parts.append(parse_transfer_offer(offer, US_REGIONS))  # outbound SKUs of this region
        del offer
    transfer = merge_transfer(transfer_parts, US_REGIONS)
    meta = {"fetched_at": datetime.now(timezone.utc).isoformat(), "index": INDEX, "offers": versions,
            "filters": "Linux, shared tenancy, no pre-installed SW, capacitystatus Used, 8 vCPU, x86 (Intel/AMD), "
                       "no GPU/accelerator/FPGA/Mac families"}
    (out / "ondemand_8vcpu_x86.json").write_text(json.dumps(rows, indent=1))
    (out / "transfer.json").write_text(json.dumps(transfer, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(rows)} on-demand rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
