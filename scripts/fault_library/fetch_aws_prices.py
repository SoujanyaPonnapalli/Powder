"""Fetch AWS on-demand and data-transfer prices for US regions (Bulk API, no credentials).

    python scripts/fault_library/fetch_aws_prices.py [--date YYYY-MM-DD]

Streams the per-region EC2 offer files (CSV form) listed in the public region
index and saves the parsed rows (not the ~200 MB raw files) to
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

from powder.faults.aws_prices import merge_transfer, offer_from_csv, parse_ondemand_offer, parse_transfer_offer  # noqa: E402

BASE = "https://pricing.us-east-1.amazonaws.com"
INDEX = BASE + "/offers/v1.0/aws/AmazonEC2/current/region_index.json"
DT_INDEX = BASE + "/offers/v1.0/aws/AWSDataTransfer/current/region_index.json"
US_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "powder-fault-library"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return json.loads(data)


def get_csv_offer(url: str) -> dict:
    import io
    req = urllib.request.Request(url, headers={"User-Agent": "powder-fault-library"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return offer_from_csv(io.TextIOWrapper(r, encoding="utf-8", newline=""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = ap.parse_args()
    out = ROOT / "data/fault_library/raw/aws_prices" / args.date
    out.mkdir(parents=True, exist_ok=True)
    index = get_json(INDEX)
    rows, transfer_parts, versions = [], [], {}
    for reg in US_REGIONS:
        rel = index["regions"][reg]["currentVersionUrl"].replace("index.json", "index.csv")
        print(f"fetching {reg}: {rel}", flush=True)
        offer = get_csv_offer(BASE + rel)  # streamed: the JSON form is ~270 MB per region
        versions[reg] = {"url": BASE + rel, "publicationDate": offer.get("publicationDate"),
                         "version": offer.get("version")}
        rows += parse_ondemand_offer(offer)
        transfer_parts.append(parse_transfer_offer(offer, US_REGIONS))  # outbound SKUs of this region
        del offer
    # Data-transfer SKUs are (increasingly) published in the separate AWSDataTransfer offer; some
    # regional EC2 files carry none (e.g. us-east-2 as of 2026-10), so merge both sources.
    dt_index = get_json(DT_INDEX)
    for reg in US_REGIONS:
        rel = dt_index["regions"][reg]["currentVersionUrl"].replace("index.json", "index.csv")
        print(f"fetching data transfer {reg}: {rel}", flush=True)
        dt_offer = get_csv_offer(BASE + rel)
        versions[f"{reg}-datatransfer"] = {"url": BASE + rel, "publicationDate": dt_offer.get("publicationDate"),
                                           "version": dt_offer.get("version")}
        transfer_parts.append(parse_transfer_offer(dt_offer, US_REGIONS))
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
