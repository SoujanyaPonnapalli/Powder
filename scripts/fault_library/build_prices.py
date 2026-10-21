"""Turn fetched AWS price/spot data into library entries (Steps B and C).

    python scripts/fault_library/fetch_aws_prices.py      # needs network
    python scripts/fault_library/fetch_spot.py advisor    # needs network
    python scripts/fault_library/fetch_spot.py history    # needs AWS credentials
    python scripts/fault_library/build_prices.py

Reads the newest snapshot of each raw input under data/fault_library/raw/ and
writes:
  prices/aws-ondemand.json      compute_ondemand entries (pool_key ids)
  prices/aws-distribution.json  price_distribution entries (+ variants)
  prices/aws-transfer.json      transfer entry
  prices/aws-spot.json          compute_spot_series entries (deployment types)
  profiles/node/aws-spot.json   spot/aws-<type>-<region> profiles
  machines/aws-spot.json        spot machine classes (native prices)
  derived/region_factors.json   deployment-type region factors

Exits with status 2 and instructions if the raw inputs are missing.
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.aws_prices import band_rates, parse_spot_advisor, price_distribution, region_factors  # noqa: E402
from powder.faults.prices import pool_key  # noqa: E402

LIB = ROOT / "data/fault_library"
RAW = LIB / "raw"
US_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]
DEPLOYMENT_TYPES = ["i4i.2xlarge", "m6id.2xlarge"]  # primary, fallback (plan §3.5)


def newest(path: Path, pattern: str):
    found = sorted(path.glob(pattern)) if path.exists() else []
    return found[-1] if found else None


def _time_weighted(points, t_end):
    """Time-weighted mean of a step price series [(t, price)] up to t_end."""
    tot = acc = 0.0
    for (t0, p0), (t1, _) in zip(points, points[1:] + [(t_end, None)]):
        dt = max(0.0, t1 - t0)
        acc += p0 * dt
        tot += dt
    return acc / tot if tot > 0 else points[-1][1]


def spot_history_summary(date_dir: Path | None):
    """{(type, region): {"latest", "summary", ...}} from describe-spot-price-history dumps.

    Spot prices are step functions per AZ: each AZ series is time-weighted,
    then summarised across AZs (median of AZ means / latest values).
    """
    from datetime import datetime
    out = {}
    if date_dir is None:
        return out
    ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # noqa: E731
    for f in sorted(date_dir.glob("*.json")):
        region = f.stem
        hist = json.loads(f.read_text()).get("SpotPriceHistory", [])
        series: dict[str, dict[str, list]] = {}
        for h in hist:
            series.setdefault(h["InstanceType"], {}).setdefault(h["AvailabilityZone"], []).append(
                (ts(h["Timestamp"]), float(h["SpotPrice"])))
        for itype, by_az in series.items():
            t_end = max(t for pts in by_az.values() for t, _ in pts)
            az_mean, az_latest = {}, {}
            for az, pts in by_az.items():
                pts.sort()
                az_mean[az] = _time_weighted(pts, t_end)
                az_latest[az] = pts[-1][1]
            t0 = min(t for pts in by_az.values() for t, _ in pts)
            out[(itype, region)] = {
                "latest": statistics.median(az_latest.values()),
                "window": [datetime.utcfromtimestamp(t0).isoformat() + "Z", datetime.utcfromtimestamp(t_end).isoformat() + "Z"],
                "n": sum(len(p) for p in by_az.values()),
                "summary": {"mean": statistics.median(az_mean.values()), "az_time_weighted_mean": az_mean,
                            "az_latest": az_latest, "min_az_mean": min(az_mean.values()),
                            "max_az_mean": max(az_mean.values())}}
    return out


def main() -> int:
    price_dir = newest(RAW / "aws_prices", "*/ondemand_8vcpu_x86.json")
    advisor_file = newest(RAW / "spot" / "advisor", "*.json")
    if price_dir is None:
        print("no raw AWS price data: run scripts/fault_library/fetch_aws_prices.py first", file=sys.stderr)
        return 2
    date = price_dir.parent.name
    rows = json.loads(price_dir.read_text())
    meta = json.loads((price_dir.parent / "meta.json").read_text())
    transfer = json.loads((price_dir.parent / "transfer.json").read_text())
    fetched = meta["fetched_at"]

    ondemand = [{"id": pool_key(r["instance_type"], r["region"]), "kind": "price", "price_kind": "compute_ondemand",
                 "price_per_hour": r["price_per_hour"], "fetched_at": fetched, "source": meta["offers"][r["region"]]["url"],
                 "attributes": {k: r[k] for k in ("family", "instance_family", "current_generation", "burstable",
                                                  "instance_store", "nvme", "storage", "memory", "processor")}}
                for r in rows]
    variants = {"all": {}, "current-gen": {"current_generation": True},
                "current-gen-non-burstable": {"current_generation": True, "burstable": False},
                "instance-store": {"instance_store": True, "current_generation": True}}
    dists = []
    for name, filt in variants.items():
        d = price_distribution(rows, **filt)
        dists.append({"id": f"price-dist/aws-us-8vcpu-x86/{name}@{date}", "kind": "price",
                      "price_kind": "price_distribution", "fetched_at": fetched, "variant": name,
                      "default": name == "current-gen-non-burstable", **d,
                      "notes": "one point per (instance type, US region); on-demand Linux $/h"})
    xfer = {"id": f"price-transfer/aws-us@{date}", "kind": "price", "price_kind": "transfer", "fetched_at": fetched,
            "per_gb": transfer, "source": "AWS Price List Bulk API (Data Transfer SKUs)"}
    factors = {t: region_factors(rows, t) for t in DEPLOYMENT_TYPES if any(r["instance_type"] == t for r in rows)}
    (LIB / "derived").mkdir(exist_ok=True)
    (LIB / "derived/region_factors.json").write_text(json.dumps({"fetched_at": fetched, "factors": factors}, indent=2) + "\n")
    (LIB / "prices/aws-ondemand.json").write_text(json.dumps(ondemand, indent=1) + "\n")
    (LIB / "prices/aws-distribution.json").write_text(json.dumps(dists, indent=1) + "\n")
    (LIB / "prices/aws-transfer.json").write_text(json.dumps([xfer], indent=1) + "\n")
    print(f"{len(ondemand)} on-demand prices, {len(dists)} distributions, transfer, factors for {list(factors)}")

    if advisor_file is None:
        print("no Spot Advisor snapshot: run fetch_spot.py advisor to build spot profiles", file=sys.stderr)
        return 0
    adv_doc = json.loads(advisor_file.read_text())
    bands = parse_spot_advisor(adv_doc, US_REGIONS)
    band_history = {}
    for f in sorted((RAW / "spot" / "advisor").glob("*.json")):
        for k, v in parse_spot_advisor(json.loads(f.read_text()), US_REGIONS).items():
            band_history.setdefault(k, []).append(v["band"])
    hist = spot_history_summary(newest(RAW / "spot" / "history", "*"))
    base = next(p for p in json.loads((LIB / "profiles/node/aws-ondemand.json").read_text())
                if p["id"] == "cloud/aws-ondemand-nvme")
    spot_prices, profiles, machines = [], [], []
    for itype in DEPLOYMENT_TYPES:
        pool_keys = {}
        for reg in US_REGIONS:
            if (reg, itype) not in bands:
                continue
            b = bands[(reg, itype)]
            pid = f"spot/aws-{itype}-{reg}"
            prof = copy.deepcopy(base)
            prof.update({"id": pid, "tags": ["aws", "spot", "local-nvme"],
                         "derived_from": base["derived_from"] + ["cloud/aws-ondemand-nvme"],
                         "built_by": "scripts/fault_library/build_prices.py",
                         "spot": {"band_index": b["band"], "band_history": band_history.get((reg, itype), []),
                                  "advisor_snapshot": advisor_file.name, "savings_pct": b["savings_pct"]},
                         "notes": ("Hardware/software causes as cloud/aws-ondemand-nvme plus spot interruption from the "
                                   "AWS Spot Advisor band (monthly interrupted fraction; lambda = -ln(1-q)/730 h; "
                                   "band low/mid/high). Interruption loses local NVMe data.")})
            prof["causes"]["spot_interrupt"] = {"rate_per_hour": band_rates(b["band"])}
            profiles.append(prof)
            key = pool_key(itype, reg, tier="spot")
            h = hist.get((itype, reg))
            od = next((r["price_per_hour"] for r in rows if r["instance_type"] == itype and r["region"] == reg), None)
            if h is not None:
                spot_prices.append({"id": key, "kind": "price", "price_kind": "compute_spot_series", "fetched_at": fetched,
                                    "latest": h["latest"], "summary": h["summary"], "window": h["window"], "n": h["n"],
                                    "source": "aws ec2 describe-spot-price-history"})
            elif od is not None and b["savings_pct"] == b["savings_pct"]:
                est = od * (1 - b["savings_pct"] / 100.0)
                spot_prices.append({"id": key, "kind": "price", "price_kind": "compute_spot_series", "fetched_at": fetched,
                                    "latest": est, "summary": {"mean": est}, "window": None, "n": 0,
                                    "source": "on-demand x (1 - Spot Advisor 30-day average savings)",
                                    "notes": "estimate until describe-spot-price-history is fetched"})
            else:
                continue
            pool_keys[reg] = key
            machines.append({"id": f"machine/spot-{itype}-{reg}", "kind": "machine", "fault_profile": pid,
                             "tier": "spot", "storage_mode": "local", "regions": [reg],
                             "price": {"route": "native", "pool_keys": {reg: key}}})
    (LIB / "prices/aws-spot.json").write_text(json.dumps(spot_prices, indent=1) + "\n")
    (LIB / "profiles/node/aws-spot.json").write_text(json.dumps(profiles, indent=1) + "\n")
    (LIB / "machines/aws-spot.json").write_text(json.dumps(machines, indent=1) + "\n")
    print(f"{len(profiles)} spot profiles, {len(spot_prices)} spot prices, {len(machines)} spot machines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
