"""Parsers for AWS price and spot data (pure functions; fetching lives in scripts/).

* :func:`parse_ondemand_offer` -- AWS Price List Bulk API per-region EC2 offer
  file -> on-demand Linux $/h rows for 8 vCPU x86 instance types
* :func:`parse_transfer_offer` -- same file -> data-transfer $/GB rows
* :func:`price_distribution` -- rows -> empirical distribution + variants
* :func:`parse_spot_advisor` -- spot-advisor-data.json -> interruption bands
* :func:`band_rates` -- band -> spot interruption rate triple (per hour)

The offer-file schema (products / terms.OnDemand / priceDimensions) is the
documented Bulk API format. Attribute values that AWS may rename are checked
explicitly, so a format change fails loudly instead of silently dropping rows.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

#: Spot Advisor band index -> (low, high) monthly interruption fraction.
#: ">20%" is open-ended; 0.20-0.40 is a documented assumption.
SPOT_BANDS = {0: (0.0, 0.05), 1: (0.05, 0.10), 2: (0.10, 0.15), 3: (0.15, 0.20), 4: (0.20, 0.40)}
HOURS_PER_MONTH = 730.0

EXCLUDED_FAMILIES = ("gpu", "machine learning", "fpga", "media accelerator", "inference", "mac")


def _usd_per_unit(term: dict[str, Any]) -> list[tuple[float, float | None, float, str]]:
    """[(begin, end, usd, unit)] for every price dimension of one term."""
    out = []
    for dim in term.get("priceDimensions", {}).values():
        usd = float(dim.get("pricePerUnit", {}).get("USD", "nan"))
        begin = float(dim.get("beginRange", 0) or 0)
        end_raw = dim.get("endRange", "Inf")
        end = None if end_raw in ("Inf", None, "") else float(end_raw)
        out.append((begin, end, usd, dim.get("unit", "")))
    return sorted(out)


def is_x86(processor: str) -> bool:
    p = processor.lower()
    return ("intel" in p or "amd" in p) and "graviton" not in p


def parse_ondemand_offer(offer: dict[str, Any], vcpu: int = 8) -> list[dict[str, Any]]:
    products = offer["products"]
    terms = offer["terms"]["OnDemand"]
    rows = []
    for sku, prod in products.items():
        a = prod.get("attributes", {})
        if prod.get("productFamily") != "Compute Instance":
            continue
        if a.get("operatingSystem") != "Linux" or a.get("tenancy") != "Shared":
            continue
        if a.get("preInstalledSw", "NA") != "NA" or a.get("capacitystatus", "Used") != "Used":
            continue
        if a.get("licenseModel", "No License required") != "No License required":
            continue
        if str(a.get("vcpu")) != str(vcpu) or not is_x86(a.get("physicalProcessor", "")):
            continue
        fam = a.get("instanceFamily", "").lower()
        if any(x in fam for x in EXCLUDED_FAMILIES) or (a.get("gpu") not in (None, "", "0", "NA")):
            continue
        if sku not in terms:
            continue
        prices = [usd for term in terms[sku].values() for (_, _, usd, unit) in _usd_per_unit(term) if unit == "Hrs"]
        if not prices or not math.isfinite(prices[0]) or prices[0] <= 0:
            continue
        itype = a["instanceType"]
        storage = a.get("storage", "")
        rows.append({
            "instance_type": itype, "region": a.get("regionCode"), "price_per_hour": prices[0],
            "family": itype.split(".")[0], "instance_family": a.get("instanceFamily"),
            "current_generation": a.get("currentGeneration") == "Yes",
            "burstable": itype.startswith(("t2.", "t3.", "t3a.")),
            "instance_store": "ebs only" not in storage.lower() and storage != "",
            "nvme": "nvme" in storage.lower(), "storage": storage,
            "memory": a.get("memory"), "processor": a.get("physicalProcessor"), "sku": sku,
        })
    # identical (type, region) rows can appear (e.g. different operation codes); keep the cheapest
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["instance_type"], r["region"])
        if k not in best or r["price_per_hour"] < best[k]["price_per_hour"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: (r["region"], r["instance_type"]))


def parse_transfer_offer(offer: dict[str, Any], regions: Iterable[str]) -> dict[str, Any]:
    """Data-transfer $/GB rows from one offer file (partial: one region's outbound SKUs).

    Returns ``{"inter_region": {src: {dst: usd}}, "inter_az": [usd...],
    "internet_egress": [usd...]}`` using the first paid tier of each SKU.
    Combine files with :func:`merge_transfer`, which validates completeness.
    """
    regions = set(regions)
    products = offer["products"]
    terms = offer["terms"]["OnDemand"]
    inter: dict[str, dict[str, float]] = {}
    intra: list[float] = []
    egress: list[float] = []
    for sku, prod in products.items():
        if prod.get("productFamily") != "Data Transfer" or sku not in terms:
            continue
        a = prod.get("attributes", {})
        src, dst = a.get("fromRegionCode"), a.get("toRegionCode")
        ttype = (a.get("transferType") or "").lower().replace(" ", "")
        tiers = [d for term in terms[sku].values() for d in _usd_per_unit(term) if d[3] == "GB"]
        paid = [d for d in tiers if d[2] > 0]
        if not paid or src not in regions:
            continue
        usd = paid[0][2]
        if dst in regions and dst != src and "interregion" in ttype:
            inter.setdefault(src, {})[dst] = usd
        elif "intraregion" in ttype:
            intra.append(usd)
        elif a.get("toLocation") == "External" or ("awsoutbound" in ttype and dst in (None, "")):
            egress.append(usd)
    return {"inter_region": inter, "inter_az": intra, "internet_egress": egress}


def merge_transfer(parts: Iterable[dict[str, Any]], regions: Iterable[str]) -> dict[str, Any]:
    """Merge per-file transfer rows and check every directed pair is priced."""
    regions = set(regions)
    inter: dict[str, dict[str, float]] = {}
    intra: list[float] = []
    egress: list[float] = []
    for p in parts:
        for src, d in p["inter_region"].items():
            inter.setdefault(src, {}).update(d)
        intra += p["inter_az"]
        egress += p["internet_egress"]
    missing = sorted(f"{s}->{d}" for s in regions for d in regions - {s} if d not in inter.get(s, {}))
    if missing:
        raise ValueError(f"inter-region transfer prices missing for {missing}; check offer attribute names")
    if not intra:
        raise ValueError("no intra-region (inter-AZ) transfer price found; check transferType values")
    return {"inter_region": inter, "inter_az": min(intra), "intra_az": 0.0,
            "internet_egress": min(egress) if egress else None}


def price_distribution(rows: list[dict[str, Any]], **filters: bool) -> dict[str, Any]:
    """Empirical $/h distribution over (type, region) rows matching filters."""
    sel = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    p = np.sort(np.array([r["price_per_hour"] for r in sel], float))
    if len(p) == 0:
        raise ValueError(f"no rows for filters {filters}")
    q = lambda v: float(np.quantile(p, v))  # noqa: E731
    lp = np.log(p)
    return {"filters": filters, "n": int(len(p)), "points": [float(v) for v in p],
            "summary": {"min": float(p[0]), "p5": q(0.05), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75),
                        "p95": q(0.95), "max": float(p[-1]), "mean": float(p.mean()),
                        "cv": float(p.std(ddof=1) / p.mean()) if len(p) > 1 else 0.0},
            "lognormal_fit": {"mu": float(lp.mean()), "sigma": float(lp.std(ddof=1)) if len(p) > 1 else 0.0},
            "instance_types": sorted({r["instance_type"] for r in sel})}


def region_factors(rows: list[dict[str, Any]], instance_type: str, base_region: str = "us-east-1") -> dict[str, float]:
    by_region = {r["region"]: r["price_per_hour"] for r in rows if r["instance_type"] == instance_type}
    if base_region not in by_region:
        raise ValueError(f"{instance_type} has no {base_region} price")
    return {reg: v / by_region[base_region] for reg, v in sorted(by_region.items())}


# -- spot ---------------------------------------------------------------------

def parse_spot_advisor(doc: dict[str, Any], regions: Iterable[str], os: str = "Linux") -> dict[tuple[str, str], dict]:
    """{(region, instance_type): {"band": i, "savings_pct": s}} from spot-advisor-data.json.

    Expected layout: ``{"ranges": [{"index", "label", "min", "max"}...],
    "spot_advisor": {region: {os: {instance_type: {"r": i, "s": s}}}}}``.
    """
    if "spot_advisor" not in doc or "ranges" not in doc:
        raise ValueError("unexpected spot advisor format (missing 'spot_advisor' or 'ranges')")
    labels = {r["index"]: r for r in doc["ranges"]}
    for i, (lo, hi) in SPOT_BANDS.items():
        if i not in labels:
            raise ValueError(f"spot advisor band {i} missing")
        if i < 4 and (abs(labels[i]["min"] / 100 - lo) > 1e-9 or abs(labels[i]["max"] / 100 - hi) > 1e-9):
            raise ValueError(f"spot advisor band {i} changed: {labels[i]}")
    out = {}
    for reg in regions:
        for itype, v in doc["spot_advisor"].get(reg, {}).get(os, {}).items():
            out[(reg, itype)] = {"band": int(v["r"]), "savings_pct": float(v.get("s", float("nan")))}
    return out


def band_rates(band: int) -> dict[str, float]:
    """Spot interruption rate triple (per hour) from a Spot Advisor band.

    Reading (plan §7.3): the band is the fraction of instances interrupted per
    month, lambda = -ln(1 - q) / 730 h. low/central/high = band low/mid/high.
    """
    lo, hi = SPOT_BANDS[band]
    f = lambda q: -math.log1p(-q) / HOURS_PER_MONTH  # noqa: E731
    return {"low": f(lo), "central": f(0.5 * (lo + hi)), "high": f(hi)}
