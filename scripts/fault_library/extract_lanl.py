"""LANL failure data (LA-UR-05-7318, 22 HPC systems, 1996-2005) -> source + hpc/lanl05 profiles.

    python scripts/fault_library/extract_lanl.py

Input: data/fault_library/raw/lanl/LA-UR-05-7318-failure-data-1996-2005.tar.gz
       (ftp://hpc-ftp.lanl.gov/data/failure/, fetched 2026-10-21; sha256 recorded in the source)
Outputs:
  data/fault_library/derived/lanl05_systems.json   per-system exposure, counts, fits
  data/fault_library/sources/lanl05.json            measurements (status V: computed from the raw data)
  data/fault_library/profiles/node/lanl05.json      hpc/lanl05-<system> (>= 1000 failures) + hpc/lanl05-pooled

Method (plan §7.12):
* Exposure per system = nodes x (production start .. min(decommission, 2005-11-30)); production
  'before tracking' starts at 1996-06-01 (remedy database start). Dates 'Jan-97' / '5-Jun' are
  month-year. 'current' = end of data.
* Causes: Hardware, Network, Facilities, Human Error, Undetermined -> `hw_reboot`; Software -> `sw`.
  LANL does not record transient vs permanent, so no data-loss cause is derived (noted).
* Rates per node-hour; per-system and pooled. Pooled profile = DerSimonian-Laird random-effects
  pool over small-node clusters (<= 4 processors per node), low/high = 90% prediction interval.
* rho_SW = n_SW / n_HW per system, pooled the same way.
* Durations: lognormal MLE of 'Down Time' (minutes) per cause class, small-node clusters.
* Hazard shape: Weibull MLE of per-node inter-failure gaps (last gap right-censored at exposure end).
* Frailty: negative-binomial MLE of per-node failure counts (zeros = nodes without failures).
* Co-failure: P(another node of the same system fails within 10 min) vs the independence
  baseline 1 - exp(-(N-1) lambda 10 min) -> inflation factor (exec #3 input).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults.fit import nelder_mead, random_effects_pool  # noqa: E402

RAW = ROOT / "data/fault_library/raw/lanl/LA-UR-05-7318-failure-data-1996-2005.tar.gz"
LIB = ROOT / "data/fault_library"
DATA_START = datetime(1996, 6, 1)
DATA_END = datetime(2005, 11, 30)
HW_CLASSES = ("Hardware", "Network", "Facilities", "Human Error", "Undetermined")
YEAR_H = 8766.0
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def month_year(s: str, default: datetime) -> datetime:
    s = s.strip()
    if not s or s.lower() in ("current",):
        return default
    if s.lower() == "before tracking":
        return DATA_START
    a, b = s.split("-")
    if a.isdigit():  # '5-Jun' -> 2005-06
        y, m = int(a), MONTHS[b]
    else:            # 'Jan-97' -> 1997-01
        m, y = MONTHS[a], int(b)
    y += 2000 if y < 50 else 1900
    return datetime(y, m, 1)


def ts(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%m/%d/%Y %H:%M")


def load():
    with tarfile.open(RAW) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".csv") and "/._" not in m.name)
        text = tf.extractfile(member).read().decode("latin-1")
    lines = text.splitlines()
    rows = list(csv.DictReader(lines[1:]))
    out = []
    for r in rows:
        if not r["System"].isdigit() or not r["nodes"]:
            continue
        cls = next((c for c in HW_CLASSES + ("Software",) if r[c].strip()), None)
        out.append({
            "system": int(r["System"]), "type": r["machine type"], "nodes": int(r["nodes"]),
            "procs": int(r["procstot"]), "ppn": int(r["procsinnode"]), "node": int(r["nodenumz"]),
            "prod": month_year(r["node prod"], DATA_START), "decom": month_year(r["node decom"], DATA_END),
            "start": ts(r["Prob Started (mm/dd/yy hh:mm)"]), "end": ts(r["Prob Fixed (mm/dd/yy hh:mm)"]),
            "down_min": float(r["Down Time"]) if r["Down Time"].strip() else None,
            "cls": cls, "sub": r[cls].strip() if cls else None,
        })
    return out, hashlib.sha256(RAW.read_bytes()).hexdigest()


def lognormal_mle(x):
    lx = np.log(np.asarray(x, float))
    return float(lx.mean()), float(lx.std(ddof=0))


def weibull_censored_mle(gaps, censored):
    g = np.asarray(gaps, float)
    c = np.asarray(censored, bool)

    def nll(p):
        k, lam = math.exp(p[0]), math.exp(p[1])
        z = g / lam
        ll = np.sum(~c * (math.log(k / lam) + (k - 1) * np.log(z))) - np.sum(z**k)
        return -float(ll)

    (lk, ll_), _ = nelder_mead(nll, [0.0, math.log(g.mean())], step=[0.3, 0.3])
    return math.exp(lk), math.exp(ll_)


def nb_mle(counts):
    n = np.asarray(counts, float)
    mean = n.mean()
    if n.var() <= mean:  # no overdispersion
        return mean, math.inf

    def nll(p):
        a = math.exp(p[0])
        mu = mean  # MLE of the mean is the sample mean for NB
        return -float(np.sum([math.lgamma(a + k) - math.lgamma(a) - math.lgamma(k + 1)
                              + a * math.log(a / (a + mu)) + k * math.log(mu / (a + mu)) for k in n]))

    (la,), _ = nelder_mead(nll, [0.0], step=[0.5])
    return mean, math.exp(la)


def analyse(rows):
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    systems = {}
    for s, rs in sorted(by_sys.items()):
        nodes, ppn = rs[0]["nodes"], rs[0]["ppn"]
        start = max(DATA_START, min(r["prod"] for r in rs))
        end = min(DATA_END, max(r["decom"] for r in rs))
        start = min(start, min(r["start"] for r in rs))  # failures before stated production date
        end = max(end, max(r["start"] for r in rs))
        years = (end - start).days / 365.25
        node_years = nodes * years
        counts = Counter(r["cls"] for r in rs)
        n_hw = sum(counts[c] for c in HW_CLASSES)
        n_sw = counts["Software"]
        # per-node gaps and counts
        per_node = defaultdict(list)
        for r in rs:
            per_node[r["node"]].append(r["start"])
        gaps, cens = [], []
        for t in per_node.values():
            t.sort()
            prev = start
            for x in t:
                gaps.append(max((x - prev).total_seconds() / 3600.0, 1 / 60))
                cens.append(False)
                prev = x
            gaps.append(max((end - prev).total_seconds() / 3600.0, 1 / 60))
            cens.append(True)
        k_shape, scale = weibull_censored_mle(gaps, cens) if len(rs) >= 100 else (None, None)
        node_counts = [len(v) for v in per_node.values()] + [0] * max(0, nodes - len(per_node))
        mu, a = nb_mle(node_counts) if nodes > 1 else (None, None)
        # co-failure within 10 minutes
        lam_node = len(rs) / (node_years * YEAR_H)  # per node-hour
        starts = sorted((r["start"], r["node"]) for r in rs)
        co = 0
        for i, (t, node) in enumerate(starts):
            j = i + 1
            hit = False
            while j < len(starts) and (starts[j][0] - t) <= timedelta(minutes=10):
                if starts[j][1] != node:
                    hit = True
                    break
                j += 1
            k = i - 1
            while not hit and k >= 0 and (t - starts[k][0]) <= timedelta(minutes=10):
                if starts[k][1] != node:
                    hit = True
                k -= 1
            co += hit
        base = 1 - math.exp(-(nodes - 1) * lam_node * (20 / 60)) if nodes > 1 else 0.0  # +/- 10 min window
        systems[s] = {
            "type": rs[0]["type"], "nodes": nodes, "procs": rs[0]["procs"], "procs_per_node": ppn,
            "start": start.date().isoformat(), "end": end.date().isoformat(), "node_years": node_years,
            "n_failures": len(rs), "counts": dict(counts), "n_hw": n_hw, "n_sw": n_sw,
            "rate_per_node_year": len(rs) / node_years, "hw_per_node_year": n_hw / node_years,
            "sw_per_node_year": n_sw / node_years, "rho_sw": n_sw / n_hw if n_hw else None,
            "weibull_k": k_shape, "weibull_scale_h": scale, "nb_mean": mu, "frailty_a": a,
            "cofail_10min": co / len(rs), "cofail_baseline": base,
            "cofail_inflation": (co / len(rs)) / base if base > 0 else None,
        }
    return systems


def main() -> int:
    rows, sha = load()
    systems = analyse(rows)
    small = [s for s, v in systems.items() if v["type"] == "cluster" and v["procs_per_node"] <= 4 and v["n_hw"] >= 20]
    pool = lambda key, nkey: random_effects_pool(  # noqa: E731
        [systems[s][key] for s in small], [1.0 / max(systems[s][nkey], 1) for s in small])
    hw_pool, sw_pool = pool("hw_per_node_year", "n_hw"), pool("sw_per_node_year", "n_sw")
    rho = random_effects_pool([systems[s]["rho_sw"] for s in small],
                              [1 / systems[s]["n_sw"] + 1 / systems[s]["n_hw"] for s in small])
    small_rows = [r for r in rows if r["system"] in small and r["down_min"] and r["down_min"] > 0]
    dur = {}
    for name, classes in (("hw", HW_CLASSES), ("sw", ("Software",))):
        x = [r["down_min"] * 60.0 for r in small_rows if r["cls"] in classes]
        mu, sig = lognormal_mle(x)
        dur[name] = {"mu": mu, "sigma": sig, "n": len(x), "median_s": float(np.median(x)), "mean_s": float(np.mean(x))}
    ks = [systems[s]["weibull_k"] for s in small if systems[s]["weibull_k"]]
    fa = [systems[s]["frailty_a"] for s in small if systems[s]["frailty_a"] and math.isfinite(systems[s]["frailty_a"])]
    infl = [systems[s]["cofail_inflation"] for s in small if systems[s]["cofail_inflation"]]
    summary = {"small_node_clusters": small, "hw_pool_per_node_year": hw_pool.__dict__,
               "sw_pool_per_node_year": sw_pool.__dict__, "rho_sw_pool": rho.__dict__, "durations": dur,
               "weibull_k_small": {"median": float(np.median(ks)), "min": min(ks), "max": max(ks)},
               "frailty_a_small": {"median": float(np.median(fa)), "min": min(fa), "max": max(fa)},
               "cofail_inflation_small": {"median": float(np.median(infl)), "min": min(infl), "max": max(infl)}}
    (LIB / "derived").mkdir(exist_ok=True)
    (LIB / "derived/lanl05_systems.json").write_text(json.dumps({"systems": systems, "summary": summary}, indent=1, default=str) + "\n")

    meas = [
        {"id": "dataset", "metric_verbatim": "LA-UR-05-7318 failure records, 22 systems, 1996-2005", "value": {"records": len(rows), "sha256": sha},
         "unit": "records", "status": "V", "location": "ftp://hpc-ftp.lanl.gov/data/failure/LA-UR-05-7318-failure-data-1996-2005.tar.gz", "covers": []},
        {"id": "pooled-small", "metric_verbatim": "computed: random-effects pooled rates over small-node clusters (<=4 procs/node)",
         "value": summary, "unit": "per node-year / seconds", "status": "V", "location": "derived/lanl05_systems.json",
         "covers": ["hw_reboot", "sw"], "conversion": "rate_per_hour = per_node_year / 8766"},
    ]
    for s, v in systems.items():
        meas.append({"id": f"system-{s}", "metric_verbatim": f"computed: system {s} ({v['type']}, {v['nodes']} nodes x {v['procs_per_node']} procs)",
                     "value": v, "unit": "per node-year / hours", "status": "V", "location": "derived/lanl05_systems.json",
                     "covers": ["hw_reboot", "sw"]})
    src = {"id": "lanl05", "kind": "source", "title": "LANL failure data LA-UR-05-7318 (22 HPC systems, 1996-2005)",
           "url": "https://usrc.lanl.gov/data/failure-data.php", "accessed": "2026-10-21",
           "population": "LANL HPC clusters/SMPs/NUMA, 4750 nodes, every failure needing an administrator",
           "notes": ("Root cause recorded per failure; whether it was transient or permanent is not (LANL FAQ). Failures "
                     "before the stated production date are counted from the first failure. Please acknowledge Los "
                     "Alamos for providing these data (README)."),
           "measurements": meas}
    (LIB / "sources/lanl05.json").write_text(json.dumps(src, indent=1, default=str) + "\n")

    tri = lambda lo, c, hi: {"low": lo, "central": c, "high": hi}  # noqa: E731
    dur_spec = lambda d: {"type": "lognormal", "mu": d["mu"], "sigma": d["sigma"]}  # noqa: E731
    k_med = summary["weibull_k_small"]
    profiles = [{
        "id": "hpc/lanl05-pooled", "kind": "node", "version": 1, "tier": "reference",
        "tags": ["hpc", "bare-metal", "1996-2005", "pooled", "small-node", "recoverability-unknown"],
        "derived_from": ["lanl05#pooled-small", "lanl05#dataset"], "built_by": "scripts/fault_library/extract_lanl.py",
        "notes": (f"Random-effects pool over {len(small)} LANL small-node clusters (<= 4 procs/node). All failures needing an "
                  "administrator, hardware-class (hw/network/facilities/human/undetermined) as hw_reboot and software as sw, "
                  "both transient (LANL does not record data loss). Durations: lognormal MLE of LANL down time. Absolute "
                  "rates are 1996-2005 bare metal: use for ratios and shapes, not as a cloud VM rate."),
        "causes": {
            "hw_reboot": {"rate_per_hour": tri(hw_pool.low / YEAR_H, hw_pool.central / YEAR_H, hw_pool.high / YEAR_H),
                          "duration": dur_spec(dur["hw"])},
            "sw": {"rate_per_hour": tri(sw_pool.low / YEAR_H, sw_pool.central / YEAR_H, sw_pool.high / YEAR_H),
                   "duration": dur_spec(dur["sw"])}},
        "frailty_a": summary["frailty_a_small"]["median"],
        "shape": {"type": "exponential"},
        "shape_variant": {"type": "weibull", "k": tri(k_med["min"], k_med["median"], k_med["max"])},
    }]
    for s, v in systems.items():
        if v["n_failures"] < 1000:
            continue
        rs = [r for r in rows if r["system"] == s and r["down_min"] and r["down_min"] > 0]
        d = {}
        for name, classes in (("hw", HW_CLASSES), ("sw", ("Software",))):
            x = [r["down_min"] * 60.0 for r in rs if r["cls"] in classes]
            mu, sig = lognormal_mle(x)
            d[name] = {"mu": mu, "sigma": sig}
        hw, sw = v["hw_per_node_year"] / YEAR_H, v["sw_per_node_year"] / YEAR_H
        profiles.append({
            "id": f"hpc/lanl05-sys{s}", "kind": "node", "version": 1, "tier": "reference",
            "tags": ["hpc", "bare-metal", "1996-2005", v["type"], "recoverability-unknown"], "derived_from": [f"lanl05#system-{s}"],
            "built_by": "scripts/fault_library/extract_lanl.py",
            "notes": f"LANL system {s}: {v['nodes']} nodes x {v['procs_per_node']} processors, {v['node_years']:.0f} node-years, {v['n_failures']} failures.",
            "causes": {"hw_reboot": {"rate_per_hour": tri(hw, hw, hw), "duration": dur_spec(d["hw"])},
                       "sw": {"rate_per_hour": tri(sw, sw, sw), "duration": dur_spec(d["sw"])}},
            "frailty_a": v["frailty_a"] if v["frailty_a"] and math.isfinite(v["frailty_a"]) else None,
            "shape": {"type": "exponential"},
            "shape_variant": {"type": "weibull", "k": tri(v["weibull_k"], v["weibull_k"], v["weibull_k"])} if v["weibull_k"] else None,
        })
    (LIB / "profiles/node/lanl05.json").write_text(json.dumps(profiles, indent=1) + "\n")
    print(json.dumps({"small": small, "hw": hw_pool.__dict__, "sw": sw_pool.__dict__, "rho_sw": rho.__dict__,
                      "durations": dur, "weibull_k": summary["weibull_k_small"], "frailty_a": summary["frailty_a_small"],
                      "cofail": summary["cofail_inflation_small"],
                      "per_system": {s: {k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in
                                         ("nodes", "procs_per_node", "node_years", "n_failures", "rate_per_node_year",
                                          "rho_sw", "weibull_k", "frailty_a", "cofail_inflation")} for s, v in systems.items()}},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
