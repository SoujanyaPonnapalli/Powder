"""Generate the machine catalogue (exec #1b §3).

    python scripts/fault_library/build_catalogue.py [--n 1000] [--seed 0] [--spot-fraction 0.3] [--horizon-days 30]
    python scripts/fault_library/build_catalogue.py --spot-twins --az-aware    # the standard 2000-VM catalogue

Writes data/fault_library/catalogue/machines-n<N>-seed<S>.json.gz and prints a summary. ``--spot-twins``
builds N on-demand machines plus one spot twin each, as machines-od<N>-spot<N>-seed<S>[-az].json.gz.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary  # noqa: E402
from powder.faults import catalogue as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spot-fraction", type=float, default=0.3)
    ap.add_argument("--horizon-days", type=float, default=30.0)
    ap.add_argument("--az-aware", action="store_true", help="bind machines across three logical AZs per region")
    ap.add_argument("--spot-twins", action="store_true", help="N on-demand machines plus a spot twin of each")
    a = ap.parse_args()
    lib = FaultLibrary.load()
    spot_fraction = 0.0 if a.spot_twins else a.spot_fraction
    cat = C.generate(lib, a.n, a.seed, spot_fraction, a.horizon_days * 24, az_aware=a.az_aware,
                     spot_twins=a.spot_twins)
    errs = C.validate(lib, cat)
    if errs:
        print("\n".join(errs[:20]), file=sys.stderr)
        return 1
    if a.spot_twins:
        name = f"machines-od{a.n}-spot{a.n}-seed{a.seed}{'-az' if a.az_aware else ''}"
    else:
        name = f"machines-n{a.n}-seed{a.seed}{'-az-v2' if a.az_aware else ''}"
    path = ROOT / f"data/fault_library/catalogue/{name}.json.gz"
    C.write(cat, path)
    ms = cat["machines"]
    yr = lambda c: np.array([m["markov_rate_per_hour"].get(c, 0.0) * 8766 for m in ms])  # noqa: E731
    q = lambda v: [round(float(x), 4) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    print(json.dumps({"path": str(path.relative_to(ROOT)), "capacity": cat["capacity"],
                      "tiers": Counter(m["tier"] for m in ms), "regions": Counter(m["region"] for m in ms),
                      "transient_bases": Counter(m["recipe"]["transient"] for m in ms),
                      "disk_components": Counter(m["recipe"]["disk"] for m in ms),
                      "per_year_p5_p50_p95": {c: q(yr(c)) for c in ("hw_reboot", "sw", "disk", "spot_interrupt")},
                      "price_assignments": cat["price_assignments"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
