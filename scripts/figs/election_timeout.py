"""What-if: Raft election timeout (availability) and elections per day under a network/stall model.

    python scripts/figs/election_timeout.py [--timeouts-s 1 2 5 10 20 30 60] [--runs 500000] [--horizon-h 24]

1. Availability (Rust MC). Election downtime after a leader failure = Uniform(T, 2T) (Raft's randomized election
   timeout) + one vote round (0.1 s, ~US cross-region RTT). Same deployments, seeds and node replacement (600 s)
   for every T (common random numbers). Deployments: 1x and 30x rho1 at the 50% budget (saved placements).
   Failure-driven elections per day come from the MC's election counter.

2. Spurious elections (analytic; parameters are ASSUMPTIONS, swept over three scenarios, not sourced numbers).
   TiKV defaults: heartbeat every T/5, pre-vote + check-quorum on, so one slow follower link cannot depose the
   leader. A spurious election needs the leader cut off from a majority for longer than a follower's randomized
   timeout U ~ Uniform(T, 2T):
   * leader-side stalls (GC pauses, disk/NIC stalls, brownouts): Poisson rate r per node, duration
     D ~ Pareto(x_m, alpha); spurious rate = r * E_U[P(D > U)].
   * independent heartbeat loss p per message: a majority of follower links each lose ceil(U/h) consecutive
     heartbeats: rate ~ (1/h) * C(n-1, q) * (p^k)^q (negligible; reported for completeness).
   A stall blocks the RSM for L = min(D, U + vote). With exec #7's client policy (every request has a 5 s SLO), a
   request arriving during the stall fails only if it would wait longer than the SLO, so the stall contributes
   max(0, L - SLO) of unavailability: r * E[max(0, L - SLO)], added to the MC unavailability as an independent term.
   (Election downtime in the MC is counted in full; with T >= 5 s nearly all of it is beyond the SLO anyway.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

VOTE_S = 0.1
SLO_S = 5.0
SCENARIOS = {  # assumed leader-side stall processes (per node): rate per hour, Pareto x_m (s), alpha
    "calm": {"rate_per_h": 1 / 24, "x_m": 0.05, "alpha": 2.0},
    "typical": {"rate_per_h": 1.0, "x_m": 0.05, "alpha": 1.5},
    "noisy": {"rate_per_h": 10.0, "x_m": 0.1, "alpha": 1.2},
}
DEPLOYMENTS = {"1x": ("outputs/figs/sensitivity/rows/x1-rho1.json", "0.5"),
               "30x": ("outputs/figs/deployments/rows/x30-rho1.json", "0.5")}


def pareto_sf(x, x_m, alpha):
    x = np.asarray(x, dtype=float)
    return np.where(x <= x_m, 1.0, (x_m / np.maximum(x, 1e-300)) ** alpha)


def pareto_mean_min(c, x_m, alpha):
    """E[min(D, c)] for D ~ Pareto(x_m, alpha)."""
    c = np.asarray(c, dtype=float)
    below = c  # c <= x_m: D >= x_m >= c
    if math.isclose(alpha, 1.0):
        above = x_m + x_m * np.log(np.maximum(c, x_m) / x_m)
    else:
        above = x_m + x_m ** alpha * (np.maximum(c, x_m) ** (1 - alpha) - x_m ** (1 - alpha)) / (1 - alpha)
    return np.where(c <= x_m, below, above)


def network_model(T: float, n: int, scen: dict, p_loss: float = 1e-3, grid: int = 2001) -> dict:
    """Spurious elections/day and stall unavailability for one RSM of n replicas at election timeout T."""
    u = np.linspace(T, 2 * T, grid)                     # follower's randomized timeout
    r = scen["rate_per_h"] / 3600.0
    spurious = r * float(np.mean(pareto_sf(u, scen["x_m"], scen["alpha"]))) * 86400
    blocked = pareto_mean_min(u + VOTE_S, scen["x_m"], scen["alpha"])            # E[min(D, U + v)]
    within_slo = pareto_mean_min(np.minimum(u + VOTE_S, SLO_S), scen["x_m"], scen["alpha"])
    stall_unavail = r * float(np.mean(blocked - within_slo))                    # E[max(0, L - SLO)]
    h = T / 5.0
    q = n // 2                                           # followers that must miss (leader + q = majority)
    k = math.ceil(T / h)
    loss = (1 / h) * math.comb(n - 1, q) * (p_loss ** k) ** q * 86400
    return {"spurious_per_day": spurious, "loss_spurious_per_day": loss, "stall_unavailability": stall_unavail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeouts-s", type=float, nargs="+", default=[1, 2, 5, 10, 20, 30, 60])
    ap.add_argument("--runs", type=int, default=500_000)
    ap.add_argument("--horizon-h", type=float, default=24.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--output", default=str(K.ROOT / "outputs/figs/whatif/election_timeout.json"))
    a = ap.parse_args()
    t0 = time.time()
    out = {"timeouts_s": a.timeouts_s, "runs": a.runs, "horizon_h": a.horizon_h, "vote_round_s": VOTE_S, "client_slo_s": SLO_S,
           "election_model": "Uniform(T, 2T) + vote round", "scenarios": SCENARIOS,
           "scenario_note": "stall parameters are assumptions, not library-sourced measurements",
           "deployments": {}}
    for tag, (path, budget) in DEPLOYMENTS.items():
        row = K.load_row(K.ROOT / path)
        scale, price = float(row["rate_scale"]), row["price"]
        dep = K.deployment(row, budget)
        counts = Counter(dep)
        by = K.machines(scale, price)
        jobs = [(T, p) for T in a.timeouts_s for p in counts]

        def run(job):
            T, p = job
            return job, K.mc_stats([by[m].node_config for m in p], timeout_s=600, horizon_s=a.horizon_h * 3600,
                                   runs=a.runs, seed=K.placement_seed(p), proto=K.protocol(T, VOTE_S))

        with ThreadPoolExecutor(a.threads) as pool:
            stats = dict(pool.map(run, jobs))
        n_rsm = len(dep)
        rows = []
        for T in a.timeouts_s:
            mean = sum(c * stats[(T, p)].mean for p, c in counts.items()) / n_rsm
            se = math.sqrt(sum((c * stats[(T, p)].se) ** 2 for p, c in counts.items())) / n_rsm
            el = sum(c * stats[(T, p)].elections_mean for p, c in counts.items()) / n_rsm * 24 / a.horizon_h
            net = {name: {k: sum(c * network_model(T, len(p), sc)[k] for p, c in counts.items()) / n_rsm
                          for k in ("spurious_per_day", "loss_spurious_per_day", "stall_unavailability")}
                   for name, sc in SCENARIOS.items()}
            lo, hi = max(mean - 1.96 * se, 0.0), min(mean + 1.96 * se, 1 - 1e-15)
            rows.append({"timeout_s": T, "mean_availability": mean, "se": se, "nines": K.nines(mean),
                         "nines_ci95": [K.nines(lo), K.nines(hi)], "failure_elections_per_rsm_day": el,
                         "network": {name: {**v, "nines_with_stalls": K.nines(mean - v["stall_unavailability"])}
                                     for name, v in net.items()}})
            print(f"[{tag}] T={T:g}s: {rows[-1]['nines']:.3f} nines, {el:.4f} failure elections/RSM/day; spurious/day "
                  + ", ".join(f"{k} {v['spurious_per_day']:.3g}" for k, v in net.items()), flush=True)
        out["deployments"][tag] = {"row": path, "budget": budget, "rate_scale": scale, "price": price,
                                   "rf_counts": row["new_ilp"][budget].get("rf_counts"), "unique_placements": len(counts),
                                   "results": rows}
    out["seconds"] = time.time() - t0
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
