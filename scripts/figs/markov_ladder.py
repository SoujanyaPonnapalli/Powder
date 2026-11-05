"""Modeling stability: a ladder of availability models vs Rust Monte Carlo (MVP Fig 13, MC as reference).

    python scripts/figs/markov_ladder.py select --row outputs/figs/sensitivity/rows/x1-rho1.json \
        --row outputs/figs/deployments/rows/x30-rho1.json          # -> outputs/figs/modeling/rsms.{json,pkl}
    python scripts/figs/markov_ladder.py markov [--backend cupy] [--n 7]   # GPU VM (infra/gpu/)
    python scripts/figs/markov_ladder.py mc [--runs 1000000]              # local, Rust MC

Ladder (each step adds one modeling feature; all use the same scaled constant-rate machine configs as MC):

1. ``indep_homog``    homogeneous erasure-coded-storage model: independent nodes, every node at the RSM's mean
                      rates (H -F-> transient -> H, H -D-> data loss -> replaced after timeout+spawn+sync);
                      available iff a majority of nodes is up.
2. ``indep_hetero``   the same with each node's own rates (Poisson-binomial).
3. ``leaderless_no``  leaderless CTMC at NO_ORPHANS: replacement pipeline, out-of-date (lagging) replicas,
                      quorum-gated replacement; one rate class per machine.
4. ``raft_no``        + leader election (Raft CTMC at NO_ORPHANS).
   ``raft_full``      extra rung where the state space fits (``--max-states``): Raft at FULL.
5. MC without region/AZ outages (same failure model as the ladder) and with them (what the optimizer uses).

Every value is the time average over ``--horizon-h`` (24 h) from an all-healthy start, the same quantity MC
estimates; steady state is recorded as well. Markov rungs are timed so the runtime table comes for free.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

OUT = HERE.parents[1] / "outputs/figs/modeling"
LADDER = ("indep_homog", "indep_hetero", "leaderless_no", "raft_no", "raft_full")


# --------------------------------------------------------------------------- independent-node models

def _node_generator(r) -> np.ndarray:
    """3-state node: 0 = up, 1 = transient failure, 2 = data loss (replaced after timeout + spawn + sync)."""
    q = np.zeros((3, 3))
    q[0, 1], q[0, 2] = r["failure"], r["data_loss"]
    q[1, 0] = r["recovery"]
    q[2, 0] = r["replace"]
    q -= np.diag(q.sum(axis=1))
    return q


def node_rates(cfg, strategy) -> dict:
    from powder.markov_builders.common import extract_rates
    e = extract_rates(cfg, strategy)
    return {"failure": e.failure_rate, "data_loss": e.data_loss_rate, "recovery": e.recovery_rate,
            "replace": e.collapsed_replace_rate}


def majority_up_probability(p_up: np.ndarray) -> np.ndarray:
    """P(at least a majority of n independent nodes up); ``p_up`` has shape (n, T) -> (T,)."""
    n, t = p_up.shape
    dist = np.zeros((n + 1, t))
    dist[0] = 1.0
    for i in range(n):  # Poisson-binomial DP
        dist[1:] = dist[1:] * (1 - p_up[i]) + dist[:-1] * p_up[i]
        dist[0] = dist[0] * (1 - p_up[i])
    return dist[n // 2 + 1:].sum(axis=0)


def independent_availability(rates: list[dict], horizon_s: float, steps: int = 40_000) -> tuple[float, float]:
    """(time-averaged availability over [0, horizon) from all-up, steady-state availability)."""
    from scipy.linalg import expm

    dt = horizon_s / steps
    p_up = np.empty((len(rates), steps + 1))
    p_inf = np.empty(len(rates))
    for i, r in enumerate(rates):
        q = _node_generator(r)
        step = expm(q * dt)
        p = np.array([1.0, 0.0, 0.0])
        for k in range(steps + 1):
            p_up[i, k] = p[0]
            p = p @ step
        # stationary: up fraction of the 3-state cycle
        p_inf[i] = 1.0 / (1.0 + r["failure"] / r["recovery"] + (r["data_loss"] / r["replace"] if r["replace"] else 0))
    a = majority_up_probability(p_up)
    avg = float(np.trapezoid(a, dx=dt) / horizon_s) if hasattr(np, "trapezoid") else float(np.trapz(a, dx=dt) / horizon_s)
    steady = float(majority_up_probability(p_inf[:, None])[0])
    return avg, steady


# --------------------------------------------------------------------------- select

def cmd_select(a) -> int:
    import common as K

    rsms = []
    for row_path in a.row:
        row = K.load_row(Path(row_path))
        scale, price = float(row["rate_scale"]), row["price"]
        memo_path = Path(a.cache_dir) / f"exact-x{scale:g}.json"
        memo = json.loads(memo_path.read_text())
        by_rf: dict[int, list] = {}
        for budget, r in row["new_ilp"].items():
            if not r.get("feasible") or "placements" not in r or "carried" in r.get("source", ""):
                continue
            for p in r["placements"]:
                by_rf.setdefault(len(p), []).append((tuple(sorted(p)), budget))
        for name, b in row.get("baselines", {}).items():   # fallback only
            for p in b.get("placements", []):
                by_rf.setdefault(len(p), []).append((tuple(sorted(p)), f"baseline:{name}"))
        for rf in a.rfs:
            pool = {}
            for p, src in by_rf.get(rf, []):
                if "|".join(p) in memo and (p not in pool or not pool[p].startswith("baseline")):
                    pool.setdefault(p, src)
            ilp = {p: s for p, s in pool.items() if not s.startswith("baseline")}
            pool = ilp or pool
            if not pool:
                print(f"x{scale:g} {price}: no RF {rf} placement", flush=True)
                continue
            med = statistics.median(memo["|".join(p)] for p in pool)
            pick = min(sorted(pool), key=lambda p: abs(memo["|".join(p)] - med))
            rsms.append({"id": f"x{scale:g}-{price}-n{rf}", "rate_scale": scale, "price": price, "n": rf,
                         "machines": list(pick), "source": pool[pick], "candidates": len(pool),
                         "exact_720h": memo["|".join(pick)]})
            print(f"{rsms[-1]['id']}: {pick} from {pool[pick]} ({len(pool)} candidates, 720 h exact "
                  f"{K.nines(memo['|'.join(pick)]):.2f} nines)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rsms.json").write_text(json.dumps(rsms, indent=1) + "\n")
    # Node configs for the GPU VM, which does not need the fault library or the catalogue.
    configs = {r["id"]: [K.machines(r["rate_scale"], r["price"])[m].node_config for m in r["machines"]] for r in rsms}
    (OUT / "rsms.pkl").write_bytes(pickle.dumps(configs))
    return 0


# --------------------------------------------------------------------------- markov

def estimated_states(n: int, rung: str) -> float:
    """Upper-bound-ish state count with one class per node (k per-node states, leader fields)."""
    k = {"leaderless_no": 6, "raft_no": 6, "raft_full": 12}[rung]
    leader = {"leaderless_no": 1, "raft_no": 1 + n, "raft_full": (1 + n) * 3}[rung]
    return float(k ** n * leader)


def cmd_markov(a) -> int:
    from powder.markov_builders import build_leaderless_model, build_raft_model
    from powder.markov_solver import availability, steady_state, time_averaged_availability_uniformization
    from powder.scenario import QualityLevel
    from powder.simulation import Exponential, LeaderlessProtocol, NodeReplacementStrategy, RaftLikeProtocol

    rsms = json.loads((OUT / "rsms.json").read_text())
    configs = pickle.loads((OUT / "rsms.pkl").read_bytes())
    strategy = NodeReplacementStrategy(failure_timeout=a.timeout_s, default_node_config=None)
    raft = RaftLikeProtocol(election_time_dist=Exponential(0.2))
    horizon_s = a.horizon_h * 3600.0
    out_dir = OUT / "markov"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rsms:
        if a.n and r["n"] not in a.n:
            continue
        cfgs = configs[r["id"]]
        for rung in a.rungs:
            path = out_dir / f"{r['id']}-{rung}.json"
            if path.exists() and not a.force:
                continue
            rec = {"id": r["id"], "n": r["n"], "rung": rung, "horizon_h": a.horizon_h, "timeout_s": a.timeout_s}
            t0 = time.time()
            try:
                if rung in ("indep_homog", "indep_hetero"):
                    rates = [node_rates(c, strategy) for c in cfgs]
                    if rung == "indep_homog":
                        mean = {k: float(np.mean([x[k] for x in rates])) for k in rates[0]}
                        rates = [mean] * len(rates)
                    rec["availability"], rec["steady_availability"] = independent_availability(rates, horizon_s)
                    rec["states"] = 3 ** len(rates)
                else:
                    est = estimated_states(r["n"], rung)
                    if est > a.max_states:
                        rec.update(status="not tractable", estimated_states=est)
                        raise StopIteration
                    level = QualityLevel.FULL if rung == "raft_full" else QualityLevel.NO_ORPHANS
                    model = (build_leaderless_model(cfgs, LeaderlessProtocol(), strategy, level)
                             if rung == "leaderless_no" else build_raft_model(cfgs, raft, strategy, level))
                    rec["build_s"] = time.time() - t0
                    rec["states"], rec["nnz"] = model.num_states, int(model.Q.nnz)
                    t1 = time.time()
                    rec["availability"], info = time_averaged_availability_uniformization(
                        model, horizon_s, backend=a.backend)
                    rec["uniformization"] = info
                    rec["solve_s"] = time.time() - t1
                    t2 = time.time()
                    try:
                        rec["steady_availability"] = availability(model, steady_state(model, backend=a.steady_backend))
                    except Exception as e:  # noqa: BLE001 - steady state is a cross-check only
                        rec["steady_error"] = repr(e)
                    rec["steady_s"] = time.time() - t2
                    rec["backend"] = a.backend
                rec["status"] = "ok"
            except StopIteration:
                pass
            except MemoryError as e:
                rec.update(status="out of memory", error=repr(e))
            rec["seconds"] = time.time() - t0
            path.write_text(json.dumps(rec, indent=1) + "\n")
            msg = (f"{1 - rec['availability']:.4e} unavail, {rec.get('states')} states" if rec["status"] == "ok"
                   else rec["status"])
            print(f"{r['id']} {rung}: {msg} ({rec['seconds']:.1f}s)", flush=True)
    return 0


# --------------------------------------------------------------------------- mc

def cmd_mc(a) -> int:
    import common as K
    from concurrent.futures import ThreadPoolExecutor

    rsms = json.loads((OUT / "rsms.json").read_text())
    out_dir = OUT / "mc"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, net) for r in rsms for net in (False, True)]

    def run(job):
        r, net = job
        path = out_dir / f"{r['id']}-{'net' if net else 'nonet'}.json"
        if path.exists() and not a.force:
            return
        by_id = K.machines(r["rate_scale"], r["price"])
        st = K.mc_stats([by_id[m].node_config for m in r["machines"]], timeout_s=a.timeout_s,
                        horizon_s=a.horizon_h * 3600, runs=a.runs, seed=K.placement_seed(r["machines"]),
                        with_network=net)
        path.write_text(json.dumps({"id": r["id"], "n": r["n"], "network": net, "horizon_h": a.horizon_h,
                                    "runs": st.runs, "availability": st.mean, "se": st.se, "std": st.std,
                                    "seconds": st.seconds}, indent=1) + "\n")
        print(f"{r['id']} MC {'with' if net else 'without'} outages: {1 - st.mean:.4e} unavail "
              f"(se {st.se:.1e}, {st.seconds:.0f}s)", flush=True)

    with ThreadPoolExecutor(a.threads) as pool:
        list(pool.map(run, jobs))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--row", action="append", required=True)
    s.add_argument("--rfs", type=int, nargs="+", default=[3, 5, 7])
    s.add_argument("--cache-dir", default=str(HERE.parents[1] / "outputs/figs/cache"))
    m = sub.add_parser("markov")
    m.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    m.add_argument("--steady-backend", default="scipy")
    m.add_argument("--n", type=int, nargs="*")
    m.add_argument("--rungs", nargs="+", default=list(LADDER))
    m.add_argument("--max-states", type=float, default=6e6)
    m.add_argument("--force", action="store_true")
    c = sub.add_parser("mc")
    c.add_argument("--runs", type=int, default=1_000_000)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--force", action="store_true")
    for p in (m, c):
        p.add_argument("--horizon-h", type=float, default=24.0)
        p.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args()
    return {"select": cmd_select, "markov": cmd_markov, "mc": cmd_mc}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
