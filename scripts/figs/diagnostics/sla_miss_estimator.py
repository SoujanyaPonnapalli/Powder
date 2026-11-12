import sys, json, numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts/figs")
import common as K
from powder.mc_backend import run_monte_carlo
from powder.simulation import ClusterState, NetworkState, NodeState
row = K.load_row("outputs/figs/sensitivity/rows/x1-rho1.json")
dep = row["new_ilp"]["0.1"]["placements"]
memo = json.load(open("outputs/figs/cache/exact-x1.json"))
by = K.machines(1.0, "rho1")
counts = Counter(tuple(sorted(p)) for p in dep)
H = 720 * 3600

def samples(p, n, seed, net=True):
    nodes = {f"n{i}": NodeState(node_id=f"n{i}", config=by[m].node_config) for i, m in enumerate(p)}
    cl = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=len(nodes))
    r = run_monte_carlo(cl, K.strategy(600), K.protocol(), num_simulations=n, max_time=H, stop_on_data_loss=False,
                        seed=seed, network_config=K.network() if net else None)
    return np.asarray(r.availability_samples), np.asarray(r.leader_election_samples)

def study(p):
    base = K.placement_seed(p) + 2 * 10**9
    a = np.concatenate([samples(p, 100_000, base + i * 100_000)[0] for i in range(10)])   # 1M runs, net on
    batch = 1 - a.reshape(10, -1).mean(axis=1)                                            # 10 x 100k estimates
    nonet, _ = samples(p, 1_000_000, base + 5 * 10**7, net=False)
    d = np.sort(1 - a)[::-1]
    return p, {"cached": 1 - memo["|".join(p)], "true": float(d.mean()), "batch": batch,
               "nonet": float(1 - nonet.mean()), "runs_with_downtime": float((d > 0).mean()),
               "top_0.1pct_share": float(d[: len(d) // 1000].sum() / d.sum()),
               "top_0.01pct_share": float(d[: len(d) // 10000].sum() / d.sum()),
               "max_run_down_h": float(d[0] * 720)}

with ThreadPoolExecutor(8) as ex:
    res = dict(ex.map(study, counts))
n = sum(counts.values())
agg = lambda k: sum(c * res[p][k] for p, c in counts.items()) / n
print(f"deployment unavailability: cached {agg('cached'):.2e} ({K.nines(1-agg('cached')):.2f} nines) | "
      f"1M fresh {agg('true'):.2e} ({K.nines(1-agg('true')):.2f}) | 1M no region/AZ outages {agg('nonet'):.2e} "
      f"({K.nines(1-agg('nonet')):.2f})")
med = sum(c * np.median(res[p]["batch"]) for p, c in counts.items()) / n
print(f"median single 100k estimate {med:.2e} ({K.nines(1-med):.2f} nines)")
for p, c in counts.items():
    r = res[p]; b = np.sort(r["batch"])
    rank = int((b < r["cached"]).sum())
    print(f"  true {r['true']:.2e} | 100k estimates {b.min():.1e}..{b.max():.1e} median {np.median(b):.1e} | "
          f"cached {r['cached']:.1e} (below {10-rank}/10 fresh) | no-outage {r['nonet']:.1e} | "
          f"runs w/ downtime {r['runs_with_downtime']:.1%} | top 0.1% runs {r['top_0.1pct_share']:.0%} "
          f"0.01% {r['top_0.01pct_share']:.0%} | worst run {r['max_run_down_h']:.1f} h down")
