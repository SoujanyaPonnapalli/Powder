import sys, json, numpy as np
from collections import Counter
sys.path.insert(0, "scripts/figs")
import common as K
from powder.mc_backend import run_monte_carlo
from powder.simulation import ClusterState, NetworkState, NodeState
row = K.load_row("outputs/figs/sensitivity/rows/x1-rho1.json")
by = K.machines(1.0, "rho1")
from powder.markov_builders.common import extract_rates
tot = Counter(); down_tot = 0.0; down_long = 0.0; N = 0
for p in sorted({tuple(sorted(p)) for p in row["new_ilp"]["0.1"]["placements"]}):
    cl = ClusterState(nodes={f"n{i}": NodeState(node_id=f"n{i}", config=by[m].node_config) for i, m in enumerate(p)},
                      network=NetworkState(), target_cluster_size=3)
    r = run_monte_carlo(cl, K.strategy(600), K.protocol(), num_simulations=300_000, max_time=720*3600,
                        stop_on_data_loss=False, seed=K.placement_seed(p) + 4 * 10**9, network_config=K.network())
    a = 1 - np.asarray(r.availability_samples); h = a * 720
    loss = np.array([t is not None for t in r.time_to_actual_loss_samples])
    ploss = np.array([t is not None for t in r.time_to_potential_loss_samples])
    long = h > 24
    N += len(h); down_tot += h.sum(); down_long += h[long].sum()
    tot["runs"] += len(h); tot["down>24h"] += int(long.sum()); tot["down>24h & actual data loss"] += int((long & loss).sum())
    tot["down>24h & potential data loss"] += int((long & ploss).sum()); tot["actual data loss"] += int(loss.sum())
    tot["permanent failures"] += int(np.sum(r.dataloss_failure_samples)); tot["spawned"] += int(np.sum(r.nodes_spawned_samples))
    rates = [extract_rates(by[m].node_config, K.strategy()) for m in p]
    tot["spot members"] += sum(int(m[1:]) >= 1000 for m in p)
print(dict(tot))
print(f"share of downtime in runs down > 24 h: {down_long / down_tot:.0%}; mean downtime/run {down_tot / N * 60:.2f} min")
print("example member rates (1st placement):", [(round(r.failure_rate*86400*365,2), round(r.data_loss_rate*86400*365,2)) for r in rates], "(transient, permanent per node-year)")
