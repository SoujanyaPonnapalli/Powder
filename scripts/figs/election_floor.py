"""Leader-election floor check: MC as configured vs (mean member failure rate x 5 s) vs MC with 5 ms elections,
for the 30x and 1x rho1 50%-budget deployments (24 h). Writes outputs/figs/whatif/election_floor.json."""
import sys, json
from collections import Counter
sys.path.insert(0, "scripts/figs")
import common as K
from powder.simulation import Exponential, RaftLikeProtocol
from powder.markov_builders.common import extract_rates

rows = {"30x": ("outputs/figs/deployments/rows/x30-rho1.json", 30.0),
        "1x": ("outputs/figs/deployments/rows/x1-rho1.json", 1.0)}
out = {}
for tag, (path, scale) in rows.items():
    dep = K.deployment(K.load_row(path), "0.5")
    counts = Counter(dep)
    by = K.machines(scale, "rho1")
    res = {"slow": 0.0, "fast": 0.0, "floor": 0.0, "fast_nonet": 0.0}
    for p, c in counts.items():
        cfgs = [by[m].node_config for m in p]
        lam = sum(extract_rates(x, K.strategy()).failure_rate + extract_rates(x, K.strategy()).data_loss_rate for x in cfgs) / len(cfgs)
        floor = lam * 5.0
        K.protocol = lambda: RaftLikeProtocol(election_time_dist=Exponential(0.2))
        slow = K.mc_stats(cfgs, timeout_s=600, horizon_s=86400, runs=200_000, seed=K.placement_seed(p))
        K.protocol = lambda: RaftLikeProtocol(election_time_dist=Exponential(200.0))
        fast = K.mc_stats(cfgs, timeout_s=600, horizon_s=86400, runs=200_000, seed=K.placement_seed(p))
        fastn = K.mc_stats(cfgs, timeout_s=600, horizon_s=86400, runs=200_000, seed=K.placement_seed(p), with_network=False)
        res["slow"] += c * (1 - slow.mean); res["fast"] += c * (1 - fast.mean)
        res["floor"] += c * floor; res["fast_nonet"] += c * (1 - fastn.mean)
        print(f"{tag} n={len(p)} x{c}: MC {1-slow.mean:.2e} | floor {floor:.2e} | fast elections {1-fast.mean:.2e} (no outages {1-fastn.mean:.2e})", flush=True)
    n = sum(counts.values())
    res = {k: v / n for k, v in res.items()}
    out[tag] = {k: {"unavailability": v, "nines": K.nines(1 - v)} for k, v in res.items()}
    print(tag, {k: round(K.nines(1 - v), 2) for k, v in res.items()}, flush=True)
json.dump(out, open("outputs/figs/whatif/election_floor.json", "w"), indent=1)
