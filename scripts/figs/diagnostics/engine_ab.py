"""Same placement, same seed, same runs: committed engine vs the rebuilt (uncommitted) engine vs the cache."""
import os, sys, json, subprocess
from collections import Counter
binary, tag = sys.argv[1], sys.argv[2]
os.environ["POWDER_MC_BINARY"] = binary
sys.path.insert(0, "scripts/figs")
import common as K
row = K.load_row("outputs/figs/sensitivity/rows/x1-rho1.json")
memo = json.load(open("outputs/figs/cache/exact-x1.json"))
by = K.machines(1.0, "rho1")
from powder.faults.mc_scoring import node_config_batch_sampler
out = {}
from concurrent.futures import ThreadPoolExecutor
def run(p):
    s = node_config_batch_sampler([by[m].node_config for m in p], K.protocol(), K.strategy(600), horizon_s=720*3600, network=K.network())
    same = s(100_000, K.placement_seed(p)).mean                      # the optimizer's exact call
    fresh = [s(100_000, K.placement_seed(p) + 3 * 10**9 + i * 100_000).mean for i in range(10)]
    return p, (same, fresh)
counts = Counter(tuple(sorted(p)) for p in row["new_ilp"]["0.1"]["placements"])
with ThreadPoolExecutor(8) as ex:
    res = dict(ex.map(run, counts))
json.dump({"|".join(p): {"cached": memo["|".join(p)], "same_seed": v[0], "fresh": v[1]} for p, v in res.items()},
          open(f"/tmp/engine_ab_{tag}.json", "w"), indent=1)
print(tag, "done")
