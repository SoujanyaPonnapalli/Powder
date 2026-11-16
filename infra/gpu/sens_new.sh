#!/usr/bin/env bash
# Sensitivity sweep with the new scoring (run ON the 32-vCPU VM, from the repo root): the 8 assumed multipliers
# besides 1x (whose row is committed under outputs/figs/sensitivity-new/rows), then the independent-truth
# evaluation of all 9 points, selecting by the optimizer's believed and by its validated availability.
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate
OUT=outputs/figs/sensitivity-new
mkdir -p $OUT/logs
git log -1 --format=%h > $OUT/logs/commit.txt 2>/dev/null || cat COMMIT > $OUT/logs/commit.txt
t0=$(date +%s)
PYTHONUNBUFFERED=1 python scripts/pilot_fleet_frontier.py --scales 0.25 0.35 0.5 0.7 1.4 2 2.8 4 --prices rho1 \
  --budget-fractions 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.4 0.5 --cheapest-budgets \
  --min-batches 3 --rounds 15 --validate-runs 1000000 \
  --workers 8 --mc-threads 16 --gurobi-threads 8 \
  --output $OUT --cache-dir $OUT/cache > $OUT/logs/sweep.log 2>&1
echo "sweep $(( $(date +%s) - t0 ))" > $OUT/logs/wall.txt
for sel in believed validated; do
  PYTHONUNBUFFERED=1 python scripts/figs/sensitivity.py --rows $OUT/rows --select-by $sel --threads 32 \
    --output $OUT/sensitivity-$sel.json > $OUT/logs/eval-$sel.log 2>&1
done
echo "total $(( $(date +%s) - t0 ))" >> $OUT/logs/wall.txt
echo done > $OUT/logs/DONE
