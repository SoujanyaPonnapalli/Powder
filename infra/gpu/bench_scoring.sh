#!/usr/bin/env bash
# Old vs new fleet-ILP scoring benchmark (run ON the 32-vCPU VM, from the repo root).
# Strategies run one after another (no core sharing), each with a fresh MC cache; then the independent truth.
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate
COMMON=(--scales 1 30 --prices rho1 --budget-fractions 0.1 0.2 0.3 0.5 --workers 2 --threads 16)
OUT=outputs/bench
mkdir -p $OUT/logs
nproc > $OUT/logs/nproc.txt; lscpu > $OUT/logs/lscpu.txt; git log -1 --format=%h > $OUT/logs/commit.txt 2>/dev/null || true
run() {  # name, extra flags...
  local name=$1; shift
  local t0=$(date +%s)
  PYTHONUNBUFFERED=1 python scripts/pilot_fleet_frontier.py "${COMMON[@]}" "$@" \
    --output $OUT/$name --cache-dir $OUT/$name/cache > $OUT/logs/$name.log 2>&1
  echo "$name $(( $(date +%s) - t0 ))" >> $OUT/logs/wall.txt
}
run old
run new --min-batches 3 --rounds 15 --validate-runs 1000000
run new_is --min-batches 3 --rounds 15 --validate-runs 1000000 --is-beta 2
PYTHONUNBUFFERED=1 python scripts/figs/bench_truth.py --strategies old=$OUT/old new=$OUT/new new_is=$OUT/new_is \
  --runs 2000000 --threads 32 --output $OUT/truth.json > $OUT/logs/truth.log 2>&1
echo done > $OUT/logs/DONE
