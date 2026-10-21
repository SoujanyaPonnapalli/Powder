#!/usr/bin/env bash
# Fetch fresh AWS price/spot data, rebuild the library, commit and push.
#   infra/aws/refresh_data.sh [advisor|prices|history|all]   (default: all)
# Runs on the VM (cron). Uses the instance role for AWS calls.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-$PWD/.venv-vm/bin/python}
WHAT=${1:-all}
LOG=$HOME/powder-logs; mkdir -p "$LOG"
exec >> "$LOG/refresh-$(date -u +%F).log" 2>&1
echo "=== $(date -u) refresh $WHAT"

git pull --rebase --autostash -q || true
case "$WHAT" in
  advisor) $PY scripts/fault_library/fetch_spot.py advisor ;;
  prices)  $PY scripts/fault_library/fetch_aws_prices.py ;;
  history) $PY scripts/fault_library/fetch_spot.py history --days 90 ;;
  all)     $PY scripts/fault_library/fetch_spot.py advisor
           $PY scripts/fault_library/fetch_aws_prices.py
           $PY scripts/fault_library/fetch_spot.py history --days 90 || echo "spot history failed (credentials?)" ;;
esac
PYTHON=$PY bash scripts/fault_library/build_all.sh
git add data/fault_library docs/fault_library.md
if ! git diff --cached --quiet; then
  git commit -q -m "Refresh fault-library data ($WHAT, $(date -u +%F))"
  git push -q origin HEAD
fi
echo "=== done $(date -u)"
