#!/usr/bin/env bash
# Rebuild every derived fault-library entry, run the tests, render the catalogue.
#   scripts/fault_library/build_all.sh
# Safe to rerun: every step is deterministic. Price steps are skipped (with a
# message) until fetch_aws_prices.py / fetch_spot.py have produced raw data.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PYTHON:-python3}

$PY scripts/fault_library/build_ford10.py > /dev/null
if [ -f data/fault_library/raw/lanl/LA-UR-05-7318-failure-data-1996-2005.tar.gz ]; then
  $PY scripts/fault_library/extract_lanl.py > /dev/null
fi
$PY scripts/fault_library/build_literature.py > /dev/null
$PY scripts/fault_library/build_region_outages.py > /dev/null
if $PY scripts/fault_library/build_prices.py; then :; else
  rc=$?; [ "$rc" -eq 2 ] || exit "$rc"   # 2 = no raw prices yet
fi
$PY scripts/fault_library/build_sg07.py > /dev/null
$PY scripts/fault_library/build_rate_models.py > /dev/null
$PY scripts/fault_library/build_machines.py > /dev/null
$PY scripts/fault_library/build_catalogue.py > /dev/null
$PY scripts/fault_library/render_catalog.py
if $PY -c "import pytest" 2>/dev/null; then
  $PY -m pytest -q tests/test_fault_library.py
else
  $PY scripts/fault_library/run_tests.py
fi
