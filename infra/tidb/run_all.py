#!/usr/bin/env python3
"""deploy -> install -> experiment(s) -> collect -> teardown, teardown guaranteed (exec #7 script 5).

Thin wrapper: the logic lives in powder.deploy.cli (tested). See docs/plans/exec-07-tidb-deployment.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from powder.deploy.cli import run_all_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_all_main())
