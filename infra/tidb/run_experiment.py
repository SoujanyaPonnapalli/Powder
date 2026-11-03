#!/usr/bin/env python3
"""Run an experiment on an installed deployment and collect results (exec #7 script 4).

Thin wrapper: the logic lives in powder.deploy.cli (tested). See docs/plans/exec-07-tidb-deployment.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from powder.deploy.cli import experiment_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(experiment_main())
