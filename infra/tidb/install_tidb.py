#!/usr/bin/env python3
"""Install and start TiDB on a deployed run with TiUP (exec #7 script 3).

Thin wrapper: the logic lives in powder.deploy.install / powder.deploy.cli (tested).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from powder.deploy.cli import install_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(install_main())
