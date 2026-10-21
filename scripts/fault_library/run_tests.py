"""Run pytest-style test modules without pytest (for minimal environments).

    python scripts/fault_library/run_tests.py [tests/test_fault_library.py ...]

Collects module-level ``test_*`` functions, runs them, prints a summary, and
exits non-zero on failure. Use real pytest where it is installed.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT = ["tests/test_fault_library.py"]


def main(paths: list[str]) -> int:
    failed = passed = 0
    for p in paths or DEFAULT:
        path = (ROOT / p).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {path.name}::{name} ({time.time() - t0:.2f}s)")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {path.name}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
