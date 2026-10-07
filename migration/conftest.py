"""Fixtures for the migration parity suite.

Builds the Rust binary once per session and keeps a single ``--stream``
process alive, so comparing dozens of scenarios costs one process start
rather than dozens.

Part of the temporary migration harness -- see ``migration/README.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUST_DIR = REPO_ROOT / "rust"
BINARY = RUST_DIR / "target" / "release" / "powder-mc"

# Make `powder` importable when pytest is pointed straight at this directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def rust_binary() -> Path:
    """Build the release binary once and return its path."""
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "powder-mc"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"cargo build failed:\n{result.stdout}\n{result.stderr}")
    if not BINARY.exists():
        pytest.fail(f"binary not found at {BINARY}")
    return BINARY


class RustEngine:
    """A long-lived ``powder-mc --stream`` process.

    Jobs go in as NDJSON and results come back the same way, one line each
    and in order, so a single process serves the whole test session.
    """

    def __init__(self, binary: Path, workers: int = 1):
        # One worker keeps results strictly ordered and the comparison
        # single-threaded, matching how the Python reference is run.
        self._proc = subprocess.Popen(
            [str(binary), "--stream", "-j", str(workers), "--batch-size", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def run(self, job: dict) -> dict:
        """Submit one job and return its result."""
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        self._proc.stdin.write(json.dumps(job) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"powder-mc produced no output; stderr:\n{stderr}")

        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(f"powder-mc rejected the job: {result['error']}")
        return result

    def close(self) -> None:
        if self._proc.poll() is None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


@pytest.fixture(scope="session")
def rust_engine(rust_binary: Path):
    """A session-wide Rust engine process."""
    engine = RustEngine(rust_binary)
    yield engine
    engine.close()
