"""Build ``powder-load`` on the controller and distribute it to the load generators (part of the install stage).

No Go toolchain is needed on the operator's machine: sources are shipped to the controller, which downloads the
pinned Go release (sha256 verified, ``infra/tidb/loadgen/GO_PIN``), builds a static binary, and copies it to every
load-generator node over the private network. The binary reports a hash of its embedded error table; it must equal
the hash of ``powder/deploy/error_classes.json`` here, so the Go and Python classifiers can never silently drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .install import CONTROLLER_KEY
from .outcomes import TABLE_PATH
from .remote import Runner
from .state import RunState

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "infra" / "tidb" / "loadgen"
GO_PIN = SRC_DIR / "GO_PIN"
# every non-test .go file: a hand-kept list missed pool.go (G3 retry) and the remote build failed
SOURCES = sorted(p.name for p in SRC_DIR.glob("*.go") if not p.name.endswith("_test.go")) + \
    ["go.mod", "go.sum", "error_classes.json"]
REMOTE_SRC = "~/powder-src/loadgen"
REMOTE_BIN = "~/powder-bin/powder-load"


class LoadgenBuildError(RuntimeError):
    pass


def go_pin() -> dict[str, str]:
    for line in GO_PIN.read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            return {"version": parts[0], "platform": parts[1], "sha256": parts[2], "url": parts[3]}
    raise LoadgenBuildError(f"no pin in {GO_PIN}")


def table_hash() -> str:
    """What ``powder-load version`` prints after ``error_classes=``."""
    return hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()[:16]


def ensure_loadgen(runner: Runner, state: RunState, log: Callable[[str], None] = print) -> str:
    """Idempotently build and distribute powder-load; returns the version string the binary reports."""
    pin = go_pin()
    ctl = lambda cmd, t=900, check=True: runner.run("controller", cmd, timeout=t, check=check)   # noqa: E731
    log(f"[{state.handle}] building powder-load on the controller ({pin['version']})")
    ctl(f"mkdir -p {REMOTE_SRC} ~/powder-bin ~/go-dl")
    ctl(f"test -x ~/go-toolchain/go/bin/go || (cd ~/go-dl && curl -fsSL -o go.tgz {pin['url']} && "
        f"echo '{pin['sha256']}  go.tgz' | sha256sum -c - && mkdir -p ~/go-toolchain && tar -xzf go.tgz -C ~/go-toolchain)")
    for f in SOURCES:
        runner.put("controller", SRC_DIR / f, f"{REMOTE_SRC}/{f}")
    ctl(f"cd {REMOTE_SRC} && CGO_ENABLED=0 GOFLAGS=-mod=mod ~/go-toolchain/go/bin/go build -o {REMOTE_BIN} .")
    want = table_hash()
    ver = ctl(f"{REMOTE_BIN} version").stdout.strip()
    if f"error_classes={want}" not in ver:
        raise LoadgenBuildError(f"powder-load embeds a different error table than powder/deploy/error_classes.json: "
                                f"binary says {ver!r}, expected error_classes={want}")
    ssh = f"-i {CONTROLLER_KEY} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    for lg in state.by_role("loadgen"):
        ctl(f"ssh {ssh} ubuntu@{lg.private_ip} 'mkdir -p ~/powder-bin' && "
            f"rsync -a -e 'ssh {ssh}' {REMOTE_BIN} ubuntu@{lg.private_ip}:~/powder-bin/powder-load")
        got = runner.run(lg.name, f"{REMOTE_BIN} version", timeout=60).stdout.strip()
        if got != ver:
            raise LoadgenBuildError(f"{lg.name}: powder-load reports {got!r}, controller built {ver!r}")
    log(f"  powder-load on {len(state.by_role('loadgen'))} load generator(s): {ver}")
    return ver
