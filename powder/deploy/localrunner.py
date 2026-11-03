"""A :class:`Runner` that executes on this machine, so the stage code can run against a local cluster or a fake.

Every "node" is this host. Used by tests (fake ``powder-load`` + fake PD API) and by ``infra/tidb/local_pipeline_check.py``
(real TiDB from ``tiup playground``); never used for AWS deployments.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .remote import RemoteError, Result


class LocalRunner:
    def __init__(self, env: dict | None = None):
        self.env = env
        self.calls: list[tuple[str, str]] = []

    def run(self, node: str, command: str, timeout: float = 600, check: bool = True) -> Result:
        self.calls.append((node, command))
        try:
            p = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout, env=self.env)
        except subprocess.TimeoutExpired as e:
            raise RemoteError(f"{node}: timed out after {timeout}s: {command!r}") from e
        r = Result(p.returncode, p.stdout, p.stderr)
        if check and r.returncode != 0:
            raise RemoteError(f"{node}: command failed ({r.returncode}): {command!r}\n{r.stderr.strip()}")
        return r

    def put(self, node: str, local: Path, remote: str, timeout: float = 600) -> None:
        dst = Path(remote).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)

    def get(self, node: str, remote: str, local: Path, timeout: float = 600) -> None:
        src = Path(remote.rstrip("/")).expanduser()
        local = Path(local)
        if not src.exists():
            raise RemoteError(f"{node}: {remote} does not exist")
        if src.is_dir():
            shutil.copytree(src, local, dirs_exist_ok=True)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local)
