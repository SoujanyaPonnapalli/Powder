"""SSH/rsync helpers that reach every node through the controller (plan §0, §6.2).

The operator's machine can only SSH to the controller's public IP (security group). Every other node is reached
by private IP over the controller (``ProxyCommand``), so TiUP, which needs private IPs, and our own commands see
the same addresses. ``Runner`` is the small interface the stages use; tests substitute a fake.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SSH_USER = "ubuntu"


class RemoteError(RuntimeError):
    pass


@dataclass
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class Runner(Protocol):
    def run(self, node: str, command: str, timeout: float = 600, check: bool = True) -> Result: ...
    def put(self, node: str, local: Path, remote: str, timeout: float = 600) -> None: ...
    def get(self, node: str, remote: str, local: Path, timeout: float = 600) -> None: ...


class SshRunner:
    """Real runner. ``hosts`` maps node name -> (private_ip, public_ip); the controller uses its public IP."""

    def __init__(self, key_path: Path, hosts: dict[str, tuple[str, str]], controller: str = "controller",
                 known_hosts: Path | None = None):
        self.key_path, self.hosts, self.controller = Path(key_path), hosts, controller
        self.known_hosts = known_hosts or self.key_path.parent / "known_hosts"

    def _base(self) -> list[str]:
        return ["-i", str(self.key_path), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "LogLevel=ERROR"]

    def _target(self, node: str) -> tuple[str, list[str]]:
        if node not in self.hosts:
            raise RemoteError(f"unknown node {node!r}")
        private, public = self.hosts[node]
        if node == self.controller:
            if not public:
                raise RemoteError("controller has no public IP recorded")
            return public, []
        if not private:
            raise RemoteError(f"node {node!r} has no private IP recorded")
        cpub = self.hosts[self.controller][1]
        proxy = "ssh " + " ".join(shlex.quote(a) for a in self._base()) + f" -W %h:%p {SSH_USER}@{cpub}"
        return private, ["-o", f"ProxyCommand={proxy}"]

    def run(self, node: str, command: str, timeout: float = 600, check: bool = True) -> Result:
        host, extra = self._target(node)
        argv = ["ssh", *self._base(), *extra, f"{SSH_USER}@{host}", command]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RemoteError(f"{node}: timed out after {timeout}s running {command!r}") from e
        r = Result(p.returncode, p.stdout, p.stderr)
        if check and r.returncode != 0:
            raise RemoteError(f"{node}: command failed ({r.returncode}): {command!r}\n{r.stderr.strip()}")
        return r

    def _rsync(self, src: str, dst: str, node: str, timeout: float) -> None:
        host, extra = self._target(node)
        ssh = "ssh " + " ".join(shlex.quote(a) for a in self._base() + extra)
        argv = ["rsync", "-az", "-e", ssh, *(src.replace("{host}", f"{SSH_USER}@{host}"),
                                              dst.replace("{host}", f"{SSH_USER}@{host}"))]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RemoteError(f"{node}: rsync failed: {p.stderr.strip()}")

    def put(self, node: str, local: Path, remote: str, timeout: float = 600) -> None:
        self._rsync(str(local), "{host}:" + remote, node, timeout)

    def get(self, node: str, remote: str, local: Path, timeout: float = 600) -> None:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._rsync("{host}:" + remote, str(local), node, timeout)


def wait_for(predicate, timeout: float, interval: float = 5.0, what: str = "condition", sleep=time.sleep,
             clock=time.monotonic):
    """Poll ``predicate`` until truthy or ``timeout`` seconds pass; raises RemoteError on timeout."""
    deadline = clock() + timeout
    last_err = None
    while True:
        try:
            v = predicate()
            if v:
                return v
        except RemoteError as e:
            last_err = e
        if clock() >= deadline:
            raise RemoteError(f"timed out after {timeout:.0f}s waiting for {what}" + (f": {last_err}" if last_err else ""))
        sleep(interval)
