"""Layout of ``outputs/tidb/<handle>/`` (plan §4, §7): the only channel between the five scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .state import RunState

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "tidb"


@dataclass(frozen=True)
class RunDir:
    root: Path
    handle: str

    @property
    def path(self) -> Path:
        return self.root / self.handle

    @property
    def state_path(self) -> Path:
        return self.path / "state.json"

    @property
    def spec_path(self) -> Path:
        return self.path / "spec.yaml"

    @property
    def spec_json(self) -> Path:
        return self.path / "spec.resolved.json"

    @property
    def key_path(self) -> Path:
        return self.path / "ssh" / "id_ed25519"

    @property
    def install_dir(self) -> Path:
        return self.path / "install"

    @property
    def experiments_dir(self) -> Path:
        return self.path / "experiments"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def teardown_path(self) -> Path:
        return self.path / "teardown.json"

    def exists(self) -> bool:
        return self.path.is_dir()

    def ensure(self) -> "RunDir":
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def load_state(self) -> RunState:
        return RunState.load(self.state_path)

    def update_manifest(self, **fields) -> dict:
        m = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        m.update(fields)
        self.manifest_path.write_text(json.dumps(m, indent=1, sort_keys=True, default=str) + "\n")
        return m


def existing_handles(root: Path = DEFAULT_ROOT) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
