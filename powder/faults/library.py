"""Load, index and validate the fault library directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import validate_entry

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
#: Sub-directories holding raw inputs or derived artefacts, not entries.
_SKIP_DIRS = {"raw", "digitized", "derived"}


class LibraryError(ValueError):
    pass


def default_root() -> Path:
    return _REPO_ROOT / "data" / "fault_library"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class FaultLibrary:
    """In-memory index of every entry under ``root``.

    Entries are found by scanning ``*.json`` recursively (skipping ``raw/``
    and ``digitized/``). A file holds one entry, or a list of entries.
    IDs must be unique across the library.
    """

    def __init__(self, entries: dict[str, dict[str, Any]], paths: dict[str, Path], root: Path):
        self.entries = entries
        self.paths = paths
        self.root = root

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, root: Path | str | None = None, validate: bool = True) -> "FaultLibrary":
        root = Path(root) if root is not None else default_root()
        if not root.is_dir():
            raise LibraryError(f"library root {root} does not exist")
        entries: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        for path in sorted(root.rglob("*.json")):
            if _SKIP_DIRS & set(path.relative_to(root).parts[:-1]):
                continue
            data = json.loads(path.read_text())
            for e in data if isinstance(data, list) else [data]:
                if not isinstance(e, dict) or "kind" not in e:
                    raise LibraryError(f"{path}: not a library entry (missing 'kind'); "
                                       "put derived files under derived/ or raw/")
                eid = e.get("id")
                if eid in entries:
                    raise LibraryError(f"duplicate id {eid!r} in {path} and {paths[eid]}")
                entries[eid] = e
                paths[eid] = path
        lib = cls(entries, paths, root)
        if validate:
            problems = lib.validate()
            if problems:
                raise LibraryError("library validation failed:\n  " + "\n  ".join(problems))
        return lib

    # -- access ------------------------------------------------------------
    def get(self, eid: str, kind: str | None = None) -> dict[str, Any]:
        if eid not in self.entries:
            raise LibraryError(f"no entry {eid!r}")
        e = self.entries[eid]
        if kind is not None and e["kind"] != kind:
            raise LibraryError(f"{eid!r} is a {e['kind']}, expected {kind}")
        return e

    def query(self, kind: str | None = None, tags: Iterable[str] = ()) -> list[dict[str, Any]]:
        tags = set(tags)
        return [
            e for e in self.entries.values()
            if (kind is None or e["kind"] == kind) and tags <= set(e.get("tags", []))
        ]

    def content_hash(self, eid: str) -> str:
        return hashlib.sha256(canonical_json(self.get(eid)).encode()).hexdigest()[:16]

    def measurement(self, ref: str) -> dict[str, Any]:
        """Resolve ``"source_id#measurement_id"``."""
        sid, _, mid = ref.partition("#")
        src = self.get(sid, "source")
        for m in src.get("measurements", []):
            if m.get("id") == mid:
                return m
        raise LibraryError(f"{sid} has no measurement {mid!r}")

    def lineage(self, eid: str, _depth: int = 0) -> list[str]:
        """Indented provenance tree (entry -> derived_from -> ...)."""
        e = self.get(eid)
        lines = ["  " * _depth + f"{eid} [{e['kind']}]"]
        for ref in e.get("derived_from", []):
            base = ref.partition("#")[0]
            if ref != base:
                m = self.measurement(ref)
                lines.append("  " * (_depth + 1) + f"{ref} ({m.get('status')}): {m.get('value')}")
            else:
                lines += self.lineage(base, _depth + 1)
        return lines

    # -- validation --------------------------------------------------------
    def validate(self) -> list[str]:
        problems: list[str] = []
        for e in self.entries.values():
            problems += validate_entry(e)
        problems += self._check_references()
        return problems

    def _ref_problems(self, eid: str, ref: str) -> list[str]:
        base, _, mid = ref.partition("#")
        if base not in self.entries:
            return [f"{eid}: derived_from {ref!r} does not resolve"]
        if mid:
            try:
                m = self.measurement(ref)
            except LibraryError as err:
                return [f"{eid}: {err}"]
            if m.get("status") == "U":
                return [f"{eid}: cites unconfirmed (U) measurement {ref}"]
        return []

    def _check_references(self) -> list[str]:
        out: list[str] = []
        for eid, e in self.entries.items():
            for ref in e.get("derived_from", []):
                out += self._ref_problems(eid, ref)
            kind = e.get("kind")
            if kind == "machine":
                fp = e.get("fault_profile")
                if fp not in self.entries or self.entries[fp]["kind"] != "node":
                    out.append(f"{eid}: fault_profile {fp!r} must be an existing node profile")
                price = e.get("price", {})
                if price.get("route") == "native":
                    for region, key in price.get("pool_keys", {}).items():
                        if key not in self.entries:
                            out.append(f"{eid}: native price {key!r} for {region} missing")
            if kind == "scenario":
                for m in e.get("machines", []):
                    if m not in self.entries or self.entries[m]["kind"] != "machine":
                        out.append(f"{eid}: machine {m!r} missing")
                for field, k in (("region_outage", "region_outage"),
                                 ("planned_downtime", "planned_downtime"),
                                 ("price_assignment", "price")):
                    ref = e.get(field)
                    if ref is not None and (ref not in self.entries or self.entries[ref]["kind"] != k):
                        out.append(f"{eid}: {field} {ref!r} must be an existing {k} entry")
                out += self._check_scenario_prices(e)
        return out

    def _check_scenario_prices(self, s: dict[str, Any]) -> list[str]:
        """Cost objectives need a price for every machine in every region."""
        if s.get("price_policy", "latest") == "none":
            return []
        out = []
        regions = s.get("deployment", {}).get("regions", [])
        assign = self.entries.get(s.get("price_assignment") or "", {})
        for mid in s.get("machines", []):
            m = self.entries.get(mid)
            if m is None:
                continue
            route = m.get("price", {}).get("route")
            m_regions = [r for r in regions if not m.get("regions") or r in m["regions"]]
            if route == "native":
                missing = [r for r in m_regions if r not in m["price"].get("pool_keys", {})]
                if missing:
                    out.append(f"{s['id']}: machine {mid} has no native price in {missing}")
            elif route == "assigned":
                if mid not in assign.get("assignments", {}):
                    out.append(f"{s['id']}: machine {mid} not priced by {s.get('price_assignment')!r}")
            elif route == "none":
                out.append(f"{s['id']}: machine {mid} is unpriced but price_policy is not 'none'")
        return out
