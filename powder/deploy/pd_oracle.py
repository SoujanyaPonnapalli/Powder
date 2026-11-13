"""Bounded, per-Region placement advice for a patched PD.

The score file is generated for one deployment: PD store IDs are not stable across
clusters. Only scored placements with matching store labels are eligible; missing
scores cause PD to use its own safe ranking. This module never issues PD operators.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from powder.placement_optimizer import PlacementCandidate


class OracleError(ValueError):
    pass


@dataclass(frozen=True)
class StoreSignature:
    machine_class: str
    region: str
    zone: str
    host: str

    @classmethod
    def parse(cls, data: Any) -> StoreSignature:
        if not isinstance(data, dict):
            raise OracleError("store labels must be an object")
        fields = ("machine_class", "region", "zone", "host")
        if any(not isinstance(data.get(key), str) or not data[key] for key in fields):
            raise OracleError("store labels require machine_class, region, zone and host")
        return cls(*(data[key] for key in fields))


@dataclass(frozen=True)
class PlacementScore:
    availability: float
    cost_per_hour: float


class ScoreTable:
    def __init__(self, model_id: str, stores: Mapping[int, StoreSignature],
                 placements: Mapping[tuple[int, ...], PlacementScore]):
        if not model_id:
            raise OracleError("model_id is required")
        self.model_id = model_id
        self.stores = dict(stores)
        self.placements = dict(placements)

    @classmethod
    def from_dict(cls, doc: Any) -> ScoreTable:
        if not isinstance(doc, dict) or doc.get("version") != 1:
            raise OracleError("score table must have version 1")
        model_id = doc.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise OracleError("model_id is required")
        raw_stores = doc.get("stores")
        raw_placements = doc.get("placements")
        if not isinstance(raw_stores, list) or not isinstance(raw_placements, list):
            raise OracleError("stores and placements must be arrays")
        stores = {}
        for store in raw_stores:
            if not isinstance(store, dict):
                raise OracleError("store must be an object")
            store_id = _store_id(store.get("store_id"))
            if store_id in stores:
                raise OracleError(f"duplicate store {store_id}")
            stores[store_id] = StoreSignature.parse(store.get("labels"))
        placements = {}
        for item in raw_placements:
            if not isinstance(item, dict):
                raise OracleError("placement must be an object")
            ids = _placement_ids(item.get("store_ids"))
            if any(store_id not in stores for store_id in ids):
                raise OracleError("placement references an unknown store")
            if ids in placements:
                raise OracleError(f"duplicate placement {ids}")
            availability = item.get("availability")
            cost = item.get("cost_per_hour", 0.0)
            if (isinstance(availability, bool) or not isinstance(availability, (int, float))
                    or not math.isfinite(availability) or not 0 <= availability <= 1):
                raise OracleError("availability must be finite and between 0 and 1")
            if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                    or not math.isfinite(cost) or cost < 0):
                raise OracleError("cost_per_hour must be finite and nonnegative")
            placements[ids] = PlacementScore(float(availability), float(cost))
        return cls(model_id, stores, placements)

    @classmethod
    def load(cls, path: str | Path) -> ScoreTable:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "model_id": self.model_id,
                "stores": [{"store_id": store_id, "labels": vars(labels)}
                           for store_id, labels in sorted(self.stores.items())],
                "placements": [{"store_ids": list(ids), "availability": score.availability,
                                "cost_per_hour": score.cost_per_hour}
                               for ids, score in sorted(self.placements.items())]}

    def recommend(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict) or request.get("version") != 1:
            raise OracleError("request must have version 1")
        if request.get("event") not in ("replace", "missing"):
            raise OracleError("only peer replacement and missing-peer repair are supported")
        if request.get("model_id") != self.model_id:
            return self._defer("model_mismatch")
        region_id = _store_id(request.get("region_id"))
        epoch = request.get("epoch")
        if (not isinstance(epoch, dict) or not all(isinstance(epoch.get(key), int)
                and not isinstance(epoch[key], bool) and epoch[key] >= 0 for key in ("version", "conf_ver"))):
            raise OracleError("region epoch is required")
        peers = request.get("peers")
        candidates = request.get("candidates")
        if not isinstance(peers, list) or not isinstance(candidates, list):
            raise OracleError("peers and candidates must be arrays")
        old = request.get("old_store_id")
        if request["event"] == "replace":
            old = _store_id(old)
        elif old is not None:
            raise OracleError("missing-peer repair cannot name an old store")
        seen = set()
        fixed = []
        for peer in peers:
            if not isinstance(peer, dict):
                raise OracleError("peer must be an object")
            store_id = _store_id(peer.get("store_id"))
            if store_id in seen:
                raise OracleError("duplicate peer store")
            seen.add(store_id)
            if peer.get("role") != "voter" or peer.get("witness") is not False:
                return self._defer("unsupported_peer_role")
            if peer.get("pending", False) or (peer.get("down", False) and store_id != old):
                return self._defer("other_peer_unavailable")
            if not self._matches(store_id, peer.get("labels")):
                return self._defer("store_labels_changed")
            if store_id != old:
                fixed.append(store_id)
        if old is not None and old not in seen:
            raise OracleError("old store is not a Region peer")
        best = None
        candidate_ids = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise OracleError("candidate must be an object")
            store_id = _store_id(candidate.get("store_id"))
            if store_id in candidate_ids or store_id in seen:
                raise OracleError("duplicate candidate or candidate already a peer")
            candidate_ids.add(store_id)
            if not self._matches(store_id, candidate.get("labels")):
                continue
            score = self.placements.get(tuple(sorted((*fixed, store_id))))
            if score is None:
                continue
            region_score = candidate.get("region_score")
            if (isinstance(region_score, bool) or not isinstance(region_score, (int, float))
                    or not math.isfinite(region_score) or region_score < 0):
                raise OracleError("candidate region_score must be finite and nonnegative")
            # PD supplied only hard-filtered destinations in its best isolation tier.
            key = (-score.availability, score.cost_per_hour, float(region_score), store_id)
            if best is None or key < best[0]:
                best = (key, store_id, score)
        if best is None:
            return self._defer("no_scored_candidate")
        _, store_id, score = best
        return {"version": 1, "model_id": self.model_id, "region_id": region_id,
                "epoch": epoch, "store_id": store_id, "availability": score.availability,
                "reason": "scored"}

    def _matches(self, store_id: int, labels: Any) -> bool:
        try:
            actual = StoreSignature.parse(labels)
        except OracleError:
            return False
        return self.stores.get(store_id) == actual

    def _defer(self, reason: str) -> dict[str, Any]:
        return {"version": 1, "model_id": self.model_id, "store_id": None, "reason": reason}


def from_candidates(candidates: Sequence[PlacementCandidate], machine_to_store: Mapping[str, int],
                    stores: Mapping[int, StoreSignature], model_id: str) -> ScoreTable:
    """Export scored ILP candidates after mapping machine IDs to this deployment's PD store IDs."""
    scores = {}
    for candidate in candidates:
        try:
            ids = _placement_ids([machine_to_store[m] for m in candidate.machine_ids])
        except KeyError as exc:
            raise OracleError(f"missing PD store mapping for {exc.args[0]}") from exc
        if any(store_id not in stores for store_id in ids):
            raise OracleError("candidate references an unknown PD store")
        if ids in scores:
            raise OracleError(f"duplicate mapped placement {ids}")
        scores[ids] = PlacementScore(candidate.availability, candidate.cost_per_hour)
    table = ScoreTable(model_id, stores, scores)
    return ScoreTable.from_dict(table.to_dict())


def _store_id(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OracleError("store and Region IDs must be positive integers")
    return value


def _placement_ids(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise OracleError("placement requires a nonempty store_ids array")
    ids = tuple(sorted(_store_id(value) for value in raw))
    if len(ids) != len(set(ids)):
        raise OracleError("placement contains a duplicate store")
    return ids
