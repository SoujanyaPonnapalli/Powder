"""Per-deployment score lookup and PD protocol (no TiUP or Gurobi needed)."""

import json
import threading
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from infra.tidb.oracle_server import handler_for
from scripts.export_pd_oracle_scores import export
from powder.deploy.pd_oracle import OracleError, ScoreTable, StoreSignature, from_candidates
from powder.placement_optimizer import PlacementCandidate


def labels(store_id):
    return {"machine_class": f"class-{store_id}", "region": "us-east-1",
            "zone": "us-east-1a", "host": f"tikv-{store_id}"}


def table():
    return ScoreTable.from_dict({"version": 1, "model_id": "toy-model-v1",
                                 "stores": [{"store_id": i, "labels": labels(i)} for i in range(1, 6)],
                                 "placements": [
                                     {"store_ids": [1, 2, 4], "availability": 0.99},
                                     {"store_ids": [1, 2, 5], "availability": 0.999},
                                 ]})


def request():
    return {"version": 1, "model_id": "toy-model-v1", "event": "replace", "region_id": 42,
            "epoch": {"version": 7, "conf_ver": 3}, "old_store_id": 3,
            "peers": [{"store_id": i, "labels": labels(i), "role": "voter", "witness": False}
                      for i in (1, 2, 3)],
            "candidates": [{"store_id": i, "labels": labels(i), "region_score": rank}
                           for rank, i in enumerate((4, 5))]}


def test_one_replacement_prefers_scored_candidate_and_preserves_peers():
    result = table().recommend(request())
    assert result == {"version": 1, "model_id": "toy-model-v1", "region_id": 42,
                      "epoch": {"version": 7, "conf_ver": 3}, "store_id": 5,
                      "availability": 0.999, "reason": "scored"}
    missing = request()
    missing["event"] = "missing"
    missing["old_store_id"] = None
    missing["peers"] = missing["peers"][:2]
    assert table().recommend(missing)["store_id"] == 5


def test_defer_on_stale_model_labels_unsupported_role_and_score_miss():
    req = request()
    req["model_id"] = "old"
    assert table().recommend(req)["reason"] == "model_mismatch"
    req = request()
    req["peers"][0]["labels"]["host"] = "different"
    assert table().recommend(req)["reason"] == "store_labels_changed"
    req = request()
    req["peers"][0]["witness"] = True
    assert table().recommend(req)["reason"] == "unsupported_peer_role"
    req = request()
    req["peers"][0]["down"] = True
    assert table().recommend(req)["reason"] == "other_peer_unavailable"
    req = request()
    req["peers"][2]["down"] = True
    assert table().recommend(req)["store_id"] == 5
    req = request()
    req["candidates"] = [dict(req["candidates"][0], store_id=2)]
    with pytest.raises(OracleError, match="already a peer"):
        table().recommend(req)
    req = request()
    req["candidates"] = [{"store_id": 3, "labels": labels(3), "region_score": 0}]
    with pytest.raises(OracleError, match="already a peer"):
        table().recommend(req)
    req = request()
    req["candidates"] = [{"store_id": 5, "labels": labels(5), "region_score": 0}]
    req["peers"][1]["store_id"] = 4
    req["peers"][1]["labels"] = labels(4)
    assert table().recommend(req)["reason"] == "no_scored_candidate"


def test_bad_request_and_score_file_are_rejected():
    req = request()
    req["old_store_id"] = 99
    with pytest.raises(OracleError, match="old store"):
        table().recommend(req)
    req = request()
    req["candidates"][0]["region_score"] = float("nan")
    with pytest.raises(OracleError, match="region_score"):
        table().recommend(req)
    doc = table().to_dict()
    assert ScoreTable.from_dict(doc).to_dict() == doc
    doc["placements"][0]["availability"] = float("nan")
    with pytest.raises(OracleError, match="availability"):
        ScoreTable.from_dict(doc)
    doc = table().to_dict()
    doc["placements"].append(doc["placements"][0])
    with pytest.raises(OracleError, match="duplicate placement"):
        ScoreTable.from_dict(doc)


def test_export_scored_candidates_requires_real_store_mapping():
    sigs = {i: StoreSignature.parse(labels(i)) for i in range(1, 6)}
    candidates = [PlacementCandidate("c1", ("a", "b", "d"), 0.99, 0.1),
                  PlacementCandidate("c2", ("a", "b", "e"), 0.999, 0.2)]
    exported = from_candidates(candidates, dict(zip("abde", (1, 2, 4, 5))), sigs, "toy-model-v1")
    assert exported.recommend(request())["store_id"] == 5
    with pytest.raises(OracleError, match="missing PD store mapping"):
        from_candidates(candidates, {"a": 1}, sigs, "toy-model-v1")
    with pytest.raises(OracleError, match="availability"):
        from_candidates([PlacementCandidate("bad", ("a", "b"), float("nan"), 0)],
                        {"a": 1, "b": 2}, sigs, "toy-model-v1")


def test_export_cli_data_binds_scores_to_pd_store_labels():
    candidates = {"candidates": [
        {"candidate_id": "a", "machine_ids": ["m1", "m2", "m4"],
         "availability": 0.99, "cost_per_hour": 0.1},
        {"candidate_id": "b", "machine_ids": ["m1", "m2", "m5"],
         "availability": 0.999, "cost_per_hour": 0.2},
    ]}
    stores = {"stores": [{"store": {"id": i, "labels":
                              [{"key": k, "value": v} for k, v in labels(i).items()]}}
                         for i in range(1, 6)]}
    doc = export(candidates, {f"m{i}": i for i in (1, 2, 4, 5)}, stores, "toy-model-v1")
    assert ScoreTable.from_dict(doc).recommend(request())["store_id"] == 5
    with pytest.raises(OracleError, match="missing PD store mapping"):
        export(candidates, {"m1": 1}, stores, "toy-model-v1")


def test_loopback_server_returns_advice_and_bounds_request_size():
    server = HTTPServer(("127.0.0.1", 0), handler_for(table()))
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/recommend"
        with urlopen(Request(url, data=json.dumps(request()).encode())) as response:
            assert json.load(response)["store_id"] == 5
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(url, data=b"x" * (64 * 1024 + 1)))
        assert exc.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
