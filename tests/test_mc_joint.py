"""Opt-in whole-fleet replay boundary and candidate-mask translation."""

from pathlib import Path

import pytest

from powder.mc_backend import joint_candidate_timeline, run_joint_replay
from powder.mc_jobs import build_joint_job


def _job():
    return build_joint_job(
        horizon_seconds=100, deadline_seconds=1, schema_lease_seconds=5,
        pd_election_seconds=2,
        hosts={id: {"region": id, "az": id} for id in
               ("p1", "p2", "p3", "m1", "m2", "m3", "a1", "a2", "a3",
                "b1", "b2", "b3", "f")},
        pd_members=["p1", "p2", "p3"], pd_leader="p1",
        frontends={"sql": "f"}, frontend="sql",
        system_regions=[["m1", "m2", "m3"]],
        shards={"a": [["a1", "a2", "a3"], ["p1", "a1"]],
                "b": [["b1", "b2", "b3"]]}, selected={"a": 0, "b": 0},
        events=[{"time": t, "scope": "host", "target": host, "down": down}
                for t, down in [(20, True), (30, False)] for host in ("p1", "p2")],
    )


def test_joint_replay_python_boundary_and_candidate_timeline(monkeypatch):
    binary = Path(__file__).resolve().parents[1] / "rust/target/debug/powder-mc"
    if not binary.is_file():
        pytest.skip("build the debug powder-mc binary for this integration test")
    monkeypatch.setenv("POWDER_MC_BINARY", str(binary))
    record = run_joint_replay(_job())
    assert record["model"] == "deterministic_replay_zero_service_time_no_repair"
    assert record["eligible_start_seconds"] == 99
    assert record["censored_seconds"] == 1
    assert record["full_access"] == pytest.approx(90 / 99)
    assert record["single_access"] == pytest.approx(90 / 99)
    assert record["class_success"]["a"]["point_read"] == 1
    assert record["class_success"]["a"]["stale_read"] == 1
    timeline = joint_candidate_timeline(record)
    assert timeline.horizon_seconds == pytest.approx(99)
    composed = timeline.compose({"a": ("a1", "a2", "a3"),
                                 "b": ("b1", "b2", "b3")})
    assert composed.full_access == pytest.approx(record["full_access"])
    assert composed.single_access == pytest.approx(record["single_access"])
    assert timeline.candidate_access("a", ("p1", "a1")) <= composed.per_shard["a"]


def test_joint_masks_feed_full_access_ilp_without_resampling(monkeypatch):
    pytest.importorskip("gurobipy")
    from powder.faults.gurobi_ilp import TransitionProblem, solve_transition_gurobi
    from powder.faults.transition_costs import ShardSpec
    from powder.placement_optimizer import PlacementCandidate

    binary = Path(__file__).resolve().parents[1] / "rust/target/debug/powder-mc"
    if not binary.is_file():
        pytest.skip("build the debug powder-mc binary for this integration test")
    monkeypatch.setenv("POWDER_MC_BINARY", str(binary))
    job = _job()
    record = run_joint_replay(job)
    timeline = joint_candidate_timeline(record)
    menus = [tuple(ids) for shard in job["shards"].values() for ids in shard]
    candidates = [PlacementCandidate(str(i), ids, 0.0, 0.0) for i, ids in enumerate(menus)]
    locations = {host: (cfg["region"], cfg["az"]) for host, cfg in job["hosts"].items()}
    transition = TransitionProblem(tuple(ShardSpec(id, 0, 0, ()) for id in job["shards"]),
                                   locations, lambda *args: 0., lambda *args: 0., 100)
    sol = solve_transition_gurobi(candidates, {host: 0. for host in job["hosts"]},
                                  capacity=2, budget=None, transition=transition,
                                  composed=timeline, composed_objective="full")
    assert sol.full_access_availability >= record["full_access"] - 1e-9
    assert sol.full_access_availability == pytest.approx(
        timeline.compose(dict(sol.shard_placements)).full_access)
    assert sol.single_access_availability == pytest.approx(
        timeline.compose(dict(sol.shard_placements)).single_access)


def test_300_shards_count_one_shared_PD_outage(monkeypatch):
    binary = Path(__file__).resolve().parents[1] / "rust/target/debug/powder-mc"
    if not binary.is_file():
        pytest.skip("build the debug powder-mc binary for this integration test")
    monkeypatch.setenv("POWDER_MC_BINARY", str(binary))
    job = _job()
    job["shards"] = {f"s{i}": [["a1", "a2", "a3"]] for i in range(300)}
    job["selected"] = {shard: 0 for shard in job["shards"]}
    result = run_joint_replay(job)
    assert len(result["per_shard"]) == 300
    assert result["single_access"] == pytest.approx(90 / 99)
    assert result["full_access"] == pytest.approx(90 / 99)
    assert result["global_access"] == pytest.approx(90 / 99)


def test_joint_replay_rejects_invalid_host(monkeypatch):
    binary = Path(__file__).resolve().parents[1] / "rust/target/debug/powder-mc"
    if not binary.is_file():
        pytest.skip("build the debug powder-mc binary for this integration test")
    monkeypatch.setenv("POWDER_MC_BINARY", str(binary))
    job = _job()
    job["shards"]["a"][0][0] = "missing"
    with pytest.raises(RuntimeError, match="distinct known physical hosts"):
        run_joint_replay(job)
