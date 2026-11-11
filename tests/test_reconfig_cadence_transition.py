"""Transition-aware cadence accounting without an AWS deployment or a Gurobi solve."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from powder.faults.gurobi_ilp import TransitionProblem
from powder.faults.transition_costs import ShardSpec
from scripts import reconfig_cadence as cadence


def test_workload_parsing_preserves_named_order_and_checks_count(tmp_path: Path):
    rows = [
        {"shard_id": "z", "write_gb_per_s": 0.001, "partition_gb": 10},
        {"shard_id": "a", "write_gb_per_s": 0.002, "partition_gb": 20},
    ]
    path = tmp_path / "workload.json"
    path.write_text(json.dumps({"shards": rows}))
    specs = cadence.workload_specs(str(path), 2, None, 100)
    assert [(s.shard_id, s.write_gb_per_s, s.partition_gb, s.old_placement) for s in specs] == [
        ("z", 0.001, 10, ()), ("a", 0.002, 20, ())]
    path.write_text(json.dumps(rows))
    assert cadence.workload_specs(str(path), 2, None, 100) == specs
    with pytest.raises(ValueError, match="expected --num-rsms=3"):
        cadence.workload_specs(str(path), 3, None, 100)
    with pytest.raises(ValueError, match="cannot be combined"):
        cadence.workload_specs(str(path), 2, 0.001, 100)
    path.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match="unique"):
        cadence.workload_specs(str(path), 2, None, 100)


def test_cap_only_builds_stable_zero_write_workload():
    specs = cadence.workload_specs(None, 2, None, 17)
    assert [(s.shard_id, s.write_gb_per_s, s.partition_gb) for s in specs] == [
        ("0", 0, 17), ("1", 0, 17)]
    with pytest.raises(ValueError, match="nonnegative"):
        cadence.workload_specs(None, 2, -1, 17)


def test_named_diagnostics_charge_recurring_and_amortized_migration():
    loc = {"a": ("east", "az1"), "b": ("west", "az2"),
           "c": ("east", "az3"), "d": ("west", "az4")}
    traffic = lambda src, dst, same_az: 0.1 if src != dst else 0.01
    region = lambda src, dst: 0.1 if src != dst else 0.0
    transition = TransitionProblem((ShardSpec("z", 0.001, 10, ("a", "b")),
                                    ShardSpec("a", 0.0, 20, ("c", "d"))),
                                   loc, traffic, region, 5, 1)
    diag = cadence.transition_diagnostics((("z", ("a", "c")), ("a", ("c", "d"))),
                                          transition, {m: 1.0 for m in loc})
    assert diag["placements"] == [["a", "c"], ["c", "d"]]
    assert diag["replica_moves"] == 1
    assert diag["replica_removals"] == 1
    assert diag["gb_moved"] == 10
    assert diag["migration_usd"] == 0  # same-region copies are free, even across AZs
    assert diag["replication_cost_per_hour"] == pytest.approx(3.6 * 0.01)
    assert diag["machine_cost_per_hour"] == 3
    assert diag["cost_per_hour"] == pytest.approx(3.036)
    assert cadence.transition_feasible(diag, 3.036, 1)
    assert not cadence.transition_feasible(diag, 3.035, 1)
    assert not cadence.transition_feasible(diag, 4, 0)
    with pytest.raises(ValueError, match="order"):
        cadence.transition_diagnostics((("a", ("a", "c")), ("z", ("c", "d"))),
                                       transition, {m: 1.0 for m in loc})
    remote = TransitionProblem((ShardSpec("z", 0.001, 10, ("a",)), transition.shards[1]),
                               loc, traffic, region, 5, 1)
    moved = cadence.transition_diagnostics((("z", ("b",)), ("a", ("c", "d"))),
                                          remote, {m: 1.0 for m in loc})
    assert moved["gb_moved"] == 10
    assert moved["migration_usd"] == pytest.approx(1)
    assert moved["cost_per_hour"] == pytest.approx(3 + 1 / 5)


def test_transition_cadence_uses_named_migration_not_hungarian(tmp_path: Path, monkeypatch):
    study = cadence.Study.__new__(cadence.Study)
    study.a = argparse.Namespace(horizon_h=24, block_h=12, gb_per_replica=100,
                                 sla_nines=5, num_rsms=2)
    study.out = tmp_path
    study.transition_mode = True
    study.transition_key = "test"
    study.t0 = time.time()
    study.log = lambda msg: None
    deployments = [
        {"placements": [["a", "b"], ["c", "d"]], "window_h": 12,
         "cost_per_hour": 10, "machine_cost_per_hour": 8, "replication_cost_per_hour": 2,
         "migration_usd": 0, "gb_moved": 0, "replica_moves": 0, "replica_removals": 0, "cpu_s": 60},
        {"placements": [["c", "d"], ["a", "b"]], "window_h": 12,
         "cost_per_hour": 12, "machine_cost_per_hour": 8, "replication_cost_per_hour": 2,
         "migration_usd": 24, "gb_moved": 100, "replica_moves": 2, "replica_removals": 2, "cpu_s": 60},
    ]
    study.solve = lambda cadence_h, t0_h, window_h, prev: deployments[int(t0_h // 12)]
    study.block_scores = lambda block, placements: {"|".join(p): 0.99 for p in placements}
    monkeypatch.setattr(cadence.R, "migration", lambda *args, **kwargs: pytest.fail("legacy matching used"))
    out = study.run_cadence(12)
    assert out["migrations"][1] == {"replicas_added": 2, "replicas_removed": 2,
                                    "gb_moved": 100, "transfer_cost": 24}
    assert out["transfer_cost_per_day"] == 24
    assert out["replicas_moved_per_day"] == 2
    assert out["gb_moved_per_day"] == 100
    assert out["mean_machine_cost_per_hour"] == 8
    assert out["mean_replication_cost_per_hour"] == 2
    assert out["mean_cost_per_hour"] == 11


def test_ascent_cannot_bypass_transition_move_cap(tmp_path: Path, monkeypatch):
    study = cadence.Study.__new__(cadence.Study)
    args = argparse.Namespace(price="rho1", capacity=5, scale=1, max_machines=0,
                              exact_runs=1, threads=1, clusters=1, table_runs=1,
                              num_rsms=1, budget_per_hour=20, fleet_sizes=1, fleet_repeats=1,
                              alpha=0.9, passes=0, ilp_time_limit=1, min_gain=0,
                              max_replica_moves=0)
    study.a, study.transition_mode, study.transition_key = args, True, "test"
    study.transition_specs = (ShardSpec("shard", 0, 10, ()),)
    study.out = tmp_path
    (tmp_path / "solves").mkdir()
    study.t0 = time.time()
    study.log = lambda msg: None
    study.protocol = study.strategy = study.network = study.lib = None
    study.score_horizon_s = 1
    ids = "abcdef"
    study.locations = {m: ("east", "az1") for m in ids}
    machines = [SimpleNamespace(machine_id=m, node_config=SimpleNamespace(cost_per_hour=1)) for m in ids]
    study.machines = lambda t, w: machines
    feature = lambda m: SimpleNamespace(machine_id=m, lam_t=1, lam_p=0, region="east")
    monkeypatch.setattr(cadence.F, "features", lambda ms: [feature(m.machine_id) for m in ms])
    monkeypatch.setattr(cadence.F, "kmeans_log", lambda feats, k, seed: ([0] * len(feats), [0]))
    monkeypatch.setattr(cadence.F, "medoids", lambda feats, labels, cent: [0])
    fleet_alphas = []
    def fleets(*args, **kw):
        fleet_alphas.append(kw["alpha"])
        return [["a", "b", "c"]]
    monkeypatch.setattr(cadence.F, "fleets_for_budget", fleets)
    monkeypatch.setattr(cadence.F, "pack_pass", lambda *args, **kw: [("a", "b", "c")] if args[1] == 3 else [])
    monkeypatch.setattr(cadence.F, "balanced_mix", lambda *args: None)
    monkeypatch.setattr(cadence.F, "split_passes", lambda *args: {})
    monkeypatch.setattr(cadence, "baseline", lambda *args, **kw: {"cost_per_hour": 3,
                                                                     "placements": [("a", "b", "c")]})
    monkeypatch.setattr(cadence, "transfer_per_gb", lambda *args, **kw: 0.0)

    class Table:
        def __init__(self, *args, **kwargs):
            self.memo = {}
            self.labels = {m: 0 for m in ids}
        def availability(self, *args):
            return 0.99
        def score(self, p):
            return 0.99

    class Exact:
        def __init__(self, *args):
            pass
        def __call__(self, p):
            return 0.999 if "d" in p else 0.99
        def batch(self, p):
            pass

    monkeypatch.setattr(cadence, "CentroidTable", Table)
    monkeypatch.setattr(cadence, "Exact", Exact)
    def solution(placement):
        return SimpleNamespace(selected=[SimpleNamespace(candidate=SimpleNamespace(machine_ids=placement), count=1)],
                               shard_placements=(("shard", placement),))
    def solve_exact(*args, **kw):
        assert kw["feat_of"] is None
        assert kw["transition"].max_replica_moves in (None, 0)
        assert kw["transition"].shards[0].old_placement == (() if kw["transition"].max_replica_moves is None
                                                               else ("a", "b", "c"))
        return solution(("a", "b", "c")), 1, 0, 0, 0
    def ascent(*args, **kw):
        assert kw["transition"].max_replica_moves in (None, 0)
        return solution(("d", "e", "f")), 1, 0, 0, 0
    monkeypatch.setattr(cadence, "solve_exact", solve_exact)
    monkeypatch.setattr(cadence, "lattice_ascent", ascent)
    rec = study.solve(12, 12, 12, [["a", "b", "c"]])
    assert rec["placements"] == [["a", "b", "c"]]
    assert rec["replica_moves"] == 0
    assert rec["cost_per_hour"] == 3
    initial = study.solve(12, 0, 12, None)
    assert initial["placements"] == [["d", "e", "f"]]
    assert initial["replica_moves"] == 0  # no old replicas to move from, no movement cap
    assert fleet_alphas == [0.0, 0.0]  # leave compute headroom for replication traffic
