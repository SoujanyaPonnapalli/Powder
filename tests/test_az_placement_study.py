"""AZ-aware baselines and MC checkpointing are reproducible and quorum-safe."""

import pytest

from powder.faults import FaultLibrary, catalogue
from powder.faults.ilp import catalogue_machines
from powder.faults.mc_scoring import MCDecision, MCScoringConfig
from powder.placement_optimizer import (CandidateScore, PlacementCandidate, PlacementSolution,
                                        SelectedCandidate, no_region_majority, no_shared_az)
from scripts import experiment_az_hybrid_frontiers as study


def test_block_baselines_cover_10000_rsms_without_reusing_hosts():
    lib = FaultLibrary.load()
    master = catalogue.load("data/fault_library/catalogue/machines-n2000-seed0-az-v2.json.gz")
    cat = {**master, "n": 1000, "machines": master["machines"][:1000]}
    for price in ("rho0", "rho1"):
        machines = catalogue_machines(lib, cat, scenario_id=study.SCENARIO,
                                       price=price, capacity=500)
        by_id = {m.machine_id: m for m in machines}
        for rf in study.RF_COUNTS:
            for cheap in (False, True):
                groups = study.block_baseline(machines, rf, cheap=cheap, seed=0)
                assert len(groups) == 20
                assert len({mid for group in groups for mid in group}) == rf * 20
                assert all(len(group) == rf and no_region_majority(group, by_id)
                           and no_shared_az(group, by_id) for group in groups)


def test_mixed_solution_counts_all_selected_rf_groups():
    a = PlacementCandidate("a", ("a", "b", "c"), .99, 3)
    b = PlacementCandidate("b", ("a", "b", "c", "d", "e"), .999, 5)
    solution = PlacementSolution("0", "optimal", "sum_availability", 0, 8, None,
                                 (SelectedCandidate(a, 1000), SelectedCandidate(b, 9000)))
    result = study.summarize(solution)
    assert result["rf_counts"] == {3: 1000, 5: 9000}
    assert sum(result["rf_counts"].values()) == 10000


def test_mc_checkpoint_reuses_only_the_matching_candidate_set(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(study, "catalogue_batch_sampler",
                        lambda *args, **kwargs: lambda n, seed: None)

    def certify(sampler, config, *, base_seed, comparisons):
        called.append((base_seed, comparisons))
        return MCDecision("pass", 100_000, .99999, .99998, .999995, .001)

    monkeypatch.setattr(study, "certify_availability", certify)
    cat = {"n": 2, "machines": [{"id": "a"}, {"id": "b"}]}
    cfg = MCScoringConfig(target_nines=4.5, max_runs=100_000, batch_runs=100_000)
    path = tmp_path / "check.jsonl"
    ids = {("a", "b")}
    first = study.mc_rows(None, cat, ids, None, None, cfg, path)
    second = study.mc_rows(None, cat, ids, None, None, cfg, path)
    assert first == second and len(called) == 1
    with pytest.raises(ValueError, match="different MC experiment"):
        study.mc_rows(None, cat, {("a",)}, None, None, cfg, path)
