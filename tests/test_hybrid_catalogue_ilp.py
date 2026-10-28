"""The ILP may use only Monte Carlo-certified candidate availability."""

from types import SimpleNamespace

import pytest

from powder.faults import FaultLibrary, catalogue
from powder.faults.ilp import mc_shortlist, solve_catalogue_placement
from powder.faults.mc_scoring import BatchSummary, MCScoringConfig, catalogue_batch_sampler
from powder.placement_optimizer import PlacementCandidate
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol


def test_markov_shortlist_preserves_cheap_and_reliable_extremes():
    cands = [PlacementCandidate(str(i), (str(i),), .99 + i / 10000, i * 3)
             for i in range(10)]
    chosen = mc_shortlist(cands, 4)
    assert len(chosen) == 4
    assert chosen == sorted(chosen, key=lambda c: c.candidate_id)
    assert {"0", "9"} <= {c.candidate_id for c in chosen}


@pytest.mark.parametrize("mean,expected", [(.999995, "pass"), (.9999, "fail"), (.99999, "undecided")])
def test_catalogue_hybrid_filters_mc_decisions_before_ilp(monkeypatch, mean, expected):
    import powder.faults.mc_scoring as scoring

    lib = FaultLibrary.load()
    cat = catalogue.generate(lib, n=60, seed=7, spot_fraction=0.0)
    sampled = []

    def sampler(lib, cat, ids, scenario_id, price, protocol, strategy, *, include_region_outages):
        sampled.append(ids)
        return lambda count, seed: BatchSummary(count, mean, .00001 if mean != .99999 else .001)

    monkeypatch.setattr(scoring, "catalogue_batch_sampler", sampler)
    result = solve_catalogue_placement(
        lib, cat, RaftLikeProtocol(election_time_dist=Exponential(.2)),
        NodeReplacementStrategy(failure_timeout=600, default_node_config=None),
        num_rsms=1, replica_counts=(3,), capacity=2, random_samples_per_size=8,
        greedy_starts_per_size=1, max_candidates=20,
        mc_config=MCScoringConfig(max_candidates=2, min_runs=100_000,
                                  max_runs=100_000, batch_runs=50_000),
    )
    assert len(sampled) == 2
    decisions = result["mc_diagnostics"]["decisions"]
    assert {d["status"] for d in decisions.values()} == {expected}
    assert all(d["runs"] == 100_000 for d in decisions.values())
    if expected == "pass":
        assert result["solution"].total_rsms == 1
        assert result["solution"].selected[0].candidate.metadata["method"] == "rust-mc-certified"
        assert result["solution"].selected[0].candidate.availability == mean
    else:
        assert result["solution"].status == "no_mc_qualified_candidates"
        assert not result["solution"].selected


def test_markov_floor_skips_monte_carlo_entirely(monkeypatch):
    import powder.faults.mc_scoring as scoring

    monkeypatch.setattr(scoring, "catalogue_batch_sampler",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MC called")))
    lib = FaultLibrary.load()
    cat = catalogue.generate(lib, n=60, seed=7, spot_fraction=0.0)
    result = solve_catalogue_placement(
        lib, cat, RaftLikeProtocol(election_time_dist=Exponential(.2)),
        NodeReplacementStrategy(failure_timeout=600, default_node_config=None),
        num_rsms=1, replica_counts=(3,), random_samples_per_size=8,
        greedy_starts_per_size=1, max_candidates=20,
        mc_config=MCScoringConfig(markov_floor_nines=8, target_nines=8,
                                  max_candidates=2, max_runs=100_000),
    )
    assert result["mc_diagnostics"]["markov_rejected"] == len(result["markov_candidates"])
    assert result["solution"].status == "no_mc_qualified_candidates"


def test_catalogue_sampler_preserves_time_varying_clocks_and_network(monkeypatch):
    import powder.mc_backend as mc_backend

    lib = FaultLibrary.load()
    cat = catalogue.generate(lib, n=12, seed=2, spot_fraction=0.0)
    seen = []

    def run(cluster, strategy, protocol, **kwargs):
        seen.append((cluster, kwargs))
        return SimpleNamespace(availability_samples=[.99] * kwargs["num_simulations"])

    monkeypatch.setattr(mc_backend, "run_monte_carlo", run)
    sampler = catalogue_batch_sampler(lib, cat, [m["id"] for m in cat["machines"][:3]],
                                      "scenario/canonical-central", "rho0",
                                      RaftLikeProtocol(election_time_dist=Exponential(.2)),
                                      NodeReplacementStrategy(failure_timeout=600, default_node_config=None),
                                      include_region_outages=True)
    summary = sampler(10, 33)
    cluster, kwargs = seen[0]
    assert summary.count == 10 and summary.mean == .99
    assert cluster.target_cluster_size == len(cluster.nodes) == 3
    assert kwargs["seed"] == 33 and kwargs["stop_on_data_loss"] is False
    assert kwargs["max_time"] == cat["horizon_h"] * 3600
    assert kwargs["network_config"] is not None
    assert any(type(node.config.failure_dist).__name__ == "TimeVaryingHazard"
               for node in cluster.nodes.values())
