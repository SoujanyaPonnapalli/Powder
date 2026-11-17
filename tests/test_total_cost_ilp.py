"""Total-cost (machines + replication) fleet ILP: scripts/pilot_fleet_frontier.py --write-gb-per-s."""

import pytest

from powder.placement_optimizer import PlacementCandidate
from scripts.pilot_fleet_frontier import min_replication_per_rsm, placements_cost, summarize_placements

AWS = {("e1", "e2"): 0.01, ("e2", "e1"): 0.01}


def aws_like(src, dst, same_az):
    return 0.02 if src == dst else AWS.get((src, dst), 0.02)  # same region = inter-AZ $0.02/GB


def rate(gb_per_s):
    def cost(ids):
        raise AssertionError("not used")
    cost.write_gb_per_s, cost.pricer = gb_per_s, aws_like
    return cost


def test_least_replication_per_rf_uses_cheapest_valid_region_pattern():
    regions = ("e1", "e2", "w1", "w2")
    # RF3: one replica per region; best is e1, e2 + one west: directed sum 2 x 0.01 + 4 x 0.02 over 6 edges.
    assert min_replication_per_rsm(rate(1.0), 3, regions) == pytest.approx(3600 * 2 * 0.10 / 6)
    # RF5 needs >= 3 regions (at most 2 per region); every valid placement costs at least this.
    rf5 = min_replication_per_rsm(rate(1.0), 5, regions)
    assert rf5 > min_replication_per_rsm(rate(1.0), 3, regions)
    assert min_replication_per_rsm(rate(2.0), 5, regions) == pytest.approx(2 * rf5)


def test_placements_cost_bills_machines_once_and_traffic_per_rsm():
    price = {"m0001": 1.0, "m0002": 2.0, "m0003": 4.0}  # catalogue-style ids (spot = index >= 1000)
    placements = [("m0001", "m0002", "m0003")] * 2
    assert placements_cost(placements, price) == (7.0, 0.0)
    assert placements_cost(placements, price, lambda p: 0.5) == (7.0, 1.0)

    class Exact:
        def __call__(self, ids):
            return 0.999
    row = summarize_placements(placements, price, Exact(), lambda p: 0.5)
    assert row["cost_per_hour"] == 8.0 and row["machine_cost_per_hour"] == 7.0
    assert row["replication_cost_per_hour"] == 1.0
    assert "replication_cost_per_hour" not in summarize_placements(placements, price, Exact())


gp = pytest.importorskip("gurobipy")
from powder.faults.gurobi_ilp import solve_unique_cost_gurobi  # noqa: E402

PRICE = {m: 1.0 for m in "abcdef"}
SPREAD = PlacementCandidate("spread", ("a", "b", "c"), 0.9999, 3.0)   # more available, cross-region traffic
LOCAL = PlacementCandidate("local", ("d", "e", "f"), 0.999, 3.0)      # cheaper to replicate


def test_running_cost_enters_the_budget_and_is_reported():
    machine_only = solve_unique_cost_gurobi([SPREAD, LOCAL], PRICE, capacity=1, num_rsms=1, budget=3.5)
    assert machine_only.selected[0].candidate is SPREAD
    assert machine_only.replication_cost_per_hour is None and machine_only.total_cost_per_hour == 3.0
    total = solve_unique_cost_gurobi([SPREAD, LOCAL], PRICE, capacity=1, num_rsms=1, budget=3.5,
                                     rsm_cost=[1.0, 0.25])
    assert total.selected[0].candidate is LOCAL
    assert total.machine_cost_per_hour == 3.0 and total.replication_cost_per_hour == 0.25
    assert total.total_cost_per_hour == pytest.approx(3.25)


def test_running_cost_counts_per_rsm_and_drives_cost_minimization():
    # Two RSMs share the same machines (capacity 2): machines billed once, traffic twice.
    sol = solve_unique_cost_gurobi([SPREAD, LOCAL], PRICE, capacity=2, num_rsms=2, budget=None,
                                   rsm_cost=[1.0, 0.25], minimize_cost=True, max_unavailability_ppm=10_000)
    assert [(it.candidate.candidate_id, it.count) for it in sol.selected] == [("local", 2)]
    assert sol.total_cost_per_hour == pytest.approx(3.0 + 2 * 0.25)
    with pytest.raises(ValueError):
        solve_unique_cost_gurobi([SPREAD, LOCAL], PRICE, capacity=1, num_rsms=1, budget=5, rsm_cost=[1.0])
