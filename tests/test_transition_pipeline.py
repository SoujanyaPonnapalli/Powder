"""Transition accounting survives the fleet exact-scoring and cost-tie-break paths."""

import pytest

pytest.importorskip("gurobipy")

from powder.faults.gurobi_ilp import TransitionProblem
from powder.faults.transition_costs import ShardSpec
from scripts.pilot_fleet_frontier import cost_tiebreak, solve_exact, summarize


class Exact:
    def __init__(self, scores):
        self.scores = scores
        self.memo = {}

    def __call__(self, ids):
        self.memo[ids] = self.scores[ids]
        return self.memo[ids]

    def needs(self, ids):
        return tuple(sorted(ids)) not in self.memo

    def batch(self, placements):
        for p in placements:
            self(tuple(sorted(p)))


def test_exact_resolve_and_cost_tiebreak_preserve_shard_mapping_and_budget():
    old, new = ("m0", "m1", "m2"), ("m0", "m1", "m3")
    pool = [old, new]
    loc = {"m0": ("r1", "z1"), "m1": ("r1", "z1"),
           "m2": ("r1", "z1"), "m3": ("r2", "z2")}
    t = TransitionProblem((ShardSpec("s", 0, 100, old),), loc,
                          lambda src, dst, same: 0 if same else 0.02,
                          lambda src, dst: 0.02, horizon_h=10, max_replica_moves=1)
    scores = {old: 0.99, new: 0.999}
    exact = Exact(scores)
    price = {m: 0.1 for m in loc}
    sol, rounds, *_ = solve_exact(pool, price, exact, lambda p: scores[p], 1, 1, 0.5,
                                  warm=[old], rounds=3, transition=t)
    assert sol.shard_placements == (("s", new),)
    assert sol.replica_moves == 1 and sol.total_cost_per_hour == pytest.approx(0.5)
    assert summarize(sol, price, exact, 1)["placements"] == [list(new)]
    assert rounds >= 1
    # The budget includes migration: a stricter budget cannot afford the new placement.
    exact.batch([old])
    cheaper, *_ = cost_tiebreak(pool, [new], price, exact, lambda p: scores[p], 1, 1, 0.4,
                                tolerance_nines=1, transition=t)
    assert cheaper.shard_placements == (("s", old),)
    assert cheaper.total_cost_per_hour == pytest.approx(0.3)
