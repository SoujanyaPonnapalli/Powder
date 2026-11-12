"""Fleet pipeline uses shared-timeline scores rather than legacy per-RSM MC."""

import pytest

pytest.importorskip("gurobipy")

from powder.faults.composed_availability import CandidateTimeline
from powder.faults.gurobi_ilp import TransitionProblem
from powder.faults.transition_costs import ShardSpec
from scripts.pilot_fleet_frontier import cost_tiebreak, lattice_ascent, solve_exact, summarize


class NotSingleRsmMC:
    memo = {}

    def __call__(self, placement):
        raise AssertionError("per-RSM MC cannot validate a composed deployment")

    def batch(self, placements):
        raise AssertionError("per-RSM MC cannot validate a composed deployment")


def test_composed_pipeline_does_not_fall_back_to_legacy_mc():
    a, b = ("m0",), ("m1",)
    timeline = CandidateTimeline((1., 1., 1., 1., 1.), (True, True, True, True, False),
                                 {"s0": {a: (True, True, True, False, True),
                                         b: (True, True, False, False, True)},
                                  "s1": {a: (True, True, False, False, True),
                                         b: (False, False, True, True, True)}})
    transition = TransitionProblem((ShardSpec("s0", 0, 0, ()), ShardSpec("s1", 0, 0, ())),
                                   {"m0": ("r1", "z1"), "m1": ("r2", "z2")},
                                   lambda *args: 0., lambda *args: 0., 10)
    exact = NotSingleRsmMC()
    price = {"m0": 0.1, "m1": 0.1}
    sol, rounds, _ilp, mc, added = solve_exact(
        [a, b], price, exact, lambda _: (_ for _ in ()).throw(AssertionError("legacy estimate")),
        2, 1, 0.2, transition=transition, composed=timeline, composed_objective="full")
    assert rounds == 1 and mc == 0 and added == 0
    assert sol.full_access_availability == pytest.approx(0.4)
    row = summarize(sol, price, exact, 2)
    assert row["mean_availability"] == pytest.approx(0.4)
    assert row["full_availability"] == pytest.approx(0.4)
    assert row["global_availability"] == pytest.approx(0.8)
    asc, asc_rounds, *_ = lattice_ascent(sol, [], {}, exact, price, 2, 1, 0.2,
                                         threads=1, seed=0, transition=transition, composed=timeline)
    assert asc is sol and asc_rounds == 0
    tiebreak, tb_rounds, _ilp, mc = cost_tiebreak([a, b], row["placements"], price, exact,
                                                  lambda _: 0, 2, 1, 0.2, tolerance_nines=0.0,
                                                  transition=transition, composed=timeline,
                                                  composed_objective="full")
    assert tb_rounds == 1 and mc == 0
    assert tiebreak.full_access_availability >= sol.full_access_availability - 1e-9
