"""Replication-cost accounting in scripts/render_pilot_frontiers.py (pure JSON rows, no catalogue)."""

from scripts.render_pilot_frontiers import points


ROW = {
    "new_ilp": {"0.5": {"feasible": True, "cost_per_hour": 10.0, "nines": 5.0, "placements": [["a", "b", "c"]]},
                "0.1": {"feasible": False}},
    "old_ilp": {"0.5": {"feasible": True, "cost_per_hour": 9.0, "nines": 4.0}},  # no saved placements
    "baselines": {"rf3-cheap": {"cost_per_hour": 3.0, "nines": 3.0, "placements": [["a", "b", "d"]] * 2}},
}


def test_machine_only_cost_is_unchanged_without_traffic():
    assert sorted(points(ROW)) == [("ILP", "50% budget", 10.0, 5.0), ("cheap", "RF 3", 3.0, 3.0),
                                   ("old", "50% budget", 9.0, 4.0)]


def test_traffic_is_added_to_ilp_and_baselines_and_unplaced_points_dropped():
    traffic = lambda placements: 0.5 * len(placements)  # noqa: E731
    assert sorted(points(ROW, traffic)) == [("ILP", "50% budget", 10.5, 5.0), ("cheap", "RF 3", 4.0, 3.0)]
