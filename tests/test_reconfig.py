"""Reconfiguration accounting (powder/faults/reconfig.py)."""

import math

from powder.faults.reconfig import cadence_label, match_rsms, migration, schedule, sla_violation_minutes

LOC = {"a": ("r1", "z1"), "b": ("r1", "z2"), "c": ("r2", "z3"), "d": ("r2", "z4"), "e": ("r1", "z1"),
       "f": ("r3", "z5")}


def price(src, dst):
    if src == dst:
        return 0.0
    return 0.02 if src[0] == dst[0] else 0.05


def test_schedule_blocks_align_with_reconfigurations():
    starts, blocks = schedule(48, 18, 6)
    assert starts == [0.0, 18.0, 36.0]
    assert blocks[0] == (0.0, 6.0) and blocks[-1][1] == 48.0
    assert all(any(math.isclose(b0, s) for b0, _ in blocks) for s in starts)
    assert math.isclose(sum(b - a for a, b in blocks), 48.0)


def test_identical_deployments_move_nothing_even_if_reordered():
    old = [("a", "b", "c"), ("b", "c", "d")]
    m = migration(old, list(reversed(old)), gb_per_replica=10, location=LOC.get, price_per_gb=price)
    assert (m.replicas_added, m.replicas_removed, m.gb_moved, m.transfer_cost, m.rsms_changed) == (0, 0, 0, 0, 0)


def test_one_replica_swap_copies_from_cheapest_member():
    old = [("a", "b", "c")]
    new = [("a", "c", "e")]  # b -> e; e shares (r1, z1) with a: free copy
    m = migration(old, new, gb_per_replica=10, location=LOC.get, price_per_gb=price)
    assert (m.replicas_added, m.replicas_removed, m.rsms_changed, m.leader_moves) == (1, 1, 1, 0)
    assert m.gb_moved == 10 and m.transfer_cost == 0.0
    m2 = migration(old, [("a", "b", "f")], gb_per_replica=10, location=LOC.get, price_per_gb=price)
    assert math.isclose(m2.transfer_cost, 0.5)


def test_matching_minimizes_moves_and_counts_leader_moves():
    old = [("a", "b", "c"), ("d", "e", "f")]
    new = [("d", "e", "a"), ("a", "b", "c")]
    pairs = dict((j, i) for i, j in match_rsms(old, new))
    assert pairs == {0: 1, 1: 0}
    m = migration(old, [("b", "c", "d"), ("a", "e", "f")], gb_per_replica=1, location=LOC.get, price_per_gb=price)
    assert m.replicas_added == 2 and m.leader_moves == 0
    m3 = migration([("a", "b", "c")], [("d", "e", "f")], gb_per_replica=1, location=LOC.get, price_per_gb=price)
    assert m3.leader_moves == 1 and m3.replicas_added == 3


def test_sla_violation_and_labels():
    month = 30 * 24 * 60
    assert sla_violation_minutes(0.1, month, 5) == 0.0
    assert math.isclose(sla_violation_minutes(1.432, month, 5), 1.432 - month * 1e-5)
    assert [cadence_label(h) for h in (6, 12, 24, 48, 168, 336, 720)] == ["6h", "12h", "1d", "2d", "1w", "2w", "1mo"]
