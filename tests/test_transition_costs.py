"""Tests for the new named-shard transition policy (not legacy cadence migration)."""

import pytest

from powder.faults.transition_costs import (
    ShardSpec,
    TransitionCost,
    edge_average_per_gb,
    migration_cost,
    replication_per_hour,
)


LOC = {
    "a": ("r1", "az1"), "b": ("r1", "az1"), "c": ("r1", "az2"),
    "d": ("r2", "az1"), "e": ("r2", "az2"),
    "f": ("r3", "az1"), "g": ("r3", "az2"),
}


def traffic(src, dst, same_az):
    assert not same_az  # same-AZ pairs never require a price lookup
    return 0.02 if src == dst else 0.05


def test_directed_replication_uses_both_directions_and_decimal_gb_per_second():
    calls = []

    def directed(src, dst, same_az):
        calls.append((src, dst, same_az))
        return {("r1", "r2"): 0.04, ("r2", "r1"): 0.09}[(src, dst)]

    assert edge_average_per_gb(("a", "d"), LOC, directed) == pytest.approx(0.065)
    assert replication_per_hour(("a", "d"), 0.5, LOC, directed) == pytest.approx(117.0)
    assert calls == [("r1", "r2", False), ("r2", "r1", False)] * 2


@pytest.mark.parametrize(
    ("machines", "sum_edge_prices"),
    [(("a", "c", "d"), 0.24),
     (("a", "c", "d", "e", "f"), 0.88),
     (("a", "b", "c", "d", "e", "f", "g"), 1.76)],
)
def test_rf_three_five_seven_average_directed_edges(machines, sum_edge_prices):
    k = len(machines)
    q = sum_edge_prices / (k * (k - 1))
    assert edge_average_per_gb(machines, LOC, traffic) == pytest.approx(q)
    assert replication_per_hour(machines, 1.0, LOC, traffic) == pytest.approx(3600 * (k - 1) * q)


def test_same_az_replication_free_but_inter_az_replication_charged():
    def no_lookup(*_):
        raise AssertionError("same AZ must not be billed")

    assert replication_per_hour(("a", "b"), 2.0, LOC, no_lookup) == 0
    assert replication_per_hour(("a", "c"), 1.0, LOC, traffic) == pytest.approx(72.0)
    assert replication_per_hour(("a",), 1.0, LOC, no_lookup) == 0
    assert replication_per_hour((), 1.0, LOC, no_lookup) == 0


def test_same_region_migration_free_even_across_az_but_bytes_and_addition_count():
    def no_lookup(*_):
        raise AssertionError("same-region migration must not look up inter-AZ prices")

    assert migration_cost(("a",), ("c",), 25.0, LOC, no_lookup) == TransitionCost(1, 1, 25.0, 0.0)
    assert migration_cost(("a",), ("a", "c"), 25.0, LOC, no_lookup) == TransitionCost(1, 0, 25.0, 0.0)
    assert migration_cost(("a", "d"), ("a", "d", "c"), 25.0, LOC, no_lookup) == TransitionCost(1, 0, 25.0, 0.0)
    assert migration_cost(("a", "d"), ("d", "a"), 25.0, LOC, no_lookup) == TransitionCost(0, 0, 0.0, 0.0)
    assert migration_cost(("a",), (), 25.0, LOC, no_lookup) == TransitionCost(0, 1, 0.0, 0.0)


def test_cross_region_chooses_cheapest_old_source_and_charges_directed_once():
    prices = {("r1", "r3"): 0.08, ("r2", "r3"): 0.03,
              ("r3", "r1"): 0.11, ("r1", "r2"): 0.04}
    calls = []

    def directed(src, dst):
        calls.append((src, dst))
        return prices[(src, dst)]

    assert migration_cost(("a", "d"), ("a", "d", "f"), 100, LOC, directed) == TransitionCost(1, 0, 100, 3)
    assert set(calls) == {("r1", "r3"), ("r2", "r3")}
    assert migration_cost(("f",), ("f", "a"), 100, LOC, directed).migration_usd == pytest.approx(11)
    assert migration_cost(("a",), ("a", "d"), 100, LOC, directed).migration_usd == pytest.approx(4)


def test_three_to_five_uses_two_additions_and_two_partition_copies():
    def region_price(src, dst):
        return {("r1", "r3"): 0.08, ("r2", "r3"): 0.03}[(src, dst)]

    old = ("a", "c", "d")
    new = ("a", "c", "d", "f", "g")
    assert migration_cost(old, new, 100, LOC, region_price) == TransitionCost(2, 0, 200, 6)
    assert migration_cost(old, old, 100, LOC, region_price) == TransitionCost(0, 0, 0, 0)


def test_empty_old_cannot_supply_nonempty_new_even_with_zero_size():
    with pytest.raises(ValueError, match="old source"):
        migration_cost((), ("a",), 0, LOC, lambda *_: 0)
    assert migration_cost((), (), 0, LOC, lambda *_: 0) == TransitionCost(0, 0, 0, 0)


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_rates_sizes_and_prices_rejected(invalid):
    with pytest.raises(ValueError):
        ShardSpec("shard-1", invalid, 1, ("a",))
    with pytest.raises(ValueError):
        ShardSpec("shard-1", 1, invalid, ("a",))
    with pytest.raises(ValueError):
        replication_per_hour(("a",), invalid, LOC, traffic)
    with pytest.raises(ValueError):
        migration_cost(("a",), ("d",), invalid, LOC, lambda *_: 0.01)
    with pytest.raises(ValueError):
        edge_average_per_gb(("a", "d"), LOC, lambda *_: invalid)
    with pytest.raises(ValueError):
        migration_cost(("a",), ("d",), 1, LOC, lambda *_: invalid)


def test_duplicate_machine_ids_and_missing_locations_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ShardSpec("shard-1", 0, 0, ("a", "a"))
    with pytest.raises(ValueError, match="duplicate"):
        replication_per_hour(("a", "a"), 1, LOC, traffic)
    for old, new in [(("a", "a"), ("d",)), (("a",), ("d", "d"))]:
        with pytest.raises(ValueError, match="duplicate"):
            migration_cost(old, new, 1, LOC, lambda *_: 0.01)
    with pytest.raises(ValueError, match="missing location"):
        replication_per_hour(("a", "unknown"), 1, LOC, traffic)
    with pytest.raises(ValueError, match="missing location"):
        migration_cost(("a",), ("a", "unknown"), 1, LOC, lambda *_: 0.01)
    with pytest.raises(ValueError, match="location"):
        edge_average_per_gb(("a",), {"a": ("", "az1")}, traffic)


def test_missing_directed_price_fails_instead_of_becoming_free():
    with pytest.raises(ValueError, match="missing traffic price"):
        replication_per_hour(("a", "d"), 0, LOC, lambda src, dst, _: {}[(src, dst)])
    with pytest.raises(ValueError, match="missing migration price"):
        migration_cost(("a",), ("d",), 0, LOC, lambda src, dst: {}[(src, dst)])
