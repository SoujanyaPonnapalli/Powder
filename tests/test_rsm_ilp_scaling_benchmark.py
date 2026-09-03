import csv
from pathlib import Path

import pytest

from notebooks.rsm_ilp_scaling_benchmark import (
    INVENTORY_FIELDS,
    MACHINE_SPECS,
    MACHINES_PER_CLASS,
    REPLICA_COUNTS,
    generate_inventory,
    generate_samples,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_inventory_uses_inverse_correlated_normal_draw(tmp_path):
    inventory_path = tmp_path / "inventory.csv"
    generate_inventory(inventory_path, seed=17)
    rows = _rows(inventory_path)

    assert tuple(rows[0]) == INVENTORY_FIELDS
    assert len(rows) == MACHINES_PER_CLASS * len(MACHINE_SPECS)
    assert {machine_class: sum(row["machine_class"] == machine_class for row in rows)
            for machine_class, *_ in MACHINE_SPECS} == {
        "low": MACHINES_PER_CLASS,
        "medium": MACHINES_PER_CLASS,
        "high": MACHINES_PER_CLASS,
    }

    for row in rows:
        z = float(row["normal_z"])
        base_rate = float(row["base_transient_failure_rate_per_second"])
        base_price = float(row["base_price_per_hour"])
        assert float(row["transient_failure_rate_per_second"]) == pytest.approx(
            max(0.0, 1.05 * base_rate + 0.05 * base_rate * z)
        )
        assert float(row["price_per_hour"]) == pytest.approx(
            max(0.0, 1.05 * base_price - 0.05 * base_price * z)
        )


def test_samples_choose_size_uniformly_and_replicas_without_replacement(tmp_path):
    inventory_path = tmp_path / "inventory.csv"
    samples_path = tmp_path / "samples.csv"
    generate_inventory(inventory_path, seed=29)
    generate_samples(samples_path, inventory_path, count=3_000, seed=29)
    rows = _rows(samples_path)

    placements = [tuple(row["machine_ids"].split(";")) for row in rows]
    assert len(set(placements)) == len(placements)
    assert all(len(placement) == len(set(placement)) for placement in placements)
    assert all(len(placement) in REPLICA_COUNTS for placement in placements)

    # A generous deterministic sanity bound catches accidental nonuniform size
    # selection without turning this into a flaky statistical test.
    counts = {
        replica_count: sum(len(placement) == replica_count for placement in placements)
        for replica_count in REPLICA_COUNTS
    }
    assert all(850 <= count <= 1_150 for count in counts.values())
