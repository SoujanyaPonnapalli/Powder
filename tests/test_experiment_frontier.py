"""The published figure may not drift from its frozen simulation outputs."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("figure,results", [
    ("placement-frontier.html", "frontier-deterministic.jsonl"),
    ("placement-frontier-10k.html", "rsm10000-ford-frontier.jsonl"),
])
def test_plotted_points_match_reproducible_frontier(figure, results):
    page = (ROOT / "outputs/exec-02" / figure).read_text()
    plotted = json.loads(page.split("const data = ", 1)[1].split(";\nconst NS", 1)[0])
    source = [json.loads(line) for line in
              (ROOT / "outputs/exec-02" / results).read_text().splitlines()]
    rows = {row["price"]: row for row in source
            if row["pool"] == 1000 and row["price_seed"] == 0}
    assert {panel["id"] for panel in plotted} == {"rho0", "rho1"}
    for panel in plotted:
        row = rows[panel["id"]]
        for mark, key, availability in (
            ("blind", "blind", "mean_availability"),
            ("cheap", "cheapest_first", "mean_availability"),
            ("ours", "min_cost_at_blind_target", "mean_availability_exact"),
            ("equal", "max_availability_at_blind_cost", "mean_availability_exact"),
        ):
            assert panel[mark] == [row[key]["cost_per_hour"], row[key][availability]]
        raw = [[p["cost_per_hour"], p["mean_availability_exact"]] for p in row["frontier"]]
        distinct = [p for i, p in enumerate(raw) if i == 0 or p != raw[i - 1]]
        assert panel["frontier"] == distinct


def test_ten_thousand_rsm_frontier_starts_at_cheapest_budget():
    rows = [json.loads(line) for line in
            (ROOT / "outputs/exec-02/rsm10000-ford-frontier.jsonl").read_text().splitlines()]
    assert {row["price"] for row in rows} == {"rho0", "rho1"}
    for row in rows:
        assert row["frontier_has_baseline_floor"] is False
        assert row["frontier"][0]["budget_per_hour"] == row["cheapest_first"]["cost_per_hour"]
        assert all(point["status"] == "0" for point in row["frontier"])
        assert row["blind"]["deployed"] == row["min_cost_at_blind_target"]["deployed"] == 60
