"""Four AZ-aware figures preserve every baseline and real RF mix."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.render_az_frontiers import RF_LABELS, render


def point(cost, availability, certified=0):
    return {"status": "0", "cost_per_hour": cost,
            "mean_availability": availability,
            "mean_ci_low": availability - 1e-7,
            "mean_ci_high": availability + 1e-7,
            "certified_rsms": certified, "active_machines": 100,
            "rf_counts": {"3": 4000, "5": 3000, "7": 3000}}


def test_four_figures_and_mixed_rf_counts_render_as_svg():
    arms = {}
    for price in ("rho0", "rho1"):
        baselines = {rf: {
            "blind": {"status": "scored", "cost_per_hour": rf * 10,
                      "mean_availability": .99997 + rf * 1e-6,
                      "certified_groups": 10},
            "cheap": {"status": "scored", "cost_per_hour": rf * 8,
                      "mean_availability": .99996 + rf * 1e-6,
                      "certified_groups": 5},
        } for rf in (3, 5, 7)}
        frontiers = {name: {
            "status": "scored", "points": [point(23, .999970), point(45, .999975, 10000)],
            "target_qualified": point(40, .999975, 10000),
        } for name in RF_LABELS}
        arms[price] = {"baselines": baselines, "frontiers": frontiers}
    result = {"scenario": "scenario/canonical-central-az-aware@v2",
              "target_nines": 4.5, "arms": arms}
    for name in RF_LABELS:
        image = render(result, name)
        ET.fromstring(image)
        assert "Price-blind" in image and "Cheapest-first" in image
        assert "500 RSM replicas/machine" in image
        if name == "choice":
            assert "RF counts" in image


def test_saved_figures_match_checkpointed_mc_source():
    root = Path(__file__).resolve().parents[1] / "outputs/exec-02"
    result = json.loads((root / "az-frontiers.json").read_text())
    assert result["scenario"] == "scenario/canonical-central-az-aware@v2"
    for name in RF_LABELS:
        image = render(result, name)
        assert (root / f"az-frontier-{name}.svg").read_text() == image
    for price in ("rho0", "rho1"):
        arm = result["arms"][price]
        assert {int(k) for k in arm["baselines"]} == {3, 5, 7}
        for point in arm["frontiers"]["choice"]["points"]:
            if point["status"] == "0":
                assert sum(point["rf_counts"].values()) == 10000
                assert 0 <= point["certified_rsms"] <= 10000
