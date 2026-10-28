"""The RF=3/5/7 plot is sourced from completed, correctly labelled runs."""

import math
import xml.etree.ElementTree as ET

from scripts.render_replica_frontiers import DEFAULT_FILES, load_rows, render


def test_replica_count_figure_uses_three_complete_workloads():
    rows = load_rows(DEFAULT_FILES)
    assert len(rows) == 6
    for replicas in (3, 5, 7):
        for price in ("rho0", "rho1"):
            row = rows[(replicas, price)]
            assert row["blind"]["deployed"] == 20 * replicas
            assert len(row["frontier"]) >= 6
            assert row["frontier"][0]["budget_per_hour"] == row["cheapest_first"]["cost_per_hour"]
    for price in ("rho0", "rho1"):
        checks = rows[(7, price)]["exact_spot_checks"]
        assert checks and all(math.isfinite(c["relative_unavailability_error"]) for c in checks)
        assert "binned" in rows[(7, price)]["score_method"]
    image = render(rows)
    ET.fromstring(image)
    assert "binned Markov screening (dashed)" in image
    assert "RF 3" in image and "RF 5" in image and "RF 7" in image
