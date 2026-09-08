"""Fixed-sample bounded confidence intervals and a tail audit of saved MC data.

Run: .venv/bin/python -m notebooks.availability_skew_diagnostics
No new simulations are needed: the saved mean, unbiased variance, sample count,
and tail observations are sufficient. Bounds are per scenario, not simultaneous
across scenarios, and are not valid under arbitrary optional stopping.
"""

import json
import math

from notebooks.availability_convergence_study import ROOT
from scipy.stats import binomtest


def empirical_bernstein(mean: float, sd: float, count: int, confidence: float = .99) -> dict:
    """Two-sided Maurer–Pontil bound for IID observations in [0, 1].

    Apply Theorem 4 to X and 1-X with failure probability delta/2 each.
    sd is the unbiased sample standard deviation (ddof=1).
    https://arxiv.org/abs/0907.3740
    """
    if count < 2 or not 0 < confidence < 1 or not 0 <= mean <= 1 or sd < 0:
        raise ValueError("Require n>=2, confidence in (0,1), mean in [0,1], sd>=0")
    if not all(math.isfinite(x) for x in (mean, sd)):
        raise ValueError("Mean and standard deviation must be finite")
    log_term = math.log(4 / (1 - confidence))
    radius = math.sqrt(2 * sd**2 * log_term / count) + 7 * log_term / (3 * (count - 1))
    return {"radius": radius, "ci": [max(0, mean - radius), min(1, mean + radius)],
            "precision_5e_6_certified": radius <= 5e-6}


def audit(row: dict) -> dict:
    count = row["runs"]
    tail = row["tail_samples_below_99pct"]
    k = len(tail)
    probability_ci = binomtest(k, count).proportion_ci(confidence_level=.99, method="exact")
    tail_downtime = sum(1 - x["availability"] for x in tail) / count
    downtime = 1 - row["mean"]
    return {
        "profile": row["profile"], "nodes": row["nodes"], "runs": count,
        "mean": row["mean"], "nominal_t_ci99": row["ci99"],
        "bounded_ci99": empirical_bernstein(row["mean"], row["std"], count),
        "tail_definition": "weekly availability < 0.99", "tail_count": k,
        "tail_probability": k / count,
        "tail_probability_exact_ci99": [probability_ci.low, probability_ci.high],
        "mean_downtime": downtime, "tail_contribution_to_mean_downtime": tail_downtime,
        "other_contribution_to_mean_downtime": downtime - tail_downtime,
        "fraction_of_observed_downtime_in_tail": tail_downtime / downtime if downtime else 0,
    }


def main() -> None:
    folder = ROOT / "outputs/availability-convergence-study"
    source = json.loads((folder / "validation.json").read_text())
    rows = [audit(row) for row in source["results"]]
    (folder / "skew_diagnostics.json").write_text(json.dumps({
        "source": "validation.json", "confidence": .99,
        "method": "Two-sided empirical Bernstein; fixed sample, IID, bounded [0,1]",
        "reference": "https://arxiv.org/abs/0907.3740", "results": rows,
    }, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
