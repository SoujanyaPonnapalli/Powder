"""One-week time-averaged Markov availability, starting healthy with a leader.

Run: .venv/bin/python -m notebooks.availability_finite_horizon
Dense solves are capped at 2,000 states by default. To measure a larger case:
  ... --profiles Standard --nodes 5 --qualities FULL --max-states 10000
Use an external time limit for large runs; N=7 FULL is too large for dense expm.

The study solver computes the same integral as time_averaged_distribution,
using a dense (n+1)-state augmentation instead of a sparse 2n-state exponential.
Production model semantics and solver defaults remain unchanged.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# This import applies the study's one-thread settings before numerical imports.
from notebooks.availability_convergence_study import (
    ROOT, THREAD_VARIABLES, VM_PROFILES, QualityLevel, build_markov_model,
    days, node_config_for, raft_protocol, replacement_strategy, np, scipy, os,
)
from powder.markov_solver import steady_state, time_averaged_distribution
from scipy import sparse
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def dense_time_average(model, horizon: float) -> tuple[np.ndarray, dict]:
    """Integral via exp([[T Q^T, p0], [0,0]])'s upper-right column.

    On the unit interval, that column is integral_0^1 exp(T Q^T s) p0 ds.
    This handles singular Q and absorbing states without a stationary-tail
    assumption. Availability is evaluated using the unavailable-state mass,
    which avoids losing small downtime probabilities by summing live states.
    """
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and nonnegative")
    n = model.num_states
    matrix = np.zeros((n + 1, n + 1))
    matrix[:n, :n] = model.Q.T.toarray() * horizon
    matrix[:n, n] = model.initial_distribution
    evolved = expm(matrix)
    raw = evolved[:n, n]
    terminal = evolved[:n, :n] @ model.initial_distribution
    mass = float(raw.sum())
    diagnostics = {
        "raw_average_mass_error": mass - 1,
        "raw_terminal_mass_error": float(terminal.sum()) - 1,
        "minimum_raw_probability": float(raw.min()),
        "integral_identity_residual_inf": float(np.max(np.abs(
            horizon * (model.Q.T @ raw) - (terminal - model.initial_distribution)
        ))),
    }
    if abs(mass - 1) > 1e-7 or raw.min() < -1e-10:
        raise RuntimeError(f"Unacceptable exponential probability error: {diagnostics}")
    return np.maximum(raw, 0) / np.maximum(raw, 0).sum(), diagnostics


def bdf_time_average(model, horizon: float) -> tuple[float, dict]:
    """Independent stiff ODE check: p'=Q^T p, u'=unavailable_mass/T."""
    n = model.num_states
    reward = (~model.live_mask).astype(float) / horizon
    jac = sparse.bmat([[model.Q.T, sparse.csr_matrix((n, 1))],
                      [sparse.csr_matrix(reward[None, :]), sparse.csr_matrix((1, 1))]], format="csc")
    initial = np.r_[model.initial_distribution, 0.]
    start = time.perf_counter()
    solution = solve_ivp(lambda _, y: jac @ y, (0., horizon), initial,
                         method="BDF", jac=jac, rtol=1e-10, atol=1e-14)
    seconds = time.perf_counter() - start
    if not solution.success:
        raise RuntimeError(solution.message)
    return 1 - float(solution.y[-1, -1]), {
        "bdf_seconds": seconds, "bdf_steps": len(solution.t),
        "bdf_mass_error": float(solution.y[:-1, -1].sum() - 1),
        "bdf_rtol": 1e-10, "bdf_atol": 1e-14,
    }


def compare_mc(profile: str, nodes: int) -> dict:
    folder = ROOT / "outputs/availability-convergence-study"
    validation = json.loads((folder / "validation.json").read_text())["results"]
    match = next((r for r in validation if r["profile"] == profile and r["nodes"] == nodes), None)
    if match:
        return {"mc_mean": match["mean"], "mc_ci99": match["ci99"],
                "mc_runs": match["runs"], "mc_source": "validation.json"}
    archive = json.loads((ROOT / "notebooks/availability_convergence_original.json").read_text())
    experiment = next(e for e in archive["experiments"] if e["name"] == "spot_pilot_3_5")
    match = next(json.loads(line) for line in experiment["output"].splitlines()
                 if json.loads(line)["n"] == nodes)
    from scipy.stats import t
    half = float(t.ppf(.995, match["runs"] - 1)) * match["std"] / np.sqrt(match["runs"])
    return {"mc_mean": match["mean"], "mc_ci99": [match["mean"]-half, match["mean"]+half],
            "mc_runs": match["runs"], "mc_source": "historical Spot pilot"}


def measure(profile, nodes, quality, max_states, repeats, verify):
    cfg = node_config_for(profile)
    start = time.perf_counter()
    model = build_markov_model([cfg] * nodes, raft_protocol(), replacement_strategy(cfg), quality)
    build_seconds = time.perf_counter() - start
    row = {"profile": profile.name, "nodes": nodes, "quality": quality.name,
           "states": model.num_states, "horizon_seconds": days(7),
           "initial_availability": float(model.initial_distribution @ model.live_mask),
           "build_seconds": build_seconds, **compare_mc(profile.name, nodes)}
    if model.num_states > max_states:
        return {**row, "skipped": True, "reason": f"Dense state cap {max_states}",
                "dense_matrix_gib": (model.num_states + 1)**2 * 8 / 1024**3}
    solve_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        average, diagnostics = dense_time_average(model, days(7))
        solve_times.append(time.perf_counter() - start)
    weekly = 1 - float(average[~model.live_mask].sum())
    start = time.perf_counter()
    stationary = steady_state(model, backend="scipy")
    stationary_seconds = time.perf_counter() - start
    stationary_availability = 1 - float(stationary[~model.live_mask].sum())
    row.update({"skipped": False, "weekly_availability": weekly,
                "steady_availability": stationary_availability,
                "weekly_minus_steady": weekly - stationary_availability,
                "weekly_minus_mc": weekly - row["mc_mean"],
                "inside_nominal_mc_ci99": bool(row["mc_ci99"][0] <= weekly <= row["mc_ci99"][1]),
                "dense_solve_seconds": float(np.median(solve_times)),
                "dense_total_seconds": build_seconds + float(np.median(solve_times)),
                "steady_solve_seconds": stationary_seconds, "repeats": repeats,
                "diagnostics": diagnostics})
    if verify and quality == QualityLevel.SIMPLIFIED:
        start = time.perf_counter()
        reference = time_averaged_distribution(model, days(7))
        row["existing_sparse_seconds"] = time.perf_counter() - start
        row["existing_sparse_max_probability_difference"] = float(np.max(np.abs(reference-average)))
        if row["existing_sparse_max_probability_difference"] > 1e-9:
            raise RuntimeError("Dense/sparse disagreement")
    if verify and quality in (QualityLevel.SIMPLIFIED, QualityLevel.NO_ORPHANS):
        bdf, check = bdf_time_average(model, days(7))
        row.update(check)
        row["bdf_availability_difference"] = bdf - weekly
        if abs(bdf - weekly) > 1e-9:
            raise RuntimeError("Dense/BDF disagreement")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=[p.name for p in VM_PROFILES], choices=[p.name for p in VM_PROFILES])
    parser.add_argument("--nodes", nargs="+", type=int, default=[3, 5, 7], choices=[3, 5, 7])
    parser.add_argument("--qualities", nargs="+", default=[q.name for q in QualityLevel], choices=[q.name for q in QualityLevel])
    parser.add_argument("--max-states", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/availability-convergence-study/finite_horizon.json")
    args = parser.parse_args()
    if not 1 <= args.max_states <= 10000 or args.repeats < 1:
        parser.error("Require 1–10000 max states and positive repeats")
    data = {"metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision_before_run": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "thread_limits": {k:os.environ[k] for k in THREAD_VARIABLES},
        "method": "Dense augmented (n+1)-state matrix exponential, normalized average distribution",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
    }, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for profile in VM_PROFILES:
        if profile.name not in args.profiles:
            continue
        for nodes in args.nodes:
            for name in args.qualities:
                row = measure(profile, nodes, QualityLevel[name], args.max_states, args.repeats, args.verify)
                data["results"].append(row)
                args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
