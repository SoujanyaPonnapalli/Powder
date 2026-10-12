"""Runs to converge: naive versus stratified Monte Carlo.

Run: .venv/bin/python -m notebooks.stratified_convergence_study [--phase ...]

Phases (each caches to outputs/stratified-convergence/ and can be rerun):

* ``markov``: finite-horizon Markov availability of each heterogeneous RSM
  at 1 day, 30 days and 365 days -- NO_ORPHANS where a dense solve fits,
  SIMPLIFIED otherwise.  RSMs below 99.99% at a horizon are dropped there:
  the simulator is only used in practice on RSMs the Markov chain already
  predicts reliable.
* ``mc``: for each kept (RSM, horizon, metric), every arm -- naive,
  proportional and Bernstein-optimal -- runs to the empirical-Bernstein
  tolerance ``(1 - A_markov) / 2`` at 99% confidence, over several disjoint
  seeds.  That is the rigorous comparison: the rule cannot stop on a sample
  that simply has not seen a rare event yet.
* ``report``: REPORT.md and results.json.  Besides the Bernstein run counts
  it reports *known-variance* run counts, ``z^2 V / tolerance^2``, for naive,
  proportional and Neyman allocation, from per-stratum moments pooled over
  every stratified run of the cell.  That is the textbook variance-reduction
  comparison, free of the Bernstein range term.

Replacement runs with ``safe_mode`` off, so a cluster that lost a quorum of
data can still promote caught-up standbys and recover -- the Markov model's
assumption too.  With it on, such runs stay down for the rest of the horizon
while standbys pile up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from notebooks.availability_convergence_study import (
    ROOT, VM_PROFILES, days, node_config_for, raft_protocol, replacement_strategy,
)
from notebooks.raft_markov_quality_benchmark import FAILURE_TIMEOUT_S
from notebooks.availability_finite_horizon import dense_time_average
from powder.mc_jobs import build_job, ensure_binary
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    ClusterState, NetworkState, NodeReplacementStrategy, NodeState, Seconds,
)

OUT = ROOT / "outputs/stratified-convergence"
CLASSES = ("Standard", "Unreliable", "Spot")
MIXES = {3: (1, 1, 1), 5: (2, 2, 1), 7: (3, 2, 2)}
HORIZONS = {"1d": days(1), "30d": days(30), "365d": days(365)}
# Dense augmented expm needs (states + 1)^2 doubles; NO_ORPHANS N=7 has
# 48,068 states (17 GiB) and sparse BDF already times out at N=5.
QUALITY = {3: "NO_ORPHANS", 5: "NO_ORPHANS", 7: "SIMPLIFIED"}
THRESHOLD = 0.9999
CONFIDENCE = 0.99
METRICS = ("availability", "data_loss_probability")
ARMS = ("naive", "proportional", "bernstein")
# Arm label -> (allocation, leader strata).  ``leader_*`` arms give the
# initial Raft leader its own axes and treat "leader never fails, at most f
# follower events" as exact, not just "no events at all".
ARM_SPECS = {
    "naive": ("naive", False),
    "proportional": ("proportional", False),
    "neyman": ("neyman", False),
    "bernstein": ("bernstein", False),
    "leader_proportional": ("proportional", True),
    "leader_neyman": ("neyman", True),
    "leader_bernstein": ("bernstein", True),
}
MAX_RUNS = 50_000_000
Z = 2.5758293035489004  # two-sided 99%


def configs_for(nodes):
    profiles = {p.name: node_config_for(p) for p in VM_PROFILES}
    return [profiles[name] for name, n in zip(CLASSES, MIXES[nodes]) for _ in range(n)]


def cluster_for(configs):
    nodes = {f"node{i}": NodeState(node_id=f"node{i}", config=c) for i, c in enumerate(configs)}
    return ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=len(configs))


def unsafe_replacement(cfg):
    return NodeReplacementStrategy(failure_timeout=Seconds(FAILURE_TIMEOUT_S),
                                   default_node_config=cfg, safe_mode=False)


def dump(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


# -- Markov filter ----------------------------------------------------------


def markov_phase():
    path = OUT / "markov.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["nodes"], r["horizon"]) for r in rows}
    for nodes in MIXES:
        configs = configs_for(nodes)
        quality = QUALITY[nodes]
        model = None
        for label, horizon in HORIZONS.items():
            if (nodes, label) in done:
                continue
            if model is None:
                start = time.perf_counter()
                # Replacements and the initial leader are Standard (configs[0]).
                model = build_markov_model(configs, raft_protocol(),
                                           replacement_strategy(configs[0]), QualityLevel[quality])
                build_seconds = time.perf_counter() - start
            start = time.perf_counter()
            distribution, diagnostics = dense_time_average(model, horizon)
            availability = 1 - float(distribution[~model.live_mask].sum())
            row = {"nodes": nodes, "mix": dict(zip(CLASSES, MIXES[nodes])), "horizon": label,
                   "horizon_seconds": horizon, "quality": quality, "states": model.num_states,
                   "availability": availability, "kept": availability >= THRESHOLD,
                   "tolerance": (1 - availability) / 2, "build_seconds": build_seconds,
                   "solve_seconds": time.perf_counter() - start}
            print(f"Markov N={nodes} {label}: {availability:.9f} ({quality}, "
                  f"{model.num_states} states, {row['solve_seconds']:.1f} s)", flush=True)
            rows.append(row)
            dump(path, rows)
    return rows


# -- Monte Carlo ------------------------------------------------------------


def seed_for(*parts):
    digest = hashlib.sha256(json.dumps(parts).encode()).digest()
    return int.from_bytes(digest[:7], "little")


def make_job(row, metric, arm, replicate):
    configs = configs_for(row["nodes"])
    job = build_job(cluster_for(configs), unsafe_replacement(configs[0]), raft_protocol(),
                    num_simulations=1, max_time=row["horizon_seconds"], stop_on_data_loss=False,
                    base_seed=seed_for(row["nodes"], row["horizon"], metric, arm, replicate),
                    mode="stratified",
                    job_id=f"N{row['nodes']}-{row['horizon']}-{metric}-{arm}-r{replicate}")
    allocation, leader = ARM_SPECS[arm]
    job["stratified"] = {"allocation": allocation, "leader_strata": leader, "exact_checks": 20,
                         "metric": metric, "confidence_level": CONFIDENCE,
                         "absolute_error": row["tolerance"], "prune_fraction": 0.1,
                         "max_runs": MAX_RUNS, "growth": 1.1,
                         "emit_strata": True}
    return job


def mc_phase(markov, replicates, jobs, only, arms, nodes_only):
    path = OUT / "mc.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            done.add(json.loads(line)["job_id"])
    pending = []
    for row in markov:
        if not row["kept"] or (only and row["horizon"] not in only):
            continue
        if nodes_only and row["nodes"] not in nodes_only:
            continue
        for metric in METRICS:
            for arm in arms:
                for replicate in range(replicates):
                    job = make_job(row, metric, arm, replicate)
                    if job["job_id"] not in done:
                        pending.append(job)
    # Shortest horizons first, so partial results are useful early.
    order = {label: i for i, label in enumerate(HORIZONS)}
    pending.sort(key=lambda j: (order[j["job_id"].split("-")[1]], j["job_id"]))
    print(f"{len(pending)} jobs pending ({len(done)} cached)", flush=True)
    if not pending:
        return
    binary = ensure_binary()
    process = subprocess.Popen([str(binary), "--stream", "-j", str(jobs), "--batch-size", "1",
                                "--unordered"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True)


    def feed():
        # A separate thread: the engine applies backpressure on input until
        # its results are read, so writing everything first would deadlock.
        for job in pending:
            process.stdin.write(json.dumps(job) + "\n")
        process.stdin.close()

    threading.Thread(target=feed, daemon=True).start()
    with path.open("a") as out:
        for count, line in enumerate(process.stdout, 1):
            result = json.loads(line)
            if result.get("error"):
                raise RuntimeError(f"{result['job_id']}: {result['error']}")
            record = result["stratified"]
            out.write(json.dumps({"job_id": result["job_id"],
                                  "elapsed_seconds": result["elapsed_seconds"], **record}) + "\n")
            out.flush()
            print(f"[{count}/{len(pending)}] {result['job_id']}: {record['total_runs']:,} runs, "
                  f"converged={record['converged']}, {result['elapsed_seconds']:.1f} s", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"engine exited {process.returncode}")


# -- Report -----------------------------------------------------------------


def pool(moments):
    """Combine (n, mean, variance) triples into one (Chan et al.)."""
    n, mean, m2 = 0, 0.0, 0.0
    for k, mu, var in moments:
        if k == 0:
            continue
        delta = mu - mean
        total = n + k
        m2 += var * (k - 1) + delta * delta * n * k / total
        mean += delta * k / total
        n = total
    return n, mean, (m2 / (n - 1) if n > 1 else 0.0)


def known_variance(records, metric, tolerance):
    """Known-variance run counts from per-stratum moments pooled over runs."""
    key = "availability" if metric == "availability" else "data_loss"
    strata = {}
    pruned = records[0]["pruned_mass"]
    for r in records:
        for s in r["strata"]:
            entry = strata.setdefault(tuple(s["counts"]), {"weight": s["weight"],
                                                          "exact": s["exact"], "moments": []})
            entry["moments"].append((s["runs"], s[f"{key}_mean"], s[f"{key}_variance"]))
    rows = []
    for entry in strata.values():
        n, mean, var = pool(entry["moments"])
        rows.append((entry["weight"], entry["exact"], n, mean, var))
    sampled = [r for r in rows if not r[1]]
    sampled_mass = sum(r[0] for r in sampled)
    overall = sum(w * m for w, _, _, m, _ in rows) + 0.5 * pruned
    within = sum(w * v for w, exact, _, _, v in rows if not exact)
    between = sum(w * (m - overall) ** 2 for w, _, _, m, _ in rows)
    neyman = sum(w * math.sqrt(v) for w, exact, _, _, v in rows if not exact) ** 2
    bias = 0.5 * pruned
    room = tolerance - bias

    def runs(variance, margin):
        return Z * Z * variance / margin ** 2

    return {
        "stratum_runs_pooled": sum(r[2] for r in rows),
        "strata_with_variance": sum(1 for r in sampled if r[4] > 0),
        "strata_sampled": len(sampled),
        "variance_total": within + between,
        "variance_within": within,
        "naive_runs": runs(within + between, tolerance),
        "proportional_runs": runs(within * sampled_mass, room),
        "neyman_runs": runs(neyman, room),
        # The pruned-mass imputation alone makes `between` positive.
        "observed_anything": any(r[4] > 0 for r in sampled),
    }


def summarise(markov):
    records = [json.loads(line) for line in (OUT / "mc.jsonl").read_text().splitlines()]
    by_cell, by_rsm = {}, {}
    for r in records:
        nodes, horizon, metric, arm, _ = r["job_id"].split("-")
        by_cell.setdefault((int(nodes[1:]), horizon, metric), {}).setdefault(arm, []).append(r)
        if arm != "naive":
            # Both metrics' jobs share strata: same tolerance, same axes.
            family = "leader" if arm.startswith("leader_") else "count"
            by_rsm.setdefault((int(nodes[1:]), horizon, family), []).append(r)
    reference = {(r["nodes"], r["horizon"]): r for r in markov}
    cells = []
    for (nodes, horizon, metric), arms in sorted(by_cell.items(),
                                                key=lambda kv: (list(HORIZONS).index(kv[0][1]), kv[0])):
        tolerance = reference[(nodes, horizon)]["tolerance"]
        cell = {"nodes": nodes, "horizon": horizon, "metric": metric,
                "markov_availability": reference[(nodes, horizon)]["availability"],
                "tolerance": tolerance, "arms": {}}
        for arm, rs in arms.items():
            runs = [r["total_runs"] for r in rs]
            cell["arms"][arm] = {
                "replicates": len(rs),
                "all_converged": all(r["converged"] for r in rs),
                "runs_median": statistics.median(runs),
                "runs": runs,
                "seconds_median": statistics.median(r["elapsed_seconds"] for r in rs),
                "estimate_median": statistics.median(r["target"]["mean"] for r in rs),
                "estimates": [r["target"]["mean"] for r in rs],
                "half_widths": [r["target"]["half_width"] for r in rs],
                "variance_terms": [r["target"]["variance_term"] for r in rs],
                "range_terms": [r["target"]["range_term"] for r in rs],
                "num_strata": rs[0]["num_strata"], "exact_mass": rs[0]["exact_mass"],
                "pruned_mass": rs[0]["pruned_mass"], "axes": rs[0]["axes"],
                "native_nodes": sum(sum(a["native_nodes"] for a in r["axes"]) for r in rs),
            }
        for family in ("count", "leader"):
            if (nodes, horizon, family) in by_rsm:
                cell.setdefault("known_variance_by_family", {})[family] = known_variance(
                    by_rsm[(nodes, horizon, family)], metric, tolerance)
        if (nodes, horizon, "count") in by_rsm:
            cell["known_variance"] = cell["known_variance_by_family"]["count"]
            naive = arms.get("naive", [])
            if naive:
                # Cross-check: naive's own sample variance, pooled over seeds.
                n, _, var = pool((r["total_runs"], r["target"]["mean"],
                                  (r["target"]["variance_term"] ** 2) * r["total_runs"]
                                  / (2 * math.log(4 / (1 - CONFIDENCE)))) for r in naive)
                cell["known_variance"]["naive_runs_from_naive_samples"] = Z * Z * var / tolerance ** 2
        cells.append(cell)
    return cells


def fmt_runs(value):
    if value is None:
        return "—"
    return f"{value:,.0f}"


def ratio(base, value):
    if not base or not value:
        return "—"
    return f"{base / value:.2f}×"


def report_phase(markov):
    cells = summarise(markov)
    lines = ["# Stratified versus naive Monte Carlo: runs to converge\n",
             "Heterogeneous Raft RSMs with replacement (`safe_mode` off). Tolerance is "
             "`(1 − A_markov) / 2` at 99% confidence for availability and data-loss probability "
             "alike.\n",
             "## Markov filter\n",
             "| N | Mix (Std/Unrel/Spot) | Horizon | Model | States | Availability | Kept | Tolerance |",
             "|---:|---|---|---|---:|---:|---|---:|"]
    for r in markov:
        mix = "/".join(str(v) for v in r["mix"].values())
        lines.append(f"| {r['nodes']} | {mix} | {r['horizon']} | {r['quality']} | {r['states']:,} | "
                     f"{100*r['availability']:.6f}% | {'yes' if r['kept'] else 'no'} | "
                     f"{r['tolerance']:.2e} |")
    for metric in METRICS:
        title = metric.replace("_", " ").capitalize()
        lines += [f"\n## {title}: empirical-Bernstein stopping (rigorous)\n",
                  "Median runs over disjoint seeds until the Bernstein half-width reaches the "
                  "tolerance.\n",
                  "| N | Horizon | Naive | Proportional | vs naive | Bernstein-optimal | vs naive | "
                  "Estimate (naive) | Strata | Exact mass | All converged |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for cell in (c for c in cells if c["metric"] == metric):
            a = cell["arms"]
            get = lambda arm, key: a.get(arm, {}).get(key)
            converged = all(x["all_converged"] for x in a.values())
            lines.append(
                f"| {cell['nodes']} | {cell['horizon']} | {fmt_runs(get('naive', 'runs_median'))} | "
                f"{fmt_runs(get('proportional', 'runs_median'))} | "
                f"{ratio(get('naive', 'runs_median'), get('proportional', 'runs_median'))} | "
                f"{fmt_runs(get('bernstein', 'runs_median'))} | "
                f"{ratio(get('naive', 'runs_median'), get('bernstein', 'runs_median'))} | "
                f"{get('naive', 'estimate_median') or 0:.3e} | {get('proportional', 'num_strata') or 0:,} | "
                f"{get('proportional', 'exact_mass') or 0:.3f} | {'yes' if converged else 'NO'} |")
        lines += [f"\n## {title}: known-variance run counts\n",
                  "`z² V / tolerance²` with per-stratum variances pooled over every stratified run "
                  "of the RSM and horizon. No range term: this is the classic variance-reduction "
                  "comparison, and it is only as good as the rare strata are sampled.\n",
                  "| N | Horizon | Naive | Naive (own samples) | Proportional | vs naive | Neyman | "
                  "vs naive | Strata with observed variance |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in (c for c in cells if c["metric"] == metric and "known_variance" in c):
            k = cell["known_variance"]
            if not k["observed_anything"]:
                lines.append(f"| {cell['nodes']} | {cell['horizon']} | no event observed in "
                             f"{k['stratum_runs_pooled']:,} stratified runs | | | | | | 0 |")
                continue
            lines.append(
                f"| {cell['nodes']} | {cell['horizon']} | {fmt_runs(k['naive_runs'])} | "
                f"{fmt_runs(k.get('naive_runs_from_naive_samples'))} | {fmt_runs(k['proportional_runs'])} | "
                f"{ratio(k['naive_runs'], k['proportional_runs'])} | {fmt_runs(k['neyman_runs'])} | "
                f"{ratio(k['naive_runs'], k['neyman_runs'])} | "
                f"{k['strata_with_variance']:,} / {k['strata_sampled']:,} |")
    for metric in METRICS:
        title = metric.replace("_", " ").capitalize()
        lines += [f"\n## {title}: every arm, Bernstein stopping\n",
                  "Median runs over seeds; ratio is naive / arm. `leader_*` arms stratify on the "
                  "initial leader separately and treat leader-survives, at-most-f-follower-events "
                  "strata as exact (each verified by 20 identical reruns).\n",
                  "| N | Horizon | Arm | Runs | vs naive | Seconds | Strata | Exact mass | Converged |",
                  "|---:|---|---|---:|---:|---:|---:|---:|---|"]
        for cell in (c for c in cells if c["metric"] == metric):
            base = cell["arms"].get("naive", {}).get("runs_median")
            for arm in ARM_SPECS:
                a = cell["arms"].get(arm)
                if not a:
                    continue
                lines.append(f"| {cell['nodes']} | {cell['horizon']} | {arm} | {fmt_runs(a['runs_median'])} | "
                             f"{ratio(base, a['runs_median'])} | {a['seconds_median']:.1f} | "
                             f"{a['num_strata']:,} | {a['exact_mass']:.3f} | "
                             f"{'yes' if a['all_converged'] else 'NO'} |")
        lines += [f"\n## {title}: known-variance run counts by strata family\n",
                  "| N | Horizon | Strata | Naive | Proportional | vs naive | Neyman | vs naive |",
                  "|---:|---|---|---:|---:|---:|---:|---:|"]
        for cell in (c for c in cells if c["metric"] == metric):
            for family, k in cell.get("known_variance_by_family", {}).items():
                if not k["observed_anything"]:
                    lines.append(f"| {cell['nodes']} | {cell['horizon']} | {family} | no event in "
                                 f"{k['stratum_runs_pooled']:,} runs | | | | |")
                    continue
                lines.append(f"| {cell['nodes']} | {cell['horizon']} | {family} | {fmt_runs(k['naive_runs'])} | "
                             f"{fmt_runs(k['proportional_runs'])} | {ratio(k['naive_runs'], k['proportional_runs'])} | "
                             f"{fmt_runs(k['neyman_runs'])} | {ratio(k['naive_runs'], k['neyman_runs'])} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    dump(OUT / "results.json", {
        "generated": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": sys.version, "confidence": CONFIDENCE, "threshold": THRESHOLD,
        "max_runs": MAX_RUNS, "markov": markov, "cells": cells})
    print((OUT / "REPORT.md").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("markov", "mc", "report", "all"), default="all")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--horizons", nargs="*", choices=list(HORIZONS), default=None)
    parser.add_argument("--nodes", nargs="*", type=int, choices=list(MIXES), default=None)
    parser.add_argument("--arms", nargs="*", choices=list(ARM_SPECS), default=list(ARMS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    markov = markov_phase()
    if args.phase in ("mc", "all"):
        mc_phase(markov, args.replicates, args.jobs, args.horizons, args.arms, args.nodes)
    if args.phase in ("report", "all"):
        report_phase(markov)


if __name__ == "__main__":
    main()
