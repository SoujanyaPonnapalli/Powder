"""How often is a 99% Monte Carlo confidence interval wrong?

Run: .venv/bin/python -m notebooks.ci_coverage_study [--phase ...]

Phases (each caches to outputs/ci-coverage/ and can be rerun):

* ``rsms``: draw random heterogeneous Raft RSMs -- N in {3, 5, 7}, one to
  three node classes with random failure, recovery and data-loss means,
  random election, replacement-timeout and spawn means -- and keep 1000
  whose one-day SIMPLIFIED Markov availability is at least 99.9%, spread
  evenly over log10(1 - A) in [-5.5, -3]: 100 per tenth of that range.
  Above 99.9997% a single Bernstein run needs over ten million samples, so
  those RSMs are left out to keep the study near an hour.
* ``test``: for each RSM and metric (availability, data-loss probability),
  one naive Monte Carlo run to tolerance ``(1 - A_markov) / 2`` at 99%
  confidence.  The run stops on the empirical-Bernstein half-width; its
  trace also records the normal (CLT) half-width at every check, so the
  CLT rule's stopping point and interval come from the same run sequence.
* ``ref``: an independent naive reference per RSM with ten times the runs
  of its longer test, giving each metric's value with its own
  Bernstein interval.
* ``strat``: the same test with leader-stratified proportional sampling,
  every sampled stratum held to at least 30 runs so its variance is not
  trusted on two samples.  Stops on Bernstein; the stratified normal
  interval ``z sqrt(V) + bias`` is read off the trace.
* ``report``: three stopping rules are read off each run's trace --
  ``bernstein`` (where the run stopped), ``clt`` (first check whose normal
  half-width met the tolerance) and ``engine_ci``, the engine's existing
  ``converged`` check at 99%: a Student-t interval for availability, and
  for data-loss probability a Wald interval with the rule of three
  (``3 / n``, converged once ``n >= 3 / tolerance``) when no loss is seen.
  Each tested interval is *covered* when it contains the whole
  reference interval's centre band, *missed* when it is disjoint from the
  reference interval, and *ambiguous* otherwise.

The simulator, not the Markov model, is the ground truth here: the two
differ, and the question is about the Monte Carlo interval.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from scipy.stats import t as scipy_t

from notebooks.availability_convergence_study import ROOT, days
from notebooks.availability_finite_horizon import dense_time_average
from powder.mc_jobs import build_job, ensure_binary
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation import (
    ClusterState, Constant, Exponential, NetworkState, NodeConfig,
    NodeReplacementStrategy, NodeState, RaftLikeProtocol, Seconds,
)

OUT = ROOT / "outputs/ci-coverage"
HORIZON = days(1)
THRESHOLD = 0.999
CONFIDENCE = 0.99
TARGET = 1000
REF_FACTOR = 5
BANDS = 10
LOG_RANGE = (-5.5, -3.0)
MAX_RUNS = 500_000_000
METRICS = ("availability", "data_loss_probability")
RULES = ("engine_ci", "clt", "bernstein")


def log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def draw_rsm(rng, index):
    nodes = rng.choice((3, 5, 7))
    classes = rng.randint(1, 3)
    # Random composition of `nodes` into `classes` positive parts.
    cuts = sorted(rng.sample(range(1, nodes), classes - 1))
    counts = [b - a for a, b in zip([0, *cuts], [*cuts, nodes])]
    return {
        "index": index, "nodes": nodes,
        "classes": [{
            "count": count,
            "mtbf_s": log_uniform(rng, days(1), days(60)),
            "recovery_s": log_uniform(rng, 300, 4 * 3600),
            "mttdl_s": log_uniform(rng, days(0.25), days(3 * 365)),
        } for count in counts],
        "election_s": log_uniform(rng, 1, 60),
        "failure_timeout_s": log_uniform(rng, 120, 1200),
        "spawn_s": log_uniform(rng, 30, 300),
    }


def configs_for(rsm):
    configs = []
    for c in rsm["classes"]:
        cfg = NodeConfig(
            region="us-east", cost_per_hour=0.1,
            failure_dist=Exponential(rate=1 / c["mtbf_s"]),
            recovery_dist=Exponential(rate=1 / c["recovery_s"]),
            data_loss_dist=Exponential(rate=1 / c["mttdl_s"]),
            log_replay_rate_dist=Constant(1e6),
            snapshot_download_time_dist=Constant(1.0),
            spawn_dist=Exponential(rate=1 / rsm["spawn_s"]),
        )
        configs += [cfg] * c["count"]
    return configs


def scenario_for(rsm, safe_mode):
    configs = configs_for(rsm)
    # node0 is the first class: the initial leader and the replacement template.
    cluster = ClusterState(
        nodes={f"node{i}": NodeState(node_id=f"node{i}", config=c) for i, c in enumerate(configs)},
        network=NetworkState(), target_cluster_size=len(configs))
    strategy = NodeReplacementStrategy(failure_timeout=Seconds(rsm["failure_timeout_s"]),
                                       default_node_config=configs[0], safe_mode=safe_mode)
    protocol = RaftLikeProtocol(election_time_dist=Exponential(rate=1 / rsm["election_s"]))
    return configs, cluster, strategy, protocol


def dump(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rsms_phase():
    path = OUT / "rsms.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(20261012)
    per_band = TARGET // BANDS
    bands = [[] for _ in range(BANDS)]
    drawn, start = 0, time.perf_counter()
    while any(len(b) < per_band for b in bands):
        rsm = draw_rsm(rng, drawn)
        drawn += 1
        configs, _, strategy, protocol = scenario_for(rsm, safe_mode=True)
        model = build_markov_model(configs, protocol, strategy, QualityLevel.SIMPLIFIED)
        distribution, _ = dense_time_average(model, HORIZON)
        rsm["markov_availability"] = 1 - float(distribution[~model.live_mask].sum())
        rsm["markov_states"] = model.num_states
        unavailability = 1 - rsm["markov_availability"]
        if rsm["markov_availability"] < THRESHOLD or unavailability <= 0:
            continue
        position = (math.log10(unavailability) - LOG_RANGE[0]) / (LOG_RANGE[1] - LOG_RANGE[0])
        band = min(int(position * BANDS), BANDS - 1)
        if 0 <= position <= 1 and len(bands[band]) < per_band:
            rsm["tolerance"] = unavailability / 2
            rsm["band"] = band
            bands[band].append(rsm)
    kept = sorted((r for b in bands for r in b), key=lambda r: r["index"])
    print(f"kept {len(kept)} of {drawn} drawn RSMs in {time.perf_counter() - start:.0f} s", flush=True)
    dump(path, kept)
    return kept


# -- Monte Carlo ------------------------------------------------------------


def job_for(rsm, job_id, metric, seed, tolerance, min_runs, max_runs, trace, stratified=None):
    _, cluster, strategy, protocol = scenario_for(rsm, safe_mode=False)
    job = build_job(cluster, strategy, protocol, num_simulations=1, max_time=HORIZON,
                    stop_on_data_loss=False, base_seed=seed, mode="stratified", job_id=job_id)
    job["stratified"] = {"allocation": "naive", "metric": metric,
                         "confidence_level": CONFIDENCE, "absolute_error": tolerance,
                         "min_runs": min_runs, "max_runs": max_runs, "growth": 1.1,
                         "trace": trace}
    job["stratified"].update(stratified or {})
    return job


def stream(jobs, path, threads):
    """Run jobs through the engine, appending each result line to `path`."""
    if not jobs:
        return
    process = subprocess.Popen([str(ensure_binary()), "--stream", "-j", str(threads),
                                "--batch-size", "1", "--unordered"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def feed():
        # The engine applies backpressure on input until results are read.
        for job in jobs:
            process.stdin.write(json.dumps(job) + "\n")
        process.stdin.close()

    threading.Thread(target=feed, daemon=True).start()
    started = time.perf_counter()
    with path.open("a") as out:
        for count, line in enumerate(process.stdout, 1):
            result = json.loads(line)
            if result.get("error"):
                # Kept, not fatal: an exact stratum that turns out not to be
                # deterministic is a finding about that RSM.
                print(f"{result['job_id']}: {result['error']}", flush=True)
                out.write(json.dumps({"job_id": result["job_id"], "error": result["error"]}) + "\n")
                out.flush()
                continue
            out.write(json.dumps({"job_id": result["job_id"],
                                  "elapsed_seconds": result["elapsed_seconds"],
                                  **result["stratified"]}) + "\n")
            out.flush()
            if count % 50 == 0 or count == len(jobs):
                print(f"[{count}/{len(jobs)}] {time.perf_counter() - started:.0f} s", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"engine exited {process.returncode}")


def load(path):
    if not path.exists():
        return {}
    return {r["job_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def test_phase(rsms, threads):
    path = OUT / "test.jsonl"
    done = load(path)
    jobs = [job_for(rsm, f"{rsm['index']}-{metric}", metric, 1_000_000 + 2 * rsm["index"] + i,
                    rsm["tolerance"], 30, MAX_RUNS, True)
            for rsm in rsms for i, metric in enumerate(METRICS)
            if f"{rsm['index']}-{metric}" not in done]
    print(f"test: {len(jobs)} pending", flush=True)
    stream(jobs, path, threads)


STRATIFIED = {"allocation": "proportional", "leader_strata": True, "min_stratum_runs": 30,
              "exact_checks": 3, "max_strata": 2_000_000}


def strat_phase(rsms, threads, limit=None, metrics=METRICS):
    path = OUT / "strat.jsonl"
    done = load(path)
    jobs = [job_for(rsm, f"{rsm['index']}-{metric}-strat", metric, 5_000_000 + 2 * rsm["index"] + i,
                    rsm["tolerance"], 30, MAX_RUNS, True, STRATIFIED)
            for rsm in rsms[:limit] for i, metric in enumerate(METRICS)
            if metric in metrics and f"{rsm['index']}-{metric}-strat" not in done]
    print(f"strat: {len(jobs)} pending", flush=True)
    stream(jobs, path, threads)


def ref_phase(rsms, threads):
    tests = load(OUT / "test.jsonl")
    path = OUT / "ref.jsonl"
    done = load(path)
    jobs = []
    for rsm in rsms:
        job_id = f"{rsm['index']}-ref"
        if job_id in done:
            continue
        longest = max(tests[f"{rsm['index']}-{m}"]["total_runs"] for m in METRICS)
        runs = min(REF_FACTOR * longest, MAX_RUNS)
        # min_runs = max_runs: one batch of exactly `runs`, no early stop.
        jobs.append(job_for(rsm, job_id, "availability", 9_000_000 + rsm["index"],
                            1e-300, runs, runs, False))
    print(f"ref: {len(jobs)} pending", flush=True)
    stream(jobs, path, threads)


# -- Report -----------------------------------------------------------------


Z99 = 2.5758293035489004


def engine_ci(row, metric, tolerance):
    """`check_convergence`'s half-width at one trace row, or None if not converged."""
    n, mean = row["runs"], row["mean"]
    if metric == "data_loss_probability":
        if mean in (0.0, 1.0):
            return 3.0 / n if n >= math.ceil(3.0 / tolerance) else None
        half = Z99 * math.sqrt(mean * (1 - mean) / n)
    else:
        # clt_half_width is z * s / sqrt(n); the engine uses Student t.
        half = row["clt_half_width"] * scipy_t.ppf(0.995, n - 1) / Z99
    return half if half <= tolerance else None


def stop_at(record, metric, rule, min_runs=0):
    """(mean, half-width, runs, zero-variance flag) where `rule` stops, or None.

    `min_runs` delays the first check: replaying the trace from the first
    check at or past it is the same seeded run sequence a larger
    ``min_runs`` would produce, up to where the checks fall."""
    for row in record["trace"]:
        if row["runs"] < min_runs:
            continue
        if rule == "clt":
            half = row["clt_half_width"] if row["clt_half_width"] <= record["tolerance"] else None
        else:
            half = engine_ci(row, metric, record["tolerance"])
        if half is not None:
            return row["mean"], half, row["runs"], row["clt_half_width"] == 0.0
    return None


def verdict(lo, hi, ref_mean, ref_half):
    if lo <= ref_mean - ref_half and ref_mean + ref_half <= hi:
        return "covered"
    if hi < ref_mean - ref_half or ref_mean + ref_half < lo:
        return "missed"
    return "ambiguous"


def wilson(k, n, z=2.5758):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


REF_HALF = "half_width"
SWEEP_FIXED = (30, 100, 300, 1_000, 3_000, 10_000, 30_000)
SWEEP_SCALED = (1, 3, 4.6, 10)


def min_runs_sweep(rsms, tests, refs, ref_half="half_width"):
    """Miss rates of the CLT and engine rules as the first check is delayed."""
    variants = [(f"{m:,}", lambda rsm, m=m: m) for m in SWEEP_FIXED]
    variants += [(f"{c} / tolerance", lambda rsm, c=c: math.ceil(c / rsm["tolerance"]))
                 for c in SWEEP_SCALED]
    global REF_HALF
    REF_HALF = ref_half
    which = "empirical-Bernstein" if ref_half == "half_width" else "normal (CLT)"
    lines = [f"\n## Delaying the first check (judged against the reference's {which} interval)\n",
             "The CLT and engine rules replayed with a larger minimum run count before the first "
             "check, on the same run sequences. `c / tolerance` scales the minimum with the "
             "tolerance, the way the rule of three does for data loss. *Unresolved* counts RSMs whose "
             "trace ended before the rule stopped.\n",
             "| Minimum runs | Metric | Rule | Missed | Miss rate (99% Wilson) | Ambiguous | Unresolved | "
             "Median runs at stop |",
             "|---|---|---|---:|---|---:|---:|---:|"]
    for label, minimum in variants:
        for metric in METRICS:
            for rule in ("engine_ci", "clt"):
                verdicts, runs, unresolved = [], [], 0
                for rsm in rsms:
                    ref = refs.get(f"{rsm['index']}-ref")
                    if not ref:
                        continue
                    hit = stop_at(tests[f"{rsm['index']}-{metric}"], metric, rule, minimum(rsm))
                    if hit is None:
                        unresolved += 1
                        continue
                    mean, half, n, _ = hit
                    truth = ref[metric]
                    verdicts.append(verdict(mean - half, mean + half, truth["mean"],
                                            truth[REF_HALF]))
                    runs.append(n)
                missed = verdicts.count("missed")
                lo, hi = wilson(missed, len(verdicts))
                lines.append(f"| {label} | {metric} | {rule} | {missed} | "
                             f"{100 * missed / max(len(verdicts), 1):.1f}% ({100*lo:.1f}–{100*hi:.1f}%) | "
                             f"{verdicts.count('ambiguous')} | {unresolved} | "
                             f"{statistics.median(runs) if runs else 0:,.0f} |")
    return lines


def sweep_minimums():
    return ([(f"{m:,}", lambda rsm, m=m: m) for m in SWEEP_FIXED]
            + [(f"{c} / tolerance", lambda rsm, c=c: math.ceil(c / rsm["tolerance"]))
               for c in SWEEP_SCALED])


def naive_vs_stratified(rsms, tests, strats, refs):
    """Availability: how often each 99% interval misses, naive versus stratified.

    Judged against the reference's normal interval (millions of runs, the
    tail sampled many times): *definitely wrong* when disjoint from it,
    *point outside* when the reference estimate falls outside the interval.
    Only RSMs with a stratified result are counted, for both arms."""
    metric = "availability"
    usable = [r for r in rsms if f"{r['index']}-{metric}-strat" in strats
              and "error" not in strats[f"{r['index']}-{metric}-strat"]]
    errors = sum(1 for r in rsms if "error" in strats.get(f"{r['index']}-{metric}-strat", {}))

    def score(hits):
        wrong = point = 0
        runs = []
        for rsm, (mean, half, n) in hits:
            truth = refs[f"{rsm['index']}-ref"][metric]
            wrong += verdict(mean - half, mean + half, truth["mean"], truth["clt_half_width"]) == "missed"
            point += abs(mean - truth["mean"]) > half
            runs.append(n)
        k = len(runs)
        return (f"{100 * wrong / k:.1f}%", f"{100 * point / k:.1f}%",
                f"{statistics.median(runs):,.0f}") if k else ("—", "—", "—")

    def first(record, rule, minimum):
        hit = stop_at(record, metric, rule, minimum)
        return None if hit is None else hit[:3]

    lines = ["\n## Availability: naive versus stratified sampling\n",
             f"{len(usable)} RSMs with both results"
             + (f" ({errors} RSMs needed more than 2,000,000 strata and are left out)" if errors else "")
             + ". Naive uses the engine's `converged` check (Student t); stratified uses leader "
             "strata, proportional allocation, at least 30 runs per sampled stratum, and the "
             "stratified normal interval `z sqrt(V) + bias`. Both stop at the first check at or past "
             "the minimum whose interval meets the tolerance. *Definitely wrong*: disjoint from the "
             "reference's normal interval. *Point outside*: the reference estimate is outside the "
             "interval. A correct 99% interval is wrong about 1% of the time.\n",
             "| Minimum runs | Naive: definitely wrong | Naive: point outside | Naive: median runs | "
             "Stratified: definitely wrong | Stratified: point outside | Stratified: median runs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, minimum in sweep_minimums():
        naive = [(r, first(tests[f"{r['index']}-{metric}"], "engine_ci", minimum(r))) for r in usable]
        strat = [(r, first(strats[f"{r['index']}-{metric}-strat"], "clt", minimum(r))) for r in usable]
        n_unres = sum(h is None for _, h in naive)
        s_unres = sum(h is None for _, h in strat)
        row = [label, *score([(r, h) for r, h in naive if h]), *score([(r, h) for r, h in strat if h])]
        note = f" ({n_unres} / {s_unres} unresolved)" if n_unres or s_unres else ""
        lines.append("| " + " | ".join(row) + " |" + note)
    bern = lambda rec: (rec["target"]["mean"], rec["target"]["half_width"], rec["total_runs"])
    lines.append("| **Bernstein** | " + " | ".join(
        [*score([(r, bern(tests[f"{r['index']}-{metric}"])) for r in usable]),
         *score([(r, bern(strats[f"{r['index']}-{metric}-strat"])) for r in usable])]) + " |")
    return lines


def report_phase(rsms):
    tests, refs = load(OUT / "test.jsonl"), load(OUT / "ref.jsonl")
    rows = []
    for rsm in rsms:
        ref = refs.get(f"{rsm['index']}-ref")
        if not ref:
            continue
        for metric in METRICS:
            test = tests[f"{rsm['index']}-{metric}"]
            truth = ref[metric]
            for rule in RULES:
                zero_variance = False
                if rule == "bernstein":
                    mean, half, runs = test["target"]["mean"], test["target"]["half_width"], test["total_runs"]
                    stopped = test["converged"]
                else:
                    hit = stop_at(test, metric, rule)
                    if hit is None:
                        mean, half, runs, stopped = test["target"]["mean"], math.inf, test["total_runs"], False
                    else:
                        (mean, half, runs, zero_variance), stopped = hit, True
                rows.append({
                    "index": rsm["index"], "nodes": rsm["nodes"], "band": rsm["band"], "classes": len(rsm["classes"]),
                    "markov_availability": rsm["markov_availability"], "tolerance": rsm["tolerance"],
                    "metric": metric, "rule": rule, "stopped": stopped, "runs": runs,
                    "zero_variance_stop": zero_variance,
                    "mean": mean, "half_width": half, "ref_mean": truth["mean"],
                    "ref_half_width": truth["half_width"], "ref_runs": ref["total_runs"],
                    "verdict": verdict(mean - half, mean + half, truth["mean"], truth["half_width"]),
                })
    dump(OUT / "results.json", {
        "generated": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": sys.version, "confidence": CONFIDENCE, "threshold": THRESHOLD,
        "horizon_seconds": HORIZON, "ref_factor": REF_FACTOR, "rsms": rsms, "rows": rows})

    def table(select, label):
        out = [f"| {label} | Rule | RSMs | Missed | Miss rate (99% Wilson) | Ambiguous | "
               "Missed after a zero-variance stop | Median runs |",
               "|---|---|---:|---:|---|---:|---:|---:|"]
        for key in sorted({select(r) for r in rows}, key=str):
            for metric in METRICS:
                for rule in RULES:
                    rs = [r for r in rows if select(r) == key and r["metric"] == metric and r["rule"] == rule]
                    if not rs:
                        continue
                    missed = sum(r["verdict"] == "missed" for r in rs)
                    ambiguous = sum(r["verdict"] == "ambiguous" for r in rs)
                    lo, hi = wilson(missed, len(rs))
                    name = f"{key} / {metric}" if key != "all" else metric
                    out.append(f"| {name} | {rule} | {len(rs)} | {missed} | {100*missed/len(rs):.1f}% "
                               f"({100*lo:.1f}–{100*hi:.1f}%) | {ambiguous} | "
                               f"{sum(r['verdict'] == 'missed' and r['zero_variance_stop'] for r in rs)} | "
                               f"{statistics.median(r['runs'] for r in rs):,.0f} |")
        return out

    lines = ["# How often is a 99% Monte Carlo interval wrong?\n",
             f"{len({r['index'] for r in rows})} random heterogeneous Raft RSMs with one-day SIMPLIFIED "
             f"Markov availability ≥ 99.9%. Each ran naive Monte Carlo once to half-width "
             f"`(1 − A_markov) / 2` at 99% confidence; **CLT** is the normal interval at the first check "
             f"it met the tolerance, **engine_ci** the engine's existing `converged` check (Student t; Wald "
             f"with the rule of three for data loss), **Bernstein** the empirical-Bernstein interval at "
             f"its stop. All three read the same run sequence. Truth is "
             f"an independent run with {REF_FACTOR}× the runs: *missed* means the interval is disjoint "
             f"from the reference's own 99% interval, *ambiguous* that it overlaps it only partly. "
             f"A correct 99% interval misses about 1% of the time.\n",
             "## Overall\n", *table(lambda r: "all", "Metric"),
             "\n## By cluster size\n", *table(lambda r: f"N={r['nodes']}", "Cluster"),
             "\n## By Markov availability\n",
             *table(lambda r: f"A ≥ {100 * (1 - 10 ** (LOG_RANGE[0] + (LOG_RANGE[1] - LOG_RANGE[0]) * (r['band'] + 1) / BANDS)):.5f}%", "Band")]
    strats = load(OUT / "strat.jsonl")
    if strats:
        lines += naive_vs_stratified(rsms, tests, strats, refs)
    lines += min_runs_sweep(rsms, tests, refs)
    # The reference has 5x the longest test's runs (millions), enough to have
    # sampled the tail many times, so its own normal interval is a sharper
    # yardstick once the tested intervals become narrow.
    lines += min_runs_sweep(rsms, tests, refs, "clt_half_width")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print((OUT / "REPORT.md").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("rsms", "test", "ref", "strat", "report", "all"),
                        default="all")
    parser.add_argument("--limit", type=int, default=None, help="strat phase: first N RSMs only")
    parser.add_argument("--metrics", nargs="*", choices=METRICS, default=list(METRICS),
                        help="strat phase: metrics to run")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rsms = rsms_phase()
    if args.phase in ("test", "all"):
        test_phase(rsms, args.jobs)
    if args.phase in ("ref", "all"):
        ref_phase(rsms, args.jobs)
    if args.phase == "strat":
        strat_phase(rsms, args.jobs, args.limit, args.metrics)
    if args.phase in ("report", "all"):
        report_phase(rsms)


if __name__ == "__main__":
    main()
