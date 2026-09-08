"""Render full and simplified reports from the weekly-window study evidence.

Run: .venv/bin/python -m notebooks.availability_weekly_window_reports
"""

from __future__ import annotations

import json
import math

from notebooks.availability_weekly_windows import FOLDER, ROOT
from scipy.stats import t


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 120:
        return f"{seconds:.2f} s"
    if seconds < 7200:
        return f"{seconds / 60:.2f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.2f} h"
    return f"{seconds / 86400:.2f} days"


def difference(windowed, control):
    """Independent-sample Welch interval; nominal, not a rare-tail guarantee."""
    a = windowed["std"] ** 2 / windowed["runs"]
    b = control["std"] ** 2 / control["runs"]
    variance = a + b
    degrees = variance ** 2 / (a*a/(windowed["runs"]-1) + b*b/(control["runs"]-1))
    delta = windowed["mean"] - control["mean"]
    half = float(t.ppf(.995, degrees)) * math.sqrt(variance)
    return delta, delta-half, delta+half


def main():
    mc = json.loads((FOLDER / "mc.json").read_text())
    markov = json.loads((FOLDER / "markov.json").read_text())
    audit = json.loads((FOLDER / "distribution_audit.json").read_text())
    tail_control = json.loads((FOLDER / "tail_control.json").read_text())
    if len(mc["results"]) != 18 or len(markov["results"]) != 45 or len(tail_control["results"]) != 1:
        raise RuntimeError("Complete both simulation and Markov suites first")
    rows = mc["results"] + tail_control["results"]
    windowed = {(r["profile"], r["nodes"]): r for r in rows if r["mode"] == "weekly_windows"}
    controls = {(r["profile"], r["nodes"]): r for r in rows if r["mode"] == "exponential_30d"}
    keys = [(p, n) for p in ("Standard", "Unreliable", "Spot") for n in (3, 5, 7)]
    solved = [r for r in markov["results"] if not r["skipped"]]
    markov_by_key = {(r["profile"], r["nodes"], r["quality"]): r for r in solved}
    total_runs = sum(r["runs"] for r in rows)
    mc_seconds = sum(r["wall_seconds"] for r in rows)
    spot7 = windowed["Spot", 7]
    early_seeds = {r["seed"] for r in spot7["early_stops"]}
    uninterrupted_severe = sum(r["seed"] not in early_seeds for r in spot7["tail_samples_below_99pct"])
    tail_downtime_share = sum(1-r["availability"] for r in spot7["tail_samples_below_99pct"]) / (spot7["runs"] * (1-spot7["mean"]))

    comparison = table(
        ["Profile", "Nodes", "Windowed runs", "Windowed mean", "Exponential mean", "Difference (ppm)", "Nominal 99% difference CI (ppm)"],
        [(p, n, f'{windowed[p,n]["runs"]:,}', f'{windowed[p,n]["mean"]:.9f}',
          f'{controls[p,n]["mean"]:.9f}', f'{difference(windowed[p,n], controls[p,n])[0]*1e6:+.3f}',
          f'[{difference(windowed[p,n], controls[p,n])[1]*1e6:+.3f}, {difference(windowed[p,n], controls[p,n])[2]*1e6:+.3f}]')
         for p,n in keys],
    )
    tails = table(
        ["Profile", "Nodes", "Mean", "Nominal 99% half-width", "Median", "Worst week", "Weeks below 99%", "Passing 1,000-run batches"],
        [(p, n, f'{windowed[p,n]["mean"]:.9f}', f'{windowed[p,n]["half_width"]:.3g}',
          f'{windowed[p,n]["quantiles"]["0.5"]:.9f}', f'{windowed[p,n]["quantiles"]["0"]:.9f}',
          windowed[p,n]["below_99pct"], f'{windowed[p,n]["passing_batches"]}/{len(windowed[p,n]["batches"])}')
         for p,n in keys],
    )
    legacy = table(
        ["Distribution", "Profile", "Nodes", "Early returns / runs", "Old stopped-duration mean", "Full-week mean"],
        [(r["mode"], r["profile"], r["nodes"], f'{r["early_stop_count"]}/{r["runs"]}',
          f'{r["legacy_mean"]:.9f}', f'{r["mean"]:.9f}') for r in rows],
    )
    markov_comparison = table(
        ["Profile", "Nodes", "Windowed MC", "Exponential MC", "One-week SIMPLIFIED proxy", "One-week NO_ORPHANS proxy", "NO_ORPHANS weekly − steady"],
        [(p, n, f'{windowed[p,n]["mean"]:.9f}', f'{controls[p,n]["mean"]:.9f}',
          f'{markov_by_key[p,n,"SIMPLIFIED"]["weekly_availability"]:.9f}',
          f'{markov_by_key[p,n,"NO_ORPHANS"]["weekly_availability"]:.9f}',
          f'{markov_by_key[p,n,"NO_ORPHANS"]["weekly_availability"]-markov_by_key[p,n,"NO_ORPHANS"]["steady_availability"]:.3g}')
         for p,n in keys],
    )
    runtime = table(
        ["Nodes", "SIMPLIFIED one week", "NO_ORPHANS one week", "FULL one week"],
        [(n, duration(markov_by_key["Standard",n,"SIMPLIFIED"]["weekly_total_seconds"]),
          duration(markov_by_key["Standard",n,"NO_ORPHANS"]["weekly_total_seconds"]),
          duration(markov_by_key["Standard",n,"FULL"]["weekly_total_seconds"]) if n == 3 else
          ("147.79 s (previous measurement)" if n == 5 else "Not computed")) for n in (3,5,7)],
    )
    bounds = table(
        ["Profile", "Nodes", "Conservative 99% CI", "±0.000005 certified?", "Measured MC time", "Pilot projected runs for nominal ±0.000005", "Projected Python time"],
        [(p,n, f'[{windowed[p,n]["bounded_ci99"]["ci"][0]:.9f}, {windowed[p,n]["bounded_ci99"]["ci"][1]:.9f}]',
          "Yes" if windowed[p,n]["bounded_ci99"]["precision_5e_6_certified"] else "No",
          duration(windowed[p,n]["wall_seconds"]), f'{windowed[p,n]["planned_precision_runs_normal"]:,}',
          duration(windowed[p,n]["projected_python_seconds"])) for p,n in keys],
    )
    all_markov = table(
        ["Profile", "Nodes", "Quality", "States", "One-week availability", "Build", "Weekly solve", "Steady solve"],
        [(r["profile"], r["nodes"], r["quality"], f'{r["states"]:,}',
          "Skipped" if r["skipped"] else f'{r["weekly_availability"]:.12f}', duration(r["build_seconds"]),
          "—" if r["skipped"] else duration(r["weekly_solve_seconds"]),
          "—" if r["skipped"] else duration(r["steady_solve_seconds"])) for r in markov["results"]],
    )
    phase_table = table(["Machine", "Weekly window begins (days after week starts)"],
                        [(i+1, f"{v:.3f}") for i,v in enumerate(audit["example_machine_window_starts_days"][0])])
    observed = table(["Profile", "Nodes", "Applied transient events", "Observed in-window share"],
                     [(p,n,windowed[p,n]["applied_transient_failures"],
                       f'{windowed[p,n]["observed_failure_fraction_in_window"]:.3%}') for p,n in keys])
    significant = [f"{p} N={n}" for p,n in keys if not difference(windowed[p,n],controls[p,n])[1] <= 0 <= difference(windowed[p,n],controls[p,n])[2]]
    comparison_conclusion = (
        "All nine nominal 99% difference intervals include zero. These samples do not establish a change in mean availability from the weekly concentration. This is not an equivalence result, especially for rare failures."
        if not significant else
        "Nominal 99% difference intervals exclude zero for " + ", ".join(significant) + ". These are exploratory per-scenario intervals without multiple-comparison correction; skew and unseen rare events limit interpretation."
    )
    summary_table = table(["Profile", "Nodes", "Windowed availability", "Exponential control"],
                          [(p,n,f'{100*windowed[p,n]["mean"]:.6f}%',f'{100*controls[p,n]["mean"]:.6f}%') for p,n in keys])
    spot = windowed["Spot",3]
    text = f"""# Weekly-window availability study — full report

## Main findings

This rerun preserves the old **30-day average transient-failure intensity**, puts **70% of that intensity in a one-day weekly window**, and assigns **each VM an independent weekly phase**. Failures remain possible throughout the week. The one-day width is a study assumption, not an inferred property of the deployment.

Completed **{total_runs:,} full-week simulations**: {sum(r['runs'] for r in windowed.values()):,} with weekly windows and {sum(r['runs'] for r in rows if r['mode'] == 'exponential_30d'):,} fresh exponential controls. Measured simulation time was **{duration(mc_seconds)}** on one worker, with numerical libraries limited to one thread. The final comparison uses an independent 100,000-run Spot N=7 control; the original 10,000-run Spot N=7 control remains in the evidence but is not pooled into that comparison.

{comparison_conclusion}

The study also found a separate accounting problem: the original simulator can return at data loss before the requested horizon. Availability divided by that shortened duration can overstate full-week availability. Both new distributions now run through the entire week. For windowed Spot N=3, the stopped-duration mean is **{spot['legacy_mean']:.6%}**, versus **{spot['mean']:.6%}** over the full week. The distribution comparison below uses the corrected horizon on both sides.

## Exact failure model

For VM i, draw a fixed phase φᵢ independently and uniformly in [0, 7 days). Its high window is `(time − φᵢ) mod 7 days < 1 day`. The transient-failure hazard while its clock is active is:

- High window: `0.7 × (7/30) / 1 = 0.163333333` per day.
- Remaining six days: `0.3 × (7/30) / 6 = 0.011666667` per day.
- Integrated hazard per week: `7/30`; calendar-average hazard: `1/30` per day.

The high rate is 14 times the low rate. This does **not** mean every machine fails once each week. Seventy percent describes integrated event intensity in the window; total failure frequency retains the original calibration. For an always-operational VM, the long-run mean interval is 30 days. Recovery downtime and VM replacement censor the failure clock, so realized failure frequency and window shares in the full simulator need not equal those ideal values exactly.

Recoveries retain the VM's phase; newly provisioned VMs draw fresh independent phases. There is no shared weekly trigger. Windows may overlap by chance; this avoids imposing artificial negative correlation through forced staggering. The fixed phase also makes repeated failures on a surviving VM calendar-dependent. Initially all nodes are healthy, with independent phases and an active clock starting at time zero.

The code draws an exponential value in **cumulative hazard**, then exactly inverts the piecewise-linear hazard integral from the current calendar time. Calendar waiting times are non-exponential even though the auxiliary hazard-space draw is exponential. [Time-transformation simulation method](https://pmc.ncbi.nlm.nih.gov/articles/PMC11581276/).

All other configuration parameters remain those of the preceding study: transient recovery mean 20 minutes; mean data-loss intervals 3 years / 1 year / 1 day for Standard / Unreliable / Spot; 5-second mean election; 5-minute replacement timeout; 60-second mean spawn; safe-mode replacement; original sync and commit semantics. Only transient failure scheduling changes. Permanent data-loss and recovery distributions remain exponential.

### Clock and phase audit

An always-operational 200,000-event clock produced **{audit['observed_fraction_in_window']:.4%}** of events in the high window, with a mean interevent interval of **{audit['mean_interevent_days']:.5f} days**. This is a generator calibration check, separate from the {total_runs:,} cluster simulations. A separate sample of 10,000 seven-VM phase vectors checks the independent phase draws; the histograms and correlation matrix are in [distribution_audit.json](distribution_audit.json).

One sampled seven-VM cluster has these one-day window starts; windows wrap across the week boundary as needed:

{phase_table}

Realized shares after recovery and replacement effects:

{observed}

## Controlled distribution comparison

Every control uses the original exponential 30-day transient clock and the same full-week accounting. Final control samples have 10,000 runs per configuration except Spot N=7, which has 100,000; the windowed samples have 100,000 except Spot N=3/5, which have 10,000 each. Seeds are independent across modes; the table is an independent-sample comparison, not a paired-seed experiment. Availability values are fractions; one ppm is 0.000001.

The original Spot N=7 control observed zero severe weeks in 10,000 runs, against ten in 100,000 windowed runs. Its nominal difference interval excluded zero, but the unequal samples poorly resolved this rare tail. We therefore ran a separate, fixed 100,000-run control with independent seeds, observing **{controls['Spot',7]['below_99pct']} weeks below 99% availability**. The follow-up size was fixed before collecting it, with no adaptive stopping. The table uses this new control; the original pilot is preserved in `mc.json`, and the follow-up is in `tail_control.json`.

{comparison}

Difference intervals use a nominal 99% Welch calculation from independent sample variances. They are per comparison, not simultaneous. Rare catastrophic weeks can make these intervals unreliable if the sample misses the tail. A zero-containing interval does not establish that the two models are interchangeable. Wider/narrower windows and correlated phases would be different experiments.

## Skew, tail events, and convergence

Weekly-window results:

{tails}

A passing batch has its nominal 99% Student-t interval entirely inside `[0.999990, 0.999999]`. This target-band test differs from estimating the mean to ±0.000005. If the true mean lies outside the band, increasing the run count cannot make the system meet that target. The smallest stored weekly values and their reproducible seeds are in [mc.json](mc.json).

Changing the input failure distribution does not remove the output's long lower tail. Quorum loss, safe-mode promotion rules, and prolonged recovery still create rare, expensive weeks. The new Unreliable N=3 sample happened to contain no weeks below 99%; the previous study did find a catastrophic path. That absence in a fresh sample is not evidence that the recovery risk has been fixed. Keep the arithmetic mean for expected availability and accompany it with tail frequency, severity, and uncertainty; medians alone hide downtime. Do not trim or winsorize away the failures to make a convergence plot look better.

For windowed Spot N=7, the ten weeks below 99% account for **{tail_downtime_share:.1%} of observed downtime**. Of those ten, **{uninterrupted_severe} already ran to the full horizon without an early data-loss return**. The severe tail therefore cannot be explained solely by the denominator correction.

{bounds}

The conservative intervals use the two-sided empirical Bernstein bound for independent observations in [0,1], at fixed sample sizes. No windowed configuration certifies the requested ±0.000005 precision under this bound. The normal-form sample projections use the pilot variance, can be much too optimistic if important paths were missed, and are **not convergence guarantees or recommended stopping rules**. Tiny projections for apparently stable configurations are particularly uninformative about unseen tails. Runtime projections assume the current measured cost per simulation, not a measured Rust speedup. [Bound method](https://arxiv.org/abs/0907.3740).

Recommended next work: verify the intended recovery behavior after quorum/data loss, then add and independently validate importance sampling or stratification for dangerous failure paths. Any importance sampler for this periodic model must include the calendar-dependent integrated hazard and phase sampling in its likelihood weights. An unweighted oversample of bad weeks would bias availability. This rerun implements the alternate failure model and bounded diagnostics; it does not implement that rare-event sampler. [Importance sampling reference](https://artowen.su.domains/mc/Ch-var-is.pdf).

## Correcting the one-week denominator

`Simulator.run_until(end_time=...)` returns immediately when actual data loss is detected. The earlier Monte Carlo path did not suppress that return even with `stop_on_data_loss=False`. Dividing accumulated available time by the early stopping time estimates a different quantity from a complete-week average.

The study adapter resumes the same simulator and event queue until seven days, retaining all existing protocol/strategy behavior. If there are no remaining events, it accounts for the final state through the horizon. It asserts that every sample contains exactly seven days of elapsed metrics. It does not assume recovery after disaster or introduce a new recovery policy. The production `Simulator` API itself is unchanged. A deterministic test with data loss at second 1 of a 100-second horizon produces legacy availability 1.0 and corrected availability 0.01.

{legacy}

The two availability columns in each row come from the same paths, before and after continuation; this isolates the accounting correction. The final row is the independent 100,000-run Spot N=7 control; the earlier 10,000-run row is the initial pilot. Comparing an old stopped-duration mean with a new full-week mean would confound the distribution change with this correction. The previous report remains archived with a correction notice.

## One-week Markov comparison and runtime

**These Markov values are exponential mean-rate proxies, not exact results for the weekly-window process.** The existing constant-generator model receives the unchanged 30-day-rate configuration. Its generators are identical to the exponential control by construction and are checked for exact sparse-matrix equality. This equality does not validate the periodic process approximation. An actual calendar-aware Markov calculation would need node phases, time-dependent transition rates, phase averaging, and explicit treatment of phase draws at replacement; the current reduced state counts do not retain those details.

The one-week calculation starts healthy with a leader and integrates state occupancy over seven days using the previously verified augmented dense matrix exponential. The table compares that finite-horizon proxy with full-week simulations:

{markov_comparison}

Markov also approximates timeout, sync, and recovery-policy semantics. In particular, its repair paths do not reproduce all of the simulator's prolonged safe-mode failures. Matching the horizon and mean failure rate does not resolve those modeling differences.

Standard-profile times include model build plus weekly solve:

{runtime}

The new run solves {len(solved)} of 45 profile/size/quality combinations, with a 2,000-state dense cap. All 36 weekly availability values match the preceding finite-horizon results exactly at stored precision. Timings are single measurements, with one numerical thread, and include first-use overhead; they are not median benchmarks. Generator-equality audit builds are excluded from the reported production build+solve times. The unchanged Standard FULL N=5 proxy took 147.79 seconds in the preceding study; that number is explicitly historical and was not rerun here. FULL N=7 remains uncomputed (68,952 states; its augmented 68,953-by-68,953 dense matrix alone is approximately 35.4 GiB). The larger skipped models would need a different numerical method and budget.

### All newly measured Markov cases

{all_markov}

## Reproduction and evidence

Run from the repository root:

```sh
.venv/bin/python -m notebooks.availability_weekly_windows --suite mc
.venv/bin/python -m notebooks.availability_weekly_windows --suite markov
.venv/bin/python -m notebooks.availability_weekly_window_tail_control
.venv/bin/python -m notebooks.availability_weekly_window_reports
.venv/bin/python -m pytest tests/test_weekly_window_study.py tests/test_availability_study_methods.py -q
```

The MC suite constructs fresh cluster, strategy, and protocol objects for each run. Its timing includes construction and full-week continuation, and excludes summary/report rendering; this differs slightly from the old deep-copy runner timing. Six numerical thread environment variables are fixed to one before numerical imports. Main-suite seeds use `100000000 + mode_index*100000000 + profile_index*10000000 + nodes*1000000 + run_index`, with mode order weekly/exponential and profile order Standard/Spot/Unreliable. The independent Spot N=7 control uses seeds 317000000–317099999. Disjoint seed ranges, source hashes, dependency versions, lower-tail samples, early-return records, summary statistics, and hashes of the full sample vectors are saved. Individual non-tail samples are regenerated from seeds rather than stored in full. The focused numerical, hazard-inversion, control-equivalence, and complete-horizon tests pass: **34 tests**.

- [Full MC evidence](mc.json)
- [Independent Spot N=7 control](tail_control.json)
- [Markov evidence and numerical diagnostics](markov.json)
- [Clock and phase audit](distribution_audit.json)
- [Study implementation](../../notebooks/availability_weekly_windows.py)
- [Independent control runner](../../notebooks/availability_weekly_window_tail_control.py)
- [Report renderer](../../notebooks/availability_weekly_window_reports.py)
- [Model and horizon tests](../../tests/test_weekly_window_study.py)
- [Previous study, with historical results](../availability-convergence-study/REPORT.md)

Recorded source revision before this run: `{mc['metadata']['git_revision_before_run']}`. The source SHA-256 in each evidence file identifies the new uncommitted implementation used for that run; the subsequent repository commit records it with these outputs.
"""
    (FOLDER / "REPORT.md").write_text(text)
    summary = f"""# Weekly-window availability — simplified report

## What changed

Each machine now has its own randomly positioned **one-day weekly window** containing **70% of its transient-failure intensity**. Failures remain possible during the other six days. The average rate stays at **one per 30 days**; it does not become one failure per week. A machine keeps its window after recovery; a new replacement draws a new window. Windows can overlap naturally, but there is no common fleet-wide trigger.

The clock audit measured **{audit['observed_fraction_in_window']:.2%}** in-window events and a **{audit['mean_interevent_days']:.2f}-day** average interval.

## Results

Completed **{total_runs:,} full-week simulations** in **{duration(mc_seconds)}** of measured simulation time on one worker. Both columns below use complete weeks.

The final Spot seven-node comparison uses 100,000 runs per distribution. Other controls use 10,000 runs; other windowed samples use 100,000 except Spot three/five-node samples, which use 10,000.

{summary_table}

{comparison_conclusion}

The initial 10,000-run Spot seven-node control missed the severe tail entirely. An independent 100,000-run control found {controls['Spot',7]['below_99pct']} severe weeks, compared with ten in the windowed sample. This is a concrete example of why a small sample can give misleadingly narrow error bars.

Standard and Unreliable five- and seven-node clusters stay close to **99.9998%** in these samples. Spot remains sensitive to severe outages. These means have sampling uncertainty; the full report includes intervals and tail counts.

## An important correction to the previous report

Some previous simulations stopped at data loss and divided availability by that shortened duration. The new study continues to the end of the week. For windowed Spot with three nodes, this changes the mean from **{spot['legacy_mean']:.3%}** to **{spot['mean']:.3%}** on the same paths. This is an accounting correction, not evidence that weekly concentration caused that entire drop. Fresh exponential controls receive the same correction.

## What to do about the skew

Changing failure timing does not eliminate rare disastrous weeks. Keep the mean, show the worst weeks and their frequency, and validate the system's recovery behavior after quorum loss. Then implement and validate a weighted rare-event sampler so those paths can be estimated efficiently. Do not remove the bad weeks from the average.

None of the new windowed samples certifies **±0.000005 at 99% confidence** using the conservative bounded interval. Narrow ordinary error bars are useful diagnostics, but they cannot prove that an important rare path was sampled. The rare-event sampler remains future work.

## One-week Markov calculations

The one-week Markov results are **mean-rate exponential proxies**. They cannot see the weekly windows or each machine's phase, so their values remain the same as before. They are useful for quick screening, but are not exact predictions for this alternate distribution or the simulator's disaster recovery behavior.

Standard profile, build plus solve, one numerical thread:

{runtime}

The five-node FULL time is the prior measurement of the identical proxy; the other shown computed times are new. A calendar-aware model would require additional state and time-dependent rates.

[Full report](REPORT.md) · [Simulation evidence](mc.json) · [Independent Spot control](tail_control.json) · [Markov evidence](markov.json) · [Clock audit](distribution_audit.json)
"""
    (FOLDER / "SUMMARY.md").write_text(summary)
    notice = (
        "> **Full-week accounting correction:** Some historical Monte Carlo runs below stopped early at data loss, "
        "so their availability averages do not cover a complete week. The "
        "[new full-week study](../availability-weekly-windows-study/REPORT.md) compares fresh exponential controls "
        "with independent weekly failure windows and reports the effect of this correction separately. "
        "Historical results below are retained for provenance.\n\n"
    )
    for filename in ("REPORT.md", "SUMMARY.md"):
        previous = ROOT / "outputs/availability-convergence-study" / filename
        original = previous.read_text()
        if not original.startswith("> **Full-week accounting correction:**"):
            previous.write_text(notice + original)
    print(f"Wrote {FOLDER / 'REPORT.md'} and {FOLDER / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
