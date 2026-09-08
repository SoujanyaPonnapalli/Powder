"""Render the simplified report and append current evidence to the full report.

Run after availability_finite_horizon and availability_skew_diagnostics:
    .venv/bin/python -m notebooks.availability_study_reports
The historical full report remains intact outside the generated section.
"""

import json

from notebooks.availability_convergence_study import ROOT

FOLDER = ROOT / "outputs/availability-convergence-study"
MARKER = "<!-- GENERATED: finite horizon and skew audit -->"


def duration(seconds):
    return f"{seconds * 1000:.2f} ms" if seconds < 1 else f"{seconds:.2f} s"


def main():
    base = json.loads((FOLDER / "finite_horizon.json").read_text())["results"]
    large = json.loads((FOLDER / "finite_horizon_full5.json").read_text())["results"]
    by_key = {(r["profile"], r["nodes"], r["quality"]): r for r in base + large}
    rows = list(by_key.values())
    audits = json.loads((FOLDER / "skew_diagnostics.json").read_text())["results"]
    solved = [r for r in rows if not r["skipped"]]
    profiles = ("Standard", "Unreliable", "Spot")
    comparison = ["| Profile | Nodes | MC weekly mean | Markov weekly SIMPLIFIED | Markov weekly NO_ORPHANS | NO_ORPHANS weekly − steady |",
                  "|---|---:|---:|---:|---:|---:|"]
    for profile in profiles:
        for nodes in (3, 5, 7):
            a = by_key[profile, nodes, "SIMPLIFIED"]
            b = by_key[profile, nodes, "NO_ORPHANS"]
            comparison.append(f"| {profile} | {nodes} | {a['mc_mean']:.9f} | {a['weekly_availability']:.9f} | {b['weekly_availability']:.9f} | {b['weekly_minus_steady']:.3e} |")
    runtime = ["| Nodes | Quality | States | One-week build + solve | Steady-state build + solve |",
               "|---:|---|---:|---:|---:|"]
    for nodes in (3, 5, 7):
        for quality in ("SIMPLIFIED", "NO_ORPHANS", "FULL"):
            row = by_key["Standard", nodes, quality]
            if row["skipped"]:
                runtime.append(f"| {nodes} | {quality} | {row['states']:,} | Not computed: dense memory limit | Not rerun |")
            else:
                runtime.append(f"| {nodes} | {quality} | {row['states']:,} | {duration(row['dense_total_seconds'])} | {duration(row['build_seconds']+row['steady_solve_seconds'])} |")
    detail_table = ["| Profile | Nodes | Quality | States | Weekly availability | Weekly build + solve | Status |",
                    "|---|---:|---|---:|---:|---:|---|"]
    for r in rows:
        detail_table.append(f"| {r['profile']} | {r['nodes']} | {r['quality']} | {r['states']:,} | " +
                            (f"{r['weekly_availability']:.12f} | {duration(r['dense_total_seconds'])} | Measured |" if not r['skipped']
                             else f"— | — | {r['reason']} |"))
    (FOLDER / "finite_horizon.md").write_text("# One-week Markov evidence\n\n" + "\n".join(detail_table) + "\n")
    sparse_errors = [r["existing_sparse_max_probability_difference"] for r in solved if "existing_sparse_max_probability_difference" in r]
    sparse_times = [r["existing_sparse_seconds"] for r in solved if "existing_sparse_seconds" in r]
    bdf_errors = [abs(r["bdf_availability_difference"]) for r in solved if "bdf_availability_difference" in r]
    standard_shift = max(abs(r["weekly_minus_steady"]) for r in solved if r["profile"] == "Standard")
    bound_rows = ["| Profile | Nodes | Nominal t half-width | Bounded 99% radius | Tail observations (<99% weekly availability) |",
                  "|---|---:|---:|---:|---:|"]
    for r in audits:
        half = (r['nominal_t_ci99'][1]-r['nominal_t_ci99'][0])/2
        bound_rows.append(f"| {r['profile']} | {r['nodes']} | {half:.3e} | {r['bounded_ci99']['radius']:.3e} | {r['tail_count']} / {r['runs']:,} |")
    full = f"""{MARKER}

## Handling skew: protect inference, then improve sampling

The target is **mean weekly downtime**, `D = 1 - availability`. Transforming
to downtime improves interpretation; it does not itself reduce variance.
Keep prolonged outages in the estimate. Dropping them, winsorizing, or reporting
only a median answers a different question. Ordinary resampling of a pilot
cannot reveal a failure path absent from that pilot.

The study now adds a conservative fixed-sample, two-sided **empirical Bernstein
interval** for IID samples in `[0,1]`. With unbiased sample variance `s²` and
`delta = 0.01`, its radius is:

```text
sqrt(2 * s² * log(4/delta) / n) + 7 * log(4/delta) / (3*(n-1))
```

This applies [Maurer and Pontil, Theorem 4](https://arxiv.org/abs/0907.3740) to
both tails. Intersect the resulting interval with `[0,1]`. The guarantee is per
scenario at a fixed, predeclared sample count; it is not simultaneous across
scenarios and does not justify repeatedly stopping at the first passing check.
It protects inference without requiring normality, but is deliberately wide.

{chr(10).join(bound_rows)}

None of these saved 100,000-run samples passes the bounded-radius requirement
`<= 5e-6`. This does not show that their means are wrong; it shows the difference
between a useful empirical estimate and a distribution-free accuracy claim.
The added range term prevents zero observed events from implying zero risk.
These diagnostics are in [skew_diagnostics.json](skew_diagnostics.json); the
production Monte Carlo convergence rule has not been changed.

For Unreliable N=3, the single severe week accounts for **80.5% of observed mean
downtime**. For Spot N=7, the five severe weeks account for **22.8%**. Their
two-sided exact 99% binomial intervals for severe-week probability are roughly
`[5.01e-8, 7.43e-5]` and `[1.08e-5, 1.41e-4]`, respectively. Even observing zero
severe weeks in 100,000 runs leaves an upper endpoint about `5.30e-5`. Tail
frequency and tail severity both need to be estimated.

### Recommended next implementation

1. Retain the arithmetic mean, tail counts, conditional outage duration, and
   both nominal and bounded intervals. Use a predeclared production sample size
   or a confidence sequence designed for sequential stopping.
2. Validate intended recovery semantics. The recorded quorum outage persists
   because safe-mode promotion requires an existing committing quorum. If the
   deployed system uses an operator restore or disaster-recovery procedure,
   model that procedure and its recovery delay/data-loss consequences explicitly.
   Changing this policy changes the system being estimated.
3. Implement **importance sampling of dangerous failure paths**, with exact
   path likelihood weights. Increase the frequency of overlapping failures in
   the proposal, and weight downtime back to the original probability law.
   Include survival/censoring terms and all active-node exposure, including
   replacement nodes. Simply increasing failure rates and averaging is biased.
4. Tune the proposal on a pilot, then freeze it for an independent validation
   run. Check likelihood normalization, weight concentration, contribution of
   tail paths, and agreement with tractable analytic models and ordinary MC in
   an easier regime. Report measured variance reduction before promising speedup.

This follows the [importance-sampling approach described by Art Owen](https://artowen.su.domains/mc/Ch-var-is.pdf).
Stratification is another option: for a predeclared dangerous event `C`,
`E[D] = P(C) E[D|C] + (1-P(C)) E[D|not C]`. It helps only when we can estimate
the stratum weights and sample the conditional paths correctly. Multilevel
splitting can target intermediate states close to quorum loss. These samplers
are recommendations; they have not yet been implemented or benchmarked here.

## One-week finite-horizon Markov comparison

All calculations start healthy with a selected leader and average availability
over **604,800 seconds**. They integrate the whole first week; they are not the
probability of being available at the end of the week.

{chr(10).join(comparison)}

MC values use the previous 100,000-run validation except Spot N=3/5, which use
the recorded 5,000-run pilots. Their uncertainty remains as documented above.
For Standard, the largest weekly-versus-stationary shift among computed quality
levels is only **{standard_shift:.3e}**. Matching the horizon therefore does not
resolve the much larger Markov/MC differences seen for Spot. The models still
differ in safe-mode recovery, timeout, and synchronization semantics. Inclusion
inside a wide nominal MC interval is not proof of model equivalence.

### One-core runtime (Standard)

{chr(10).join(runtime)}

The weekly figures use the **study's dense augmented matrix exponential**.
They include model construction and matrix assembly/normalization, exclude
interpreter startup, and do not include the independent validation checks.
Small-model solve times are medians of three repeats; construction and
steady-state solves are single measurements. The expensive N=5 FULL case was
measured once; the first model build can include initialization overhead.
These timings are not timings of the unchanged production
`time_averaged_distribution` implementation.

The existing sparse augmented-exponential routine was also measured for all
nine SIMPLIFIED cases: about **{min(sparse_times):.2f}–{max(sparse_times):.2f} seconds per weekly solve**, compared
with milliseconds including build for the study's dense SIMPLIFIED method.
It agrees with the dense average distributions to
**{max(sparse_errors):.2e}** maximum absolute state-probability difference.
Independent stiff BDF integration checked SIMPLIFIED and NO_ORPHANS for all
profiles/sizes; the largest availability difference was **{max(bdf_errors):.2e}**.
Detailed timings, probability-mass errors, and integral residuals are saved in
[finite_horizon.json](finite_horizon.json), with the larger Standard N=5 FULL
run in [finite_horizon_full5.json](finite_horizon_full5.json). All quality rows,
including explicit skips, are in [finite_horizon.md](finite_horizon.md).

The dense method uses the upper-right column of
`exp([[T*Q.T, p0], [0, 0]])`, which is the exact time-average integral in exact
arithmetic. It handles singular generators and absorbing states without a
stationary-tail approximation. Floating-point residuals and independent
checks measure numerical agreement, not modeling accuracy. Tests also check a
two-state closed form, zero horizon, and an absorbing-outage model.

There are **{len(solved)} computed scenario/quality combinations**. The routine
sweep caps dense models at 2,000 states; Standard N=5 FULL was added separately.
N=7 FULL alone needs about **35.4 GiB for one dense matrix**, before exponential
work arrays, exceeding this host's 32 GB RAM. N=7 MERGED_PIPELINE and the other
two N=5 FULL profiles were also omitted from this bounded study. They are marked
uncomputed, not assigned extrapolated finite-horizon results.

### Reproduce the new evidence and reports

```sh
.venv/bin/python -m notebooks.availability_skew_diagnostics
.venv/bin/python -m notebooks.availability_finite_horizon --verify
.venv/bin/python -m notebooks.availability_finite_horizon --profiles Standard --nodes 5 --qualities FULL --max-states 10000 --repeats 1 --output outputs/availability-convergence-study/finite_horizon_full5.json
.venv/bin/python -m notebooks.availability_study_reports
.venv/bin/python -m pytest -q tests/test_availability_study_methods.py
```

The reports are [SUMMARY.md](SUMMARY.md) for a concise reading and this full
report for assumptions, historical measurements, and reproducibility details.
"""
    original = (FOLDER / "REPORT.md").read_text().split(MARKER)[0].rstrip()
    (FOLDER / "REPORT.md").write_text(original + "\n\n" + full)
    short_runtime = ["| Nodes | SIMPLIFIED weekly | NO_ORPHANS weekly | FULL weekly |",
                     "|---:|---:|---:|---:|"]
    for nodes in (3,5,7):
        values = []
        for quality in ("SIMPLIFIED","NO_ORPHANS","FULL"):
            row = by_key["Standard",nodes,quality]
            values.append("Not computed" if row["skipped"] else duration(row["dense_total_seconds"]))
        short_runtime.append(f"| {nodes} | " + " | ".join(values) + " |")
    summary = f"""# Availability study — simplified report

The fastest Markov model is useful for screening. Reliable Monte Carlo error
bars require treating rare, prolonged outages explicitly.

## What the simulations say

- **Standard/Unreliable, 5–7 nodes:** all 100 fresh batches of 1,000 weekly runs
  per configuration passed the nominal target-band test. Availability is near
  `0.999998`. This is an empirical starting budget, not a guaranteed 99% result.
- **Standard, 3 nodes:** only 93/100 batches passed.
- **Unreliable, 3 nodes:** only 91/100 passed. A week with **4.43% availability**
  exposed a quorum-recovery deadlock and accounted for 80.5% of observed downtime.
- **Spot:** rare outages make the mean expensive to estimate. The seven-node
  pilot suggests about **2.1 million runs / 2.35 hours in Python** for a nominal
  99% half-width of `0.000005`; only five severe weeks informed that estimate.

## How to handle the skew

Keep the arithmetic mean and report tail frequency and outage duration.
The new conservative bounded confidence intervals do **not** certify the
requested precision from any saved 100,000-run sample. Narrow Student-t
intervals alone are insufficient evidence when influential paths may be missed.

Next, validate the intended disaster-recovery policy and implement **importance
sampling with correct likelihood weights** so dangerous failure paths are
sampled more often without biasing the mean. Validate the sampler independently
before using it to claim convergence. Changing recovery policy changes the
modeled system; it must reflect how the deployment actually recovers.

The bounded-interval diagnostic is implemented; the importance sampler is a
recommended next step. [Method sources: empirical Bernstein](https://arxiv.org/abs/0907.3740)
and [importance sampling](https://artowen.su.domains/mc/Ch-var-is.pdf).

## Does a one-week Markov horizon change the answer?

Very little for Standard: the largest measured shift from steady state is
**{standard_shift:.2e}**. Finite-horizon calculation fixes the observation window;
it does not fix differences in recovery-policy assumptions.

{chr(10).join(comparison)}

MC means have sampling uncertainty; Spot N=3/5 use older 5,000-run pilots,
while the other rows use 100,000 runs. Markov's optimistic Spot results remain
a modeling issue after matching the horizon.

## What does the one-week calculation cost?

Measured on one core, Standard profile, including build and solve:

{chr(10).join(short_runtime)}

These use a verified dense matrix-exponential method in the study code.
The first build includes initialization overhead; that affects the 3-node row.
Steady-state SIMPLIFIED is still roughly 1 ms; five-node FULL is roughly 13 s
at steady state. Seven-node FULL is too large for this dense method on the host.
Agreement between Markov quality levels does not establish simulator accuracy.
Rust speedups remain hypothetical: no Rust port was measured.

[Full report and reproducibility](REPORT.md) · [All finite-horizon results](finite_horizon.md)
"""
    (FOLDER / "SUMMARY.md").write_text(summary)


if __name__ == "__main__":
    main()
