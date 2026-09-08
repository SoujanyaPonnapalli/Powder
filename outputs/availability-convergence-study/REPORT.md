> **Full-week accounting correction:** Some historical Monte Carlo runs below stopped early at data loss, so their availability averages do not cover a complete week. The [new full-week study](../availability-weekly-windows-study/REPORT.md) compares fresh exponential controls with independent weekly failure windows and reports the effect of this correction separately. Historical results below are retained for provenance.

# Availability Estimation: Quality, Runtime, and Monte Carlo Convergence

Date: 2026-09-08

For a concise reading, see [the simplified report](SUMMARY.md). The final
sections add one-week Markov calculations and a concrete plan for skewed data.

## Executive summary

This study compares Powder's discrete-event Monte Carlo simulator with its
Markov availability models. It focuses on estimating weekly or annual Raft
availability near five to six nines on one CPU core.

The main conclusions are:

1. For the canonical **Standard** hardware profile, 1,000 independent
   one-week simulations are a useful starting budget at 5 or 7 nodes: all
   100 fresh batches per size had a nominal 99% interval contained within
   `[0.999990, 0.999999]`. At 3 nodes, only **93/100** fresh batches passed.
   The original claim that 1,000 runs were comfortably sufficient at every
   size was too strong. Batch pass rates are not confidence-interval coverage.
2. For the **Unreliable** profile, which changes per-node mean time to data loss
   from 3 years to 1 year, 5- and 7-node batches also passed 100/100. However,
   the 3-node validation found a week with only **4.43% availability** in
   100,000 runs. Its combined nominal 99% interval is wider than the requested
   precision. **Withdraw the 1,000-run recommendation for Unreliable N=3.**
3. This does **not** generalize to the **Spot** profile, whose per-node mean time
   to data loss is only 1 day. Spot results have a heavy lower tail caused by
   rare, prolonged quorum/data-loss episodes. Brute-force Monte Carlo then
   can require millions to billions of runs for a nominal 99% half-width of
   `0.000005`.
4. Markov `SIMPLIFIED` is also the repository default. On the canonical
   homogeneous exponential scenario, its availability differs from `FULL` by
   only about `1.3e-12` at 5 nodes while taking about 1 ms instead of 13 s.
   This is agreement between Markov approximations, not an error bound against
   the discrete-event simulator.
5. A highly optimized Rust simulation port is estimated to be 20–50 times
   faster than the Python simulator. This is an engineering extrapolation,
   not a measured Rust result.
6. One-week Markov averages barely differ from steady state for Standard;
   matching the horizon does not reconcile recovery-policy differences.
   Conservative bounded confidence intervals do not certify five-decimal
   accuracy from the saved 100,000-run MC samples. Importance sampling is the
   recommended next efficiency improvement, after validating recovery semantics.

## Scope and interpretation

Two precision requirements appear in this report:

- **Target-band containment:** the entire two-sided 99% confidence interval
  must lie inside `[0.999990, 0.999999]`.
- **Five-decimal absolute precision:** the two-sided nominal 99% confidence
  interval must have a half-width no larger than `0.000005`. This does not
  guarantee stable rounding: even a narrower interval can straddle a rounding
  boundary. To report a stable rounded value, the entire interval (intersected
  with the known `[0, 1]` support) must round to the same five-decimal value.

These requirements are different. Target-band containment also depends on
where the true availability lies within the band. Five-decimal precision only
constrains interval width.

## Canonical simulation parameters

The scenarios use the profiles and factory functions in
[`notebooks/raft_markov_quality_benchmark.py`](../../notebooks/raft_markov_quality_benchmark.py).

| Parameter | Value |
|---|---:|
| Protocol | Raft-like majority quorum |
| Cluster sizes | 3, 5, and 7 homogeneous nodes |
| Initial condition | All nodes healthy and synchronized; initial leader selected |
| Transient failure distribution | Exponential, mean 30 days per node |
| Recovery distribution | Exponential, mean 20 minutes |
| Leader election distribution | Exponential, mean 5 seconds |
| Failure replacement timeout | 5 minutes |
| Replacement spawn distribution | Exponential, mean 60 seconds |
| Replacement safe mode | Enabled |
| Log replay rate | Constant 1,000,000 units/second |
| Commit rate | 1 unit/second |
| Snapshot interval | 0; snapshots disabled |
| Log retention | 0; infinite retention |
| Network outages | None |
| Monte Carlo execution | Sequential, one core |
| Stop on data loss | False |
| Availability metric | Fraction of simulated time during which Raft can commit |

The three hardware profiles differ as follows:

| Profile | Transient MTBF | Mean time to data loss per node | Cost/hour |
|---|---:|---:|---:|
| Standard | 30 days | 3 years | $0.10 |
| Unreliable | 30 days | 1 year | $0.05 |
| Spot | 30 days | 1 day | $0.03 |

Consequently, the repository's lower-reliability profiles change permanent
data-loss reliability, not transient-failure MTBF.

## Confidence-interval method

Each simulation produces a continuous time-average availability sample
`X_i` in `[0, 1]`. Powder constructs the availability interval using the
Student-t distribution:

```text
sample mean +/- t * sample standard deviation / sqrt(n)
```

For large-sample planning, the required run count is approximated by:

```text
n ~= (z * sigma / epsilon)^2
```

where:

- `sigma` is the standard deviation among simulation results;
- `epsilon` is the desired confidence-interval half-width;
- `z = 2.576` for a two-sided 99% interval.

Powder's implementation is in
[`powder/monte_carlo.py`](../../powder/monte_carlo.py).

These are nominal intervals: the Student-t formula does not supply exact
finite-sample coverage for this strongly skewed distribution. Increasing the
sample count helps only if the influential tail is sampled adequately. See
[NIST's discussion of non-normality and skew](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/conflimi.htm).
Passing the target-band test is also different from covering the true mean;
covering an independent pilot mean does not establish coverage of the truth.
Repeatedly stopping at the first passing ordinary interval does not preserve
a fixed-sample 99% coverage claim; use a predeclared final sample size with an
independent validation sample, or a method designed for sequential inference.

## Standard hardware: one-year simulations

A one-core pilot of 1,000 one-year simulations produced:

| Nodes | Mean availability | Sample standard deviation | Python time/run |
|---:|---:|---:|---:|
| 3 | 0.999997804 | 3.359e-6 | 2.72 ms |
| 5 | 0.999998017 | 7.828e-7 | 5.12 ms |
| 7 | 0.999998002 | 7.825e-7 | 8.04 ms |

The estimated run counts needed to place the entire 99% interval inside the
target band were 57, 30, and 30 respectively. A 30-run minimum was retained
where the formula returned a smaller value so that variance could be estimated
with a minimally useful pilot.

| Nodes | Planned runs | Projected 99% interval | Python total | Estimated Rust total |
|---:|---:|---:|---:|---:|
| 3 | 57 | `[0.999996618, 0.999998991]` | 155 ms | 3.1–7.8 ms |
| 5 | 30 | `[0.999997623, 0.999998411]` | 154 ms | 3.1–7.7 ms |
| 7 | 30 | `[0.999997608, 0.999998395]` | 241 ms | 4.8–12.1 ms |

## Standard hardware: one-week simulations

Reducing the simulated horizon from one year to one week makes each run
approximately 19–29 times faster, but increases sample variance because fewer
failure and election events occur in each run.

A one-core pilot of 10,000 weekly simulations produced:

| Nodes | Mean availability | Sample standard deviation | Runs with no unavailability | Python time/run |
|---:|---:|---:|---:|---:|
| 3 | 0.999997944 | 1.057e-5 | 78.23% | 142 us |
| 5 | 0.999998063 | 5.682e-6 | 79.19% | 204 us |
| 7 | 0.999997977 | 5.880e-6 | 78.04% | 281 us |

Estimated minimum counts for target-band containment were:

| Nodes | Estimated runs | Projected 99% interval | Python total | Estimated Rust total |
|---:|---:|---:|---:|---:|
| 3 | 670 | `[0.999996889, 0.999998999]` | 95 ms | 1.9–4.8 ms |
| 5 | 248 | `[0.999997126, 0.999999000]` | 51 ms | 1.0–2.5 ms |
| 7 | 224 | `[0.999996957, 0.999998998]` | 63 ms | 1.3–3.1 ms |

Although the shorter horizon requires more independent runs, its lower cost
per run makes the total compute time similar or lower.

### Direct validation of the 1,000-run recommendation

Ten non-overlapping 1,000-run batches were executed for each cluster size. In
all 30 batches:

- the computed 99% confidence interval was contained in the target band; and
- the interval contained the corresponding 10,000-run pilot mean.

Representative batches were:

| Nodes | Estimate | 99% interval | Python time for 1,000 | Passing batches |
|---:|---:|---:|---:|---:|
| 3 | 0.999997790 | `[0.999997270, 0.999998310]` | 132 ms | 10/10 |
| 5 | 0.999997845 | `[0.999997364, 0.999998326]` | 198 ms | 10/10 |
| 7 | 0.999998104 | `[0.999997661, 0.999998548]` | 271 ms | 10/10 |

Across the ten batches, observed 99% half-widths ranged from:

- N=3: `0.417e-6` to `0.822e-6`;
- N=5: `0.411e-6` to `0.567e-6`;
- N=7: `0.410e-6` to `0.514e-6`.

The original ten batches support 1,000 weekly runs as a starting budget.
They do not establish a universal pass rate or calibrated 99% coverage.

## Less-reliable hardware

### Unreliable profile: one-year per-node MTTDL

Fresh 1,000-run weekly batches for the Unreliable profile produced:

| Nodes | Mean availability | 99% interval | Target band satisfied? |
|---:|---:|---:|---:|
| 3 | 0.999998001 | `[0.999997521, 0.999998481]` | Yes |
| 5 | 0.999997676 | `[0.999997148, 0.999998204]` | Yes |
| 7 | 0.999997723 | `[0.999997220, 0.999998225]` | Yes |

These three batches support testing a 1,000-run starting budget; they are
too few to establish the reliability of that recommendation across seeds.

### Spot profile: one-day MTTDL

Initial 1,000-run Spot tests did not reliably reveal the distribution's lower
tail. Larger pilots produced:

| Nodes | Pilot runs | Mean availability | Sample standard deviation | Below 99% | Below 90% |
|---:|---:|---:|---:|---:|---:|
| 3 | 5,000 | 0.959883510 | 0.129133 | 15.82% | 10.80% |
| 5 | 5,000 | 0.998499191 | 0.025573 | 0.46% | 0.40% |
| 7 | 10,000 | 0.999897688 | 0.004262 | 0.01% | 0.01% |

The N=7 pilot contained one run with availability around `0.574`. The earlier
1,000-run batch missed such events and underestimated standard deviation by
roughly 135 times. This demonstrates why convergence checks based only on a
small pilot can be overconfident for rare-event systems.

Using the observed pilot variances, the approximate counts for a two-sided 99%
half-width of `0.000005` are:

| Nodes | Approximate simulations | Measured Python time/run | Python, one core | Estimated Rust, one core |
|---:|---:|---:|---:|---:|
| 3 | 4.43 billion | 1.33 ms | 68 days | 1.4–3.4 days |
| 5 | 174 million | 2.62 ms | 5.3 days | 2.5–6.3 hours |
| 7 | 4.82 million | 4.10 ms | 5.5 hours | 6.6–16.5 minutes |

These estimates should be treated as orders of magnitude. In particular, the
N=7 variance estimate is controlled by a very small number of catastrophic
samples and may move substantially with a larger pilot. If the acceptable
half-width is `0.000010` instead, counts and times are one quarter as large.
The fresh 100,000-run N=7 validation below revises its planning estimate to
about 2.10 million runs; retain the table above as the original pilot evidence.

No number of simulations can make the Spot confidence interval fit inside
`[0.999990, 0.999999]` when the true mean itself is below the lower endpoint.
More simulations would make that failure more certain, not move the result
into the band.

## Reliability and Monte Carlo sample complexity

### Binary observations

If each simulation returned only `1` for available or `0` for unavailable,
with true availability `x`, then:

```text
variance = x * (1 - x)
```

For fixed absolute accuracy, variance decreases as `x` approaches one, and so
does the required sample count.

### Time-average observations

Powder instead returns the fraction of the horizon during which the system can
commit. The result is continuous, and its variance is not determined by the
mean alone. Since each result lies in `[0, 1]`, it nevertheless obeys the bound:

```text
variance <= x * (1 - x)
```

Two systems with the same mean availability can have very different Monte
Carlo complexity:

- a system unavailable for exactly one second every week has almost no
  between-run variance;
- a system that is normally perfect but occasionally unavailable for an
  entire week can have the same mean and much larger variance.

Spot hardware has elements of the second case. Rare, long outages inflate the
second moment of downtime and can materially affect its mean; the fresh N=7
pilot attributes 22.8% of observed mean downtime to its five severe weeks.

### Absolute versus relative precision

For fixed absolute precision on availability, increasing reliability generally
reduces required samples. For relative precision on the rare unavailability
probability `q = 1 - x`, the relationship reverses. For a binary estimator and
relative error `r`:

```text
n ~= z^2 / (r^2 * q)
```

Therefore, measuring a progressively rarer failure probability to a fixed
percentage requires a sample count that grows approximately as `1/q`.

Longer simulation horizons can also reduce variance by allowing more events to
self-average within each run. After startup and mixing effects become small,
variance often scales approximately as `1 / horizon`, while runtime per run
scales approximately with the horizon. Total compute for a fixed confidence
width may therefore remain roughly constant.

More simulations reduce statistical error but do not correct finite-horizon
startup bias, model error, an incorrect failure distribution, or missing
failure modes.

## Markov quality and runtime comparison

The quality levels are defined in
[`powder/scenario.py`](../../powder/scenario.py). `SIMPLIFIED` is both the lowest
quality level and the API default. `NO_ORPHANS` is documented as the recommended
balanced Raft model.

Measured on an Apple M1 Pro with numerical libraries restricted to one thread:

| Nodes | Default / `SIMPLIFIED` | Balanced / `NO_ORPHANS` | `FULL` |
|---:|---:|---:|---:|
| 3 | 16 states, 0.81 ms | 76 states, 1.47 ms | 598 states, 13.6 ms |
| 5 | 36 states, 1.02 ms | 377 states, 5.98 ms | 8,463 states, 13.2 s |
| 7 | 64 states, 1.39 ms | 1,253 states, 37.8 ms | 68,952 states, approximately 1–2 hours |

The N=7 full model built in about 1.16 seconds but its sparse steady-state solve
did not finish during a 5.5-minute bounded test. The 1–2 hour figure is an
extrapolation from measured state-count and solve-time scaling.

For the canonical N=5 Standard scenario:

| Method | One-core time | Absolute deviation from `FULL` |
|---|---:|---:|
| Markov default / `SIMPLIFIED` | 1.02 ms | 1.34e-12 |
| Markov `COLLAPSED_PIPELINE` | 1.68 ms | 1.31e-12 |
| Markov balanced / `NO_ORPHANS` | 5.98 ms | 1.58e-14 |
| Markov `MERGED_PIPELINE` | 85.1 ms | 5.6e-15 |
| Markov `FULL` | 13.2 s | Reference |

These deviations apply to the homogeneous Standard scenario and
should not be assumed for Spot hardware or arbitrary distributions without a
fresh comparison.

`FULL` is a reference within the Markov family, not an exact representation of
this simulator. [`extract_rates`](../../powder/markov_builders/common.py)
converts the fixed five-minute timeout to an exponential rate and models sync
using the snapshot-download mean, rather than the simulator's backlog-dependent
log replay. The [Raft builder](../../powder/markov_builders/raft.py) also allows
data-loss replacement in all-down states to prevent an absorbing state; the
simulator cancels replacement timeouts when all nodes are unavailable and uses
safe-mode constraints on promotion. Those differences matter particularly for
Spot's long outages. The tiny Standard differences between quality levels do
not quantify these shared modeling errors.

Normal Markov availability is a steady-state quantity and does not depend on a
one-week versus one-year observation horizon. A comparison to a weekly
simulation starting all healthy should use the finite-horizon time-averaged
distribution in
[`powder/markov_solver.py`](../../powder/markov_solver.py), rather than the
ordinary steady-state result. Matching the horizon alone does not remove the
modeling differences described above.

The new consolidated runner also re-executed all 14 tractable Markov solves;
their availabilities match the historical results within `2e-15`. It rebuilt
N=7 FULL and confirmed 68,952 states, while deliberately skipping that solve.
Fresh timings and results are saved in [`markov.json`](markov.json); the
historical timing tables above remain representative (fresh N=5 FULL: 12.96 s).

## Rust performance estimate

The estimated optimized Rust timings assume a 20–50x speedup over the measured
Python discrete-event simulator. The rationale is removal of Python object
dispatch, repeated deep copies, allocation-heavy event handling, and
Python-to-NumPy random-sampling overhead.

The estimate excludes compilation and process startup and has not been
validated with a Rust implementation. For short batches, executable startup
may dominate the event-processing time. A real port should be benchmarked
before using the estimate for capacity planning.

## Recommendations

1. Use **1,000 weekly simulations as a starting budget** for Standard and
   Unreliable profiles at N=5 and N=7, with the empirical limitations in the
   expanded validation below. Standard N=3 needs more care; Unreliable N=3
   needs rare-event treatment. Do not label a starting budget a guaranteed
   99% result.
2. Retain adaptive confidence checking, but do not trust a small pilot when the
   model can enter rare, long-lived unavailable states.
3. For Spot hardware, report distribution quantiles and tail-event counts in
   addition to the mean and Student-t confidence interval.
4. Track catastrophic/quorum-loss observations and variance stability across
   independent batches. A minimum tail count is a diagnostic, not by itself a
   proof of convergence or calibrated coverage.
5. Use importance sampling, rare-event splitting, stratification, regenerative
   analysis, or a finite-horizon Markov computation with matching failure and
   recovery semantics for five-decimal Spot results. These alternatives need
   their own validation; brute-force Monte Carlo is impractical for N=3 and N=5.
6. Use Markov `SIMPLIFIED` for routine analysis of the canonical homogeneous
   exponential scenario, and `NO_ORPHANS` when replacement-pipeline fidelity is
   important. Reserve `FULL` primarily for small-model validation.

## Expanded validation and corrected conclusions

A fresh fixed-size validation uses disjoint seed ranges and 100,000 weekly
simulations for each Standard/Unreliable cluster size, partitioned into 100
non-overlapping batches of 1,000. The results are saved in
[`validation.json`](validation.json), with a generated table in
[`validation.md`](validation.md). Batch timings in the JSON are allocated from
the full scenario runtime, not separately timed individual batches.

| Profile | Nodes | Combined mean | Combined nominal 99% interval | Passing 1,000-run batches |
|---|---:|---:|---|---:|
| Standard | 3 | 0.999997910 | `[0.999997798, 0.999998022]` | 93/100 |
| Standard | 5 | 0.999998001 | `[0.999997954, 0.999998047]` | 100/100 |
| Standard | 7 | 0.999998008 | `[0.999997961, 0.999998055]` | 100/100 |
| Unreliable | 3 | 0.999988135 | `[0.999963518, 1.000012753]` | 91/100 |
| Unreliable | 5 | 0.999997922 | `[0.999997875, 0.999997970]` | 100/100 |
| Unreliable | 7 | 0.999997893 | `[0.999997844, 0.999997941]` | 100/100 |

Intervals are displayed without clipping, matching the arithmetic used in
the original report. An upper bound above one indicates a limitation of this
interval construction; clipping it at one does not fix missing-tail coverage.

For Standard N=3, the observed standard deviation grew to `1.377e-5`, and the
fixed-variance target-band planning count rose from 670 to **1,064**. This is
only a plug-in estimate, not evidence that precisely 1,064 future runs suffice.
All seven failing 1,000-run batches should remain visible in the results,
rather than selecting only batches that pass.

For Unreliable N=3, seed **23070693** produced availability
`0.044323613179782305`: approximately **6.69 days unavailable** in a seven-day
run. Replaying it with event logging reproduced the same result. Two nodes
lost data at about 26,559 and 26,807 seconds, before the first node's five-minute
replacement timeout expired. Standby replacements spawned and synchronized,
but [`_promote_eligible_standbys`](../../powder/simulation/strategy.py) requires
the active cluster to be able to commit in safe mode. Once the active quorum
was lost, promotion could not restore it. The event trace ends with surviving
data and synchronized standbys, but no ability to commit. This is a prolonged
quorum outage under the configured recovery policy, not recorded total data
destruction. See [`unreliable-3-tail-trace.json`](unreliable-3-tail-trace.json).

That one observation contributes about `9.56e-6` to the sample's mean
unavailability. The 100,000-run interval has half-width **`2.46e-5`**, almost
five times the requested `5e-6`. Its pilot variance implies about **2.42 million
runs** for that half-width (roughly 5.6 minutes of measured Python compute),
but a plan dominated by one catastrophic observation is highly unstable.
The interval does not establish whether the true mean lies inside the target
band. Passing 91 individual batches demonstrates how easily small samples can
miss this tail; it does not override the combined evidence.

For N=5 and N=7, all 100 batches passed in both profiles. This supports a
1,000-run practical starting budget for these configurations, while leaving
rarer unobserved outage paths unresolved. The original report's universal
recommendation for Standard/Unreliable N=3, 5, and 7 is therefore withdrawn.

The fresh **Spot N=7** validation added 100,000 weekly simulations using base
seed `40000000`. It found **five runs below 90% availability**, versus one in
the original 10,000-run pilot:

| Quantity | Fresh Spot N=7 result |
|---|---:|
| Mean availability | 0.999922662 |
| Nominal 99% interval | `[0.999899764, 0.999945560]` |
| Sample standard deviation | 0.002811052 |
| Nominal 99% half-width | 0.000022898 |
| Measured Python time/run | 4.03 ms |
| Planned runs for half-width 0.000005 | 2,097,162 |
| Projected Python time, one core | 2.35 hours |
| Projected Rust time at assumed 20–50x | 2.8–7.0 minutes |

The revised count is less than half the original 4.82-million estimate,
illustrating its sensitivity to tail sampling. Five severe observations still
do not establish a stable variance estimate or calibrated 99% coverage. The
mean and the whole computed interval are below the target band; more sampling
does not improve the underlying availability. This new pilot remains too
imprecise for the requested half-width.

## Reproducibility notes

- Host: Apple M1 Pro, 10 cores, 32 GB RAM.
- Benchmarks constrained numerical-library thread counts to one.
- Monte Carlo runs used one sequential worker and deterministic base seeds.
- No Rust simulator was written for this study.
- No repository production code was changed. The reproducible study runner,
  historical source archive, and new evidence are described below.

The prior task, **Estimate simulation quality tradeoff**, ran inline Python
snippets and composed the report manually. Those exact experiment sources and
their recorded numeric outputs are now preserved in
[`availability_convergence_original.json`](../../notebooks/availability_convergence_original.json).
The archive includes the interrupted N=7 FULL solve, explicitly marked as
failed/terminated; its build output is not a measured solve time. The historical
weekly Rust calculation used 15–50x, inconsistent with the report's stated
20–50x assumption. The weekly table above now consistently uses 20–50x.

[`availability_convergence_study.py`](../../notebooks/availability_convergence_study.py)
consolidates the experiments, saves seeds, runtime/dependency metadata, sample
hashes, quantiles, low-availability samples, batch statistics, confidence
intervals, and planning estimates. It generates JSON and Markdown evidence;
the explanatory text in this report remains manually maintained.

Run from the repository root using the existing environment:

```sh
# Original Monte Carlo scenarios and seeds (45 batches/scenarios).
.venv/bin/python -m notebooks.availability_convergence_study --suite original

# 600,000 Standard/Unreliable runs plus 100,000 fresh Spot N=7 runs.
.venv/bin/python -m notebooks.availability_convergence_study --suite validation

# Markov timing sweep; builds N=7 FULL but skips its expensive solve.
.venv/bin/python -m notebooks.availability_convergence_study --suite markov

# Replay an exact historical source snippet with a child-process time limit.
.venv/bin/python -m notebooks.availability_convergence_study --replay standard_weekly_pilot
.venv/bin/python -m notebooks.availability_convergence_study --replay markov_full_7_bounded --timeout 330

# Reproduce the newly discovered quorum outage and its event log.
.venv/bin/python -m notebooks.availability_convergence_tail_trace
```

Original seed schedules (`i` is the profile index: Spot=1, Unreliable=2;
`b` is a zero-based batch index):

| Experiment | Base seed | Runs per scenario |
|---|---|---:|
| Standard annual pilot | `73000 + N*1000` | 1,000 |
| Standard weekly pilot | `173000 + N*10000` | 10,000 |
| Standard weekly validation | `904000 + N*10000 + b*1000` | 1,000 × 10 |
| Original Spot/Unreliable small batches | `1204000 + i*100000 + N*10000` | 1,000 |
| Original larger Spot pilots | `2200000 + N*100000` | 5,000 at N=3/5; 10,000 at N=7 |
| Fresh Standard validation | `10000000 + N*1000000` | 100,000 |
| Fresh Unreliable validation | `20000000 + N*1000000` | 100,000 |
| Fresh Spot N=7 validation | `40000000` | 100,000 |

Each run uses `base_seed + run_index`. Preserve the logged dependency versions
and source revision when comparing exact seeded results; wall-clock timings
can vary across executions. The study artifacts are explicitly tracked even
though general benchmark output directories remain ignored.

Verification completed: 700,000 fresh validation simulations; exact replay of
the original three 10,000-run weekly pilots (all saved statistics agree except
runtime); deterministic replay of the Unreliable N=3 tail; 14 Markov solves and
the N=7 FULL build; source parsing and sample-planning/rounding-boundary checks.

<!-- GENERATED: finite horizon and skew audit -->

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

| Profile | Nodes | Nominal t half-width | Bounded 99% radius | Tail observations (<99% weekly availability) |
|---|---:|---:|---:|---:|
| Standard | 3 | 1.122e-07 | 1.400e-04 | 0 / 100,000 |
| Standard | 5 | 4.677e-08 | 1.399e-04 | 0 / 100,000 |
| Standard | 7 | 4.694e-08 | 1.399e-04 | 0 / 100,000 |
| Unreliable | 3 | 2.462e-05 | 1.729e-04 | 1 / 100,000 |
| Unreliable | 5 | 4.772e-08 | 1.399e-04 | 0 / 100,000 |
| Unreliable | 7 | 4.823e-08 | 1.399e-04 | 0 / 100,000 |
| Spot | 7 | 2.290e-05 | 1.706e-04 | 5 / 100,000 |

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

| Profile | Nodes | MC weekly mean | Markov weekly SIMPLIFIED | Markov weekly NO_ORPHANS | NO_ORPHANS weekly − steady |
|---|---:|---:|---:|---:|---:|
| Standard | 3 | 0.999997910 | 0.999997981 | 0.999997979 | 4.083e-11 |
| Standard | 5 | 0.999998001 | 0.999998018 | 0.999998018 | 1.640e-11 |
| Standard | 7 | 0.999998008 | 0.999998018 | 0.999998018 | 1.638e-11 |
| Unreliable | 3 | 0.999988135 | 0.999997870 | 0.999997868 | 4.549e-11 |
| Unreliable | 5 | 0.999997922 | 0.999997912 | 0.999997912 | 1.727e-11 |
| Unreliable | 7 | 0.999997893 | 0.999997912 | 0.999997912 | 1.726e-11 |
| Spot | 3 | 0.959883510 | 0.999885782 | 0.999885688 | 4.198e-08 |
| Spot | 5 | 0.998499191 | 0.999939434 | 0.999939432 | 1.209e-09 |
| Spot | 7 | 0.999922662 | 0.999940193 | 0.999940193 | 5.064e-10 |

MC values use the previous 100,000-run validation except Spot N=3/5, which use
the recorded 5,000-run pilots. Their uncertainty remains as documented above.
For Standard, the largest weekly-versus-stationary shift among computed quality
levels is only **4.224e-11**. Matching the horizon therefore does not
resolve the much larger Markov/MC differences seen for Spot. The models still
differ in safe-mode recovery, timeout, and synchronization semantics. Inclusion
inside a wide nominal MC interval is not proof of model equivalence.

### One-core runtime (Standard)

| Nodes | Quality | States | One-week build + solve | Steady-state build + solve |
|---:|---|---:|---:|---:|
| 3 | SIMPLIFIED | 16 | 2.92 ms | 3.09 ms |
| 3 | NO_ORPHANS | 76 | 1.29 ms | 1.26 ms |
| 3 | FULL | 598 | 52.22 ms | 13.75 ms |
| 5 | SIMPLIFIED | 36 | 0.89 ms | 0.96 ms |
| 5 | NO_ORPHANS | 377 | 16.61 ms | 5.89 ms |
| 5 | FULL | 8,463 | 147.79 s | 13.12 s |
| 7 | SIMPLIFIED | 64 | 1.28 ms | 1.33 ms |
| 7 | NO_ORPHANS | 1,253 | 456.27 ms | 32.70 ms |
| 7 | FULL | 68,952 | Not computed: dense memory limit | Not rerun |

The weekly figures use the **study's dense augmented matrix exponential**.
They include model construction and matrix assembly/normalization, exclude
interpreter startup, and do not include the independent validation checks.
Small-model solve times are medians of three repeats; construction and
steady-state solves are single measurements. The expensive N=5 FULL case was
measured once; the first model build can include initialization overhead.
These timings are not timings of the unchanged production
`time_averaged_distribution` implementation.

The existing sparse augmented-exponential routine was also measured for all
nine SIMPLIFIED cases: about **2.08–2.34 seconds per weekly solve**, compared
with milliseconds including build for the study's dense SIMPLIFIED method.
It agrees with the dense average distributions to
**5.11e-15** maximum absolute state-probability difference.
Independent stiff BDF integration checked SIMPLIFIED and NO_ORPHANS for all
profiles/sizes; the largest availability difference was **0.00e+00**.
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

There are **37 computed scenario/quality combinations**. The routine
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
