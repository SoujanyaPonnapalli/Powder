# Weekly-window availability study — full report

## Main findings

This rerun preserves the old **30-day average transient-failure intensity**, puts **70% of that intensity in a one-day weekly window**, and assigns **each VM an independent weekly phase**. Failures remain possible throughout the week. The one-day width is a study assumption, not an inferred property of the deployment.

Completed **910,000 full-week simulations**: 720,000 with weekly windows and 190,000 fresh exponential controls. Measured simulation time was **15.87 min** on one worker, with numerical libraries limited to one thread. The final comparison uses an independent 100,000-run Spot N=7 control; the original 10,000-run Spot N=7 control remains in the evidence but is not pooled into that comparison.

All nine nominal 99% difference intervals include zero. These samples do not establish a change in mean availability from the weekly concentration. This is not an equivalence result, especially for rare failures.

The study also found a separate accounting problem: the original simulator can return at data loss before the requested horizon. Availability divided by that shortened duration can overstate full-week availability. Both new distributions now run through the entire week. For windowed Spot N=3, the stopped-duration mean is **96.028961%**, versus **91.643182%** over the full week. The distribution comparison below uses the corrected horizon on both sides.

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

An always-operational 200,000-event clock produced **70.1595%** of events in the high window, with a mean interevent interval of **30.01209 days**. This is a generator calibration check, separate from the 910,000 cluster simulations. A separate sample of 10,000 seven-VM phase vectors checks the independent phase draws; the histograms and correlation matrix are in [distribution_audit.json](distribution_audit.json).

One sampled seven-VM cluster has these one-day window starts; windows wrap across the week boundary as needed:

| Machine | Weekly window begins (days after week starts) |
| --- | --- |
| 1 | 1.664 |
| 2 | 4.706 |
| 3 | 5.621 |
| 4 | 0.122 |
| 5 | 4.423 |
| 6 | 6.262 |
| 7 | 2.115 |

Realized shares after recovery and replacement effects:

| Profile | Nodes | Applied transient events | Observed in-window share |
| --- | --- | --- | --- |
| Standard | 3 | 70651 | 70.081% |
| Standard | 5 | 116540 | 69.926% |
| Standard | 7 | 163711 | 70.091% |
| Unreliable | 3 | 70284 | 69.972% |
| Unreliable | 5 | 116994 | 69.820% |
| Unreliable | 7 | 162321 | 69.789% |
| Spot | 3 | 6420 | 69.143% |
| Spot | 5 | 11299 | 68.634% |
| Spot | 7 | 159891 | 69.546% |

## Controlled distribution comparison

Every control uses the original exponential 30-day transient clock and the same full-week accounting. Final control samples have 10,000 runs per configuration except Spot N=7, which has 100,000; the windowed samples have 100,000 except Spot N=3/5, which have 10,000 each. Seeds are independent across modes; the table is an independent-sample comparison, not a paired-seed experiment. Availability values are fractions; one ppm is 0.000001.

The original Spot N=7 control observed zero severe weeks in 10,000 runs, against ten in 100,000 windowed runs. Its nominal difference interval excluded zero, but the unequal samples poorly resolved this rare tail. We therefore ran a separate, fixed 100,000-run control with independent seeds, observing **8 weeks below 99% availability**. The follow-up size was fixed before collecting it, with no adaptive stopping. The table uses this new control; the original pilot is preserved in `mc.json`, and the follow-up is in `tail_control.json`.

| Profile | Nodes | Windowed runs | Windowed mean | Exponential mean | Difference (ppm) | Nominal 99% difference CI (ppm) |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | 3 | 100,000 | 0.999997727 | 0.999997417 | +0.310 | [-0.873, +1.493] |
| Standard | 5 | 100,000 | 0.999998039 | 0.999997987 | +0.052 | [-0.106, +0.209] |
| Standard | 7 | 100,000 | 0.999998049 | 0.999998029 | +0.019 | [-0.134, +0.172] |
| Unreliable | 3 | 100,000 | 0.999997751 | 0.999997828 | -0.077 | [-0.359, +0.204] |
| Unreliable | 5 | 100,000 | 0.999997888 | 0.999997893 | -0.005 | [-0.163, +0.153] |
| Unreliable | 7 | 100,000 | 0.999997888 | 0.999997900 | -0.012 | [-0.171, +0.147] |
| Spot | 3 | 10,000 | 0.916431817 | 0.916968419 | -536.601 | [-8620.805, +7547.603] |
| Spot | 5 | 10,000 | 0.998051545 | 0.997974425 | +77.120 | [-1241.168, +1395.408] |
| Spot | 7 | 100,000 | 0.999893284 | 0.999904981 | -11.697 | [-71.728, +48.335] |

Difference intervals use a nominal 99% Welch calculation from independent sample variances. They are per comparison, not simultaneous. Rare catastrophic weeks can make these intervals unreliable if the sample misses the tail. A zero-containing interval does not establish that the two models are interchangeable. Wider/narrower windows and correlated phases would be different experiments.

## Skew, tail events, and convergence

Weekly-window results:

| Profile | Nodes | Mean | Nominal 99% half-width | Median | Worst week | Weeks below 99% | Passing 1,000-run batches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | 3 | 0.999997727 | 2.34e-07 | 1.000000000 | 0.995156786 | 0 | 90/100 |
| Standard | 5 | 0.999998039 | 4.6e-08 | 1.000000000 | 0.999920321 | 0 | 100/100 |
| Standard | 7 | 0.999998049 | 4.58e-08 | 1.000000000 | 0.999913935 | 0 | 100/100 |
| Unreliable | 3 | 0.999997751 | 1.37e-07 | 1.000000000 | 0.997017453 | 0 | 92/100 |
| Unreliable | 5 | 0.999997888 | 4.82e-08 | 1.000000000 | 0.999915778 | 0 | 100/100 |
| Unreliable | 7 | 0.999997888 | 4.85e-08 | 1.000000000 | 0.999912044 | 0 | 100/100 |
| Spot | 3 | 0.916431817 | 0.00572 | 0.999937364 | 0.000784927 | 1596 | 0/10 |
| Spot | 5 | 0.998051545 | 0.000904 | 0.999944598 | 0.061273132 | 37 | 0/10 |
| Spot | 7 | 0.999893284 | 4.54e-05 | 0.999944471 | 0.130419952 | 10 | 0/100 |

A passing batch has its nominal 99% Student-t interval entirely inside `[0.999990, 0.999999]`. This target-band test differs from estimating the mean to ±0.000005. If the true mean lies outside the band, increasing the run count cannot make the system meet that target. The smallest stored weekly values and their reproducible seeds are in [mc.json](mc.json).

Changing the input failure distribution does not remove the output's long lower tail. Quorum loss, safe-mode promotion rules, and prolonged recovery still create rare, expensive weeks. The new Unreliable N=3 sample happened to contain no weeks below 99%; the previous study did find a catastrophic path. That absence in a fresh sample is not evidence that the recovery risk has been fixed. Keep the arithmetic mean for expected availability and accompany it with tail frequency, severity, and uncertainty; medians alone hide downtime. Do not trim or winsorize away the failures to make a convergence plot look better.

For windowed Spot N=7, the ten weeks below 99% account for **44.0% of observed downtime**. Of those ten, **3 already ran to the full horizon without an early data-loss return**. The severe tail therefore cannot be explained solely by the denominator correction.

| Profile | Nodes | Conservative 99% CI | ±0.000005 certified? | Measured MC time | Pilot projected runs for nominal ±0.000005 | Projected Python time |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | 3 | [0.999857610, 1.000000000] | No | 10.37 s | 219 | 22.70 ms |
| Standard | 5 | [0.999858174, 1.000000000] | No | 15.90 s | 9 | 1.43 ms |
| Standard | 7 | [0.999858185, 1.000000000] | No | 22.36 s | 9 | 2.01 ms |
| Unreliable | 3 | [0.999857765, 1.000000000] | No | 11.43 s | 76 | 8.69 ms |
| Unreliable | 5 | [0.999858021, 1.000000000] | No | 16.66 s | 10 | 1.67 ms |
| Unreliable | 7 | [0.999858020, 1.000000000] | No | 23.06 s | 10 | 2.31 ms |
| Spot | 3 | [0.907343359, 0.925520276] | No | 12.61 s | 13,098,419,028 | 191.15 days |
| Spot | 5 | [0.995438369, 1.000000000] | No | 24.91 s | 326,966,766 | 9.43 days |
| Spot | 7 | [0.999692436, 1.000000000] | No | 6.30 min | 8,253,759 | 8.66 h |

The conservative intervals use the two-sided empirical Bernstein bound for independent observations in [0,1], at fixed sample sizes. No windowed configuration certifies the requested ±0.000005 precision under this bound. The normal-form sample projections use the pilot variance, can be much too optimistic if important paths were missed, and are **not convergence guarantees or recommended stopping rules**. Tiny projections for apparently stable configurations are particularly uninformative about unseen tails. Runtime projections assume the current measured cost per simulation, not a measured Rust speedup. [Bound method](https://arxiv.org/abs/0907.3740).

Recommended next work: verify the intended recovery behavior after quorum/data loss, then add and independently validate importance sampling or stratification for dangerous failure paths. Any importance sampler for this periodic model must include the calendar-dependent integrated hazard and phase sampling in its likelihood weights. An unweighted oversample of bad weeks would bias availability. This rerun implements the alternate failure model and bounded diagnostics; it does not implement that rare-event sampler. [Importance sampling reference](https://artowen.su.domains/mc/Ch-var-is.pdf).

## Correcting the one-week denominator

`Simulator.run_until(end_time=...)` returns immediately when actual data loss is detected. The earlier Monte Carlo path did not suppress that return even with `stop_on_data_loss=False`. Dividing accumulated available time by the early stopping time estimates a different quantity from a complete-week average.

The study adapter resumes the same simulator and event queue until seven days, retaining all existing protocol/strategy behavior. If there are no remaining events, it accounts for the final state through the horizon. It asserts that every sample contains exactly seven days of elapsed metrics. It does not assume recovery after disaster or introduce a new recovery policy. The production `Simulator` API itself is unchanged. A deterministic test with data loss at second 1 of a 100-second horizon produces legacy availability 1.0 and corrected availability 0.01.

| Distribution | Profile | Nodes | Early returns / runs | Old stopped-duration mean | Full-week mean |
| --- | --- | --- | --- | --- | --- |
| weekly_windows | Standard | 3 | 0/100000 | 0.999997727 | 0.999997727 |
| weekly_windows | Standard | 5 | 0/100000 | 0.999998039 | 0.999998039 |
| weekly_windows | Standard | 7 | 0/100000 | 0.999998049 | 0.999998049 |
| weekly_windows | Spot | 3 | 1399/10000 | 0.960289611 | 0.916431817 |
| weekly_windows | Spot | 5 | 29/10000 | 0.998869040 | 0.998051545 |
| weekly_windows | Spot | 7 | 7/100000 | 0.999906011 | 0.999893284 |
| weekly_windows | Unreliable | 3 | 0/100000 | 0.999997751 | 0.999997751 |
| weekly_windows | Unreliable | 5 | 0/100000 | 0.999997888 | 0.999997888 |
| weekly_windows | Unreliable | 7 | 0/100000 | 0.999997888 | 0.999997888 |
| exponential_30d | Standard | 3 | 0/10000 | 0.999997417 | 0.999997417 |
| exponential_30d | Standard | 5 | 0/10000 | 0.999997987 | 0.999997987 |
| exponential_30d | Standard | 7 | 0/10000 | 0.999998029 | 0.999998029 |
| exponential_30d | Spot | 3 | 1420/10000 | 0.959806349 | 0.916968419 |
| exponential_30d | Spot | 5 | 28/10000 | 0.998566429 | 0.997974425 |
| exponential_30d | Spot | 7 | 0/10000 | 0.999940299 | 0.999940299 |
| exponential_30d | Unreliable | 3 | 0/10000 | 0.999997828 | 0.999997828 |
| exponential_30d | Unreliable | 5 | 0/10000 | 0.999997893 | 0.999997893 |
| exponential_30d | Unreliable | 7 | 0/10000 | 0.999997900 | 0.999997900 |
| exponential_30d | Spot | 7 | 6/100000 | 0.999912548 | 0.999904981 |

The two availability columns in each row come from the same paths, before and after continuation; this isolates the accounting correction. The final row is the independent 100,000-run Spot N=7 control; the earlier 10,000-run row is the initial pilot. Comparing an old stopped-duration mean with a new full-week mean would confound the distribution change with this correction. The previous report remains archived with a correction notice.

## One-week Markov comparison and runtime

**These Markov values are exponential mean-rate proxies, not exact results for the weekly-window process.** The existing constant-generator model receives the unchanged 30-day-rate configuration. Its generators are identical to the exponential control by construction and are checked for exact sparse-matrix equality. This equality does not validate the periodic process approximation. An actual calendar-aware Markov calculation would need node phases, time-dependent transition rates, phase averaging, and explicit treatment of phase draws at replacement; the current reduced state counts do not retain those details.

The one-week calculation starts healthy with a leader and integrates state occupancy over seven days using the previously verified augmented dense matrix exponential. The table compares that finite-horizon proxy with full-week simulations:

| Profile | Nodes | Windowed MC | Exponential MC | One-week SIMPLIFIED proxy | One-week NO_ORPHANS proxy | NO_ORPHANS weekly − steady |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | 3 | 0.999997727 | 0.999997417 | 0.999997981 | 0.999997979 | 4.08e-11 |
| Standard | 5 | 0.999998039 | 0.999997987 | 0.999998018 | 0.999998018 | 1.64e-11 |
| Standard | 7 | 0.999998049 | 0.999998029 | 0.999998018 | 0.999998018 | 1.64e-11 |
| Unreliable | 3 | 0.999997751 | 0.999997828 | 0.999997870 | 0.999997868 | 4.55e-11 |
| Unreliable | 5 | 0.999997888 | 0.999997893 | 0.999997912 | 0.999997912 | 1.73e-11 |
| Unreliable | 7 | 0.999997888 | 0.999997900 | 0.999997912 | 0.999997912 | 1.73e-11 |
| Spot | 3 | 0.916431817 | 0.916968419 | 0.999885782 | 0.999885688 | 4.2e-08 |
| Spot | 5 | 0.998051545 | 0.997974425 | 0.999939434 | 0.999939432 | 1.21e-09 |
| Spot | 7 | 0.999893284 | 0.999904981 | 0.999940193 | 0.999940193 | 5.06e-10 |

Markov also approximates timeout, sync, and recovery-policy semantics. In particular, its repair paths do not reproduce all of the simulator's prolonged safe-mode failures. Matching the horizon and mean failure rate does not resolve those modeling differences.

Standard-profile times include model build plus weekly solve:

| Nodes | SIMPLIFIED one week | NO_ORPHANS one week | FULL one week |
| --- | --- | --- | --- |
| 3 | 6.41 ms | 1.36 ms | 53.83 ms |
| 5 | 0.98 ms | 17.39 ms | 147.79 s (previous measurement) |
| 7 | 1.50 ms | 464.65 ms | Not computed |

The new run solves 36 of 45 profile/size/quality combinations, with a 2,000-state dense cap. All 36 weekly availability values match the preceding finite-horizon results exactly at stored precision. Timings are single measurements, with one numerical thread, and include first-use overhead; they are not median benchmarks. Generator-equality audit builds are excluded from the reported production build+solve times. The unchanged Standard FULL N=5 proxy took 147.79 seconds in the preceding study; that number is explicitly historical and was not rerun here. FULL N=7 remains uncomputed (68,952 states; its augmented 68,953-by-68,953 dense matrix alone is approximately 35.4 GiB). The larger skipped models would need a different numerical method and budget.

### All newly measured Markov cases

| Profile | Nodes | Quality | States | One-week availability | Build | Weekly solve | Steady solve |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | 3 | SIMPLIFIED | 16 | 0.999997981293 | 5.32 ms | 1.08 ms | 1.32 ms |
| Standard | 3 | COLLAPSED_PIPELINE | 30 | 0.999997981247 | 0.67 ms | 0.19 ms | 0.23 ms |
| Standard | 3 | NO_ORPHANS | 76 | 0.999997978944 | 0.93 ms | 0.43 ms | 0.33 ms |
| Standard | 3 | MERGED_PIPELINE | 192 | 0.999997978918 | 2.10 ms | 3.79 ms | 0.89 ms |
| Standard | 3 | FULL | 598 | 0.999997978908 | 6.88 ms | 46.95 ms | 6.44 ms |
| Standard | 5 | SIMPLIFIED | 36 | 0.999998018145 | 0.76 ms | 0.22 ms | 0.22 ms |
| Standard | 5 | COLLAPSED_PIPELINE | 91 | 0.999998018145 | 1.14 ms | 0.51 ms | 0.39 ms |
| Standard | 5 | NO_ORPHANS | 377 | 0.999998018143 | 3.68 ms | 13.71 ms | 2.35 ms |
| Standard | 5 | MERGED_PIPELINE | 1,452 | 0.999998018143 | 15.99 ms | 720.50 ms | 65.69 ms |
| Standard | 5 | FULL | 8,463 | Skipped | 112.11 ms | — | — |
| Standard | 7 | SIMPLIFIED | 64 | 0.999998018158 | 1.14 ms | 0.36 ms | 0.33 ms |
| Standard | 7 | COLLAPSED_PIPELINE | 204 | 0.999998018158 | 2.14 ms | 2.93 ms | 0.94 ms |
| Standard | 7 | NO_ORPHANS | 1,253 | 0.999998018158 | 12.00 ms | 452.65 ms | 22.94 ms |
| Standard | 7 | MERGED_PIPELINE | 6,864 | Skipped | 82.01 ms | — | — |
| Standard | 7 | FULL | 68,952 | Skipped | 1.11 s | — | — |
| Spot | 3 | SIMPLIFIED | 16 | 0.999885782420 | 1.33 ms | 0.17 ms | 0.20 ms |
| Spot | 3 | COLLAPSED_PIPELINE | 30 | 0.999885780683 | 0.55 ms | 0.17 ms | 0.21 ms |
| Spot | 3 | NO_ORPHANS | 76 | 0.999885688210 | 0.88 ms | 0.41 ms | 0.33 ms |
| Spot | 3 | MERGED_PIPELINE | 192 | 0.999885687311 | 2.34 ms | 2.49 ms | 1.12 ms |
| Spot | 3 | FULL | 598 | 0.999885686938 | 7.24 ms | 47.61 ms | 6.27 ms |
| Spot | 5 | SIMPLIFIED | 36 | 0.999939433522 | 0.76 ms | 0.23 ms | 0.23 ms |
| Spot | 5 | COLLAPSED_PIPELINE | 91 | 0.999939433486 | 1.19 ms | 0.52 ms | 0.38 ms |
| Spot | 5 | NO_ORPHANS | 377 | 0.999939431536 | 3.42 ms | 13.09 ms | 2.47 ms |
| Spot | 5 | MERGED_PIPELINE | 1,452 | 0.999939431524 | 15.67 ms | 712.75 ms | 62.14 ms |
| Spot | 5 | FULL | 8,463 | Skipped | 111.57 ms | — | — |
| Spot | 7 | SIMPLIFIED | 64 | 0.999940193216 | 1.15 ms | 0.35 ms | 0.34 ms |
| Spot | 7 | COLLAPSED_PIPELINE | 204 | 0.999940193215 | 2.14 ms | 2.92 ms | 0.84 ms |
| Spot | 7 | NO_ORPHANS | 1,253 | 0.999940193177 | 11.23 ms | 455.35 ms | 21.71 ms |
| Spot | 7 | MERGED_PIPELINE | 6,864 | Skipped | 82.39 ms | — | — |
| Spot | 7 | FULL | 68,952 | Skipped | 1.11 s | — | — |
| Unreliable | 3 | SIMPLIFIED | 16 | 0.999997870348 | 1.36 ms | 0.16 ms | 0.20 ms |
| Unreliable | 3 | COLLAPSED_PIPELINE | 30 | 0.999997870300 | 0.55 ms | 0.16 ms | 0.20 ms |
| Unreliable | 3 | NO_ORPHANS | 76 | 0.999997867840 | 0.90 ms | 0.44 ms | 0.35 ms |
| Unreliable | 3 | MERGED_PIPELINE | 192 | 0.999997867812 | 2.22 ms | 2.44 ms | 0.87 ms |
| Unreliable | 3 | FULL | 598 | 0.999997867802 | 6.84 ms | 44.56 ms | 6.48 ms |
| Unreliable | 5 | SIMPLIFIED | 36 | 0.999997912444 | 0.75 ms | 0.21 ms | 0.22 ms |
| Unreliable | 5 | COLLAPSED_PIPELINE | 91 | 0.999997912444 | 1.10 ms | 0.51 ms | 0.40 ms |
| Unreliable | 5 | NO_ORPHANS | 377 | 0.999997912442 | 3.52 ms | 13.16 ms | 2.37 ms |
| Unreliable | 5 | MERGED_PIPELINE | 1,452 | 0.999997912442 | 15.78 ms | 709.61 ms | 63.60 ms |
| Unreliable | 5 | FULL | 8,463 | Skipped | 109.27 ms | — | — |
| Unreliable | 7 | SIMPLIFIED | 64 | 0.999997912460 | 1.12 ms | 0.36 ms | 0.33 ms |
| Unreliable | 7 | COLLAPSED_PIPELINE | 204 | 0.999997912460 | 2.16 ms | 3.06 ms | 0.90 ms |
| Unreliable | 7 | NO_ORPHANS | 1,253 | 0.999997912460 | 11.78 ms | 452.45 ms | 22.30 ms |
| Unreliable | 7 | MERGED_PIPELINE | 6,864 | Skipped | 82.58 ms | — | — |
| Unreliable | 7 | FULL | 68,952 | Skipped | 1.14 s | — | — |

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

Recorded source revision before this run: `ae48bafda440ca25846f49e1f6e0b15cb865cd95`. The source SHA-256 in each evidence file identifies the new uncommitted implementation used for that run; the subsequent repository commit records it with these outputs.
