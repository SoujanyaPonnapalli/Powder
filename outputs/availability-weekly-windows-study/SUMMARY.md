# Weekly-window availability — simplified report

## What changed

Each machine now has its own randomly positioned **one-day weekly window** containing **70% of its transient-failure intensity**. Failures remain possible during the other six days. The average rate stays at **one per 30 days**; it does not become one failure per week. A machine keeps its window after recovery; a new replacement draws a new window. Windows can overlap naturally, but there is no common fleet-wide trigger.

The clock audit measured **70.16%** in-window events and a **30.01-day** average interval.

## Results

Completed **910,000 full-week simulations** in **15.87 min** of measured simulation time on one worker. Both columns below use complete weeks.

The final Spot seven-node comparison uses 100,000 runs per distribution. Other controls use 10,000 runs; other windowed samples use 100,000 except Spot three/five-node samples, which use 10,000.

| Profile | Nodes | Windowed availability | Exponential control |
| --- | --- | --- | --- |
| Standard | 3 | 99.999773% | 99.999742% |
| Standard | 5 | 99.999804% | 99.999799% |
| Standard | 7 | 99.999805% | 99.999803% |
| Unreliable | 3 | 99.999775% | 99.999783% |
| Unreliable | 5 | 99.999789% | 99.999789% |
| Unreliable | 7 | 99.999789% | 99.999790% |
| Spot | 3 | 91.643182% | 91.696842% |
| Spot | 5 | 99.805155% | 99.797443% |
| Spot | 7 | 99.989328% | 99.990498% |

All nine nominal 99% difference intervals include zero. These samples do not establish a change in mean availability from the weekly concentration. This is not an equivalence result, especially for rare failures.

The initial 10,000-run Spot seven-node control missed the severe tail entirely. An independent 100,000-run control found 8 severe weeks, compared with ten in the windowed sample. This is a concrete example of why a small sample can give misleadingly narrow error bars.

Standard and Unreliable five- and seven-node clusters stay close to **99.9998%** in these samples. Spot remains sensitive to severe outages. These means have sampling uncertainty; the full report includes intervals and tail counts.

## An important correction to the previous report

Some previous simulations stopped at data loss and divided availability by that shortened duration. The new study continues to the end of the week. For windowed Spot with three nodes, this changes the mean from **96.029%** to **91.643%** on the same paths. This is an accounting correction, not evidence that weekly concentration caused that entire drop. Fresh exponential controls receive the same correction.

## What to do about the skew

Changing failure timing does not eliminate rare disastrous weeks. Keep the mean, show the worst weeks and their frequency, and validate the system's recovery behavior after quorum loss. Then implement and validate a weighted rare-event sampler so those paths can be estimated efficiently. Do not remove the bad weeks from the average.

None of the new windowed samples certifies **±0.000005 at 99% confidence** using the conservative bounded interval. Narrow ordinary error bars are useful diagnostics, but they cannot prove that an important rare path was sampled. The rare-event sampler remains future work.

## One-week Markov calculations

The one-week Markov results are **mean-rate exponential proxies**. They cannot see the weekly windows or each machine's phase, so their values remain the same as before. They are useful for quick screening, but are not exact predictions for this alternate distribution or the simulator's disaster recovery behavior.

Standard profile, build plus solve, one numerical thread:

| Nodes | SIMPLIFIED one week | NO_ORPHANS one week | FULL one week |
| --- | --- | --- | --- |
| 3 | 6.41 ms | 1.36 ms | 53.83 ms |
| 5 | 0.98 ms | 17.39 ms | 147.79 s (previous measurement) |
| 7 | 1.50 ms | 464.65 ms | Not computed |

The five-node FULL time is the prior measurement of the identical proxy; the other shown computed times are new. A calendar-aware model would require additional state and time-dependent rates.

[Full report](REPORT.md) · [Simulation evidence](mc.json) · [Independent Spot control](tail_control.json) · [Markov evidence](markov.json) · [Clock audit](distribution_audit.json)
