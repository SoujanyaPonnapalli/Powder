# One-day availability: three Spot replicas

The FULL Markov model predicts **99.988594%**, meeting **3 nines (99.900000%)**. The independent adaptive Monte Carlo run **disproved** this threshold for the simulator at overall 99% confidence across its scheduled checks.

| Method | Mean one-day availability | Runs / states | Nominal simulator 99% CI | Bounded decision interval | Runtime |
|---|---:|---:|---|---|---:|
| FULL Markov | 99.988594% | 598 states | — | Numerical check, not statistical CI | 0.0567 s |
| Fixed MC baseline | 98.770661% | 100,000 runs | 98.697858%–98.843464% | 98.658844%–98.882479% | 22.472 s |
| Adaptive MC | 98.751425% | 256,000 runs | 98.705702%–98.797147% | 98.660553%–98.842296% | 57.007 s |

Baseline MC minus Markov is **-12,179.3 ppm**. The fixed baseline's nominal half-width is **±728.0 ppm**; the adaptive run's is **±457.2 ppm**.

The threshold was first resolved at **4,000 adaptive trials (0.877 s)**. The run continued to meet the additional native precision target.

## Matching the experiment

Both methods start with three healthy Spot replicas and a leader and estimate the expected fraction of the complete first 24 hours during which the cluster can commit. This is neither availability at the end of the day nor the probability of a completely outage-free day. Transient failures are exponential with mean 30 days, recovery mean is 20 minutes, permanent per-machine data-loss mean is 1 day, the replacement timeout is 5 minutes, spawning averages 60 seconds, and elections average 5 seconds. The simulator uses the existing safe replacement strategy.

Each MC trial accounts for all 86,400 seconds, including time after the simulator first reports actual data loss. This uses the previously validated continuation adapter, avoiding the earlier stopped-duration bias. Baseline and adaptive runs use separate, predetermined seed ranges. Neither run reuses the other’s samples.

## Stopping rule and confidence

The Markov result selects k=3 and threshold 1−10⁻ᵏ=99.900000% before MC begins. The native simulator convergence calculation is configured for 99% Student-t confidence and absolute half-width ≤500 ppm (half the scale of k nines).

The adaptive run checks after 1,000, 2,000, 4,000, …, 1,024,000 trials. It stops only when **both** the native half-width target is met and a bounded interval lies entirely above or below the nines threshold. A narrow interval that still straddles the threshold is inconclusive.

For check j, the two-sided empirical Bernstein interval spends δⱼ=0.01/[j(j+1)]. Since the sum over all checks is at most 0.01, a union bound provides at least 99% simultaneous coverage at the scheduled checks for independent, identically distributed trial availabilities in [0,1]. This protects the threshold decision against repeated inspection and skewed samples. The fixed 100,000-run baseline uses an ordinary fixed-sample 99% empirical Bernstein interval. These guarantees are per experiment, not a joint 99% guarantee for both experiments.

The native simulator Student-t interval is also reported, but its nominal coverage depends on the approximation for the sample mean; it alone is not the basis of the sequential threshold claim. The bound concerns the simulator’s mean, not the correctness of its physical assumptions.

| Adaptive checkpoint | Mean | Native 99% half-width (ppm) | Bounded interval | Threshold decision | Native precision met? |
|---:|---:|---:|---|---|---|
| 1,000 | 98.583865% | 7,946.7 | 95.896680%–100.000000% | unresolved | False |
| 2,000 | 98.630302% | 5,288.4 | 96.912542%–100.000000% | unresolved | False |
| 4,000 | 98.825893% | 3,564.8 | 97.761762%–99.890024% | disproved | False |
| 8,000 | 98.801264% | 2,531.0 | 98.122625%–99.479902% | disproved | False |
| 16,000 | 98.802638% | 1,776.2 | 98.366822%–99.238455% | disproved | False |
| 32,000 | 98.802376% | 1,258.6 | 98.515904%–99.088848% | disproved | False |
| 64,000 | 98.739969% | 919.9 | 98.543611%–98.936327% | disproved | False |
| 128,000 | 98.768817% | 639.6 | 98.637568%–98.900065% | disproved | False |
| 256,000 | 98.751425% | 457.2 | 98.660553%–98.842296% | disproved | True |

## Why the models can disagree

The fixed baseline observed **855 actual-data-loss trials**, **2,583 physical-quorum-loss trials**, and **2,473 trials below 99% daily availability**. The independent adaptive sample observed 2,365 actual-data-loss trials. These events can contribute substantial downtime despite short routine election interruptions.

FULL retains the most states among the existing Markov quality levels, but still approximates the simulator’s synchronization, deterministic timeout, standby promotion, and safe replacement behavior. It permits replacement recovery in all-data-loss states and does not preserve the full history of latest-copy ownership. Consequently, FULL is not an exact Markov encoding of this simulator. This experiment establishes the mean-availability discrepancy; it does not isolate the contribution of each modeling approximation.

## Numerical validation and reproduction

The Markov calculation uses a dense augmented matrix exponential for the time integral. An independent sparse BDF integration differs in availability by 0; its additional verification runtime is 0.6289 s. The table reports Markov build plus dense solve, excluding BDF verification. MC runtimes report trajectory generation; confidence calculations and file writing are excluded. Everything runs through Python, with one-thread NumPy/SciPy settings; no Rust implementation is used.

Run `.venv/bin/python -m notebooks.availability_one_day_spot`. `results.json` contains the complete checkpoint history and source hashes. `baseline_samples.npz` and `adaptive_samples.npz` preserve per-trial availability and first data/quorum loss times; NaN means no such event during the day.
