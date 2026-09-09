# Heterogeneous one-week versus steady-state Markov benchmarks

**Measured heterogeneous Markov costs.** New solver benchmarks mix three machine profiles within each RSM. The rates remain constant and exponential, so these models are heterogeneous across machines but still homogeneous over time. All nodes start healthy and the initial leader is Standard. The mixes are:

| Nodes | Standard | Unreliable | Spot |
|---|---|---|---|
| 3 | 1 | 1 | 1 |
| 5 | 2 | 2 | 1 |
| 7 | 3 | 2 | 2 |

| Nodes | Model | States | One-week build + solve | Steady-state build + solve |
|---|---|---|---|---|
| 3 | SIMPLIFIED | 54 | 1.14 ms | 1.18 ms |
| 3 | NO_ORPHANS | 323 | 12.91 ms | 5.95 ms |
| 3 | FULL | 3,024 | 7.04 s | 583.66 ms |
| 5 | SIMPLIFIED | 252 | 7.22 ms | 4.33 ms |
| 5 | NO_ORPHANS | 4,598 | 24.64 s | 2.45 s |
| 5 | FULL | 158,652 | Not attempted: state cap | Stopped: memory guard |
| 7 | SIMPLIFIED | 936 | 161.32 ms | 27.83 ms |
| 7 | NO_ORPHANS | 48,068 | Stopped: memory guard | Stopped: memory guard |
| 7 | FULL | Not enumerated | Not built | Not built |

The seven-node FULL model was not built: its combinatorial state upper bound is **5,682,456**, above the 200,000-state preflight build budget. This is an upper bound, not an enumerated state count. The five-node FULL model was built and contains 158,652 states; its one-week solve was not attempted under the 60,000-state finite-solve cap.

All timings use Python/SciPy and one numerical thread. The main benchmark uses a **1.5 GiB resident-memory guard** and a **120-second wall budget per worker**, including build and validation. The five-node NO_ORPHANS dense retry uses a **3 GiB guard**. “Memory guard” means the run was stopped by this study's budget, not that the model is impossible to solve on a larger host. The guard is sampled, so recorded peak memory may overshoot its trigger.

Completed one-week calculations use the same dense augmented matrix exponential as the homogeneous table. Five-node NO_ORPHANS initially exceeded 1.5 GiB; sparse BDF integration then reached its 120-second budget. Its dense retry with the 3 GiB guard had status **complete** and is used in the table if complete. Seven-node NO_ORPHANS exceeded the 1.5 GiB guard for both sparse BDF integration and the stationary solver. Finite and stationary build times are independently measured; startup, small-model warmup, and validation are excluded from reported totals. Solves up to 2,000 states use the median of three repetitions; larger solves use one measurement. Historical homogeneous times are context, not a controlled interleaved speed comparison.

The state growth explains why homogeneous runtimes do not transfer directly. Five-node NO_ORPHANS grows from **377 to 4,598 states**; seven-node NO_ORPHANS grows from **1,253 to 48,068 states**. Three-node FULL grows from **598 to 3,024 states**, with a newly measured one-week total of **7.04 s** versus the earlier homogeneous 53.83 ms. The finite-horizon method and resource budget matter alongside state count.

These measurements do not include heterogeneous Monte Carlo or non-exponential clocks. Markov's existing recovery-policy approximations still apply. [Machine-readable measurements](results.json).

## Availability estimates

| Nodes | Model | One-week availability | Steady-state availability |
|---|---|---|---|
| 3 | SIMPLIFIED | 99.999620511% | 99.999601173% |
| 3 | NO_ORPHANS | 99.999617281% | 99.999598004% |
| 3 | FULL | 99.999617240% | 99.999597964% |
| 5 | SIMPLIFIED | 99.999760198% | 99.999748178% |
| 5 | NO_ORPHANS | 99.999760173% | 99.999748170% |
| 5 | FULL | — | — |
| 7 | SIMPLIFIED | 99.999738211% | 99.999720944% |
| 7 | NO_ORPHANS | — | — |
| 7 | FULL | — | — |

These are model predictions, not validated real-system availabilities. The healthy start and Standard initial leader are relevant to the one-week values. Each replacement retains its slot's fixed rate class; aging and software calendars are not represented. The three profiles retain their original 30-day mean transient interval, 20-minute mean recovery, and 3-year / 1-year / 1-day mean permanent-loss intervals for Standard / Unreliable / Spot. Other protocol and replacement settings match the earlier study.

## Numerical checks and reproduction

For 5 completed dense cases, independent sparse BDF integration agrees within **1.11e-15** absolute availability. BDF uses rtol=1e-10 and atol=1e-14. Probability-mass, generator, and stationary residual diagnostics are stored in the JSON evidence. The five-node NO_ORPHANS retry retains the dense solver's probability diagnostics; its independent BDF attempt timed out, so that particular result does not have the additional cross-method check. All **88 targeted numerical and heterogeneous-state regression tests passed** for this change.

```sh
.venv/bin/python -m notebooks.availability_heterogeneous_markov
.venv/bin/python -m notebooks.availability_heterogeneous_bdf
.venv/bin/python -m notebooks.availability_heterogeneous_dense_retry
.venv/bin/python -m notebooks.availability_heterogeneous_markov_report
.venv/bin/python -m pytest tests/test_availability_study_methods.py tests/test_markov_hetero_state_counts.py -q
```

- [All primary attempts, including resource stops](results.json)
- [Five-node sparse-BDF fallback](n5-NO_ORPHANS-finite-bdf.json)
- [Five-node dense retry with 3 GiB guard](n5-NO_ORPHANS-finite-dense-3gib.json)
- [Benchmark implementation](../../notebooks/availability_heterogeneous_markov.py)
- [Fallback implementation](../../notebooks/availability_heterogeneous_bdf.py)
- [Dense retry implementation](../../notebooks/availability_heterogeneous_dense_retry.py)
- [Report renderer](../../notebooks/availability_heterogeneous_markov_report.py)

Source hashes, dependency versions, numerical thread settings, mixture definitions, actual state counts where built, solve samples, and observed memory peaks are preserved. The 120-second budget includes verification, so a timeout is not automatically a lower bound on solve time alone.
