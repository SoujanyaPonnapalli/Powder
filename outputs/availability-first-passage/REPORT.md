# Time to quorum loss and data loss

All clusters start healthy, contain only the named profile, and use the exponential transient baseline: 30-day mean transient interval, 20-minute mean recovery, 5-minute replacement timeout, 60-second mean spawn, and the existing synchronization approximation. Per-machine permanent-loss mean is 1 day for Spot and 365 days for Unreliable. A year is 365 days. No new MC trajectories were run.

## Mean time to first quorum unavailability

The main metric is first loss of a majority of **available, up-to-date** replicas. A leader election alone does not count. Physical quorum instead counts available lagging replicas too, matching the simulator’s `has_potential_data_loss` predicate. These are first-passage times from an initially healthy cluster, not outage durations or reciprocals of unavailability.

| Profile | Nodes | Required quorum | MTT up-to-date quorum unavailable | MTT physical quorum unavailable |
|---|---:|---:|---:|---:|
| Spot | 3 | 2 | 38.37 days | 38.37 days |
| Spot | 5 | 3 | 4.94 years | 4.94 years |
| Spot | 7 | 4 | 249 years | 249 years |
| Unreliable | 3 | 2 | 104 years | 104 years |
| Unreliable | 5 | 3 | 170,000 years | 170,000 years |
| Unreliable | 7 | 4 | 299 million years | 299 million years |

These are NO_ORPHANS Markov approximations, not validated simulator predictions. The existing model approximates synchronization, timeout shape, and safe replacement. Very large Unreliable values especially depend on independent faults and exclude common-cause events.

For each configuration, removing the leader label was verified to preserve all count-transition rates. The reduced generator solves `-Q_T t = 1`, with zero time at the target. The reported quorum solves use dense mpmath arithmetic at 50 decimal digits and agree with independent 80-digit solves to 13 relative digits. Diagonals are rebuilt by summing positive transition rates at high precision; ordinary floating-point diagonals can overwhelm the rare hitting rate. Input rates still have their original floating-point precision. These solves differ from the earlier sparse steady-state and dense finite-horizon calculations.

## MTTDL: what the saved data support

**A trustworthy MTTDL is not available from the existing availability Markov model.** It does not track which failed or lagging copies retain the latest committed data, and it permits recovery after all data are lost. Making only the all-permanently-failed state absorbing would therefore measure a different event.

The simulator records actual loss when no active member retains the latest committed data, including temporarily unavailable members in that check. Its predicate checks active members, not standby copies. The saved exponential runs give the following evidence. The extrapolated mean assumes a constant *cluster first-data-loss hazard* at every future time. Exponential per-machine faults do **not** imply this assumption; a cluster has repair pipelines and starts healthy.

| Profile | Nodes | Data-loss weeks / simulated weeks | Extrapolated MTTDL, constant cluster hazard | Conditional 99% interval |
|---|---:|---:|---:|---:|
| Spot | 3 | 1,420 / 10,000 | 45.71 days | 42.70 days – 49.00 days |
| Spot | 5 | 28 / 10,000 | 6.84 years | 4.28 years – 11.8 years |
| Spot | 7 | 6 / 100,000 | 320 years | 122 years – 1,250 years |
| Unreliable | 3 | 0 / 10,000 | Not estimated (zero events) | > 41.6 years, one-sided; same hazard assumption |
| Unreliable | 5 | 0 / 10,000 | Not estimated (zero events) | > 41.6 years, one-sided; same hazard assumption |
| Unreliable | 7 | 0 / 10,000 | Not estimated (zero events) | > 41.6 years, one-sided; same hazard assumption |

The point extrapolation is `MTTDL = -7 days / log(1 - losses/runs)`. The two-sided intervals transform exact 99% binomial intervals for seven-day loss probability. Zero-event lower limits use `runs × 7 days / log(100)` and are one-sided 99% limits under that same assumption. These intervals quantify sampling uncertainty, not extrapolation/model error. Without a survival-tail assumption, one-week observations do not determine the unrestricted mean, so no numerical Unreliable MTTDL is justified here.

Seven-node Spot uses the independent 100,000-run control: six actual-data-loss stops, distinct from its eight severe-availability weeks. The earlier zero-loss 10,000-run control is not used.

For defensible MTTDL across all six configurations, the next model must track latest-copy ownership and preserve the simulator’s safe replacement behavior, then solve an absorbing first-passage problem. Merely extending a weekly availability average or averaging loss times only among observed losses is insufficient.

## Reproduction

Run `.venv/bin/python -m notebooks.availability_first_passage` from the repository root. `results.json` preserves calculations, residuals, numerical verification, runtimes, and hashes of the saved input samples.
