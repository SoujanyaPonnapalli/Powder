> **Full-week accounting correction:** Some historical Monte Carlo runs below stopped early at data loss, so their availability averages do not cover a complete week. The [new full-week study](../availability-weekly-windows-study/REPORT.md) compares fresh exponential controls with independent weekly failure windows and reports the effect of this correction separately. Historical results below are retained for provenance.

# Availability study — simplified report

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
**4.22e-11**. Finite-horizon calculation fixes the observation window;
it does not fix differences in recovery-policy assumptions.

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

MC means have sampling uncertainty; Spot N=3/5 use older 5,000-run pilots,
while the other rows use 100,000 runs. Markov's optimistic Spot results remain
a modeling issue after matching the horizon.

## What does the one-week calculation cost?

Measured on one core, Standard profile, including build and solve:

| Nodes | SIMPLIFIED weekly | NO_ORPHANS weekly | FULL weekly |
|---:|---:|---:|---:|
| 3 | 2.92 ms | 1.29 ms | 52.22 ms |
| 5 | 0.89 ms | 16.61 ms | 147.79 s |
| 7 | 1.28 ms | 456.27 ms | Not computed |

These use a verified dense matrix-exponential method in the study code.
The first build includes initialization overhead; that affects the 3-node row.
Steady-state SIMPLIFIED is still roughly 1 ms; five-node FULL is roughly 13 s
at steady state. Seven-node FULL is too large for this dense method on the host.
Agreement between Markov quality levels does not establish simulator accuracy.
Rust speedups remain hypothetical: no Rust port was measured.

[Full report and reproducibility](REPORT.md) · [All finite-horizon results](finite_horizon.md)
