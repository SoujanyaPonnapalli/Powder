# One-day availability: three Spot replicas

The FULL Markov model predicts **99.988594%**, meeting **3 nines (99.900000%)**. The independent adaptive Monte Carlo run **disproved** this threshold for the simulator at overall 99% confidence across its scheduled checks.

| Method | Mean one-day availability | Runs / states | Nominal simulator 99% CI | Bounded decision interval | Runtime |
|---|---:|---:|---|---|---:|
| FULL Markov | 99.988594% | 598 states | — | Numerical check, not statistical CI | 0.0567 s |
| Fixed MC baseline | 98.770661% | 100,000 runs | 98.697858%–98.843464% | 98.658844%–98.882479% | 22.472 s |
| Adaptive MC | 98.751425% | 256,000 runs | 98.705702%–98.797147% | 98.660553%–98.842296% | 57.007 s |

Baseline MC minus Markov is **-12,179.3 ppm**. The fixed baseline's nominal half-width is **±728.0 ppm**; the adaptive run's is **±457.2 ppm**.

The threshold was first resolved at **4,000 adaptive trials (0.877 s)**. The run continued to meet the additional native precision target.


The adaptive decision uses a bounded interval corrected for repeated checks. FULL retains modeling approximations; the comparison tests agreement with the simulator. See REPORT.md for assumptions and the full stopping rule.
