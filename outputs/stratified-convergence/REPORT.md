# Stratified versus naive Monte Carlo: runs to converge

Heterogeneous Raft RSMs with replacement (`safe_mode` off). Tolerance is `(1 − A_markov) / 2` at 99% confidence for availability and data-loss probability alike.

## Markov filter

| N | Mix (Std/Unrel/Spot) | Horizon | Model | States | Availability | Kept | Tolerance |
|---:|---|---|---|---:|---:|---|---:|
| 3 | 1/1/1 | 1d | NO_ORPHANS | 323 | 99.999664% | yes | 1.68e-06 |
| 3 | 1/1/1 | 30d | NO_ORPHANS | 323 | 99.999605% | yes | 1.98e-06 |
| 3 | 1/1/1 | 365d | NO_ORPHANS | 323 | 99.999599% | yes | 2.01e-06 |
| 5 | 2/2/1 | 1d | NO_ORPHANS | 4,598 | 99.999784% | yes | 1.08e-06 |
| 5 | 2/2/1 | 30d | NO_ORPHANS | 4,598 | 99.999753% | yes | 1.23e-06 |
| 5 | 2/2/1 | 365d | NO_ORPHANS | 4,598 | 99.999749% | yes | 1.26e-06 |
| 7 | 3/2/2 | 1d | SIMPLIFIED | 936 | 99.999777% | yes | 1.12e-06 |
| 7 | 3/2/2 | 30d | SIMPLIFIED | 936 | 99.999727% | yes | 1.36e-06 |
| 7 | 3/2/2 | 365d | SIMPLIFIED | 936 | 99.999722% | yes | 1.39e-06 |

## Availability: empirical-Bernstein stopping (rigorous)

Median runs over disjoint seeds until the Bernstein half-width reaches the tolerance.

| N | Horizon | Naive | Proportional | vs naive | Bernstein-optimal | vs naive | Estimate (naive) | Strata | Exact mass | All converged |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 1d | 10,050,992 | 6,865,068 | 1.46× | 7,551,558 | 1.33× | 1.000e+00 | 186 | 0.289 | NO |
| 5 | 1d | 13,377,873 | 10,900,413 | 1.23× | 10,900,413 | 1.23× | 1.000e+00 | 269 | 0.252 | NO |
| 7 | 1d | 13,377,873 | 13,075,806 | 1.02× | 13,075,806 | 1.02× | 1.000e+00 | 430 | 0.081 | NO |
| 3 | 30d | 14,715,661 | 16,382,867 | 0.90× | 13,502,941 | 1.09× | 1.000e+00 | 16,690 | 0.000 | yes |
| 5 | 30d | 12,161,702 | 13,034,593 | 0.93× | 13,034,593 | 0.93× | 1.000e+00 | 31,277 | 0.000 | yes |

## Availability: known-variance run counts

`z² V / tolerance²` with per-stratum variances pooled over every stratified run of the RSM and horizon. No range term: this is the classic variance-reduction comparison, and it is only as good as the rare strata are sampled.

| N | Horizon | Naive | Naive (own samples) | Proportional | vs naive | Neyman | vs naive | Strata with observed variance |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1d | 106,128 | 78,169 | 83,330 | 1.27× | 2,862 | 37.08× | 169 / 185 |
| 5 | 1d | 1,308 | 1,315 | 1,033 | 1.27× | 418 | 3.13× | 236 / 268 |
| 7 | 1d | 1,221 | 1,223 | 1,200 | 1.02× | 503 | 2.43× | 388 / 429 |
| 3 | 30d | 1,862,340 | 2,095,108 | 2,063,149 | 0.90× | 202,991 | 9.17× | 16,511 / 16,690 |
| 5 | 30d | 34 | 34 | 36 | 0.95× | 35 | 0.97× | 31,107 / 31,277 |

## Data loss probability: empirical-Bernstein stopping (rigorous)

Median runs over disjoint seeds until the Bernstein half-width reaches the tolerance.

| N | Horizon | Naive | Proportional | vs naive | Bernstein-optimal | vs naive | Estimate (naive) | Strata | Exact mass | All converged |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 1d | 9,137,265 | 6,240,987 | 1.46× | 6,240,987 | 1.46× | 0.000e+00 | 186 | 0.289 | yes |
| 5 | 1d | 13,377,873 | 10,900,413 | 1.23× | 10,900,413 | 1.23× | 0.000e+00 | 269 | 0.252 | yes |
| 7 | 1d | 13,377,873 | 13,075,806 | 1.02× | 13,075,806 | 1.02× | 0.000e+00 | 430 | 0.081 | yes |
| 3 | 30d | 7,551,458 | 7,649,778 | 0.99× | 7,649,778 | 0.99× | 0.000e+00 | 16,690 | 0.000 | yes |
| 5 | 30d | — | — | — | 13,034,593 | — | 0.000e+00 | 0 | 0.000 | yes |

## Data loss probability: known-variance run counts

`z² V / tolerance²` with per-stratum variances pooled over every stratified run of the RSM and horizon. No range term: this is the classic variance-reduction comparison, and it is only as good as the rare strata are sampled.

| N | Horizon | Naive | Naive (own samples) | Proportional | vs naive | Neyman | vs naive | Strata with observed variance |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1d | no event observed in 136,003,312 stratified runs | | | | | | 0 |
| 5 | 1d | no event observed in 218,008,260 stratified runs | | | | | | 0 |
| 7 | 1d | no event observed in 261,516,120 stratified runs | | | | | | 0 |
| 3 | 30d | no event observed in 224,063,717 stratified runs | | | | | | 0 |
| 5 | 30d | no event observed in 195,518,895 stratified runs | | | | | | 0 |

## Availability: every arm, Bernstein stopping

Median runs over seeds; ratio is naive / arm. `leader_*` arms stratify on the initial leader separately and treat leader-survives, at-most-f-follower-events strata as exact (each verified by 20 identical reruns).

| N | Horizon | Arm | Runs | vs naive | Seconds | Strata | Exact mass | Converged |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 3 | 1d | naive | 10,050,992 | 1.00× | 10.3 | 1 | 0.000 | yes |
| 3 | 1d | proportional | 6,865,068 | 1.46× | 8.7 | 186 | 0.289 | yes |
| 3 | 1d | bernstein | 7,551,558 | 1.33× | 11.7 | 186 | 0.289 | yes |
| 3 | 1d | leader_proportional | 3,815,726 | 2.63× | 6.1 | 581 | 0.638 | yes |
| 3 | 1d | leader_neyman | 50,118,612 | 0.20× | 101.3 | 581 | 0.638 | NO |
| 5 | 1d | naive | 13,377,873 | 1.00× | 19.7 | 1 | 0.000 | yes |
| 5 | 1d | proportional | 10,900,413 | 1.23× | 19.7 | 269 | 0.252 | yes |
| 5 | 1d | bernstein | 10,900,413 | 1.23× | 19.8 | 269 | 0.252 | yes |
| 5 | 1d | leader_proportional | 2,626,002 | 5.09× | 5.9 | 865 | 0.818 | yes |
| 5 | 1d | leader_neyman | 50,000,784 | 0.27× | 139.1 | 865 | 0.818 | NO |
| 7 | 1d | naive | 13,377,873 | 1.00× | 33.2 | 1 | 0.000 | yes |
| 7 | 1d | proportional | 13,075,806 | 1.02× | 36.3 | 430 | 0.081 | yes |
| 7 | 1d | bernstein | 13,075,806 | 1.02× | 35.2 | 430 | 0.081 | yes |
| 7 | 1d | leader_proportional | 3,538,933 | 3.78× | 13.0 | 1,423 | 0.736 | yes |
| 7 | 1d | leader_neyman | 50,001,473 | 0.27× | 183.6 | 1,423 | 0.736 | NO |
| 3 | 30d | naive | 14,715,661 | 1.00× | 71.0 | 1 | 0.000 | yes |
| 3 | 30d | proportional | 16,382,867 | 0.90× | 94.9 | 16,690 | -0.000 | yes |
| 3 | 30d | bernstein | 13,502,941 | 1.09× | 86.3 | 16,690 | -0.000 | yes |
| 5 | 30d | naive | 12,161,702 | 1.00× | 96.6 | 1 | 0.000 | yes |
| 5 | 30d | proportional | 13,034,593 | 0.93× | 123.1 | 31,277 | -0.000 | yes |
| 5 | 30d | bernstein | 13,034,593 | 0.93× | 125.8 | 31,277 | -0.000 | yes |

## Availability: known-variance run counts by strata family

| N | Horizon | Strata | Naive | Proportional | vs naive | Neyman | vs naive |
|---:|---|---|---:|---:|---:|---:|---:|
| 3 | 1d | count | 106,128 | 83,330 | 1.27× | 2,862 | 37.08× |
| 3 | 1d | leader | 121,962 | 48,637 | 2.51× | 2,224 | 54.85× |
| 5 | 1d | count | 1,308 | 1,033 | 1.27× | 418 | 3.13× |
| 5 | 1d | leader | 1,310 | 135 | 9.72× | 28 | 46.82× |
| 7 | 1d | count | 1,221 | 1,200 | 1.02× | 503 | 2.43× |
| 7 | 1d | leader | 1,221 | 182 | 6.72× | 24 | 51.88× |
| 3 | 30d | count | 1,862,340 | 2,063,149 | 0.90× | 202,991 | 9.17× |
| 5 | 30d | count | 34 | 36 | 0.95× | 35 | 0.97× |

## Data loss probability: every arm, Bernstein stopping

Median runs over seeds; ratio is naive / arm. `leader_*` arms stratify on the initial leader separately and treat leader-survives, at-most-f-follower-events strata as exact (each verified by 20 identical reruns).

| N | Horizon | Arm | Runs | vs naive | Seconds | Strata | Exact mass | Converged |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 3 | 1d | naive | 9,137,265 | 1.00× | 9.4 | 1 | 0.000 | yes |
| 3 | 1d | proportional | 6,240,987 | 1.46× | 8.8 | 186 | 0.289 | yes |
| 3 | 1d | bernstein | 6,240,987 | 1.46× | 8.0 | 186 | 0.289 | yes |
| 3 | 1d | leader_proportional | 3,468,889 | 2.63× | 5.5 | 581 | 0.638 | yes |
| 3 | 1d | leader_neyman | 3,468,889 | 2.63× | 5.3 | 581 | 0.638 | yes |
| 5 | 1d | naive | 13,377,873 | 1.00× | 17.8 | 1 | 0.000 | yes |
| 5 | 1d | proportional | 10,900,413 | 1.23× | 19.2 | 269 | 0.252 | yes |
| 5 | 1d | bernstein | 10,900,413 | 1.23× | 20.7 | 269 | 0.252 | yes |
| 5 | 1d | leader_proportional | 2,626,002 | 5.09× | 6.0 | 865 | 0.818 | yes |
| 5 | 1d | leader_neyman | 2,626,002 | 5.09× | 6.1 | 865 | 0.818 | yes |
| 7 | 1d | naive | 13,377,873 | 1.00× | 31.4 | 1 | 0.000 | yes |
| 7 | 1d | proportional | 13,075,806 | 1.02× | 34.8 | 430 | 0.081 | yes |
| 7 | 1d | bernstein | 13,075,806 | 1.02× | 37.7 | 430 | 0.081 | yes |
| 7 | 1d | leader_proportional | 3,538,933 | 3.78× | 12.7 | 1,423 | 0.736 | yes |
| 7 | 1d | leader_neyman | 3,538,933 | 3.78× | 13.1 | 1,423 | 0.736 | yes |
| 3 | 30d | naive | 7,551,458 | 1.00× | 34.7 | 1 | 0.000 | yes |
| 3 | 30d | proportional | 7,649,778 | 0.99× | 45.5 | 16,690 | -0.000 | yes |
| 3 | 30d | bernstein | 7,649,778 | 0.99× | 44.2 | 16,690 | -0.000 | yes |
| 5 | 30d | bernstein | 13,034,593 | — | 124.0 | 31,277 | -0.000 | yes |

## Data loss probability: known-variance run counts by strata family

| N | Horizon | Strata | Naive | Proportional | vs naive | Neyman | vs naive |
|---:|---|---|---:|---:|---:|---:|---:|
| 3 | 1d | count | no event in 136,003,312 runs | | | | |
| 3 | 1d | leader | 2,107 | 844 | 2.50× | 0 | 9251.60× |
| 5 | 1d | count | no event in 218,008,260 runs | | | | |
| 5 | 1d | leader | no event in 290,339,877 runs | | | | |
| 7 | 1d | count | no event in 261,516,120 runs | | | | |
| 7 | 1d | leader | no event in 303,077,397 runs | | | | |
| 3 | 30d | count | no event in 224,063,717 runs | | | | |
| 5 | 30d | count | no event in 195,518,895 runs | | | | |
