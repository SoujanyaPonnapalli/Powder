# How often is a 99% Monte Carlo interval wrong?

1000 random heterogeneous Raft RSMs with one-day SIMPLIFIED Markov availability ≥ 99.9%. Each ran naive Monte Carlo once to half-width `(1 − A_markov) / 2` at 99% confidence; **CLT** is the normal interval at the first check it met the tolerance, **engine_ci** the engine's existing `converged` check (Student t; Wald with the rule of three for data loss), **Bernstein** the empirical-Bernstein interval at its stop. All three read the same run sequence. Truth is an independent run with 5× the runs: *missed* means the interval is disjoint from the reference's own 99% interval, *ambiguous* that it overlaps it only partly. A correct 99% interval misses about 1% of the time.

## Overall

| Metric | Rule | RSMs | Missed | Miss rate (99% Wilson) | Ambiguous | Missed after a zero-variance stop | Median runs |
|---|---|---:|---:|---|---:|---:|---:|
| availability | engine_ci | 1000 | 244 | 24.4% (21.1–28.1%) | 121 | 28 | 77 |
| availability | clt | 1000 | 254 | 25.4% (22.0–29.1%) | 115 | 28 | 77 |
| availability | bernstein | 1000 | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 696,964 |
| data_loss_probability | engine_ci | 1000 | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| data_loss_probability | clt | 1000 | 74 | 7.4% (5.5–9.8%) | 925 | 74 | 30 |
| data_loss_probability | bernstein | 1000 | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 576,002 |

## By cluster size

| Cluster | Rule | RSMs | Missed | Miss rate (99% Wilson) | Ambiguous | Missed after a zero-variance stop | Median runs |
|---|---|---:|---:|---|---:|---:|---:|
| N=3 / availability | engine_ci | 361 | 173 | 47.9% (41.2–54.7%) | 56 | 10 | 94 |
| N=3 / availability | clt | 361 | 174 | 48.2% (41.5–54.9%) | 56 | 10 | 85 |
| N=3 / availability | bernstein | 361 | 0 | 0.0% (0.0–1.8%) | 4 | 0 | 576,002 |
| N=3 / data_loss_probability | engine_ci | 361 | 0 | 0.0% (0.0–1.8%) | 4 | 0 | 77,831 |
| N=3 / data_loss_probability | clt | 361 | 74 | 20.5% (15.6–26.5%) | 286 | 74 | 30 |
| N=3 / data_loss_probability | bernstein | 361 | 0 | 0.0% (0.0–1.8%) | 1 | 0 | 393,416 |
| N=5 / availability | engine_ci | 314 | 39 | 12.4% (8.4–18.0%) | 34 | 8 | 70 |
| N=5 / availability | clt | 314 | 43 | 13.7% (9.4–19.4%) | 29 | 8 | 63 |
| N=5 / availability | bernstein | 314 | 0 | 0.0% (0.0–2.1%) | 0 | 0 | 665,284 |
| N=5 / data_loss_probability | engine_ci | 314 | 0 | 0.0% (0.0–2.1%) | 0 | 0 | 137,887 |
| N=5 / data_loss_probability | clt | 314 | 0 | 0.0% (0.0–2.1%) | 314 | 0 | 30 |
| N=5 / data_loss_probability | bernstein | 314 | 0 | 0.0% (0.0–2.1%) | 0 | 0 | 633,603 |
| N=7 / availability | engine_ci | 325 | 32 | 9.8% (6.4–14.9%) | 31 | 10 | 85 |
| N=7 / availability | clt | 325 | 37 | 11.4% (7.6–16.7%) | 30 | 10 | 77 |
| N=7 / availability | bernstein | 325 | 0 | 0.0% (0.0–2.0%) | 0 | 0 | 766,661 |
| N=7 / data_loss_probability | engine_ci | 325 | 0 | 0.0% (0.0–2.0%) | 0 | 0 | 166,844 |
| N=7 / data_loss_probability | clt | 325 | 0 | 0.0% (0.0–2.0%) | 325 | 0 | 30 |
| N=7 / data_loss_probability | bernstein | 325 | 0 | 0.0% (0.0–2.0%) | 0 | 0 | 766,661 |

## By Markov availability

| Band | Rule | RSMs | Missed | Miss rate (99% Wilson) | Ambiguous | Missed after a zero-variance stop | Median runs |
|---|---|---:|---:|---|---:|---:|---:|
| A ≥ 99.90000% / availability | engine_ci | 100 | 22 | 22.0% (13.3–34.2%) | 11 | 0 | 57 |
| A ≥ 99.90000% / availability | clt | 100 | 23 | 23.0% (14.0–35.3%) | 12 | 0 | 51 |
| A ≥ 99.90000% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 1 | 0 | 53,158 |
| A ≥ 99.90000% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 3 | 0 | 9,556 |
| A ≥ 99.90000% / data_loss_probability | clt | 100 | 26 | 26.0% (16.5–38.5%) | 73 | 26 | 30 |
| A ≥ 99.90000% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 1 | 0 | 48,325 |
| A ≥ 99.94377% / availability | engine_ci | 100 | 18 | 18.0% (10.2–29.8%) | 14 | 0 | 51 |
| A ≥ 99.94377% / availability | clt | 100 | 18 | 18.0% (10.2–29.8%) | 14 | 0 | 51 |
| A ≥ 99.94377% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 1 | 0 | 94,177 |
| A ≥ 99.94377% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 16,934 |
| A ≥ 99.94377% / data_loss_probability | clt | 100 | 12 | 12.0% (5.9–22.8%) | 88 | 12 | 30 |
| A ≥ 99.94377% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 77,831 |
| A ≥ 99.96838% / availability | engine_ci | 100 | 15 | 15.0% (8.0–26.3%) | 12 | 0 | 70 |
| A ≥ 99.96838% / availability | clt | 100 | 16 | 16.0% (8.7–27.5%) | 13 | 0 | 57 |
| A ≥ 99.96838% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 137,887 |
| A ≥ 99.96838% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 1 | 0 | 24,796 |
| A ≥ 99.96838% / data_loss_probability | clt | 100 | 10 | 10.0% (4.6–20.4%) | 90 | 10 | 30 |
| A ≥ 99.96838% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 125,351 |
| A ≥ 99.98222% / availability | engine_ci | 100 | 21 | 21.0% (12.5–33.1%) | 11 | 0 | 77 |
| A ≥ 99.98222% / availability | clt | 100 | 24 | 24.0% (14.8–36.4%) | 8 | 0 | 74 |
| A ≥ 99.98222% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 268,707 |
| A ≥ 99.98222% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 48,325 |
| A ≥ 99.98222% / data_loss_probability | clt | 100 | 11 | 11.0% (5.3–21.6%) | 89 | 11 | 30 |
| A ≥ 99.98222% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 244,279 |
| A ≥ 99.99000% / availability | engine_ci | 100 | 27 | 27.0% (17.3–39.6%) | 16 | 1 | 85 |
| A ≥ 99.99000% / availability | clt | 100 | 29 | 29.0% (18.9–41.7%) | 14 | 1 | 81 |
| A ≥ 99.99000% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 1 | 0 | 476,034 |
| A ≥ 99.99000% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 85,615 |
| A ≥ 99.99000% / data_loss_probability | clt | 100 | 8 | 8.0% (3.4–17.9%) | 92 | 8 | 30 |
| A ≥ 99.99000% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 432,758 |
| A ≥ 99.99438% / availability | engine_ci | 100 | 27 | 27.0% (17.3–39.6%) | 13 | 1 | 77 |
| A ≥ 99.99438% / availability | clt | 100 | 27 | 27.0% (17.3–39.6%) | 12 | 1 | 70 |
| A ≥ 99.99438% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 766,661 |
| A ≥ 99.99438% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 151,676 |
| A ≥ 99.99438% / data_loss_probability | clt | 100 | 6 | 6.0% (2.2–15.3%) | 94 | 6 | 30 |
| A ≥ 99.99438% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 696,964 |
| A ≥ 99.99684% / availability | engine_ci | 100 | 33 | 33.0% (22.3–45.8%) | 9 | 4 | 77 |
| A ≥ 99.99684% / availability | clt | 100 | 34 | 34.0% (23.1–46.9%) | 9 | 4 | 74 |
| A ≥ 99.99684% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 1 | 0 | 1,358,191 |
| A ≥ 99.99684% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 268,707 |
| A ≥ 99.99684% / data_loss_probability | clt | 100 | 1 | 1.0% (0.1–8.0%) | 99 | 1 | 30 |
| A ≥ 99.99684% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 1,234,719 |
| A ≥ 99.99822% / availability | engine_ci | 100 | 30 | 30.0% (19.7–42.7%) | 12 | 7 | 85 |
| A ≥ 99.99822% / availability | clt | 100 | 31 | 31.0% (20.6–43.8%) | 9 | 7 | 85 |
| A ≥ 99.99822% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 2,187,385 |
| A ≥ 99.99822% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 476,034 |
| A ≥ 99.99822% / data_loss_probability | clt | 100 | 0 | 0.0% (0.0–6.2%) | 100 | 0 | 30 |
| A ≥ 99.99822% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 2,187,385 |
| A ≥ 99.99900% / availability | engine_ci | 100 | 23 | 23.0% (14.0–35.3%) | 9 | 8 | 140 |
| A ≥ 99.99900% / availability | clt | 100 | 23 | 23.0% (14.0–35.3%) | 10 | 8 | 140 |
| A ≥ 99.99900% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 4,262,599 |
| A ≥ 99.99900% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 843,328 |
| A ≥ 99.99900% / data_loss_probability | clt | 100 | 0 | 0.0% (0.0–6.2%) | 100 | 0 | 30 |
| A ≥ 99.99900% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 3,875,090 |
| A ≥ 99.99944% / availability | engine_ci | 100 | 28 | 28.0% (18.1–40.7%) | 14 | 7 | 228 |
| A ≥ 99.99944% / availability | clt | 100 | 29 | 29.0% (18.9–41.7%) | 14 | 7 | 218 |
| A ≥ 99.99944% / availability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 7,551,458 |
| A ≥ 99.99944% / data_loss_probability | engine_ci | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 1,494,011 |
| A ≥ 99.99944% / data_loss_probability | clt | 100 | 0 | 0.0% (0.0–6.2%) | 100 | 0 | 30 |
| A ≥ 99.99944% / data_loss_probability | bernstein | 100 | 0 | 0.0% (0.0–6.2%) | 0 | 0 | 6,864,961 |

## Availability: naive versus stratified sampling

970 RSMs with both results (30 RSMs needed more than 2,000,000 strata and are left out). Naive uses the engine's `converged` check (Student t); stratified uses leader strata, proportional allocation, at least 30 runs per sampled stratum, and the stratified normal interval `z sqrt(V) + bias`. Both stop at the first check at or past the minimum whose interval meets the tolerance. *Definitely wrong*: disjoint from the reference's normal interval. *Point outside*: the reference estimate is outside the interval. A correct 99% interval is wrong about 1% of the time.

| Minimum runs | Naive: definitely wrong | Naive: point outside | Naive: median runs | Stratified: definitely wrong | Stratified: point outside | Stratified: median runs |
|---|---:|---:|---:|---:|---:|---:|
| 30 | 29.0% | 30.1% | 85 | 1.4% | 2.3% | 317,698 |
| 100 | 19.9% | 20.8% | 140 | 1.4% | 2.3% | 317,698 |
| 300 | 15.6% | 16.7% | 305 | 1.4% | 2.3% | 317,698 |
| 1,000 | 10.6% | 12.2% | 1,063 | 1.4% | 2.3% | 317,698 |
| 3,000 | 7.5% | 9.8% | 3,041 | 1.4% | 2.3% | 317,698 |
| 10,000 | 5.4% | 7.8% | 10,512 | 1.1% | 2.0% | 317,698 |
| 30,000 | 2.5% | 5.3% | 30,004 | 0.9% | 1.6% | 317,698 |
| 1 / tolerance | 3.7% | 6.7% | 58,474 | 0.7% | 1.6% | 449,938 | (0 / 25 unresolved)
| 3 / tolerance | 0.7% | 3.4% | 151,676 | 0.7% | 1.4% | 722,250 | (0 / 73 unresolved)
| 4.6 / tolerance | 0.6% | 3.5% | 222,071 | 0.7% | 1.6% | 848,670 | (0 / 117 unresolved)
| 10 / tolerance | 0.5% | 4.0% | 432,758 | 0.3% | 1.1% | 1,094,445 | (0 / 229 unresolved)
| **Bernstein** | 0.0% | 0.0% | 696,964 | 0.0% | 0.5% | 1,472,866 |

## Delaying the first check (judged against the reference's empirical-Bernstein interval)

The CLT and engine rules replayed with a larger minimum run count before the first check, on the same run sequences. `c / tolerance` scales the minimum with the tolerance, the way the rule of three does for data loss. *Unresolved* counts RSMs whose trace ended before the rule stopped.

| Minimum runs | Metric | Rule | Missed | Miss rate (99% Wilson) | Ambiguous | Unresolved | Median runs at stop |
|---|---|---|---:|---|---:|---:|---:|
| 30 | availability | engine_ci | 244 | 24.4% (21.1–28.1%) | 121 | 0 | 77 |
| 30 | availability | clt | 254 | 25.4% (22.0–29.1%) | 115 | 0 | 77 |
| 30 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 30 | data_loss_probability | clt | 74 | 7.4% (5.5–9.8%) | 925 | 0 | 30 |
| 100 | availability | engine_ci | 153 | 15.3% (12.6–18.5%) | 126 | 0 | 140 |
| 100 | availability | clt | 156 | 15.6% (12.9–18.8%) | 129 | 0 | 127 |
| 100 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 100 | data_loss_probability | clt | 73 | 7.3% (5.5–9.7%) | 925 | 0 | 104 |
| 300 | availability | engine_ci | 114 | 11.4% (9.1–14.2%) | 175 | 0 | 305 |
| 300 | availability | clt | 113 | 11.3% (9.0–14.1%) | 178 | 0 | 305 |
| 300 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 300 | data_loss_probability | clt | 68 | 6.8% (5.0–9.1%) | 925 | 0 | 305 |
| 1,000 | availability | engine_ci | 73 | 7.3% (5.5–9.7%) | 337 | 0 | 1,063 |
| 1,000 | availability | clt | 73 | 7.3% (5.5–9.7%) | 338 | 0 | 1,063 |
| 1,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 1,000 | data_loss_probability | clt | 59 | 5.9% (4.3–8.1%) | 923 | 0 | 1,063 |
| 3,000 | availability | engine_ci | 40 | 4.0% (2.7–5.9%) | 533 | 0 | 3,041 |
| 3,000 | availability | clt | 40 | 4.0% (2.7–5.9%) | 533 | 0 | 3,041 |
| 3,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 3,000 | data_loss_probability | clt | 41 | 4.1% (2.8–6.0%) | 924 | 0 | 3,041 |
| 10,000 | availability | engine_ci | 27 | 2.7% (1.7–4.4%) | 701 | 0 | 10,512 |
| 10,000 | availability | clt | 27 | 2.7% (1.7–4.4%) | 701 | 0 | 10,512 |
| 10,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 10,000 | data_loss_probability | clt | 25 | 2.5% (1.5–4.1%) | 918 | 0 | 10,512 |
| 30,000 | availability | engine_ci | 12 | 1.2% (0.6–2.5%) | 770 | 0 | 30,004 |
| 30,000 | availability | clt | 12 | 1.2% (0.6–2.5%) | 770 | 0 | 30,004 |
| 30,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 11 | 0 | 125,351 |
| 30,000 | data_loss_probability | clt | 12 | 1.2% (0.6–2.5%) | 909 | 0 | 30,004 |
| 1 / tolerance | availability | engine_ci | 10 | 1.0% (0.5–2.2%) | 802 | 0 | 53,158 |
| 1 / tolerance | availability | clt | 10 | 1.0% (0.5–2.2%) | 802 | 0 | 53,158 |
| 1 / tolerance | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 1 / tolerance | data_loss_probability | clt | 27 | 2.7% (1.7–4.4%) | 920 | 0 | 43,931 |
| 3 / tolerance | availability | engine_ci | 2 | 0.2% (0.0–1.0%) | 791 | 0 | 137,887 |
| 3 / tolerance | availability | clt | 2 | 0.2% (0.0–1.0%) | 791 | 0 | 137,887 |
| 3 / tolerance | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 4 | 0 | 125,351 |
| 3 / tolerance | data_loss_probability | clt | 8 | 0.8% (0.3–1.9%) | 902 | 0 | 125,351 |
| 4.6 / tolerance | availability | engine_ci | 0 | 0.0% (0.0–0.7%) | 808 | 0 | 201,882 |
| 4.6 / tolerance | availability | clt | 0 | 0.0% (0.0–0.7%) | 808 | 0 | 201,882 |
| 4.6 / tolerance | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 6 | 0 | 183,529 |
| 4.6 / tolerance | data_loss_probability | clt | 7 | 0.7% (0.3–1.8%) | 896 | 0 | 183,529 |
| 10 / tolerance | availability | engine_ci | 0 | 0.0% (0.0–0.7%) | 856 | 0 | 393,416 |
| 10 / tolerance | availability | clt | 0 | 0.0% (0.0–0.7%) | 856 | 0 | 393,416 |
| 10 / tolerance | data_loss_probability | engine_ci | 1 | 0.1% (0.0–0.8%) | 28 | 0 | 393,416 |
| 10 / tolerance | data_loss_probability | clt | 2 | 0.2% (0.0–1.0%) | 899 | 0 | 393,416 |

## Delaying the first check (judged against the reference's normal (CLT) interval)

The CLT and engine rules replayed with a larger minimum run count before the first check, on the same run sequences. `c / tolerance` scales the minimum with the tolerance, the way the rule of three does for data loss. *Unresolved* counts RSMs whose trace ended before the rule stopped.

| Minimum runs | Metric | Rule | Missed | Miss rate (99% Wilson) | Ambiguous | Unresolved | Median runs at stop |
|---|---|---|---:|---|---:|---:|---:|
| 30 | availability | engine_ci | 285 | 28.5% (25.0–32.3%) | 24 | 0 | 77 |
| 30 | availability | clt | 289 | 28.9% (25.4–32.7%) | 26 | 0 | 77 |
| 30 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 30 | data_loss_probability | clt | 126 | 12.6% (10.1–15.6%) | 83 | 0 | 30 |
| 100 | availability | engine_ci | 193 | 19.3% (16.3–22.7%) | 26 | 0 | 140 |
| 100 | availability | clt | 194 | 19.4% (16.4–22.8%) | 26 | 0 | 127 |
| 100 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 100 | data_loss_probability | clt | 125 | 12.5% (10.1–15.4%) | 83 | 0 | 104 |
| 300 | availability | engine_ci | 151 | 15.1% (12.4–18.2%) | 29 | 0 | 305 |
| 300 | availability | clt | 149 | 14.9% (12.2–18.0%) | 31 | 0 | 305 |
| 300 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 300 | data_loss_probability | clt | 120 | 12.0% (9.6–14.9%) | 83 | 0 | 305 |
| 1,000 | availability | engine_ci | 103 | 10.3% (8.1–13.0%) | 34 | 0 | 1,063 |
| 1,000 | availability | clt | 102 | 10.2% (8.0–12.9%) | 35 | 0 | 1,063 |
| 1,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 1,000 | data_loss_probability | clt | 110 | 11.0% (8.7–13.8%) | 83 | 0 | 1,063 |
| 3,000 | availability | engine_ci | 74 | 7.4% (5.5–9.8%) | 50 | 0 | 3,041 |
| 3,000 | availability | clt | 74 | 7.4% (5.5–9.8%) | 50 | 0 | 3,041 |
| 3,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 3,000 | data_loss_probability | clt | 92 | 9.2% (7.1–11.8%) | 83 | 0 | 3,041 |
| 10,000 | availability | engine_ci | 52 | 5.2% (3.7–7.3%) | 63 | 0 | 10,512 |
| 10,000 | availability | clt | 52 | 5.2% (3.7–7.3%) | 63 | 0 | 10,512 |
| 10,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 10,000 | data_loss_probability | clt | 72 | 7.2% (5.4–9.6%) | 81 | 0 | 10,512 |
| 30,000 | availability | engine_ci | 24 | 2.4% (1.4–4.0%) | 79 | 0 | 30,004 |
| 30,000 | availability | clt | 24 | 2.4% (1.4–4.0%) | 79 | 0 | 30,004 |
| 30,000 | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 5 | 0 | 125,351 |
| 30,000 | data_loss_probability | clt | 53 | 5.3% (3.8–7.4%) | 80 | 0 | 30,004 |
| 1 / tolerance | availability | engine_ci | 36 | 3.6% (2.4–5.4%) | 80 | 0 | 53,158 |
| 1 / tolerance | availability | clt | 36 | 3.6% (2.4–5.4%) | 80 | 0 | 53,158 |
| 1 / tolerance | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 1 / tolerance | data_loss_probability | clt | 77 | 7.7% (5.8–10.2%) | 81 | 0 | 43,931 |
| 3 / tolerance | availability | engine_ci | 7 | 0.7% (0.3–1.8%) | 101 | 0 | 137,887 |
| 3 / tolerance | availability | clt | 7 | 0.7% (0.3–1.8%) | 101 | 0 | 137,887 |
| 3 / tolerance | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 125,351 |
| 3 / tolerance | data_loss_probability | clt | 46 | 4.6% (3.2–6.6%) | 75 | 0 | 125,351 |
| 4.6 / tolerance | availability | engine_ci | 7 | 0.7% (0.3–1.8%) | 100 | 0 | 201,882 |
| 4.6 / tolerance | availability | clt | 7 | 0.7% (0.3–1.8%) | 100 | 0 | 201,882 |
| 4.6 / tolerance | data_loss_probability | engine_ci | 0 | 0.0% (0.0–0.7%) | 1 | 0 | 183,529 |
| 4.6 / tolerance | data_loss_probability | clt | 43 | 4.3% (2.9–6.3%) | 71 | 0 | 183,529 |
| 10 / tolerance | availability | engine_ci | 5 | 0.5% (0.2–1.5%) | 169 | 0 | 393,416 |
| 10 / tolerance | availability | clt | 5 | 0.5% (0.2–1.5%) | 169 | 0 | 393,416 |
| 10 / tolerance | data_loss_probability | engine_ci | 2 | 0.2% (0.0–1.0%) | 3 | 0 | 393,416 |
| 10 / tolerance | data_loss_probability | clt | 32 | 3.2% (2.0–5.0%) | 72 | 0 | 393,416 |
