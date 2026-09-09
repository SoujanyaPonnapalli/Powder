# Computing one-week RSM availability: accuracy versus runtime

**Recommendation:** screen configurations with a small, one-week Markov model, then evaluate finalists with detailed Monte Carlo under the intended fault curves and recovery policy. Add Markov states when they represent a behavior that could change the decision. Ordinary Monte Carlo can become prohibitively expensive for rare catastrophic outages; tight precision remains unresolved in those cases.

The homogeneous and Monte Carlo sections use prior results. The heterogeneous Markov section adds new bounded solver benchmarks; no additional Monte Carlo simulations were run.

The quantity we want is the **expected fraction of a complete week during which the RSM can commit**:

$$
A_7 = \frac{1}{T}\,\mathbb{E}\!\left[\int_0^T \mathbf{1}\{\text{RSM can commit at }t\}\,dt\right],\qquad T=604{,}800\text{ seconds}.
$$

This differs from the probability of an outage-free week or of avoiding permanent data loss. Initial ages, software schedules, and cluster state matter; existing benchmarks start healthy with a leader. An absolute error of **0.000005 (5 ppm)** means **3.024 seconds of expected downtime per week**.

Both methods inherit error from fault data and system assumptions. Markov calculations additionally approximate state and timing laws; Monte Carlo adds sampling uncertainty. A solver's small residual or a simulation's narrow confidence interval measures only part of the total error.

| Choice | What it can improve | Runtime/memory cost | What it cannot establish |
|---|---|---|---|
| More Markov states | Represent fault causes, elections, timeout/spawn/sync stages, and replacement behavior | Larger matrices; solving can grow much faster than state count | Greater accuracy is not automatic if the added states retain incorrect rates or recovery rules |
| Richer timing model | Capture hardware age, software schedules, and non-exponential recovery | Time-dependent solves, age bins, or additional distribution phases | Matching a mean failure interval does not preserve outage overlap or tail behavior |
| One-week instead of steady-state Markov | Answer the requested horizon from the specified initial state | Transient integration can cost much more than a stationary solve | Matching the horizon does not repair other model assumptions |
| More independent MC runs | Reduce sampling uncertainty for the simulated model | Approximately linear runtime; ordinary error bars scale as `1/√N` once variance is represented | More runs do not fix incorrect input curves, correlations, or simulator behavior |
| Weighted rare-event sampling | Estimate influential rare paths more efficiently | Additional implementation and validation work | No such sampler was benchmarked here; its speedup is unknown |

One-week Markov integrates availability from the initial state; steady state assumes the long-run distribution. For ordinary MC, halving sampling error generally costs four times as many runs; tenfold improvement costs about 100 times as many. These are planning relationships, not guarantees when a pilot misses rare paths.

**Measured Markov costs.** The table compares Standard-profile **build-plus-solve runtime** for one-week availability and long-term, steady-state availability. Both columns include model construction; steady-state totals add the recorded build and stationary-solve times. Long term means solving for the stationary distribution, not simulating a longer horizon. Measurements use Python/SciPy with one numerical thread. “FULL” means the most detailed implemented Markov model, not the complete physical system.

| Nodes | Markov model | States | Finite horizon: one week | Long term: steady state |
|---:|---|---:|---:|---:|
| 3 | SIMPLIFIED | 16 | 6.41 ms | 6.64 ms |
| 3 | NO_ORPHANS | 76 | 1.36 ms | 1.26 ms |
| 3 | FULL | 598 | 53.83 ms | 13.32 ms |
| 5 | SIMPLIFIED | 36 | 0.98 ms | 0.98 ms |
| 5 | NO_ORPHANS | 377 | 17.39 ms | 6.04 ms |
| 5 | FULL | 8,463 | 147.79 s¹ | 13.12 s¹ |
| 7 | SIMPLIFIED | 64 | 1.50 ms | 1.47 ms |
| 7 | NO_ORPHANS | 1,253 | 464.65 ms | 34.95 ms |
| 7 | FULL | 68,952 | Not computed | Not computed |

¹ Previous measurements of the identical five-node FULL model. Other times are individual measurements from the latest run; first-use overhead affects three-node SIMPLIFIED, and tiny timing differences should not be interpreted as stable speed rankings. Seven-node FULL needs about **35.4 GiB for one augmented dense matrix**, before workspace; other numerical methods may use less memory. Neither solve is reported for that configuration in these saved comparisons. [Timing evidence](../outputs/availability-weekly-windows-study/markov.json), [FULL five-node measurement](../outputs/availability-convergence-study/finite_horizon_full5.json).

For Standard with five nodes, SIMPLIFIED and FULL one-week availability differ by only **1.34 × 10⁻¹²**. FULL's weekly and steady-state values differ by **1.64 × 10⁻¹¹**; its stationary calculation took **13.1 seconds including build**, versus 147.79 seconds for one week. Extra detail buys little here, but agreement does not validate the models' shared assumptions.

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

These measurements do not include heterogeneous Monte Carlo or non-exponential clocks. Markov's existing recovery-policy approximations still apply. [Full heterogeneous benchmark](../outputs/availability-heterogeneous-markov/REPORT.md).

**Measured Monte Carlo costs — exponential transient failures.** All rows use the original exponential transient-failure law with a 30-day mean interval and complete-week accounting. The profile names describe permanent-loss reliability: **High = Standard** (3-year mean data-loss interval), **Medium = Unreliable** (1 year), and **Low = Spot** (1 day). Recovery and the other simulator settings remain those of the existing study.

The measured columns describe completed runs. Projection columns estimate **total runs and total runtime**, not additional work, for a nominal 99% interval. They use `N ≈ ceil((2.576 × s / ε)²)` and the measured cost per run, holding the pilot standard deviation `s` constant. They do not assert that a smaller rerun would reproduce the pilot's precision.

| Profile / RSM size | Mean availability | Nominal 99% half-width (ppm) | Measured runs / runtime | Projection for ±5 ppm: total runs / runtime | Projection for nines precision: total runs / runtime |
|---|---|---|---|---|---|
| High — Standard, 3 nodes | 99.999742% | 1.160 | 10,000 / 944.57 ms | 538 runs / 50.82 ms† | 538 runs / 50.82 ms† |
| High — Standard, 5 nodes | 99.999799% | 0.151 | 10,000 / 1.41 s | 10 runs / 1.41 ms† | 10 runs / 1.41 ms† |
| High — Standard, 7 nodes | 99.999803% | 0.146 | 10,000 / 1.97 s | 9 runs / 1.77 ms† | 9 runs / 1.77 ms† |
| Medium — Unreliable, 3 nodes | 99.999783% | 0.246 | 10,000 / 948.37 ms | 25 runs / 2.37 ms† | 25 runs / 2.37 ms† |
| Medium — Unreliable, 5 nodes | 99.999789% | 0.151 | 10,000 / 1.46 s | 10 runs / 1.46 ms† | 10 runs / 1.46 ms† |
| Medium — Unreliable, 7 nodes | 99.999790% | 0.152 | 10,000 / 2.05 s | 10 runs / 2.05 ms† | 10 runs / 2.05 ms† |
| Low — Spot, 3 nodes | 91.696842% | 5710.366 | 10,000 / 11.64 s | 13.04 billion runs / 175.70 days | 131 runs / 152.52 ms† |
| Low — Spot, 5 nodes | 99.797443% | 959.419 | 10,000 / 22.86 s | 368.05 million runs / 9.74 days | 369 runs / 843.42 ms† |
| Low — Spot, 7 nodes | 99.990498% | 39.247 | 100,000 / 5.97 min | 6.16 million runs / 6.13 h | 61,611 runs / 3.68 min |

“Nines precision” follows the requested convention: for a baseline with `k` nines, use `ε = 0.5 × 10⁻ᵏ` in availability units. Thus 99.9% means three nines and an error tolerance of **±0.0005 = ±500 ppm**, not nine seconds of downtime. We take `k = floor(−log10(1−μ̂))` from each exponential pilot and keep the same per-configuration tolerance in the non-exponential comparison:

| Exponential baseline configurations | Nines used for planning | Absolute error tolerance | Tolerance in ppm |
|---|---|---|---|
| Standard and Unreliable, all sizes | 5 | ±0.000005 | ±5 |
| Spot, 3 nodes | 1 | ±0.05 | ±50,000 |
| Spot, 5 nodes | 2 | ±0.005 | ±5,000 |
| Spot, 7 nodes | 4 | ±0.00005 | ±50 |

These are tolerances for estimating the mean at the selected scale, **not certifications that a configuration meets a particular nines threshold**. The planning classification comes from an uncertain pilot mean. In particular, the seven-node Spot windowed point estimate falls below the four-nines threshold, but the comparison retains ±50 ppm so it does not silently relax the accuracy requirement.

† Projections below 1,000 runs are raw outputs of the asymptotic variance formula, **not recommended sampling budgets**. Rare-event variance is poorly determined by a short sample; the single-digit estimates are especially unreliable. No exponential baseline sample certifies ±5 ppm under the saved conservative 99% bounded interval. All projections assume the observed variance and cost per run persist; neither is guaranteed. The seven-node Spot baseline uses the independent 100,000-run control, which observed eight severe weeks, rather than the earlier 10,000-run control that observed none. [Exponential measurements](../outputs/availability-weekly-windows-study/mc.json), [independent Spot control](../outputs/availability-weekly-windows-study/tail_control.json).

**How mean availability constrains worst-case sampling cost.** For weekly availability `A ∈ [0,1]`, `A² ≤ A`, so `Var(A) = E[A²] − μ² ≤ μ(1−μ)`. At a fixed absolute error tolerance and confidence, variance-based sample budgets scale with this variance. If unavailability is `u = 1−μ`, then the maximum variance is `u(1−u) ≈ u` near perfect availability: **10× less unavailability gives approximately a 10× smaller worst-case variance-based budget**.

| Mean availability | Maximum variance μ(1−μ) | Worst-case variance-based budget relative to 99% |
|---|---|---|
| 99% | 0.0099 | 1× |
| 99.9% | 0.000999 | 0.101× |
| 99.99% | 0.00009999 | 0.0101× |
| 99.999% | 0.0000099999 | 0.00101× |

For example, moving from 99% to 99.999% availability reduces the maximum variance by approximately **990×**. This is a comparison of upper bounds, not a promise that two systems' actual simulation counts have that ratio. The nominal formula above is a planning approximation. A conservative, distribution-free sufficient budget from Chebyshev's inequality is `N ≥ μ(1−μ)/(0.01 × ε²)` for 99% confidence, provided the variance bound is valid; it has the same dependence on mean availability. A pilot estimate of μ alone does not provide a guaranteed variance bound for the true system.

Total runtime also includes cost per run: larger clusters can require more event processing despite their better availability. The roughly 10× improvement applies at **fixed absolute precision**, such as ±5 ppm. Nines-based precision tightens the tolerance as availability improves. When ε scales with unavailability, the worst-case variance-based budget instead grows approximately as `1/u`. This is why the two projection columns answer different questions.

**Non-exponential transient failures: comparison with the Markov approximation.** The alternate model preserves the original 30-day average transient-failure intensity but places **70% of it in a one-day weekly window**, independently phased for each VM. The hazard is 0.16333/day inside the window and 0.01167/day outside, versus a constant 0.03333/day in the exponential model. Failures remain possible throughout the week. Recovering VMs retain their phase; newly provisioned VMs draw an independent phase. Windows can overlap by chance, but there is no imposed fleet-wide trigger. Permanent-loss and recovery distributions are unchanged.

The Markov column uses the **one-week NO_ORPHANS exponential mean-rate proxy**, with build-plus-solve runtime on one numerical thread. Its constant-generator equations cannot represent the weekly phases, so these are **not exact Markov results for the non-exponential process**. The MC column uses the windowed model. Its sample size is 100,000 per configuration except Spot with three/five nodes, which uses 10,000. The MC runtime projection uses exactly the same tolerance for each configuration as in the exponential table.

| Profile / RSM size | One-week Markov availability (proxy) | Windowed MC availability | MC uncertainty: nominal 99% half-width (ppm) | Projected MC total runs / runtime for nines precision | Markov build + solve runtime |
|---|---|---|---|---|---|
| High — Standard, 3 nodes | 99.999798% | 99.999773% | 0.234 | 219 runs / 22.70 ms† | 1.36 ms |
| High — Standard, 5 nodes | 99.999802% | 99.999804% | 0.046 | 9 runs / 1.43 ms† | 17.39 ms |
| High — Standard, 7 nodes | 99.999802% | 99.999805% | 0.046 | 9 runs / 2.01 ms† | 464.65 ms |
| Medium — Unreliable, 3 nodes | 99.999787% | 99.999775% | 0.137 | 76 runs / 8.69 ms† | 1.34 ms |
| Medium — Unreliable, 5 nodes | 99.999791% | 99.999789% | 0.048 | 10 runs / 1.67 ms† | 16.68 ms |
| Medium — Unreliable, 7 nodes | 99.999791% | 99.999789% | 0.048 | 10 runs / 2.31 ms† | 464.23 ms |
| Low — Spot, 3 nodes | 99.988569% | 91.643182% | 5723.509 | 131 runs / 165.18 ms† | 1.28 ms |
| Low — Spot, 5 nodes | 99.993943% | 99.805155% | 904.284 | 327 runs / 814.63 ms† | 16.50 ms |
| Low — Spot, 7 nodes | 99.994019% | 99.989328% | 45.426 | 82,538 runs / 5.20 min | 466.58 ms |

MC uncertainty above is the nominal 99% Student-t half-width in ppm. It describes sampling uncertainty only; it does not include incorrect fault inputs or recovery-policy assumptions. † has the same meaning as in the exponential table. No windowed sample certifies ±5 ppm under the conservative bounded interval. [Windowed MC, Markov results, and diagnostics](../outputs/availability-weekly-windows-study/REPORT.md).

All nine nominal 99% intervals for the **windowed-minus-exponential MC difference include zero**. Thus the saved experiments do not establish a change in mean availability from this particular weekly concentration, but they also do not prove equivalence. Seven-node Spot has ten severe windowed weeks versus eight in the exponential control; its mean difference is −11.7 ppm, with a nominal 99% interval of approximately −71.7 to +48.3 ppm. At the shared ±50 ppm nines tolerance, the pilot runtime projections are **3.68 minutes exponential** and **5.20 minutes windowed**. These estimates remain sensitive to a small number of severe outcomes.

The large three-node Spot Markov–MC discrepancy is not explained by weekly concentration: exponential MC is also only 91.696842%, whereas the Markov proxy predicts 99.988569%. Their recovery paths differ, and a more detailed exponential state model does not automatically fix that mismatch. The latest simulation evidence also corrects early data-loss stops that shortened some older “weekly” denominators. The tables here consistently use complete weeks. More general hardware aging, correlated software rollouts, and other recovery laws remain unmeasured scenarios.

**Representing fault curves correctly.** Given survival to age `a`, the next-week failure probability is `[F(a+T)−F(a)]/[1−F(a)]`; the unconditioned CDF difference answers a different question. The hazard is `h(a)=F′(a)/[1−F(a)]`. Hardware age, upgrade schedules, and elapsed recovery time are distinct clocks. Calendar-dependent rates can use a time-dependent Markov generator, approximated in time slices. When faults or replacements reset clocks, elapsed age/history must also be represented—for example through age bins or extra distribution phases. Time slicing alone does not remove that requirement. General semi-Markov models need not have closed-form solutions.

Cause-specific hazards can be added when defined consistently for the same surviving population. This does not make machines independent: shared rollouts or rack outages need explicit shared causes. Merging fault types into one recovery law introduces another approximation. **The existing studies do not measure the accuracy/runtime tradeoff for age-aware or time-sliced Markov models**, so assigning them an error percentage would be speculative.

The practical choice is therefore:

1. **Screen with one-week SIMPLIFIED; check finalists with NO_ORPHANS.** Use FULL only when its additional behavior can change the decision. Use steady state only as an explicitly checked shortcut.
2. **Evaluate finalists with complete-week MC under the intended system model.** Hardware ages, correlated software faults, and recovery rules must be represented; existing experiments cover only their stated assumptions.
3. **Choose precision around the decision.** Seek an interval on the required side of a threshold, or a resolved difference between candidates. Rejecting a clearly poor configuration does not require ±5 ppm. When uncertainty spans the decision boundary, report it as unresolved; validate a weighted rare-event method before promising affordable tight precision for Spot.
