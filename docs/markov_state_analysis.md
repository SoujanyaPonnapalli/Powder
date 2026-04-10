# Markov State Space Analysis for RSM Replacement Strategies

How many states does a continuous-time Markov chain (CTMC) need to properly
model the `NodeReplacementStrategy` from `powder/simulation/strategy.py`?

This document works from the full model to a minimal tractable
model, quantifying what each simplification costs and saves.

## System overview

An N-node replicated state machine with `NodeReplacementStrategy` (N = 5 and
N = 7 are computed throughout):

1. A node becomes unavailable (transient failure, network outage, or data loss).
2. A **failure timeout** countdown begins.
3. If the node recovers before the timeout, the countdown is cancelled.
4. If the timeout fires and the node is still unavailable, a **replacement**
  is spawned as a standby, synced, promoted, and the worst node is removed.

Special rule: if **all** nodes are unavailable (`num_available == 0`), all
pending replacement timers are cancelled.

---

## Per-node state dimensions

To fully capture the system, each node's state has three dimensions:

### 1. Availability


| Sub-state         | Meaning                                               |
| ----------------- | ----------------------------------------------------- |
| **Healthy (H)**   | Available, has data, up to date                       |
| **Failed (F)**    | Unavailable, has data (transient failure)             |
| **Lagging (L)**   | Available, has data, not yet synced to `commit_index` |
| **Data-loss (D)** | No data, needs full replacement                       |


### 2. Replacement timer


| Sub-state       | Meaning                                              |
| --------------- | ---------------------------------------------------- |
| **None**        | No timer (healthy/lagging, or all-down cancellation) |
| **Waiting (w)** | Timer running, has not fired yet                     |
| **Expired (e)** | Timer fired, replacement spawned                     |


### 3. Replacement pipeline stage


| Sub-state            | Meaning                         |
| -------------------- | ------------------------------- |
| **None**             | No replacement in flight        |
| **Provisioning (P)** | VM being created                |
| **Syncing (S)**      | Standby syncing data from donor |


A synced standby (Ready) is not a separate state — once synced, promotion is
near-instantaneous, so a ready replacement is effectively a new healthy node.

---

## Full model: 12 per-node states

Not all combinations of (availability × timer × pipeline) are valid. The
valid combinations are:


| Availability | Timer | Pipeline | Description                                       |
| ------------ | ----- | -------- | ------------------------------------------------- |
| H            | —     | None     | Normal healthy node                               |
| H            | —     | P        | Node recovered; orphaned replacement provisioning |
| H            | —     | S        | Node recovered; orphaned replacement syncing      |
| F            | w     | None     | Failed, timer running, no replacement yet         |
| F            | e     | P        | Failed, replacement provisioning                  |
| F            | e     | S        | Failed, replacement syncing                       |
| L            | —     | None     | Recovering from failure, syncing back             |
| L            | —     | P        | Recovering; orphaned replacement provisioning     |
| L            | —     | S        | Recovering; orphaned replacement syncing          |
| D            | w     | None     | Data loss, timer running                          |
| D            | e     | P        | Data loss, replacement provisioning               |
| D            | e     | S        | Data loss, replacement syncing                    |


**k = 12 valid per-node states.**

### Full model state counts


|                   | Formula             | N = 5                 | N = 7                  |
| ----------------- | ------------------- | --------------------- | ---------------------- |
| **Homogeneous**   | C(N + k − 1, k − 1) | C(16, 11) = **4,368** | C(18, 11) = **31,824** |
| **Heterogeneous** | k^N                 | 12^5 = **248,832**    | 12^7 = **35,831,808**  |


---

## Simplification 1: Merge Provisioning and Syncing

**Insight:** Provisioning (spinning up a VM) and Syncing (copying data to the
standby) are sequential stages of the same process: creating a replacement.
We can merge them into a single **Replacing (R)** pipeline state.

**What changes:**

- Pipeline dimension: `{None, P, S}` → `{None, R}`
- Every state that distinguished P from S merges into one.

**Before:** `{H, H_P, H_S, Fw, Fe_P, Fe_S, L, L_P, L_S, Dw, De_P, De_S}` → **k = 12**

**After:** `{H, H_R, Fw, Fe_R, L, L_R, Dw, De_R}` → **k = 8**

**Accuracy cost:** Loses the ability to model different durations for
provisioning vs. syncing. The combined stage uses a single rate
`μ_R = 1 / (mean_provision_time + mean_sync_time)`.


|                   | N = 5              | N = 7                | Savings vs. full (N = 5) | Savings vs. full (N = 7) |
| ----------------- | ------------------ | -------------------- | ------------------------ | ------------------------ |
| **Homogeneous**   | C(12, 7) = **792** | C(14, 7) = **3,432** | 5.5× (from 4,368)        | 9.3× (from 31,824)       |
| **Heterogeneous** | 8^5 = **32,768**   | 8^7 = **2,097,152**  | 7.6× (from 248,832)      | 17.1× (from 35,831,808)  |


---

## Simplification 2: Remove orphaned replacement tracking

**Insight:** When a failed node recovers while its replacement is still in
the pipeline, the replacement becomes "orphaned" — it will eventually promote,
and `_maintain_cluster_size` will remove the excess. From the perspective of
the recovered node, an orphaned replacement does not affect its own
availability or behavior. We can merge these into their base states.

**What changes:**

- `H_R` → `H` (healthy node with an orphaned replacement ≈ healthy node)
- `L_R` → `L` (lagging node with an orphaned replacement ≈ lagging node)

**Before:** `{H, H_R, Fw, Fe_R, L, L_R, Dw, De_R}` → **k = 8**

**After:** `{H, Fw, Fe_R, L, Dw, De_R}` → **k = 6**

**Accuracy cost:** Loses visibility into temporary over-provisioning (>N
physical nodes). The orphaned replacement still runs in reality, consuming
resources and eventually promoting, but the model treats the system as if
the replacement were never spawned once the original recovers.


|                   | N = 5              | N = 7              | Savings vs. previous (N = 5) | Savings vs. previous (N = 7) |
| ----------------- | ------------------ | ------------------ | ---------------------------- | ---------------------------- |
| **Homogeneous**   | C(10, 5) = **252** | C(12, 5) = **792** | 3.1× (from 792)              | 4.3× (from 3,432)            |
| **Heterogeneous** | 6^5 = **7,776**    | 6^7 = **279,936**  | 4.2× (from 32,768)           | 7.5× (from 2,097,152)        |


---

## Simplification 3: Collapse the replacement pipeline into a transition rate

**Insight:** Each failed or data-loss node goes through two phases: waiting
for the timeout to fire (Fw/Dw), then having a replacement in flight (Fe_R/De_R).
We can collapse both phases into a single state by modeling the entire
timeout-then-replace sequence as a single exponential rate. For failed nodes
this rate competes with the direct recovery rate.

**What changes:**

- `Fw` and `Fe_R` → `F` (a single "failed" state with effective rate
`μ_f = μ_direct + μ_replacement` out to H)
- `Dw` and `De_R` → `D` (a single "data loss" state with rate
`μ_d = 1 / (mean_timeout + mean_replace)` out to H)

**Before:** `{H, Fw, Fe_R, L, Dw, De_R}` → **k = 6**

**After:** `{H, F, L, D}` → **k = 4**

**Accuracy cost:** The actual time from failure to replacement is a sum of
stages (timeout + provision + sync), which is closer to Erlang than
exponential. A single exponential has higher variance (CV = 1), overweighting
both very fast and very slow replacements.


|                   | N = 5            | N = 7              | Savings vs. previous (N = 5) | Savings vs. previous (N = 7) |
| ----------------- | ---------------- | ------------------ | ---------------------------- | ---------------------------- |
| **Homogeneous**   | C(8, 3) = **56** | C(10, 3) = **120** | 4.5× (from 252)              | 6.6× (from 792)              |
| **Heterogeneous** | 4^5 = **1,024**  | 4^7 = **16,384**   | 7.6× (from 7,776)            | 17.1× (from 279,936)         |


---

## Simplification 4: Remove Lagging

**Insight:** The Lagging state exists only because a transiently failed node
recovers but has not yet synced to `commit_index`. We can absorb the sync-back
delay into the F → H transition rate, making recovery go directly from Failed
to Healthy.

**What changes:**

- Remove L from the per-node state space.
- The F → H rate becomes `μ_f = 1 / (mean_recovery_time + mean_sync_time)`,
encoding both the time to come back online and the time to catch up.

**Before:** `{H, F, L, D}` → **k = 4**

**After:** `{H, F, D}` → **k = 3**

**Accuracy cost:** A lagging node hasn't caught up on recent commits. By jumping
F → H directly, the model slightly underestimates availability during the
sync window. This matters most when sync time is long relative to failure
intervals.


|                   | N = 5            | N = 7            | Savings vs. previous (N = 5) | Savings vs. previous (N = 7) |
| ----------------- | ---------------- | ---------------- | ---------------------------- | ---------------------------- |
| **Homogeneous**   | C(7, 2) = **21** | C(9, 2) = **36** | 2.7× reduction (from 56)     | 3.3× reduction (from 120)    |
| **Heterogeneous** | 3^5 = **243**    | 3^7 = **2,187**  | 4.2× reduction (from 1,024)  | 7.5× reduction (from 16,384) |


---

## Summary of cumulative simplifications

### N = 5


| Model                                  | k (per-node) | Homogeneous | Heterogeneous |
| -------------------------------------- | ------------ | ----------- | ------------- |
| Full (pipeline + timer + availability) | 12           | 4,368       | 248,832       |
| 1: Merge P and S → R                   | 8            | 792         | 32,768        |
| 2: Remove orphaned replacements        | 6            | 252         | 7,776         |
| 3: Collapse pipeline into rate         | 4            | 56          | 1,024         |
| 4: Remove Lagging                      | 3            | 21          | 243           |


### N = 7


| Model                                  | k (per-node) | Homogeneous | Heterogeneous |
| -------------------------------------- | ------------ | ----------- | ------------- |
| Full (pipeline + timer + availability) | 12           | 31,824      | 35,831,808    |
| 1: Merge P and S → R                   | 8            | 3,432       | 2,097,152     |
| 2: Remove orphaned replacements        | 6            | 792         | 279,936       |
| 3: Collapse pipeline into rate         | 4            | 120         | 16,384        |
| 4: Remove Lagging                      | 3            | 36          | 2,187         |


### Total reduction (full → simplified)


|                   | N = 5 Homogeneous | N = 5 Heterogeneous | N = 7 Homogeneous | N = 7 Heterogeneous |
| ----------------- | ----------------- | ------------------- | ----------------- | ------------------- |
| Full → Simplified | **208×**          | **1,024×**          | **884×**          | **16,384×**         |


---

## Homogeneous vs. heterogeneous scaling

The symmetry reduction ratio `k^N / C(N + k − 1, k − 1)` grows rapidly
with N:


| N   | Homogeneous C(N+2, 2) | Heterogeneous 3^N | Ratio  |
| --- | --------------------- | ----------------- | ------ |
| 3   | 10                    | 27                | 2.7×   |
| 5   | 21                    | 243               | 11.6×  |
| 7   | 36                    | 2,187             | 60.8×  |
| 9   | 55                    | 19,683            | 357.9× |
| 11  | 78                    | 177,147           | 2,271× |


The homogeneous state space grows **O(N²)** while the heterogeneous grows
**O(3^N)**. Both are tractable for symbolic CTMC solvers (e.g. Wolfram) at
N = 5. By N ≈ 11, the heterogeneous model pushes the limits of analytical
solution.

---

## Encoding the "all-down" rule

The `strategy.py` cancels all replacement timers when `num_available == 0`
(line 233). In the simplified model this is encoded as state-dependent
transition rates, not additional states:


| Condition        | F → H rate                       | D → H rate                          |
| ---------------- | -------------------------------- | ----------------------------------- |
| n_H ≥ 1          | μ_direct + μ_replacement         | μ_d                                 |
| n_H = 0, n_F ≥ 1 | μ_direct only (timers cancelled) | 0 (stuck)                           |
| n_H = 0, n_F = 0 | N/A                              | 0 (absorbing; needs human recovery) |


For N = 5, 21 homogeneous states suffice—6 of them (the n_H = 0 states)
have different transition rates. For N = 7, 36 states suffice—8 of them
(n_H = 0) have different rates.

---

## Comparison with existing models in `rsm_model_utils.py`


| Model                          | N = 5  | N = 7    | Tracks replacement?   | Tracks data loss?      |
| ------------------------------ | ------ | -------- | --------------------- | ---------------------- |
| `get_cmm` (birth-death)        | 4      | 5        | No                    | No                     |
| `get_dr_cmm` (durability)      | ~15–21 | ~28–36   | No                    | Via out-of-date counts |
| `get_dr_good_bad_cmm` (2-type) | ~50–60 | ~100–130 | No                    | Via 2-type out-of-date |
| **Simplified full model**      | **21** | **36**   | **Yes (via rate)**    | **Yes (D state)**      |
| Full pipeline model            | 4,368  | 31,824   | Yes (explicit stages) | Yes                    |


The simplified model achieves a comparable state count to `get_dr_cmm` while
capturing the replacement strategy dynamics through the transition rates
rather than explicit pipeline states.