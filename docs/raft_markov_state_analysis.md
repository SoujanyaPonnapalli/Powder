# Markov State Space Analysis for Raft with Node Replacement

This extends the [base RSM analysis](markov_state_analysis.md) by adding Raft
leader election. The same per-node state dimensions and simplifications apply;
the only addition is a **leader role** dimension and the transitions it
introduces.

## What Raft adds

In the base model, the system can commit whenever a majority of nodes are
healthy. Raft requires an additional condition: one healthy node must be the
**leader**. This introduces two new dynamics:

1. **Leader role:** Exactly one healthy node is the leader. The system can
  commit only when it has a leader AND a majority quorum.
2. **Leader election:** When the leader becomes unavailable, the remaining
  healthy nodes must elect a new leader. Election takes time (modeled as
   exponential rate μ_election), during which the system **cannot commit**
   even if quorum is available.

---

## Per-node state dimensions

The three dimensions from the base model (availability, replacement timer,
replacement pipeline) remain unchanged. Raft adds a fourth:

### 4. Role


| Sub-state      | Meaning                                               |
| -------------- | ----------------------------------------------------- |
| **Leader (★)** | This node is the cluster leader and processes commits |
| **Non-leader** | Follower or candidate                                 |


Constraints:

- At most **1** node can be leader at any time.
- Only **Healthy** nodes can be leader (F, L, D nodes are always non-leader).

### System-level state: leader status

For a given assignment of per-node states, the system is in one of two
leader statuses:


| Status         | Condition                     | Can commit?                                      |
| -------------- | ----------------------------- | ------------------------------------------------ |
| **Has leader** | One H node is leader, n_H ≥ 1 | Yes, if n_H ≥ majority                           |
| **No leader**  | No node is leader             | No (electing if n_H ≥ majority, stuck otherwise) |


### Key Raft transitions


| Transition         | Rate                             | Effect                                     |
| ------------------ | -------------------------------- | ------------------------------------------ |
| Leader fails       | λ (single node)                  | has_leader → no_leader; n_H -= 1, n_F += 1 |
| Follower fails     | (n_H - 1) × λ                    | n_H -= 1, n_F += 1; leader unchanged       |
| Election completes | μ_election (when n_H ≥ majority) | no_leader → has_leader                     |


---

## Counting states with the leader constraint

The "at most 1 leader" constraint means we cannot simply use k^N or
C(N+k-1, k-1). Instead, we split into states with and without a leader.

Let k_nl = number of non-leader per-node states and k_l = number of leader
per-node states (variants of Healthy that can be leader).

### Homogeneous formula

A leader is one distinguished node drawn from the Healthy pool. Since all
nodes are identical, the leader's specific sub-state matters but not its
identity.

- **No leader:** distribute N among k_nl types → C(N + k_nl - 1, k_nl - 1)
- **Leader of type j** (for each of k_l leader types): 1 leader + remaining
N-1 among k_nl types → C(N + k_nl - 2, k_nl - 1) per type

**Total = C(N + k_nl - 1, k_nl - 1) + k_l × C(N + k_nl - 2, k_nl - 1)**

### Heterogeneous formula

Nodes are distinguishable, so the leader's identity matters.

- **No leader:** each of N nodes in k_nl states → k_nl^N
- **Node i is leader type j:** 1 specific node in 1 of k_l leader states,
remaining N-1 in k_nl states → N × k_l × k_nl^(N-1)

**Total = k_nl^(N-1) × (k_nl + N × k_l)**

---

## Full model: 15 per-node states

The 12 non-leader states from the base model are unchanged. Leader variants
apply only to Healthy nodes (the leader must be available and up to date):


| Role | Availability | Timer | Pipeline | Description                                   |
| ---- | ------------ | ----- | -------- | --------------------------------------------- |
| ★    | H            | —     | None     | Leader, healthy                               |
| ★    | H            | —     | P        | Leader; orphaned replacement provisioning     |
| ★    | H            | —     | S        | Leader; orphaned replacement syncing          |
|      | H            | —     | None     | Follower, healthy                             |
|      | H            | —     | P        | Follower; orphaned replacement provisioning   |
|      | H            | —     | S        | Follower; orphaned replacement syncing        |
|      | F            | w     | None     | Failed, timer running                         |
|      | F            | e     | P        | Failed, replacement provisioning              |
|      | F            | e     | S        | Failed, replacement syncing                   |
|      | L            | —     | None     | Recovering, syncing back                      |
|      | L            | —     | P        | Recovering; orphaned replacement provisioning |
|      | L            | —     | S        | Recovering; orphaned replacement syncing      |
|      | D            | w     | None     | Data loss, timer running                      |
|      | D            | e     | P        | Data loss, replacement provisioning           |
|      | D            | e     | S        | Data loss, replacement syncing                |


**k_nl = 12 non-leader states, k_l = 3 leader states.**

### Full model state counts


|                   | N = 5       | N = 7          | Raft overhead |
| ----------------- | ----------- | -------------- | ------------- |
| **Homogeneous**   | **8,463**   | **68,952**     | 1.9× / 2.2×   |
| **Heterogeneous** | **559,872** | **98,537,472** | 2.2× / 2.8×   |


(Raft overhead = ratio vs. base model without leader tracking.)

---

## Simplification 1: Merge Provisioning and Syncing

Pipeline `{None, P, S}` → `{None, R}`. Leader types follow the same merge.

**k_nl = 8, k_l = 2** (leader states: {H★, H★_R})


|                   | N = 5      | N = 7         | Raft overhead |
| ----------------- | ---------- | ------------- | ------------- |
| **Homogeneous**   | **1,452**  | **6,864**     | 1.8× / 2.0×   |
| **Heterogeneous** | **73,728** | **5,767,168** | 2.2× / 2.8×   |


---

## Simplification 2: Remove orphaned replacement tracking

`H_R` → `H` and `L_R` → `L`. For the leader: `H★_R` → `H★`.

**k_nl = 6, k_l = 1** (single leader state: {H★})


|                   | N = 5      | N = 7       | Raft overhead |
| ----------------- | ---------- | ----------- | ------------- |
| **Homogeneous**   | **378**    | **1,254**   | 1.5× / 1.6×   |
| **Heterogeneous** | **14,256** | **606,528** | 1.8× / 2.2×   |


---

## Simplification 3: Collapse the replacement pipeline into a transition rate

`Fw` + `Fe_R` → `F`, `Dw` + `De_R` → `D`.

**k_nl = 4, k_l = 1** (leader state: {H★})


|                   | N = 5     | N = 7      | Raft overhead |
| ----------------- | --------- | ---------- | ------------- |
| **Homogeneous**   | **91**    | **204**    | 1.6× / 1.7×   |
| **Heterogeneous** | **2,304** | **45,056** | 2.2× / 2.8×   |


---

## Simplification 4: Remove Lagging

`L` absorbed into F → H transition rate.

**k_nl = 3, k_l = 1** (leader state: {H★})

### Homogeneous

The formula simplifies to a clean closed form:

C(N+2, 2) + C(N+1, 2) = **(N + 1)²**


|                   | N = 5            | N = 7                | vs. base model |
| ----------------- | ---------------- | -------------------- | -------------- |
| **Homogeneous**   | 6² = **36**      | 8² = **64**          | 1.7× / 1.8×    |
| **Heterogeneous** | 81 × 8 = **648** | 729 × 10 = **7,290** | 2.7× / 3.3×    |


The final per-node states are: **{H★, H, F, D}** with at most one H★.

---

## Summary of cumulative simplifications

### N = 5


| Model                           | k_nl / k_l | Homogeneous | Heterogeneous |
| ------------------------------- | ---------- | ----------- | ------------- |
| Full model                      | 12 / 3     | 8,463       | 559,872       |
| 1: Merge P and S → R            | 8 / 2      | 1,452       | 73,728        |
| 2: Remove orphaned replacements | 6 / 1      | 378         | 14,256        |
| 3: Collapse pipeline into rate  | 4 / 1      | 91          | 2,304         |
| 4: Remove Lagging               | 3 / 1      | 36          | 648           |


### N = 7


| Model                           | k_nl / k_l | Homogeneous | Heterogeneous |
| ------------------------------- | ---------- | ----------- | ------------- |
| Full model                      | 12 / 3     | 68,952      | 98,537,472    |
| 1: Merge P and S → R            | 8 / 2      | 6,864       | 5,767,168     |
| 2: Remove orphaned replacements | 6 / 1      | 1,254       | 606,528       |
| 3: Collapse pipeline into rate  | 4 / 1      | 204         | 45,056        |
| 4: Remove Lagging               | 3 / 1      | 64          | 7,290         |


### Total reduction (full → simplified)


|                   | N = 5 Homo | N = 5 Hetero | N = 7 Homo | N = 7 Hetero |
| ----------------- | ---------- | ------------ | ---------- | ------------ |
| Full → Simplified | **235×**   | **864×**     | **1,077×** | **13,516×**  |


---

## Raft overhead: side-by-side with base model

How much does tracking the leader cost compared to the leaderless base model?

### Simplified model (after all 4 simplifications)


|                         | Base model | Raft model | Raft overhead |
| ----------------------- | ---------- | ---------- | ------------- |
| **N = 5 Homogeneous**   | 21         | 36         | **1.71×**     |
| **N = 5 Heterogeneous** | 243        | 648        | **2.67×**     |
| **N = 7 Homogeneous**   | 36         | 64         | **1.78×**     |
| **N = 7 Heterogeneous** | 2,187      | 7,290      | **3.33×**     |


### Scaling with N (simplified model)


| N   | Base Homo | Raft Homo | Overhead | Base Hetero | Raft Hetero | Overhead |
| --- | --------- | --------- | -------- | ----------- | ----------- | -------- |
| 3   | 10        | 16        | 1.60×    | 27          | 54          | 2.00×    |
| 5   | 21        | 36        | 1.71×    | 243         | 648         | 2.67×    |
| 7   | 36        | 64        | 1.78×    | 2,187       | 7,290       | 3.33×    |
| 9   | 55        | 100       | 1.82×    | 19,683      | 78,732      | 4.00×    |
| 11  | 78        | 144       | 1.85×    | 177,147     | 826,686     | 4.67×    |


The homogeneous overhead converges to **2×** as N grows (from (N+1)² vs
(N+1)(N+2)/2, ratio → 2(N+1)/(N+2) → 2). The heterogeneous overhead grows
as **(N+3)/3** — roughly one extra state per node for the leader identity.

---

## Availability with Raft

In the base model, the system is available whenever n_H ≥ majority.

With Raft, availability requires **both**:

1. A leader exists (has_leader = true)
2. The leader can reach a majority: n_H ≥ majority

The **election gap** — time spent in the "no leader, electing" state — is
the key new source of unavailability that Raft introduces. Every leader
failure triggers an election with mean duration 1/μ_election, during which
the system cannot commit even if quorum is available.

### Encoding leader transitions in the simplified model

For the simplified homogeneous Raft model with (N+1)² states, the
transitions involving the leader are:


| From                        | To                              | Rate           | Condition                            |
| --------------------------- | ------------------------------- | -------------- | ------------------------------------ |
| (n_H, n_F, n_D, has_leader) | (n_H-1, n_F+1, n_D, no_leader)  | λ              | Leader fails                         |
| (n_H, n_F, n_D, has_leader) | (n_H-1, n_F+1, n_D, has_leader) | (n_H - 1) × λ  | Follower fails                       |
| (n_H, n_F, n_D, no_leader)  | (n_H, n_F, n_D, has_leader)     | μ_election     | n_H ≥ majority                       |
| (n_H, n_F, n_D, has_leader) | (n_H-1, n_F, n_D+1, has_leader) | 0              | Leader can't lose data (it has data) |
| (n_H, n_F, n_D, has_leader) | (n_H, n_F-1, n_D, has_leader)   | n_F × μ_direct | Follower recovers                    |


All other transitions (D → H via replacement, F → H via recovery, the
"all-down" timer cancellation rule) remain identical to the base model.