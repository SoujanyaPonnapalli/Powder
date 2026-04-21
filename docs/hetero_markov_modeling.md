# Heterogeneous Markov Modeling

The leaderless and Raft CTMC builders in `powder/markov_builders/` accept
a `list[NodeConfig]` that may contain nodes with different failure,
recovery, data-loss, replacement, or cost parameters. This document
describes how heterogeneity is collapsed into a tractable state space
and what invariants the builders preserve.

Homogeneous inputs (`[cfg] * N`) remain a first-class case: they
collapse to a single rate class and produce the same state space, the
same transitions, and the same availability / MTTF / cost values as
the pre-change implementation. See
`[tests/test_markov_hetero_reduces_to_homogeneous.py](../tests/test_markov_hetero_reduces_to_homogeneous.py)`
for the golden-value pins.

## Rate classes

Each config is passed through `extract_rates` to recover the scalar
rates that the chain actually consumes
(failure / data-loss / recovery / sync / spawn / timeout). Those rates,
plus the per-second cost, are combined into a **rate signature** that
is specific to the chosen `QualityLevel`:


| QualityLevel       | Signature (tuple of floats)                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| SIMPLIFIED         | `(lam, lam_d, recovery_with_sync_rate, collapsed_replace_rate, cost_per_second)`               |
| COLLAPSED_PIPELINE | `(lam, lam_d, recovery_rate, sync_rate, collapsed_replace_rate, cost_per_second)`              |
| NO_ORPHANS         | `(lam, lam_d, recovery_rate, sync_rate, timeout_rate, replace_pipeline_rate, cost_per_second)` |
| MERGED_PIPELINE    | same as NO_ORPHANS                                                                             |
| FULL               | `(lam, lam_d, recovery_rate, sync_rate, timeout_rate, spawn_rate, cost_per_second)`            |


At SIMPLIFIED and COLLAPSED_PIPELINE the signature uses already-combined
composites (`collapsed_replace_rate = 1 / (1/timeout + 1/spawn + 1/sync)`,
`recovery_with_sync_rate = 1 / (1/recovery + 1/sync)`). Two configs whose
individual spawn / sync / recovery rates differ but whose composites
coincide live in the same class at those levels -- which is what the
transition code actually consumes. The FULL signature separates every
raw rate so no such collapsing is possible.

`group_configs_into_classes(configs, strategy, quality)` partitions
configs into rate classes:

- Signatures are compared entrywise with `np.isclose` (`rtol = 1e-9`,
`atol = 1e-15`) so arithmetically-equivalent floats are merged.
- Infinities compare only with equal infinities to avoid silently
collapsing `sync_rate = inf` (instantaneous sync) with finite values.
- Union-find produces a canonical, input-order-stable ordering of
classes. Class `0` always contains the first input config.

The output is a `list[RateClass]`, each carrying `rates`, `size`,
`cost_per_second`, `class_idx`, and the list of input-config positions
that map into it.

## State encoding

Let `C` be the number of rate classes and `k` the per-node-state count
at the chosen quality level (3, 4, 6, 8, or 12). Per-class counts
occupy the leading `C * k` slots of the state tuple; Raft appends
leader fields afterwards.

```
Leaderless:
    (c_0 counts..., ..., c_{C-1} counts...)
    length = C * k

Raft SIMPLIFIED / COLLAPSED_PIPELINE / NO_ORPHANS:
    (c_0 counts..., ..., has_leader, leader_class)
    length = C * k + 2

Raft MERGED_PIPELINE:
    (c_0 counts..., ..., has_leader, leader_class, leader_orphan)
    length = C * k + 3

Raft FULL:
    (c_0 counts..., ..., has_leader, leader_class, leader_pipe)
    length = C * k + 3
```

`leader_class` ranges over `{0, ..., C-1}`. When `has_leader == 0` we
reset `leader_class` (and any `leader_orphan` / `leader_pipe` flags) to
`0` so every no-leader state hashes to a canonical tuple.

## State-count bounds

With `N = sum_c n_c` total slots, the reachable state space is bounded
above by

```
sum_leader   prod_c   C(n_c^* + k - 1, k - 1)
```

where `n_c^* = n_c - [class c currently holds the leader]`. For
leaderless the leader term drops out and the bound is
`prod_c C(n_c + k - 1, k - 1)`.

Two limiting cases:

- `C = 1`: the product collapses to `C(N + k - 1, k - 1)` -- exactly
the pre-change homogeneous count.
- `C = N`: each class holds a single slot and the bound becomes
`k^N` -- the full per-slot upper bound.

Intermediate partitions grow smoothly between those two extremes.
`[tests/test_markov_hetero_state_counts.py](../tests/test_markov_hetero_state_counts.py)`
pins the SIMPLIFIED counts at all three points.

## Transitions

For every reachable state the transition generator iterates classes
and replicates, for class `c`, the transitions the homogeneous
builder would emit with class-`c` rates and class-`c` counts. Only
two cross-class couplings exist:

1. `all_down` predicates sum the healthy-like per-node states across
  all classes before deciding whether replacement timers fire.
2. Raft elections fire at total rate `mu_election` split across
  classes in proportion to the class's contribution to the healthy
   pool (`nH_c / total_H`, or the merged-pipeline / full equivalent).
   The sum of the per-class election rates out of a no-leader state
   stays exactly `mu_election`, so the `C = 1` collapse is preserved.

Leader-specific transitions -- leader failure, leader data loss,
leader orphan completion, leader-pipe advancement -- look up rates by
`leader_class`, so they use the correct class's rates when the cluster
is heterogeneous.

## Replacement preserves class membership

A physical slot's rate profile is fixed: a worn-out disk does not
become a fresh disk when its node is replaced. In the Monte Carlo
simulator (`powder/simulation/cluster.py`) replacement reuses the
slot's existing `NodeConfig`. The Markov builders assume the same:
replacements reconstitute slots inside the original rate class, so
`size_c` is conserved across every transition.

## Cost model

`heterogeneous_cost_fn` bills each class at its own rate:

```
cost_rate(state) = sum_c (size_c - unbilled_count_c(state_c)) * cost_per_second_c
```

`size_c` already includes the leader slot when the leader belongs to
class `c` (the per-class counts sum to `size_c - 1` in that case), so
the leader is billed exactly once at its class's cost. Homogeneous
clusters collapse to `(N - unbilled_count) * cost_per_second`, which
matches the pre-change formula.

## Solver

The generated `MarkovModel` uses the same sparse `Q` representation
and the same steady-state solver as before. Any future GPU acceleration
operates on the same object and is unaffected by the heterogeneity
generalization.