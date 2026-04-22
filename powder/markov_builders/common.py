"""
Shared utilities for Markov chain state-space builders.

Provides rate extraction from MC objects, BFS state enumeration, the
cost model that maps states to billing rates, and rate-class grouping
for heterogeneous clusters.

Heterogeneous modeling:
    A cluster is described as ``list[NodeConfig]``. We partition configs
    into *rate classes* where two configs share a class iff the rates
    actually consumed by the chosen ``QualityLevel`` (plus cost) agree
    within a floating-point tolerance. At ``QualityLevel.SIMPLIFIED``
    and ``QualityLevel.COLLAPSED_PIPELINE`` the signature uses
    already-combined composite rates (``collapsed_replace_rate`` etc.)
    so two configs whose individual spawn/sync rates differ but whose
    composites match are grouped together.

    State tuples are a concatenation of per-class count tuples. When all
    configs land in a single class (``C = 1``) the encoding reduces to
    the homogeneous count layout used by the original builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from ..markov import MarkovModel
from ..markov_solver import build_q_from_triples
from ..scenario import QualityLevel
from ..simulation.node import NodeConfig
from ..simulation.strategy import ClusterStrategy


@dataclass(frozen=True)
class ExtractedRates:
    """Scalar rates extracted from a NodeConfig and strategy for Markov builders.

    All rates are in events-per-second. A rate of 0 means the event
    never occurs (e.g. no replacement when using NoOpStrategy).
    """

    failure_rate: float
    recovery_rate: float
    data_loss_rate: float
    spawn_rate: float
    sync_rate: float
    timeout_rate: float

    @property
    def replace_pipeline_rate(self) -> float:
        """Effective rate for the merged replacement pipeline (spawn + sync).

        Models the sequential spawn-then-sync process as a single
        exponential with mean = mean_spawn + mean_sync.
        Returns 0 if either stage has rate 0.
        """
        if self.spawn_rate <= 0 or self.sync_rate <= 0:
            return 0.0
        return 1.0 / (1.0 / self.spawn_rate + 1.0 / self.sync_rate)

    @property
    def collapsed_replace_rate(self) -> float:
        """Effective rate for the fully collapsed replacement path (timeout + spawn + sync).

        Models the sequential timeout-wait, spawn, sync process as a single
        exponential with mean = mean_timeout + mean_spawn + mean_sync.
        Returns 0 if timeout_rate is 0 (no replacement strategy).
        """
        if self.timeout_rate <= 0:
            return 0.0
        total_mean = 1.0 / self.timeout_rate
        if self.spawn_rate > 0:
            total_mean += 1.0 / self.spawn_rate
        if self.sync_rate > 0:
            total_mean += 1.0 / self.sync_rate
        return 1.0 / total_mean

    @property
    def recovery_with_sync_rate(self) -> float:
        """Effective rate for direct recovery including sync-back time.

        Models recovery + sync as sequential: mean = mean_recovery + mean_sync.
        Used by the SIMPLIFIED quality level where lagging is absorbed.
        """
        total_mean = 1.0 / self.recovery_rate if self.recovery_rate > 0 else float("inf")
        if self.sync_rate > 0:
            total_mean += 1.0 / self.sync_rate
        if total_mean == float("inf"):
            return 0.0
        return 1.0 / total_mean


def extract_rates(node_config: NodeConfig, strategy: ClusterStrategy) -> ExtractedRates:
    """Extract scalar exponential rates from MC simulation objects.

    For non-exponential distributions, uses 1/mean as the rate (the
    exponential approximation with matching mean).
    """
    return ExtractedRates(
        failure_rate=node_config.failure_dist.approx_rate,
        recovery_rate=node_config.recovery_dist.approx_rate,
        data_loss_rate=node_config.data_loss_dist.approx_rate,
        spawn_rate=node_config.spawn_dist.approx_rate,
        sync_rate=1.0 / node_config.snapshot_download_time_dist.mean
        if node_config.snapshot_download_time_dist.mean > 0
        else float("inf"),
        timeout_rate=strategy.replacement_rate,
    )


def majority(n: int) -> int:
    """Minimum number of nodes needed for majority quorum."""
    return n // 2 + 1


def weak_compositions(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """Generate all weak compositions of n into k non-negative parts.

    Yields tuples (a_1, ..., a_k) with a_i >= 0 and sum = n.
    These represent homogeneous Markov states where a_i is the count
    of nodes in per-node state i.
    """
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in weak_compositions(n - i, k - 1):
            yield (i,) + rest


def state_name(counts: tuple[int, ...]) -> str:
    """Format a state tuple as ':'-separated string."""
    return ":".join(str(c) for c in counts)


# Type aliases for builder callbacks -----------------------------------------

# Transition function: given a state tuple, returns (target_tuple, rate) pairs.
TransitionFn = Callable[[tuple[int, ...]], list[tuple[tuple[int, ...], float]]]

# Liveness function: given a state tuple, returns True if live.
LivenessFn = Callable[[tuple[int, ...]], bool]

# Cost function: given a state tuple, returns the cost rate in $/second.
CostFn = Callable[[tuple[int, ...]], float]


def _zero_cost(_: tuple[int, ...]) -> float:
    return 0.0


def build_model_bfs(
    initial_state: tuple[int, ...],
    transition_fn: TransitionFn,
    liveness_fn: LivenessFn,
    cost_fn: CostFn = _zero_cost,
) -> MarkovModel:
    """Enumerate reachable states via BFS and build a MarkovModel.

    Uses integer state IDs throughout; only keeps string names for
    debugging output. Produces a sparse Q matrix sized to the reachable
    state count.

    Args:
        initial_state: Starting state as a count tuple.
        transition_fn: Given a state tuple, returns list of
            (target_tuple, rate) pairs. Zero-rate transitions must be
            filtered by the caller.
        liveness_fn: Given a state tuple, returns True if the state is live.
        cost_fn: Given a state tuple, returns the cost rate ($/second)
            while in that state.

    Returns:
        A MarkovModel with sparse Q, live mask, cost vector, and integer IDs.
    """
    state_to_id: dict[tuple[int, ...], int] = {initial_state: 0}
    state_tuples: list[tuple[int, ...]] = [initial_state]
    queue: list[tuple[int, ...]] = [initial_state]
    triples: list[tuple[int, int, float]] = []

    while queue:
        s = queue.pop()
        src_id = state_to_id[s]
        for target_tuple, rate in transition_fn(s):
            dst_id = state_to_id.get(target_tuple)
            if dst_id is None:
                dst_id = len(state_tuples)
                state_to_id[target_tuple] = dst_id
                state_tuples.append(target_tuple)
                queue.append(target_tuple)
            triples.append((src_id, dst_id, float(rate)))

    n = len(state_tuples)
    Q = build_q_from_triples(n, triples)

    live_mask = np.fromiter(
        (liveness_fn(s) for s in state_tuples),
        dtype=bool,
        count=n,
    )
    state_costs = np.fromiter(
        (cost_fn(s) for s in state_tuples),
        dtype=np.float64,
        count=n,
    )
    initial = np.zeros(n, dtype=np.float64)
    initial[0] = 1.0
    names = [state_name(s) for s in state_tuples]

    return MarkovModel(
        Q=Q,
        initial_distribution=initial,
        live_mask=live_mask,
        state_costs=state_costs,
        state_names=names,
    )


# ---------------------------------------------------------------------------
# Rate-class grouping for heterogeneous clusters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateClass:
    """A group of slots that share the same effective rates.

    Attributes:
        rates: The ExtractedRates that every member of this class uses.
            Builders consume these rates directly; members may have had
            individually different raw rates that collapse to the same
            composite values at the chosen quality level.
        size: Number of node slots in this class.
        cost_per_second: Billing rate of one slot in this class ($/s).
        class_idx: Zero-based position of this class in the canonical
            ordering produced by ``group_configs_into_classes``.
        member_indices: Positions of the member NodeConfigs in the
            original ``list[NodeConfig]`` input, for debugging.
    """

    rates: ExtractedRates
    size: int
    cost_per_second: float
    class_idx: int
    member_indices: tuple[int, ...] = ()


def _rate_signature(
    rates: ExtractedRates,
    quality: QualityLevel,
    cost_per_second: float,
) -> tuple[float, ...]:
    """Return the float tuple that defines rate-class equivalence at ``quality``.

    The signature lists exactly the rates that the chosen quality level's
    transition generator consumes. For SIMPLIFIED/COLLAPSED_PIPELINE we
    use already-combined composites so two configs whose raw rates differ
    but whose composites match end up in the same class.
    """
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    if quality == QualityLevel.SIMPLIFIED:
        return (
            lam,
            lam_d,
            rates.recovery_with_sync_rate,
            rates.collapsed_replace_rate,
            cost_per_second,
        )
    if quality == QualityLevel.COLLAPSED_PIPELINE:
        return (
            lam,
            lam_d,
            rates.recovery_rate,
            rates.sync_rate,
            rates.collapsed_replace_rate,
            cost_per_second,
        )
    if quality in (QualityLevel.NO_ORPHANS, QualityLevel.MERGED_PIPELINE):
        return (
            lam,
            lam_d,
            rates.recovery_rate,
            rates.sync_rate,
            rates.timeout_rate,
            rates.replace_pipeline_rate,
            cost_per_second,
        )
    if quality == QualityLevel.FULL:
        return (
            lam,
            lam_d,
            rates.recovery_rate,
            rates.sync_rate,
            rates.timeout_rate,
            rates.spawn_rate,
            cost_per_second,
        )
    raise ValueError(f"Unsupported QualityLevel: {quality}")


def _signatures_close(
    a: tuple[float, ...],
    b: tuple[float, ...],
    rtol: float,
    atol: float,
) -> bool:
    """Entry-wise near-equality with numpy semantics, treating infs specially."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if np.isinf(x) or np.isinf(y):
            if x != y:
                return False
            continue
        if not np.isclose(x, y, rtol=rtol, atol=atol):
            return False
    return True


def group_configs_into_classes(
    node_configs: list[NodeConfig],
    strategy: ClusterStrategy,
    quality: QualityLevel,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-15,
) -> list[RateClass]:
    """Partition ``node_configs`` into rate classes for ``quality``.

    Two configs share a class iff their rate signatures at ``quality``
    agree within the tolerance. The partition is stable: classes are
    ordered by the first occurrence of a member in ``node_configs``.

    With ``len(node_configs) == 1`` or all-identical configs the result
    is a single class and the downstream state space collapses to the
    original homogeneous shape.
    """
    if not node_configs:
        return []

    rates_list = [extract_rates(c, strategy) for c in node_configs]
    cps_list = [c.cost_per_hour / 3600.0 for c in node_configs]
    sigs = [
        _rate_signature(r, quality, cps)
        for r, cps in zip(rates_list, cps_list)
    ]

    # Union-find over the O(N) configs; N is small (typically <= 20) so
    # O(N^2) pairwise comparison is fine and keeps the tolerance check
    # explicit rather than dependent on a hashing round-trip.
    n = len(node_configs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if _signatures_close(sigs[i], sigs[j], rtol, atol):
                union(i, j)

    # Build classes, preserving input order by first occurrence.
    root_to_class: dict[int, int] = {}
    members: list[list[int]] = []
    for i in range(n):
        r = find(i)
        if r not in root_to_class:
            root_to_class[r] = len(members)
            members.append([])
        members[root_to_class[r]].append(i)

    classes: list[RateClass] = []
    for cid, idxs in enumerate(members):
        rep = idxs[0]
        classes.append(
            RateClass(
                rates=rates_list[rep],
                size=len(idxs),
                cost_per_second=cps_list[rep],
                class_idx=cid,
                member_indices=tuple(idxs),
            )
        )
    return classes


# ---------------------------------------------------------------------------
# Per-class state helpers
# ---------------------------------------------------------------------------


def class_counts(
    state: tuple[int, ...], class_idx: int, k: int,
) -> tuple[int, ...]:
    """Extract the k-tuple of per-node-state counts for class ``class_idx``."""
    off = class_idx * k
    return state[off : off + k]


def delta_in_class(
    state: tuple[int, ...],
    class_idx: int,
    k: int,
    *changes: tuple[int, int],
) -> tuple[int, ...]:
    """Return ``state`` with per-node-state deltas applied inside one class.

    Each change is ``(per_node_idx, delta)``. Other classes and any
    trailing protocol flags (e.g. Raft leader fields) are left untouched.
    """
    new = list(state)
    off = class_idx * k
    for idx, d in changes:
        new[off + idx] += d
    return tuple(new)


def set_trailing(
    state: tuple[int, ...],
    trailing_values: dict[int, int],
) -> tuple[int, ...]:
    """Return ``state`` with specific trailing flag slots set.

    Keys are absolute indices; typically ``len(state) - k`` for the k-th
    trailing flag. Used by Raft builders to update ``has_leader``,
    ``leader_class``, and pipeline/orphan flags in one step.
    """
    new = list(state)
    for idx, val in trailing_values.items():
        new[idx] = val
    return tuple(new)


def total_per_node_state(
    state: tuple[int, ...], per_node_idx: int, num_classes: int, k: int,
) -> int:
    """Sum the count at ``per_node_idx`` across all classes."""
    return sum(state[c * k + per_node_idx] for c in range(num_classes))


def make_initial_counts(classes: list[RateClass], k: int) -> list[int]:
    """Build the per-class section of an initial state with all slots in H (index 0)."""
    flat = [0] * (len(classes) * k)
    for rc in classes:
        flat[rc.class_idx * k + 0] = rc.size
    return flat


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def heterogeneous_cost_fn(
    classes: list[RateClass],
    k: int,
    unbilled_indices: tuple[int, ...],
    *,
    leader_has_leader_idx: int | None = None,
    leader_class_idx: int | None = None,
) -> CostFn:
    """Cost function for a (possibly heterogeneous) cluster.

    Bills each class's slots at that class's ``cost_per_second``:
    ``(size_c - unbilled_count_c(state)) * cost_per_second_c``. The
    ``size_c`` already includes the leader slot when it belongs to
    class ``c`` (the per-class counts sum to ``size_c - 1`` while
    ``has_leader == 1 and leader_class == c``), so Raft builders do
    not add the leader separately. The ``leader_has_leader_idx`` and
    ``leader_class_idx`` parameters are accepted for future use and
    validation but are not consulted here.

    Args:
        classes: The rate classes in canonical order.
        k: Per-node state count (columns per class within the state tuple).
        unbilled_indices: Per-node-state indices that are not billed
            (e.g. the Dw slot when the replacement VM has not yet
            been spawned).
        leader_has_leader_idx: Absolute state index of the ``has_leader``
            trailing flag, or ``None`` for leaderless builders.
        leader_class_idx: Absolute state index of the ``leader_class``
            trailing flag. Must be supplied together with
            ``leader_has_leader_idx``.

    Reduces to the original homogeneous billing when ``len(classes) == 1``.
    """
    unbilled = tuple(unbilled_indices)
    C = len(classes)
    cps = tuple(c.cost_per_second for c in classes)
    sizes = tuple(c.size for c in classes)

    if (leader_has_leader_idx is None) != (leader_class_idx is None):
        raise ValueError(
            "leader_has_leader_idx and leader_class_idx must be supplied together",
        )

    def _cost(s: tuple[int, ...]) -> float:
        total = 0.0
        for c in range(C):
            off = c * k
            unbilled_count = sum(s[off + i] for i in unbilled)
            total += (sizes[c] - unbilled_count) * cps[c]
        return total

    return _cost
