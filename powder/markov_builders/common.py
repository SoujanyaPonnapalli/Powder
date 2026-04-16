"""
Shared utilities for Markov chain state-space builders.

Provides rate extraction from MC objects, BFS state enumeration, and
the homogeneous cost model that maps states to billing rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from ..markov import MarkovModel
from ..markov_solver import build_q_from_triples
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
# Homogeneous cost model
# ---------------------------------------------------------------------------


def homogeneous_cost_fn(
    cost_per_second: float,
    total_nodes: int,
    unbilled_indices: tuple[int, ...],
) -> CostFn:
    """Cost function for a homogeneous cluster.

    Bills N - (data-lost-without-replacement count) replicas per unit time.
    Mirrors the MC billing rule of summing cost_per_hour over nodes with
    has_data == True: a slot whose data is lost and whose replacement
    hasn't been spawned yet contributes $0; every other slot contributes
    cost_per_second.

    For SIMPLIFIED/COLLAPSED levels, the unbilled index is the D state
    (pipeline is collapsed, so we treat D as unbilled uniformly; this
    slightly underestimates cost while a replacement is transiently
    running). For NO_ORPHANS/MERGED/FULL, the unbilled index is Dw,
    since De_R/De_P/De_S states have a live replacement VM.

    Args:
        cost_per_second: Hourly cost of one node, converted to $/second.
        total_nodes: The configured cluster size N.
        unbilled_indices: Indices into the state tuple that are not
            billed (slots with no active VM).

    Returns:
        A callable mapping a state tuple to its $/second cost rate.
    """
    unbilled = tuple(unbilled_indices)

    def _cost(counts: tuple[int, ...]) -> float:
        unbilled_count = sum(counts[i] for i in unbilled)
        return (total_nodes - unbilled_count) * cost_per_second

    return _cost
