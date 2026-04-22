"""
Raft Markov chain builders at 5 quality levels.

Each builder generates a CTMC for an N-node Raft cluster. The state
space extends the leaderless model with a leader dimension: at most
one Healthy node is the designated leader, and the system can only
commit when a leader exists AND a majority quorum of healthy nodes
is available. Leader failure triggers an election; election completes
at rate mu_election only when quorum is available.

Heterogeneity:
    Inputs are a ``list[NodeConfig]`` that is partitioned into rate
    classes (see ``common.group_configs_into_classes``). The leading
    part of the state tuple is the concatenation of per-class count
    tuples; the trailing part holds leader fields. ``leader_class``
    records which rate class the current leader belongs to, so leader
    failure / data-loss / orphan-completion transitions use the
    correct rates. Election fires at rate ``mu_election`` split across
    classes in proportion to their healthy counts so the total
    no-leader exit rate stays ``mu_election`` at ``C = 1``.
    When every config collapses to a single class the state space
    matches the original homogeneous builder exactly (``leader_class``
    is always 0).

State encoding:
    SIMPLIFIED / COLLAPSED_PIPELINE / NO_ORPHANS:
        (c_0 counts..., ..., c_{C-1} counts..., has_leader, leader_class)
    MERGED_PIPELINE:
        (..., has_leader, leader_class, leader_orphan)
    FULL:
        (..., has_leader, leader_class, leader_pipe)

    Convention: ``leader_class`` is set to 0 when ``has_leader == 0``
    so equivalent no-leader states hash to the same tuple.

Modeling note -- data-loss recovery in all-down states:
    See leaderless.py module docstring for the rationale.  The same
    approximation is applied here: data-loss replacement proceeds even
    when all nodes are unavailable, preventing the all-data-loss state
    from becoming absorbing.
"""

from __future__ import annotations

import logging

from ..markov import MarkovModel
from ..scenario import QualityLevel
from ..simulation.node import NodeConfig
from ..simulation.protocol import Protocol, RaftLikeProtocol
from ..simulation.strategy import ClusterStrategy
from .common import (
    RateClass,
    build_model_bfs,
    delta_in_class,
    group_configs_into_classes,
    heterogeneous_cost_fn,
    majority,
    total_per_node_state,
)

_logger = logging.getLogger(__name__)


Transition = tuple[tuple[int, ...], float]


def build_raft_model(
    node_configs: list[NodeConfig],
    protocol: Protocol,
    strategy: ClusterStrategy,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> MarkovModel:
    """Build a Raft CTMC at the requested quality level.

    Supports heterogeneous ``node_configs``: configs are partitioned
    into rate classes using the effective rates consumed by ``quality``.
    When every config maps to a single class the state space reduces
    to the homogeneous encoding (with a trailing ``leader_class`` field
    that is always 0).
    """
    n = len(node_configs)
    classes = group_configs_into_classes(node_configs, strategy, quality)

    if not isinstance(protocol, RaftLikeProtocol):
        raise TypeError(f"Expected RaftLikeProtocol, got {type(protocol).__name__}")

    mu_election = protocol.election_rate

    builders = {
        QualityLevel.SIMPLIFIED: _build_simplified,
        QualityLevel.COLLAPSED_PIPELINE: _build_collapsed_pipeline,
        QualityLevel.NO_ORPHANS: _build_no_orphans,
        QualityLevel.MERGED_PIPELINE: _build_merged_pipeline,
        QualityLevel.FULL: _build_full,
    }
    model = builders[quality](n, classes, mu_election)
    _logger.debug(
        "raft/%s N=%d classes=%d (sizes=%s) states=%d",
        quality.name,
        n,
        len(classes),
        [c.size for c in classes],
        model.num_states,
    )
    return model


# ---------------------------------------------------------------------------
# SIMPLIFIED (k=3, trailing=2): {H, F, D} + (has_leader, leader_class)
#   Per-class indices: 0=H, 1=F, 2=D
# ---------------------------------------------------------------------------


def _build_simplified(
    n: int, classes: list[RateClass], mu_election: float,
) -> MarkovModel:
    k = 3
    C = len(classes)
    TRAILING = 2
    HL = C * k
    LC = C * k + 1
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        has_leader = s[HL]
        leader_class = s[LC]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        all_down = total_H == 0

        if has_leader:
            lc_rates = classes[leader_class].rates
            lam = lc_rates.failure_rate
            lam_d = lc_rates.data_loss_rate
            # Leader fails -> class `leader_class` F += 1, drop leader.
            new = list(delta_in_class(s, leader_class, k, (1, 1)))
            new[HL] = 0
            new[LC] = 0
            trans.append((tuple(new), lam))
            if lam_d > 0:
                new = list(delta_in_class(s, leader_class, k, (2, 1)))
                new[HL] = 0
                new[LC] = 0
                trans.append((tuple(new), lam_d))

        for rc in classes:
            c = rc.class_idx
            off = c * k
            nH, nF, nD = s[off], s[off + 1], s[off + 2]
            lam = rc.rates.failure_rate
            lam_d = rc.rates.data_loss_rate
            mu_f_direct = rc.rates.recovery_with_sync_rate
            mu_f_replace = rc.rates.collapsed_replace_rate
            mu_d_replace = rc.rates.collapsed_replace_rate

            if nH > 0:
                trans.append((delta_in_class(s, c, k, (0, -1), (1, 1)), nH * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, -1), (2, 1)), nH * lam_d),
                    )
            if nF > 0:
                rate = nF * mu_f_direct
                if not all_down and mu_f_replace > 0:
                    rate += nF * mu_f_replace
                if rate > 0:
                    trans.append((delta_in_class(s, c, k, (0, 1), (1, -1)), rate))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (2, 1)), nF * lam_d),
                    )
            if nD > 0 and mu_d_replace > 0:
                trans.append(
                    (delta_in_class(s, c, k, (0, 1), (2, -1)), nD * mu_d_replace),
                )

        if not has_leader and total_H >= q and mu_election > 0:
            # Split mu_election across classes proportionally to their H counts.
            for rc in classes:
                c = rc.class_idx
                nH_c = s[c * k]
                if nH_c > 0:
                    rate = mu_election * nH_c / total_H
                    new = list(delta_in_class(s, c, k, (0, -1)))
                    new[HL] = 1
                    new[LC] = c
                    trans.append((tuple(new), rate))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        has_leader = s[HL]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        return has_leader == 1 and total_H >= q

    initial = _initial_with_leader(classes, k, TRAILING)
    cost_fn = heterogeneous_cost_fn(
        classes, k, unbilled_indices=(2,),
        leader_has_leader_idx=HL, leader_class_idx=LC,
    )
    return build_model_bfs(initial, transitions, liveness, cost_fn)


def _initial_with_leader(
    classes: list[RateClass], k: int, trailing: int,
) -> tuple[int, ...]:
    """Initial state: all healthy, leader drawn from class 0."""
    C = len(classes)
    flat = [0] * (C * k + trailing)
    for rc in classes:
        flat[rc.class_idx * k + 0] = rc.size
    flat[0 * k + 0] -= 1  # class 0 contributes the leader
    flat[C * k] = 1  # has_leader
    flat[C * k + 1] = 0  # leader_class = 0
    # remaining trailing slots already zero (leader_orphan / leader_pipe)
    return tuple(flat)


# ---------------------------------------------------------------------------
# COLLAPSED_PIPELINE (k=4, trailing=2): {H, F, L, D} + (has_leader, leader_class)
#   Per-class indices: 0=H, 1=F, 2=L, 3=D
# ---------------------------------------------------------------------------


def _build_collapsed_pipeline(
    n: int, classes: list[RateClass], mu_election: float,
) -> MarkovModel:
    k = 4
    C = len(classes)
    TRAILING = 2
    HL = C * k
    LC = C * k + 1
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        has_leader = s[HL]
        leader_class = s[LC]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        total_L = total_per_node_state(s, 2, C, k)
        n_available = total_H + total_L
        all_down = n_available == 0

        if has_leader:
            lc_rates = classes[leader_class].rates
            lam = lc_rates.failure_rate
            lam_d = lc_rates.data_loss_rate
            new = list(delta_in_class(s, leader_class, k, (1, 1)))
            new[HL] = 0
            new[LC] = 0
            trans.append((tuple(new), lam))
            if lam_d > 0:
                new = list(delta_in_class(s, leader_class, k, (3, 1)))
                new[HL] = 0
                new[LC] = 0
                trans.append((tuple(new), lam_d))

        for rc in classes:
            c = rc.class_idx
            off = c * k
            nH, nF, nL, nD = s[off], s[off + 1], s[off + 2], s[off + 3]
            lam = rc.rates.failure_rate
            lam_d = rc.rates.data_loss_rate
            mu_rec = rc.rates.recovery_rate
            mu_sync = (
                rc.rates.sync_rate if rc.rates.sync_rate != float("inf") else 1e12
            )
            mu_replace = rc.rates.collapsed_replace_rate

            if nH > 0:
                trans.append((delta_in_class(s, c, k, (0, -1), (1, 1)), nH * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, -1), (3, 1)), nH * lam_d),
                    )
            if nF > 0:
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (2, 1)), nF * mu_rec),
                    )
                if not all_down and mu_replace > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (1, -1)), nF * mu_replace),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (3, 1)), nF * lam_d),
                    )
            if nL > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (2, -1)), nL * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (1, 1), (2, -1)), nL * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (3, 1)), nL * lam_d),
                    )
            if nD > 0 and mu_replace > 0:
                trans.append(
                    (delta_in_class(s, c, k, (0, 1), (3, -1)), nD * mu_replace),
                )

        if not has_leader and total_H >= q and mu_election > 0:
            for rc in classes:
                c = rc.class_idx
                nH_c = s[c * k]
                if nH_c > 0:
                    rate = mu_election * nH_c / total_H
                    new = list(delta_in_class(s, c, k, (0, -1)))
                    new[HL] = 1
                    new[LC] = c
                    trans.append((tuple(new), rate))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        has_leader = s[HL]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        return has_leader == 1 and total_H >= q

    initial = _initial_with_leader(classes, k, TRAILING)
    cost_fn = heterogeneous_cost_fn(
        classes, k, unbilled_indices=(3,),
        leader_has_leader_idx=HL, leader_class_idx=LC,
    )
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# NO_ORPHANS (k=6, trailing=2): {H, Fw, Fe_R, L, Dw, De_R} + (has_leader, leader_class)
#   Per-class indices: 0=H, 1=Fw, 2=Fe_R, 3=L, 4=Dw, 5=De_R
# ---------------------------------------------------------------------------


def _build_no_orphans(
    n: int, classes: list[RateClass], mu_election: float,
) -> MarkovModel:
    k = 6
    C = len(classes)
    TRAILING = 2
    HL = C * k
    LC = C * k + 1
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        has_leader = s[HL]
        leader_class = s[LC]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        total_L = total_per_node_state(s, 3, C, k)
        n_available = total_H + total_L
        all_down = n_available == 0

        if has_leader:
            lc_rates = classes[leader_class].rates
            lam = lc_rates.failure_rate
            lam_d = lc_rates.data_loss_rate
            new = list(delta_in_class(s, leader_class, k, (1, 1)))
            new[HL] = 0
            new[LC] = 0
            trans.append((tuple(new), lam))
            if lam_d > 0:
                new = list(delta_in_class(s, leader_class, k, (4, 1)))
                new[HL] = 0
                new[LC] = 0
                trans.append((tuple(new), lam_d))

        for rc in classes:
            c = rc.class_idx
            off = c * k
            nH = s[off]
            nFw = s[off + 1]
            nFeR = s[off + 2]
            nL = s[off + 3]
            nDw = s[off + 4]
            nDeR = s[off + 5]
            lam = rc.rates.failure_rate
            lam_d = rc.rates.data_loss_rate
            mu_rec = rc.rates.recovery_rate
            mu_sync = (
                rc.rates.sync_rate if rc.rates.sync_rate != float("inf") else 1e12
            )
            mu_timeout = rc.rates.timeout_rate
            mu_R = rc.rates.replace_pipeline_rate

            if nH > 0:
                trans.append((delta_in_class(s, c, k, (0, -1), (1, 1)), nH * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, -1), (4, 1)), nH * lam_d),
                    )
            if nFw > 0:
                if not all_down and mu_timeout > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (2, 1)), nFw * mu_timeout),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (3, 1)), nFw * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (4, 1)), nFw * lam_d),
                    )
            if nFeR > 0:
                if mu_R > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (2, -1)), nFeR * mu_R),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (3, 1)), nFeR * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (5, 1)), nFeR * lam_d),
                    )
            if nL > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (3, -1)), nL * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (1, 1), (3, -1)), nL * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (3, -1), (4, 1)), nL * lam_d),
                    )
            if nDw > 0 and mu_timeout > 0:
                trans.append(
                    (delta_in_class(s, c, k, (4, -1), (5, 1)), nDw * mu_timeout),
                )
            if nDeR > 0 and mu_R > 0:
                trans.append(
                    (delta_in_class(s, c, k, (0, 1), (5, -1)), nDeR * mu_R),
                )

        if not has_leader and total_H >= q and mu_election > 0:
            for rc in classes:
                c = rc.class_idx
                nH_c = s[c * k]
                if nH_c > 0:
                    rate = mu_election * nH_c / total_H
                    new = list(delta_in_class(s, c, k, (0, -1)))
                    new[HL] = 1
                    new[LC] = c
                    trans.append((tuple(new), rate))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        has_leader = s[HL]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        return has_leader == 1 and total_H >= q

    initial = _initial_with_leader(classes, k, TRAILING)
    cost_fn = heterogeneous_cost_fn(
        classes, k, unbilled_indices=(4,),
        leader_has_leader_idx=HL, leader_class_idx=LC,
    )
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# MERGED_PIPELINE (k=8, trailing=3): {H, H_R, Fw, Fe_R, L, L_R, Dw, De_R}
#   + (has_leader, leader_class, leader_orphan)
#   Per-class indices: 0=H, 1=H_R, 2=Fw, 3=Fe_R, 4=L, 5=L_R, 6=Dw, 7=De_R
# ---------------------------------------------------------------------------


def _build_merged_pipeline(
    n: int, classes: list[RateClass], mu_election: float,
) -> MarkovModel:
    k = 8
    C = len(classes)
    TRAILING = 3
    HL = C * k
    LC = C * k + 1
    LO = C * k + 2  # leader_orphan
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        has_leader = s[HL]
        leader_class = s[LC]
        leader_orphan = s[LO]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        total_HR = total_per_node_state(s, 1, C, k)
        total_L = total_per_node_state(s, 4, C, k)
        total_LR = total_per_node_state(s, 5, C, k)
        total_healthy = total_H + total_HR
        n_available = total_healthy + total_L + total_LR
        all_down = n_available == 0

        if has_leader and not leader_orphan:
            lc_rates = classes[leader_class].rates
            lam = lc_rates.failure_rate
            lam_d = lc_rates.data_loss_rate
            new = list(delta_in_class(s, leader_class, k, (2, 1)))
            new[HL] = 0
            new[LC] = 0
            new[LO] = 0
            trans.append((tuple(new), lam))
            if lam_d > 0:
                new = list(delta_in_class(s, leader_class, k, (6, 1)))
                new[HL] = 0
                new[LC] = 0
                new[LO] = 0
                trans.append((tuple(new), lam_d))

        if has_leader and leader_orphan:
            lc_rates = classes[leader_class].rates
            lam = lc_rates.failure_rate
            lam_d = lc_rates.data_loss_rate
            mu_R = lc_rates.replace_pipeline_rate
            # Leader with orphan fails -> Fe_R += 1, drop leader and orphan.
            new = list(delta_in_class(s, leader_class, k, (3, 1)))
            new[HL] = 0
            new[LC] = 0
            new[LO] = 0
            trans.append((tuple(new), lam))
            if lam_d > 0:
                new = list(delta_in_class(s, leader_class, k, (7, 1)))
                new[HL] = 0
                new[LC] = 0
                new[LO] = 0
                trans.append((tuple(new), lam_d))
            if mu_R > 0:
                # Orphan pipeline completes for the leader: flip sub-state
                # H*_R -> H*; discard the pending replacement.
                new = list(s)
                new[LO] = 0
                trans.append((tuple(new), mu_R))

        for rc in classes:
            c = rc.class_idx
            off = c * k
            nH = s[off]
            nHR = s[off + 1]
            nFw = s[off + 2]
            nFeR = s[off + 3]
            nL = s[off + 4]
            nLR = s[off + 5]
            nDw = s[off + 6]
            nDeR = s[off + 7]
            lam = rc.rates.failure_rate
            lam_d = rc.rates.data_loss_rate
            mu_rec = rc.rates.recovery_rate
            mu_sync = (
                rc.rates.sync_rate if rc.rates.sync_rate != float("inf") else 1e12
            )
            mu_timeout = rc.rates.timeout_rate
            mu_R = rc.rates.replace_pipeline_rate

            if nH > 0:
                trans.append((delta_in_class(s, c, k, (0, -1), (2, 1)), nH * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, -1), (6, 1)), nH * lam_d),
                    )
            if nHR > 0:
                trans.append((delta_in_class(s, c, k, (1, -1), (3, 1)), nHR * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (7, 1)), nHR * lam_d),
                    )
                if mu_R > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (1, -1)), nHR * mu_R),
                    )
            if nFw > 0:
                if not all_down and mu_timeout > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (3, 1)), nFw * mu_timeout),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (4, 1)), nFw * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (6, 1)), nFw * lam_d),
                    )
            if nFeR > 0:
                if mu_R > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (3, -1)), nFeR * mu_R),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (3, -1), (5, 1)), nFeR * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (3, -1), (7, 1)), nFeR * lam_d),
                    )
            if nL > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (4, -1)), nL * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (2, 1), (4, -1)), nL * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (4, -1), (6, 1)), nL * lam_d),
                    )
            if nLR > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, 1), (5, -1)), nLR * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (3, 1), (5, -1)), nLR * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (5, -1), (7, 1)), nLR * lam_d),
                    )
                if mu_R > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (4, 1), (5, -1)), nLR * mu_R),
                    )
            if nDw > 0 and mu_timeout > 0:
                trans.append(
                    (delta_in_class(s, c, k, (6, -1), (7, 1)), nDw * mu_timeout),
                )
            if nDeR > 0 and mu_R > 0:
                trans.append(
                    (delta_in_class(s, c, k, (0, 1), (7, -1)), nDeR * mu_R),
                )

        if not has_leader and total_healthy >= q and mu_election > 0:
            # Election: promote a class-c healthy node. Use H first, then H_R.
            # Rate is proportional to the class's contribution to total_healthy.
            for rc in classes:
                c = rc.class_idx
                nH_c = s[c * k]
                nHR_c = s[c * k + 1]
                contrib = nH_c + nHR_c
                if contrib == 0:
                    continue
                rate_total = mu_election * contrib / total_healthy
                if nH_c > 0:
                    # Fraction of rate going to H is nH_c / contrib.
                    r_h = rate_total * nH_c / contrib
                    new = list(delta_in_class(s, c, k, (0, -1)))
                    new[HL] = 1
                    new[LC] = c
                    new[LO] = 0
                    trans.append((tuple(new), r_h))
                if nHR_c > 0:
                    r_hr = rate_total * nHR_c / contrib
                    new = list(delta_in_class(s, c, k, (1, -1)))
                    new[HL] = 1
                    new[LC] = c
                    new[LO] = 1
                    trans.append((tuple(new), r_hr))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        has_leader = s[HL]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        total_HR = total_per_node_state(s, 1, C, k)
        return has_leader == 1 and (total_H + total_HR) >= q

    initial = _initial_with_leader(classes, k, TRAILING)
    cost_fn = heterogeneous_cost_fn(
        classes, k, unbilled_indices=(6,),
        leader_has_leader_idx=HL, leader_class_idx=LC,
    )
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# FULL (k=12, trailing=3): {H, H_P, H_S, Fw, Fe_P, Fe_S, L, L_P, L_S,
#                           Dw, De_P, De_S} + (has_leader, leader_class, leader_pipe)
#   Per-class indices: 0=H, 1=H_P, 2=H_S, 3=Fw, 4=Fe_P, 5=Fe_S,
#                      6=L, 7=L_P, 8=L_S, 9=Dw, 10=De_P, 11=De_S
#   leader_pipe ∈ {0: none, 1: P, 2: S}
# ---------------------------------------------------------------------------


def _build_full(
    n: int, classes: list[RateClass], mu_election: float,
) -> MarkovModel:
    k = 12
    C = len(classes)
    TRAILING = 3
    HL = C * k
    LC = C * k + 1
    LP = C * k + 2  # leader_pipe
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        has_leader = s[HL]
        leader_class = s[LC]
        leader_pipe = s[LP]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        total_HP = total_per_node_state(s, 1, C, k)
        total_HS = total_per_node_state(s, 2, C, k)
        total_L = total_per_node_state(s, 6, C, k)
        total_LP = total_per_node_state(s, 7, C, k)
        total_LS = total_per_node_state(s, 8, C, k)
        total_healthy = total_H + total_HP + total_HS
        n_available = total_healthy + total_L + total_LP + total_LS
        all_down = n_available == 0

        if has_leader:
            lc_rates = classes[leader_class].rates
            lam = lc_rates.failure_rate
            lam_d = lc_rates.data_loss_rate
            mu_spawn = lc_rates.spawn_rate
            mu_sync_lc = (
                lc_rates.sync_rate if lc_rates.sync_rate != float("inf") else 1e12
            )
            if leader_pipe == 0:
                new = list(delta_in_class(s, leader_class, k, (3, 1)))
                new[HL] = 0
                new[LC] = 0
                new[LP] = 0
                trans.append((tuple(new), lam))
                if lam_d > 0:
                    new = list(delta_in_class(s, leader_class, k, (9, 1)))
                    new[HL] = 0
                    new[LC] = 0
                    new[LP] = 0
                    trans.append((tuple(new), lam_d))
            elif leader_pipe == 1:
                new = list(delta_in_class(s, leader_class, k, (4, 1)))
                new[HL] = 0
                new[LC] = 0
                new[LP] = 0
                trans.append((tuple(new), lam))
                if lam_d > 0:
                    new = list(delta_in_class(s, leader_class, k, (10, 1)))
                    new[HL] = 0
                    new[LC] = 0
                    new[LP] = 0
                    trans.append((tuple(new), lam_d))
                if mu_spawn > 0:
                    new = list(s)
                    new[LP] = 2
                    trans.append((tuple(new), mu_spawn))
            elif leader_pipe == 2:
                new = list(delta_in_class(s, leader_class, k, (5, 1)))
                new[HL] = 0
                new[LC] = 0
                new[LP] = 0
                trans.append((tuple(new), lam))
                if lam_d > 0:
                    new = list(delta_in_class(s, leader_class, k, (11, 1)))
                    new[HL] = 0
                    new[LC] = 0
                    new[LP] = 0
                    trans.append((tuple(new), lam_d))
                if mu_sync_lc > 0:
                    # Leader's orphaned replacement finishes sync: H*_S -> H*;
                    # discard the pending replacement.
                    new = list(s)
                    new[LP] = 0
                    trans.append((tuple(new), mu_sync_lc))

        for rc in classes:
            c = rc.class_idx
            off = c * k
            nH = s[off]
            nHP = s[off + 1]
            nHS = s[off + 2]
            nFw = s[off + 3]
            nFeP = s[off + 4]
            nFeS = s[off + 5]
            nL = s[off + 6]
            nLP = s[off + 7]
            nLS = s[off + 8]
            nDw = s[off + 9]
            nDeP = s[off + 10]
            nDeS = s[off + 11]
            lam = rc.rates.failure_rate
            lam_d = rc.rates.data_loss_rate
            mu_rec = rc.rates.recovery_rate
            mu_sync = (
                rc.rates.sync_rate if rc.rates.sync_rate != float("inf") else 1e12
            )
            mu_spawn = rc.rates.spawn_rate
            mu_timeout = rc.rates.timeout_rate

            if nH > 0:
                trans.append((delta_in_class(s, c, k, (0, -1), (3, 1)), nH * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, -1), (9, 1)), nH * lam_d),
                    )
            if nHP > 0:
                trans.append((delta_in_class(s, c, k, (1, -1), (4, 1)), nHP * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (10, 1)), nHP * lam_d),
                    )
                if mu_spawn > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, -1), (2, 1)), nHP * mu_spawn),
                    )
            if nHS > 0:
                trans.append((delta_in_class(s, c, k, (2, -1), (5, 1)), nHS * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (2, -1), (11, 1)), nHS * lam_d),
                    )
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (2, -1)), nHS * mu_sync),
                    )
            if nFw > 0:
                if not all_down and mu_timeout > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (3, -1), (4, 1)), nFw * mu_timeout),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (3, -1), (6, 1)), nFw * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (3, -1), (9, 1)), nFw * lam_d),
                    )
            if nFeP > 0:
                if mu_spawn > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (4, -1), (5, 1)), nFeP * mu_spawn),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (4, -1), (7, 1)), nFeP * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (4, -1), (10, 1)), nFeP * lam_d),
                    )
            if nFeS > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (5, -1)), nFeS * mu_sync),
                    )
                if mu_rec > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (5, -1), (8, 1)), nFeS * mu_rec),
                    )
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (5, -1), (11, 1)), nFeS * lam_d),
                    )
            if nL > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (0, 1), (6, -1)), nL * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (3, 1), (6, -1)), nL * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (6, -1), (9, 1)), nL * lam_d),
                    )
            if nLP > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, 1), (7, -1)), nLP * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (4, 1), (7, -1)), nLP * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (7, -1), (10, 1)), nLP * lam_d),
                    )
                if mu_spawn > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (7, -1), (8, 1)), nLP * mu_spawn),
                    )
            if nLS > 0:
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (1, 1), (8, -1)), nLS * mu_sync),
                    )
                trans.append((delta_in_class(s, c, k, (5, 1), (8, -1)), nLS * lam))
                if lam_d > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (8, -1), (11, 1)), nLS * lam_d),
                    )
                if mu_sync > 0:
                    trans.append(
                        (delta_in_class(s, c, k, (6, 1), (8, -1)), nLS * mu_sync),
                    )
            if nDw > 0 and mu_timeout > 0:
                trans.append(
                    (delta_in_class(s, c, k, (9, -1), (10, 1)), nDw * mu_timeout),
                )
            if nDeP > 0 and mu_spawn > 0:
                trans.append(
                    (delta_in_class(s, c, k, (10, -1), (11, 1)), nDeP * mu_spawn),
                )
            if nDeS > 0 and mu_sync > 0:
                trans.append(
                    (delta_in_class(s, c, k, (0, 1), (11, -1)), nDeS * mu_sync),
                )

        if not has_leader and total_healthy >= q and mu_election > 0:
            # Election: promote from H, then H_P, then H_S per class.
            for rc in classes:
                c = rc.class_idx
                nH_c = s[c * k]
                nHP_c = s[c * k + 1]
                nHS_c = s[c * k + 2]
                contrib = nH_c + nHP_c + nHS_c
                if contrib == 0:
                    continue
                rate_total = mu_election * contrib / total_healthy
                if nH_c > 0:
                    r = rate_total * nH_c / contrib
                    new = list(delta_in_class(s, c, k, (0, -1)))
                    new[HL] = 1
                    new[LC] = c
                    new[LP] = 0
                    trans.append((tuple(new), r))
                if nHP_c > 0:
                    r = rate_total * nHP_c / contrib
                    new = list(delta_in_class(s, c, k, (1, -1)))
                    new[HL] = 1
                    new[LC] = c
                    new[LP] = 1
                    trans.append((tuple(new), r))
                if nHS_c > 0:
                    r = rate_total * nHS_c / contrib
                    new = list(delta_in_class(s, c, k, (2, -1)))
                    new[HL] = 1
                    new[LC] = c
                    new[LP] = 2
                    trans.append((tuple(new), r))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        has_leader = s[HL]
        total_H = total_per_node_state(s, 0, C, k) + has_leader
        total_HP = total_per_node_state(s, 1, C, k)
        total_HS = total_per_node_state(s, 2, C, k)
        return has_leader == 1 and (total_H + total_HP + total_HS) >= q

    initial = _initial_with_leader(classes, k, TRAILING)
    cost_fn = heterogeneous_cost_fn(
        classes, k, unbilled_indices=(9,),
        leader_has_leader_idx=HL, leader_class_idx=LC,
    )
    return build_model_bfs(initial, transitions, liveness, cost_fn)
