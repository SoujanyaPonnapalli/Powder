"""
Leaderless Markov chain builders at 5 quality levels.

Each builder generates a CTMC for an N-node replicated state machine
with majority quorum via BFS from the initial (all-healthy) state.
Only reachable states are included. Quality levels correspond to the
simplification steps in docs/markov_state_analysis.md.

Heterogeneity:
    Inputs are a ``list[NodeConfig]`` that is partitioned into rate
    classes (see ``common.group_configs_into_classes``). The state
    tuple is a concatenation of per-class count tuples; transitions
    are emitted per class using that class's rates; global predicates
    (``all_down``, quorum) sum across classes. Replacement preserves
    class membership -- a slot's physical rate profile is fixed.
    When every config collapses to a single class, the state layout
    matches the original homogeneous builder exactly.

Per-node states at each level (k columns per class):
    SIMPLIFIED (k=3):         H, F, D
    COLLAPSED_PIPELINE (k=4): H, F, L, D
    NO_ORPHANS (k=6):         H, Fw, Fe_R, L, Dw, De_R
    MERGED_PIPELINE (k=8):    H, H_R, Fw, Fe_R, L, L_R, Dw, De_R
    FULL (k=12):              H, H_P, H_S, Fw, Fe_P, Fe_S, L, L_P, L_S, Dw, De_P, De_S

Modeling note -- data-loss recovery in all-down states:
    When every node is unavailable, replacement timers for *transient*
    failures are cancelled (the cluster cannot observe liveness to trigger
    a replacement).  However, data-loss nodes are still allowed to proceed
    through the replacement pipeline even when all nodes are down.
    In reality, total data loss is unrecoverable, but this state is reached
    with negligible probability.  Allowing recovery here eliminates the
    absorbing all-data-loss state, which would otherwise cause steady-state
    availability to collapse to 0 and produce unusable results.
    The all-transient-failure state is *not* absorbing because each failed
    node can self-recover at rate mu_recovery without any healthy peers.
"""

from __future__ import annotations

import logging

from ..markov import MarkovModel
from ..scenario import QualityLevel
from ..simulation.node import NodeConfig
from ..simulation.protocol import Protocol, LeaderlessProtocol
from ..simulation.strategy import ClusterStrategy
from .common import (
    RateClass,
    build_model_bfs,
    delta_in_class,
    group_configs_into_classes,
    heterogeneous_cost_fn,
    make_initial_counts,
    majority,
    total_per_node_state,
)

_logger = logging.getLogger(__name__)


Transition = tuple[tuple[int, ...], float]


def build_leaderless_model(
    node_configs: list[NodeConfig],
    protocol: Protocol,
    strategy: ClusterStrategy,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> MarkovModel:
    """Build a leaderless CTMC at the requested quality level.

    Supports heterogeneous ``node_configs``: configs are partitioned
    into rate classes using the effective rates consumed by ``quality``.
    When every config maps to a single class the state space reduces to
    the homogeneous count layout.
    """
    n = len(node_configs)
    classes = group_configs_into_classes(node_configs, strategy, quality)

    up_to_date_quorum = True
    if isinstance(protocol, LeaderlessProtocol):
        up_to_date_quorum = protocol.up_to_date_quorum

    builders = {
        QualityLevel.SIMPLIFIED: _build_simplified,
        QualityLevel.COLLAPSED_PIPELINE: _build_collapsed_pipeline,
        QualityLevel.NO_ORPHANS: _build_no_orphans,
        QualityLevel.MERGED_PIPELINE: _build_merged_pipeline,
        QualityLevel.FULL: _build_full,
    }
    model = builders[quality](n, classes, up_to_date_quorum)
    _logger.debug(
        "leaderless/%s N=%d classes=%d (sizes=%s) states=%d",
        quality.name,
        n,
        len(classes),
        [c.size for c in classes],
        model.num_states,
    )
    return model


# ---------------------------------------------------------------------------
# SIMPLIFIED (k=3): {H, F, D}
#   Per-class indices: 0=H, 1=F, 2=D
# ---------------------------------------------------------------------------


def _build_simplified(
    n: int, classes: list[RateClass], up_to_date_quorum: bool,
) -> MarkovModel:
    k = 3
    C = len(classes)
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        total_H = total_per_node_state(s, 0, C, k)
        all_down = total_H == 0
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
        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        # SIMPLIFIED has no L state, so up_to_date_quorum is irrelevant.
        return total_per_node_state(s, 0, C, k) >= q

    initial = tuple(make_initial_counts(classes, k))
    cost_fn = heterogeneous_cost_fn(classes, k, unbilled_indices=(2,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# COLLAPSED_PIPELINE (k=4): {H, F, L, D}
#   Per-class indices: 0=H, 1=F, 2=L, 3=D
# ---------------------------------------------------------------------------


def _build_collapsed_pipeline(
    n: int, classes: list[RateClass], up_to_date_quorum: bool,
) -> MarkovModel:
    k = 4
    C = len(classes)
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        total_H = total_per_node_state(s, 0, C, k)
        total_L = total_per_node_state(s, 2, C, k)
        all_down = (total_H + total_L) == 0
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
        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        total_H = total_per_node_state(s, 0, C, k)
        if up_to_date_quorum:
            return total_H >= q
        total_L = total_per_node_state(s, 2, C, k)
        return (total_H + total_L) >= q

    initial = tuple(make_initial_counts(classes, k))
    cost_fn = heterogeneous_cost_fn(classes, k, unbilled_indices=(3,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# NO_ORPHANS (k=6): {H, Fw, Fe_R, L, Dw, De_R}
#   Per-class indices: 0=H, 1=Fw, 2=Fe_R, 3=L, 4=Dw, 5=De_R
# ---------------------------------------------------------------------------


def _build_no_orphans(
    n: int, classes: list[RateClass], up_to_date_quorum: bool,
) -> MarkovModel:
    k = 6
    C = len(classes)
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        total_H = total_per_node_state(s, 0, C, k)
        total_L = total_per_node_state(s, 3, C, k)
        all_down = (total_H + total_L) == 0
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
        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        total_H = total_per_node_state(s, 0, C, k)
        if up_to_date_quorum:
            return total_H >= q
        total_L = total_per_node_state(s, 3, C, k)
        return (total_H + total_L) >= q

    initial = tuple(make_initial_counts(classes, k))
    cost_fn = heterogeneous_cost_fn(classes, k, unbilled_indices=(4,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# MERGED_PIPELINE (k=8): {H, H_R, Fw, Fe_R, L, L_R, Dw, De_R}
#   Per-class indices: 0=H, 1=H_R, 2=Fw, 3=Fe_R, 4=L, 5=L_R, 6=Dw, 7=De_R
# ---------------------------------------------------------------------------


def _build_merged_pipeline(
    n: int, classes: list[RateClass], up_to_date_quorum: bool,
) -> MarkovModel:
    k = 8
    C = len(classes)
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        total_H = total_per_node_state(s, 0, C, k)
        total_HR = total_per_node_state(s, 1, C, k)
        total_L = total_per_node_state(s, 4, C, k)
        total_LR = total_per_node_state(s, 5, C, k)
        all_down = (total_H + total_HR + total_L + total_LR) == 0
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
        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        total_H = total_per_node_state(s, 0, C, k)
        total_HR = total_per_node_state(s, 1, C, k)
        if up_to_date_quorum:
            return (total_H + total_HR) >= q
        total_L = total_per_node_state(s, 4, C, k)
        total_LR = total_per_node_state(s, 5, C, k)
        return (total_H + total_HR + total_L + total_LR) >= q

    initial = tuple(make_initial_counts(classes, k))
    cost_fn = heterogeneous_cost_fn(classes, k, unbilled_indices=(6,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# FULL (k=12): {H, H_P, H_S, Fw, Fe_P, Fe_S, L, L_P, L_S, Dw, De_P, De_S}
#   Per-class indices: 0=H, 1=H_P, 2=H_S, 3=Fw, 4=Fe_P, 5=Fe_S,
#                      6=L, 7=L_P, 8=L_S, 9=Dw, 10=De_P, 11=De_S
# ---------------------------------------------------------------------------


def _build_full(
    n: int, classes: list[RateClass], up_to_date_quorum: bool,
) -> MarkovModel:
    k = 12
    C = len(classes)
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        trans: list[Transition] = []
        total_H = total_per_node_state(s, 0, C, k)
        total_HP = total_per_node_state(s, 1, C, k)
        total_HS = total_per_node_state(s, 2, C, k)
        total_L = total_per_node_state(s, 6, C, k)
        total_LP = total_per_node_state(s, 7, C, k)
        total_LS = total_per_node_state(s, 8, C, k)
        all_down = (
            total_H + total_HP + total_HS + total_L + total_LP + total_LS
        ) == 0
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
        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        total_H = total_per_node_state(s, 0, C, k)
        total_HP = total_per_node_state(s, 1, C, k)
        total_HS = total_per_node_state(s, 2, C, k)
        if up_to_date_quorum:
            return (total_H + total_HP + total_HS) >= q
        total_L = total_per_node_state(s, 6, C, k)
        total_LP = total_per_node_state(s, 7, C, k)
        total_LS = total_per_node_state(s, 8, C, k)
        return (
            total_H + total_HP + total_HS + total_L + total_LP + total_LS
        ) >= q

    initial = tuple(make_initial_counts(classes, k))
    cost_fn = heterogeneous_cost_fn(classes, k, unbilled_indices=(9,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)
