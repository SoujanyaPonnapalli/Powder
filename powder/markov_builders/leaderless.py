"""
Leaderless Markov chain builders at 5 quality levels.

Each builder generates a homogeneous CTMC for an N-node replicated state
machine with majority quorum via BFS from the initial (all-healthy) state.
Only reachable states are included. Quality levels correspond to the
simplification steps in docs/markov_state_analysis.md.

Per-node states at each level:
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

from ..markov import MarkovModel
from ..scenario import QualityLevel
from ..simulation.node import NodeConfig
from ..simulation.protocol import Protocol, LeaderlessProtocol
from ..simulation.strategy import ClusterStrategy
from .common import (
    ExtractedRates,
    build_model_bfs,
    extract_rates,
    homogeneous_cost_fn,
    majority,
)


def build_leaderless_model(
    node_configs: list[NodeConfig],
    protocol: Protocol,
    strategy: ClusterStrategy,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> MarkovModel:
    """Build a leaderless CTMC at the requested quality level.

    Uses the first node config for rates (homogeneous model).
    """
    n = len(node_configs)
    rates = extract_rates(node_configs[0], strategy)
    cost_per_second = node_configs[0].cost_per_hour / 3600.0

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
    return builders[quality](n, rates, up_to_date_quorum, cost_per_second)


def _delta(state: tuple[int, ...], *changes: tuple[int, int]) -> tuple[int, ...]:
    """Return a new state tuple with the given (index, delta) shifts applied."""
    s = list(state)
    for idx, d in changes:
        s[idx] += d
    return tuple(s)


Transition = tuple[tuple[int, ...], float]


# ---------------------------------------------------------------------------
# SIMPLIFIED (k=3): {H, F, D}
#   Indices: 0=H, 1=F, 2=D
# ---------------------------------------------------------------------------


def _build_simplified(
    n: int, rates: ExtractedRates, up_to_date_quorum: bool, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_f_direct = rates.recovery_with_sync_rate
    mu_f_replace = rates.collapsed_replace_rate
    mu_d_replace = rates.collapsed_replace_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nF, nD = s
        trans: list[Transition] = []
        all_down = nH == 0

        if nH > 0:
            trans.append((_delta(s, (0, -1), (1, 1)), nH * lam))
            if lam_d > 0:
                trans.append((_delta(s, (0, -1), (2, 1)), nH * lam_d))
        if nF > 0:
            rate = nF * mu_f_direct
            if not all_down and mu_f_replace > 0:
                rate += nF * mu_f_replace
            if rate > 0:
                trans.append((_delta(s, (0, 1), (1, -1)), rate))
            if lam_d > 0:
                trans.append((_delta(s, (1, -1), (2, 1)), nF * lam_d))
        if nD > 0 and mu_d_replace > 0:
            trans.append((_delta(s, (0, 1), (2, -1)), nD * mu_d_replace))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        return s[0] >= q

    initial = tuple([n] + [0] * 2)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(2,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# COLLAPSED_PIPELINE (k=4): {H, F, L, D}
#   Indices: 0=H, 1=F, 2=L, 3=D
# ---------------------------------------------------------------------------


def _build_collapsed_pipeline(
    n: int, rates: ExtractedRates, up_to_date_quorum: bool, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_replace = rates.collapsed_replace_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nF, nL, nD = s
        trans: list[Transition] = []
        all_down = (nH + nL) == 0

        if nH > 0:
            trans.append((_delta(s, (0, -1), (1, 1)), nH * lam))
            if lam_d > 0:
                trans.append((_delta(s, (0, -1), (3, 1)), nH * lam_d))
        if nF > 0:
            if mu_rec > 0:
                trans.append((_delta(s, (1, -1), (2, 1)), nF * mu_rec))
            if not all_down and mu_replace > 0:
                trans.append((_delta(s, (0, 1), (1, -1)), nF * mu_replace))
            if lam_d > 0:
                trans.append((_delta(s, (1, -1), (3, 1)), nF * lam_d))
        if nL > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (0, 1), (2, -1)), nL * mu_sync))
            trans.append((_delta(s, (1, 1), (2, -1)), nL * lam))
            if lam_d > 0:
                trans.append((_delta(s, (2, -1), (3, 1)), nL * lam_d))
        if nD > 0 and mu_replace > 0:
            trans.append((_delta(s, (0, 1), (3, -1)), nD * mu_replace))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, _nF, nL, _nD = s
        if up_to_date_quorum:
            return nH >= q
        return (nH + nL) >= q

    initial = tuple([n] + [0] * 3)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(3,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# NO_ORPHANS (k=6): {H, Fw, Fe_R, L, Dw, De_R}
#   Indices: 0=H, 1=Fw, 2=Fe_R, 3=L, 4=Dw, 5=De_R
# ---------------------------------------------------------------------------


def _build_no_orphans(
    n: int, rates: ExtractedRates, up_to_date_quorum: bool, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_timeout = rates.timeout_rate
    mu_R = rates.replace_pipeline_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nFw, nFeR, nL, nDw, nDeR = s
        trans: list[Transition] = []
        n_available = nH + nL
        all_down = n_available == 0

        if nH > 0:
            trans.append((_delta(s, (0, -1), (1, 1)), nH * lam))
            if lam_d > 0:
                trans.append((_delta(s, (0, -1), (4, 1)), nH * lam_d))
        if nFw > 0:
            if not all_down and mu_timeout > 0:
                trans.append((_delta(s, (1, -1), (2, 1)), nFw * mu_timeout))
            if mu_rec > 0:
                trans.append((_delta(s, (1, -1), (3, 1)), nFw * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (1, -1), (4, 1)), nFw * lam_d))
        if nFeR > 0:
            if mu_R > 0:
                trans.append((_delta(s, (0, 1), (2, -1)), nFeR * mu_R))
            if mu_rec > 0:
                trans.append((_delta(s, (2, -1), (3, 1)), nFeR * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (2, -1), (5, 1)), nFeR * lam_d))
        if nL > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (0, 1), (3, -1)), nL * mu_sync))
            trans.append((_delta(s, (1, 1), (3, -1)), nL * lam))
            if lam_d > 0:
                trans.append((_delta(s, (3, -1), (4, 1)), nL * lam_d))
        if nDw > 0 and mu_timeout > 0:
            trans.append((_delta(s, (4, -1), (5, 1)), nDw * mu_timeout))
        if nDeR > 0 and mu_R > 0:
            trans.append((_delta(s, (0, 1), (5, -1)), nDeR * mu_R))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, _nFw, _nFeR, nL, _nDw, _nDeR = s
        if up_to_date_quorum:
            return nH >= q
        return (nH + nL) >= q

    initial = tuple([n] + [0] * 5)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(4,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# MERGED_PIPELINE (k=8): {H, H_R, Fw, Fe_R, L, L_R, Dw, De_R}
#   Indices: 0=H, 1=H_R, 2=Fw, 3=Fe_R, 4=L, 5=L_R, 6=Dw, 7=De_R
# ---------------------------------------------------------------------------


def _build_merged_pipeline(
    n: int, rates: ExtractedRates, up_to_date_quorum: bool, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_timeout = rates.timeout_rate
    mu_R = rates.replace_pipeline_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nHR, nFw, nFeR, nL, nLR, nDw, nDeR = s
        trans: list[Transition] = []
        n_available = nH + nHR + nL + nLR
        all_down = n_available == 0

        if nH > 0:
            trans.append((_delta(s, (0, -1), (2, 1)), nH * lam))
            if lam_d > 0:
                trans.append((_delta(s, (0, -1), (6, 1)), nH * lam_d))

        if nHR > 0:
            trans.append((_delta(s, (1, -1), (3, 1)), nHR * lam))
            if lam_d > 0:
                trans.append((_delta(s, (1, -1), (7, 1)), nHR * lam_d))
            if mu_R > 0:
                trans.append((_delta(s, (0, 1), (1, -1)), nHR * mu_R))

        if nFw > 0:
            if not all_down and mu_timeout > 0:
                trans.append((_delta(s, (2, -1), (3, 1)), nFw * mu_timeout))
            if mu_rec > 0:
                trans.append((_delta(s, (2, -1), (4, 1)), nFw * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (2, -1), (6, 1)), nFw * lam_d))

        if nFeR > 0:
            if mu_R > 0:
                trans.append((_delta(s, (0, 1), (3, -1)), nFeR * mu_R))
            if mu_rec > 0:
                trans.append((_delta(s, (3, -1), (5, 1)), nFeR * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (3, -1), (7, 1)), nFeR * lam_d))

        if nL > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (0, 1), (4, -1)), nL * mu_sync))
            trans.append((_delta(s, (2, 1), (4, -1)), nL * lam))
            if lam_d > 0:
                trans.append((_delta(s, (4, -1), (6, 1)), nL * lam_d))

        if nLR > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (1, 1), (5, -1)), nLR * mu_sync))
            trans.append((_delta(s, (3, 1), (5, -1)), nLR * lam))
            if lam_d > 0:
                trans.append((_delta(s, (5, -1), (7, 1)), nLR * lam_d))
            if mu_R > 0:
                trans.append((_delta(s, (4, 1), (5, -1)), nLR * mu_R))

        if nDw > 0 and mu_timeout > 0:
            trans.append((_delta(s, (6, -1), (7, 1)), nDw * mu_timeout))

        if nDeR > 0 and mu_R > 0:
            trans.append((_delta(s, (0, 1), (7, -1)), nDeR * mu_R))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, nHR, _nFw, _nFeR, nL, nLR, _nDw, _nDeR = s
        if up_to_date_quorum:
            return (nH + nHR) >= q
        return (nH + nHR + nL + nLR) >= q

    initial = tuple([n] + [0] * 7)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(6,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# FULL (k=12): {H, H_P, H_S, Fw, Fe_P, Fe_S, L, L_P, L_S, Dw, De_P, De_S}
#   Indices: 0=H, 1=H_P, 2=H_S, 3=Fw, 4=Fe_P, 5=Fe_S,
#            6=L,  7=L_P, 8=L_S, 9=Dw, 10=De_P, 11=De_S
# ---------------------------------------------------------------------------


def _build_full(
    n: int, rates: ExtractedRates, up_to_date_quorum: bool, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_spawn = rates.spawn_rate
    mu_timeout = rates.timeout_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nHP, nHS, nFw, nFeP, nFeS, nL, nLP, nLS, nDw, nDeP, nDeS = s
        trans: list[Transition] = []
        n_available = nH + nHP + nHS + nL + nLP + nLS
        all_down = n_available == 0

        if nH > 0:
            trans.append((_delta(s, (0, -1), (3, 1)), nH * lam))
            if lam_d > 0:
                trans.append((_delta(s, (0, -1), (9, 1)), nH * lam_d))

        if nHP > 0:
            trans.append((_delta(s, (1, -1), (4, 1)), nHP * lam))
            if lam_d > 0:
                trans.append((_delta(s, (1, -1), (10, 1)), nHP * lam_d))
            if mu_spawn > 0:
                trans.append((_delta(s, (1, -1), (2, 1)), nHP * mu_spawn))

        if nHS > 0:
            trans.append((_delta(s, (2, -1), (5, 1)), nHS * lam))
            if lam_d > 0:
                trans.append((_delta(s, (2, -1), (11, 1)), nHS * lam_d))
            if mu_sync > 0:
                trans.append((_delta(s, (0, 1), (2, -1)), nHS * mu_sync))

        if nFw > 0:
            if not all_down and mu_timeout > 0:
                trans.append((_delta(s, (3, -1), (4, 1)), nFw * mu_timeout))
            if mu_rec > 0:
                trans.append((_delta(s, (3, -1), (6, 1)), nFw * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (3, -1), (9, 1)), nFw * lam_d))

        if nFeP > 0:
            if mu_spawn > 0:
                trans.append((_delta(s, (4, -1), (5, 1)), nFeP * mu_spawn))
            if mu_rec > 0:
                trans.append((_delta(s, (4, -1), (7, 1)), nFeP * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (4, -1), (10, 1)), nFeP * lam_d))

        if nFeS > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (0, 1), (5, -1)), nFeS * mu_sync))
            if mu_rec > 0:
                trans.append((_delta(s, (5, -1), (8, 1)), nFeS * mu_rec))
            if lam_d > 0:
                trans.append((_delta(s, (5, -1), (11, 1)), nFeS * lam_d))

        if nL > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (0, 1), (6, -1)), nL * mu_sync))
            trans.append((_delta(s, (3, 1), (6, -1)), nL * lam))
            if lam_d > 0:
                trans.append((_delta(s, (6, -1), (9, 1)), nL * lam_d))

        if nLP > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (1, 1), (7, -1)), nLP * mu_sync))
            trans.append((_delta(s, (4, 1), (7, -1)), nLP * lam))
            if lam_d > 0:
                trans.append((_delta(s, (7, -1), (10, 1)), nLP * lam_d))
            if mu_spawn > 0:
                trans.append((_delta(s, (7, -1), (8, 1)), nLP * mu_spawn))

        if nLS > 0:
            if mu_sync > 0:
                trans.append((_delta(s, (1, 1), (8, -1)), nLS * mu_sync))
            trans.append((_delta(s, (5, 1), (8, -1)), nLS * lam))
            if lam_d > 0:
                trans.append((_delta(s, (8, -1), (11, 1)), nLS * lam_d))
            if mu_sync > 0:
                trans.append((_delta(s, (6, 1), (8, -1)), nLS * mu_sync))

        if nDw > 0 and mu_timeout > 0:
            trans.append((_delta(s, (9, -1), (10, 1)), nDw * mu_timeout))

        if nDeP > 0 and mu_spawn > 0:
            trans.append((_delta(s, (10, -1), (11, 1)), nDeP * mu_spawn))

        if nDeS > 0 and mu_sync > 0:
            trans.append((_delta(s, (0, 1), (11, -1)), nDeS * mu_sync))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, nHP, nHS = s[0], s[1], s[2]
        nL, nLP, nLS = s[6], s[7], s[8]
        if up_to_date_quorum:
            return (nH + nHP + nHS) >= q
        return (nH + nHP + nHS + nL + nLP + nLS) >= q

    initial = tuple([n] + [0] * 11)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(9,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)
