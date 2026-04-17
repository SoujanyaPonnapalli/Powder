"""
Raft Markov chain builders at 5 quality levels.

Each builder generates a homogeneous CTMC for an N-node Raft cluster.
The state space extends the leaderless model with a leader dimension:
at most one Healthy node is the designated leader, and the system
can only commit when a leader exists AND a majority quorum of healthy
nodes is available. Leader failure triggers an election; election
completes at rate mu_election only when quorum is available.

Quality levels correspond to docs/raft_markov_state_analysis.md.

Per-node states (non-leader / leader):
    SIMPLIFIED (k_nl=3, k_l=1):         {H, F, D} + {H*}
    COLLAPSED_PIPELINE (k_nl=4, k_l=1): {H, F, L, D} + {H*}
    NO_ORPHANS (k_nl=6, k_l=1):         {H, Fw, Fe_R, L, Dw, De_R} + {H*}
    MERGED_PIPELINE (k_nl=8, k_l=2):    {H, H_R, Fw, Fe_R, L, L_R, Dw, De_R} + {H*, H*_R}
    FULL (k_nl=12, k_l=3):              all 12 base + {H*, H*_P, H*_S}

State encoding: each state is a tuple of counts with a trailing
has_leader flag (0 or 1). For example, SIMPLIFIED states are
(nH, nF, nD, has_leader) where nH counts non-leader healthy nodes
and has_leader indicates whether one additional healthy node is leader.

Modeling note -- data-loss recovery in all-down states:
    See leaderless.py module docstring for the rationale.  The same
    approximation is applied here: data-loss replacement proceeds even
    when all nodes are unavailable, preventing the all-data-loss state
    from becoming absorbing.
"""

from __future__ import annotations

from ..markov import MarkovModel
from ..scenario import QualityLevel
from ..simulation.node import NodeConfig
from ..simulation.protocol import Protocol, RaftLikeProtocol
from ..simulation.strategy import ClusterStrategy
from .common import (
    ExtractedRates,
    build_model_bfs,
    extract_rates,
    homogeneous_cost_fn,
    majority,
)


def build_raft_model(
    node_configs: list[NodeConfig],
    protocol: Protocol,
    strategy: ClusterStrategy,
    quality: QualityLevel = QualityLevel.SIMPLIFIED,
) -> MarkovModel:
    """Build a Raft CTMC at the requested quality level.

    Uses the first node config for rates (homogeneous model).
    """
    n = len(node_configs)
    rates = extract_rates(node_configs[0], strategy)
    cost_per_second = node_configs[0].cost_per_hour / 3600.0

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
    return builders[quality](n, rates, mu_election, cost_per_second)


def _delta(state: tuple[int, ...], *changes: tuple[int, int]) -> tuple[int, ...]:
    """Return a new state tuple with the given (index, delta) shifts applied."""
    s = list(state)
    for idx, d in changes:
        s[idx] += d
    return tuple(s)


Transition = tuple[tuple[int, ...], float]


# ---------------------------------------------------------------------------
# SIMPLIFIED (k_nl=3, k_l=1): {H, F, D, has_leader}
#   Indices: 0=H (non-leader), 1=F, 2=D, 3=has_leader
#   Total healthy = H + has_leader
# ---------------------------------------------------------------------------


def _build_simplified(
    n: int, rates: ExtractedRates, mu_election: float, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_f_direct = rates.recovery_with_sync_rate
    mu_f_replace = rates.collapsed_replace_rate
    mu_d_replace = rates.collapsed_replace_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nF, nD, has_leader = s
        total_healthy = nH + has_leader
        trans: list[Transition] = []
        all_down = total_healthy == 0

        if has_leader:
            trans.append((_delta(s, (1, 1), (3, -1)), lam))
            if lam_d > 0:
                trans.append((_delta(s, (2, 1), (3, -1)), lam_d))

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

        if not has_leader and total_healthy >= q and mu_election > 0:
            trans.append((_delta(s, (0, -1), (3, 1)), mu_election))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, _nF, _nD, has_leader = s
        return has_leader == 1 and (nH + has_leader) >= q

    initial = (n - 1, 0, 0, 1)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(2,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# COLLAPSED_PIPELINE (k_nl=4, k_l=1): {H, F, L, D, has_leader}
#   Indices: 0=H, 1=F, 2=L, 3=D, 4=has_leader
# ---------------------------------------------------------------------------


def _build_collapsed_pipeline(
    n: int, rates: ExtractedRates, mu_election: float, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_replace = rates.collapsed_replace_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nF, nL, nD, has_leader = s
        total_healthy = nH + has_leader
        n_available = total_healthy + nL
        all_down = n_available == 0
        trans: list[Transition] = []

        if has_leader:
            trans.append((_delta(s, (1, 1), (4, -1)), lam))
            if lam_d > 0:
                trans.append((_delta(s, (3, 1), (4, -1)), lam_d))

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

        if not has_leader and total_healthy >= q and mu_election > 0:
            trans.append((_delta(s, (0, -1), (4, 1)), mu_election))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, _nF, _nL, _nD, has_leader = s
        return has_leader == 1 and (nH + has_leader) >= q

    initial = (n - 1, 0, 0, 0, 1)
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(3,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# NO_ORPHANS (k_nl=6, k_l=1): {H, Fw, Fe_R, L, Dw, De_R, has_leader}
#   Indices: 0=H, 1=Fw, 2=Fe_R, 3=L, 4=Dw, 5=De_R, 6=has_leader
# ---------------------------------------------------------------------------


def _build_no_orphans(
    n: int, rates: ExtractedRates, mu_election: float, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_timeout = rates.timeout_rate
    mu_R = rates.replace_pipeline_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nFw, nFeR, nL, nDw, nDeR, has_leader = s
        total_healthy = nH + has_leader
        n_available = total_healthy + nL
        all_down = n_available == 0
        trans: list[Transition] = []

        if has_leader:
            trans.append((_delta(s, (1, 1), (6, -1)), lam))
            if lam_d > 0:
                trans.append((_delta(s, (4, 1), (6, -1)), lam_d))

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

        if not has_leader and total_healthy >= q and mu_election > 0:
            trans.append((_delta(s, (0, -1), (6, 1)), mu_election))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH = s[0]
        has_leader = s[6]
        return has_leader == 1 and (nH + has_leader) >= q

    initial = tuple([n - 1] + [0] * 5 + [1])
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(4,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# MERGED_PIPELINE (k_nl=8, k_l=2):
#   Non-leader: {H, H_R, Fw, Fe_R, L, L_R, Dw, De_R}
#   Leader: {H*, H*_R}
#   Indices: 0=H, 1=H_R, 2=Fw, 3=Fe_R, 4=L, 5=L_R, 6=Dw, 7=De_R,
#            8=has_leader, 9=leader_has_orphan
# ---------------------------------------------------------------------------


def _build_merged_pipeline(
    n: int, rates: ExtractedRates, mu_election: float, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_timeout = rates.timeout_rate
    mu_R = rates.replace_pipeline_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nHR, nFw, nFeR, nL, nLR, nDw, nDeR, has_leader, leader_orphan = s
        total_healthy = nH + nHR + has_leader
        n_available = total_healthy + nL + nLR
        all_down = n_available == 0
        trans: list[Transition] = []

        if has_leader and not leader_orphan:
            trans.append((_delta(s, (2, 1), (8, -1)), lam))
            if lam_d > 0:
                trans.append((_delta(s, (6, 1), (8, -1)), lam_d))

        if has_leader and leader_orphan:
            trans.append((_delta(s, (3, 1), (8, -1), (9, -1)), lam))
            if lam_d > 0:
                trans.append((_delta(s, (7, 1), (8, -1), (9, -1)), lam_d))
            if mu_R > 0:
                # Orphaned replacement pipeline completes for the current leader:
                # the leader's sub-state flips H*_R -> H*. The pending replacement
                # is discarded (the node is already healthy and serving as leader),
                # so no new slot joins the cluster.
                trans.append((_delta(s, (9, -1)), mu_R))

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

        if not has_leader and total_healthy >= q and mu_election > 0:
            if nH > 0:
                trans.append((_delta(s, (0, -1), (8, 1)), mu_election))
            elif nHR > 0:
                trans.append((_delta(s, (1, -1), (8, 1), (9, 1)), mu_election))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, nHR = s[0], s[1]
        has_leader = s[8]
        return has_leader == 1 and (nH + nHR + has_leader) >= q

    initial = tuple([n - 1] + [0] * 7 + [1, 0])
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(6,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)


# ---------------------------------------------------------------------------
# FULL (k_nl=12, k_l=3):
#   Non-leader: {H, H_P, H_S, Fw, Fe_P, Fe_S, L, L_P, L_S, Dw, De_P, De_S}
#   Leader: {H*, H*_P, H*_S}
#   Indices: 0=H, 1=H_P, 2=H_S, 3=Fw, 4=Fe_P, 5=Fe_S,
#            6=L, 7=L_P, 8=L_S, 9=Dw, 10=De_P, 11=De_S,
#            12=has_leader, 13=leader_pipeline (0=none, 1=P, 2=S)
# ---------------------------------------------------------------------------


def _build_full(
    n: int, rates: ExtractedRates, mu_election: float, cost_per_second: float,
) -> MarkovModel:
    lam = rates.failure_rate
    lam_d = rates.data_loss_rate
    mu_rec = rates.recovery_rate
    mu_sync = rates.sync_rate if rates.sync_rate != float("inf") else 1e12
    mu_spawn = rates.spawn_rate
    mu_timeout = rates.timeout_rate
    q = majority(n)

    def transitions(s: tuple[int, ...]) -> list[Transition]:
        nH, nHP, nHS, nFw, nFeP, nFeS, nL, nLP, nLS, nDw, nDeP, nDeS, \
            has_leader, leader_pipe = s
        total_healthy = nH + nHP + nHS + has_leader
        n_available = total_healthy + nL + nLP + nLS
        all_down = n_available == 0
        trans: list[Transition] = []

        if has_leader:
            if leader_pipe == 0:
                trans.append((_delta(s, (3, 1), (12, -1)), lam))
                if lam_d > 0:
                    trans.append((_delta(s, (9, 1), (12, -1)), lam_d))
            elif leader_pipe == 1:
                trans.append((_delta(s, (4, 1), (12, -1), (13, -1)), lam))
                if lam_d > 0:
                    trans.append((_delta(s, (10, 1), (12, -1), (13, -1)), lam_d))
                if mu_spawn > 0:
                    trans.append((_delta(s, (13, 1)), mu_spawn))
            elif leader_pipe == 2:
                trans.append((_delta(s, (5, 1), (12, -1), (13, -2)), lam))
                if lam_d > 0:
                    trans.append((_delta(s, (11, 1), (12, -1), (13, -2)), lam_d))
                if mu_sync > 0:
                    # Leader's orphaned replacement finishes its sync phase:
                    # the leader's sub-state flips H*_S -> H*. The pending
                    # replacement is discarded (the leader is healthy); no new
                    # slot joins the cluster.
                    trans.append((_delta(s, (13, -2)), mu_sync))

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

        if not has_leader and total_healthy >= q and mu_election > 0:
            if nH > 0:
                trans.append((_delta(s, (0, -1), (12, 1)), mu_election))
            elif nHP > 0:
                trans.append((_delta(s, (1, -1), (12, 1), (13, 1)), mu_election))
            elif nHS > 0:
                trans.append((_delta(s, (2, -1), (12, 1), (13, 2)), mu_election))

        return trans

    def liveness(s: tuple[int, ...]) -> bool:
        nH, nHP, nHS = s[0], s[1], s[2]
        has_leader = s[12]
        return has_leader == 1 and (nH + nHP + nHS + has_leader) >= q

    initial = tuple([n - 1] + [0] * 11 + [1, 0])
    cost_fn = homogeneous_cost_fn(cost_per_second, n, unbilled_indices=(9,))
    return build_model_bfs(initial, transitions, liveness, cost_fn)
