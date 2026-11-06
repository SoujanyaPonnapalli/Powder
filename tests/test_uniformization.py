"""Uniformization time-averaged availability agrees with the augmented-expm time average."""

import numpy as np

from powder.markov_builders import build_raft_model
from powder.markov_solver import steady_state, time_averaged_availability_uniformization, time_averaged_distribution
from powder.scenario import QualityLevel
from powder.simulation import Exponential, NodeConfig, NodeReplacementStrategy, RaftLikeProtocol


def _configs():
    out = []
    for f, r, d in ((1 / 86400, 1 / 1200, 1 / (365 * 86400)), (3 / 86400, 1 / 1800, 1 / (30 * 86400)),
                    (1 / (5 * 86400), 1 / 600, 1 / 86400)):
        out.append(NodeConfig(region="r", cost_per_hour=1.0, failure_dist=Exponential(f), recovery_dist=Exponential(r),
                              data_loss_dist=Exponential(d), log_replay_rate_dist=Exponential(1.0),
                              snapshot_download_time_dist=Exponential(1 / 10), spawn_dist=Exponential(1 / 120)))
    return out


def test_uniformization_matches_expm():
    m = build_raft_model(_configs(), RaftLikeProtocol(election_time_dist=Exponential(0.2)),
                         NodeReplacementStrategy(failure_timeout=600.0), QualityLevel.NO_ORPHANS)
    for horizon in (600.0, 6 * 3600.0):
        ref = float(time_averaged_distribution(m, horizon)[m.live_mask].sum())
        got, info = time_averaged_availability_uniformization(m, horizon)
        assert info["iterations"] > 0
        assert abs((1 - got) - (1 - ref)) <= 1e-7 * (1 - ref) + 1e-14
    assert np.isclose(time_averaged_availability_uniformization(m, 60.0)[0],
                      float(time_averaged_distribution(m, 60.0)[m.live_mask].sum()), rtol=0, atol=1e-12)


def test_pinned_steady_state_matches_dense_row():
    m = build_raft_model(_configs(), RaftLikeProtocol(election_time_dist=Exponential(0.2)),
                         NodeReplacementStrategy(failure_timeout=600.0), QualityLevel.NO_ORPHANS)
    a = steady_state(m, backend="scipy")
    b = steady_state(m, backend="scipy", normalization="pin")
    assert np.allclose(a, b, rtol=1e-9, atol=1e-15)
