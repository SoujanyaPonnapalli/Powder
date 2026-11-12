"""Optional single-RSM outage timing crosses the Python/Rust job boundary."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from powder.mc_backend import (
    MonteCarloConfig, run_monte_carlo, run_monte_carlo_summary,
)
from powder.mc_jobs import build_job
from powder.simulation import (
    ClusterState, Constant, LeaderlessProtocol, NetworkState, NodeConfig,
    NodeState, NoOpStrategy,
)


def _scenario():
    config = NodeConfig(
        region="r", cost_per_hour=0.0,
        failure_dist=Constant(10.0), recovery_dist=Constant(3.0),
        data_loss_dist=Constant(1e9), log_replay_rate_dist=Constant(100.0),
        snapshot_download_time_dist=Constant(0.0), spawn_dist=Constant(0.0),
    )
    cluster = ClusterState(nodes={
        f"n{i}": NodeState(f"n{i}", config) for i in range(3)
    }, network=NetworkState(), target_cluster_size=3)
    return cluster, NoOpStrategy(), LeaderlessProtocol()


def test_threshold_job_is_opt_in_and_validated():
    cluster, strategy, protocol = _scenario()
    kwargs = dict(num_simulations=2, max_time=38.0, stop_on_data_loss=False,
                  base_seed=7)
    job = build_job(cluster, strategy, protocol, **kwargs)
    assert "episode_threshold_seconds" not in job["run"]
    assert build_job(cluster, strategy, protocol, **kwargs,
                     episode_threshold_seconds=0.0)["run"]["episode_threshold_seconds"] == 0.0
    for value in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="episode_threshold_seconds"):
            build_job(cluster, strategy, protocol, **kwargs,
                      episode_threshold_seconds=value)
        with pytest.raises(ValueError, match="episode_threshold_seconds"):
            MonteCarloConfig(2, max_time=38, stop_on_data_loss=False,
                             episode_threshold_seconds=value)
    with pytest.raises(ValueError, match="stop_on_data_loss"):
        build_job(cluster, strategy, protocol, **(kwargs | {"stop_on_data_loss": True}),
                  episode_threshold_seconds=5.0)


def test_python_runner_and_summary_only_report_measured_tail(monkeypatch):
    binary = Path(__file__).resolve().parents[1] / "rust/target/debug/powder-mc"
    if not binary.is_file():
        pytest.skip("build the debug powder-mc binary for this integration test")
    monkeypatch.setenv("POWDER_MC_BINARY", str(binary))
    cluster, strategy, protocol = _scenario()
    kwargs = dict(num_simulations=2, max_time=38.0, stop_on_data_loss=False, seed=7)
    baseline = run_monte_carlo(cluster, strategy, protocol, **kwargs)
    assert baseline.episode_samples is None
    measured = run_monte_carlo(cluster, strategy, protocol, **kwargs,
                               episode_threshold_seconds=3.0)
    assert measured.availability_samples == baseline.availability_samples
    assert measured.episode_samples is not None
    for sample in measured.episode_samples:
        assert sample["threshold_seconds"] == 3.0
        assert sample["count_above_threshold"] == 0  # strict > T
        assert sample["total_unavailable_seconds"] == 8.0
        assert sample["max_duration_seconds"] == 3.0
        assert sample["post_threshold_seconds"] == 0.0
        assert sample["end_censored_duration_seconds"] == 2.0
    zero = run_monte_carlo(cluster, strategy, protocol, **kwargs,
                           episode_threshold_seconds=0.0)
    for sample in zero.episode_samples:
        assert sample["count_above_threshold"] == 3
        assert sample["post_threshold_seconds"] == 8.0
        assert sample["long_episode_seconds"] == 8.0
    summary = run_monte_carlo_summary(cluster, strategy, protocol, **kwargs,
                                      episode_threshold_seconds=0.0)["episode_summary"]
    assert summary["post_threshold_seconds_mean"] == 8.0
    assert summary["end_censored_fraction"] == 1.0


def test_old_engine_reports_rebuild_instruction(monkeypatch):
    import powder.mc_backend as backend

    monkeypatch.setattr(backend, "ensure_binary", lambda: Path("/unused/powder-mc"))
    monkeypatch.setattr(backend.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"summary": {"availability_mean": 1.0}}), stderr=""))
    cluster, strategy, protocol = _scenario()
    with pytest.raises(RuntimeError, match="rebuild it"):
        run_monte_carlo_summary(cluster, strategy, protocol, num_simulations=1, max_time=38.0,
                                stop_on_data_loss=False, episode_threshold_seconds=3.0)
