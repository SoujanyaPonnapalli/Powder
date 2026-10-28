"""AZ outages affect one logical zone, while region outages still span zones."""

from dataclasses import replace

import pytest

from powder.mc_backend import run_monte_carlo
from powder.mc_jobs import EngineNotFound, binary_path, network_config_spec, node_config_spec
from powder.simulation import (ClusterState, Constant, LeaderlessProtocol, NetworkConfig,
                               NetworkState, NodeConfig, NodeState, NoOpStrategy)


def _cfg(region, az):
    return NodeConfig(region=region, az=az, cost_per_hour=1,
                      failure_dist=Constant(1e9), recovery_dist=Constant(1),
                      data_loss_dist=Constant(1e9), log_replay_rate_dist=Constant(100),
                      snapshot_download_time_dist=Constant(0), spawn_dist=Constant(0))


def test_az_and_region_outages_have_distinct_scopes():
    configs = [_cfg("r1", "r1:az1"), _cfg("r1", "r1:az2"), _cfg("r2", "r2:az1")]
    cluster = ClusterState(nodes={f"n{i}": NodeState(f"n{i}", cfg) for i, cfg in enumerate(configs)},
                           network=NetworkState(), target_cluster_size=3)
    assert cluster.num_available() == 3
    cluster.network.add_outage("r1:az1")
    assert cluster.num_available() == 2
    cluster.network.add_outage("r1")
    assert cluster.num_available() == 1
    cluster.network.remove_outage("r1:az1")
    assert cluster.num_available() == 1
    cluster.network.remove_outage("r1")
    assert cluster.num_available() == 3


def test_optional_az_fields_round_trip_into_rust_job():
    net = NetworkConfig(outage_dist=Constant(1e9), outage_duration_dist=Constant(10),
                        regions=["r1", "r2"], azs=["r1:az1", "r1:az2"],
                        per_az={"r1:az1": (Constant(2), Constant(5))})
    spec = network_config_spec(net)
    assert spec["azs"] == ["r1:az1", "r1:az2"]
    assert spec["per_az"]["r1:az1"]["outage_duration_dist"]["value"] == 5
    assert node_config_spec(_cfg("r1", "r1:az1"))["az"] == "r1:az1"
    assert "az" not in node_config_spec(replace(_cfg("r1", "r1:az1"), az=None))


def test_rust_az_outage_leaves_quorum_but_region_outage_does_not():
    try:
        binary_path()
    except EngineNotFound:
        pytest.skip("powder-mc release binary unavailable")
    cfg = _cfg("r1", "r1:az1")
    cluster = ClusterState(nodes={
        "a": NodeState("a", cfg),
        "b": NodeState("b", replace(cfg, az="r1:az2")),
        "c": NodeState("c", replace(cfg, region="r2", az="r2:az1")),
    }, network=NetworkState(), target_cluster_size=3)
    az = NetworkConfig(Constant(1e9), Constant(1000), regions=["r1", "r2"],
                       azs=["r1:az1"], per_az={"r1:az1": (Constant(1), None)})
    region = NetworkConfig(Constant(1e9), Constant(1000), regions=["r1", "r2"],
                           per_region={"r1": (Constant(1), None)})
    def availability(net):
        return run_monte_carlo(cluster, NoOpStrategy(), LeaderlessProtocol(), num_simulations=5,
                               max_time=100, stop_on_data_loss=False, seed=1,
                               network_config=net).availability_mean()
    assert availability(az) == 1.0
    assert availability(region) == pytest.approx(.01)
