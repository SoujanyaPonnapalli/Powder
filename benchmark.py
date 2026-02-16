"""Benchmark: compare Cython vs pure Python simulation performance."""

import time
import copy
import argparse
import os
import sys
import contextlib
from pathlib import Path

# Context manager to temporarily hide .so files
@contextlib.contextmanager
def hide_so_files(simulation_dir):
    so_files = list(Path(simulation_dir).glob("*.so"))
    hidden = []
    try:
        for so in so_files:
            bak = so.with_name("." + so.name + ".bak")
            so.rename(bak)
            hidden.append((so, bak))
        yield
    finally:
        for so, bak in hidden:
            try:
                if bak.exists():
                    bak.rename(so)
            except Exception as e:
                print(f"Error restoring {so}: {e}")

def run_benchmark(n_runs=100, warmup=5):
    # Imports must be here to allow mode switching to take effect before import
    try:
        # Force reload if already imported (unlikely in script, but good practice)
        if 'powder.simulation' in sys.modules:
            del sys.modules['powder.simulation']

        from powder.simulation import (
            ClusterState, NodeConfig, NodeState, NetworkState,
            Normal, Uniform, Exponential, Constant, Simulator,
            Seconds, hours, days, minutes,
            LeaderlessUpToDateQuorumProtocol, NodeReplacementStrategy,
        )
    except ImportError:
        # Fallback if specific imports fail
        print("Import failed. Ensure powder package is installed/accessible.")
        return

    def make_standard_node_config(region="us-east"):
        return NodeConfig(
            region=region, cost_per_hour=0.14,
            failure_dist=Constant(days(1e10)),
            recovery_dist=Constant(Seconds(0)),
            data_loss_dist=Exponential(rate=1 / days(50)),
            log_replay_rate_dist=Constant(1e10),
            snapshot_download_time_dist=Constant(0),
            spawn_dist=Exponential(1/hours(5)),
        )

    def create_cluster(num_nodes=3):
        regions = ["us-east", "us-west", "eu-central"]
        nodes = {}
        for i in range(num_nodes):
            config = make_standard_node_config(regions[i % len(regions)])
            nodes[f"node{i}"] = NodeState(node_id=f"node{i}", config=config)
        return ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=num_nodes)

    protocol = LeaderlessUpToDateQuorumProtocol(commit_rate=1, snapshot_interval=24*60*60, log_retention_ops=2*24*60*60)
    strategy = NodeReplacementStrategy(
        failure_timeout=minutes(30),
        default_node_config=make_standard_node_config(),
    )
    cluster = create_cluster(3)

    # print(f"Running warmup ({warmup} runs)...")
    # for i in range(warmup):
    #     c = copy.deepcopy(cluster)
    #     s = copy.deepcopy(strategy)
    #     p = copy.deepcopy(protocol)
    #     sim = Simulator(initial_cluster=c, strategy=s, protocol=p, seed=i)
    #     sim.run_until_data_loss(max_time=sim_duration)

    # print(f"Running benchmark ({n_runs} runs)...")
    start = time.perf_counter()
    # for i in range(n_runs):
    #     c = copy.deepcopy(cluster)
    #     s = copy.deepcopy(strategy)
    #     p = copy.deepcopy(protocol)
    #     sim = Simulator(initial_cluster=c, strategy=s, protocol=p, seed=42 + i)
    #     sim.run_until_data_loss(max_time=sim_duration)

    from powder.monte_carlo import (
        run_monte_carlo_converged, ConvergenceMetric,
    )
    print("starting runner")
    results = run_monte_carlo_converged(
        cluster=cluster,
        strategy=strategy,
        protocol=protocol,
        max_time=None,
        confidence_level=0.99,
        relative_error=0.1,
        metrics=[
            ConvergenceMetric.MEAN_TIME_TO_DATA_LOSS,
        ],
        stop_on_data_loss=True,
        seed=42,
        min_runs=30,
        max_runs=5000,
        batch_size=100,
    )

    print(results)
    print(results.summary())
    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark simulation performance.")
    parser.add_argument("--mode", choices=["cython", "python"], default="cython",
                      help="Run mode: 'cython' (extensions) or 'python' (pure python). Default: cython")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs")
    args = parser.parse_args()

    sim_dir = Path(__file__).parent / "powder" / "simulation"

    if args.mode == "python":
        print("Running in PURE PYTHON mode (hiding .so files)...")
        with hide_so_files(sim_dir):
            run_benchmark(n_runs=args.runs)
    else:
        # Check if extensions exist
        so_files = list(sim_dir.glob("*.so"))
        if not so_files:
            print("WARNING: No .so files found in powder/simulation/. Running in python mode effectively (or build failed).")
            print("Run 'python setup.py build_ext --inplace' to compile Cython extensions.")
        else:
            print(f"Running in CYTHON mode (found {len(so_files)} modules)...")

        run_benchmark(n_runs=args.runs)

