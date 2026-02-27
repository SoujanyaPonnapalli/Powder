# %% [markdown]
# # Availability Sanity Checks
#
# This notebook validates that RSM availability degrades monotonically as the
# proportion of "bad" (unreliable) machines increases, for 3, 5, and 7 node
# clusters using the `NodeReplacementStrategy`.
#
# Two protocols are compared:
# - **RaftLikeProtocol**: Leader-based, with election downtime on leader failure.
# - **LeaderlessUpToDateQuorumProtocol**: Leaderless, commits when a quorum is up-to-date.
#
# **Good machines**: Baseline failure and data-loss rates.
# **Bad machines**: 2× higher transient-failure and data-loss rates.

# %%
import math
import time

import numpy as np
import plotly.graph_objects as go
from scipy import stats as scipy_stats

from powder.monte_carlo import (
    ConvergenceCriteria,
    ConvergenceMetric,
    MonteCarloConfig,
    MonteCarloRunner,
)
from powder.simulation import (
    ClusterState,
    Constant,
    Exponential,
    Normal,
    LeaderlessUpToDateQuorumProtocol,
    NetworkState,
    NodeConfig,
    NodeReplacementStrategy,
    NodeState,
    RaftLikeProtocol,
    Seconds,
    days,
    hours,
    minutes,
)

# %% [markdown]
# ## 1. Define Good and Bad Machine Configs

# %%
# --- Good machine: baseline rates ---
good_config = NodeConfig(
    region="us-east",
    cost_per_hour=1.0,
    failure_dist=Exponential(rate=1 / hours(720)),          # MTBF ~30 days
    recovery_dist=Exponential(rate=1 / minutes(10)),        # ~10 min recovery
    data_loss_dist=Exponential(rate=1 / days(365 * 3)),     # MTTDL ~3 years
    log_replay_rate_dist=Constant(1000000 / 5000 / 2),
    snapshot_download_time_dist=Normal(minutes(5), minutes(1)),
    spawn_dist=Normal(Seconds(50), Seconds(5)),
)

# --- Bad machine: 2x failure and data-loss rates ---
bad_config = NodeConfig(
    region="us-east",
    cost_per_hour=1.0,
    failure_dist=Exponential(rate=2 / hours(720)),          # MTBF ~15 days
    recovery_dist=Exponential(rate=1 / minutes(10)),        # same recovery
    data_loss_dist=Exponential(rate=2 / days(365 * 3)),     # MTTDL ~1.5 years
    log_replay_rate_dist=Constant(1000000 / 5000 / 2),
    snapshot_download_time_dist=Normal(minutes(5), minutes(1)),
    spawn_dist=Normal(Seconds(50), Seconds(5)),
)

# %% [markdown]
# ## 2. Build Cluster Configurations
#
# For each cluster size N ∈ {3, 5, 7}, we create all combinations from
# (N good, 0 bad) to (0 good, N bad).

# %%
CLUSTER_SIZES = [3, 5, 7]

def make_cluster(num_good: int, num_bad: int) -> ClusterState:
    """Create a cluster with the specified mix of good and bad nodes."""
    total = num_good + num_bad
    nodes = {}
    for i in range(num_good):
        nodes[f"good_{i}"] = NodeState(node_id=f"good_{i}", config=good_config)
    for i in range(num_bad):
        nodes[f"bad_{i}"] = NodeState(node_id=f"bad_{i}", config=bad_config)
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=total,
    )

# %% [markdown]
# ## 3. Run Monte Carlo Simulations

# %%
def run_all_experiments():
    """Run simulations for both protocols and all cluster configurations."""
    # Simulation parameters
    SIM_DURATION = days(365)  # 1 year per simulation
    FAILURE_TIMEOUT = hours(1)  # replacement triggered after 1h of unavailability
    ELECTION_TIME = Exponential(1 / Seconds(5))  # 5-second leader elections

    # Convergence: 95% CI with ±0.001 absolute error on availability
    convergence = ConvergenceCriteria(
        confidence_level=0.95,
        absolute_error=1e-8,
        metrics=[ConvergenceMetric.AVAILABILITY],
        min_runs=100000,
        max_runs=500000,
        batch_size=50000,
    )

    mc_config = MonteCarloConfig(
        num_simulations=500000,  # upper bound, convergence may stop earlier
        max_time=SIM_DURATION,
        stop_on_data_loss=False,  # run the full year every time for consistent time windows
        base_seed=42,
    )

    protocols = {
        "Raft": lambda: RaftLikeProtocol(election_time_dist=ELECTION_TIME, snapshot_interval=hours(24), log_retention_ops=hours(24) * 7),
        "Leaderless": lambda: LeaderlessUpToDateQuorumProtocol(snapshot_interval=hours(24), log_retention_ops=hours(24) * 7),
    }

    # Storage for results: (protocol_name, cluster_size, num_bad) -> dict
    results_data = {}

    print(f"Good machine: MTBF={good_config.failure_dist.mean/3600:.0f}h, "
          f"MTTDL={good_config.data_loss_dist.mean/86400:.0f}d")
    print(f"Bad machine:  MTBF={bad_config.failure_dist.mean/3600:.0f}h, "
          f"MTTDL={bad_config.data_loss_dist.mean/86400:.0f}d")
    print()

    for proto_name, proto_factory in protocols.items():
        print(f"=== {proto_name} Protocol ===")
        print(f"{'Size':>4} | {'Good':>4} | {'Bad':>3} | {'Availability':>14} | {'95% CI':>24} | {'Runs':>5} | {'Time':>6}")
        print("-" * 80)

        for N in CLUSTER_SIZES:
            for num_bad in range(N + 1):
                num_good = N - num_bad

                cluster = make_cluster(num_good, num_bad)
                strategy = NodeReplacementStrategy(
                    failure_timeout=Seconds(FAILURE_TIMEOUT),
                    default_node_config=good_config,
                )
                protocol = proto_factory()
                runner = MonteCarloRunner(mc_config)

                t0 = time.time()
                conv_result = runner.run_until_converged(
                    cluster=cluster,
                    strategy=strategy,
                    protocol=protocol,
                    convergence=convergence,
                )
                elapsed = time.time() - t0

                res = conv_result.results
                n_runs = len(res.availability_samples)
                avail_mean = res.availability_mean()
                avail_std = res.availability_std()

                # Compute 95% CI using t-distribution
                if n_runs >= 2:
                    t_crit = scipy_stats.t.ppf(0.975, df=n_runs - 1)
                    ci_half = t_crit * avail_std / math.sqrt(n_runs)
                else:
                    ci_half = float("inf")

                ci_lo = avail_mean - ci_half
                ci_hi = avail_mean + ci_half

                results_data[(proto_name, N, num_bad)] = {
                    "num_good": num_good,
                    "num_bad": num_bad,
                    "cluster_size": N,
                    "protocol": proto_name,
                    "availability_mean": avail_mean,
                    "availability_std": avail_std,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "ci_half": ci_half,
                    "n_runs": n_runs,
                    "converged": conv_result.converged,
                }

                print(
                    f"{N:>4} | {num_good:>4} | {num_bad:>3} | "
                    f"{avail_mean*100:>13.6f}% | "
                    f"[{ci_lo*100:.6f}%, {ci_hi*100:.6f}%] | "
                    f"{n_runs:>5} | "
                    f"{elapsed:>5.1f}s"
                )

        print()

    print("All simulations complete!")
    return results_data

# %%
if __name__ == "__main__":
    results_data = run_all_experiments()

    # %% [markdown]
    # ## 4. Verify Monotonic Decrease

    # %%
    print("\nMonotonicity check:")
    all_monotonic = True
    for proto_name in ["Raft", "Leaderless"]:
        print(f"\n  --- {proto_name} Protocol ---")
        for N in CLUSTER_SIZES:
            means = [results_data[(proto_name, N, b)]["availability_mean"] for b in range(N + 1)]
            is_monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
            status = "✓ PASS" if is_monotonic else "✗ FAIL"

            if not is_monotonic:
                all_monotonic = False
                violations = []
                for i in range(len(means) - 1):
                    if means[i] < means[i + 1]:
                        r1 = results_data[(proto_name, N, i)]
                        r2 = results_data[(proto_name, N, i + 1)]
                        overlap = r1["ci_lo"] <= r2["ci_hi"] and r2["ci_lo"] <= r1["ci_hi"]
                        violations.append(
                            f"    {i}bad ({means[i]*100:.4f}%) < {i+1}bad ({means[i+1]*100:.4f}%) "
                            f"{'(CIs overlap - not significant)' if overlap else '(SIGNIFICANT)'}"
                        )
                print(f"  {N}-node: {status}")
                for v in violations:
                    print(v)
            else:
                print(f"  {N}-node: {status} — availability decreases: "
                      f"{means[0]*100:.4f}% → {means[-1]*100:.4f}%")

    if all_monotonic:
        print("\n✓ All cluster sizes and protocols show monotonically decreasing availability!")

    # %% [markdown]
    # ## 5. Plot Results with Plotly

    # %%
    # Colors per cluster size, line style per protocol
    size_colors = {
        3: "#636EFA",   # blue
        5: "#EF553B",   # red
        7: "#00CC96",   # green
    }

    proto_dash = {
        "Raft": "solid",
        "Leaderless": "dash",
    }

    proto_symbol = {
        "Raft": "circle",
        "Leaderless": "diamond",
    }

    fig = go.Figure()

    for proto_name in ["Raft", "Leaderless"]:
        for N in CLUSTER_SIZES:
            x_vals = list(range(N + 1))
            y_vals = [results_data[(proto_name, N, b)]["availability_mean"] * 100 for b in x_vals]
            ci_lo = [results_data[(proto_name, N, b)]["ci_lo"] * 100 for b in x_vals]
            ci_hi = [results_data[(proto_name, N, b)]["ci_hi"] * 100 for b in x_vals]
            n_runs = [results_data[(proto_name, N, b)]["n_runs"] for b in x_vals]
            error_y = [results_data[(proto_name, N, b)]["ci_half"] * 100 for b in x_vals]

            hover_text = [
                f"<b>{N}-node {proto_name}</b><br>"
                f"Good: {N - b}, Bad: {b}<br>"
                f"Availability: {y:.6f}%<br>"
                f"95% CI: [{lo:.6f}%, {hi:.6f}%]<br>"
                f"Runs: {nr}"
                for b, y, lo, hi, nr in zip(x_vals, y_vals, ci_lo, ci_hi, n_runs)
            ]

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                error_y=dict(type="data", array=error_y, visible=True),
                mode="lines+markers",
                name=f"{N}-node {proto_name}",
                marker=dict(size=10, symbol=proto_symbol[proto_name]),
                line=dict(color=size_colors[N], width=2, dash=proto_dash[proto_name]),
                hovertext=hover_text,
                hoverinfo="text",
            ))

    fig.update_layout(
        title=dict(
            text="RSM Availability vs. Number of Bad Machines<br>"
                 "<sub>Solid = Raft, Dashed = Leaderless | NodeReplacementStrategy | 95% CI error bars</sub>",
            x=0.5,
        ),
        xaxis=dict(
            title="Number of Bad Machines (2× failure/data-loss rate)",
            dtick=1,
        ),
        yaxis=dict(
            title="Availability (%)",
            rangemode="tozero",
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
        ),
        template="plotly_white",
        width=900,
        height=550,
    )

    fig.show()

    # Save to HTML for later viewing
    output_path = "availability-sanity-checks.html"
    fig.write_html(output_path)
    print(f"\nChart saved to {output_path}")
