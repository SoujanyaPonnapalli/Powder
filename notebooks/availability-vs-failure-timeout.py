# %% [markdown]
# # Availability vs Failure Timeout Duration
#
# This notebook measures how RSM availability changes as the
# `failure_timeout` parameter varies from 1 second to 30 days.
#
# The failure_timeout controls how long the `NodeReplacementStrategy` waits
# before replacing a failed node.  Shorter timeouts trigger faster replacement
# but may cause unnecessary churn; longer timeouts leave the cluster degraded.
#
# **Protocol**: RaftLikeProtocol (leader-based, with election downtime).
# **Machine config**: All good machines (baseline failure/recovery rates).
# **Cluster size**: 3 nodes.
#
# The timeout values are log-spaced so that most data points fall in the
# 1 second – 1 day range where the interesting transitions happen.

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
# ## 1. Machine & Protocol Configuration

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

CLUSTER_SIZE = 3
ELECTION_TIME = Exponential(1 / Seconds(5))  # ~5-second leader elections

# %% [markdown]
# ## 2. Generate Failure Timeout Values
#
# We use a log-spaced distribution from 1 second to 30 days.
# To concentrate points in the 1 second – 1 day interval, we split
# the range into two segments with different point densities.

# %%
def generate_timeout_values() -> list[float]:
    """Generate failure timeout values from 1s to 30 days, log-spaced.

    Concentrates ~75% of points in the 1s–1day range.
    """
    one_second = 1.0
    one_day = days(1)
    thirty_days = days(30)

    # Dense sampling in 1s – 1 day  (20 points)
    dense = np.geomspace(one_second, one_day, num=20)

    # Sparse sampling in 1 day – 30 days  (6 points, skip the first to avoid overlap)
    sparse = np.geomspace(one_day, thirty_days, num=7)[1:]

    all_timeouts = np.concatenate([dense, sparse])
    return sorted(set(float(t) for t in all_timeouts))



# %% [markdown]
# ## 3. Helper: Compute 95% CI

# %%
def compute_ci(samples: list, confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Compute mean and 95% CI for a list of numeric samples.

    Returns (mean, ci_lo, ci_hi, ci_half).
    """
    arr = np.array(samples, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean, 0.0
    std = float(np.std(arr, ddof=1))
    t_crit = scipy_stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    ci_half = t_crit * std / math.sqrt(n)
    return mean, mean - ci_half, mean + ci_half, ci_half


# %% [markdown]
# ## 4. Run Monte Carlo Simulations

# %%
def make_cluster() -> ClusterState:
    """Create a 3-node cluster with all good machines."""
    nodes = {}
    for i in range(CLUSTER_SIZE):
        nodes[f"node_{i}"] = NodeState(node_id=f"node_{i}", config=good_config)
    return ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=CLUSTER_SIZE,
    )


def run_sweep():
    """Run Monte Carlo simulations for each failure_timeout value."""
    timeout_values = generate_timeout_values()

    print(f"Number of timeout values: {len(timeout_values)}")
    for t in timeout_values:
        if t < 60:
            print(f"  {t:.1f}s")
        elif t < 3600:
            print(f"  {t/60:.1f}m")
        elif t < 86400:
            print(f"  {t/3600:.2f}h")
        else:
            print(f"  {t/86400:.2f}d")
    print()

    SIM_DURATION = days(365)  # 1 year per simulation

    convergence = ConvergenceCriteria(
        confidence_level=0.95,
        absolute_error=1e-8,
        metrics=[ConvergenceMetric.AVAILABILITY],
        min_runs=10000,
        max_runs=100000,
        batch_size=20000,
    )

    mc_config = MonteCarloConfig(
        num_simulations=50000,
        max_time=SIM_DURATION,
        stop_on_data_loss=False,
        base_seed=42,
    )

    results = []

    print(f"Good machine: MTBF={good_config.failure_dist.mean/3600:.0f}h, "
          f"MTTDL={good_config.data_loss_dist.mean/86400:.0f}d")
    print(f"Cluster size: {CLUSTER_SIZE}, Protocol: Raft")
    print(f"Election time: ~{1/ELECTION_TIME.rate:.0f}s mean")
    print()

    header = (
        f"{'Timeout':>12} | {'Availability':>14} | {'95% CI':>24} | "
        f"{'Elections':>9} | {'Unavail':>10} | {'TransFail':>9} | "
        f"{'DLFail':>6} | {'Spawned':>7} | {'Runs':>5} | {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    for timeout in timeout_values:
        cluster = make_cluster()
        strategy = NodeReplacementStrategy(
            failure_timeout=Seconds(timeout),
            default_node_config=good_config,
        )
        protocol = RaftLikeProtocol(
            election_time_dist=ELECTION_TIME,
            snapshot_interval=hours(24),
            log_retention_ops=hours(24) * 7,
        )
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

        # Availability CI
        avail_mean, avail_lo, avail_hi, avail_ci_half = compute_ci(res.availability_samples)

        # Event counter CIs
        elections_mean, elections_lo, elections_hi, _ = compute_ci(res.leader_election_samples)
        unavail_mean, unavail_lo, unavail_hi, _ = compute_ci(res.unavailability_incident_samples)
        trans_mean, trans_lo, trans_hi, _ = compute_ci(res.transient_failure_samples)
        dl_mean, dl_lo, dl_hi, _ = compute_ci(res.dataloss_failure_samples)
        spawned_mean, spawned_lo, spawned_hi, _ = compute_ci(res.nodes_spawned_samples)

        # Cost CI
        cost_mean, cost_lo, cost_hi, _ = compute_ci(res.cost_samples)

        # Format timeout label
        if timeout < 60:
            timeout_label = f"{timeout:.1f}s"
        elif timeout < 3600:
            timeout_label = f"{timeout/60:.1f}m"
        elif timeout < 86400:
            timeout_label = f"{timeout/3600:.2f}h"
        else:
            timeout_label = f"{timeout/86400:.2f}d"

        row = {
            "timeout_seconds": timeout,
            "timeout_label": timeout_label,
            "n_runs": n_runs,
            "converged": conv_result.converged,
            "elapsed": elapsed,
            # Availability
            "avail_mean": avail_mean,
            "avail_lo": avail_lo,
            "avail_hi": avail_hi,
            "avail_ci_half": avail_ci_half,
            # Leader elections
            "elections_mean": elections_mean,
            "elections_lo": elections_lo,
            "elections_hi": elections_hi,
            # Unavailability incidents
            "unavail_mean": unavail_mean,
            "unavail_lo": unavail_lo,
            "unavail_hi": unavail_hi,
            # Transient failures
            "trans_mean": trans_mean,
            "trans_lo": trans_lo,
            "trans_hi": trans_hi,
            # Dataloss failures
            "dl_mean": dl_mean,
            "dl_lo": dl_lo,
            "dl_hi": dl_hi,
            # Nodes spawned
            "spawned_mean": spawned_mean,
            "spawned_lo": spawned_lo,
            "spawned_hi": spawned_hi,
            # Cost
            "cost_mean": cost_mean,
            "cost_lo": cost_lo,
            "cost_hi": cost_hi,
        }
        results.append(row)

        print(
            f"{timeout_label:>12} | "
            f"{avail_mean*100:>13.6f}% | "
            f"[{avail_lo*100:.6f}%, {avail_hi*100:.6f}%] | "
            f"{elections_mean:>9.1f} | {unavail_mean:>10.4f} | "
            f"{trans_mean:>9.1f} | {dl_mean:>6.2f} | "
            f"{spawned_mean:>7.1f} | "
            f"{n_runs:>5} | "
            f"{elapsed:>5.1f}s"
        )

    print("\nAll simulations complete!")
    return results


# %%
if __name__ == "__main__":
    results = run_sweep()

    # %% [markdown]
    # ## 5. Plot Results with Plotly

    # %%
    x_vals = [r["timeout_seconds"] for r in results]
    y_vals = [r["avail_mean"] * 100 for r in results]
    error_y = [r["avail_ci_half"] * 100 for r in results]

    hover_text = []
    for r in results:
        text = (
            f"<b>Failure Timeout: {r['timeout_label']}</b> ({r['timeout_seconds']:.1f}s)<br>"
            f"<br>"
            f"<b>Availability</b><br>"
            f"  Mean: {r['avail_mean']*100:.8f}%<br>"
            f"  95% CI: [{r['avail_lo']*100:.8f}%, {r['avail_hi']*100:.8f}%]<br>"
            f"<br>"
            f"<b>Leader Elections</b> (per year)<br>"
            f"  Mean: {r['elections_mean']:.2f}<br>"
            f"  95% CI: [{r['elections_lo']:.2f}, {r['elections_hi']:.2f}]<br>"
            f"<br>"
            f"<b>Unavailability Incidents</b> (per year)<br>"
            f"  Mean: {r['unavail_mean']:.4f}<br>"
            f"  95% CI: [{r['unavail_lo']:.4f}, {r['unavail_hi']:.4f}]<br>"
            f"<br>"
            f"<b>Transient Failures</b> (per year)<br>"
            f"  Mean: {r['trans_mean']:.2f}<br>"
            f"  95% CI: [{r['trans_lo']:.2f}, {r['trans_hi']:.2f}]<br>"
            f"<br>"
            f"<b>Dataloss Failures</b> (per year)<br>"
            f"  Mean: {r['dl_mean']:.4f}<br>"
            f"  95% CI: [{r['dl_lo']:.4f}, {r['dl_hi']:.4f}]<br>"
            f"<br>"
            f"<b>Nodes Spawned</b> (per year)<br>"
            f"  Mean: {r['spawned_mean']:.2f}<br>"
            f"  95% CI: [{r['spawned_lo']:.2f}, {r['spawned_hi']:.2f}]<br>"
            f"<br>"
            f"<b>Cost</b> (per year)<br>"
            f"  Mean: ${r['cost_mean']:.2f}<br>"
            f"  95% CI: [${r['cost_lo']:.2f}, ${r['cost_hi']:.2f}]<br>"
            f"<br>"
            f"Runs: {r['n_runs']} | Converged: {r['converged']}"
        )
        hover_text.append(text)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        error_y=dict(type="data", array=error_y, visible=True, thickness=1.5, width=4),
        mode="lines+markers",
        name="Raft (3-node, good machines)",
        marker=dict(size=8, color="#636EFA", symbol="circle"),
        line=dict(color="#636EFA", width=2),
        hovertext=hover_text,
        hoverinfo="text",
    ))

    # Add reference lines for key time intervals
    ref_lines = [
        (1, "1s"),
        (60, "1m"),
        (600, "10m"),
        (3600, "1h"),
        (86400, "1d"),
        (86400 * 7, "7d"),
        (86400 * 30, "30d"),
    ]
    for ref_x, ref_label in ref_lines:
        fig.add_vline(
            x=ref_x,
            line_dash="dot",
            line_color="rgba(150, 150, 150, 0.4)",
            annotation_text=ref_label,
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color="gray",
        )

    fig.update_layout(
        title=dict(
            text=(
                "RSM Availability vs. Failure Timeout Duration<br>"
                "<sub>Raft Protocol | 3-node cluster | Good machines | "
                "NodeReplacementStrategy | 95% CI error bars</sub>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            title="Failure Timeout (seconds, log scale)",
            type="log",
            tickvals=[1, 10, 60, 600, 3600, 86400, 86400*7, 86400*30],
            ticktext=["1s", "10s", "1m", "10m", "1h", "1d", "7d", "30d"],
        ),
        yaxis=dict(
            title="Availability (%)",
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
        ),
        template="plotly_white",
        width=1000,
        height=600,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            align="left",
        ),
    )

    fig.show()

    # Save to HTML
    output_path = "availability-vs-failure-timeout.html"
    fig.write_html(output_path)
    print(f"\nChart saved to {output_path}")
