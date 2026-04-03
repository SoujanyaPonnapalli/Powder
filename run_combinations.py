import csv
import itertools
import math
import os
import sys
import multiprocessing

import numpy as np
from scipy import stats as scipy_stats

from powder.simulation.distributions import Exponential, Normal, Constant, days, hours, minutes, Seconds
from powder.simulation.node import NodeConfig, NodeState
from powder.simulation.cluster import ClusterState
from powder.simulation.network import NetworkState
from powder.simulation.protocol import RaftLikeProtocol
from powder.simulation.strategy import NodeReplacementStrategy
from powder.monte_carlo import MonteCarloRunner, MonteCarloConfig


ELECTION_TIME = Exponential(1 / Seconds(5))  # ~5-second leader elections


def compute_ci(samples, confidence=0.95):
    """Compute mean and 95% CI for a list of numeric samples.

    Returns (mean, ci_lo, ci_hi).
    """
    arr = np.array(samples, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean
    std = float(np.std(arr, ddof=1))
    t_crit = scipy_stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    hw = t_crit * std / math.sqrt(n)
    return mean, float(mean - hw), float(mean + hw)


def compute_ci_proportion(p, n, confidence=0.95):
    """Compute mean and 95% CI for a proportion (Wald interval).

    Returns (mean, ci_lo, ci_hi).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    hw = z * math.sqrt((p * (1 - p)) / n)
    return float(p), max(0.0, float(p - hw)), min(1.0, float(p + hw))


def _run_single_combination(args):
    combo_indices, node_configs, node_names = args
    combo_name = "-".join([node_names[idx] for idx in combo_indices])
    
    nodes = {}
    for j, node_idx in enumerate(combo_indices):
        node_id = f"node_{j}"
        nodes[node_id] = NodeState(node_id=node_id, config=node_configs[node_idx])
        
    cluster_state = ClusterState(
        nodes=nodes,
        network=NetworkState(),
        target_cluster_size=len(combo_indices),
    )
    
    # Run sequentially inside to avoid nesting overhead
    mc_config = MonteCarloConfig(
        num_simulations=100_000,
        max_time=days(365),
        stop_on_data_loss=True,
        parallel_workers=1, 
    )

    mc_runner = MonteCarloRunner(mc_config)
    strategy = NodeReplacementStrategy(
        failure_timeout=hours(1), 
        safe_mode=False, 
        default_node_config=None
    )
    protocol = RaftLikeProtocol(
        election_time_dist=ELECTION_TIME,
        snapshot_interval=hours(24),
        log_retention_ops=hours(24) * 7,
    )
    
    results = mc_runner.run(
        cluster=cluster_state,
        strategy=strategy,
        protocol=protocol,
    )
    
    n_samples = len(results.availability_samples)

    # Availability CI
    avail_m, avail_lo, avail_hi = compute_ci(results.availability_samples)

    # Cost CI
    cost_m, cost_lo, cost_hi = compute_ci(results.cost_samples)

    # Data loss probability CI (proportion)
    prob_dl = results.data_loss_probability()
    dl_m, dl_lo, dl_hi = compute_ci_proportion(prob_dl, n_samples)

    # MTTL CI
    mttl = results.mean_time_to_actual_loss()
    mttl_ci = results.ci_time_to_actual_loss()
    mttl_d = mttl / 86400 if mttl is not None else float('inf')
    mttl_lo = mttl_ci[0] / 86400 if mttl_ci else mttl_d
    mttl_hi = mttl_ci[1] / 86400 if mttl_ci else mttl_d

    # Transient failures CI
    trans_m, trans_lo, trans_hi = compute_ci(results.transient_failure_samples)

    # Dataloss failures CI
    dlfail_m, dlfail_lo, dlfail_hi = compute_ci(results.dataloss_failure_samples)

    # Nodes spawned CI
    spawned_m, spawned_lo, spawned_hi = compute_ci(results.nodes_spawned_samples)

    # Unavailability incidents CI
    unavail_m, unavail_lo, unavail_hi = compute_ci(results.unavailability_incident_samples)

    # Leader elections CI
    elections_m, elections_lo, elections_hi = compute_ci(results.leader_election_samples)

    return {
        'name': combo_name,
        # Availability
        'availability_mean': avail_m,
        'availability_ci_lower': avail_lo,
        'availability_ci_upper': avail_hi,
        # Cost
        'cost_mean': cost_m,
        'cost_ci_lower': cost_lo,
        'cost_ci_upper': cost_hi,
        # Data loss probability
        'prob_dl_mean': dl_m,
        'prob_dl_ci_lower': dl_lo,
        'prob_dl_ci_upper': dl_hi,
        # Mean time to data loss (days)
        'mttl_days_mean': mttl_d,
        'mttl_days_ci_lower': mttl_lo,
        'mttl_days_ci_upper': mttl_hi,
        # Transient failures
        'transient_failures_mean': trans_m,
        'transient_failures_ci_lower': trans_lo,
        'transient_failures_ci_upper': trans_hi,
        # Dataloss failures
        'dataloss_failures_mean': dlfail_m,
        'dataloss_failures_ci_lower': dlfail_lo,
        'dataloss_failures_ci_upper': dlfail_hi,
        # Nodes spawned
        'nodes_spawned_mean': spawned_m,
        'nodes_spawned_ci_lower': spawned_lo,
        'nodes_spawned_ci_upper': spawned_hi,
        # Unavailability incidents
        'unavail_incidents_mean': unavail_m,
        'unavail_incidents_ci_lower': unavail_lo,
        'unavail_incidents_ci_upper': unavail_hi,
        # Leader elections
        'leader_elections_mean': elections_m,
        'leader_elections_ci_lower': elections_lo,
        'leader_elections_ci_upper': elections_hi,
    }


FIELDNAMES = [
    'name',
    'availability_mean', 'availability_ci_lower', 'availability_ci_upper',
    'cost_mean', 'cost_ci_lower', 'cost_ci_upper',
    'prob_dl_mean', 'prob_dl_ci_lower', 'prob_dl_ci_upper',
    'mttl_days_mean', 'mttl_days_ci_lower', 'mttl_days_ci_upper',
    'transient_failures_mean', 'transient_failures_ci_lower', 'transient_failures_ci_upper',
    'dataloss_failures_mean', 'dataloss_failures_ci_lower', 'dataloss_failures_ci_upper',
    'nodes_spawned_mean', 'nodes_spawned_ci_lower', 'nodes_spawned_ci_upper',
    'unavail_incidents_mean', 'unavail_incidents_ci_lower', 'unavail_incidents_ci_upper',
    'leader_elections_mean', 'leader_elections_ci_lower', 'leader_elections_ci_upper',
]


def compute_and_print_pareto(results_list):
    print(f"\n--- Current Interesting Results (Pareto Optimal) from {len(results_list)} total ---")
    if not results_list:
        return
        
    pareto_optimal = []
    for r1 in results_list:
        dominated = False
        for r2 in results_list:
            if r1 is r2:
                continue
            better_or_eq = (
                r2['cost_mean'] <= r1['cost_mean'] and 
                r2['availability_mean'] >= r1['availability_mean'] and 
                r2['prob_dl_mean'] <= r1['prob_dl_mean']
            )
            strictly_better = (
                r2['cost_mean'] < r1['cost_mean'] or 
                r2['availability_mean'] > r1['availability_mean'] or 
                r2['prob_dl_mean'] < r1['prob_dl_mean']
            )
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(r1)

    pareto_optimal.sort(key=lambda x: (-x['availability_mean'], x['cost_mean']))
    print(f"Found {len(pareto_optimal)} Pareto optimal (interesting) configurations:")
    for r in pareto_optimal:
        print(f"  {r['name']} | Avail: {r['availability_mean']*100:.4f}% | DL: {r['prob_dl_mean']*100:.2f}% | Cost: ${r['cost_mean']:.2f}")
    print("-" * 50, flush=True)

def main():
    failure_dists = [Exponential(1 / days(1)), Exponential(1 / days(7)), Exponential(1 / days(31))]
    failure_price = [0.03, 0.15, 0.30]
    
    data_loss_dists = [Exponential(1 / days(31)), Exponential(1 / days(365)), Exponential(1 / days(3 * 365))]
    data_loss_price = [0.06, 0.3, 0.6]
    
    cost_per_hour_base = 0.14
    recovery_dist = Exponential(rate=1 / minutes(1))
    log_replay_rate_dist = Constant(value=1000000 / 5000 / 2)
    snapshot_download_time_dist = Normal(mean=minutes(5), std=Seconds(30))
    spawn_dist = Normal(mean=Seconds(50), std=Seconds(5))
    
    node_configs = []
    node_names = []
    
    idx = 0
    for f_idx, (f_dist, f_price) in enumerate(zip(failure_dists, failure_price)):
        for d_idx, (d_dist, d_price) in enumerate(zip(data_loss_dists, data_loss_price)):
            cost = cost_per_hour_base + f_price + d_price
            cfg = NodeConfig(
                region=f"region_{idx}",
                cost_per_hour=cost,
                failure_dist=f_dist,
                recovery_dist=recovery_dist,
                data_loss_dist=d_dist,
                log_replay_rate_dist=log_replay_rate_dist,
                snapshot_download_time_dist=snapshot_download_time_dist,
                spawn_dist=spawn_dist,
            )
            node_configs.append(cfg)
            node_names.append(f"F{f_idx}D{d_idx}")
            idx += 1
            
    indices = list(range(len(node_configs)))
    combinations = []
    for cluster_size in [3, 5, 7]:
        combinations.extend(list(itertools.combinations_with_replacement(indices, cluster_size)))
    
    print(f"Total possible cluster combinations: {len(combinations)}", flush=True)
    
    csv_file = 'rsm_combinations.csv'
    existing_results = []
    existing_names = set()
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = {'name': row['name']}
                for key in FIELDNAMES:
                    if key == 'name':
                        continue
                    result[key] = float(row[key])
                existing_results.append(result)
                existing_names.add(row['name'])
                
    print(f"Loaded {len(existing_results)} existing results from {csv_file}.", flush=True)
    
    worker_args = []
    for combo in combinations:
        combo_name = "-".join([node_names[idx] for idx in combo])
        if combo_name not in existing_names:
            worker_args.append((combo, node_configs, node_names))
            
    print(f"{len(worker_args)} combinations remaining to run.", flush=True)
    
    results_list = list(existing_results)
    
    if len(worker_args) == 0:
        compute_and_print_pareto(results_list)
        return

    write_header = not os.path.exists(csv_file)
    
    pool = multiprocessing.Pool(processes=os.cpu_count())
    new_results_batch = []
    
    for idx, res in enumerate(pool.imap_unordered(_run_single_combination, worker_args)):
        new_results_batch.append(res)
        results_list.append(res)
        
        print(f"Completed {idx+1}/{len(worker_args)}: {res['name']} "
              f"| Avail: {res['availability_mean']*100:.4f}% "
              f"[{res['availability_ci_lower']*100:.4f}%, {res['availability_ci_upper']*100:.4f}%] "
              f"| DL: {res['prob_dl_mean']*100:.2f}% "
              f"| Cost: ${res['cost_mean']:.2f} "
              f"| Elections: {res['leader_elections_mean']:.1f}", flush=True)
              
        if len(new_results_batch) >= 100:
            mode = 'w' if write_header else 'a'
            with open(csv_file, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerows(new_results_batch)
            new_results_batch = []
            
            compute_and_print_pareto(results_list)
            
    if new_results_batch:
        mode = 'w' if write_header else 'a'
        with open(csv_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(new_results_batch)
    
    pool.close()
    pool.join()
        
    compute_and_print_pareto(results_list)

if __name__ == '__main__':
    main()
