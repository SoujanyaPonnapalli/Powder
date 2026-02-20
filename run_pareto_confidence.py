import csv
import os
import math
import numpy as np
import scipy.stats as stats

from powder.simulation.distributions import Exponential, Normal, Constant, days, hours, minutes, Seconds
from powder.simulation.node import NodeConfig, NodeState
from powder.simulation.cluster import ClusterState
from powder.simulation.network import NetworkState
from powder.simulation.protocol import LeaderlessUpToDateQuorumProtocol
from powder.simulation.strategy import NodeReplacementStrategy
from powder.monte_carlo import MonteCarloRunner, MonteCarloConfig

def compute_ci_mean(samples, conf=0.95):
    n = len(samples)
    if n < 2:
        val = float(np.mean(samples)) if n > 0 else 0.0
        return val, val, val
    m = np.mean(samples)
    std = np.std(samples, ddof=1)
    z = stats.norm.ppf(1 - (1 - conf)/2)
    hw = z * (std / math.sqrt(n))
    return float(m), float(m - hw), float(m + hw)

def compute_ci_proportion(p, n, conf=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - conf)/2)
    hw = z * math.sqrt((p * (1 - p)) / n)
    return float(p), max(0.0, float(p - hw)), min(1.0, float(p + hw))

def read_pareto_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
        
    results = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'name': row['name'],
                'cost': float(row['cost']),
                'availability': float(row['availability']),
                'prob_dataloss': float(row['prob_dataloss'])
            })
            
    pareto_optimal = []
    for r1 in results:
        dominated = False
        for r2 in results:
            if r1 is r2:
                continue
            better_or_eq = (
                r2['cost'] <= r1['cost'] and 
                r2['availability'] >= r1['availability'] and 
                r2['prob_dataloss'] <= r1['prob_dataloss']
            )
            strictly_better = (
                r2['cost'] < r1['cost'] or 
                r2['availability'] > r1['availability'] or 
                r2['prob_dataloss'] < r1['prob_dataloss']
            )
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(r1)

    pareto_optimal.sort(key=lambda x: (-x['availability'], x['cost']))
    return pareto_optimal

def main():
    input_csv = 'rsm_combinations.csv'
    output_csv = 'pareto_confidence.csv'
    
    pareto_configs = read_pareto_from_csv(input_csv)
    print(f"Discovered {len(pareto_configs)} pareto-optimal configurations from '{input_csv}'.", flush=True)
    if not pareto_configs:
        return

    # Re-build Node Configurations Map
    failure_dists = [Exponential(1 / days(1)), Exponential(1 / days(7)), Exponential(1 / days(31))]
    failure_price = [0.03, 0.15, 0.30]
    
    data_loss_dists = [Exponential(1 / days(31)), Exponential(1 / days(365)), Exponential(1 / days(3 * 365))]
    data_loss_price = [0.06, 0.3, 0.6]
    
    cost_per_hour_base = 0.14
    recovery_dist = Exponential(rate=1 / minutes(1))
    log_replay_rate_dist = Constant(value=1000000 / 5000 / 2)
    snapshot_download_time_dist = Normal(mean=minutes(5), std=Seconds(30))
    spawn_dist = Normal(mean=Seconds(50), std=Seconds(5))
    
    node_configs_map = {}
    
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
            node_name = f"F{f_idx}D{d_idx}"
            node_configs_map[node_name] = cfg
            idx += 1
            
    existing_runs = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_runs.add(row['name'])
    
    fieldnames = [
        'name', 
        'cost_mean', 'cost_ci_lower', 'cost_ci_upper', 
        'availability_mean', 'availability_ci_lower', 'availability_ci_upper', 
        'prob_dl_mean', 'prob_dl_ci_lower', 'prob_dl_ci_upper',
        'mttl_days_mean', 'mttl_days_ci_lower', 'mttl_days_ci_upper'
    ]
    
    write_header = not os.path.exists(output_csv)
    
    # 50,000 simulations target
    run_count = 50000
    mc_config = MonteCarloConfig(
        num_simulations=run_count,
        max_time=days(365),
        stop_on_data_loss=True,
        parallel_workers=os.cpu_count() or 1,
    )
    mc_runner = MonteCarloRunner(mc_config)
    strategy = NodeReplacementStrategy(
        failure_timeout=hours(1), 
        safe_mode=False, 
        default_node_config=None
    )
    protocol = LeaderlessUpToDateQuorumProtocol()
    
    completed = 0
    for combo in pareto_configs:
        c_name = combo['name']
        if c_name in existing_runs:
            completed += 1
            continue
            
        print(f"Running {run_count} simulations for {c_name}...", flush=True)
        nodes = {}
        target_cluster_size = 0
        
        for j, nn in enumerate(c_name.split('-')):
            nodes[f"node_{j}"] = NodeState(node_id=f"node_{j}", config=node_configs_map[nn])
            target_cluster_size += 1
            
        cluster_state = ClusterState(
            nodes=nodes,
            network=NetworkState(),
            target_cluster_size=target_cluster_size,
        )
        
        results = mc_runner.run(
            cluster=cluster_state,
            strategy=strategy,
            protocol=protocol,
        )
        
        avail_samples = results.availability_samples
        cost_samples = results.cost_samples
        dl_prob = results.data_loss_probability()
        n_samples = len(cost_samples)
        
        cost_m, cost_l, cost_u = compute_ci_mean(cost_samples)
        avail_m, avail_l, avail_u = compute_ci_mean(avail_samples)
        dl_m, dl_l, dl_u = compute_ci_proportion(dl_prob, n_samples)
        
        mttl_m = results.mean_time_to_actual_loss()
        mttl_ci_tuple = results.ci_time_to_actual_loss()
        
        mttl_d = mttl_m / 86400 if mttl_m is not None else float('inf')
        mttl_l = mttl_ci_tuple[0] / 86400 if mttl_ci_tuple else mttl_d
        mttl_u = mttl_ci_tuple[1] / 86400 if mttl_ci_tuple else mttl_d
        
        out_dict = {
            'name': c_name,
            'cost_mean': cost_m,
            'cost_ci_lower': cost_l,
            'cost_ci_upper': cost_u,
            'availability_mean': avail_m,
            'availability_ci_lower': avail_l,
            'availability_ci_upper': avail_u,
            'prob_dl_mean': dl_m,
            'prob_dl_ci_lower': dl_l,
            'prob_dl_ci_upper': dl_u,
            'mttl_days_mean': mttl_d,
            'mttl_days_ci_lower': mttl_l,
            'mttl_days_ci_upper': mttl_u,
        }
        
        mode = 'w' if write_header else 'a'
        with open(output_csv, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(out_dict)
        
        completed += 1
        print(f"[{completed}/{len(pareto_configs)}] Checkpointed {c_name} to {output_csv}. "
              f"Cost: ~${cost_m:.2f}, Avail: ~{avail_m*100:.4f}%, DL: ~{dl_m*100:.2f}%.", flush=True)

    print("Finished extracting confident pareto optimal simulations!", flush=True)

if __name__ == '__main__':
    main()
