import csv
import itertools
import os
import sys
import multiprocessing

from powder.simulation.distributions import Exponential, Normal, Constant, days, hours, minutes, Seconds
from powder.simulation.node import NodeConfig, NodeState
from powder.simulation.cluster import ClusterState
from powder.simulation.network import NetworkState
from powder.simulation.protocol import LeaderlessUpToDateQuorumProtocol
from powder.simulation.strategy import NodeReplacementStrategy
from powder.monte_carlo import MonteCarloRunner, MonteCarloConfig

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
        num_simulations=1000,
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
    protocol = LeaderlessUpToDateQuorumProtocol()
    
    results = mc_runner.run(
        cluster=cluster_state,
        strategy=strategy,
        protocol=protocol,
    )
    
    cost = results.cost_mean()
    availability = results.availability_mean()
    prob_dl = results.data_loss_probability()
    mttl = results.mean_time_to_actual_loss()
    
    return {
        'name': combo_name,
        'cost': cost,
        'availability': availability,
        'prob_dataloss': prob_dl,
        'mttl_days': mttl / 86400 if mttl is not None else float('inf')
    }

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
    print(f"Found {len(pareto_optimal)} Pareto optimal (interesting) configurations:")
    for r in pareto_optimal:
        print(f"  {r['name']} | Avail: {r['availability']*100:.4f}% | DL: {r['prob_dataloss']*100:.2f}% | Cost: ${r['cost']:.2f}")
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
                existing_results.append({
                    'name': row['name'],
                    'cost': float(row['cost']),
                    'availability': float(row['availability']),
                    'prob_dataloss': float(row['prob_dataloss']),
                    'mttl_days': float(row['mttl_days'])
                })
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
              f"| Avail: {res['availability']*100:.4f}% | DL: {res['prob_dataloss']*100:.2f}% | Cost: ${res['cost']:.2f}", flush=True)
              
        if len(new_results_batch) >= 100:
            mode = 'w' if write_header else 'a'
            with open(csv_file, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['name', 'cost', 'availability', 'prob_dataloss', 'mttl_days'])
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerows(new_results_batch)
            new_results_batch = []
            
            compute_and_print_pareto(results_list)
            
    if new_results_batch:
        mode = 'w' if write_header else 'a'
        with open(csv_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'cost', 'availability', 'prob_dataloss', 'mttl_days'])
            if write_header:
                writer.writeheader()
            writer.writerows(new_results_batch)
    
    pool.close()
    pool.join()
        
    compute_and_print_pareto(results_list)

if __name__ == '__main__':
    main()
