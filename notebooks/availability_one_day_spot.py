"""One-day FULL Markov versus fixed and adaptive Spot N=3 Monte Carlo.

Run: .venv/bin/python -m notebooks.availability_one_day_spot
The adaptive stopping rule and disjoint seeds are fixed before any MC runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from notebooks.availability_convergence_study import (
    ROOT, THREAD_VARIABLES, VM_PROFILES, np, scipy, os, make_cluster,
    node_config_for, raft_protocol, replacement_strategy,
)
from notebooks.availability_finite_horizon import dense_time_average, bdf_time_average
from notebooks.availability_weekly_windows import run_fixed_week
from notebooks.availability_skew_diagnostics import empirical_bernstein
from powder.mc_backend import MonteCarloResults, ConvergenceCriteria, _check_convergence
from powder.scenario import QualityLevel, build_markov_model
from powder.simulation.simulator import Simulator

HORIZON = 86400.0
BASELINE_RUNS = 100_000
BASELINE_SEED = 420_000_000
ADAPTIVE_SEED = 430_000_000
CHECKPOINTS = [1000 * 2**i for i in range(11)]
ALPHA = .01
OUT = ROOT/'outputs/availability-one-day-spot'


def threshold_verdict(interval, threshold):
    if interval[1] < threshold:
        return 'disproved'
    if interval[0] >= threshold:
        return 'verified'
    return 'unresolved'


def look_alpha(look):
    if look < 1:
        raise ValueError('Look index starts at one')
    return ALPHA / (look * (look + 1))


def summary(samples, elapsed, criteria, threshold, look=None):
    values = np.asarray(samples['availability'])
    status = _check_convergence(MonteCarloResults(availability_samples=values.tolist()), criteria)[0]
    mean, sd, half = float(status.current_mean), float(status.current_std), float(status.ci_half_width)
    nominal = [mean-half, mean+half]
    delta = ALPHA if look is None else look_alpha(look)
    bounded = empirical_bernstein(mean, sd, len(values), confidence=1-delta)
    verdict = threshold_verdict(bounded['ci'], threshold)
    return {
        'runs': len(values), 'mean': mean, 'std': sd, 'runtime_seconds': elapsed,
        'nominal_simulator_ci99': nominal, 'nominal_simulator_half_width_ppm': half*1e6,
        'native_precision_converged': bool(status.converged),
        'bounded_ci': bounded['ci'], 'bounded_radius_ppm': bounded['radius']*1e6,
        'look': look, 'look_failure_probability': delta,
        'threshold_verdict': verdict,
        'adaptive_stop': bool(status.converged and verdict != 'unresolved'),
        'actual_data_loss_runs': int(np.count_nonzero(np.isfinite(samples['first_data_loss_seconds']))),
        'quorum_loss_runs': int(np.count_nonzero(np.isfinite(samples['first_physical_quorum_loss_seconds']))),
        'below_99pct_runs': int(np.count_nonzero(values < .99)),
        'sample_sha256': hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(),
    }


def add_runs(samples, target, seed, cfg, label):
    started = time.perf_counter()
    for i in range(len(samples['availability']), target):
        simulator = Simulator(make_cluster(3, cfg), replacement_strategy(cfg), raft_protocol(), seed=seed+i)
        # The existing adapter resumes early data-loss returns and accounts for
        # the complete horizon; its historical function name says "week".
        metrics, _, _, _ = run_fixed_week(simulator, horizon=HORIZON)
        value = metrics.availability_fraction()
        assert 0 <= value <= 1 and abs(float(metrics.total_time())-HORIZON) < 1e-6
        samples['availability'].append(value)
        for key, value in (
            ('first_data_loss_seconds', metrics.time_to_actual_data_loss),
            ('first_physical_quorum_loss_seconds', metrics.time_to_potential_data_loss),
        ):
            samples[key].append(float(value) if value is not None else float('nan'))
        if (i+1) % 10000 == 0:
            print(f'{label}: {i+1:,} runs', flush=True)
    return time.perf_counter()-started


def save(data):
    (OUT/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def percent(value):
    return f'{100*value:.6f}%'


def report(data):
    markov, baseline, adaptive = data['markov'], data['baseline'], data['adaptive']
    threshold = markov['nines_threshold']
    decisive = next((r for r in data['adaptive_checkpoints'] if r['threshold_verdict'] != 'unresolved'), None)
    def interval(values):
        return '–'.join(percent(v) for v in values)
    rows = [
        '# One-day availability: three Spot replicas\n',
        f"The FULL Markov model predicts **{percent(markov['availability'])}**, meeting **{markov['nines']} nines ({percent(threshold)})**. The independent adaptive Monte Carlo run **{adaptive['threshold_verdict']}** this threshold for the simulator at overall 99% confidence across its scheduled checks.\n",
        '| Method | Mean one-day availability | Runs / states | Nominal simulator 99% CI | Bounded decision interval | Runtime |',
        '|---|---:|---:|---|---|---:|',
        f"| FULL Markov | {percent(markov['availability'])} | {markov['states']} states | — | Numerical check, not statistical CI | {markov['total_seconds']:.4f} s |",
    ]
    for label, r in [('Fixed MC baseline', baseline), ('Adaptive MC', adaptive)]:
        rows.append(f"| {label} | {percent(r['mean'])} | {r['runs']:,} runs | {interval(r['nominal_simulator_ci99'])} | {interval(r['bounded_ci'])} | {r['runtime_seconds']:.3f} s |")
    rows += [
        f"\nBaseline MC minus Markov is **{(baseline['mean']-markov['availability'])*1e6:,.1f} ppm**. The fixed baseline's nominal half-width is **±{baseline['nominal_simulator_half_width_ppm']:,.1f} ppm**; the adaptive run's is **±{adaptive['nominal_simulator_half_width_ppm']:,.1f} ppm**.\n",
        (f"The threshold was first resolved at **{decisive['runs']:,} adaptive trials ({decisive['runtime_seconds']:.3f} s)**. The run continued to meet the additional native precision target.\n" if decisive else 'No adaptive checkpoint resolved the threshold.\n'),
        '## Matching the experiment\n',
        'Both methods start with three healthy Spot replicas and a leader and estimate the expected fraction of the complete first 24 hours during which the cluster can commit. This is neither availability at the end of the day nor the probability of a completely outage-free day. Transient failures are exponential with mean 30 days, recovery mean is 20 minutes, permanent per-machine data-loss mean is 1 day, the replacement timeout is 5 minutes, spawning averages 60 seconds, and elections average 5 seconds. The simulator uses the existing safe replacement strategy.\n',
        'Each MC trial accounts for all 86,400 seconds, including time after the simulator first reports actual data loss. This uses the previously validated continuation adapter, avoiding the earlier stopped-duration bias. Baseline and adaptive runs use separate, predetermined seed ranges. Neither run reuses the other’s samples.\n',
        '## Stopping rule and confidence\n',
        f"The Markov result selects k={markov['nines']} and threshold 1−10⁻ᵏ={percent(threshold)} before MC begins. The native simulator convergence calculation is configured for 99% Student-t confidence and absolute half-width ≤{markov['nines_precision']*1e6:,.0f} ppm (half the scale of k nines).\n",
        'The adaptive run checks after 1,000, 2,000, 4,000, …, 1,024,000 trials. It stops only when **both** the native half-width target is met and a bounded interval lies entirely above or below the nines threshold. A narrow interval that still straddles the threshold is inconclusive.\n',
        'For check j, the two-sided empirical Bernstein interval spends δⱼ=0.01/[j(j+1)]. Since the sum over all checks is at most 0.01, a union bound provides at least 99% simultaneous coverage at the scheduled checks for independent, identically distributed trial availabilities in [0,1]. This protects the threshold decision against repeated inspection and skewed samples. The fixed 100,000-run baseline uses an ordinary fixed-sample 99% empirical Bernstein interval. These guarantees are per experiment, not a joint 99% guarantee for both experiments.\n',
        'The native simulator Student-t interval is also reported, but its nominal coverage depends on the approximation for the sample mean; it alone is not the basis of the sequential threshold claim. The bound concerns the simulator’s mean, not the correctness of its physical assumptions.\n',
        '| Adaptive checkpoint | Mean | Native 99% half-width (ppm) | Bounded interval | Threshold decision | Native precision met? |',
        '|---:|---:|---:|---|---|---|',
    ]
    for r in data['adaptive_checkpoints']:
        rows.append(f"| {r['runs']:,} | {percent(r['mean'])} | {r['nominal_simulator_half_width_ppm']:,.1f} | {interval(r['bounded_ci'])} | {r['threshold_verdict']} | {r['native_precision_converged']} |")
    rows += [
        '\n## Why the models can disagree\n',
        f"The fixed baseline observed **{baseline['actual_data_loss_runs']:,} actual-data-loss trials**, **{baseline['quorum_loss_runs']:,} physical-quorum-loss trials**, and **{baseline['below_99pct_runs']:,} trials below 99% daily availability**. The independent adaptive sample observed {adaptive['actual_data_loss_runs']:,} actual-data-loss trials. These events can contribute substantial downtime despite short routine election interruptions.\n",
        'FULL retains the most states among the existing Markov quality levels, but still approximates the simulator’s synchronization, deterministic timeout, standby promotion, and safe replacement behavior. It permits replacement recovery in all-data-loss states and does not preserve the full history of latest-copy ownership. Consequently, FULL is not an exact Markov encoding of this simulator. This experiment establishes the mean-availability discrepancy; it does not isolate the contribution of each modeling approximation.\n',
        '## Numerical validation and reproduction\n',
        f"The Markov calculation uses a dense augmented matrix exponential for the time integral. An independent sparse BDF integration differs in availability by {markov['bdf_difference']:.3g}; its additional verification runtime is {markov['bdf_diagnostics']['bdf_seconds']:.4f} s. The table reports Markov build plus dense solve, excluding BDF verification. MC runtimes report trajectory generation; confidence calculations and file writing are excluded. Everything runs through Python, with one-thread NumPy/SciPy settings; no Rust implementation is used.\n",
        'Run `.venv/bin/python -m notebooks.availability_one_day_spot`. `results.json` contains the complete checkpoint history and source hashes. `baseline_samples.npz` and `adaptive_samples.npz` preserve per-trial availability and first data/quorum loss times; NaN means no such event during the day.\n',
    ]
    (OUT/'REPORT.md').write_text('\n'.join(rows))
    short = '\n'.join(rows).split('\n## Matching the experiment')[0]
    (OUT/'SUMMARY.md').write_text(short+'\n\nThe adaptive decision uses a bounded interval corrected for repeated checks. FULL retains modeling approximations; the comparison tests agreement with the simulator. See REPORT.md for assumptions and the full stopping rule.\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = node_config_for(next(p for p in VM_PROFILES if p.name == 'Spot'))
    started = time.perf_counter()
    model = build_markov_model([cfg]*3, raft_protocol(), replacement_strategy(cfg), QualityLevel.FULL)
    built = time.perf_counter()-started
    started = time.perf_counter()
    average, diagnostics = dense_time_average(model, HORIZON)
    solved = time.perf_counter()-started
    availability = 1-float(average[~model.live_mask].sum())
    reference, bdf_diagnostics = bdf_time_average(model, HORIZON)
    assert abs(reference-availability) < 1e-9
    k = math.floor(-math.log10(1-availability))
    threshold, epsilon = 1-10.**(-k), .5*10.**(-k)
    criteria = ConvergenceCriteria(confidence_level=.99, absolute_error=epsilon,
                                   min_runs=1000, max_runs=CHECKPOINTS[-1], batch_size=1000)
    dependencies = [Path(__file__), ROOT/'notebooks/availability_weekly_windows.py',
        ROOT/'notebooks/availability_finite_horizon.py', ROOT/'notebooks/availability_skew_diagnostics.py',
        ROOT/'notebooks/raft_markov_quality_benchmark.py', ROOT/'powder/monte_carlo.py']
    dependencies += list((ROOT/'powder/simulation').glob('*.py'))
    dependencies += list((ROOT/'powder/markov_builders').glob('*.py'))
    data = {
        'metadata': {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
            'threads': {v: os.environ[v] for v in THREAD_VARIABLES},
            'baseline_seed': BASELINE_SEED, 'adaptive_seed': ADAPTIVE_SEED,
            'baseline_runs': BASELINE_RUNS, 'adaptive_checkpoints': CHECKPOINTS,
            'overall_adaptive_alpha': ALPHA, 'horizon_seconds': HORIZON,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}},
        'markov': {'states': model.num_states, 'availability': availability, 'nines': k,
            'nines_threshold': threshold, 'nines_precision': epsilon,
            'build_seconds': built, 'dense_solve_seconds': solved, 'total_seconds': built+solved,
            'diagnostics': diagnostics, 'bdf_difference': reference-availability, 'bdf_diagnostics': bdf_diagnostics},
        'adaptive_checkpoints': [],
    }
    print('Markov:', json.dumps(data['markov']), flush=True)
    save(data)
    empty = lambda: {'availability': [], 'first_data_loss_seconds': [], 'first_physical_quorum_loss_seconds': []}
    baseline = empty()
    elapsed = add_runs(baseline, BASELINE_RUNS, BASELINE_SEED, cfg, 'Baseline')
    data['baseline'] = summary(baseline, elapsed, criteria, threshold)
    np.savez_compressed(OUT/'baseline_samples.npz', **baseline)
    save(data)
    adaptive, elapsed = empty(), 0.0
    for look, target in enumerate(CHECKPOINTS, 1):
        elapsed += add_runs(adaptive, target, ADAPTIVE_SEED, cfg, 'Adaptive')
        row = summary(adaptive, elapsed, criteria, threshold, look)
        data['adaptive_checkpoints'].append(row)
        data['adaptive'] = row
        save(data)
        print('Adaptive checkpoint:', json.dumps(row), flush=True)
        if row['adaptive_stop']:
            break
    np.savez_compressed(OUT/'adaptive_samples.npz', **adaptive)
    data['adaptive_stop_reason'] = 'precision_and_threshold_resolved' if row['adaptive_stop'] else 'maximum_runs_inconclusive'
    save(data)
    report(data)
    print('Finished:', data['adaptive_stop_reason'], flush=True)


if __name__ == '__main__':
    main()
