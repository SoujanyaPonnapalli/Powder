"""Quorum first passage and explicitly conditional MTTDL extrapolations.

Run: .venv/bin/python -m notebooks.availability_first_passage
No new Monte Carlo trajectories. Uses the saved exponential one-week samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

# Establish the same single-thread environment before numerical imports.
from notebooks.availability_convergence_study import (
    ROOT, VM_PROFILES, node_config_for, raft_protocol, replacement_strategy,
)
import mpmath as mp
import numpy as np
from scipy.stats import binomtest

from powder.scenario import QualityLevel, build_markov_model
from powder.simulation.protocol import LeaderlessProtocol


def counts(model):
    return [tuple(map(int, name.split(":"))) for name in model.state_names]


def check_raft_lumping(raft, aggregate):
    """Removing the leader label preserves every count-transition rate.

    This is a projection of the Raft model, not a switch of RSM protocol.
    Election transitions stay inside a count state and cancel from its Q.
    """
    ac = counts(aggregate)
    lookup = {s: i for i, s in enumerate(ac)}
    mapping = []
    for state in counts(raft):
        assert len(state) == 8 and state[7] == 0
        state_counts = list(state[:6])
        state_counts[0] += state[6]
        mapping.append(lookup[tuple(state_counts)])
    worst = 0.0
    for i, ai in enumerate(mapping):
        actual = {}
        row = raft.Q.getrow(i)
        for j, rate in zip(row.indices, row.data):
            aj = mapping[j]
            if ai != aj:
                actual[aj] = actual.get(aj, 0.0) + rate
        row = aggregate.Q.getrow(ai)
        expected = {j: rate for j, rate in zip(row.indices, row.data) if j != ai}
        assert actual.keys() == expected.keys()
        for j in actual:
            worst = max(worst, abs(actual[j] - expected[j]))
            assert math.isclose(actual[j], expected[j], rel_tol=1e-13, abs_tol=1e-18)
    assert mapping[raft.initial_state_id] == aggregate.initial_state_id
    return worst


def precise_passage(model, target, digits):
    """Solve -Q_T t=1; rebuild diagonals from positive rates at high precision.

    Reusing already-rounded float diagonals can swamp a very rare hitting
    rate. Input off-diagonal rates retain their original float precision.
    """
    live = [i for i in range(model.num_states) if not target[i]]
    lookup = {s: i for i, s in enumerate(live)}
    with mp.workdps(digits):
        matrix = mp.matrix(len(live))
        for old_i, new_i in lookup.items():
            row = model.Q.getrow(old_i)
            for old_j, rate in zip(row.indices, row.data):
                if old_i == old_j:
                    continue
                assert rate > 0
                value = mp.mpf(float(rate))
                matrix[new_i, new_i] += value
                if old_j in lookup:
                    matrix[new_i, lookup[old_j]] -= value
        rhs = mp.matrix([1] * len(live))
        started = time.perf_counter()
        result = mp.lu_solve(matrix, rhs)
        seconds = time.perf_counter() - started
        residual = max(abs(x) for x in matrix * result - rhs)
        assert min(result) > 0 and residual < mp.mpf('1e-25')
        return {
            "mean_seconds": float(result[lookup[model.initial_state_id]]),
            "mean_days": float(result[lookup[model.initial_state_id]] / 86400),
            "transient_states": len(live), "decimal_precision": digits,
            "residual": str(residual), "solve_seconds": seconds,
        }


def loss_evidence(row):
    n = row['runs']
    stops = row['early_stops']
    assert all(s['reason'] == 'data_loss' and 0 <= s['time'] < 7*86400 for s in stops)
    k = len(stops)
    evidence = {"runs": n, "losses": k, "weekly_loss_probability": k/n}
    if k:
        interval = binomtest(k, n).proportion_ci(.99, method='exact')
        # Exact binomial probability interval, transformed under the ADDITIONAL
        # assumption that the cluster first-loss hazard is constant forever.
        mean = lambda p: -7 / math.log1p(-p)
        evidence.update({
            "constant_hazard_mttdl_days": mean(k/n),
            "constant_hazard_mttdl_ci99_days": [mean(interval.high), mean(interval.low)],
        })
    else:
        evidence['constant_hazard_mttdl_one_sided99_lower_days'] = n*7/math.log(100)
    return evidence


def duration(days):
    if days < 365:
        return f'{days:,.2f} days'
    years = days/365
    if years >= 1e6:
        return f'{years/1e6:,.3g} million years'
    if years >= 1000:
        rounded = round(years, 2-int(math.floor(math.log10(years))))
        return f'{rounded:,.0f} years'
    return f'{years:,.3g} years'


def main():
    source_dir = ROOT/'outputs/availability-weekly-windows-study'
    sources = [source_dir/'mc.json', source_dir/'tail_control.json']
    saved = json.loads(sources[0].read_text())['results']
    rows = {(r['profile'], r['nodes']): r for r in saved if r['mode'] == 'exponential_30d'}
    rows[('Spot', 7)] = json.loads(sources[1].read_text())['results'][0]
    results = []
    for profile in [p for p in VM_PROFILES if p.name in ('Spot', 'Unreliable')]:
        cfg = node_config_for(profile)
        for n in (3, 5, 7):
            started = time.perf_counter()
            raft = build_markov_model([cfg]*n, raft_protocol(), replacement_strategy(cfg), QualityLevel.NO_ORPHANS)
            aggregate = build_markov_model([cfg]*n, LeaderlessProtocol(), replacement_strategy(cfg), QualityLevel.NO_ORPHANS)
            lump_error = check_raft_lumping(raft, aggregate)
            states = np.array(counts(aggregate))
            # A quorum of up-to-date, available replicas; leader absence alone
            # is excluded. Also report physical/voting quorum for disambiguation.
            target = states[:, 0] < n//2+1
            first = precise_passage(aggregate, target, 50)
            checked = precise_passage(aggregate, target, 80)
            assert math.isclose(first['mean_seconds'], checked['mean_seconds'], rel_tol=1e-13)
            physical = precise_passage(aggregate, states[:, 0]+states[:, 3] < n//2+1, 50)
            row = {
                'profile': profile.name, 'nodes': n, 'quorum': n//2+1,
                'raft_states': raft.num_states, 'aggregate_states': aggregate.num_states,
                'lumping_max_rate_error': lump_error,
                'up_to_date_quorum_first_passage': first,
                'precision_check_80_digits': checked,
                'physical_quorum_first_passage': physical,
                'data_loss': loss_evidence(rows[(profile.name, n)]),
                'total_seconds': time.perf_counter()-started,
            }
            results.append(row)
            print(profile.name, n, duration(first['mean_days']), flush=True)
    out = ROOT/'outputs/availability-first-passage'
    out.mkdir(parents=True, exist_ok=True)
    data = {
        'method': 'Homogeneous exponential NO_ORPHANS Raft count projection; high-precision dense first-passage solve. MTTDL is a conditional extrapolation of saved MC loss incidence, not a Markov MTTDL.',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *sources]},
        'results': results,
    }
    (out/'results.json').write_text(json.dumps(data, indent=2)+'\n')
    text = [
        '# Time to quorum loss and data loss\n',
        'All clusters start healthy, contain only the named profile, and use the exponential transient baseline: 30-day mean transient interval, 20-minute mean recovery, 5-minute replacement timeout, 60-second mean spawn, and the existing synchronization approximation. Per-machine permanent-loss mean is 1 day for Spot and 365 days for Unreliable. A year is 365 days. No new MC trajectories were run.\n',
        '## Mean time to first quorum unavailability\n',
        'The main metric is first loss of a majority of **available, up-to-date** replicas. A leader election alone does not count. Physical quorum instead counts available lagging replicas too, matching the simulator’s `has_potential_data_loss` predicate. These are first-passage times from an initially healthy cluster, not outage durations or reciprocals of unavailability.\n',
        '| Profile | Nodes | Required quorum | MTT up-to-date quorum unavailable | MTT physical quorum unavailable |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in results:
        text.append(f"| {r['profile']} | {r['nodes']} | {r['quorum']} | {duration(r['up_to_date_quorum_first_passage']['mean_days'])} | {duration(r['physical_quorum_first_passage']['mean_days'])} |")
    text += [
        '\nThese are NO_ORPHANS Markov approximations, not validated simulator predictions. The existing model approximates synchronization, timeout shape, and safe replacement. Very large Unreliable values especially depend on independent faults and exclude common-cause events.\n',
        'For each configuration, removing the leader label was verified to preserve all count-transition rates. The reduced generator solves `-Q_T t = 1`, with zero time at the target. The reported quorum solves use dense mpmath arithmetic at 50 decimal digits and agree with independent 80-digit solves to 13 relative digits. Diagonals are rebuilt by summing positive transition rates at high precision; ordinary floating-point diagonals can overwhelm the rare hitting rate. Input rates still have their original floating-point precision. These solves differ from the earlier sparse steady-state and dense finite-horizon calculations.\n',
        '## MTTDL: what the saved data support\n',
        '**A trustworthy MTTDL is not available from the existing availability Markov model.** It does not track which failed or lagging copies retain the latest committed data, and it permits recovery after all data are lost. Making only the all-permanently-failed state absorbing would therefore measure a different event.\n',
        'The simulator records actual loss when no active member retains the latest committed data, including temporarily unavailable members in that check. Its predicate checks active members, not standby copies. The saved exponential runs give the following evidence. The extrapolated mean assumes a constant *cluster first-data-loss hazard* at every future time. Exponential per-machine faults do **not** imply this assumption; a cluster has repair pipelines and starts healthy.\n',
        '| Profile | Nodes | Data-loss weeks / simulated weeks | Extrapolated MTTDL, constant cluster hazard | Conditional 99% interval |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in results:
        d = r['data_loss']
        if d['losses']:
            value = duration(d['constant_hazard_mttdl_days'])
            ci = ' – '.join(duration(x) for x in d['constant_hazard_mttdl_ci99_days'])
        else:
            value = 'Not estimated (zero events)'
            ci = f"> {duration(d['constant_hazard_mttdl_one_sided99_lower_days'])}, one-sided; same hazard assumption"
        text.append(f"| {r['profile']} | {r['nodes']} | {d['losses']:,} / {d['runs']:,} | {value} | {ci} |")
    text += [
        '\nThe point extrapolation is `MTTDL = -7 days / log(1 - losses/runs)`. The two-sided intervals transform exact 99% binomial intervals for seven-day loss probability. Zero-event lower limits use `runs × 7 days / log(100)` and are one-sided 99% limits under that same assumption. These intervals quantify sampling uncertainty, not extrapolation/model error. Without a survival-tail assumption, one-week observations do not determine the unrestricted mean, so no numerical Unreliable MTTDL is justified here.\n',
        'Seven-node Spot uses the independent 100,000-run control: six actual-data-loss stops, distinct from its eight severe-availability weeks. The earlier zero-loss 10,000-run control is not used.\n',
        'For defensible MTTDL across all six configurations, the next model must track latest-copy ownership and preserve the simulator’s safe replacement behavior, then solve an absorbing first-passage problem. Merely extending a weekly availability average or averaging loss times only among observed losses is insufficient.\n',
        '## Reproduction\n',
        'Run `.venv/bin/python -m notebooks.availability_first_passage` from the repository root. `results.json` preserves calculations, residuals, numerical verification, runtimes, and hashes of the saved input samples.\n',
    ]
    (out/'REPORT.md').write_text('\n'.join(text))


if __name__ == '__main__':
    main()
