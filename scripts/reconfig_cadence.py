"""How often must we reconfigure? (MVP Fig 8). Expensive: smoke-test only unless explicitly launched.

    python scripts/reconfig_cadence.py --cadences-h 6 12 24 48 168 336 720 --horizon-h 4380 \
        [--scale 1] [--price rho1] [--budget-per-hour 15.3] [--output outputs/reconfig-cadence]
    python scripts/reconfig_cadence.py --smoke          # minutes: 2 cadences, 2 days, 10 RSMs, few MC runs

Catalogue hazards vary over time (calendar curves for hardware reboots, software and spot reclaim; age curves
for disks). For each cadence c the deployment is re-optimized at t = 0, c, 2c, ... on the hazards averaged over
the coming window [t, t + c) (``catalogue.node_config(t0_h=...)``), i.e. the optimizer knows the hazard model
but not the realized faults. Each solve is the fleet ILP of scripts/pilot_fleet_frontier.py at a fixed $/h
budget: knapsack fleets, packing passes, centroid table, Gurobi + exact MC, lattice ascent. The previous
deployment's placements are offered as columns, so staying put is always feasible.

Ground truth is evaluated on fixed blocks (``--block-h``, the finest cadence) for every cadence: block
availability = long-run availability of each placement under the block-average hazards (Rust MC over the
catalogue horizon, region/AZ outages on), i.e. a quasi-stationary approximation. Expected downtime per RSM is
summed per 30-day month; SLA violation = downtime above the ``--sla-nines`` error budget, averaged over RSMs.

Reconfiguration overhead in the legacy mode: replicas added between consecutive deployments (RSMs
matched to minimize moves; powder.faults.reconfig), x ``--gb-per-replica`` GB, billed at the library's
AWS transfer prices from the cheapest surviving member. With ``--workload-json``, ``--write-gb-per-s``,
or ``--max-replica-moves``, named shards instead incur recurring replication and one-time migration in
the optimization budget; the one-time bill is amortized over the next decision window. Both modes report
optimizer CPU time per solve.

Resumable: every solve and every block score is cached under ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary  # noqa: E402
from powder.faults import catalogue as C  # noqa: E402
from powder.faults import fleets as F  # noqa: E402
from powder.faults import reconfig as R  # noqa: E402
from powder.faults.centroid_table import CentroidTable  # noqa: E402
from powder.faults.convert import scenario_network_config  # noqa: E402
from powder.faults.ilp import catalogue_machines  # noqa: E402
from powder.faults.gurobi_ilp import TransitionProblem  # noqa: E402
from powder.faults.prices import transfer_per_gb  # noqa: E402
from powder.faults.transition_costs import ShardSpec, migration_cost, replication_per_hour  # noqa: E402
from powder.simulation import Exponential, NodeReplacementStrategy, RaftLikeProtocol  # noqa: E402
from scripts.experiment_sizing import baseline  # noqa: E402
from scripts.pilot_fleet_frontier import CATALOGUE, REGIONS, SCENARIO, Exact, lattice_ascent, nines, solve_exact  # noqa: E402

TRANSFER = "price-transfer/aws-us@2026-10-21"
MONTH_H = 30 * 24.0


def workload_specs(path: str | None, num_rsms: int, write_gb_per_s: float | None,
                   gb_per_replica: float) -> tuple[ShardSpec, ...]:
    """Load the stable shard order, or make a uniform workload for the opt-in mode."""
    if path is not None and write_gb_per_s is not None:
        raise ValueError("--workload-json and --write-gb-per-s cannot be combined")
    if num_rsms <= 0:
        raise ValueError("--num-rsms must be positive")
    if path is not None:
        data = json.loads(Path(path).read_text())
        records = data.get("shards") if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError("--workload-json must contain an array of shards or a {'shards': [...]} object")
        if len(records) != num_rsms:
            raise ValueError(f"workload has {len(records)} shards, expected --num-rsms={num_rsms}")
        if any(not isinstance(r, dict) for r in records):
            raise ValueError("every shard must be a JSON object")
        try:
            specs = tuple(ShardSpec(r["shard_id"], float(r["write_gb_per_s"]),
                                    float(r["partition_gb"]), ()) for r in records)
        except KeyError as exc:
            raise ValueError(f"missing workload field {exc.args[0]}") from exc
    else:
        rate = 0.0 if write_gb_per_s is None else write_gb_per_s
        specs = tuple(ShardSpec(str(i), rate, gb_per_replica, ()) for i in range(num_rsms))
    if len({s.shard_id for s in specs}) != len(specs):
        raise ValueError("workload shard_id values must be unique")
    return specs


def transition_diagnostics(shard_placements: tuple[tuple[str, tuple[str, ...]], ...],
                           transition: TransitionProblem, price: dict[str, float]) -> dict:
    """Independently check actual named-shard moves and all-in cost, never rematching RSMs."""
    if len(shard_placements) != len(transition.shards):
        raise ValueError("transition solution must contain one placement per shard")
    used: set[str] = set()
    replication = migration = gb = 0.0
    moves = removals = 0
    placements = []
    for (sid, ids), shard in zip(shard_placements, transition.shards):
        if sid != shard.shard_id:
            raise ValueError("transition solution shard order differs from workload order")
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"invalid placement for shard {sid}")
        used.update(ids)
        placements.append(list(ids))
        replication += replication_per_hour(ids, shard.write_gb_per_s,
                                            transition.locations, transition.traffic_price)
        if shard.old_placement:
            change = migration_cost(shard.old_placement, ids, shard.partition_gb,
                                    transition.locations, transition.region_price)
            moves += change.replica_additions
            removals += change.replica_removals
            gb += change.gb_moved
            migration += change.migration_usd
    machine = sum(price[m] for m in used)
    return {"placements": placements, "machine_cost_per_hour": machine,
            "replication_cost_per_hour": replication, "migration_usd": migration,
            "gb_moved": gb, "replica_moves": moves, "replica_removals": removals,
            "horizon_h": transition.horizon_h,
            "cost_per_hour": machine + replication + migration / transition.horizon_h}


def transition_feasible(diag: dict, budget: float, max_replica_moves: int | None) -> bool:
    """Guard post-solve choices against the *actual* budget and aggregate move cap."""
    return (diag["cost_per_hour"] <= budget + 1e-7 * max(1.0, budget)
            and (max_replica_moves is None or diag["replica_moves"] <= max_replica_moves))


def cpu_seconds() -> float:
    """CPU time of this process and its finished children (every Rust MC call is a child process)."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


class Study:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.lib = FaultLibrary.load()
        self.cat = C.load(ROOT / CATALOGUE)
        self.protocol = RaftLikeProtocol(election_time_dist=Exponential(0.2))
        self.strategy = NodeReplacementStrategy(failure_timeout=a.timeout_s, default_node_config=None)
        self.network = scenario_network_config(self.lib, SCENARIO).as_network_config(engine=True)
        self.score_horizon_s = self.cat["horizon_h"] * 3600.0
        self.out = ROOT / a.output
        (self.out / "solves").mkdir(parents=True, exist_ok=True)
        (self.out / "blocks").mkdir(parents=True, exist_ok=True)
        loc = {m["id"]: (m["region"], m.get("az") or m["region"]) for m in self.cat["machines"]}
        self.location = loc.__getitem__
        self.locations = loc
        self.transition_mode = (getattr(a, "workload_json", None) is not None
                                or getattr(a, "write_gb_per_s", None) is not None
                                or getattr(a, "max_replica_moves", None) is not None)
        self.transition_specs = (workload_specs(a.workload_json, a.num_rsms, a.write_gb_per_s, a.gb_per_replica)
                                 if self.transition_mode else ())
        self.transition_key = ""
        if self.transition_mode:
            config = {k: v for k, v in vars(a).items() if k not in ("threads", "output")}
            config["workload"] = [(s.shard_id, s.write_gb_per_s, s.partition_gb) for s in self.transition_specs]
            self.transition_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        print(f"[{time.time() - self.t0:7.0f}s] {msg}", flush=True)

    def machines(self, t0_h: float, window_h: float):
        ms = catalogue_machines(self.lib, self.cat, scenario_id=SCENARIO, price=self.a.price,
                                capacity=self.a.capacity, horizon_h=window_h, rate_scale=self.a.scale, t0_h=t0_h)
        if self.a.max_machines:
            ms = ms[: self.a.max_machines // 2] + [m for m in ms if int(m.machine_id[1:]) >= 1000][: self.a.max_machines // 2]
        return ms

    # ----------------------------------------------------------------------------------------- solve

    def solve(self, cadence_h: float, t0_h: float, window_h: float, previous: list | None) -> dict:
        """Fleet ILP at the fixed budget on hazards averaged over [t0, t0 + window).

        Hysteresis: the new deployment replaces ``previous`` only if its believed downtime is at least
        ``--min-gain`` (fraction) below the previous deployment's, both scored on this window's hazards."""
        a = self.a
        name = f"c{cadence_h:g}-t{t0_h:g}-w{window_h:g}"
        if self.transition_mode:
            name += f"-transition-{self.transition_key}"
        path = self.out / "solves" / f"{name}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if not self.transition_mode or cached.get("old_placements") == (previous or []):
                return cached
        wall, cpu = time.time(), cpu_seconds()
        machines = self.machines(t0_h, window_h)
        by_id = {m.machine_id: m for m in machines}
        price = {m.machine_id: m.node_config.cost_per_hour for m in machines}
        transition = None
        if self.transition_mode:
            old = previous or [() for _ in self.transition_specs]
            if len(old) != len(self.transition_specs):
                raise ValueError("previous deployment has a different shard count")
            shards = tuple(ShardSpec(s.shard_id, s.write_gb_per_s, s.partition_gb, tuple(p))
                           for s, p in zip(self.transition_specs, old))
            traffic_price = lambda src, dst, same_az: transfer_per_gb(  # noqa: E731
                self.lib, TRANSFER, src, dst, same_az=same_az)
            region_price = lambda src, dst: (0.0 if src == dst else transfer_per_gb(  # noqa: E731
                self.lib, TRANSFER, src, dst))
            transition = TransitionProblem(shards, self.locations, traffic_price, region_price, window_h,
                                           a.max_replica_moves if previous else None)
        exact = Exact(by_id, self.protocol, self.strategy, self.score_horizon_s, self.network, a.exact_runs, a.threads)
        feats = F.features(machines)
        labels, cent = F.kmeans_log(feats, k=a.clusters, seed=0)
        med = F.medoids(feats, labels, cent)
        table = CentroidTable([machines[i].node_config for i in med], {f.machine_id: int(l) for f, l in zip(feats, labels)},
                              REGIONS, {f.machine_id: f.region for f in feats}, self.protocol, self.strategy,
                              horizon_s=self.score_horizon_s, network=self.network, runs=a.table_runs)
        with ThreadPoolExecutor(a.threads) as tp:
            u_c = [1 - x for x in tp.map(lambda j: table.availability([j, j, j], (1, 1, 1)), range(len(cent)))]
        lam = np.array([f.lam_t + f.lam_p for f in feats])
        u = np.array([u_c[l] * lam[i] / max(lam[med[l]], 1e-300) for i, l in enumerate(labels)])
        u_of = {f.machine_id: float(x) for f, x in zip(feats, u)}
        feat_of = {f.machine_id: f for f in feats}
        dummy = lambda ms: type("S", (), {"availability": 0.0})()  # noqa: E731
        base_pl = []
        for rf in (3, 5, 7):
            for cheap in (False, True):
                b = baseline(machines, a.num_rsms, rf, a.capacity, cheap=cheap, seed=0, scorer=dummy)
                if b["cost_per_hour"] <= a.budget_per_hour:
                    base_pl += [tuple(sorted(p)) for p in b["placements"]]

        def table_batch(placements) -> None:
            keys = {}
            for ids in placements:
                cids, pat = [table.labels[m] for m in ids], table.pattern_of(ids)
                keys.setdefault(CentroidTable.key(cids, pat), (cids, pat))
            todo = [v for k, v in keys.items() if k not in table.memo]
            with ThreadPoolExecutor(a.threads) as tp:
                list(tp.map(lambda kv: table.availability(*kv), todo))

        # The legacy 90%-of-budget compute lower bound leaves no room for
        # replication traffic. In transition mode the knapsack is a compute-only
        # candidate heuristic; enforce the full bill in the shard-aware MILP.
        fleets = F.fleets_for_budget(feats, u, a.budget_per_hour, num_rsms=a.num_rsms, capacity=a.capacity,
                                     sizes=a.fleet_sizes, repeats=a.fleet_repeats,
                                     alpha=0.0 if transition is not None else a.alpha)
        placements, covers = set(), []
        for i, fl in enumerate(fleets):
            rng = random.Random(i)
            ff = [feat_of[m] for m in sorted(fl)]
            rfs = [r for r in (3, 5, 7) if F.pack_pass(ff, r, 1, a.capacity, u_of, rng)]
            mix = F.balanced_mix(a.num_rsms, a.capacity * len(ff), rfs)
            rand = set()
            for r, n_pass in F.split_passes(mix, a.passes).items():
                c = F.pack_pass(ff, r, a.num_rsms, a.capacity, u_of, rng)
                placements.update(c)
                if len(c) == a.num_rsms:
                    covers.append(c)
                for _ in range(n_pass):
                    rand.update(F.pack_pass(ff, r, a.num_rsms, a.capacity, u_of, rng, noise=0.5, near=0.1))
            table_batch({n for p in rand for n in (p, *F.swap_neighbors(p, ff))})
            placements |= {F.improve(p, ff, table.score) for p in rand}
        prev = [tuple(sorted(p)) for p in previous] if previous else []
        prev_cost = sum(price[m] for m in {m for p in prev for m in p})
        prev_diag = (transition_diagnostics(tuple((s.shard_id, p) for s, p in zip(self.transition_specs, prev)),
                                            transition, price) if transition is not None and prev else None)
        prev_ok = (transition_feasible(prev_diag, a.budget_per_hour, transition.max_replica_moves)
                   if prev_diag is not None else prev_cost <= a.budget_per_hour)
        pool = list(dict.fromkeys([*placements, *base_pl, *prev]))
        warm = prev if prev and prev_ok else (
            max(covers, key=lambda c: sum(map(table.score, c))) if covers else [])
        if not pool:
            raise RuntimeError(f"t={t0_h}: no candidate placements under ${a.budget_per_hour}/h")
        sol, rounds, t_ilp, t_mc, _ = solve_exact(pool, price, exact, table.score, a.num_rsms, a.capacity,
                                                  a.budget_per_hour, warm=warm, threads=a.threads,
                                                  feat_of=None if transition is not None else feat_of,
                                                  seed=int(t0_h), u_of=u_of, prefetch=table_batch,
                                                  time_limit=a.ilp_time_limit,
                                                  **({"transition": transition} if transition is not None else {}))
        best = sol
        if sol.selected:
            asc, *_ = lattice_ascent(sol, [feat_of[m] for m in {m for ids in pool for m in ids}], u_of, exact, price,
                                     a.num_rsms, a.capacity, a.budget_per_hour, threads=a.threads, seed=int(t0_h),
                                     **({"transition": transition} if transition is not None else {}))
            score = lambda s: sum(exact(it.candidate.machine_ids) * it.count for it in s.selected)  # noqa: E731
            asc_ok = True
            if transition is not None and asc.selected:
                asc_diag = transition_diagnostics(asc.shard_placements, transition, price)
                asc_ok = transition_feasible(asc_diag, a.budget_per_hour, transition.max_replica_moves)
            if asc.selected and asc_ok and score(asc) >= score(sol):
                best = asc
        if not best.selected:
            detail = (f", max_replica_moves={transition.max_replica_moves}, horizon_h={window_h:g}"
                      if transition is not None else "")
            raise RuntimeError(f"t={t0_h}: ILP {best.status}; no incumbent within "
                               f"${a.budget_per_hour}/h{detail} (also check candidate capacity)")
        diag = None
        if transition is not None:
            diag = transition_diagnostics(best.shard_placements, transition, price)
            if not transition_feasible(diag, a.budget_per_hour, transition.max_replica_moves):
                raise RuntimeError(f"t={t0_h}: ILP result violates all-in budget or replica-move cap")
            for reported, calculated in (("total_cost_per_hour", "cost_per_hour"),
                                          ("machine_cost_per_hour", "machine_cost_per_hour"),
                                          ("replication_cost_per_hour", "replication_cost_per_hour"),
                                          ("migration_usd", "migration_usd"),
                                          ("gb_moved", "gb_moved"),
                                          ("replica_moves", "replica_moves"),
                                          ("replica_removals", "replica_removals")):
                value = getattr(best, reported, None)
                if value is not None and not math.isclose(value, diag[calculated], rel_tol=1e-7, abs_tol=1e-7):
                    raise RuntimeError(f"t={t0_h}: ILP {reported} disagrees with selected deployment")
            dep = diag["placements"]
        else:
            dep = [list(it.candidate.machine_ids) for it in best.selected for _ in range(it.count)]
        believed = sum(exact(p) for p in dep) / len(dep)
        kept = False
        if prev and prev_ok:
            exact.batch(prev)
            prev_believed = sum(exact(p) for p in prev) / len(prev)
            if (1 - believed) > (1 - prev_believed) * (1 - a.min_gain):
                dep, believed, kept = [list(p) for p in prev], prev_believed, True
                if transition is not None:
                    diag = prev_diag
        used = {m for p in dep for m in p}
        rec = {"t0_h": t0_h, "window_h": window_h, "placements": dep,
               "cost_per_hour": diag["cost_per_hour"] if diag is not None else sum(price[m] for m in used),
               "believed_availability": believed, "believed_nines": nines(believed), "machines_used": len(used),
               "kept_previous": kept,
               "candidates": len(pool), "rounds": rounds, "ilp_s": t_ilp, "mc_s": t_mc,
               "wall_s": time.time() - wall, "cpu_s": cpu_seconds() - cpu}
        if diag is not None:
            rec.update({k: v for k, v in diag.items() if k != "placements"})
            rec["shard_ids"] = [s.shard_id for s in self.transition_specs]
            rec["old_placements"] = previous or []
        path.write_text(json.dumps(rec) + "\n")
        self.log(f"solve t={t0_h:g}h w={window_h:g}h: {rec['believed_nines']:.2f} believed nines"
                 f"{' (kept previous)' if kept else ''}, "
                 f"${rec['cost_per_hour']:.2f}/h, {rec['wall_s']:.0f}s wall, {rec['cpu_s']:.0f}s CPU")
        return rec

    # ----------------------------------------------------------------------------------------- evaluate

    def block_scores(self, block: tuple[float, float], placements: list[tuple[str, ...]]) -> dict:
        """Long-run availability of each placement under the block-average hazards (cached per block)."""
        t0, t1 = block
        path = self.out / "blocks" / f"t{t0:g}-{t1:g}.json"
        memo = json.loads(path.read_text()) if path.exists() else {}
        todo = sorted({"|".join(p) for p in placements} - set(memo))
        if todo:
            by_id = {m.machine_id: m for m in self.machines(t0, t1 - t0)}
            ex = Exact(by_id, self.protocol, self.strategy, self.score_horizon_s, self.network,
                       self.a.eval_runs, self.a.threads)
            ex.batch([tuple(k.split("|")) for k in todo])
            memo.update({k: ex(tuple(k.split("|"))) for k in todo})
            path.write_text(json.dumps(memo) + "\n")
        return memo

    # ----------------------------------------------------------------------------------------- one cadence

    def run_cadence(self, cadence_h: float) -> dict:
        a = self.a
        starts, blocks = R.schedule(a.horizon_h, cadence_h, a.block_h)
        solves, prev = [], None
        for t in starts:
            window = min(cadence_h, a.horizon_h - t)
            rec = self.solve(cadence_h, t, window, prev)
            solves.append(rec)
            prev = rec["placements"]
        # Named transition metrics come from the chosen, guarded solve, not legacy Hungarian matching.
        if self.transition_mode:
            migs = [None] + [{"replicas_added": s["replica_moves"],
                              "replicas_removed": s["replica_removals"],
                              "gb_moved": s["gb_moved"], "transfer_cost": s["migration_usd"]}
                             for s in solves[1:]]
        else:
            tprice = lambda s, d: transfer_per_gb(self.lib, TRANSFER, s[0], d[0], same_az=(s == d))  # noqa: E731
            migs = [None] + [R.migration([tuple(p) for p in x["placements"]], [tuple(p) for p in y["placements"]],
                                         gb_per_replica=a.gb_per_replica, location=self.location, price_per_gb=tprice)
                             for x, y in zip(solves, solves[1:])]
        # ground truth per block
        block_rows = []
        for b0, b1 in blocks:
            k = max(i for i, s in enumerate(starts) if s <= b0 + 1e-9)
            dep = [tuple(sorted(p)) for p in solves[k]["placements"]]
            memo = self.block_scores((b0, b1), dep)
            avail = [memo["|".join(p)] for p in dep]
            block_rows.append({"t0_h": b0, "t1_h": b1, "deployment": k, "mean_availability": float(np.mean(avail)),
                               "downtime_min_per_rsm": [(1 - x) * (b1 - b0) * 60.0 for x in avail]})
        # monthly aggregates (expected downtime per RSM; SLA violation averaged over RSMs)
        months = []
        n_months = max(1, int(math.ceil(a.horizon_h / MONTH_H - 1e-9)))
        for mi in range(n_months):
            m0, m1 = mi * MONTH_H, min((mi + 1) * MONTH_H, a.horizon_h)
            rows = [r for r in block_rows if m0 <= r["t0_h"] < m1]
            if not rows:
                continue
            down = np.sum([r["downtime_min_per_rsm"] for r in rows], axis=0)
            period = (m1 - m0) * 60.0
            viol = [R.sla_violation_minutes(d, period, a.sla_nines) for d in down]
            months.append({"month": mi, "hours": m1 - m0, "mean_downtime_min": float(np.mean(down)),
                           "sla_violation_min": float(np.mean(viol)),
                           "rsms_violating": int(sum(v > 0 for v in viol))})
        days = a.horizon_h / 24.0
        if self.transition_mode:
            gb = sum(m["gb_moved"] for m in migs if m)
            transfer = sum(m["transfer_cost"] for m in migs if m)
            moved = sum(m["replicas_added"] for m in migs if m)
        else:
            gb = sum(m.gb_moved for m in migs if m)
            transfer = sum(m.transfer_cost for m in migs if m)
            moved = sum(m.replicas_added for m in migs if m)
        out = {
            "cadence_h": cadence_h, "label": R.cadence_label(cadence_h), "reconfigurations": len(starts),
            "sla_violation_min_per_month": float(np.mean([m["sla_violation_min"] * MONTH_H / m["hours"]
                                                          for m in months])),
            "downtime_min_per_month": float(np.mean([m["mean_downtime_min"] * MONTH_H / m["hours"] for m in months])),
            "mean_availability": float(np.mean([r["mean_availability"] for r in block_rows])),
            "gb_moved_per_day": gb / days,
            "transfer_cost_per_day": transfer / days,
            "replicas_moved_per_day": moved / days,
            "optimizer_cpu_h_per_day": sum(s["cpu_s"] for s in solves) / 3600.0 / days,
            "deployment_changes": sum(not s.get("kept_previous") for s in solves[1:]),
            "mean_cost_per_hour": float(np.mean([s["cost_per_hour"] for s in solves])),
            "months": months,
            "solves": [{k: v for k, v in s.items() if k != "placements"} for s in solves],
            "migrations": [m if self.transition_mode else m.__dict__ if m else None for m in migs],
            "blocks": [{k: v for k, v in r.items() if k != "downtime_min_per_rsm"} for r in block_rows],
        }
        if self.transition_mode:
            out.update({
                "transition_aware": True,
                "mean_cost_per_hour": sum(s["cost_per_hour"] * s["window_h"] for s in solves) / a.horizon_h,
                "mean_machine_cost_per_hour": sum(s["machine_cost_per_hour"] * s["window_h"]
                                                  for s in solves) / a.horizon_h,
                "mean_replication_cost_per_hour": sum(s["replication_cost_per_hour"] * s["window_h"]
                                                      for s in solves) / a.horizon_h,
                "migration_cost_total": transfer,
                "gb_moved_total": gb,
            })
            out["replication_cost_per_day"] = out["mean_replication_cost_per_hour"] * 24.0
        suffix = f"-transition-{self.transition_key}" if self.transition_mode else ""
        (self.out / f"cadence-{out['label']}{suffix}.json").write_text(json.dumps(out, indent=1) + "\n")
        self.log(f"cadence {out['label']}: {out['sla_violation_min_per_month']:.3f} SLA-violation min/month, "
                 f"{out['downtime_min_per_month']:.3f} downtime min/month, {out['gb_moved_per_day']:.1f} GB/day, "
                 f"{out['optimizer_cpu_h_per_day']:.2f} optimizer CPU-h/day")
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences-h", type=float, nargs="+", default=[6, 12, 24, 48, 168, 336, 720])
    ap.add_argument("--horizon-h", type=float, default=4380.0, help="6 months")
    ap.add_argument("--block-h", type=float, default=6.0, help="ground-truth resolution (finest cadence)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--price", default="rho1")
    ap.add_argument("--budget-per-hour", type=float, default=15.3, help="~50%% of the 1x rho1 frontier's range")
    ap.add_argument("--num-rsms", type=int, default=50)
    ap.add_argument("--capacity", type=int, default=5)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--sla-nines", type=float, default=5.0)
    ap.add_argument("--gb-per-replica", type=float, default=100.0)
    ap.add_argument("--workload-json", help="named shard workload: JSON array or {'shards': [...]} object")
    ap.add_argument("--write-gb-per-s", type=float, default=None,
                    help="uniform shard write rate (opt in to transition-aware optimization)")
    ap.add_argument("--max-replica-moves", type=int, default=None,
                    help="total replicas added per reconfiguration; inactive on the initial solve")
    ap.add_argument("--min-gain", type=float, default=0.05,
                    help="switch deployments only if believed downtime drops by at least this fraction")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--fleet-sizes", type=int, default=4)
    ap.add_argument("--fleet-repeats", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--exact-runs", type=int, default=100_000)
    ap.add_argument("--table-runs", type=int, default=20_000)
    ap.add_argument("--eval-runs", type=int, default=50_000)
    ap.add_argument("--ilp-time-limit", type=float, default=120.0)
    ap.add_argument("--max-machines", type=int, default=0, help="subsample the catalogue (smoke tests)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--output", default="outputs/reconfig-cadence")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (minutes)")
    a = ap.parse_args()
    if a.smoke:
        a.cadences_h, a.horizon_h, a.block_h = [12.0, 24.0], 48.0, 12.0
        a.num_rsms, a.max_machines, a.budget_per_hour = 10, 300, 6.0
        a.fleet_sizes, a.fleet_repeats, a.passes, a.clusters = 2, 1, 2, 8
        a.exact_runs, a.table_runs, a.eval_runs, a.ilp_time_limit = 2_000, 1_000, 2_000, 20.0
        if a.output == "outputs/reconfig-cadence":
            a.output = "outputs/reconfig-cadence-smoke"
    if a.max_replica_moves is not None and a.max_replica_moves < 0:
        ap.error("--max-replica-moves must be nonnegative")
    if a.workload_json is not None or a.write_gb_per_s is not None or a.max_replica_moves is not None:
        try:
            workload_specs(a.workload_json, a.num_rsms, a.write_gb_per_s, a.gb_per_replica)
        except ValueError as exc:
            ap.error(str(exc))
        default_output = "outputs/reconfig-cadence-smoke" if a.smoke else "outputs/reconfig-cadence"
        if a.output == default_output:
            a.output += "-transition"
    s = Study(a)
    meta = {k: v for k, v in vars(a).items() if k not in ("threads",)}
    suffix = f"-transition-{s.transition_key}" if s.transition_mode else ""
    (s.out / f"params{suffix}.json").write_text(json.dumps({**meta, "catalogue": CATALOGUE, "scenario": SCENARIO,
                                                       "transfer": TRANSFER}, indent=1) + "\n")
    rows = [s.run_cadence(c) for c in sorted(a.cadences_h)]
    (s.out / f"summary{suffix}.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("solves", "migrations", "blocks", "months")} for r in rows],
        indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
