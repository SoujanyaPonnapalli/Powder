"""Metric self-test (gate G3): stop a quorum of one RSM's TiKV nodes for a known time and check the metrics say so.

No fault emulator is needed (plain ``systemctl stop tikv-20160``). Stopping stores ``A`` and ``B`` makes every Raft
group whose voters include **both** of them lose its quorum, so the *expected* unavailable RSMs are computed from the
live Region map, not guessed. The check then compares what the prober measured per RSM with
``hold_s x (rounds that hit it)``, and requires the RSMs that kept their quorum to stay available (apart from the
leader-election gap of Regions that merely lost a leader, bounded per round).

Pure planning/evaluation functions live here (unit-tested); :func:`run_self_test` wires them to ``run_experiment``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .experiment import ScheduledAction, run_experiment
from .regionmap import Region, RegionMap, from_pd_json, row_key
from .remote import Runner
from .rundir import RunDir
from .spec import ClusterSpec, ExperimentConfig, Workload
from .state import RunState

TIKV_UNIT = "tikv-20160"
# A crash, not `systemctl stop`: SIGTERM starts a graceful shutdown that tries to evict leaders for up to 20 s, which
# at G3 delayed the quorum loss by ~20 s. The follow-up stop keeps systemd's Restart=always from reviving it.
STOP_COMMAND = f"sudo systemctl kill --signal=SIGKILL {TIKV_UNIT}; sudo systemctl stop {TIKV_UNIT}"
ELECTION_GAP_S = 20.0           # allowed unavailability per round for RSMs that only lost a leader
RECOVERY_GAP_S = 45.0           # allowed extra unavailability per round after the stores restart (re-election, retries)


@dataclass
class Round:
    stop_stores: tuple[int, int]
    stop_nodes: tuple[str, str]
    expected_rsms: list[str]            # voter-set labels that lose quorum
    start_offset_s: float
    hold_s: float


def rsm_label(voters) -> str:
    return ",".join(map(str, sorted(voters)))


def losing_quorum(m: RegionMap, stopped: set[int], rf: int = 3, regions: list[Region] | None = None) -> list[str]:
    """RSMs (voter sets) with a majority of voters on stopped stores."""
    need = rf // 2 + 1
    regs = m.regions if regions is None else regions
    return sorted({rsm_label(r.voters) for r in regs if len(set(r.voters) & stopped) >= need})


def probe_regions(m: RegionMap, table_id: int, keys: int, buckets: int) -> list[Region]:
    """The Regions holding the prober's rows (one per bucket, see ``powder-load probe``): the only RSMs it measures."""
    found: dict[int, Region] = {}
    for b in range(buckets):
        r = m.find(row_key(table_id, int((b + 0.5) * keys / buckets)))
        if r is not None:
            found[r.id] = r
    return sorted(found.values(), key=lambda r: r.start)


def planned_duration(rounds: int, hold_s: float, gap_s: float = 90.0, first_offset_s: float = 60.0,
                     tail_s: float = 90.0) -> float:
    """What :func:`experiment_duration` gives for :func:`plan_rounds` output, known before the stores are chosen."""
    return first_offset_s + (rounds - 1) * (hold_s + gap_s) + hold_s + tail_s


def plan_rounds(m: RegionMap, store_node: dict[int, str], rounds: int, hold_s: float, gap_s: float = 90.0,
                first_offset_s: float = 60.0, rf: int = 3, regions: list[Region] | None = None) -> list[Round]:
    """Pick ``rounds`` different pairs of stores, each shared by at least one RSM, and schedule them in turn.

    ``regions`` restricts planning to the Regions that are measured (default: every Region in ``m``)."""
    regs = m.regions if regions is None else regions
    pairs: list[tuple[int, int]] = []
    for r in sorted(regs, key=lambda r: r.start):
        v = sorted(r.voters)
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                p = (v[i], v[j])
                if p not in pairs and all(s in store_node for s in p):
                    pairs.append(p)
    if len(pairs) < rounds:
        raise ValueError(f"only {len(pairs)} distinct store pairs share an RSM; need {rounds}")
    out = []
    for k in range(rounds):
        a, b = pairs[k * max(1, len(pairs) // rounds)]
        out.append(Round((a, b), (store_node[a], store_node[b]), losing_quorum(m, {a, b}, rf, regs),
                         first_offset_s + k * (hold_s + gap_s), hold_s))
    return out


def schedule_for(rounds: list[Round]) -> list[ScheduledAction]:
    acts = []
    for i, r in enumerate(rounds, start=1):
        for n in r.stop_nodes:
            acts.append(ScheduledAction(r.start_offset_s, n, STOP_COMMAND, f"round{i}-stop"))
            acts.append(ScheduledAction(r.start_offset_s + r.hold_s, n, f"sudo systemctl start {TIKV_UNIT}", f"round{i}-start"))
    return acts


def experiment_duration(rounds: list[Round], tail_s: float = 90.0) -> float:
    return max(r.start_offset_s + r.hold_s for r in rounds) + tail_s


def evaluate(summary: dict, rounds: list[Round], interval_s: float = 1.0) -> dict:
    """Compare measured per-RSM probe unavailability with what the rounds should have caused."""
    probe = summary.get("probe") or {}
    by_rsm = probe.get("by_rsm", {})
    # by_rsm sums partition-seconds over the RSM's partitions; compare the mean per partition with the hold
    part_bins = next(iter(probe.get("by_partition", {}).values()), {}).get("bins")
    hits: dict[str, int] = {}
    hold = rounds[0].hold_s if rounds else 0.0
    for r in rounds:
        for label in r.expected_rsms:
            hits[label] = hits.get(label, 0) + 1
    rows, ok = [], True
    for label in sorted(set(by_rsm) | set(hits)):
        n = hits.get(label, 0)
        g = by_rsm.get(label, {})
        parts = g["bins"] / part_bins if part_bins and g.get("bins") else 1.0
        measured = g.get("unavailable_seconds_all", 0.0) / parts
        if n:
            lo, hi = 0.8 * n * hold - interval_s, n * (hold + RECOVERY_GAP_S)
            good = lo <= measured <= hi
        else:
            lo, hi = 0.0, len(rounds) * ELECTION_GAP_S
            good = measured <= hi
        ok &= good
        rows.append({"rsm": label, "rounds_hit": n, "expected_s": n * hold, "measured_s": measured,
                     "allowed_range_s": [max(lo, 0.0), hi], "ok": good})
    missing = [l for l in hits if l not in by_rsm]
    if missing:
        ok = False
    return {"passed": ok, "hold_s": hold, "rounds": len(rounds), "rsms": rows,
            "expected_rsms_not_observed": missing}


REBALANCE_WAIT_S = 300.0        # after the split, before planning: let PD spread the probe table's Regions (G3)
PRE_KILL_S = 6.0               # a probe issued this long before the kill can still time out because of it


def unavailable_seconds(fail_by_sec: dict[int, bool], lo: float, hi: float) -> int:
    """Whole seconds in [lo, hi) in which every probe of a partition failed (``fail_by_sec``: second -> all failed)."""
    return sum(1 for s, failed in fail_by_sec.items() if failed and lo <= s < hi)


def evaluate_rounds(rounds: list[Round], kills: list[tuple[float, float]], lost_by_round: list[set[int]],
                    fail_by_bucket: dict[int, dict[int, bool]], buckets: int,
                    moving_by_round: list[set[int]] | None = None) -> dict:
    """Per round and partition: did exactly the partitions whose Region lost quorum go down for about the hold?

    ``kills[k]`` is (kill done, restart done) of round k, ``lost_by_round[k]`` the buckets whose Region had a majority
    of voters on the stopped stores at the kill; ``moving_by_round[k]`` buckets left out because their Region moved
    around the kill. Unlike :func:`evaluate` this does not average over RSMs, whose membership changes while PD moves
    the freshly split Regions (G3)."""
    out, ok = [], True
    moving_by_round = moving_by_round or [set() for _ in rounds]
    for k, (r, (t_kill, t_up), lost, moving) in enumerate(zip(rounds, kills, lost_by_round, moving_by_round), start=1):
        lo, hi = t_kill - PRE_KILL_S, t_up + RECOVERY_GAP_S
        down = {b: unavailable_seconds(fail_by_bucket.get(b, {}), lo, hi) for b in range(buckets) if b not in moving}
        bad_lost = {b: s for b, s in down.items() if b in lost and not 0.8 * r.hold_s <= s <= r.hold_s + RECOVERY_GAP_S}
        bad_kept = {b: s for b, s in down.items() if b not in lost and s > ELECTION_GAP_S}
        passed = bool(lost) and not bad_lost and not bad_kept
        ok &= passed
        lost_s = sorted(down[b] for b in lost)
        out.append({"round": k, "stores": list(r.stop_stores), "passed": passed, "lost_buckets": len(lost),
                    "moving_buckets": sorted(moving),
                    "lost_down_s": {"min": lost_s[0] if lost_s else None, "max": lost_s[-1] if lost_s else None},
                    "kept_down_s_max": max((s for b, s in down.items() if b not in lost), default=0),
                    "allowed_lost_s": [0.8 * r.hold_s, r.hold_s + RECOVERY_GAP_S], "allowed_kept_s": ELECTION_GAP_S,
                    "bad_lost": bad_lost, "bad_kept": bad_kept})
    return {"passed": ok, "rounds": out}


def evaluate_experiment(exp: Path, rounds: list[Round], rf: int = 3) -> dict:
    """Load the raw probes, fault log and Region-map snapshots of ``exp`` and run :func:`evaluate_rounds`."""
    import gzip
    import re
    prep = json.loads((exp / "prepare.json").read_text())
    table_id, keys, buckets = prep["table_id"], prep["keys"], prep["regions"]
    applied = [json.loads(l) for l in (exp / "faults" / "applied.jsonl").read_text().splitlines() if l.strip()]
    kills = []
    for k in range(1, len(rounds) + 1):
        t = lambda kind: max(a.get("t_done", a["t_applied"]) for a in applied if a["label"] == f"round{k}-{kind}")
        kills.append((t("stop"), t("start")))
    snaps = sorted((float(re.search(r"regions-([0-9.]+)\.json", p.name).group(1)), p)
                   for p in (exp / "pd" / "snapshots").glob("regions-*.json.gz"))
    def lost_in(path: Path, stopped: set[int]) -> set[int]:
        m = from_pd_json(json.load(gzip.open(path)), 0.0, None)
        lost = set()
        for b in range(buckets):
            reg = m.find(row_key(table_id, int((b + 0.5) * keys / buckets)))
            if reg is not None and len(set(reg.voters) & stopped) >= rf // 2 + 1:
                lost.add(b)
        return lost

    # snapshots are ~10 s apart and PD keeps moving the new Regions: classify with the snapshots on both sides of the
    # kill and leave out buckets whose answer differs (their Region moved around the kill)
    lost_by_round, moving_by_round = [], []
    for r, (t_kill, _) in zip(rounds, kills):
        before = [p for ts, p in snaps if ts <= t_kill] or [snaps[0][1]]
        after = [p for ts, p in snaps if ts > t_kill] or [snaps[-1][1]]
        a, b = lost_in(before[-1], set(r.stop_stores)), lost_in(after[0], set(r.stop_stores))
        lost_by_round.append(a & b)
        moving_by_round.append(a ^ b)
    per_sec: dict[int, dict[int, list[bool]]] = {}
    for f in (exp / "workload").glob("*/probes.jsonl.gz"):
        for line in gzip.open(f, "rt"):
            rec = json.loads(line)
            b = min(buckets - 1, rec["row"] * buckets // keys)
            per_sec.setdefault(b, {}).setdefault(int(rec["ts"]), []).append(rec["outcome"] != "ok")
    fail_by_bucket = {b: {s: all(v) for s, v in secs.items()} for b, secs in per_sec.items()}
    return evaluate_rounds(rounds, kills, lost_by_round, fail_by_bucket, buckets, moving_by_round)


def _current_map(runner: Runner, pd: str) -> RegionMap:
    regions = json.loads(runner.run("controller", f"curl -sf --max-time 60 {pd}/pd/api/v1/regions", timeout=90).stdout)
    stores = json.loads(runner.run("controller", f"curl -sf --max-time 20 {pd}/pd/api/v1/stores", timeout=30).stdout)
    return from_pd_json(regions, 0.0, stores)


def run_self_test(spec: ClusterSpec, rundir: RunDir, runner: Runner, rounds: int = 3, hold_s: float = 60.0,
                  rate_per_region: float = 200.0, rebalance_wait_s: float = REBALANCE_WAIT_S,
                  **kw) -> tuple[Path, dict]:
    from .experiment import Context, nested_get  # noqa: F401  (kept for symmetry with run_experiment)
    from .install import pd_url
    from .topology import Host
    state: RunState = rundir.load_state()
    state.require_installed()
    hosts = {n: Host(n, r.private_ip, r.region, r.az) for n, r in state.instances.items()}
    pd = pd_url(spec, hosts)
    ip_node = {r.private_ip: n for n, r in state.instances.items() if r.role == "tikv"}
    cfg = ExperimentConfig(name="selftest", workload=Workload(duration_s=planned_duration(rounds, hold_s),
                                                              rate_per_region=rate_per_region))
    planned: list[Round] = []

    def make_schedule(ctx, prep: dict) -> list[ScheduledAction]:
        # after prepare: plan against the probe table's own Regions (G3: planning on the pre-prepare map picked
        # system-table RSMs that the prober never touches)
        m = _current_map(runner, pd)
        store_node = {sid: ip_node[s.address.rsplit(":", 1)[0]] for sid, s in m.stores.items()
                      if s.address.rsplit(":", 1)[0] in ip_node}
        regs = probe_regions(m, prep["table_id"], prep["keys"], int(cfg.workload.regions_target))
        planned.extend(plan_rounds(m, store_node, rounds, hold_s, regions=regs))
        ctx.log(f"  self-test: {len(regs)} probed Regions on {len({rsm_label(r.voters) for r in regs})} RSM(s); "
                + "; ".join(f"stop {r.stop_nodes} -> {r.expected_rsms}" for r in planned))
        return schedule_for(planned)

    exp = run_experiment(spec, cfg, rundir, runner, schedule=make_schedule, rebalance_wait_s=rebalance_wait_s, **kw)
    plan = planned
    summary = json.loads((exp / "summary.json").read_text())
    verdict = evaluate_experiment(exp, plan)                      # pass/fail: per round and partition
    verdict["by_rsm"] = evaluate(summary, plan, cfg.prober_interval_s)   # informational: per-RSM averages
    verdict["plan"] = [{"stores": r.stop_stores, "nodes": r.stop_nodes, "expected_rsms": r.expected_rsms,
                        "start_offset_s": r.start_offset_s} for r in plan]
    (exp / "selftest.json").write_text(json.dumps(verdict, indent=1))
    return exp, verdict
