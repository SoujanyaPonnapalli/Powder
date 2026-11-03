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
from .regionmap import RegionMap, from_pd_json
from .remote import Runner
from .rundir import RunDir
from .spec import ClusterSpec, ExperimentConfig, Workload
from .state import RunState

TIKV_UNIT = "tikv-20160"
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


def losing_quorum(m: RegionMap, stopped: set[int], rf: int = 3) -> list[str]:
    """RSMs (voter sets) with a majority of voters on stopped stores."""
    need = rf // 2 + 1
    return sorted({rsm_label(r.voters) for r in m.regions if len(set(r.voters) & stopped) >= need})


def plan_rounds(m: RegionMap, store_node: dict[int, str], rounds: int, hold_s: float, gap_s: float = 90.0,
                first_offset_s: float = 60.0, rf: int = 3) -> list[Round]:
    """Pick ``rounds`` different pairs of stores, each shared by at least one RSM, and schedule them in turn."""
    pairs: list[tuple[int, int]] = []
    for r in sorted(m.regions, key=lambda r: r.start):
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
        out.append(Round((a, b), (store_node[a], store_node[b]), losing_quorum(m, {a, b}, rf),
                         first_offset_s + k * (hold_s + gap_s), hold_s))
    return out


def schedule_for(rounds: list[Round]) -> list[ScheduledAction]:
    acts = []
    for i, r in enumerate(rounds, start=1):
        for n in r.stop_nodes:
            acts.append(ScheduledAction(r.start_offset_s, n, f"sudo systemctl stop {TIKV_UNIT}", f"round{i}-stop"))
            acts.append(ScheduledAction(r.start_offset_s + r.hold_s, n, f"sudo systemctl start {TIKV_UNIT}", f"round{i}-start"))
    return acts


def experiment_duration(rounds: list[Round], tail_s: float = 90.0) -> float:
    return max(r.start_offset_s + r.hold_s for r in rounds) + tail_s


def evaluate(summary: dict, rounds: list[Round], interval_s: float = 1.0) -> dict:
    """Compare measured per-RSM probe unavailability with what the rounds should have caused."""
    by_rsm = (summary.get("probe") or {}).get("by_rsm", {})
    hits: dict[str, int] = {}
    hold = rounds[0].hold_s if rounds else 0.0
    for r in rounds:
        for label in r.expected_rsms:
            hits[label] = hits.get(label, 0) + 1
    rows, ok = [], True
    for label in sorted(set(by_rsm) | set(hits)):
        n = hits.get(label, 0)
        measured = by_rsm.get(label, {}).get("unavailable_seconds_all", 0.0)
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


def _current_map(runner: Runner, pd: str) -> RegionMap:
    regions = json.loads(runner.run("controller", f"curl -sf --max-time 60 {pd}/pd/api/v1/regions", timeout=90).stdout)
    stores = json.loads(runner.run("controller", f"curl -sf --max-time 20 {pd}/pd/api/v1/stores", timeout=30).stdout)
    return from_pd_json(regions, 0.0, stores)


def run_self_test(spec: ClusterSpec, rundir: RunDir, runner: Runner, rounds: int = 3, hold_s: float = 60.0,
                  rate_per_region: float = 200.0, **kw) -> tuple[Path, dict]:
    from .experiment import Context, nested_get  # noqa: F401  (kept for symmetry with run_experiment)
    from .install import pd_url
    from .topology import Host
    state: RunState = rundir.load_state()
    state.require_installed()
    hosts = {n: Host(n, r.private_ip, r.region, r.az) for n, r in state.instances.items()}
    m = _current_map(runner, pd_url(spec, hosts))
    ip_node = {r.private_ip: n for n, r in state.instances.items() if r.role == "tikv"}
    store_node = {sid: ip_node[s.address.rsplit(":", 1)[0]] for sid, s in m.stores.items()
                  if s.address.rsplit(":", 1)[0] in ip_node}
    plan = plan_rounds(m, store_node, rounds, hold_s)
    cfg = ExperimentConfig(name="selftest", workload=Workload(duration_s=experiment_duration(plan),
                                                              rate_per_region=rate_per_region))
    exp = run_experiment(spec, cfg, rundir, runner, schedule=schedule_for(plan), **kw)
    summary = json.loads((exp / "summary.json").read_text())
    verdict = evaluate(summary, plan, cfg.prober_interval_s)
    verdict["plan"] = [{"stores": r.stop_stores, "nodes": r.stop_nodes, "expected_rsms": r.expected_rsms,
                        "start_offset_s": r.start_offset_s} for r in plan]
    (exp / "selftest.json").write_text(json.dumps(verdict, indent=1))
    return exp, verdict
