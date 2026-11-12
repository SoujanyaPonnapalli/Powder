"""Turn seeded physical-host fault traces into shared joint-job toggles.

Each slot is ONE physical host, even when it carries replicas of many shards.
Overlapping outages of a host, region or AZ are unioned before toggles are
emitted; one process cannot recover while another overlapping fault remains.
Permanent faults have no recovery in this no-repair joint replay pilot.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .convert import scenario_network_config
from .library import FaultLibrary
from .trace import FaultEvent, generate, stream


def fault_events_to_toggles(events: Sequence[FaultEvent], horizon_seconds: float) -> list[dict]:
    """Emit one down/up pair for every maximal unioned outage interval."""
    if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be finite and positive")
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        scope = "host" if event.kind in {"transient", "permanent", "planned"} else event.kind
        if scope not in {"host", "region", "az", "global"}:
            raise ValueError(f"unknown fault scope {event.kind!r}")
        target = (event.slot if scope == "host" else event.region if scope in {"region", "az"}
                  else "")
        if scope != "global" and (not target or target == "*"):
            raise ValueError(f"{scope} fault needs a physical target")
        start = float(event.start)
        duration = event.duration
        if not math.isfinite(start) or start < 0 or (duration is not None and
                                                      (not math.isfinite(duration) or duration < 0)):
            raise ValueError("fault start and duration must be finite and nonnegative")
        end = horizon_seconds if duration is None else min(horizon_seconds, start + duration)
        if start < end and start < horizon_seconds:
            intervals[(scope, target)].append((start, end))
    out: list[dict] = []
    for (scope, target), parts in sorted(intervals.items()):
        merged: list[list[float]] = []
        for start, end in sorted(parts):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        for start, end in merged:
            base = {"scope": scope, **({"target": target} if scope != "global" else {})}
            out.append({**base, "time": start, "down": True})
            if end < horizon_seconds:
                out.append({**base, "time": end, "down": False})
    return sorted(out, key=lambda e: (e["time"], e["scope"], e.get("target", ""), not e["down"]))


def seeded_joint_toggles(lib: FaultLibrary, scenario_id: str, slots: dict[str, tuple],
                         horizon_seconds: float, seed: int, *, include_az: bool = True,
                         level: str | None = None) -> list[dict]:
    """One seeded draw per host and fault domain, suitable for a joint job.

    ``slots`` maps unique physical host ID to ``(machine_class, region, az)``.
    The canonical trace already draws host, planned, region and global events
    once; AZs are added here using independent seeded streams. A permanent
    host event lasts through the horizon because this pilot has no repair.
    """
    if not slots or any(len(v) < 3 for v in slots.values()):
        raise ValueError("each physical host needs a class, region and AZ")
    events = generate(lib, scenario_id, slots, horizon_seconds, seed, level=level)
    model = scenario_network_config(lib, scenario_id, level=level) if include_az else None
    if model is not None and model.az_rate_per_second > 0:
        for az in sorted({v[2] for v in slots.values()}):
            rng = stream(seed, "az", az)
            t = 0.0
            while True:
                t += float(rng.exponential(1.0 / model.az_rate_per_second))
                if t >= horizon_seconds:
                    break
                d = float(model.az_duration.sample(rng))
                events.append(FaultEvent("*", az, "*", "az", "az", t, d))
    return fault_events_to_toggles(events, horizon_seconds)


def run_seeded_joint_replay(job: dict, lib: FaultLibrary, scenario_id: str,
                            slots: dict[str, tuple], seed: int, *, include_az: bool = True) -> dict:
    """Run one whole-fleet Rust replay using one seeded fault draw per host/domain.

    The fault library specifies hardware/region distributions; PD election,
    schema freshness and request timing are the explicitly declared joint-job
    model, not calibrated TiDB protocols. Every host in the job must have one
    physical slot with the same region and AZ; never draw once per replica.
    """
    if set(slots) != set(job.get("hosts", {})) or any(
        len(slot) < 3 or (slot[1], slot[2]) != (job["hosts"][host]["region"], job["hosts"][host]["az"])
        for host, slot in slots.items()
    ):
        raise ValueError("joint host slots must match the job's physical hosts and locations")
    from powder.mc_backend import run_joint_replay

    events = seeded_joint_toggles(lib, scenario_id, slots, job["horizon_seconds"],
                                  seed, include_az=include_az)
    return run_joint_replay({**job, "events": events})


def sample_joint_replays(job: dict, lib: FaultLibrary, scenario_id: str,
                         slots: dict[str, tuple], seeds: Sequence[int], *, include_az: bool = True):
    """Run independent *whole-fleet* fault draws; aggregate without losing covariance.

    Returns ``(records, merged_timeline, composed_result)``. This deliberately
    does not multiply independent per-shard probabilities or claim an accurate
    TiDB SLA: no repair, real request latency, or PD network behavior is modeled.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("at least one distinct Monte Carlo seed is required")
    from powder.mc_backend import joint_candidate_timeline
    from .composed_availability import merge_timelines

    records = [run_seeded_joint_replay(job, lib, scenario_id, slots, seed,
                                       include_az=include_az) for seed in seeds]
    timeline = merge_timelines(tuple(joint_candidate_timeline(record) for record in records))
    selected = {shard: tuple(job["shards"][shard][index]) for shard, index in job["selected"].items()}
    return records, timeline, timeline.compose(selected)
