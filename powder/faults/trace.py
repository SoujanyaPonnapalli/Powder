"""Seeded fault traces for emulation (plan §3.5).

``generate(lib, scenario_id, slots, horizon, seed)`` produces the fault events
to inject into identical AWS instances. The node-level process is the one the
simulator runs for the same machine class (``convert.machine_node_config``):

* one transient clock per slot with hazard ``lambda_T`` (exponential, or a
  single Weibull with the same mean when the profile's shape is Weibull); when
  it fires, the cause is drawn with probability ``lambda_i / lambda_T`` and the
  slot is down for that cause's duration; the clock restarts at recovery
  (renewal), as in the simulator.
* one permanent (data-loss) clock per slot, exponential at ``lambda_P``; it
  keeps running during transient downtime (the simulator's data-loss clock
  does too). After a permanent event the emulator replaces the node; both
  clocks restart after a ``spawn`` delay.
* planned downtime per slot from :mod:`planned_downtime` (phase independent
  per node by default; ``rolling`` profiles take a whole AZ/region down
  together).
* region outages per region at the regional rate, plus *global* outages that
  hit every region at once (the separate correlated process of
  ``region_outage`` profiles).
* frailty is **opt-in** (``use_frailty=True``): each node incarnation draws
  ``Z ~ Gamma(a, 1/a)`` and its unplanned rates are multiplied by Z (a
  replacement node draws a fresh Z). The matching simulator configs come from
  ``machine_node_config(..., rate_multiplier=Z)`` with :func:`frailty_draws`.

Random streams are independent per (layer, slot/region), derived from the seed,
so adding or removing a slot does not change the other slots' events.

Output (``write_jsonl``): a header line (scenario, seed, horizon, library
content hashes), then one event per line:
``{"slot", "region", "machine", "kind", "cause", "start", "duration"}``
(seconds from experiment start; ``duration`` is null for permanent events;
``slot``/``region`` are ``"*"`` for events that hit every slot/region).
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .convert import _hazard, node_profile_rates, scenario_network_config
from .dists import from_spec
from .library import FaultLibrary
from .planned_downtime import schedule as planned_schedule
from .schema import PERMANENT_CAUSES, TRANSIENT_CAUSES


@dataclass(frozen=True)
class FaultEvent:
    slot: str
    region: str
    machine: str
    kind: str          # transient | permanent | planned | region | global
    cause: str
    start: float
    duration: float | None


def stream(seed: int, layer: str, key: str = "") -> np.random.Generator:
    """Independent, reproducible generator for one (layer, key)."""
    return np.random.default_rng([int(seed), zlib.crc32(f"{layer}:{key}".encode())])


def _frailty(profile: dict[str, Any], rng: np.random.Generator) -> float:
    a = profile.get("frailty_a")
    if a is None or not math.isfinite(a) or a <= 0:
        return 1.0
    return float(rng.gamma(a, 1.0 / a))


def frailty_draws(lib: FaultLibrary, slots: dict[str, tuple], seed: int) -> dict[str, float]:
    """First-incarnation frailty multiplier per slot (same draws as the trace)."""
    out = {}
    for slot, (machine_id, *_rest) in slots.items():
        profile = lib.get(lib.get(machine_id, "machine")["fault_profile"], "node")
        out[slot] = _frailty(profile, stream(seed, "frailty", slot))
    return out


def _slot_events(slot, machine_id, region, profile, horizon, level, seed, spawn, use_frailty) -> list[FaultEvent]:
    rng = stream(seed, "node", slot)
    frng = stream(seed, "frailty", slot)
    rates = node_profile_rates(profile, level)
    trans = [(c, r) for c, r in rates.items() if c in TRANSIENT_CAUSES and r.rate_per_second > 0]
    perm = [(c, r) for c, r in rates.items() if c in PERMANENT_CAUSES and r.rate_per_second > 0]
    lam_t = sum(r.rate_per_second for _, r in trans)
    lam_p = sum(r.rate_per_second for _, r in perm)
    p_t = np.array([r.rate_per_second for _, r in trans]) / lam_t if trans else None
    p_p = np.array([r.rate_per_second for _, r in perm]) / lam_p if perm else None
    shape = profile.get("shape", {"type": "exponential"})

    def clocks(t0: float, z: float):
        nt = t0 + (_hazard(lam_t * z, shape, level).sample(rng) if lam_t > 0 else math.inf)
        np_ = t0 + (rng.exponential(1.0 / (lam_p * z)) if lam_p > 0 else math.inf)
        return nt, np_

    z = _frailty(profile, frng) if use_frailty else 1.0
    next_t, next_p = clocks(0.0, z)
    out: list[FaultEvent] = []
    while min(next_t, next_p) < horizon:
        if next_p <= next_t:
            cause = perm[int(rng.choice(len(perm), p=p_p))][0]
            out.append(FaultEvent(slot, region, machine_id, "permanent", cause, float(next_p), None))
            restart = next_p + float(spawn.sample(rng))
            z = _frailty(profile, frng) if use_frailty else 1.0  # replacement node: fresh frailty
            next_t, next_p = clocks(restart, z)
        else:
            cause, r = trans[int(rng.choice(len(trans), p=p_t))]
            d = float(r.duration.sample(rng))
            out.append(FaultEvent(slot, region, machine_id, "transient", cause, float(next_t), d))
            next_t = next_t + d + _hazard(lam_t * z, shape, level).sample(rng)
    return out


def generate(
    lib: FaultLibrary,
    scenario_id: str,
    slots: dict[str, tuple],
    horizon: float,
    seed: int,
    level: str | None = None,
    use_frailty: bool = False,
) -> list[FaultEvent]:
    """Fault events for ``slots`` = {slot: (machine_id, region[, az])} over ``horizon`` seconds."""
    s = lib.get(scenario_id, "scenario")
    lvl = level or s.get("level", "central")
    events: list[FaultEvent] = []
    for slot in sorted(slots):
        machine_id, region = slots[slot][0], slots[slot][1]
        profile = lib.get(lib.get(machine_id, "machine")["fault_profile"], "node")
        spawn = from_spec(profile.get("spawn", s.get("system", {}).get("spawn", {"type": "exponential", "rate": 1 / 60})))
        events += _slot_events(slot, machine_id, region, profile, horizon, lvl, seed, spawn, use_frailty)
    events += scenario_events(lib, scenario_id, slots, horizon, seed, lvl)
    events.sort(key=lambda e: (e.start, e.slot, e.kind))
    return events


def scenario_events(lib: FaultLibrary, scenario_id: str, slots: dict[str, tuple], horizon: float, seed: int,
                    lvl: str) -> list[FaultEvent]:
    """Scenario-level events (planned downtime, region and global outages) for ``slots``."""
    s = lib.get(scenario_id, "scenario")
    events: list[FaultEvent] = []
    if s.get("planned_downtime"):
        pd = lib.get(s["planned_downtime"], "planned_downtime")
        groups = {slot: (v[2] if len(v) > 2 else v[1]) for slot, v in slots.items()}  # AZ if given, else region
        if pd.get("phase") == "rolling":  # groups are staggered jointly: one stream for the schedule
            planned = planned_schedule(sorted(slots), horizon, pd, stream(seed, "planned"), level=lvl, groups=groups)
        else:  # independent phases: one stream per slot, so slots do not perturb each other
            planned = [e for slot in sorted(slots)
                       for e in planned_schedule([slot], horizon, pd, stream(seed, "planned", slot), level=lvl)]
        for e in planned:
            m, r = slots[e.node][0], slots[e.node][1]
            events.append(FaultEvent(e.node, r, m, "planned", "planned", e.start, e.end - e.start))
    ro = scenario_network_config(lib, scenario_id, level=lvl)
    if ro is not None:
        for region in sorted({v[1] for v in slots.values()}):
            rng = stream(seed, "region", region)
            rate, t = ro.rate_per_second.get(region, 0.0), 0.0
            while rate > 0:
                t += float(rng.exponential(1.0 / rate))
                if t >= horizon:
                    break
                events.append(FaultEvent("*", region, "*", "region", "region", t, float(ro.duration.sample(rng))))
        rng = stream(seed, "global")
        t = 0.0
        while ro.global_rate_per_second > 0:
            t += float(rng.exponential(1.0 / ro.global_rate_per_second))
            if t >= horizon:
                break
            events.append(FaultEvent("*", "*", "*", "global", "global", t, float(ro.global_duration.sample(rng))))
    return events


def write_jsonl(path: Path | str, lib: FaultLibrary, scenario_id: str, slots, horizon, seed, events,
                level: str | None = None, use_frailty: bool = False) -> None:
    s = lib.get(scenario_id, "scenario")
    used = {scenario_id, s.get("region_outage"), s.get("planned_downtime")} - {None}
    for v in slots.values():
        used |= {v[0], lib.get(v[0])["fault_profile"]}
    header = {"type": "header", "scenario": scenario_id, "seed": seed, "horizon_s": horizon,
              "level": level or s.get("level", "central"), "use_frailty": use_frailty,
              "slots": {k: list(v) for k, v in slots.items()},
              "library_hashes": {e: lib.content_hash(e) for e in sorted(used)}}
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for e in events:
            f.write(json.dumps(asdict(e)) + "\n")
