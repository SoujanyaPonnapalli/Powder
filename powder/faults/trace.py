"""Seeded fault traces for emulation (plan §3.5).

``generate(lib, scenario_id, slots, horizon, seed)`` produces the list of
fault events to inject into identical AWS instances, using exactly the
distributions the simulator uses:

* node-level causes per slot (a slot is one replica position, held by a
  machine class in a region). Unplanned causes are competing risks: each cause
  draws its next arrival (exponential, or Weibull with the same mean when the
  profile asks for it); the earliest one fires. A transient event takes the
  slot down for the cause's duration; all clocks restart at recovery (renewal,
  as in the simulator). A permanent event (disk / host loss / spot) loses the
  slot's data; the emulator replaces the node, and the slot's clocks restart
  after a ``spawn`` delay.
* planned downtime per slot from :mod:`planned_downtime` (independent layer;
  if the slot is already down the emulator treats it as a no-op).
* region outages per region: Poisson arrivals at the region rate with the
  region-outage duration (independent layer; network isolation of every slot
  in the region). Correlation across regions is exec #3.
* frailty: if the profile has a finite ``frailty_a``, each slot draws
  ``Z ~ Gamma(a, 1/a)`` once and all its unplanned rates are multiplied by Z.

Output (``write_jsonl``): a header line with scenario, seed, horizon and
library content hashes, then one event per line:
``{"slot", "region", "machine", "kind", "cause", "start", "duration"}``
(seconds from experiment start; ``duration`` is null for permanent events).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .convert import node_profile_rates, scenario_network_config
from .dists import from_spec
from .library import FaultLibrary
from .planned_downtime import schedule as planned_schedule
from .schema import PERMANENT_CAUSES, TRANSIENT_CAUSES, pick


@dataclass(frozen=True)
class FaultEvent:
    slot: str
    region: str
    machine: str
    kind: str          # transient | permanent | planned | region
    cause: str
    start: float
    duration: float | None


def _arrival(rate: float, shape: dict[str, Any], level: str, rng: np.random.Generator) -> float:
    if rate <= 0:
        return math.inf
    if shape.get("type") == "weibull":
        k = float(pick(shape["k"], level, "weibull_k"))
        scale = (1.0 / rate) / math.gamma(1.0 + 1.0 / k)
        return float(scale * rng.weibull(k))
    return float(rng.exponential(1.0 / rate))


def _slot_events(slot, machine_id, region, profile, horizon, level, rng, spawn) -> list[FaultEvent]:
    rates = node_profile_rates(profile, level)
    z = 1.0
    a = profile.get("frailty_a")
    if a is not None and math.isfinite(a) and a > 0:
        z = float(rng.gamma(a, 1.0 / a))
    shape = profile.get("shape", {"type": "exponential"})
    out = []
    t = 0.0
    while t < horizon:
        draws = {c: _arrival(r.rate_per_second * z, shape, level, rng) for c, r in rates.items()}
        cause = min(draws, key=draws.get)
        dt = draws[cause]
        if not math.isfinite(dt) or t + dt >= horizon:
            break
        t += dt
        if cause in TRANSIENT_CAUSES:
            d = float(rates[cause].duration.sample(rng))
            out.append(FaultEvent(slot, region, machine_id, "transient", cause, t, d))
            t += d
        elif cause in PERMANENT_CAUSES:
            out.append(FaultEvent(slot, region, machine_id, "permanent", cause, t, None))
            t += float(spawn.sample(rng))
    return out


def generate(
    lib: FaultLibrary,
    scenario_id: str,
    slots: dict[str, tuple[str, str]],
    horizon: float,
    seed: int,
    level: str | None = None,
) -> list[FaultEvent]:
    """Fault events for ``slots`` = {slot_id: (machine_id, region)} over ``horizon`` seconds."""
    s = lib.get(scenario_id, "scenario")
    lvl = level or s.get("level", "central")
    rng = np.random.default_rng(seed)
    events: list[FaultEvent] = []
    for slot in sorted(slots):
        machine_id, region = slots[slot]
        profile = lib.get(lib.get(machine_id, "machine")["fault_profile"], "node")
        spawn = from_spec(profile.get("spawn", s.get("system", {}).get("spawn", {"type": "exponential", "rate": 1 / 60})))
        events += _slot_events(slot, machine_id, region, profile, horizon, lvl, rng, spawn)
    if s.get("planned_downtime"):
        pd = lib.get(s["planned_downtime"], "planned_downtime")
        for e in planned_schedule(sorted(slots), horizon, pd, rng, level=lvl):
            m, r = slots[e.node]
            events.append(FaultEvent(e.node, r, m, "planned", "planned", e.start, e.end - e.start))
    ro = scenario_network_config(lib, scenario_id, level=lvl)
    if ro is not None:
        for region in sorted({r for _, r in slots.values()}):
            rate = ro.rate_per_second.get(region, 0.0)
            t = 0.0
            while rate > 0:
                t += float(rng.exponential(1.0 / rate))
                if t >= horizon:
                    break
                events.append(FaultEvent("*", region, "*", "region", "region", t, float(ro.duration.sample(rng))))
    events.sort(key=lambda e: (e.start, e.slot, e.kind))
    return events


def write_jsonl(path: Path | str, lib: FaultLibrary, scenario_id: str, slots, horizon, seed, events,
                level: str | None = None) -> None:
    s = lib.get(scenario_id, "scenario")
    used = {scenario_id, s.get("region_outage"), s.get("planned_downtime")} - {None}
    for m, _ in slots.values():
        used |= {m, lib.get(m)["fault_profile"]}
    header = {"type": "header", "scenario": scenario_id, "seed": seed, "horizon_s": horizon,
              "level": level or s.get("level", "central"), "slots": {k: list(v) for k, v in slots.items()},
              "library_hashes": {e: lib.content_hash(e) for e in sorted(used)}}
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for e in events:
            f.write(json.dumps(asdict(e)) + "\n")
