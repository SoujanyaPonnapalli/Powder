"""Machine catalogue: >= 1000 unique machines composed from the library (exec #1b §3).

A machine is a **recipe** plus **draws**:

* recipe: ``transient`` base (a node profile with known-recoverable transient
  causes), ``disk`` (a disk component, and for population / curve-population
  components one member, i.e. a drive model), ``spot`` (``None`` for
  on-demand, else a SpotLake pool), ``region``;
* draws (seeded): each cause's rate from its rate model (a level from a
  population, an age at t = 0, a calendar offset). Spot pools share one offset
  per catalogue (pool-wide reclaim waves).

A recipe repeats only after every recipe of its tier is used, and then with
independent draws, so no two machines share all their rates. Every cause records its source entry and
member so each number traces back to a source measurement.

Consumers:

* :func:`node_config` -- the exponential approximation used by the Markov model
  and today's Monte Carlo engines: each cause's average hazard over
  ``horizon_h`` from t = 0 (exec #1b decision), planned downtime folded in.
* :func:`machine_hazards` -- the full per-cause hazards (constant or
  piecewise curves) for NHPP sampling (traces, and the simulator once it
  supports time-varying hazards).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from powder.simulation.distributions import Exponential, Mixture, PiecewiseRate, TimeVaryingHazard
from powder.simulation.node import NodeConfig

from .convert import NEVER, DEFAULT_SYSTEM, CauseRate, planned_as_cause
from .dists import from_spec
from .library import FaultLibrary
from .ratemodels import MachineRate, instantiate, machine_rate_from_dict, member_ids
from .schema import PERMANENT_CAUSES, TRANSIENT_CAUSES, pick
from .system import REBUILD_S, SYSTEM

US_REGIONS = ("us-east-1", "us-east-2", "us-west-1", "us-west-2")
#: Node profiles whose transient causes may seed catalogue machines: recoverability known, and no
#: overlap with the weekly planned window (cluster/ford10-google-storage counts rollout restarts).
TRANSIENT_BASES = ("cloud/aws-ondemand-nvme", "cluster/ford10-google-storage-ge15min")
#: Disk components usable in recipes (all count drives removed as failed: permanent).
DISK_COMPONENTS = ("disk/backblaze-models-age", "disk/backblaze-models", "disk/sg07-age", "disk/sg07-systems",
                   "disk/pooled", "disk/schroeder-gibson-fast07", "disk/vishwanath10", "disk/ford10")
SPOT_COMPONENT = "spot/spotlake-pools"
DEFAULT_HORIZON_H = 30 * 24.0


def stream(seed: int, *parts: str) -> np.random.Generator:
    return np.random.default_rng([seed & 0xFFFFFFFF] + [zlib.crc32(p.encode()) for p in parts])


@dataclass(frozen=True)
class Recipe:
    transient: str
    disk: str
    disk_member: str | None
    spot_pool: str | None
    region: str

    @property
    def key(self) -> str:
        return "|".join(str(x) for x in (self.transient, self.disk, self.disk_member, self.spot_pool, self.region))


def disk_options(lib: FaultLibrary) -> list[tuple[str, str | None]]:
    """(component, member) pairs; Backblaze models appear once (age curve if available, else constant)."""
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for cid in DISK_COMPONENTS:
        if cid not in lib.entries:
            continue
        for m in member_ids(lib.get(cid).get("rate_model")):
            if m is not None:
                if m in seen:
                    continue
                seen.add(m)
            out.append((cid, m))
    return out


def spot_pools(lib: FaultLibrary) -> list[str]:
    if SPOT_COMPONENT not in lib.entries:
        return []
    return [m["id"] for m in lib.get(SPOT_COMPONENT)["rate_model"]["members"]]


def recipe_capacity(lib: FaultLibrary) -> dict[str, int]:
    """Distinct recipes per tier (machines beyond this differ only by their draws)."""
    d, s = len(disk_options(lib)), len(spot_pools(lib))
    od = len(TRANSIENT_BASES) * d * len(US_REGIONS)
    sp = len(TRANSIENT_BASES) * d * s
    return {"disk_options": d, "spot_pools": s, "ondemand_recipes": od, "spot_recipes": sp, "total": od + sp}


def all_recipes(lib: FaultLibrary, spot: bool) -> list[Recipe]:
    disks = disk_options(lib)
    if spot:
        return [Recipe(t, c, m, p, p.split(":", 1)[0]) for t in TRANSIENT_BASES for c, m in disks for p in spot_pools(lib)]
    return [Recipe(t, c, m, None, r) for t in TRANSIENT_BASES for c, m in disks for r in US_REGIONS]


def sample_recipes(lib: FaultLibrary, n: int, seed: int, spot_fraction: float) -> list[tuple[Recipe, int]]:
    """(recipe, draw) pairs. Each tier cycles through a shuffled list of all its recipes, so a recipe
    repeats (with an independent draw: a different Ford cell level, drive age, ...) only once every
    recipe of the tier has been used."""
    n_spot = int(round(n * spot_fraction)) if spot_pools(lib) else 0
    out: list[tuple[Recipe, int]] = []
    for spot, count in ((False, n - n_spot), (True, n_spot)):
        pool = all_recipes(lib, spot)
        if count and not pool:
            raise ValueError("no recipes for tier " + ("spot" if spot else "on-demand"))
        order = stream(seed, "recipes", str(spot)).permutation(len(pool)) if pool else []
        out += [(pool[order[i % len(pool)]], i // len(pool)) for i in range(count)]
    return out


def _source_of(entry: dict[str, Any]) -> list[str]:
    return list(entry.get("derived_from", [])) or [entry["id"]]


def build_machine(lib: FaultLibrary, recipe: Recipe, draw: int, idx: int, seed: int,
                  pool_offsets: dict[str, float]) -> dict[str, Any]:
    base = lib.get(recipe.transient, "node")
    rng = stream(seed, "machine", recipe.key, str(draw))
    causes: dict[str, Any] = {}
    durations: dict[str, Any] = {}
    for c in TRANSIENT_CAUSES:
        spec = base["causes"].get(c)
        if not spec or float(spec["rate_per_hour"]["central"]) <= 0:
            continue
        mr = instantiate(spec.get("rate_model"), spec["rate_per_hour"], rng)
        causes[c] = {**mr.to_dict(), "source": recipe.transient}
        durations[c] = pick(spec["duration"], "central", "duration")
    disk = lib.get(recipe.disk)
    mr = instantiate(disk.get("rate_model"), disk["rate_per_hour"], rng, member_id=recipe.disk_member)
    causes["disk"] = {**mr.to_dict(), "source": recipe.disk}
    host = base["causes"].get("hw_host_loss")
    if host and float(host["rate_per_hour"]["central"]) > 0:
        causes["hw_host_loss"] = {**instantiate(host.get("rate_model"), host["rate_per_hour"], rng).to_dict(),
                                  "source": recipe.transient}
    price: dict[str, Any]
    if recipe.spot_pool:
        comp = lib.get(SPOT_COMPONENT)
        mr = instantiate(comp["rate_model"], comp["rate_per_hour"], rng, member_id=recipe.spot_pool,
                         offset_h=pool_offsets[recipe.spot_pool])
        causes["spot_interrupt"] = {**mr.to_dict(), "source": SPOT_COMPONENT}
        mem = next(m for m in comp["rate_model"]["members"] if m["id"] == recipe.spot_pool)
        price = {"route": "native", "per_hour": mem.get("spot_price_p50"), "source": "spotlake-aws#pools"}
    else:
        price = {"route": "assigned"}
    return {"id": f"m{idx:05d}", "recipe": recipe.__dict__, "recipe_key": recipe.key, "draw": draw,
            "region": recipe.region,
            "tier": "spot" if recipe.spot_pool else "ondemand", "causes": causes, "durations": durations,
            "price": price}


def machine_rates(machine: dict[str, Any]) -> dict[str, MachineRate]:
    return {c: machine_rate_from_dict(d) for c, d in machine["causes"].items()}


def horizon_rates(machine: dict[str, Any], horizon_h: float = DEFAULT_HORIZON_H) -> dict[str, float]:
    """Average hazard per hour of each cause over [0, horizon] (the Markov approximation)."""
    return {c: mr.mean_rate(horizon_h) for c, mr in machine_rates(machine).items()}


def reliability_score(machine: dict[str, Any], horizon_h: float = DEFAULT_HORIZON_H, rebuild_s: float = REBUILD_S) -> float:
    """Expected unavailable hours per node-year (node-level causes), as assign.reliability_score."""
    r = horizon_rates(machine, horizon_h)
    down_h = 0.0
    for c, lam in r.items():
        if c in TRANSIENT_CAUSES:
            down_h += lam * from_spec(machine["durations"][c]).mean / 3600.0
        elif c in PERMANENT_CAUSES:
            down_h += lam * rebuild_s / 3600.0
    return down_h * 8766.0


def piecewise_rate(mr: MachineRate) -> PiecewiseRate:
    """A machine's curve hazard in simulator units (seconds)."""
    h = mr.hazard
    start = mr.draws.get("start_h")
    age_start = None
    if mr.clock == "age" and start and start.get("type") == "weights":
        age_start = ([e * 3600.0 for e in start["edges_h"]], list(start["weights"]))
    return PiecewiseRate([e * 3600.0 for e in h.edges], [r / 3600.0 for r in h.rates], h.periodic,
                         h.offset * 3600.0, age_start)


def time_varying_hazard(machine: dict[str, Any], causes: Iterable[str], horizon_h: float = DEFAULT_HORIZON_H,
                        extra_constant_per_s: float = 0.0) -> TimeVaryingHazard | None:
    parts, const = [], extra_constant_per_s
    for c, mr in machine_rates(machine).items():
        if c not in causes:
            continue
        if mr.kind == "constant":
            const += mr.rate_per_hour / 3600.0
        else:
            parts.append(piecewise_rate(mr))
    if not parts and const <= 0:
        return None
    return TimeVaryingHazard(parts, const, mean_horizon=horizon_h * 3600.0)


def node_config(machine: dict[str, Any], *, horizon_h: float = DEFAULT_HORIZON_H, cost_per_hour: float | None = None,
                planned: Iterable[CauseRate] = (), system: dict[str, Any] | None = None,
                time_varying: bool = False) -> NodeConfig:
    """A catalogue machine as a NodeConfig.

    Default: the exponential approximation (each cause's average hazard over
    ``horizon_h``) for the Markov model and constant-rate engines.
    ``time_varying=True``: failure and data-loss clocks are the machine's full
    hazards (:class:`TimeVaryingHazard`, sampled from the current simulation
    time; age clocks restart with a fresh age on replacement) for the Python
    simulator. Planned downtime stays a constant-rate transient cause; recovery
    durations are mixed by horizon-average weights in both modes.
    """
    sysd = {**DEFAULT_SYSTEM, **{k: v for k, v in (system or SYSTEM).items() if k != "source"}}
    planned = list(planned)
    r = horizon_rates(machine, horizon_h)
    trans = [(r[c] / 3600.0, from_spec(machine["durations"][c])) for c in TRANSIENT_CAUSES if r.get(c, 0) > 0]
    trans += [(p.rate_per_second, p.duration) for p in planned if p.rate_per_second > 0]
    lam_t = sum(x for x, _ in trans)
    lam_p = sum(r.get(c, 0.0) for c in PERMANENT_CAUSES) / 3600.0
    if not trans:
        recovery = from_spec({"type": "constant", "value": 0.0})
    elif len(trans) == 1:
        recovery = trans[0][1]
    else:
        recovery = Mixture([x for x, _ in trans], [d for _, d in trans])
    price = machine["price"].get("per_hour") if cost_per_hour is None else cost_per_hour
    fail = Exponential(lam_t) if lam_t > 0 else NEVER
    loss = Exponential(lam_p) if lam_p > 0 else NEVER
    if time_varying:
        fail = time_varying_hazard(machine, TRANSIENT_CAUSES, horizon_h,
                                   sum(p.rate_per_second for p in planned)) or NEVER
        loss = time_varying_hazard(machine, PERMANENT_CAUSES, horizon_h) or NEVER
    return NodeConfig(
        region=machine["region"], cost_per_hour=float(price or 0.0),
        failure_dist=fail, recovery_dist=recovery,
        data_loss_dist=loss,
        log_replay_rate_dist=from_spec(sysd["log_replay_rate_dist"]),
        snapshot_download_time_dist=from_spec(sysd["snapshot_download_time_dist"]),
        spawn_dist=from_spec(sysd["spawn"]),
    )


def assign_catalogue_prices(lib: FaultLibrary, machines: list[dict[str, Any]], rhos=(0.0, 1.0), seed: int = 0,
                            horizon_h: float = DEFAULT_HORIZON_H) -> dict[str, Any]:
    """On-demand machines: price from the 8 vCPU x86 distribution (Gaussian copula, plan §3.4), x the
    deployment type's region factor. Spot machines keep their pool's SpotLake price."""
    from .assign import assign_prices
    dist = next((d for d in lib.query(kind="price") if d.get("price_kind") == "price_distribution" and d.get("default")), None)
    if dist is None:
        return {}
    rf_path = lib.root / "derived/region_factors.json"
    factors = json.loads(rf_path.read_text())["factors"].get("i4i.2xlarge", {}) if rf_path.exists() else {}
    od = {m["id"]: reliability_score(m, horizon_h) for m in machines if m["tier"] == "ondemand"}
    out = {}
    for rho in rhos:
        a = assign_prices(od, dist["points"], rho, seed)
        out[f"rho{rho:g}"] = {"rho": rho, "seed": seed, "distribution": dist["id"], "realised_spearman": a.realised_spearman}
        for m in machines:
            if m["id"] in a.prices:
                m["price"][f"rho{rho:g}"] = a.prices[m["id"]] * factors.get(m["region"], 1.0)
    return out


def generate(lib: FaultLibrary, n: int = 1000, seed: int = 0, spot_fraction: float = 0.3,
             horizon_h: float = DEFAULT_HORIZON_H) -> dict[str, Any]:
    recipes = sample_recipes(lib, n, seed, spot_fraction)
    pools = sorted({r.spot_pool for r, _ in recipes if r.spot_pool})
    comp = lib.get(SPOT_COMPONENT) if pools else None
    spans = {m["id"]: m["edges_h"][-1] for m in comp["rate_model"]["members"]} if comp else {}
    offsets = {p: float(stream(seed, "pool", p).uniform(0, spans[p])) for p in pools}
    machines = [build_machine(lib, r, d, i, seed, offsets) for i, (r, d) in enumerate(recipes)]
    prices = assign_catalogue_prices(lib, machines, seed=seed, horizon_h=horizon_h)
    for m in machines:
        m["markov_rate_per_hour"] = horizon_rates(m, horizon_h)
    hashes = {eid: lib.content_hash(eid) for eid in sorted({*TRANSIENT_BASES, *DISK_COMPONENTS, SPOT_COMPONENT})
              if eid in lib.entries}
    return {"kind": "machine_catalogue", "seed": seed, "n": n, "spot_fraction": spot_fraction,
            "horizon_h": horizon_h, "capacity": recipe_capacity(lib), "pool_offsets_h": offsets,
            "price_assignments": prices, "library_hashes": hashes, "system": SYSTEM, "machines": machines}


def validate(lib: FaultLibrary, cat: dict[str, Any]) -> list[str]:
    errs = []
    keys = [(m["recipe_key"], m["draw"]) for m in cat["machines"]]
    if len(set(keys)) != len(keys):
        errs.append("duplicate (recipe, draw)")
    vecs = [tuple(sorted((c, round(v, 15)) for c, v in m["markov_rate_per_hour"].items())) + (m["region"],)
            for m in cat["machines"]]
    if len(set(vecs)) != len(vecs):
        errs.append(f"{len(vecs) - len(set(vecs))} machines have identical rates and region")
    for m in cat["machines"]:
        for c, d in m["causes"].items():
            src = lib.get(d["source"])
            if "recoverability-unknown" in src.get("tags", []):
                errs.append(f"{m['id']}.{c}: source {src['id']} has unknown recoverability")
            mr = machine_rate_from_dict(d)
            if mr.mean_rate(cat["horizon_h"]) < 0:
                errs.append(f"{m['id']}.{c}: negative rate")
        if m["tier"] == "spot" and "spot_interrupt" not in m["causes"]:
            errs.append(f"{m['id']}: spot machine without spot_interrupt")
    return errs


def write(cat: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(cat, separators=(",", ":"), sort_keys=True).encode()
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:  # mtime=0: reproducible bytes
        fh.write(data)


def load(path: Path | str) -> dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def digest(cat: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cat, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- traces (emulation)

def _slot_events(machine: dict[str, Any], slot: str, horizon: float, seed: int, spawn) -> list:
    from .trace import FaultEvent
    import math
    rng = stream(seed, "trace-node", slot)
    rates = machine_rates(machine)
    clocks: dict[str, Any] = {}
    for c, mr in rates.items():
        if mr.kind == "constant":
            clocks[c] = mr.rate_per_hour / 3600.0
        else:
            clocks[c] = piecewise_rate(mr)

    def nxt(c: str, now: float) -> float:
        k = clocks[c]
        if isinstance(k, float):
            return now + rng.exponential(1.0 / k) if k > 0 else math.inf
        return k.next_event(now, rng)

    trans = [c for c in rates if c in TRANSIENT_CAUSES]
    perm = [c for c in rates if c in PERMANENT_CAUSES]
    durs = {c: from_spec(machine["durations"][c]) for c in trans}
    out = []

    def next_of(causes, now):
        best = (math.inf, None)
        for c in causes:
            t = nxt(c, now)
            if t < best[0]:
                best = (t, c)
        return best

    t_next, c_t = next_of(trans, 0.0)
    p_next, c_p = next_of(perm, 0.0)
    while min(t_next, p_next) < horizon:
        if p_next <= t_next:
            out.append(FaultEvent(slot, machine["region"], machine["id"], "permanent", c_p, float(p_next), None))
            restart = p_next + float(spawn.sample(rng))
            for c, k in clocks.items():  # replacement unit: fresh age on age clocks
                if not isinstance(k, float):
                    clocks[c] = k.respawned(restart, rng)
            t_next, c_t = next_of(trans, restart)
            p_next, c_p = next_of(perm, restart)
        else:
            d = float(durs[c_t].sample(rng))
            out.append(FaultEvent(slot, machine["region"], machine["id"], "transient", c_t, float(t_next), d))
            t_next, c_t = next_of(trans, t_next + d)
    return out


def trace(lib: FaultLibrary, cat: dict[str, Any], slots: dict[str, str], horizon_s: float, seed: int,
          scenario_id: str | None = "scenario/canonical-central") -> list:
    """Fault events for ``slots`` = {slot: catalogue machine id} (emulation).

    Node events follow each machine's full hazards (constant or piecewise, the
    same objects as ``node_config(time_varying=True)``); planned downtime and
    region/global outages come from ``scenario_id`` exactly as in
    :func:`powder.faults.trace.generate`. Different seeds give different
    timings from the same distributions.
    """
    from .trace import scenario_events
    by_id = {m["id"]: m for m in cat["machines"]}
    spawn = from_spec(cat.get("system", SYSTEM)["spawn"])
    events = []
    for slot in sorted(slots):
        events += _slot_events(by_id[slots[slot]], slot, horizon_s, seed, spawn)
    if scenario_id:
        s = lib.get(scenario_id, "scenario")
        events += scenario_events(lib, scenario_id, {k: (v, by_id[v]["region"]) for k, v in slots.items()},
                                  horizon_s, seed, s.get("level", "central"))
    events.sort(key=lambda e: (e.start, e.slot, e.kind))
    return events
