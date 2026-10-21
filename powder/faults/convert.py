"""Turn library entries into simulator objects.

* :func:`node_profile_rates` -- per-cause rates/durations at a fault level
* :func:`machine_node_config` -- one machine class in one region -> NodeConfig
* :func:`scenario_node_configs` -- every (machine, region) of a scenario
* :func:`scenario_network_config` -- region-outage model for a scenario

Assembly (plan §7.9, AWS + local NVMe):

    lambda_T = lambda_hw_reboot + lambda_sw + lambda_pause      (data kept)
    lambda_P = lambda_disk + lambda_hw_host_loss + lambda_spot  (data lost)
    recovery = mixture of transient-cause durations, weights lambda_i/lambda_T

With ``engine=True`` every distribution is mapped to a family the Rust
engine accepts (same mean and median), see ``dists.engine_compatible``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from powder.simulation.distributions import Constant, Distribution, Exponential, Mixture, Weibull
from powder.simulation.network import NetworkConfig
from powder.simulation.node import NodeConfig

from . import prices
from .dists import engine_compatible, from_spec
from .library import FaultLibrary, LibraryError
from .schema import PERMANENT_CAUSES, TRANSIENT_CAUSES, pick

SECONDS_PER_HOUR = 3600.0
#: Defaults for the non-fault NodeConfig fields; they match the legacy
#: ``node_config_for`` in notebooks/raft_markov_quality_benchmark.py.
DEFAULT_SYSTEM = {
    "log_replay_rate_dist": {"type": "constant", "value": 1e6},
    "snapshot_download_time_dist": {"type": "constant", "value": 1.0},
    "spawn": {"type": "exponential", "rate": 1.0 / 60.0},
}
NEVER = Constant(math.inf)


@dataclass(frozen=True)
class CauseRate:
    cause: str
    rate_per_second: float
    duration: Distribution | None


def node_profile_rates(profile: dict[str, Any], level: str = "central") -> dict[str, CauseRate]:
    """Per-cause rate (per second) and duration distribution at ``level``."""
    out = {}
    for cause, spec in profile["causes"].items():
        rate_h = float(pick(spec["rate_per_hour"], level, "rate"))
        dur = None
        if "duration" in spec and spec["duration"] is not None:
            dur = from_spec(pick(spec["duration"], level, "duration"))
        out[cause] = CauseRate(cause, rate_h / SECONDS_PER_HOUR, dur)
    return out


def _hazard(rate: float, shape: dict[str, Any], level: str) -> Distribution:
    if rate <= 0:
        return NEVER
    if shape.get("type", "exponential") == "weibull":
        k = float(pick(shape["k"], level, "weibull_k"))
        return Weibull(k, (1.0 / rate) / math.gamma(1.0 + 1.0 / k))
    return Exponential(rate)


def transient_and_permanent(profile: dict[str, Any], level: str = "central",
                            rate_multiplier: float = 1.0, extra_transient: list[CauseRate] | None = None):
    """(lambda_T, recovery distribution, lambda_P), rates per second.

    ``rate_multiplier`` scales the profile's own rates (frailty);
    ``extra_transient`` adds scenario-level transient processes (planned
    downtime folded in as an exponential cause), which are not scaled.
    """
    rates = node_profile_rates(profile, level)
    rates = {c: CauseRate(c, r.rate_per_second * rate_multiplier, r.duration) for c, r in rates.items()}
    transient = [r for c, r in rates.items() if c in TRANSIENT_CAUSES and r.rate_per_second > 0]
    transient += [r for r in (extra_transient or []) if r.rate_per_second > 0]
    lam_t = sum(r.rate_per_second for r in transient)
    lam_p = sum(r.rate_per_second for c, r in rates.items() if c in PERMANENT_CAUSES)
    if not transient:
        recovery: Distribution = Constant(0.0)
    elif len(transient) == 1:
        recovery = transient[0].duration
    else:
        recovery = Mixture([r.rate_per_second for r in transient], [r.duration for r in transient])
    return lam_t, recovery, lam_p


def machine_node_config(
    lib: FaultLibrary,
    machine_id: str,
    region: str,
    *,
    level: str = "central",
    cost_per_hour: float = 0.0,
    system: dict[str, Any] | None = None,
    engine: bool = False,
    rate_multiplier: float = 1.0,
    extra_transient: list[CauseRate] | None = None,
) -> NodeConfig:
    """Build the NodeConfig for one machine class in one region.

    ``rate_multiplier``: per-node frailty Z (see trace.frailty_draws).
    ``extra_transient``: scenario-level transient processes (planned downtime).
    """
    machine = lib.get(machine_id, "machine")
    profile = lib.get(machine["fault_profile"], "node")
    sysd = {**DEFAULT_SYSTEM, **(system or {})}
    lam_t, recovery, lam_p = transient_and_permanent(profile, level, rate_multiplier, extra_transient)
    spawn = from_spec(profile.get("spawn", sysd["spawn"]))
    fix = engine_compatible if engine else (lambda d: d)
    return NodeConfig(
        region=region,
        cost_per_hour=float(cost_per_hour),
        failure_dist=fix(_hazard(lam_t, profile.get("shape", {}), level)),
        recovery_dist=fix(recovery),
        data_loss_dist=Exponential(lam_p) if lam_p > 0 else NEVER,
        log_replay_rate_dist=fix(from_spec(sysd["log_replay_rate_dist"])),
        snapshot_download_time_dist=fix(from_spec(sysd["snapshot_download_time_dist"])),
        spawn_dist=fix(spawn),
    )


def planned_as_cause(lib: FaultLibrary, scenario: dict[str, Any], level: str) -> list[CauseRate]:
    """The scenario's planned downtime as an exponential transient cause.

    Rate 1/E[W], duration = the planned-duration distribution. This is the
    Markov/ILP/engine approximation of the (quasi-periodic) planned process
    that traces inject exactly; long-run downtime matches E[D]/E[W].
    """
    if not scenario.get("planned_downtime"):
        return []
    pd = lib.get(scenario["planned_downtime"], "planned_downtime")
    period = from_spec(pd["period"])
    return [CauseRate("planned", 1.0 / period.mean, from_spec(pick(pd["duration"], level, "duration")))]


def scenario_node_configs(
    lib: FaultLibrary, scenario_id: str, *, engine: bool = False, level: str | None = None,
    price_assignment: str | None = None, include_planned: bool = True,
) -> dict[tuple[str, str], NodeConfig]:
    """NodeConfig for every (machine, region) pair in a scenario.

    ``price_assignment`` overrides the scenario's assignment (e.g. to sweep
    the 20 rho = 0 seeds without a scenario file per seed).
    ``include_planned`` folds the scenario's planned downtime into each node's
    transient process (default), so simulation sees the planned downtime that
    emulation traces inject.
    """
    s = lib.get(scenario_id, "scenario")
    if price_assignment is not None:
        s = {**s, "price_assignment": price_assignment,
             "price_policy": s["price_policy"] if s.get("price_policy") != "none" else "latest"}
    lvl = level or s.get("level", "central")
    out = {}
    for mid in s["machines"]:
        m_regions = lib.get(mid, "machine").get("regions")
        for region in s["deployment"]["regions"]:
            if m_regions and region not in m_regions:
                continue  # machine class not offered in this region (e.g. a regional spot pool)
            price = prices.price_per_hour(lib, s, mid, region)
            out[(mid, region)] = machine_node_config(
                lib, mid, region, level=lvl, cost_per_hour=price,
                system=s.get("system"), engine=engine,
                extra_transient=planned_as_cause(lib, s, lvl) if include_planned else None,
            )
    return out


@dataclass(frozen=True)
class RegionOutageModel:
    """Per-region outage hazards, one shared duration distribution, and a
    separate global process (outages that hit every region at once)."""

    rate_per_second: dict[str, float]
    duration: Distribution
    source_regions: dict[str, str]
    global_rate_per_second: float = 0.0
    global_duration: Distribution | None = None

    def as_network_config(self, rule: str = "max", engine: bool = False) -> NetworkConfig:
        """Collapse to today's single-distribution NetworkConfig.

        The current engines take one outage distribution for all regions.
        ``rule='max'`` (default) uses the largest regional rate, which is
        conservative; ``'mean'`` uses the average. Per-region outage specs in
        the engines are a planned follow-up (plan Step F).
        """
        rates = list(self.rate_per_second.values())
        # The engines have no correlated (all-region) outages: the global
        # process is added to the per-region rate, which keeps each region's
        # marginal outage rate right but loses the correlation.
        rate = (max(rates) if rule == "max" else sum(rates) / len(rates)) + self.global_rate_per_second
        dur = engine_compatible(self.duration) if engine else self.duration
        return NetworkConfig(
            outage_dist=Exponential(rate) if rate > 0 else NEVER,
            outage_duration_dist=dur,
            regions=sorted(self.rate_per_second),
        )


def scenario_network_config(lib: FaultLibrary, scenario_id: str, level: str | None = None) -> RegionOutageModel | None:
    s = lib.get(scenario_id, "scenario")
    if not s.get("region_outage"):
        return None
    lvl = level or s.get("level", "central")
    ro = lib.get(s["region_outage"], "region_outage")
    rates, src = {}, {}
    for region in s["deployment"]["regions"]:
        if region not in ro["regions"]:
            raise LibraryError(f"{ro['id']} has no outage rate for region {region}")
        r = ro["regions"][region]
        rates[region] = float(pick(r["rate_per_hour"], lvl, "rate")) / SECONDS_PER_HOUR
        src[region] = r.get("azure_region", region)
    dur = from_spec(pick(ro["duration"], lvl, "duration"))
    g = ro.get("global")
    g_rate = float(pick(g["rate_per_hour"], lvl, "rate")) / SECONDS_PER_HOUR if g else 0.0
    g_dur = from_spec(pick(g.get("duration", ro["duration"]), lvl, "duration")) if g else None
    return RegionOutageModel(rates, dur, src, g_rate, g_dur)
