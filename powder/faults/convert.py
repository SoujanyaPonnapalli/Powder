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


def transient_and_permanent(profile: dict[str, Any], level: str = "central"):
    """(lambda_T, recovery distribution, lambda_P), rates per second."""
    rates = node_profile_rates(profile, level)
    transient = [r for c, r in rates.items() if c in TRANSIENT_CAUSES and r.rate_per_second > 0]
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
) -> NodeConfig:
    """Build the NodeConfig for one machine class in one region."""
    machine = lib.get(machine_id, "machine")
    profile = lib.get(machine["fault_profile"], "node")
    sysd = {**DEFAULT_SYSTEM, **(system or {})}
    lam_t, recovery, lam_p = transient_and_permanent(profile, level)
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


def scenario_node_configs(
    lib: FaultLibrary, scenario_id: str, *, engine: bool = False, level: str | None = None
) -> dict[tuple[str, str], NodeConfig]:
    """NodeConfig for every (machine, region) pair in a scenario."""
    s = lib.get(scenario_id, "scenario")
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
            )
    return out


@dataclass(frozen=True)
class RegionOutageModel:
    """Per-region outage hazards plus one shared duration distribution."""

    rate_per_second: dict[str, float]
    duration: Distribution
    source_regions: dict[str, str]

    def as_network_config(self, rule: str = "max", engine: bool = False) -> NetworkConfig:
        """Collapse to today's single-distribution NetworkConfig.

        The current engines take one outage distribution for all regions.
        ``rule='max'`` (default) uses the largest regional rate, which is
        conservative; ``'mean'`` uses the average. Per-region outage specs in
        the engines are a planned follow-up (plan Step F).
        """
        rates = list(self.rate_per_second.values())
        rate = max(rates) if rule == "max" else sum(rates) / len(rates)
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
    return RegionOutageModel(rates, from_spec(pick(ro["duration"], lvl, "duration")), src)
