"""Tests for the fault + price library (powder.faults).

NumPy-only: these run without SciPy or the Rust binary. Under pytest they
are collected normally; ``python scripts/fault_library/run_tests.py`` runs
them without pytest.
"""

from __future__ import annotations

import json
import math

import numpy as np

from powder.faults import FaultLibrary, scenario_node_configs
from powder.faults.convert import machine_node_config, transient_and_permanent
from powder.faults.dists import engine_compatible, from_spec, to_spec, weibull_from_mean_median
from powder.faults.library import LibraryError
from powder.faults.schema import CAUSES, pick, validate_entry
from powder.mc_jobs import distribution_spec, node_config_spec
from powder.simulation.distributions import (
    Empirical, Exponential, LogNormal, Mixture, Weibull, days, minutes, normal_cdf, normal_ppf,
)

LIB = FaultLibrary.load()


# -- library-wide invariants -------------------------------------------------

def test_library_validates():
    assert LIB.validate() == []


def test_every_node_profile_converts_for_both_engines():
    for p in LIB.query(kind="node"):
        mid = f"__probe__{p['id']}"
        LIB.entries[mid] = {"id": mid, "kind": "machine", "fault_profile": p["id"],
                            "price": {"route": "none"}}
        try:
            for level in ("optimistic", "central", "pessimistic"):
                cfg = machine_node_config(LIB, mid, "us-east-1", level=level, engine=True)
                node_config_spec(cfg)  # raises if the Rust engine can't take it
                machine_node_config(LIB, mid, "us-east-1", level=level, engine=False)
        finally:
            del LIB.entries[mid]


def test_every_scenario_builds():
    for s in LIB.query(kind="scenario"):
        cfgs = scenario_node_configs(LIB, s["id"], engine=True)
        assert len(cfgs) == len(s["machines"]) * len(s["deployment"]["regions"])


def test_no_unconfirmed_citations():
    for e in LIB.entries.values():
        for ref in e.get("derived_from", []):
            if "#" in ref:
                assert LIB.measurement(ref)["status"] != "U", (e["id"], ref)


def test_triples_ordered():
    for p in LIB.query(kind="node"):
        for c, spec in p["causes"].items():
            assert c in CAUSES
            t = spec["rate_per_hour"]
            assert t["low"] <= t["central"] <= t["high"], (p["id"], c)


# -- legacy reproduction -----------------------------------------------------

def _legacy_reference(mtbf_s, rec_s, mttdl_s, cost):
    """Mirror of notebooks/raft_markov_quality_benchmark.node_config_for."""
    return {
        "region": "us-east", "cost_per_hour": cost,
        "failure_dist": {"type": "exponential", "rate": 1.0 / mtbf_s},
        "recovery_dist": {"type": "exponential", "rate": 1.0 / rec_s},
        "data_loss_dist": {"type": "exponential", "rate": 1.0 / mttdl_s},
        "log_replay_rate_dist": {"type": "constant", "value": 1e6},
        "snapshot_download_time_dist": {"type": "constant", "value": 1.0},
        "spawn_dist": {"type": "exponential", "rate": 1.0 / 60.0},
    }


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12)
    return a == b


def test_legacy_scenario_reproduces_vm_profiles():
    ref = {
        "machine/legacy-standard": _legacy_reference(days(30), minutes(20), days(365 * 3), 0.10),
        "machine/legacy-spot": _legacy_reference(days(30), minutes(20), days(1), 0.03),
        "machine/legacy-unreliable": _legacy_reference(days(30), minutes(20), days(365), 0.05),
    }
    cfgs = scenario_node_configs(LIB, "scenario/legacy", engine=True)
    for (mid, region), cfg in cfgs.items():
        assert _close(node_config_spec(cfg), ref[mid]), mid


def test_legacy_matches_notebook_when_importable():
    try:
        from notebooks.raft_markov_quality_benchmark import VM_PROFILES, node_config_for
    except Exception:  # SciPy or the notebook deps are missing
        return
    cfgs = scenario_node_configs(LIB, "scenario/legacy", engine=True)
    for p in VM_PROFILES:
        mid = f"machine/legacy-{p.name.lower()}"
        assert _close(node_config_spec(cfgs[(mid, "us-east")]), node_config_spec(node_config_for(p)))


# -- distributions -----------------------------------------------------------

def test_normal_ppf_inverts_cdf():
    for p in (1e-8, 1e-3, 0.02, 0.3, 0.5, 0.77, 0.99, 1 - 1e-7):
        assert abs(normal_cdf(normal_ppf(p)) - p) < 1e-8 * max(1.0, 1.0 / p)


def test_spec_round_trip():
    specs = [
        {"type": "exponential", "rate": 0.5},
        {"type": "weibull", "shape": 0.7, "scale": 100.0},
        {"type": "lognormal", "mu": 5.0, "sigma": 1.2},
        {"type": "mixture", "components": [
            {"weight": 0.25, "dist": {"type": "constant", "value": 3.0}},
            {"weight": 0.75, "dist": {"type": "exponential", "rate": 0.1}}]},
        {"type": "empirical", "x": [10.0, 100.0, 1000.0], "F": [0.2, 0.6, 0.9], "log_x": True,
         "tail": {"type": "lognormal", "mu": 4.5, "sigma": 1.5}},
    ]
    for s in specs:
        again = to_spec(from_spec(s))
        assert json.dumps(again, sort_keys=True) == json.dumps(to_spec(from_spec(again)), sort_keys=True)


def test_empirical_quantiles_and_sampling():
    tail = LogNormal(math.log(300), 1.3)
    e = Empirical([10, 60, 600, 3600], [0.1, 0.4, 0.8, 0.95], tail=tail)
    assert abs(e.ppf(0.8) - 600) < 1e-6 and abs(e.cdf(60) - 0.4) < 1e-12
    for u in (0.01, 0.05, 0.97, 0.999):  # continuity / monotone into the tails
        assert e.ppf(u) > 0
    assert e.ppf(0.05) < 10 < e.ppf(0.11) and e.ppf(0.96) > 3600
    rng = np.random.default_rng(7)
    s = np.array([e.sample(rng) for _ in range(60_000)])
    assert abs((s <= 600).mean() - 0.8) < 0.01
    assert abs(s.mean() / e.mean - 1) < 0.05


def test_weibull_from_mean_median():
    w = Weibull(0.75, 1000.0)
    fit = weibull_from_mean_median(w.mean, w.ppf(0.5))
    assert isinstance(fit, Weibull) and abs(fit.shape - 0.75) < 1e-6 and abs(fit.scale / 1000 - 1) < 1e-6


def test_engine_compatible_preserves_mean_and_median():
    ln = LogNormal(math.log(900), 1.0)
    fit = engine_compatible(ln)
    distribution_spec(fit)
    assert abs(fit.mean / ln.mean - 1) < 1e-9
    assert abs(fit.ppf(0.5) / ln.ppf(0.5) - 1) < 1e-6


def test_mixture_recovery_is_rate_weighted():
    p = {"causes": {
        "hw_reboot": {"rate_per_hour": {"low": 1, "central": 1, "high": 1},
                      "duration": {"type": "constant", "value": 100.0}},
        "sw": {"rate_per_hour": {"low": 3, "central": 3, "high": 3},
               "duration": {"type": "constant", "value": 20.0}},
        "disk": {"rate_per_hour": {"low": 0.5, "central": 0.5, "high": 0.5}}}}
    lam_t, rec, lam_p = transient_and_permanent(p)
    assert math.isclose(lam_t, 4 / 3600) and math.isclose(lam_p, 0.5 / 3600)
    assert isinstance(rec, Mixture) and math.isclose(rec.mean, (100 + 3 * 20) / 4)


def test_pick_levels():
    t = {"low": 1, "central": 2, "high": 3}
    assert pick(t, "pessimistic", "rate") == 3 and pick(t, "optimistic", "rate") == 1
    assert pick(t, "pessimistic", "weibull_k") == 1 and pick(t, "central") == 2


def test_schema_rejects_bad_entries():
    assert validate_entry({"id": "x", "kind": "node", "tier": "cloud", "causes": {
        "hw_reboot": {"rate_per_hour": {"low": 2, "central": 1, "high": 3},
                      "duration": {"type": "constant", "value": 1}}}})
    assert validate_entry({"id": "x", "kind": "node", "tier": "cloud", "causes": {
        "sw": {"rate_per_hour": {"low": 1, "central": 1, "high": 1}}}})  # missing duration
    assert validate_entry({"id": "x", "kind": "bogus"})


def test_unconfirmed_citation_is_rejected():
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    lib.entries["src/u"] = {"id": "src/u", "kind": "source", "measurements": [
        {"id": "m", "value": 1, "status": "U"}]}
    lib.entries["bad"] = {**lib.entries["legacy/vm-spot"], "id": "bad", "derived_from": ["src/u#m"]}
    assert any("unconfirmed" in p for p in lib.validate())
