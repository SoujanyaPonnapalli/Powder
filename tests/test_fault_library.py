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
        expected = sum(len([r for r in s["deployment"]["regions"]
                            if not LIB.get(m)["regions"] or r in LIB.get(m)["regions"]]) for m in s["machines"])
        assert len(cfgs) == expected


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


def test_engine_compatible_native_and_fallback():
    ln = LogNormal(math.log(900), 1.0)
    assert engine_compatible(ln) is ln  # native in powder-mc
    distribution_spec(ln)
    tailed = Empirical([60.0, 600.0], [0.2, 0.9], tail=ln)  # tails are Python-only -> Weibull fallback
    fit = engine_compatible(tailed)
    distribution_spec(fit)
    assert abs(fit.mean / tailed.mean - 1) < 1e-6
    assert abs(fit.ppf(0.5) / tailed.ppf(0.5) - 1) < 1e-4


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


# -- Ford et al. OSDI'10 digitization ------------------------------------------

def test_ford_digitization_checks_pass():
    src = LIB.get("ford10", "source")
    chk = src["digitization"]["checks"]
    assert chk["fig1_F_15min_gt_0.90"]  # "less than 10% of events last longer than 15 minutes"
    assert chk["node_restart_spans_lt1min_to_gt1h"] and chk["unplanned_reboot_has_largest_mean"]
    assert chk["raster_fig1_within_0.02"] >= 0.99
    assert all(v >= 0.99 for v in chk["raster_fig2_within_0.02"].values())
    # independent consistency: summed >=15-min rates vs Table 2 node MTTF (4.3 months)
    assert 0.8 < chk["ratio_sum_ge15min_to_table2_node_mttf"] < 1.25


def test_ford_axis_calibration_is_tight():
    cal = LIB.get("ford10", "source")["digitization"]["calibration"]
    for fig in ("fig1", "fig2"):
        assert cal[fig]["x"]["max_residual_pt"] < 0.2  # log10 axis, "1mon" = 30 days
        assert cal[fig]["y"]["max_residual_pt"] < 0.05


def test_ford_planned_duration_matches_digitized_median():
    pd = LIB.get("planned_downtime/weekly-ford10", "planned_downtime")
    d = from_spec(pd["duration"]["central"])
    med = next(m for m in LIB.get("ford10")["measurements"] if m["id"] == "digitized-planned_reboot")
    assert abs(d.ppf(0.5) / med["value"]["median_s"] - 1) < 0.02
    lo, hi = from_spec(pd["duration"]["low"]), from_spec(pd["duration"]["high"])
    assert lo.mean < d.mean < hi.mean


def test_ford_fig3_rates_inside_table1_range():
    for c in ("node_restart", "planned_reboot", "unplanned_reboot"):
        m = next(x for x in LIB.get("ford10")["measurements"] if x["id"] == f"derived-rate-{c}")["value"]
        assert m["ge15min"]["low"] <= m["fig3_one_cell_ge15min"] <= m["ge15min"]["high"], c


# -- planned downtime (subtask 1.1) -------------------------------------------

from powder.faults.planned_downtime import as_transient, long_run_fraction, schedule  # noqa: E402


def _planned():
    return LIB.get("planned_downtime/weekly-ford10", "planned_downtime")


def test_planned_schedule_no_overlap_and_gap_distribution():
    p = _planned()
    ev = schedule([f"n{i}" for i in range(300)], days(7 * 30), p, np.random.default_rng(3))
    by_node = {}
    for e in ev:
        by_node.setdefault(e.node, []).append(e)
    gaps = []
    for evs in by_node.values():
        for a, b in zip(evs, evs[1:]):
            assert b.start >= a.end  # no overlapping downtime on one node
            gaps.append(b.start - a.start)
    g = np.sort(np.array(gaps))
    model = np.array([normal_cdf((v - days(7)) / (12 * 3600.0)) for v in g])
    ks = np.max(np.abs(model - np.arange(1, len(g) + 1) / len(g)))
    assert ks < 1.63 / math.sqrt(len(g))  # KS at alpha = 0.01


def test_planned_equilibrium_phase():
    p = _planned()
    rng = np.random.default_rng(11)
    firsts = [schedule(["x"], days(30), p, rng)[0].start for _ in range(4000)]
    mu, sd = days(7), 12 * 3600.0
    expected = (mu**2 + sd**2) / (2 * mu)  # mean residual life of the renewal process
    assert abs(np.mean(firsts) / expected - 1) < 0.03
    assert max(firsts) < days(7) + 8 * sd


def test_planned_long_run_fraction():
    p = _planned()
    horizon = days(7 * 52)
    ev = schedule([f"n{i}" for i in range(400)], horizon, p, np.random.default_rng(5))
    down = sum(min(e.end, horizon) - e.start for e in ev)
    frac = down / (400 * horizon)
    assert abs(frac / long_run_fraction(p) - 1) < 0.06
    rate, mean_d = as_transient(p)
    assert math.isclose(rate, 1 / days(7)) and 600 < mean_d < 3600


def test_planned_events_per_day_matches_n_over_7():
    p = _planned()
    rng = np.random.default_rng(9)
    counts = [len(schedule([f"n{i}" for i in range(9)], days(1), p, rng)) for _ in range(3000)]
    assert abs(np.mean(counts) - 9 / 7) < 0.06


# -- region outages (subtask 1.2) ----------------------------------------------

from powder.faults.regions import region_map  # noqa: E402


def test_region_map_decisions():
    m = region_map()
    assert m["us-east-1"]["azure_region"] == "eastus2"  # fixed decision, no pooling
    assert m["us-east-2"]["azure_region"] == "eastus"
    assert m["us-west-1"]["azure_region"] == "westus"
    assert m["us-west-2"]["azure_region"] == "westus2"
    for v in m.values():
        assert v["distance_km"] < 500


def test_region_outage_profile():
    ro = LIB.get("region_outage/aws-us-from-azure", "region_outage")
    assert set(ro["regions"]) == {"us-east-1", "us-east-2", "us-west-1", "us-west-2"}
    for aws, r in ro["regions"].items():
        per_year = {k: v * 8766 for k, v in r["rate_per_hour"].items()}
        assert 0.05 < per_year["central"] < 3.0, (aws, per_year)
        assert r["azure_region"] == region_map()[aws]["azure_region"]
    d = from_spec(ro["duration"]["central"])
    assert 3600 < d.mean < 2 * 86400  # region outages last hours, not minutes or weeks
    assert ro["duration_fit"]["ks_bootstrap_p"] > 0.05
    pess = LIB.get("region_outage/aws-us-from-azure-incl-single-az", "region_outage")
    assert pess["regions"]["us-east-1"]["rate_per_hour"]["central"] >= ro["regions"]["us-east-1"]["rate_per_hour"]["central"]


def test_region_incident_citations_are_confirmed():
    ro = LIB.get("region_outage/aws-us-from-azure", "region_outage")
    for ref in ro["derived_from"]:
        assert LIB.measurement(ref)["status"] in ("V", "S")


# -- literature profiles (Step D) ----------------------------------------------

def test_disk_pool_brackets_inputs():
    pooled = LIB.get("disk/pooled", "component")["rate_per_hour"]
    inputs = [LIB.get(c, "component")["rate_per_hour"]["central"]
              for c in ("disk/backblaze-hdd-2025", "disk/backblaze-ssd-2023", "disk/schroeder-gibson-fast07", "disk/ford10")]
    assert pooled["low"] < min(inputs) * 1.5 and pooled["high"] > max(inputs) * 0.5
    assert min(inputs) <= pooled["central"] <= max(inputs)


def test_aws_ondemand_within_sla_and_sensible():
    from powder.faults.schema import pick
    p = LIB.get("cloud/aws-ondemand-nvme", "node")
    down_h = 0.0
    for c in ("hw_reboot", "sw", "pause"):
        spec = p["causes"][c]
        down_h += spec["rate_per_hour"]["high"] * 8766 * from_spec(pick(spec["duration"], "high", "duration")).mean / 3600
    assert down_h < 0.005 * 8766  # AWS instance-level SLA 99.5%
    lam_t, rec, lam_p = transient_and_permanent(p, "central")
    assert 0.01 < lam_t * 3600 * 8766 < 2.0 and 0.005 < lam_p * 3600 * 8766 < 1.0  # per node-year


def test_vishwanath_frailty_reproduces_inputs():
    a = LIB.get("host/vishwanath10", "node")["frailty_a"]
    mu = 0.2  # per 14 months
    p0 = (1 + mu / a) ** (-a)
    assert abs((1 - p0) - 0.09) < 1e-6 and abs(mu / (1 - p0) - 20 / 9) < 1e-6


# -- AWS price / spot ingestion (Steps B and C), on format fixtures -------------

from pathlib import Path  # noqa: E402

from powder.faults.aws_prices import (  # noqa: E402
    band_rates, merge_transfer, parse_ondemand_offer, parse_spot_advisor, parse_transfer_offer, price_distribution,
    region_factors,
)

FIX = Path(__file__).parent / "fixtures" / "fault_library"
US = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]


def _offers():
    return json.loads((FIX / "aws_offers_sample.json").read_text())


def test_parse_ondemand_filters():
    rows = [r for reg in US for r in parse_ondemand_offer(_offers()[reg])]
    types = {r["instance_type"] for r in rows}
    assert "m7g.2xlarge" not in types and "g4dn.2xlarge" not in types and "m6i.xlarge" not in types
    m6i = [r for r in rows if r["instance_type"] == "m6i.2xlarge" and r["region"] == "us-east-1"]
    assert len(m6i) == 1 and m6i[0]["price_per_hour"] == 0.384  # Windows / dedicated variants dropped
    i4i = next(r for r in rows if r["instance_type"] == "i4i.2xlarge" and r["region"] == "us-east-1")
    assert i4i["instance_store"] and i4i["nvme"]
    assert next(r for r in rows if r["instance_type"] == "t3.2xlarge")["burstable"]


def test_transfer_merge_and_validation():
    parts = [parse_transfer_offer(_offers()[reg], US) for reg in US]
    t = merge_transfer(parts, US)
    assert t["inter_region"]["us-east-1"]["us-west-2"] == 0.02
    assert t["inter_az_per_side"] == 0.01 and t["inter_az"] == 0.02  # billed on both sides
    assert t["internet_egress"] == 0.09  # first paid tier
    try:
        merge_transfer(parts[:2], US)
        raise AssertionError("incomplete transfer table must fail")
    except ValueError:
        pass


def test_price_distribution_and_factors():
    rows = [r for reg in US for r in parse_ondemand_offer(_offers()[reg])]
    d = price_distribution(rows, current_generation=True, burstable=False)
    assert d["n"] == len([r for r in rows if r["current_generation"] and not r["burstable"]])
    assert d["summary"]["min"] <= d["summary"]["p50"] <= d["summary"]["max"]
    f = region_factors(rows, "i4i.2xlarge")
    assert f["us-east-1"] == 1.0 and f["us-west-1"] > 1.0


def test_spot_advisor_and_band_rates():
    doc = json.loads((FIX / "spot_advisor_sample.json").read_text())
    assert all("min" not in r and "dots" in r for r in doc["ranges"])  # real layout
    bands = parse_spot_advisor(doc, US)
    bad = {**doc, "ranges": [dict(r, label="?") if r["index"] == 2 else r for r in doc["ranges"]]}
    try:
        parse_spot_advisor(bad, US)
        raise AssertionError("changed band labels must fail")
    except ValueError:
        pass
    assert bands[("us-east-1", "i4i.2xlarge")]["band"] == 0
    r = band_rates(1)
    assert r["low"] < r["central"] < r["high"]
    assert abs(r["central"] - (-math.log(1 - 0.075) / 730)) < 1e-15


def test_build_prices_pipeline_on_fixtures(tmp_path=None):
    import importlib.util
    import shutil
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    lib_dir = tmp / "fault_library"
    shutil.copytree(LIB.root, lib_dir, ignore=shutil.ignore_patterns("raw"))
    rows, parts = [], []
    for reg in US:
        rows += parse_ondemand_offer(_offers()[reg])
        parts.append(parse_transfer_offer(_offers()[reg], US))
    pdir = lib_dir / "raw/aws_prices/2026-10-21"
    pdir.mkdir(parents=True)
    (pdir / "ondemand_8vcpu_x86.json").write_text(json.dumps(rows))
    (pdir / "transfer.json").write_text(json.dumps(merge_transfer(parts, US)))
    (pdir / "meta.json").write_text(json.dumps({"fetched_at": "2026-10-21T00:00:00Z",
                                                "offers": {r: {"url": "fixture"} for r in US}}))
    adir = lib_dir / "raw/spot/advisor"
    adir.mkdir(parents=True)
    shutil.copy(FIX / "spot_advisor_sample.json", adir / "2026-10-21.json")
    spec = importlib.util.spec_from_file_location(
        "build_prices", Path(__file__).resolve().parents[1] / "scripts/fault_library/build_prices.py")
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    bp.LIB, bp.RAW = lib_dir, lib_dir / "raw"
    assert bp.main() == 0
    lib = FaultLibrary.load(lib_dir)
    spot = lib.get("spot/aws-i4i.2xlarge-us-west-1", "node")
    assert "spot_interrupt" in spot["causes"]
    cfg = machine_node_config(lib, "machine/spot-i4i.2xlarge-us-west-1", "us-west-1", engine=True)
    assert cfg.data_loss_dist.rate > 0
    assert lib.get("price-transfer/aws-us@2026-10-21")["per_gb"]["inter_az"] == 0.02
    # ... then machine classes, price assignments and priced scenarios
    spec = importlib.util.spec_from_file_location(
        "build_machines", Path(__file__).resolve().parents[1] / "scripts/fault_library/build_machines.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    bm.LIB = lib_dir
    assert bm.main() == 0
    lib = FaultLibrary.load(lib_dir)
    cfgs = scenario_node_configs(lib, "scenario/canonical-central-rho1", engine=True)
    assert all(c.cost_per_hour > 0 for c in cfgs.values())
    assert ("machine/spot-i4i.2xlarge-us-west-1", "us-west-1") in cfgs
    assert ("machine/spot-i4i.2xlarge-us-west-1", "us-east-1") not in cfgs  # regional spot pool
    rho0 = [e for e in lib.query(kind="price") if e.get("price_kind") == "price_assignment" and e["rho"] == 0.0]
    assert len(rho0) == 20
    alt = scenario_node_configs(lib, "scenario/canonical-central-rho1", price_assignment=rho0[3]["id"])
    assert any(alt[k].cost_per_hour != cfgs[k].cost_per_hour for k in alt)


# -- price assignment and traces (Step G) ---------------------------------------

from powder.faults.assign import assign_prices, reliability_score, stratified_prices  # noqa: E402
from powder.faults.trace import generate  # noqa: E402


def test_assignment_extremes_and_multiset():
    scores = {f"m{i}": float(s) for i, s in enumerate([5.0, 0.2, 1.1, 3.3, 0.05, 2.0, 7.5])}
    pts = np.random.default_rng(0).lognormal(math.log(0.4), 0.4, 300)
    ranked = assign_prices(scores, pts, rho=1.0)
    assert abs(ranked.realised_spearman - 1.0) < 1e-12
    most_reliable = min(scores, key=scores.get)
    assert ranked.prices[most_reliable] == max(ranked.prices.values())
    expected = sorted(stratified_prices(pts, len(scores)))
    rhos = []
    for seed in range(200):
        a = assign_prices(scores, pts, rho=0.0, seed=seed)
        assert np.allclose(sorted(a.prices.values()), expected)
        rhos.append(a.realised_spearman)
    assert abs(np.mean(rhos)) < 0.1


def test_reliability_score_units():
    p = LIB.get("cloud/aws-ondemand-nvme", "node")
    lam_t, rec, lam_p = transient_and_permanent(p)
    from powder.faults.system import REBUILD_S
    manual = (lam_t * rec.mean + lam_p * REBUILD_S) * 8766
    assert abs(reliability_score(p) / manual - 1) < 1e-9


def test_trace_rates_round_trip():
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    lib.entries["__s__"] = {"id": "__s__", "kind": "scenario", "machines": ["machine/aws-ondemand-x10"],
                            "region_outage": "region_outage/aws-us-from-azure",
                            "planned_downtime": "planned_downtime/weekly-ford10", "level": "central",
                            "price_assignment": None, "price_policy": "none",
                            "deployment": {"regions": ["us-east-1", "us-west-2"]}}
    slots = {f"s{i}": ("machine/aws-ondemand-x10", "us-east-1" if i % 2 else "us-west-2") for i in range(40)}
    T = days(365 * 25)
    ev = generate(lib, "__s__", slots, T, seed=4)
    prof = LIB.get("synthetic/aws-ondemand-x10", "node")
    rates = node_profile_rates_ps(prof)
    for cause, lam in rates.items():
        if lam <= 0:
            continue
        n = sum(1 for e in ev if e.cause == cause)
        mu = lam * T * len(slots)
        assert abs(n - mu) < 4 * math.sqrt(mu) + 2, (cause, n, mu)  # downtime fraction is negligible here
    n_planned = sum(1 for e in ev if e.kind == "planned")
    assert abs(n_planned / (len(slots) * T / days(7)) - 1) < 0.02
    ro = LIB.get("region_outage/aws-us-from-azure")
    lam_r = ro["regions"]["us-east-1"]["rate_per_hour"]["central"] / 3600
    n_r = sum(1 for e in ev if e.kind == "region" and e.region == "us-east-1")
    assert abs(n_r - lam_r * T) < 4 * math.sqrt(lam_r * T) + 2


def node_profile_rates_ps(profile):
    from powder.faults.convert import node_profile_rates
    return {c: r.rate_per_second for c, r in node_profile_rates(profile).items()}


def test_trace_is_reproducible():
    slots = {"a": ("machine/aws-ondemand-nvme", "us-east-1"), "b": ("machine/ford10-google-storage", "us-west-1")}
    e1 = generate(LIB, "scenario/canonical-central", slots, days(30), seed=7)
    e2 = generate(LIB, "scenario/canonical-central", slots, days(30), seed=7)
    assert e1 == e2 and len(e1) > 0


def test_scenario_machine_types_for_ilp():
    try:
        import powder.placement_optimizer  # noqa: F401  (needs SciPy)
    except Exception:
        return
    from powder.faults.ilp import scenario_machine_types
    types = scenario_machine_types(LIB, "scenario/canonical-central", copies=4)
    assert types and all(t.copies == 4 and "@" in t.type_id for t in types)


# -- review fixes: trace/simulator consistency, global outages, rolling planned ---

def _scenario(lib, sid, **kw):
    base = {"id": sid, "kind": "scenario", "machines": ["machine/aws-ondemand-x10"], "region_outage": None,
            "planned_downtime": None, "level": "central", "price_assignment": None, "price_policy": "none",
            "deployment": {"regions": ["us-east-1", "us-west-2"]}}
    base.update(kw)
    lib.entries[sid] = base
    return sid


def test_trace_weibull_uses_single_transient_clock():
    """With a Weibull shape the trace must fire transients at lambda_T (one clock), like the simulator."""
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    prof = json.loads(json.dumps(LIB.get("synthetic/aws-ondemand-x10")))
    prof["id"] = "__weib__"
    prof["shape"] = {"type": "weibull", "k": {"low": 0.7, "central": 0.7, "high": 0.7}}
    # no permanent causes: replacements restart the clock and (for k < 1) legitimately raise
    # the transient rate in both the trace and the simulator, so isolate the renewal process
    for c in ("disk", "hw_host_loss"):
        prof["causes"].pop(c, None)
    # many renewals per slot: an ordinary renewal process with k < 1 front-loads events after t = 0, a
    # fixed per-slot excess over lambda*T; with ~hundreds of renewals per slot it is negligible
    for c in prof["causes"].values():
        c["rate_per_hour"] = {k: v * 50 for k, v in c["rate_per_hour"].items()}
    lib.entries["__weib__"] = prof
    lib.entries["__m__"] = {"id": "__m__", "kind": "machine", "fault_profile": "__weib__", "price": {"route": "none"}}
    sid = _scenario(lib, "__sw__", machines=["__m__"])
    slots = {f"s{i}": ("__m__", "us-east-1") for i in range(200)}
    T = days(365 * 20)
    ev = generate(lib, sid, slots, T, seed=1)
    lam_t, _, _ = transient_and_permanent(prof)
    n = sum(e.kind == "transient" for e in ev)
    mu = lam_t * T * len(slots)
    assert abs(n / mu - 1) < 0.05, (n, mu)


def test_trace_slot_streams_are_independent():
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    sid = _scenario(lib, "__ind__", planned_downtime="planned_downtime/weekly-ford10")
    a = {"x": ("machine/aws-ondemand-x10", "us-east-1"), "y": ("machine/aws-ondemand-x10", "us-west-2")}
    b = {**a, "z": ("machine/aws-ondemand-x10", "us-west-2")}
    b2 = {"a0": ("machine/aws-ondemand-x10", "us-east-1"), **b}  # sorts before x as well
    ea = [e for e in generate(lib, sid, a, days(4000), seed=3) if e.slot == "x"]
    eb = [e for e in generate(lib, sid, b2, days(4000), seed=3) if e.slot == "x"]
    assert ea == eb and any(e.kind != "planned" for e in ea)


def test_frailty_is_opt_in_and_matches_draws():
    from powder.faults.trace import frailty_draws
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    # transient-only profile with strong frailty: same host, repeated faults (lemons)
    prof = json.loads(json.dumps(LIB.get("synthetic/aws-ondemand-x10")))
    prof.update({"id": "__frail__", "frailty_a": 0.1})
    for c in ("disk", "hw_host_loss"):
        prof["causes"].pop(c, None)
    lib.entries["__frail__"] = prof
    lib.entries["__mf__"] = {"id": "__mf__", "kind": "machine", "fault_profile": "__frail__", "price": {"route": "none"}}
    sid = _scenario(lib, "__fr__", machines=["__mf__"])
    slots = {f"s{i}": ("__mf__", "us-east-1") for i in range(300)}
    T = days(365 * 30)
    plain = generate(lib, sid, slots, T, seed=2)
    frail = generate(lib, sid, slots, T, seed=2, use_frailty=True)
    per_slot = lambda ev: np.array([sum(e.slot == s for e in ev) for s in slots])  # noqa: E731
    assert per_slot(frail).var() > 3 * per_slot(plain).var()  # lemons appear only with frailty
    z = frailty_draws(lib, slots, seed=2)
    assert abs(np.mean(list(z.values())) - 1) < 0.5 and len(z) == len(slots)


def test_planned_downtime_folded_into_node_configs():
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    sid = _scenario(lib, "__pl__", machines=["machine/aws-ondemand-nvme"], planned_downtime="planned_downtime/weekly-ford10")
    with_p = scenario_node_configs(lib, sid)[("machine/aws-ondemand-nvme", "us-east-1")]
    without = scenario_node_configs(lib, sid, include_planned=False)[("machine/aws-ondemand-nvme", "us-east-1")]
    assert math.isclose(1 / with_p.failure_dist.mean - 1 / without.failure_dist.mean, 1 / days(7), rel_tol=1e-9)


def test_rolling_planned_groups_go_down_together():
    pd = LIB.get("planned_downtime/weekly-ford10-rolling", "planned_downtime")
    nodes = ["a1", "a2", "b1", "b2"]
    groups = {"a1": "az-a", "a2": "az-a", "b1": "az-b", "b2": "az-b"}
    ev = schedule(nodes, days(70), pd, np.random.default_rng(0), groups=groups)
    starts = {}
    for e in ev:
        starts.setdefault(e.node, []).append(round(e.start, 6))
    assert starts["a1"] == starts["a2"] and starts["b1"] == starts["b2"] and starts["a1"] != starts["b1"]


def test_global_outage_process():
    from powder.faults.convert import scenario_network_config
    ro = LIB.get("region_outage/aws-us-from-azure")
    assert ro["global"]["n_incidents"] >= 1 and "2023-01-25-wan" in ro["global"]["incidents"]
    for r in ro["regions"].values():  # the global event is not counted per region
        assert "2023-01-25-wan" not in r["incidents"]
    m = scenario_network_config(LIB, "scenario/canonical-central")
    nc = m.as_network_config("max")
    assert math.isclose(nc.outage_dist.rate, max(m.rate_per_second.values()) + m.global_rate_per_second)
    exact = m.as_network_config()
    for r, rate in m.rate_per_second.items():
        assert math.isclose(exact.outage_for(r).rate, rate)
    assert math.isclose(exact.global_outage_dist.rate, m.global_rate_per_second)
    lib = FaultLibrary(dict(LIB.entries), dict(LIB.paths), LIB.root)
    sid = _scenario(lib, "__gl__", region_outage="region_outage/aws-us-from-azure")
    ev = generate(lib, sid, {"x": ("machine/aws-ondemand-nvme", "us-east-1")}, days(365 * 200), seed=5)
    assert any(e.kind == "global" and e.region == "*" for e in ev)


def test_aws_host_faults_consistent_per_level():
    p = LIB.get("cloud/aws-ondemand-nvme", "node")
    a = p["assumptions"]
    for lvl in ("low", "central", "high"):
        total = (p["causes"]["hw_reboot"]["rate_per_hour"][lvl] + p["causes"]["hw_host_loss"]["rate_per_hour"][lvl]) * 8766
        assert math.isclose(total, a["host_repair_rate_per_year"][lvl] * (1 - a["f_LM"][lvl]), rel_tol=1e-6)


# -- LANL (raw data) -------------------------------------------------------------

def test_lanl_pooled_profile_and_shapes():
    if "hpc/lanl05-pooled" not in LIB.entries:
        return  # raw LANL data not fetched
    p = LIB.get("hpc/lanl05-pooled", "node")
    hw = p["causes"]["hw_reboot"]["rate_per_hour"]
    assert 0.05 < hw["central"] * 8766 < 5  # failures needing an administrator, per node-year
    k = p["shape_variant"]["k"]
    assert 0.4 < k["central"] < 0.9  # decreasing hazard, consistent with DSN'06 (0.7-0.8)
    summ = LIB.measurement("lanl05#pooled-small")["value"]
    assert 0.05 < summ["rho_sw_pool"]["central"] < 1.0
    assert summ["cofail_inflation_small"]["median"] > 1  # failures cluster in time across nodes
    aws = LIB.get("cloud/aws-ondemand-nvme", "node")["assumptions"]["sw_ratio"]
    assert aws["low"] < 0.5 < aws["central"] < aws["high"]



def test_engine_runs_library_scenario_natively():
    """End to end: library scenario -> powder-mc job with native lognormal/empirical/mixture -> run."""
    import os
    import subprocess
    from powder.mc_jobs import binary_path, build_job, EngineNotFound
    from powder.simulation import ClusterState, NetworkState, NodeState, NodeReplacementStrategy, RaftLikeProtocol
    try:
        binary = binary_path()
        subprocess.run([str(binary), "--help"], capture_output=True, check=True, timeout=30)
    except (EngineNotFound, OSError, subprocess.SubprocessError):
        return  # no runnable engine here (e.g. a Mac binary on Linux)
    cfgs = scenario_node_configs(LIB, "scenario/canonical-central", engine=True)
    cfg = cfgs[("machine/ford10-google-storage", "us-east-1")]
    kinds = {distribution_spec(cfg.recovery_dist)["type"]}
    assert kinds & {"mixture", "empirical", "lognormal"}  # not approximated
    nodes = {f"n{i}": NodeState(node_id=f"n{i}", config=cfg) for i in range(3)}
    cluster = ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=3)
    job = build_job(cluster, NodeReplacementStrategy(failure_timeout=300.0, default_node_config=cfg),
                    RaftLikeProtocol(election_time_dist=Exponential(0.2)), num_simulations=20,
                    max_time=days(30), stop_on_data_loss=False, base_seed=1)
    out = subprocess.run([str(binary), "--stream"], input=json.dumps(job) + "\n", capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()



def test_python_simulator_per_region_and_global_outages():
    """Python simulator: per-region rates, a global process, and overlap counting."""
    from powder.simulation import ClusterState, NetworkConfig, NetworkState, NodeState, NoOpStrategy, Simulator
    from powder.simulation.network import NetworkState as NS
    ns = NS()
    ns.add_outage("a"); ns.add_outage("a"); ns.remove_outage("a")
    assert ns.is_region_down("a")  # overlapping outages: still down after one ends
    ns.remove_outage("a")
    assert not ns.is_region_down("a")
    never = Exponential(1e-12)
    cfg = machine_node_config(LIB, "machine/legacy-standard", "r1")
    from dataclasses import replace
    nodes = {}
    for i, r in enumerate(["r1", "r2", "r3"]):
        c = replace(cfg, region=r, failure_dist=never, data_loss_dist=never)
        nodes[f"n{i}"] = NodeState(node_id=f"n{i}", config=c)
    net = NetworkConfig(outage_dist=never, outage_duration_dist=Exponential(1 / 3600.0), regions=["r1", "r2", "r3"],
                        per_region={"r1": (Exponential(1 / days(10)), None)},
                        global_outage_dist=Exponential(1 / days(40)))
    from powder.simulation import RaftLikeProtocol
    sim = Simulator(ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=3), NoOpStrategy(),
                    RaftLikeProtocol(election_time_dist=Exponential(0.2)), network_config=net, seed=3, log_events=True)
    from powder.simulation import EventType
    sim.run_for(days(4000))
    starts = {"r1": 0, "r2": 0, "r3": 0, "__global__": 0}
    for ev in sim.event_log:
        if ev.event_type == EventType.NETWORK_OUTAGE_START:
            starts[ev.metadata.get("region", ev.target_id)] += 1
    assert 300 < starts["r1"] < 500 and starts["r2"] == 0  # r1 every ~10 d, r2 never
    assert 70 < starts["__global__"] < 130  # ~100 global events in 4000 d


def test_engine_global_outages_take_quorum_down():
    """powder-mc: a global outage process lowers 3-region availability by ~rate x duration;
    outages confined to one region do not break quorum."""
    import subprocess
    from dataclasses import replace
    from powder.mc_jobs import EngineNotFound, binary_path, build_job
    from powder.simulation import (ClusterState, NetworkConfig, NetworkState, NodeReplacementStrategy, NodeState,
                                   RaftLikeProtocol)
    try:
        binary = binary_path()
        subprocess.run([str(binary), "--help"], capture_output=True, check=True, timeout=30)
    except (EngineNotFound, OSError, subprocess.SubprocessError):
        return
    never = Exponential(1e-12)
    base = machine_node_config(LIB, "machine/legacy-standard", "r1")

    def run(net):
        nodes = {f"n{i}": NodeState(node_id=f"n{i}", config=replace(base, region=r, failure_dist=never,
                                                                     data_loss_dist=never))
                 for i, r in enumerate(["r1", "r2", "r3"])}
        job = build_job(ClusterState(nodes=nodes, network=NetworkState(), target_cluster_size=3),
                        NodeReplacementStrategy(1e12, None), RaftLikeProtocol(election_time_dist=Exponential(1.0)),
                        num_simulations=300, max_time=days(365), stop_on_data_loss=False, base_seed=7, network_config=net)
        out = subprocess.run([str(binary), "--stream"], input=json.dumps(job) + "\n", capture_output=True, text=True,
                             timeout=150)
        assert out.returncode == 0, out.stderr[-800:]
        return json.loads(out.stdout.splitlines()[0])["summary"]["availability_mean"]

    dur = Exponential(1 / 36000.0)  # 10 h
    regions = ["r1", "r2", "r3"]
    one_region = run(NetworkConfig(never, dur, regions, per_region={"r1": (Exponential(1 / days(5)), None)}))
    glob = run(NetworkConfig(never, dur, regions, global_outage_dist=Exponential(1 / days(20))))
    assert one_region > 0.99999
    expected = 1 - (365 / 20) * 10 / 8760
    assert abs(glob - expected) < 0.005, (glob, expected)



def test_exec10_simple_system_timings():
    from powder.faults.system import REBUILD_S, SNAPSHOT_S
    assert abs(SNAPSHOT_S - 8.589934592) < 1e-9  # 256 MiB over 1 Gbps
    assert abs(REBUILD_S - (120 + SNAPSHOT_S)) < 1e-9
    s = LIB.get("scenario/canonical-central", "scenario")
    assert s["system"]["source"] == "exec10-simple-estimate"
    cfg = scenario_node_configs(LIB, "scenario/canonical-central")[("machine/aws-ondemand-nvme", "us-east-1")]
    assert abs(cfg.snapshot_download_time_dist.value - SNAPSHOT_S) < 1e-12
    assert (cfg.spawn_dist.low, cfg.spawn_dist.high) == (60.0, 180.0)
    legacy = scenario_node_configs(LIB, "scenario/legacy")[("machine/legacy-standard", "us-east")]
    assert legacy.spawn_dist.mean == 60.0  # legacy profiles keep their own timings


def test_spot_rates_from_band_history():
    from powder.faults.aws_prices import band_rates, band_rates_from_history
    r = band_rates_from_history({"0": 0.5, "2": 0.5})
    assert r["low"] == band_rates(0)["low"] and r["high"] == band_rates(2)["high"]
    assert abs(r["central"] - 0.5 * (band_rates(0)["central"] + band_rates(2)["central"])) < 1e-15
    assert band_rates_from_history({"3": 1.0}) == band_rates(3)
