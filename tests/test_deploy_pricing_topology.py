"""Cost/quota arithmetic and TiUP topology rendering (exec #7 step 2)."""

import json
import os
from pathlib import Path

import pytest
import yaml

from powder.deploy import pricing as P
from powder.deploy import spec as S
from powder.deploy import topology as T
from powder.faults import FaultLibrary
from powder.faults.regions import AWS_US_REGIONS

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "infra" / "tidb" / "specs" / "smoke.yaml"
GOLDEN = Path(__file__).parent / "fixtures" / "deploy" / "topology_smoke.yaml"


@pytest.fixture(scope="module")
def lib():
    return FaultLibrary.load()


def raw_i4i(region):
    """i4i.2xlarge $/h read straight from the price file (independent of powder.deploy)."""
    for e in json.loads((ROOT / "data/fault_library/prices/aws-ondemand.json").read_text()):
        if e["id"] == f"aws:i4i.2xlarge:{region}:ondemand:linux":
            return e["price_per_hour"]
    raise AssertionError(region)


def raw_cp(itype, region):
    return json.loads((ROOT / "data/tidb/control_plane_prices.json").read_text())["prices"][itype][region]


# ---------------------------------------------------------------- pricing
def test_smoke_bill_matches_hand_computation(lib):
    sp = S.parse_spec(SMOKE)
    e = P.estimate(sp, lib)
    tikv = 2 * (raw_i4i("us-east-1") + raw_i4i("us-east-2") + raw_i4i("us-west-2"))
    ctl_per_region = {r: raw_cp("m6i.xlarge", r) for r in ("us-east-1", "us-east-2", "us-west-2")}
    # control plane: controller (use1) + 3 PD + 3 system TiKV + 3 TiDB + 3 loadgen (one of each per region)
    cp = ctl_per_region["us-east-1"] + sum(ctl_per_region.values()) * 4
    assert e.hourly_by_role["systikv"] == pytest.approx(sum(ctl_per_region.values()))
    assert e.hourly_by_role["tikv"] == pytest.approx(tikv)
    assert e.hourly_total == pytest.approx(tikv + cp)
    assert sum(e.hourly_by_region.values()) == pytest.approx(e.hourly_total)
    assert e.total_for(3) == pytest.approx(3 * e.hourly_total)


def test_vcpu_by_region_for_quota_check(lib):
    e = P.estimate(S.parse_spec(SMOKE), lib)
    # use1: 2 TiKV (8 each) + controller + PD + system TiKV + TiDB + loadgen (4 each) = 36; use2/usw2: 2x8 + 4x4 = 32
    assert e.vcpu_by_region == {"us-east-1": 36, "us-east-2": 32, "us-west-2": 32}
    assert P.check_quotas(e, {"us-east-1": 36, "us-east-2": 32, "us-west-2": 32}) == []
    problems = P.check_quotas(e, {"us-east-1": 36, "us-east-2": 20})
    assert any("us-east-2" in p and "needs 32" in p for p in problems)
    assert any("us-west-2" in p and "unknown" in p for p in problems)


def test_modeled_cost_priced_vs_unpriced_scenario(lib):
    sp = S.parse_spec(SMOKE)
    priced = P.estimate(sp, lib)
    assert priced.modeled_hourly and priced.modeled_hourly != pytest.approx(priced.hourly_by_role["tikv"])
    d = S.load_yaml(SMOKE)
    d["scenario"] = "scenario/canonical-central"          # price_policy none
    unpriced = P.estimate(S.parse_spec(d), lib)
    assert unpriced.modeled_hourly is None and "no prices" in unpriced.modeled_note


def test_unknown_instance_type_is_an_error_not_zero(lib):
    d = S.load_yaml(SMOKE)
    d["control_plane"]["instance_type"] = "m9.mega"
    with pytest.raises(P.UnknownPrice, match="m9.mega"):
        P.estimate(S.parse_spec(d), lib)


def test_missing_control_plane_price_file_reports_clearly(lib, tmp_path):
    with pytest.raises(P.UnknownPrice, match="fetch_control_plane_prices"):
        P.estimate(S.parse_spec(SMOKE), lib, cpp=P.load_control_plane_prices(tmp_path / "none.json"))


def test_limits_and_transfer_rates(lib):
    sp = S.parse_spec(SMOKE)
    e = P.estimate(sp, lib)
    assert P.check_limits(e, sp.limits) == []
    d = S.load_yaml(SMOKE)
    d["limits"]["max_hourly_usd"] = 1
    sp2 = S.parse_spec(d)
    assert "exceeds limits.max_hourly_usd" in P.check_limits(P.estimate(sp2, lib), sp2.limits)[0]
    lo, hi = e.inter_region_per_gb
    assert 0 < lo <= hi <= 0.05


# ---------------------------------------------------------------- topology
def fake_hosts(spec):
    """Deterministic fake IPs: 10.<region index>.<az index * 16>.<n>."""
    hosts = {}
    for i, n in enumerate(S.plan_nodes(spec), start=1):
        az = n.az if n.az != "auto" else n.region + "a"
        idx = list(AWS_US_REGIONS).index(n.region)
        hosts[n.name] = T.Host(n.name, f"10.{idx}.0.{i}", n.region, az)
    return hosts


def test_golden_topology():
    sp = S.parse_spec({**S.load_yaml(SMOKE), "handle": "OrangePanda"})
    text = T.render_topology(sp, fake_hosts(sp))
    if os.environ.get("UPDATE_GOLDEN") or not GOLDEN.exists():
        GOLDEN.write_text(text)
    assert text == GOLDEN.read_text()


def test_topology_structure():
    sp = S.parse_spec(SMOKE)
    topo = yaml.safe_load(T.render_topology(sp, fake_hosts(sp)))
    assert len(topo["pd_servers"]) == 3 and len(topo["tikv_servers"]) == 9 and len(topo["tidb_servers"]) == 3
    classes = [kv["config"]["server.labels"]["machine_class"] for kv in topo["tikv_servers"]]
    assert classes.count("system") == 3                              # dedicated stores for TiDB's own data
    pd = topo["server_configs"]["pd"]
    assert pd["replication.location-labels"] == ["region", "zone", "host", "machine_class"]
    assert pd["replication.max-replicas"] == 3
    tidb = topo["server_configs"]["tidb"]
    assert tidb == {}                                     # TiDB defaults: powder-load kills requests past their SLO
    for kv in topo["tikv_servers"]:
        labels = kv["config"]["server.labels"]
        assert set(labels) == {"region", "zone", "host", "machine_class"}
        assert set(labels) == set(pd["replication.location-labels"])       # tiup deploy rejects any label not listed
        assert kv["data_dir"] == "/data/tikv"
        assert all(T._LABEL_RE.match(v) for v in labels.values())
    assert {kv["config"]["server.labels"]["machine_class"] for kv in topo["tikv_servers"]} == {
        "ford10-google-storage", "ford10-google-storage-ge15min", "system"}
    regions = [kv["config"]["server.labels"]["region"] for kv in topo["tikv_servers"]]
    assert sorted(set(regions)) == ["us-east-1", "us-east-2", "us-west-2"]
    # PD spread over regions: the three PD hosts sit in three different VPCs (second octet = region index)
    assert len({h["host"].split(".")[1] for h in topo["pd_servers"]}) == 3
    assert topo["monitoring_servers"][0]["host"] == topo["grafana_servers"][0]["host"]


def test_spec_pd_config_overrides_defaults():
    d = S.load_yaml(SMOKE)
    d["pd_config"] = {"schedule.max-store-down-time": "10m", "replication.max-replicas": 5}
    sp = S.parse_spec(d)
    pd = T.topology_dict(sp, fake_hosts(sp))["server_configs"]["pd"]
    assert pd["schedule.max-store-down-time"] == "10m"
    assert pd["replication.max-replicas"] == 5


def test_missing_host_is_reported():
    sp = S.parse_spec(SMOKE)
    hosts = fake_hosts(sp)
    del hosts["tikv-3"], hosts["pd-2"]
    with pytest.raises(T.TopologyError, match="pd-2.*tikv-3|tikv-3.*pd-2"):
        T.topology_dict(sp, hosts)


@pytest.mark.parametrize("raw, want", [
    ("machine/ford10-google-storage", "ford10-google-storage"),
    ("machine/spot-i4i.2xlarge-us-east-1", "spot-i4i.2xlarge-us-east-1"),
    ("machine/ford10-ge15min-x0.1", "ford10-ge15min-x0.1"),
    ("weird name!/a b", "a_b"),
])
def test_machine_class_label_is_a_valid_label_value(raw, want):
    assert T.machine_class_label(raw) == want
    assert T._LABEL_RE.match(want)


def test_label_value_rejects_unsalvageable_input():
    with pytest.raises(T.TopologyError):
        T.label_value("///")
