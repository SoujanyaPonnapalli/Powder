"""Pinning TiDB's own data to the system stores (powder.deploy.placement)."""

from pathlib import Path

from powder.deploy import placement as PL
from powder.deploy import spec as S
from powder.deploy.regionmap import Region, RegionMap, StoreInfo, row_key

SMOKE = Path(__file__).resolve().parents[1] / "infra" / "tidb" / "specs" / "smoke.yaml"
LABELS = ["region", "zone", "host", "machine_class"]


def stores():
    sys_ = {i: StoreInfo(i, machine_class="system") for i in (1, 2, 3)}
    exp = {i: StoreInfo(i, machine_class="ford10-google-storage") for i in (4, 5, 6, 7, 8, 9)}
    return {**sys_, **exp}


def test_rules_pin_everything_to_system_stores_except_the_user_range():
    pd, powder = PL.rule_bundles(114, 3, LABELS)
    default = pd["rules"][0]
    assert (pd["group_id"], default["id"], default["start_key"], default["end_key"]) == ("pd", "default", "", "")
    assert default["label_constraints"] == [{"key": "machine_class", "op": "in", "values": ["system"]}]
    # no isolation level on the system rule: with the orphan peers it deadlocked PD at the first live run
    assert default["count"] == 3 and default["isolation_level"] == ""
    data = powder["rules"][0]
    assert powder["group_override"] and powder["group_index"] > pd["group_index"]       # overrides the default rule
    assert data["label_constraints"] == [{"key": "machine_class", "op": "notIn", "values": ["system"]}]
    lo, hi = PL.user_range(114)
    assert (data["start_key"], data["end_key"]) == (lo.hex(), hi.hex())
    assert data["isolation_level"] == "region"                 # experiment data keeps one replica per AWS region


def test_user_range_holds_user_rows_but_not_system_tables_or_reserved_ids():
    lo, hi = PL.user_range(114)
    assert lo <= row_key(114, 0) < hi and lo <= row_key(5000, 7) < hi
    assert not lo <= row_key(113, 999) < hi                                      # last system table
    assert not lo <= row_key((1 << 48) - 5, 1) < hi                             # tidb_mdl_info's reserved id


def regions(*specs):
    return RegionMap(0.0, [Region(i, a, b, tuple(v)) for i, (a, b, v) in enumerate(specs, start=1)], stores())


def test_placement_problems_accepts_a_correctly_placed_map():
    lo, hi = PL.user_range(114)
    m = regions((b"", lo, (1, 2, 3)), (lo, hi, (4, 6, 8)), (hi, b"", (1, 2, 3)))
    assert PL.placement_problems(m, 114, 3) == []


def test_placement_problems_flags_misplaced_and_straddling_regions():
    lo, hi = PL.user_range(114)
    mid = row_key(114, 500)
    m = regions((b"", lo, (1, 2, 4)), (lo, mid, (4, 6, 1)), (mid, hi, (4, 6, 8)), (hi, b"", (1, 2, 3)))
    p = PL.placement_problems(m, 114, 3)
    assert any("system Region 1" in x for x in p) and any("user Region 2" in x for x in p)
    straddle = regions((b"", mid, (1, 2, 3)), (mid, b"", (4, 6, 8)))
    assert any("straddles" in x for x in PL.placement_problems(straddle, 114, 3))
    no_sys = RegionMap(0.0, [Region(1, b"", b"", (4, 5, 6))], {i: StoreInfo(i) for i in (4, 5, 6)})
    assert PL.placement_problems(no_sys, 114, 3) == ["no store is labelled machine_class=system"]


def test_plan_has_one_system_store_per_region_and_enough_for_the_replicas():
    sp = S.parse_spec(SMOKE)
    sys_nodes = [n for n in S.plan_nodes(sp) if n.role == "systikv"]
    assert sorted(n.region for n in sys_nodes) == ["us-east-1", "us-east-2", "us-west-2"]
    assert all(n.instance_type == sp.control_plane.instance_type and n.slot is None for n in sys_nodes)
    two = S.parse_spec({**S.load_yaml(SMOKE), "slots": {
        "tikv-1": {"machine": "machine/ford10-google-storage@us-east-1"},
        "tikv-2": {"machine": "machine/ford10-google-storage@us-east-1"},
        "tikv-3": {"machine": "machine/ford10-google-storage@us-west-2"}}})
    names = sorted(n.name for n in S.plan_nodes(two) if n.role == "systikv")
    assert names == ["systikv-us-east-1-1", "systikv-us-east-1-2", "systikv-us-west-2-1"]   # 3 replicas, 2 regions
    off = S.parse_spec({**S.load_yaml(SMOKE), "control_plane": {**S.load_yaml(SMOKE)["control_plane"],
                                                               "system_tikv_per_region": 0}})
    assert not [n for n in S.plan_nodes(off) if n.role == "systikv"]
