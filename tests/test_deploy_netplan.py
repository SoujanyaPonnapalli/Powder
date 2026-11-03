"""Network planning (exec #7 step 2)."""

import dataclasses

import pytest

from powder.deploy import netplan as N

AZS = {r: ["a", "b", "c"] for r in N.REGION_INDEX}


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_peering_is_a_full_mesh(n):
    regions = list(N.REGION_INDEX)[:n]
    plan = N.plan_network(regions, AZS, "1.2.3.4/32")
    assert len(plan.peerings) == n * (n - 1) // 2
    assert len(plan.routes) == 2 * len(plan.peerings)                   # both sides
    assert N.validate_plan(plan) == []


def test_addressing_is_stable_by_region_regardless_of_input_order():
    a = N.plan_network(["us-west-2", "us-east-1"], AZS, "1.2.3.4/32")
    b = N.plan_network(["us-east-1", "us-west-2"], AZS, "1.2.3.4/32")
    assert a.vpcs == b.vpcs == {"us-east-1": "10.0.0.0/16", "us-west-2": "10.3.0.0/16"}
    assert a.peerings == b.peerings == [("us-east-1", "us-west-2")]      # requester has the lower index


def test_subnets_one_per_az_inside_the_vpc():
    plan = N.plan_network(["us-east-2"], {"us-east-2": ["c", "a", "b"]}, "1.2.3.4/32")
    assert plan.subnets["us-east-2"] == {"a": "10.1.0.0/20", "b": "10.1.16.0/20", "c": "10.1.32.0/20"}


def test_security_group_never_world_open_and_includes_every_vpc():
    plan = N.plan_network(list(N.REGION_INDEX), AZS, "203.0.113.7/32")
    for region, rules in plan.sg_rules.items():
        cidrs = {r.cidr for r in rules}
        assert "0.0.0.0/0" not in cidrs
        assert set(plan.vpcs.values()) <= cidrs                          # whole mesh may talk to the node
        ssh = [r for r in rules if r.from_port == 22]
        assert [r.cidr for r in ssh] == ["203.0.113.7/32"]


def test_world_open_operator_cidr_rejected():
    for bad in ("0.0.0.0/0", "1.2.3.4"):
        with pytest.raises(N.NetPlanError):
            N.plan_network(["us-east-1"], AZS, bad)


def test_unknown_region_and_missing_azs_rejected():
    with pytest.raises(N.NetPlanError):
        N.plan_network(["eu-west-1"], {"eu-west-1": ["a"]}, "1.2.3.4/32")
    with pytest.raises(N.NetPlanError):
        N.plan_network(["us-east-1"], {}, "1.2.3.4/32")


def test_validate_plan_detects_a_missing_route_and_overlap():
    plan = N.plan_network(["us-east-1", "us-east-2", "us-west-2"], AZS, "1.2.3.4/32")
    broken = dataclasses.replace(plan, routes=plan.routes[:-1])
    assert any("missing route" in p for p in N.validate_plan(broken))
    overlapping = dataclasses.replace(plan, vpcs={**plan.vpcs, "us-east-2": plan.vpcs["us-east-1"]})
    assert any("overlapping" in p for p in N.validate_plan(overlapping))
