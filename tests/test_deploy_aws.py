"""Deploy and teardown against moto's emulated EC2 (exec #7 step 3), plus pure tests of the AWS-facing builders.

moto stands in for AWS because this repo's CI has no credentials; the live gates (G1, G5) still run on real AWS.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

moto = pytest.importorskip("moto")
import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from powder.deploy import aws, aws_net, aws_nodes, cli, deploy as dp, spec as S, teardown as td  # noqa: E402
from powder.deploy.remote import Result  # noqa: E402
from powder.deploy.rundir import RunDir  # noqa: E402
from powder.deploy.state import RunState  # noqa: E402

REGIONS = ["us-east-1", "us-east-2", "us-west-2"]
# controller + 3 PD + 3 TiKV + 3 system TiKV + 3 TiDB + 3 load generators
N_NODES = 16
_AMIS: dict = {}


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, node, command, timeout=600, check=True):
        self.calls.append((node, command))
        return Result(0, "", "")

    def put(self, node, local, remote, timeout=600):
        pass

    def get(self, node, remote, local, timeout=600):
        pass


def small_spec(handle="auto"):
    slots = {f"tikv-{i + 1}": {"machine": f"machine/ford10-google-storage@{r}"} for i, r in enumerate(REGIONS)}
    return S.parse_spec({"handle": handle, "slots": slots, "limits": {"max_hourly_usd": 50}})


@pytest.fixture
def aws_env(monkeypatch, tmp_path):
    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_SESSION_TOKEN": "testing",
                 "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("POWDER_ALLOW_SPEND", raising=False)
    with moto.mock_aws():
        # moto refuses to store the reserved /aws/service/... SSM parameter, so only the AMI lookup is stubbed
        if not _AMIS:           # moto's default AMI ids are static: look them up once per session
            _AMIS.update({r: boto3.client("ec2", region_name=r).describe_images(Owners=["amazon"])["Images"][0]["ImageId"]
                          for r in aws.ALL_REGIONS})
        amis = _AMIS
        monkeypatch.setattr(aws_nodes, "resolve_ami", lambda ctx, region: amis[region])
        yield tmp_path


def do_deploy(tmp_path, spec=None, runner=None, **kw):
    spec = spec or small_spec("OrangePanda")
    rd = RunDir(tmp_path / "out", spec.handle)
    clients = aws.Clients()
    runner = runner or FakeRunner()
    state = dp.deploy(spec, rd, clients, lambda st: runner, "203.0.113.7/32", quotas=kw.pop("quotas", None), yes=True,
                      log=lambda m: None, sleep=lambda s: None, status_checks=False, **kw)
    return spec, rd, clients, state, runner


def count(ec2, what, **kw):
    return {"v": lambda: ec2.describe_vpcs(**kw)["Vpcs"], "i": lambda: [i for r in ec2.describe_instances(**kw)["Reservations"]
            for i in r["Instances"] if i["State"]["Name"] != "terminated"]}[what]()


# ---------------------------------------------------------------- pure builders
def _ctx():
    return aws_net.Ctx(aws.Clients.__new__(aws.Clients), "OrangePanda", "reggie", "2026-11-03T00:00:00+00:00", RunState("OrangePanda", "h"))


def test_run_params_security_and_tags():
    node = S.NodePlan("tikv-2", "tikv", "us-east-1", "i4i.2xlarge", "tikv-2", "machine/ford10-google-storage")
    p = aws_nodes.run_params(_ctx(), node, "us-east-1b", "subnet-1", "sg-1", "ami-1", "powder-OrangePanda", "#!/bin/sh")
    assert p["MetadataOptions"]["HttpTokens"] == "required"                       # IMDSv2 only
    assert p["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert p["NetworkInterfaces"][0]["AssociatePublicIpAddress"] is True
    assert p["BlockDeviceMappings"][0]["Ebs"]["VolumeType"] == "gp3"
    assert {t["ResourceType"] for t in p["TagSpecifications"]} == {"instance", "volume", "network-interface"}
    tags = aws.tags_to_dict(p["TagSpecifications"][0]["Tags"])
    assert tags["Project"] == "powder" and tags["Run"] == "OrangePanda" and tags["Owner"] == "reggie"
    assert tags["Name"] == "OrangePanda-tikv-2" and tags["Slot"] == "tikv-2" and tags["Role"] == "tikv"
    assert tags["Machine"] == "machine/ford10-google-storage" and tags["CreatedAt"]
    assert p["Placement"] == {"AvailabilityZone": "us-east-1b"} and p["UserData"] == "#!/bin/sh"


def test_client_token_is_stable_unique_and_short():
    a = aws.client_token("OrangePanda", "tikv-1")
    assert a == aws.client_token("OrangePanda", "tikv-1")
    assert len({a, aws.client_token("OrangePanda", "tikv-2"), aws.client_token("BlueFalcon", "tikv-1"),
                aws.client_token("OrangePanda", "tikv-1", "us-east-1b")}) == 4
    assert len(a) <= 64


def test_tags_are_sanitised_and_complete():
    t = aws.tags_to_dict(aws.tag_list("OrangePanda", "role/we<ird>", "2026-01-01T00:00:00+00:00", "vpc"))
    assert set(t) == {"Project", "Run", "Owner", "CreatedAt", "Name"} and "<" not in t["Owner"]


def _runs():
    now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
    mk = lambda h, hours: td.RunResources(h, "me", (now - timedelta(hours=hours)).isoformat())     # noqa: E731
    return {"OrangePanda": mk("OrangePanda", 1), "QuietFalcon": mk("QuietFalcon", 30), "BoldHeron": mk("BoldHeron", 50)}, now


def test_select_by_handle_all_and_age():
    runs, now = _runs()
    assert td.select(runs, ["OrangePanda", "OrangePanda"]) == ["OrangePanda"]
    assert td.select(runs, all_runs=True) == ["BoldHeron", "OrangePanda", "QuietFalcon"]
    assert td.select(runs, older_than_s=24 * 3600, now=now) == ["BoldHeron", "QuietFalcon"]
    assert td.select(runs, older_than_s=100 * 3600, now=now) == []


def test_select_rejects_unknown_handles_and_ambiguous_requests():
    runs, _ = _runs()
    with pytest.raises(td.SelectionError, match="NopeNope"):
        td.select(runs, ["OrangePanda", "NopeNope"])
    for kw in ({}, {"handles": ["OrangePanda"], "all_runs": True}, {"all_runs": True, "older_than_s": 5}):
        with pytest.raises(td.SelectionError):
            td.select(runs, **kw)


def test_fmt_age():
    assert [td.fmt_age(x) for x in (None, 90, 3 * 3600 + 600, 26 * 3600)] == ["?", "1m", "3h10m", "1d2h"]


def test_spend_guard():
    with pytest.raises(dp.SpendNotAllowed):
        dp.check_spend_allowed(False, {})
    dp.check_spend_allowed(True, {})
    dp.check_spend_allowed(False, {"POWDER_ALLOW_SPEND": "1"})
    with pytest.raises(dp.SpendNotAllowed):
        dp.check_spend_allowed(False, {"POWDER_ALLOW_SPEND": "0"})


def test_bootstrap_script_is_valid_bash():
    import subprocess
    assert subprocess.run(["bash", "-n", str(aws_nodes.BOOTSTRAP_SCRIPT)]).returncode == 0
    text = aws_nodes.BOOTSTRAP_SCRIPT.read_text()
    assert "nodelalloc,noatime" in text and "bootstrap.done" in text and "swapoff" in text


# ---------------------------------------------------------------- deploy against moto
def test_deploy_creates_the_whole_tagged_mesh(aws_env):
    spec, rd, clients, state, runner = do_deploy(aws_env)
    assert state.stages["infra"] == "done" and len(state.instances) == N_NODES
    assert state.est_hourly_usd > 0 and rd.state_path.exists() and rd.manifest_path.exists()
    for r in REGIONS:
        ec2 = clients.ec2(r)
        assert len(ec2.describe_vpcs(Filters=aws.run_filter("OrangePanda"))["Vpcs"]) == 1
        assert len(ec2.describe_internet_gateways(Filters=aws.run_filter("OrangePanda"))["InternetGateways"]) == 1
        assert len(ec2.describe_key_pairs(Filters=aws.run_filter("OrangePanda"))["KeyPairs"]) == 1
        net = state.regions[r]
        assert net.vpc_id and net.security_group_id and net.route_table_id and net.subnets and net.key_name
    # every instance: tagged at launch, running, with addresses recorded
    for rec in state.instances.values():
        assert rec.state == "running" and rec.private_ip and rec.instance_id
    insts = [i for r in REGIONS for i in count(clients.ec2(r), "i")]
    assert len(insts) == N_NODES
    for i in insts:
        t = aws.tags_to_dict(i["Tags"])
        assert t["Project"] == "powder" and t["Run"] == "OrangePanda" and t["Node"] and t["Role"]
    tikv = [aws.tags_to_dict(i["Tags"]) for i in insts if aws.tags_to_dict(i["Tags"])["Role"] == "tikv"]
    assert sorted(t["Slot"] for t in tikv) == ["tikv-1", "tikv-2", "tikv-3"]
    # bootstrap was awaited on every node, controller first
    waited = [n for n, c in runner.calls if "bootstrap.done" in c]
    assert waited[0] == "controller" and set(waited) == set(state.instances)


def test_peering_mesh_is_active_with_routes_on_both_sides(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    ids = set()
    for r in REGIONS:
        ec2 = clients.ec2(r)
        pcx = ec2.describe_vpc_peering_connections(Filters=aws.run_filter("OrangePanda"))["VpcPeeringConnections"]
        assert all(p["Status"]["Code"] == "active" for p in pcx)
        ids |= {p["VpcPeeringConnectionId"] for p in pcx}
        routes = ec2.describe_route_tables(RouteTableIds=[state.regions[r].route_table_id])["RouteTables"][0]["Routes"]
        peer_routes = [x for x in routes if x.get("VpcPeeringConnectionId")]
        assert len(peer_routes) == 2                                   # one per other region
        assert any(x.get("GatewayId", "").startswith("igw-") for x in routes)
    assert len(ids) == 3                                               # C(3,2)


def test_peering_waits_while_peer_region_says_not_found(aws_env, monkeypatch):
    # real AWS (gate G1): the accepter region raises NotFound for a few seconds after a cross-region request
    real_ec2 = aws.Clients.ec2
    misses = {"n": 0}

    class Lagging:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def describe_vpc_peering_connections(self, **kw):
            if "VpcPeeringConnectionIds" in kw and misses["n"] < 6:
                misses["n"] += 1
                raise ClientError({"Error": {"Code": "InvalidVpcPeeringConnectionID.NotFound", "Message": "lag"}},
                                  "DescribeVpcPeeringConnections")
            return self._inner.describe_vpc_peering_connections(**kw)

    monkeypatch.setattr(aws.Clients, "ec2", lambda self, region: Lagging(real_ec2(self, region)))
    _, _, clients, state, _ = do_deploy(aws_env)
    assert misses["n"] == 6 and state.stages["infra"] == "done"


def test_security_group_is_never_world_open_for_ingress(aws_env):
    _, _, clients, state, _ = do_deploy(aws_env)
    for r in REGIONS:
        sg = clients.ec2(r).describe_security_groups(GroupIds=[state.regions[r].security_group_id])["SecurityGroups"][0]
        cidrs = [rng["CidrIp"] for perm in sg["IpPermissions"] for rng in perm["IpRanges"]]
        assert "0.0.0.0/0" not in cidrs and "203.0.113.7/32" in cidrs


def test_deploy_is_idempotent(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    before = {r: (len(count(clients.ec2(r), "v")), len(count(clients.ec2(r), "i"))) for r in REGIONS}
    do_deploy(aws_env, spec=spec)                                      # same spec, same handle, same directory
    after = {r: (len(count(clients.ec2(r), "v")), len(count(clients.ec2(r), "i"))) for r in REGIONS}
    assert before == after


def test_resume_after_partial_failure_launches_only_what_is_missing(aws_env, monkeypatch):
    spec = small_spec("OrangePanda")
    real = aws_nodes.launch_node
    n = {"c": 0}

    def flaky(ctx, node, *a, **k):
        n["c"] += 1
        if n["c"] == 6:
            raise RuntimeError("boom")
        return real(ctx, node, *a, **k)

    monkeypatch.setattr(aws_nodes, "launch_node", flaky)
    with pytest.raises(RuntimeError):
        do_deploy(aws_env, spec=spec)
    rd = RunDir(aws_env / "out", "OrangePanda")
    partial = rd.load_state()
    assert partial.stages["infra"] == "partial" and 0 < len(partial.instances) < N_NODES
    monkeypatch.setattr(aws_nodes, "launch_node", real)
    _, _, clients, state, _ = do_deploy(aws_env, spec=spec)
    assert len(state.instances) == N_NODES
    assert sum(len(count(clients.ec2(r), "i")) for r in REGIONS) == N_NODES        # nothing double-launched


def test_state_survives_a_lost_state_file_via_tags(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    rd.state_path.unlink()
    runs = td.discover(clients)
    assert set(runs) == {"OrangePanda"} and runs["OrangePanda"].instance_count() == N_NODES


def test_changed_spec_with_same_handle_is_refused(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    other = S.parse_spec({**spec.to_dict() | {"slots": {f"tikv-{i + 1}": {"machine": f"machine/ford10-google-storage-ge15min@{r}"}
                                                          for i, r in enumerate(REGIONS)}}, "handle": "OrangePanda"} if False else
                         {"handle": "OrangePanda", "slots": {f"tikv-{i + 1}": {"machine": f"machine/ford10-google-storage-ge15min@{r}"}
                                                              for i, r in enumerate(REGIONS)}, "limits": {"max_hourly_usd": 50}})
    with pytest.raises(dp.DeployError, match="different spec"):
        do_deploy(aws_env, spec=other)


def test_spend_guard_blocks_before_any_aws_call(aws_env):
    spec = small_spec("OrangePanda")
    rd = RunDir(aws_env / "out", "OrangePanda")
    with pytest.raises(dp.SpendNotAllowed):
        dp.deploy(spec, rd, aws.Clients(), lambda s: FakeRunner(), "203.0.113.7/32", yes=False, environ={},
                  log=lambda m: None)
    assert count(boto3.client("ec2", region_name="us-east-1"), "v") == count(boto3.client("ec2", region_name="us-east-1"), "v")
    assert td.discover(aws.Clients()) == {} and not rd.exists()


def test_preflight_failures_create_nothing(aws_env):
    with pytest.raises(dp.DeployError, match="quota"):
        do_deploy(aws_env, quotas={"us-east-1": 8, "us-east-2": 8, "us-west-2": 8})
    d = S.load_yaml(__import__("pathlib").Path("infra/tidb/specs/smoke.yaml"))
    d.update(handle="OrangePanda", limits={"max_hourly_usd": 1})
    with pytest.raises(dp.DeployError, match="max_hourly_usd"):
        do_deploy(aws_env, spec=S.parse_spec(d))
    assert td.discover(aws.Clients()) == {}


class _CapacityProxy:
    """Wraps an EC2 client: the first RunInstances in a given AZ fails with InsufficientInstanceCapacity."""

    def __init__(self, inner, bad_az):
        self._inner, self._bad, self.failures = inner, bad_az, 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def run_instances(self, **kw):
        if kw["Placement"]["AvailabilityZone"] == self._bad:
            self.failures += 1
            raise ClientError({"Error": {"Code": "InsufficientInstanceCapacity", "Message": "no capacity"}}, "RunInstances")
        return self._inner.run_instances(**kw)


def test_capacity_error_falls_back_to_another_az(aws_env):
    clients = aws.Clients()
    ec2 = clients.ec2("us-east-1")
    ctx_azs = None
    spec = small_spec("OrangePanda")
    rd = RunDir(aws_env / "out", "OrangePanda")
    # discover the AZ the first TiKV would pick, then make exactly that AZ fail
    state = RunState("OrangePanda", spec.spec_hash())
    ctx = aws_net.Ctx(clients, "OrangePanda", "me", state.created_at, state)
    azs = aws_nodes.choose_azs(ctx, "us-east-1", S.plan_nodes(spec))
    assert len(azs) >= 2
    proxy = _CapacityProxy(ec2, azs[0])
    clients._cache[("ec2", "us-east-1")] = proxy
    s = dp.deploy(spec, rd, clients, lambda st: FakeRunner(), "203.0.113.7/32", yes=True, log=lambda m: None,
                  sleep=lambda x: None, status_checks=False)
    assert proxy.failures >= 1 and len(s.instances) == N_NODES
    assert all(r.az != azs[0] for r in s.instances.values() if r.region == "us-east-1")


def test_pinned_az_does_not_fall_back(aws_env):
    clients = aws.Clients()
    d = small_spec("OrangePanda").to_dict()
    sp = S.parse_spec({"handle": "OrangePanda", "limits": {"max_hourly_usd": 50}, "slots": {
        "tikv-1": {"machine": "machine/ford10-google-storage@us-east-1", "az": "us-east-1a"},
        "tikv-2": {"machine": "machine/ford10-google-storage@us-east-2"},
        "tikv-3": {"machine": "machine/ford10-google-storage@us-west-2"}}})
    clients._cache[("ec2", "us-east-1")] = _CapacityProxy(clients.ec2("us-east-1"), "us-east-1a")
    with pytest.raises(ClientError):
        dp.deploy(sp, RunDir(aws_env / "out", "OrangePanda"), clients, lambda st: FakeRunner(), "203.0.113.7/32",
                  yes=True, log=lambda m: None, sleep=lambda x: None, status_checks=False)


# ---------------------------------------------------------------- teardown against moto
def test_teardown_removes_everything_and_verify_is_clean(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    runs = td.discover(clients)
    res = td.teardown_run(clients, runs["OrangePanda"], sleep=lambda s: None, log=lambda m: None)
    assert res["errors"] == [] and res["deleted"]["instances"] == N_NODES and res["deleted"]["vpcs"] == 3
    assert td.verify(clients) == []
    for r in REGIONS:
        assert count(clients.ec2(r), "i") == [] and count(clients.ec2(r), "v", Filters=aws.run_filter()) == []


def test_teardown_is_tolerant_when_resources_are_already_gone(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    runs = td.discover(clients)
    td.teardown_run(clients, runs["OrangePanda"], sleep=lambda s: None, log=lambda m: None)
    again = td.teardown_run(clients, runs["OrangePanda"], sleep=lambda s: None, log=lambda m: None)   # stale snapshot
    assert again["errors"] == []


def test_verify_reports_a_leaked_resource(aws_env):
    clients = aws.Clients()
    ec2 = clients.ec2("us-west-2")
    ec2.create_vpc(CidrBlock="10.9.0.0/16", TagSpecifications=aws.tag_spec("vpc", aws.tag_list(
        "GhostRun", "me", aws.utcnow(), "vpc")))
    left = td.verify(clients)
    assert len(left) == 1 and left[0].startswith("GhostRun in us-west-2")


def test_teardown_cli_by_handle_leaves_other_deployment_untouched(aws_env):
    spec_a, rd_a, clients, st_a, _ = do_deploy(aws_env, spec=small_spec("OrangePanda"))
    spec_b, rd_b, _, st_b, _ = do_deploy(aws_env, spec=small_spec("QuietFalcon"))
    out = []
    rc = cli.teardown_main(["OrangePanda", "--yes", "--out-root", str(aws_env / "out")], clients=clients,
                           out=out.append, sleep=lambda s: None)
    assert rc == 0, out
    runs = td.discover(clients)
    assert set(runs) == {"QuietFalcon"} and runs["QuietFalcon"].instance_count() == N_NODES
    assert json.loads(rd_a.teardown_path.read_text())["errors"] == []
    assert rd_a.load_state().instances["controller"].state == "terminated"
    assert not rd_b.teardown_path.exists()


def test_teardown_cli_unknown_handle_confirmation_and_list(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    out = []
    root = str(aws_env / "out")
    assert cli.teardown_main(["NopeNope", "--yes", "--out-root", root], clients=clients, out=out.append) == cli.REFUSED
    assert cli.teardown_main(["OrangePanda", "--out-root", root], clients=clients, out=out.append,
                             input_fn=lambda p: "no") == cli.REFUSED
    assert len(td.discover(clients)["OrangePanda"].regions) == 3                       # nothing deleted
    assert cli.teardown_main(["OrangePanda", "--dry-run", "--out-root", root], clients=clients, out=out.append) == 0
    assert td.discover(clients)["OrangePanda"].instance_count() == N_NODES
    out.clear()
    assert cli.teardown_main(["--list", "--out-root", root], clients=clients, out=out.append) == 0
    table = out[0].splitlines()
    assert table[0].split()[:2] == ["HANDLE", "AGE"] and table[1].startswith("OrangePanda")
    assert f"={N_NODES}" in table[1] and table[1].rstrip().endswith("yes")
    assert cli.teardown_main(["--verify", "--out-root", root], clients=clients, out=out.append) == 1     # still deployed
    assert cli.teardown_main(["OrangePanda", "--yes", "--out-root", root], clients=clients, out=out.append,
                             sleep=lambda s: None) == 0
    assert cli.teardown_main(["--verify", "--out-root", root], clients=clients, out=out.append) == 0


def test_list_flags_a_deployment_without_a_local_directory(aws_env):
    spec, rd, clients, state, _ = do_deploy(aws_env)
    text = td.format_list(td.discover(clients), aws_env / "elsewhere")
    assert "OrangePanda" in text and text.rstrip().endswith("no")


# ---------------------------------------------------------------- deploy CLI
def test_deploy_cli_end_to_end_and_resume(aws_env, monkeypatch):
    monkeypatch.setattr(aws, "detect_operator_cidr", lambda: "203.0.113.7/32")
    specfile = aws_env / "spec.yaml"
    import yaml
    d = small_spec("auto").to_dict()
    specfile.write_text(yaml.safe_dump({"slots": {s["name"]: {"machine": f"{s['machine']}@{s['region']}"} for s in d["slots"]},
                                        "limits": {"max_hourly_usd": 50}}))
    out = []
    root = str(aws_env / "out")
    args = ["--spec", str(specfile), "--out-root", root, "--skip-quota-check"]
    assert cli.deploy_main(args, clients=aws.Clients(), runner_factory=lambda st: FakeRunner(), out=out.append,
                           sleep=lambda s: None) == cli.NO_SPEND                         # no --yes
    assert td.discover(aws.Clients()) == {}
    assert cli.deploy_main(args + ["--yes"], clients=aws.Clients(), runner_factory=lambda st: FakeRunner(),
                           out=out.append, sleep=lambda s: None) == 0, out
    handle = next(l.split(": ")[1] for l in out if l.startswith("deployment handle:"))
    assert handle in td.discover(aws.Clients()) and any(f"teardown_infra.py {handle}" in l for l in out)
    assert cli.deploy_main(args + ["--yes", "--resume"], clients=aws.Clients(), runner_factory=lambda st: FakeRunner(),
                           out=out.append, sleep=lambda s: None) == cli.BAD_INPUT         # resume needs --handle
    assert cli.deploy_main(args + ["--yes", "--resume", "--handle", handle], clients=aws.Clients(),
                           runner_factory=lambda st: FakeRunner(), out=out.append, sleep=lambda s: None) == 0
    assert td.discover(aws.Clients())[handle].instance_count() == N_NODES


def test_deploy_cli_rejects_a_taken_explicit_handle(aws_env, monkeypatch):
    monkeypatch.setattr(aws, "detect_operator_cidr", lambda: "203.0.113.7/32")
    do_deploy(aws_env, spec=small_spec("OrangePanda"))
    import yaml
    d = small_spec("auto").to_dict()
    specfile = aws_env / "spec.yaml"
    specfile.write_text(yaml.safe_dump({"slots": {s["name"]: {"machine": f"{s['machine']}@{s['region']}"} for s in d["slots"]},
                                        "limits": {"max_hourly_usd": 50}}))
    from powder.deploy import names
    with pytest.raises(names.HandleError):
        cli.deploy_main(["--spec", str(specfile), "--out-root", str(aws_env / "out"), "--handle", "OrangePanda", "--yes",
                         "--skip-quota-check"], clients=aws.Clients(), runner_factory=lambda st: FakeRunner(),
                        out=lambda m: None, sleep=lambda s: None)


def test_deploy_cli_dry_run_makes_no_aws_calls(aws_env):
    out = []
    assert cli.deploy_main(["--spec", "infra/tidb/specs/smoke.yaml", "--dry-run"], out=out.append) == 0
    assert "AWS bill" in out[0] and td.discover(aws.Clients()) == {}
