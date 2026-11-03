"""Install stage: health gate on real-shaped PD documents, TiUP output parsing, and the stage sequence (exec #7 step 4)."""

import json
import re
from pathlib import Path

import pytest

from powder.deploy import install as I
from powder.deploy import spec as S
from powder.deploy.remote import Result
from powder.deploy.rundir import RunDir
from powder.deploy.state import InstanceRecord, RunState

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "infra" / "tidb" / "specs" / "smoke.yaml"
REAL_STORES = json.loads((ROOT / "tests/fixtures/tidb/stores/00-baseline.json").read_text())

DISPLAY = """Cluster type:       tidb
Cluster name:       OrangePanda
Cluster version:    v8.5.8
Deploy user:        tidb
Dashboard URL:      http://10.0.0.2:2379/dashboard
Grafana URL:        http://10.0.0.1:3000
ID              Role        Host      Ports        OS/Arch       Status   Data Dir               Deploy Dir
--              ----        ----      -----        -------       ------   --------               ----------
10.0.0.1:3000   grafana     10.0.0.1  3000         linux/x86_64  Up       -                      /tidb-deploy/grafana-3000
10.0.0.2:2379   pd          10.0.0.2  2379/2380    linux/x86_64  Up|L|UI  /tidb-data/pd-2379     /tidb-deploy/pd-2379
10.0.0.5:20160  tikv        10.0.0.5  20160/20180  linux/x86_64  Up       /data/tikv             /tidb-deploy/tikv-20160
10.0.0.11:4000  tidb        10.0.0.11 4000/10080   linux/x86_64  Up       -                      /tidb-deploy/tidb-4000
Total nodes: 4
"""


def make_state(tmp_path, infra="done"):
    spec = S.parse_spec({**S.load_yaml(SMOKE), "handle": "OrangePanda"})
    st = RunState("OrangePanda", spec.spec_hash())
    st.set_stage("infra", infra)
    ips = {}
    for i, n in enumerate(S.plan_nodes(spec), start=1):
        idx = list(__import__("powder.faults.regions", fromlist=["x"]).AWS_US_REGIONS).index(n.region)
        ip = f"10.{idx}.0.{i}"
        ips[n.name] = ip
        st.add_instance(InstanceRecord(n.name, n.role, n.region, n.instance_type, az=n.region + "a", instance_id=f"i-{i}",
                                       private_ip=ip, public_ip=f"54.0.0.{i}", slot=n.slot, machine=n.machine, state="running"))
    rd = RunDir(tmp_path, "OrangePanda").ensure()
    key = rd.key_path
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text("PRIVATE KEY")
    st.key_path = str(key)
    st.save(rd.state_path)
    return spec, rd, st, ips


def good_docs(spec, st):
    hosts = I.hosts_from_state(st)
    want = I.expected_store_labels(spec, hosts)
    stores = {"count": len(want), "stores": []}
    for i, (ip, labels) in enumerate(want.items(), start=1):
        s = json.loads(json.dumps(REAL_STORES["stores"][0]))                 # real store document shape
        s["store"].update(id=i, address=f"{ip}:20160", state_name="Up",
                          labels=[{"key": k, "value": v} for k, v in labels.items()])
        stores["stores"].append(s)
    members = {"members": [{"name": f"pd-{i}"} for i in range(1, spec.control_plane.pd + 1)], "leader": {"name": "pd-1"}}
    sql = {n.name: "1\n" for n in S.plan_nodes(spec) if n.role == "tidb"}
    return hosts, members, stores, sql


# ---------------------------------------------------------------- pure parsing / gate
def test_parse_display_rows_and_unhealthy_detection():
    rows = I.parse_display(DISPLAY)
    assert set(rows) == {"10.0.0.1:3000", "10.0.0.2:2379", "10.0.0.5:20160", "10.0.0.11:4000"}
    assert rows["10.0.0.2:2379"] == {"role": "pd", "host": "10.0.0.2", "status": "Up|L|UI"}
    assert I.display_problems(rows) == []
    rows["10.0.0.5:20160"]["status"] = "Down"
    assert I.display_problems(rows) == ["10.0.0.5:20160 (tikv) is Down"]
    assert I.display_problems({}) == ["tiup cluster display returned no instances"]


def test_health_gate_accepts_a_healthy_cluster(tmp_path):
    spec, rd, st, _ = make_state(tmp_path)
    hosts, members, stores, sql = good_docs(spec, st)
    assert I.check_health(spec, hosts, members, stores, sql, "200") == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda m, s, q, g: m["members"].pop(), "PD has 2 members, expected 3"),
    (lambda m, s, q, g: m.update(leader=None), "no leader"),
    (lambda m, s, q, g: s["stores"][0]["store"].update(state_name="Down"), "is Down"),
    (lambda m, s, q, g: s["stores"][1]["store"].update(state_name="Disconnected"), "is Disconnected"),
    (lambda m, s, q, g: s["stores"].pop(), "no store registered"),
    (lambda m, s, q, g: s["stores"][2]["store"]["labels"].pop(3), "labels differ"),
    (lambda m, s, q, g: s["stores"][0]["store"]["labels"][0].update(value="eu-west-1"), "labels differ"),
    (lambda m, s, q, g: s["stores"].append(json.loads(json.dumps(s["stores"][0]))) or
     s["stores"][-1]["store"].update(address="10.9.9.9:20160", id=99), "unexpected store"),
    (lambda m, s, q, g: q.update({"tidb-us-east-1-1": "error: refused"}), "SELECT 1"),
    (lambda m, s, q, g: g.__setitem__(0, "502"), "Grafana"),
])
def test_health_gate_problems(tmp_path, mutate, needle):
    spec, rd, st, _ = make_state(tmp_path)
    hosts, members, stores, sql = good_docs(spec, st)
    graf = ["200"]
    mutate(members, stores, sql, graf)
    problems = I.check_health(spec, hosts, members, stores, sql, graf[0])
    assert problems and any(needle in p for p in problems), problems


def test_expected_labels_use_the_nodes_az_and_machine_class(tmp_path):
    spec, rd, st, _ = make_state(tmp_path)
    want = I.expected_store_labels(spec, I.hosts_from_state(st))
    assert len(want) == 6
    ip = st.by_slot("tikv-2").private_ip
    assert want[ip] == {"region": "us-east-1", "zone": "us-east-1a", "host": "tikv-2",
                        "machine_class": "ford10-google-storage-ge15min"}


def test_pinned_versions_come_from_the_versions_file():
    p = I.pinned_versions()
    assert p["tidb"] == "v8.5.8" and p["tiup"].startswith("v1.") and p["cluster"].startswith("v1.")


# ---------------------------------------------------------------- stage sequence
class Scripted:
    """Runner whose answers are chosen by regex over the command; records everything."""

    def __init__(self, spec, st, docs, fail=None, unhealthy_first=0):
        self.spec, self.st, self.docs, self.fail = spec, st, docs, fail or {}
        self.calls, self.puts, self.health_calls, self.unhealthy_first = [], [], 0, unhealthy_first

    def run(self, node, command, timeout=600, check=True):
        self.calls.append((node, command))
        for rx, res in self.fail.items():
            if re.search(rx, command):
                if check and res.returncode:
                    from powder.deploy.remote import RemoteError
                    raise RemoteError(f"{command}: {res.stderr}")
                return res
        hosts, members, stores, sql = self.docs
        if "/pd/api/v1/members" in command:
            self.health_calls += 1
            if self.health_calls <= self.unhealthy_first:
                return Result(0, json.dumps({"members": [], "leader": None}))
            return Result(0, json.dumps(members))
        if "/pd/api/v1/stores" in command:
            return Result(0, json.dumps(stores))
        if command.startswith("mysql"):
            return Result(0, "1\n")
        if "api/health" in command:
            return Result(0, "200")
        if " display " in command:
            return Result(0, DISPLAY)
        if "tiup" in command and " check " in command:
            return Result(0, "Node  Check  Result  Message\n10.0.0.5  cpu-governor  Fail  advisory\n10.0.0.5  swap  Pass  ok\n")
        return Result(0, "")

    def put(self, node, local, remote, timeout=600):
        self.puts.append((node, Path(local).name, remote))

    def get(self, *a, **k):
        pass


def run_install(tmp_path, **kw):
    spec, rd, st, _ = make_state(tmp_path, infra=kw.pop("infra", "done"))
    docs = good_docs(spec, st)
    runner = Scripted(spec, st, docs, **kw)
    out = I.install(spec, rd, runner, log=lambda m: None, sleep=lambda s: None, health_timeout=30)
    return spec, rd, runner, out


def test_install_sequence_and_artifacts(tmp_path):
    spec, rd, runner, state = run_install(tmp_path)
    cmds = [c for _, c in runner.calls]
    order = [next(i for i, c in enumerate(cmds) if re.search(rx, c)) for rx in (
        r"fetch_tidb\.sh tiup cluster tidb tikv pd ctl", r"tar xzf .*tiup-v1\.", r"tiup install cluster:v1\.",
        r" check ", r" deploy OrangePanda v8\.5\.8 ", r" start OrangePanda", r"/pd/api/v1/members", r" display OrangePanda")]
    assert order == sorted(order)                                              # verify -> tiup -> check -> deploy -> start -> health -> display
    assert all(n == "controller" for n, _ in runner.calls)                      # everything runs on the controller
    assert ("controller", "fetch_tidb.sh", "~/powder/infra/tidb/fetch_tidb.sh") in runner.puts
    assert any(p[1] == "VERSIONS" for p in runner.puts) and any(p[2] == I.CONTROLLER_KEY for p in runner.puts)
    topo = (rd.install_dir / "topology.yaml").read_text()
    assert "machine_class: ford10-google-storage-ge15min" in topo and "OrangePanda" in topo
    assert (rd.install_dir / "check.txt").exists() and (rd.install_dir / "display.txt").exists()
    assert state.stages["install"] == "done" and rd.load_state().stages["install"] == "done"
    m = json.loads(rd.manifest_path.read_text())
    assert m["tidb_version"] == "v8.5.8" and m["check_advisory_failures"] == 1 and m["pins"]["pd"] == "v8.5.8"


def test_health_gate_retries_until_healthy(tmp_path):
    _, _, runner, state = run_install(tmp_path, unhealthy_first=3)
    assert runner.health_calls == 4 and state.stages["install"] == "done"


def test_unhealthy_cluster_fails_the_gate_and_stays_partial(tmp_path):
    spec, rd, st, _ = make_state(tmp_path)
    runner = Scripted(spec, st, good_docs(spec, st), unhealthy_first=10 ** 6)
    ticks = iter(range(0, 10 ** 6, 20))                                        # fake clock: +20 s per look
    with pytest.raises(I.InstallError, match="health gate failed"):
        I.install(spec, rd, runner, log=lambda m: None, sleep=lambda s: None, health_timeout=60,
                  clock=lambda: float(next(ticks)))
    assert rd.load_state().stages["install"] == "partial"


def test_artifact_hash_failure_stops_before_anything_is_deployed(tmp_path):
    spec, rd, st, _ = make_state(tmp_path)
    runner = Scripted(spec, st, good_docs(spec, st),
                      fail={r"fetch_tidb\.sh": Result(1, "", "HASH MISMATCH for pd v8.5.8")})
    with pytest.raises(I.InstallError, match="HASH MISMATCH"):
        I.install(spec, rd, runner, log=lambda m: None, sleep=lambda s: None)
    assert not any(" deploy " in c for _, c in runner.calls) and not any(" start " in c for _, c in runner.calls)
    assert rd.load_state().stages["install"] == "partial"


def test_deploy_failure_is_fatal(tmp_path):
    spec, rd, st, _ = make_state(tmp_path)
    runner = Scripted(spec, st, good_docs(spec, st), fail={r" deploy ": Result(1, "", "ssh: connect failed")})
    from powder.deploy.remote import RemoteError
    with pytest.raises(RemoteError, match="connect failed"):
        I.install(spec, rd, runner, log=lambda m: None, sleep=lambda s: None)


def test_install_requires_finished_infra_and_matching_version(tmp_path):
    spec, rd, st, _ = make_state(tmp_path, infra="partial")
    with pytest.raises(I.InstallError, match="no finished infrastructure"):
        I.install(spec, rd, Scripted(spec, st, good_docs(spec, st)), log=lambda m: None)
    spec2, rd2, st2, _ = make_state(tmp_path / "b")
    spec2.tidb_version = "v8.1.0"
    with pytest.raises(I.InstallError, match="pins v8.5.8"):
        I.install(spec2, rd2, Scripted(spec2, st2, good_docs(spec2, st2)), log=lambda m: None)


def test_install_cli_loads_the_runs_own_spec_and_reports_failures(tmp_path):
    from powder.deploy import cli
    spec, rd, st, _ = make_state(tmp_path)
    rd.spec_path.write_text(SMOKE.read_text())                       # what deploy_infra copies into the run directory
    out = []
    runner = Scripted(spec, st, good_docs(spec, st))
    assert cli.install_main(["--handle", "OrangePanda", "--out-root", str(tmp_path)], runner_factory=lambda s: runner,
                            out=out.append, sleep=lambda s: None) == cli.OK
    assert rd.load_state().stages["install"] == "done"
    assert cli.install_main(["--handle", "NopeNope", "--out-root", str(tmp_path)]) == cli.BAD_INPUT
    bad = Scripted(spec, st, good_docs(spec, st), fail={r"fetch_tidb\.sh": Result(1, "", "HASH MISMATCH")})
    assert cli.install_main(["--handle", "OrangePanda", "--out-root", str(tmp_path)], runner_factory=lambda s: bad,
                            out=out.append, sleep=lambda s: None) == cli.FAIL
