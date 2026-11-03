"""run_experiment end to end through LocalRunner: fake PD API (real captured JSON) + fake powder-load (known outage)."""

import gzip
import http.server
import json
import threading
import time
from pathlib import Path

import pytest

from powder.deploy import experiment as E
from powder.deploy import spec as S
from powder.deploy.localrunner import LocalRunner
from powder.deploy.remote import RemoteError
from powder.deploy.rundir import RunDir
from powder.deploy.state import InstanceRecord, RunState, StateError

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SMOKE = ROOT / "infra" / "tidb" / "specs" / "smoke.yaml"
REGIONS_DOC = json.load(gzip.open(FIX / "tidb" / "regions" / "02-split-t31.json.gz"))
STORES_DOC = json.loads((FIX / "tidb" / "stores" / "00-baseline.json").read_text())


SETTLE = {"miss": [], "ops": []}          # scripted answers for the settle gate, consumed one per poll


class FakePD(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/pd/api/v1/regions/check/miss-peer":
            doc = {"count": SETTLE["miss"].pop(0) if SETTLE["miss"] else 0, "regions": []}
        elif path == "/pd/api/v1/operators":
            doc = [{}] * (SETTLE["ops"].pop(0) if SETTLE["ops"] else 0) or []
        else:
            doc = {"/pd/api/v1/regions": REGIONS_DOC, "/pd/api/v1/stores": STORES_DOC}.get(path)
        body = json.dumps(doc if doc is not None else {}).encode()
        self.send_response(200 if doc is not None else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def pd_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakePD)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make_run(tmp_path, installed=True, loadgens=True):
    spec = S.parse_spec({**S.load_yaml(SMOKE), "handle": "OrangePanda"})
    st = RunState("OrangePanda", spec.spec_hash())
    st.set_stage("infra", "done")
    st.set_stage("install", "done" if installed else "partial")
    for n in S.plan_nodes(spec):
        if n.role == "loadgen" and not loadgens:
            continue
        st.add_instance(InstanceRecord(n.name, n.role, n.region, n.instance_type, az=n.region + "a", instance_id="i-" + n.name,
                                       private_ip="127.0.0.1", public_ip="127.0.0.1", slot=n.slot, machine=n.machine, state="running"))
    rd = RunDir(tmp_path / "out", "OrangePanda").ensure()
    st.save(rd.state_path)
    return spec, rd


def cfg(duration=40, **kw):
    return S.ExperimentConfig(name="t", workload=S.Workload(duration_s=duration, keys=100000, regions_target=8, rate_per_region=1),
                              region_map_interval_s=1.0, **kw)


class FastClock:
    """Our clock advances by the time the stage asks to sleep, so scheduled actions fire without real waiting."""
    def __init__(self):
        self.t = time.time()

    def now(self):
        return self.t

    def sleep(self, s):
        time.sleep(min(s, 0.05))
        self.t += s


def run(tmp_path, pd, monkeypatch, schedule=None, config=None, **kw):
    monkeypatch.setattr(E, "WORK_ROOT", str(tmp_path / "work"))
    spec, rd = make_run(tmp_path)
    clk = FastClock()
    exp = E.run_experiment(spec, config or cfg(), rd, LocalRunner(), schedule=schedule, log=lambda m: None, sleep=clk.sleep,
                           clock=clk.now, loadgen_bin=str(FIX / "fake_powder_load.py"), pdctl=kw.pop("pdctl", "echo"),
                           pd_url_override=pd, exp_id="exp1", **kw)
    return spec, rd, exp


def test_full_run_collects_everything_and_the_numbers_match_the_known_outage(tmp_path, pd_server, monkeypatch):
    spec, rd, exp = run(tmp_path, pd_server, monkeypatch)
    for rel in ("config.json", "prepare.json", "summary.json", "faults/applied.jsonl", "pd/snapshotter.sh"):
        assert (exp / rel).exists(), rel
    for region in ("us-east-1", "us-east-2", "us-west-2"):
        assert (exp / "workload" / region / "requests.jsonl.gz").exists() and (exp / "workload" / region / "probes.jsonl.gz").exists()
    assert list((exp / "pd" / "snapshots").glob("regions-*.json.gz")) and list((exp / "pd" / "snapshots").glob("stores-*.json.gz"))

    cfgj = json.loads((exp / "config.json").read_text())
    assert cfgj["handle"] == "OrangePanda" and cfgj["spec_hash"] == spec.spec_hash() and "git_sha" in cfgj

    s = json.loads((exp / "summary.json").read_text())
    assert s["handle"] == "OrangePanda" and s["experiment_id"] == "exp1" and s["window"]["seconds"] == pytest.approx(40)
    wl = s["workload"]["overall"]
    # 3 client regions x 40 s x 2 requests; the row-80000 request of each region fails for 10 s -> 30 timeouts of 240
    assert (wl["total"], wl["timeout"], wl["ok"]) == (240, 30, 210)
    assert wl["unavailability"] == pytest.approx(30 / 240)
    assert wl["conflict"] == 0 and s["workload"]["causes"]["tidb_9005_region_unavailable"]["count"] == 30
    parts = s["workload"]["by_partition"]
    assert parts["0"]["unavailability"] == 0.0 and parts["6"]["unavailability"] == pytest.approx(10 / 40)
    # RSM attribution used the captured Region map: partition 6 (row 80000) lives in exactly one RSM, which carries it all
    bad_rsm = [k for k, v in s["workload"]["by_rsm"].items() if v["timeout"]]
    assert len(bad_rsm) == 1 and s["workload"]["by_rsm"][bad_rsm[0]]["timeout"] == 30
    assert set(s["workload"]["by_client_region"]) == {"us-east-1", "us-east-2", "us-west-2"}

    pr = s["probe"]
    for b in range(8):
        down = b >= 4
        assert pr["by_partition"][str(b)]["unavailability_all"] == pytest.approx(0.25 if down else 0.0)
        assert pr["by_partition"][str(b)]["unavailable_seconds_all"] == (10 if down else 0)
    assert pr["aggregate"]["worst_unavailability_all"] == pytest.approx(0.25)
    assert pr["aggregate"]["mean_unavailability_all"] == pytest.approx(0.125)
    assert pr["flags"]["low_coverage"] is False

    assert s["snapshots"] >= 1 and s["repartition"]["counts"]["split"] == 0
    # nothing about a cluster that does not exist here may be silently missing
    errs = " | ".join(s["collection_errors"])
    assert "Prometheus" in errs and "logs from" in errs and "no PD logs collected" in errs

    assert S.parse_experiment({"injector": "none"}).injector == "none"
    st = rd.load_state()
    assert st.experiments == ["exp1"] and json.loads(rd.manifest_path.read_text())["last_experiment"] == "exp1"


def test_scheduled_actions_run_on_time_and_are_logged(tmp_path, pd_server, monkeypatch):
    marker = tmp_path / "marker"
    sched = [E.ScheduledAction(3, "loadgen-us-east-1-1", f"echo late > {marker}.b", "second"),
             E.ScheduledAction(1, "loadgen-us-east-1-1", f"echo hi > {marker}.a", "first")]
    _, _, exp = run(tmp_path, pd_server, monkeypatch, schedule=sched)
    assert Path(f"{marker}.a").read_text().strip() == "hi" and Path(f"{marker}.b").read_text().strip() == "late"
    applied = [json.loads(l) for l in (exp / "faults" / "applied.jsonl").read_text().splitlines()]
    assert [a["label"] for a in applied] == ["first", "second"]                 # executed in offset order
    assert all(a["returncode"] == 0 and a["t_applied"] >= a["t_planned"] for a in applied)
    planned = [json.loads(l) for l in (exp / "faults" / "planned.jsonl").read_text().splitlines()]
    assert {p["label"] for p in planned} == {"first", "second"}


def test_pd_runtime_config_is_applied_and_restored(tmp_path, pd_server, monkeypatch):
    log = tmp_path / "pdctl.log"
    fake = tmp_path / "fakepd.py"
    fake.write_text(f"""#!/usr/bin/env python3
import sys, json
a = sys.argv[1:]
if 'show' in a and 'all' in a:
    print(json.dumps({{"schedule": {{"max-store-down-time": "30m0s"}}, "replication": {{"max-replicas": 3}}}}))
else:
    open({str(log)!r}, 'a').write(' '.join(a[2:]) + '\\n')
""")
    c = cfg(pd_runtime_config={"schedule.max-store-down-time": "10m"})
    _, _, exp = run(tmp_path, pd_server, monkeypatch, config=c, pdctl=f"python3 {fake}")
    assert log.read_text().splitlines()[:2] == ["config set max-store-down-time 10m", "config set max-store-down-time 30m0s"]
    assert json.loads((exp / "pd" / "config-before.json").read_text())["schedule"]["max-store-down-time"] == "30m0s"


def test_unknown_pd_option_fails_before_anything_starts(tmp_path, pd_server, monkeypatch):
    fake = tmp_path / "fakepd.py"
    fake.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps({'schedule': {}}))\n")
    with pytest.raises(E.ExperimentError, match="not found"):
        run(tmp_path, pd_server, monkeypatch, config=cfg(pd_runtime_config={"schedule.nope": "1"}), pdctl=f"python3 {fake}")


def test_refuses_without_an_installed_deployment_or_with_reserved_injector(tmp_path, pd_server, monkeypatch):
    monkeypatch.setattr(E, "WORK_ROOT", str(tmp_path / "work"))
    spec, rd = make_run(tmp_path, installed=False)
    with pytest.raises(StateError, match="not installed"):
        E.run_experiment(spec, cfg(), rd, LocalRunner(), log=lambda m: None, pd_url_override=pd_server)
    spec, rd = make_run(tmp_path / "b")
    with pytest.raises(E.ExperimentError, match="exec #8"):
        E.run_experiment(spec, cfg(injector="trace:x.jsonl"), rd, LocalRunner(), log=lambda m: None)
    assert not rd.experiments_dir.exists()                                      # nothing was created


def test_existing_experiment_directory_is_never_overwritten(tmp_path, pd_server, monkeypatch):
    spec, rd, exp = run(tmp_path, pd_server, monkeypatch)
    clk = FastClock()
    with pytest.raises(E.ExperimentError, match="already exists"):
        E.run_experiment(spec, cfg(), rd, LocalRunner(), log=lambda m: None, sleep=clk.sleep, clock=clk.now,
                         loadgen_bin=str(FIX / "fake_powder_load.py"), pdctl="echo", pd_url_override=pd_server, exp_id="exp1")


def test_a_failed_generator_is_reported_not_hidden(tmp_path, pd_server, monkeypatch):
    broken = tmp_path / "broken.sh"
    broken.write_text("#!/bin/sh\nif [ \"$1\" = prepare ]; then echo '{\"table_id\":114,\"keys\":100000,\"regions\":8}'; exit 0; fi\n"
                      "echo boom >&2; exit 3\n")
    broken.chmod(0o755)
    monkeypatch.setattr(E, "WORK_ROOT", str(tmp_path / "work"))
    spec, rd = make_run(tmp_path)
    clk = FastClock()
    exp = E.run_experiment(spec, cfg(), rd, LocalRunner(), log=lambda m: None, sleep=clk.sleep, clock=clk.now,
                           loadgen_bin=str(broken), pdctl="echo", pd_url_override=pd_server, exp_id="exp1")
    s = json.loads((exp / "summary.json").read_text())
    assert "error" in s and any("exited with status 3" in e for e in s["collection_errors"])


def test_nested_get_finds_pd_options():
    doc = {"schedule": {"max-store-down-time": "30m0s"}, "replication": {"max-replicas": 3}, "pd-server": {"x": 1}}
    assert E.nested_get(doc, "schedule.max-store-down-time") == "30m0s"
    assert E.nested_get(doc, "max-replicas") == 3 and E.nested_get(doc, "replication.max-replicas") == 3
    assert E.nested_get(doc, "nope") is None


def test_experiment_waits_for_the_cluster_to_settle_before_starting_load(tmp_path, pd_server, monkeypatch):
    SETTLE["miss"][:] = [5, 2, 0, 0, 0]          # still up-replicating for two polls
    SETTLE["ops"][:] = [0, 3, 0, 0, 0]
    seen = []
    monkeypatch.setattr(E, "WORK_ROOT", str(tmp_path / "work"))
    spec, rd = make_run(tmp_path)
    clk = FastClock()
    E.run_experiment(spec, cfg(), rd, LocalRunner(), log=seen.append, sleep=clk.sleep, clock=clk.now,
                     loadgen_bin=str(FIX / "fake_powder_load.py"), pdctl="echo", pd_url_override=pd_server, exp_id="exp1")
    waits = [l for l in seen if "waiting for the cluster to settle" in l]
    assert len(waits) >= 2 and "5 Regions missing a replica" in waits[0] and "3 operators running" in " ".join(waits)
    assert any("cluster settled" in l for l in seen)
    assert seen.index(next(l for l in seen if "cluster settled" in l)) < seen.index(next(l for l in seen if "starting PD snapshotter" in l))


def test_an_unsettled_cluster_aborts_before_any_load_is_generated(tmp_path, pd_server, monkeypatch):
    SETTLE["miss"][:] = [4] * 100
    monkeypatch.setattr(E, "WORK_ROOT", str(tmp_path / "work"))
    spec, rd = make_run(tmp_path)
    clk = FastClock()
    with pytest.raises(E.ExperimentError, match="did not settle within 20s .*4 Regions missing a replica"):
        E.run_experiment(spec, cfg(), rd, LocalRunner(), log=lambda m: None, sleep=clk.sleep, clock=clk.now,
                         loadgen_bin=str(FIX / "fake_powder_load.py"), pdctl="echo", pd_url_override=pd_server,
                         exp_id="exp1", settle_timeout_s=20, settle_interval_s=10)
    SETTLE["miss"][:] = []
    assert not (rd.experiments_dir / "exp1" / "workload").exists()          # nothing ran
    assert rd.load_state().experiments == []


def test_settle_can_be_disabled(tmp_path, pd_server, monkeypatch):
    SETTLE["miss"][:] = [9] * 100                 # would never settle
    try:
        _, _, exp = run(tmp_path, pd_server, monkeypatch, settle_timeout_s=0)
    finally:
        SETTLE["miss"][:] = []
    assert (exp / "summary.json").exists()
