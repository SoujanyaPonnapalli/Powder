#!/usr/bin/env python3
"""Developer check: run the REAL experiment pipeline against a real TiDB on this machine (no AWS).

    python3 infra/tidb/local_pipeline_check.py [--duration 70] [--pause-at 20] [--pause-for 20]

Starts ``tiup playground`` (3 TiKV, so every Region has all three replicas) from the hash-pinned tiup in
``~/.cache/powder`` (fetch_tidb.sh), builds ``powder-load`` with the local Go toolchain, then runs
``powder.deploy.experiment.run_experiment`` through ``LocalRunner``: ``prepare``, PD snapshotter, probers and load
generators, collection and ``summary.json``. Two TiKV processes are paused (SIGSTOP) for ``--pause-for`` seconds, so
quorum is lost for every Region; the summary must show roughly that much unavailability, with causes recorded.
This validates the Go tool and the metric chain on real TiDB output; AWS-specific behaviour (SSH, TiUP deploy,
cross-region networking) is covered by the live gates G1-G5, not here. Stops the playground on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.deploy import experiment as E, spec as S  # noqa: E402
from powder.deploy.install import pinned_versions  # noqa: E402
from powder.deploy.localrunner import LocalRunner  # noqa: E402
from powder.deploy.rundir import RunDir  # noqa: E402
from powder.deploy.state import InstanceRecord, RunState  # noqa: E402

CACHE = Path.home() / ".cache" / "powder"
TIUP_HOME = CACHE / "tiup-home"


def wait_http(url: str, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"timed out waiting for {url}")


def wait_sql(timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        r = subprocess.run(["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/4000"], capture_output=True)
        if r.returncode == 0:
            time.sleep(3)
            return
        time.sleep(2)
    raise SystemExit("timed out waiting for TiDB on :4000")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--duration", type=int, default=70)
    ap.add_argument("--pause-at", type=int, default=20)
    ap.add_argument("--pause-for", type=int, default=20)
    ap.add_argument("--rate", type=float, default=100)
    ap.add_argument("--keep-dir", type=Path, help="keep results here instead of a temp dir")
    a = ap.parse_args()

    tidb_v = pinned_versions()["tidb"]
    env = {**os.environ, "TIUP_HOME": str(TIUP_HOME)}
    go = shutil.which("go") or str(CACHE / "go-root" / "go" / "bin" / "go")
    work = Path(a.keep_dir or tempfile.mkdtemp(prefix="powder-local-"))
    work.mkdir(parents=True, exist_ok=True)

    print("== building powder-load")
    binary = work / "powder-load"
    subprocess.run([go, "build", "-o", str(binary), "."], cwd=ROOT / "infra/tidb/loadgen", check=True,
                   env={**os.environ, "GOFLAGS": "-mod=mod", "CGO_ENABLED": "0"})

    tiup = TIUP_HOME / "bin" / "tiup"
    if not tiup.exists():
        subprocess.run([str(ROOT / "infra/tidb/fetch_tidb.sh"), "--extract", "tiup"], check=True)
        TIUP_HOME.mkdir(parents=True, exist_ok=True)
        (TIUP_HOME / "bin").mkdir(exist_ok=True)
        shutil.copy(CACHE / "tidb" / tidb_v / f"tiup-{pinned_versions()['tiup']}" / "tiup", tiup)
    print("== starting a local TiDB (tiup playground: 1 PD, 3 TiKV, 1 TiDB)")
    log = open(work / "playground.log", "w")
    pg = subprocess.Popen([str(tiup), "playground", tidb_v, "--db", "1", "--pd", "1", "--kv", "3", "--tiflash", "0",
                           "--without-monitor", "--host", "127.0.0.1", "--tag", "powderlocal"], stdout=log, stderr=log,
                          env=env, start_new_session=True)
    try:
        wait_http("http://127.0.0.1:2379/pd/api/v1/stores", 300)
        wait_sql(300)
        pdctl = str(CACHE / "tidb" / tidb_v / f"ctl-{tidb_v}" / "pd-ctl")
        if not Path(pdctl).exists():
            subprocess.run([str(ROOT / "infra/tidb/fetch_tidb.sh"), "--extract", "ctl"], check=True)

        spec = S.parse_spec({**S.load_yaml(ROOT / "infra/tidb/specs/smoke.yaml"), "handle": "LocalCheck"})
        st = RunState("LocalCheck", spec.spec_hash())
        st.set_stage("infra", "done")
        st.set_stage("install", "done")
        for n in S.plan_nodes(spec):
            st.add_instance(InstanceRecord(n.name, n.role, n.region, n.instance_type, az=n.region + "a", instance_id=n.name,
                                           private_ip="127.0.0.1", public_ip="127.0.0.1", slot=n.slot, machine=n.machine,
                                           state="running"))
        rd = RunDir(work / "out", "LocalCheck").ensure()
        st.save(rd.state_path)
        E.WORK_ROOT = str(work / "remote")

        cfg = S.ExperimentConfig(name="localcheck", workload=S.Workload(
            duration_s=a.duration, keys=100000, regions_target=8, rate_per_region=a.rate), region_map_interval_s=2.0)
        stop = "pkill -STOP -f 'tikv-server --addr=127.0.0.1:2016[01]'"
        cont = "pkill -CONT -f 'tikv-server --addr=127.0.0.1:2016[01]'"
        sched = [E.ScheduledAction(a.pause_at, "controller", stop, "pause 2 of 3 TiKV"),
                 E.ScheduledAction(a.pause_at + a.pause_for, "controller", cont, "resume")]
        print(f"== running the experiment: {a.duration}s, pausing 2 of 3 TiKV at +{a.pause_at}s for {a.pause_for}s")
        runner = LocalRunner()
        exp = E.run_experiment(spec, cfg, rd, runner, schedule=sched, loadgen_bin=str(binary), pdctl=f"{pdctl}",
                               pd_url_override="http://127.0.0.1:2379", log=print)

        # PD log from the playground feeds the operator-lifecycle parser (on AWS collect_everything fetches it)
        pdlog = TIUP_HOME / "data" / "powderlocal" / "pd-0" / "pd.log"
        if pdlog.exists():
            dst = exp / "logs" / "pd-1" / "pd.log"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pdlog, dst)
        from powder.deploy import collect
        prep = json.loads((exp / "prepare.json").read_text())
        summary = collect.summarize_experiment(exp, cfg, int(prep["table_id"]), 3)
        summary["handle"] = "LocalCheck"
        collect.write_summary(exp, summary)

        wl, pr = summary["workload"]["overall"], summary["probe"]["aggregate"]
        print("\n== result")
        print(f"workload: {wl['total']} requests, {wl['timeout']} timeouts -> unavailability {wl['unavailability']:.3f} "
              f"(conflicts {wl['conflict']}, other {wl['other']})")
        print("causes:", {k: v["count"] for k, v in summary["workload"]["causes"].items()})
        print(f"probes: worst partition unavailability {pr['worst_unavailability_all']:.3f}, mean {pr['mean_unavailability_all']:.3f}, "
              f"coverage {pr['min_coverage']:.2f}; expected ~{a.pause_for / a.duration:.3f} plus recovery")
        print("repartition:", {k: v for k, v in summary["repartition"]["counts"].items() if v},
              "| operators from PD log:", summary["pd_log"]["operators"], "| unparsed log lines:", summary["pd_log"]["unparsed"])
        print("collection problems:", *summary["collection_errors"], sep="\n  ")
        print(f"\nresults kept in {exp}")
        return 0
    finally:
        os.killpg(os.getpgid(pg.pid), signal.SIGINT)
        try:
            pg.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(pg.pid), signal.SIGKILL)
        subprocess.run(["pkill", "-CONT", "-f", "tikv-server"], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
