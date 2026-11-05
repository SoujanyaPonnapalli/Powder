"""Stage 3: install and start TiDB on a deployed run with TiUP, then gate on cluster health (plan §6.2).

Everything runs on the **controller** (inside the peering mesh) over SSH:

1. ship ``fetch_tidb.sh`` + ``third_party/tidb/VERSIONS`` and re-verify every pinned tarball against its sha256
   (a changed artifact stops the run before anything is deployed);
2. install the pinned ``tiup`` binary, then the pinned ``tiup-cluster`` component;
3. render the TiUP topology from spec + state (private IPs, ``region/zone/host/machine_class`` store labels);
4. ``tiup cluster check --apply`` (output saved; warnings are logged, not fatal, see below), ``deploy``, ``start``;
5. **health gate**: PD members and a leader; every store ``Up`` with exactly the expected labels; ``SELECT 1``
   through every TiDB; Grafana reachable.

``tiup cluster check`` reports many advisory items on cloud VMs (CPU governor, disk IO ...); failing the run on all
of them would make it undeployable, so they are saved to ``install/check.txt`` and counted, while deploy/start
errors and the health gate are fatal. Tune this list at live gate G2.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from .remote import RemoteError, Runner, wait_for
from .rundir import RunDir
from .spec import ClusterSpec, plan_nodes
from .state import RunState, utcnow
from .placement import MAX_SYSTEM_ID_SQL, placement_problems, rule_bundles
from .regionmap import from_pd_json
from .topology import (LOCATION_LABELS, TIKV_DATA_DIR, Host, label_value, machine_class_label, render_topology,
                       store_labels)

ROOT = Path(__file__).resolve().parents[2]
VERSIONS_FILE = ROOT / "third_party" / "tidb" / "VERSIONS"
FETCH_SCRIPT = ROOT / "infra" / "tidb" / "fetch_tidb.sh"
CONTROLLER_KEY = "~/.ssh/powder_run"
CONTROLLER_HOME = "~/powder"
TIUP_MIRROR = "https://tiup-mirrors.pingcap.com"       # same default as fetch_tidb.sh
START_WAIT_S = 1200                                    # tiup cluster start --wait-timeout (default 120)
PD_PORT, TIDB_PORT, TIDB_STATUS_PORT, GRAFANA_PORT, PROM_PORT = 2379, 4000, 10080, 3000, 9090


class InstallError(RuntimeError):
    pass


def pinned_versions(path: Path = VERSIONS_FILE) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            out[parts[0]] = parts[1]
    return out


def hosts_from_state(state: RunState) -> dict[str, Host]:
    return {n: Host(n, r.private_ip, r.region, r.az) for n, r in state.instances.items()}


# ------------------------------------------------------------------ parsing / health gate (pure)
def parse_display(text: str) -> dict[str, dict]:
    """``tiup cluster display`` -> {instance id: {role, host, status}} (status like 'Up', 'Up|L|UI', 'N/A')."""
    rows, in_table = {}, False
    for line in text.splitlines():
        if re.match(r"^-{2,}\s+-{2,}", line):
            in_table = True
            continue
        if not in_table or not line.strip():
            continue
        f = line.split()                    # no column contains spaces; robust to uneven padding
        if len(f) >= 6:
            rows[f[0]] = {"role": f[1], "host": f[2], "status": f[5]}
    return rows


def display_problems(rows: dict[str, dict]) -> list[str]:
    if not rows:
        return ["tiup cluster display returned no instances"]
    return [f"{i} ({r['role']}) is {r['status']}" for i, r in rows.items()
            if not r["status"].startswith("Up") and r["role"] not in ("alertmanager",)]


def expected_store_labels(spec: ClusterSpec, hosts: dict[str, Host]) -> dict[str, dict[str, str]]:
    """private IP of each TiKV node -> the labels its store must carry."""
    out = {}
    for n in plan_nodes(spec):
        if n.role in ("tikv", "systikv"):
            h = hosts[n.name]
            out[h.private_ip] = store_labels(n, h)
    return out


def check_health(spec: ClusterSpec, hosts: dict[str, Host], members: dict, stores: dict,
                 tidb_sql: dict[str, str], grafana_status: str) -> list[str]:
    """All problems found (empty list = healthy). Inputs are PD's ``/members`` and ``/stores`` JSON documents."""
    p: list[str] = []
    want_pd = spec.control_plane.pd
    got = members.get("members", [])
    if len(got) != want_pd:
        p.append(f"PD has {len(got)} members, expected {want_pd}")
    if not members.get("leader"):
        p.append("PD has no leader")

    want = expected_store_labels(spec, hosts)
    seen = {}
    for s in stores.get("stores", []):
        st = s["store"]
        ip = st["address"].rsplit(":", 1)[0]
        seen[ip] = st
        if st.get("state_name") != "Up":
            p.append(f"store {st['id']} at {ip} is {st.get('state_name')}")
    for ip, labels in want.items():
        st = seen.get(ip)
        if st is None:
            p.append(f"no store registered for TiKV node at {ip}")
            continue
        have = {l["key"]: l["value"] for l in st.get("labels", [])}
        bad = {k: (have.get(k), v) for k, v in labels.items() if have.get(k) != v}
        if bad:
            p.append(f"store at {ip} labels differ (have, want): {bad}")
    for ip in set(seen) - set(want):
        p.append(f"unexpected store registered at {ip}")

    for name, out in sorted(tidb_sql.items()):
        if out.strip() != "1":
            p.append(f"SELECT 1 through {name} returned {out.strip()!r}")
    if grafana_status.strip() != "200":
        p.append(f"Grafana health returned HTTP {grafana_status.strip() or 'nothing'}")
    return p


# ------------------------------------------------------------------ remote helpers
def pd_url(spec: ClusterSpec, hosts: dict[str, Host]) -> str:
    pds = sorted(n.name for n in plan_nodes(spec) if n.role == "pd")
    return f"http://{hosts[pds[0]].private_ip}:{PD_PORT}"


def _tiup(cluster_v: str, args: str) -> str:
    return f"~/.tiup/bin/tiup cluster:{cluster_v} {args}"


def gather_health(runner: Runner, spec: ClusterSpec, hosts: dict[str, Host]) -> tuple[dict, dict, dict, str]:
    base = pd_url(spec, hosts)
    members = json.loads(runner.run("controller", f"curl -sf --max-time 10 {base}/pd/api/v1/members", timeout=30).stdout)
    stores = json.loads(runner.run("controller", f"curl -sf --max-time 10 {base}/pd/api/v1/stores", timeout=30).stdout)
    sql = {}
    for n in plan_nodes(spec):
        if n.role == "tidb":
            r = runner.run("controller", f"mysql -h {hosts[n.name].private_ip} -P {TIDB_PORT} -u root -N -e 'select 1'",
                           timeout=30, check=False)
            sql[n.name] = r.stdout if r.returncode == 0 else f"error: {r.stderr.strip()[:100]}"
    ctl = hosts["controller"].private_ip
    g = runner.run("controller", f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 10 http://{ctl}:{GRAFANA_PORT}/api/health",
                   timeout=30, check=False)
    return members, stores, sql, g.stdout


def pin_system_data(spec: ClusterSpec, hosts: dict[str, Host], rundir: RunDir, runner: Runner,
                    log: Callable[[str], None], sleep, clock, timeout: float) -> int:
    """Put TiDB's own data on the system stores and user tables on the experiment stores (powder.deploy.placement);
    wait until PD has moved every Region. Returns the first user table id."""
    tidb = next(hosts[n.name].private_ip for n in plan_nodes(spec) if n.role == "tidb")
    r = runner.run("controller", f'mysql -h {tidb} -P {TIDB_PORT} -u root -N -e "{MAX_SYSTEM_ID_SQL}"', timeout=60)
    first_user = int(r.stdout.strip().splitlines()[-1]) + 1
    sys_regions = len({n.region for n in plan_nodes(spec) if n.role == "systikv"})
    rules = rule_bundles(first_user, spec.replication_factor, LOCATION_LABELS + ["machine_class"], sys_regions)
    path = rundir.install_dir / "placement-rules.json"
    path.write_text(json.dumps(rules, indent=1))
    runner.put("controller", path, f"{CONTROLLER_HOME}/placement-rules.json")
    pd = pd_url(spec, hosts)
    log(f"[{spec.handle}] pinning TiDB's data (ids < {first_user}, meta, reserved) to the system stores")
    runner.run("controller", f"curl -sf -X POST -H 'Content-Type: application/json' "
                             f"--data @{CONTROLLER_HOME}/placement-rules.json {pd}/pd/api/v1/config/placement-rule",
               timeout=60)
    last: list[str] = []

    def placed() -> bool:
        nonlocal last
        regions = json.loads(runner.run("controller", f"curl -sf --max-time 60 {pd}/pd/api/v1/regions", timeout=90).stdout)
        stores = json.loads(runner.run("controller", f"curl -sf --max-time 20 {pd}/pd/api/v1/stores", timeout=30).stdout)
        last = placement_problems(from_pd_json(regions, 0.0, stores), first_user, spec.replication_factor)
        if last:
            log(f"  placement: {len(last)} Regions still moving (e.g. {last[0]})")
        return not last
    try:
        wait_for(placed, timeout=timeout, interval=15, what="system-data placement", sleep=sleep, clock=clock)
    except RemoteError as e:
        raise InstallError(f"placement did not converge: {last[:5] or e}") from e
    rundir.update_manifest(first_user_table_id=first_user, system_data_pinned=True)
    return first_user


def install(spec: ClusterSpec, rundir: RunDir, runner: Runner, log: Callable[[str], None] = print,
            sleep=time.sleep, health_timeout: float = 300.0, deploy_timeout: float = 1800.0,
            clock=time.monotonic, placement_timeout: float = 900.0) -> RunState:
    state = rundir.load_state()
    if state.stages.get("infra") != "done":
        raise InstallError(f"deployment {state.handle} has no finished infrastructure (infra: {state.stages.get('infra')})")
    pins = pinned_versions()
    tidb_v, tiup_v, cluster_v = pins["tidb"], pins["tiup"], pins["cluster"]
    if spec.tidb_version != tidb_v:
        raise InstallError(f"spec wants TiDB {spec.tidb_version} but third_party/tidb/VERSIONS pins {tidb_v}; "
                           "re-pin (infra/tidb/pin_versions.py) or change the spec")
    hosts = hosts_from_state(state)
    rundir.install_dir.mkdir(parents=True, exist_ok=True)
    state.set_stage("install", "partial")
    state.save(rundir.state_path)

    log(f"[{state.handle}] shipping fetch script and pins to the controller; verifying tarball hashes")
    runner.run("controller", f"mkdir -p {CONTROLLER_HOME}/infra/tidb {CONTROLLER_HOME}/third_party/tidb ~/.ssh", timeout=60)
    runner.put("controller", FETCH_SCRIPT, f"{CONTROLLER_HOME}/infra/tidb/fetch_tidb.sh")
    runner.put("controller", VERSIONS_FILE, f"{CONTROLLER_HOME}/third_party/tidb/VERSIONS")
    r = runner.run("controller", f"bash {CONTROLLER_HOME}/infra/tidb/fetch_tidb.sh tiup cluster tidb tikv pd ctl",
                   timeout=deploy_timeout, check=False)
    if r.returncode != 0:
        raise InstallError(f"artifact verification failed on the controller:\n{r.stdout}\n{r.stderr}")

    log(f"[{state.handle}] installing tiup {tiup_v} and tiup-cluster {cluster_v}")
    cache = f"~/.cache/powder/tidb/{tidb_v}"
    runner.run("controller", f"mkdir -p ~/.tiup/bin && tar xzf {cache}/tiup-{tiup_v}-linux-amd64.tar.gz -C ~/.tiup/bin "
                             f"&& chmod +x ~/.tiup/bin/tiup && ~/.tiup/bin/tiup --version", timeout=120)
    # tiup needs the mirror's root.json (its trust anchor) next to the binary, as the official installer puts it (G2)
    runner.run("controller", f"curl -fsSL --retry 3 -o ~/.tiup/bin/root.json {TIUP_MIRROR}/root.json", timeout=120)
    runner.run("controller", f"~/.tiup/bin/tiup install cluster:{cluster_v} ctl:{tidb_v}", timeout=600)
    runner.run("controller", "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-client-core-8.0 >/dev/null",
               timeout=600)

    log(f"[{state.handle}] rendering topology")
    topo = render_topology(spec, hosts)
    topo_path = rundir.install_dir / "topology.yaml"
    topo_path.write_text(topo)
    key = Path(state.key_path)
    runner.put("controller", key, CONTROLLER_KEY)
    runner.run("controller", f"chmod 600 {CONTROLLER_KEY}", timeout=30)
    runner.put("controller", topo_path, f"{CONTROLLER_HOME}/topology.yaml")

    ssh = f"--user ubuntu -i {CONTROLLER_KEY}"
    log(f"[{state.handle}] tiup cluster check --apply")
    chk = runner.run("controller", _tiup(cluster_v, f"check {CONTROLLER_HOME}/topology.yaml {ssh} --apply"),
                     timeout=deploy_timeout, check=False)
    (rundir.install_dir / "check.txt").write_text(chk.stdout + "\n" + chk.stderr)
    fails = sum(1 for l in chk.stdout.splitlines() if re.search(r"\bFail\b", l))
    log(f"  check finished (exit {chk.returncode}, {fails} advisory failures, see install/check.txt)")

    # a re-run after a failed start continues: tiup refuses to deploy a cluster name it already has
    known = runner.run("controller", f"test -d ~/.tiup/storage/cluster/clusters/{state.handle}", timeout=30, check=False)
    if known.returncode == 0:
        log(f"[{state.handle}] tiup already has cluster {state.handle}; skipping deploy")
    else:
        log(f"[{state.handle}] tiup cluster deploy {state.handle} {tidb_v}")
        runner.run("controller", _tiup(cluster_v, f"deploy {state.handle} {tidb_v} {CONTROLLER_HOME}/topology.yaml {ssh} -y"),
                   timeout=deploy_timeout)
    # node_bootstrap.sh creates the NVMe data dir as root before tiup creates the tidb user, and tiup does not chown
    # an existing data dir: without this TiKV cannot open RocksDB (G2)
    for n in plan_nodes(spec):
        if n.role in ("tikv", "systikv"):
            runner.run(n.name, f"sudo chown -R tidb:tidb {TIKV_DATA_DIR}", timeout=60)
    log(f"[{state.handle}] tiup cluster start")
    # the first TiDB bootstraps the system schema with one cross-region DDL round per step (~2-3 s each, G2): far
    # beyond tiup's default 120 s port wait
    runner.run("controller", _tiup(cluster_v, f"start {state.handle} --wait-timeout {START_WAIT_S}"),
               timeout=deploy_timeout)

    log(f"[{state.handle}] health gate")
    last: list[str] = []

    def healthy() -> bool:
        nonlocal last
        members, stores, sql, graf = gather_health(runner, spec, hosts)
        last = check_health(spec, hosts, members, stores, sql, graf)
        return not last
    try:
        wait_for(healthy, timeout=health_timeout, interval=10, what="cluster health", sleep=sleep, clock=clock)
    except RemoteError as e:
        raise InstallError(f"health gate failed: {last or e}") from e

    if any(n.role == "systikv" for n in plan_nodes(spec)):
        pin_system_data(spec, hosts, rundir, runner, log, sleep, clock, placement_timeout)

    disp = runner.run("controller", _tiup(cluster_v, f"display {state.handle}"), timeout=120).stdout
    (rundir.install_dir / "display.txt").write_text(disp)
    problems = display_problems(parse_display(disp))
    if problems:
        raise InstallError("tiup cluster display shows unhealthy instances: " + "; ".join(problems))
    comps = runner.run("controller", "ls ~/.tiup/components/ ; for c in ~/.tiup/components/*; do echo $c: $(ls $c); done",
                       timeout=60, check=False).stdout
    (rundir.install_dir / "components.txt").write_text(comps)

    from .loadgen_build import LoadgenBuildError, ensure_loadgen
    try:
        loadgen_version = ensure_loadgen(runner, state, log)
    except LoadgenBuildError as e:
        raise InstallError(f"load generator build failed: {e}") from e

    state.set_stage("install", "done")
    state.save(rundir.state_path)
    rundir.update_manifest(installed_at=utcnow(), loadgen=loadgen_version, tidb_version=tidb_v, tiup_version=tiup_v, tiup_cluster_version=cluster_v,
                           pins=pins, check_advisory_failures=fails)
    log(f"[{state.handle}] installed: {len(parse_display(disp))} instances up")
    return state
