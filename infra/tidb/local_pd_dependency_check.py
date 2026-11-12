#!/usr/bin/env python3
"""Off-cloud TiDB v8.5.8 request behavior with one PD process stopped.

Starts a uniquely tagged local tiup playground (1 PD, 3 TiKV, 1 TiDB),
exercises SQL using sessions established while PD is healthy, suspends only
that playground's PD PID, repeats requests, resumes PD, and tears down the
process group in finally. This measures observed SQL success against a 5 s
client deadline; it does not model a 3-member PD failover or AWS networking.

    ~/.cache/powder/venv/bin/python infra/tidb/local_pd_dependency_check.py \
        --output /tmp/powder-pd-check.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parents[2]
TIUP_HOME = Path.home() / ".cache/powder/tiup-home"
TIUP = TIUP_HOME / "bin/tiup"
TIDB_VERSION = "v8.5.8"  # third_party/tidb/VERSIONS


def wait_ready(pg: subprocess.Popen, seconds: float = 180) -> None:
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if pg.poll() is not None:
            raise RuntimeError(f"playground exited with {pg.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:2379/pd/api/v1/health", timeout=1) as r:
                if r.status == 200:
                    c = connect()
                    c.close()
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("local PD/TiDB did not become ready")


def connect() -> pymysql.connections.Connection:
    return pymysql.connect(host="127.0.0.1", port=4000, user="root", password="",
                           database=None, autocommit=True, connect_timeout=3,
                           read_timeout=5, write_timeout=5)


def pd_pid_for_tag(tag: str) -> int:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if not args or Path(os.fsdecode(args[0])).name != "pd-server":
            continue
        if any(tag.encode() in arg for arg in args[1:] if arg.startswith((b"--data-dir", b"--name"))):
            matches.append(int(proc.name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one PD for tag {tag}, found {matches}; refusing to signal any")
    return matches[0]


def request(label: str, phase: str, sql: str, conn=None, since_stop: float | None = None) -> dict:
    own = conn is None
    t0 = time.monotonic()
    r = {"phase": phase, "label": label, "sql": sql, "since_pd_stop_s": since_stop}
    try:
        c = conn or connect()
        with c.cursor() as cur:
            cur.execute(sql)
            data = cur.fetchall() if cur.description else []
            r["rows"] = [list(row) for row in data[:3]]
            r["rowcount"] = cur.rowcount
        r["outcome"] = "ok"
    except Exception as exc:
        r["outcome"] = "failed"
        r["error_type"] = type(exc).__name__
        r["error"] = str(exc)[:300]
    finally:
        r["elapsed_s"] = round(time.monotonic() - t0, 3)
        if own and "c" in locals():
            try:
                c.close()
            except Exception:
                pass
    return r


def pd_health(phase: str, since_stop: float | None = None) -> dict:
    start = time.monotonic()
    r = {"phase": phase, "label": "pd_http_health", "since_pd_stop_s": since_stop}
    try:
        with urllib.request.urlopen("http://127.0.0.1:2379/pd/api/v1/health", timeout=1) as response:
            r["http_status"] = response.status
            r["outcome"] = "ok" if response.status == 200 else "failed"
    except Exception as exc:
        r.update(outcome="failed", error_type=type(exc).__name__, error=str(exc)[:300])
    r["elapsed_s"] = round(time.monotonic() - start, 3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--hold-until-s", type=float, default=0.0,
                    help="keep PD stopped this many seconds before late probes (e.g. 60)")
    a = ap.parse_args()
    if not TIUP.is_file():
        ap.error(f"pinned TiUP not found: {TIUP}; see infra/tidb/local_pipeline_check.py")
    tag = f"powderpdcheck{os.getpid()}"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = a.output.with_suffix(".playground.log")
    records: list[dict] = []
    sessions = {}
    pd_pid = None
    pg = None
    started = time.monotonic()
    with log_path.open("w") as log:
        pg = subprocess.Popen([str(TIUP), "playground", TIDB_VERSION, "--db", "1", "--pd", "1",
                               "--kv", "3", "--tiflash", "0", "--without-monitor",
                               "--host", "127.0.0.1", "--tag", tag],
                              cwd=ROOT, env={**os.environ, "TIUP_HOME": str(TIUP_HOME)},
                              stdout=log, stderr=log, start_new_session=True)
        try:
            wait_ready(pg)
            pd_pid = pd_pid_for_tag(tag)
            records.append(pd_health("healthy"))
            db = f"pdcheck_{os.getpid()}"
            for statement in (f"CREATE DATABASE {db}",
                              f"CREATE TABLE {db}.t (id INT PRIMARY KEY, v INT)",
                              f"INSERT INTO {db}.t VALUES " + ",".join(
                                  f"({i},{10*i})" for i in range(1, 12))):
                r = request("setup", "healthy", statement)
                records.append(r)
                if r["outcome"] != "ok":
                    raise RuntimeError(f"setup failed: {r}")
            snap = connect()
            with snap.cursor() as cur:
                cur.execute("SELECT NOW(6)")
                snapshot = str(cur.fetchone()[0])
            snap.close()
            time.sleep(2)  # historical timestamp safely precedes the AS OF query
            queries = {
                "constant": "SELECT 1",
                "show_tables": f"SHOW TABLES FROM {db}",
                "point_get": f"SELECT v FROM {db}.t WHERE id=1",
                "scan_read": f"SELECT SUM(v) FROM {db}.t",
                "scan_read_late": f"SELECT SUM(v) FROM {db}.t",
                "scan_read_very_late": f"SELECT SUM(v) FROM {db}.t",
                "fixed_snapshot_read": f"SELECT v FROM {db}.t AS OF TIMESTAMP '{snapshot}' WHERE id=1",
                "fixed_snapshot_scan": f"SELECT SUM(v) FROM {db}.t AS OF TIMESTAMP '{snapshot}'",
                "cached_tso_scan_early": f"SELECT SUM(v) FROM {db}.t",
                "cached_tso_scan_late": f"SELECT SUM(v) FROM {db}.t",
                "constant_late": "SELECT 1",
                "autocommit_write": f"UPDATE {db}.t SET v=v+1 WHERE id=2",
                "autocommit_write_very_late": f"UPDATE {db}.t SET v=v+1 WHERE id=8",
            }
            for name in queries:
                sessions[name] = connect()
                if name.startswith("cached_tso"):
                    r = request("enable_cached_tso_" + name, "healthy",
                                "SET @@tidb_low_resolution_tso=ON", sessions[name])
                    records.append(r)
            for name, sql in queries.items():
                records.append(request(name, "healthy", sql, sessions[name]))
            # Both transactions begin and read while PD is healthy. Baseline uses
            # a separate session; the second session stays open across the fault.
            txn = connect()
            records.append(request("existing_txn_begin", "healthy", "BEGIN", txn))
            records.append(request("existing_txn_read", "healthy", f"SELECT v FROM {db}.t WHERE id=3", txn))
            baseline_txn = connect()
            for label, sql in (("txn_begin", "BEGIN"), ("txn_read", f"SELECT v FROM {db}.t WHERE id=1"),
                               ("txn_update", f"UPDATE {db}.t SET v=v+1 WHERE id=1"),
                               ("txn_commit", "COMMIT")):
                records.append(request(label, "healthy", sql, baseline_txn))
            baseline_txn.close()
            # Determine which commit protocol TiDB *actually* picked in the
            # healthy case. Enabling a protocol does not guarantee its use.
            modes = {
                "two_phase": ("OFF", "OFF", "BEGIN", 4),
                "one_phase_eligible": ("ON", "OFF", "BEGIN", 5),
                "async_eligible": ("OFF", "ON", "BEGIN", 6),
                "causal_consistency": ("ON", "ON", "START TRANSACTION WITH CAUSAL CONSISTENCY ONLY", 7),
            }
            mode_sessions = {}
            for mode, (one_pc, async_commit, begin_sql, row_id) in modes.items():
                for phase, holder in (("healthy", None), ("healthy_prepared", mode_sessions)):
                    c = connect()
                    if holder is not None:
                        holder[mode] = c
                    for label, sql in (("enable_1pc", f"SET SESSION tidb_enable_1pc={one_pc}"),
                                       ("enable_async", f"SET SESSION tidb_enable_async_commit={async_commit}"),
                                       ("begin", begin_sql),
                                       ("snapshot_read", f"SELECT v FROM {db}.t WHERE id={row_id}")):
                        records.append(request(f"{mode}_{phase}_{label}", "healthy", sql, c))
                    if holder is None:
                        records.append(request(f"{mode}_healthy_update", "healthy",
                                               f"UPDATE {db}.t SET v=v+1 WHERE id={row_id}", c))
                        records.append(request(f"{mode}_healthy_commit", "healthy", "COMMIT", c))
                        records.append(request(f"{mode}_healthy_actual_mode", "healthy",
                                               "SELECT @@tidb_last_txn_info", c))
                        c.close()
            late_causal = connect()
            for label, sql in (("enable_1pc", "SET SESSION tidb_enable_1pc=ON"),
                               ("enable_async", "SET SESSION tidb_enable_async_commit=ON"),
                               ("begin", "START TRANSACTION WITH CAUSAL CONSISTENCY ONLY"),
                               ("snapshot_read", f"SELECT v FROM {db}.t WHERE id=9")):
                records.append(request(f"late_causal_{label}", "healthy", sql, late_causal))
            if any(r["outcome"] != "ok" for r in records if r["phase"] == "healthy"):
                print("Some baseline cases failed; their PD-down outcomes cannot be attributed to PD alone")
            os.kill(pd_pid, signal.SIGSTOP)
            stopped = time.monotonic()
            records.append(pd_health("pd_stopped", 0.0))
            for name in ("cached_tso_scan_early", "constant", "point_get", "scan_read",
                         "fixed_snapshot_read", "fixed_snapshot_scan", "show_tables", "autocommit_write",
                         "cached_tso_scan_late", "scan_read_late", "constant_late"):
                records.append(request(name, "pd_stopped", queries[name], sessions[name],
                                       round(time.monotonic() - stopped, 2)))
            # The existing transaction's update and commit are separate requests;
            # do not attempt COMMIT after a timed-out update on a broken session.
            upd = request("existing_txn_update", "pd_stopped", f"UPDATE {db}.t SET v=v+1 WHERE id=3",
                          txn, round(time.monotonic() - stopped, 2))
            records.append(upd)
            if upd["outcome"] == "ok":
                records.append(request("existing_txn_commit", "pd_stopped", "COMMIT", txn,
                                       round(time.monotonic() - stopped, 2)))
            for mode, (_one_pc, _async_commit, _begin, row_id) in modes.items():
                c = mode_sessions[mode]
                upd = request(f"{mode}_update", "pd_stopped",
                              f"UPDATE {db}.t SET v=v+1 WHERE id={row_id}", c,
                              round(time.monotonic() - stopped, 2))
                records.append(upd)
                if upd["outcome"] == "ok":
                    committed = request(f"{mode}_commit", "pd_stopped", "COMMIT", c,
                                        round(time.monotonic() - stopped, 2))
                    records.append(committed)
                    if committed["outcome"] == "ok":
                        records.append(request(f"{mode}_actual_mode", "pd_stopped",
                                               "SELECT @@tidb_last_txn_info", c,
                                               round(time.monotonic() - stopped, 2)))
            records.append(request("new_connection_constant", "pd_stopped", "SELECT 1",
                                   since_stop=round(time.monotonic() - stopped, 2)))
            if a.hold_until_s > time.monotonic() - stopped:
                time.sleep(a.hold_until_s - (time.monotonic() - stopped))
                records.append(pd_health("pd_stopped", round(time.monotonic() - stopped, 2)))
                for name in ("constant", "point_get", "fixed_snapshot_scan", "show_tables",
                             "cached_tso_scan_late", "scan_read_very_late", "autocommit_write_very_late"):
                    records.append(request(name + "_after_hold", "pd_stopped", queries[name], sessions[name],
                                           round(time.monotonic() - stopped, 2)))
                late_update = request("late_causal_update", "pd_stopped",
                                      f"UPDATE {db}.t SET v=v+1 WHERE id=9", late_causal,
                                      round(time.monotonic() - stopped, 2))
                records.append(late_update)
                if late_update["outcome"] == "ok":
                    late_commit = request("late_causal_commit", "pd_stopped", "COMMIT", late_causal,
                                          round(time.monotonic() - stopped, 2))
                    records.append(late_commit)
                    if late_commit["outcome"] == "ok":
                        records.append(request("late_causal_actual_mode", "pd_stopped",
                                               "SELECT @@tidb_last_txn_info", late_causal,
                                               round(time.monotonic() - stopped, 2)))
            stopped_duration = round(time.monotonic() - stopped, 2)
            os.kill(pd_pid, signal.SIGCONT)
            time.sleep(3)
            wait_ready(pg, 45)
            records.append(pd_health("restored"))
            for name in ("constant", "point_get", "scan_read", "fixed_snapshot_read",
                         "fixed_snapshot_scan", "autocommit_write"):
                records.append(request(name, "restored", queries[name]))
            result = {"tag": tag, "version": TIDB_VERSION, "pd_member_count": 1,
                      "tikv_member_count": 3, "request_deadline_s": 5,
                      "snapshot": snapshot, "pd_stopped_for_s": stopped_duration,
                      "elapsed_s": round(time.monotonic() - started, 2), "records": records}
            a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
            for phase in ("healthy", "pd_stopped", "restored"):
                print(f"\n{phase}:")
                for r in records:
                    if r["phase"] == phase:
                        print(f"  {r['label']:<28} {r['outcome']:<6} {r['elapsed_s']:>5.2f}s "
                              f"{r.get('error','')[:90]}")
            print(f"\nresults: {a.output}; playground log: {log_path}")
            return 0
        finally:
            if pd_pid is not None:
                try:
                    os.kill(pd_pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            for c in sessions.values():
                try:
                    c.close()
                except Exception:
                    pass
            if "txn" in locals():
                try:
                    txn.close()
                except Exception:
                    pass
            for c in mode_sessions.values() if "mode_sessions" in locals() else []:
                try:
                    c.close()
                except Exception:
                    pass
            if "late_causal" in locals():
                try:
                    late_causal.close()
                except Exception:
                    pass
            if pg is not None and pg.poll() is None:
                os.killpg(pg.pid, signal.SIGINT)
                try:
                    pg.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(pg.pid, signal.SIGKILL)
                    pg.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
