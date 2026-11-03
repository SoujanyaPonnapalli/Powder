#!/usr/bin/env python3
"""Assemble the committed TiDB fixtures from a raw capture directory.

The raw capture is produced by running a throwaway local cluster
(``tiup playground v8.5.8 --db 1 --pd 1 --kv 4 --tiflash 0``) and the shell
steps described in README.md; this script only trims/compresses/indexes what
was captured so the fixtures stay small and reproducible. It is NOT needed to
use the fixtures.

    python3 -I make_fixtures.py RAW_DIR PD_LOG OUT_DIR

RAW_DIR holds ``<name>.regions.json`` / ``<name>.stores.json`` / ``<name>.ts``
snapshots, ``*.issue.ts`` files for each operator command, metrics dumps
(``pd.metrics.final``, ``tikv{1,2,3}.metrics.final``) and operator history dumps.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sys
from pathlib import Path

RAW = Path(sys.argv[1])
PD_LOG = Path(sys.argv[2])
OUT = Path(sys.argv[3])

TABLE_REGIONS_OF_INTEREST = {"316", "320", "324", "328", "332", "336", "27", "312"}
LIFECYCLE_MSGS = (
    "add operator", "operator finish", "operator timeout", "operator canceled",
    "operator canceled related merge region", "operator is stale", "operator removed",
    "leader changed", "region ConfVer changed", "region Version changed",
    "region batch split, generate new regions", "send schedule command",
    "send keepalive message fail, store maybe disconnected",
)


def ts(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.read_text().strip()[:26], "%Y-%m-%dT%H:%M:%S.%f")


def write_gz_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:  # mtime=0: reproducible bytes
        f.write(json.dumps(obj, separators=(",", ":")).encode())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Region-map snapshots (full maps, gzipped) + index -----------------------------------
    names = sorted(p.name[: -len(".regions.json")] for p in RAW.glob("*.regions.json"))
    keep = []
    for n in names:
        if n.startswith(("00-", "01-", "02-")) or re.match(r"0[3-9]-.*-(pre|t1|t8|t20)$", n) or n.startswith("10-down-t") or n.startswith("12-late-"):
            keep.append(n)
    index = []
    for n in keep:
        d = json.loads((RAW / f"{n}.regions.json").read_text())
        write_gz_json(OUT / "regions" / f"{n}.json.gz", d)
        index.append({"name": n, "utc": (RAW / f"{n}.ts").read_text().strip(), "count": d["count"]})

    # ---- Store lists (small): Up / Disconnected / Down ---------------------------------------
    (OUT / "stores").mkdir(exist_ok=True)
    for n in ("00-baseline", "10-down-t15", "10-down-t70"):
        d = json.loads((RAW / f"{n}.stores.json").read_text())
        (OUT / "stores" / f"{n}.json").write_text(json.dumps(d, indent=1))

    # ---- Commands issued, with timestamps ---------------------------------------------------
    commands = []
    for p in sorted(RAW.glob("*.issue.ts")):
        label = p.name[: -len(".issue.ts")]
        out = RAW / f"{label}.cmd.out"
        commands.append({
            "label": label, "utc": p.read_text().strip(),
            "pd_ctl_output": out.read_text().strip() if out.exists() else None,
        })
    kill = RAW / "10-down.kill.ts"
    if kill.exists():
        commands.append({"label": "10-down", "utc": kill.read_text().strip(), "pd_ctl_output": "kill -9 of the tikv-server for store 6"})
    (OUT / "timeline.json").write_text(json.dumps({"snapshots": index, "commands": commands}, indent=1))

    # ---- PD log samples ---------------------------------------------------------------------
    pat = re.compile(r"^\[(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d\.\d+) \+00:00\] \[(\w+)\] \[([\w.]+:\d+)\] \[\"?([^\"\]]*)\"?\]")
    t_ops0 = ts(RAW / "03-transfer-leader.issue.ts") - dt.timedelta(seconds=1)
    t_ops1 = (ts(RAW / "09.end.ts") if (RAW / "09.end.ts").exists() else t_ops0 + dt.timedelta(minutes=15)) + dt.timedelta(seconds=90)  # +90 s: the 1-minute split-region timeout fires after the last command
    t_kill = ts(kill) if kill.exists() else None
    t_start = None
    sections: dict[str, list[str]] = {"startup split + first operators (first 40 lifecycle lines)": [],
                                      "manual operators on table regions": [],
                                      "store down window (kill .. +16 min), non-routine lines": []}
    startup = 0
    emitted: set[int] = set()  # raw line indices already written, so sections never repeat a line
    for idx, line in enumerate(PD_LOG.read_text().splitlines()):
        m = pat.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S.%f")
        msg = m.group(4)
        t_start = t_start or t
        rid = re.search(r"\[region-id=(\d+)\]", line)
        if msg in LIFECYCLE_MSGS and startup < 40 and t < t_start + dt.timedelta(minutes=1):
            sections["startup split + first operators (first 40 lifecycle lines)"].append(line)
            emitted.add(idx)
            startup += 1
        if t_ops0 <= t <= t_ops1 and rid and rid.group(1) in TABLE_REGIONS_OF_INTEREST and msg in LIFECYCLE_MSGS:
            if idx not in emitted:
                sections["manual operators on table regions"].append(line)
                emitted.add(idx)
        if t_kill and t_kill <= t <= t_kill + dt.timedelta(minutes=16) and idx not in emitted:
            if msg in LIFECYCLE_MSGS and msg not in ("send schedule command", "leader changed", "region ConfVer changed") or msg == "audit log":
                sections["store down window (kill .. +16 min), non-routine lines"].append(line)
            elif msg in ("leader changed", "region ConfVer changed", "send schedule command") and sum(len(x) for x in sections["store down window (kill .. +16 min), non-routine lines"]) < 60000:
                sections["store down window (kill .. +16 min), non-routine lines"].append(line)
    with open(OUT / "pd_log_samples.log", "w") as f:
        f.write("# Raw PD v8.5.8 log lines, unmodified. Sections are separated by '##' lines.\n")
        for title, lines in sections.items():
            f.write(f"## {title} ({len(lines)} lines)\n")
            f.writelines(l + "\n" for l in lines)

    # ---- Operator history + operator show ---------------------------------------------------
    for src, dst in (("operator_history.cli.json", "operator_history.cli.json"),
                     ("operator_history.rest.json", "operator_history.rest.json")):
        p = RAW / src
        if p.exists():
            (OUT / dst).write_text(p.read_text())
    shows = {}
    for p in sorted(RAW.glob("*.show0.json")) + sorted(RAW.glob("10-down-t*.show.json")) + sorted(RAW.glob("12-late-*.show.json")):
        txt = p.read_text().strip()
        if txt and txt != "[]":
            shows[p.name] = txt
    (OUT / "operator_show_nonempty.json").write_text(json.dumps(shows, indent=1))

    # ---- Metric names + sample lines --------------------------------------------------------
    def digest(path: Path, prefixes: tuple[str, ...], per_metric: int = 4) -> str:
        types, samples = {}, {}
        for line in path.read_text().splitlines():
            if line.startswith("# TYPE "):
                _, _, name, typ = line.split()
                types[name] = typ
            elif line and not line.startswith("#"):
                name = re.split(r"[{ ]", line, maxsplit=1)[0]
                base = re.sub(r"_(bucket|sum|count)$", "", name)
                if base in types or name in types:
                    samples.setdefault(base if base in types else name, []).append(line)
        out = []
        for name in sorted(types):
            if name.startswith(prefixes):
                out.append(f"# TYPE {name} {types[name]}")
                lines = samples.get(name, [])
                nonbucket = [l for l in lines if "_bucket{" not in l]
                out += nonbucket[:per_metric] if nonbucket else lines[:per_metric]
        return "\n".join(out) + "\n"

    pdm = RAW / "pd.metrics.final"
    if pdm.exists():
        (OUT / "pd_metrics_samples.txt").write_text(digest(pdm, (
            "pd_schedule_operators_count", "pd_schedule_finish_operators_duration_seconds",
            "pd_schedule_finish_operator_steps_duration_seconds", "pd_schedule_operator_region_size",
            "pd_cluster_status", "pd_regions_status", "pd_regions_abnormal_peer_duration_seconds",
            "pd_scheduler_store_status", "pd_cluster_region_event", "pd_scheduler_event_count")))
    tk = RAW / "tikv1.metrics.final"
    if tk.exists():
        (OUT / "tikv_metrics_samples.txt").write_text(digest(tk, (
            "tikv_raftstore_admin_cmd_total", "tikv_raftstore_region_count", "tikv_raftstore_snapshot_total",
            "tikv_raftstore_snapshot_duration_seconds", "tikv_raftstore_snapshot_traffic_total",
            "tikv_snapshot_size", "tikv_snapshot_limit_transport_bytes", "tikv_server_send_snapshot_duration_seconds",
            "tikv_server_snapshot_task_total", "tikv_raftstore_leader_missing", "tikv_raftstore_peer_pending_duration_seconds",
            "tikv_raftstore_raft_sent_message_total")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
