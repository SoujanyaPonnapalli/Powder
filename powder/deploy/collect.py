"""Turn an experiment directory into ``summary.json`` (plan §7).

Layout produced by ``experiment.run_experiment`` under ``outputs/tidb/<handle>/experiments/<name>-<utc>/``::

    config.json  prepare.json
    workload/<client-region>/{requests,probes}.jsonl.gz   load.out  probe.out
    pd/snapshots/{regions,stores}-<epoch>.json.gz         pd/{store,config,placement-rules}.json
    logs/<node>/*.log                                     (PD logs feed the operator lifecycle parser)
    metrics/<name>.json                                   (Prometheus query_range exports)
    summary.json

Collection problems (a missing Prometheus, an unreadable log) never abort a finished experiment: they are listed
in ``summary.json["collection_errors"]`` so a human can see exactly what is absent.
"""

from __future__ import annotations

import glob
import gzip
import json
import re
import zlib
from pathlib import Path
from typing import Iterator

from . import availability as A
from . import pdlog, repartition as R
from .regionmap import RegionMapSeries, from_pd_json
from .spec import ExperimentConfig

#: Prometheus queries exported to ``metrics/`` (names verified against v8.5.8, tests/fixtures/tidb/*_metrics_samples.txt;
#: the two ``tidb_*`` entries are standard TiDB metrics not covered by the fixtures).
METRICS: dict[str, str] = {
    "pd_operators_count": "sum by (event, type) (pd_schedule_operators_count)",
    "pd_operator_finish_duration_sum": "sum by (type) (pd_schedule_finish_operators_duration_seconds_sum)",
    "pd_operator_finish_duration_count": "sum by (type) (pd_schedule_finish_operators_duration_seconds_count)",
    "pd_operator_region_size_sum": "sum by (type) (pd_schedule_operator_region_size_sum)",
    "pd_cluster_status": "pd_cluster_status",
    "pd_regions_status": "pd_regions_status",
    "tikv_admin_cmd_success": 'sum by (type) (tikv_raftstore_admin_cmd_total{status="success"})',
    "tikv_region_count": "tikv_raftstore_region_count",
    "tikv_snapshot_bytes_sent": "sum by (instance) (tikv_snapshot_size_sum)",
    "tikv_snapshot_transport_bytes": "sum by (instance, type) (tikv_snapshot_limit_transport_bytes)",
    "tidb_query_total": "sum by (instance) (tidb_server_query_total)",
    "tidb_query_duration_p99": "histogram_quantile(0.99, sum by (le) (rate(tidb_server_handle_query_duration_seconds_bucket[1m])))",
    # G3 follow-up: is the healthy-Region stall connection churn? (server side of powder-load's pool stats)
    "tidb_connections": "sum by (instance) (tidb_server_connections)",
    "tidb_handshake_errors": "sum by (instance) (tidb_server_handshake_error_total)",
    "tidb_tokens": "sum by (instance) (tidb_server_tokens)",
    "tidb_kv_backoff_count": "sum by (type) (tidb_tikvclient_backoff_seconds_count)",
}


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """Records of a (possibly truncated) gzip or plain JSONL file; a torn tail is ignored, not fatal."""
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        return                                  # partial last line of a killed writer
    except (EOFError, zlib.error, gzip.BadGzipFile):
        return                                                  # truncated gzip stream: keep what was read


def _epoch_of(path: str) -> float:
    return float(re.search(r"-(\d+(?:\.\d+)?)\.json", path).group(1))


def load_snapshots(snap_dir: Path) -> tuple[RegionMapSeries | None, list[str]]:
    """Pair each ``regions-<ts>`` snapshot with the nearest earlier ``stores-<ts>`` one."""
    errors: list[str] = []
    stores = sorted((_epoch_of(p), p) for p in glob.glob(str(snap_dir / "stores-*.json*")))
    maps = []
    for p in sorted(glob.glob(str(snap_dir / "regions-*.json*")), key=_epoch_of):
        ts = _epoch_of(p)
        try:
            doc = json.load((gzip.open if p.endswith(".gz") else open)(p, "rt"))
            sd = None
            earlier = [sp for st, sp in stores if st <= ts + 5]
            if earlier:
                sd = json.load((gzip.open if earlier[-1].endswith(".gz") else open)(earlier[-1], "rt"))
            maps.append(from_pd_json(doc, ts, sd))
        except (OSError, ValueError, KeyError, EOFError, zlib.error) as e:
            errors.append(f"snapshot {Path(p).name}: {e}")
    return (RegionMapSeries(maps) if maps else None), errors


def _window(exp: Path, errors: list[str]) -> tuple[float, float] | None:
    starts, ends = [], []
    for out in glob.glob(str(exp / "workload" / "*" / "load.out")):
        try:
            d = json.loads(Path(out).read_text().strip().splitlines()[-1])
            starts.append(d["start"]), ends.append(d["end"])
        except (OSError, ValueError, KeyError, IndexError) as e:
            errors.append(f"{out}: {e}")
    return (min(starts), max(ends)) if starts else None


def summarize_experiment(exp: Path, cfg: ExperimentConfig, table_id: int, replication_factor: int = 3,
                         inter_region_usd_per_gb: float | None = None) -> dict:
    errors: list[str] = []
    series, serr = load_snapshots(exp / "pd" / "snapshots")
    errors += serr
    if series is None:
        raise ValueError(f"{exp}: no Region-map snapshots; cannot attribute requests to RSMs")

    # PD logs -> operator lifecycles (lossless, unlike `operator history`)
    log = pdlog.PdLog()
    for p in sorted(glob.glob(str(exp / "logs" / "**" / "pd*.log*"), recursive=True)):
        try:
            opener = gzip.open if p.endswith(".gz") else open
            part = pdlog.parse(opener(p, "rt", errors="replace"))
            log.operators += part.operators
            log.splits += part.splits
            log.leader_changes += part.leader_changes
            log.conf_changes += part.conf_changes
            log.lines += part.lines
            log.duplicates += part.duplicates
            log.unparsed += part.unparsed
        except OSError as e:
            errors.append(f"{p}: {e}")
    have_log = log.lines > 0

    window = _window(exp, errors)
    workload_files = sorted(glob.glob(str(exp / "workload" / "*" / "requests.jsonl*")))
    probe_files = sorted(glob.glob(str(exp / "workload" / "*" / "probes.jsonl*")))
    if window is None:
        ts = [r["ts"] for f in workload_files + probe_files for r in read_jsonl(f)]
        if not ts:
            raise ValueError(f"{exp}: no workload or probe records found")
        window = (min(ts), max(ts))
        errors.append("window taken from record timestamps (no load.out)")
    t0, t1 = window

    ops = [o for o in log.operators if t0 - 1 <= o.t_create <= t1 + 1] if have_log else None
    index = R.RepartitionIndex(events=R.events_from_series(series), operators=ops,
                               region_for_key=R.region_for_key_fn(series))
    wl = A.WorkloadAggregator(series, table_id, cfg.workload.keys, cfg.workload.regions_target, repartition_index=index)
    for f in workload_files:
        wl.consume(read_jsonl(f))
    pr = A.ProbeAggregator(series, table_id, cfg.workload.keys, cfg.workload.regions_target, cfg.prober_interval_s, window,
                           max_lag_s=cfg.request_timeout_s + 10.0)
    if probe_files:
        pr.consume(A.merge_sorted(*[read_jsonl(f) for f in probe_files]))

    log_splits = sum(s.total for s in log.splits if t0 <= s.ts <= t1) if have_log else None
    rep = R.summarize(series, replication_factor, window, operators=ops, log_splits=log_splits,
                      inter_region_usd_per_gb=inter_region_usd_per_gb)
    if not have_log:
        errors.append("no PD logs collected: operator lifecycle metrics come from Region-map snapshots only")
    if log.unparsed:
        errors.append(f"{len(log.unparsed)} PD log lines looked like lifecycle events but did not parse")

    summary = {
        "experiment": cfg.to_dict(),
        "window": {"start": t0, "end": t1, "seconds": t1 - t0},
        "workload": wl.result() if workload_files else None,
        "probe": pr.result() if probe_files else None,
        "repartition": rep,
        "pd_log": {"lines": log.lines, "duplicates_removed": log.duplicates, "unparsed": len(log.unparsed),
                   "operators": len(log.operators), "leader_changes": len(log.leader_changes)},
        "snapshots": len(series),
        "collection_errors": errors,
    }
    return summary


def write_summary(exp: Path, summary: dict) -> Path:
    p = exp / "summary.json"
    p.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    return p
