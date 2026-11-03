"""Parse PD's log into operator lifecycles and repartitioning bookkeeping (plan §7, measurement 3).

PD's ``operator history`` API is **lossy** (10-minute ring, one record per Region, so a newer operator overwrites
the older), so the PD log is the lossless source. The grammar below was captured from a real v8.5.8 cluster
(``tests/fixtures/tidb/``):

    [2026/11/03 00:56:01.960 +00:00] [INFO] [operator_controller.go:724] ["operator finish"] [region-id=316]
        [takes=3.746801ms] [operator="\"admin-transfer-leader {transfer leader: store 5 to 6} (kind:admin,leader,
        region:316(73, 17), createAt:2026-11-03 00:56:01.956472306 +0000 UTC m=+549.37, startAt:..., currentStep:1,
        size:1, steps:[0:{...}], timeout:[1m0s]) finished\""] [additional-info="{\"cancel-reason\":\"\"}"]

There is no separate "operator start" line: ``add operator`` carries create and start. Terminal lines are
``operator finish`` / ``operator timeout`` / ``operator canceled`` / ``operator removed``. Some lines are logged twice
with identical text (deduplicated here). Lines that look like lifecycle events but do not parse are *counted*
(``PdLog.unparsed``), never dropped silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from .repartition import OperatorEvent

_HEADER = re.compile(r'^\[(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d\.\d+) ([+-]\d\d:\d\d)\] \[(\w+)\] \[([^\]]+)\] \["((?:[^"\\]|\\.)*)"\]')
_FIELD = re.compile(r'\[([A-Za-z][\w-]*)=("(?:[^"\\]|\\.)*"|[^\]]*)\]')
_OPERATOR = re.compile(
    r"^(?P<name>\S+) \{(?P<brief>.*?)\} \(kind:(?P<kind>.*?), region:(?P<rid>\d+)\((?P<ver>\d+), (?P<cv>\d+)\), "
    r"createAt:(?P<create>.*?), startAt:(?P<start>.*?), currentStep:(?P<cur>\d+), size:(?P<size>\d+), "
    r"steps:\[(?P<steps>.*)\], timeout:\[(?P<timeout>[^\]]*)\]\)(?P<suffix>.*)$")
_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))? ([+-]\d{4})")
_DUR = re.compile(r"(\d+(?:\.\d+)?)(ns|µs|us|ms|s|m|h)")
_UNIT = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}
_PEER = re.compile(r"(Add|Remove) peer:\{id:(\d+) store_id:(\d+)(?: role:(\w+))? ?\}")

TERMINAL = {"operator finish": "finished", "operator timeout": "timeout", "operator canceled": "canceled",
            "operator removed": "removed"}
LIFECYCLE_PREFIX = "operator "


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return json.loads(v)
    return v


def parse_header_ts(date: str, tz: str) -> float:
    base, _, frac = date.partition(".")
    dt = datetime.strptime(base, "%Y/%m/%d %H:%M:%S")
    sign = 1 if tz[0] == "+" else -1
    off = sign * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
    return dt.replace(tzinfo=timezone.utc).timestamp() - off + float("0." + (frac or "0"))


def parse_go_time(text: str) -> float | None:
    """``2026-11-03 00:47:22.88431359 +0000 UTC m=+30.3`` -> epoch; the zero time ``0001-01-01`` -> None."""
    m = _TS.match(text.strip())
    if not m or m.group(1).startswith("0001-"):
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    sign = 1 if m.group(3)[0] == "+" else -1
    off = sign * (int(m.group(3)[1:3]) * 3600 + int(m.group(3)[3:5]) * 60)
    return dt.timestamp() - off + float("0." + (m.group(2) or "0"))


def parse_go_duration(text: str) -> float:
    """``1m0.532946769s`` / ``3.746801ms`` / ``0s`` -> seconds."""
    parts = _DUR.findall(text)
    if not parts or "".join(a + u for a, u in parts) != text.strip():
        raise ValueError(f"bad Go duration {text!r}")
    return sum(float(a) * _UNIT[u] for a, u in parts)


def _stores(brief: str) -> tuple[int | None, int | None]:
    """from/to store ids out of an operator brief such as ``mv peer: store [4] to [6]``."""
    nums = [int(x) for x in re.findall(r"\d+", brief.split(":", 1)[-1])] if ":" in brief else []
    if brief.startswith("transfer leader") or brief.startswith("mv peer"):
        return (nums[0], nums[1]) if len(nums) >= 2 else (None, None)
    if brief.startswith("add peer") or brief.startswith("add learner"):
        return (None, nums[0]) if nums else (None, None)
    if brief.startswith("rm peer"):
        return (nums[0], None) if nums else (None, None)
    return None, None


def kind_of(name: str, brief: str) -> str:
    """Map a PD operator to one of repartition.KINDS where possible (by brief), else a stable raw label."""
    for prefix, kind in (("transfer leader", "transfer_leader"), ("mv peer", "move_peer"), ("add peer", "add_peer"),
                         ("add learner", "add_peer"), ("rm peer", "remove_peer"), ("merge", "merge"),
                         ("split", "split")):
        if brief.startswith(prefix):
            return kind
    return name.replace("-", "_")


@dataclass
class SplitLog:
    ts: float
    region_id: int
    total: int          # number of new Regions generated = boundaries added


@dataclass
class LeaderChange:
    ts: float
    region_id: int
    from_store: int
    to_store: int


@dataclass
class ConfChange:
    ts: float
    region_id: int
    changes: list[tuple[str, int, int, str]]    # (Add|Remove, peer id, store id, role or "")
    old_confver: int
    new_confver: int


@dataclass
class PdLog:
    operators: list[OperatorEvent] = field(default_factory=list)
    splits: list[SplitLog] = field(default_factory=list)
    leader_changes: list[LeaderChange] = field(default_factory=list)
    conf_changes: list[ConfChange] = field(default_factory=list)
    lines: int = 0
    duplicates: int = 0
    unparsed: list[str] = field(default_factory=list)       # lifecycle-looking lines that did not parse (first 200 chars)


def _fields(line: str, header_end: int) -> dict[str, str]:
    return {k: _unquote(v) for k, v in _FIELD.findall(line[header_end:])}


def _operator(text: str):
    inner = text.strip()
    if len(inner) >= 2 and inner[0] == inner[-1] == '"':          # the operator value is quoted twice
        inner = inner[1:-1]
    return _OPERATOR.match(inner)


def parse(lines: Iterable[str], t0: float | None = None, t1: float | None = None) -> PdLog:
    out = PdLog()
    seen: set[str] = set()
    live: dict[tuple, OperatorEvent] = {}                       # (region, name, createAt text) -> started, not finished
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.startswith("[20"):
            continue
        out.lines += 1
        if line in seen:
            out.duplicates += 1
            continue
        seen.add(line)
        h = _HEADER.match(line)
        if not h:
            continue
        ts = parse_header_ts(h.group(1), h.group(2))
        if (t0 is not None and ts < t0) or (t1 is not None and ts > t1):
            continue
        msg = h.group(5)
        f = _fields(line, h.end())

        try:
            if msg == "add operator":
                m = _operator(f["operator"])
                if not m:
                    raise ValueError("operator string")
                key = (int(f["region-id"]), m["name"], m["create"])
                live[key] = _event(m, f, ts, outcome="unknown", t_finish=None)
            elif msg in TERMINAL:
                m = _operator(f["operator"])
                if not m:
                    raise ValueError("operator string")
                key = (int(f["region-id"]), m["name"], m["create"])
                live.pop(key, None)
                out.operators.append(_event(m, f, ts, outcome=TERMINAL[msg], t_finish=ts))
            elif msg == "region batch split, generate new regions":
                out.splits.append(SplitLog(ts, int(f["region-id"]), int(f["total"])))
            elif msg == "leader changed":
                out.leader_changes.append(LeaderChange(ts, int(f["region-id"]), int(f["from"]), int(f["to"])))
            elif msg == "region ConfVer changed":
                ch = [(a, int(pid), int(sid), role or "") for a, pid, sid, role in _PEER.findall(f.get("detail", ""))]
                out.conf_changes.append(ConfChange(ts, int(f["region-id"]), ch, int(f["old-confver"]), int(f["new-confver"])))
            elif msg.startswith(LIFECYCLE_PREFIX) and msg not in ("operator canceled related merge region", "operator is stale"):
                raise ValueError(f"unrecognised lifecycle message {msg!r}")
        except (KeyError, ValueError) as e:
            out.unparsed.append(f"{e}: {line[:200]}")
    out.operators.extend(live.values())                          # still running when the log ended
    out.operators.sort(key=lambda o: (o.t_create, o.region_id or 0))
    return out


def _event(m, f: dict, log_ts: float, outcome: str, t_finish: float | None) -> OperatorEvent:
    brief = m["brief"]
    frm, to = _stores(brief)
    create = parse_go_time(m["create"]) or log_ts
    start = parse_go_time(m["start"])
    takes = parse_go_duration(f["takes"]) if "takes" in f else None
    try:
        info = json.loads(f.get("additional-info") or "{}")
    except json.JSONDecodeError:
        info = {}
    return OperatorEvent(
        kind=kind_of(m["name"], brief), region_id=int(m["rid"]), t_create=create, t_start=start,
        t_finish=(create + takes) if (takes is not None and t_finish is not None and start is not None) else t_finish,
        outcome=outcome, trigger=m["name"], from_store=frm, to_store=to, desc=f"{m['name']} {{{brief}}}",
        size_mb=float(m["size"]), cancel_reason=str(info.get("cancel-reason", "")), name=m["name"])
