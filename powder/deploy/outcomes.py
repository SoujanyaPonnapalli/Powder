"""Classify a request result into ``ok | timeout | conflict | other`` with a recorded cause (plan §7).

The code table lives in ``error_classes.json`` and is shared with the Go load generator (``powder-load``,
step 5), so the two implementations cannot drift. Every non-``ok`` outcome carries a ``cause`` slug, the
numeric ``code`` when there is one, and the ``layer`` it surfaced at, whatever its class, so conflicts and
unclassified errors are visible too.

``workload_unavailability = timeout / (ok + timeout)``; conflicts and ``other`` never enter that ratio.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

OK, TIMEOUT, CONFLICT, OTHER = "ok", "timeout", "conflict", "other"
OUTCOMES = (OK, TIMEOUT, CONFLICT, OTHER)
ERR_TEXT_MAX = 200
TABLE_PATH = Path(__file__).with_name("error_classes.json")

_TABLE: dict | None = None
_PATTERNS: list[tuple[re.Pattern, dict]] | None = None


def table() -> dict:
    global _TABLE, _PATTERNS
    if _TABLE is None:
        _TABLE = json.loads(TABLE_PATH.read_text())
        _PATTERNS = [(re.compile(p["regex"], re.I), p) for p in _TABLE["error_text_patterns"]]
    return _TABLE


@dataclass(frozen=True)
class Outcome:
    outcome: str
    cause: str | None = None
    code: int | None = None
    layer: str | None = None


OK_OUTCOME = Outcome(OK)


def truncate(text: str | None, limit: int = ERR_TEXT_MAX) -> str | None:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def classify(code: int | None = None, text: str | None = None, deadline_exceeded: bool = False,
             failed: bool = True) -> Outcome:
    """Classify one finished request.

    ``failed=False`` means it succeeded. Otherwise precedence is: the client's own deadline, then a known
    MySQL/TiDB error ``code``, then recognised connection/timeout wording in ``text``, then ``other`` with a
    cause that still carries the code (``unclassified_<code>``) so nothing is anonymous.
    """
    t = table()
    if not failed:
        return OK_OUTCOME
    if deadline_exceeded:
        d = t["client_deadline"]
        return Outcome(d["outcome"], d["cause"], code, d["layer"])
    if code is not None:
        e = t["mysql_codes"].get(str(code))
        if e:
            return Outcome(e["outcome"], e["cause"], code, e["layer"])
    if text:
        for rx, p in _PATTERNS:
            if rx.search(text):
                return Outcome(p["outcome"], p["cause"], code, p["layer"])
    suffix = f"{code}" if code is not None else "no_code"
    return Outcome(OTHER, f"{t['unclassified_cause_prefix']}_{suffix}", code, None)


@dataclass
class Counts:
    ok: int = 0
    timeout: int = 0
    conflict: int = 0
    other: int = 0

    def add(self, outcome: str, n: int = 1) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        setattr(self, outcome, getattr(self, outcome) + n)

    @property
    def total(self) -> int:
        return self.ok + self.timeout + self.conflict + self.other

    @property
    def served(self) -> int:
        """Requests that count toward availability: those that succeeded or timed out."""
        return self.ok + self.timeout

    def unavailability(self) -> float | None:
        """timeout / (ok + timeout); None when no request counted (never 0 by default)."""
        return self.timeout / self.served if self.served else None

    def to_dict(self) -> dict:
        u = self.unavailability()
        tot = self.total
        return {"ok": self.ok, "timeout": self.timeout, "conflict": self.conflict, "other": self.other,
                "total": tot, "unavailability": u,
                "conflict_rate": self.conflict / tot if tot else None,
                "other_rate": self.other / tot if tot else None}
