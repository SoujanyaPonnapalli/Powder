"""Outcome classification and the availability ratio (exec #7 step 2)."""

import pytest

from powder.deploy import outcomes as O
from powder.deploy.availability import normalize


@pytest.mark.parametrize("code, outcome, cause", [
    (9001, "timeout", "tidb_9001_pd_server_timeout"),
    (9002, "timeout", "tidb_9002_tikv_server_timeout"),
    (9003, "timeout", "tidb_9003_tikv_server_busy"),
    (9005, "timeout", "tidb_9005_region_unavailable"),
    (9010, "timeout", "tidb_9010_tikv_stale_command"),
    (1053, "timeout", "mysql_1053_server_shutdown"),
    (9007, "conflict", "tidb_9007_write_conflict"),
    (8005, "conflict", "tidb_8005_write_conflict_in_tidb"),
    (8022, "conflict", "tidb_8022_txn_retryable"),
    (1213, "conflict", "mysql_1213_deadlock"),
    (1062, "conflict", "mysql_1062_duplicate_key"),
    (1205, "conflict", "mysql_1205_lock_wait_timeout"),
    (9006, "other", "tidb_9006_gc_too_early"),
    (8004, "other", "tidb_8004_txn_too_large"),
])
def test_known_codes(code, outcome, cause):
    r = O.classify(code=code)
    assert (r.outcome, r.cause, r.code) == (outcome, cause, code)
    assert r.layer in {"client", "tidb", "tikv", "pd"}


def test_success_has_no_cause():
    r = O.classify(failed=False)
    assert r == O.OK_OUTCOME and r.cause is None and r.code is None


def test_client_deadline_wins_over_any_code():
    r = O.classify(code=9007, deadline_exceeded=True)       # the client gave up first
    assert (r.outcome, r.cause, r.layer, r.code) == ("timeout", "client_deadline", "client", 9007)


@pytest.mark.parametrize("text, cause", [
    ("dial tcp 10.0.0.1:4000: connect: connection refused", "conn_refused"),
    ("read tcp 10.0.0.2:5/4000: read: connection reset by peer", "conn_reset"),
    ("write: broken pipe", "conn_reset"),
    ("driver: bad connection", "conn_reset"),
    ("unexpected EOF", "conn_reset"),
    ("context deadline exceeded", "client_deadline"),
    ("read tcp: i/o timeout", "client_deadline"),
    ("dial tcp: lookup tidb.local: no such host", "conn_unreachable"),
])
def test_text_patterns_for_client_side_failures(text, cause):
    r = O.classify(text=text)
    assert (r.outcome, r.cause) == ("timeout", cause)


def test_known_code_beats_text():
    # a conflict message that happens to contain the word "timeout" is still a conflict
    assert O.classify(code=1205, text="Lock wait timeout exceeded; try restarting transaction").outcome == "conflict"


def test_unknown_error_is_other_but_keeps_its_code():
    assert O.classify(code=1146, text="Table doesn't exist") == O.Outcome("other", "unclassified_1146", 1146, None)
    assert O.classify(text="something odd").cause == "unclassified_no_code"


def test_table_is_internally_consistent():
    import re
    t = O.table()
    assert all(e["outcome"] in set(O.OUTCOMES) - {"ok"} for e in t["mysql_codes"].values())
    assert len({e["cause"] for e in t["mysql_codes"].values()}) == len(t["mysql_codes"])     # unique per code
    causes = [e["cause"] for e in t["mysql_codes"].values()] + [p["cause"] for p in t["error_text_patterns"]]
    assert all(re.fullmatch(r"[a-z0-9_]+", c) for c in causes)                                # stable slugs
    for code, e in t["mysql_codes"].items():                                                 # numeric causes name their code
        digits = re.search(r"(\d{4})", e["cause"])
        assert digits is None or digits.group(1) == code


def test_unavailability_excludes_conflicts_and_other():
    c = O.Counts(ok=90, timeout=10, conflict=50, other=7)
    assert c.unavailability() == pytest.approx(10 / 100)           # (ok + timeout) only
    d = c.to_dict()
    assert d["conflict_rate"] == pytest.approx(50 / 157) and d["other_rate"] == pytest.approx(7 / 157)
    assert d["total"] == 157


def test_no_served_requests_is_none_not_zero():
    assert O.Counts(conflict=5).unavailability() is None
    assert O.Counts().to_dict()["conflict_rate"] is None


def test_counts_reject_unknown_outcome():
    with pytest.raises(ValueError):
        O.Counts().add("weird")


def test_truncate_collapses_whitespace_and_limits_length():
    assert O.truncate(None) is None
    assert O.truncate("a  b\n c") == "a b c"
    assert len(O.truncate("x" * 500)) == O.ERR_TEXT_MAX


def test_normalize_classifies_raw_records_only_when_needed():
    assert normalize({"outcome": "ok", "ts": 1})["outcome"] == "ok"
    n = normalize({"ts": 1, "code": 9005})
    assert (n["outcome"], n["cause"]) == ("timeout", "tidb_9005_region_unavailable")
    assert normalize({"ts": 1, "failed": False})["outcome"] == "ok"
    assert normalize({"ts": 1, "deadline": True})["cause"] == "client_deadline"
