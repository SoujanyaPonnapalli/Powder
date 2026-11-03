"""Parsers against REAL TiDB v8.5.8 output captured from a 4-store cluster (tests/fixtures/tidb/README.md).

Every expectation below is ground truth from that capture (the commands issued are in timeline.json), not from
hand-built data. Timings are loopback-fast; only formats and event semantics matter here.
"""

import datetime
import gzip
import json
from pathlib import Path

import pytest

from powder.deploy import pdlog
from powder.deploy import regionmap as RM
from powder.deploy import repartition as R
from powder.deploy.regionmap import RegionMapSeries

FIX = Path(__file__).parent / "fixtures" / "tidb"
TIMELINE = json.loads((FIX / "timeline.json").read_text())
SNAPS = {s["name"]: s for s in TIMELINE["snapshots"]}
KILL_TS = datetime.datetime.fromisoformat("2026-11-03T00:59:28.06+00:00").timestamp()      # kill -9 of store 6


def epoch(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def stores(name):
    return json.loads((FIX / "stores" / f"{name}.json").read_text())


def load(name, store_doc="00-baseline"):
    doc = json.load(gzip.open(FIX / "regions" / f"{name}.json.gz"))
    return RM.from_pd_json(doc, epoch(SNAPS[name]["utc"]), stores(store_doc))


def diff(a, b):
    return R.diff_maps(load(a), load(b))


@pytest.fixture(scope="module")
def log():
    return pdlog.parse((FIX / "pd_log_samples.log").open())


# ---------------------------------------------------------------- Region maps
@pytest.mark.parametrize("name", sorted(SNAPS))
def test_every_snapshot_tiles_the_key_space_after_sorting(name):
    m = load(name)
    assert m.coverage_problems() == []
    assert len(m.regions) == SNAPS[name]["count"]


def test_row_keys_use_pds_memcomparable_encoding():
    m = load("02-split-t31")
    r316 = next(r for r in m.regions if r.id == 316)
    assert r316.start == RM.row_key(114, 12500)                       # the capture's documented fact
    assert m.find(RM.row_key(114, 12600)).id == 316
    assert m.find(RM.row_key(114, 0)).id == 312 and m.find(RM.row_key(114, 99999)).id == 27   # parent = rightmost piece
    assert RM.mc_encode(b"") == b"\x00" * 8 + b"\xf7"                 # empty key: one all-padding group
    assert RM.mc_encode(b"12345678") == b"12345678\xff" + b"\x00" * 8 + b"\xf7"


def test_stores_document_and_voters():
    st = RM.parse_stores(stores("10-down-t70"))
    assert {i: s.state for i, s in st.items()} == {1: "Up", 4: "Up", 5: "Up", 6: "Down"}
    r = load("03-transfer-leader-pre").regions[0]
    assert all(isinstance(v, int) for v in r.voters) and len(r.voters) == 3 and r.leader in r.voters


# ---------------------------------------------------------------- diffs of real events
def kinds(ev):
    return [(e.kind, e.region_id) for e in ev]


def test_table_split_is_seven_splits_and_ids_do_not_matter():
    ev = diff("01-after-load", "02-split-t0")
    splits = [e for e in ev if e.kind == "split"]
    assert len(splits) == 7 and load("02-split-t0").boundaries() - load("01-after-load").boundaries() == {e.key for e in splits}
    assert len(load("02-split-t0").regions) - len(load("01-after-load").regions) == 7
    assert not any(e.kind == "merge" for e in ev)


def test_leader_transfer_316_from_store_5_to_6():
    ev = diff("03-transfer-leader-pre", "03-transfer-leader-t1")
    assert [(e.kind, e.region_id, e.from_store, e.to_store) for e in ev] == [("transfer_leader", 316, 5, 6)]


def test_move_peer_328_from_store_4_to_6():
    ev = diff("06-transfer-peer-pre", "06-transfer-peer-t1")
    assert [(e.kind, e.region_id, e.from_store, e.to_store) for e in ev] == [("move_peer", 328, 4, 6)]
    assert ev[0].rsm_before == (1, 4, 5) or 4 in ev[0].rsm_before


def test_merge_deletes_the_source_and_target_keeps_its_id_and_takes_its_own_store_set():
    pre, post = load("07-merge-pre"), load("07-merge-t8")
    ev = R.diff_maps(pre, post)
    assert [e.kind for e in ev] == ["merge"] and len(pre.regions) - len(post.regions) == 1
    assert 332 in {r.id for r in pre.regions} and 332 not in {r.id for r in post.regions}
    target = next(r for r in post.regions if r.id == 336)
    assert target.voters == (1, 4, 6)                                   # the target's store set, not the source's (4,5,6)
    assert target.version > next(r for r in pre.regions if r.id == 336).version


@pytest.mark.parametrize("pre, post, region", [("04-remove-peer-pre", "04-remove-peer-t1", 320),
                                               ("05-add-peer-pre", "05-add-peer-t1", 324)])
def test_membership_changes_shorter_than_the_snapshot_interval_show_up_as_peer_churn(pre, post, region):
    # the peer set is unchanged after the operator (it finished in < 1 s) but epoch.conf_ver moved by 3
    ev = diff(pre, post)
    assert [(e.kind, e.region_id, e.conf_delta) for e in ev] == [("peer_churn", region, 3)]


def test_unchanged_pair_has_no_events():
    assert R.diff_maps(load("03-transfer-leader-pre"), load("03-transfer-leader-pre")) == []


# ---------------------------------------------------------------- store failure
def down_series():
    names = sorted([n for n in SNAPS if n.startswith(("10-down", "12-late"))], key=lambda n: SNAPS[n]["utc"])
    maps = []
    for n in names:
        t = epoch(SNAPS[n]["utc"])
        doc = "00-baseline" if t < KILL_TS else "10-down-t15" if t < KILL_TS + 40 else "10-down-t70"   # nearest captured store state
        maps.append(load(n, doc))
    return RegionMapSeries(maps), names


def test_dead_store_marks_its_regions_under_replicated_until_they_are_re_replicated():
    series, names = down_series()
    first = series.maps[0]
    assert sum(r.under_replicated(3) for r in first.regions) == 63            # store 6 held 63 Regions
    assert all(6 in r.down for r in first.regions if r.under_replicated(3))
    assert sum(bool(r.pending) for r in series.maps[2].regions) >= 6          # pending_peers appears, `down_peers` never does
    last = series.maps[-1]
    assert sum(r.under_replicated(3) for r in last.regions) == 0
    assert all(6 not in r.voters for r in last.regions)                        # every peer moved off the dead store

    u = R.under_replication(series, 3, window_end=last.ts)
    assert u["unresolved"] == 0 and len(u["restore_times_s"]) == 63
    # PD waited ~9 min (TiKV max-peer-down-duration 10m) before replacing peers: restore takes 9-11 min
    assert 530 < min(u["restore_times_s"]) and max(u["restore_times_s"]) < 700
    assert u["any_share"] > 0.7


def test_move_events_account_for_the_rereplication():
    series, _ = down_series()
    s = R.summarize(series, 3, (series.maps[0].ts, series.maps[-1].ts))
    assert s["counts"]["move_peer"] >= 55                                     # 63 peers moved; some changes fall in one window
    assert s["under_replication"]["any_seconds"] > 600


# ---------------------------------------------------------------- PD log
def test_log_parses_completely(log):
    assert log.unparsed == []                                                  # nothing lifecycle-like was skipped
    assert log.duplicates >= 1                                                 # PD logs some lines twice
    assert len(log.operators) == 88
    outcomes = {}
    for o in log.operators:
        outcomes[o.outcome] = outcomes.get(o.outcome, 0) + 1
    assert outcomes == {"finished": 79, "timeout": 4, "canceled": 4, "removed": 1}


def test_transfer_leader_operator_316(log):
    op = next(o for o in log.operators if o.region_id == 316 and o.name == "admin-transfer-leader")
    assert (op.kind, op.from_store, op.to_store, op.outcome) == ("transfer_leader", 5, 6, "finished")
    assert op.duration == pytest.approx(0.003746801, abs=1e-4)
    assert op.trigger == "admin-transfer-leader"
    assert [(c.from_store, c.to_store) for c in log.leader_changes if c.region_id == 316][0] == (5, 6)
    assert sum(1 for c in log.leader_changes if c.region_id == 316 and (c.from_store, c.to_store) == (5, 6)) == 1  # deduped


def test_merge_logs_two_operators_only_the_target_finishes(log):
    merges = [o for o in log.operators if o.kind == "merge"]
    assert {(o.region_id, o.outcome) for o in merges} == {(332, "canceled"), (336, "finished")}
    assert next(o for o in merges if o.region_id == 332).cancel_reason == "related merge region"


def test_timeouts_and_cancel_reasons(log):
    split = next(o for o in log.operators if o.name == "admin-split-region")
    assert (split.region_id, split.outcome, split.cancel_reason) == (27, "timeout", "timeout")
    assert split.duration == pytest.approx(60.5, abs=0.6)
    reasons = {o.cancel_reason for o in log.operators if o.outcome in ("canceled", "timeout")}
    assert {"timeout", "already exist", "related merge region"} <= reasons


def test_store_down_repair_operators(log):
    repairs = [o for o in log.operators if o.name == "replace-rule-down-peer"]
    assert len(repairs) == 63 and all(o.outcome == "finished" and o.kind == "move_peer" for o in repairs)
    assert all(o.from_store == 6 for o in repairs) and {o.to_store for o in repairs} <= {1, 4, 5}
    assert max(o.duration for o in repairs) < 0.5                              # each took ~50-125 ms
    assert sum(1 for o in log.operators if o.kind == "move_peer") == 70        # + 6 balance-region + 1 admin-move-peer


def test_batch_split_lines_are_parsed(log):
    # the sample keeps the startup splits (the 00:49 table split is not in the sample, so no map-diff comparison here)
    assert (log.splits[0].region_id, log.splits[0].total) == (2, 4)           # labeler-split of Region 2 -> 4 new Regions
    assert sum(s.total for s in log.splits) == 4 + 17                          # then 17 single splits of the bootstrap Regions
    assert all(s.total >= 1 for s in log.splits)


def test_conf_changes_are_parsed(log):
    c = [x for x in log.conf_changes if x.region_id == 320]
    assert c[0].changes == [("Remove", 389, 1, "")] and (c[0].old_confver, c[0].new_confver) == (29, 30)
    assert c[1].changes[0][:3] == ("Add", 403, 1) and c[1].changes[0][3] == "Learner"


def test_window_filter(log):
    t = epoch(SNAPS["03-transfer-leader-pre"]["utc"])
    sub = pdlog.parse((FIX / "pd_log_samples.log").open(), t0=t, t1=t + 5)
    assert [o.name for o in sub.operators if o.outcome == "finished"] == ["admin-transfer-leader"]


def test_operator_summary_and_cross_check_on_a_real_window(log):
    a, b = load("03-transfer-leader-pre"), load("03-transfer-leader-t20")
    ops = [o for o in log.operators if a.ts <= o.t_create <= b.ts]
    s = R.summarize(RegionMapSeries([a, load("03-transfer-leader-t1"), load("03-transfer-leader-t8"), b]), 3,
                    (a.ts, b.ts), operators=ops)
    assert s["cross_check"]["transfer_leader"] == {"region_maps": 1, "operators": 1}
    assert s["operators"]["by_kind_finished"]["transfer_leader"] == 1


def test_timestamp_and_duration_parsers():
    assert pdlog.parse_go_duration("1m0.532946769s") == pytest.approx(60.532946769)
    assert pdlog.parse_go_duration("3.746801ms") == pytest.approx(0.003746801) and pdlog.parse_go_duration("0s") == 0
    with pytest.raises(ValueError):
        pdlog.parse_go_duration("soon")
    assert pdlog.parse_go_time("0001-01-01 00:00:00 +0000 UTC") is None
    assert pdlog.parse_go_time("2026-11-03 00:56:01.956472306 +0000 UTC m=+549.37") == pytest.approx(1793667361.956472)


def test_unrecognised_lifecycle_lines_are_counted_not_dropped():
    bad = ('[2026/11/03 00:00:00.000 +00:00] [INFO] [operator_controller.go:1] ["operator exploded"] [region-id=1] '
           '[operator="\\"x\\""]')
    out = pdlog.parse([bad])
    assert len(out.unparsed) == 1 and out.operators == []
