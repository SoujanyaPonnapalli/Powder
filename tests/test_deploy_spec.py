"""Spec, experiment config, handle names and node expansion (exec #7 step 2)."""

import random
from pathlib import Path

import pytest

from powder.deploy import names
from powder.deploy import spec as S

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "infra" / "tidb" / "specs" / "smoke.yaml"


def smoke_dict():
    return S.load_yaml(SMOKE)


# ---------------------------------------------------------------- handles
def test_generate_avoids_taken_case_insensitively():
    rng = random.Random(0)
    first = names.generate(rng=random.Random(0))
    again = names.generate(taken=[first.upper()], rng=random.Random(0))
    assert again.lower() != first.lower()
    assert again[0].isupper() and again.isalpha()


def test_generate_is_two_capitalised_words():
    for seed in range(20):
        h = names.generate(rng=random.Random(seed))
        parts = [w for w in __import__("re").findall(r"[A-Z][a-z]+", h)]
        assert len(parts) == 2 and "".join(parts) == h
        assert parts[0].lower() in names.ADJECTIVES and parts[1].lower() in names.NOUNS


def test_explicit_handle_validation():
    assert names.validate("MyRun-2") == "MyRun-2"
    for bad in ("", "ab", "1abc", "has space", "x" * 40, "auto", "AUTO"):
        with pytest.raises(names.HandleError):
            names.validate(bad)
    with pytest.raises(names.HandleError):
        names.ensure_new("OrangePanda", ["orangepanda"])


# ---------------------------------------------------------------- cluster spec
def test_smoke_spec_parses_and_defaults_to_pinned_version():
    sp = S.parse_spec(SMOKE)
    assert sp.tidb_version == S.pinned_tidb_version() == "v8.5.8"
    assert [s.name for s in sp.slots] == [f"tikv-{i}" for i in range(1, 7)]
    assert sp.regions == ["us-east-1", "us-east-2", "us-west-2"]
    assert S.spec_warnings(sp) == []


def test_hash_ignores_handle_and_label_but_not_content():
    d = smoke_dict()
    base = S.parse_spec(d).spec_hash()
    d2 = {**d, "handle": "OrangePanda", "label": "other"}
    assert S.parse_spec(d2).spec_hash() == base
    d3 = smoke_dict()
    d3["slots"]["tikv-1"] = {"machine": "machine/ford10-google-storage-ge15min@us-east-1"}
    assert S.parse_spec(d3).spec_hash() != base
    d4 = {**smoke_dict(), "seed": 2}
    assert S.parse_spec(d4).spec_hash() != base


def test_unknown_keys_rejected_at_every_level():
    d = smoke_dict()
    d["tikv_instnce_type"] = "x"
    d["control_plane"]["pdd"] = 3
    d["slots"]["tikv-1"]["azz"] = "x"
    with pytest.raises(S.SpecError) as e:
        S.parse_spec(d)
    msg = str(e.value)
    assert "tikv_instnce_type" in msg and "pdd" in msg and "azz" in msg      # all reported together


def test_duplicate_yaml_keys_rejected():
    first = '  tikv-1: {machine: "machine/ford10-google-storage@us-east-1"}\n'
    text = SMOKE.read_text()
    assert first in text
    with pytest.raises(S.SpecError, match="duplicate key"):
        S.parse_spec(text.replace(first, first + first))
    with pytest.raises(S.SpecError, match="YAML syntax"):
        S.parse_spec("slots: [unclosed\nfoo: bar\n")


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["control_plane"].update(pd=2), "must be odd"),
    (lambda d: d.update(replication_factor=4), "odd positive"),
    (lambda d: d.update(slots={"tikv-1": d["slots"]["tikv-1"]}), "cannot hold 3 replicas"),
    (lambda d: d["slots"]["tikv-1"].update(machine="machine/ford10-google-storage@eu-west-1"), "not one of"),
    (lambda d: d["slots"]["tikv-1"].update(machine="machine/nope@us-east-1"), "no entry"),
    (lambda d: d["slots"]["tikv-1"].update(machine="machine/spot-i4i.2xlarge-us-east-1@us-west-2"),
     "not available in us-west-2"),
    (lambda d: d.update(scenario="scenario/does-not-exist"), "scenario"),
    (lambda d: d.update(tidb_version="8.5.8"), "must look like"),
    (lambda d: d.update(handle="bad handle"), "invalid handle"),
    (lambda d: d["limits"].update(max_hours=0), "must be positive"),
    (lambda d: d["slots"]["tikv-1"].update(az="us-west-2a"), "not in region"),
])
def test_validation_errors(mutate, needle):
    d = smoke_dict()
    mutate(d)
    with pytest.raises(S.SpecError, match=needle):
        S.parse_spec(d)


def test_warnings_for_single_region_and_single_pd():
    d = smoke_dict()
    d["slots"] = {f"tikv-{i}": {"machine": "machine/ford10-google-storage@us-east-1"} for i in range(1, 4)}
    d["control_plane"].update(pd=1)
    sp = S.parse_spec(d)
    w = " | ".join(S.spec_warnings(sp))
    assert "span 1 region" in w and "single PD" in w


def test_resolve_handle_generates_or_checks():
    sp = S.parse_spec(SMOKE)
    r = S.resolve_handle(sp, taken=set(), rng=random.Random(3))
    assert r.handle != "auto" and sp.handle == "auto"                        # copy, original untouched
    with pytest.raises(names.HandleError):
        d = smoke_dict(); d["handle"] = "TakenName"
        S.resolve_handle(S.parse_spec(d), taken={"takenname"})


def test_plan_nodes_counts_roles_and_pd_spread():
    sp = S.parse_spec(SMOKE)
    nodes = S.plan_nodes(sp)
    by = {}
    for n in nodes:
        by.setdefault(n.role, []).append(n)
    assert {k: len(v) for k, v in by.items()} == {"controller": 1, "pd": 3, "tikv": 6, "systikv": 3, "tidb": 3, "loadgen": 3}
    assert [n.region for n in by["pd"]] == ["us-east-1", "us-east-2", "us-west-2"]     # one per region
    assert all(n.instance_type == "i4i.2xlarge" and n.machine for n in by["tikv"])
    assert all(n.instance_type == "m6i.xlarge" for n in nodes if n.role != "tikv")
    assert len({n.name for n in nodes}) == len(nodes)


def test_plan_nodes_pd_wraps_when_more_pd_than_regions():
    d = smoke_dict()
    d["control_plane"].update(pd=5)
    regions = [n.region for n in S.plan_nodes(S.parse_spec(d)) if n.role == "pd"]
    assert regions == ["us-east-1", "us-east-2", "us-west-2", "us-east-1", "us-east-2"]


# ---------------------------------------------------------------- durations / experiment config
@pytest.mark.parametrize("text, secs", [("30s", 30), ("10m", 600), ("24h", 86400), ("1.5h", 5400), ("2d", 172800),
                                        (45, 45), ("5 s", 5)])
def test_parse_duration(text, secs):
    assert S.parse_duration(text) == secs


@pytest.mark.parametrize("bad", ["", "10", "m", "-5s", "5x", True])
def test_parse_duration_rejects(bad):
    with pytest.raises(S.SpecError):
        S.parse_duration(bad)


def test_smoke_experiment_parses():
    e = S.parse_experiment(ROOT / "infra" / "tidb" / "experiments" / "smoke.yaml")
    assert (e.name, e.request_timeout_s, e.prober_interval_s, e.region_map_interval_s) == ("smoke", 5.0, 1.0, 10.0)
    assert e.workload.duration_s == 600 and e.workload.regions_target == 50 and e.injector == "none"


def test_experiment_errors_are_collected():
    with pytest.raises(S.SpecError) as err:
        S.parse_experiment({"name": "x y", "workload": {"type": "diurnal-hot", "rate_per_region": -1,
                                                       "duration": "ten"}, "injector": "kill", "oops": 1})
    m = str(err.value)
    for needle in ("name", "workload.type", "rate_per_region", "duration", "injector", "oops"):
        assert needle in m


def test_experiment_trace_injector_syntax_accepted():
    assert S.parse_experiment({"injector": "trace:out/trace.jsonl"}).injector == "trace:out/trace.jsonl"


def test_warns_when_only_pd_down_time_is_tuned():
    d = smoke_dict()
    d["pd_config"] = {"schedule.max-store-down-time": "10m"}
    assert any("max-peer-down-duration" in w for w in S.spec_warnings(S.parse_spec(d)))
    d["tikv_config"] = {"raftstore.max-peer-down-duration": "10m"}
    assert not any("max-peer-down-duration" in w for w in S.spec_warnings(S.parse_spec(d)))
