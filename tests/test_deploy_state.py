"""RunState / state.json (exec #7 step 2)."""

import json
import os

import pytest

from powder.deploy import state as ST


def make_state():
    s = ST.RunState(handle="OrangePanda", spec_hash="abc")
    s.region("us-east-1").vpc_id = "vpc-1"
    s.region("us-east-1").peerings["us-east-2"] = "pcx-1"
    s.add_instance(ST.InstanceRecord("controller", "controller", "us-east-1", "m6i.xlarge", instance_id="i-0"))
    s.add_instance(ST.InstanceRecord("tikv-1", "tikv", "us-east-1", "i4i.2xlarge", slot="tikv-1",
                                     machine="machine/x", private_ip="10.0.0.5"))
    return s


def test_round_trip(tmp_path):
    s = make_state()
    p = tmp_path / "state.json"
    s.save(p)
    assert ST.RunState.load(p).to_dict() == s.to_dict()
    assert json.loads(p.read_text())["schema"] == ST.SCHEMA_VERSION


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    s = make_state()
    s.save(p)
    before = p.read_text()
    s.set_stage("infra", "done")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        s.save(p)
    assert p.read_text() == before                       # old file intact
    assert [x.name for x in tmp_path.iterdir()] == ["state.json"]   # temp file cleaned up


def test_load_errors(tmp_path):
    with pytest.raises(ST.StateError, match="has this handle been deployed"):
        ST.RunState.load(tmp_path / "missing.json")
    p = tmp_path / "state.json"
    d = make_state().to_dict()
    d["schema"] = 99
    p.write_text(json.dumps(d))
    with pytest.raises(ST.StateError, match="schema"):
        ST.RunState.load(p)


def test_queries_and_guards():
    s = make_state()
    assert s.controller.name == "controller" and s.by_slot("tikv-1").private_ip == "10.0.0.5"
    assert [i.name for i in s.by_role("tikv")] == ["tikv-1"]
    with pytest.raises(ST.StateError):
        s.by_slot("tikv-9")
    with pytest.raises(ST.StateError):
        s.add_instance(ST.InstanceRecord("tikv-1", "tikv", "us-east-1", "i4i.2xlarge"))
    with pytest.raises(ST.StateError):
        s.set_stage("infra", "bogus")
    with pytest.raises(ST.StateError, match="not installed"):
        s.require_installed()
    s.set_stage("install", "done")
    s.require_installed()
