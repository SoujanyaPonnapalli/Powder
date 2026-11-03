"""Dry-run preview (exec #7 step 2)."""

import json
from pathlib import Path

import pytest

from powder.deploy import plan as PL
from powder.deploy import spec as S

SMOKE = Path(__file__).resolve().parents[1] / "infra" / "tidb" / "specs" / "smoke.yaml"


def test_smoke_preview_text(capsys):
    assert PL.main([str(SMOKE)]) == 0
    out = capsys.readouterr().out
    assert "16 machines in 3 regions" in out and "3 peerings" in out
    assert "AWS bill: ~$6.0" in out and "cap for 3 h: ~$18" in out
    assert "us-east-1 32" in out and "us-east-2 28" in out          # vCPUs for the quota check
    assert "Modeled cost of the emulated machine classes" in out
    assert "random AdjectiveNoun" in out                              # handle is generated at deploy, not here
    assert "not included: root EBS volumes" in out


def test_json_output_is_machine_readable(capsys):
    assert PL.main([str(SMOKE), "--json"]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["cost"]["hourly_usd"] == pytest.approx(6.036, abs=0.01)
    assert p["network"]["peering_count"] == 3 and p["problems"] == []
    assert sum(1 for n in p["nodes"] if n["role"] == "tikv") == 6


def test_exceeding_the_spend_limit_is_a_problem(tmp_path, capsys):
    d = S.load_yaml(SMOKE)
    d["limits"]["max_hourly_usd"] = 2
    f = tmp_path / "cheap.yaml"
    import yaml
    f.write_text(yaml.safe_dump(d))
    assert PL.main([str(f)]) == 1
    assert "PROBLEM: estimated" in capsys.readouterr().out


def test_invalid_spec_exits_2_listing_every_problem(tmp_path, capsys):
    f = tmp_path / "bad.yaml"
    f.write_text("slots: {tikv-1: {machine: 'machine/nope@us-east-1'}}\nbogus: 1\n")
    assert PL.main([str(f)]) == 2
    err = capsys.readouterr().err
    assert "bogus" in err and "machine/nope" in err and "cannot hold 3 replicas" in err


def test_unpriced_scenario_reports_modeled_cost_na(tmp_path, capsys):
    d = S.load_yaml(SMOKE)
    d["scenario"] = "scenario/canonical-central"
    import yaml
    f = tmp_path / "unpriced.yaml"
    f.write_text(yaml.safe_dump(d))
    assert PL.main([str(f)]) == 0
    assert "Modeled cost: n/a" in capsys.readouterr().out
