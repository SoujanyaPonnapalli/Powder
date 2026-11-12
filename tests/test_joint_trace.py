"""One stochastic fault per physical host/domain, with overlap-preserving replay."""

from types import SimpleNamespace

import pytest

from powder.faults import joint_trace
from powder.faults.trace import FaultEvent


def ev(slot, region, kind, start, duration):
    return FaultEvent(slot, region, "machine", kind, kind, start, duration)


def test_host_faults_overlap_without_premature_recovery():
    events = [ev("h1", "r1", "transient", 1, 4),
              ev("h1", "r1", "planned", 3, 4),
              ev("h1", "r1", "planned", 7, 2),
              ev("h2", "r1", "permanent", 5, None)]
    assert joint_trace.fault_events_to_toggles(events, 10) == [
        {"scope": "host", "target": "h1", "time": 1, "down": True},
        {"scope": "host", "target": "h2", "time": 5, "down": True},
        {"scope": "host", "target": "h1", "time": 9, "down": False},
    ]


def test_global_region_az_outages_keep_distinct_shared_domains():
    toggles = joint_trace.fault_events_to_toggles([
        ev("*", "r1", "region", 1, 2),
        ev("*", "*", "global", 2, 2),
        ev("*", "r1:az1", "az", 3, 1),
    ], 5)
    assert len(toggles) == 6
    assert {(e["scope"], e.get("target")) for e in toggles if e["down"]} == {
        ("region", "r1"), ("global", None), ("az", "r1:az1")}
    assert len([e for e in toggles if not e["down"]]) == 3


def test_invalid_faults_fail_closed():
    with pytest.raises(ValueError, match="physical target"):
        joint_trace.fault_events_to_toggles([ev("*", "r", "transient", 1, 2)], 3)
    with pytest.raises(ValueError, match="finite"):
        joint_trace.fault_events_to_toggles([ev("h", "r", "transient", 1, -1)], 3)
    with pytest.raises(ValueError, match="unknown"):
        joint_trace.fault_events_to_toggles([ev("h", "r", "moon", 1, 1)], 3)


def test_seeded_az_draws_are_stable_and_do_not_duplicate_host_events(monkeypatch):
    monkeypatch.setattr(joint_trace, "generate", lambda *a, **k: [ev("h", "r", "transient", 0.3, 0.2)])
    monkeypatch.setattr(joint_trace, "scenario_network_config", lambda *a, **k: SimpleNamespace(
        az_rate_per_second=100, az_duration=SimpleNamespace(sample=lambda rng: 0.05)))
    slots = {"h": ("machine", "r", "r:az1")}
    first = joint_trace.seeded_joint_toggles(None, "scenario", slots, 0.5, 42)
    again = joint_trace.seeded_joint_toggles(None, "scenario", slots, 0.5, 42)
    assert first == again
    assert any(e["scope"] == "az" for e in first)
    assert len([e for e in first if e["scope"] == "host" and e["down"]]) == 1
    with pytest.raises(ValueError, match="region and AZ"):
        joint_trace.seeded_joint_toggles(None, "scenario", {"h": ("machine", "r")}, 1, 1)


def test_seeded_joint_draw_uses_exact_job_host_ids_and_regions(monkeypatch):
    from powder import mc_backend

    captured = []
    monkeypatch.setattr(joint_trace, "seeded_joint_toggles", lambda *a, **k: [
        {"time": 1, "scope": "region", "target": "r1", "down": True}])
    monkeypatch.setattr(mc_backend, "run_joint_replay", lambda job: captured.append(job) or {"ok": True})
    job = {"mode": "joint", "horizon_seconds": 10,
           "hosts": {"h1": {"region": "r1", "az": "r1:az1"}}}
    slots = {"h1": ("machine/c1", "r1", "r1:az1")}
    assert joint_trace.run_seeded_joint_replay(job, None, "s", slots, 3) == {"ok": True}
    assert captured[0]["events"][0]["target"] == "r1"
    with pytest.raises(ValueError, match="must match"):
        joint_trace.run_seeded_joint_replay(job, None, "s",
                                            {"h1": ("machine/c1", "r2", "r1:az1")}, 3)
