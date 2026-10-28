"""24-hour one-shot replays inject shared outages and overlay the choice figure."""

import xml.etree.ElementTree as ET

import numpy as np

from powder.faults import FaultLibrary
from powder.faults.convert import scenario_network_config
from scripts.experiment_az_24h_oneshot import SCENARIO, network_for, shared_events
from scripts.render_az_frontiers import render
from tests.test_az_frontier_renderer import point


def test_shared_events_inject_only_the_sampled_domains():
    model = scenario_network_config(FaultLibrary.load(), SCENARIO)
    assert network_for([], model) is None
    events = [{"domain": "us-east-1:az2", "kind": "az", "start_s": 3600.0,
               "duration_s": 600.0, "may_repeat": True}]
    cfg = network_for(events, model)
    assert cfg.per_az["us-east-1:az2"][0].value == 3600.0
    assert all(d.value == float("inf") for az, (d, _) in cfg.per_az.items() if az != "us-east-1:az2")
    assert all(d.value == float("inf") for d, _ in cfg.per_region.values())
    days = [shared_events(model, np.random.default_rng(i)) for i in range(200)]
    assert sum(map(bool, days)) < 10  # ~0.27% of days see any regional/AZ event


def test_oneshot_overlay_draws_every_day_and_perfect_days():
    arms = {}
    for price in ("rho0", "rho1"):
        baselines = {rf: {policy: {"status": "scored", "cost_per_hour": rf * 10,
                                   "mean_availability": .99997, "certified_groups": 10}
                          for policy in ("blind", "cheap")} for rf in (3, 5, 7)}
        arms[price] = {"baselines": baselines, "frontiers": {"choice": {
            "status": "scored", "points": [point(23, .99997), point(45, .999975, 10000)],
            "target_qualified": point(40, .999975, 10000)}}}
    result = {"scenario": SCENARIO, "target_nines": 4.5, "arms": arms}
    oneshot = {"scenario": SCENARIO, "frontier": "choice", "trials": 3, "horizon_h": 24.0,
               "arms": {price: [{"cost_per_hour": 23, "oneshot_availability": [.99999, 1.0, .999995]}]
                        for price in arms}}
    image = render(result, "choice", oneshot)
    ET.fromstring(image)
    assert image.count("<title>24 h day") == 6 and image.count("no RSM downtime") == 2
