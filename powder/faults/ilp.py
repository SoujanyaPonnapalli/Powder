"""Scenario -> placement-optimizer machine catalogue.

    from powder.faults import FaultLibrary
    from powder.faults.ilp import scenario_machine_types
    types = scenario_machine_types(FaultLibrary.load(), "scenario/canonical-central-rho1", copies=10)

Each (machine class, region) pair of the scenario becomes one
``placement_optimizer.MachineType`` with ``type_id = "<machine>@<region>"``.
The optimizer module is imported lazily because it depends on SciPy.
"""

from __future__ import annotations

from .convert import scenario_node_configs
from .library import FaultLibrary


def scenario_machine_types(lib: FaultLibrary, scenario_id: str, *, copies: int = 1, capacity: int = 1,
                           engine: bool = False, level: str | None = None,
                           price_assignment: str | None = None):
    from powder.placement_optimizer import MachineType

    cfgs = scenario_node_configs(lib, scenario_id, engine=engine, level=level, price_assignment=price_assignment)
    return [MachineType(type_id=f"{m}@{r}", node_config=cfg, copies=copies, capacity=capacity)
            for (m, r), cfg in sorted(cfgs.items())]
