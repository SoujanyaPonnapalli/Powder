"""Fault + price library (October Goals, execution order #1).

The library is a directory of JSON entries (default ``data/fault_library``)
organised in layers:

* ``source``            -- numbers extracted from a paper/dataset/assumption
* ``node`` / ``component`` / ``planned_downtime`` / ``region_outage``
                         -- fault profiles built from sources
* ``price``             -- AWS prices, price distributions, price assignments
* ``machine``           -- a fault profile joined with a price route
* ``scenario``          -- what an experiment (simulation or emulation) loads

See ``docs/plans/exec-01-canonical-failure-data.md`` for the design and
``docs/fault_library.md`` for the generated catalogue.

Everything here depends on NumPy only (no SciPy), so it runs in minimal
environments, including the emulation controller.
"""

from .library import FaultLibrary, LibraryError, default_root
from .convert import (
    machine_node_config,
    node_profile_rates,
    scenario_node_configs,
    scenario_network_config,
)

__all__ = [
    "FaultLibrary",
    "LibraryError",
    "default_root",
    "machine_node_config",
    "node_profile_rates",
    "scenario_node_configs",
    "scenario_network_config",
]
