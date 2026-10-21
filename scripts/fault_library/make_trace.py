"""Generate a fault-injection trace for an emulation run.

    python scripts/fault_library/make_trace.py \
        --scenario scenario/canonical-central \
        --slot tikv-1=machine/aws-ondemand-nvme@us-east-1 \
        --slot tikv-2=machine/ford10-google-storage@us-west-2 \
        --horizon-hours 24 --seed 1 --out trace.jsonl

Each ``--slot NAME=MACHINE@REGION`` is one replica position. The output is
JSONL: a header (scenario, seed, horizon, library hashes) followed by one
event per line (see powder.faults.trace). The emulator (exec #8) replays it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from powder.faults import FaultLibrary  # noqa: E402
from powder.faults.trace import generate, write_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--slot", action="append", required=True, help="NAME=MACHINE@REGION")
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--level", choices=["optimistic", "central", "pessimistic"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    slots = {}
    for spec in a.slot:
        name, rest = spec.split("=", 1)
        machine, region = rest.rsplit("@", 1)
        slots[name] = (machine, region)
    lib = FaultLibrary.load()
    horizon = a.horizon_hours * 3600.0
    events = generate(lib, a.scenario, slots, horizon, a.seed, a.level)
    write_jsonl(a.out, lib, a.scenario, slots, horizon, a.seed, events, a.level)
    print(f"{len(events)} events -> {a.out}: {dict(Counter(e.kind for e in events))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
