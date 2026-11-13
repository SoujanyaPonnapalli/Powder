#!/usr/bin/env python3
"""Export scored physical placements for one TiDB cluster's PD store IDs.

The machine mapping must correspond to the same physical machines used when
scoring the candidates. A mismatched mapping or changed PD store labels causes
the runtime oracle to defer to PD; do not map a sampled catalogue machine to an
unrelated deployed machine just to fill the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from powder.deploy.pd_oracle import StoreSignature, from_candidates  # noqa: E402
from powder.deploy.regionmap import parse_stores  # noqa: E402
from powder.placement_optimizer import PlacementCandidate  # noqa: E402


def export(candidates_doc: dict, mapping: dict, stores_doc: dict, model_id: str) -> dict:
    raw_candidates = candidates_doc["candidates"]
    machine_to_store = dict(mapping)
    stores = {store_id: StoreSignature(s.machine_class, s.region, s.zone, s.host)
              for store_id, s in parse_stores(stores_doc).items()}
    candidates = [PlacementCandidate(str(c["candidate_id"]), tuple(c["machine_ids"]),
                                     c["availability"], c["cost_per_hour"])
                  for c in raw_candidates]
    return from_candidates(candidates, machine_to_store, stores, model_id).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="JSON object with scored candidates")
    parser.add_argument("--machine-map", type=Path, required=True, help="JSON: optimizer machine ID -> PD store ID")
    parser.add_argument("--stores", type=Path, required=True, help="PD /pd/api/v1/stores JSON snapshot")
    parser.add_argument("--model-id", required=True, help="immutable score/fault-library version for this run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(json.loads(args.candidates.read_text()), json.loads(args.machine_map.read_text()),
                    json.loads(args.stores.read_text()), args.model_id)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
