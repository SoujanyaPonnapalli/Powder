"""Independent fixed-size follow-up to the zero-tail 10k Spot N=7 control.

Run: .venv/bin/python -m notebooks.availability_weekly_window_tail_control
Keep the original pilot evidence; use this independent 100k sample for the
final Spot N=7 distribution comparison. No data-dependent stopping is used.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from notebooks.availability_weekly_windows import FOLDER, VM_PROFILES, simulate_case


def main():
    row = simulate_case(VM_PROFILES[1], 7, "exponential_30d", 100000, 317000000)
    implementation = Path(__file__).with_name("availability_weekly_windows.py")
    data = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Original 10k control observed zero severe weeks; independent fixed 100k control avoids relying on that under-sampled tail",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "simulator_source_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
            "environment": json.loads((FOLDER / "mc.json").read_text())["metadata"],
            "sampling": "Independent fixed sample; disjoint seeds 317000000 through 317099999; no adaptive stopping",
        },
        "results": [row],
    }
    (FOLDER / "tail_control.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in row.items() if k not in ("batches", "early_stops", "tail_samples_below_99pct")}), flush=True)


if __name__ == "__main__":
    main()
