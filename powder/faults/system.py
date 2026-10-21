"""System timings shared by machine classes (exec #10 placeholders).

Simple estimate agreed on 2026-10-21 (source ``exec10-simple-estimate``), to be
replaced by measured AWS timings as ``@v2`` entries:

* snapshot / partition transfer: 1 GiB over 1 Gbps = 8.59 s
* spawning a replacement node: Uniform(1, 3) minutes
* rebuild of a permanently failed node = E[spawn] + snapshot = 128.6 s
"""

from __future__ import annotations

PARTITION_BYTES = 2**30  # 1 GiB (2026-10-21; was 256 MiB)
NETWORK_BPS = 1e9
SNAPSHOT_S = PARTITION_BYTES * 8 / NETWORK_BPS
SPAWN = {"type": "uniform", "low": 60.0, "high": 180.0}
SYSTEM = {
    "snapshot_download_time_dist": {"type": "constant", "value": SNAPSHOT_S},
    "spawn": SPAWN,
    "source": "exec10-simple-estimate",
}
REBUILD_S = 0.5 * (SPAWN["low"] + SPAWN["high"]) + SNAPSHOT_S
REBUILD_H = REBUILD_S / 3600.0
