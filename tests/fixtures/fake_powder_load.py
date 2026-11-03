#!/usr/bin/env python3
"""Stand-in for the powder-load binary in tests: deterministic records with a known outage.

keys=100000 split into 8 partitions of 12500 rows (the same layout as the captured TiDB fixtures, table 114).
Outage: for t in [10, 20) s after the start, every request/probe to rows >= 50000 (partitions 4..7) times out.
"""
import gzip, json, sys, time

args = sys.argv[1:]
cmd, flags = args[0], {}
i = 1
while i < len(args):
    flags[args[i].lstrip("-")] = args[i + 1]
    i += 2

def dur(s): return float(s.rstrip("s"))

if cmd == "version":
    print("powder-load 1 error_classes=fake")
elif cmd == "prepare":
    print(json.dumps({"table_id": 114, "keys": int(flags["keys"]), "regions": int(flags["regions"]), "preloaded": 0}))
else:
    start, D, region = float(flags["start-at"]), dur(flags["duration"]), flags["region"]
    keys = int(flags["keys"])
    def rec(ts, row, op):
        down = 10 <= ts - start < 20 and row >= 50000
        r = {"ts": ts, "client_region": region, "row": row, "op": op, "latency_ms": 5000.0 if down else 3.0}
        if down:
            r.update(outcome="timeout", cause="tidb_9005_region_unavailable", code=9005, layer="tikv")
        else:
            r["outcome"] = "ok"
        return r
    with gzip.open(flags["out"], "wt") as f:
        if cmd == "load":
            for t in range(int(D)):
                f.write(json.dumps(rec(start + t + 0.25, 6000, "read")) + "\n")      # partition 0
                f.write(json.dumps(rec(start + t + 0.75, 80000, "write")) + "\n")    # partition 6
        else:
            b = int(flags["buckets"])
            for t in range(int(D)):
                for k in range(b):
                    f.write(json.dumps(rec(start + t + 0.3, int((k + 0.5) * keys / b), "probe")) + "\n")
    print(json.dumps({"start": start, "end": start + D, "records": 0}))
