"""Bounded sparse-BDF fallback for the five-node mixed NO_ORPHANS model.

Run after the main benchmark:
  .venv/bin/python -m notebooks.availability_heterogeneous_bdf
The dense method exceeded the benchmark's 1.5-GiB resident-memory budget.
This is a separate numerical method, not a replacement of that measurement.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from notebooks.availability_heterogeneous_markov import (
    FOLDER, ROOT, MIXES, VM_PROFILES, MEMORY_GIB, WALL_SECONDS, peak_gib, dump,
    node_config_for, build_markov_model, raft_protocol, replacement_strategy,
    QualityLevel, bdf_time_average, days, os, np,
)


def worker():
    path = FOLDER / "n5-NO_ORPHANS-finite-bdf.json"
    profiles = {p.name:p for p in VM_PROFILES}
    configs = [node_config_for(profiles[name]) for name,n in zip(
        ("Standard","Unreliable","Spot"),MIXES[5]) for _ in range(n)]
    row = {"nodes":5,"quality":"NO_ORPHANS","operation":"finite",
           "method":"Sparse BDF integration; rtol=1e-10, atol=1e-14",
           "mix":{"Standard":2,"Unreliable":2,"Spot":1},
           "status":"building","memory_limit_gib":MEMORY_GIB,
           "wall_limit_seconds":WALL_SECONDS,"initial_leader_class":"Standard",
           "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    lock = threading.Lock()
    def save():
        with lock:
            dump(path,row)
    def guard():
        while True:
            if peak_gib() > MEMORY_GIB:
                row.update(status="memory_limit",peak_resident_gib=peak_gib())
                save()
                os._exit(86)
            time.sleep(.05)
    save()
    threading.Thread(target=guard,daemon=True).start()
    cfg=configs[0]
    warm=build_markov_model([cfg]*3,raft_protocol(),replacement_strategy(cfg),QualityLevel.SIMPLIFIED)
    bdf_time_average(warm,days(7))
    del warm
    start=time.perf_counter()
    model=build_markov_model(configs,raft_protocol(),replacement_strategy(cfg),QualityLevel.NO_ORPHANS)
    row.update(build_seconds=time.perf_counter()-start,states=model.num_states,
               nonzeros=model.Q.nnz,status="solving")
    save()
    start=time.perf_counter()
    value,diagnostics=bdf_time_average(model,days(7))
    seconds=time.perf_counter()-start
    if not 0 <= value <= 1 or abs(diagnostics["bdf_mass_error"]) > 1e-7:
        raise RuntimeError("Unacceptable BDF availability or mass error")
    row.update(status="complete",availability=value,solve_seconds=seconds,
               total_seconds=row["build_seconds"]+seconds,repeats=1,
               diagnostics=diagnostics,peak_resident_gib=peak_gib())
    save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    if args.worker:
        worker()
        return
    path=FOLDER / "n5-NO_ORPHANS-finite-bdf.json"
    try:
        result=subprocess.run([sys.executable,"-m","notebooks.availability_heterogeneous_bdf","--worker"],
                              cwd=ROOT,capture_output=True,text=True,timeout=WALL_SECONDS)
        row=json.loads(path.read_text())
        if result.returncode and row["status"] != "memory_limit":
            row.update(status="error",returncode=result.returncode,error=result.stderr[-3000:])
    except subprocess.TimeoutExpired:
        row=json.loads(path.read_text())
        row.update(status="timeout",reason=f"{WALL_SECONDS}s worker wall limit")
    dump(path,row)
    print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
