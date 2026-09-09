"""Bounded, single-thread benchmarks of heterogeneous one-week Markov models.

Run: .venv/bin/python -m notebooks.availability_heterogeneous_markov
Each build/solve runs in an isolated process with a 120-second wall budget and
a 1.5-GiB resident-memory guard. Dense and sparse-BDF finite-horizon methods
are reported separately; no Monte Carlo simulations are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from notebooks.availability_convergence_study import (
    ROOT, VM_PROFILES, THREAD_VARIABLES, np, scipy, os, days, node_config_for,
    raft_protocol, replacement_strategy,
)
from notebooks.availability_finite_horizon import dense_time_average, bdf_time_average
from powder.scenario import QualityLevel, build_markov_model
from powder.markov_solver import steady_state

FOLDER = ROOT / "outputs/availability-heterogeneous-markov"
MIXES = {3: (1, 1, 1), 5: (2, 2, 1), 7: (3, 2, 2)}
QUALITIES = ("SIMPLIFIED", "NO_ORPHANS", "FULL")
MEMORY_GIB = 1.5
WALL_SECONDS = 120


def state_bound(sizes, quality):
    k = {"SIMPLIFIED": 3, "NO_ORPHANS": 6, "FULL": 12}[quality]
    no_leader = math.prod(math.comb(n+k-1, k-1) for n in sizes)
    leader = sum(math.prod(math.comb(n-(i==c)+k-1, k-1)
                           for i,n in enumerate(sizes)) for c in range(len(sizes)))
    return no_leader + (3 if quality == "FULL" else 1)*leader


def dump(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def peak_gib():
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale/1024**3


def worker(nodes, quality, operation, output):
    sizes = MIXES[nodes]
    profiles = {p.name:p for p in VM_PROFILES}
    configs = [node_config_for(profiles[name]) for name,n in zip(
        ("Standard", "Unreliable", "Spot"), sizes) for _ in range(n)]
    row = {"nodes":nodes,"quality":quality,"mix":dict(zip(
        ("Standard","Unreliable","Spot"),sizes)),"operation":operation,
        "state_upper_bound":state_bound(sizes,quality),"status":"building",
        "memory_limit_gib":MEMORY_GIB,"wall_limit_seconds":WALL_SECONDS,
        "initial_leader_class":"Standard","horizon_seconds":days(7)}
    lock = threading.Lock()

    def save():
        with lock:
            dump(output,row)

    def guard():
        while True:
            peak = peak_gib()
            if peak > MEMORY_GIB:
                row.update(status="memory_limit",peak_resident_gib=peak)
                save()
                os._exit(86)
            time.sleep(.05)

    save()
    threading.Thread(target=guard,daemon=True).start()
    # Warm small numerical calls outside measured builds/solves.
    cfg = configs[0]
    warm = build_markov_model([cfg]*3,raft_protocol(),replacement_strategy(cfg),QualityLevel.SIMPLIFIED)
    dense_time_average(warm,days(7))
    steady_state(warm,backend="scipy")
    del warm
    start = time.perf_counter()
    model = build_markov_model(configs,raft_protocol(),replacement_strategy(cfg),QualityLevel[quality])
    row.update(build_seconds=time.perf_counter()-start,states=model.num_states,
               nonzeros=model.Q.nnz,dense_matrix_gib=(model.num_states+1)**2*8/1024**3,
               initial_availability=float(model.initial_distribution@model.live_mask),
               generator_row_sum_error=float(np.max(np.abs(np.asarray(model.Q.sum(axis=1)).ravel()))),
               status="solving")
    save()
    if operation == "finite" and model.num_states > 60000:
        row.update(status="state_limit",reason="Finite solve cap: 60000 states",peak_resident_gib=peak_gib())
        save()
        return
    repeats = 3 if model.num_states <= 2000 else 1
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        if operation == "steady":
            distribution = steady_state(model,backend="scipy")
            value = 1-float(distribution[~model.live_mask].sum())
            diagnostics = {"mass_error":float(distribution.sum()-1),
                           "minimum_probability":float(distribution.min()),
                           "stationary_residual_inf":float(np.max(np.abs(model.Q.T@distribution)))}
            method = "Sparse stationary solve (SciPy)"
        elif model.num_states <= 5000:
            distribution,diagnostics = dense_time_average(model,days(7))
            value = 1-float(distribution[~model.live_mask].sum())
            method = "Dense augmented matrix exponential"
        else:
            value,diagnostics = bdf_time_average(model,days(7))
            method = "Sparse BDF integration; rtol=1e-10, atol=1e-14"
        times.append(time.perf_counter()-start)
    row.update(status="complete",method=method,availability=value,
               solve_seconds=float(np.median(times)),solve_samples_seconds=times,
               total_seconds=row["build_seconds"]+float(np.median(times)),
               repeats=repeats,diagnostics=diagnostics,peak_resident_gib=peak_gib())
    if not 0 <= value <= 1:
        raise RuntimeError("Invalid availability")
    # Independent integration check outside benchmark timing for smaller cases.
    if operation == "finite" and model.num_states <= 5000:
        row["status"] = "verifying"
        save()
        other,check = bdf_time_average(model,days(7))
        row.update(independent_bdf_availability=other,independent_bdf_diagnostics=check,
                   independent_difference=other-value,status="complete",peak_resident_gib=peak_gib())
        if abs(other-value) > 1e-8:
            raise RuntimeError("Dense/BDF disagreement")
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--nodes",type=int,choices=(3,5,7))
    parser.add_argument("--quality",choices=QUALITIES)
    parser.add_argument("--operation",choices=("finite","steady"))
    args = parser.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True)
    if args.worker:
        path=FOLDER/f"n{args.nodes}-{args.quality}-{args.operation}.json"
        worker(args.nodes,args.quality,args.operation,path)
        return
    data={"metadata":{
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
        "platform":platform.platform(),"threads":{k:os.environ[k] for k in THREAD_VARIABLES},
        "description":"Three machine classes, constant exponential rates, initial Standard leader, healthy start",
        "comparison":"Total runtimes include independently measured build plus solve; startup/warmup/verification excluded",
    },"results":[]}
    for nodes,sizes in MIXES.items():
        for quality in QUALITIES:
            upper=state_bound(sizes,quality)
            for operation in ("finite","steady"):
                if upper > 200000:
                    row={"nodes":nodes,"quality":quality,"operation":operation,"state_upper_bound":upper,
                         "mix":dict(zip(("Standard","Unreliable","Spot"),sizes)),
                         "status":"build_limit","reason":"Preflight upper bound exceeds 200000-state build budget"}
                else:
                    path=FOLDER/f"n{nodes}-{quality}-{operation}.json"
                    command=[sys.executable,"-m","notebooks.availability_heterogeneous_markov",
                             "--worker","--nodes",str(nodes),"--quality",quality,"--operation",operation]
                    print(f"Starting N={nodes} {quality} {operation}",flush=True)
                    try:
                        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=WALL_SECONDS)
                        row=json.loads(path.read_text())
                        if result.returncode and row["status"] != "memory_limit":
                            row.update(status="error",returncode=result.returncode,error=result.stderr[-3000:])
                    except subprocess.TimeoutExpired:
                        row=json.loads(path.read_text())
                        row.update(interrupted_phase=row["status"],status="timeout",reason=f"{WALL_SECONDS}s worker wall limit, including build and verification")
                    dump(path,row)
                data["results"].append(row)
                dump(FOLDER/"results.json",data)
                print(json.dumps({k:v for k,v in row.items() if k not in ("diagnostics","independent_bdf_diagnostics")}),flush=True)


if __name__ == "__main__":
    main()
