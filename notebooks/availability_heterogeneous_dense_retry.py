"""Retry mixed five-node NO_ORPHANS dense integration with a 3-GiB guard.

Run: .venv/bin/python -m notebooks.availability_heterogeneous_dense_retry
The original dense 1.5-GiB attempt and 120-second BDF attempt remain recorded.
Reuse the bounded worker without its BDF validation step, which timed out in
the separate probe. The dense method still enforces probability diagnostics.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from notebooks import availability_heterogeneous_markov as benchmark


def main():
    output=benchmark.FOLDER/"n5-NO_ORPHANS-finite-dense-3gib.json"
    if "--worker" in sys.argv:
        benchmark.MEMORY_GIB=3.0
        # 'finite_probe' uses dense integration but bypasses the independent
        # BDF verification branch, whose separate measured attempt timed out.
        benchmark.worker(5,"NO_ORPHANS","finite_probe",output)
        return
    try:
        result=subprocess.run([sys.executable,"-m","notebooks.availability_heterogeneous_dense_retry","--worker"],
                              cwd=benchmark.ROOT,capture_output=True,text=True,timeout=benchmark.WALL_SECONDS)
        row=json.loads(output.read_text())
        if result.returncode and row["status"] != "memory_limit":
            row.update(status="error",returncode=result.returncode,error=result.stderr[-3000:])
    except subprocess.TimeoutExpired:
        row=json.loads(output.read_text())
        row.update(status="timeout",reason="120s worker wall limit")
    row.update(operation="finite",source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               independent_validation="Separate BDF attempt timed out; dense probability diagnostics retained")
    benchmark.dump(output,row)
    print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
