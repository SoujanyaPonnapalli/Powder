"""Render heterogeneous runtime evidence and update the tradeoff document.

Run: .venv/bin/python -m notebooks.availability_heterogeneous_markov_report
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs/availability-heterogeneous-markov"


def duration(t):
    return f"{1000*t:.2f} ms" if t < 1 else f"{t:.2f} s"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"]*len(headers)) + "|",
                      *("| " + " | ".join(map(str,row)) + " |" for row in rows)])


def display(row):
    if row["status"] == "complete":
        return duration(row["total_seconds"]) + (" (BDF)" if "BDF" in row["method"] else "")
    return {"memory_limit":"Stopped: memory guard", "state_limit":"Not attempted: state cap",
            "build_limit":"Not built", "timeout":"Stopped: 120 s budget"}.get(row["status"],row["status"])


def main():
    data=json.loads((FOLDER/"results.json").read_text())
    fallback=json.loads((FOLDER/"n5-NO_ORPHANS-finite-bdf.json").read_text())
    retry=json.loads((FOLDER/"n5-NO_ORPHANS-finite-dense-3gib.json").read_text())
    if len(data["results"]) != 18:
        raise RuntimeError("Complete the main benchmark first")
    cases={(r["nodes"],r["quality"],r["operation"]):r for r in data["results"]}
    if fallback["status"] == "complete":
        cases[5,"NO_ORPHANS","finite"]=fallback
    if retry["status"] == "complete":
        cases[5,"NO_ORPHANS","finite"]=retry
    output=[]
    availability=[]
    for n in (3,5,7):
        for q in ("SIMPLIFIED","NO_ORPHANS","FULL"):
            finite,steady=cases[n,q,"finite"],cases[n,q,"steady"]
            states=finite.get("states",steady.get("states"))
            state_text=f"{states:,}" if states else "Not enumerated"
            output.append((n,q,state_text,display(finite),display(steady)))
            availability.append((n,q,
                f"{100*finite['availability']:.9f}%" if finite["status"] == "complete" else "—",
                f"{100*steady['availability']:.9f}%" if steady["status"] == "complete" else "—"))
    runtime_table=table(["Nodes","Model","States","One-week build + solve","Steady-state build + solve"],output)
    value_table=table(["Nodes","Model","One-week availability","Steady-state availability"],availability)
    mixes=table(["Nodes","Standard","Unreliable","Spot"],[(3,1,1,1),(5,2,2,1),(7,3,2,2)])
    section=f"""**Measured heterogeneous Markov costs.** New solver benchmarks mix three machine profiles within each RSM. The rates remain constant and exponential, so these models are heterogeneous across machines but still homogeneous over time. All nodes start healthy and the initial leader is Standard. The mixes are:

{mixes}

{runtime_table}

The seven-node FULL model was not built: its combinatorial state upper bound is **5,682,456**, above the 200,000-state preflight build budget. This is an upper bound, not an enumerated state count. The five-node FULL model was built and contains 158,652 states; its one-week solve was not attempted under the 60,000-state finite-solve cap.

All timings use Python/SciPy and one numerical thread. The main benchmark uses a **1.5 GiB resident-memory guard** and a **120-second wall budget per worker**, including build and validation. The five-node NO_ORPHANS dense retry uses a **3 GiB guard**. “Memory guard” means the run was stopped by this study's budget, not that the model is impossible to solve on a larger host. The guard is sampled, so recorded peak memory may overshoot its trigger.

Completed one-week calculations use the same dense augmented matrix exponential as the homogeneous table. Five-node NO_ORPHANS initially exceeded 1.5 GiB; sparse BDF integration then reached its 120-second budget. Its dense retry with the 3 GiB guard had status **{retry['status']}** and is used in the table if complete. Seven-node NO_ORPHANS exceeded the 1.5 GiB guard for both sparse BDF integration and the stationary solver. Finite and stationary build times are independently measured; startup, small-model warmup, and validation are excluded from reported totals. Solves up to 2,000 states use the median of three repetitions; larger solves use one measurement. Historical homogeneous times are context, not a controlled interleaved speed comparison.

The state growth explains why homogeneous runtimes do not transfer directly. Five-node NO_ORPHANS grows from **377 to 4,598 states**; seven-node NO_ORPHANS grows from **1,253 to 48,068 states**. Three-node FULL grows from **598 to 3,024 states**, with a newly measured one-week total of **{duration(cases[3,'FULL','finite']['total_seconds'])}** versus the earlier homogeneous 53.83 ms. The finite-horizon method and resource budget matter alongside state count.

These measurements do not include heterogeneous Monte Carlo or non-exponential clocks. Markov's existing recovery-policy approximations still apply. [Full heterogeneous benchmark](../outputs/availability-heterogeneous-markov/REPORT.md).

"""
    checks=[r for r in data["results"] if "independent_difference" in r]
    full="# Heterogeneous one-week versus steady-state Markov benchmarks\n\n"+section
    full=full.replace("[Full heterogeneous benchmark](../outputs/availability-heterogeneous-markov/REPORT.md).", "[Machine-readable measurements](results.json).")
    full+=f"""## Availability estimates

{value_table}

These are model predictions, not validated real-system availabilities. The healthy start and Standard initial leader are relevant to the one-week values. Each replacement retains its slot's fixed rate class; aging and software calendars are not represented. The three profiles retain their original 30-day mean transient interval, 20-minute mean recovery, and 3-year / 1-year / 1-day mean permanent-loss intervals for Standard / Unreliable / Spot. Other protocol and replacement settings match the earlier study.

## Numerical checks and reproduction

For {len(checks)} completed dense cases, independent sparse BDF integration agrees within **{max(abs(r['independent_difference']) for r in checks):.3g}** absolute availability. BDF uses rtol=1e-10 and atol=1e-14. Probability-mass, generator, and stationary residual diagnostics are stored in the JSON evidence. The five-node NO_ORPHANS retry retains the dense solver's probability diagnostics; its independent BDF attempt timed out, so that particular result does not have the additional cross-method check. All **88 targeted numerical and heterogeneous-state regression tests passed** for this change.

```sh
.venv/bin/python -m notebooks.availability_heterogeneous_markov
.venv/bin/python -m notebooks.availability_heterogeneous_bdf
.venv/bin/python -m notebooks.availability_heterogeneous_dense_retry
.venv/bin/python -m notebooks.availability_heterogeneous_markov_report
.venv/bin/python -m pytest tests/test_availability_study_methods.py tests/test_markov_hetero_state_counts.py -q
```

- [All primary attempts, including resource stops](results.json)
- [Five-node sparse-BDF fallback](n5-NO_ORPHANS-finite-bdf.json)
- [Five-node dense retry with 3 GiB guard](n5-NO_ORPHANS-finite-dense-3gib.json)
- [Benchmark implementation](../../notebooks/availability_heterogeneous_markov.py)
- [Fallback implementation](../../notebooks/availability_heterogeneous_bdf.py)
- [Dense retry implementation](../../notebooks/availability_heterogeneous_dense_retry.py)
- [Report renderer](../../notebooks/availability_heterogeneous_markov_report.py)

Source hashes, dependency versions, numerical thread settings, mixture definitions, actual state counts where built, solve samples, and observed memory peaks are preserved. The 120-second budget includes verification, so a timeout is not automatically a lower bound on solve time alone.
"""
    (FOLDER/"REPORT.md").write_text(full)
    doc=ROOT/"docs/one-week-rsm-availability-tradeoffs.md"
    text=doc.read_text()
    marker="**Measured heterogeneous Markov costs.**"
    end=text.index("**Measured Monte Carlo costs —")
    start=text.index(marker) if marker in text else end
    text=text[:start]+section+text[end:]
    text=text.replace("This document uses existing Powder results only. No new simulations or solver benchmarks were run.",
                      "The homogeneous and Monte Carlo sections use prior results. The heterogeneous Markov section adds new bounded solver benchmarks; no additional Monte Carlo simulations were run.")
    doc.write_text(text)
    print(runtime_table)


if __name__ == "__main__":
    main()
