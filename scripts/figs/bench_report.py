"""Report for the scoring benchmark (outputs/bench): timing and error vs independent truth, table + figure.

    python scripts/figs/bench_report.py [--bench outputs/bench]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render as Rr  # noqa: E402  (palette + save; needs matplotlib)

plt = Rr.plt
CLIP = 0.3  # nines; larger errors are drawn clipped and labelled
LABEL = {"old": "old (1 fixed-seed batch)", "new": "new (re-selection batches + validation)",
         "new_is": "new + importance sampling (beta 2)"}
COLOR = {"old": Rr.MUTED, "new": Rr.BLUE, "new_is": Rr.ORANGE}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(Rr.K.ROOT / "outputs/bench"))
    a = ap.parse_args()
    bench = Path(a.bench)
    d = json.loads((bench / "truth.json").read_text())
    wall = dict(line.split() for line in (bench / "logs/wall.txt").read_text().splitlines() if line.strip())
    deps = [x for x in d["deployments"] if not x["carried"]]
    names = [n for n in ("old", "new", "new_is") if any(x["strategy"] == n for x in deps)]

    lines = ["# Fleet-ILP scoring benchmark: old vs new", "",
             f"32-vCPU c7i.8xlarge; 1x and 30x faults, rho1, budgets 10/20/30/50%. Truth: plain MC, "
             f"{d['runs']:,} runs per placement on independent seeds. Carried-forward budgets excluded.", "",
             "| strategy | total wall (s) | optimize (s) | validate (s) | mean abs error, believed | worst | "
             "mean abs error, validated | mean truth (nines) | mean cost ($/h) |", "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for n in names:
        xs = [x for x in deps if x["strategy"] == n]
        err = np.array([x["error_nines"] for x in xs])
        verr = [x["validated_error_nines"] for x in xs if x["validated_error_nines"] is not None]
        summary[n] = {"wall_s": float(wall.get(n, "nan")), "optimize_s": sum(x["optimize_s"] or 0 for x in xs),
                      "validate_s": sum(x["validate_s"] or 0 for x in xs), "mae": float(np.mean(np.abs(err))),
                      "worst": float(err[np.argmax(np.abs(err))]), "mean_signed": float(err.mean()),
                      "vmae": float(np.mean(np.abs(verr))) if verr else None,
                      "truth": float(np.mean([x["truth_nines"] for x in xs])),
                      "cost": float(np.mean([x["cost_per_hour"] for x in xs]))}
        s = summary[n]
        lines.append(f"| {LABEL[n]} | {s['wall_s']:.0f} | {s['optimize_s']:.0f} | {s['validate_s']:.0f} | "
                     f"{s['mae']:.2f} (signed {s['mean_signed']:+.2f}) | {s['worst']:+.2f} | "
                     f"{'-' if s['vmae'] is None else format(s['vmae'], '.2f')} | "
                     f"{s['truth']:.2f} | {s['cost']:.2f} |")
    lines += ["", "| strategy | faults | budget | believed | validated | truth | error | cost $/h | RF mix | optimize s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(deps, key=lambda x: (x["rate_scale"], float(x["budget"]), names.index(x["strategy"]))):
        lines.append(f"| {x['strategy']} | {x['rate_scale']:g}x | {float(x['budget']):.0%} | {x['believed_nines']:.2f} | "
                     f"{'-' if x['validated_nines'] is None else format(x['validated_nines'], '.2f')} | "
                     f"{x['truth_nines']:.2f} | {x['error_nines']:+.2f} | {x['cost_per_hour']:.2f} | {x['rf_counts']} | "
                     f"{x['optimize_s'] or 0:.0f} |")
    (bench / "report.md").write_text("\n".join(lines) + "\n")
    (bench / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    keys = sorted({(x["rate_scale"], float(x["budget"])) for x in deps})
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        by = {(x["rate_scale"], float(x["budget"])): x for x in deps if x["strategy"] == n}
        ys = [by[k]["error_nines"] if k in by else np.nan for k in keys]
        xs_ = np.arange(len(keys)) + (i - (len(names) - 1) / 2) * width
        ax.bar(xs_, np.clip(ys, -CLIP, CLIP), width * 0.9, color=COLOR[n], label=LABEL[n])
        for x_, y_ in zip(xs_, ys):
            if abs(y_) > CLIP:
                ax.text(x_, np.sign(y_) * CLIP, f"{y_:+.1f}", ha="center", va="top" if y_ < 0 else "bottom",
                        fontsize=6.5, color=Rr.INK2, rotation=90)
        vs = [by[k]["validated_error_nines"] if k in by and by[k]["validated_error_nines"] is not None else np.nan
              for k in keys]
        if not all(np.isnan(vs)):
            ax.scatter(np.arange(len(keys)) + (i - (len(names) - 1) / 2) * width, vs, marker="_", s=120,
                       color=Rr.INK, zorder=3, label="validated estimate" if n == "new" else None)
    ax.axhline(0, color=Rr.INK, linewidth=0.8)
    ax.set_ylim(-CLIP * 1.25, CLIP * 1.25)
    ax.set_xticks(range(len(keys)), [f"{s:g}x\n{b:.0%}" for s, b in keys], fontsize=7.5)
    ax.set_ylabel(f"Believed - truth (nines; clipped at +/-{CLIP:g})")
    ax.set_title("Estimate error vs independent truth", fontsize=9, loc="left", color=Rr.INK)
    ax.legend(fontsize=7)
    ax2.bar(range(len(names)), [summary[n]["optimize_s"] for n in names], color=[COLOR[n] for n in names],
            label="optimize")
    ax2.bar(range(len(names)), [summary[n]["validate_s"] for n in names], bottom=[summary[n]["optimize_s"] for n in names],
            color=Rr.GRID, edgecolor=[COLOR[n] for n in names], label="validate")
    ax2.set_xticks(range(len(names)), [n.replace("_", "+") for n in names])
    ax2.set_ylabel("Optimizer wall time (s, 8 budgets)")
    ax2.set_title("Runtime on 32 vCPUs", fontsize=9, loc="left", color=Rr.INK)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    Rr.save(fig, bench / "scoring_benchmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
