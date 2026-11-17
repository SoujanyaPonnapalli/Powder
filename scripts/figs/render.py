"""Render the paper figures from the JSON the figure scripts write (PNG + SVG next to the data).

    python scripts/figs/render.py whatif outputs/figs/whatif/x30-rho1-b0.5-h24.json
    python scripts/figs/render.py ladder            # outputs/figs/modeling/{markov,mc}
    python scripts/figs/render.py sensitivity       # outputs/figs/sensitivity/sensitivity.json
    python scripts/figs/render.py scalability       # outputs/figs/scalability/past_runs.json
    python scripts/figs/render.py reconfig DIR      # outputs/reconfig-cadence*/summary.json

Palette: the dataviz reference categorical order (blue, orange, aqua, yellow, ...); text in ink tokens.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GB_PER_REPLICA = 100.0  # copied per replacement (same assumption as reconfig_cadence.py)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "svg.fonttype": "none",
})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):  # append (names like b0.5-h24 contain dots)
        fig.savefig(path.parent / f"{path.name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.parent / (path.name + '.png')}")


# ------------------------------------------------------------------------------------------------ Fig 4

def whatif(path: str) -> None:
    d = json.loads(Path(path).read_text())
    rs = d["results"]
    labels = [f"{r['timeout_min']:g} min" for r in rs]
    y = [r["nines"] for r in rs]
    lo = [r["nines"] - r["nines_ci95"][0] for r in rs]
    hi = [r["nines_ci95"][1] - r["nines"] for r in rs]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    x = range(len(rs))
    ax.bar(x, y, width=0.55, color="white", edgecolor=BLUE, hatch="////", linewidth=1.2, label="Expected (MC estimator)")
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=4, linewidth=1)
    for i, r in enumerate(rs):
        ax.text(i, r["nines_ci95"][1] + 0.01, f"{r['nines']:.2f}", ha="center", va="bottom", color=INK, fontsize=8)
    floor = min(r["nines_ci95"][0] for r in rs)
    ax.set_ylim(max(0, floor - 0.25), max(r["nines_ci95"][1] for r in rs) + 0.15)
    ax.set_ylabel("Availability (nines)\n-log10(1-A)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"What-if: node replacement timeout ({d['rate_scale']:g}x faults, {d['price']}, "
                 f"{float(d['budget']):.0%} budget, {d['horizon_h']:g} h, {d['runs'] // 1000}k runs/RSM)",
                 fontsize=9, color=INK, loc="left")
    ax2.bar(x, [r["replacements_per_day"] for r in rs], width=0.55, color=ORANGE)
    for i, r in enumerate(rs):
        ax2.text(i, r["replacements_per_day"], f"{r['replacements_per_day'] * GB_PER_REPLICA / 1000:.1f} TB/day",
                 ha="center", va="bottom", fontsize=7, color=INK2)
    ax2.set_ylabel("Replacements\nper day")
    ax2.set_ylim(0, max(r["replacements_per_day"] for r in rs) * 1.25)
    ax2.set_xticks(list(x), labels)
    ax2.set_xlabel("Node replacement timeout")
    save(fig, Path(path).parent / Path(path).name.removesuffix(".json"))


# ------------------------------------------------------------------------------------------------ Fig 13

RUNG_LABEL = {"indep_homog": "Homog.\nEC", "indep_hetero": "+Hetero-\ngeneous", "leaderless_no": "+Up /\nstale",
              "raft_no": "+Leader\nelection", "raft_full": "+Full\npipeline", "mc_nonet": "MC, no\noutages",
              "mc_net": "MC"}


def ladder() -> None:
    base = K.ROOT / "outputs/figs/modeling"
    rsms = json.loads((base / "rsms.json").read_text())
    scales = sorted({r["rate_scale"] for r in rsms})
    ns = sorted({r["n"] for r in rsms})
    fig, axes = plt.subplots(len(scales), len(ns), figsize=(3.2 * len(ns), 2.8 * len(scales)), squeeze=False,
                             sharey=True)
    summary = {}
    for r in rsms:
        ax = axes[scales.index(r["rate_scale"])][ns.index(r["n"])]
        mcs = {}
        for net in ("net", "nonet"):
            p = base / "mc" / f"{r['id']}-{net}.json"
            if p.exists():
                mcs[net] = json.loads(p.read_text())
        if "net" not in mcs:
            ax.set_title(f"{r['id']}: MC missing", fontsize=8)
            continue
        ref = K.nines(mcs["net"]["availability"])
        names, vals, colors, notes = [], [], [], []
        for rung in ("indep_homog", "indep_hetero", "leaderless_no", "raft_no", "raft_full"):
            p = base / "markov" / f"{r['id']}-{rung}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())
            names.append(RUNG_LABEL[rung])
            if m["status"] != "ok":
                vals.append(0.0)
                colors.append("none")
                notes.append(m["status"])
                continue
            vals.append(K.nines(m["availability"]) - ref)
            colors.append(BLUE)
            notes.append("")
            summary.setdefault(rung, []).append(abs(vals[-1]))
        if "nonet" in mcs:
            names.append(RUNG_LABEL["mc_nonet"])
            vals.append(K.nines(mcs["nonet"]["availability"]) - ref)
            colors.append(AQUA)
            notes.append("")
        x = range(len(vals))
        ax.bar(x, vals, color=colors, edgecolor=[c if c != "none" else MUTED for c in colors], width=0.6)
        for i, (v, note) in enumerate(zip(vals, notes)):
            ax.text(i, v if not note else 0, note or f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.5, color=INK2, rotation=90 if note else 0)
        se = mcs["net"]["se"] / max(1e-300, (1 - mcs["net"]["availability"])) / math.log(10)
        ax.axhspan(-1.96 * se, 1.96 * se, color=GRID, zorder=0)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks(list(x), names, fontsize=6.5, linespacing=0.95)
        ax.set_title(f"{r['rate_scale']:g}x faults, {r['n']}-node RSM (MC {ref:.2f} nines)", fontsize=8.5,
                     color=INK, loc="left")
    for row in axes:
        row[0].set_ylabel("Estimate - MC (nines)")
    fig.suptitle("Modeling stability: availability estimate error vs Monte Carlo (24 h, region/AZ outages)",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, base / "ladder")
    (base / "ladder_summary.json").write_text(json.dumps(
        {k: {"mean_abs_error_nines": sum(v) / len(v), "rsms": len(v)} for k, v in summary.items()}, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------ Fig 11

def sensitivity() -> None:
    """Fig 11. With outputs/figs/sensitivity-new/sensitivity-validated.json (new scoring: re-selection batches +
    independent validation, chosen by validated availability) that curve is the main one and the old scoring is
    drawn faded for comparison; otherwise the old-scoring figure (with any 1x overlay) is drawn."""
    old_path = K.ROOT / "outputs/figs/sensitivity/sensitivity.json"
    new_path = K.ROOT / "outputs/figs/sensitivity-new/sensitivity-validated.json"
    if not new_path.exists():
        return _sensitivity_old(old_path)
    new = json.loads(new_path.read_text())
    old = json.loads(old_path.read_text())
    pts = sorted(new["points"], key=lambda p: p["assumed_multiplier"])
    opts = sorted(old["points"], key=lambda p: p["assumed_multiplier"])
    ms = [p["assumed_multiplier"] for p in pts]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True)
    ax.plot([p["assumed_multiplier"] for p in opts], [p["true_nines"] for p in opts], color=MUTED, marker="o",
            markersize=3, linewidth=1, alpha=0.7, label="True availability, old scoring")
    ax.plot(ms, [p["validated_nines"] if p.get("validated_nines") is not None else p["believed_nines"] for p in pts],
            color=BLUE, linestyle=":", marker="o", markersize=3, linewidth=1, alpha=0.8,
            label="Believed: validated at the assumed rates (new scoring)")
    err = [[p["true_nines"] - p["true_nines_ci95"][0] for p in pts], [p["true_nines_ci95"][1] - p["true_nines"] for p in pts]]
    ax.errorbar(ms, [p["true_nines"] for p in pts], yerr=err, color=BLUE, marker="o", markersize=5, linewidth=2,
                capsize=3, label="True availability, new scoring")
    ax.axhline(new["sla_nines"], color=RED, linestyle="--", linewidth=1)
    ax.text(ms[0], new["sla_nines"], f" SLA {new['sla_nines']:g} nines", color=INK2, va="bottom", fontsize=8)
    bad = [m for m, p in zip(ms, pts) if not p["true_meets_sla"]]
    if bad:
        ax.axvspan(min(bad) / 1.1, max(bad) * 1.1, color=RED, alpha=0.08, linewidth=0)
    for a in (ax, ax2):
        a.axvline(1.0, color=INK, linestyle=":", linewidth=0.8)
    ax.set_ylabel("Availability (nines)")
    ax.set_title("Sensitivity to reliability input error (truth = 1x, rho1 prices)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, loc="lower right")
    ax2.plot([p["assumed_multiplier"] for p in opts], [p["cost_per_hour"] for p in opts], color=MUTED, marker="o",
             markersize=3, linewidth=1, alpha=0.7, label="old scoring")
    ax2.plot(ms, [p["cost_per_hour"] for p in pts], color=ORANGE, marker="o", markersize=5, linewidth=2,
             label="new scoring")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_ylabel("Cluster cost ($/h)")
    ax2.set_xscale("log")
    ax2.set_xticks(ms, [f"{m:g}x" for m in ms])
    ax2.minorticks_off()
    ax2.set_xlabel("Assumed failure rate / true failure rate   (<- optimistic | pessimistic ->)")
    fig.text(0.01, -0.02, "New scoring: 3 fresh 100k-run batches per re-selected placement; deployment chosen by its "
             "1M-run independent validation. Truth: 1M runs/placement on separate seeds.", fontsize=6.5, color=MUTED)
    save(fig, new_path.parent / "sensitivity")


def _sensitivity_old(path) -> None:
    d = json.loads(path.read_text())
    pts = sorted(d["points"], key=lambda p: p["assumed_multiplier"])
    ms = [p["assumed_multiplier"] for p in pts]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 5), sharex=True)
    y = [p["true_nines"] for p in pts]
    err = [[p["true_nines"] - p["true_nines_ci95"][0] for p in pts], [p["true_nines_ci95"][1] - p["true_nines"] for p in pts]]
    ax.errorbar(ms, y, yerr=err, color=BLUE, marker="o", markersize=5, linewidth=2, capsize=3,
                label="True availability (1x rates)")
    ax.plot(ms, [p["believed_nines"] for p in pts], color=MUTED, linestyle=":", marker="o", markersize=3,
            linewidth=1, label="Believed by the optimizer")
    ax.axhline(d["sla_nines"], color=RED, linestyle="--", linewidth=1)
    ax.text(ms[0], d["sla_nines"], f" SLA {d['sla_nines']:g} nines", color=INK2, va="bottom", fontsize=8)
    bad = [m for m, p in zip(ms, pts) if not p["true_meets_sla"]]
    if bad:
        ax.axvspan(min(bad) / 1.1, max(bad) * 1.1, color=RED, alpha=0.08, linewidth=0)
    for a in (ax, ax2):
        a.axvline(1.0, color=INK, linestyle=":", linewidth=0.8)
    ax.set_ylabel("Availability (nines)")
    ax.set_title("Sensitivity to reliability input error (truth = 1x, rho1 prices)", fontsize=9, loc="left", color=INK)
    ax2.plot(ms, [p["cost_per_hour"] for p in pts], color=ORANGE, marker="o", markersize=5, linewidth=2)
    # optional overlay: 1x re-run with the new scoring (re-selection batches + independent validation)
    for fname, marker, label in (("sensitivity-x1-new-believed.json", "D", "1x, new scoring, chosen by believed"),
                                 ("sensitivity-x1-new-validated.json", "*", "1x, new scoring, chosen by validated")):
        q = path.parent / fname
        if not q.exists():
            continue
        for p in json.loads(q.read_text())["points"]:
            off = 1.06 if marker == "*" else 0.94     # nudge so the two overlays and the original do not overlap
            ax.errorbar([p["assumed_multiplier"] * off], [p["true_nines"]],
                        yerr=[[p["true_nines"] - p["true_nines_ci95"][0]], [p["true_nines_ci95"][1] - p["true_nines"]]],
                        color=GREEN, marker=marker, markersize=8 if marker == "*" else 5, linestyle="none",
                        capsize=2, label=label)
            ax2.plot([p["assumed_multiplier"] * off], [p["cost_per_hour"]], color=GREEN, marker=marker,
                     markersize=8 if marker == "*" else 5, linestyle="none")
    ax.legend(fontsize=7, loc="lower right")
    ax2.set_ylabel("Cluster cost ($/h)")
    ax2.set_xscale("log")
    ax2.set_xticks(ms, [f"{m:g}x" for m in ms])
    ax2.minorticks_off()
    ax2.set_xlabel("Assumed failure rate / true failure rate   (<- optimistic | pessimistic ->)")
    save(fig, path.parent / path.name.removesuffix(".json"))


# ------------------------------------------------------------------------------------------------ Fig 10

def scalability() -> None:
    path = K.ROOT / "outputs/figs/scalability/past_runs.json"
    recs = json.loads(path.read_text())["records"]
    groups = {}
    for r in recs:
        if r["cluster_nodes"]:
            groups.setdefault(f"{r['pipeline']}, {r['num_rsms']} RSMs", []).append(r)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for color, (name, rs) in zip((BLUE, ORANGE, AQUA, VIOLET), sorted(groups.items())):
        ax.scatter([r["cluster_nodes"] for r in rs], [r["wall_s"] / 3600 for r in rs], s=22, color=color,
                   edgecolor="white", linewidth=0.8, label=name)
        ax2.scatter([r["cluster_nodes"] for r in rs], [r["optimizer_usd_per_month_daily"] for r in rs], s=22,
                    color=color, edgecolor="white", linewidth=0.8, label=f"optimizer: {name}")
        sv = [r for r in rs if r["savings_usd_per_month"] and r["savings_usd_per_month"] > 0]
        ax2.scatter([r["cluster_nodes"] for r in sv], [r["savings_usd_per_month"] for r in sv], s=22, marker="^",
                    facecolor="none", edgecolor=color, linewidth=1, label=f"savings: {name}")
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("Cluster size (machines used)")
    ax.set_ylabel("Optimizer wall time per solve (h)")
    ax2.set_ylabel("$ per month (daily reconfiguration)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax2.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Optimizer cost vs savings, past runs (wall time on one 8-vCPU VM)", fontsize=9.5, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout()
    save(fig, path.with_name("scalability_past"))


# ------------------------------------------------------------------------------------------------ Fig 8

def reconfig(directory: str) -> None:
    rows = sorted(json.loads((Path(directory) / "summary.json").read_text()), key=lambda r: r["cadence_h"])
    x = [r["cadence_h"] for r in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True)
    ax.plot(x, [r["sla_violation_min_per_month"] for r in rows], color=BLUE, marker="o", markersize=6, linewidth=2,
            label="SLA violation")
    ax.plot(x, [r["downtime_min_per_month"] for r in rows], color=MUTED, marker="o", markersize=3, linewidth=1,
            linestyle=":", label="Expected downtime")
    ax.set_ylabel("Minutes / month / RSM")
    ax.legend(fontsize=8)
    ax.set_title("How often must we reconfigure?", fontsize=9, loc="left", color=INK)
    ax2.plot(x, [max(r["gb_moved_per_day"], 1e-3) for r in rows], color=ORANGE, marker="o", markersize=6, linewidth=2)
    ax2.set_yscale("log")
    ax2.set_ylabel("Reconfig overhead\n(GB moved / day)")
    ax2.set_xscale("log")
    ax2.set_xticks(x, [r["label"] for r in rows])
    ax2.minorticks_off()
    ax2.set_xlabel("Reconfiguration interval")
    save(fig, Path(directory) / "reconfig_cadence")


# ------------------------------------------------------------------------------------------------ election timeout

def _per(rate_per_h: float) -> str:
    return f"{rate_per_h * 24:g}/day" if rate_per_h < 1 else f"{rate_per_h:g}/h"


def election(path: str = str(K.ROOT / "outputs/figs/whatif/election_timeout.json")) -> None:
    d = json.loads(Path(path).read_text())
    T = d["timeouts_s"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    colors = {"1x": BLUE, "30x": ORANGE}
    for tag, dep in d["deployments"].items():
        rs = dep["results"]
        y = [r["nines"] for r in rs]
        err = [[r["nines"] - r["nines_ci95"][0] for r in rs], [r["nines_ci95"][1] - r["nines"] for r in rs]]
        ax.errorbar(T, y, yerr=err, color=colors[tag], marker="o", markersize=5, linewidth=2, capsize=2,
                    label=f"{tag} faults (MC)")
        ax.plot(T, [r["network"]["noisy"]["nines_with_stalls"] for r in rs], color=colors[tag], linestyle="--",
                linewidth=1.2, label=f"{tag} + noisy-network stalls")
        ax2.plot(T, [r["failure_elections_per_rsm_day"] for r in rs], color=colors[tag], marker="o", markersize=5,
                 linewidth=2, label=f"failure-driven, {tag} faults (MC)")
    scen_col = {"calm": AQUA, "typical": VIOLET, "noisy": MAGENTA}
    rs = d["deployments"]["1x"]["results"]
    for name, col in scen_col.items():
        sc = d["scenarios"][name]
        ax2.plot(T, [max(r["network"][name]["spurious_per_day"], 1e-7) for r in rs], color=col, linestyle=":",
                 marker="s", markersize=4, linewidth=1.5,
                 label=f"spurious, {name} (stalls {_per(sc['rate_per_h'])}, Pareto x_m={sc['x_m'] * 1000:g} ms, "
                       f"a={sc['alpha']:g})")
    for a in (ax, ax2):
        a.axvline(10, color=INK, linestyle=":", linewidth=0.8)
    ax.text(10, ax.get_ylim()[1], " TiKV default (10 ticks x 1 s)", fontsize=7, color=INK2, va="top")
    ax.set_ylabel("Availability (nines)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"What-if: Raft election timeout T (downtime Uniform(T, 2T) + {d['vote_round_s']:g} s vote; "
                 f"{d['horizon_h']:g} h, {d['runs'] // 1000}k runs/RSM, 50% budget)", fontsize=8.5, loc="left", color=INK)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xticks(T, [f"{t:g}s" for t in T])
    ax2.minorticks_off()
    ax2.set_ylabel("Leader elections\nper RSM per day")
    ax2.set_xlabel("Election timeout T")
    ax2.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.text(0.01, -0.13, "Stall scenarios are assumptions (pre-vote + check-quorum on; heartbeat T/5). Stall downtime "
             f"counts only waits beyond the {d.get('client_slo_s', 5):g} s client SLO.", fontsize=6.5, color=MUTED)
    save(fig, Path(path).parent / "election_timeout")


def main() -> int:
    cmd, *rest = sys.argv[1:] or ["help"]
    fns = {"whatif": whatif, "ladder": ladder, "sensitivity": sensitivity, "scalability": scalability,
           "reconfig": reconfig, "election": election}
    if cmd not in fns:
        print(__doc__)
        return 2
    fns[cmd](*rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
