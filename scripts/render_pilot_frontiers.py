"""Render the rate-scale pilot as cost vs availability frontiers (one panel per fault scale x price model).

    python scripts/render_pilot_frontiers.py [--input outputs/pilot-ilp-rate-scale/results-fleet.json]
    rsvg-convert --zoom 2 outputs/pilot-ilp-rate-scale/frontiers.svg -o outputs/pilot-ilp-rate-scale/frontiers.png

Reuses the pilot's stored values; nothing is re-solved. The ILP line joins its feasible budget points
(fractions of the unrestricted cost) and the unrestricted point. Exact values are in each marker's <title>.
For ``results-fleet.json`` (scripts/pilot_fleet_frontier.py) the blue line is the budget-targeted fleet ILP
and the grey line the old random-menu ILP, all exact MC. For the older Markov ``results.json`` the dashed
grey step is the Pareto envelope over every plotted point.
"""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = "DejaVu Sans,Arial,sans-serif"
PRICES = {"rho0": "Independent prices", "rho1": "Reliability-ranked prices"}
PW, PH = 250, 210                       # plot area per panel
ML, MT, GX, GY = 92, 118, 62, 74         # left/top margins, gaps between panels


def text(x, y, s, size=11, color=INK2, anchor="start", weight=400, rotate=None):
    rot = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-family="{FONT}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{rot}>{escape(str(s))}</text>')


def marker(x, y, shape, color, title):
    t = f"<title>{escape(title)}</title>"
    ring = f'stroke="{SURFACE}" stroke-width="2"'
    if shape == "hollow":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{SURFACE}" stroke="{color}" stroke-width="1.8">{t}</circle>'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" {ring}>{t}</circle>'
    if shape == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="1.5" fill="{color}" {ring}>{t}</rect>'
    return (f'<path d="M{x:.1f},{y - 6.5:.1f} L{x + 6:.1f},{y + 4.5:.1f} L{x - 6:.1f},{y + 4.5:.1f} Z" '
            f'fill="{color}" {ring}>{t}</path>')


def nice_ticks(lo, hi, n=5):
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    step *= min((m for m in (1, 2, 2.5, 5, 10) if (hi - lo) / (step * m) <= n), default=10)
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 6) for i in range(int((hi - start) / step + 1e-9) + 1)]


def points(row):
    if "new_ilp" in row:  # fleet pilot: exact MC for every point
        pts = [(kind, "no budget" if f == "1" else f"{float(f):.0%} budget", v)
               for kind, key in (("ILP", "new_ilp"), ("old", "old_ilp"), ("prev", "prev_ilp"))
               for f, v in row.get(key, {}).items() if v["feasible"]]
        for name, b in row["baselines"].items():
            rf, policy = name.split("-")
            pts.append((policy, rf.upper().replace("RF", "RF "), b))
        return [(kind, label, p["cost_per_hour"], p["nines"]) for kind, label, p in pts]
    pts = [("ILP", f"{float(key[7:]):.0%} budget", v) for key, v in row.items()
           if key.startswith("budget_") and key != "budget_fractions" and v.get("feasible", True)]
    pts.append(("ILP", "no budget", row["unrestricted"]))
    for name, b in row["baselines"].items():
        rf, policy = name.split("-")
        pts.append((policy, rf.upper().replace("RF", "RF "), b))
    return [(kind, label, p["cost_per_hour"], p["nines"]) for kind, label, p in pts]


def pareto(pts):
    best, out = -math.inf, []
    for c, a in sorted((c, a) for _, _, c, a in pts):
        if a > best:
            out.append((c, a))
            best = a
    return out


def render(data, compare=None, show_main=False) -> str:
    rows = data["rows"]
    fleet = "new_ilp" in rows[0]
    prev_label = "Fleet ILP, previous sweep"
    if compare:  # overlay an earlier fleet sweep of the same configurations
        prev = {(r["rate_scale"], r["price"]): r["new_ilp"] for r in compare["rows"]}
        rows = [{**r, "prev_ilp": prev[(r["rate_scale"], r["price"])]} for r in rows
                if (r["rate_scale"], r["price"]) in prev]
    elif show_main:  # the same run's main ILP, before lattice ascent
        rows = [{**r, "prev_ilp": {f: v["main_ilp"] for f, v in r["new_ilp"].items() if "main_ilp" in v}}
                for r in rows]
        prev_label = "Fleet ILP"
    compare = compare or show_main
    scales = sorted({r["rate_scale"] for r in rows})
    prices = [p for p in PRICES if any(r["price"] == p for r in rows)]
    by = {(r["rate_scale"], r["price"]): r for r in rows}
    W = ML + len(scales) * PW + (len(scales) - 1) * GX + 24
    W = max(W, ML + 1000)  # room for the title, subtitle, legend and footnote when there are few columns
    H = MT + len(prices) * PH + (len(prices) - 1) * GY + 78
    xmax = max(c for r in rows for _, _, c, _ in points(r)) * 1.08
    xt = nice_ticks(0, xmax, 4)
    if xt[-1] < xmax:
        xt.append(round(2 * xt[-1] - xt[-2], 6))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           text(ML, 28, "Cost vs. availability under scaled fault rates (50 RSMs, 2000-VM catalogue)", 16, INK,
                weight=600),
           text(ML, 48, ("Exact Monte Carlo per-RSM availability (region/AZ outages on); budget-targeted fleet ILP "
                 "vs. random-menu ILP and fixed-RF baselines") if fleet else
                ("Expected per-RSM availability from the Markov model (no region/AZ outages); "
                 "ILP at several budgets vs. fixed-RF baselines"), 11, INK2)]
    # legend
    lx = ML
    items = ((("circle", BLUE, ("Fleet ILP + lattice ascent" if show_main else "Fleet ILP + lattice ascent")
                if compare else "Fleet ILP across budgets"),)
             + ((("hollow", BLUE, prev_label),) if compare else ())
             + (("hollow", MUTED, "Random-menu ILP"),) if fleet else (("circle", BLUE, "ILP across budgets"),))
    for shape, color, label in items + (("square", ORANGE, "Price-blind baseline"),
                                        ("triangle", AQUA, "Cheapest-first baseline")):
        out.append(marker(lx + 6, 68, shape, color, label))
        out.append(text(lx + 18, 72, label, 11, INK2))
        lx += 34 + 6.9 * len(label)
    if not fleet:
        out.append(f'<line x1="{lx}" y1="68" x2="{lx + 22}" y2="68" stroke="{MUTED}" stroke-width="1.5" '
                   f'stroke-dasharray="4 3"/>')
        out.append(text(lx + 28, 72, "Pareto envelope of shown points", 11, INK2))
    for j, k in enumerate(scales):
        col = [p for pr in prices for p in points(by[(k, pr)])]
        lo, hi = min(a for *_, a in col), max(a for *_, a in col)
        pad = (hi - lo) * 0.12
        yt = nice_ticks(lo - pad, hi + pad, 5)
        ylo, yhi = min(yt[0], lo - pad), max(yt[-1], hi + pad)
        for i, pr in enumerate(prices):
            x0, y0 = ML + j * (PW + GX), MT + i * (PH + GY)
            sx = lambda c: x0 + c / xt[-1] * PW  # noqa: E731
            sy = lambda a: y0 + PH - (a - ylo) / (yhi - ylo) * PH  # noqa: E731
            if i == 0:
                out.append(text(x0 + PW / 2, y0 - 12, f"{k:g}× fault rates", 13, INK, "middle", 600))
            if j == 0:
                out.append(text(x0 - 58, y0 + PH / 2, PRICES[pr], 12, INK, "middle", 600, rotate=-90))
            for v in yt:
                out.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + PW}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
                out.append(text(x0 - 6, sy(v) + 4, f"{v:.2f}".rstrip("0").rstrip("."), 10, MUTED, "end"))
            for v in xt:
                out.append(f'<line x1="{sx(v):.1f}" y1="{y0}" x2="{sx(v):.1f}" y2="{y0 + PH}" stroke="{GRID}"/>')
                out.append(text(sx(v), y0 + PH + 15, f"${v:g}", 10, MUTED, "middle"))
            out.append(f'<line x1="{x0}" y1="{y0 + PH}" x2="{x0 + PW}" y2="{y0 + PH}" stroke="{AXIS}"/>')
            out.append(text(x0 + PW / 2, y0 + PH + 32, "Cluster cost ($/h)", 10, INK2, "middle"))
            if j == 0:
                out.append(text(x0 - 34, y0 + PH / 2, "Availability (nines)", 10, INK2, "middle", rotate=-90))
            pts = points(by[(k, pr)])
            if fleet:
                old = sorted((c, a) for kind, _, c, a in pts if kind == "old")
                out.append(f'<polyline points="{" ".join(f"{sx(c):.1f},{sy(a):.1f}" for c, a in old)}" fill="none" '
                           f'stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 3"/>')
                prv = sorted((c, a) for kind, _, c, a in pts if kind == "prev")
                if prv:
                    out.append(f'<polyline points="{" ".join(f"{sx(c):.1f},{sy(a):.1f}" for c, a in prv)}" '
                               f'fill="none" stroke="{BLUE}" stroke-width="1.2" stroke-dasharray="2 3"/>')
            else:
                env = pareto(pts)
                d = f"M{sx(env[0][0]):.1f},{sy(env[0][1]):.1f}"
                for (c0, a0), (c1, a1) in zip(env, env[1:]):
                    d += f" H{sx(c1):.1f} V{sy(a1):.1f}"
                out.append(f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 3"/>')
            ilp = sorted((c, a) for kind, _, c, a in pts if kind == "ILP")
            out.append(f'<polyline points="{" ".join(f"{sx(c):.1f},{sy(a):.1f}" for c, a in ilp)}" fill="none" '
                       f'stroke="{BLUE}" stroke-width="2"/>')
            for kind, label, c, a in sorted(pts, key=lambda p: p[0] == "ILP"):  # fleet ILP drawn on top
                shape, color = {"ILP": ("circle", BLUE), "old": ("hollow", MUTED), "prev": ("hollow", BLUE),
                                "blind": ("square", ORANGE), "cheap": ("triangle", AQUA)}[kind]
                name = {"ILP": ("Lattice" if compare else "Fleet ILP") if fleet else "ILP", "old": "Random-menu ILP",
                        "prev": prev_label,
                        "blind": "Price-blind", "cheap": "Cheapest-first"}[kind]
                out.append(marker(sx(c), sy(a), shape, color, f"{name} {label}: {a:.2f} nines, ${c:.2f}/h"))
                if kind in ("blind", "cheap"):
                    dy = 16 if kind == "blind" else -7  # blind markers sit just under the ILP line
                    out.append(text(sx(c) + 7, sy(a) + dy, label.replace("RF ", ""), 10, INK2))
    out.append(text(ML, H - 12, "Numbers beside baseline markers are the replication factor (3/5/7). " +
                    ("Budgets: 10-100% of the random-menu ILP's unrestricted cost." if fleet else
                     "Markov screening scores; RF 7 uses binned rates."), 10, MUTED))
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/pilot-ilp-rate-scale/results-fleet.json")
    ap.add_argument("--output", default="outputs/pilot-ilp-rate-scale/frontiers.svg")
    ap.add_argument("--compare", default=None, help="earlier results-fleet.json to overlay (dotted blue)")
    ap.add_argument("--show-main", action="store_true",
                    help="overlay this run's main ILP before lattice ascent (dotted blue)")
    a = ap.parse_args()
    data = json.loads((ROOT / a.input).read_text())
    compare = json.loads((ROOT / a.compare).read_text()) if a.compare else None
    (ROOT / a.output).write_text(render(data, compare, a.show_main) + "\n")
    print(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
