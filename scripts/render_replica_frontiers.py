"""Render the six-panel RF=3/5/7 cost-frontier comparison as a static SVG.

The 3- and 5-replica rows use exact machine-rate Markov scores. The 7-replica
row uses rate-binned screening scores and is drawn dashed; its error probes
are reported in the row's source file, not silently treated as exact.
"""
from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

DEFAULT_FILES = {
    3: "outputs/exec-02/rsm10000-ford-frontier.jsonl",
    5: "outputs/exec-02/rsm10000-rf5-frontier.jsonl",
    7: "outputs/exec-02/rsm10000-rf7-binned.jsonl",
}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#142235", "#66778a", "#e1e0d9", "#fcfcfb"


def availability(entry):
    return entry.get("mean_availability_exact", entry.get("mean_availability_model",
                       entry.get("mean_availability")))


def nines(entry):
    return -math.log10(1 - availability(entry))


def load_rows(paths):
    result = {}
    for replicas, path in paths.items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if {row["price"] for row in rows} != {"rho0", "rho1"}:
            raise ValueError(f"need both pricing assignments for RF{replicas}")
        for row in rows:
            placements = row["blind"]["placements"]
            if len(placements) != 10000 or any(len(ids) != replicas for ids in placements):
                raise ValueError(f"wrong workload in {path}")
            if row["blind"]["deployed"] != 20 * replicas:
                raise ValueError(f"wrong fleet size in {path}")
            if any(p["status"] != "0" for p in row["frontier"]):
                raise ValueError(f"frontier has unfinished solve in {path}")
            if not math.isclose(row["frontier"][0]["budget_per_hour"],
                                row["cheapest_first"]["cost_per_hour"], abs_tol=1e-8):
                raise ValueError(f"frontier does not reach cheapest-first price in {path}")
            result[(replicas, row["price"])] = row
    return result


def line(x1, y1, x2, y2, *, stroke=GRID, width=1, dash=""):
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" {f"stroke-dasharray={dash!r}" if dash else ""}/>')


def text(x, y, value, *, size=13, color=MUTED, anchor="start", weight=400):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
            f'font-family="DejaVu Sans, Arial, sans-serif" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(str(value))}</text>')


def marker(x, y, label, cost, avail, *, kind="circle", color=BLUE):
    tip = escape(f"{label}: ${cost:.3f}/h; availability {avail:.10f}; {-math.log10(1-avail):.5f} nines")
    if kind == "hollow":
        shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{SURFACE}" stroke="{color}" stroke-width="2.5"/>'
    elif kind == "diamond":
        shape = f'<path d="M{x:.2f} {y-7:.2f}L{x+7:.2f} {y:.2f}L{x:.2f} {y+7:.2f}L{x-7:.2f} {y:.2f}Z" fill="{color}" stroke="{SURFACE}" stroke-width="1.6"/>'
    else:
        shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
    return f'<g><title>{tip}</title>{shape}</g>'


def plot(row, *, x0, y0, x_min, x_max, y_min, y_max, x_step, y_step, approximate):
    left, right = x0 + 57, x0 + 530
    top, bottom = y0 + 56, y0 + 263
    xf = lambda v: left + (v - x_min) / (x_max - x_min) * (right - left)
    yf = lambda a: top + (y_max - (-math.log10(1 - a))) / (y_max - y_min) * (bottom - top)
    parts = []
    for i in range(round((y_max - y_min) / y_step) + 1):
        v = y_min + i * y_step
        yy = top + (y_max - v) / (y_max - y_min) * (bottom - top)
        parts.append(line(left, yy, right, yy))
        parts.append(text(left - 9, yy + 4, f"{v:.2f}", size=11, anchor="end"))
    for v in range(x_min, x_max + 1, x_step):
        xx = xf(v)
        parts.append(line(xx, top, xx, bottom))
        parts.append(text(xx, bottom + 20, v, size=11, anchor="middle"))
    parts.append(line(left, bottom, right, bottom, stroke="#9aa9b8"))
    parts.append(text(left, top - 16, "Mean RSM availability (nines)", size=11))
    points = [(p["cost_per_hour"], availability(p)) for p in row["frontier"]]
    coords = [(xf(c), yf(a)) for c, a in points]
    path = " ".join(f'{"M" if i == 0 else "L"}{x:.2f},{y:.2f}' for i, (x, y) in enumerate(coords))
    dashed = 'stroke-dasharray="6 4"' if approximate else ""
    parts.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="2" stroke-linecap="round" {dashed}/>')
    for x, y in coords:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{BLUE}" stroke="{SURFACE}" stroke-width="1.5"/>')
    for key, label, kind, color in (
        ("cheapest_first", "Cheapest-first", "circle", AQUA),
        ("max_availability_at_blind_cost", "ILP under blind budget + floor", "diamond", BLUE),
        ("blind", "Price-blind", "circle", ORANGE),
        ("min_cost_at_blind_target", "ILP min cost at blind target", "hollow", BLUE),
    ):
        item = row[key]
        parts.append(marker(xf(item["cost_per_hour"]), yf(availability(item)), label,
                            item["cost_per_hour"], availability(item), kind=kind, color=color))
    parts.append(text(xf(row["blind"]["cost_per_hour"]) - 9,
                      yf(availability(row["blind"])) + 18, "Blind", size=11, anchor="end"))
    parts.append(text(xf(row["min_cost_at_blind_target"]["cost_per_hour"]) + 9,
                      yf(availability(row["min_cost_at_blind_target"])) + 25, "ILP target", size=11))
    if abs(xf(row["cheapest_first"]["cost_per_hour"]) -
           xf(row["min_cost_at_blind_target"]["cost_per_hour"])) > 20:
        parts.append(text(xf(row["cheapest_first"]["cost_per_hour"]) + 9,
                          yf(availability(row["cheapest_first"])) + 23, "Cheapest", size=11))
    parts.append(text((left + right) / 2, bottom + 43, "Modeled cost ($/hour)", size=12, anchor="middle"))
    return "".join(parts)


def render(rows):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1330" viewBox="0 0 1280 1330">',
             f'<rect width="1280" height="1330" fill="{SURFACE}"/>',
             text(36, 47, "DB placement · 10,000 RSMs", size=30, color=INK, weight=700),
             text(36, 76, "Separate replica-count workloads  ·  500 replicas/machine  ·  current composed fault catalogue  ·  seed 0", size=14),
             line(36, 102, 61, 102, stroke=BLUE, width=2),
             marker(49, 102, "ILP target", 0, .99, kind="hollow"),
             text(71, 106, "ILP frontier / target", size=12),
             marker(324, 102, "Price-blind", 0, .99, color=ORANGE),
             text(338, 106, "Price-blind", size=12),
             marker(491, 102, "Cheapest-first", 0, .99, color=AQUA),
             text(505, 106, "Cheapest-first", size=12),
             marker(693, 102, "ILP at blind budget", 0, .99, kind="diamond"),
             text(709, 106, "ILP at blind budget + floor", size=12),
             text(75, 155, "Independent price assignment (ρ = 0)", size=18, color=INK, weight=700),
             text(665, 155, "Reliability-ranked price (ρ = 1)", size=18, color=INK, weight=700)]
    for idx, replicas in enumerate((3, 5, 7)):
        left_row, right_row = rows[(replicas, "rho0")], rows[(replicas, "rho1")]
        panel = [left_row, right_row]
        all_costs = [item["cost_per_hour"] for row in panel for item in
                     [row["blind"], row["cheapest_first"], row["min_cost_at_blind_target"],
                      row["max_availability_at_blind_cost"], *row["frontier"]]]
        all_nines = [nines(item) for row in panel for item in
                     [row["blind"], row["cheapest_first"], row["min_cost_at_blind_target"],
                      row["max_availability_at_blind_cost"], *row["frontier"]]]
        x_step = 5 if max(all_costs) - min(all_costs) < 35 else 10
        x_min = x_step * math.floor((min(all_costs) - .5) / x_step)
        x_max = x_step * math.ceil((max(all_costs) + .5) / x_step)
        y_step = .05 if max(all_nines) - min(all_nines) < .20 else .1
        y_min = round(y_step * math.floor((min(all_nines) - .005) / y_step), 6)
        y_max = round(y_step * math.ceil((max(all_nines) + .005) / y_step), 6)
        y = 185 + idx * 365
        approximate = replicas == 7
        label = f"RF {replicas}  ·  {20 * replicas} active machines"
        if approximate:
            label += "  ·  binned Markov screening (dashed)"
        parts.append(text(36, y, label, size=16, color=INK, weight=700))
        for col, row in enumerate(panel):
            parts.append(plot(row, x0=36 + col * 590, y0=y + 15, x_min=x_min, x_max=x_max,
                              y_min=y_min, y_max=y_max, x_step=x_step, y_step=y_step,
                              approximate=approximate))
    parts.append(text(36, 1288, "Axes match within each RF row, but rescale across rows; dashed RF7 is a binned screening estimate.", size=12))
    parts.append(text(36, 1309, "Model prices, not AWS bills · mean RSM availability, not whole-deployment uptime · no measured TiDB or oracle point", size=12))
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for replicas, default in DEFAULT_FILES.items():
        ap.add_argument(f"--rf{replicas}", default=default)
    ap.add_argument("--output", default="outputs/exec-02/placement-frontier-rf357.svg")
    args = ap.parse_args()
    rows = load_rows({k: getattr(args, f"rf{k}") for k in DEFAULT_FILES})
    Path(args.output).write_text(render(rows))
    print(args.output)


if __name__ == "__main__":
    main()
