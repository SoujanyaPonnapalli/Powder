"""Render four AZ-aware, MC-scored 10,000-RSM frontiers as SVG figures."""
from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#142235", "#66778a", "#e1e0d9", "#fcfcfb"
RF_LABELS = {"rf3": "Uniform RF 3", "rf5": "Uniform RF 5",
             "rf7": "Uniform RF 7", "choice": "ILP chooses RF 3 / 5 / 7"}


def label(x, y, value, size=12, *, color=MUTED, anchor="start", weight=400):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(str(value))}</text>')


def nines(a):
    if not 0 <= a < 1:
        raise ValueError("availability at 1.0 is unresolved; cannot plot finite nines")
    return -math.log10(1 - a)


def glyph(x, y, policy, rf, detail, *, selected=False):
    color = BLUE if selected else ORANGE if policy == "blind" else AQUA
    title = f'<title>{escape(detail)}</title>'
    shape = rf if not selected else "target"
    if shape == "target":
        mark = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{SURFACE}" stroke="{color}" stroke-width="2.5"/>'
    elif shape == 3:
        mark = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
    elif shape == 5:
        mark = f'<rect x="{x-6:.2f}" y="{y-6:.2f}" width="12" height="12" rx="2" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
    else:
        mark = f'<path d="M{x:.2f},{y-7:.2f}L{x+7:.2f},{y+6:.2f}H{x-7:.2f}Z" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
    return f'<g>{title}{mark}</g>'


def usable(point):
    return point.get("status") == "0" and 0 <= point.get("mean_availability", -1) < 1


def oneshot_marks(rows, x, y, top, ymax):
    """Each simulated day of each deployment, at that deployment's price.

    Days above the axis (including days with no downtime) are pinned at the
    top. With many days the dots are faint and jittered by up to ±0.6 $/h so
    their density shows; a bar marks the mean of the days.
    """
    parts = []
    for row in rows:
        days = row["oneshot_availability"]
        many = len(days) > 50
        cx = x(row["cost_per_hour"])
        spread = (x(row["cost_per_hour"] + .6) - cx) if many else 0
        for trial, a in enumerate(days):
            xx = cx + spread * (((trial * 0.6180339887) % 1) * 2 - 1)
            if a >= 1 or nines(a) > ymax:
                detail = f'24 h day {trial} · ${row["cost_per_hour"]:.3f}/h · ' + (
                    "no RSM downtime" if a >= 1 else f"{nines(a):.3f} nines (above axis)")
                mark = (f'<path d="M{xx:.2f},{top + 2:.2f}L{xx + 4:.2f},{top + 9:.2f}H{xx - 4:.2f}Z" '
                        f'fill="none" stroke="{BLUE}" stroke-width="1.2"/>') if not many else (
                        f'<circle cx="{xx:.2f}" cy="{top + 5:.2f}" r="2" fill="{BLUE}" fill-opacity=".12"/>')
            else:
                detail = f'24 h day {trial} · ${row["cost_per_hour"]:.3f}/h · {nines(a):.3f} nines'
                mark = (f'<circle cx="{xx:.2f}" cy="{y(a):.2f}" r="2" fill="{BLUE}" fill-opacity=".12"/>'
                        if many else
                        f'<circle cx="{xx:.2f}" cy="{y(a):.2f}" r="3" fill="{BLUE}" fill-opacity=".15" '
                        f'stroke="{BLUE}" stroke-width="1.2"/>')
            parts.append(f'<g><title>{escape(detail)}</title>{mark}</g>' if not many else mark)
        if many:
            mean = sum(days) / len(days)
            if mean < 1 and nines(mean) <= ymax:
                detail = (f'Mean of {len(days)} 24 h days · ${row["cost_per_hour"]:.3f}/h · '
                          f'{nines(mean):.3f} nines')
                parts.append(f'<g><title>{escape(detail)}</title><path d="M{cx - 9:.2f},{y(mean):.2f}H{cx + 9:.2f}" '
                             f'stroke="{INK}" stroke-width="2.5"/></g>')
    return parts


def draw_panel(arm, name, rf, *, offset, xmin, xmax, ymin, ymax, xstep, ystep, target, oneshot=None):
    data = arm["frontiers"][name]
    baselines = arm["baselines"]
    left, right, top, bottom = offset + 57, offset + 533, 225, 552
    x = lambda cost: left + (cost - xmin) / (xmax - xmin) * (right - left)
    y = lambda a: top + (ymax - nines(a)) / (ymax - ymin) * (bottom - top)
    parts = []
    for i in range(round((ymax - ymin) / ystep) + 1):
        value = ymin + i * ystep
        yy = top + (ymax - value) / (ymax - ymin) * (bottom - top)
        parts.append(f'<path d="M{left},{yy:.2f}H{right}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(label(left - 8, yy + 4, f"{value:.2f}", 11, anchor="end"))
    for value in range(xmin, xmax + 1, xstep):
        xx = x(value)
        parts.append(f'<path d="M{xx:.2f},{top}V{bottom}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(label(xx, bottom + 20, value, 11, anchor="middle"))
    if ymin <= target <= ymax:
        yy = top + (ymax - target) / (ymax - ymin) * (bottom - top)
        parts.append(f'<path d="M{left},{yy:.2f}H{right}" stroke="#8b979d" stroke-width="1" stroke-dasharray="5 4"/>')
        parts.append(label(right - 3, yy - 6, "4.5-nines target", 11, anchor="end"))
    parts.append(label(left, top - 17, "Mean RSM availability (nines)", 11))
    if oneshot:
        parts.extend(oneshot_marks(oneshot, x, y, top, ymax))
    points = [p for p in data.get("points", []) if usable(p)]
    for i in range(1, len(points)):
        p, q = points[i-1], points[i]
        dashed = 'stroke-dasharray="6 4"' if (p.get("certified_rsms", 0) < 10000 or q.get("certified_rsms", 0) < 10000) else ''
        parts.append(f'<path d="M{x(p["cost_per_hour"]):.2f},{y(p["mean_availability"]):.2f} '
                     f'L{x(q["cost_per_hour"]):.2f},{y(q["mean_availability"]):.2f}" '
                     f'stroke="{BLUE}" stroke-width="2" fill="none" {dashed}/>')
    for p in points:
        xx, yy = x(p["cost_per_hour"]), y(p["mean_availability"])
        if "mean_ci_low" in p and "mean_ci_high" in p and p["mean_ci_high"] < 1:
            lo, hi = y(p["mean_ci_low"]), y(p["mean_ci_high"])
            parts.append(f'<path d="M{xx:.2f},{lo:.2f}V{hi:.2f}" stroke="{BLUE}" stroke-width="1"/>')
        mix = p.get("rf_counts", {})
        detail = (f'ILP ${p["cost_per_hour"]:.3f}/h · {nines(p["mean_availability"]):.5f} nines · '
                  f'{p.get("certified_rsms", 0)}/10000 RSMs certified · RF mix {mix}')
        parts.append(f'<g><title>{escape(detail)}</title><circle cx="{xx:.2f}" cy="{yy:.2f}" r="4" fill="{BLUE}" stroke="{SURFACE}" stroke-width="1.5"/></g>')
    for fixed in (3, 5, 7) if name == "choice" else (rf,):
        for policy in ("blind", "cheap"):
            p = baselines[fixed][policy]
            if p["status"] != "scored" or p["mean_availability"] >= 1:
                continue
            detail = (f'RF{fixed} {policy} baseline · ${p["cost_per_hour"]:.3f}/h · '
                      f'{nines(p["mean_availability"]):.5f} nines · '
                      f'{p["certified_groups"]}/20 groups certified')
            parts.append(glyph(x(p["cost_per_hour"]), y(p["mean_availability"]), policy, fixed, detail))
    qualified = data.get("target_qualified", {})
    if usable(qualified):
        detail = (f'ILP target-qualified · ${qualified["cost_per_hour"]:.3f}/h · '
                  f'{nines(qualified["mean_availability"]):.5f} nines · '
                  f'RF counts {qualified.get("rf_counts", {})}')
        parts.append(glyph(x(qualified["cost_per_hour"]), y(qualified["mean_availability"]),
                           "ilp", rf, detail, selected=True))
        parts.append(label(x(qualified["cost_per_hour"]) + 9,
                           y(qualified["mean_availability"]) + 24, "ILP target", 11))
    else:
        parts.append(label(left, bottom - 13, "No MC-certified placement at 4.5 nines", 11))
    parts.append(label((left + right) / 2, bottom + 46, "Assigned-model cost ($/hour)", 12,
                       anchor="middle"))
    return "".join(parts)


def render(result, name, oneshot=None):
    target = float(result["target_nines"])
    # @v2 (weekly planned cadence, removed 2026-11-17) only in saved results; new runs use @v3.
    if result["scenario"] not in ("scenario/canonical-central-az-aware@v2", "scenario/canonical-central-az-aware@v3"):
        raise ValueError("expected AZ-aware sensitivity scenario")
    arms = {price: {**arm, "baselines": {int(rf): data for rf, data in arm["baselines"].items()}}
            for price, arm in result["arms"].items()}
    relevant = []
    rf = int(name[-1]) if name.startswith("rf") else None
    for arm in arms.values():
        data = arm["frontiers"][name]
        relevant.extend(p for p in data.get("points", []) if usable(p))
        relevant += [p for size in ((3, 5, 7) if rf is None else (rf,))
                     for p in arm["baselines"][size].values() if p["status"] == "scored"]
        if usable(data.get("target_qualified", {})):
            relevant.append(data["target_qualified"])
    if not relevant:
        raise ValueError("no scored points to plot")
    costs = [p["cost_per_hour"] for p in relevant]
    nines_values = [nines(p[key]) for p in relevant
                    for key in ("mean_availability", "mean_ci_low", "mean_ci_high")
                    if key in p and 0 <= p[key] < 1]
    if oneshot is not None:
        if oneshot["scenario"] != result["scenario"] or oneshot["frontier"] != name:
            raise ValueError("one-shot replays belong to a different figure")
        days = sorted(nines(a) for rows in oneshot["arms"].values() for row in rows
                      for a in row["oneshot_availability"] if a < 1)
        nines_values += days[:max(1, len(days) * 99 // 100)]  # top 1% of days pin to the axis
    xstep = 5 if max(costs) - min(costs) < 35 else 10
    xmin = xstep * math.floor((min(costs) - .5) / xstep)
    xmax = xstep * math.ceil((max(costs) + .5) / xstep)
    spread = max(nines_values) - min(nines_values)
    ystep = .05 if spread < .25 else .1 if spread < 1 else .25
    ymin = round(ystep * math.floor((min(min(nines_values), target) - .005) / ystep), 6)
    ymax = round(ystep * math.ceil((max(nines_values) + .005) / ystep), 6)
    extra = 0 if oneshot is None else 26
    height = (865 if name == "choice" else 685) + extra
    subtitle = ("MC ≥ 100,000 runs/candidate; dashed ILP segments contain uncertified RSMs"
                if oneshot is None else
                f"Line: 30-day MC mean · dots: {oneshot['trials']} simulated {oneshot['horizon_h']:g} h days of each "
                "ILP deployment, ±0.6 $/h jitter (RSM-weighted; top edge = above axis or no downtime) · "
                "black bar: mean of days"
                if oneshot["trials"] > 50 else
                f"Line: 30-day MC mean · dots: {oneshot['trials']} simulated {oneshot['horizon_h']:g} h days of each "
                "ILP deployment (RSM-weighted one-shot availability; △ at top = no downtime)")
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
        f'<rect width="1200" height="{height}" fill="{SURFACE}"/>',
        label(27, 43, f"{RF_LABELS[name]} · 10,000 RSMs", 27, color=INK, weight=700),
        label(27, 70, "AZ-aware sensitivity · 4 regions × 3 zones · 500 RSM replicas/machine · seed 0", 13),
        label(27, 95, subtitle, 12),
        f'<path d="M27,121H57" stroke="{BLUE}" stroke-width="2"/>',
        label(67, 125, "ILP frontier", 12),
        f'<circle cx="225" cy="121" r="5" fill="{ORANGE}"/>',
        label(238, 125, "Price-blind", 12),
        f'<circle cx="385" cy="121" r="5" fill="{AQUA}"/>',
        label(398, 125, "Cheapest-first", 12),
        label(610, 125, "Baseline shapes: ○ RF 3 · □ RF 5 · △ RF 7", 12),
        label(75, 161, "Independent price assignment (ρ = 0)", 18, color=INK, weight=700),
        label(665, 161, "Reliability-ranked price (ρ = 1)", 18, color=INK, weight=700),
    ]
    for offset, price in ((27, "rho0"), (617, "rho1")):
        header.append(draw_panel(arms[price], name, rf, offset=offset,
                                 xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                 xstep=xstep, ystep=ystep, target=target,
                                 oneshot=None if oneshot is None else oneshot["arms"][price]))
    footer = (845 if name == "choice" else 665) + extra
    if name == "choice":
        header.append(label(27, 628, "Actual ILP replica choices at each budget (no mix imposed)", 15, color=INK, weight=700))
        for offset, price in ((27, "rho0"), (617, "rho1")):
            header.append(label(offset + 32, 652, "budget $/h", 11))
            for dx, heading in ((154, "active"), (218, "RF3"), (284, "RF5"), (350, "RF7"), (414, "certified")):
                header.append(label(offset + dx, 652, heading, 11))
            for index, point in enumerate(arms[price]["frontiers"][name].get("points", [])):
                yy = 677 + index * 26
                header.append(label(offset + 32, yy,
                                    f"{point.get('budget_per_hour', point.get('cost_per_hour', 0)):.2f}", 12))
                if not usable(point):
                    header.append(label(offset + 154, yy, "infeasible", 12))
                    continue
                counts = point.get("rf_counts", {})
                header.append(label(offset + 154, yy, point["active_machines"], 12))
                for dx, rf_count in ((218, 3), (284, 5), (350, 7)):
                    header.append(label(offset + dx, yy, counts.get(str(rf_count), counts.get(rf_count, 0)), 12))
                header.append(label(offset + 414, yy, point.get("certified_rsms", 0), 12))
    header.append(label(27, footer - 22, "Blue = ILP · orange = price-blind · green = cheapest-first · shapes encode baseline RF 3/5/7", 12))
    if oneshot is not None:
        header.append(label(27, footer - 44, "24 h days share AZ/region outage draws across deployments; "
                            "a machine set sees the same day wherever it is placed.", 12))
    header.append(label(27, footer, "Whole-deployment uptime is not shown. Partial WAN impact remains a stress-model gap; assigned costs are not AWS bills.", 12))
    header.append('</svg>')
    return "\n".join(header)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("outputs/exec-02/az-frontiers.json"))
    ap.add_argument("--oneshot", type=Path, default=None,
                    help="24 h replay JSON; also writes az-frontier-<name>-24h.svg")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/exec-02"))
    args = ap.parse_args()
    result = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in RF_LABELS:
        dest = args.output_dir / f"az-frontier-{name}.svg"
        dest.write_text(render(result, name))
        print(dest)
    if args.oneshot is not None:
        oneshot = json.loads(args.oneshot.read_text())
        name = oneshot["frontier"]
        dest = args.output_dir / f"az-frontier-{name}-24h.svg"
        dest.write_text(render(result, name, oneshot))
        print(dest)


if __name__ == "__main__":
    main()
