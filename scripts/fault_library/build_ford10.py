"""Digitize Ford et al., OSDI'10 and build the ford10 library entries.

    python scripts/fault_library/build_ford10.py

Inputs:  docs/reading/Ford_OSDI10.pdf (all plots are vector graphics)
Outputs:
  data/fault_library/digitized/ford10/
      fig1_all_causes.csv, fig2_<cause>.csv     (jumps: x_s, F_before, F_after)
      fig3_daily.csv                             (events/1000 nodes/day by cause)
      calibration.json                           (axis fits + residuals)
      raster_check.json                          (independent raster pass)
      fig1_overlay.png, fig2_overlay.png, fig3_overlay.png
  data/fault_library/sources/ford10.json
  data/fault_library/profiles/node/ford10.json
  data/fault_library/profiles/component/ford10-disk.json
  data/fault_library/profiles/planned_downtime/ford10.json

Deterministic: rerunning produces identical JSON.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pdfplumber  # noqa: E402

from digitize_figure import (  # noqa: E402
    AxisMap, StepCurve, fit_axis, raster_frame, raster_series_top, raster_ticks_x,
    render_page, step_curve_from_pts, tick_positions,
)
from powder.faults.fit import best_cdf_fit  # noqa: E402

PDF = ROOT / "docs/reading/Ford_OSDI10.pdf"
OUT = ROOT / "data/fault_library/digitized/ford10"
LIB = ROOT / "data/fault_library"
DPI = 600
H = 3600.0
DAY = 86400.0
MONTH_DAYS = 30.4375  # mean Gregorian month, for "4.3 months"

#: Tick labels of Figs 1-2, left to right. "1mon" is taken as 30 days; the
#: least-squares residual over all 9 ticks checks that assumption.
X_TICKS_S = [1, 10, 60, 900, 3600, 21600, 86400, 7 * 86400, 30 * 86400]
Y_TICKS_PCT = [0, 20, 40, 60, 80, 100]

#: Legend-derived colour maps (swatch line/rect sits on its label's baseline).
FIG2_COLORS = {"node_restart": (1.0, 0.0, 0.0), "planned_reboot": (0.0, 1.0, 0.0),
               "unplanned_reboot": (0.0, 0.0, 1.0)}
FIG3_COLORS = {"unknown": (0.0, 0.0, 1.0), "node_restart": (1.0, 1.0, 0.0),
               "planned_reboot": (0.0, 1.0, 0.0), "unplanned_reboot": (1.0, 0.0, 0.0)}

#: Table 1 (all cells): unavailability %, average / min / max  (p. 4)
TABLE1 = {"node_restart": (0.0139, 0.0004, 0.1295), "planned_reboot": (0.0154, 0.0050, 0.0563),
          "unplanned_reboot": (0.0025, 0.0000, 0.0122), "unknown": (0.0142, 0.0013, 0.0454)}
#: Table 2 MTTFs (p. 5)
TABLE2 = {"disk_years": (10.0, 50.0), "node_months": 4.3, "rack_years": 10.2}

CAUSE_MAP = {"node_restart": "sw", "unplanned_reboot": "hw_reboot"}


def _round(x, nd=6):
    if isinstance(x, float):
        return float(f"{x:.{nd}g}")
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


def _color_eq(c, ref, tol=1e-6):
    try:
        return c is not None and len(c) == 3 and all(abs(a - b) < tol for a, b in zip(c, ref))
    except TypeError:
        return False


# -- vector extraction ---------------------------------------------------------

def axes_for_cdf_figure(page, bbox):
    x0, top, x1, bottom = bbox
    lines = [l for l in page.lines if x0 <= l["x0"] <= x1 and top <= l["top"] <= bottom]
    horiz = [l for l in lines if abs(l["top"] - l["bottom"]) < 0.05 and l["x1"] - l["x0"] > 50]
    vert = [l for l in lines if abs(l["x0"] - l["x1"]) < 0.05 and l["bottom"] - l["top"] > 50]
    ybot = max(l["top"] for l in horiz)
    ytop = min(l["top"] for l in horiz)
    xleft = min(l["x0"] for l in vert)
    xt = tick_positions(page.lines, "x", ybot, (xleft - 1, x1))
    yt = tick_positions(page.lines, "y", xleft, (ytop - 1, ybot + 1))
    if len(xt) != len(X_TICKS_S):
        raise RuntimeError(f"expected {len(X_TICKS_S)} x ticks, found {xt}")
    if len(yt) != len(Y_TICKS_PCT):
        raise RuntimeError(f"expected {len(Y_TICKS_PCT)} y ticks, found {yt}")
    xmap = fit_axis(xt, X_TICKS_S, log=True)
    # y grows downward in page coordinates: tick at largest 'top' is 0%.
    ymap = fit_axis(sorted(yt, reverse=True), [v / 100 for v in Y_TICKS_PCT], log=False)
    return xmap, ymap


def curve_by_color(page, bbox, rgb):
    x0, top, x1, bottom = bbox
    cs = [c for c in page.curves if _color_eq(c.get("stroking_color"), rgb)
          and x0 <= c["x0"] and c["x1"] <= x1 and top <= c["top"] and c["bottom"] <= bottom]
    if len(cs) != 1:
        raise RuntimeError(f"expected one {rgb} curve in {bbox}, got {len(cs)}")
    return cs[0]


def save_steps(path: Path, sc: StepCurve):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_seconds", "F_before", "F_after"])
        for x, a, b in zip(sc.x, sc.F_before, sc.F_after):
            w.writerow([f"{x:.6g}", f"{a:.6f}", f"{b:.6f}"])


def empirical_spec(sc: StepCurve, quantum_decades: float, shift_decades: float = 0.0) -> dict:
    """ECDF as an `empirical` spec, each jump spread over one coordinate quantum.

    The PDF stores x to one coordinate quantum (~0.002 decades), so a jump at
    x_j is only known to lie in [x_j 10^-q, x_j]. Each jump's mass is spread
    log-uniformly over that interval: F rises from F_before to F_after across
    it and is flat between jumps. This keeps the ECDF's mean and quantiles
    (unlike interpolating across the gaps between jumps, which drags tail mass
    toward shorter durations).
    """
    x = sc.x * 10**shift_decades
    Fb = np.clip(sc.F_before, 0, 1)
    Fa = np.clip(sc.F_after, 0, 1)
    if Fa[-1] > 0.999:  # curve ends on the top frame: normalise, don't snap
        Fb, Fa = Fb / Fa[-1], Fa / Fa[-1]
    r8 = lambda v: float(f"{v:.8g}")  # noqa: E731  (round before comparing)
    out_x, out_F = [], []
    for xj, fb, fa in zip(x, Fb, Fa):
        lo, hi = r8(xj * 10 ** (-quantum_decades)), r8(xj)
        if not out_x or lo > out_x[-1]:
            out_x.append(lo)
            out_F.append(max(round(float(fb), 6), out_F[-1] if out_F else 0.0))
        if hi > out_x[-1]:
            out_x.append(hi)
            out_F.append(max(round(float(fa), 6), out_F[-1]))
        else:
            out_F[-1] = max(out_F[-1], round(float(fa), 6))
    if out_F[0] != 0.0:  # distribution must start at F = 0
        out_x.insert(0, r8(out_x[0] * 10 ** (-quantum_decades)))
        out_F.insert(0, 0.0)
    return {"type": "empirical", "x": out_x, "F": out_F, "log_x": True, "tail": None}


def fig3_bars(page, bbox):
    x0, top, x1, bottom = bbox
    lines = [l for l in page.lines if x0 <= l["x0"] <= x1 and top <= l["top"] <= bottom]
    # left axis: vertical line; ticks are short horizontal lines ending on it
    vert = [l for l in lines if abs(l["x0"] - l["x1"]) < 0.05 and l["bottom"] - l["top"] > 50]
    xaxis = vert[0]["x0"]
    ybot = max(l["bottom"] for l in vert)
    yt = tick_positions(page.lines, "y", xaxis, (top, ybot + 1))
    ymap = fit_axis(sorted(yt, reverse=True), [0, 10, 20, 30, 40], log=False)
    xt = tick_positions(page.lines, "x", ybot, (x0, x1))
    xmap = fit_axis(xt, [0, 1, 2, 3], log=False)
    rects = [r for r in page.rects if x0 <= r["x0"] and r["x1"] <= x1 and top <= r["top"] <= bottom
             and (r["x1"] - r["x0"]) < 3]
    bars = {}
    for r in rects:
        key = round(r["x0"], 2)
        cause = next((c for c, rgb in FIG3_COLORS.items() if _color_eq(r.get("non_stroking_color"), rgb)), None)
        if cause is None:
            continue
        h = float(ymap.to_value(r["top"]) - ymap.to_value(r["bottom"]))
        b = bars.setdefault(key, {"x0": r["x0"], "x1": r["x1"], **{c: 0.0 for c in FIG3_COLORS}})
        b[cause] += h
    rows = []
    for key in sorted(bars):
        b = bars[key]
        rows.append({"month": float(xmap.to_value(0.5 * (b["x0"] + b["x1"]))),
                     **{c: b[c] for c in FIG3_COLORS}})
    return rows, xmap, ymap


# -- bootstrap -------------------------------------------------------------------

BOOT = 2000


def min_events(sc: StepCurve, f_quantum: float) -> int:
    """Lower bound on the events behind an ECDF drawn with coordinate quantum f_quantum (in F).

    Jump heights are rounded to whole quanta. The smallest observed jump is one
    quantum, so a single event's true height is below 1.5 quanta, i.e.
    N > 1 / (1.5 * f_quantum). (N cannot be pinned down further from the PDF.)
    """
    smallest = float(np.min(sc.F_after - sc.F_before))
    return int(math.ceil(1.0 / (1.5 * max(smallest, f_quantum))))


def bootstrap_curve(sc: StepCurve, seed: int, n_events: int, B: int = BOOT) -> dict[str, np.ndarray]:
    """Resample the digitized ECDF with ``n_events`` draws (a lower bound on the
    true event count, see min_events), so the intervals are conservative.
    Returns arrays of E[D], P(D >= 15 min), E[D 1{D >= 15 min}], E[D | D >= 15 min].
    """
    rng = np.random.default_rng(seed)
    w = (sc.F_after - sc.F_before)
    w = w / w.sum()
    n = int(n_events)
    idx = rng.choice(len(sc.x), size=(B, n), p=w)
    x = sc.x[idx]
    ge = x >= 900.0
    mean = x.mean(axis=1)
    p15 = ge.mean(axis=1)
    m15 = (x * ge).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c15 = np.where(p15 > 0, m15 / np.maximum(p15, 1e-300), np.nan)
    return {"mean": mean, "p15": p15, "m15": m15, "c15": c15}


# -- raster cross-check --------------------------------------------------------

def raster_check(page_no, bbox_pt, colors, step_curves, exclude_pt=()):
    import cv2
    img = render_page(PDF, page_no, DPI)
    s = DPI / 72.0
    x0, top, x1, bottom = bbox_pt
    crop = img[int(top * s):int(bottom * s), int(x0 * s):int(x1 * s)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ftop, fbot, fleft, fright = raster_frame(gray)
    ticks = raster_ticks_x(gray, fbot, fleft, fright)
    # The frame's right edge reads as a tick when the axis extends past the
    # last labelled tick (Fig 2); drop edge groups until the count matches.
    while len(ticks) > len(X_TICKS_S) and abs(ticks[-1] - fright) <= 3:
        ticks = ticks[:-1]
    if len(ticks) != len(X_TICKS_S):
        return {"status": "tick_detection_failed", "ticks_found": len(ticks)}, crop
    xmap = fit_axis(ticks, X_TICKS_S, log=True)
    excl = [(int((t - top) * s), int((b - top) * s), int((l - x0) * s), int((r - x0) * s))
            for (l, t, r, b) in exclude_pt]
    halfw = 0.5 * 0.52086 * s + 1.0  # half the stroke width plus one pixel, in px
    report = {"status": "ok", "dpi": DPI, "frame_px": [ftop, fbot, fleft, fright],
              "x_axis_max_residual_px": xmap.max_residual_pt, "series": {}}
    for name, color in colors.items():
        tops = raster_series_top(crop, color, (ftop, fbot, fleft, fright), excl)
        sc = step_curves[name]
        gaps = []
        for col, row in tops.items():
            F_r = (fbot - row) / (fbot - ftop)
            xv_lo = float(xmap.to_value(col - halfw))
            xv_hi = float(xmap.to_value(col + halfw))
            lo = float(sc.cdf(xv_lo)) if xv_lo >= sc.x[0] else 0.0
            hi = float(sc.cdf(xv_hi)) if xv_hi >= sc.x[0] else 0.0
            gaps.append(0.0 if lo - 0.004 <= F_r <= hi + 0.004 else min(abs(F_r - lo), abs(F_r - hi)))
        g = np.array(gaps)
        report["series"][name] = {"columns": int(len(g)), "max_gap": float(g.max()),
                                  "p99_gap": float(np.quantile(g, 0.99)),
                                  "frac_within_0.02": float((g <= 0.02).mean())}
    return report, crop


def overlay(crop, bbox_pt, xmap: AxisMap, ymap: AxisMap, curves: dict, path: Path):
    import cv2
    s = DPI / 72.0
    x0, top = bbox_pt[0], bbox_pt[1]
    img = crop.copy()
    for sc in curves.values():
        for x, f in zip(sc.x, sc.F_after):
            px = int((float(xmap.to_pt(x)) - x0) * s)
            py = int((float(ymap.to_pt(f)) - top) * s)
            cv2.circle(img, (px, py), 3, (0, 0, 0), -1)
    cv2.imwrite(str(path), img)


# -- main ------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(PDF) as pdf:
        p2, p4 = pdf.pages[1], pdf.pages[3]
        fig1_bbox = (345, 75, 535, 175)
        fig2_bbox = (100, 75, 290, 175)
        fig3_bbox = (95, 240, 300, 412)
        x1m, y1m = axes_for_cdf_figure(p2, fig1_bbox)
        x2m, y2m = axes_for_cdf_figure(p4, fig2_bbox)
        fig1 = step_curve_from_pts(curve_by_color(p2, fig1_bbox, (1.0, 0.0, 0.0))["pts"], x1m, y1m)
        fig2 = {c: step_curve_from_pts(curve_by_color(p4, fig2_bbox, rgb)["pts"], x2m, y2m)
                for c, rgb in FIG2_COLORS.items()}
        bars, x3m, y3m = fig3_bars(p4, fig3_bbox)
        stroke_w = 0.52086

    save_steps(OUT / "fig1_all_causes.csv", fig1)
    for c, sc in fig2.items():
        save_steps(OUT / f"fig2_{c}.csv", sc)
    with (OUT / "fig3_daily.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["month"] + list(FIG3_COLORS))
        w.writeheader()
        for r in bars:
            w.writerow({k: f"{v:.5g}" for k, v in r.items()})

    # digitization band: +/- max(axis residual, half stroke) in decades
    def band_decades(xm: AxisMap) -> float:
        return max(xm.max_residual_pt, 0.5 * stroke_w) / xm.b

    calib = {"fig1": {"x": x1m.as_dict(), "y": y1m.as_dict(), "band_decades": band_decades(x1m)},
             "fig2": {"x": x2m.as_dict(), "y": y2m.as_dict(), "band_decades": band_decades(x2m)},
             "fig3": {"x_months": x3m.as_dict(), "y_events": y3m.as_dict()},
             "note": "x tick '1mon' assumed 30 days; residuals confirm the log10 axis."}
    (OUT / "calibration.json").write_text(json.dumps(_round(calib), indent=2) + "\n")

    # independent raster pass
    legend_excl = [(250.0, 84.0, 281.0, 105.0)]
    r1, crop1 = raster_check(2, fig1_bbox, {"all": "red"}, {"all": fig1})
    r2, crop2 = raster_check(4, fig2_bbox, {"node_restart": "red", "planned_reboot": "green",
                                             "unplanned_reboot": "blue"}, fig2, legend_excl)
    (OUT / "raster_check.json").write_text(json.dumps(_round({"fig1": r1, "fig2": r2}), indent=2) + "\n")
    overlay(crop1, fig1_bbox, x1m, y1m, {"all": fig1}, OUT / "fig1_overlay.png")
    overlay(crop2, fig2_bbox, x2m, y2m, fig2, OUT / "fig2_overlay.png")

    # statistics per cause
    q = 0.052086 / x2m.b  # PDF coordinate quantum in decades
    stats = {}
    for c, sc in {**fig2, "all_causes_fig1": fig1}.items():
        best, fits = best_cdf_fit(sc.x, sc.F_after, sc.F_before)
        stats[c] = {
            "n_jumps": int(len(sc.x)), "mean_s": sc.mean(), "median_s": sc.quantile(0.5),
            "p10_s": sc.quantile(0.1), "p90_s": sc.quantile(0.9),
            "F_15min": float(sc.cdf(900.0)), "P_gt_15min": 1 - float(sc.cdf(900.0)),
            "min_s": float(sc.x[0]), "max_s": float(sc.x[-1]),
            "fits": {f.family: {"params": f.params, "ks": f.ks} for f in fits},
            "best_fit": best.family, "best_fit_spec": best.spec(),
        }
    days = np.array([[r[c] for c in FIG3_COLORS] for r in bars])
    fig3_avg = {c: float(days[:, i].mean()) for i, c in enumerate(FIG3_COLORS)}
    fig3_sd = {c: float(days[:, i].std(ddof=1)) for i, c in enumerate(FIG3_COLORS)}

    # Rates. Table 1 and Fig 3 count only events >= 15 min (the paper's
    # convention; Fig 2 alone includes shorter events). So
    #   lambda_ge15 = U / E[D | D >= 15 min]   and   lambda_all = lambda_ge15 / P(D >= 15 min)
    # with E[D | D >= 15 min] and P(D >= 15 min) from Fig 2.
    rates = {}
    boots = {}
    for i, c in enumerate(("node_restart", "planned_reboot", "unplanned_reboot")):
        sc = fig2[c]
        dF = sc.F_after - sc.F_before
        ge = sc.x >= 900.0
        p15 = float(dF[ge].sum())
        m15 = float((sc.x[ge] * dF[ge]).sum())  # E[D 1{D >= 15 min}], seconds
        ed15_h = m15 / p15 / H
        n_ev = min_events(sc, 0.052086 / (y2m.b * -1 if y2m.b < 0 else y2m.b))
        bt = bootstrap_curve(sc, seed=100 + i, n_events=n_ev)
        boots[c] = bt
        qt = lambda a, p: float(np.nanquantile(a, p))  # noqa: E731
        U_avg, U_min, U_max = (v / 100.0 for v in TABLE1[c])
        # central: point estimate; low/high: Table 1 cell range x bootstrap duration uncertainty
        ge15 = {"low": U_min / (qt(bt["c15"], 0.95) / H), "central": U_avg / ed15_h,
                "high": U_max / (qt(bt["c15"], 0.05) / H)}
        allv = {"low": U_min / (qt(bt["m15"], 0.95) / H), "central": U_avg / (m15 / H),
                "high": U_max / (max(qt(bt["m15"], 0.05), 1e-9) / H)}
        fig3_rate = fig3_avg[c] / 1000.0 / 24.0
        stats[c]["P_ge_15min"] = p15
        stats[c]["mean_ge_15min_s"] = ed15_h * H
        stats[c]["bootstrap"] = {"N_events_assumed": n_ev, "N_rule": "smallest jump = 1 quantum -> N > 1/(1.5 quantum)",
                                 "B": BOOT,
                                 "mean_s_5_95": [qt(bt["mean"], 0.05), qt(bt["mean"], 0.95)],
                                 "P_ge_15min_5_95": [qt(bt["p15"], 0.05), qt(bt["p15"], 0.95)],
                                 "mean_ge_15min_s_5_95": [qt(bt["c15"], 0.05), qt(bt["c15"], 0.95)],
                                 "n_jumps_ge_15min": int(ge.sum())}
        rates[c] = {"ge15min": ge15, "all_events": allv,
                    "fig3_one_cell_ge15min": fig3_rate,
                    "fig3_over_table1": fig3_rate / ge15["central"],
                    "fig3_within_table1_range": ge15["low"] <= fig3_rate <= ge15["high"]}
    # Table 2 check with bootstrap uncertainty (causes resampled independently)
    node_rate = 1.0 / (TABLE2["node_months"] * MONTH_DAYS * 24)
    ratio_draws = sum(TABLE1[c][0] / 100.0 / (boots[c]["c15"] / H) for c in boots) / node_rate
    ge15 = sum(rates[c]["ge15min"]["central"] for c in rates)
    node_mttf_rate = node_rate

    checks = {
        "fig1_F_15min_gt_0.90": stats["all_causes_fig1"]["F_15min"] > 0.90,
        "node_restart_spans_lt1min_to_gt1h": stats["node_restart"]["min_s"] < 60 < 3600 < stats["node_restart"]["max_s"],
        "unplanned_reboot_has_largest_mean": stats["unplanned_reboot"]["mean_s"] == max(
            stats[c]["mean_s"] for c in FIG2_COLORS),
        "raster_fig1_within_0.02": r1.get("series", {}).get("all", {}).get("frac_within_0.02", 0),
        "raster_fig2_within_0.02": {k: v["frac_within_0.02"] for k, v in r2.get("series", {}).items()},
        "sum_ge15min_rates_table1_per_node_hour": ge15,
        "table2_node_mttf_rate_per_node_hour": node_mttf_rate,
        "ratio_sum_ge15min_to_table2_node_mttf": ge15 / node_mttf_rate,
        "ratio_bootstrap_5_95": [float(np.nanquantile(ratio_draws, 0.05)), float(np.nanquantile(ratio_draws, 0.95))],
        "ratio_note": ("node-restart E[D | D >= 15 min] rests on few jumps (one ~25 h event dominates); "
                       "treat the Table 2 agreement as weak evidence"),
    }
    write_entries(stats, rates, fig3_avg, fig3_sd, calib, checks, q)
    print(json.dumps(_round({"stats": {c: {k: v for k, v in s.items() if k not in ("fits", "best_fit_spec")}
                                      for c, s in stats.items()},
                             "rates_per_node_hour": rates, "fig3_avg_per_1000_day": fig3_avg,
                             "checks": checks}, 4), indent=1))


def write_entries(stats, rates, fig3_avg, fig3_sd, calib, checks, q):
    tri = lambda lo, c, hi: {"low": lo, "central": c, "high": hi}  # noqa: E731
    url = "https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Ford.pdf"
    meas = []

    def m(mid, verbatim, value, unit, status, loc, covers=(), conversion=None, notes=None):
        d = {"id": mid, "metric_verbatim": verbatim, "value": value, "unit": unit, "status": status,
             "location": loc, "covers": list(covers)}
        if conversion:
            d["conversion"] = conversion
        if notes:
            d["notes"] = notes
        meas.append(d)

    m("pop", "tens of Google storage cells, each with 1000 to 7000 nodes, over a one year period",
      "tens of cells x 1000-7000 nodes x 1 yr", "text", "V", "p.2 §2.1")
    m("lt10pct-gt15min", "less than 10% of events last longer than 15 minutes", 0.10, "fraction", "V", "p.2 §2.1 / Fig 1")
    m("fig2-scope", "Figure 2 ... one of our storage cells ... we do not exclude events that last less than "
      "15 minutes, but we still end the unavailability period when the system reconstructs all the data",
      "one cell; includes <15 min; truncated by reconstruction", "text", "V", "p.3 §3.1")
    for c, (avg, lo, hi) in TABLE1.items():
        m(f"table1-{c}", f"Table 1 unavailability (%) average / min / max: {c.replace('_', ' ')}",
          {"average": avg, "min": lo, "max": hi}, "percent of node time", "V", "p.4 Table 1",
          covers=[CAUSE_MAP.get(c, c)])
    m("table2-disk-mttf", "Table 2 Disk MTTF 10-50 years", list(TABLE2["disk_years"]), "years", "V", "p.5 Table 2", ["disk"],
      "rate_per_hour = 1 / (years * 8766)")
    m("table2-node-mttf", "Table 2 Node MTTF 4.3 months", TABLE2["node_months"], "months", "V", "p.5 Table 2")
    m("table2-rack-mttf", "Table 2 Rack MTTF 10.2 years", TABLE2["rack_years"], "years", "V", "p.5 Table 2",
      notes="input for exec #3 (correlated failures)")
    for c, s in stats.items():
        fig = "Fig 1" if c.startswith("all") else "Fig 2"
        m(f"digitized-{c}", f"{fig} CDF of unavailability duration ({c.replace('_', ' ')}), digitized from vector paths",
          {k: s[k] for k in ("n_jumps", "mean_s", "median_s", "p10_s", "p90_s", "F_15min", "min_s", "max_s")},
          "seconds / fraction", "V", f"p.{2 if c.startswith('all') else 4} {fig}",
          covers=[CAUSE_MAP.get(c, "planned")] if not c.startswith("all") else [],
          conversion="vector path -> step ECDF jumps; see digitized/ford10/*.csv and calibration.json",
          notes=f"best CDF fit {s['best_fit']}; fits: " + json.dumps(_round(s["fits"], 4)))
    for c in FIG3_COLORS:
        m(f"fig3-avg-{c}", f"Fig 3 events per 1000 nodes per day ({c.replace('_', ' ')}), mean over 92 days, one cell",
          {"mean": fig3_avg[c], "sd_daily": fig3_sd[c]}, "events/1000 nodes/day", "V", "p.4 Fig 3",
          covers=[CAUSE_MAP.get(c, c)], conversion="sum of stacked-bar heights per colour, legend-mapped")
    m("fig3-daily", "Fig 3 events per 1000 nodes per day by cause, daily series over 92 days, one cell",
      "digitized/ford10/fig3_daily.csv", "events/1000 nodes/day", "V", "p.4 Fig 3",
      covers=[CAUSE_MAP.get(c, c) for c in FIG3_COLORS],
      conversion="stacked-bar heights per day and colour (vector rectangles), legend-mapped")
    for c, rr in rates.items():
        m(f"derived-rate-{c}", f"Derived {c.replace('_', ' ')} rate per node-hour (>=15 min and all events)",
          {"ge15min": rr["ge15min"], "all_events": rr["all_events"],
           "fig3_one_cell_ge15min": rr["fig3_one_cell_ge15min"], "fig3_over_table1": rr["fig3_over_table1"],
           "P_ge_15min": stats[c]["P_ge_15min"], "mean_ge_15min_s": stats[c]["mean_ge_15min_s"]},
          "per node-hour", "V", "derived from Table 1 + Fig 2 (+ Fig 3 check)", [CAUSE_MAP.get(c, "planned")],
          "lambda_ge15 = U/E[D|D>=15min]; lambda_all = lambda_ge15/P(D>=15min)")
    src = {"id": "ford10", "kind": "source",
           "title": "Availability in Globally Distributed Storage Systems",
           "authors": "D. Ford, F. Labelle, F. I. Popovici, M. Stokely, V.-A. Truong, L. Barroso, C. Grimes, S. Quinlan",
           "year": 2010, "venue": "OSDI", "url": url, "local_path": "docs/reading/Ford_OSDI10.pdf",
           "population": "Google storage nodes (bare metal), ~2010", "accessed": "2026-10-21",
           "digitization": {"calibration": calib, "checks": checks, "coordinate_quantum_decades": q},
           "measurements": meas}

    # durations (empirical, with digitization band)
    band = calib["fig2"]["band_decades"]

    def load_steps(c):
        rows = list(csv.DictReader((OUT / f"fig2_{c}.csv").open()))
        return StepCurve(np.array([float(r["x_seconds"]) for r in rows]),
                         np.array([float(r["F_before"]) for r in rows]),
                         np.array([float(r["F_after"]) for r in rows]))

    def conditional_ge15(sc: StepCurve) -> StepCurve:
        """Duration distribution given D >= 15 min (renormalised).

        The curves end on the top frame (F = 1); the raw final value is
        ~0.9999 from axis-fit rounding, which renormalising a 1% tail would
        amplify, so the curve is normalised by its final value first.
        """
        if sc.F_after[-1] > 0.999:
            top = sc.F_after[-1]
            sc = StepCurve(sc.x, sc.F_before / top, sc.F_after / top)
        keep = sc.x >= 900.0
        f0 = sc.F_before[keep][0]
        scale = 1.0 - f0
        return StepCurve(sc.x[keep], (sc.F_before[keep] - f0) / scale, (sc.F_after[keep] - f0) / scale)

    def dur_triple(c, ge15=False):
        """central = digitized curve; low/high = curve rescaled in time so its mean sits at the
        bootstrap 5th/95th percentile of the mean (or at least the digitization band)."""
        sc = load_steps(c)
        b = stats[c]["bootstrap"]
        if ge15:
            sc = conditional_ge15(sc)
            m, (lo_m, hi_m) = stats[c]["mean_ge_15min_s"], b["mean_ge_15min_s_5_95"]
        else:
            m, (lo_m, hi_m) = stats[c]["mean_s"], b["mean_s_5_95"]
        lo_shift = min(-band, math.log10(lo_m / m))
        hi_shift = max(band, math.log10(hi_m / m))
        return tri(empirical_spec(sc, q, lo_shift), empirical_spec(sc, q, 0.0), empirical_spec(sc, q, hi_shift))

    disk_rate = tri(1 / (50 * 8766), 1 / (math.sqrt(10 * 50) * 8766), 1 / (10 * 8766))
    common_notes = ("Rates: Table 1 (all cells) gives unavailability from events >= 15 min, so "
                    "lambda_ge15 = U / E[D | D >= 15 min] and lambda_all = lambda_ge15 / P(D >= 15 min), with the "
                    "conditional mean and P from Fig 2 (one cell). Table 1 avg/min/max -> central/low/high. "
                    "Check: the summed >=15-min rate matches Table 2's node MTTF (4.3 months) within ~5%. Planned "
                    "reboots are a planned_downtime profile, not a cause here. 'Unknown' events (Table 1: 0.0142% "
                    "avg) are excluded: their durations are not reported. No host-loss cause: Ford reports "
                    "transient unavailability; data loss enters only through the disk term (Table 2).")
    nodes = []
    for variant, ge15, key in (("", False, "all_events"), ("-ge15min", True, "ge15min")):
        nodes.append({
            "id": f"cluster/ford10-google-storage{variant}", "kind": "node", "version": 1, "tier": "reference",
            "tags": ["google", "bare-metal", "2010", "storage"] + (["ge15min"] if ge15 else []),
            "derived_from": ["ford10#table1-node_restart", "ford10#table1-unplanned_reboot",
                             "ford10#digitized-node_restart", "ford10#digitized-unplanned_reboot",
                             "ford10#table2-node-mttf", "disk/ford10"],
            "built_by": "scripts/fault_library/build_ford10.py",
            "notes": common_notes + (" This variant keeps only events >= 15 min (durations conditioned on "
                                     "D >= 15 min), i.e. GFS's view that shorter blips are not failures."
                                     if ge15 else " This variant includes all events (full Fig 2 durations)."),
            "causes": {
                "sw": {"rate_per_hour": rates["node_restart"][key], "duration": dur_triple("node_restart", ge15)},
                "hw_reboot": {"rate_per_hour": rates["unplanned_reboot"][key],
                              "duration": dur_triple("unplanned_reboot", ge15)},
                "disk": {"rate_per_hour": disk_rate},
            },
            "shape": {"type": "exponential"}})
    disk = {"id": "disk/ford10", "kind": "component", "component": "disk", "version": 1,
            "derived_from": ["ford10#table2-disk-mttf"], "built_by": "scripts/fault_library/build_ford10.py",
            "rate_per_hour": disk_rate,
            "notes": "Table 2: disk MTTF 10-50 years (application-visible disk errors). Central = geometric mean."}
    pr = rates["planned_reboot"]["all_events"]
    planned = [
        {"id": "planned_downtime/weekly-ford10", "kind": "planned_downtime", "version": 1,
         "tags": ["default"], "derived_from": ["ford10#digitized-planned_reboot"],
         "built_by": "scripts/fault_library/build_ford10.py",
         "period": {"type": "normal", "mean": 7 * DAY, "std": 12 * H, "min_val": 1 * DAY},
         "duration": dur_triple("planned_reboot"), "phase": "equilibrium", "graceful": False,
         "notes": "Decision 2026-10-20: weekly gap N(7 d, 12 h); duration = Ford Fig 2 planned-reboot CDF."},
        {"id": "planned_downtime/weekly-ford10-rolling", "kind": "planned_downtime", "version": 1,
         "tags": ["sensitivity", "rolling"], "derived_from": ["ford10#digitized-planned_reboot"],
         "built_by": "scripts/fault_library/build_ford10.py",
         "period": {"type": "normal", "mean": 7 * DAY, "std": 12 * H, "min_val": 1 * DAY},
         "duration": dur_triple("planned_reboot"), "phase": "rolling", "graceful": False,
         "notes": ("Rolling variant (decision 2026-10-21: independent phases stay the default): all nodes in one "
                   "AZ (or region, if slots carry no AZ) go down together; groups staggered evenly through the week.")},
        {"id": "planned_downtime/ford10-rate", "kind": "planned_downtime", "version": 1,
         "tags": ["sensitivity"], "derived_from": ["ford10#table1-planned_reboot", "ford10#digitized-planned_reboot"],
         "built_by": "scripts/fault_library/build_ford10.py",
         "period": {"type": "normal", "mean": float(f"{H / pr['central']:.6g}"), "std": 12 * H, "min_val": 1 * DAY},
         "duration": dur_triple("planned_reboot"), "phase": "equilibrium", "graceful": False,
         "notes": ("Google-cadence variant: mean gap = 1 / lambda_all(planned reboot) from Table 1 (see the node "
                   "profile notes for the >=15-min correction), sigma 12 h.")},
    ]
    (LIB / "sources/ford10.json").write_text(json.dumps(_round(src, 8), indent=2) + "\n")
    (LIB / "profiles/node/ford10.json").write_text(json.dumps(_round(nodes, 8), indent=2) + "\n")
    (LIB / "profiles/component/ford10-disk.json").write_text(json.dumps(_round(disk, 8), indent=2) + "\n")
    (LIB / "profiles/planned_downtime/ford10.json").write_text(json.dumps(_round(planned, 8), indent=2) + "\n")


if __name__ == "__main__":
    main()
