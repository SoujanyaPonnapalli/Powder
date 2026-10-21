"""Digitize plots from PDFs: vector extraction first, raster as an independent check.

Vector path (exact): ``pdfplumber`` exposes line/curve/rect objects with page
coordinates. Axis maps are fitted from tick-mark positions to tick values
(log10 for log axes) by least squares; the max residual is the calibration
error. Step-function curves (empirical CDFs) are reduced to their jumps.

Raster path (independent): render with ``pdftoppm``, find the plot frame and
tick marks from dark pixels, find each series by colour, and read the
top-most pixel per column. It shares nothing with the vector path except the
tick *values*, so agreement between the two validates both.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AxisMap:
    """value = f(pt): linear in value, or in log10(value) when ``log``."""

    a: float
    b: float
    log: bool
    max_residual_pt: float
    ticks: list  # [(pt, value), ...]

    def to_value(self, pt):
        t = (np.asarray(pt, float) - self.a) / self.b
        return 10.0**t if self.log else t

    def to_pt(self, value):
        v = np.log10(value) if self.log else np.asarray(value, float)
        return self.a + self.b * v

    def as_dict(self):
        return {"a": self.a, "b": self.b, "log10": self.log, "max_residual_pt": self.max_residual_pt,
                "ticks": [{"pt": p, "value": v} for p, v in self.ticks]}


def fit_axis(positions, values, log: bool) -> AxisMap:
    p = np.asarray(positions, float)
    v = np.log10(values) if log else np.asarray(values, float)
    A = np.vstack([np.ones_like(v), v]).T
    (a, b), *_ = np.linalg.lstsq(A, p, rcond=None)
    resid = float(np.max(np.abs(A @ np.array([a, b]) - p)))
    return AxisMap(float(a), float(b), log, resid, list(zip(map(float, positions), map(float, values))))


def tick_positions(lines, orientation: str, axis_coord: float, span: tuple[float, float],
                   max_len: float = 5.0, tol: float = 0.05) -> list[float]:
    """Positions of short tick marks touching an axis line.

    orientation 'x': vertical ticks on the horizontal axis at y=axis_coord
    (returns their x); 'y': horizontal ticks on the vertical axis at
    x=axis_coord (returns their y).
    """
    out = set()
    for l in lines:
        if orientation == "x":
            vertical = abs(l["x0"] - l["x1"]) < tol and 0 < (l["bottom"] - l["top"]) <= max_len
            if vertical and span[0] - tol <= l["x0"] <= span[1] + tol and \
                    (abs(l["bottom"] - axis_coord) < tol or abs(l["top"] - axis_coord) < tol):
                out.add(round(l["x0"], 3))
        else:
            horizontal = abs(l["top"] - l["bottom"]) < tol and 0 < (l["x1"] - l["x0"]) <= max_len
            if horizontal and span[0] - tol <= l["top"] <= span[1] + tol and \
                    (abs(l["x0"] - axis_coord) < tol or abs(l["x1"] - axis_coord) < tol):
                out.add(round(l["top"], 3))
    return sorted(out)


@dataclass
class StepCurve:
    """Empirical CDF read from a step path: jumps at x with F before/after."""

    x: np.ndarray
    F_before: np.ndarray
    F_after: np.ndarray

    def mean(self) -> float:
        return float(np.sum(self.x * (self.F_after - self.F_before)) /
                     max(self.F_after[-1] - self.F_before[0], 1e-12))

    def cdf(self, v) -> np.ndarray:
        idx = np.searchsorted(self.x, np.asarray(v, float), side="right") - 1
        out = np.where(idx >= 0, self.F_after[np.clip(idx, 0, None)], self.F_before[0] if len(self.x) else 0)
        return out

    def quantile(self, p: float) -> float:
        i = int(np.searchsorted(self.F_after, p, side="left"))
        return float(self.x[min(i, len(self.x) - 1)])


def step_curve_from_pts(pts, xmap: AxisMap, ymap: AxisMap, eps_pt: float = 1e-6) -> StepCurve:
    """Reduce a staircase path to its vertical jumps."""
    pts = np.asarray(pts, float)
    xs, Fb, Fa = [], [], []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if abs(x1 - x0) < eps_pt and abs(y1 - y0) > eps_pt:  # vertical segment
            f0, f1 = float(ymap.to_value(y0)), float(ymap.to_value(y1))
            if f1 < f0:
                continue
            x = float(xmap.to_value(x0))
            if xs and abs(x - xs[-1]) < 1e-12 * max(1.0, x):
                Fa[-1] = f1
            else:
                xs.append(x)
                Fb.append(f0)
                Fa.append(f1)
    order = np.argsort(xs)
    return StepCurve(np.array(xs)[order], np.array(Fb)[order], np.array(Fa)[order])


# -- raster ----------------------------------------------------------------

def render_page(pdf: Path, page_1based: int, dpi: int = 600) -> np.ndarray:
    import cv2
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "p"
        subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page_1based), "-l", str(page_1based),
                        "-png", str(pdf), str(out)], check=True)
        png = next(Path(d).glob("p*.png"))
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)  # BGR
    return img


def raster_frame(gray: np.ndarray, dark: int = 140, min_frac: float = 0.6):
    """Bounding frame of a plot crop: outermost long dark rows/cols."""
    d = gray < dark
    rows = np.where(d.mean(axis=1) > min_frac)[0]
    cols = np.where(d.mean(axis=0) > min_frac)[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def raster_ticks_x(gray: np.ndarray, bottom_row: int, left: int, right: int, dark: int = 140,
                   probe_px: tuple[int, int] = (6, 20)) -> list[float]:
    """Columns with dark pixels just above the bottom frame (tick marks)."""
    band = gray[bottom_row - probe_px[1]:bottom_row - probe_px[0], left:right + 1] < dark
    hits = np.where(band.mean(axis=0) > 0.8)[0] + left
    groups, cur = [], [hits[0]] if len(hits) else []
    for h in hits[1:]:
        if h - cur[-1] <= 2:
            cur.append(h)
        else:
            groups.append(cur)
            cur = [h]
    if cur:
        groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def raster_series_top(img_bgr: np.ndarray, color: str, frame, exclude=()) -> dict[int, int]:
    """For each column, the top-most pixel of a pure-colour series."""
    top, bottom, left, right = frame
    b, g, r = (img_bgr[..., i].astype(int) for i in range(3))
    masks = {
        "red": (r > 180) & (g < 110) & (b < 110),
        "green": (g > 150) & (r < 110) & (b < 110),
        "blue": (b > 180) & (r < 110) & (g < 110),
    }
    m = masks[color].copy()
    m[:top, :] = m[bottom + 1:, :] = False
    m[:, :left] = m[:, right + 1:] = False
    for (r0, r1, c0, c1) in exclude:
        m[r0:r1, c0:c1] = False
    out = {}
    for c in range(left, right + 1):
        rows = np.where(m[:, c])[0]
        if len(rows):
            out[c] = int(rows.min())
    return out
