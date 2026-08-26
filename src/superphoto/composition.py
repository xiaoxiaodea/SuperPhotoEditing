"""Composition: horizon tilt, subject box, and a conservative crop/rotate hint.

Default is do nothing. Only straighten a clearly tilted horizon, and only
trim when the edges are empty and the subject is compact.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .color import linear_to_srgb, luma, srgb_to_linear
from .models import GeometryHint, TechnicalReport


def measure_composition(rgb: np.ndarray) -> dict[str, float]:
    y = luma(np.clip(rgb, 0.0, 1.0))
    tilt, conf = _horizon_tilt(y)
    box = _subject_box(y)
    waste = _margin_waste(y, box)
    thirds = _thirds_error(box)
    return {
        "horizon_tilt_deg": tilt,
        "horizon_confidence": conf,
        "subject_left": box[0],
        "subject_top": box[1],
        "subject_right": box[2],
        "subject_bottom": box[3],
        "margin_waste": waste,
        "thirds_error": thirds,
    }


def suggest_geometry(technical: TechnicalReport, genre: str, slot: str) -> GeometryHint:
    """Slot A: straighten only. B/C: optional modest crop. Skip if unsure."""
    rotate = 0.0
    notes: list[str] = []
    landscapeish = genre in {"landscape", "architecture"} or technical.top_third_blue > 0.12
    tilt = float(technical.horizon_tilt_deg)
    conf = float(technical.horizon_confidence)
    if landscapeish and conf >= 0.32 and 0.55 <= abs(tilt) <= 2.7:
        rotate = float(np.clip(-tilt, -2.3, 2.3))
        notes.append(f"地平线校正 {rotate:+.1f}°")
    elif conf >= 0.45 and 0.8 <= abs(tilt) <= 2.2 and genre != "portrait":
        rotate = float(np.clip(-tilt, -2.0, 2.0))
        notes.append(f"画面倾斜 {rotate:+.1f}°")

    left, top, right, bottom = 0.0, 0.0, 1.0, 1.0
    cropped = False
    if slot != "A":
        sub_w = technical.subject_right - technical.subject_left
        sub_h = technical.subject_bottom - technical.subject_top
        compact = sub_w < 0.92 and sub_h < 0.92
        empty = technical.margin_waste > 0.22 or technical.thirds_error > 0.12
        if compact and empty and (sub_w * sub_h) < 0.78:
            pad = 0.10 if slot == "B" else 0.07
            max_trim = 0.14 if slot == "B" else 0.18
            left, top, right, bottom = _padded_aspect_box(
                technical.subject_left,
                technical.subject_top,
                technical.subject_right,
                technical.subject_bottom,
                pad=pad,
                max_trim=max_trim,
            )
            trimmed = (right - left) < 0.97 or (bottom - top) < 0.97
            if trimmed:
                cropped = True
                notes.append("收掉空边，主体放到更稳的位置")

    apply = abs(rotate) > 0.2 or cropped
    if not apply:
        return GeometryHint()
    return GeometryHint(
        rotate_deg=rotate,
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        apply=True,
        why="；".join(notes),
    )


def apply_geometry(linear: np.ndarray, geom: GeometryHint) -> np.ndarray:
    if geom.is_noop():
        return linear
    srgb = np.clip(linear_to_srgb(np.maximum(linear, 0.0)), 0.0, 1.0)
    u8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8, mode="RGB")
    if abs(geom.rotate_deg) > 0.05:
        im = im.rotate(geom.rotate_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    w, h = im.size
    l = int(round(float(np.clip(geom.crop_left, 0.0, 1.0)) * w))
    t = int(round(float(np.clip(geom.crop_top, 0.0, 1.0)) * h))
    r = int(round(float(np.clip(geom.crop_right, 0.0, 1.0)) * w))
    b = int(round(float(np.clip(geom.crop_bottom, 0.0, 1.0)) * h))
    l, t = max(0, min(l, w - 8)), max(0, min(t, h - 8))
    r, b = max(l + 8, min(r, w)), max(t + 8, min(b, h))
    if (r - l) < w or (b - t) < h:
        im = im.crop((l, t, r, b))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return srgb_to_linear(arr)


def clamp_geometry(geom: GeometryHint) -> GeometryHint:
    """Reject heroic crops and dutch angles from a VLM."""
    rot = float(np.clip(geom.rotate_deg, -2.5, 2.5))
    l = float(np.clip(geom.crop_left, 0.0, 0.22))
    t = float(np.clip(geom.crop_top, 0.0, 0.22))
    r = float(np.clip(geom.crop_right, 0.78, 1.0))
    b = float(np.clip(geom.crop_bottom, 0.78, 1.0))
    if (r - l) < 0.72 or (b - t) < 0.72:
        l, t, r, b = 0.0, 0.0, 1.0, 1.0
    apply = geom.apply and (abs(rot) > 0.2 or l > 0.004 or t > 0.004 or r < 0.996 or b < 0.996)
    return GeometryHint(
        rotate_deg=rot if apply else 0.0,
        crop_left=l if apply else 0.0,
        crop_top=t if apply else 0.0,
        crop_right=r if apply else 1.0,
        crop_bottom=b if apply else 1.0,
        apply=apply,
        why=geom.why if apply else "",
    )


def _horizon_tilt(y: np.ndarray) -> tuple[float, float]:
    gy, gx = np.gradient(y.astype(np.float32))
    mag = np.hypot(gx, gy)
    if float(np.max(mag)) < 1e-6:
        return 0.0, 0.0
    thr = float(np.percentile(mag, 88))
    horiz = (np.abs(gy) > np.abs(gx) * 1.35) & (mag >= thr)
    # Horizon usually sits in the middle band, not the very top or dirt at the bottom.
    h, w = y.shape
    band = np.zeros_like(horiz)
    band[int(h * 0.18) : int(h * 0.72)] = True
    horiz = horiz & band
    ys, xs = np.where(horiz)
    if xs.size < 50:
        return 0.0, 0.0
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    A = np.vstack([xs, np.ones(xs.size)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-6)
    deg = float(np.degrees(np.arctan(slope)))
    conf = float(np.clip(r2, 0.0, 1.0)) * float(np.clip(xs.size / 400.0, 0.2, 1.0))
    if abs(deg) > 8.0:
        return 0.0, 0.0
    return float(np.clip(deg, -8.0, 8.0)), float(np.clip(conf, 0.0, 1.0))


def _subject_box(y: np.ndarray) -> tuple[float, float, float, float]:
    gy, gx = np.gradient(y.astype(np.float32))
    contrast = np.hypot(gx, gy)
    h, w = y.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt(((yy - cy) / max(h, 1)) ** 2 + ((xx - cx) / max(w, 1)) ** 2)
    sal = contrast * (1.15 - np.clip(dist, 0.0, 1.0))
    # Prefer brighter midtones a little — faces, flowers, peaks.
    sal = sal * (0.55 + 0.45 * np.clip(y, 0.0, 1.0))
    thr = float(np.percentile(sal, 80))
    mask = sal >= thr
    if float(np.mean(mask)) > 0.72:
        return 0.0, 0.0, 1.0, 1.0
    ys, xs = np.where(mask)
    if xs.size < 12:
        return 0.08, 0.08, 0.92, 0.92
    l = float(xs.min()) / max(w - 1, 1)
    r = float(xs.max()) / max(w - 1, 1)
    t = float(ys.min()) / max(h - 1, 1)
    b = float(ys.max()) / max(h - 1, 1)
    return l, t, r, b


def _margin_waste(y: np.ndarray, box: tuple[float, float, float, float]) -> float:
    h, w = y.shape
    gy, gx = np.gradient(y.astype(np.float32))
    energy = np.hypot(gx, gy)
    m = 0.10
    outer = np.ones_like(energy, dtype=bool)
    outer[int(h * m) : int(h * (1 - m)), int(w * m) : int(w * (1 - m))] = False
    inner = ~outer
    o = float(np.mean(energy[outer])) if np.any(outer) else 0.0
    i = float(np.mean(energy[inner])) if np.any(inner) else 1e-6
    edge_ratio = o / max(i, 1e-6)
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    empty = float(np.clip(1.0 - area, 0.0, 1.0))
    return float(np.clip(0.55 * empty + 0.45 * (1.0 - np.clip(edge_ratio, 0.0, 1.2)), 0.0, 1.0))


def _thirds_error(box: tuple[float, float, float, float]) -> float:
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    targets = (1.0 / 3.0, 0.5, 2.0 / 3.0)
    dx = min(abs(cx - t) for t in targets)
    dy = min(abs(cy - t) for t in targets)
    return float(np.clip(np.hypot(dx, dy) / 0.5, 0.0, 1.0))


def _padded_aspect_box(
    sl: float,
    st: float,
    sr: float,
    sb: float,
    pad: float,
    max_trim: float,
) -> tuple[float, float, float, float]:
    sl = max(0.0, sl - pad)
    st = max(0.0, st - pad)
    sr = min(1.0, sr + pad)
    sb = min(1.0, sb + pad)
    width = sr - sl
    height = sb - st
    side = max(width, height)
    side = min(side, 1.0 - 1e-6)
    # Keep original aspect by using equal normalized span.
    cx = 0.5 * (sl + sr)
    cy = 0.5 * (st + sb)
    half = side / 2.0
    left = cx - half
    top = cy - half
    right = cx + half
    bottom = cy + half
    if left < 0:
        right -= left
        left = 0.0
    if top < 0:
        bottom -= top
        top = 0.0
    if right > 1:
        left -= right - 1.0
        right = 1.0
    if bottom > 1:
        top -= bottom - 1.0
        bottom = 1.0
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(1.0, right)
    bottom = min(1.0, bottom)
    # Do not trim more than max_trim on any side.
    left = min(left, max_trim)
    top = min(top, max_trim)
    right = max(right, 1.0 - max_trim)
    bottom = max(bottom, 1.0 - max_trim)
    return left, top, right, bottom
