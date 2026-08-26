from __future__ import annotations

from pathlib import Path

import numpy as np

from .color import luma, rgb_to_hsl, srgb_to_linear
from .composition import measure_composition
from .ingest import read_exif, source_kind
from .models import CorrectionHint, TechnicalReport


def analyze(linear: np.ndarray, original_path: Path) -> TechnicalReport:
    work = _thumb(linear, 768)
    y = luma(work)
    mean_luma = float(np.mean(y))
    median_luma = float(np.median(y))
    std_luma = float(np.std(y))
    highlight_clip = float(np.mean(y > 0.98))
    shadow_clip = float(np.mean(y < 0.02))
    p5, p95 = np.percentile(y, [5, 95])
    dynamic_spread = float(p95 - p5)
    shadow_p05 = float(p5)
    highlight_p95 = float(p95)

    gy = np.abs(np.diff(y, axis=0, append=y[-1:]))
    gx = np.abs(np.diff(y, axis=1, append=y[:, -1:]))
    sharpness = float(np.mean(gx + gy))

    # High-frequency residual as a crude noise proxy, normalized by sharpness.
    hh, ww = y.shape
    a = y[1:hh:2, 1:ww:2]
    b = y[0:hh:2, 0:ww:2]
    hmin, wmin = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    noise = float(np.mean(np.abs(a[:hmin, :wmin] - b[:hmin, :wmin]))) if hmin and wmin else 0.0

    h, s, l = rgb_to_hsl(np.clip(work, 0, 1))
    sat_mean = float(np.mean(s))

    # Temperature bias: more red vs blue in midtones.
    mid = (l > 0.15) & (l < 0.85)
    if np.any(mid):
        r = float(np.mean(work[..., 0][mid]))
        b = float(np.mean(work[..., 2][mid]))
        temperature_bias = float(np.clip((r - b) * 2.0, -1, 1))
    else:
        temperature_bias = 0.0

    top = work[: max(1, work.shape[0] // 3)]
    top_h, top_s, _ = rgb_to_hsl(np.clip(top, 0, 1))
    blue_mask = (top_h > 0.50) & (top_h < 0.72) & (top_s > 0.12)
    top_third_blue = float(np.mean(blue_mask))

    skin = (
        (h > 0.02)
        & (h < 0.13)
        & (s > 0.12)
        & (s < 0.65)
        & (l > 0.18)
        & (l < 0.85)
    )
    skin_fraction = float(np.mean(skin))

    green = (h > 0.22) & (h < 0.45) & (s > 0.15) & (l > 0.08) & (l < 0.8)
    green_fraction = float(np.mean(green))
    aqua = (h > 0.42) & (h < 0.58) & (s > 0.10) & (l > 0.08) & (l < 0.88)
    aqua_fraction = float(np.mean(aqua))
    magenta = ((h > 0.82) | (h < 0.05)) & (s > 0.12) & (l > 0.08) & (l < 0.85)
    magenta_fraction = float(np.mean(magenta))
    orange = (h > 0.02) & (h < 0.13) & (s > 0.10) & (l > 0.10) & (l < 0.88)
    orange_fraction = float(np.mean(orange))
    comp = measure_composition(work)

    hist, _ = np.histogram(y, bins=32, range=(0.0, 1.0), density=True)
    hist = (hist / (hist.sum() + 1e-6)).astype(float).tolist()

    suggested = _suggest_exposure(
        mean_luma=mean_luma,
        median_luma=median_luma,
        highlight_clip=highlight_clip,
        shadow_clip=shadow_clip,
        source_kind=source_kind(original_path),
    )

    return TechnicalReport(
        width=int(linear.shape[1]),
        height=int(linear.shape[0]),
        source_kind=source_kind(original_path),
        mean_luma=mean_luma,
        median_luma=median_luma,
        std_luma=std_luma,
        highlight_clip=highlight_clip,
        shadow_clip=shadow_clip,
        dynamic_spread=dynamic_spread,
        sharpness=sharpness,
        noise=noise,
        saturation_mean=sat_mean,
        temperature_bias=temperature_bias,
        top_third_blue=top_third_blue,
        skin_fraction=skin_fraction,
        green_fraction=green_fraction,
        histogram=hist,
        shadow_p05=shadow_p05,
        highlight_p95=highlight_p95,
        exif=read_exif(original_path),
        suggested_exposure_ev=suggested,
        aqua_fraction=aqua_fraction,
        magenta_fraction=magenta_fraction,
        orange_fraction=orange_fraction,
        horizon_tilt_deg=comp["horizon_tilt_deg"],
        horizon_confidence=comp["horizon_confidence"],
        subject_left=comp["subject_left"],
        subject_top=comp["subject_top"],
        subject_right=comp["subject_right"],
        subject_bottom=comp["subject_bottom"],
        margin_waste=comp["margin_waste"],
        thirds_error=comp["thirds_error"],
    )


def correction_from_technical(tech: TechnicalReport) -> CorrectionHint:
    ev = tech.suggested_exposure_ev
    highlights = 0.0
    shadows = 0.0
    whites = 0.0
    blacks = 0.0
    temperature = 0.0
    contrast = 0.0
    saturation = 0.0
    vibrance = 0.0
    dehaze = 0.0
    clarity = 0.0
    developed = tech.source_kind in {"jpeg", "png"}

    # Heaton-style: recover clipped highlights first. Do not open a low-key frame.
    if tech.highlight_clip > 0.012:
        highlights = -min(0.35, 0.08 + tech.highlight_clip * 5)
        whites = -min(0.16, tech.highlight_clip * 2.5)
    truly_dark = tech.mean_luma < 0.10 and tech.median_luma < 0.12
    if truly_dark and tech.shadow_clip > 0.06 and not developed:
        shadows = min(0.22, 0.08 + tech.shadow_clip * 0.5)
    if tech.dynamic_spread < 0.28 and not developed:
        contrast += 0.08
        clarity += 0.05
    elif tech.dynamic_spread > 0.75 and tech.highlight_clip > 0.01:
        contrast -= 0.04

    if developed and tech.saturation_mean > 0.35:
        saturation -= 0.04

    # Don't "correct" cyan water, warm skin, or sunsets — those are the photo.
    scenic_color = tech.top_third_blue > 0.18 or tech.green_fraction > 0.16 or tech.skin_fraction > 0.08
    if not scenic_color:
        if tech.temperature_bias < -0.18:
            temperature = min(12, (-tech.temperature_bias) * 18)
        elif tech.temperature_bias > 0.28:
            temperature = max(-10, -tech.temperature_bias * 12)

    if tech.top_third_blue > 0.22 and tech.highlight_clip > 0.008:
        highlights = min(highlights, -0.08)
        if not developed:
            dehaze += 0.05

    if tech.skin_fraction > 0.08:
        contrast = min(contrast, 0.06)
        clarity = min(clarity, 0.04)
        shadows = min(shadows, 0.08)

    return CorrectionHint(
        exposure_ev=float(np.clip(ev, -1.2, 1.4)),
        temperature=float(np.clip(temperature, -20, 20)),
        highlights=float(np.clip(highlights, -1, 0.3)),
        shadows=float(np.clip(shadows, -0.15, 0.45)),
        whites=float(np.clip(whites, -0.4, 0.25)),
        blacks=float(np.clip(blacks, -0.25, 0.18)),
        contrast=float(np.clip(contrast, -0.2, 0.2)),
        saturation=float(np.clip(saturation, -0.15, 0.12)),
        vibrance=float(np.clip(vibrance, -0.1, 0.15)),
        dehaze=float(np.clip(dehaze, 0, 0.2)),
        clarity=float(np.clip(clarity, -0.05, 0.16)),
    )


def guess_genre(tech: TechnicalReport) -> str:
    landscape_signal = tech.green_fraction > 0.16 or tech.top_third_blue > 0.20
    if landscape_signal and tech.skin_fraction < 0.14:
        return "landscape"
    if tech.skin_fraction > 0.10:
        return "portrait"
    if tech.green_fraction > 0.18 or tech.top_third_blue > 0.22:
        return "landscape"
    if tech.mean_luma < 0.10:
        return "night"
    if tech.exif.get("FocalLength"):
        try:
            fl = str(tech.exif.get("FocalLength", "0"))
            num = float(fl.split("/")[0]) if "/" in fl else float(fl.split()[0])
            if num >= 85 and tech.skin_fraction > 0.02:
                return "portrait"
        except ValueError:
            pass
    return "general"


def _suggest_exposure(
    mean_luma: float,
    median_luma: float,
    highlight_clip: float,
    shadow_clip: float,
    source_kind: str,
) -> float:
    ref = median_luma if median_luma > 1e-5 else mean_luma
    # Finished JPEGs already have a tone curve. Only nudge obvious mistakes.
    if source_kind in {"jpeg", "png"}:
        if ref < 0.07:
            return float(np.clip(np.log2(0.11 / max(ref, 1e-4)), 0.0, 1.1))
        if highlight_clip > 0.04:
            return float(np.clip(-0.25 - highlight_clip * 2.0, -0.8, 0.0))
        return 0.0
    target = 0.16
    if ref <= 1e-4:
        ev = 0.8
    else:
        ev = float(np.log2(target / ref))
    ev = float(np.clip(ev, -1.2, 1.3))
    if highlight_clip > 0.02:
        ev -= min(0.45, highlight_clip * 6)
    if shadow_clip > 0.16 and highlight_clip < 0.01 and mean_luma < 0.12:
        ev += 0.12
    return float(np.clip(ev, -1.2, 1.4))


def _thumb(linear: np.ndarray, edge: int) -> np.ndarray:
    h, w = linear.shape[:2]
    m = max(h, w)
    if m <= edge:
        return linear
    scale = edge / m
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    ys = np.linspace(0, h - 1, nh).astype(np.int32)
    xs = np.linspace(0, w - 1, nw).astype(np.int32)
    return linear[ys][:, xs]


# Imported so analysis tests can build linear JPEGs without a circular ingest import.
def jpeg_bytes_to_linear(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return srgb_to_linear(arr)
