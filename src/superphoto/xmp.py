"""Lightroom Camera Raw XMP → SuperPhoto Recipe.

The packs in 调色素材 are labelled LUT but they are CRS presets: tone curve,
HSL, split toning, and basic sliders. They were authored for RAW (+1..2 EV).
We never apply that exposure to an already-developed JPEG.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .models import ColorGradingControls, HueShift, Recipe, ToneCurve

_HSL_HUES = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")

# JPEG-safe mapping from Lightroom -100..100 into our slider units.
_K = {
    "contrast": 0.55,
    "highlights": 0.45,
    "shadows": 0.22,
    "whites": 0.40,
    "blacks": 0.42,
    "clarity": 0.45,
    "dehaze": 0.32,
    "vibrance": 0.42,
    "saturation": 0.48,
    "vignette": 0.55,
    "grain": 0.28,
}


def parse_xmp(path: Path, *, apply_exposure: bool = False) -> Recipe:
    text = path.read_text(encoding="utf-8", errors="replace")
    n = _crs(text)

    hsl: dict[str, HueShift] = {}
    for name in _HSL_HUES:
        cap = name.capitalize()
        hue = n.get(f"HueAdjustment{cap}", 0.0) * 0.30
        raw_sat = n.get(f"SaturationAdjustment{cap}", 0.0) / 100.0
        if raw_sat > 0:
            sat = float(np.clip(raw_sat * 0.62, 0.0, 0.50))
        else:
            sat = float(np.clip(raw_sat, -0.95, 0.0))
        lum = float(np.clip(n.get(f"LuminanceAdjustment{cap}", 0.0) / 100.0, -0.90, 0.90))
        if abs(hue) + abs(sat) + abs(lum) > 1e-4:
            hsl[name] = HueShift(hue=hue, sat=sat, lum=lum)

    color_grading = _native_color_grade(n)
    curve = ToneCurve(
        rgb=_curve_points(text, "ToneCurvePV2012"),
        r=_curve_points(text, "ToneCurvePV2012Red"),
        g=_curve_points(text, "ToneCurvePV2012Green"),
        b=_curve_points(text, "ToneCurvePV2012Blue"),
    )

    shadows = float(np.clip(n.get("Shadows2012", 0.0) / 100.0 * _K["shadows"], -0.22, 0.12))
    exposure = n.get("Exposure2012", 0.0) if apply_exposure else 0.0

    return Recipe(
        exposure_ev=float(np.clip(exposure, -0.45, 0.45)),
        highlights=float(np.clip(n.get("Highlights2012", 0.0) / 100.0 * _K["highlights"], -1, 1)),
        shadows=shadows,
        whites=float(np.clip(n.get("Whites2012", 0.0) / 100.0 * _K["whites"], -1, 1)),
        blacks=float(np.clip(n.get("Blacks2012", 0.0) / 100.0 * _K["blacks"], -1, 1)),
        contrast=float(np.clip(n.get("Contrast2012", 0.0) / 100.0 * _K["contrast"], -0.6, 0.6)),
        saturation=float(np.clip(n.get("Saturation", 0.0) / 100.0 * _K["saturation"], -0.7, 0.4)),
        vibrance=float(np.clip(n.get("Vibrance", 0.0) / 100.0 * _K["vibrance"], -0.5, 0.45)),
        clarity=float(np.clip(n.get("Clarity2012", 0.0) / 100.0 * _K["clarity"], -0.4, 0.5)),
        dehaze=float(np.clip(n.get("Dehaze", 0.0) / 100.0 * _K["dehaze"], -0.32, 0.35)),
        vignette=float(np.clip(-n.get("PostCropVignetteAmount", 0.0) / 100.0 * _K["vignette"], -0.4, 0.4)),
        grain=float(np.clip(n.get("GrainAmount", 0.0) / 100.0 * _K["grain"], 0.0, 0.35)),
        hsl=hsl,
        color_grading=color_grading,
        curve=curve,
    )


def preset_dir() -> Path:
    return Path(__file__).resolve().parent / "presets"


def _crs(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, raw in re.findall(r'crs:([A-Za-z0-9]+)="([^"]+)"', text):
        try:
            value = float(raw.replace("+", "").strip())
            if math.isfinite(value):
                out[key] = value
        except ValueError:
            continue
    return out


def _curve_points(text: str, name: str) -> list[tuple[float, float]]:
    block = re.search(rf"<crs:{name}>\s*<rdf:Seq>(.*?)</rdf:Seq>", text, re.S)
    if not block:
        return []
    pts: list[tuple[float, float]] = []
    for item in re.findall(r"<rdf:li>([^<]+)</rdf:li>", block.group(1)):
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            continue
        x, y = float(parts[0]) / 255.0, float(parts[1]) / 255.0
        pts.append((x, y))
    if len(pts) < 2:
        return []
    if pts == [(0.0, 0.0), (1.0, 1.0)]:
        return []
    return pts


def _native_color_grade(n: dict[str, float]) -> ColorGradingControls:
    """Preserve supported Camera Raw wheels in their native units, once.

    Lightroom still writes shadow/highlight hue and saturation under the
    SplitToning names. Alternate ColorGrade names describe the same controls,
    not an extra layer. Canonical names, including explicit zero, take priority.
    Missing controls stay unset so a partial preset does not reset other wheels.
    The global wheel is not represented by Recipe and is not approximated here.
    """
    values: dict[str, float] = {}
    for field, keys, low, high in (
        ("shadow_hue", ("SplitToningShadowHue", "ColorGradeShadowHue"), 0, 360),
        ("shadow_sat", ("SplitToningShadowSaturation", "ColorGradeShadowSat"), 0, 100),
        ("shadow_lum", ("ColorGradeShadowLum",), -100, 100),
        ("midtone_hue", ("ColorGradeMidtoneHue",), 0, 360),
        ("midtone_sat", ("ColorGradeMidtoneSat",), 0, 100),
        ("midtone_lum", ("ColorGradeMidtoneLum",), -100, 100),
        ("highlight_hue", ("SplitToningHighlightHue", "ColorGradeHighlightHue"), 0, 360),
        ("highlight_sat", ("SplitToningHighlightSaturation", "ColorGradeHighlightSat"), 0, 100),
        ("highlight_lum", ("ColorGradeHighlightLum",), -100, 100),
        ("blending", ("ColorGradeBlending",), 0, 100),
        ("balance", ("SplitToningBalance", "ColorGradeBalance"), -100, 100),
    ):
        for key in keys:
            if key in n:
                values[field] = max(low, min(high, n[key]))
                break
    return ColorGradingControls(**values)
