from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .color import (
    EPS,
    HUE_CENTERS,
    hue_weights,
    linear_to_srgb,
    luma,
    rgb_to_hsl,
    hsl_to_rgb,
    srgb_to_linear,
    smoothstep,
)
from .layers import LayerMasks
from .local import apply_local
from .models import Recipe, RegionBox, TechnicalReport, ToneCurve


def apply_recipe(
    linear: np.ndarray,
    recipe: Recipe,
    layers: LayerMasks | None = None,
    technical: TechnicalReport | None = None,
    clutter_boxes: list[RegionBox] | None = None,
) -> np.ndarray:
    """Tone in display-referred sRGB; only exposure/WB stay in linear.

    Near-black pixels must not be revalued by huge channel ratios — that is
    what turned dark pond water into neon red.
    """
    r = recipe.clamped()
    rgb = np.maximum(linear.astype(np.float32, copy=True), 0.0)
    rgb = _white_balance(rgb, r.temperature, r.tint)
    if abs(r.exposure_ev) > 1e-4:
        rgb *= np.float32(2.0 ** r.exposure_ev)
        rgb = np.maximum(rgb, 0.0)

    img = np.clip(linear_to_srgb(rgb), 0.0, 1.0)
    if abs(r.highlights) > 1e-4 or abs(r.whites) > 1e-4:
        img = _highlights_whites(img, r.highlights, r.whites)
    if abs(r.shadows) > 1e-4 or abs(r.blacks) > 1e-4:
        img = _shadows_blacks(img, r.shadows, r.blacks)
    if abs(r.dehaze) > 1e-4:
        img = _dehaze(img, r.dehaze)
    if abs(r.contrast) > 1e-4:
        img = _contrast(img, r.contrast)
    if not r.curve.is_identity():
        img = _tone_curve(img, r.curve)
    if r.hsl:
        img = _hsl(img, r)
    if abs(r.saturation) > 1e-4 or abs(r.vibrance) > 1e-4:
        img = _sat_vib(img, r.saturation, r.vibrance)
    img = _grade(img, r)
    if r.color_grading.model_dump(exclude_none=True):
        img = _color_wheels(img, r)
    if abs(r.clarity) > 1e-4:
        img = _clarity(img, r.clarity)
    if abs(r.vignette) > 1e-4:
        img = _vignette(img, r.vignette)
    if r.grain > 1e-4:
        img = _grain(img, r.grain)
    if not r.local.is_noop() or r.masks:
        img = apply_local(img, r, layers=layers, technical=technical, clutter_boxes=clutter_boxes)
    img = np.clip(img, 0.0, 1.0)
    return srgb_to_linear(img)


def _color_wheels(img: np.ndarray, recipe: Recipe) -> np.ndarray:
    """Approximate native wheels for previews; final Lightroom pixels take precedence."""
    import colorsys

    y = luma(img)
    grading = recipe.color_grading
    out = img.copy()
    for prefix, weight in (("shadow", (1-y)**2), ("midtone", 4*y*(1-y)), ("highlight", y**2)):
        hue = getattr(grading, f"{prefix}_hue") or 0
        sat = getattr(grading, f"{prefix}_sat") or 0
        luminance = getattr(grading, f"{prefix}_lum") or 0
        color = np.array(colorsys.hsv_to_rgb(hue/360, 1, 1), dtype=np.float32)
        color -= float(color @ np.array([.2126, .7152, .0722]))
        # Fade at both endpoints to keep pure black and specular white neutral.
        chroma_weight = weight * (4*y*(1-y))
        out += chroma_weight[..., None] * color * (sat/100)*.42
        out += weight[..., None] * (luminance/100)*.15
    return np.clip(out, 0, 1)


def export_srgb_jpeg(linear: np.ndarray, path: Path, quality: int = 92, max_edge: int | None = None) -> None:
    from .ingest import srgb_profile_bytes
    rgb = linear
    if max_edge is not None:
        from .ingest import downscale_linear

        rgb = downscale_linear(rgb, max_edge)
    srgb = np.clip(linear_to_srgb(np.maximum(rgb, 0.0)), 0.0, 1.0)
    arr = (srgb * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=quality, optimize=True,
                                        subsampling=0, icc_profile=srgb_profile_bytes())


def _white_balance(rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    r_gain = 1.0 + 0.018 * temperature
    b_gain = 1.0 - 0.018 * temperature
    g_gain = 1.0 + 0.010 * tint
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    gains = np.clip(gains, 0.65, 1.5)
    gains /= float(np.mean(gains))
    return rgb * gains


def _highlights_whites(img: np.ndarray, highlights: float, whites: float) -> np.ndarray:
    y = luma(img)
    if highlights < 0:
        recover = -highlights
        shoulder = smoothstep(0.52, 0.98, y)
        new_y = y - recover * 0.55 * shoulder * y
        img = _revalue(img, y, new_y)
        y = luma(img)
    elif highlights > 0:
        shoulder = smoothstep(0.52, 0.95, y)
        img = _revalue(img, y, y + highlights * 0.32 * shoulder)
        y = luma(img)
    if whites != 0:
        wmask = smoothstep(0.74, 1.0, y)
        img = _revalue(img, y, y + whites * 0.32 * wmask)
    return np.clip(img, 0.0, 1.0)


def _shadows_blacks(img: np.ndarray, shadows: float, blacks: float) -> np.ndarray:
    y = luma(img)
    if shadows != 0:
        smask = 1.0 - smoothstep(0.05, 0.50, y)
        if shadows > 0:
            # Open mid-shadows, not the pond / hair / night dirt.
            smask = smask * smoothstep(0.035, 0.12, y)
        new_y = y + shadows * 0.52 * smask
        img = _revalue(img, y, new_y)
        y = luma(img)
    if blacks != 0:
        bmask = 1.0 - smoothstep(0.0, 0.20, y)
        if blacks > 0:
            bmask = bmask * smoothstep(0.02, 0.09, y)
        img = _revalue(img, y, y + blacks * 0.45 * bmask)
    return np.clip(img, 0.0, 1.0)


def _revalue(rgb: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Lift/crush luma without exploding channel ratios in the dirt."""
    new = np.clip(new, 0.0, 1.0)
    delta = (new - old)[..., None]
    # Equal-channel add keeps hue of near-black (pond, hair, night) stable.
    neutral = np.clip(rgb + delta, 0.0, 1.0)
    scale = np.clip(new / np.maximum(old, 0.08), 0.55, 1.7)
    ratio = np.clip(rgb * scale[..., None], 0.0, 1.0)
    w = smoothstep(0.04, 0.18, old)[..., None]
    return ratio * w + neutral * (1.0 - w)


def _dehaze(img: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, -0.30, 0.40))
    # Microcontrast + sat only. Do not stack another S-curve — that is how
    # water highlights went yellow and already-graded JPEGs went HDR.
    img = _clarity(img, amount * 0.55)
    y = luma(img)[..., None]
    img = y + (img - y) * (1.0 + amount * 0.32)
    return np.clip(img, 0.0, 1.0)


def _contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    y = luma(img)
    amount = float(np.clip(contrast, -0.85, 0.85))
    pivot = 0.45
    # Slider 0.20 should read like a finished grade, not a 2% nudge.
    gain = 1.0 + amount * 1.85
    new_y = (y - pivot) * gain + pivot
    if amount > 0:
        t = np.clip(y, 0.0, 1.0)
        s = t * t * (3.0 - 2.0 * t)
        new_y = new_y + amount * 0.28 * (s - t)
    return np.clip(_revalue(img, y, new_y), 0.0, 1.0)


def _tone_curve(img: np.ndarray, curve: ToneCurve) -> np.ndarray:
    out = img
    if curve.rgb:
        lut = _lut_from_points(curve.rgb)
        if lut is not None:
            out = _map_lut(out, lut)
    for i, pts in enumerate((curve.r, curve.g, curve.b)):
        if not pts:
            continue
        lut = _lut_from_points(pts)
        if lut is None:
            continue
        ch = out[..., i]
        out[..., i] = _map_lut(ch, lut)
    return np.clip(out, 0.0, 1.0)


def _lut_from_points(points: list[tuple[float, float]]) -> np.ndarray | None:
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    grid = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    return np.interp(grid, xs, ys).astype(np.float32)


def _map_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    x = np.clip(img, 0.0, 1.0) * 255.0
    i0 = np.minimum(x.astype(np.int32), 254)
    f = (x - i0.astype(np.float32)).astype(np.float32)
    return lut[i0] * (1.0 - f) + lut[np.minimum(i0 + 1, 255)] * f


def _sat_vib(img: np.ndarray, sat: float, vib: float) -> np.ndarray:
    y = luma(img)[..., None]
    # Saturation follows light: unlit regions stay quieter (Gore / 通透).
    light = np.clip((y - 0.05) / 0.52, 0.0, 1.0)
    sat_w = light if sat > 0 else np.maximum(light, 0.35)
    out = y + (img - y) * (1.0 + sat * sat_w)
    if abs(vib) > 1e-4:
        mag = np.linalg.norm(img - y, axis=-1, keepdims=True)
        weight = 1.0 - np.clip(mag * 2.4, 0.0, 1.0)
        if vib > 0:
            weight = weight * light
        out = y + (out - y) * (1.0 + vib * weight)
    return np.clip(out, 0.0, 1.0)


def _hsl(img: np.ndarray, recipe: Recipe) -> np.ndarray:
    h, s, l = rgb_to_hsl(img)
    alive = (l > 0.05) & (s > 0.04)
    if not np.any(alive):
        return img
    h2, s2, l2 = h.copy(), s.copy(), l.copy()
    touched = np.zeros(h.shape, dtype=np.float32)
    for name, shift in recipe.hsl.items():
        center = HUE_CENTERS.get(name)
        if center is None:
            continue
        if max(abs(shift.hue), abs(shift.sat), abs(shift.lum)) < 1e-4:
            continue
        w = hue_weights(h, center) * alive
        touched = np.maximum(touched, w)
        if abs(shift.hue) > 1e-4:
            h2 = (h2 + (shift.hue / 360.0) * w) % 1.0
        if abs(shift.sat) > 1e-4:
            if shift.sat > 0:
                light = np.clip((l - 0.06) / 0.50, 0.0, 1.0)
                s2 = np.clip(s2 * (1.0 + shift.sat * w * 1.15 * light), 0.0, 1.0)
            else:
                s2 = np.clip(s2 * (1.0 + shift.sat * w * 1.15), 0.0, 1.0)
        if abs(shift.lum) > 1e-4:
            l2 = np.clip(l2 + shift.lum * 0.34 * w, 0.0, 1.0)
    if float(np.max(touched)) < 1e-4:
        return img
    out = hsl_to_rgb(h2, s2, l2)
    mix = touched[..., None]
    return img * (1.0 - mix) + out * mix


def _grade(img: np.ndarray, recipe: Recipe) -> np.ndarray:
    g = recipe.grade
    y = luma(img)
    sm = 1.0 - smoothstep(0.06, 0.38, y)
    mm = smoothstep(0.22, 0.48, y) * (1.0 - smoothstep(0.55, 0.82, y))
    hm = smoothstep(0.58, 0.92, y)
    offset = (
        np.array(g.shadows, dtype=np.float32) * sm[..., None]
        + np.array(g.midtones, dtype=np.float32) * mm[..., None]
        + np.array(g.highlights, dtype=np.float32) * hm[..., None]
    )
    return np.clip(img + offset * 1.20, 0.0, 1.0)


def _clarity(img: np.ndarray, amount: float) -> np.ndarray:
    y = luma(img)
    u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32) / 255.0
    highpass = y - blurred
    new_y = y + amount * 1.05 * highpass
    return np.clip(_revalue(img, y, new_y), 0.0, 1.0)


def _vignette(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (yy - cy) / max(cy, 1.0)
    nx = (xx - cx) / max(cx, 1.0)
    d = np.sqrt(nx * nx + ny * ny)
    falloff = np.clip(d / 1.35, 0.0, 1.0)
    falloff = falloff * falloff
    if amount >= 0:
        factor = 1.0 - amount * 0.95 * falloff
    else:
        factor = 1.0 + (-amount) * 0.32 * falloff
    return np.clip(img * factor[..., None].astype(np.float32), 0.0, 1.0)


def _grain(img: np.ndarray, amount: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    y = luma(img)
    noise = rng.normal(0.0, 1.0, y.shape).astype(np.float32)
    envelope = smoothstep(0.08, 0.22, y) * (1.0 - smoothstep(0.7, 0.96, y))
    grain = noise * amount * 0.045 * envelope
    return np.clip(img + grain[..., None], 0.0, 1.0)
