"""Layered local finish: clean color, subject pop, suppress clutter.

This is photographic suppression, not generative erase. Distractions are
darkened/desaturated (and cropped when the geometry pass agrees). Tiny
specks are healed by blending in a blur of their neighborhood.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .color import circular_mean_hue, hue_weights, luma, rgb_to_hsl, hsl_to_rgb, linear_to_srgb, srgb_to_linear
from .layers import LayerMasks, build_layers
from .models import LocalEdit, Recipe, RegionBox, TechnicalReport


def apply_local(
    img: np.ndarray,
    recipe: Recipe,
    layers: LayerMasks | None = None,
    technical: TechnicalReport | None = None,
    clutter_boxes: list[RegionBox] | None = None,
) -> np.ndarray:
    loc = recipe.local
    # Explicit per-picture intent replaces, rather than doubles, an automatic
    # heuristic for the same region, matching the Lightroom mask adapter.
    groups = {"subject": ("subject_lift", "subject_vibrance", "subject_clarity", "subject_sharpen"),
              "sky": ("sky_dehaze",), "background": ("background_darken", "background_desat")}
    loc = loc.model_copy(update={key: 0.0 for adjustment in recipe.masks for key in groups[adjustment.target]})
    if loc.is_noop() and not recipe.masks:
        return img
    img = np.clip(img.astype(np.float32, copy=True), 0.0, 1.0)
    if layers is None:
        layers = build_layers(img, technical, clutter_boxes)
    out = img
    if loc.defect_heal > 1e-4 and float(np.max(layers.defects)) > 1e-4:
        out = _heal(out, layers.defects * loc.defect_heal)
    if loc.color_clean > 1e-4:
        out = _color_clean(out, layers, loc.color_clean)
    if loc.sky_dehaze > 1e-4 and float(np.max(layers.sky)) > 1e-4:
        out = _sky(out, layers.sky, loc.sky_dehaze)
    if abs(loc.background_darken) > 1e-4 or loc.background_desat > 1e-4:
        out = _background(out, layers, loc)
    if loc.clutter_suppress > 1e-4 and float(np.max(layers.clutter)) > 1e-4:
        out = _clutter(out, layers.clutter, loc.clutter_suppress)
    if loc.subject_lift or loc.subject_vibrance or loc.subject_clarity or loc.subject_sharpen:
        out = _subject(out, layers.subject, loc)
    for adjustment in recipe.masks:
        mask = getattr(layers, adjustment.target)
        from .render import _white_balance, _highlights_whites, _shadows_blacks, _dehaze, _contrast

        adjusted = srgb_to_linear(out)
        adjusted = _white_balance(adjusted, adjustment.temperature, adjustment.tint)
        adjusted = linear_to_srgb(adjusted * (2.0 ** adjustment.exposure_ev))
        adjusted = _highlights_whites(adjusted, adjustment.highlights, adjustment.whites)
        adjusted = _shadows_blacks(adjusted, adjustment.shadows, adjustment.blacks)
        if adjustment.contrast:
            adjusted = _contrast(adjusted, adjustment.contrast)
        if adjustment.dehaze:
            adjusted = _dehaze(adjusted, adjustment.dehaze)
        # The fallback preview uses the same EV semantics. Native Lightroom
        # masks remain authoritative when Lightroom is the delivery engine.
        if adjustment.saturation:
            gray = luma(adjusted)[..., None]
            adjusted = gray + (adjusted - gray) * (1.0 + adjustment.saturation)
        out = out * (1.0 - mask[..., None]) + adjusted * mask[..., None]
        if adjustment.clarity:
            out = _masked_highpass(out, mask, adjustment.clarity, radius=9)
        if adjustment.texture:
            out = _masked_highpass(out, mask, adjustment.texture, radius=2)
    return np.clip(out, 0.0, 1.0)


def _color_clean(img: np.ndarray, layers: LayerMasks, amount: float) -> np.ndarray:
    h, s, l = rgb_to_hsl(img)
    weights = layers.subject * np.clip(s, 0.0, 1.0)
    if float(np.mean(layers.subject)) < 0.02 or float(np.sum(weights)) < 1e-3:
        center = 0.08
    else:
        center = circular_mean_hue(h, weights)
    keep = hue_weights(h, center * 360.0, 75.0)
    keep = np.maximum(keep, layers.skin * hue_weights(h, 30.0, 45.0))
    keep = np.maximum(keep, layers.sky * hue_weights(h, 220.0, 55.0))
    keep = np.maximum(keep, layers.water * hue_weights(h, 180.0, 50.0) * 0.7)
    reduce = (1.0 - keep) * (0.28 + 0.72 * layers.background)
    s2 = np.clip(s * (1.0 - amount * 0.82 * reduce), 0.0, 1.0)
    out = hsl_to_rgb(h, s2, l)
    mix = np.clip(amount * (0.35 + 0.65 * layers.background), 0.0, 1.0)[..., None]
    return img * (1.0 - mix) + out * mix


def _sky(img: np.ndarray, sky: np.ndarray, amount: float) -> np.ndarray:
    y = luma(img)
    gray = y[..., None]
    contrast = gray + (img - gray) * (1.0 + amount * 0.22)
    y2 = np.clip(y - amount * 0.08 * sky * np.clip(y - 0.55, 0.0, 1.0), 0.0, 1.0)
    lifted = _add_luma(contrast, y2)
    m = sky[..., None] * amount
    return img * (1.0 - m) + lifted * m


def _background(img: np.ndarray, layers: LayerMasks, loc: LocalEdit) -> np.ndarray:
    y = luma(img)
    bg = layers.background * (1.0 - 0.35 * layers.sky)
    out = img
    if loc.background_desat > 1e-4:
        gray = y[..., None]
        desat = gray + (img - gray) * (1.0 - loc.background_desat * 0.7)
        m = bg[..., None] * loc.background_desat
        out = out * (1.0 - m) + desat * m
    if loc.background_darken > 1e-4:
        y = luma(out)
        new_y = y - loc.background_darken * 0.22 * bg
        out = _add_luma(out, new_y)
    return np.clip(out, 0.0, 1.0)


def _clutter(img: np.ndarray, clutter: np.ndarray, amount: float) -> np.ndarray:
    y = luma(img)
    gray = y[..., None]
    quiet = gray + (img - gray) * (1.0 - amount * 0.85)
    quiet = _add_luma(quiet, y - amount * 0.16 * clutter)
    blurred = _blur_rgb(img, 2.4)
    quiet = quiet * (1.0 - amount * 0.35) + blurred * (amount * 0.35)
    m = (clutter * amount)[..., None]
    return np.clip(img * (1.0 - m) + quiet * m, 0.0, 1.0)


def _subject(img: np.ndarray, subject: np.ndarray, loc: LocalEdit) -> np.ndarray:
    out = img
    m = subject[..., None]
    if loc.subject_lift > 1e-4:
        y = luma(out)
        out = _add_luma(out, y + loc.subject_lift * 0.16 * subject)
    if loc.subject_vibrance > 1e-4:
        y = luma(out)[..., None]
        vib = y + (out - y) * (1.0 + loc.subject_vibrance * 0.55)
        out = out * (1.0 - m) + vib * m
    if loc.subject_clarity > 1e-4:
        out = _masked_highpass(out, subject, loc.subject_clarity, radius=9)
    if loc.subject_sharpen > 1e-4:
        out = _masked_highpass(out, subject, loc.subject_sharpen * 0.85, radius=1.4)
    return np.clip(out, 0.0, 1.0)


def _heal(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blurred = _blur_rgb(img, 5.0)
    m = np.clip(mask, 0.0, 1.0)[..., None]
    return img * (1.0 - m) + blurred * m


def _masked_highpass(img: np.ndarray, mask: np.ndarray, amount: float, radius: float) -> np.ndarray:
    y = luma(img)
    u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32) / 255.0
    new_y = y + amount * 0.95 * (y - blurred)
    mixed = _add_luma(img, new_y)
    m = mask[..., None]
    return img * (1.0 - m) + mixed * m


def _add_luma(img: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    old = luma(img)
    delta = (np.clip(new_y, 0.0, 1.0) - old)[..., None]
    # Equal-channel add keeps near-black hue stable.
    return np.clip(img + delta, 0.0, 1.0)


def _blur_rgb(img: np.ndarray, radius: float) -> np.ndarray:
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(out, dtype=np.float32) / 255.0
