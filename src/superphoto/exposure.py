"""Evidence and composable tone correction for developed photographs."""
from __future__ import annotations

import numpy as np

from .models import TechnicalReport, Recipe, ToneCurve
from .color import linear_to_srgb, srgb_to_linear


def developed_overexposure(t: TechnicalReport) -> bool:
    """Require bright usable content, not just white sky, snow or a white wall."""
    usable = t.unclipped_median_luma or 0
    chromatic = t.chromatic_median_luma or 0
    clipped_bright = (t.highlight_clip > .04 and usable > .30
                      and t.chromatic_fraction > .12 and chromatic > .28)
    broad_bright = (usable > .60 and t.chromatic_fraction > .35 and chromatic > .55)
    return (t.source_kind in {"jpeg", "png"} and t.unclipped_fraction > .35
            and t.shadow_clip < .10 and (clipped_bright or broad_bright))


def deep_low_key_headroom(t: TechnicalReport) -> bool:
    """Allow a small requested lift only when almost the whole JPEG is dark.

    This measures tonal headroom, not whether darkness was the photographer's
    intention. Ordinary low-key frames retain the existing stricter limit.
    """
    return (t.source_kind in {"jpeg", "png"} and t.mean_luma < .025
            and t.median_luma < .012 and t.shadow_clip > .65
            and t.highlight_p95 < .12 and t.highlight_clip < .005
            and t.black_clip < .55)


def developed_lift_ev(t: TechnicalReport) -> float:
    """Bound a local-fallback lift to measured, unused highlight headroom.

    This is tonal evidence, not recognition of a person or photographic intent.
    Exclude deep night, strong silhouettes, high-key scenes and old reports that
    did not measure usable pixels. Explicit intent/zero controls remain upstream.
    """
    if not (t.source_kind in {"jpeg", "png"}
            and .075 <= t.median_luma < .16 and .08 <= t.mean_luma < .24
            and .15 <= t.highlight_p95 < .40 and t.highlight_clip < .002
            and t.white_clip_fraction < .002 and t.black_clip < .02
            and t.shadow_clip < .30 and t.unclipped_fraction > .65):
        return 0.0
    return float(np.clip(min(np.log2(.18 / t.median_luma),
                            np.log2(.48 / t.highlight_p95)), 0, .55))


def has_tone_correction(primary: Recipe, t: TechnicalReport) -> bool:
    return developed_overexposure(t) and t.white_clip_fraction >= .03 and not primary.curve.is_identity()


def white_preserving_curve(ev: float) -> ToneCurve:
    """Negative EV through midtones, returning to white without inventing detail."""
    x = np.array([0, .10, .18, .35, .50, .65, .80, .90, 1.0])
    y = linear_to_srgb(srgb_to_linear(x) * 2 ** ev).astype(float)
    shoulder = np.clip((x - .75) / .25, 0, 1)
    shoulder = shoulder * shoulder * (3 - 2 * shoulder)
    y += (x - y) * shoulder
    return ToneCurve(rgb=list(zip(x.tolist(), y.tolist())))


def _points(points):
    return np.array(points or [(0, 0), (1, 1)], dtype=float).T


def compose_rgb_curves(base: list, style: list) -> list[tuple[float, float]]:
    """Exact piecewise-linear style(base(x)), including preimages of style knots."""
    bx, by = _points(base)
    sx, sy = _points(style)
    xs = set(bx.tolist())
    for x0, x1, y0, y1 in zip(bx[:-1], bx[1:], by[:-1], by[1:]):
        if y1 > y0:
            for knot in sx[(sx > y0) & (sx < y1)]:
                xs.add(float(x0 + (knot - y0) * (x1 - x0) / (y1 - y0)))
    xs = np.array(sorted(xs))
    ys = np.interp(np.interp(xs, bx, by), sx, sy)
    return list(zip(xs.tolist(), ys.tolist()))


def merge_correction_style(primary: Recipe, style: Recipe, t: TechnicalReport) -> Recipe:
    """Compose the base and look once. Later guards only project, never reapply."""
    result = primary.merged(style).clamped()
    if has_tone_correction(primary, t):
        result.curve = style.curve.model_copy(deep=True)
        result.curve.rgb = compose_rgb_curves(primary.curve.rgb, style.curve.rgb)
    return result


def blend_tone_curves(base: ToneCurve, full: ToneCurve, amount: float) -> ToneCurve:
    """Reduce style toward the corrected base, not toward uncorrected identity."""
    result = ToneCurve()
    for channel in ("rgb", "r", "g", "b"):
        a, b = getattr(base, channel), getattr(full, channel)
        if not a and not b:
            continue
        ax, ay = _points(a)
        bx, by = _points(b)
        xs = np.unique(np.r_[ax, bx])
        ys = np.interp(xs, ax, ay) * (1 - amount) + np.interp(xs, bx, by) * amount
        setattr(result, channel, list(zip(xs.tolist(), ys.tolist())))
    return result


def preserve_exposure_correction(recipe: Recipe, primary: Recipe, t: TechnicalReport) -> Recipe:
    """Protect correction headroom while retaining each look's tone and palette.

    Project an already composed curve onto an envelope around the base. This
    operation is idempotent: retrying or validating cannot compound exposure.
    Endpoint anchoring protects clipped whites; actual exports still pass QA.
    """
    if not has_tone_correction(primary, t):
        return recipe
    result = recipe.model_copy(deep=True)
    bx, by = _points(primary.curve.rgb)
    cx, cy = _points(result.curve.rgb or primary.curve.rgb)
    xs = np.unique(np.r_[bx, cx])
    base_y = np.interp(xs, bx, by)
    # Permit meaningful style density without losing the common exposure move.
    width = .085 * np.sin(np.pi * xs)
    lower = np.maximum(0, base_y - width)
    upper = np.minimum(1, base_y + width)
    upper[xs == 0] = .02
    ys = np.maximum.accumulate(np.clip(np.interp(xs, cx, cy), lower, upper))
    ys[-1] = 1.0
    result.curve.rgb = list(zip(xs.tolist(), ys.tolist()))
    # Keep channel palettes in the body, but don't tint a clipped white plateau.
    for channel in ("r", "g", "b"):
        points = getattr(result.curve, channel)
        if points:
            px, py = _points(points)
            knots = np.unique(np.r_[px, 0, 1])
            values = np.interp(knots, px, py)
            values[0] = min(values[0], .02)
            values[-1] = 1.0
            setattr(result.curve, channel, list(zip(knots.tolist(), np.maximum.accumulate(values).tolist())))
    result.exposure_ev = 0.0
    # On a clipped developed plateau these sliders grey whites before the RGB
    # curve can anchor its endpoint. Shape highlight density in the composed
    # curve, while leaving color, shadow density and local contrast available.
    result.highlights = 0.0
    result.whites = 0.0
    result.parametric_tone.highlights = 0.0
    result.contrast = float(np.clip(result.contrast, -.12, .22))
    result.shadows = float(np.clip(result.shadows, -.12, .22))
    result.blacks = float(np.clip(result.blacks, -.12, .08))
    return result
