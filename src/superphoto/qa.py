"""Perceptual safety checks for an automatic photographic grade.

The score rewards a useful, restrained improvement. It intentionally does not
reward a large difference from the source: subtle work is valid, while crushed
shadows, clipped highlights, channel clipping and excessive contrast are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .color import linear_to_srgb, luma, rgb_to_hsl
from .models import HueShift, Recipe, TechnicalReport, VariantPlan
from .preference import (
    _adapt_to_scene,
    _fit_lut_to_scene,
    _scale_recipe,
    _why,
    local_edit_for,
    protect_look,
    honor_locked_controls,
    sanitize_correction,
)
from .recipes import get_look
from .workflow import apply_ethics, guard_hues

_TARGET = {"A": 82.0, "B": 84.0, "C": 84.0}


@dataclass
class QAReport:
    score: float
    presence: float
    safety: float
    finish: float
    mae: float
    too_weak: bool
    too_strong: bool
    notes: list[str] = field(default_factory=list)


def score_grade(
    source_linear: np.ndarray,
    graded_linear: np.ndarray,
    technical: TechnicalReport,
    slot: str,
) -> QAReport:
    src = _thumb(linear_to_srgb(np.maximum(source_linear, 0.0)))
    dst = _thumb(linear_to_srgb(np.maximum(graded_linear, 0.0)))
    h = min(src.shape[0], dst.shape[0])
    w = min(src.shape[1], dst.shape[1])
    src, dst = src[:h, :w], dst[:h, :w]

    mae = float(np.mean(np.abs(dst - src)))
    y0, y1 = luma(src), luma(dst)
    std0 = float(np.std(y0))
    std1 = float(np.std(y1))
    gain = std1 / std0 if std0 > 1e-5 else 1.0
    d_luma = float(np.mean(y1) - np.mean(y0))
    sat0, sat1 = _sat(src), _sat(dst)

    presence = _presence_score(mae, gain, slot)
    safety, notes = _safety_score(src, dst, y0, y1, technical, d_luma, sat0, sat1, gain)
    finish = _finish_score(gain, mae, slot, y0, y1)
    # Safety dominates. A conspicuous but damaged image must never win.
    score = 0.20 * presence + 0.58 * safety + 0.22 * finish
    too_weak = slot != "A" and mae < 0.012
    sky_blown = any("天空过曝" in note for note in notes)
    too_strong = safety < 70 or mae > 0.145 or sky_blown
    if too_strong:
        too_weak = False
    if too_weak:
        notes.append("调整很轻，仍接近原片")
    if 0.02 <= mae <= 0.10:
        notes.append("调整幅度自然")

    return QAReport(
        score=float(np.clip(score, 0, 100)),
        presence=presence,
        safety=safety,
        finish=finish,
        mae=mae,
        too_weak=too_weak,
        too_strong=too_strong,
        notes=notes[:6],
    )


def target_for(slot: str) -> float:
    return _TARGET.get(slot, 84.0)


def refine_plan(
    plan: VariantPlan,
    report: QAReport,
    critique,
    technical: TechnicalReport,
    attempt: int,
) -> VariantPlan:
    """Prefer backing off. Only an almost-identical look receives a mild lift."""
    if plan.lut_mix == 0 or plan.method_id.startswith("preset:"):
        if not report.too_strong:
            return plan
        from .preference import reduce_style_strength
        recipe = (reduce_style_strength(plan.recipe, plan.primary, .78) if plan.method_id.startswith("preset:")
                  else _scale_recipe(plan.recipe, .78))
        recipe = recipe.model_copy(update={name: getattr(plan.recipe, name).model_copy(deep=True)
                                           for name in ("detail", "optics", "crop")})
        return plan.model_copy(update={"recipe": honor_locked_controls(recipe, plan.intent)})
    look = get_look(plan.look_id)
    mix = float(plan.lut_mix)
    if report.too_strong or report.safety < target_for(plan.slot):
        mix = max(0.18, mix * (0.72 if attempt == 0 else 0.62))
        extra = Recipe(
            highlights=-0.04,
            shadows=0.035 if technical.mean_luma > 0.12 else 0.0,
            blacks=0.025,
            contrast=-0.035,
            saturation=-0.025,
            vibrance=-0.015,
            clarity=-0.025,
            dehaze=-0.015,
        )
        if any("天空" in note for note in report.notes):
            extra = extra.merged(
                Recipe(
                    highlights=-0.16,
                    whites=-0.12,
                    exposure_ev=-0.04,
                    hsl={
                        "aqua": HueShift(lum=-0.10, sat=0.02),
                        "blue": HueShift(lum=-0.08),
                    },
                )
            )
    else:
        # A small difference is not evidence of an unsuccessful edit.
        return plan

    if look.source == "master":
        from .masters import recipe_for_photo
        base_recipe = recipe_for_photo(look.id, technical)
    else:
        base_recipe = look.recipe
    look_recipe = _scale_recipe(base_recipe, mix)
    look_recipe = _fit_lut_to_scene(look_recipe, look.id, technical).merged(extra)
    look_recipe = protect_look(look_recipe, plan.intent)
    correction = sanitize_correction(critique.correction, technical).as_recipe()
    correction = correction.model_copy(update={"masks": [m.model_copy(deep=True) for m in plan.recipe.masks]})
    recipe = _adapt_to_scene(correction.merged(look_recipe), technical, look.id)
    recipe = recipe.model_copy(
        update={
            "local": recipe.local.merged(
                local_edit_for(
                    plan.slot,
                    technical,
                    genre=plan.genre,
                    has_clutter=bool(getattr(critique, "clutter", None)),
                )
            )
        }
    ).clamped()
    recipe = apply_ethics(guard_hues(recipe, technical, plan.ethics), plan.ethics)
    recipe = honor_locked_controls(recipe, plan.intent)
    recipe = recipe.model_copy(update={
        "detail": plan.recipe.detail.model_copy(deep=True),
        "optics": plan.recipe.optics.model_copy(deep=True),
        "crop": plan.recipe.crop.model_copy(deep=True),
    })
    why = _why(plan.slot, look, critique, mix, plan.geometry)
    return plan.model_copy(update={"recipe": recipe, "lut_mix": mix, "why": why, "primary": correction})


def _presence_score(mae: float, gain: float, slot: str) -> float:
    # Broad plateau: restrained edits are not punished for being restrained.
    ideal = 0.028 if slot == "A" else 0.055
    spread = 0.075 if slot == "A" else 0.085
    score = 96.0 - abs(mae - ideal) / spread * 34.0
    if gain < 0.72 or gain > 1.32:
        score -= 12.0
    return float(np.clip(score, 45, 98))


def _safety_score(
    src: np.ndarray,
    dst: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    technical: TechnicalReport,
    d_luma: float,
    sat0: float,
    sat1: float,
    gain: float,
) -> tuple[float, list[str]]:
    score = 100.0
    notes: list[str] = []

    # Measure newly lost tonal detail, not just absolute black/white pixels.
    source_detail = (y0 > 0.025) & (y0 < 0.22)
    if np.any(source_detail):
        crushed = float(np.mean(y1[source_detail] < 0.018))
        if crushed > 0.08:
            score -= min(42.0, 12.0 + crushed * 80.0)
            notes.append("暗部层次损失明显")

    source_high = (y0 > 0.65) & (y0 < 0.985)
    if np.any(source_high):
        blown = float(np.mean(y1[source_high] > 0.992))
        if blown > 0.06:
            score -= min(34.0, 10.0 + blown * 65.0)
            notes.append("亮部纹理被冲白")

    sky0 = _sky_mask(src, y0, technical)
    if float(np.mean(sky0)) > 0.02:
        y0s, y1s = y0[sky0], y1[sky0]
        lift = float(np.mean(y1s) - np.mean(y0s))
        hot0 = float(np.mean(y0s > 0.90))
        hot1 = float(np.mean(y1s > 0.90))
        clip_sky = float(np.mean(y1s > 0.97))
        # Blown sky often desaturates, so hue-based sky checks miss it.
        if hot1 > hot0 + 0.08 or lift > 0.035 or clip_sky > 0.04:
            score -= min(52.0, 18.0 + max(0.0, hot1 - hot0) * 55.0 + max(0.0, lift) * 200.0 + clip_sky * 90.0)
            notes.append("天空过曝，亮部没有层次")

    clip0 = float(np.mean((src < 0.004) | (src > 0.996)))
    clip1 = float(np.mean((dst < 0.004) | (dst > 0.996)))
    if clip1 > clip0 + 0.055:
        score -= min(28.0, 10.0 + (clip1 - clip0) * 90.0)
        notes.append("颜色通道出现额外裁切")

    dark = y0 < 0.05
    if np.any(dark):
        src_w, dst_w = src[dark], dst[dark]
        g0 = max(float(np.mean(src_w[..., 1])), 1e-6)
        g1 = max(float(np.mean(dst_w[..., 1])), 1e-6)
        rg0 = float(np.mean(src_w[..., 0])) / g0
        rg1 = float(np.mean(dst_w[..., 0])) / g1
        if float(np.mean(y1[dark])) > 0.045 and rg1 > max(rg0 * 1.45, 1.25):
            score -= 38
            notes.append("暗部出现明显偏色")

    if gain > 1.34:
        score -= min(32.0, 12.0 + (gain - 1.34) * 70.0)
        notes.append("整体反差过硬")
    elif gain < 0.68:
        score -= 18.0
        notes.append("层次被压平")
    if abs(d_luma) > 0.12:
        score -= 16.0
        notes.append("整体亮度变化过大")
    if sat1 - sat0 > 0.11:
        score -= 18.0
        notes.append("颜色过浓")
    if sat0 - sat1 > 0.12:
        score -= min(32.0, 8.0 + (sat0 - sat1) * 140.0)
        notes.append("颜色被抽干")
    if d_luma < -0.08 and float(np.mean(y0)) > 0.50:
        score -= min(26.0, 8.0 + abs(d_luma) * 180.0)
        notes.append("画面被压得发闷")
    h_top = max(1, y0.shape[0] // 3)
    top0, top1 = y0[:h_top], y1[:h_top]
    std0_top = float(np.std(top0))
    std1_top = float(np.std(top1))
    if (
        technical.source_kind in {"jpeg", "png"}
        and std0_top > 0.04
        and std1_top > std0_top * 1.22
        and technical.highlight_clip < 0.04
    ):
        score -= 14.0
        notes.append("天空或雾气被修得过硬")

    h0, s0, l0 = rgb_to_hsl(src)
    h1, s1, _l1 = rgb_to_hsl(dst)
    hue_delta = np.abs(((h1 - h0 + 0.5) % 1.0) - 0.5)

    if technical.skin_fraction > 0.08:
        hue, sat, lig = rgb_to_hsl(dst)
        skin = (hue > 0.02) & (hue < 0.13) & (sat > 0.08) & (lig > 0.18) & (lig < 0.85)
        if float(np.mean(skin)) < technical.skin_fraction * 0.45:
            score -= 24.0
            notes.append("肤色自然度下降")
        skin0 = (h0 > 0.02) & (h0 < 0.13) & (s0 > 0.08) & (l0 > 0.18) & (l0 < 0.85)
        if np.any(skin0) and float(np.mean(hue_delta[skin0])) > 0.035:
            score -= 16.0
            notes.append("肤色偏离自然范围")

    if technical.green_fraction > 0.12:
        foliage = (h0 > 0.22) & (h0 < 0.45) & (s0 > 0.15) & (l0 > 0.08)
        if np.any(foliage):
            if float(np.mean(hue_delta[foliage])) > 0.045:
                score -= 16.0
                notes.append("植被颜色偏离自然色")
            if float(np.mean(s1[foliage]) - np.mean(s0[foliage])) > 0.08:
                score -= 14.0
                notes.append("绿色过饱和")

    if technical.top_third_blue > 0.14:
        sky = (h0 > 0.50) & (h0 < 0.72) & (s0 > 0.08)
        if np.any(sky) and float(np.mean(hue_delta[sky])) > 0.05:
            score -= 12.0
            notes.append("天空颜色偏离自然色")
    return float(np.clip(score, 0, 100)), notes


def _finish_score(gain: float, mae: float, slot: str, y0: np.ndarray, y1: np.ndarray) -> float:
    h, w = y0.shape
    inner = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    pop0 = float(np.mean(y0[inner]) - np.mean(y0))
    pop1 = float(np.mean(y1[inner]) - np.mean(y1))
    subject_gain = float(np.clip((pop1 - pop0) * 260, -8, 10))
    natural_contrast = max(0.0, abs(gain - 1.10) - 0.18) * 90.0
    return float(np.clip(86 + subject_gain - natural_contrast - max(0.0, mae - 0.11) * 140, 35, 96))


def _sky_mask(src: np.ndarray, y0: np.ndarray, technical: TechnicalReport) -> np.ndarray:
    """Sky from the *source*, so a washed-out destination still counts."""
    h, w = y0.shape
    h_top = max(1, h // 3)
    hue, sat, _lig = rgb_to_hsl(src)
    floor = max(0.52, float(np.percentile(y0, 55)))
    bright = y0[:h_top] > floor
    blueish = (
        (hue[:h_top] > 0.42)
        & (hue[:h_top] < 0.80)
        & (sat[:h_top] > 0.03)
        & (y0[:h_top] > 0.38)
    )
    top = bright | blueish
    if technical.top_third_blue > 0.10:
        top = top | ((y0[:h_top] > 0.55) & (sat[:h_top] < 0.38))
    mask = np.zeros_like(y0, dtype=bool)
    mask[:h_top] = top
    return mask


def _sat(img: np.ndarray) -> float:
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    return float(np.mean((mx - mn) / np.maximum(mx, 1e-6)))


def _thumb(srgb: np.ndarray, edge: int = 512) -> np.ndarray:
    h, w = srgb.shape[:2]
    m = max(h, w)
    if m <= edge:
        return np.clip(srgb, 0.0, 1.0).astype(np.float32)
    scale = edge / m
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    ys = np.linspace(0, h - 1, nh).astype(np.int32)
    xs = np.linspace(0, w - 1, nw).astype(np.int32)
    return np.clip(srgb[ys][:, xs], 0.0, 1.0).astype(np.float32)
