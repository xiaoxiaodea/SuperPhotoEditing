"""Self-check a grade the way a picker would: finished, not broken.

Score is 0-100. Slot B/C should land in the high 80s; if not, the pipeline
retries with a stronger or safer recipe instead of shipping a 30-point frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .color import linear_to_srgb, luma, rgb_to_hsl
from .models import Recipe, TechnicalReport, VariantPlan
from .preference import (
    _adapt_to_scene,
    _fit_lut_to_scene,
    _scale_recipe,
    _why,
    local_edit_for,
    sanitize_correction,
)
from .recipes import get_look

# Presence floor: below this, a viewer says "nothing happened".
_WEAK_MAE = 0.046
_STRONG_MAE = 0.17
_TARGET = {"A": 72.0, "B": 88.0, "C": 90.0}
_MAX_TRIES = 3

_PRESENCE = Recipe(
    contrast=0.16,
    clarity=0.16,
    blacks=-0.10,
    whites=0.06,
    dehaze=0.08,
    vignette=0.12,
    vibrance=0.06,
)


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
    y0 = luma(src)
    y1 = luma(dst)
    std0 = float(np.std(y0) + 1e-6)
    std1 = float(np.std(y1))
    gain = std1 / std0
    d_luma = float(np.mean(y1) - np.mean(y0))
    sat0 = _sat(src)
    sat1 = _sat(dst)

    presence = _presence_score(mae, gain, slot)
    safety, s_notes = _safety_score(src, dst, y0, y1, technical, d_luma, sat0, sat1)
    finish = _finish_score(gain, mae, slot, y0, y1)

    score = 0.40 * presence + 0.36 * safety + 0.24 * finish
    target = _TARGET.get(slot, 88.0)
    too_weak = slot != "A" and (mae < _WEAK_MAE or (gain < 1.05 and mae < 0.07) or score < target - 18)
    too_strong = safety < 58 or mae > _STRONG_MAE
    if too_strong:
        too_weak = False

    notes = list(s_notes)
    if mae < _WEAK_MAE and slot != "A":
        notes.append(f"和原片几乎一样（Δ={mae:.3f}），看起来没调")
    elif mae > 0.06:
        notes.append(f"成片差可见（Δ={mae:.3f}）")
    if gain > 1.08:
        notes.append("中灰对比起来了")
    elif gain < 1.03 and slot != "A":
        notes.append("对比几乎没动")

    return QAReport(
        score=float(np.clip(score, 0, 100)),
        presence=presence,
        safety=safety,
        finish=finish,
        mae=mae,
        too_weak=too_weak,
        too_strong=too_strong,
        notes=notes[:5],
    )


def target_for(slot: str) -> float:
    return _TARGET.get(slot, 88.0)


def refine_plan(
    plan: VariantPlan,
    report: QAReport,
    critique,
    technical: TechnicalReport,
    attempt: int,
) -> VariantPlan:
    """Nudge the recipe after a failed self-check."""
    look = get_look(plan.look_id)
    mix = float(plan.lut_mix)
    extra = Recipe()
    if report.too_strong:
        mix = max(0.22, mix * 0.70)
        extra = Recipe(saturation=-0.04, vibrance=-0.04, contrast=-0.03)
    elif report.too_weak:
        mix = min(0.88, mix * (1.28 + 0.12 * attempt) if look.source == "xmp" else mix)
        extra = _PRESENCE if attempt == 0 else Recipe(
            contrast=0.20,
            clarity=0.18,
            blacks=-0.12,
            whites=0.08,
            dehaze=0.10,
            vignette=0.14,
            vibrance=0.08,
        )
        if look.id == "faithful":
            extra = Recipe(contrast=0.10, blacks=-0.06, whites=0.04, clarity=0.08)
    else:
        # Close to the target: add print snap, don't yank the LUT back.
        mix = min(0.88, mix * 1.12 if look.source == "xmp" else mix)
        extra = Recipe(contrast=0.10, clarity=0.12, blacks=-0.07, whites=0.05, dehaze=0.06)
    look_recipe = look.recipe if mix >= 0.995 else _scale_recipe(look.recipe, mix)
    look_recipe = _fit_lut_to_scene(look_recipe, look.id, technical).merged(extra)
    correction = sanitize_correction(critique.correction, technical).as_recipe()
    recipe = _adapt_to_scene(correction.merged(look_recipe), technical, look.id)
    recipe = recipe.model_copy(update={"local": recipe.local.merged(local_edit_for(plan.slot, technical))}).clamped()
    why = _why(plan.slot, look, critique, mix, plan.geometry)
    return plan.model_copy(update={"recipe": recipe, "lut_mix": mix, "why": why})


def _presence_score(mae: float, gain: float, slot: str) -> float:
    if slot == "A":
        # Control: a modest change is enough.
        if mae < 0.012:
            return 55.0
        return float(np.clip(70 + (mae - 0.02) * 400, 60, 92))
    # 0.02 → 28, 0.05 → 72, 0.08 → 92, 0.13 → 88, 0.20 → 55
    if mae < 0.02:
        mae_s = 20 + mae * 400
    elif mae < 0.08:
        mae_s = 28 + (mae - 0.02) / 0.06 * 64
    elif mae < 0.14:
        mae_s = 92 - (mae - 0.08) / 0.06 * 8
    else:
        mae_s = 84 - (mae - 0.14) * 200
    gain_s = float(np.clip(40 + (gain - 1.0) * 350, 20, 95))
    if gain > 1.35:
        gain_s = 70
    return float(np.clip(0.7 * mae_s + 0.3 * gain_s, 0, 100))


def _safety_score(
    src: np.ndarray,
    dst: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    technical: TechnicalReport,
    d_luma: float,
    sat0: float,
    sat1: float,
) -> tuple[float, list[str]]:
    score = 100.0
    notes: list[str] = []
    dark = y0 < 0.05
    if np.any(dark):
        src_w = src[dark]
        dst_w = dst[dark]
        g0 = max(float(np.mean(src_w[..., 1])), 1e-6)
        g1 = max(float(np.mean(dst_w[..., 1])), 1e-6)
        rg0 = float(np.mean(src_w[..., 0])) / g0
        rg1 = float(np.mean(dst_w[..., 0])) / g1
        lift = float(np.mean(y1[dark]))
        visible = lift > 0.045 and float(np.mean(dst_w[..., 0])) > 0.06
        if visible and rg1 > max(rg0 * 1.45, 1.25):
            score -= 38
            notes.append("暗部被提红，已判不合格")
        if lift > 0.14:
            score -= 16
            notes.append("暗部被提得太开")
    clip0 = float(np.mean(y0 > 0.98))
    clip1 = float(np.mean(y1 > 0.98))
    if clip1 > 0.03 and clip1 > clip0 + 0.04:
        score -= 14
        notes.append("高光裁切加重")
    if abs(d_luma) > 0.14:
        score -= 16
        notes.append("整体亮度跳太狠")
    if sat1 - sat0 > 0.14:
        score -= 14
        notes.append("饱和冲了")
    if technical.skin_fraction > 0.08:
        h, s, l = rgb_to_hsl(dst)
        skin = (h > 0.02) & (h < 0.13) & (s > 0.08) & (l > 0.18) & (l < 0.85)
        if np.mean(skin) < technical.skin_fraction * 0.35:
            # hue left the skin band
            score -= 20
            notes.append("皮肤色相跑偏")
    return float(np.clip(score, 0, 100)), notes


def _finish_score(gain: float, mae: float, slot: str, y0: np.ndarray, y1: np.ndarray) -> float:
    if slot == "A":
        return float(np.clip(70 + mae * 200, 60, 88))
    h, w = y0.shape
    inner = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    pop0 = float(np.mean(y0[inner]) - np.mean(y0))
    pop1 = float(np.mean(y1[inner]) - np.mean(y1))
    pop = float(np.clip((pop1 - pop0) * 420, -8, 16))
    return float(np.clip(50 + mae * 320 + (gain - 1.0) * 110 + pop, 30, 96))


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
