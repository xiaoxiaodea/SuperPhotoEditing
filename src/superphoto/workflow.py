"""Two-axis photographic workflow: restoration first, retouch second.

Restoration is intentionally color-neutral. Retouch is intentionally forbidden
from denoising or sharpening. Keeping these axes explicit prevents a style
recipe from disguising technical damage and makes each stage independently
safe to accept or roll back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from .color import linear_to_srgb, luma, srgb_to_linear
from .models import EthicsRegime, HueShift, Recipe, TechnicalReport, VariantPlan

# Look groups the retouch axis may keep, in display order.
_ATTRIBUTE_ORDER = ("tone", "color", "atmosphere", "subject", "background")
_GENRE_INTENTS: dict[str, tuple[str, ...]] = {
    "landscape": ("tone", "color", "background"),
    "portrait": ("tone", "color", "subject"),
    "street": ("tone", "color", "atmosphere"),
    "night": ("tone", "atmosphere", "subject"),
    "product": ("tone", "color", "background"),
    "indoor": ("tone", "color", "subject"),
    "documentary": ("tone", "color", "atmosphere"),
}
_HUE_CLAMP = {
    "green": 12.0,
    "yellow": 10.0,
    "aqua": 12.0,
    "blue": 16.0,
    "orange": 8.0,
    "red": 8.0,
    "purple": 18.0,
    "magenta": 14.0,
}


@dataclass
class RestorationPlan:
    denoise: float = 0.0
    sharpen: float = 0.0
    defect_heal: float = 0.0
    actions: list[str] = field(default_factory=list)
    protections: list[str] = field(default_factory=lambda: ["颜色和明暗关系保持不变"])

    def is_noop(self) -> bool:
        return max(self.denoise, self.sharpen, self.defect_heal) < 1e-4


@dataclass
class StageGate:
    accepted: bool
    score: float
    notes: list[str] = field(default_factory=list)


def plan_restoration(technical: TechnicalReport) -> RestorationPlan:
    """Conservative technical cleanup based only on measured defects."""
    actions: list[str] = []
    denoise = 0.0
    sharpen = 0.0
    defect = 0.0
    # Water ripple, snow sparkle and foliage all inflate the noise proxy.
    # A developed JPEG is already denoised in-camera — leave it unless ISO is high.
    iso = _iso(technical)
    developed = technical.source_kind in {"jpeg", "png"}
    if developed and iso < 1600:
        return RestorationPlan()
    if technical.noise > 0.065 and (not developed or iso >= 1600):
        denoise = float(np.clip((technical.noise - 0.055) * 2.2, 0.04, 0.22))
        actions.append("轻微清理画面噪点")
    if technical.sharpness < 0.038 and technical.noise < 0.10 and not developed:
        sharpen = float(np.clip((0.05 - technical.sharpness) * 3.0, 0.03, 0.14))
        actions.append("轻微恢复主体细节")
    return RestorationPlan(denoise=denoise, sharpen=sharpen, defect_heal=defect, actions=actions)


def _iso(technical: TechnicalReport) -> float:
    for key in ("ISOSpeedRatings", "PhotographicSensitivity", "ISO"):
        raw = technical.exif.get(key)
        if not raw:
            continue
        try:
            return float(str(raw).split("/")[0].split()[0])
        except ValueError:
            continue
    return 0.0


def apply_restoration(linear: np.ndarray, plan: RestorationPlan) -> np.ndarray:
    """Color-neutral luma restoration; chroma and global tone are frozen."""
    if plan.is_noop():
        return linear
    srgb = np.clip(linear_to_srgb(np.maximum(linear, 0.0)), 0.0, 1.0)
    y = luma(srgb)
    chroma = srgb - y[..., None]
    out_y = y
    if plan.denoise > 1e-4:
        blur = _blur_luma(out_y, 1.15)
        # Prefer denoising shadows, where the proxy is most useful.
        shadow = np.clip((0.55 - out_y) / 0.5, 0.0, 1.0)
        mix = plan.denoise * (0.35 + 0.65 * shadow)
        out_y = out_y * (1.0 - mix) + blur * mix
    if plan.defect_heal > 1e-4:
        blur = _blur_luma(out_y, 2.4)
        hot = (out_y - blur > 0.24) & (out_y > 0.62)
        if float(np.mean(hot)) < 0.01:
            mask = _blur_luma(hot.astype(np.float32), 0.8) * plan.defect_heal
            out_y = out_y * (1.0 - mask) + blur * mask
    if plan.sharpen > 1e-4:
        blur = _blur_luma(out_y, 1.25)
        high = np.clip(out_y - blur, -0.06, 0.06)
        out_y = out_y + plan.sharpen * high
    # Re-attach the original chroma exactly; restoration cannot alter WB/sat.
    restored = np.clip(out_y[..., None] + chroma, 0.0, 1.0)
    return srgb_to_linear(restored)


def score_restoration(source_linear: np.ndarray, restored_linear: np.ndarray) -> StageGate:
    src = _thumb(linear_to_srgb(np.maximum(source_linear, 0.0)))
    dst = _thumb(linear_to_srgb(np.maximum(restored_linear, 0.0)))
    y0, y1 = luma(src), luma(dst)
    chroma0 = src - y0[..., None]
    chroma1 = dst - y1[..., None]
    chroma_drift = float(np.mean(np.abs(chroma1 - chroma0)))
    tone_drift = float(abs(np.mean(y1) - np.mean(y0)))
    mae = float(np.mean(np.abs(dst - src)))
    score = 100.0 - chroma_drift * 1800.0 - tone_drift * 650.0 - max(0.0, mae - 0.035) * 500.0
    notes: list[str] = []
    if chroma_drift > 0.006:
        notes.append("修复阶段改变了颜色")
    if tone_drift > 0.025:
        notes.append("修复阶段改变了整体明暗")
    accepted = chroma_drift <= 0.008 and tone_drift <= 0.03 and mae <= 0.075
    return StageGate(accepted=accepted, score=float(np.clip(score, 0, 100)), notes=notes)


def ethics_regime_for(genre: str) -> EthicsRegime:
    if genre in {"street", "documentary"}:
        return "documentary"
    if genre in {"landscape", "nature"}:
        return "landscape"
    if genre == "portrait":
        return "portrait"
    if genre == "product":
        return "product"
    return "commercial"


def apply_ethics(recipe: Recipe, ethics: EthicsRegime) -> Recipe:
    """Freeze content-changing retouch according to the job's ethical regime."""
    data = recipe.model_dump()
    loc = data["local"]
    if ethics == "documentary":
        loc["clutter_suppress"] = 0.0
        loc["defect_heal"] = 0.0
        loc["color_clean"] = float(loc.get("color_clean") or 0.0) * 0.35
        loc["background_desat"] = float(loc.get("background_desat") or 0.0) * 0.45
        data["grain"] = min(float(data.get("grain") or 0.0), 0.04)
    elif ethics == "product":
        g = data.get("grade") or {}
        data["grade"] = {
            "shadows": tuple(c * 0.35 for c in g.get("shadows", (0.0, 0.0, 0.0))),
            "midtones": tuple(c * 0.35 for c in g.get("midtones", (0.0, 0.0, 0.0))),
            "highlights": tuple(c * 0.35 for c in g.get("highlights", (0.0, 0.0, 0.0))),
        }
        data["temperature"] = float(np.clip(data.get("temperature") or 0.0, -6.0, 6.0))
    data["local"] = loc
    return Recipe.model_validate(data)


def guard_hues(recipe: Recipe, technical: TechnicalReport, ethics: EthicsRegime = "commercial") -> Recipe:
    """Keep foliage, sky and skin inside expected hue corridors."""
    if not recipe.hsl:
        return recipe
    hsl = {k: v.model_copy() for k, v in recipe.hsl.items()}
    documentary = ethics == "documentary"
    for name, shift in list(hsl.items()):
        limit = _HUE_CLAMP.get(name, 20.0)
        if documentary:
            limit = min(limit, 6.0)
        hue = float(np.clip(shift.hue, -limit, limit))
        sat = shift.sat
        if name in {"green", "yellow", "aqua"} and technical.green_fraction > 0.10:
            sat = min(sat, 0.0 if documentary else 0.04)
        if name in {"orange", "red"} and technical.skin_fraction > 0.06:
            sat = min(sat, 0.06 if documentary else 0.08)
            hue = float(np.clip(hue, -6.0, 6.0))
        if name in {"blue", "aqua"} and technical.top_third_blue > 0.12:
            hue = float(np.clip(hue, -12.0, 12.0))
        # Blue in the upper frame can be a mountain, jacket or building.
        # Its presence cannot veto a deliberate luminance lift.
        hsl[name] = HueShift(hue=hue, sat=sat, lum=shift.lum)
    return recipe.model_copy(update={"hsl": hsl})


def split_retouch_recipe(
    recipe: Recipe,
    max_attributes: int | None = None,
    *,
    primary: Recipe | None = None,
    genre: str = "general",
    ethics: EthicsRegime = "commercial",
) -> tuple[Recipe, list[str]]:
    """Preserve a coherent plan; optional limits shorten its explanation only.

    A small sky or subject adjustment must not disappear just because a curve
    has a larger numeric magnitude. Capture sharpening and healing remain in
    the restoration stage; mild positive/negative clarity is a creative move.
    """
    recipe = apply_ethics(recipe, ethics)
    primary = primary or Recipe()
    look_only = _delta_recipe(recipe, primary)
    preferred = set(_GENRE_INTENTS.get(genre, ("tone", "color", "atmosphere")))
    ranked = sorted(
        _attribute_groups(look_only),
        key=lambda item: item[1] + (0.10 if item[0] in preferred and item[1] > 0.04 else 0.0),
        reverse=True,
    )
    selected_look: list[str] = []
    for name, strength in ranked:
        if strength <= 1e-4:
            continue
        selected_look.append(name)
        if max_attributes is not None and len(selected_look) >= max_attributes:
            break
    data = recipe.model_dump()
    data["clarity"] = float(np.clip(data["clarity"], -0.15, 0.12))
    data["local"]["subject_clarity"] = min(data["local"]["subject_clarity"], 0.15)
    data["local"]["subject_sharpen"] = 0.0
    data["local"]["defect_heal"] = 0.0
    if ethics == "documentary":
        data["local"]["clutter_suppress"] = 0.0
    out = Recipe.model_validate(data).clamped()
    attributes = [name for name in _ATTRIBUTE_ORDER if name in selected_look]
    return out, attributes


def apply_retouch(
    restored_linear: np.ndarray,
    plan: VariantPlan,
    *,
    layers,
    technical: TechnicalReport,
    clutter_boxes,
):
    from .render import apply_recipe

    recipe, attributes = split_retouch_recipe(
        plan.recipe,
        primary=plan.primary,
        genre=plan.genre,
        ethics=plan.ethics,
    )
    recipe = guard_hues(recipe, technical, plan.ethics)
    rendered = apply_recipe(
        restored_linear,
        recipe,
        layers=layers,
        technical=technical,
        clutter_boxes=clutter_boxes,
    )
    return rendered, recipe, attributes


def simple_explanation(
    restoration: RestorationPlan,
    attributes: list[str],
    semantic_used: bool,
    ethics: EthicsRegime = "commercial",
) -> dict[str, list[str]]:
    labels = {
        "tone": "整理明暗层次",
        "color": "让颜色更协调",
        "atmosphere": "调整整体氛围",
        "subject": "轻微突出主体",
        "background": "减少背景干扰",
    }
    beautified = [labels[x] for x in attributes if x in labels]
    protected = ["不改变画面内容", "美化阶段不额外磨皮或锐化", "技术校正始终保留"]
    if semantic_used:
        protected.append("按主体、天空和背景分别保护")
    else:
        protected.append("使用保守分区避免误伤主体")
    if ethics == "documentary":
        protected.append("按纪实规则：不抹掉画面里的东西")
    elif ethics == "product":
        protected.append("商品颜色保持可辨认")
    elif ethics == "landscape":
        protected.append("草木和天空不离开自然色")
    elif ethics == "portrait":
        protected.append("肤色锁在自然范围")
    return {
        "repaired": restoration.actions or ["原片清晰度正常，未强行修复"],
        "beautified": beautified or ["仅做保守校正"],
        "protected": restoration.protections + protected,
    }


def _delta_recipe(full: Recipe, base: Recipe) -> Recipe:
    """Look-only residual used to rank aesthetic intents."""
    data = full.model_dump()
    b = base.model_dump()
    skip = {"hsl", "grade", "curve", "local"}
    for key, value in list(data.items()):
        if key in skip:
            continue
        if isinstance(value, (int, float)):
            data[key] = value - float(b.get(key) or 0.0)
    hsl = {}
    for name, shift in full.hsl.items():
        other = base.hsl.get(name, HueShift())
        hsl[name] = HueShift(
            hue=shift.hue - other.hue,
            sat=shift.sat - other.sat,
            lum=shift.lum - other.lum,
        )
    data["hsl"] = hsl
    data["grade"] = {
        "shadows": tuple(a - c for a, c in zip(full.grade.shadows, base.grade.shadows)),
        "midtones": tuple(a - c for a, c in zip(full.grade.midtones, base.grade.midtones)),
        "highlights": tuple(a - c for a, c in zip(full.grade.highlights, base.grade.highlights)),
    }
    loc = full.local.model_dump()
    bloc = base.local.model_dump()
    for key, value in loc.items():
        loc[key] = value - float(bloc.get(key) or 0.0)
    data["local"] = loc
    return Recipe.model_validate(data)


def _attribute_groups(recipe: Recipe) -> list[tuple[str, float]]:
    d = recipe
    tone = max(
        abs(d.exposure_ev) / 1.2,
        abs(d.highlights), abs(d.shadows), abs(d.whites), abs(d.blacks),
        abs(d.contrast) * 2.0,
        0.18 if not d.curve.is_identity() else 0.0,
    )
    color = max(
        abs(d.temperature) / 35.0,
        abs(d.tint) / 25.0,
        abs(d.saturation) * 2.0,
        abs(d.vibrance) * 2.0,
        max((abs(x.hue) / 30 + abs(x.sat) + abs(x.lum)) / 2 for x in d.hsl.values()) if d.hsl else 0.0,
        max(abs(v) for triplet in (d.grade.shadows, d.grade.midtones, d.grade.highlights) for v in triplet) * 4.0,
        max(d.color_grading.shadow_sat or 0, d.color_grading.midtone_sat or 0, d.color_grading.highlight_sat or 0) / 50,
    )
    atmosphere = max(abs(d.dehaze) * 2.0, abs(d.vignette) * 1.5, d.grain * 1.5)
    subject = max(abs(d.local.subject_lift) * 5.0, abs(d.local.subject_vibrance) * 2.0)
    background = max(
        abs(d.local.background_darken) * 4.0,
        abs(d.local.background_desat) * 2.0,
        abs(d.local.color_clean) * 1.2,
        abs(d.local.clutter_suppress),
        abs(d.local.sky_dehaze) * 1.5,
    )
    for mask in d.masks:
        if mask.is_noop():
            continue
        if mask.target == "subject":
            subject = max(subject, 0.3)
        else:
            background = max(background, 0.3)
    return [("tone", tone), ("color", color), ("atmosphere", atmosphere), ("subject", subject), ("background", background)]


def _blur_luma(gray: np.ndarray, radius: float) -> np.ndarray:
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, "L").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(out, dtype=np.float32) / 255.0


def _thumb(srgb: np.ndarray, edge: int = 512) -> np.ndarray:
    h, w = srgb.shape[:2]
    if max(h, w) <= edge:
        return np.clip(srgb, 0.0, 1.0).astype(np.float32)
    scale = edge / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    ys = np.linspace(0, h - 1, nh).astype(np.int32)
    xs = np.linspace(0, w - 1, nw).astype(np.int32)
    return np.clip(srgb[ys][:, xs], 0.0, 1.0).astype(np.float32)
