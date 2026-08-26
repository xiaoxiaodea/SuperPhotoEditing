from __future__ import annotations

import sqlite3

import numpy as np

from .composition import clamp_geometry, suggest_geometry
from .db import look_stats
from .models import CorrectionHint, Critique, HueShift, LocalEdit, Recipe, TechnicalReport, VariantPlan
from .recipes import LOOKS, get_look, xmp_look_ids


def preferred_look_ids(conn: sqlite3.Connection) -> list[str]:
    stats = look_stats(conn)
    ranked = sorted(stats["looks"], key=lambda r: (r["wins"], r["style"] or 0), reverse=True)
    return [r["look_id"] for r in ranked]


def build_plans(
    look_ids: list[str],
    critique: Critique,
    technical: TechnicalReport,
) -> list[VariantPlan]:
    correction = sanitize_correction(critique.correction, technical).as_recipe()
    slots = ("A", "B", "C")
    plans: list[VariantPlan] = []
    for slot, look_id in zip(slots, look_ids, strict=False):
        look = get_look(look_id)
        match = look_match(look.id, technical, critique.genre)
        amount = lut_amount(look, technical, slot, match, critique.lut_mix)
        look_recipe = look.recipe if amount >= 0.995 else _scale_recipe(look.recipe, amount)
        look_recipe = _fit_lut_to_scene(look_recipe, look.id, technical)
        recipe = _adapt_to_scene(correction.merged(look_recipe), technical, look_id)
        recipe = recipe.model_copy(update={"local": recipe.local.merged(local_edit_for(slot, technical))}).clamped()
        geom = suggest_geometry(technical, critique.genre, slot)
        if critique.geometry.apply:
            merged = critique.geometry
            if slot == "A":
                merged = merged.model_copy(update={"crop_left": 0.0, "crop_top": 0.0, "crop_right": 1.0, "crop_bottom": 1.0})
            geom = clamp_geometry(merged) if geom.is_noop() else geom
        geom = clamp_geometry(geom)
        why = _why(slot, look, critique, amount, geom)
        plans.append(
            VariantPlan(
                slot=slot,  # type: ignore[arg-type]
                look_id=look.id,
                title=look.title,
                why=why,
                recipe=recipe,
                moves=look.moves,
                lut_mix=amount,
                geometry=geom,
            )
        )
    return plans


def look_match(look_id: str, technical: TechnicalReport, genre: str) -> float:
    """How well a preset belongs on this picture. 0 = skip, 1 = strong fit."""
    look = LOOKS.get(look_id)
    if look is None or look_id == "faithful":
        return 0.0
    score = 0.28
    if genre in look.genre_tags:
        score += 0.22
    lid = look_id
    if lid == "dune":
        if technical.skin_fraction > 0.10:
            score -= 0.40
        if technical.green_fraction > 0.22 and technical.saturation_mean > 0.38:
            score -= 0.28
        if technical.top_third_blue > 0.16 or technical.aqua_fraction > 0.08:
            score -= 0.18
        if genre == "landscape" and technical.temperature_bias > 0.04:
            score += 0.22
    elif lid in {"kdrama_tears", "romance_gold", "filmic_portrait_warm"}:
        score += 0.32 if technical.skin_fraction > 0.08 else -0.42
    elif lid == "hanabi":
        score += 0.40 if technical.top_third_blue > 0.08 and technical.mean_luma < 0.16 else -0.50
    elif lid == "tokyo_ghoul":
        score += 0.28 if technical.mean_luma < 0.12 else -0.35
    elif lid in {"natsume", "five_cm"}:
        if technical.skin_fraction > 0.08 or genre in {"landscape", "portrait", "general"}:
            score += 0.18
        if technical.magenta_fraction > 0.05:
            score -= 0.12
    elif lid in {"skam", "shameless", "wong_kar_wai", "jp_drama"}:
        if genre in {"street", "urban", "indoor", "portrait"}:
            score += 0.16
        if technical.green_fraction > 0.28:
            score -= 0.20
    elif lid == "moody_lowkey":
        score += 0.30 if technical.mean_luma < 0.14 else -0.15
    elif lid == "landscape_filmic":
        score += 0.20 if technical.green_fraction > 0.12 or technical.top_third_blue > 0.15 else -0.10
    return float(np.clip(score, 0.0, 1.0))


def lut_amount(
    look,
    technical: TechnicalReport,
    slot: str,
    match: float,
    critique_mix: float,
) -> float:
    """Never 100% of an XMP pack on a developed JPEG."""
    if look.id == "faithful":
        return 1.0
    if look.source != "xmp":
        return float({"A": 1.0, "B": 1.12, "C": 1.28}.get(slot, 1.0))
    amount = 0.50 + 0.30 * float(np.clip(match, 0.0, 1.0))
    if technical.source_kind in {"jpeg", "png"}:
        amount *= 0.96
    if technical.saturation_mean > 0.46:
        amount *= 0.90
    if technical.dynamic_spread > 0.78:
        amount *= 0.92
    if slot == "B":
        amount *= 0.92
    amount *= float(np.clip(critique_mix, 0.40, 0.85)) / 0.55
    return float(np.clip(amount, 0.42, 0.86))


def ranked_look_ids(technical: TechnicalReport, genre: str) -> list[str]:
    scored = []
    for look in LOOKS.values():
        if look.id == "faithful":
            continue
        scored.append((look_match(look.id, technical, genre), look.id))
    scored.sort(reverse=True)
    return [lid for s, lid in scored if s >= 0.20]


def sanitize_correction(hint: CorrectionHint, technical: TechnicalReport) -> CorrectionHint:
    """VLM likes to 'fix' low-key photos. Don't let it."""
    data = hint.model_dump()
    developed = technical.source_kind in {"jpeg", "png"}
    low_key = technical.mean_luma < 0.12 and technical.shadow_clip > 0.12
    if developed:
        data["exposure_ev"] = float(np.clip(data["exposure_ev"], -0.30, 0.22))
        if low_key:
            data["exposure_ev"] = min(data["exposure_ev"], 0.06)
            data["shadows"] = min(data["shadows"], 0.08)
            data["contrast"] = min(data["contrast"], 0.04)
            data["saturation"] = min(data["saturation"], 0.03)
        data["shadows"] = float(np.clip(data["shadows"], -0.10, 0.14))
        data["highlights"] = float(np.clip(data["highlights"], -0.30, 0.12))
        data["whites"] = float(np.clip(data["whites"], -0.20, 0.12))
        data["blacks"] = float(np.clip(data["blacks"], -0.16, 0.10))
        data["contrast"] = float(np.clip(data["contrast"], -0.10, 0.10))
        data["saturation"] = float(np.clip(data["saturation"], -0.10, 0.06))
        data["vibrance"] = float(np.clip(data["vibrance"], -0.08, 0.08))
        data["clarity"] = float(np.clip(data["clarity"], -0.04, 0.10))
        data["dehaze"] = float(np.clip(data["dehaze"], 0.0, 0.08))
        data["temperature"] = float(np.clip(data["temperature"], -8.0, 8.0))
        data["tint"] = float(np.clip(data["tint"], -6.0, 6.0))
    else:
        data["exposure_ev"] = float(np.clip(data["exposure_ev"], -0.8, 0.8))
        data["shadows"] = float(np.clip(data["shadows"], -0.15, 0.25))
    return CorrectionHint.model_validate(data)


def _fit_lut_to_scene(recipe: Recipe, look_id: str, technical: TechnicalReport) -> Recipe:
    """Drop LUT pieces that fight the photograph."""
    if look_id not in xmp_look_ids():
        return recipe
    hsl = {k: v.model_copy() for k, v in recipe.hsl.items()}

    def scale_hsl(name: str, sat_k: float, lum_k: float = 1.0) -> None:
        if name not in hsl:
            return
        s = hsl[name]
        hsl[name] = HueShift(hue=s.hue, sat=s.sat * sat_k, lum=s.lum * lum_k)

    water_or_sky = technical.top_third_blue > 0.08 or technical.aqua_fraction > 0.04
    lush = technical.green_fraction > 0.18 and technical.saturation_mean > 0.34
    if look_id in {"dune", "downton", "resident_evil", "wong_kar_wai"}:
        if water_or_sky:
            scale_hsl("aqua", 0.30)
            scale_hsl("blue", 0.45)
        if lush:
            scale_hsl("green", 0.35, 0.5)
    if look_id in {"natsume", "unnatural", "five_cm", "jujutsu"} and technical.magenta_fraction > 0.03:
        scale_hsl("magenta", 0.28)
        scale_hsl("purple", 0.35)
    if look_id in {"kdrama_tears", "romance_gold"} and technical.skin_fraction < 0.04:
        scale_hsl("orange", 0.45)
        scale_hsl("yellow", 0.55)
    return recipe.model_copy(update={"hsl": hsl})


def _adapt_to_scene(recipe: Recipe, technical: TechnicalReport, look_id: str) -> Recipe:
    """Keep grades visible, but don't crush a low-key frame or HDR an already-loud JPEG."""
    data = recipe.model_dump()
    if look_id in xmp_look_ids():
        data["exposure_ev"] = 0.0
        if technical.mean_luma < 0.12:
            data["shadows"] = min(float(data["shadows"]), 0.04)
            curve = dict(data.get("curve") or {})
            rgb_pts = list(curve.get("rgb") or [])
            if rgb_pts:
                damped = []
                for x, y in rgb_pts:
                    xx, yy = float(x), float(y)
                    if xx < 0.10 and yy > xx:
                        yy = xx + (yy - xx) * 0.35
                    damped.append((xx, yy))
                curve["rgb"] = damped
                data["curve"] = curve
        if technical.saturation_mean > 0.52:
            data["vibrance"] = min(float(data["vibrance"]), 0.14)
        if technical.skin_fraction > 0.08:
            hsl = dict(data.get("hsl") or {})
            orange = dict(hsl.get("orange") or {})
            if orange:
                cap = 0.34 if technical.temperature_bias > 0.04 else 0.46
                orange["sat"] = min(float(orange.get("sat", 0.0)), cap)
                hsl["orange"] = orange
                data["hsl"] = hsl
        return Recipe.model_validate(data)
    if technical.mean_luma < 0.12:
        data["blacks"] = max(float(data["blacks"]), -0.16)
        data["shadows"] = float(np.clip(data["shadows"], -0.10, 0.06))
        data["vignette"] = min(float(data["vignette"]), 0.20)
        data["contrast"] = min(float(data["contrast"]), 0.16)
        data["dehaze"] = min(float(data["dehaze"]), 0.08)
        data["clarity"] = min(float(data["clarity"]), 0.12)
        data["exposure_ev"] = min(float(data["exposure_ev"]), 0.05)
    if technical.saturation_mean > 0.48:
        data["dehaze"] = min(float(data["dehaze"]), 0.08)
        data["vibrance"] = min(float(data["vibrance"]), 0.06)
        data["saturation"] = min(float(data["saturation"]), 0.02)
    if look_id == "faithful":
        data["vignette"] = 0.0
        data["grain"] = min(float(data["grain"]), 0.04)
    return Recipe.model_validate(data)


def _scale_recipe(recipe: Recipe, scale: float) -> Recipe:
    if scale == 1.0:
        return recipe
    data = recipe.model_dump()
    skip = {"hsl", "grade", "curve", "local"}
    for key, value in list(data.items()):
        if key in skip:
            continue
        if isinstance(value, (int, float)):
            data[key] = value * scale
    hsl = {}
    for name, shift in recipe.hsl.items():
        hsl[name] = {
            "hue": shift.hue * scale,
            "sat": shift.sat * scale,
            "lum": shift.lum * scale,
        }
    data["hsl"] = hsl
    g = recipe.grade
    data["grade"] = {
        "shadows": tuple(c * scale for c in g.shadows),
        "midtones": tuple(c * scale for c in g.midtones),
        "highlights": tuple(c * scale for c in g.highlights),
    }
    if not recipe.curve.is_identity():
        mix = lambda pts: [(x, x + (y - x) * scale) for x, y in pts]
        data["curve"] = {
            "rgb": mix(recipe.curve.rgb),
            "r": mix(recipe.curve.r),
            "g": mix(recipe.curve.g),
            "b": mix(recipe.curve.b),
        }
    loc = recipe.local.model_dump()
    for key, value in loc.items():
        loc[key] = value * scale
    data["local"] = loc
    return Recipe.model_validate(data)


def local_edit_for(slot: str, technical: TechnicalReport) -> LocalEdit:
    low = technical.mean_luma < 0.12
    portrait = technical.skin_fraction > 0.08
    if slot == "A":
        loc = LocalEdit(
            color_clean=0.18,
            subject_sharpen=0.14,
            clutter_suppress=0.20,
            defect_heal=0.55,
            sky_dehaze=0.08,
        )
    elif slot == "B":
        loc = LocalEdit(
            color_clean=0.34,
            subject_sharpen=0.30,
            subject_clarity=0.22,
            subject_lift=0.0 if low else 0.045,
            subject_vibrance=0.10,
            background_darken=0.025 if low else 0.08,
            background_desat=0.14,
            sky_dehaze=0.16,
            clutter_suppress=0.44,
            defect_heal=0.7,
        )
    else:
        loc = LocalEdit(
            color_clean=0.42,
            subject_sharpen=0.38,
            subject_clarity=0.28,
            subject_lift=0.01 if low else 0.06,
            subject_vibrance=0.14,
            background_darken=0.03 if low else 0.12,
            background_desat=0.18,
            sky_dehaze=0.20,
            clutter_suppress=0.56,
            defect_heal=0.85,
        )
    if portrait:
        loc = loc.model_copy(
            update={
                "subject_sharpen": loc.subject_sharpen * 0.72,
                "subject_lift": loc.subject_lift + 0.015,
                "background_desat": min(0.24, loc.background_desat + 0.04),
            }
        )
    return loc


def _why(slot: str, look, critique: Critique, amount: float = 1.0, geom=None) -> str:
    if look.id == "faithful":
        bits = ["对照基准：只做技术校正，方便你判断另外两套味道有没有过。"]
        if geom is not None and not geom.is_noop() and abs(geom.rotate_deg) > 0.2:
            bits.append(geom.why or "只做了水平校正。")
        return " ".join(bits)
    top = critique.issues[0].note if critique.issues else critique.summary
    extra = ""
    if geom is not None and not geom.is_noop() and geom.why:
        extra = f" {geom.why}。"
    if look.source == "xmp":
        mix = f"按画面只用了预设的 {int(round(amount * 100))}% ，不是整份套 LUT。"
        return f"{look.description} {mix} 分层：干净颜色、主体清晰、压干扰。针对：{top}{extra}"
    return f"{look.description} 分层：干净颜色、主体清晰、压干扰。针对：{top}{extra}"


def describe_taste(conn: sqlite3.Connection) -> str:
    stats = look_stats(conn)
    if not stats["total_picks"]:
        return "还没有打分记录。选一张你更想要的，系统会记住你的味道。"
    ranked = sorted(stats["looks"], key=lambda r: r["wins"], reverse=True)
    top = LOOKS.get(ranked[0]["look_id"])
    name = top.title if top else ranked[0]["look_id"]
    return f"已积累 {stats['total_picks']} 次选择，目前更常留下「{name}」。"
