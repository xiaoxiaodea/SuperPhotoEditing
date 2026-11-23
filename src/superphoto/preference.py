from __future__ import annotations

import sqlite3

import numpy as np

from .composition import clamp_geometry, suggest_geometry
from .db import look_stats
from .models import (
    CorrectionHint,
    Critique,
    EditIntent,
    GeometryHint,
    HueShift,
    LocalEdit,
    Recipe,
    TechnicalReport,
    VariantPlan,
)
from .look_routes import route_similarity
from .recipes import LOOKS, get_look, xmp_look_ids
from .workflow import apply_ethics, ethics_regime_for, guard_hues


def preferred_look_ids(conn: sqlite3.Connection) -> list[str]:
    stats = look_stats(conn)
    ranked = sorted(stats["looks"], key=lambda r: (r["wins"], r["style"] or 0), reverse=True)
    return [r["look_id"] for r in ranked if r["wins"] > 0]


def build_plans(
    look_ids: list[str],
    critique: Critique,
    technical: TechnicalReport,
    *, slots: tuple[str, ...] = ("A", "B", "C"),
) -> list[VariantPlan]:
    from .exposure import merge_correction_style
    correction = correction_recipe(critique.correction, technical, critique.intent)
    correction = correction.model_copy(update={"masks": [m.model_copy(deep=True) for m in critique.intent.masks]})
    ethics = ethics_regime_for(critique.genre)
    plans: list[VariantPlan] = []
    for slot, look_id in zip(slots, look_ids, strict=False):
        look = get_look(look_id)
        direction = next((d for d in critique.directions if d.slot == slot), None)
        if direction is not None:
            from .directions import bound_direction
            if direction.look_id and direction.look_id != look_id:
                raise ValueError(f"{slot}: direction preset {direction.look_id} differs from selected {look_id}")
            direction = bound_direction(direction, critique, technical)
            recipe = merge_correction_style(correction, direction.recipe, technical)
            # RAW recovery is the common correction beneath the three looks.
            # Do not let a weaker sky style erase the diagnosis's paired light
            # controls. Keep any scene-specific local color choices.
            if technical.source_kind == "raw":
                recovery = next((m for m in correction.masks if m.target == "sky"
                    and m.exposure_ev <= -.5 and (m.highlights < -.1 or m.contrast > .1)), None)
                if recovery is not None:
                    for mask in recipe.masks:
                        if mask.target == "sky" and mask.exposure_ev > recovery.exposure_ev:
                            for key in ("exposure_ev", "contrast", "whites", "blacks", "highlights", "shadows"):
                                setattr(mask, key, getattr(recovery, key))
                            mask.reason = recovery.reason
            from .exposure import preserve_exposure_correction
            recipe = preserve_exposure_correction(recipe, correction, technical)
            recipe = honor_locked_controls(apply_ethics(guard_hues(recipe, technical, ethics), ethics), critique.intent)
            recipe.masks = [m.for_source(technical.source_kind) for m in recipe.masks]
            intent = critique.intent.model_copy(update={"goal": direction.goal, "masks": recipe.masks})
            geom = GeometryHint()
            if slot != "A" and critique.geometry.apply:
                geom = critique.geometry.model_copy(deep=True)
                if technical.horizon_confidence < .32:
                    geom.rotate_deg = 0.0
                geom = clamp_geometry(geom)
            plans.append(VariantPlan(slot=slot, look_id=look_id, title=direction.title, why=direction.goal,
                                     method_id=direction.method_id,
                                     recipe=recipe, moves=direction.moves, lut_mix=direction.preset_amount, primary=correction,
                                     ethics=ethics, genre=critique.genre, intent=intent, geometry=geom))
            continue
        match = look_match(look.id, technical, critique.genre)
        amount = lut_amount(look, technical, slot, match, critique.lut_mix)
        if look.id == "celebrity_rescue":
            amount = 0.25
        elif critique.intent.natural_color:
            amount = min(amount, 0.40 if slot == "B" else 0.55)
        if look.source == "master":
            from .masters import recipe_for_photo

            base_recipe = recipe_for_photo(look.id, technical)
        else:
            base_recipe = look.recipe
        look_recipe = _scale_recipe(base_recipe, amount)
        look_recipe = _fit_lut_to_scene(look_recipe, look.id, technical)
        look_recipe = protect_look(look_recipe, critique.intent)
        from .resource_knowledge import apply_resource_guards
        look_recipe = apply_resource_guards(look_recipe, critique, technical)
        recipe = _adapt_to_scene(merge_correction_style(correction, look_recipe, technical), technical, look_id)
        recipe = recipe.model_copy(
            update={
                "local": recipe.local.merged(
                    local_edit_for(
                        slot,
                        technical,
                        genre=critique.genre,
                        has_clutter=bool(critique.clutter),
                    )
                )
            }
        ).clamped()
        from .exposure import preserve_exposure_correction
        recipe = preserve_exposure_correction(recipe, correction, technical)
        recipe = honor_locked_controls(apply_ethics(guard_hues(recipe, technical, ethics), ethics), critique.intent)
        recipe.masks = [m.for_source(technical.source_kind) for m in recipe.masks]
        geom = suggest_geometry(technical, critique.genre, slot)
        if slot == "A":
            # Reference slot keeps the original frame. VLM tilt guesses left black corners.
            geom = GeometryHint()
        else:
            if critique.geometry.apply:
                merged = critique.geometry
                if technical.horizon_confidence < 0.32:
                    merged = merged.model_copy(update={"rotate_deg": 0.0})
                geom = clamp_geometry(merged) if geom.is_noop() else geom
            geom = clamp_geometry(geom)
        master_meta = None
        if look.source == "master":
            from .masters import match_report

            master_meta = match_report(technical, look.id)
        why = _why(slot, look, critique, amount, geom, match)
        plans.append(
            VariantPlan(
                slot=slot,  # type: ignore[arg-type]
                look_id=look.id,
                title={"A": "自然精修", "B": "光影层次", "C": "克制氛围"}[slot] if critique.intent.natural_color else look.title,
                why=(critique.intent.goal + "。" + "；".join(m.reason for m in recipe.masks if m.reason)) if critique.intent.masks else why,
                recipe=recipe,
                moves=look.moves,
                lut_mix=amount,
                geometry=geom,
                primary=correction,
                ethics=ethics,
                genre=critique.genre,
                intent=critique.intent,
                master=master_meta,
            )
        )
    return plans


def honor_locked_controls(recipe: Recipe, intent: EditIntent) -> Recipe:
    updates = {key: 0.0 for key in intent.locked_controls if key != "hue"}
    if "hue" in intent.locked_controls:
        updates["hsl"] = {k: v.model_copy(update={"hue": 0.0}) for k, v in recipe.hsl.items()}
    return recipe.model_copy(update=updates)


def reduce_style_strength(recipe: Recipe, primary: Recipe, amount: float) -> Recipe:
    """Back off this preset without reducing confirmed exposure/local recovery."""
    from .workflow import _delta_recipe
    residual = _delta_recipe(recipe, primary)
    residual.masks = []
    result = primary.merged(_scale_recipe(residual, amount))
    if not primary.curve.is_identity():
        from .exposure import blend_tone_curves
        result.curve = blend_tone_curves(primary.curve, recipe.curve, float(np.clip(amount, 0, 1)))
    result.masks = [m.model_copy(deep=True) for m in recipe.masks]
    for key in ("detail", "optics", "crop"):
        setattr(result, key, getattr(recipe, key).model_copy(deep=True))
    return result.clamped()


def protect_look(recipe: Recipe, intent: EditIntent) -> Recipe:
    """Preserve the same photographic intent through every refinement."""
    if intent.natural_color:
        recipe = recipe.model_copy(update={
            "temperature": 0.0, "tint": 0.0, "grade": Recipe().grade,
            "curve": recipe.curve.model_copy(update={"r": [], "g": [], "b": []}),
            "hsl": {k: v.model_copy(update={"hue": float(np.clip(v.hue, -4, 4)),
                                            "sat": float(np.clip(v.sat, -0.15, 0.05))})
                    for k, v in recipe.hsl.items()},
        })
    if intent.preserve_black_depth:
        recipe = recipe.model_copy(update={
            "blacks": min(recipe.blacks, 0.08), "shadows": min(recipe.shadows, 0.12),
            "curve": recipe.curve.model_copy(update={
                "rgb": [(x, min(y, x + 0.02) if x < 0.10 else y) for x, y in recipe.curve.rgb]
            }),
        })
    if intent.preserve_atmosphere:
        recipe = recipe.model_copy(update={"dehaze": min(recipe.dehaze, 0.04)})
    return honor_locked_controls(recipe, intent)


def look_match(look_id: str, technical: TechnicalReport, genre: str) -> float:
    """How well a preset belongs on this picture. 0 = skip, 1 = strong fit.

    Hasselblad Masters are matched by scene (light, subject, environment).
    XMP looks use 练习图 centroids. Genre tags are a small bonus.
    """
    if look_id.startswith("hb:"):
        from .masters import master_similarity

        return master_similarity(look_id, technical)
    look = LOOKS.get(look_id)
    if look is None or look_id == "faithful":
        return 0.0
    sim = route_similarity(look_id, technical)
    if sim is None:
        score = 0.28
        score += 0.22 if genre in look.genre_tags else -0.16
        return float(np.clip(score, 0.0, 1.0))
    score = 0.10 + 0.80 * sim
    if genre in look.genre_tags:
        score += 0.06
    else:
        score -= 0.04
    return float(np.clip(score, 0.0, 1.0))


def lut_amount(
    look,
    technical: TechnicalReport,
    slot: str,
    match: float,
    critique_mix: float,
) -> float:
    """Looks are mixes. A developed JPEG never gets a full second grade."""
    if look.id == "faithful":
        return 1.0
    if look.id == "celebrity_rescue":
        return 0.82 if technical.source_kind in {"jpeg", "png"} else 1.0
    if look.source == "master":
        amount = 0.40 + 0.48 * float(np.clip(match, 0.0, 1.0))
        if technical.source_kind in {"jpeg", "png"}:
            amount *= 0.94
        if technical.saturation_mean > 0.48:
            amount *= 0.90
        return float(np.clip(amount, 0.34, 0.88))
    developed = technical.source_kind in {"jpeg", "png"}
    if look.source != "xmp":
        amount = {"A": 0.45, "B": 0.88, "C": 1.0}.get(slot, 0.85)
        if developed:
            amount *= 0.94
        if technical.saturation_mean > 0.52:
            amount *= 0.92
        return float(np.clip(amount, 0.55, 1.0))
    amount = 0.50 + 0.30 * float(np.clip(match, 0.0, 1.0))
    if developed:
        amount *= 0.92
    if technical.saturation_mean > 0.50:
        amount *= 0.90
    if technical.dynamic_spread > 0.78:
        amount *= 0.94
    if slot == "B":
        amount *= 0.94
    amount *= float(np.clip(critique_mix, 0.40, 0.85)) / 0.55
    return float(np.clip(amount, 0.38, 0.82))


def ranked_look_ids(technical: TechnicalReport, genre: str) -> list[str]:
    from .masters import ranked_master_ids

    ordered: list[str] = []
    for look_id in ranked_master_ids(technical):
        if look_match(look_id, technical, genre) >= 0.20:
            ordered.append(look_id)
    scored = []
    for look in LOOKS.values():
        if look.id == "faithful" or look.id in ordered:
            continue
        scored.append((look_match(look.id, technical, genre), look.id))
    scored.sort(reverse=True)
    ordered.extend(lid for s, lid in scored if s >= 0.20)
    return ordered


def sanitize_correction(hint: CorrectionHint, technical: TechnicalReport) -> CorrectionHint:
    """VLM likes to 'fix' low-key photos. Don't let it."""
    data = hint.model_dump()
    developed = technical.source_kind in {"jpeg", "png"}
    low_key = technical.mean_luma < 0.12 and technical.shadow_clip > 0.12
    if developed:
        from .exposure import developed_overexposure, deep_low_key_headroom, developed_lift_ev
        floor = -.8 if developed_overexposure(technical) else -.30
        measured_lift = developed_lift_ev(technical)
        ceiling = (max(.22, measured_lift) if measured_lift > 0 else
                   .45 if deep_low_key_headroom(technical) else (.06 if low_key else .22))
        data["exposure_ev"] = float(np.clip(data["exposure_ev"], floor, ceiling))
        if low_key:
            data["shadows"] = min(data["shadows"], 0.08)
            data["contrast"] = min(data["contrast"], 0.04)
            data["saturation"] = min(data["saturation"], 0.03)
        data["shadows"] = float(np.clip(data["shadows"], -0.10, 0.14))
        data["highlights"] = float(np.clip(data["highlights"], -0.30, 0.12))
        data["whites"] = float(np.clip(data["whites"], -0.20, 0.12))
        data["blacks"] = float(np.clip(data["blacks"], -0.16, 0.10))
        data["contrast"] = float(np.clip(data["contrast"], -0.10, 0.10))
        data["saturation"] = float(np.clip(data["saturation"], -0.03, 0.03))
        data["vibrance"] = float(np.clip(data["vibrance"], -0.08, 0.08))
        data["clarity"] = float(np.clip(data["clarity"], -0.04, 0.10))
        data["dehaze"] = float(np.clip(data["dehaze"], 0.0, 0.08))
        data["temperature"] = float(np.clip(data["temperature"], -8.0, 8.0))
        data["tint"] = float(np.clip(data["tint"], -6.0, 6.0))
    else:
        data["exposure_ev"] = float(np.clip(data["exposure_ev"], -0.8, 0.8))
        data["shadows"] = float(np.clip(data["shadows"], -0.15, 0.25))
        if technical.highlight_p95 > .93:
            data["whites"] = min(data["whites"], 0.0)
    return CorrectionHint.model_validate(data)


def correction_recipe(hint: CorrectionHint, technical: TechnicalReport, intent: EditIntent | None = None) -> Recipe:
    """Build shared correction; clipped developed whites need an anchored curve."""
    from .exposure import developed_overexposure, white_preserving_curve
    recipe = sanitize_correction(hint, technical).as_recipe()
    if (developed_overexposure(technical) and technical.white_clip_fraction >= .03
            and recipe.exposure_ev < -.30
            and not (intent and "exposure_ev" in intent.locked_controls)):
        recipe.curve = white_preserving_curve(recipe.exposure_ev)
        recipe.exposure_ev = 0.0
        recipe.highlights = 0.0
        recipe.whites = 0.0
        recipe.contrast = 0.0
    return recipe


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


def _soften_developed(data: dict, technical: TechnicalReport) -> dict:
    """Keep JPEG looks visible, but stop short of a second HDR pass."""
    if technical.source_kind not in {"jpeg", "png"}:
        return data
    data["contrast"] = min(float(data.get("contrast") or 0.0), 0.16)
    data["dehaze"] = min(float(data.get("dehaze") or 0.0), 0.06)
    data["clarity"] = min(float(data.get("clarity") or 0.0), 0.08)
    data["vignette"] = min(float(data.get("vignette") or 0.0), 0.16)
    if technical.saturation_mean > 0.48:
        data["vibrance"] = min(float(data.get("vibrance") or 0.0), 0.08)
        data["saturation"] = min(float(data.get("saturation") or 0.0), 0.04)
    if technical.mean_luma > 0.62:
        data["blacks"] = max(float(data.get("blacks") or 0.0), -0.08)
        data["contrast"] = min(float(data.get("contrast") or 0.0), 0.12)
    return data


def _protect_sky(data: dict, technical: TechnicalReport) -> dict:
    """A bright sky must not be lifted. Global p95 misses a small hot sky over dark ground."""
    if technical.top_third_blue <= 0.10:
        return data
    data["highlights"] = min(float(data.get("highlights") or 0.0), -0.08)
    data["whites"] = min(float(data.get("whites") or 0.0), 0.0)
    data["exposure_ev"] = min(float(data.get("exposure_ev") or 0.0), 0.02)
    hsl = dict(data.get("hsl") or {})
    for name in ("aqua", "blue"):
        node = dict(hsl.get(name) or {})
        node["lum"] = min(float(node.get("lum") or 0.0), 0.0)
        hsl[name] = node
    data["hsl"] = hsl
    return data


def _adapt_to_scene(recipe: Recipe, technical: TechnicalReport, look_id: str) -> Recipe:
    """Keep grades visible, but don't crush a low-key frame or HDR an already-loud JPEG."""
    data = recipe.model_dump()
    data = _protect_sky(data, technical)
    if look_id.startswith("hb:"):
        data["exposure_ev"] = float(np.clip(float(data.get("exposure_ev") or 0.0), -0.12, 0.12))
        if technical.skin_fraction > 0.10:
            data["saturation"] = float(np.clip(float(data["saturation"]), -0.16, 0.08))
            data["temperature"] = float(np.clip(float(data["temperature"]), -18.0, 22.0))
        if technical.mean_luma < 0.12:
            data["shadows"] = min(float(data["shadows"]), 0.06)
            data["blacks"] = max(float(data["blacks"]), -0.28)
    if look_id in xmp_look_ids():
        data["exposure_ev"] = 0.0
        if technical.dynamic_spread > 0.72:
            data["contrast"] = min(float(data["contrast"]), 0.10)
            data["clarity"] = min(float(data["clarity"]), 0.08)
            data["dehaze"] = min(float(data["dehaze"]), 0.05)
            data["blacks"] = max(float(data["blacks"]), -0.07)
            data["whites"] = min(float(data["whites"]), 0.03)
        if technical.highlight_p95 > 0.90:
            data["whites"] = min(float(data["whites"]), 0.01)
            data["highlights"] = min(float(data["highlights"]), -0.04)
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
        return Recipe.model_validate(_soften_developed(data, technical))
    if technical.mean_luma < 0.12:
        data["blacks"] = max(float(data["blacks"]), -0.10)
        data["shadows"] = float(np.clip(data["shadows"], -0.06, 0.06))
        data["vignette"] = min(float(data["vignette"]), 0.12)
        data["contrast"] = min(float(data["contrast"]), 0.10)
        data["dehaze"] = min(float(data["dehaze"]), 0.05)
        data["clarity"] = min(float(data["clarity"]), 0.08)
        data["exposure_ev"] = min(float(data["exposure_ev"]), 0.05)
    if technical.dynamic_spread > 0.72:
        data["contrast"] = min(float(data["contrast"]), 0.12)
        data["clarity"] = min(float(data["clarity"]), 0.10)
        data["dehaze"] = min(float(data["dehaze"]), 0.06)
        data["blacks"] = max(float(data["blacks"]), -0.08)
        data["whites"] = min(float(data["whites"]), 0.04)
    if technical.highlight_p95 > 0.90:
        data["whites"] = min(float(data["whites"]), 0.02)
        data["highlights"] = min(float(data["highlights"]), -0.04)
    if technical.saturation_mean > 0.48:
        data["dehaze"] = min(float(data["dehaze"]), 0.08)
        data["vibrance"] = min(float(data["vibrance"]), 0.06)
        data["saturation"] = min(float(data["saturation"]), 0.02)
    if look_id == "faithful":
        data["vignette"] = 0.0
        data["grain"] = min(float(data["grain"]), 0.04)
        data["contrast"] = min(float(data["contrast"]), 0.06 if technical.source_kind in {"jpeg", "png"} else 0.12)
    return Recipe.model_validate(_soften_developed(data, technical))


def _scale_recipe(recipe: Recipe, scale: float) -> Recipe:
    # Capture detail and geometry come from this photograph, not a look.
    scale = float(np.clip(scale, 0.0, 1.0))
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
    data["masks"] = [dict(m.model_dump(), **{key: value * scale for key, value in
                         m.model_dump(exclude={"target", "reason"}).items()}) for m in recipe.masks]
    data["parametric_tone"] = recipe.parametric_tone.model_dump()
    for key in ("shadows", "darks", "lights", "highlights"):
        data["parametric_tone"][key] *= scale
    data["color_grading"] = recipe.color_grading.model_dump()
    for key, value in data["color_grading"].items():
        if value is not None and (key.endswith("_sat") or key.endswith("_lum")):
            data["color_grading"][key] = value * scale
    data["detail"] = {}
    data["optics"] = {}
    data["crop"] = {}
    return Recipe.model_validate(data)


def local_edit_for(
    slot: str,
    technical: TechnicalReport,
    *,
    genre: str = "general",
    has_clutter: bool = False,
) -> LocalEdit:
    """Retouch-only local finish. Landscape color and atmosphere are not 'dirt'."""
    developed = technical.source_kind in {"jpeg", "png"}
    portrait = technical.skin_fraction > 0.08 or genre == "portrait"
    landscape = genre in {"landscape", "nature"} or (
        not portrait and (technical.green_fraction > 0.16 or technical.top_third_blue > 0.20)
    )
    atmospheric = technical.top_third_blue > 0.12 and technical.highlight_clip < 0.035
    low = technical.mean_luma < 0.12
    high_key = technical.mean_luma > 0.55

    # Reference slot on a finished file: do not add a second local grade.
    if slot == "A" and developed:
        return LocalEdit()

    loc = LocalEdit()
    if portrait:
        loc = LocalEdit(
            color_clean={"B": 0.16, "C": 0.20}.get(slot, 0.10),
            subject_lift=0.0 if low else (0.035 if slot == "B" else 0.045),
            subject_vibrance=0.06 if slot == "B" else 0.08,
            background_desat=0.08 if slot == "B" else 0.11,
            background_darken=0.0 if low or high_key else (0.03 if slot == "B" else 0.045),
            clutter_suppress=0.18 if has_clutter else 0.0,
        )
        return loc

    if landscape:
        sky_dehaze = 0.0
        if technical.highlight_clip > 0.025 and not atmospheric:
            sky_dehaze = 0.06 if slot == "B" else 0.08
        loc = LocalEdit(
            sky_dehaze=sky_dehaze,
            clutter_suppress=0.16 if has_clutter else 0.0,
            background_darken=0.0 if high_key or low else (0.02 if slot == "B" else 0.03),
        )
        return loc

    if has_clutter:
        loc = loc.model_copy(update={"clutter_suppress": 0.18 if slot == "B" else 0.24})
    return loc


def _why(slot: str, look, critique: Critique, amount: float = 1.0, geom=None, match: float = 0.0) -> str:
    labels = {"A": "稳妥救片", "B": "最佳匹配", "C": "方向分叉"}
    top = critique.issues[0].note if critique.issues else critique.summary
    extra = ""
    if geom is not None and not geom.is_noop() and geom.why:
        extra = f" {geom.why}。"
    score = int(round(match * 100)) if look.id != "celebrity_rescue" else max(58, int(round(amount * 72)))
    mix = f"按当前画面只保留约 {int(round(amount * 100))}% 的风格强度"
    return f"{labels.get(slot, '候选路线')}：模仿「{look.title}」，适配度约 {score}%。{look.description} {mix}，不是照抄示范参数。优先处理：{top}{extra}"


def describe_taste(conn: sqlite3.Connection) -> str:
    from .scene_preferences import taste_summary
    return taste_summary(conn)
