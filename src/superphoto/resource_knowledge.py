"""Reviewed local tutorial lessons, conditional guards and new starting looks.

No media is read and no original commercial preset is loaded at runtime.
Numerical bounds and recipes below are our conservative interpretations; source
hashes and observed timestamps describe the evidence, not exact author sliders.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from .models import ColorGradingControls, Critique, HueShift, Look, Recipe, TechnicalReport, ToneCurve

BUNDLE_PATH = Path(__file__).with_name("resource_lessons.json")
RESOURCE_FAMILIES = {"resource_matte": "matte", "resource_snow": "snow", "resource_night": "night_color"}
RESOURCE_LESSONS = {"resource_matte": "matte_structure", "resource_snow": "snow_luminous", "resource_night": "night_light_hierarchy"}

RESOURCE_LOOKS = {
    "resource_matte": Look(id="resource_matte", title="柔灰哑光", title_en="Quiet matte",
        description="小幅抬黑与柔和肩部保留中间调骨架，轻微冷影暖光，不靠全局压暗制造哑光。",
        genre_tags=["landscape", "nature", "street", "urban", "architecture", "documentary"],
        moves=["柔和黑位与高光过渡", "保留中间调密度", "轻量组织已有冷暖"],
        recipe=Recipe(contrast=.06, clarity=-.02, saturation=-.035,
            curve=ToneCurve(rgb=[(0, .012), (.18, .16), (.5, .49), (.8, .81), (1, .98)]),
            color_grading=ColorGradingControls(shadow_hue=215, shadow_sat=4, highlight_hue=40, highlight_sat=5, blending=55, balance=0))),
    "resource_snow": Look(id="resource_snow", title="雪景清透", title_en="Luminous snow",
        description="保留雪面明度和纹理，轻整蓝色阴影，避免亮雪被压成灰色。",
        genre_tags=["landscape", "nature"], moves=["舒展雪面中间调", "柔和保护亮部", "克制整理蓝色阴影"],
        recipe=Recipe(highlights=-.08, contrast=.03, vibrance=.025,
            curve=ToneCurve(rgb=[(0, 0), (.18, .18), (.5, .54), (.8, .84), (1, .995)]),
            hsl={"blue": HueShift(sat=-.10, lum=.025)},
            color_grading=ColorGradingControls(shadow_hue=215, shadow_sat=3))),
    "resource_night": Look(id="resource_night", title="夜色光序", title_en="Night light hierarchy",
        description="保留夜色密度和灯光亮度，用轻量亮度色轮区分冷环境与暖光。",
        genre_tags=["night", "street", "urban"], moves=["保留暗部密度", "柔和灯光过渡", "区分环境与亮部光色"],
        recipe=Recipe(highlights=-.12, blacks=-.02, contrast=.04,
            curve=ToneCurve(rgb=[(0, 0), (.18, .155), (.5, .50), (.8, .82), (1, .985)]),
            color_grading=ColorGradingControls(shadow_hue=215, shadow_sat=6, highlight_hue=40, highlight_sat=7, blending=65, balance=-5))),
}


@lru_cache(maxsize=1)
def _bundle() -> tuple[dict, str]:
    content = BUNDLE_PATH.read_bytes()
    data = json.loads(content)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported resource lesson schema")
    ids = [lesson["id"] for lesson in data["lessons"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate resource lesson ID")
    for lesson in data["lessons"]:
        if not lesson.get("evidence") or any(item["source_id"] not in data["sources"] for item in lesson["evidence"]):
            raise ValueError("Resource lesson lacks source evidence")
    return data, hashlib.sha256(content).hexdigest()


def _has(text: str, chinese: tuple[str, ...], english: tuple[str, ...]) -> bool:
    return any(word in text for word in chinese) or any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in english)


def _positive_scene_text(text: str) -> str:
    # Remove negated evidence, retaining positive clauses elsewhere. This is a
    # bounded vocabulary filter, not a claim of general language understanding.
    text = re.sub(r"(?:没有|并无|不含|不存在|不是|非|无)\s*(?:任何|明显|可见|一点)?\s*(?:薄雾|浓雾|雾气|薄霭|云海|雾|积雪|白雪|冰雪|雪景|雪地|雪山|雪|夜景|夜晚|夜街|霓虹)", "", text)
    return re.sub(r"\b(?:no|without|not)\s+(?:(?:a|an|any|the)\s+)?(?:nighttime|night|snowfields?|snowy|snow|foggy|fog|misty|mist|hazy|haze)\b", "", text)


def resource_scene_flags(critique: Critique, t: TechnicalReport) -> dict[str, bool]:
    text = _positive_scene_text(critique.subject_note.casefold())
    primary = re.split(r"及|与|对岸|背景|background| with | and |[,，;；。]", text, maxsplit=1)[0]
    landscape = critique.genre in {"landscape", "nature"}
    night = critique.genre == "night" or _has(primary, ("夜景", "夜晚", "夜街", "霓虹"), ("night", "nighttime"))
    snow = landscape and _has(primary, ("雪景", "雪地", "雪山", "白雪", "积雪", "冰雪"), ("snow", "snowy", "snowfield", "snowfields"))
    # Intent defaults alone do not prove that a scene contains fog.
    fog = landscape and _has(text, ("雾", "薄霭", "云海"), ("fog", "foggy", "mist", "misty", "haze", "hazy"))
    portrait = critique.genre in {"portrait", "family", "fashion"}
    forest = landscape and _has(primary, ("森林", "树林", "绿叶", "林地", "绿植", "草地"), ("forest", "woodland", "foliage", "leaves", "grass"))
    twilight = night or _has(primary, ("暮色", "黄昏", "黎明", "蓝调时刻"), ("twilight", "dusk", "dawn"))
    matte = critique.genre in {"landscape", "nature", "street", "urban", "architecture", "documentary"} and not (snow or night) and t.mean_luma < .38
    return {"night": night, "snow": snow, "fog": fog, "portrait": portrait, "forest": forest,
            "twilight": twilight, "matte": matte, "flat_landscape": landscape and t.std_luma < .09,
            "vintage": critique.genre in {"street", "urban", "documentary", "portrait"}}


def resource_context(critique: Critique, t: TechnicalReport) -> dict:
    data, digest = _bundle()
    flags = resource_scene_flags(critique, t)
    applicable = [lesson for lesson in data["lessons"] if "always" in lesson["triggers"] or any(flags.get(tag) for tag in lesson["triggers"])]
    # Keep all scene-specific lessons, rather than filling the prompt with an
    # entire tutorial library. Current bundle is bounded to eleven short items.
    source_ids = {item["source_id"] for lesson in applicable for item in lesson["evidence"]}
    checks = []
    # Region/material checks have priority over shared workflow bookkeeping.
    for lesson in sorted(applicable, key=lambda lesson: "always" in lesson["triggers"]):
        checks.extend(lesson["review_checks"])
    return {"version": data["schema_version"], "bundle_sha256": digest, "coverage": data["coverage"],
            "scene_evidence": flags, "lessons": applicable,
            "sources": {key: data["sources"][key] for key in sorted(source_ids)},
            "review_checks": list(dict.fromkeys(checks))[:4],
            "numeric_policy_note": data["numeric_policy_note"]}


def resource_incompatibility(look_id: str, critique: Critique, t: TechnicalReport) -> str:
    flags = resource_scene_flags(critique, t)
    needed = {"resource_snow": "snow", "resource_night": "night", "resource_matte": "matte"}.get(look_id)
    if needed and not flags[needed]:
        return {"snow": "没有明确雪景主体；白墙、亮天空和冷色像素不是雪景依据。",
                "night": "缺少夜景光线；低亮度日光照片不能当作夜景。",
                "matte": "缺少适合柔灰哑光的中低调场景，不压暗雪景或高调画面。"}[needed]
    return ""


def apply_resource_guards(recipe: Recipe, critique: Critique, t: TechnicalReport, *, audit: list | None = None, primary: Recipe | None = None) -> Recipe:
    """Bound style extremes; an optional merged correction retains scalar headroom.

    Masks are absolute settings, so their existing exposure/recovery is retained.
    Curves belong to the style; technical correction uses separate sliders.
    """
    flags = resource_scene_flags(critique, t)
    result = recipe.model_copy(deep=True)

    def limit(obj, field, low, high, policy, path=""):
        old = getattr(obj, field)
        if old is None:
            return
        base = getattr(primary, field) if primary is not None and obj is result else 0.0
        value = max(low + base, min(high + base, old))
        if value != old:
            setattr(obj, field, value)
            if audit is not None:
                audit.append({"lesson_id": policy, "control": path + field, "before": old, "after": value})

    if flags["fog"] and critique.intent.preserve_atmosphere:
        for field, high in (("dehaze", .025), ("clarity", .06), ("contrast", .12)):
            limit(result, field, -1, high, "fog_depth")
        for index, mask in enumerate(result.masks):
            if mask.target in {"sky", "background"}:
                for field, high in (("dehaze", .04), ("clarity", .08)):
                    limit(mask, field, -1, high, "fog_depth", f"masks.{index}.")
    if flags["portrait"]:
        for field, low, high in (("clarity", -.08, .06), ("texture", -.08, .10)):
            limit(result, field, low, high, "portrait_material")
        for index, mask in enumerate(result.masks):
            if mask.target == "subject":
                for field, low, high in (("clarity", -.06, .06), ("texture", -.08, .10)):
                    limit(mask, field, low, high, "portrait_material", f"masks.{index}.")
    if flags["snow"] and critique.intent.natural_color:
        old = result.curve.rgb
        # Include the interval boundary: protecting only supplied knots can
        # leave .70..1 dark when the previous knot lies just below .70.
        points = sorted(old)
        if points:
            import numpy as np
            boundary = float(np.interp(.7, *np.array(points).T))
            if boundary < .66 and not any(x == .7 for x, _ in points):
                points = sorted([*points, (.7, boundary)])
        # Only limit upper-tone darkening of the style. Exposure/highlight
        # correction from this RAW remains independently controlled downstream.
        result.curve.rgb = [(x, max(y, x - .04) if x >= .7 else y) for x, y in points]
        if old != result.curve.rgb and audit is not None:
            audit.append({"lesson_id": "snow_luminous", "control": "curve.rgb", "before": old, "after": result.curve.rgb})
    return result
