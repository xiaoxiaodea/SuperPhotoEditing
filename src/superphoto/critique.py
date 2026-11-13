from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import correction_from_technical, guess_genre
from .composition import clamp_geometry, suggest_geometry
from .config import Settings
from .models import CorrectionHint, Critique, EditIntent, GeometryHint, Issue, MaskAdjustment, RegionBox, TechnicalReport
from .ollama import chat_vision, probe
from .look_routes import route_legend
from .preference import look_match, ranked_look_ids
from .recipes import LOOKS, allowed_look_ids

SCHEMA_HINT = """
{
  "genre": "portrait|landscape|street|night|indoor|product|general",
  "summary": "中文，两到三句，指出这张图真正的问题，不要空夸",
  "issues": [{"dim": "exposure|wb|highlights|shadows|color|composition|noise|sharpness|contrast", "severity": 0.0, "note": "中文"}],
  "do_not": ["中文，禁止的过度处理"],
  "correction": {
    "exposure_ev": 0.0, "temperature": 0.0, "tint": 0.0,
    "highlights": 0.0, "shadows": 0.0, "whites": 0.0, "blacks": 0.0,
    "contrast": 0.0, "saturation": 0.0, "vibrance": 0.0, "clarity": 0.0, "dehaze": 0.0
  },
  "looks": ["faithful", "kdrama_tears"],
  "lut_mix": 0.5,
  "geometry": {"rotate_deg": 0.0, "crop": null, "why": ""},
  "subject_note": "画面主体是什么",
  "intent": {
    "goal": "这张照片的视觉重点，以及如何组织人物、前景和远景的关系",
    "preserve": ["必须保留的现场光色或空间关系"],
    "natural_color": true, "preserve_atmosphere": true, "preserve_black_depth": true, "locked_controls": [],
    "subject_box": null, "subject_confidence": 0.0,
    "masks": [{"target": "subject", "exposure_ev": 0.0, "shadows": 0.0, "highlights": 0.0,
               "contrast": 0.0, "whites": 0.0, "blacks": 0.0,
               "temperature": 0.0, "saturation": 0.0, "clarity": 0.0, "reason": "区域问题与处理目的"}]
  },
  "clutter": [{"box": [0.0, 0.0, 0.2, 0.3], "note": "边缘干扰，可压暗或裁掉"}]
}
"""


def critique_photo(
    settings: Settings,
    preview_path: Path,
    technical: TechnicalReport,
) -> Critique:
    local = local_critique(technical)
    status = probe(settings)
    vision_ready = bool(status.get("ready") and status.get("resolved_model"))
    if vision_ready:
        try:
            remote = ollama_critique(settings, preview_path, technical, status["resolved_model"])
            return _merge_with_local(remote, local)
        except Exception as exc:
            local.summary = (local.summary + f" （Ollama 诊断失败：{_format_critique_error(exc)}）").strip()
    if settings.xai_api_key:
        try:
            remote = grok_critique(settings, preview_path, technical)
            return _merge_with_local(remote, local)
        except Exception:
            pass
    note = (
        "（视觉模型输出无法使用，已用本地技术分析）"
        if vision_ready
        else "（视觉模型未就绪，已用本地技术分析）"
    )
    if note not in local.summary:
        local.summary = (local.summary + " " + note).strip()
    return local


def local_critique(technical: TechnicalReport) -> Critique:
    genre = guess_genre(technical)
    correction = correction_from_technical(technical)
    issues: list[Issue] = []
    do_not: list[str] = []

    if technical.highlight_clip > 0.01:
        issues.append(
            Issue(
                dim="highlights",
                severity=min(1.0, 0.4 + technical.highlight_clip * 12),
                note="高光已有裁切，优先压高光而不是继续加曝光。",
            )
        )
        do_not.append("不要为了提亮暗部把天空或窗户拉爆")
    if technical.mean_luma < 0.08 and correction.exposure_ev > 0.15:
        issues.append(
            Issue(
                dim="exposure",
                severity=min(1.0, 0.5 + (0.08 - technical.mean_luma) * 4),
                note=f"整体偏暗（线性均值 {technical.mean_luma:.2f}），建议约 +{correction.exposure_ev:.2f} EV。",
            )
        )
    elif technical.highlight_clip > 0.03 and correction.exposure_ev < 0:
        issues.append(
            Issue(
                dim="exposure",
                severity=0.45,
                note="高光偏多，先收回曝光再谈风格。",
            )
        )
    if technical.black_clip > 0.12 and technical.mean_luma < 0.12:
        issues.append(
            Issue(
                dim="shadows",
                severity=min(1.0, technical.black_clip * 3),
                note="阴影丢失细节，适量抬暗部，但不要做成雾蒙蒙的平图。",
            )
        )
    scenic = (
        technical.top_third_blue > 0.18
        or technical.green_fraction > 0.16
        or technical.skin_fraction > 0.08
    )
    if not scenic:
        if technical.temperature_bias < -0.18:
            issues.append(
                Issue(
                    dim="wb",
                    severity=min(1.0, abs(technical.temperature_bias)),
                    note="画面偏冷/偏青，白平衡可略暖。",
                )
            )
        elif technical.temperature_bias > 0.28:
            issues.append(
                Issue(
                    dim="wb",
                    severity=min(1.0, technical.temperature_bias),
                    note="色温偏暖，注意是否过橙。",
                )
            )
    if technical.saturation_mean > 0.32:
        issues.append(
            Issue(
                dim="color",
                severity=0.4,
                note="饱和已经不低，风格层请克制，避免再加全局饱和。",
            )
        )
        do_not.append("不要再提高全局饱和度")
    if technical.sharpness < 0.035:
        issues.append(
            Issue(
                dim="sharpness",
                severity=0.35,
                note="锐度偏低，可能是微距失焦或预览缩小，清晰度滑条只能适度补偿。",
            )
        )
    if technical.noise > 0.08 and technical.exif.get("ISOSpeedRatings"):
        issues.append(
            Issue(
                dim="noise",
                severity=0.4,
                note="高感噪声明显，不要用强清晰度放大颗粒。",
            )
        )
        do_not.append("不要加高清晰度")
    if abs(technical.horizon_tilt_deg) > 0.7 and technical.horizon_confidence > 0.32:
        issues.append(
            Issue(
                dim="composition",
                severity=min(0.7, 0.3 + abs(technical.horizon_tilt_deg) / 6),
                note=f"地平线大约倾斜 {technical.horizon_tilt_deg:.1f}°，可以小幅校正，不要矫枉过正。",
            )
        )
    if technical.margin_waste > 0.32 and technical.thirds_error > 0.12:
        issues.append(
            Issue(
                dim="composition",
                severity=min(0.6, technical.margin_waste),
                note="边缘偏空、主体位置不稳，可适度裁切，不要裁掉环境和呼吸感。",
            )
        )
        do_not.append("不要为了填满画面把主体裁死")

    if not issues:
        issues.append(
            Issue(dim="contrast", severity=0.25, note="技术面比较完整，重点放在风格而不是抢救。")
        )

    looks = _looks_for_genre(genre, technical)
    summary = _summary(genre, issues, technical)
    if not do_not:
        do_not.append("校正与风格分开，先做中性再上味道")
        do_not.append("不要整份套 LUT，按画面减淡")
    lut_mix = 0.62
    if technical.saturation_mean > 0.46:
        lut_mix = 0.54
    if technical.dynamic_spread > 0.78:
        lut_mix = min(lut_mix, 0.56)
    if technical.source_kind in {"jpeg", "png"}:
        lut_mix *= 0.98

    return Critique(
        genre=genre,
        summary=summary,
        issues=issues[:6],
        do_not=do_not[:5],
        correction=correction,
        looks=looks,
        source="local",
        lut_mix=float(np.clip(lut_mix, 0.48, 0.72)),
        geometry=suggest_geometry(technical, genre, "B"),
    )


def _look_legend() -> str:
    return "\n".join(f"- {look.id}: {look.description}" for look in LOOKS.values())


def _vlm_prompt(technical: TechnicalReport) -> str:
    from .style_selection import preset_legend
    looks = preset_legend()
    measurements = technical.model_dump(exclude={"histogram", "shadow_clip"})
    measurements["low_tone_area_not_clipping"] = technical.shadow_clip
    return f"""You are a photography director following published master practice:
Thomas Heaton (correct highlights, then shadows, then WB; almost never global saturation),
Ted Gore (do not blast every green; saturate only where light already is; stay inside the context of nature),
portrait colorists (fix skin WB first; often KEEP shadows for chiaroscuro).

You receive a JPEG viewing preview of a {technical.source_kind.upper()} SOURCE plus lab measurements.
The preview format does not change the source file's adjustment headroom.
First name the picture's INTENT (low-key pond, open landscape, studio portrait).
Then say what is actually weak. Separate correction from look.
Inspect each of sky, focal subject, foreground and edges before choosing the main weakness.
Dark but textured hills are not lost detail. black_clip measures near-black output; low_tone_area_not_clipping
only describes normal dark tones. Never say shadows are clipped based on that area or a dark-looking thumbnail alone.
For a pale RAW sky, consider a meaningful recovery rather than lifting the whole photo:
global highlights -0.3..-0.8 as needed, then sky exposure roughly -0.7..-1.6 with contrast +.15..+.35
and whites +.2..+.6 to keep clouds luminous. These are conditional recovery options, not a universal preset.
For a JPEG or an intentionally pale/hazy sky, preserve atmosphere and use smaller local changes.
The default is natural photographic finishing. First design the visual hierarchy:
- intent.goal names the focal subject, the light direction and foreground/background relationship.
- A person in a landscape is still a subject. Do not overlook a dark face because most pixels are bright mountains.
- Preserve aerial perspective: distant ridges SHOULD be paler and softer than near ridges. Haze is not automatically a defect.
- Use intent.masks for native Lightroom subject/sky/background masks, at most one per target.
- Subject exposure typically +0.15..+0.40 EV when needed, background -0.05..-0.20 EV only if it competes.
- Local exposure normally -0.75..+0.75 EV; a visibly washed-out, recoverable RAW sky may use -2..0 EV
  with positive local contrast (up to .50) and whites (up to .75) to preserve luminous clouds.
  For JPEG, subject and background keep contrast/whites within -.30..+.30 and blacks -.25..+.25.
  Local highlights -0.40..+0.20, shadows -0.25..+0.35,
  saturation -0.25..+0.15, clarity -0.15..+0.15, temperature -12..+12.
- Empty masks if they would not help. Do not lift an intentional silhouette. Do not change every color to make an edit visible.
- subject_box is a normalized object with left/top/right/bottom, or null if uncertain; report subject_confidence.
- Translate do_not into the intent protection flags. Keep natural_color/preserve_atmosphere/preserve_black_depth true by default.
- Inspect each named region before writing its preserved color. Reflections can be warm while the
  surrounding city is cool. Whole-frame temperature_bias does not describe a small reflected highlight.
  If a region's hue is uncertain, preserve its existing color relationship without inventing a hue.
- locked_controls explicitly forbids global look moves: allowed names are exposure_ev, temperature, tint, hue, grain, clarity, dehaze.
Choose a visible, justified improvement in light hierarchy; a tiny edit is not automatically a good edit.
Do not label a whole hillside as a selectable subject. Never suggest generative fill or face-changing retouch.

JPEG/PNG files are already developed. Be conservative:
- exposure_ev is usually 0. Never go beyond -0.3..+0.25 on a JPEG.
- Low-key is a choice. If the subject (flower, face, peak) is already well exposed, leave exposure at 0. Do not lift a dark pond.
- shadows at most +0.10. highlights at most -0.25.
- Distinguish ambient light from reflected sunlight; preserve the colors actually visible in each region.
- Do not raise global saturation. If color is loud, lower it.

Executable presets with their visual character and usage conditions (suggest 3 compatible IDs, no mandatory slot):
{looks}

Choose from the scene's subject, light, palette and intended mood, not just its genre or a famous name.
Warm earth is not skin, a dark daylight picture is not necessarily night, and buildings behind a person
do not make the subject architectural. Different photographs may need entirely different preset families.
These recommendations enter preset ranking; diagnosis and local light come first.
LUT / XMP packs are REFERENCES, never apply at 100%. Do not pick a look just because the title is cinematic.
- lut_mix is 0.3-0.7: how much of the look to keep after you read THIS picture.
- If the photo is already saturated, contrasty, or a finished JPEG, stay near 0.35-0.5.
- Drop LUT pieces that fight the scene (don't khaki a lush green pond; don't orange a landscape).

Geometry: only if it clearly helps. Otherwise rotate_deg=0 and crop=null.
- rotate_deg is typically -2..2. 0 if unsure. Never dutch-angle on purpose.
- crop is [left, top, right, bottom] in 0-1, or null. Do not crop faces off. Do not crop a low-key pond just to fill the frame. Max trim about 15%.
- clutter boxes mark distractions to suppress (desaturate/darken) or crop. Empty list if none.
We do NOT generative-erase objects. Name the subject in subject_note.

Correction units (same as Lightroom-like sliders, 0 = no-op):
- exposure_ev: stops, typically -1.2..1.2
- temperature: -30..30, positive = warmer
- highlights/shadows/whites/blacks/contrast/saturation/vibrance/clarity/dehaze: about -1..1

Technical measurements (use black_clip/highlight_clip to distinguish clipping from ordinary dark/bright tones;
mean exposure is not an artistic target):
{json.dumps(measurements, ensure_ascii=False, indent=2)}

/no_think
Return JSON only, matching this shape (Chinese for summary/notes/do_not). No markdown, no commentary:
{SCHEMA_HINT}
"""


def ollama_critique(
    settings: Settings,
    preview_path: Path,
    technical: TechnicalReport,
    model: str,
) -> Critique:
    import base64

    image_b64 = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    text = chat_vision(settings, model, _vlm_prompt(technical), image_b64)
    data = _extract_json(text)
    return _critique_from_dict(data, source="ollama")


def grok_critique(
    settings: Settings,
    preview_path: Path,
    technical: TechnicalReport,
) -> Critique:
    import base64

    from openai import OpenAI

    image_b64 = base64.b64encode(preview_path.read_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{image_b64}"
    prompt = _vlm_prompt(technical)

    client = OpenAI(
        api_key=settings.xai_api_key,
        base_url=settings.xai_base_url,
        timeout=90.0,
    )
    response = client.responses.create(
        model=settings.xai_model,
        store=False,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )
    text = getattr(response, "output_text", None) or _output_text_fallback(response)
    data = _extract_json(text)
    return _critique_from_dict(data, source="grok")


def compose_look_ids(
    critique: Critique,
    preferred: list[str],
    technical: TechnicalReport | None = None,
) -> list[str]:
    """Always 3 unique looks. Slot A is faithful. B/C must match this picture."""
    picked: list[str] = []
    allowed = set(allowed_look_ids())
    if "faithful" in allowed:
        picked.append("faithful")

    def ok(look_id: str) -> bool:
        if look_id in picked or look_id not in allowed or look_id == "faithful":
            return False
        if technical is None:
            return True
        return look_match(look_id, technical, critique.genre) >= 0.18

    for look_id in list(critique.looks) + preferred:
        if not ok(look_id):
            continue
        picked.append(look_id)
        if len(picked) == 3:
            break
    if technical is not None:
        for look_id in ranked_look_ids(technical, critique.genre):
            if len(picked) == 3:
                break
            if ok(look_id):
                picked.append(look_id)
    for fallback in _fallbacks_for(critique.genre):
        if len(picked) == 3:
            break
        if fallback not in picked and fallback in allowed and ok(fallback):
            picked.append(fallback)
    for fallback in _fallbacks_for(critique.genre):
        if len(picked) == 3:
            break
        if fallback not in picked and fallback in allowed:
            picked.append(fallback)
    return picked[:3]


def _fallbacks_for(genre: str) -> tuple[str, ...]:
    return {
        "landscape": ("landscape_filmic", "dune", "five_cm", "moody_lowkey"),
        "portrait": ("filmic_portrait_warm", "kdrama_tears", "romance_gold", "editorial_crisp"),
        "street": ("cinematic_teal_orange", "skam", "wong_kar_wai", "moody_lowkey"),
        "night": ("moody_lowkey", "cinematic_teal_orange", "hanabi", "tokyo_ghoul"),
        "indoor": ("filmic_portrait_warm", "kdrama_tears", "unnatural"),
        "product": ("editorial_crisp", "filmic_portrait_warm"),
        "general": ("landscape_filmic", "filmic_portrait_warm", "cinematic_teal_orange"),
    }.get(genre, ("landscape_filmic", "filmic_portrait_warm", "cinematic_teal_orange"))


def _looks_for_genre(genre: str, technical: TechnicalReport) -> list[str]:
    ranked = ranked_look_ids(technical, genre)
    picked = ["faithful"]
    allowed = set(allowed_look_ids())
    for look_id in ranked:
        if look_id in picked or look_id not in allowed:
            continue
        picked.append(look_id)
        if len(picked) == 3:
            break
    for look_id in _fallbacks_for(genre):
        if len(picked) == 3:
            break
        if look_id not in picked and look_id in allowed:
            picked.append(look_id)
    return picked[:3]


def _summary(genre: str, issues: list[Issue], technical: TechnicalReport) -> str:
    top = "；".join(i.note for i in issues[:2])
    kind = {
        "portrait": "人像",
        "landscape": "风光",
        "night": "夜景/低照度",
        "street": "街头",
        "indoor": "室内",
        "product": "产品",
        "general": "未分类",
    }.get(genre, genre)
    src = "RAW" if technical.source_kind == "raw" else technical.source_kind.upper()
    return f"{kind}，{src}。{top}"


def _merge_with_local(remote: Critique, local: Critique) -> Critique:
    # A deliberate zero adjustment is valid. Do not replace the photographer's
    # decision to preserve exposure or framing with histogram heuristics.
    looks = [lid for lid in remote.looks if lid in LOOKS]
    if "faithful" not in looks:
        looks = ["faithful", *looks]
    remote.looks = looks[:4] or local.looks
    if not remote.issues:
        remote.issues = local.issues
    if not remote.summary:
        remote.summary = local.summary
    remote.lut_mix = float(np.clip(remote.lut_mix or local.lut_mix, 0.28, 0.75))
    if remote.geometry.apply:
        remote.geometry = clamp_geometry(remote.geometry)
    if not remote.clutter:
        remote.clutter = local.clutter
    if not remote.subject_note:
        remote.subject_note = local.subject_note
    return remote


def _critique_from_dict(data: dict[str, Any], source: str) -> Critique:
    issues = []
    for item in data.get("issues") or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        try:
            row["severity"] = min(1.0, max(0.0, float(row.get("severity", 0))))
        except (TypeError, ValueError):
            row["severity"] = 0.3
        try:
            issues.append(Issue.model_validate(row))
        except Exception:
            continue
    looks = [x for x in (data.get("looks") or []) if isinstance(x, str) and x in LOOKS]
    corr_raw = data.get("correction") or {}
    if isinstance(corr_raw, dict):
        corr_raw = {k: v for k, v in corr_raw.items() if k in CorrectionHint.model_fields}
        correction = CorrectionHint.model_validate(corr_raw)
    else:
        correction = CorrectionHint()
    lut_mix = data.get("lut_mix", 0.55)
    try:
        lut_mix = float(lut_mix)
    except (TypeError, ValueError):
        lut_mix = 0.55
    geometry = _geometry_from_dict(data.get("geometry"))
    clutter = _clutter_from_dict(data.get("clutter"))
    return Critique(
        genre=str(data.get("genre") or "general"),
        summary=str(data.get("summary") or ""),
        issues=issues,
        do_not=[str(x) for x in (data.get("do_not") or [])][:6],
        correction=correction,
        looks=looks,
        source=source if source in {"ollama", "grok", "local"} else "local",
        lut_mix=float(np.clip(lut_mix, 0.25, 0.8)),
        geometry=geometry,
        clutter=clutter,
        subject_note=str(data.get("subject_note") or ""),
        intent=_intent_from_dict(data.get("intent")),
    )


def _intent_from_dict(raw: object) -> EditIntent:
    if not isinstance(raw, dict):
        return EditIntent()
    data = dict(raw)
    masks = []
    raw_masks = data.pop("masks", [])
    for item in raw_masks if isinstance(raw_masks, list) else []:
        try:
            mask = MaskAdjustment.model_validate(item)
            if not mask.is_noop() and mask.target not in {m.target for m in masks}:
                masks.append(mask)
        except (ValueError, TypeError):
            continue
    # An invalid optional box must not discard the goal or color protections.
    valid = {}
    for key, value in data.items():
        if key not in EditIntent.model_fields:
            continue
        try:
            EditIntent.model_validate({key: value})
            valid[key] = value
        except (ValueError, TypeError):
            continue
    intent = EditIntent.model_validate(valid)
    if data.get("subject_confidence") == 0 and intent.subject_box is None:
        # An explicit absence of a selectable object cannot support a native
        # subject mask (or its inverse). Empty selections can also block LR's UI.
        masks = [m for m in masks if m.target == "sky"]
    return intent.model_copy(update={"masks": masks[:3]})


def _clutter_from_dict(raw: object) -> list[RegionBox]:
    if not isinstance(raw, list):
        return []
    out: list[RegionBox] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        box = item.get("box") or item.get("rect")
        try:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                left, top, right, bottom = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            else:
                left = float(item.get("left", 0))
                top = float(item.get("top", 0))
                right = float(item.get("right", 1))
                bottom = float(item.get("bottom", 1))
        except (TypeError, ValueError):
            continue
        if right - left < 0.04 or bottom - top < 0.04:
            continue
        out.append(
            RegionBox(
                left=float(np.clip(left, 0, 1)),
                top=float(np.clip(top, 0, 1)),
                right=float(np.clip(right, 0, 1)),
                bottom=float(np.clip(bottom, 0, 1)),
                note=str(item.get("note") or ""),
                kind="clutter",
            )
        )
    return out[:6]


def _geometry_from_dict(raw: object) -> GeometryHint:
    if not isinstance(raw, dict):
        return GeometryHint()
    rotate = 0.0
    try:
        rotate = float(raw.get("rotate_deg") or 0.0)
    except (TypeError, ValueError):
        rotate = 0.0
    crop = raw.get("crop")
    left, top, right, bottom = 0.0, 0.0, 1.0, 1.0
    if isinstance(crop, (list, tuple)) and len(crop) == 4:
        try:
            left, top, right, bottom = (float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3]))
        except (TypeError, ValueError):
            left, top, right, bottom = 0.0, 0.0, 1.0, 1.0
    elif all(k in raw for k in ("crop_left", "crop_top", "crop_right", "crop_bottom")):
        try:
            left = float(raw.get("crop_left", 0))
            top = float(raw.get("crop_top", 0))
            right = float(raw.get("crop_right", 1))
            bottom = float(raw.get("crop_bottom", 1))
        except (TypeError, ValueError):
            pass
    why = str(raw.get("why") or "")
    apply = abs(rotate) > 0.15 or left > 0.004 or top > 0.004 or right < 0.996 or bottom < 0.996
    return clamp_geometry(
        GeometryHint(
            rotate_deg=rotate,
            crop_left=left,
            crop_top=top,
            crop_right=right,
            crop_bottom=bottom,
            apply=apply,
            why=why,
        )
    )


def _format_critique_error(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, json.JSONDecodeError) or any(
        token in msg for token in ("Expecting", "Unterminated", "delimiter", "no JSON")
    ):
        return "模型输出不是完整 JSON（多半被截断）"
    if len(msg) > 120:
        return msg[:117] + "…"
    return msg


_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("no JSON object in model output")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```.*$", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    blob = text[start:].strip()
    for candidate in (blob, _repair_json(blob)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data:
            return data
    raise ValueError("no JSON object in model output")


def _repair_json(text: str) -> str:
    """Close a truncated JSON object, keeping the last complete value."""
    s = text.strip().translate(_SMART_QUOTES)
    in_str = False
    esc = False
    stack: list[str] = []
    complete_at = 0
    i = 0
    n = len(s)

    def after_value(pos: int) -> None:
        nonlocal complete_at
        complete_at = pos

    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if stack and stack[-1] == "{key":
                    stack[-1] = "{val"
                else:
                    after_value(i + 1)
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "{":
            stack.append("{key")
            i += 1
            continue
        if ch == "[":
            stack.append("[")
            i += 1
            continue
        if ch == "}":
            if stack and stack[-1] in {"{key", "{val"}:
                stack.pop()
                after_value(i + 1)
            i += 1
            continue
        if ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                after_value(i + 1)
            i += 1
            continue
        if ch == ":":
            i += 1
            continue
        if ch == ",":
            if stack and stack[-1] == "{val":
                stack[-1] = "{key"
            i += 1
            continue
        if ch in "tfn":
            for word in ("true", "false", "null"):
                if s.startswith(word, i):
                    after_value(i + len(word))
                    i += len(word)
                    break
            else:
                break
            continue
        if ch == "-" or ch.isdigit():
            j = i + 1
            while j < n and s[j] in "0123456789.eE+-":
                j += 1
            token = s[i:j]
            if not token or token[-1] in ".eE+-":
                break
            after_value(j)
            i = j
            continue
        break

    prefix = s[:complete_at].rstrip()
    prefix = re.sub(r",\s*$", "", prefix)
    if not prefix:
        return "{}"
    return prefix + _unclosed_suffix(prefix)


def _unclosed_suffix(text: str) -> str:
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return "".join(reversed(stack))


def _output_text_fallback(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "text", None):
                chunks.append(content.text)
            elif getattr(content, "type", "") == "output_text":
                chunks.append(getattr(content, "text", ""))
    if not chunks and isinstance(response, dict):
        return json.dumps(response)
    return "\n".join(chunks)
