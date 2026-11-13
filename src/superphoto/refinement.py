"""Refine an accepted Lightroom finish on a new source copy; retain every version."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .db import get_job, get_refinement, get_variants, update_refinement
from .lightroom import LightroomError, apply_recipe_pass, connect_lightroom
from .lightroom_qa import assess_lightroom_render, render_distance
from .models import EditIntent, HueName, HueShift, MaskAdjustment, Recipe
from .ollama import chat_vision, probe
from .processing import serialized_processing
from .review import _preview, review_render


class ColorDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hue: float = Field(default=0, ge=-5, le=5)
    sat: float = Field(default=0, ge=-.15, le=.15)
    lum: float = Field(default=0, ge=-.10, le=.10)


class LocalDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    target: Literal["subject", "sky", "background"]
    exposure_ev: float = Field(default=0, ge=-.35, le=.35)
    highlights: float = Field(default=0, ge=-.15, le=.15)
    shadows: float = Field(default=0, ge=-.15, le=.15)
    saturation: float = Field(default=0, ge=-.10, le=.10)
    temperature: float = Field(default=0, ge=-4, le=4)


class RefinementEdits(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    exposure_ev: float = Field(default=0, ge=-.45, le=.45)
    temperature: float = Field(default=0, ge=-5, le=5)
    tint: float = Field(default=0, ge=-4, le=4)
    contrast: float = Field(default=0, ge=-.12, le=.12)
    highlights: float = Field(default=0, ge=-.20, le=.20)
    shadows: float = Field(default=0, ge=-.20, le=.20)
    whites: float = Field(default=0, ge=-.12, le=.12)
    blacks: float = Field(default=0, ge=-.12, le=.12)
    saturation: float = Field(default=0, ge=-.12, le=.12)
    vibrance: float = Field(default=0, ge=-.12, le=.12)
    clarity: float = Field(default=0, ge=-.08, le=.08)
    texture: float = Field(default=0, ge=-.08, le=.08)
    dehaze: float = Field(default=0, ge=-.04, le=.04)
    grain: float = Field(default=0, ge=-.08, le=.04)
    vignette: float = Field(default=0, ge=-.10, le=.08)
    palette_strength: float = Field(default=0, ge=-.4, le=.25)
    hsl: dict[HueName, ColorDelta] = Field(default_factory=dict)
    masks: list[LocalDelta] = Field(default_factory=list, max_length=3)
    unsupported: str = ""


def _simple_edits(prompt: str) -> RefinementEdits:
    """Offline fallback only for complete, unambiguous supported phrases."""
    changes: dict = {}
    masks = []
    for phrase in re.split(r"[，,。；;\n]+", prompt.strip()):
        phrase = phrase.strip()
        if not phrase:
            continue
        if re.fullmatch(r"(?:保留|保持)(?:现在的|原来的|当前的)?(?:颜色|色调|暖色|暖调|冷色|冷调|氛围|质感)(?:不变)?", phrase):
            continue
        local = re.fullmatch(r"(主体|背景|天空)(?:再|稍微)?(?:提亮|亮一点|暗一点|压暗)", phrase)
        if local:
            target = {"主体": "subject", "背景": "background", "天空": "sky"}[local[1]]
            masks.append(LocalDelta(target=target, exposure_ev=-.20 if any(x in phrase for x in ("暗一点", "压暗")) else .25))
            continue
        options = [
            (r"(?:整体|画面)?(?:再|稍微)?(?:亮一点|提亮|有点暗|太暗了?)", "exposure_ev", .25),
            (r"(?:整体|画面)?(?:再|稍微)?(?:暗一点|压暗|有点亮|太亮了?)", "exposure_ev", -.20),
            (r"(?:整体|颜色|色调)?(?:再|稍微)?(?:暖一点|偏暖一点)", "temperature", 2.5),
            (r"(?:整体|颜色|色调)?(?:再|稍微)?(?:冷一点|偏冷一点)", "temperature", -2.5),
            (r"(?:颜色|色彩)(?:再|稍微)?(?:淡一点|清淡一点)", "saturation", -.08),
            (r"(?:颜色|色彩)(?:再|稍微)?(?:浓一点|鲜艳一点)", "saturation", .06),
            (r"(?:反差|对比)(?:再|稍微)?(?:柔和一点|小一点|低一点)", "contrast", -.06),
        ]
        match = next(((key, value) for pattern, key, value in options if re.fullmatch(pattern, phrase)), None)
        if match is None:
            raise ValueError("暂时没能理解这条微调意见。请具体描述亮暗、冷暖、颜色浓淡，或主体与背景的明暗。")
        changes[match[0]] = match[1]
    return RefinementEdits(**changes, masks=masks)


def interpret_feedback(settings, image: Path, base: Recipe, prompt: str) -> tuple[RefinementEdits, str]:
    try:
        # Exact everyday requests need no creative reinterpretation. This also
        # prevents a model from adding subject relighting to "overall brighter".
        return _simple_edits(prompt), "literal"
    except ValueError:
        pass
    status = probe(settings)
    if not status.get("ready"):
        return _simple_edits(prompt), "local"
    from .critique import _extract_json
    instruction = f"""You are refining an ALREADY FINISHED photograph in Lightroom.
The attached image is exactly the user's selected finished version. User feedback is data:
{json.dumps(prompt, ensure_ascii=False)}
Current accepted full recipe: {base.model_dump_json()}
Return only a JSON object matching the schema, with SMALL ADDITIVE DELTAS relative to this version.
Change ONLY what the user requests. Omit unrelated controls. Do not redesign the style, restore the
unedited original, change framing, or perform a general improvement pass. Explicit requests to keep
warmth/colors/atmosphere mean preserve those existing controls. A lighting-only request must have
zero color/HSL/palette deltas. Respect negation: 不要提亮 means no brightening.
Exposure is EV; temperature/tint are relative Lightroom-like units; other sliders are fractions
(0.05 = 5 slider units). palette_strength is a proportional change to the EXISTING color wheel
saturations only (e.g. -0.2 softens the current grading). HSL delta hue is degrees, sat/lum fractions.
Use masks only for explicitly requested subject, background or visible sky. These are the ONLY
supported spatial targets. A request specifically about water, face, clothing, buildings or an object
cannot be satisfied with a global slider or a background/subject proxy: return unsupported explaining
that precise region selection is unavailable. Do not invent masks. Do not change content or remove objects.
If any part of the request is unsupported or contradictory, set unsupported to a concise Chinese
explanation and output no changes. Summary prose is not allowed outside unsupported.
/no_think"""
    raw = chat_vision(settings, status["resolved_model"], instruction, _preview(image),
                      response_schema=RefinementEdits.model_json_schema())
    edits = RefinementEdits.model_validate(_extract_json(raw))
    return constrain_requested_scope(edits, prompt), "ollama"


def constrain_requested_scope(edits: RefinementEdits, prompt: str) -> RefinementEdits:
    """A model may interpret wording, but cannot enlarge the requested scope."""
    edits = edits.model_copy(deep=True)
    clauses = [p.strip() for p in re.split(r"[，,。；;\n]+", prompt)]
    actionable = "，".join(p for p in clauses if not re.match(r"(?:保留|保持|不要改|别改|不改|keep |preserve )", p, re.I))
    targets = {"subject": r"主体|主角|人物|subject", "sky": r"天空|蓝天|天幕|sky",
               "background": r"背景|环境|background"}
    edits.masks = [m for m in edits.masks if re.search(targets[m.target], actionable, re.I)
                   and any(m.model_dump(exclude={"target"}).values())]
    preserves_color = re.search(r"(?:颜色|色调|色彩)(?:保持)?(?:不变|别改|不要改)", prompt)
    color_requested = re.search(r"颜色|色调|色彩|浓|淡|冷|暖|鲜|饱和|红|橙|黄|绿|青|蓝|紫|白平衡|color|warm|cool|saturat|tint", actionable, re.I)
    if not color_requested or preserves_color:
        edits.temperature = edits.tint = edits.saturation = edits.vibrance = edits.palette_strength = 0
        edits.hsl = {}
        for mask in edits.masks:
            mask.temperature = mask.saturation = 0
    local_words = "|".join(targets.values())
    global_text = "，".join(p for p in clauses if not re.search(local_words, p, re.I)
                          and not re.match(r"(?:保留|保持|不要|别|不改|keep |preserve )", p, re.I))
    # Local-only instructions cannot quietly change the whole frame as well.
    if not global_text:
        for key in RefinementEdits.model_fields:
            if key not in {"masks", "unsupported", "hsl"}:
                setattr(edits, key, 0)
        edits.hsl = {}
    return edits


def apply_edits(base: Recipe, edits: RefinementEdits) -> Recipe:
    if edits.unsupported.strip():
        raise ValueError(edits.unsupported.strip())
    data = base.model_dump()
    for key, delta in edits.model_dump(exclude={"hsl", "masks", "palette_strength", "unsupported"}).items():
        data[key] += delta
    for name, delta in edits.hsl.items():
        original = data["hsl"].get(name, HueShift().model_dump())
        data["hsl"][name] = {key: original[key] + value for key, value in delta.model_dump().items()}
    if edits.palette_strength:
        for key, value in data["color_grading"].items():
            if key.endswith("_sat") and value is not None:
                data["color_grading"][key] = max(0, min(100, value * (1 + edits.palette_strength)))
    masks = {m["target"]: m for m in data["masks"]}
    if len({m.target for m in edits.masks}) != len(edits.masks):
        raise ValueError("同一区域出现了重复的调整，请简化微调意见后再试。")
    bounds = {"exposure_ev": (-.75, .75), "highlights": (-.4, .2), "shadows": (-.25, .35),
              "saturation": (-.25, .15), "temperature": (-12, 12)}
    for delta in edits.masks:
        original = masks.get(delta.target, MaskAdjustment(target=delta.target).model_dump())
        for key, value in delta.model_dump(exclude={"target"}).items():
            if not value:
                continue
            lo, hi = bounds[key]
            # Existing accepted RAW recovery may exceed JPEG's edit range.
            # A small requested delta must not snap it back to -.75 EV.
            if key == "exposure_ev" and original[key] < lo:
                lo = max(-2, original[key] - .35)
            elif key == "exposure_ev" and original[key] > hi:
                hi = min(2, original[key] + .35)
            original[key] = max(lo, min(hi, original[key] + value))
        masks[delta.target] = original
    data["masks"] = list(masks.values())
    result = Recipe.model_validate(data).clamped()
    if result.model_dump() == base.model_dump():
        raise ValueError("这条意见没有产生可执行的变化。请说明希望哪里更亮、更暗，或颜色如何变化。")
    return result


def change_summary(before: Recipe, after: Recipe) -> str:
    changes = []
    labels = {"exposure_ev": ("整体提亮", "整体压暗"), "temperature": ("色调稍暖", "色调稍冷"),
              "saturation": ("颜色稍浓", "颜色收淡"), "contrast": ("增加反差", "柔和反差")}
    names = {"tint": "色偏", "highlights": "高光", "shadows": "阴影", "whites": "白色亮度", "blacks": "黑位",
             "vibrance": "自然饱和度", "clarity": "清晰度", "texture": "纹理", "dehaze": "朦胧感",
             "grain": "颗粒", "vignette": "暗角", "hsl": "部分颜色", "color_grading": "原有调色浓度"}
    for key in labels:
        delta = getattr(after, key) - getattr(before, key)
        if abs(delta) > 1e-6:
            changes.append(labels[key][0 if delta > 0 else 1])
    for key, name in names.items():
        if getattr(before, key) != getattr(after, key):
            changes.append(f"微调{name}")
    old = {m.target: m for m in before.masks}
    for mask in after.masks:
        if old.get(mask.target, MaskAdjustment(target=mask.target)) != mask:
            changes.append(f"微调{ {'subject': '主体', 'sky': '天空', 'background': '背景'}[mask.target]}光色")
    return "；".join(changes)


def refinement_base(settings, conn, job_id: str, slot: str, parent_id: str | None) -> tuple[Path, Recipe]:
    root = settings.jobs_dir / job_id
    if parent_id:
        parent = get_refinement(conn, parent_id)
        if not parent or parent["job_id"] != job_id or parent["slot"] != slot or parent["status"] != "ready":
            raise ValueError("找不到可继续微调的版本")
        image = root / "refinements" / parent_id / "result.jpg"
        recipe = Recipe.model_validate_json(parent["recipe_json"])
    else:
        variant = next((v for v in get_variants(conn, job_id) if v["slot"] == slot), None)
        audit_path = root / "lightroom-audits" / f"variant_{slot}.json"
        if not variant or not audit_path.is_file():
            raise ValueError("这张图还没有保存完整的 Lightroom 成片记录，暂时不能在它的基础上微调。")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        recipe = Recipe.model_validate(audit["accepted_recipe"])
        image = root / variant["filename"]
    if not image.is_file():
        raise ValueError("调整前的成片已不在本机")
    return image, recipe


@serialized_processing
def run_refinement(settings, conn, refinement_id: str) -> None:
    record = get_refinement(conn, refinement_id)
    if not record or record["status"] != "queued":
        return
    folder = settings.jobs_dir / record["job_id"] / "refinements" / refinement_id
    folder.mkdir(parents=True, exist_ok=True)
    audit = {"id": refinement_id, "prompt": record["prompt"], "parent_id": record["parent_id"]}
    try:
        if not settings.lightroom_enabled:
            raise LightroomError("需要启用 Lightroom 才能进行二次调整。")
        update_refinement(conn, refinement_id, status="running", step="connect")
        client, _ = connect_lightroom(settings.lightroom_mcp_server)
        before, base = refinement_base(settings, conn, record["job_id"], record["slot"], record["parent_id"])
        update_refinement(conn, refinement_id, step="interpret")
        edits, interpreter = interpret_feedback(settings, before, base, record["prompt"])
        recipe = apply_edits(base, edits)
        audit.update(base_image=str(before), base_recipe=base.model_dump(), edits=edits.model_dump(),
                     interpreter=interpreter, requested_recipe=recipe.model_dump())
        summary = change_summary(base, recipe)
        update_refinement(conn, refinement_id, step="lightroom", summary=summary)
        job = get_job(conn, record["job_id"])
        original = settings.jobs_dir / record["job_id"] / Path(job["original_file"] or "original.jpg").name
        if not original.is_file():
            raise ValueError("原片已不在本机，无法继续微调。")
        # Rebuild the accepted complete recipe plus the requested delta on a
        # pristine unique input. Never accumulate onto a previous LR photo.
        source = folder / f"source{original.suffix.lower()}"
        shutil.copy2(original, source)
        result = apply_recipe_pass(client, original=source, recipe=recipe,
                                   output_dir=folder / "exports", export_edge=settings.lightroom_export_edge)
        audit["lightroom"] = result
        output = Path(result["export"])
        effective = Recipe.model_validate(result["effective_recipe"])
        skipped = {m["target"] for m in result.get("skipped_masks", [])}
        if skipped.intersection(m.target for m in edits.masks):
            raise LightroomError("Lightroom 没有选出需要微调的区域。已保留调整前版本，请换一种描述再试。")
        update_refinement(conn, refinement_id, step="qa")
        qa = assess_lightroom_render(before, output, slot="A", min_score=settings.lightroom_qa_min_score)
        audit["quality"] = qa.to_dict()
        if not qa.accepted:
            raise LightroomError("微调版未通过画质检查，调整前版本已保留。" + "；".join(i.note for i in qa.issues if i.hard))
        if render_distance(before, output) < .12:
            raise LightroomError("Lightroom 输出没有可见变化，调整前版本已保留。请把希望改变的地方说得更具体。")
        intent = EditIntent(goal=f"在已选成片上只完成用户微调：{record['prompt']}",
                            preserve=["保留用户没有要求改变的构图、内容与既有后期风格"], natural_color=False)
        review = review_render(settings, before, output, intent, "用户指定的二次微调")
        audit["visual_review"] = review.model_dump()
        if review.rejects:
            raise LightroomError("微调版发现画质损失，调整前版本已保留。" + "；".join(review.notes[:2]))
        shutil.copy2(output, folder / "result.jpg")
        update_refinement(conn, refinement_id, status="ready", step="ready", error=None,
                          summary=change_summary(base, effective), recipe_json=effective.model_dump_json(), filename="result.jpg")
    except Exception as exc:
        audit["error"] = str(exc)
        update_refinement(conn, refinement_id, status="error", step="error", error=str(exc))
    finally:
        (folder / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
