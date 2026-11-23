"""Compare actual rendered photographs against the per-picture editing intent.

This is an evidence-bearing preference judgment, separate from technical QA.
An unavailable model is reported as unavailable, never as a perfect score.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, Field, computed_field

from .config import Settings
from .ingest import image_to_srgb, srgb_profile_bytes
from .models import CorrectionHint, EditIntent, Recipe
from .vision import chat_vision, probe, provider_from, safe_error


class VisualDamage(BaseModel):
    kind: Literal["clipping", "shadow_crush", "halos", "color_cast", "detail_loss"]
    region: str
    evidence: str
    severity: float = Field(ge=0, le=1)


class VisualReview(BaseModel):
    decision: Literal["better", "same", "worse", "unavailable"] = "unavailable"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list, max_length=6)
    correction: CorrectionHint = Field(default_factory=CorrectionHint)
    source: str = "unavailable"
    damage: list[VisualDamage] = Field(default_factory=list, max_length=4)
    damage_confirmed: bool = False
    verification: dict = Field(default_factory=dict)
    transport_audit: dict = Field(default_factory=dict)
    verification_transport_audit: dict = Field(default_factory=dict)

    @computed_field
    @property
    def rejects(self) -> bool:
        return self.decision == "worse" and self.confidence >= 0.80 and self.damage_confirmed


def _preview(path: Path) -> str:
    with Image.open(path) as image:
        image = image_to_srgb(ImageOps.exif_transpose(image))
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=90, icc_profile=srgb_profile_bytes())
    return base64.b64encode(stream.getvalue()).decode("ascii")


def review_render(settings: Settings, before: Path, after: Path, intent: EditIntent, stage: str) -> VisualReview:
    if not settings.visual_review_enabled:
        return VisualReview(notes=["视觉复核未启用"])
    status = probe(settings)
    if not status.get("ready") or not status.get("resolved_model"):
        return VisualReview(notes=["视觉模型不可用；仅完成技术检查"], transport_audit={"selection": status})
    prompt = f"""Compare the two numbered panels on this SINGLE image board of the SAME photograph.
LEFT panel 1 (first) = reference before this step; RIGHT panel 2 (second) = completed proposed result.
Judge whether panel 2 realizes its specific photographic design. First/second always refer to these panels.
Stage: {stage}. The automatically inferred editing plan (NOT verified scene facts):
{json.dumps(intent.model_dump(exclude={"masks"}), ensure_ascii=False)}
First inspect the reference itself. Its pixels are authoritative about existing colors and detail.
The plan's preserve descriptions can be wrong: do not claim a warm reflection was originally cool
because the text says so. Ignore an inferred protection when the reference contradicts it.
reference_checks and resource_checks are conditional lessons from separately studied photographs and
tutorials, not facts about these two images. Apply only checks relevant to what is actually visible here; never demand imitation of
a master's colors, staging, light direction or subject position. Do not reject a valid image for
failing an inapplicable study check.
Inspect focal hierarchy, subject material, direction of light, foreground vs distant layers,
skin/sky/foliage coherence and highlight transitions. A warm luminous design and a deeper sculpted
design should differ. A darker midtone, warmer highlight, or softer reflection can be intentional.
Do not call an intended mood change damage. Equally, almost unchanged is NOT a successful finished
design if the named weaknesses remain. Report the visible benefit or remaining problem, not identical
object positions: geometry/content preservation is not the aesthetic question.
Separate preference from DAMAGE. Damage requires a specific region and visible loss (clipped detail,
blocked shadows, edge halos, an unintended color cast, or destroyed texture). Generic 'less natural'
or 'weaker layers' is not evidence. Empty damage when unsure. Do not hallucinate detail loss from a
brightness change. More visible distant detail is not detail_loss; require actual halos or lost texture.
A subtle successful edit can be better; same means no convincing improvement.
Return JSON only, Chinese notes with visible evidence:
{{"decision":"better|same|worse","confidence":0.0,"notes":["specific visual observation"],
"damage":[{{"kind":"clipping|shadow_crush|halos|color_cast|detail_loss","region":"具体位置",
"evidence":"可见损坏，而非风格偏好","severity":0.0}}],
"correction":{{"exposure_ev":0.0,"highlights":0.0,"shadows":0.0,"contrast":0.0,
"saturation":0.0,"vibrance":0.0,"temperature":0.0,"tint":0.0}}}}
Correction is a SMALL delta only if worse: exposure +/-0.12 EV, tone/color sliders +/-0.08,
temperature/tint +/-2. Never change protected color or atmosphere. Otherwise all deltas zero.
/no_think"""
    transport_audit: dict = {}
    verification_transport_audit: dict = {}
    try:
        from .critique import _extract_json
        from .pair_review import build_pair_board

        before_preview, after_preview = _preview(before), _preview(after)
        if before_preview == after_preview:
            return VisualReview(decision="same", confidence=1.0, source="identical-render",
                                notes=["参考与输出相同，没有新增调整"])
        raw = chat_vision(settings, status["resolved_model"], prompt,
                          [build_pair_board(before_preview, after_preview)], audit=transport_audit,
                          selection=status)
        data = _extract_json(raw)
        data["source"] = provider_from(status, transport_audit)
        data["damage_confirmed"] = False
        data["verification"] = {}
        data["transport_audit"] = transport_audit
        data["verification_transport_audit"] = verification_transport_audit
        review = VisualReview.model_validate(data)
        severe = [d for d in review.damage if d.severity >= .65 and d.region.strip() and d.evidence.strip()]
        if review.decision == "worse" and review.confidence >= .80 and severe:
            # Do not feed the first review's allegations back to the verifier:
            # a false scene description otherwise anchors both judgments.
            verification = """Independently inspect two numbered panels on this SINGLE image board.
LEFT panel 1 (FIRST) is the proposed result; RIGHT panel 2 (SECOND) is the original reference.
First/second always refer to these panels of the same photograph. Start by describing the actual
colors and detail in SECOND, then compare FIRST. No prior description is authoritative.
Only confirm newly introduced objective damage: clipped texture, blocked shadows, edge halos,
unintended color contamination, or destroyed texture. A mood preference, different color intensity,
warmer light, less haze, or stronger contrast alone is not damage. Existing warmth in SECOND
cannot be called a new warm cast in FIRST. More visible detail is not destroyed detail.
If uncertain, confirmed=false. Report what is actually visible, not a hypothetical edit.
Return JSON {"confirmed":false,"confidence":0.0,
"kind":"none|clipping|shadow_crush|halos|color_cast|detail_loss",
"reference_observation":"原片中具体区域的实际颜色与细节",
"result_observation":"成片中同一区域的实际颜色与细节","evidence":"新损坏的可见依据或没有发现损坏"}.
/no_think"""
            try:
                checked = _extract_json(chat_vision(settings, status["resolved_model"], verification,
                    [build_pair_board(after_preview, before_preview)], audit=verification_transport_audit,
                    selection=status))
                review.verification = checked
                review.damage_confirmed = (
                    checked.get("confirmed") is True
                    and float(checked.get("confidence", 0)) >= .80
                    and checked.get("kind") in {d.kind for d in severe}
                    and all(isinstance(checked.get(key), str) and checked[key].strip()
                            for key in ("reference_observation", "result_observation", "evidence"))
                )
            except (ValueError, TypeError, OSError, RuntimeError):
                # Keep the initial opinion, but an unavailable verification
                # cannot authorize undoing a completed artistic decision.
                review.damage_confirmed = False
        # The model cannot escape the small-step contract even if it misreads
        # slider units. Artistic judgments never override pixel safety checks.
        correction = review.correction.model_dump()
        for key, value in correction.items():
            limit = 0.12 if key == "exposure_ev" else (2.0 if key in {"temperature", "tint"} else 0.08)
            correction[key] = max(-limit, min(limit, value)) if review.rejects else 0.0
        if intent.natural_color:
            correction["temperature"] = correction["tint"] = 0.0
        if intent.preserve_atmosphere:
            correction["dehaze"] = min(0.0, correction["dehaze"])
        for key in intent.locked_controls:
            if key in correction:
                correction[key] = 0.0
        return review.model_copy(update={"correction": CorrectionHint.model_validate(correction),
                                         "transport_audit": transport_audit,
                                         "verification_transport_audit": verification_transport_audit})
    except Exception as exc:
        transport_audit["stage_error"] = {"type": type(exc).__name__}
        return VisualReview(notes=[f"视觉复核不可用：{safe_error(exc)}"],
                            transport_audit=transport_audit,
                            verification_transport_audit=verification_transport_audit)


def repair_visual_recipe(recipe: Recipe, review: VisualReview, attempt: int) -> Recipe:
    """Remove the implicated edits on a fresh source, retaining its own design.

    An additive zero cannot undo an existing warm WB or sharpening. The caller
    renders this complete repair recipe on an independent source copy.
    """
    from .preference import _scale_recipe

    strength = (.5, .2, 0.0)[min(max(attempt, 0), 2)]
    reduced = _scale_recipe(recipe, strength).model_dump()
    data = recipe.model_dump()
    kinds = {damage.kind for damage in review.damage if damage.severity >= .65}
    if "color_cast" in kinds:
        for key in ("hsl", "color_grading", "grade", "saturation", "vibrance"):
            data[key] = reduced[key]
        data["temperature"] = data["tint"] = 0.0
        for channel in ("r", "g", "b"):
            data["curve"][channel] = reduced["curve"][channel]
        for mask in data["masks"]:
            for key in ("temperature", "tint", "saturation"):
                mask[key] *= strength
    if kinds & {"detail_loss", "halos"}:
        for key in ("texture", "clarity", "dehaze"):
            data[key] = reduced[key]
        data["detail"] = {}
        for mask in data["masks"]:
            for key in ("texture", "clarity", "dehaze"):
                mask[key] *= strength
    if kinds & {"clipping", "shadow_crush"}:
        for key in ("exposure_ev", "highlights", "shadows", "whites", "blacks", "contrast", "curve", "parametric_tone"):
            data[key] = reduced[key]
        for mask in data["masks"]:
            for key in ("exposure_ev", "highlights", "shadows", "contrast", "whites", "blacks"):
                mask[key] *= strength
    return Recipe.model_validate(data)
