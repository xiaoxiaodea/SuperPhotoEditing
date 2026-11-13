"""One bounded, evidence-led finishing pass on an actual Lightroom export.

Technical safety and artistic completeness are separate questions. This module
can suggest a further edit, but only a fresh comparison can authorize retaining
it. It never uses a master's finished image or invents additional native masks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .config import Settings
from .critique import _extract_json
from .models import EditIntent, HueName, HueShift, MaskAdjustment, Recipe
from .ollama import chat_vision, probe
from .pair_review import PairJudgment, compare_finished_pair
from .review import _preview


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("notes", mode="before", check_fields=False)
    @classmethod
    def note_list(cls, value):
        # JSON-mode vision models sometimes return one explanatory sentence
        # here. Normalize only this optional prose, never decision/evidence.
        return [value] if isinstance(value, str) else value


class FinishingWeakness(_Response):
    region: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    goal: str = Field(min_length=1)

    @field_validator("region", "evidence", "goal")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A finishing weakness needs visible, regional evidence")
        return value.strip()


class _HSLDelta(_Response):
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0


class _MaskDelta(_Response):
    target: Literal["subject", "sky", "background"]
    exposure_ev: float = 0.0
    contrast: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0
    clarity: float = 0.0
    texture: float = 0.0
    dehaze: float = 0.0


class _FinishingDelta(_Response):
    exposure_ev: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    contrast: float = 0.0
    texture: float = 0.0
    hsl: dict[HueName, _HSLDelta] = Field(default_factory=dict)
    masks: list[_MaskDelta] = Field(default_factory=list, max_length=3)


class _ProposalResponse(_Response):
    confidence: float = Field(gt=0, le=1)
    observations: list[str] = Field(min_length=1, max_length=5)
    weaknesses: list[FinishingWeakness] = Field(max_length=3)
    delta: _FinishingDelta
    notes: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("observations")
    @classmethod
    def actual_observations(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("A completed diagnosis needs actual image observations")
        return [value.strip() for value in values]


class FinishingProposal(BaseModel):
    source: str = "unavailable"
    confidence: float = Field(default=0.0, ge=0, le=1)
    observations: list[str] = Field(default_factory=list)
    weaknesses: list[FinishingWeakness] = Field(default_factory=list, max_length=3)
    recipe: Recipe | None = None
    notes: list[str] = Field(default_factory=list)
    raw_response: str | None = None


class FinishingEvidence(_Response):
    region: str = Field(min_length=1)
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)
    benefit: str = Field(min_length=1)

    @field_validator("region", "before", "after", "benefit")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A comparison needs actual regional observations")
        return value.strip()




class FinishingDecision(BaseModel):
    decision: Literal["better", "same", "worse", "uncertain", "unavailable"] = "unavailable"
    confidence: float = Field(default=0.0, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)
    evidence: list[FinishingEvidence] = Field(default_factory=list)
    source: str = "unavailable"
    raw_response: str | None = None
    pair_comparison: dict = Field(default_factory=dict)

    @computed_field
    @property
    def accepts(self) -> bool:
        return self.decision == "better" and self.confidence >= .75 and bool(self.evidence)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _toward_neutral(value: float, delta: float) -> float:
    """Natural color can undo an existing hue cast, without adding a new hue."""
    return _clip(value + delta, min(0.0, value), max(0.0, value))


def _apply_delta(base: Recipe, delta: _FinishingDelta, intent: EditIntent,
                 source_kind: str) -> Recipe | None:
    data = base.model_dump()
    locked = set(intent.locked_controls)
    global_limits = {"exposure_ev": (.15, -2.5, 2.5), "contrast": (.20, -.8, .8),
                     "texture": (.12, -.8, .8)}
    for key in ("highlights", "shadows", "whites", "blacks"):
        global_limits[key] = (.20, -1.0, 1.0)
    for key, (step, low, high) in global_limits.items():
        change = getattr(delta, key)
        if key in locked or abs(change) < 1e-4:
            continue
        value = _clip(data[key] + _clip(change, -step, step), low, high)
        if intent.preserve_black_depth and key in {"blacks", "shadows"}:
            value = min(value, max(data[key], .08 if key == "blacks" else .12))
        data[key] = value

    for name, changes in delta.hsl.items():
        original = data["hsl"].get(name, HueShift().model_dump())
        shifted = dict(original)
        for key, change in changes.model_dump().items():
            if abs(change) < 1e-4 or (key == "hue" and "hue" in locked):
                continue
            step, absolute = (3.0, 40.0) if key == "hue" else (.20, 1.0)
            change = _clip(change, -step, step)
            value = _clip(original[key] + change, -absolute, absolute)
            if intent.natural_color:
                if key == "hue":
                    value = _toward_neutral(original[key], change)
                elif key == "sat":
                    value = _clip(value, min(original[key], -.15), max(original[key], .05))
            shifted[key] = value
        if shifted != original:
            data["hsl"][name] = shifted

    targets = [m.target for m in delta.masks]
    existing = {m["target"]: m for m in data["masks"]}
    # A duplicate mask could apply the step twice; an absent mask has no proven
    # selection. Neither can be inferred from the model's prose.
    if len(targets) != len(set(targets)) or any(target not in existing for target in targets):
        raise ValueError("Finishing may adjust each existing native mask only once")
    if len(existing) != len(data["masks"]):
        raise ValueError("Existing mask targets are ambiguous")
    for adjustment in delta.masks:
        original = existing[adjustment.target]
        for key, change in adjustment.model_dump(exclude={"target"}).items():
            if abs(change) < 1e-4 or key in locked:
                continue
            if intent.natural_color and key in {"temperature", "tint"}:
                continue
            if intent.preserve_atmosphere and key == "dehaze" and change > 0:
                continue
            step = {"exposure_ev": .65 if source_kind == "raw" else .20,
                    "contrast": .20, "whites": .20, "highlights": .20,
                    "shadows": .15, "blacks": .15, "texture": .12,
                    "temperature": 2.0, "tint": 2.0,
                    "saturation": .10, "clarity": .08, "dehaze": .06}[key]
            field = MaskAdjustment.model_fields[key]
            bounds = {type(m).__name__: m for m in field.metadata}
            value = _clip(original[key] + _clip(change, -step, step),
                          bounds["Ge"].ge, bounds["Le"].le)
            if intent.preserve_black_depth and key in {"blacks", "shadows"}:
                value = min(value, max(original[key], .08 if key == "blacks" else .12))
            original[key] = value
        # Preserve target, original reason, order and every unrequested control.
        existing[adjustment.target] = MaskAdjustment.model_validate(original).for_source(source_kind).model_dump()
    data["masks"] = [existing[m.target] for m in base.masks]
    candidate = Recipe.model_validate(data)
    return candidate if candidate != base else None


def _intent_context(intent: EditIntent) -> str:
    # Learned reference checks and prior masks are not observations of this
    # finished image and must not anchor a second visual judgment.
    return intent.model_dump_json(exclude={"masks", "reference_ids", "reference_checks", "subject_box",
                                           "subject_confidence"})


def propose_finishing(settings: Settings, original: Path, current: Path, recipe: Recipe,
                      intent: EditIntent, source_kind: str) -> FinishingProposal:
    """Inspect the actual export and propose at most one bounded full recipe."""
    if not settings.visual_review_enabled:
        return FinishingProposal(notes=["成片精修复核未启用"])
    raw_response = None
    try:
        status = probe(settings)
        if not status.get("ready") or not status.get("resolved_model"):
            return FinishingProposal(notes=["视觉模型不可用，保留当前成片"])
        prompt = f"""请直接检查这张当前成片，判断它作为一张独立照片还欠缺什么。
先观察主要区域的明暗、颜色、质感与主体关系，再决定是否值得小范围精修。
observations 必须写出至少一条具体区域的真实观察；不要只说干净、自然、专业。
若有明确不足，最多列三处，每处说明位置 region、可见依据 evidence、改善目的 goal。
没有明显不足可以 weaknesses=[]、delta={{}}，但仍要用观察解释为何无需调整。
柔光、淡色、薄雾或暗部可能合理；不是每张照片都应强烈，不要默认压暗天空或增加饱和度。
也不要把灰淡、缺少光层次误当成完成度高。无法判断请如实降低 confidence，不能编造不足。

只返回 JSON，必填 confidence（0到1）、observations（字符串数组）、weaknesses（上述对象数组）、
delta（只写需要改变的参数）；可选 notes（字符串数组）。delta 是在现有配方上增加的量。
支持全局 exposure_ev（±0.15 EV）、highlights/shadows/whites/blacks/contrast（±0.20）、texture（±0.12）；
hsl 按 red/orange/yellow/green/aqua/blue/purple/magenta 写 hue（±3度）、sat/lum（±0.20）。
masks 数组只能用现有 target，每个一次；局部 exposure_ev 单轮±{'.65' if source_kind == 'raw' else '.20'} EV，
contrast/whites/highlights ±0.20，其余只作小调整。只写变化字段，不输出完整零值模板。
不新增蒙版，不改归属、构图、内容、色轮、曲线和全局白平衡。
锁定参数不改；natural_color 保持自然色，preserve_atmosphere 保留氛围，preserve_black_depth 保留黑位。
每项变化须对应可见不足，随后会实际渲染比较，没改善就保留当前版本。
拍摄与后期意图：{_intent_context(intent)}
源文件类型：{source_kind}。现有非默认参数仅作诊断后的调整依据，不能替代看图：
{recipe.model_dump_json(exclude_defaults=True, exclude_none=True)}
/no_think"""
        raw_response = chat_vision(
            settings, status["resolved_model"], prompt, [_preview(current)])
        response = _ProposalResponse.model_validate(_extract_json(raw_response))
        proposal = FinishingProposal(source="ollama", confidence=response.confidence,
                                     observations=response.observations, weaknesses=response.weaknesses,
                                     notes=response.notes,
                                     raw_response=raw_response)
        if not response.weaknesses or response.confidence < .70:
            proposal.notes.append("没有足够明确的剩余问题，保留当前成片")
            return proposal
        proposal.recipe = _apply_delta(recipe, response.delta, intent, source_kind)
        if proposal.recipe is None:
            proposal.notes.append("约束后没有有效调整，保留当前成片")
        return proposal
    except Exception as exc:
        return FinishingProposal(notes=[f"成片精修建议不可用：{type(exc).__name__}；保留当前成片"],
                                 raw_response=raw_response)


def compare_finishing(settings: Settings, before: Path, after: Path,
                      intent: EditIntent) -> FinishingDecision:
    """Accept only a gain that survives exchanging the photographs' positions."""
    return decision_from_pair(compare_finished_pair(settings, before, after, intent))


def decision_from_pair(judgment: PairJudgment) -> FinishingDecision:
    """Map neutral physical-image choices to the pipeline's retained/candidate roles."""
    decision = {"tie": "same", "uncertain": "uncertain", "unavailable": "unavailable"}.get(judgment.outcome)
    if judgment.outcome == "winner":
        decision = "better" if judgment.winner_id == judgment.second_id else "worse"
    evidence = []
    if judgment.orders:
        # The first order is the caller's first/second order. The complete
        # reverse evidence remains in pair_comparison for inspection.
        evidence = [FinishingEvidence(region=item.region, before=item.image_1,
                                      after=item.image_2, benefit=item.reason)
                    for item in judgment.orders[0].evidence]
    return FinishingDecision(decision=decision or "unavailable", confidence=judgment.confidence,
                             notes=judgment.notes, evidence=evidence, source=judgment.source,
                             raw_response=json.dumps([order.raw_response for order in judgment.orders],
                                                     ensure_ascii=False) if judgment.orders else None,
                             pair_comparison=judgment.model_dump())
