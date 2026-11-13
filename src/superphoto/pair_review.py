"""Order-checked comparison of two anonymous photographs on one numbered board.

Agreement checks presentation consistency, not aesthetic truth. Model evidence
is retained as a claim for inspection, never relabeled as a verified fact.
"""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import Settings
from .critique import _extract_json
from .ingest import image_to_srgb, srgb_profile_bytes
from .models import EditIntent
from .ollama import chat_vision, probe
from .review import _preview


class PairEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str = Field(min_length=1)
    image_1: str = Field(min_length=1)
    image_2: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    reference_observation: str | None = None

    @field_validator("region", "image_1", "image_2", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pair evidence requires actual regional observations")
        return value.strip()

    @field_validator("reference_observation")
    @classmethod
    def nonblank_reference(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Reference evidence cannot be blank")
        return value.strip() if value is not None else None


class _ReferenceEvidence(PairEvidence):
    reference_observation: str = Field(min_length=1)


class _OrderAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    choice: Literal[1, 2, "tie", "uncertain"]
    confidence: float = Field(ge=0, le=1, strict=True)
    evidence: list[PairEvidence] = Field(min_length=1, max_length=5)

    @field_validator("choice", mode="before")
    @classmethod
    def numbered_choice(cls, value):
        if isinstance(value, bool):
            raise ValueError("A boolean is not an image number")
        return value


class _ReferenceOrderAnswer(_OrderAnswer):
    evidence: list[_ReferenceEvidence] = Field(min_length=1, max_length=5)


class PairOrder(BaseModel):
    """image_ids[i] identifies the physical photograph labeled i+1."""
    image_ids: list[str]
    preview_hashes: list[str]
    prompt: str
    board_sha256: str = ""
    raw_response: str | None = None
    choice: Literal[1, 2, "tie", "uncertain"] | None = None
    confidence: float = 0.0
    evidence: list[PairEvidence] = Field(default_factory=list)
    transport_audit: dict = Field(default_factory=dict)
    error: str | None = None


class PairJudgment(BaseModel):
    outcome: Literal["winner", "tie", "uncertain", "unavailable"] = "unavailable"
    winner_id: str | None = None
    first_id: str = ""
    second_id: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)
    orders: list[PairOrder] = Field(default_factory=list)
    source: str = "unavailable"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preview_bytes(preview: str) -> bytes:
    return base64.b64decode(preview, validate=True)


def _decode(preview: str) -> Image.Image:
    with Image.open(BytesIO(_preview_bytes(preview))) as source:
        return image_to_srgb(ImageOps.exif_transpose(source)).copy()


def build_pair_board(first_preview: str, second_preview: str,
                     reference_preview: str | None = None) -> str:
    """Return base64 JPEG: equal 950px cells, numbered 1/2, original ratios.

    Width is 1920px: 5px margin + 950 + 10px gap + 950 + 5px margin.
    Each row has a 50px label strip and 5px top/bottom margins. The image
    area uses the larger fitted height, capped at 650px. Optional reference R
    is centered in a separate lower row and never changes position on reversal.
    No source image is modified or written to disk.
    """
    images = [ImageOps.contain(_decode(value), (950, 650), Image.Resampling.LANCZOS)
              for value in (first_preview, second_preview)]
    row_height = max(image.height for image in images) + 60
    reference = (ImageOps.contain(_decode(reference_preview), (1910, 650), Image.Resampling.LANCZOS)
                 if reference_preview is not None else None)
    height = row_height + (reference.height + 70 if reference is not None else 0)
    board = Image.new("RGB", (1920, height), (128, 128, 128))
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.load_default(size=32)
    except TypeError:  # Older Pillow versions still get unambiguous ASCII labels.
        font = ImageFont.load_default()
    for index, photo in enumerate(images):
        cell_left = 5 + index * 960
        draw.text((cell_left + 475, 30), str(index + 1), font=font, fill="white", anchor="mm")
        board.paste(photo, (cell_left + (950 - photo.width) // 2,
                            55 + (row_height - 60 - photo.height) // 2))
    if reference is not None:
        reference_top = row_height + 10
        draw.text((960, reference_top + 30), "R", font=font, fill="white", anchor="mm")
        board.paste(reference, ((1920 - reference.width) // 2, reference_top + 55))
    stream = BytesIO()
    board.save(stream, format="JPEG", quality=95, icc_profile=srgb_profile_bytes())
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _comparison_prompt(intent: EditIntent, *, has_reference: bool) -> str:
    context = intent.model_dump_json(exclude={"masks", "reference_ids", "reference_checks",
                                               "subject_box", "subject_confidence"})
    if has_reference:
        return f"""这是参考接近程度比较。上排标号1和2是同一照片的两个版本，下排标记R是参考目标。
任务是判断哪版更接近R可见的明暗关系、光带位置与强弱、颜色深浅及区域组织，而非抽象评价哪版更自然。
先看R，再逐区域比较1和2。每条evidence先写reference_observation，具体描述同一区域R实际的光影或颜色；
再写image_1和image_2的实际表现，reason必须解释两版与R的关系，不能只谈自然、保留或泛泛的完成度。
共同意图只作保护约束，不能替代R观察，也不能因为意图含natural_color或preserve就忽略参考的实际光色。
不要机械模仿参考的压缩、光晕、噪点或裁切；更暗、更浓、更锐不自动更接近R，也不预设任何编号获胜。
两版都与R有差距时仍比较相对接近程度；有明确整体优势且无明显抵消不足，choice填对应编号1或2。
明确同样接近填tie；看不清R、证据不足或无法权衡填uncertain。不能用不观察R来得出tie。
返回JSON：choice、confidence（0到1）、evidence数组。每条必填region、reference_observation、image_1、image_2、reason。
所有观察都是模型判断，不能编造细节。保护约束：{context}
/no_think"""
    reference = ""
    return f"""画布上方标号1和2是同一照片的两个版本。请只依据可见画面比较，不预设任何编号更好。
观察主体关系、明暗组织、色彩协调、材质与氛围。更暗、更亮、更浓、更锐或变化更大都不自动代表更好。
若一张有具体整体优势且没有明显抵消的不足，choice填对应编号1或2；两张明确相当填tie；看不清、证据不足或各有得失无法选出填uncertain。
返回JSON：choice、confidence（0到1）、evidence数组。每条包含region（区域）、image_1（图1实际表现）、image_2（图2实际表现）、reason（该差异为何构成优势、相当或无法判断）。所有判断都是待验证的观察，不能编造细节或用泛泛赞美代替证据。{reference}
共同意图：{context}
/no_think"""


def compare_finished_pair(settings: Settings, first: Path, second: Path, intent: EditIntent,
                          reference_preview: str | None = None) -> PairJudgment:
    """A winner needs two independent orders selecting the same physical file."""
    judgment = PairJudgment()
    try:
        judgment.first_id, judgment.second_id = _file_hash(first), _file_hash(second)
        if not settings.visual_review_enabled:
            judgment.notes = ["成片比较未启用"]
            return judgment
        previews = [_preview(first), _preview(second)]
        preview_hashes = [hashlib.sha256(_preview_bytes(value)).hexdigest() for value in previews]
        if preview_hashes[0] == preview_hashes[1]:
            return judgment.model_copy(update={"outcome": "tie", "confidence": 1.0,
                "source": "identical-render", "notes": ["评审预览相同，无法据此区分优劣"]})
        status = probe(settings)
        if not status.get("ready") or not status.get("resolved_model"):
            judgment.notes = ["视觉模型不可用，未确定成片优劣"]
            return judgment
        prompt = _comparison_prompt(intent, has_reference=reference_preview is not None)
        answer_type = _ReferenceOrderAnswer if reference_preview is not None else _OrderAnswer
        image_ids = [judgment.first_id, judgment.second_id]
        for indices in ((0, 1), (1, 0)):
            order = PairOrder(image_ids=[image_ids[index] for index in indices],
                              preview_hashes=[preview_hashes[index] for index in indices], prompt=prompt)
            judgment.orders.append(order)
            try:
                board = build_pair_board(previews[indices[0]], previews[indices[1]], reference_preview)
                order.board_sha256 = hashlib.sha256(_preview_bytes(board)).hexdigest()
                order.raw_response = chat_vision(settings, status["resolved_model"], prompt, [board],
                    response_schema=answer_type.model_json_schema(), audit=order.transport_audit)
                response = answer_type.model_validate(_extract_json(order.raw_response))
                order.choice, order.confidence, order.evidence = response.choice, response.confidence, response.evidence
            except Exception as exc:
                order.error = f"{type(exc).__name__}: {exc}"
                judgment.notes = ["至少一个呈现顺序的比较不可用，未确定成片优劣"]
                return judgment
        judgment.source = "ollama-pair"
        reliable = all(order.confidence >= .75 for order in judgment.orders)
        if reliable and all(order.choice == "tie" for order in judgment.orders):
            judgment.outcome = "tie"
            judgment.confidence = min(order.confidence for order in judgment.orders)
            judgment.notes = ["两种呈现顺序均判断相当"]
        elif reliable and all(order.choice in (1, 2) for order in judgment.orders):
            winners = [order.image_ids[order.choice - 1] for order in judgment.orders]
            if winners[0] == winners[1]:
                judgment.outcome, judgment.winner_id = "winner", winners[0]
                judgment.confidence = min(order.confidence for order in judgment.orders)
                judgment.notes = ["两种呈现顺序的模型判断指向同一张照片"]
            else:
                judgment.outcome = "uncertain"
                judgment.notes = ["交换呈现顺序后胜者改变，未确定成片优劣"]
        else:
            judgment.outcome = "uncertain"
            judgment.notes = ["比较证据或置信度不足，或两种顺序的判断不一致"]
        return judgment
    except Exception as exc:
        judgment.notes = [f"成片比较不可用：{type(exc).__name__}；未确定成片优劣"]
        return judgment
