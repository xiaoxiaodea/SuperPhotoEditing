from __future__ import annotations

import json
from pathlib import Path

import pytest

from superphoto.config import Settings
from superphoto.critique import _extract_json, _format_critique_error, critique_photo, local_critique
from superphoto.models import TechnicalReport
from superphoto.ollama import _assistant_text


TRUNCATED = (
    '{"genre": "landscape", "summary": "画面主体黑天鹅曝光正常，但水面反光过亮导致高光细节丢失，'
    '岸边植被色彩偏灰。整体影调偏平，缺乏自然光影层次。", "issues": [{"dim": "highlights", '
    '"severity": 0.3, "note": "水面反光高光溢出，细节丢失"}, {"dim": "color", '
    '"severity": 0.2, "note": "植被绿色饱和度不足，缺乏自然生机"}, {"dim": "contrast", '
    '"severity": 0.2, "note": "整体影调偏平，明暗过渡不自然"}], "do_not": ["过度提亮暗部", '
    '"全局提高饱和度", "改变水体自然色调"], "correction": {"exposure_ev": 0.0, '
    '"temperature": 0.0, "tint": 0.0, "highlights": -0.2, "shadows": 0.0, "wh'
)


def _tech() -> TechnicalReport:
    return TechnicalReport(
        width=64,
        height=64,
        source_kind="jpeg",
        mean_luma=0.25,
        median_luma=0.22,
        std_luma=0.1,
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_spread=0.5,
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.3,
        temperature_bias=0.0,
        top_third_blue=0.0,
        skin_fraction=0.0,
        green_fraction=0.18,
        histogram=[0.0] * 32,
    )


@pytest.mark.parametrize("black_clip,expected_severity", [(0.0, None), (.20, .60)])
def test_local_critique_separates_dark_tone_area_from_confirmed_near_black(black_clip, expected_severity):
    technical = _tech().model_copy(update={"mean_luma": .08, "median_luma": .07,
                                           "shadow_clip": .75, "black_clip": black_clip})
    shadows = [issue for issue in local_critique(technical).issues if issue.dim == "shadows"]
    if expected_severity is None:
        assert shadows == []
    else:
        assert len(shadows) == 1
        assert shadows[0].severity == pytest.approx(expected_severity)


def test_extract_json_repairs_truncated_vl_output() -> None:
    data = _extract_json(TRUNCATED)
    assert data["genre"] == "landscape"
    assert "黑天鹅" in data["summary"]
    assert data["correction"]["highlights"] == -0.2
    assert data["correction"]["shadows"] == 0.0
    assert "whites" not in data["correction"]
    assert data["do_not"][0] == "过度提亮暗部"


def test_explicitly_absent_subject_cannot_create_subject_or_inverse_mask():
    from superphoto.critique import _intent_from_dict
    intent = _intent_from_dict({"subject_confidence": 0, "subject_box": None, "masks": [
        {"target": "subject", "exposure_ev": .2}, {"target": "background", "exposure_ev": -.1},
        {"target": "sky", "exposure_ev": -.5}]})
    assert [m.target for m in intent.masks] == ["sky"]


def test_old_greedy_brace_cut_is_invalid() -> None:
    match = __import__("re").search(r"\{.*\}", TRUNCATED, __import__("re").DOTALL)
    assert match is not None
    try:
        json.loads(match.group(0))
        raise AssertionError("greedy brace match should fail on this blob")
    except json.JSONDecodeError as exc:
        assert "delimiter" in str(exc) or "Expecting" in str(exc)


def test_extract_json_ignores_trailing_commentary() -> None:
    data = _extract_json('{"genre": "portrait", "summary": "人像"} 备注：已对照 {schema}')
    assert data["genre"] == "portrait"


def test_assistant_text_prefers_thinking_json() -> None:
    text = _assistant_text(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": '{"genre": "landscape", "summary": "ok"}',
            }
        }
    )
    assert json.loads(text)["genre"] == "landscape"


def test_format_hides_raw_json_decode_error() -> None:
    try:
        json.loads("{")
    except json.JSONDecodeError as exc:
        note = _format_critique_error(exc)
    assert "Expecting" not in note
    assert "截断" in note


def test_ready_but_unusable_output_does_not_say_model_missing(
    tmp_path: Path, monkeypatch
) -> None:
    preview = tmp_path / "preview.jpg"
    preview.write_bytes(b"jpeg")
    monkeypatch.setattr(
        "superphoto.critique.probe",
        lambda settings: {"ready": True, "resolved_model": "qwen3-vl:32b"},
    )

    def boom(*_args, **_kwargs):
        raise json.JSONDecodeError("Expecting ',' delimiter", "x", 281)

    monkeypatch.setattr("superphoto.critique.ollama_critique", boom)
    settings = Settings(ollama_enabled=True, xai_api_key="")
    critique = critique_photo(settings, preview, _tech())
    assert "视觉模型未就绪" not in critique.summary
    assert "本地技术分析" in critique.summary
    assert "完整 JSON" in critique.summary
