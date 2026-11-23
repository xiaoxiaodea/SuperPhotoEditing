import json

import httpx
from PIL import Image
import pytest

from superphoto import critique, directions, finishing_review, pair_review, refinement, review, vision
from superphoto.models import Critique, EditIntent, Recipe
from test_celebrity_styles import _tech
from test_vision import cloud, completion, configured, no_real_http, response


STAGES = ("diagnosis", "directions", "review", "finishing", "pair", "refinement")


@pytest.fixture
def photos(tmp_path):
    paths = (tmp_path / "before.jpg", tmp_path / "after.jpg")
    Image.new("RGB", (80, 60), (180, 100, 70)).save(paths[0])
    Image.new("RGB", (80, 60), (70, 100, 180)).save(paths[1])
    return paths


def answers_for(stage):
    if stage == "diagnosis":
        return [{"genre": "landscape", "summary": "主体明确，天空细节需要保留", "looks": ["faithful"]}]
    if stage == "directions":
        return [{item.slot: item.model_dump(exclude={"slot", "source"})
                 for item in directions.scene_directions(Critique(), _tech())}]
    if stage == "review":
        return [{"decision": "better", "confidence": .9, "notes": ["主体层次更清晰"], "damage": []}]
    if stage == "finishing":
        return [{"confidence": .9, "observations": ["中间主体与暗部层次接近"],
                 "weaknesses": [{"region": "中央主体", "evidence": "主体层次较平", "goal": "明确受光面"}],
                 "delta": {"contrast": .1}}]
    if stage == "pair":
        return [{"choice": choice, "confidence": .9, "evidence": [{"region": "主体",
                 "image_1": "纹理与亮部可见", "image_2": "亮部过渡与阴影可见", "reason": "可见层次不同"}]}
                for choice in (2, 1)]
    return [{"masks": [{"target": "subject", "exposure_ev": .1}]}]


def run_stage(stage, settings, photos, tmp_path):
    before, after = photos
    if stage == "diagnosis":
        result = critique.critique_photo(settings, before, _tech())
        return result, result.transport_audit
    if stage == "directions":
        audit_path = tmp_path / "directions.json"
        result = directions.design_directions(settings, before, Critique(), _tech(), audit_path=audit_path)
        return result, json.loads(audit_path.read_text(encoding="utf-8"))["transport_audit"]
    if stage == "review":
        result = review.review_render(settings, before, after, EditIntent(), "test")
        return result, result.transport_audit
    if stage == "finishing":
        result = finishing_review.propose_finishing(settings, before, after, Recipe(), EditIntent(), "jpeg")
        return result, result.transport_audit
    if stage == "pair":
        result = pair_review.compare_finished_pair(settings, before, after, EditIntent())
        return result, result.orders[0].transport_audit if result.orders else result.vision_status
    audit = {}
    result = refinement.interpret_feedback(settings, before, Recipe(), "主体层次再明确一些，保留现在的颜色", audit=audit)
    return result, audit


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("protocol", ["chat/completions", "responses"])
@pytest.mark.parametrize("provider", ["openai", "xai"])
def test_every_visual_stage_uses_selected_cloud_protocol_and_audits_usage(
        cloud, monkeypatch, photos, tmp_path, stage, protocol, provider):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("explicit cloud probed Ollama"))
    answers = iter(answers_for(stage))
    def handler(_):
        text = json.dumps(next(answers), ensure_ascii=False)
        return httpx.Response(200, json=response(text) if protocol == "responses" else completion(text))
    calls, _ = cloud(handler)
    result, audit = run_stage(stage, configured(provider, protocol), photos, tmp_path)
    assert len(calls) == (2 if stage == "pair" else 1)
    assert {path for path, _ in calls} == {"/v1/" + protocol}
    assert audit["provider"] == provider and audit["model"] == "actual-model"
    assert audit["usage"]["total_tokens"] == 30
    assert "mock-api-secret" not in json.dumps(audit)
    if stage == "diagnosis":
        assert result.vision_provider == provider and result.source == provider
        restored = Critique.model_validate_json(result.model_dump_json())
        assert restored.source == provider and restored.vision_provider == provider
        assert restored.transport_audit["model"] == "actual-model"
        assert restored.transport_audit["usage"]["total_tokens"] == 30
    elif stage == "directions":
        assert all(item.vision_provider == provider and item.source == provider for item in result)
        assert all(item.transport_audit["transport"]["provider"] == provider for item in result)
        restored = Critique.model_validate_json(Critique(directions=result).model_dump_json())
        assert all(item.source == provider and item.vision_provider == provider for item in restored.directions)
        assert all(item.transport_audit["transport"]["usage"]["total_tokens"] == 30 for item in restored.directions)
    elif stage == "review":
        assert result.source == provider and result.decision == "better" and not result.rejects
    elif stage == "finishing":
        assert result.source == provider and result.recipe.contrast == .1
    elif stage == "pair":
        assert result.source == f"{provider}-pair" and result.outcome == "winner"
        assert result.winner_id == result.second_id
        assert result.orders[1].transport_audit["provider"] == provider
    else:
        edits, interpreter = result
        assert interpreter == provider and edits.masks[0].exposure_ev == .1


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("failure", ["http", "malformed"])
def test_cloud_failure_is_recorded_and_never_masquerades_as_a_visual_success(
        cloud, monkeypatch, photos, tmp_path, stage, failure, caplog):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("cloud failure fell to Ollama"))
    calls, _ = cloud(lambda _: httpx.Response(503, json={"error": {"message": "mock-api-secret"}})
                     if failure == "http" else httpx.Response(200, json=completion("not JSON mock-api-secret")))
    if stage == "refinement":
        audit = {}
        with pytest.raises(ValueError, match="未应用调整") as error:
            refinement.interpret_feedback(configured(), photos[0], Recipe(), "主体层次再明确一些", audit=audit)
        assert "mock-api-secret" not in str(error.value)
        assert audit["interpretation_error"]
    else:
        result, audit = run_stage(stage, configured(), photos, tmp_path)
        if stage == "diagnosis":
            assert result.vision_provider == "local" and result.source == "local"
            assert "本地技术分析" in result.summary and "诊断失败" in result.summary
        elif stage == "directions":
            assert all(item.vision_provider == "local" and item.source == "scene" for item in result)
            assert all(item.transport_audit["fallback"] for item in result)
        elif stage == "review":
            assert result.decision == "unavailable" and not result.rejects
        elif stage == "finishing":
            assert result.source == "unavailable" and result.recipe is None
        else:
            assert result.outcome == "unavailable" and result.winner_id is None
            assert result.orders[0].error
    assert len(calls) == 1
    assert "mock-api-secret" not in json.dumps(audit) + caplog.text


@pytest.mark.parametrize("protocol", ["chat/completions", "responses"])
def test_independent_damage_verification_also_uses_cloud(cloud, photos, protocol):
    replies = iter([
        {"decision": "worse", "confidence": .95, "notes": ["云层纹理丢失"],
         "damage": [{"kind": "clipping", "region": "上方天空", "evidence": "云层新增纯白块", "severity": .9}]},
        {"confirmed": True, "confidence": .95, "kind": "clipping",
         "reference_observation": "原片云层纹理可见", "result_observation": "成片云层出现纯白块", "evidence": "纹理消失"},
    ])
    def handler(_):
        text = json.dumps(next(replies))
        return httpx.Response(200, json=response(text) if protocol == "responses" else completion(text))
    calls, _ = cloud(handler)
    result = review.review_render(configured(protocol=protocol), *photos, EditIntent(), "test")
    assert len(calls) == 2 and result.rejects
    assert result.verification_transport_audit["provider"] == "openai"
    assert result.verification_transport_audit["usage"]["total_tokens"] == 30


def test_invalid_initial_ollama_json_can_fall_to_xai_and_is_labeled(cloud, monkeypatch, photos):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: {"ready": True, "resolved_model": "local-vl"})
    monkeypatch.setattr(vision.ollama, "chat_vision", lambda *args, **kwargs: "invalid JSON")
    cloud(lambda _: httpx.Response(200, json=response(json.dumps(answers_for("diagnosis")[0]))))
    settings = configured("xai", "responses").model_copy(update={"vision_provider": "auto"})
    result = critique.critique_photo(settings, photos[0], _tech())
    assert result.vision_provider == "xai" and result.transport_audit["fallback"]
    assert len(result.transport_audit["attempts"]) == 2
    assert "备用模型" in result.summary


def test_local_provider_keeps_all_visual_fallbacks_offline(monkeypatch, photos, tmp_path):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("local mode probed Ollama"))
    monkeypatch.setattr(vision, "OpenAI", lambda **kwargs: pytest.fail("local mode contacted cloud"))
    settings = configured().model_copy(update={"vision_provider": "local"})
    for stage in STAGES:
        if stage == "refinement":
            edits, provider = refinement.interpret_feedback(settings, photos[0], Recipe(), "整体亮一点")
            assert provider == "literal" and edits.exposure_ev == .25
        else:
            result, _ = run_stage(stage, settings, photos, tmp_path)
            if stage == "diagnosis":
                assert result.source == "local"
            elif stage == "directions":
                assert all(item.source == "scene" for item in result)
            elif stage == "review":
                assert result.decision == "unavailable"
            elif stage == "finishing":
                assert result.recipe is None
            else:
                assert result.outcome == "unavailable"
