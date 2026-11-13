import json
from pathlib import Path

import pytest

from superphoto import finishing_review as finishing
from superphoto.config import Settings
from superphoto.models import ColorGradingControls, CropControls, EditIntent, HueShift, MaskAdjustment, Recipe
from superphoto.pair_review import PairJudgment, PairOrder, PairEvidence


@pytest.fixture
def vision(monkeypatch):
    calls = []
    monkeypatch.setattr(finishing, "probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr(finishing, "_preview", lambda path: str(path))

    def install(response):
        def chat(settings, model, prompt, images, **kwargs):
            calls.append({"prompt": prompt, "images": images, **kwargs})
            return json.dumps(response)
        monkeypatch.setattr(finishing, "chat_vision", chat)
        return calls
    return install


def _proposal(delta, **updates):
    return {"confidence": .9,
            "observations": ["上方白云与淡蓝天明暗接近，地面受光条带仍可读"],
            "weaknesses": [{"region": "上方天空", "evidence": "云与天空明暗接近，原有蓝色变灰",
                            "goal": "让云层与原有天色的差异重新可读"}],
            "delta": delta, **updates}


def _propose(recipe=Recipe(), intent=EditIntent(), source="raw"):
    return finishing.propose_finishing(Settings(), Path("original"), Path("export"), recipe, intent, source)


def test_finishing_can_restore_muted_color_and_strengthen_actual_sky_without_redesign(vision):
    base = Recipe(hsl={"blue": HueShift(sat=-.45)},
                  crop=CropControls(left=.15, angle=2),
                  color_grading=ColorGradingControls(shadow_hue=205, shadow_sat=3),
                  masks=[MaskAdjustment(target="sky", exposure_ev=-.7, highlights=-.4,
                                        reason="由本图实际云层诊断得到的天空选择")])
    saved = base.model_dump()
    calls = vision(_proposal({"hsl": {"blue": {"sat": .20}},
                              "masks": [{"target": "sky", "exposure_ev": -.5}]}))
    result = _propose(base)
    assert result.recipe is not None
    assert result.recipe.hsl["blue"].sat == pytest.approx(-.25)
    assert result.recipe.masks[0].exposure_ev == pytest.approx(-1.2)
    assert result.recipe.masks[0].reason == base.masks[0].reason
    assert result.recipe.masks[0].highlights == -.4
    assert result.recipe.crop == base.crop and result.recipe.color_grading == base.color_grading
    assert base.model_dump() == saved
    assert calls[0]["images"] == ["export"]
    assert "response_schema" not in calls[0]
    result.recipe.masks[0].reason = "改动候选"
    assert base.model_dump() == saved


@pytest.mark.parametrize("source,exposure,expected", [("raw", -1.7, -2.0), ("jpeg", -.7, -.75),
                                                       ("raw", -.2, -.85), ("jpeg", -.2, -.4)])
def test_local_recovery_respects_both_single_pass_and_source_headroom(vision, source, exposure, expected):
    vision(_proposal({"exposure_ev": -100, "contrast": 100,
                      "masks": [{"target": "sky", "exposure_ev": -100, "contrast": 100}]}))
    result = _propose(Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=exposure)]), source=source)
    assert result.recipe.exposure_ev == -.15
    assert result.recipe.contrast == .20
    assert result.recipe.masks[0].exposure_ev == pytest.approx(expected)
    assert result.recipe.masks[0].contrast == .20


def test_protected_controls_remain_existing_values_even_with_adversarial_deltas(vision):
    base = Recipe(exposure_ev=.3, blacks=.04, shadows=.1, hsl={"blue": HueShift(hue=-2, sat=-.45)},
                  masks=[MaskAdjustment(target="sky", exposure_ev=-.9, temperature=4,
                                        dehaze=.03, clarity=.04, blacks=.05)])
    vision(_proposal({"exposure_ev": .15, "blacks": .2, "shadows": .2,
                      "hsl": {"blue": {"hue": 3, "sat": .2}},
                      "masks": [{"target": "sky", "exposure_ev": .65, "temperature": 2,
                                 "dehaze": .06, "clarity": .08, "blacks": .15}]}))
    result = _propose(base, EditIntent(locked_controls=["exposure_ev", "hue", "clarity"]))
    assert result.recipe.exposure_ev == .3
    assert result.recipe.blacks == .08 and result.recipe.shadows == .12
    assert result.recipe.hsl["blue"].hue == -2
    mask = result.recipe.masks[0]
    assert mask.exposure_ev == -.9 and mask.temperature == 4
    assert mask.dehaze == .03 and mask.clarity == .04 and mask.blacks == .08


def test_natural_color_can_undo_hue_cast_but_cannot_add_a_different_cast(vision):
    vision(_proposal({"hsl": {"blue": {"hue": 30}, "green": {"hue": -30, "sat": 10}}}))
    result = _propose(Recipe(hsl={"blue": HueShift(hue=-2)}))
    assert result.recipe.hsl["blue"].hue == 0
    assert result.recipe.hsl["green"].hue == 0
    assert result.recipe.hsl["green"].sat == .05


@pytest.mark.parametrize("delta", [
    {"masks": [{"target": "subject", "exposure_ev": .2}]},
    {"masks": [{"target": "sky", "exposure_ev": -.4}, {"target": "sky", "exposure_ev": -.4}]},
    {"masks": [{"target": "sky", "exposure_ev": -.4, "reason": "替换原始意图"}]},
    {"crop": {"left": .3}},
    {"color_grading": {"shadow_sat": 10}},
    {"hsl": {"sky": {"sat": .2}}},
    {"exposure_ev": float("nan")},
])
def test_unsupported_or_malformed_proposal_is_unavailable_not_a_partial_edit(vision, delta):
    vision(_proposal(delta))
    result = _propose(Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=-.7)]))
    assert result.recipe is None and result.source == "unavailable"


@pytest.mark.parametrize("updates", [{"weaknesses": []}, {"confidence": .6},
                                      {"weaknesses": [{"region": " ", "evidence": "变差", "goal": "改善"}]}])
def test_no_evidence_or_uncertain_criticism_cannot_trigger_adjustment(vision, updates):
    vision(_proposal({"contrast": .15}, **updates))
    assert _propose().recipe is None


def test_fully_locked_or_zero_delta_is_not_reported_as_effective(vision):
    vision(_proposal({"exposure_ev": .1, "hsl": {"blue": {"hue": 0, "sat": 0}}}))
    result = _propose(intent=EditIntent(locked_controls=["exposure_ev"]))
    assert result.recipe is None
    assert "没有有效调整" in result.notes[-1]










@pytest.mark.parametrize("failure", ["unavailable", "probe-error", "chat-error", "invalid-json"])
def test_model_failures_preserve_finished_version(vision, monkeypatch, failure):
    vision({})
    if failure == "unavailable":
        monkeypatch.setattr(finishing, "probe", lambda _: {"ready": False})
    elif failure == "probe-error":
        def probe_error(*args, **kwargs):
            raise OSError("test failure")
        monkeypatch.setattr(finishing, "probe", probe_error)
    elif failure == "chat-error":
        def chat_error(*args, **kwargs):
            raise RuntimeError("test failure")
        monkeypatch.setattr(finishing, "chat_vision", chat_error)
    else:
        monkeypatch.setattr(finishing, "chat_vision", lambda *args, **kwargs: "not JSON")
    proposal = _propose()
    assert proposal.source == "unavailable" and proposal.recipe is None


@pytest.mark.parametrize("missing", ["confidence", "observations", "weaknesses", "delta", "all"])
def test_missing_proposal_fields_are_unavailable_and_preserve_raw_output(vision, missing):
    response = _proposal({})
    response = {} if missing == "all" else {key: value for key, value in response.items() if key != missing}
    calls = vision(response)
    result = _propose()
    assert result.source == "unavailable" and result.recipe is None
    assert json.loads(result.raw_response) == response
    assert "response_schema" not in calls[0]
    assert set(finishing._ProposalResponse.model_json_schema()["required"]) == {
        "confidence", "observations", "weaknesses", "delta"}




def test_explicit_no_weakness_is_valid_but_not_an_effective_edit(vision):
    response = {"confidence": .9, "observations": ["远处山坡有薄雾，前景仍有明暗层次，柔光一致"],
                "weaknesses": [], "delta": {}, "notes": ["保留现有薄雾与柔光"]}
    vision(response)
    result = _propose()
    assert result.source == "ollama" and result.confidence == .9 and result.recipe is None
    assert result.observations == response["observations"]
    assert json.loads(result.raw_response) == response


@pytest.mark.parametrize("updates", [{"confidence": 0}, {"observations": []}, {"observations": [" "]}])
def test_no_actual_observation_is_incomplete_diagnosis_not_successful_no_edit(vision, updates):
    response = _proposal({}, weaknesses=[], **updates)
    vision(response)
    result = _propose()
    assert result.source == "unavailable" and result.recipe is None
    assert json.loads(result.raw_response) == response




def test_current_image_only_and_sparse_recipe_prevent_baseline_and_zero_template_anchoring(vision):
    calls = vision(_proposal({}))
    _propose(Recipe(hsl={"blue": HueShift(sat=-.45)}))
    assert calls[0]["images"] == ["export"]
    assert '"hsl":{"blue":{"sat":-0.45}}' in calls[0]["prompt"]
    assert '"exposure_ev":0.0' not in calls[0]["prompt"]


@pytest.mark.parametrize("outcome,winner,expected,accepted", [
    ("winner", "second-hash", "better", True),
    ("winner", "first-hash", "worse", False),
    ("tie", None, "same", False),
    ("uncertain", None, "uncertain", False),
    ("unavailable", None, "unavailable", False),
])
def test_finishing_maps_neutral_comparison_and_preserves_both_order_audits(
        monkeypatch, outcome, winner, expected, accepted):
    evidence = PairEvidence(region="云层", image_1="云天明暗接近",
                            image_2="云底与亮云可分辨", reason="云内层次的可见差异")
    order = PairOrder(image_ids=["first-hash", "second-hash"], preview_hashes=["p1", "p2"],
                      prompt="neutral", board_sha256="board", raw_response="raw-forward",
                      choice=2, confidence=.9, evidence=[evidence])
    reverse = order.model_copy(update={"image_ids": ["second-hash", "first-hash"],
                                       "raw_response": "raw-reverse", "choice": 1})
    judgment = PairJudgment(outcome=outcome, winner_id=winner, first_id="first-hash",
                            second_id="second-hash", confidence=.9, orders=[order, reverse],
                            source="ollama-pair", notes=["顺序核对的实际结果"])
    calls = []
    def compare(settings, first, second, intent):
        calls.append((first, second, intent))
        return judgment
    monkeypatch.setattr(finishing, "compare_finished_pair", compare)
    intent = EditIntent()
    result = finishing.compare_finishing(Settings(), Path("old"), Path("candidate"), intent)
    assert result.decision == expected and result.accepts is accepted
    assert calls == [(Path("old"), Path("candidate"), intent)]
    assert result.evidence[0].before == evidence.image_1
    assert result.evidence[0].after == evidence.image_2
    assert result.pair_comparison == judgment.model_dump()
    assert json.loads(result.raw_response) == ["raw-forward", "raw-reverse"]
