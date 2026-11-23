import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from superphoto.color import linear_to_srgb
from superphoto.config import Settings
from superphoto.critique import _critique_from_dict
from superphoto.ingest import downscale_linear, load_linear
from superphoto.lightroom import apply_professional_recipe, native_mask_settings, LightroomError
from superphoto.models import (ColorGradingControls, Critique, EditIntent, LocalEdit, MaskAdjustment, Recipe,
                               ParametricTone, DetailControls, ToneCurve)
from superphoto.preference import _scale_recipe
from superphoto.review import review_render


def test_embedded_icc_is_converted_before_linearization(tmp_path):
    # LAB has different channel semantics: dropping its ICC/mode cannot pass.
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))
    image = Image.new("LAB", (16, 16), (145, 145, 118))
    path = tmp_path / "tagged.tiff"
    image.save(path, icc_profile=profile.tobytes())
    expected = ImageCms.profileToProfile(image, profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
    linear, meta = load_linear(path)
    assert meta["icc_converted"]
    assert np.allclose(linear_to_srgb(linear), np.asarray(expected) / 255, atol=1e-5)


def test_float_resizing_preserves_deep_shadows_and_bright_scene_values():
    dark = np.full((48, 64, 3), 0.0012, dtype=np.float32)
    assert np.allclose(downscale_linear(dark, 16), 0.0012, atol=1e-7)
    bright = np.full((48, 64, 3), 1.8, dtype=np.float32)
    assert np.allclose(downscale_linear(bright, 16), 1.8, atol=1e-5)


def test_look_scaling_cannot_inherit_reference_capture_sharpening():
    recipe = Recipe(contrast=0.2, parametric_tone=ParametricTone(shadows=-0.2, lights=0.1),
                    detail=DetailControls(sharpness=120, luminance_nr=60))
    scaled = _scale_recipe(recipe, 0.2)
    assert scaled.parametric_tone.shadows == pytest.approx(-0.04)
    assert scaled.parametric_tone.lights == pytest.approx(0.02)
    assert scaled.detail.sharpness is None
    assert scaled.detail.luminance_nr is None
    assert _scale_recipe(recipe, 1.0).detail.sharpness is None


def test_visual_diagnosis_can_lift_a_person_in_a_landscape():
    critique = _critique_from_dict({"genre": "landscape", "subject_note": "登山者",
        "intent": {"goal": "保留远山层次，改善人物面部可读性", "masks": [
            {"target": "subject", "exposure_ev": 0.30, "shadows": 0.12},
            {"target": "sky", "exposure_ev": -0.10},
        ]}}, "ollama")
    assert critique.intent.masks[0].target == "subject"
    assert critique.intent.masks[0].exposure_ev == 0.3
    assert len(critique.intent.masks) == 2


def test_invalid_optional_mask_does_not_destroy_valid_diagnosis():
    critique = _critique_from_dict({"intent": {"masks": [
        {"target": "subject", "exposure_ev": 9}, {"target": "sky", "highlights": -0.2}
    ]}}, "ollama")
    assert [m.target for m in critique.intent.masks] == ["sky"]


def test_explicit_subject_intent_overrides_legacy_heuristic():
    recipe = Recipe(local=LocalEdit(subject_lift=0.02), masks=[MaskAdjustment(target="subject", exposure_ev=0.35)])
    specs = native_mask_settings(recipe)
    assert len(specs) == 1
    assert specs[0][1]["local_Exposure"] == 0.35
    assert specs[0][1]["local_Saturation"] == 0


@pytest.mark.parametrize(("field", "value", "native_key", "native_value"), [
    ("contrast", -0.32, "local_Contrast", -32),
    ("whites", 0.63, "local_Whites", 63),
    ("blacks", -0.28, "local_Blacks", -28),
])
def test_local_tonal_control_alone_keeps_mask_and_clears_previous_controls(
    field, value, native_key, native_value,
):
    # These edits must not disappear as no-ops when no exposure lift is needed.
    mask = MaskAdjustment.model_validate({"target": "background", field: value})
    specs = native_mask_settings(Recipe(masks=[mask]))
    assert len(specs) == 1 and specs[0][0] == "background"
    settings = specs[0][1]
    assert settings[native_key] == native_value
    for key in ("local_Exposure", "local_Contrast", "local_Whites", "local_Blacks"):
        if key != native_key:
            assert settings[key] == 0, "absolute updates must clear old local tonal settings"


def test_bad_optional_box_does_not_discard_photographic_goal():
    critique = _critique_from_dict({"intent": {"goal": "保留清晨雾气", "subject_box": "unknown", "masks": 2}}, "ollama")
    assert critique.intent.goal == "保留清晨雾气"
    assert critique.intent.subject_box is None
    assert critique.intent.masks == []


def test_neutral_visual_decision_is_not_replaced_with_global_brightening():
    from superphoto.critique import _merge_with_local
    from superphoto.models import CorrectionHint, GeometryHint
    remote = Critique(source="ollama", intent=EditIntent(goal="只提亮人物，保留远山"))
    local = Critique(correction=CorrectionHint(exposure_ev=0.2), geometry=GeometryHint(rotate_deg=2, apply=True))
    merged = _merge_with_local(remote, local)
    assert merged.correction.exposure_ev == 0
    assert merged.geometry.is_noop()


def test_loopback_ollama_bypasses_system_proxy(monkeypatch):
    from superphoto.ollama import probe
    def get(url, **kwargs):
        assert url == "http://127.0.0.1:11434/api/tags"
        assert kwargs["trust_env"] is False
        from types import SimpleNamespace
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"models": [{"name": "qwen3-vl:32b"}]})
    monkeypatch.setattr("superphoto.ollama.httpx.get", get)
    assert probe(Settings(ollama_enabled=True))["ready"]


class _Client:
    tools = {"get_photo_metadata", "set_develop_settings", "export_photos", "apply_ai_mask"}

    def __init__(self):
        self.settings = {"Contrast2012": 13, "ToneCurvePV2012": [0, 0, 128, 134, 255, 255],
                         "ToneCurveName2012": "Custom"}
        self.calls = []

    def call_tool(self, name, arguments=None, **kwargs):
        args = arguments or {}
        self.calls.append((name, args))
        if name == "get_photo_metadata":
            return {"developSettings": {"sdk": dict(self.settings)}}
        if name == "set_develop_settings":
            self.settings.update(args["settings"])
            return {"success": True}
        if name == "apply_ai_mask":
            return {"success": True, "mask_id": "native-123", "target": args["target"], "settings": args["settings"]}
        if name == "export_photos":
            (Path(args["destination"]) / "frame.jpg").write_bytes(b"fresh export")
            return {"success": True}
        raise AssertionError(name)


def test_native_local_light_reaches_lightroom_before_style(tmp_path):
    photo = tmp_path / "original.jpg"
    photo.write_bytes(b"source")
    client = _Client()
    result = apply_professional_recipe(client, original=photo, output_dir=tmp_path / "out",
        recipe=Recipe(contrast=0.1, masks=[MaskAdjustment(target="subject", exposure_ev=0.3)]))
    assert [s["stage"] for s in result["stages"]] == ["local-light", "tonal-shape"]
    mask_call = next(args for name, args in client.calls if name == "apply_ai_mask")
    assert mask_call["settings"]["local_Exposure"] == 0.3
    assert result["masks"][0]["mask_id"] == "native-123"
    assert Path(result["export"]).read_bytes() == b"fresh export"


def test_background_light_shaping_reaches_native_mask_without_global_changes(tmp_path):
    photo = tmp_path / "original.jpg"
    photo.write_bytes(b"source")
    client = _Client()
    global_before = dict(client.settings)
    mask = MaskAdjustment(target="background", exposure_ev=-1.58,
                          contrast=0.28, whites=-0.61, blacks=-0.23)
    result = apply_professional_recipe(client, original=photo, output_dir=tmp_path / "out",
                                      recipe=Recipe(masks=[mask]))
    assert [stage["stage"] for stage in result["stages"]] == ["local-light"]
    mask_call = next(args for name, args in client.calls if name == "apply_ai_mask")
    assert mask_call["target"] == "background"
    expected = {"local_Exposure": -1.58, "local_Contrast": 28,
                "local_Whites": -61, "local_Blacks": -23}
    for key, value in expected.items():
        assert mask_call["settings"][key] == value
        assert result["masks"][0]["settings"][key] == value
    assert client.settings == global_before
    assert not any(name == "set_develop_settings" for name, _ in client.calls)
    assert Recipe.model_validate(result["effective_recipe"]).masks == [mask]


def test_absent_sky_keeps_subject_mask_and_removes_unapplied_recipe(tmp_path):
    class NoSkyClient(_Client):
        def call_tool(self, name, arguments=None, **kwargs):
            if name == "apply_ai_mask" and arguments["target"] == "sky":
                return {"success": False, "skipped": True, "target": "sky", "reason": "no_mask_created"}
            return super().call_tool(name, arguments, **kwargs)
    photo = tmp_path / "boat.jpg"
    photo.write_bytes(b"source")
    result = apply_professional_recipe(NoSkyClient(), original=photo, output_dir=tmp_path / "out",
        recipe=Recipe(local=LocalEdit(sky_dehaze=.05), masks=[
            MaskAdjustment(target="subject", exposure_ev=.2), MaskAdjustment(target="sky", highlights=-.1)]))
    assert [mask["target"] for mask in result["masks"]] == ["subject"]
    assert result["skipped_masks"] == [{"success": False, "skipped": True, "target": "sky", "reason": "no_mask_created"}]
    effective = Recipe.model_validate(result["effective_recipe"])
    assert [mask.target for mask in effective.masks] == ["subject"]
    assert effective.local.sky_dehaze == 0
    assert result["stages"][0]["skipped_masks"]


def test_unrecognized_mask_failure_is_not_silently_skipped(tmp_path):
    class FailedClient(_Client):
        def call_tool(self, name, arguments=None, **kwargs):
            if name == "apply_ai_mask":
                return {"success": False, "skipped": True, "target": "sky", "reason": "write_failed"}
            return super().call_tool(name, arguments, **kwargs)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"source")
    with pytest.raises(LightroomError, match="未确认完成"):
        apply_professional_recipe(FailedClient(), original=photo, output_dir=tmp_path / "out",
                                  recipe=Recipe(masks=[MaskAdjustment(target="sky", highlights=-.1)]))


@pytest.mark.parametrize("always_reject", [False, True])
def test_curve_backoff_uses_prior_state_and_records_effective_recipe(tmp_path, always_reject):
    from superphoto.models import ToneCurve
    photo = tmp_path / "original.jpg"
    photo.write_bytes(b"source")
    client = _Client()
    original_curve = client.settings["ToneCurvePV2012"][:]
    def review(stage, before, after):
        return {"rejects": always_reject or client.settings["Contrast2012"] > 21}
    result = apply_professional_recipe(client, original=photo, output_dir=tmp_path / "out",
        recipe=Recipe(contrast=.20, curve=ToneCurve(rgb=[(0, 0), (.5, .4), (1, 1)])), stage_reviewer=review)
    stage = result["stages"][0]
    assert [a["strength"] for a in stage["attempts"]] == [1, .65, .35]
    effective = Recipe.model_validate(result["effective_recipe"])
    if always_reject:
        assert stage["rolled_back"] and stage["accepted_strength"] == 0
        assert client.settings["Contrast2012"] == 13 and client.settings["ToneCurvePV2012"] == original_curve
        assert effective.curve.rgb == [] and effective.contrast == 0
    else:
        assert not stage["rolled_back"] and stage["accepted_strength"] == .35
        assert client.settings["Contrast2012"] == 20
        assert effective.contrast == pytest.approx(.07)
        # The accepted curve approaches the prior non-linear curve, not identity;
        # previous failed attempts cannot compound the applied contrast.
        xs, ys = zip(*effective.curve.rgb)
        old_mid = .5 * 134 / 128
        assert np.interp(.5, xs, ys) == pytest.approx(old_mid + (.4 - old_mid) * .35)


def test_missing_native_mask_support_cannot_silently_deliver_global_grade(tmp_path):
    client = _Client()
    client.tools = client.tools - {"apply_ai_mask"}
    photo = tmp_path / "original.jpg"
    photo.write_bytes(b"source")
    with pytest.raises(LightroomError, match="局部蒙版"):
        apply_professional_recipe(client, original=photo, output_dir=tmp_path / "out",
            recipe=Recipe(masks=[MaskAdjustment(target="subject", exposure_ev=0.3)]))


def test_native_color_grading_must_be_confirmed_by_lightroom_readback(tmp_path):
    photo = tmp_path / "original.jpg"
    photo.write_bytes(b"source")
    class IgnoredColorClient(_Client):
        def call_tool(self, name, arguments=None, **kwargs):
            if name == "set_develop_settings":
                return {"success": True}
            return super().call_tool(name, arguments, **kwargs)
    with pytest.raises(LightroomError, match="颜色分级未正确回读"):
        apply_professional_recipe(IgnoredColorClient(), original=photo, output_dir=tmp_path / "out",
            recipe=Recipe(color_grading=ColorGradingControls(highlight_hue=42, highlight_sat=12)))


def test_visual_regression_restores_exact_prior_curve_and_contrast(tmp_path):
    photo = tmp_path / "original.jpg"
    photo.write_bytes(b"source")
    client = _Client()
    before = dict(client.settings)
    def reviewer(stage, previous, proposed):
        assert Path(proposed).is_file()
        return {"decision": "worse", "rejects": True, "confidence": 1, "notes": ["暗部裁切超出保护线"]}
    result = apply_professional_recipe(client, original=photo, output_dir=tmp_path / "out",
        recipe=Recipe(contrast=0.2, curve=ToneCurve(rgb=[(0, 0.08), (0.5, 0.4), (1, 1)])), stage_reviewer=reviewer)
    assert client.settings == before
    assert result["stages"][0]["rolled_back"]
    assert result["effective_recipe"]["contrast"] == 0
    assert result["effective_recipe"]["curve"]["rgb"] == []


def test_missing_visual_model_is_not_reported_as_aesthetic_success(tmp_path):
    review = review_render(Settings(ollama_enabled=False), tmp_path / "a", tmp_path / "b", EditIntent(), "final")
    assert review.decision == "unavailable"
    assert review.confidence == 0
    assert not review.rejects


def test_pairwise_review_uses_one_numbered_board_and_bounds_corrections(tmp_path, monkeypatch):
    before, after = tmp_path / "before.jpg", tmp_path / "after.jpg"
    Image.new("RGB", (24, 24), (80, 90, 100)).save(before)
    Image.new("RGB", (24, 24), (100, 90, 80)).save(after)
    monkeypatch.setattr("superphoto.review.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    calls = []
    def vision(settings, model, prompt, images, *, audit, selection=None):
        assert len(images) == 1
        with Image.open(BytesIO(base64.b64decode(images[0]))) as board:
            left = board.getpixel((board.width // 4, board.height // 2))
            right = board.getpixel((board.width * 3 // 4, board.height // 2))
        calls.append((left, right))
        assert "LEFT panel 1" in prompt and "RIGHT panel 2" in prompt
        audit.update(response={"call": len(calls)}, response_metadata={"done_reason": "stop"})
        if len(calls) == 2:
            assert np.allclose(calls[1], tuple(reversed(calls[0])), atol=2)
            assert '肤色出现黄斑' not in prompt
            return '{"confirmed":true,"confidence":0.9,"kind":"color_cast","reference_observation":"人脸中性","result_observation":"人脸出现黄斑","evidence":"人脸肤色有明显偏黄"}'
        assert left[2] > left[0] and right[0] > right[2]
        return '{"decision":"worse","confidence":0.9,"notes":["肤色偏黄"],"damage":[{"kind":"color_cast","region":"人脸","evidence":"肤色出现黄斑","severity":0.9}],"correction":{"temperature":-30,"contrast":-2}}'
    monkeypatch.setattr("superphoto.review.chat_vision", vision)
    review = review_render(Settings(), before, after, EditIntent(), "creative-look")
    assert review.rejects
    assert review.correction.contrast == -0.08
    assert review.correction.temperature == 0
    assert review.transport_audit["response"]["call"] == 1
    assert review.verification_transport_audit["response"]["call"] == 2


def test_negative_style_opinion_cannot_erase_finished_design(tmp_path, monkeypatch):
    path = tmp_path / "render.jpg"
    Image.new("RGB", (24, 24), (80, 90, 100)).save(path)
    after = tmp_path / "after.jpg"
    Image.new("RGB", (24, 24), (85, 92, 100)).save(after)
    monkeypatch.setattr("superphoto.review.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.review.chat_vision", lambda *args, **kwargs:
        '{"decision":"worse","confidence":0.99,"damage_confirmed":true,"notes":["反射更柔和"],"correction":{"contrast":-0.08}}')
    review = review_render(Settings(), path, after, EditIntent(goal="柔和光感"), "final")
    assert review.decision == "worse"
    assert not review.rejects
    assert review.correction.contrast == 0


@pytest.mark.parametrize("verification", ['{"confirmed":false,"confidence":0.99}', 'invalid JSON',
                                         '{"confirmed":true,"confidence":0.99}'])
def test_unconfirmed_damage_retains_opinion_without_rollback(tmp_path, monkeypatch, verification):
    path = tmp_path / "render.jpg"
    Image.new("RGB", (24, 24), (80, 90, 100)).save(path)
    after = tmp_path / "after.jpg"
    Image.new("RGB", (24, 24), (85, 92, 100)).save(after)
    monkeypatch.setattr("superphoto.review.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    responses = iter(['{"decision":"worse","confidence":0.9,"damage":[{"kind":"detail_loss","region":"水面","evidence":"反光变弱","severity":0.8}]}', verification])
    def vision(*args, audit, selection=None):
        raw = next(responses)
        audit["response"] = {"message": {"content": raw}}
        return raw
    monkeypatch.setattr("superphoto.review.chat_vision", vision)
    review = review_render(Settings(), path, after, EditIntent(), "final")
    assert review.decision == "worse"
    assert not review.rejects
    assert review.verification_transport_audit["response"]["message"]["content"] == verification


def test_identical_renders_cannot_have_new_visual_damage(tmp_path, monkeypatch):
    path = tmp_path / "render.jpg"
    Image.new("RGB", (24, 24), (120, 90, 60)).save(path)
    monkeypatch.setattr("superphoto.review.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.review.chat_vision", lambda *a, **kw: pytest.fail("identical render"))
    review = review_render(Settings(), path, path, EditIntent(), "final")
    assert review.decision == "same" and not review.rejects


def test_false_preserved_color_does_not_anchor_verification(tmp_path, monkeypatch):
    before, after = tmp_path / "before.jpg", tmp_path / "after.jpg"
    Image.new("RGB", (24, 24), (170, 120, 80)).save(before)
    Image.new("RGB", (24, 24), (175, 125, 85)).save(after)
    monkeypatch.setattr("superphoto.review.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    calls = []
    def board(first, second):
        calls.append((first, second))
        return f"mock-board-{len(calls)}"
    monkeypatch.setattr("superphoto.pair_review.build_pair_board", board)
    def vision(settings, model, prompt, images, *, audit, selection=None):
        assert images == [f"mock-board-{len(calls)}"]
        if len(calls) == 1:
            return '{"decision":"worse","confidence":0.9,"damage":[{"kind":"color_cast","region":"水面","evidence":"反光从冷变暖","severity":0.7}]}'
        assert calls[1] == tuple(reversed(calls[0]))
        assert "反光从冷变暖" not in prompt and "保留冷色反光" not in prompt
        return '{"confirmed":false,"confidence":0.95,"kind":"none","reference_observation":"原片反光是橙色","result_observation":"成片同样是橙色","evidence":"没有新增色偏"}'
    monkeypatch.setattr("superphoto.review.chat_vision", vision)
    review = review_render(Settings(), before, after, EditIntent(preserve=["保留冷色反光"]), "final")
    assert not review.rejects
    assert review.verification["reference_observation"] == "原片反光是橙色"


@pytest.mark.parametrize("failure_call", [1, 2])
def test_review_keeps_transport_failure_audit_without_authorizing_rollback(tmp_path, monkeypatch, failure_call):
    monkeypatch.setattr("superphoto.review.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.review._preview", lambda path: str(path))
    monkeypatch.setattr("superphoto.pair_review.build_pair_board", lambda first, second: "mock-board")
    calls = []
    failure = {"done": False, "message": {"content": '{"confirmed":true}'}}
    def vision(settings, model, prompt, images, *, audit, selection=None):
        calls.append(images)
        if len(calls) == failure_call:
            audit.update(response=failure, error={"type": "ValueError"})
            raise ValueError("Ollama response is not complete (done=false)")
        audit["response"] = {"done": True, "done_reason": "stop"}
        return '{"decision":"worse","confidence":0.9,"damage":[{"kind":"detail_loss","region":"水面","evidence":"纹理减少","severity":0.8}]}'
    monkeypatch.setattr("superphoto.review.chat_vision", vision)

    review = review_render(Settings(), tmp_path / "before", tmp_path / "after", EditIntent(), "final")

    assert not review.rejects
    assert review.decision == ("unavailable" if failure_call == 1 else "worse")
    failed = review.transport_audit if failure_call == 1 else review.verification_transport_audit
    assert failed["response"] == failure
    assert failed["error"] == {"type": "ValueError"}


def test_color_damage_repair_undoes_shared_wb_without_borrowing_another_curve():
    from superphoto.models import HueShift
    from superphoto.review import VisualDamage, VisualReview, repair_visual_recipe
    recipe = Recipe(temperature=2, contrast=.12, texture=.1,
                    hsl={"blue": HueShift(sat=-.3)},
                    curve=ToneCurve(rgb=[(0, 0), (.5, .45), (1, 1)]),
                    color_grading=ColorGradingControls(highlight_hue=40, highlight_sat=8))
    review = VisualReview(decision="worse", confidence=.9, damage_confirmed=True,
                          damage=[VisualDamage(kind="color_cast", region="水面", evidence="新增色偏", severity=.8)])
    repaired = repair_visual_recipe(recipe, review, 0)
    assert repaired.temperature == 0
    assert repaired.hsl["blue"].sat == -.15
    assert repaired.color_grading.highlight_sat == 4
    assert repaired.curve == recipe.curve and repaired.contrast == recipe.contrast
    assert repair_visual_recipe(recipe, review, 2).color_grading.highlight_sat == 0
