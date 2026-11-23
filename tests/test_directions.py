import json
from itertools import combinations

import numpy as np
import pytest

from superphoto.config import Settings
from superphoto.directions import _signature, _signature_distance, bound_direction, design_directions, scene_directions
from superphoto.lightroom import recipe_to_develop_settings
from superphoto.models import ArtDirection, ColorGradingControls, Critique, EditIntent, Recipe, ToneCurve
from superphoto.preference import _scale_recipe, build_plans
from test_celebrity_styles import _tech


def test_three_directions_have_different_tone_and_light_even_without_model(tmp_path):
    critique = Critique(subject_note="湖边白色建筑")
    directions = design_directions(Settings(ollama_enabled=False), tmp_path / "unused", critique, _tech(mean_luma=.64))
    assert [d.slot for d in directions] == ["A", "B", "C"]
    for first, second in combinations(directions, 2):
        assert _signature_distance(_signature(first.recipe), _signature(second.recipe)) > .018
    assert directions[1].recipe.hsl != directions[2].recipe.hsl
    assert directions[1].recipe.curve.rgb[2][1] < directions[2].recipe.curve.rgb[2][1]
    assert all(any(m.target == "sky" for m in d.recipe.masks) for d in directions)


def test_existing_silhouette_and_environment_intent_are_not_replaced_by_generic_relighting():
    from superphoto.models import MaskAdjustment
    silhouette = Critique(genre="street", intent=EditIntent(masks=[
        MaskAdjustment(target="subject", exposure_ev=-.15, reason="保留人物剪影"),
        MaskAdjustment(target="background", exposure_ev=.10, reason="让背景建筑的受光面可读")]))
    for direction in scene_directions(silhouette, _tech(top_third_blue=0)):
        masks = {m.target: m for m in direction.recipe.masks}
        assert masks["subject"].exposure_ev == -.15
        assert masks["background"].exposure_ev == .10
    lifted = silhouette.model_copy(deep=True)
    lifted.intent.masks[0].exposure_ev = .2
    middle = scene_directions(lifted, _tech(top_third_blue=0))[1]
    assert next(m for m in middle.recipe.masks if m.target == "background").exposure_ev == .10


@pytest.mark.parametrize("controls", [
    {"exposure_ev": -1.58, "contrast": .36, "whites": .72, "blacks": -.40},
    {},
])
def test_fallback_preserves_explicit_sky_recovery_and_neutral_intent(tmp_path, controls):
    from superphoto.models import MaskAdjustment

    sky = MaskAdjustment(target="sky", reason="保留明确的天空调整", **controls)
    critique = Critique(genre="landscape", intent=EditIntent(masks=[sky]))
    directions = design_directions(Settings(ollama_enabled=False), tmp_path / "unused", critique,
                                   _tech(source_kind="raw", top_third_blue=.3))
    for direction in directions:
        assert next(m for m in direction.recipe.masks if m.target == "sky") == sky


@pytest.mark.parametrize("target", ["subject", "sky", "background"])
def test_planner_preserves_directions_distinguished_only_by_local_tonal_controls(tmp_path, monkeypatch, target):
    from superphoto.models import MaskAdjustment

    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    curve = ToneCurve(rgb=[(0, 0), (.2, .2), (.5, .5), (.8, .8), (1, 1)])
    response = {}
    for slot, (control, value) in zip("ABC", (("contrast", .30), ("whites", .30), ("blacks", -.25))):
        recipe = Recipe(curve=curve, masks=[MaskAdjustment(target=target, **{control: value})])
        response[slot] = {"title": control, "goal": "区域光影层次", "moves": ["按区域调整明暗"],
                          "recipe": recipe.model_dump()}
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.directions.chat_vision", lambda *args, **kwargs: json.dumps(response))

    critique = Critique(intent=EditIntent(masks=[MaskAdjustment(target="subject")]))
    directions = design_directions(Settings(), image, critique, _tech(top_third_blue=0),
                                   methods=("natural", "natural", "natural"))
    assert [d.title for d in directions] == ["contrast", "whites", "blacks"]
    for direction in directions:
        assert direction.source == "ollama"
        assert direction.recipe == Recipe.model_validate(response[direction.slot]["recipe"])


def test_model_cannot_return_three_differently_named_noops(tmp_path, monkeypatch):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.directions.chat_vision", lambda *args, **kwargs: json.dumps({"directions": [
        {"slot": slot, "title": "不同的名字" + slot, "goal": "好看", "recipe": {"contrast": .001}}
        for slot in "ABC"]}))
    directions = design_directions(Settings(), image, Critique(), _tech())
    assert all(not d.recipe.curve.is_identity() for d in directions)
    assert directions[1].recipe.hsl and directions[2].recipe.hsl
    assert len({tuple(d.recipe.curve.rgb) for d in directions}) == 3


def test_art_direction_is_applied_with_its_own_goal_and_native_color():
    critique = Critique(intent=EditIntent(goal="整体目标", locked_controls=["dehaze"]))
    critique.directions = scene_directions(critique, _tech())
    plans = build_plans(["faithful", "downton", "faithful"], critique, _tech())
    assert all(p.lut_mix == 0 for p in plans)
    assert plans[1].intent.goal == critique.directions[1].goal
    assert plans[1].recipe.color_grading == critique.directions[1].recipe.color_grading
    assert plans[1].recipe.curve.rgb == critique.directions[1].recipe.curve.rgb
    assert all(p.recipe.dehaze == 0 and p.recipe.detail.sharpness is None for p in plans)


def test_structured_planner_accepts_three_complete_scene_specific_recipes(tmp_path, monkeypatch):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    critique, technical = Critique(), _tech()
    expected = scene_directions(critique, technical)
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    def vision(*args, response_schema, **kwargs):
        assert response_schema["required"] == ["A", "B", "C"]
        assert response_schema["$defs"]["direction"]["required"] == ["title", "goal", "moves", "recipe"]
        return json.dumps({d.slot: d.model_dump(exclude={"slot", "source"}) for d in expected})
    monkeypatch.setattr("superphoto.directions.chat_vision", vision)
    audit = tmp_path / "audit.json"
    actual = design_directions(Settings(), image, critique, technical, audit_path=audit)
    assert all(d.source == "ollama" for d in actual)
    assert [d.recipe for d in actual] == [d.recipe for d in expected]
    assert json.loads(audit.read_text(encoding="utf-8"))["issues"] == []


def test_planner_prompt_grounds_color_and_local_moves_and_accepts_shared_natural_palette(tmp_path, monkeypatch):
    from superphoto.models import HueShift

    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    hsl = {name: HueShift() for name in ("blue", "aqua", "green")}
    curves = (
        [(0, 0), (.2, .15), (.5, .515), (.8, .815), (1, 1)],
        [(0, 0), (.2, .105), (.5, .43), (.8, .76), (1, 1)],
        [(0, 0), (.2, .24), (.5, .56), (.8, .85), (1, 1)],
    )
    response = {slot: {"title": "自然配色与光影" + slot, "goal": "保留现场颜色，整理明暗结构",
                       "moves": ["用全局曲线整理亮暗面"],
                       "recipe": Recipe(contrast=.08, hsl=hsl, curve=ToneCurve(rgb=curve)).model_dump()}
                for slot, curve in zip("ABC", curves)}
    prompts = []

    def vision(settings, model, prompt, image, *, response_schema, **kwargs):
        prompts.append(prompt)
        return json.dumps(response)

    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.directions.chat_vision", vision)
    actual = design_directions(Settings(), image, Critique(), _tech(top_third_blue=0),
                               methods=("natural", "sculpted", "soft_warm"))

    assert len(prompts) == 1
    for instruction in (
        "Global HSL affects every region sharing that hue",
        '\"background blue/green\" is not a mask',
        "For each substantial desaturation, name the visible competing region",
        "Preserve or recover a pale main color",
        "The three directions may share the same natural palette",
        "Every move must map to controls actually present in its recipe",
        "Do not promise local hillside, skin, eye or foreground edits",
    ):
        assert instruction in prompts[0]
    assert "A/B/C must differ in palette" not in prompts[0]
    assert "收敛背景的蓝绿色" not in prompts[0]
    assert "减弱背景蓝青" not in prompts[0]
    assert all(d.source == "ollama" and d.title == response[d.slot]["title"] for d in actual)
    assert all(d.recipe.hsl == hsl and d.recipe.color_grading == ColorGradingControls() for d in actual)
    assert len({tuple(d.recipe.curve.rgb) for d in actual}) == 3


def test_natural_color_protects_neutrals_without_removing_restrained_wheels():
    direction = ArtDirection(slot="B", title="光影", goal="暖光", recipe=Recipe(temperature=12,
        color_grading=ColorGradingControls(highlight_sat=60, highlight_hue=42),
        curve=ToneCurve(rgb=[(0, .08), (.5, .85), (1, 1)], r=[(0, .05), (1, 1)])))
    bounded = bound_direction(direction, Critique(), _tech()).recipe
    assert bounded.temperature == 0
    assert bounded.curve.r == []
    assert bounded.curve.rgb[0][1] <= .02
    assert bounded.curve.rgb[1][1] <= .58
    assert bounded.color_grading.highlight_sat == 24


def test_native_color_wheels_use_absolute_units_preserve_zero_and_scale_chroma():
    recipe = Recipe(color_grading=ColorGradingControls(shadow_hue=218, shadow_sat=9,
        highlight_hue=43, highlight_sat=7, midtone_lum=0, blending=55, balance=10))
    settings = recipe_to_develop_settings(recipe, {"SplitToningShadowSaturation": 80, "ColorGradeMidtoneLum": 9})
    assert settings["SplitToningShadowSaturation"] == 9
    assert settings["ColorGradeMidtoneLum"] == 0
    assert settings["ColorGradeBlending"] == 55
    assert settings["SplitToningBalance"] == 10
    scaled = _scale_recipe(recipe, .5).color_grading
    assert scaled.shadow_sat == 4.5 and scaled.shadow_hue == 218
    merged = recipe.merged(Recipe(color_grading=ColorGradingControls(shadow_sat=0)))
    assert merged.color_grading.shadow_hue == 218
    assert merged.color_grading.shadow_sat == 0


def test_planner_gets_study_sources_and_can_preserve_selective_color_changes(tmp_path, monkeypatch):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    from superphoto.models import MaskAdjustment
    critique = Critique(genre="landscape", subject_note="雾山中的人物",
                        intent=EditIntent(masks=[MaskAdjustment(target="subject", exposure_ev=.2)]))
    technical = _tech()
    expected = scene_directions(critique, technical)
    expected[2].recipe.curve = ToneCurve(rgb=[(0, 0), (.2, .17), (.5, .42), (.8, .78), (1, 1)])
    next(m for m in expected[1].recipe.masks if m.target == "subject").exposure_ev = .05
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    def vision(settings, model, prompt, image, *, response_schema, **kwargs):
        assert "名人仿色调色.md" in prompt and "Hasselblad-REPORT.md" in prompt and "调色素材" in prompt
        assert "hsl" in response_schema["$defs"]["direction"]["properties"]["recipe"]["properties"]
        response = {d.slot: d.model_dump(exclude={"slot", "source"}) for d in expected}
        response["C"]["recipe"]["hsl"]["green"] = {"sat": 0}
        return json.dumps(response)
    monkeypatch.setattr("superphoto.directions.chat_vision", vision)
    actual = design_directions(Settings(), image, critique, technical)
    assert actual[2].recipe.curve.rgb[2][1] > .5  # A luminous promise cannot deliver a darkening curve.
    assert actual[2].recipe.hsl["blue"].sat < actual[1].recipe.hsl["blue"].sat
    assert actual[2].recipe.color_grading.shadow_sat == 0
    assert actual[2].recipe.hsl["green"].sat == 0
    assert next(m for m in actual[1].recipe.masks if m.target == "subject").exposure_ev >= .2


def test_night_and_forest_use_relevant_study_methods_without_vision():
    from superphoto.finishing_knowledge import method_ids
    assert method_ids(Critique(genre="night"), _tech(mean_luma=.05))[2] == "documentary"
    assert method_ids(Critique(genre="landscape"), _tech(green_fraction=.4))[1] == "forest"
    assert method_ids(Critique(subject_note="白色建筑"), _tech())[1] == "graphic"
    assert method_ids(Critique(subject_note="江面游船及对岸建筑"), _tech())[1] == "sculpted"
    assert method_ids(Critique(genre="portrait", subject_note="人物与背景建筑"), _tech())[1] == "sculpted"


def test_planning_signature_recognizes_hue_and_local_light():
    first = Recipe(color_grading=ColorGradingControls(shadow_hue=210, shadow_sat=10))
    warm = Recipe(color_grading=ColorGradingControls(shadow_hue=40, shadow_sat=10))
    assert _signature_distance(_signature(first), _signature(warm)) > .018


def test_low_key_daylight_preserves_color_and_normal_shadow_contrast(tmp_path):
    from superphoto.finishing_knowledge import method_ids
    critique = Critique(genre="landscape", subject_note="城市建筑与金色反光")
    technical = _tech(mean_luma=.08, shadow_clip=.22, aqua_fraction=.27)
    assert method_ids(critique, technical) == ("natural", "graphic", "soft_warm")
    directions = design_directions(Settings(ollama_enabled=False), tmp_path / "unused", critique, technical)
    natural, neutral, warm = [d.recipe for d in directions]
    assert -.35 <= neutral.hsl["aqua"].sat < 0
    assert -.30 <= warm.hsl["aqua"].sat < 0
    assert 0 < warm.color_grading.midtone_sat <= 10
    assert warm.color_grading.shadow_sat == 0
    assert neutral.color_grading != warm.color_grading
    assert neutral.contrast > 0  # Dark tones are not proof of clipping.
    assert natural.saturation > neutral.saturation


def test_model_scene_palette_and_explicit_zeros_survive_defaults(tmp_path, monkeypatch):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    critique = Critique(genre="landscape", subject_note="城市建筑")
    technical = _tech(mean_luma=.08, shadow_clip=.22, aqua_fraction=.27)
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    response = {slot: {"title": slot, "goal": "保持原样", "moves": [], "recipe": {
        "hsl": {"blue": {"sat": -.1}, "aqua": {"sat": 0}},
        "color_grading": {"midtone_sat": 0, "highlight_sat": 0}}} for slot in "ABC"}
    monkeypatch.setattr("superphoto.directions.chat_vision", lambda *args, **kwargs: json.dumps(response))
    actual = design_directions(Settings(), image, critique, technical)
    assert actual[1].title == "B" and actual[2].title == "C"
    assert actual[1].recipe.hsl["aqua"].sat == 0
    assert actual[2].recipe.color_grading.midtone_sat == 0
    assert actual[2].recipe.color_grading.highlight_sat == 0


def test_raw_sky_has_recovery_headroom_but_jpeg_and_subject_do_not():
    from superphoto.models import MaskAdjustment
    recipe = Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=-1.58, contrast=.36, whites=.72),
                           MaskAdjustment(target="subject", exposure_ev=-1.58, whites=.72)])
    direction = ArtDirection(slot="B", title="云层", goal="天空有密度，云保持明亮", recipe=recipe)
    critique = Critique(intent=EditIntent(masks=[recipe.masks[1]]))
    raw = bound_direction(direction, critique, _tech(source_kind="raw")).recipe
    jpeg = bound_direction(direction, critique, _tech(source_kind="jpeg")).recipe
    assert raw.masks[0].exposure_ev == -1.58 and raw.masks[0].whites == .72
    assert raw.masks[1].exposure_ev == -.75 and raw.masks[1].whites == .30
    assert jpeg.masks[0].exposure_ev == -.75 and jpeg.masks[0].whites == .30


@pytest.mark.parametrize("subject_evidence", ["none", "explicit-mask", "confidence", "box"])
def test_direction_requires_subject_evidence_before_inventing_subject_or_inverse_mask(subject_evidence):
    from superphoto.models import MaskAdjustment, RegionBox

    sky = MaskAdjustment(target="sky", exposure_ev=-1.58, contrast=.36, whites=.72)
    subject = MaskAdjustment(target="subject", exposure_ev=-.15, reason="保留人物剪影")
    background = MaskAdjustment(target="background", exposure_ev=.10)
    intent = EditIntent(masks=[sky])
    if subject_evidence == "explicit-mask":
        intent.masks.append(subject)
    elif subject_evidence == "confidence":
        intent.subject_confidence = .8
    elif subject_evidence == "box":
        intent.subject_box = RegionBox(left=.3, top=.2, right=.7, bottom=.9)
    direction = ArtDirection(slot="B", title="区域光影", goal="保留现场光比",
                             recipe=Recipe(masks=[sky, subject, background]))

    result = bound_direction(direction, Critique(intent=intent), _tech(source_kind="raw")).recipe
    assert next(m for m in result.masks if m.target == "sky") == sky
    if subject_evidence == "none":
        assert [m.target for m in result.masks] == ["sky"]
    else:
        assert result.masks == [sky, subject, background]


def test_confirmed_black_clipping_still_limits_shadow_consumption():
    direction = scene_directions(Critique(), _tech())[1]
    result = bound_direction(direction, Critique(), _tech(black_clip=.22)).recipe
    assert result.exposure_ev >= 0 and result.blacks >= 0 and result.contrast <= 0
    assert all(y >= x for x, y in result.curve.rgb if x <= .5)


def test_art_direction_keeps_explicit_crop_without_unconfirmed_rotation():
    from superphoto.models import GeometryHint
    critique = Critique(geometry=GeometryHint(apply=True, crop_bottom=.90, rotate_deg=1.5))
    critique.directions = scene_directions(critique, _tech())
    plans = build_plans(["faithful"] * 3, critique, _tech(horizon_confidence=0))
    assert plans[0].geometry.is_noop()
    assert plans[1].geometry.crop_bottom == .90 and plans[1].geometry.rotate_deg == 0
