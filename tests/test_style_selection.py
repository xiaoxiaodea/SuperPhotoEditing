import json

import pytest

from superphoto.celebrity_styles import build_style_decision
from superphoto.config import Settings
from superphoto.db import connect, init_db, set_rating
from superphoto.directions import design_directions, preset_direction
from superphoto.lightroom import recipe_to_develop_settings
from superphoto.models import Critique, EditIntent, MaskAdjustment
from superphoto.preference import build_plans
from superphoto.recipes import LOOKS
from superphoto.scene_preferences import preset_feedback, scene_profile, variant_method
from superphoto.style_selection import rank_presets
from test_celebrity_styles import _tech
from test_scene_preferences import saved_job


def test_all_executable_presets_participate_with_explicit_exclusions():
    ranked, excluded = rank_presets(_tech(), Critique(genre="portrait", subject_note="日光人像"))
    eligible = {m.look_id for m in ranked}
    assert eligible | excluded.keys() == LOOKS.keys() - {"faithful", "celebrity_rescue"}
    assert eligible.isdisjoint(excluded)
    assert any(m.source == "xmp" for m in ranked)
    assert all(m.evidence and m.family for m in ranked)


def test_semantic_conflict_cannot_be_overridden_by_model_or_previous_likes():
    critique = Critique(genre="landscape", subject_note="湖边白色建筑及水面", source="ollama",
                        looks=["dune", "hanabi", "celebrity_portra_hamada"])
    # Warm stone can trigger the pixel-level skin detector; it is not a face.
    t = _tech(skin_fraction=.4, orange_fraction=.45, green_fraction=0, aqua_fraction=0)
    d = build_style_decision(t, critique, feedback={lid: {"like": 100} for lid in critique.looks})
    assert set(critique.looks).isdisjoint(d.look_ids)
    assert all(lid in d.excluded for lid in critique.looks)
    assert "downton" in d.look_ids


def test_different_scenes_choose_different_families_deterministically():
    cases = [
        (_tech(green_fraction=.38), Critique(genre="landscape", subject_note="森林光线")),
        (_tech(green_fraction=0, orange_fraction=.5, aqua_fraction=.01), Critique(genre="landscape", subject_note="沙丘")),
        (_tech(mean_luma=.05, green_fraction=0), Critique(genre="night", subject_note="夜空烟火")),
        (_tech(skin_fraction=.28, green_fraction=0), Critique(genre="portrait", subject_note="窗边人像")),
    ]
    selections = []
    for t, c in cases:
        decision = build_style_decision(t, c)
        assert len(set(decision.look_ids)) == 3
        assert decision.look_ids == build_style_decision(t, c).look_ids
        selections.append(tuple(decision.look_ids))
    assert len(set(selections)) == 4
    assert "celebrity_forest_michael_kagerer" in selections[0]
    assert "dune" in selections[1]
    assert "hanabi" in selections[2]
    assert "celebrity_portra_hamada" in selections[3]


def test_preserve_atmosphere_excludes_dehaze_route_even_for_flat_photo():
    d = build_style_decision(_tech(std_luma=.05), Critique(genre="landscape"))
    assert "celebrity_extreme_dehaze" in d.excluded


def test_selected_xmp_survives_sparse_design_and_reaches_native_lightroom(tmp_path, monkeypatch):
    preview = tmp_path / "input.jpg"
    preview.write_bytes(b"test-preview")
    critique = Critique(genre="architecture", subject_note="湖边白建筑",
        intent=EditIntent(masks=[MaskAdjustment(target="sky", exposure_ev=-.2)]))
    technical = _tech()
    ids = ["downton", "editorial_crisp", "skam"]
    defaults = [preset_direction(slot, lid, critique, technical) for slot, lid in zip("ABC", ids)]
    def vision(settings, model, prompt, image, *, response_schema):
        assert all(lid in prompt for lid in ids)
        assert "SPARSE overrides" in prompt
        assert response_schema["$defs"]["direction"]["properties"]["recipe"]["required"] == []
        return json.dumps({slot: {"title": "模型泛称", "goal": "保留建筑光线", "moves": ["轻整理材质"],
                                   "recipe": {"texture": .06}} for slot in "ABC"})
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.directions.chat_vision", vision)
    critique.directions = design_directions(Settings(), preview, critique, technical, look_ids=ids)
    plans = build_plans(ids, critique, technical)
    for plan, default in zip(plans, defaults):
        assert plan.look_id == default.look_id
        assert plan.method_id == "preset:" + plan.look_id
        assert plan.title == LOOKS[plan.look_id].title
        assert plan.recipe.curve == default.recipe.curve
        assert plan.recipe.color_grading == default.recipe.color_grading
        assert plan.recipe.masks == default.recipe.masks
        assert plan.lut_mix > 0
    native = recipe_to_develop_settings(plans[0].recipe)
    assert plans[0].recipe.curve.r or plans[0].recipe.curve.b
    assert native["ToneCurvePV2012"] != recipe_to_develop_settings(plans[1].recipe).get("ToneCurvePV2012")
    assert any(key.startswith(("SplitToning", "ColorGrade")) for key in native)
    with pytest.raises(ValueError, match="differs from selected"):
        build_plans(list(reversed(ids)), critique, technical)


def test_model_unavailable_uses_selected_preset_not_three_method_templates(tmp_path):
    ids = ["downton", "skam", "five_cm"]
    c, t = Critique(genre="architecture"), _tech()
    actual = design_directions(Settings(ollama_enabled=False), tmp_path / "unused", c, t, look_ids=ids)
    assert [d.look_id for d in actual] == ids
    assert [d.recipe for d in actual] == [preset_direction(slot, lid, c, t).recipe for slot, lid in zip("ABC", ids)]
    assert all(d.source == "scene" for d in actual)


def test_explicit_color_zero_overrides_selected_preset(tmp_path, monkeypatch):
    preview = tmp_path / "input.jpg"
    preview.write_bytes(b"test-preview")
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    response = {slot: {"title": slot, "goal": "保留现场蓝色", "moves": [],
                       "recipe": {"hsl": {"blue": {"sat": 0}}, "color_grading": {"shadow_sat": 0}}} for slot in "ABC"}
    monkeypatch.setattr("superphoto.directions.chat_vision", lambda *a, **k: json.dumps(response))
    ds = design_directions(Settings(), preview, Critique(), _tech(), look_ids=["downton", "skam", "five_cm"])
    assert all(d.recipe.hsl["blue"].sat == 0 and d.recipe.color_grading.shadow_sat == 0 for d in ds)


def test_dislike_tracks_executed_preset_only_in_similar_scenes(tmp_path):
    conn = connect(tmp_path / "prefs.db")
    init_db(conn)
    c, t = saved_job(conn)
    method = "preset:landscape_filmic"
    assert variant_method({"title": "风格名", "recipe_json": json.dumps({"style_workflow": {"method_id": method}})}) == method
    before = build_style_decision(t, c, feedback=preset_feedback(conn, c, t)["evidence"])
    set_rating(conn, "photo", "B", -1, method, scene_profile(c, t))
    votes = preset_feedback(conn, c, t)
    assert votes["matched_feedback"] == 1
    after = build_style_decision(t, c, feedback=votes["evidence"])
    score = lambda d: next(m.match_score for m in d.ranked_candidates if m.look_id == "landscape_filmic")
    assert score(after) < score(before) - .15
    assert preset_feedback(conn, Critique(genre="portrait", subject_note="室内人像"), t)["matched_feedback"] == 0
    set_rating(conn, "photo", "B", 0, method, scene_profile(c, t))
    assert preset_feedback(conn, c, t)["matched_feedback"] == 0


def test_preset_strength_repair_preserves_confirmed_raw_recovery():
    from superphoto.models import Recipe, ToneCurve
    from superphoto.preference import reduce_style_strength
    primary = Recipe(exposure_ev=-.37, highlights=-.72, masks=[
        MaskAdjustment(target="sky", exposure_ev=-1.58, contrast=.36, whites=.72)])
    full = primary.merged(Recipe(contrast=.2, curve=ToneCurve(rgb=[(0, 0), (.5, .56), (1, 1)])))
    softened = reduce_style_strength(full, primary, .4)
    assert softened.exposure_ev == primary.exposure_ev
    assert softened.highlights == primary.highlights
    assert softened.masks == primary.masks
    assert softened.contrast == pytest.approx(.08)
    assert softened.curve.rgb[1][1] == pytest.approx(.524)
