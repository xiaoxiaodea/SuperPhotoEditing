import json

import pytest

from superphoto.config import Settings
from superphoto.directions import design_directions, scene_directions
from superphoto.models import Critique
from test_celebrity_styles import _tech


IDENTITY = [(0, 0), (.18, .18), (.5, .5), (.8, .8), (1, 1)]


def _disabled_recipe(**updates):
    return {
        "contrast": 0,
        "blacks": 0,
        "texture": 0,
        "saturation": 0,
        "hsl": {color: {"sat": 0} for color in ("blue", "aqua", "green")},
        "color_grading": {"shadow_sat": 0, "highlight_sat": 0},
        "curve": {"rgb": IDENTITY},
        **updates,
    }


def _model_response(monkeypatch, recipe):
    response = {
        slot: {"title": f"Model {slot}", "goal": "Keep the scene colors", "moves": [], "recipe": recipe}
        for slot in "ABC"
    }
    monkeypatch.setattr("superphoto.directions.probe", lambda _: {"ready": True, "resolved_model": "vision"})
    monkeypatch.setattr("superphoto.directions.chat_vision", lambda *args, **kwargs: json.dumps(response))


@pytest.mark.parametrize("fallback", ["near-identity", "duplicate"])
def test_fallback_retains_explicit_zeros_and_unspecified_defaults(tmp_path, monkeypatch, fallback):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    recipe = _disabled_recipe(contrast=.25 if fallback == "duplicate" else 0)
    _model_response(monkeypatch, recipe)
    critique, technical = Critique(), _tech(top_third_blue=0)
    methods = ("sculpted",) * 3
    defaults = scene_directions(critique, technical, methods=methods)

    directions = design_directions(Settings(), image, critique, technical, methods=methods)

    selected, default = directions[1], defaults[1]
    assert any("B: near identity or duplicate" in issue for issue in selected.transport_audit["issues"])
    assert selected.source == "scene"
    assert selected.recipe.blacks == selected.recipe.texture == selected.recipe.saturation == 0
    assert all(selected.recipe.hsl[color].sat == 0 for color in ("blue", "aqua", "green"))
    assert selected.recipe.color_grading.shadow_sat == selected.recipe.color_grading.highlight_sat == 0
    assert selected.recipe.color_grading.shadow_hue == default.recipe.color_grading.shadow_hue
    assert selected.recipe.color_grading.blending == default.recipe.color_grading.blending
    assert selected.recipe.curve == default.recipe.curve
    assert selected.recipe.contrast == (default.recipe.contrast if fallback == "duplicate" else 0)
    if fallback == "duplicate":
        assert directions[0].source == "ollama"
        assert directions[0].recipe.contrast == .25


@pytest.mark.parametrize("control,value", [("highlights", -.3), ("blacks", -.2)])
def test_global_recovery_with_disabled_palette_is_not_a_noop(tmp_path, monkeypatch, control, value):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    _model_response(monkeypatch, _disabled_recipe(**{control: value}))

    directions = design_directions(Settings(), image, Critique(), _tech(top_third_blue=0),
                                   methods=("sculpted",) * 3)

    assert directions[0].source == "ollama"
    assert directions[0].title == "Model A"
    assert getattr(directions[0].recipe, control) == value
    assert directions[0].recipe.color_grading.shadow_sat == 0
    assert directions[0].recipe.color_grading.highlight_sat == 0
    assert not any("A: near identity or duplicate" in issue for issue in directions[0].transport_audit["issues"])


def test_resource_night_explicitly_disabled_wheels_survive_selection(tmp_path, monkeypatch):
    image = tmp_path / "preview.jpg"
    image.write_bytes(b"preview")
    _model_response(monkeypatch, {
        "contrast": 0,
        "color_grading": {"shadow_sat": 0, "highlight_sat": 0},
        "curve": {"rgb": IDENTITY},
    })

    directions = design_directions(
        Settings(), image, Critique(genre="night", subject_note="夜街灯光"), _tech(mean_luma=.08),
        look_ids=["resource_night", "celebrity_low_sat_shinagawa", "wong_kar_wai"],
    )

    assert directions[0].recipe.color_grading.shadow_sat == 0
    assert directions[0].recipe.color_grading.highlight_sat == 0
    assert directions[0].recipe.highlights < 0
    assert directions[0].recipe.blacks < 0
