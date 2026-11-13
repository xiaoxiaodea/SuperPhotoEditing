from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from superphoto.lightroom import (
    LightroomError,
    apply_professional_recipe,
    native_color_recipe,
    professional_recipe_stages,
    recipe_to_develop_settings,
)
from superphoto.models import ColorGradingControls, Grade, Recipe
from superphoto.xmp import parse_xmp, preset_dir


_NATIVE_KEYS = {
    "SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum",
    "ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum",
    "SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum",
    "ColorGradeBlending", "SplitToningBalance",
}


def _preset(tmp_path: Path, **controls) -> Path:
    path = tmp_path / "grade.xmp"
    attributes = "\n".join(f'crs:{name}="{value}"' for name, value in controls.items())
    path.write_text(f'<rdf:Description {attributes}/>', encoding="utf-8")
    return path


def test_xmp_wheels_reach_lightroom_in_native_units_without_rgb_duplicate(tmp_path: Path) -> None:
    expected = {
        "SplitToningShadowHue": 225, "SplitToningShadowSaturation": 22,
        "ColorGradeShadowLum": -13, "ColorGradeMidtoneHue": 39,
        "ColorGradeMidtoneSat": 16, "ColorGradeMidtoneLum": 0,
        "SplitToningHighlightHue": 45, "SplitToningHighlightSaturation": 32,
        "ColorGradeHighlightLum": 29, "ColorGradeBlending": 68, "SplitToningBalance": -24,
    }
    recipe = parse_xmp(_preset(tmp_path, **expected))

    assert recipe.grade == Grade()
    assert recipe_to_develop_settings(recipe) == expected
    stages = professional_recipe_stages(recipe)
    assert [name for name, _ in stages] == ["color-correction"]
    assert recipe_to_develop_settings(stages[0][1]) == expected


def test_partial_xmp_preserves_unset_controls_and_explicit_zero(tmp_path: Path) -> None:
    recipe = parse_xmp(_preset(tmp_path, ColorGradeMidtoneSat=0, ColorGradeBlending=0))
    assert recipe.color_grading.shadow_hue is None
    assert recipe.color_grading.balance is None
    assert recipe_to_develop_settings(recipe, {"ColorGradeMidtoneSat": 40}) == {
        "ColorGradeMidtoneSat": 0, "ColorGradeBlending": 0,
    }


def test_xmp_aliases_describe_one_wheel_and_canonical_zero_wins(tmp_path: Path) -> None:
    recipe = parse_xmp(_preset(
        tmp_path,
        SplitToningShadowHue=0, ColorGradeShadowHue=220,
        SplitToningShadowSaturation=0, ColorGradeShadowSat=30,
        ColorGradeHighlightHue=45, ColorGradeHighlightSat=12,
        SplitToningBalance=0, ColorGradeBalance=35,
    ))
    assert recipe.grade == Grade()
    assert recipe_to_develop_settings(recipe) == {
        "SplitToningShadowHue": 0, "SplitToningShadowSaturation": 0,
        "SplitToningHighlightHue": 45, "SplitToningHighlightSaturation": 12,
        "SplitToningBalance": 0,
    }


def test_invalid_xmp_wheel_values_do_not_leak_nonfinite_settings(tmp_path: Path) -> None:
    recipe = parse_xmp(_preset(
        tmp_path, ColorGradeMidtoneHue="NaN", ColorGradeMidtoneSat="Inf",
        ColorGradeShadowLum=-200, ColorGradeHighlightSat=150, ColorGradeBlending=120,
    ))
    assert recipe_to_develop_settings(recipe) == {
        "ColorGradeShadowLum": -100, "SplitToningHighlightSaturation": 100,
        "ColorGradeBlending": 100,
    }


@pytest.mark.parametrize("path", sorted(preset_dir().glob("*.xmp")), ids=lambda p: p.stem)
def test_packaged_presets_preserve_all_supported_native_color_controls(path: Path) -> None:
    # Read expected native values independently of the production XMP parser.
    namespace = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
    attributes = {key.removeprefix(namespace): value for element in ET.parse(path).iter()
                  for key, value in element.attrib.items() if key.startswith(namespace)}
    expected = {key: float(attributes[key]) for key in _NATIVE_KEYS if key in attributes}
    recipe = parse_xmp(path)
    actual = recipe_to_develop_settings(recipe)
    assert recipe.grade == Grade()
    assert {key: actual[key] for key in expected} == expected


def test_legacy_rgb_grade_reaches_native_stage_with_direction_and_bounded_strength() -> None:
    recipe = Recipe(grade=Grade(
        shadows=(-.04, .01, .06), midtones=(.02, .02, .02), highlights=(.06, .02, -.02),
    ))
    stages = professional_recipe_stages(recipe)
    assert [name for name, _ in stages] == ["color-correction"]
    settings = recipe_to_develop_settings(stages[0][1])
    assert 210 <= settings["SplitToningShadowHue"] <= 250
    assert 10 <= settings["SplitToningHighlightHue"] <= 50
    assert 0 < settings["SplitToningShadowSaturation"] <= 25
    assert 0 < settings["SplitToningHighlightSaturation"] <= 25
    assert settings["ColorGradeMidtoneSat"] == 0
    assert settings["ColorGradeMidtoneLum"] == 2
    assert recipe_to_develop_settings(recipe) == settings
    # Wheel controls are absolute; an existing native grade is never added twice.
    assert recipe_to_develop_settings(recipe, settings) == settings


@pytest.mark.parametrize("native", [
    {"shadow_sat": 0}, {"shadow_hue": 0}, {"shadow_lum": 0},
    {"shadow_hue": 120, "shadow_sat": 8, "shadow_lum": 2},
])
def test_explicit_native_control_owns_its_whole_wheel(native: dict) -> None:
    recipe = Recipe(
        grade=Grade(shadows=(-.04, .01, .06), highlights=(.06, .02, -.02)),
        color_grading=ColorGradingControls(**native, blending=65),
    )
    audit: dict = {}
    normalized = native_color_recipe(recipe, audit=audit)
    shadow = {key: value for key, value in normalized.color_grading.model_dump(exclude_none=True).items()
              if key.startswith("shadow_")}
    assert shadow == native
    assert normalized.color_grading.highlight_sat > 0
    assert normalized.color_grading.blending == 65
    assert audit["color_grading_conversion"]["native_overrides"] == ["shadows"]
    assert set(audit["color_grading_conversion"]["converted"]) == {"highlights"}


def test_normalization_copies_input_and_is_idempotent_and_audited() -> None:
    recipe = Recipe(exposure_ev=.2, grade=Grade(shadows=(-.04, .01, .06)))
    saved = recipe.model_dump()
    audit: dict = {}
    normalized = native_color_recipe(recipe, audit=audit)
    assert recipe.model_dump() == saved
    assert normalized.grade == Grade()
    assert normalized.exposure_ev == recipe.exposure_ev
    assert native_color_recipe(normalized) == normalized
    assert audit["color_grading_conversion"]["approximate"] is True
    assert audit["color_grading_conversion"]["legacy_grade"] == saved["grade"]
    mapper_audit: dict = {}
    recipe_to_develop_settings(recipe, audit=mapper_audit)
    assert mapper_audit == audit


def test_empty_grade_does_not_invent_a_native_color_stage() -> None:
    assert recipe_to_develop_settings(Recipe()) == {}
    assert professional_recipe_stages(Recipe()) == []
    assert native_color_recipe(Recipe()) == Recipe()


def test_extreme_legacy_values_are_bounded_and_neutral_offsets_have_no_color_cast() -> None:
    normalized = native_color_recipe(Recipe(grade=Grade(
        shadows=(-20, 0, 20), midtones=(20, 20, 20), highlights=(-20, -20, -20),
    )))
    assert normalized.color_grading.shadow_sat == 25
    assert normalized.color_grading.midtone_sat == 0
    assert normalized.color_grading.highlight_sat == 0
    assert normalized.color_grading.midtone_lum == 15
    assert normalized.color_grading.highlight_lum == -15


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_legacy_color_is_rejected(invalid: float) -> None:
    recipe = Recipe(grade=Grade(shadows=(invalid, 0, .02)))
    with pytest.raises(ValueError, match="Non-finite legacy color grade"):
        recipe_to_develop_settings(recipe)


def test_hasselblad_legacy_recipe_has_executable_native_color(monkeypatch) -> None:
    from superphoto import masters

    catalog = json.loads((Path(__file__).parents[1] / "src/superphoto/hasselblad_index.json").read_text())
    monkeypatch.setattr(masters, "load_index", lambda: catalog)
    # Use a real stored master with nonzero inferred RGB grading.
    for look_id in catalog["looks"]:
        look = masters.look_for(look_id)
        if look is not None and look.recipe.grade != Grade():
            stages = dict(professional_recipe_stages(look.recipe))
            assert "color-correction" in stages
            assert any(value > 0 for key, value in recipe_to_develop_settings(stages["color-correction"]).items()
                       if key in {"SplitToningShadowSaturation", "SplitToningHighlightSaturation", "ColorGradeMidtoneSat"})
            break
    else:
        pytest.fail("No stored Hasselblad master with legacy color grading")


class _NativeClient:
    tools = {"get_photo_metadata", "set_develop_settings", "export_photos"}

    def __init__(self, *, drop_color: bool = False):
        self.settings = {key: 0.0 for key in _NATIVE_KEYS}
        self.settings["ColorGradeBlending"] = 50
        self.applied: list[dict] = []
        self.drop_color = drop_color

    def call_tool(self, name, arguments=None, *, timeout=None):
        args = arguments or {}
        if name == "get_photo_metadata":
            return {"developSettings": dict(self.settings)}
        if name == "set_develop_settings":
            self.applied.append(dict(args["settings"]))
            if not self.drop_color:
                self.settings.update(args["settings"])
            return {"ok": True}
        if name == "export_photos":
            destination = Path(args["destination"])
            destination.mkdir(parents=True, exist_ok=True)
            # The fake export captures exactly the developed state at export.
            (destination / "render.jpg").write_text(json.dumps(self.settings))
            return {"count": 1}
        raise AssertionError(name)


@pytest.mark.parametrize("reject", [False, True])
def test_native_execution_effective_recipe_export_and_rollback_match(tmp_path: Path, reject: bool) -> None:
    original = tmp_path / "original.jpg"
    original.write_bytes(b"source")
    client = _NativeClient()
    initial = dict(client.settings)
    recipe = Recipe(grade=Grade(shadows=(-.04, .01, .06), highlights=(.06, .02, -.02)))
    result = apply_professional_recipe(
        client, original=original, recipe=recipe, output_dir=tmp_path / "exports",
        stage_reviewer=lambda *_: {"rejects": reject},
    )

    assert len(result["stages"]) == 1
    stage = result["stages"][0]
    assert stage["stage"] == "color-correction"
    assert stage["rolled_back"] is reject
    effective = Recipe.model_validate(result["effective_recipe"])
    assert effective.grade == Grade()
    assert effective == (Recipe() if reject else native_color_recipe(recipe))
    exported = json.loads(Path(result["export"]).read_text())
    assert exported == client.settings == result["develop_after"]
    assert exported == (initial if reject else initial | recipe_to_develop_settings(effective))
    assert len(client.applied) == (2 if reject else 1)
    assert result["color_grading_conversion"]["approximate"] is True


def test_failed_native_readback_is_not_reported_as_applied(tmp_path: Path) -> None:
    original = tmp_path / "original.jpg"
    original.write_bytes(b"source")
    with pytest.raises(LightroomError, match="颜色分级未正确回读"):
        apply_professional_recipe(
            _NativeClient(drop_color=True), original=original,
            recipe=Recipe(grade=Grade(shadows=(-.04, .01, .06))), output_dir=tmp_path / "exports",
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, False])
def test_invalid_native_readback_cannot_confirm_explicit_zero(tmp_path: Path, invalid) -> None:
    original = tmp_path / "original.jpg"
    original.write_bytes(b"source")
    client = _NativeClient(drop_color=True)
    client.settings["ColorGradeMidtoneSat"] = invalid
    with pytest.raises(LightroomError, match="颜色分级未正确回读"):
        apply_professional_recipe(
            client, original=original,
            recipe=Recipe(color_grading=ColorGradingControls(midtone_sat=0)), output_dir=tmp_path / "exports",
        )
