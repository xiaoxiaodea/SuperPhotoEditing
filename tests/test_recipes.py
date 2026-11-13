from __future__ import annotations

from superphoto.analysis import analyze, correction_from_technical
from superphoto.critique import compose_look_ids, local_critique
from superphoto.ingest import load_linear
from superphoto.models import Recipe
from superphoto.preference import build_plans
from superphoto.recipes import LOOKS, get_look


def test_look_library_complete() -> None:
    assert "faithful" in LOOKS
    assert len(LOOKS) >= 16
    assert get_look("filmic_portrait_warm").recipe.temperature != 0
    assert get_look("dune").source == "xmp"
    assert not get_look("dune").recipe.curve.is_identity()
    assert get_look("faithful").recipe.contrast == 0
    assert get_look("faithful").recipe.local.is_noop()


def test_recipe_merge_adds() -> None:
    a = Recipe(exposure_ev=0.3, contrast=0.1)
    b = Recipe(exposure_ev=-0.1, saturation=0.2)
    m = a.merged(b)
    assert abs(m.exposure_ev - 0.2) < 1e-6
    assert abs(m.saturation - 0.2) < 1e-6


def test_adapt_blocks_sky_lift_on_landscape() -> None:
    from superphoto.models import HueShift, Recipe, TechnicalReport
    from superphoto.preference import _adapt_to_scene

    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.28,
        median_luma=0.10,
        std_luma=0.29,
        highlight_clip=0.0,
        shadow_clip=0.06,
        dynamic_spread=0.76,
        sharpness=0.08,
        noise=0.04,
        saturation_mean=0.33,
        temperature_bias=0.04,
        top_third_blue=0.24,
        skin_fraction=0.04,
        green_fraction=0.01,
        histogram=[0.0] * 32,
        highlight_p95=0.78,
    )
    raw = Recipe(
        highlights=0.05,
        whites=0.08,
        hsl={"aqua": HueShift(lum=0.12), "blue": HueShift(lum=0.08)},
    )
    out = _adapt_to_scene(raw, tech, "hb:benjamin-everett")
    assert out.highlights <= -0.08
    assert out.whites <= 0.0
    assert out.hsl["aqua"].lum <= 0.0
    assert out.hsl["blue"].lum <= 0.0


def test_sanitize_caps_low_key_jpeg_exposure() -> None:
    from superphoto.models import CorrectionHint, TechnicalReport
    from superphoto.preference import sanitize_correction

    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.06,
        median_luma=0.04,
        std_luma=0.05,
        highlight_clip=0.0,
        shadow_clip=0.4,
        dynamic_spread=0.5,
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.2,
        temperature_bias=0.0,
        top_third_blue=0.0,
        skin_fraction=0.0,
        green_fraction=0.3,
        histogram=[0.0] * 32,
        shadow_p05=0.01,
        highlight_p95=0.7,
    )
    out = sanitize_correction(CorrectionHint(exposure_ev=0.8, shadows=0.3, saturation=0.1), tech)
    assert out.exposure_ev <= 0.06
    assert out.shadows <= 0.08


def test_plans_are_three_and_include_faithful(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    critique = local_critique(tech)
    critique.correction = correction_from_technical(tech)
    ids = compose_look_ids(critique, [])
    plans = build_plans(ids, critique, tech)
    assert len(plans) == 3
    assert plans[0].look_id == "faithful"
    assert {p.slot for p in plans} == {"A", "B", "C"}
    assert plans[1].recipe.exposure_ev != 0 or plans[0].recipe.exposure_ev != 0
    # B/C looks must carry a finished grade, not a 2% nudge.
    def punch(r: Recipe) -> float:
        hsl = sum(abs(s.sat) + abs(s.lum) + abs(s.hue) / 40.0 for s in r.hsl.values())
        g = sum(abs(x) for x in (*r.grade.shadows, *r.grade.midtones, *r.grade.highlights))
        curve = 0.4 if not r.curve.is_identity() else 0.0
        return (
            abs(r.contrast)
            + abs(r.vignette)
            + abs(r.clarity)
            + abs(r.dehaze)
            + abs(r.blacks)
            + abs(r.whites)
            + hsl
            + g
            + curve
        )

    assert punch(plans[1].recipe) > 0.15
    assert punch(plans[2].recipe) > 0.15


def _tech(**kwargs):
    from superphoto.models import TechnicalReport

    base = dict(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.5,
        median_luma=0.48,
        std_luma=0.18,
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_spread=0.6,
        sharpness=0.08,
        noise=0.04,
        saturation_mean=0.4,
        temperature_bias=0.0,
        top_third_blue=0.2,
        skin_fraction=0.04,
        green_fraction=0.1,
        histogram=[0.0] * 32,
        aqua_fraction=0.04,
        orange_fraction=0.08,
        magenta_fraction=0.02,
    )
    base.update(kwargs)
    return TechnicalReport(**base)


def test_desert_photo_prefers_dune_over_filmic_green() -> None:
    from superphoto.preference import look_match

    desert = _tech(
        mean_luma=0.36,
        saturation_mean=0.34,
        temperature_bias=0.29,
        top_third_blue=0.20,
        skin_fraction=0.30,
        green_fraction=0.0,
        aqua_fraction=0.05,
        orange_fraction=0.55,
        shadow_clip=0.01,
    )
    assert look_match("dune", desert, "landscape") > look_match("landscape_filmic", desert, "landscape")
    lush = _tech(
        mean_luma=0.53,
        saturation_mean=0.52,
        temperature_bias=-0.31,
        top_third_blue=0.65,
        skin_fraction=0.06,
        green_fraction=0.29,
        aqua_fraction=0.30,
        orange_fraction=0.10,
    )
    assert look_match("dune", lush, "landscape") < 0.18
    assert look_match("landscape_filmic", lush, "landscape") > look_match("jp_drama", lush, "landscape")


def test_fireworks_photo_routes_to_hanabi() -> None:
    from superphoto.preference import look_match

    night = _tech(
        mean_luma=0.05,
        std_luma=0.12,
        saturation_mean=0.46,
        temperature_bias=0.55,
        top_third_blue=0.11,
        skin_fraction=0.02,
        green_fraction=0.0,
        orange_fraction=0.05,
        shadow_clip=0.74,
        dynamic_spread=0.28,
    )
    assert look_match("hanabi", night, "night") > 0.35
    assert look_match("hanabi", night, "night") > look_match("dune", night, "night")


def test_jp_drama_is_rejected_for_landscape() -> None:
    from superphoto.models import Critique, TechnicalReport
    from superphoto.critique import compose_look_ids
    from superphoto.preference import look_match

    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.5,
        median_luma=0.48,
        std_luma=0.18,
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_spread=0.6,
        sharpness=0.08,
        noise=0.04,
        saturation_mean=0.4,
        temperature_bias=0.0,
        top_third_blue=0.4,
        skin_fraction=0.02,
        green_fraction=0.22,
        histogram=[0.0] * 32,
    )
    assert look_match("jp_drama", tech, "landscape") < 0.18
    critique = Critique(genre="landscape", looks=["faithful", "landscape_filmic", "jp_drama"])
    ids = compose_look_ids(critique, [], tech)
    assert ids[0] == "faithful"
    assert "jp_drama" not in ids
    assert "landscape_filmic" in ids


def test_styled_slots_keep_a_visible_look_on_jpeg() -> None:
    from superphoto.models import CorrectionHint, Critique, TechnicalReport
    from superphoto.preference import build_plans

    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.53,
        median_luma=0.50,
        std_luma=0.18,
        highlight_clip=0.002,
        shadow_clip=0.0,
        dynamic_spread=0.64,
        sharpness=0.09,
        noise=0.05,
        saturation_mean=0.40,
        temperature_bias=-0.1,
        top_third_blue=0.40,
        skin_fraction=0.02,
        green_fraction=0.22,
        histogram=[0.0] * 32,
    )
    critique = Critique(
        genre="landscape",
        summary="ok",
        correction=CorrectionHint(),
        looks=["faithful", "landscape_filmic", "moody_lowkey"],
    )
    plans = build_plans(["faithful", "landscape_filmic", "moody_lowkey"], critique, tech)
    assert plans[0].recipe.contrast == 0
    punch = abs(plans[1].recipe.contrast) + sum(
        abs(x) for x in (*plans[1].recipe.grade.shadows, *plans[1].recipe.grade.highlights)
    )
    punch += sum(abs(s.sat) + abs(s.lum) for s in plans[1].recipe.hsl.values())
    assert punch > 0.12
    assert 0 < plans[1].lut_mix <= 0.40


def test_landscape_jpeg_does_not_get_portrait_cleanup() -> None:
    from superphoto.models import TechnicalReport
    from superphoto.preference import local_edit_for, look_match

    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.53,
        median_luma=0.50,
        std_luma=0.18,
        highlight_clip=0.002,
        shadow_clip=0.0,
        dynamic_spread=0.64,
        sharpness=0.09,
        noise=0.05,
        saturation_mean=0.52,
        temperature_bias=-0.3,
        top_third_blue=0.65,
        skin_fraction=0.06,
        green_fraction=0.29,
        histogram=[0.0] * 32,
    )
    a = local_edit_for("A", tech, genre="landscape")
    b = local_edit_for("B", tech, genre="landscape")
    assert a.is_noop()
    assert b.color_clean == 0
    assert b.sky_dehaze == 0
    assert look_match("natsume", tech, "landscape") < 0.20


def test_low_key_look_does_not_crush_blacks_away() -> None:
    from superphoto.models import CorrectionHint, Critique, TechnicalReport
    from superphoto.preference import build_plans

    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.05,
        median_luma=0.03,
        std_luma=0.04,
        highlight_clip=0.0,
        shadow_clip=0.4,
        dynamic_spread=0.4,
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.3,
        temperature_bias=0.0,
        top_third_blue=0.0,
        skin_fraction=0.0,
        green_fraction=0.3,
        histogram=[0.0] * 32,
        shadow_p05=0.01,
        highlight_p95=0.5,
    )
    critique = Critique(
        genre="landscape",
        summary="low key",
        correction=CorrectionHint(),
        looks=["faithful", "moody_lowkey", "landscape_filmic"],
    )
    plans = build_plans(["faithful", "moody_lowkey", "landscape_filmic"], critique, tech)
    assert plans[1].recipe.blacks >= -0.16
    assert plans[1].recipe.vignette <= 0.20
    assert plans[1].recipe.contrast <= 0.16
