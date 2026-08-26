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


def test_recipe_merge_adds() -> None:
    a = Recipe(exposure_ev=0.3, contrast=0.1)
    b = Recipe(exposure_ev=-0.1, saturation=0.2)
    m = a.merged(b)
    assert abs(m.exposure_ev - 0.2) < 1e-6
    assert abs(m.saturation - 0.2) < 1e-6


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
