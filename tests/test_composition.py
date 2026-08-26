from __future__ import annotations

import numpy as np

from superphoto.analysis import analyze, correction_from_technical
from superphoto.composition import apply_geometry, measure_composition, suggest_geometry
from superphoto.critique import compose_look_ids, local_critique
from superphoto.ingest import load_linear
from superphoto.models import GeometryHint, Recipe
from superphoto.preference import build_plans, look_match, lut_amount
from superphoto.recipes import get_look
from superphoto.xmp import parse_xmp, preset_dir


def test_horizon_tilt_detects_slope() -> None:
    h, w = 180, 280
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = 70.0 + 0.07 * xx
    sky = np.where(yy < horizon, 0.82, 0.16)
    rgb = np.stack([sky * 0.7, sky * 0.8, sky], axis=-1).astype(np.float32)
    comp = measure_composition(rgb)
    assert abs(comp["horizon_tilt_deg"]) > 0.3
    assert comp["horizon_confidence"] > 0.15


def test_centered_flat_scene_does_not_demand_crop() -> None:
    rgb = np.full((120, 160, 3), 0.4, dtype=np.float32)
    rgb[40:80, 50:110] = 0.7
    from superphoto.models import TechnicalReport

    comp = measure_composition(rgb)
    tech = TechnicalReport(
        width=160,
        height=120,
        source_kind="jpeg",
        mean_luma=0.4,
        median_luma=0.4,
        std_luma=0.1,
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_spread=0.4,
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.2,
        temperature_bias=0.0,
        top_third_blue=0.0,
        skin_fraction=0.0,
        green_fraction=0.0,
        histogram=[0.0] * 32,
        **comp,
    )
    geom = suggest_geometry(tech, "general", "B")
    assert geom.is_noop() or (geom.crop_right - geom.crop_left) > 0.84


def test_apply_geometry_crops() -> None:
    rgb = np.zeros((80, 100, 3), dtype=np.float32)
    rgb[..., 1] = 0.2
    geom = GeometryHint(crop_left=0.1, crop_top=0.1, crop_right=0.9, crop_bottom=0.9, apply=True)
    out = apply_geometry(rgb, geom)
    assert out.shape[0] < 80
    assert out.shape[1] < 100


def test_xmp_look_is_not_applied_full_on_jpeg(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    critique = local_critique(tech)
    critique.correction = correction_from_technical(tech)
    ids = compose_look_ids(critique, [], tech)
    plans = build_plans(ids, critique, tech)
    assert plans[0].look_id == "faithful"
    for plan in plans[1:]:
        if get_look(plan.look_id).source == "xmp":
            assert plan.lut_mix < 0.90
            assert plan.lut_mix > 0.30


def test_dune_aqua_kill_is_softened_on_lush_water() -> None:
    from superphoto.models import TechnicalReport
    from superphoto.preference import _fit_lut_to_scene

    raw = parse_xmp(preset_dir() / "dune.xmp")
    aqua0 = raw.hsl["aqua"].sat
    tech = TechnicalReport(
        width=100,
        height=100,
        source_kind="jpeg",
        mean_luma=0.3,
        median_luma=0.25,
        std_luma=0.1,
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_spread=0.5,
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.45,
        temperature_bias=0.0,
        top_third_blue=0.22,
        skin_fraction=0.0,
        green_fraction=0.25,
        histogram=[0.0] * 32,
        aqua_fraction=0.12,
    )
    fitted = _fit_lut_to_scene(raw, "dune", tech)
    assert fitted.hsl["aqua"].sat > aqua0  # less negative = less kill
    assert look_match("dune", tech, "landscape") < 0.55
    look = get_look("dune")
    amount = lut_amount(look, tech, "B", look_match("dune", tech, "landscape"), 0.55)
    assert amount < 0.65


def test_recipe_punch_still_visible(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    critique = local_critique(tech)
    plans = build_plans(compose_look_ids(critique, [], tech), critique, tech)

    def punch(r: Recipe) -> float:
        hsl = sum(abs(s.sat) + abs(s.lum) + abs(s.hue) / 40.0 for s in r.hsl.values())
        g = sum(abs(x) for x in (*r.grade.shadows, *r.grade.midtones, *r.grade.highlights))
        curve = 0.3 if not r.curve.is_identity() else 0.0
        return abs(r.contrast) + abs(r.blacks) + abs(r.whites) + hsl + g + curve

    assert punch(plans[1].recipe) > 0.08
    assert punch(plans[2].recipe) > 0.08
