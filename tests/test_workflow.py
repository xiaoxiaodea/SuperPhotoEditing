from __future__ import annotations

import numpy as np

from superphoto.color import linear_to_srgb, srgb_to_linear
from superphoto.models import HueShift, LocalEdit, Recipe, TechnicalReport
from superphoto.workflow import (
    apply_ethics,
    apply_restoration,
    ethics_regime_for,
    guard_hues,
    plan_restoration,
    score_restoration,
    simple_explanation,
    split_retouch_recipe,
)


def _tech(**kwargs) -> TechnicalReport:
    base = dict(
        width=64,
        height=64,
        source_kind="jpeg",
        mean_luma=0.25,
        median_luma=0.22,
        std_luma=0.1,
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_spread=0.5,
        sharpness=0.05,
        noise=0.02,
        saturation_mean=0.3,
        temperature_bias=0.0,
        top_third_blue=0.0,
        skin_fraction=0.0,
        green_fraction=0.1,
        histogram=[0.0] * 32,
    )
    base.update(kwargs)
    return TechnicalReport(**base)


def test_restoration_is_color_neutral() -> None:
    rng = np.random.default_rng(2)
    srgb = np.clip(rng.normal((0.42, 0.28, 0.18), 0.035, (64, 64, 3)), 0, 1).astype(np.float32)
    linear = srgb_to_linear(srgb)
    plan = plan_restoration(_tech(noise=0.11, sharpness=0.025, source_kind="raw"))
    restored = apply_restoration(linear, plan)
    gate = score_restoration(linear, restored)
    src = linear_to_srgb(linear)
    dst = linear_to_srgb(restored)
    src_chroma = src - np.mean(src, axis=-1, keepdims=True)
    dst_chroma = dst - np.mean(dst, axis=-1, keepdims=True)
    assert float(np.mean(np.abs(src_chroma - dst_chroma))) < 0.01
    assert gate.accepted


def test_upper_frame_blue_does_not_veto_luminance_for_a_blue_object():
    recipe = Recipe(hsl={"blue": HueShift(hue=20, lum=.10), "aqua": HueShift(lum=.03)})
    guarded = guard_hues(recipe, _tech(top_third_blue=.8))
    assert guarded.hsl["blue"].hue == 12
    assert guarded.hsl["blue"].lum == .10
    assert guarded.hsl["aqua"].lum == .03


def test_restoration_planner_does_not_force_cleanup() -> None:
    plan = plan_restoration(_tech(noise=0.01, sharpness=0.08, highlight_clip=0.0))
    assert plan.is_noop()
    assert not plan.actions


def test_developed_jpeg_skips_restoration() -> None:
    plan = plan_restoration(_tech(noise=0.12, sharpness=0.02, source_kind="jpeg"))
    assert plan.is_noop()
    raw = plan_restoration(_tech(noise=0.12, sharpness=0.02, source_kind="raw"))
    assert not raw.is_noop()


def test_retouch_preserves_coherent_local_and_global_intents() -> None:
    recipe = Recipe(
        exposure_ev=0.4,
        temperature=14,
        saturation=0.18,
        dehaze=0.16,
        vignette=0.18,
        clarity=0.5,
        local=LocalEdit(
            subject_lift=0.08,
            subject_vibrance=0.18,
            subject_clarity=0.5,
            subject_sharpen=0.7,
            background_darken=0.16,
            background_desat=0.24,
            color_clean=0.4,
            defect_heal=0.8,
        ),
    )
    retouch, attributes = split_retouch_recipe(recipe)
    assert set(attributes) == {"tone", "color", "atmosphere", "subject", "background"}
    assert 0 < retouch.clarity <= 0.12
    assert 0 < retouch.local.subject_clarity <= 0.15
    assert retouch.local.subject_lift == recipe.local.subject_lift
    assert retouch.vignette == recipe.vignette
    assert retouch.local.subject_sharpen == 0
    assert retouch.local.defect_heal == 0


def test_simple_explanation_is_nontechnical() -> None:
    plan = plan_restoration(_tech(noise=0.10))
    explanation = simple_explanation(plan, ["tone", "subject"], True, ethics="portrait")
    text = "".join(sum(explanation.values(), []))
    assert "整理明暗层次" in text
    assert "轻微突出主体" in text
    assert "按主体、天空和背景分别保护" in text
    assert "肤色锁在自然范围" in text
    assert "exposure" not in text.lower()
    assert "sharpness" not in text.lower()


def test_primary_correction_survives_attribute_limit() -> None:
    primary = Recipe(exposure_ev=0.32, temperature=9)
    look = Recipe(
        exposure_ev=0.32,
        temperature=9,
        saturation=0.22,
        dehaze=0.20,
        vignette=0.22,
        local=LocalEdit(
            subject_lift=0.10,
            subject_vibrance=0.16,
            background_darken=0.18,
            background_desat=0.24,
            color_clean=0.40,
            clutter_suppress=0.50,
            sky_dehaze=0.30,
        ),
    )
    out, attributes = split_retouch_recipe(look, primary=primary, genre="landscape", max_attributes=3)
    assert abs(out.exposure_ev - 0.32) < 1e-6
    assert abs(out.temperature - 9) < 1e-6
    assert len(attributes) <= 3
    assert out.local.subject_lift == look.local.subject_lift
    assert out.dehaze == look.dehaze
    assert "tone" in attributes or abs(out.exposure_ev - 0.32) < 1e-6


def test_documentary_ethics_freezes_clutter() -> None:
    recipe = Recipe(local=LocalEdit(clutter_suppress=0.6, color_clean=0.4, defect_heal=0.8))
    out = apply_ethics(recipe, "documentary")
    assert out.local.clutter_suppress == 0
    assert out.local.defect_heal == 0
    assert out.local.color_clean < recipe.local.color_clean
    frozen, _ = split_retouch_recipe(recipe, ethics="documentary")
    assert frozen.local.clutter_suppress == 0


def test_hue_guard_keeps_foliage_inside_nature() -> None:
    recipe = Recipe(hsl={"green": HueShift(hue=28, sat=0.4)})
    tech = _tech(green_fraction=0.3)
    guarded = guard_hues(recipe, tech, "landscape")
    assert abs(guarded.hsl["green"].hue) <= 12
    assert guarded.hsl["green"].sat <= 0.04


def test_landscape_keeps_background_and_subject_together() -> None:
    recipe = Recipe(
        contrast=0.14,
        saturation=0.12,
        dehaze=0.14,
        vignette=0.16,
        local=LocalEdit(subject_lift=0.08, background_darken=0.10, background_desat=0.14),
    )
    _out, attributes = split_retouch_recipe(recipe, genre="landscape")
    assert "background" in attributes
    assert "subject" in attributes
    assert "atmosphere" in attributes


def test_ethics_regime_follows_genre() -> None:
    assert ethics_regime_for("street") == "documentary"
    assert ethics_regime_for("landscape") == "landscape"
    assert ethics_regime_for("portrait") == "portrait"
    assert ethics_regime_for("product") == "product"
    assert ethics_regime_for("night") == "commercial"
