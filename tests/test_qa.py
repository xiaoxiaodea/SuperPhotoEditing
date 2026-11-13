from __future__ import annotations

import numpy as np

from superphoto.color import srgb_to_linear
from superphoto.models import Recipe, TechnicalReport
from superphoto.qa import refine_plan, score_grade, target_for
from superphoto.recipes import get_look
from superphoto.models import VariantPlan


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
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.3,
        temperature_bias=0.0,
        top_third_blue=0.0,
        skin_fraction=0.0,
        green_fraction=0.1,
        histogram=[0.0] * 32,
    )
    base.update(kwargs)
    return TechnicalReport(**base)


def test_refinement_keeps_local_intent_and_capture_detail():
    from superphoto.models import Critique, EditIntent, MaskAdjustment, DetailControls, CropControls
    from superphoto.preference import build_plans
    from superphoto.qa import QAReport
    critique = Critique(intent=EditIntent(locked_controls=["grain", "temperature"], masks=[
        MaskAdjustment(target="subject", exposure_ev=0.3, reason="人物面部过暗")]))
    plan = build_plans(["celebrity_rescue", "celebrity_twilight_kirinoma"], critique, _tech())[-1]
    plan.recipe.detail = DetailControls(sharpness=38)
    plan.recipe.crop = CropControls(left=0.02, right=0.98)
    report = QAReport(score=60, presence=90, safety=60, finish=70, mae=0.15, too_weak=False, too_strong=True)
    refined = refine_plan(plan, report, critique, _tech(), 0)
    assert refined.recipe.masks == plan.recipe.masks
    assert refined.recipe.detail.sharpness == 38
    assert refined.recipe.crop == plan.recipe.crop
    assert refined.recipe.temperature == refined.recipe.grain == 0
    assert not refined.recipe.curve.r and not refined.recipe.curve.b


def test_small_visual_difference_does_not_force_more_style():
    from superphoto.models import Critique
    from superphoto.preference import build_plans
    from superphoto.qa import QAReport
    critique = Critique()
    plan = build_plans(["faithful", "celebrity_twilight_kirinoma"], critique, _tech())[-1]
    report = QAReport(score=83, presence=75, safety=100, finish=80, mae=0.005, too_weak=True, too_strong=False)
    assert refine_plan(plan, report, critique, _tech(), 0) is plan


def test_identity_is_only_too_weak_for_styled_slot() -> None:
    rgb = np.full((48, 48, 3), 0.25, dtype=np.float32)
    lin = srgb_to_linear(rgb)
    tech = _tech()
    styled = score_grade(lin, lin, tech, "B")
    natural = score_grade(lin, lin, tech, "A")
    assert styled.too_weak
    assert not natural.too_weak
    assert natural.safety == 100


def test_red_water_fails_safety() -> None:
    src = np.zeros((40, 40, 3), dtype=np.float32)
    src[..., 1] = 0.003
    dst = src.copy()
    dst[..., 0] = 0.35
    dst[..., 1] = 0.04
    dst[..., 2] = 0.04
    qa = score_grade(src, dst, _tech(mean_luma=0.05, shadow_clip=0.4), "B")
    assert qa.safety < 80
    assert qa.too_strong or qa.score < 70


def test_visible_but_safe_grade_scores_higher_than_crushed_grade() -> None:
    from superphoto.render import apply_recipe

    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[:, :32] = 0.18
    rgb[:, 32:] = 0.55
    lin = srgb_to_linear(rgb)
    safe = apply_recipe(lin, Recipe(contrast=0.10, blacks=-0.03, whites=0.03, clarity=0.05))
    crushed = apply_recipe(lin, Recipe(contrast=0.65, blacks=-0.75, whites=0.55, clarity=0.6))
    safe_qa = score_grade(lin, safe, _tech(), "B")
    crushed_qa = score_grade(lin, crushed, _tech(), "B")
    assert safe_qa.score > crushed_qa.score
    assert crushed_qa.too_strong


def test_sky_blowout_fails_self_check() -> None:
    src = np.zeros((90, 60, 3), dtype=np.float32)
    src[:30] = (0.72, 0.80, 0.90)
    src[30:] = (0.22, 0.18, 0.12)
    dst = src.copy()
    dst[:30] = (0.97, 0.98, 0.99)
    qa = score_grade(
        srgb_to_linear(src),
        srgb_to_linear(dst),
        _tech(top_third_blue=0.35, highlight_clip=0.0, highlight_p95=0.78),
        "C",
    )
    assert qa.too_strong
    assert qa.safety < target_for("C")
    assert any("天空过曝" in note for note in qa.notes)


def test_source_sky_identity_is_not_blown() -> None:
    src = np.zeros((90, 60, 3), dtype=np.float32)
    src[:30] = (0.72, 0.80, 0.90)
    src[30:] = (0.22, 0.18, 0.12)
    lin = srgb_to_linear(src)
    qa = score_grade(lin, lin, _tech(top_third_blue=0.35), "C")
    assert not qa.too_strong
    assert not any("天空过曝" in note for note in qa.notes)


def test_shadow_crush_fails_safety() -> None:
    src = np.full((48, 48, 3), 0.12, dtype=np.float32)
    dst = src.copy()
    dst[:32] = 0.0
    qa = score_grade(srgb_to_linear(src), srgb_to_linear(dst), _tech(), "B")
    assert qa.safety < 70
    assert qa.too_strong


def test_saturation_collapse_fails_safety() -> None:
    src = np.full((40, 40, 3), 0.62, dtype=np.float32)
    src[..., 1] = 0.48
    src[..., 2] = 0.30
    dst = np.full((40, 40, 3), 0.50, dtype=np.float32)
    qa = score_grade(
        srgb_to_linear(src),
        srgb_to_linear(dst),
        _tech(mean_luma=0.79, saturation_mean=0.35, source_kind="jpeg"),
        "A",
    )
    assert qa.safety < 85
    assert any("抽干" in n or "发闷" in n for n in qa.notes)


def test_nuclear_greens_fail_safety() -> None:
    src = np.zeros((40, 40, 3), dtype=np.float32)
    src[..., 0] = 0.12
    src[..., 1] = 0.28
    src[..., 2] = 0.10
    dst = src.copy()
    dst[..., 1] = 0.62
    qa = score_grade(
        srgb_to_linear(src),
        srgb_to_linear(dst),
        _tech(green_fraction=0.4, saturation_mean=0.2),
        "B",
    )
    assert qa.safety < 90
    assert any("绿" in n for n in qa.notes)


def test_refine_backs_off_unsafe_plan() -> None:
    from superphoto.models import Critique

    look = get_look("landscape_filmic")
    plan = VariantPlan(
        slot="B",
        look_id=look.id,
        title=look.title,
        why="x",
        recipe=look.recipe,
        lut_mix=0.7,
    )
    qa = score_grade(
        srgb_to_linear(np.full((16, 16, 3), 0.3, dtype=np.float32)),
        srgb_to_linear(np.zeros((16, 16, 3), dtype=np.float32)),
        _tech(),
        "B",
    )
    qa.too_weak = False
    qa.too_strong = True
    out = refine_plan(plan, qa, Critique(), _tech(), 0)
    assert out.lut_mix < plan.lut_mix
    assert out.recipe.contrast < plan.recipe.contrast


def test_targets_prioritize_natural_safety() -> None:
    assert target_for("A") <= target_for("B")
    assert target_for("B") == target_for("C")
    assert target_for("A") >= 80
