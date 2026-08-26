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


def test_identity_is_too_weak_for_look_slot() -> None:
    rgb = np.full((48, 48, 3), 0.25, dtype=np.float32)
    lin = srgb_to_linear(rgb)
    tech = _tech()
    qa = score_grade(lin, lin, tech, "B")
    assert qa.too_weak
    assert qa.score < 80


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


def test_visible_contrast_grade_scores_higher() -> None:
    from superphoto.render import apply_recipe

    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[:, :32] = 0.18
    rgb[:, 32:] = 0.55
    lin = srgb_to_linear(rgb)
    out = apply_recipe(lin, Recipe(contrast=0.28, blacks=-0.12, whites=0.08, clarity=0.16))
    qa = score_grade(lin, out, _tech(), "B")
    ident = score_grade(lin, lin, _tech(), "B")
    assert qa.score > ident.score
    assert qa.mae > 0.03


def test_refine_strengthens_weak_plan() -> None:
    from superphoto.models import Critique

    look = get_look("landscape_filmic")
    plan = VariantPlan(
        slot="B",
        look_id=look.id,
        title=look.title,
        why="x",
        recipe=look.recipe,
        lut_mix=0.3,
    )
    qa = score_grade(
        srgb_to_linear(np.full((16, 16, 3), 0.3, dtype=np.float32)),
        srgb_to_linear(np.full((16, 16, 3), 0.3, dtype=np.float32)),
        _tech(),
        "B",
    )
    qa.too_weak = True
    qa.too_strong = False
    out = refine_plan(plan, qa, Critique(), _tech(), 0)
    assert abs(out.recipe.contrast) + abs(out.recipe.clarity) > abs(plan.recipe.contrast) + abs(plan.recipe.clarity) - 1e-6
    assert out.recipe.contrast >= plan.recipe.contrast


def test_targets() -> None:
    assert target_for("A") < target_for("B") <= target_for("C")
    assert target_for("C") >= 88
