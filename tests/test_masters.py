from __future__ import annotations

from superphoto.analysis import analyze
from superphoto.critique import compose_look_ids, local_critique
from superphoto.ingest import load_linear
from superphoto.masters import (
    environment_label,
    look_for,
    ranked_master_ids,
    recipe_from_print,
)
from superphoto.models import Recipe
from superphoto.preference import build_plans, look_match


def test_recipe_from_print_is_parametric(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    from superphoto.color import linear_to_srgb
    import numpy as np

    srgb = np.clip(linear_to_srgb(np.maximum(linear, 0.0)), 0.0, 1.0)
    recipe = recipe_from_print(srgb, tech)
    assert isinstance(recipe, Recipe)
    punch = abs(recipe.contrast) + abs(recipe.blacks) + abs(recipe.vignette)
    assert punch >= 0.0


def test_environment_label(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    label = environment_label(tech)
    assert label in {"night", "portrait", "nature", "water", "landscape", "warm-urban", "architecture"}


def test_compose_still_three_slots_with_masters(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    critique = local_critique(tech)
    ids = compose_look_ids(critique, [], tech)
    assert ids[0] == "faithful"
    assert len(ids) == 3
    plans = build_plans(ids, critique, tech)
    assert len(plans) == 3
    assert plans[0].look_id == "faithful"
    if any(p.look_id.startswith("hb:") for p in plans):
        master = next(p for p in plans if p.look_id.startswith("hb:"))
        assert master.master is not None
        assert "哈苏" in master.title or "Hasselblad" in master.why or "组图" in master.why
        assert look_for(master.look_id) is not None
        assert look_match(master.look_id, tech, critique.genre) >= 0.18


def test_ranked_masters_unique_photographers() -> None:
    from superphoto.models import TechnicalReport

    tech = TechnicalReport(
        width=64,
        height=64,
        source_kind="jpeg",
        mean_luma=0.3,
        median_luma=0.3,
        std_luma=0.15,
        highlight_clip=0.0,
        shadow_clip=0.05,
        dynamic_spread=0.5,
        sharpness=0.1,
        noise=0.01,
        saturation_mean=0.25,
        temperature_bias=0.0,
        top_third_blue=0.2,
        skin_fraction=0.02,
        green_fraction=0.2,
        histogram=[0.0] * 32,
    )
    ids = ranked_master_ids(tech)
    assert len(ids) == len(set(ids))
