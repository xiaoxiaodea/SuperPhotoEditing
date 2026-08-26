from __future__ import annotations

import numpy as np

from superphoto.color import linear_to_srgb, luma
from superphoto.recipes import LOOKS, get_look
from superphoto.render import apply_recipe
from superphoto.xmp import parse_xmp, preset_dir


def test_all_packaged_presets_parse() -> None:
    root = preset_dir()
    files = list(root.glob("*.xmp"))
    assert len(files) >= 16
    for path in files:
        recipe = parse_xmp(path)
        assert recipe.exposure_ev == 0.0
        has_look = (
            not recipe.curve.is_identity()
            or recipe.hsl
            or any(abs(x) > 1e-4 for x in (*recipe.grade.shadows, *recipe.grade.highlights))
            or abs(recipe.blacks) + abs(recipe.whites) + abs(recipe.contrast) > 0.05
        )
        assert has_look, path.name


def test_dune_has_rgb_curve_and_dead_aqua() -> None:
    look = get_look("dune")
    assert look.recipe.curve.rgb
    assert look.recipe.curve.r and look.recipe.curve.g and look.recipe.curve.b
    aqua = look.recipe.hsl.get("aqua")
    assert aqua is not None and aqua.sat < -0.3


def test_kdrama_boosts_orange() -> None:
    look = get_look("kdrama_tears")
    orange = look.recipe.hsl["orange"]
    assert orange.sat > 0.4


def test_xmp_look_does_not_turn_black_water_red() -> None:
    rgb = np.zeros((48, 48, 3), dtype=np.float32)
    rgb[..., 0] = 0.0015
    rgb[..., 1] = 0.0025
    rgb[..., 2] = 0.0018
    rgb[18:30, 18:30] = 0.35
    out = apply_recipe(rgb, LOOKS["dune"].recipe)
    water = out[:10, :10]
    assert float(np.mean(luma(water))) < 0.08
    assert float(np.mean(water[..., 0])) <= float(np.mean(water[..., 1])) * 1.5


def test_xmp_look_changes_a_normal_frame() -> None:
    rgb = np.full((32, 32, 3), 0.16, dtype=np.float32)
    rgb[:, 16:] = 0.42
    out = apply_recipe(rgb, LOOKS["skam"].recipe)
    src = linear_to_srgb(rgb)
    dst = linear_to_srgb(out)
    assert float(np.mean(np.abs(dst - src))) > 0.02
