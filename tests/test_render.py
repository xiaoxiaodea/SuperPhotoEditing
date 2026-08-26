from __future__ import annotations

import numpy as np

from superphoto.color import linear_to_srgb, luma, srgb_to_linear
from superphoto.models import Recipe
from superphoto.render import apply_recipe


def test_srgb_roundtrip_midgray() -> None:
    x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    y = linear_to_srgb(srgb_to_linear(x))
    assert np.allclose(x, y, atol=1e-5)


def test_positive_exposure_raises_luma() -> None:
    rgb = np.full((32, 32, 3), 0.12, dtype=np.float32)
    out = apply_recipe(rgb, Recipe(exposure_ev=1.0))
    assert float(np.mean(luma(out))) > float(np.mean(luma(rgb))) * 1.5


def test_highlight_recovery_reduces_clip() -> None:
    rgb = np.full((48, 48, 3), 0.08, dtype=np.float32)
    rgb[:, 24:] = 1.2
    recovered = apply_recipe(rgb, Recipe(highlights=-0.8))
    assert float(np.mean(linear_to_srgb(recovered)[:, 24:])) < float(np.mean(linear_to_srgb(np.clip(rgb, 0, None))[:, 24:]))


def test_shadow_lift_does_not_turn_black_water_red() -> None:
    rgb = np.zeros((48, 48, 3), dtype=np.float32)
    rgb[..., 0] = 0.0015
    rgb[..., 1] = 0.0025
    rgb[..., 2] = 0.0018
    rgb[18:30, 18:30] = 0.35
    out = apply_recipe(rgb, Recipe(exposure_ev=0.8, shadows=0.35, saturation=0.12))
    water = out[:10, :10]
    # Pond must stay dark and not become a red field.
    assert float(np.mean(luma(water))) < 0.08
    assert float(np.mean(water[..., 0])) <= float(np.mean(water[..., 1])) * 1.35


def test_contrast_is_visibly_stronger() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[:, :32] = 0.22
    rgb[:, 32:] = 0.62
    out = apply_recipe(rgb, Recipe(contrast=0.28))
    src = linear_to_srgb(rgb)
    dst = linear_to_srgb(out)
    spread_in = float(np.std(luma(src)))
    spread_out = float(np.std(luma(dst)))
    assert spread_out > spread_in * 1.12


def test_cinematic_grade_shifts_shadows_cool() -> None:
    from superphoto.recipes import LOOKS

    rgb = np.full((48, 48, 3), 0.05, dtype=np.float32)
    out = apply_recipe(rgb, LOOKS["cinematic_teal_orange"].recipe)
    src = linear_to_srgb(rgb)
    dst = linear_to_srgb(out)
    mae = float(np.mean(np.abs(dst - src)))
    assert mae > 0.02
    # Shadows of the look are cooler: more blue than red vs source.
    assert float(np.mean(dst[..., 2])) > float(np.mean(dst[..., 0]))
