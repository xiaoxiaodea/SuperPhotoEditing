from __future__ import annotations

import numpy as np

from superphoto.color import linear_to_srgb, luma, srgb_to_linear
from superphoto.layers import build_layers
from superphoto.local import apply_local
from superphoto.models import LocalEdit, Recipe
from superphoto.render import apply_recipe
from superphoto.semantic import SemanticMasks


def test_subject_mask_prefers_center_blob() -> None:
    rgb = np.full((80, 80, 3), 0.15, dtype=np.float32)
    rgb[28:52, 28:52] = (0.62, 0.42, 0.32)
    layers = build_layers(rgb)
    inner = float(np.mean(layers.subject[32:48, 32:48]))
    outer = float(np.mean(layers.subject[:10, :10]))
    assert inner > outer + 0.15


def test_semantic_subject_overrides_center_bias() -> None:
    rgb = np.full((80, 80, 3), 0.2, dtype=np.float32)
    zeros = np.zeros((80, 80), dtype=np.float32)
    subject = zeros.copy()
    subject[28:52, 4:24] = 1.0
    semantic = SemanticMasks(
        subject=subject,
        sky=zeros,
        foliage=zeros,
        water=zeros,
        skin=zeros,
        confidence=0.9,
        model="test",
    )
    layers = build_layers(rgb, semantic=semantic)
    assert float(np.mean(layers.subject[32:48, 8:20])) > 0.7
    assert float(np.mean(layers.subject[32:48, 32:48])) < 0.35


def test_semantic_sky_rejects_blue_object_outside_sky() -> None:
    rgb = np.full((80, 80, 3), 0.2, dtype=np.float32)
    rgb[52:72, 20:60] = (0.15, 0.45, 0.8)
    zeros = np.zeros((80, 80), dtype=np.float32)
    sky = zeros.copy()
    sky[:20] = 1.0
    semantic = SemanticMasks(
        subject=zeros,
        sky=sky,
        foliage=zeros,
        water=zeros,
        skin=zeros,
        confidence=0.9,
        model="test",
    )
    layers = build_layers(rgb, semantic=semantic)
    assert float(np.mean(layers.sky[:12])) > 0.7
    assert float(np.mean(layers.sky[58:68, 28:52])) < 0.3


def test_color_clean_reduces_background_saturation() -> None:
    rgb = np.full((64, 64, 3), 0.22, dtype=np.float32)
    rgb[20:44, 20:44] = (0.70, 0.48, 0.36)
    rgb[:12, :] = (0.15, 0.55, 0.20)
    recipe = Recipe(local=LocalEdit(color_clean=0.7, background_desat=0.4))
    out = apply_local(rgb, recipe)
    bg_in = rgb[:12]
    bg_out = out[:12]
    sat = lambda im: float(np.mean(np.max(im, axis=-1) - np.min(im, axis=-1)))
    assert sat(bg_out) < sat(bg_in) * 0.92


def test_local_does_not_turn_black_water_red() -> None:
    rgb = np.zeros((48, 48, 3), dtype=np.float32)
    rgb[..., 0] = 0.0015
    rgb[..., 1] = 0.0025
    rgb[..., 2] = 0.0018
    rgb[16:32, 16:32] = 0.4
    lin = srgb_to_linear(rgb)
    out = apply_recipe(
        lin,
        Recipe(local=LocalEdit(color_clean=0.4, subject_lift=0.06, background_darken=0.1, clutter_suppress=0.4)),
    )
    water = out[:10, :10]
    assert float(np.mean(luma(water))) < 0.08
    assert float(np.mean(water[..., 0])) <= float(np.mean(water[..., 1])) * 1.4


def test_subject_sharpen_increases_local_contrast() -> None:
    rgb = np.full((48, 48, 3), 0.3, dtype=np.float32)
    rgb[::2, :, :] = 0.5
    recipe = Recipe(local=LocalEdit(subject_sharpen=0.6, subject_clarity=0.5))
    out = apply_local(rgb, recipe)
    assert float(np.std(luma(out))) >= float(np.std(luma(rgb))) * 0.98
    assert float(np.mean(np.abs(out - rgb))) > 0.002
