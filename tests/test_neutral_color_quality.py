"""Natural-color intent protects measured source neutrals without semantic claims."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from superphoto.lightroom_qa import assess_lightroom_render, reduce_color_cast
from superphoto.models import ColorGradingControls, HueShift, MaskAdjustment, Recipe, ToneCurve


def _pair(tmp_path: Path, *, source_cast=(0, 0, 0), added_cast=(0, 0, 0), area=1.0):
    values = np.broadcast_to(np.linspace(.55, .9, 160)[None, :, None], (100, 160, 3)).copy()
    before = np.clip(values + source_cast, 0, 1)
    after = before.copy()
    after[:round(100 * area)] = np.clip(after[:round(100 * area)] + added_cast, 0, 1)
    paths = [tmp_path/'before.png', tmp_path/'after.png']
    for path, pixels in zip(paths, (before, after)):
        Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(path)
    return paths


@pytest.mark.parametrize('cast', [(-.025, .015, .005), (0, .02, .04), (.04, .02, -.02)])
def test_natural_intent_rejects_new_cast_on_source_neutral_brights(tmp_path, cast):
    before, after = _pair(tmp_path, added_cast=cast)
    checked = assess_lightroom_render(before, after, slot='C', creative_palette=True, natural_color=True)
    assert not checked.accepted
    issue = next(i for i in checked.issues if i.code == 'neutral_color_cast')
    assert issue.hard and not issue.advisory
    assert checked.neutral_assessment['fraction'] > .5
    assert checked.neutral_assessment['median_shift'] > 3
    # A deliberate palette is allowed when preserving natural color is not requested.
    assert not any(i.code == 'neutral_color_cast' for i in
        assess_lightroom_render(before, after, slot='C', creative_palette=True, natural_color=False).issues)


def test_original_warm_light_and_small_color_changes_are_not_neutral_damage(tmp_path):
    before, after = _pair(tmp_path, source_cast=(.04, .01, -.06))
    checked = assess_lightroom_render(before, after, slot='A', natural_color=True)
    assert checked.accepted and not any(i.code == 'neutral_color_cast' for i in checked.issues)
    before, after = _pair(tmp_path, added_cast=(.004, .002, -.003))
    assert assess_lightroom_render(before, after, slot='C', natural_color=True).accepted


def test_neutral_hue_swap_cannot_hide_behind_unchanged_chroma(tmp_path):
    before, after = _pair(tmp_path, source_cast=(0, 0, .03), added_cast=(-.025, .015, -.025))
    checked = assess_lightroom_render(before, after, slot='C', natural_color=True)
    assert any(i.code == 'neutral_color_cast' for i in checked.issues)
    assert checked.neutral_assessment['chroma_after'] < checked.neutral_assessment['chroma_before'] + .8


def test_moving_source_cool_neutrals_toward_gray_is_allowed(tmp_path):
    before, after = _pair(tmp_path, source_cast=(0, 0, .03), added_cast=(0, 0, -.03))
    checked = assess_lightroom_render(before, after, slot='C', natural_color=True)
    assert not any(i.code == 'neutral_color_cast' for i in checked.issues)
    assert checked.neutral_assessment['introduced_chroma'] < .1


def test_cast_on_tiny_area_is_not_a_global_color_repair_signal(tmp_path):
    before, after = _pair(tmp_path, added_cast=(-.03, .02, 0), area=.01)
    checked = assess_lightroom_render(before, after, slot='C', natural_color=True)
    assert not any(i.code == 'neutral_color_cast' for i in checked.issues)


def test_color_repair_preserves_light_design_and_native_local_exposure():
    primary = Recipe(exposure_ev=-.55)
    original = Recipe(exposure_ev=-.55, contrast=.18, shadows=.12,
        curve=ToneCurve(rgb=[(0, .01), (.5, .42), (1, 1)],
                        r=[(0, 0), (.35, .32), (1, 1)], b=[(0, .02), (.55, .51), (1, .96)]),
        hsl={'blue': HueShift(hue=3, sat=-.2, lum=.12)},
        color_grading=ColorGradingControls(shadow_hue=210, shadow_sat=8, highlight_lum=5),
        masks=[MaskAdjustment(target='subject', exposure_ev=.25, temperature=3, saturation=.1)])
    repaired = reduce_color_cast(original, primary, .4)
    assert repaired.exposure_ev == original.exposure_ev and repaired.contrast == original.contrast
    assert repaired.shadows == original.shadows and repaired.curve.rgb == original.curve.rgb
    assert repaired.hsl['blue'].lum == original.hsl['blue'].lum
    assert repaired.masks[0].exposure_ev == .25
    assert repaired.color_grading.highlight_lum == 5
    assert repaired.color_grading.shadow_sat == pytest.approx(3.2)
    xs = np.linspace(0, 1, 301)
    before_channels = np.array([np.interp(xs, *np.array(getattr(original.curve, c) or [(0, 0), (1, 1)]).T) for c in 'rgb'])
    after_channels = np.array([np.interp(xs, *np.array(getattr(repaired.curve, c)).T) for c in 'rgb'])
    weights = np.array([.2126, .7152, .0722])
    # Preserve the channels' common density while reducing their divergence.
    assert np.allclose(weights @ before_channels, weights @ after_channels)
    assert np.allclose(after_channels[0]-after_channels[1], .4*(before_channels[0]-before_channels[1]))
    assert original.color_grading.shadow_sat == 8 and original.masks[0].temperature == 3


def test_color_repair_preserves_corrected_base_palette_and_point_budget():
    primary = Recipe(curve=ToneCurve(r=[(0, 0), (.5, .49), (1, 1)]), tint=2)
    assert reduce_color_cast(primary, primary, 0) == primary
    original = primary.model_copy(deep=True)
    original.curve.b = [(float(x), float(x*.96)) for x in np.linspace(0, 1, 32)]
    original.curve.g = [(float(x), float(x)) for x in np.linspace(0, 1, 31)]
    repaired = reduce_color_cast(original, primary, .4)
    assert all(len(getattr(repaired.curve, c)) <= 32 for c in 'rgb')
    assert repaired.tint == primary.tint


@pytest.mark.parametrize('geometry', [
    {'crop': {'left': .1}}, {'crop': {'angle': 2}},
    {'optics': {'perspective_vertical': 8}}, {'optics': {'manual_distortion': -3}},
])
def test_geometry_changes_skip_unaligned_neutral_comparison(tmp_path, geometry):
    before, after = _pair(tmp_path, added_cast=(-.025, .015, .005))
    checked = assess_lightroom_render(before, after, slot='C', natural_color=True,
                                      render_recipe=Recipe.model_validate(geometry))
    assert not any(i.code == 'neutral_color_cast' for i in checked.issues)
    assert checked.neutral_assessment == {'skipped': 'geometry_changes'}


def test_identity_geometry_still_checks_neutral_color(tmp_path):
    from superphoto.models import CropControls, OpticsControls
    before, after = _pair(tmp_path, added_cast=(-.025, .015, .005))
    identity = Recipe(crop=CropControls(top=0, left=0, right=1, bottom=1, angle=0),
                      optics=OpticsControls(perspective_scale=100, perspective_vertical=0))
    checked = assess_lightroom_render(before, after, slot='C', natural_color=True, render_recipe=identity)
    assert any(i.code == 'neutral_color_cast' for i in checked.issues)
