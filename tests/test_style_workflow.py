"""A style must survive exposure correction, bounded retries and backoff."""
import numpy as np
import pytest

from superphoto.analysis import correction_from_technical
from superphoto.exposure import (
    blend_tone_curves, compose_rgb_curves, merge_correction_style,
    preserve_exposure_correction,
)
from superphoto.lightroom_qa import separate_candidate_recipe
from superphoto.models import Recipe, ToneCurve, HueShift, ParametricTone
from superphoto.preference import correction_recipe, reduce_style_strength
from test_developed_exposure import measured, assert_recovery_preserved


def interp(xs, points):
    return np.interp(xs, *np.array(points).T)


def test_curve_composition_keeps_style_knots_between_base_knots():
    base = [(0, 0), (.3, .16), (.8, .59), (1, 1)]
    style = [(0, .01), (.25, .14), (.53, .62), (1, .98)]
    xs = np.linspace(0, 1, 2001)
    combined = compose_rgb_curves(base, style)
    assert np.allclose(interp(xs, combined), interp(interp(xs, base), style), atol=1e-12)
    # Chaining the curve is materially different from replacing the base.
    assert not np.allclose(interp(xs, combined), interp(xs, style))


@pytest.mark.parametrize('mid', [.43, .57])
def test_style_and_channel_palette_survive_repeated_protection(tmp_path, mid):
    _, tech = measured(tmp_path)
    primary = correction_recipe(correction_from_technical(tech), tech)
    style = Recipe(curve=ToneCurve(rgb=[(0, .01), (.5, mid), (1, .98)],
                                  b=[(0, .015), (.5, .475), (1, .97)]),
                   contrast=.14, hsl={'green': HueShift(lum=.10)},
                   parametric_tone=ParametricTone(darks=-.05))
    composed = merge_correction_style(primary, style, tech)
    guarded = preserve_exposure_correction(composed, primary, tech)
    assert_recovery_preserved(guarded, primary, tech)
    assert abs(interp(.5, guarded.curve.rgb) - interp(.5, primary.curve.rgb)) > .015
    assert guarded.curve.b[-1] == (1, 1)
    assert interp(.5, guarded.curve.b) == pytest.approx(.475)
    assert guarded.contrast == .14
    assert guarded.hsl['green'].lum == .10
    assert guarded.parametric_tone.darks == -.05
    assert style.curve.b[-1] == (1, .97)  # Input not mutated.


def test_strength_backoff_returns_to_corrected_base_not_identity(tmp_path):
    _, tech = measured(tmp_path)
    primary = correction_recipe(correction_from_technical(tech), tech)
    style = Recipe(curve=ToneCurve(rgb=[(0, .01), (.5, .43), (1, .98)]), contrast=.16)
    full = preserve_exposure_correction(merge_correction_style(primary, style, tech), primary, tech)
    xs = np.linspace(0, 1, 501)
    for amount in (0, .4, 1):
        reduced = reduce_style_strength(full, primary, amount)
        expected = interp(xs, primary.curve.rgb) * (1-amount) + interp(xs, full.curve.rgb) * amount
        assert np.allclose(interp(xs, reduced.curve.rgb), expected)
    assert np.allclose(interp(xs, blend_tone_curves(primary.curve, full.curve, 0).rgb), interp(xs, primary.curve.rgb))


@pytest.mark.parametrize('mid,direction', [(.43, -1), (.57, 1)])
def test_diversity_retry_uses_style_relative_to_corrected_base(tmp_path, mid, direction):
    _, tech = measured(tmp_path)
    primary = correction_recipe(correction_from_technical(tech), tech)
    style = Recipe(curve=ToneCurve(rgb=[(0, 0), (.5, mid), (1, 1)]))
    full = preserve_exposure_correction(merge_correction_style(primary, style, tech), primary, tech)
    changed = separate_candidate_recipe(full, slot='B', attempt=0, method='preset:test', technical=tech, primary=primary)
    changed = preserve_exposure_correction(changed, primary, tech)
    assert direction * (interp(.5, changed.curve.rgb) - interp(.5, full.curve.rgb)) > .015
    assert_recovery_preserved(changed, primary, tech)
    assert changed.hsl == full.hsl and changed.color_grading == full.color_grading


@pytest.mark.parametrize('points', [[(0, 0), (1, 1)], [(0, .02), (.82, .79), (1, .96)]])
def test_sparse_curve_retry_inserts_a_real_midtone_control(tmp_path, points):
    _, tech = measured(tmp_path, (.07, .12, .04))
    original = Recipe(curve=ToneCurve(rgb=points, b=[(0, .01), (1, .97)]))
    changed = separate_candidate_recipe(original, slot='C', attempt=0, method='preset:test', technical=tech)
    assert abs(interp(.5, changed.curve.rgb) - interp(.5, original.curve.rgb)) >= .011
    assert changed.curve.rgb[0] == original.curve.rgb[0]
    assert changed.curve.rgb[-1] == original.curve.rgb[-1]
    assert changed.curve.b == original.curve.b
    assert original.curve.rgb == points
    assert np.all(np.diff(interp(np.linspace(0, 1, 501), changed.curve.rgb)) >= 0)


def test_ordinary_scene_guard_does_not_change_style(tmp_path):
    _, tech = measured(tmp_path, (.07, .12, .04))
    style = Recipe(curve=ToneCurve(rgb=[(0, .02), (.5, .4), (1, .98)]), contrast=.2)
    assert preserve_exposure_correction(style, Recipe(), tech) == style
