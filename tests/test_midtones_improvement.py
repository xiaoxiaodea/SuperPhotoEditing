"""A safe preset must also resolve the measured problem on the delivered pixels."""
import numpy as np
import pytest
from PIL import Image

from superphoto.analysis import analyze
from superphoto.color import linear_to_srgb
from superphoto.critique import local_critique
from superphoto.exposure import developed_lift_ev
from superphoto.lightroom_qa import assess_lightroom_render, correction_for_report
from superphoto.models import CorrectionHint, EditIntent, Recipe
from superphoto.preference import build_plans, sanitize_correction


def dark_developed(tmp_path):
    ramp = np.linspace(.035, .23, 180, dtype=np.float32)[None, :, None]
    pixels = np.broadcast_to(ramp * np.array([.88, .94, 1.0]), (120, 180, 3)).copy()
    return pixels, analyze(pixels, tmp_path/'dark.jpg')


def test_broad_dark_midtones_get_meaningful_common_correction(tmp_path):
    _, tech = dark_developed(tmp_path)
    critique = local_critique(tech)
    lift = developed_lift_ev(tech)
    assert .3 < lift <= .55
    assert critique.correction.exposure_ev == pytest.approx(lift)
    assert any('亮部仍有余量' in issue.note for issue in critique.issues)
    # Directions carry the common correction; it is not merely display prose.
    from superphoto.directions import preset_direction
    critique.directions = [preset_direction(slot, look, critique, tech)
                          for slot, look in zip('ABC', ['landscape_filmic', 'resource_matte', 'moody_lowkey'])]
    plans = build_plans(['landscape_filmic', 'resource_matte', 'moody_lowkey'], critique, tech)
    assert all(plan.primary.exposure_ev == pytest.approx(lift) for plan in plans)
    assert all(plan.recipe.exposure_ev == pytest.approx(lift) for plan in plans)
    assert sanitize_correction(CorrectionHint(exposure_ev=0), tech).exposure_ev == 0
    critique.intent = EditIntent(locked_controls=['exposure_ev'])
    assert all(plan.recipe.exposure_ev == 0 for plan in build_plans(
        ['landscape_filmic', 'resource_matte', 'moody_lowkey'], critique, tech))


@pytest.mark.parametrize('changes', [
    {'median_luma':.004, 'mean_luma':.01, 'highlight_p95':.04},  # night
    {'median_luma':.12, 'highlight_p95':.75},  # bright subject/sky
    {'median_luma':.04, 'shadow_clip':.6, 'black_clip':.15},  # silhouette
    {'median_luma':.45, 'highlight_p95':.8},  # high key
    {'median_luma':.24, 'highlight_p95':.48},  # existing haze
    {'white_clip_fraction':.01},
    {'unclipped_fraction':0.0},  # older report lacks measured support
])
def test_lift_requires_actual_headroom_and_is_not_universal(tmp_path, changes):
    _, tech = dark_developed(tmp_path)
    assert developed_lift_ev(tech.model_copy(update=changes)) == 0


def save_linear(path, linear):
    Image.fromarray(np.round(np.clip(linear_to_srgb(linear),0,1)*255).astype('uint8')).save(path)


def test_unchanged_damage_free_export_fails_its_actual_improvement_goal(tmp_path):
    pixels, tech = dark_developed(tmp_path)
    before, after = tmp_path/'before.png', tmp_path/'after.png'
    save_linear(before, pixels)
    lift = developed_lift_ev(tech)
    assert assess_lightroom_render(before,before,slot='A').accepted
    unchanged = assess_lightroom_render(before,before,slot='A',exposure_lift_ev=lift)
    assert not unchanged.accepted
    assert 'exposure_lift_unresolved' in [i.code for i in unchanged.issues]
    assert correction_for_report(unchanged).exposure_ev > 0
    save_linear(after, pixels * 2**lift)
    improved = assess_lightroom_render(before,after,slot='A',exposure_lift_ev=lift)
    assert improved.accepted, improved.to_dict()
    assert improved.exposure_assessment['lift_ratio'] > improved.exposure_assessment['lift_min_ratio']
    # Fixed coordinates cannot be used following geometry changes.
    moved = assess_lightroom_render(before,before,slot='A',exposure_lift_ev=lift,
        render_recipe=Recipe(crop={'left':.1}))
    assert moved.exposure_assessment['lift_skipped'] == 'geometry_changes'
