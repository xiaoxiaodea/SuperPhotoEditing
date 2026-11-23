"""Regressions for observed local travel-photo routing and exposure failures."""
import numpy as np
import pytest

from superphoto.analysis import analyze
from superphoto.celebrity_styles import build_style_decision, _color_card
from superphoto.critique import local_critique
from superphoto.models import CorrectionHint, Critique, EditIntent
from superphoto.preference import correction_recipe, sanitize_correction
from test_celebrity_styles import _tech


@pytest.mark.parametrize('blue,expected', [(1.0, 'landscape'), (0.0, 'general')])
def test_warm_stone_does_not_establish_portrait(blue, expected):
    t = _tech(mean_luma=.44, skin_fraction=.46, orange_fraction=.48,
              top_third_blue=blue, green_fraction=.01, exif={'FocalLength':'120'})
    c = local_critique(t)
    assert c.genre == expected
    d = build_style_decision(t, c)
    assert {'celebrity_portra_hamada', 'filmic_portrait_warm', 'kdrama_tears'} <= d.excluded.keys()
    assert '肤色' not in _color_card(t)


def test_explicit_visual_portrait_still_admits_portrait_styles():
    t = _tech(skin_fraction=.28, green_fraction=0)
    d = build_style_decision(t, Critique(genre='portrait', subject_note='湖边人物', source='openai'))
    assert 'celebrity_portra_hamada' not in d.excluded


def test_bright_colored_jpeg_is_corrected_before_clipping(tmp_path):
    # Bright sand-colored pixels retain headroom and texture; no white plateau.
    x = np.linspace(.60, .88, 180, dtype=np.float32)[None, :, None]
    rgb = np.broadcast_to(x * np.array([1.0, .92, .78], dtype=np.float32), (120, 180, 3)).copy()
    t = analyze(rgb, tmp_path/'bright.jpg')
    assert t.highlight_clip == 0
    c = local_critique(t)
    r = correction_recipe(c.correction, t)
    assert -.6 <= r.exposure_ev <= -.3
    assert r.curve.is_identity()
    assert correction_recipe(CorrectionHint(exposure_ev=0), t).exposure_ev == 0

    # Final-pixel assessment must detect a look that undoes scalar correction.
    from PIL import Image
    from superphoto.color import linear_to_srgb
    from superphoto.lightroom_qa import assess_lightroom_render
    before, after = tmp_path/'before.png', tmp_path/'after.png'
    Image.fromarray(np.round(linear_to_srgb(rgb)*255).astype('uint8')).save(before)
    Image.fromarray(np.round(linear_to_srgb(rgb*2**r.exposure_ev)*255).astype('uint8')).save(after)
    unchanged = assess_lightroom_render(before, before, slot='A', exposure_recovery=True)
    assert 'exposure_unresolved' in [i.code for i in unchanged.issues]
    corrected = assess_lightroom_render(before, after, slot='A', exposure_recovery=True)
    assert corrected.accepted, corrected.to_dict()


@pytest.mark.parametrize('scene', ['neutral_high_key', 'blue_sky', 'normal_color'])
def test_bright_tone_recovery_does_not_target_every_bright_scene(tmp_path, scene):
    rgb = np.empty((120,180,3), dtype=np.float32)
    if scene == 'neutral_high_key':
        rgb[:] = [.76,.76,.76]
    elif scene == 'blue_sky':
        rgb[:] = [.12,.35,.70]
        rgb[:40] = [1,1,1]
    else:
        rgb[:] = [.42,.35,.25]
    t = analyze(rgb, tmp_path/'control.jpg')
    assert correction_recipe(local_critique(t).correction, t).exposure_ev == 0


def test_deep_low_key_gets_bounded_lift_without_reopening_ordinary_low_key():
    dark = _tech(source_kind='jpeg', mean_luma=.009, median_luma=.0022, shadow_clip=.90,
                 black_clip=.33, highlight_p95=.031, highlight_clip=.0003,
                 suggested_exposure_ev=1.1)
    hint = CorrectionHint(exposure_ev=1.1)
    r = sanitize_correction(hint, dark)
    assert .25 <= r.exposure_ev <= .45
    ordinary = dark.model_copy(update={'mean_luma':.06, 'median_luma':.04, 'highlight_p95':.7})
    assert sanitize_correction(hint, ordinary).exposure_ev <= .06
    assert sanitize_correction(CorrectionHint(exposure_ev=0), dark).exposure_ev == 0
    assert local_critique(dark).correction.exposure_ev == r.exposure_ev
    assert '+0.45 EV' in local_critique(dark).summary
    # Explicit locks remain authoritative at the execution-plan boundary.
    from superphoto.preference import honor_locked_controls
    assert honor_locked_controls(r.as_recipe(), EditIntent(locked_controls=['exposure_ev'])).exposure_ev == 0
