from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from superphoto.analysis import analyze, correction_from_technical
from superphoto.color import linear_to_srgb, srgb_to_linear
from superphoto.critique import local_critique
from superphoto.exposure import developed_overexposure, preserve_exposure_correction
from superphoto.models import ArtDirection, Critique, EditIntent, Recipe, ToneCurve
from superphoto.preference import build_plans, correction_recipe, sanitize_correction
from superphoto.lightroom_qa import assess_lightroom_render


def measured(tmp_path, foreground=(.42, .52, .23)):
    linear = np.full((120, 180, 3), foreground, dtype=np.float32)
    linear[:36] = 1
    # Include textured midtones, rather than measuring only a single color.
    linear[36:] *= np.linspace(.65, 1, 84)[:, None, None]
    path = tmp_path / 'source.jpg'
    Image.fromarray(np.round(linear_to_srgb(linear) * 255).astype('uint8')).save(path, quality=100)
    return path, analyze(linear, path)


def assert_recovery_preserved(recipe, primary, tech):
    xs = np.linspace(0, 1, 101)
    base = np.interp(xs, *np.array(primary.curve.rgb).T)
    actual = np.interp(xs, *np.array(recipe.curve.rgb).T)
    assert np.max(np.abs(actual - base)) <= .086
    assert recipe.curve.rgb[-1] == (1, 1)
    assert np.all(np.diff(actual) >= 0)
    again = preserve_exposure_correction(recipe, primary, tech)
    assert again.curve == recipe.curve


def test_bright_usable_content_gets_shared_white_preserving_correction(tmp_path):
    _, tech = measured(tmp_path)
    assert developed_overexposure(tech)
    hint = correction_from_technical(tech)
    assert hint.exposure_ev < -.3
    assert sanitize_correction(hint, tech).exposure_ev == hint.exposure_ev
    primary = correction_recipe(hint, tech)
    assert primary.exposure_ev == 0
    assert primary.highlights == primary.whites == 0
    assert primary.curve.rgb[-1] == (1, 1)
    assert np.interp(.5, *np.array(primary.curve.rgb).T) < .43
    critique = Critique(genre='landscape', correction=hint)
    for directions in [[], [ArtDirection(slot='A', title='style', goal='style', look_id='faithful',
            recipe=Recipe(exposure_ev=.2, highlights=-.3,
                curve=ToneCurve(rgb=[(0, .05), (.5, .6), (1, .90)])))]]:
        critique.directions = directions
        plan = build_plans(['faithful'], critique, tech)[0]
        assert_recovery_preserved(plan.recipe, primary, tech)
        assert plan.recipe.exposure_ev == 0
        from superphoto.qa import QAReport, refine_plan
        report = QAReport(score=60, presence=50, safety=50, finish=50, mae=.1, too_strong=True, too_weak=False, notes=[])
        retried = refine_plan(plan, report, critique, tech, 0)
        assert_recovery_preserved(retried.recipe, primary, tech)
        repaired = preserve_exposure_correction(Recipe(curve=ToneCurve(rgb=[(0, 0), (1, .8)])), primary, tech)
        assert_recovery_preserved(repaired, primary, tech)


@pytest.mark.parametrize('foreground', [(.07, .12, .04), (.55, .55, .55), (.01, .015, .005)])
def test_white_sky_or_neutral_high_key_does_not_prove_overexposure(tmp_path, foreground):
    _, tech = measured(tmp_path, foreground)
    assert not developed_overexposure(tech)
    assert tech.suggested_exposure_ev == 0
    assert correction_recipe(correction_from_technical(tech), tech).curve.is_identity()


def test_model_zero_locks_and_old_reports_do_not_enable_recovery(tmp_path):
    _, tech = measured(tmp_path)
    hint = correction_from_technical(tech)
    assert correction_recipe(hint.model_copy(update={'exposure_ev': 0}), tech).curve.is_identity()
    assert correction_recipe(hint, tech, EditIntent(locked_controls=['exposure_ev'])).curve.is_identity()
    old = tech.model_copy(update={'unclipped_median_luma': None})
    assert sanitize_correction(hint, old).exposure_ev == -.3
    raw = tech.model_copy(update={'source_kind': 'raw'})
    assert correction_recipe(hint, raw).curve.is_identity()


def test_qa_checks_actual_correction_and_does_not_reward_grey_white_plateau(tmp_path):
    source, tech = measured(tmp_path)
    primary = correction_recipe(correction_from_technical(tech), tech)
    rgb = np.asarray(Image.open(source).convert('RGB'), dtype=np.float32) / 255
    x, y = np.array(primary.curve.rgb).T
    corrected = tmp_path / 'corrected.png'
    Image.fromarray(np.round(np.interp(rgb, x, y) * 255).astype('uint8')).save(corrected)
    unchanged = assess_lightroom_render(source, source, slot='A', exposure_recovery=True)
    assert not unchanged.accepted
    assert 'exposure_unresolved' in [i.code for i in unchanged.issues]
    report = assess_lightroom_render(source, corrected, slot='A', exposure_recovery=True)
    assert report.accepted, report.to_dict()
    grey = tmp_path / 'grey.png'
    Image.fromarray(np.round(linear_to_srgb(srgb_to_linear(rgb) * 2 ** -.8) * 255).astype('uint8')).save(grey)
    report = assess_lightroom_render(source, grey, slot='A', exposure_recovery=True)
    assert not report.accepted
    assert 'clipped_whites_greyed' in [i.code for i in report.issues]
    black = tmp_path / 'black.png'
    Image.fromarray(np.zeros_like(rgb, dtype='uint8')).save(black)
    assert not assess_lightroom_render(source, black, slot='A', exposure_recovery=True).accepted


@pytest.mark.parametrize('goal_rejected', [False, True])
def test_pipeline_replacement_retains_exposure_curve_and_checks_goal(tmp_path, monkeypatch, goal_rejected):
    import json
    from superphoto.config import Settings
    from superphoto.db import connect, init_db, insert_job, get_variants
    from superphoto.pipeline import _render_variants_with_lightroom
    from test_lightroom_pipeline import _FakeClient, _report

    source, tech = measured(tmp_path)
    settings = Settings(data_dir=tmp_path / 'data', semantic_enabled=False,
                        vision_provider='local', ollama_enabled=False, finishing_review_enabled=False)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, 'exposure-job', 'source.jpg')
    root = settings.jobs_dir / 'exposure-job'
    root.mkdir(parents=True)
    (root / 'technical.json').write_text(tech.model_dump_json(), encoding='utf-8')
    critique = Critique(genre='landscape', correction=correction_from_technical(tech))
    primary = correction_recipe(critique.correction, tech)
    sidecar = {**Recipe(highlights=-.3).model_dump(), 'style_workflow': {'method_id': 'preset:faithful'}}
    variants = [dict(id='exposure-A', slot='A', look_id='faithful', title='A', why='test',
                     recipe_json=json.dumps(sidecar), filename='variant_A.jpg')]
    monkeypatch.setattr('superphoto.lightroom.get_lightroom_client', lambda *_: _FakeClient())
    seen = []

    def render(client, *, recipe, output_dir, **kwargs):
        seen.append(recipe.model_copy(deep=True))
        assert_recovery_preserved(recipe, primary, tech)
        assert recipe.exposure_ev == 0
        assert recipe.highlights >= -.12 and recipe.whites >= -.08
        output_dir.mkdir(parents=True)
        export = output_dir / 'render.jpg'
        export.write_bytes(b'output')
        return dict(export=export, settings={}, develop_before={}, develop_after={})

    qa_calls = []

    def quality(*args, **kwargs):
        from types import SimpleNamespace
        assert kwargs['exposure_recovery'] is True
        qa_calls.append(args)
        if goal_rejected and len(qa_calls) == 1:
            return _report(accepted=False, issues=[SimpleNamespace(code='clipped_whites_greyed', note='grey white plateau')])
        return _report()

    decisions = iter([(True, 8.0)] if goal_rejected else [(False, 2.0), (True, 8.0)])
    monkeypatch.setattr('superphoto.pipeline.apply_recipe_pass', render)
    monkeypatch.setattr('superphoto.pipeline.assess_lightroom_render', quality)
    monkeypatch.setattr('superphoto.pipeline.is_distinct_candidate', lambda *a, **k: next(decisions))
    monkeypatch.setattr('superphoto.pipeline.separate_candidate_recipe',
                        lambda *a, **k: Recipe(exposure_ev=.3, curve=ToneCurve(rgb=[(0, 0), (1, .8)])))
    try:
        _render_variants_with_lightroom(settings, conn, 'exposure-job', source, root, variants,
                                       baseline_result={'export': source}, critique=critique)
        assert len(seen) == 2
        saved = Recipe.model_validate_json(get_variants(conn, 'exposure-job')[0]['recipe_json'])
        assert_recovery_preserved(saved, primary, tech)
    finally:
        conn.close()


def test_low_iso_foliage_is_not_called_high_iso_noise(tmp_path):
    _, tech = measured(tmp_path)
    tech.noise = .11
    tech.exif['ISOSpeedRatings'] = '100'
    critique = local_critique(tech)
    assert not any(i.dim == 'noise' for i in critique.issues)
    assert any('无法恢复' in i.note for i in critique.issues)
