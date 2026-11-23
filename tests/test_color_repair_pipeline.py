import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from superphoto.db import get_variants
from superphoto.models import Recipe, ToneCurve
from superphoto.pipeline import _render_variants_with_lightroom
from test_lightroom_pipeline import _report, _similar_candidate_job


@pytest.mark.parametrize('slot', list('ABC'))
def test_color_failure_repairs_effective_design_on_fresh_source_and_delivers_matching_audit(tmp_path, monkeypatch, slot):
    import superphoto.pipeline as pipeline
    args = _similar_candidate_job(tmp_path, monkeypatch, [8.] * 4)
    settings, conn, job_id, original, root, variants = args
    settings.visual_review_enabled = False
    settings.finishing_review_enabled = False
    requested = Recipe(exposure_ev=-.55, contrast=.16,
        curve=ToneCurve(rgb=[(0, 0), (.5, .43), (1, 1)], r=[(0, 0), (.5, .46), (1, 1)]))
    # A stage already rolled back to a milder RGB curve: repair the actual
    # effective design instead of resurrecting the requested one.
    effective = requested.model_copy(deep=True)
    effective.curve.rgb = [(0, 0), (.5, .47), (1, 1)]
    for row in variants:
        row['recipe_json'] = requested.model_dump_json()
    fake_pass = pipeline.apply_recipe_pass
    calls = []
    def render(client, *, original, recipe, output_dir, **kwargs):
        result = fake_pass(client, original=original, recipe=recipe, output_dir=output_dir, **kwargs)
        if output_dir.parent.name == slot:
            calls.append((original, recipe.model_copy(deep=True)))
            if output_dir.name == 'round-0':
                result['effective_recipe'] = effective.model_dump()
        return result
    def quality(before, after, **kwargs):
        assert kwargs['natural_color']
        rejected = after.parent.parent.name == slot and after.parent.name == 'round-0'
        return _report(accepted=not rejected, issues=[SimpleNamespace(code='neutral_color_cast',
            note='new cast', severity=1, hard=True, advisory=False)] if rejected else [])
    monkeypatch.setattr(pipeline, 'apply_recipe_pass', render)
    monkeypatch.setattr(pipeline, 'assess_lightroom_render', quality)
    try:
        _render_variants_with_lightroom(*args)
        assert len(calls) == 2 and calls[0][0] != calls[1][0]
        assert calls[1][1].exposure_ev == -.55 and calls[1][1].contrast == .16
        assert calls[1][1].curve.rgb == effective.curve.rgb
        audit = json.loads((root/f'lightroom-audits/variant_{slot}.json').read_text(encoding='utf-8'))
        assert audit['rounds'][1]['mode'] == 'neutral-color-repair'
        assert audit['rounds'][1]['color_style_weight'] == .5
        stored = next(row for row in get_variants(conn, job_id) if row['slot'] == slot)
        assert Recipe.model_validate_json(stored['recipe_json']) == calls[1][1]
        assert Recipe.model_validate(audit['accepted_recipe']) == calls[1][1]
        assert (root/f'variant_{slot}.jpg').read_bytes() == Path(audit['rounds'][1]['export']).read_bytes()
    finally:
        conn.close()


def test_explicit_creative_intent_reaches_pixel_gate(tmp_path, monkeypatch):
    args = _similar_candidate_job(tmp_path, monkeypatch, [8.] * 4)
    settings, conn, _, _, _, variants = args
    settings.visual_review_enabled = False
    for row in variants:
        row['recipe_json'] = json.dumps({**Recipe().model_dump(), 'intent': {'natural_color': False}})
    seen = []
    def quality(before, after, **kwargs):
        seen.append(kwargs['natural_color'])
        return _report()
    monkeypatch.setattr('superphoto.pipeline.assess_lightroom_render', quality)
    try:
        _render_variants_with_lightroom(*args)
        assert seen == [False, False, False]
    finally:
        conn.close()


def test_mixed_darkness_and_cast_repairs_exposure_before_color(tmp_path, monkeypatch):
    from superphoto.lightroom_qa import LightroomQualityIssue
    args = _similar_candidate_job(tmp_path, monkeypatch, [8.] * 4)
    settings, conn, job_id, _, root, variants = args
    settings.visual_review_enabled = False
    settings.lightroom_qa_max_refinements = 2
    for row in variants:
        row['recipe_json'] = Recipe(exposure_ev=-.55).model_dump_json()
    color = LightroomQualityIssue('neutral_color_cast', 1, 'new cast')
    def quality(before, after, **kwargs):
        if after.parent.parent.name == 'B':
            if after.parent.name == 'round-0':
                return _report(accepted=False, issues=[color, LightroomQualityIssue('too_dark', 1, 'dark')])
            if after.parent.name == 'round-1':
                return _report(accepted=False, issues=[color])
        return _report()
    monkeypatch.setattr('superphoto.pipeline.assess_lightroom_render', quality)
    try:
        _render_variants_with_lightroom(*args)
        audit = json.loads((root/'lightroom-audits/variant_B.json').read_text(encoding='utf-8'))
        assert [r['mode'] for r in audit['rounds']] == ['initial', 'refine', 'neutral-color-repair']
        assert audit['accepted_recipe']['exposure_ev'] == pytest.approx(-.45)
    finally:
        conn.close()
