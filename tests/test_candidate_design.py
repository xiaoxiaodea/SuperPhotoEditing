import json

import pytest

from superphoto.candidate_design import alternative_preset_plan
from superphoto.db import get_variants
from superphoto.models import Critique, StyleDecision, StyleMatch
from superphoto.pipeline import _render_variants_with_lightroom
from test_developed_exposure import measured
from test_lightroom_pipeline import _similar_candidate_job


def ranking():
    return [StyleMatch(look_id='resource_snow', title='Snow', match_score=1, reason='snow only'),
            StyleMatch(look_id='celebrity_german_maurice', title='Vintage', match_score=.6, reason='eligible landscape')]


def test_alternatives_recheck_scene_eligibility_and_preserve_requested_slot(tmp_path):
    _, tech = measured(tmp_path, (.07, .12, .04))
    critique = Critique(genre='landscape', subject_note='绿树与池塘')
    plan = alternative_preset_plan('C', ranking(), critique, tech,
                                  excluded_ids=set(), accepted_recipes=[])
    assert plan.look_id == 'celebrity_german_maurice'
    assert plan.slot == 'C'
    assert plan.method_id == 'preset:celebrity_german_maurice'
    assert alternative_preset_plan('C', ranking(), critique, tech,
        excluded_ids={'celebrity_german_maurice'}, accepted_recipes=[]) is None


@pytest.mark.parametrize('alternative_safe', [True, False])
def test_rendered_alternative_updates_identity_only_when_selected(tmp_path, monkeypatch, alternative_safe):
    _, tech = measured(tmp_path, (.07, .12, .04))
    args = _similar_candidate_job(tmp_path, monkeypatch, [1, 2, 8, 8],
                                  invalid_rounds=() if alternative_safe else (2, 3))
    settings, conn, job_id, original, root, variants = args
    settings.semantic_enabled = False
    settings.finishing_review_enabled = False
    critique = Critique(genre='landscape', subject_note='绿树与池塘')
    (root / 'technical.json').write_text(tech.model_dump_json(), encoding='utf-8')
    decision = StyleDecision(tone='', color='', purpose='', route_id='', route_title='', route_reason='',
                             ranked_candidates=ranking())
    (root / 'style-decision.json').write_text(decision.model_dump_json(), encoding='utf-8')
    for row, lid in zip(variants, ['celebrity_forest_michael_kagerer', 'landscape_filmic', 'celebrity_tim_northey']):
        row['look_id'] = lid
        payload = json.loads(row['recipe_json'])
        payload['style_workflow'] = {'method_id': 'preset:' + lid, 'preset_id': lid}
        row['recipe_json'] = json.dumps(payload)
    try:
        _render_variants_with_lightroom(*args, critique=critique)
        saved = get_variants(conn, job_id)
        assert len(saved) == 3
        b = next(row for row in saved if row['slot'] == 'B')
        payload = json.loads(b['recipe_json'])
        audit = json.loads((root / 'lightroom-audits/variant_B.json').read_text(encoding='utf-8'))
        replacement = audit['rounds'][2]
        assert replacement['mode'] == 'preset-direction-replacement'
        assert replacement['design']['look_id'] == 'celebrity_german_maurice'
        if alternative_safe:
            assert b['look_id'] == 'celebrity_german_maurice'
            assert payload['style_workflow']['preset_id'] == b['look_id']
            assert payload['style_workflow']['method_id'] == 'preset:' + b['look_id']
            assert payload['style_workflow']['requested_preset_id'] == 'landscape_filmic'
            assert payload['candidate_selection']['distinct']
            assert payload['candidate_selection']['selected_round'] == 2
        else:
            assert b['look_id'] == 'landscape_filmic'
            assert payload['style_workflow']['preset_id'] == 'landscape_filmic'
            assert not payload['candidate_selection']['distinct']
            assert payload['candidate_selection']['selected_round'] == 0
        chosen = audit['rounds'][payload['candidate_selection']['selected_round']]
        from pathlib import Path
        assert (root / 'variant_B.jpg').read_bytes() == Path(chosen['export']).read_bytes()
    finally:
        conn.close()
