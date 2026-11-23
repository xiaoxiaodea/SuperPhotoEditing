"""Ensure a measured improvement goal reaches the real delivery QA boundary."""
import pytest

from superphoto.critique import local_critique
from superphoto.db import get_variants
from superphoto.models import Recipe
from superphoto.pipeline import _render_variants_with_lightroom
from test_lightroom_pipeline import _report, _similar_candidate_job
from test_midtones_improvement import dark_developed


@pytest.mark.parametrize('locked',[False,True])
def test_common_lift_goal_reaches_each_final_render_and_respects_lock(tmp_path,monkeypatch,locked):
    settings,conn,job_id,original,root,variants=_similar_candidate_job(tmp_path,monkeypatch,[8.]*4)
    settings.visual_review_enabled=False
    settings.finishing_review_enabled=False
    _,tech=dark_developed(tmp_path)
    critique=local_critique(tech)
    if locked:
        critique.intent.locked_controls=['exposure_ev']
    (root/'technical.json').write_text(tech.model_dump_json(),encoding='utf-8')
    for row in variants:
        row['recipe_json']=Recipe(exposure_ev=critique.correction.exposure_ev).model_dump_json()
    seen=[]
    def quality(before,after,**kwargs):
        if 'exposure_lift_ev' in kwargs:
            seen.append((kwargs['slot'],kwargs['exposure_lift_ev']))
        return _report()
    monkeypatch.setattr('superphoto.pipeline.assess_lightroom_render',quality)
    try:
        _render_variants_with_lightroom(settings,conn,job_id,original,root,variants,critique=critique)
        assert [slot for slot,_ in seen]==list('ABC')
        assert all(goal==pytest.approx(0 if locked else critique.correction.exposure_ev) for _,goal in seen)
        rows=get_variants(conn,job_id)
        assert len(rows)==3
        for row in rows:
            recipe=Recipe.model_validate_json(row['recipe_json'])
            assert recipe.exposure_ev==pytest.approx(0 if locked else critique.correction.exposure_ev)
    finally:
        conn.close()
