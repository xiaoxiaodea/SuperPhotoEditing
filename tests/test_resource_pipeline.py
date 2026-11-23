"""A generated diversity replacement must pass material rules before rendering."""
import json

import pytest

from superphoto.config import Settings
from superphoto.db import connect, init_db, insert_job
from superphoto.models import CorrectionHint, Critique, Recipe, ToneCurve
from superphoto.pipeline import _render_variants_with_lightroom
from test_celebrity_styles import _tech
from test_lightroom_pipeline import _FakeClient, _report


def test_snow_replacement_reapplies_curve_guard_and_retains_raw_correction(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, semantic_enabled=False, vision_provider='local', ollama_enabled=False)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, 'snow-job', 'snow.dng')
    root = settings.jobs_dir / 'snow-job'
    root.mkdir(parents=True)
    original = root / 'original.dng'
    original.write_bytes(b'raw source')
    technical = _tech(source_kind='raw', mean_luma=.3)
    (root / 'technical.json').write_text(technical.model_dump_json(), encoding='utf-8')
    critique = Critique(genre='landscape', subject_note='积雪覆盖的山峰', correction=CorrectionHint(exposure_ev=-.6, highlights=-.65))
    initial = Recipe(exposure_ev=-.6, highlights=-.65, curve=ToneCurve(rgb=[(0, 0), (.18, .18), (.5, .48), (.7, .66), (1, .96)]))
    sidecar = {**initial.model_dump(), 'style_workflow': {'method_id': 'preset:resource_snow'}}
    variants = [{'id': 'snow-A', 'slot': 'A', 'look_id': 'resource_snow', 'title': '雪景', 'why': '保留雪面', 'recipe_json': json.dumps(sidecar), 'filename': 'variant_A.jpg'}]
    monkeypatch.setattr('superphoto.lightroom.get_lightroom_client', lambda *_: _FakeClient())

    def baseline(client, *, original, output_dir, export_edge):
        output_dir.mkdir(parents=True)
        export = output_dir / 'baseline.jpg'
        export.write_bytes(b'baseline')
        return {'export': export, 'metadata': {}}

    seen = []

    def render(client, *, original, recipe, output_dir, **kwargs):
        seen.append(recipe.model_copy(deep=True))
        assert dict(recipe.curve.rgb)[.7] >= .66 - 1e-9
        assert recipe.exposure_ev == -.6 and recipe.highlights == -.65
        output_dir.mkdir(parents=True)
        export = output_dir / 'render.jpg'
        export.write_bytes(b'output')
        return {'export': export, 'settings': {}, 'develop_before': {}, 'develop_after': {}}

    decisions = iter([(False, 2.0), (True, 8.0)])
    monkeypatch.setattr('superphoto.pipeline.export_lightroom_baseline', baseline)
    monkeypatch.setattr('superphoto.pipeline.apply_recipe_pass', render)
    monkeypatch.setattr('superphoto.pipeline.assess_lightroom_render', lambda *args, **kwargs: _report())
    monkeypatch.setattr('superphoto.pipeline.is_distinct_candidate', lambda *args, **kwargs: next(decisions))
    try:
        _render_variants_with_lightroom(settings, conn, 'snow-job', original, root, variants, critique=critique)
        assert len(seen) == 2
        audit = json.loads((root / 'lightroom-audits/variant_A.json').read_text(encoding='utf-8'))
        adjustments = audit['rounds'][-1]['resource_adjustments']
        assert any(item['lesson_id'] == 'snow_luminous' for item in adjustments)
    finally:
        conn.close()
