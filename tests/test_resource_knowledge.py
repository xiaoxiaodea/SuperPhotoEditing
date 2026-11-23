from pathlib import Path
import json
from zipfile import ZipFile

import pytest

from superphoto.models import ArtDirection, Critique, EditIntent, MaskAdjustment, Recipe, ToneCurve
from superphoto.resource_knowledge import (
    RESOURCE_LOOKS, apply_resource_guards, resource_context, resource_scene_flags,
)
from superphoto.resource_materials import inspect_xmp, inventory_resources
from test_celebrity_styles import _tech


def xmp(fields='', body=''):
    return f'''<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" {fields}>{body}</rdf:Description></rdf:RDF></x:xmpmeta>'''.encode()


def test_inventory_deduplicates_files_but_never_activates_presets(tmp_path):
    payload = xmp('crs:Exposure2012="1.1" crs:Temperature="5300"')
    (tmp_path / 'look.xmp').write_bytes(payload)
    with ZipFile(tmp_path / 'look.zip', 'w') as archive:
        archive.writestr('look.xmp', payload)
        archive.writestr('__MACOSX/._look.xmp', b'not XML')
        archive.writestr('../escape.xmp', payload)
    result = inventory_resources(tmp_path)
    assert result['summary']['xmp_entries'] == 2
    assert result['summary']['unique_xmp_bytes'] == 1
    assert result['summary']['duplicate_xmp_entries'] == 1
    assert not result['summary']['auto_registered']
    assert result['errors'] == [{'source': 'look.zip!../escape.xmp', 'error': 'unsafe_member_path'}]
    assert result['presets'][0]['values']['Exposure2012'] == 1.1
    assert not (tmp_path.parent / 'escape.xmp').exists()


def test_corrupt_archive_is_reported_without_losing_other_materials(tmp_path):
    (tmp_path / 'broken.zip').write_bytes(b'not a zip')
    (tmp_path / 'valid.xmp').write_bytes(xmp('crs:Saturation="0"'))
    result = inventory_resources(tmp_path)
    assert result['summary']['xmp_entries'] == 1
    assert result['errors'] == [{'source': 'broken.zip', 'error': 'BadZipFile'}]


def test_inspection_reads_top_level_fields_without_profile_contamination():
    data = inspect_xmp(xmp('crs:Saturation="0"', '''<crs:Dehaze>12</crs:Dehaze><crs:Look><rdf:Description crs:Saturation="90" /></crs:Look>'''), 'preset.xmp', 'preset.xmp')
    assert data['values']['Saturation'] == 0
    assert data['values']['Dehaze'] == 12
    assert 'camera_or_look_profile' in data['unsupported']


def test_original_mask_and_calibration_are_reported_as_untranslated():
    data = inspect_xmp(xmp('crs:RedHue="10"', '<crs:MaskGroupBasedCorrections><rdf:Seq /></crs:MaskGroupBasedCorrections>'), 'module.xmp', 'module.xmp')
    assert data['role'] == 'mask'
    assert set(data['unsupported']) == {'camera_calibration', 'embedded_local_masks'}


@pytest.mark.parametrize('payload', [b'<!DOCTYPE x [<!ENTITY e "value">]><x/>', b'x' * (2 * 1024 * 1024 + 1)], ids=['entity', 'oversize'])
def test_inspection_rejects_entity_documents_and_oversize(payload):
    with pytest.raises(ValueError):
        inspect_xmp(payload, 'bad.xmp', 'bad.xmp')


def test_bundled_lessons_are_source_bound_and_scene_filtered():
    portrait = resource_context(Critique(genre='portrait', subject_note='窗边人像'), _tech())
    ids = {item['id'] for item in portrait['lessons']}
    assert 'portrait_material' in ids
    assert 'night_light_hierarchy' not in ids
    assert 'snow_luminous' not in ids
    assert len(portrait['bundle_sha256']) == 64
    assert portrait['coverage']['video_audio_transcribed'] is False
    assert portrait['coverage']['duplicate_xmp_entries'] > 0
    assert 1 <= len(portrait['review_checks']) <= 4
    for lesson in portrait['lessons']:
        for ref in lesson['evidence']:
            assert len(portrait['sources'][ref['source_id']]['sha256']) == 64
            assert ref['locator'] and ref['observation']


def test_scene_evidence_does_not_turn_bright_pixels_or_low_luma_into_snow_or_night():
    flags = resource_scene_flags(Critique(genre='architecture', subject_note='湖边白色建筑与远山积雪'), _tech(mean_luma=.05, skin_fraction=.6))
    assert not flags['snow'] and not flags['night'] and not flags['portrait']
    assert not resource_scene_flags(Critique(genre='landscape', subject_note='无雾的山峰'), _tech(std_luma=.05))['fog']


@pytest.mark.parametrize('description,flag', [
    ('No snow on green hills', 'snow'), ('没有积雪的山峰', 'snow'),
    ('not a night scene', 'night'), ('日光风景而非夜景', 'night'),
    ('mountains without mist', 'fog'), ('没有薄雾的晴天山峰', 'fog'),
    ('mountains without fog', 'fog'), ('not foggy mountains', 'fog'),
])
def test_negated_scene_evidence_does_not_activate_guards(description, flag):
    c = Critique(genre='landscape', subject_note=description)
    assert not resource_scene_flags(c, _tech())[flag]
    original = Recipe(contrast=.2, clarity=.12, dehaze=.08)
    assert apply_resource_guards(original, c, _tech()) == original


def test_resource_scene_gates_override_model_and_many_likes():
    from superphoto.style_selection import rank_presets
    c = Critique(genre='architecture', subject_note='日光下白色建筑', source='openai', looks=['resource_snow', 'resource_night'])
    ranked, excluded = rank_presets(_tech(), c, feedback={key: {'like': 100} for key in RESOURCE_LOOKS})
    assert 'resource_snow' in excluded and 'resource_night' in excluded
    assert not {'resource_snow', 'resource_night'} & {item.look_id for item in ranked}


def test_snow_and_night_knowledge_reach_real_selection():
    from superphoto.celebrity_styles import build_style_decision
    snow = build_style_decision(_tech(mean_luma=.4), Critique(genre='landscape', subject_note='积雪覆盖的山峰和雪地'))
    night = build_style_decision(_tech(mean_luma=.08), Critique(genre='night', subject_note='夜街灯光'))
    assert 'resource_snow' in snow.look_ids
    # Compatible options compete on score and diversity; none owns a slot.
    assert 'resource_night' in {item.look_id for item in night.ranked_candidates}
    liked = build_style_decision(_tech(mean_luma=.08), Critique(genre='night', subject_note='夜街灯光'), feedback={'resource_night': {'like': 3}})
    assert 'resource_night' in liked.look_ids
    assert 'resource_night' in snow.excluded
    assert 'resource_snow' in night.excluded


def test_fog_guard_retains_confirmed_raw_recovery_and_records_actual_changes():
    c = Critique(genre='landscape', subject_note='雾中山林', intent=EditIntent(preserve_atmosphere=True))
    mask = MaskAdjustment(target='sky', exposure_ev=-1.5, contrast=.35, whites=.65, dehaze=.12)
    original = Recipe(exposure_ev=-.3, highlights=-.6, contrast=.28, clarity=.15, dehaze=.08, masks=[mask])
    audit = []
    result = apply_resource_guards(original, c, _tech(source_kind='raw'), audit=audit)
    assert result.dehaze <= .025 and result.clarity <= .06 and result.contrast <= .12
    assert result.exposure_ev == original.exposure_ev and result.highlights == original.highlights
    assert result.masks[0].exposure_ev == -1.5 and result.masks[0].whites == .65
    assert result.masks[0].contrast == .35
    assert original.dehaze == .08 and original.masks[0].dehaze == .12
    assert audit and all(item['lesson_id'] == 'fog_depth' for item in audit)
    assert apply_resource_guards(result, c, _tech(source_kind='raw')).model_dump() == result.model_dump()


def test_flat_nonfog_scene_does_not_get_fog_constraints():
    original = Recipe(contrast=.25, clarity=.14, dehaze=.08)
    c = Critique(genre='landscape', subject_note='晴天山峰')
    assert apply_resource_guards(original, c, _tech(std_luma=.03)) == original


def test_portrait_global_and_subject_material_guards_are_not_skin_masks():
    original = Recipe(clarity=.15, texture=-.18, masks=[MaskAdjustment(target='subject', texture=-.2), MaskAdjustment(target='background', texture=.2)])
    c = Critique(genre='portrait')
    result = apply_resource_guards(original, c, _tech())
    assert result.clarity <= .06 and result.texture >= -.08
    assert result.masks[0].texture >= -.08
    assert result.masks[1].texture == .2
    assert {mask.target for mask in result.masks} == {'subject', 'background'}


def test_snow_guard_does_not_modify_correction_or_force_a_palette():
    original = Recipe(exposure_ev=-.4, highlights=-.7, curve=ToneCurve(rgb=[(0, 0), (.5, .48), (.8, .71), (1, .92)]))
    c = Critique(genre='landscape', subject_note='白雪覆盖的山峰')
    result = apply_resource_guards(original, c, _tech(source_kind='raw'))
    assert result.curve.rgb[-1][1] >= .96
    assert result.exposure_ev == -.4 and result.highlights == -.7
    c.intent.natural_color = False
    assert apply_resource_guards(original, c, _tech()) == original


def test_snow_floor_covers_interpolation_after_a_knot_below_the_boundary():
    import numpy as np
    recipe = Recipe(curve=ToneCurve(rgb=[(0, 0), (.18, .18), (.5, .4), (.69, .59), (1, .96)]))
    c = Critique(genre='landscape', subject_note='snow-covered mountains')
    guarded = apply_resource_guards(recipe, c, _tech())
    points = np.array(guarded.curve.rgb)
    inputs = np.linspace(.7, 1, 101)
    assert np.all(np.interp(inputs, *points.T) >= inputs - .04000001)
    assert np.all(np.diff(points[:, 1]) >= 0)
    assert apply_resource_guards(guarded, c, _tech()) == guarded


def test_replacement_guard_limits_style_without_clamping_primary_recovery():
    primary = Recipe(exposure_ev=-.6, highlights=-.65, contrast=.08, clarity=.03, dehaze=.02)
    c = Critique(genre='landscape', subject_note='雾中树林')
    recipe = primary.merged(Recipe(contrast=.2, clarity=.12, dehaze=.1))
    guarded = apply_resource_guards(recipe, c, _tech(), primary=primary)
    assert guarded.contrast == pytest.approx(.08 + .12)
    assert guarded.clarity == pytest.approx(.03 + .06)
    assert guarded.dehaze == pytest.approx(.02 + .025)
    assert guarded.exposure_ev == -.6 and guarded.highlights == -.65


def test_plan_without_directions_still_guards_portrait_material():
    from superphoto.models import CorrectionHint
    from superphoto.preference import build_plans
    c = Critique(genre='portrait', subject_note='窗边人像', correction=CorrectionHint(exposure_ev=-.3, highlights=-.5))
    plan = build_plans(['editorial_crisp'], c, _tech(top_third_blue=0))[0]
    assert plan.recipe.clarity <= .06
    assert plan.recipe.exposure_ev == plan.primary.exposure_ev == -.3
    assert plan.recipe.highlights == plan.primary.highlights == -.5


def test_local_model_fallback_keeps_executable_recipe_provenance_and_native_wheels(tmp_path):
    from superphoto.config import Settings
    from superphoto.directions import design_directions
    from superphoto.lightroom import recipe_to_develop_settings
    c = Critique(genre='night', subject_note='夜街灯光')
    audit = tmp_path / 'directions.json'
    directions = design_directions(Settings(vision_provider='local', ollama_enabled=False), tmp_path / 'unused', c, _tech(mean_luma=.08), audit_path=audit, look_ids=['resource_night', 'celebrity_low_sat_shinagawa', 'wong_kar_wai'])
    assert directions[0].look_id == 'resource_night'
    native = recipe_to_develop_settings(directions[0].recipe)
    assert native['SplitToningHighlightSaturation'] > 0
    assert native['SplitToningShadowSaturation'] > 0
    saved = json.loads(audit.read_text(encoding='utf-8'))
    assert saved['knowledge']['resource_materials']['bundle_sha256']
    assert any(item['id'] == 'night_light_hierarchy' for item in saved['knowledge']['resource_materials']['lessons'])


def test_model_planner_receives_only_applicable_lessons_and_keeps_explicit_zero(tmp_path, monkeypatch):
    from superphoto.config import Settings
    from superphoto.directions import design_directions
    preview = tmp_path / 'preview.jpg'
    preview.write_bytes(b'preview')
    calls = []
    monkeypatch.setattr('superphoto.directions.probe', lambda _: {'ready': True, 'resolved_model': 'vision'})

    def chat(settings, model, prompt, images, **kwargs):
        calls.append(prompt)
        return json.dumps({slot: {'title': '保留雪面' + slot, 'goal': '保留明亮的雪地与原有光色', 'moves': ['轻整颜色'], 'recipe': {'hsl': {'blue': {'sat': 0}}}} for slot in 'ABC'})

    monkeypatch.setattr('superphoto.directions.chat_vision', chat)
    directions = design_directions(Settings(), preview, Critique(genre='landscape', subject_note='积雪覆盖的山峰'), _tech(mean_luma=.4), look_ids=['resource_snow', 'landscape_filmic', 'celebrity_tim_northey'])
    assert len(calls) == 1 and 'snow_luminous' in calls[0]
    assert 'bundle_sha256' in calls[0] and 'night_light_hierarchy' not in calls[0]
    assert directions[0].source != 'local'
    assert directions[0].recipe.hsl['blue'].sat == 0


def test_visual_review_receives_conditional_resource_checks(tmp_path, monkeypatch):
    from PIL import Image
    from superphoto.config import Settings
    from superphoto.review import review_render
    before, after = tmp_path / 'before.jpg', tmp_path / 'after.jpg'
    Image.new('RGB', (32, 32), (110, 120, 130)).save(before)
    Image.new('RGB', (32, 32), (120, 130, 140)).save(after)
    context = resource_context(Critique(genre='landscape', subject_note='积雪覆盖的山峰'), _tech())
    intent = EditIntent(resource_lesson_ids=[x['id'] for x in context['lessons']], resource_checks=context['review_checks'])
    prompts = []
    monkeypatch.setattr('superphoto.review.probe', lambda _: {'ready': True, 'resolved_model': 'vision'})

    def chat(settings, model, prompt, images, **kwargs):
        prompts.append(prompt)
        return json.dumps({'decision': 'better', 'confidence': .9, 'notes': ['亮部层次可辨'], 'damage': []})

    monkeypatch.setattr('superphoto.review.chat_vision', chat)
    result = review_render(Settings(), before, after, intent, 'final')
    assert result.decision == 'better'
    assert len(prompts) == 1 and 'conditional lessons' in prompts[0]
    assert context['review_checks'][0] in prompts[0]


def test_adjustment_audit_survives_direction_roundtrip():
    from superphoto.directions import bound_direction
    c = Critique(genre='portrait')
    direction = ArtDirection(slot='A', title='人像', goal='保留材质', recipe=Recipe(texture=.18))
    bounded = bound_direction(direction, c, _tech())
    assert bounded.resource_adjustments
    restored = ArtDirection.model_validate_json(bounded.model_dump_json())
    assert restored.resource_adjustments == bounded.resource_adjustments
    assert bound_direction(restored, c, _tech()).resource_adjustments == restored.resource_adjustments
