from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superphoto.config import Settings
from superphoto.db import connect, get_job, get_variants, init_db, insert_job
from superphoto.models import Recipe
from superphoto.lightroom import LightroomError
from superphoto.pipeline import _render_variants_with_lightroom


class _FakeClient:
    def status(self) -> dict:
        return {"connected": True, "tools": []}


def _report(*, accepted: bool = True, score: float = 96.0, issues: list | None = None):
    return SimpleNamespace(
        accepted=accepted,
        score=score,
        issues=issues or [],
        to_dict=lambda: {"accepted": accepted, "score": score, "issues": []},
    )


def test_lightroom_renders_baseline_and_three_independent_sources(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data"
    settings = Settings(data_dir=data, lightroom_enabled=True)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job-1", "photo.jpg")
    root = settings.jobs_dir / "job-1"
    root.mkdir(parents=True, exist_ok=True)
    original = root / "original.jpg"
    original.write_bytes(b"source")
    variants = [
        {
            "id": f"id-{slot}",
            "slot": slot,
            "look_id": f"look-{slot}",
            "title": f"Look {slot}",
            "why": "reason",
            "recipe_json": Recipe(exposure_ev=offset).model_dump_json(),
            "filename": f"variant_{slot}.jpg",
        }
        for slot, offset in (("A", 0.1), ("B", 0.2), ("C", 0.3))
    ]
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr("superphoto.lightroom.get_lightroom_client", lambda *_: _FakeClient())

    def fake_baseline(_client, *, original, output_dir, export_edge):
        assert original.name == "baseline.jpg"
        assert original.read_bytes() == b"source"
        output_dir.mkdir(parents=True)
        exported = output_dir / "baseline.jpg"
        exported.write_bytes(b"baseline")
        return {"export": exported, "metadata": {}}

    def fake_pass(_client, *, original, recipe, output_dir, export_edge):
        assert original.read_bytes() == b"source"
        output_dir.mkdir(parents=True)
        slot = original.stem.split("_")[1]
        exported = output_dir / f"rendered_{slot}.jpg"
        exported.write_bytes(f"lightroom-{slot}".encode())
        calls.append((slot, original))
        return {
            "export": exported,
            "settings": {"Exposure2012": recipe.exposure_ev},
            "develop_before": {},
            "develop_after": {},
            "stages": [{"stage": "technical-correction"}],
        }

    monkeypatch.setattr("superphoto.pipeline.export_lightroom_baseline", fake_baseline)
    monkeypatch.setattr("superphoto.pipeline.apply_recipe_pass", fake_pass)
    monkeypatch.setattr("superphoto.pipeline.assess_lightroom_render", lambda *args, **kwargs: _report())
    monkeypatch.setattr("superphoto.pipeline.is_distinct_candidate", lambda *args, **kwargs: (True, 8.0))

    _render_variants_with_lightroom(settings, conn, "job-1", original, root, variants)

    assert [slot for slot, _ in calls] == ["A", "B", "C"]
    assert len({path for _, path in calls}) == 3
    assert (root / "variant_A.jpg").read_bytes() == b"lightroom-A"
    assert (root / "variant_B.jpg").read_bytes() == b"lightroom-B"
    assert (root / "variant_C.jpg").read_bytes() == b"lightroom-C"
    summary = json.loads((root / "lightroom-qa.json").read_text(encoding="utf-8"))
    assert set(summary["slots"]) == {"A", "B", "C"}
    assert all(item["score"] == 96.0 for item in summary["slots"].values())
    assert all(item["stage_count"] == 1 for item in summary["slots"].values())
    saved = get_variants(conn, "job-1")
    assert all("LightCraft 技术检查通过" in row["why"] for row in saved)
    assert get_job(conn, "job-1")["step"] == "lightroom"


def test_quality_replacement_switches_to_own_light_design_once_safe(tmp_path: Path, monkeypatch) -> None:
    """After quality recovery, similarity must not mix the result back into A."""
    data = tmp_path / "data"
    settings = Settings(data_dir=data, lightroom_enabled=True, lightroom_qa_max_refinements=2,
                        lightroom_require_distinct_candidates=True)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job-3", "photo.jpg")
    root = settings.jobs_dir / "job-3"
    root.mkdir(parents=True, exist_ok=True)
    original = root / "original.jpg"
    original.write_bytes(b"source")
    variants = [
        {
            "id": f"id-{slot}",
            "slot": slot,
            "look_id": f"look-{slot}",
            "title": f"Look {slot}",
            "why": "reason",
            "recipe_json": Recipe(exposure_ev=0.1).model_dump_json(),
            "filename": f"variant_{slot}.jpg",
        }
        for slot in "ABC"
    ]

    monkeypatch.setattr("superphoto.lightroom.get_lightroom_client", lambda *_: _FakeClient())

    def fake_baseline(_client, *, original, output_dir, export_edge):
        output_dir.mkdir(parents=True)
        exported = output_dir / "baseline.jpg"
        exported.write_bytes(b"baseline")
        return {"export": exported, "metadata": {}}

    def fake_pass(_client, *, original, recipe, output_dir, export_edge):
        output_dir.mkdir(parents=True)
        exported = output_dir / "rendered.jpg"
        exported.write_bytes(b"pixels")
        return {"export": exported, "settings": {}, "develop_before": {}, "develop_after": {}}

    monkeypatch.setattr("superphoto.pipeline.export_lightroom_baseline", fake_baseline)
    monkeypatch.setattr("superphoto.pipeline.apply_recipe_pass", fake_pass)

    # A passes QA and is distinct.
    # B fails QA (shadow clipping) on all initial+refine rounds,
    # then replacement-0 passes QA but is NOT distinct (distance 1.5),
    # then replacement-1 passes QA AND is distinct (distance 5.0).
    qa_results = iter([
        _report(),                                                           # A round-0
        _report(accepted=False, score=65, issues=[SimpleNamespace(code="shadow_clipping", severity=1.0, note="clipped", hard=True)]),  # B round-0
        _report(accepted=False, score=70, issues=[SimpleNamespace(code="shadow_clipping", severity=0.8, note="clipped", hard=True)]),  # B round-1
        _report(accepted=False, score=75, issues=[SimpleNamespace(code="shadow_clipping", severity=0.6, note="clipped", hard=True)]),  # B round-2
        _report(),                                                           # B replacement-0 (QA pass)
        _report(),                                                           # B replacement-1 (QA pass)
        _report(),                                                           # C round-0
    ])

    # is_distinct_candidate is only called when report.accepted is True.
    # B rounds 0-2 have accepted=False, so no distinct calls for those.
    distinct_results = iter([
        (True, None),          # A: accepted, distinct
        (False, 1.5),          # B replacement-0: QA pass but too close to A
        (True, 5.0),           # B replacement-1: QA pass and distinct
        (True, 8.0),           # C
    ])

    monkeypatch.setattr("superphoto.pipeline.assess_lightroom_render", lambda *a, **kw: next(qa_results))
    monkeypatch.setattr("superphoto.pipeline.is_distinct_candidate", lambda *a, **kw: next(distinct_results))
    monkeypatch.setattr("superphoto.pipeline.correction_for_report", lambda report, strength: Recipe(vibrance=-0.12))

    _render_variants_with_lightroom(settings, conn, "job-3", original, root, variants)

    audit_b = json.loads((root / "lightroom-audits" / "variant_B.json").read_text(encoding="utf-8"))
    replacement_rounds = [r for r in audit_b["rounds"] if r["mode"].endswith("replacement")]
    assert len(replacement_rounds) == 2
    assert replacement_rounds[0]["candidate_distinct"] is False
    assert replacement_rounds[1]["candidate_distinct"] is True
    assert replacement_rounds[0]["mode"] == "safe-style-replacement"
    assert replacement_rounds[1]["mode"] == "distinctness-replacement"
    assert replacement_rounds[1]["style_weight"] is None
    assert audit_b["replaced"] is True


def test_lightroom_refines_failed_candidate_before_delivery(tmp_path: Path, monkeypatch) -> None:
    """A candidate that fails initial QA should be refined, and the refined result delivered."""
    data = tmp_path / "data"
    settings = Settings(data_dir=data, lightroom_enabled=True, lightroom_qa_max_refinements=2)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job-2", "photo.jpg")
    root = settings.jobs_dir / "job-2"
    root.mkdir(parents=True, exist_ok=True)
    original = root / "original.jpg"
    original.write_bytes(b"source")
    variants = [
        {
            "id": f"id-{slot}",
            "slot": slot,
            "look_id": f"look-{slot}",
            "title": f"Look {slot}",
            "why": "reason",
            "recipe_json": Recipe(exposure_ev=0.1).model_dump_json(),
            "filename": f"variant_{slot}.jpg",
        }
        for slot in "ABC"
    ]
    scores = iter([
        _report(),
        _report(accepted=False, score=65.0, issues=[SimpleNamespace(code="oversaturation", severity=1.0, note="too strong", hard=True)]),
        _report(accepted=True, score=91.0),
        _report(),
    ])

    monkeypatch.setattr("superphoto.lightroom.get_lightroom_client", lambda *_: _FakeClient())

    def fake_baseline(_client, *, original, output_dir, export_edge):
        output_dir.mkdir(parents=True)
        exported = output_dir / "baseline.jpg"
        exported.write_bytes(b"baseline")
        return {"export": exported, "metadata": {}}

    def fake_pass(_client, *, original, recipe, output_dir, export_edge):
        output_dir.mkdir(parents=True)
        exported = output_dir / "rendered.jpg"
        exported.write_bytes(b"pixels")
        return {"export": exported, "settings": {}, "develop_before": {}, "develop_after": {}}

    monkeypatch.setattr("superphoto.pipeline.export_lightroom_baseline", fake_baseline)
    monkeypatch.setattr("superphoto.pipeline.apply_recipe_pass", fake_pass)
    monkeypatch.setattr("superphoto.pipeline.assess_lightroom_render", lambda *args, **kwargs: next(scores))
    monkeypatch.setattr("superphoto.pipeline.correction_for_report", lambda report, strength: Recipe(vibrance=-0.12))
    monkeypatch.setattr("superphoto.pipeline.is_distinct_candidate", lambda *args, **kwargs: (True, 8.0))

    _render_variants_with_lightroom(settings, conn, "job-2", original, root, variants)

    audit_b = json.loads((root / "lightroom-audits" / "variant_B.json").read_text(encoding="utf-8"))
    assert audit_b["refined"] is True
    assert audit_b["accepted_score"] == 91.0
    assert len(audit_b["rounds"]) == 2
    recipe_b = Recipe.model_validate_json(get_variants(conn, "job-2")[1]["recipe_json"])
    assert recipe_b.vibrance == -0.12


def _similar_candidate_job(tmp_path, monkeypatch, distances, *, invalid_rounds=(), broken_export=False, scores=None):
    settings = Settings(data_dir=tmp_path / "data", lightroom_enabled=True,
                        lightroom_require_distinct_candidates=True, lightroom_qa_max_refinements=0)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "similar", "photo.jpg")
    root = settings.jobs_dir / "similar"
    root.mkdir(parents=True)
    original = root / "original.jpg"
    original.write_bytes(b"source")
    variants = [{"id": slot, "slot": slot, "look_id": "faithful", "title": slot, "why": "光影目标",
                 "recipe_json": Recipe(exposure_ev=.1).model_dump_json(), "filename": f"variant_{slot}.jpg"}
                for slot in "ABC"]
    monkeypatch.setattr("superphoto.pipeline._connected_lightroom", lambda _: (_FakeClient(), {"tools": []}))
    def baseline(_client, *, original, output_dir, **kwargs):
        output_dir.mkdir(parents=True)
        exported = output_dir / "baseline.jpg"
        exported.write_bytes(b"baseline")
        return {"export": exported}
    def render(_client, *, original, recipe, output_dir, **kwargs):
        if broken_export and output_dir.parent.name == "B" and output_dir.name.startswith("replacement"):
            raise LightroomError("optional export failed")
        output_dir.mkdir(parents=True)
        exported = output_dir / "rendered.jpg"
        exported.write_bytes(str(output_dir.relative_to(root)).encode())
        return {"export": exported, "settings": {"Exposure2012": recipe.exposure_ev},
                "develop_before": {}, "develop_after": {"Exposure2012": recipe.exposure_ev},
                "effective_recipe": recipe.model_dump()}
    def index(path):
        return 0 if path.parent.name == "round-0" else int(path.parent.name.rsplit("-", 1)[1]) + 1
    def quality(_before, after, **kwargs):
        if after.parent.parent.name == "B" and index(after) in invalid_rounds:
            return _report(accepted=False, score=50, issues=[SimpleNamespace(note="clipped")])
        if after.parent.parent.name == "B" and scores is not None:
            return _report(score=scores[index(after)])
        return _report()
    def distinct(after, accepted, **kwargs):
        if after.parent.parent.name == "B":
            distance = distances[index(after)]
            return distance >= settings.lightroom_candidate_min_distance, distance
        return True, 8.0 if accepted else None
    monkeypatch.setattr("superphoto.pipeline.export_lightroom_baseline", baseline)
    monkeypatch.setattr("superphoto.pipeline.apply_recipe_pass", render)
    monkeypatch.setattr("superphoto.pipeline.assess_lightroom_render", quality)
    monkeypatch.setattr("superphoto.pipeline.is_distinct_candidate", distinct)
    return settings, conn, "similar", original, root, variants


@pytest.mark.parametrize("distances,invalid_rounds,selected", [
    ([2.76, 1.89, 1.92, 1.99], (), 0),  # Exact user failure: later attempts are worse choices.
    ([1.40, 2.60, 2.30, 2.00], (3,), 0),  # Greater difference does not prove a better photograph.
    ([1.40, 2.60, 2.30, 2.00], (0, 3), 1),  # First safe repair remains paired with its actual recipe.
    ([0.00, 0.00, 0.00, 0.00], (), 0),  # Even identical outputs cannot abort C or the entire job.
])
def test_similarity_retries_preserve_first_valid_design_and_finish_c(tmp_path, monkeypatch, distances, invalid_rounds, selected):
    args = _similar_candidate_job(tmp_path, monkeypatch, distances, invalid_rounds=invalid_rounds)
    _render_variants_with_lightroom(*args)
    settings, conn, _, _, root, _ = args
    saved = get_variants(conn, "similar")
    assert [v["slot"] for v in saved] == ["A", "B", "C"]
    audit = json.loads((root / "lightroom-audits/variant_B.json").read_text())
    assert audit["candidate_selection"]["distinct"] is False
    assert audit["candidate_selection"]["selected_round"] == selected
    assert audit["candidate_selection"]["fallback_policy"] == "first_valid_design"
    assert audit["accepted_score"] == 96
    assert audit["nearest_candidate_distance"] == distances[selected]
    chosen = audit["rounds"][selected]
    assert (root / "variant_B.jpg").read_bytes() == Path(chosen["export"]).read_bytes()
    recipe = json.loads(saved[1]["recipe_json"])
    assert recipe["exposure_ev"] == chosen["settings_applied"]["Exposure2012"]
    assert "差别较细微" in saved[1]["why"]
    assert (root / "variant_C.jpg").is_file()
    modes = [r["mode"] for r in audit["rounds"][1:]]
    assert modes[0] == ("safe-style-replacement" if 0 in invalid_rounds else "distinctness-replacement")
    assert all(mode == "distinctness-replacement" for mode in modes[1:])


def test_higher_passing_qa_score_and_larger_difference_do_not_prove_an_aesthetic_gain(tmp_path, monkeypatch):
    args = _similar_candidate_job(tmp_path, monkeypatch, [1., 2., 3., 4.], scores=[96., 98., 99., 100.])
    _render_variants_with_lightroom(*args)
    root = args[4]
    audit = json.loads((root / "lightroom-audits/variant_B.json").read_text())
    assert audit["accepted_score"] == 96
    assert audit["candidate_selection"]["selected_round"] == 0
    assert (root / "variant_B.jpg").read_bytes() == Path(audit["rounds"][0]["export"]).read_bytes()


@pytest.mark.parametrize("failure", ["pixels", "confirmed-damage"])
def test_similarity_fallback_never_delivers_quality_rejections(tmp_path, monkeypatch, failure):
    args = _similar_candidate_job(tmp_path, monkeypatch, [1.99] * 4,
                                 invalid_rounds=range(4) if failure == "pixels" else ())
    if failure == "confirmed-damage":
        from superphoto.review import VisualReview
        monkeypatch.setattr("superphoto.pipeline.review_render", lambda settings, before, after, *args:
            VisualReview(decision="worse", confidence=.95, damage_confirmed=True, notes=["confirmed damage"])
            if after.parent.parent.name == "B" else VisualReview())
    _render_variants_with_lightroom(*args)
    root = args[4]
    assert not (root / "variant_B.jpg").exists()
    audit = json.loads((root / "lightroom-audits/variant_B.json").read_text())
    assert audit["status"] == "failed"
    assert [row["slot"] for row in get_variants(args[1], "similar")] == ["A", "C"]
    assert (root / "variant_C.jpg").is_file()
    summary = json.loads((root / "lightroom-qa.json").read_text())
    assert summary["delivery"]["delivered"] == 2
    assert summary["delivery"]["omitted"][0]["slot"] == "B"


def test_all_unsafe_directions_still_fail_without_delivering_previews(tmp_path, monkeypatch):
    args = _similar_candidate_job(tmp_path, monkeypatch, [0] * 4)
    monkeypatch.setattr("superphoto.pipeline.assess_lightroom_render", lambda *a, **kw:
                        _report(accepted=False, score=20, issues=[SimpleNamespace(note="clipped")]))
    with pytest.raises(LightroomError, match="所有方案均未通过"):
        _render_variants_with_lightroom(*args)
    assert get_variants(args[1], "similar") == []
    assert all(not (args[4] / f"variant_{slot}.jpg").exists() for slot in "ABC")


def test_one_native_render_error_keeps_other_completed_directions(tmp_path, monkeypatch):
    from superphoto import pipeline
    args = _similar_candidate_job(tmp_path, monkeypatch, [0] * 4)
    render = pipeline.apply_recipe_pass
    def render_with_failure(client, *, original, **kwargs):
        if original.stem == "variant_B":
            raise LightroomError("Local adjustment did not read back")
        return render(client, original=original, **kwargs)
    monkeypatch.setattr(pipeline, "apply_recipe_pass", render_with_failure)
    _render_variants_with_lightroom(*args)
    assert [row["slot"] for row in get_variants(args[1], "similar")] == ["A", "C"]
    audit = json.loads((args[4] / "lightroom-audits/variant_B.json").read_text())
    assert audit["status"] == "failed" and "did not read back" in audit["reason"]


def test_optional_diversity_export_failure_preserves_valid_candidate(tmp_path, monkeypatch):
    args = _similar_candidate_job(tmp_path, monkeypatch, [1.99] * 4, broken_export=True)
    _render_variants_with_lightroom(*args)
    root = args[4]
    audit = json.loads((root / "lightroom-audits/variant_B.json").read_text())
    assert audit["candidate_selection"]["selected_round"] == 0
    assert audit["rounds"][-1]["error"] == "optional export failed"
    assert (root / "variant_C.jpg").is_file()


@pytest.mark.parametrize("failed_slot", list("ABC"))
def test_selected_presets_share_quality_policy_and_repair_their_own_curves(tmp_path, monkeypatch, failed_slot):
    from superphoto.models import HueShift, ToneCurve
    args = _similar_candidate_job(tmp_path, monkeypatch, [8.] * 4)
    settings, conn, _, _, root, variants = args
    settings.lightroom_qa_max_refinements = 0
    settings.visual_review_enabled = False
    curves = {}
    for slot, row in zip("ABC", variants):
        middle = {"A": .55, "B": .45, "C": .52}[slot]
        curves[slot] = [(0, 0), (.2, .2), (.5, middle), (.8, .8), (1, 1)]
        row["look_id"] = {"A": "downton", "B": "skam", "C": "five_cm"}[slot]
        recipe = Recipe(curve=ToneCurve(rgb=curves[slot]), hsl={"blue": HueShift(sat=-.2)})
        row["recipe_json"] = json.dumps({**recipe.model_dump(), "style_workflow": {
            "method_id": "preset:" + row["look_id"], "preset_id": row["look_id"]}})
    def quality(before, after, **kwargs):
        assert kwargs["creative_palette"] is True
        rejected = after.parent.parent.name == failed_slot and after.parent.name == "round-0"
        return _report(accepted=not rejected, issues=[SimpleNamespace(note="too strong")] if rejected else [])
    monkeypatch.setattr("superphoto.pipeline.assess_lightroom_render", quality)
    _render_variants_with_lightroom(*args)
    audit = json.loads((root / f"lightroom-audits/variant_{failed_slot}.json").read_text())
    assert audit["rounds"][1]["mode"] == "preset-strength-replacement"
    middle = audit["accepted_recipe"]["curve"]["rgb"][2][1]
    assert middle == pytest.approx(.5 + (curves[failed_slot][2][1] - .5) * .60)
    saved = get_variants(conn, "similar")
    assert len(saved) == 3
    for row in saved:
        workflow = json.loads(row["recipe_json"])["style_workflow"]
        assert workflow["method_id"] == "preset:" + row["look_id"]
