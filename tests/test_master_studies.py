import json
from pathlib import Path
import sys

import pytest

from superphoto import master_studies as studies
from superphoto.models import Critique, Recipe
from test_celebrity_styles import _tech


def reference(rid="one", *, genre="landscape", tags=None, principle="atmospheric_depth", kind="photograph", role="portfolio", author="Artist"):
    return {"id": rid, "schema_version": 3, "status": "complete",
            "sources": [{"role": role, "relative_path": f"{rid}.jpg", "photographer": author}],
            "features": {"mean_luma": .30, "green_fraction": .08, "aqua_fraction": .05, "dynamic_spread": .62},
            "analysis": {"image_kind": kind, "genre": genre, "scene_tags": tags or ["fog", "vegetation"],
                "caption": "雾中林木逐渐变淡，近处树干深而清晰", "strengths": [{"evidence": "远处树干逐渐隐入雾气", "benefit": "保留空间层次"}],
                "lessons": [{"principle": principle, "applies_when": "当前原片确实具有远近雾气层次", "action": "保留低反差远景，仅整理近处影调",
                             "avoid": "不要用统一去雾压实全部远景", "review": "远景是否仍比近景柔和、淡一些？"}],
                "capture_dependencies": ["现场雾气和前后景距离决定空间层次"], "uncertainty": "无法从成片得知原始去雾参数"}}


def test_source_studies_require_matching_scene_and_verified_record_version(monkeypatch):
    records = [reference(), reference("portrait", genre="portrait", tags=["people"]),
               reference("bts", kind="behind_scenes"), reference("jury", role="jury_or_program"),
               reference("layout", kind="layout"), reference("obsolete")]
    records[-1]["schema_version"] = 1
    monkeypatch.setattr(studies, "load_studies", lambda: {"records": records})
    critique = Critique(genre="landscape", subject_note="雾中树林")
    chosen = studies.select_studies(critique, _tech())
    assert [r["id"] for r in chosen] == ["one"]
    assert chosen[0]["capture_dependencies"]
    clear = critique.model_copy(update={"subject_note": "树林，天空晴朗，没有雾气"})
    assert studies.select_studies(clear, _tech()) == []
    assert "fog" not in studies.scene_description(clear, _tech())["tags"]


def test_reference_guidance_changes_only_applicable_protection_and_reaches_review(monkeypatch):
    monkeypatch.setattr(studies, "load_studies", lambda: {"records": [reference()]})
    critique, technical = Critique(genre="landscape", subject_note="雾中树林"), _tech()
    original = Recipe(dehaze=.08, clarity=.12, exposure_ev=.15, temperature=3)
    protected = studies.apply_study_guards(original, critique, technical)
    assert protected.dehaze == .02 and protected.clarity == .08
    assert protected.exposure_ev == original.exposure_ev and protected.temperature == original.temperature
    assert original.dehaze == .08
    from superphoto.finishing_knowledge import finishing_context
    context = finishing_context(critique, technical)["per_image_studies"]
    assert context["references"][0]["id"] == "one"
    assert "远景" in context["review_checks"][0]
    portrait = Critique(genre="portrait", subject_note="室内肖像")
    assert studies.apply_study_guards(original, portrait, technical) == original


def test_reference_selection_keeps_distinct_authors_and_does_not_force_monochrome(monkeypatch):
    records = [reference("first", principle="highlight_rolloff"), reference("same-author", principle="highlight_rolloff", author="master ARTIST"),
               reference("other", principle="black_depth", author="Another"), reference("bw", principle="monochrome_structure", author="BW")]
    monkeypatch.setattr(studies, "load_studies", lambda: {"records": records})
    selected = studies.select_studies(Critique(genre="landscape", subject_note="山间树林"), _tech())
    assert len(selected) == 2
    assert len({r["photographer"] for r in selected}) == 2
    assert all(r["id"] != "bw" for r in selected)


def test_non_grounded_duplicate_scene_tags_fail_validation():
    schema = studies.StudyAnalysis.model_json_schema()
    assert "behind_scenes" in schema["properties"]["image_kind"]["enum"]
    with pytest.raises(ValueError, match="distinct visible"):
        studies.StudyAnalysis.unique_tags(["fog", "water", "fog"])


def test_model_prose_cannot_become_destructive_executable_advice():
    proposed = {"lessons": [{"principle": "skin_integrity", "action": "把肤色饱和度降到最低"},
                            {"principle": "atmospheric_depth", "action": "全局压暗远景，使用精细深度蒙版"}]}
    lessons = studies.workflow_lessons(proposed)
    assert "自然血色" in lessons[0]["action"] and "降到最低" not in lessons[0]["action"]
    assert "控制全局去雾" in lessons[1]["action"]
    assert "精细远近蒙版" in lessons[1]["avoid"]


def test_workflow_lessons_require_source_evidence_and_indoor_scenes_match(monkeypatch):
    glass = reference("glass", genre="portrait", tags=["people", "interior"])
    assert studies.workflow_lessons(glass["analysis"]) == []
    glass["analysis"]["lessons"][0]["principle"] = "subject_separation"
    room = reference("room", genre="architecture", tags=["architecture", "interior"],
                     principle="graphic_geometry", author="Another")
    monkeypatch.setattr(studies, "load_studies", lambda: {"records": [glass, room]})
    portrait = Critique(genre="indoor", subject_note="窗边人物，室内人像")
    interior = Critique(genre="indoor", subject_note="室内空间与楼梯")
    assert [r["id"] for r in studies.select_studies(portrait, _tech())] == ["glass"]
    assert [r["id"] for r in studies.select_studies(interior, _tech())] == ["room"]


def test_misty_city_reserves_a_relevant_depth_lesson_without_copying_forest_colors(monkeypatch):
    city = [reference(str(i), genre="architecture", tags=["architecture", "water"],
                      principle="graphic_geometry", author=str(i)) for i in range(3)]
    forest = reference("forest", tags=["fog"], author="Forest")
    forest["analysis"]["lessons"].append({"principle": "color_hierarchy"})
    monkeypatch.setattr(studies, "load_studies", lambda: {"records": [*city, forest]})
    critique = Critique(genre="landscape", subject_note="雾中城市建筑群与河水")
    result = studies.select_studies(critique, _tech())
    assert len(result) == 3 and result[0]["id"] == "forest"
    assert [lesson["principle"] for lesson in result[0]["lessons"]] == ["atmospheric_depth"]
    critique.subject_note = "城市建筑群与河水"
    critique.intent.goal = "保留远景建筑的大气透视与自然朦胧感"
    assert studies.select_studies(critique, _tech())[0]["id"] == "forest"
    critique.intent.goal = "整理建筑线条"
    critique.intent.reference_checks = ["远景雾气是否保留？"]
    assert "fog" not in studies.scene_description(critique, _tech())["tags"]


def test_coverage_rejects_stale_or_invalid_records(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_master_study_report import collect
    folder = tmp_path / "study"
    folder.mkdir()
    (folder / "records").mkdir()
    inventory = [{"id": "pending", "sha256": "one", "sources": [{"corpus": "ph-main"}]},
                 {"id": "stale", "sha256": "two", "sources": [{"corpus": "legacy"}]}]
    (folder / "inventory.json").write_text(json.dumps(inventory))
    (folder / "records" / "stale.json").write_text(json.dumps({"sha256": "old", "schema_version": 3, "status": "complete"}))
    records, pending, coverage = collect(folder)
    assert records == [] and len(pending) == 2
    assert coverage["status"] == "partial" and coverage["analyzed"] == 0
    assert coverage["corpora"] == {"ph-main": 1, "legacy": 1}


def test_bundled_index_is_complete_and_every_record_is_individually_valid():
    payload = json.loads(Path(studies.__file__).with_name("master_studies.json").read_text())
    records, coverage = payload["records"], payload["coverage"]
    assert coverage["status"] == "complete" and coverage["pending"] == 0 and not coverage["invalid"]
    assert len(records) == coverage["analyzed"] == coverage["unique_images"]
    assert len({r["sha256"] for r in records}) == len(records)
    for record in records:
        studies.StudyAnalysis.model_validate(record["analysis"])
        assert record["id"] == "hb-" + record["sha256"][:20]
        assert record["schema_version"] == 3 and record["status"] == "complete"
        assert record["sources"] and record["model"] and record["analyzed_at"]
