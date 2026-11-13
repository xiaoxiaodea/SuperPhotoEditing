"""A finishing attempt must replace a complete accepted state, or none of it."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superphoto import pipeline
from superphoto.config import Settings
from superphoto.db import connect, get_variants, init_db, insert_job
from superphoto.finishing_review import FinishingDecision, FinishingEvidence, FinishingProposal, FinishingWeakness
from superphoto.lightroom import LightroomError
from superphoto.models import Critique, Recipe
from superphoto.review import VisualReview


@pytest.fixture
def finishing_job(tmp_path, monkeypatch):
    connections = []

    def setup(outcomes, *, suffix=".cr3"):
        settings = Settings(data_dir=tmp_path / "data", lightroom_enabled=True,
                            semantic_enabled=False, lightroom_qa_max_refinements=0,
                            finishing_review_enabled=True, finishing_max_rounds=len(outcomes))
        conn = connect(settings.db_path)
        connections.append(conn)
        init_db(conn)
        insert_job(conn, "finishing", "photo" + suffix)
        root = settings.jobs_dir / "finishing"
        root.mkdir(parents=True)
        original = root / ("original" + suffix)
        original.write_bytes(b"untouched-source")
        initial_recipe = Recipe(exposure_ev=.1, contrast=.1)
        variants = [{"id": "A", "slot": "A", "look_id": "faithful", "title": "A", "why": "分区光影",
                     "recipe_json": initial_recipe.model_dump_json(), "filename": "variant_A.jpg"}]
        state = SimpleNamespace(root=root, conn=conn, settings=settings, original=original,
                                variants=variants, proposals=[], renders=[], comparisons=[], checkpoints=[],
                                native_states={}, outcomes=outcomes)

        monkeypatch.setattr(pipeline, "_connected_lightroom", lambda _: (object(), {"tools": ["create_develop_preset"]}))

        def baseline(client, *, original, output_dir, **kwargs):
            output_dir.mkdir(parents=True)
            exported = output_dir / "baseline.jpg"
            exported.write_bytes(b"baseline")
            return {"export": exported}

        def render(client, *, original, recipe, output_dir, **kwargs):
            assert original.read_bytes() == b"untouched-source"
            finishing = output_dir.name.startswith("finishing-")
            index = int(output_dir.name.rsplit("-", 1)[1]) if finishing else -1
            outcome = outcomes[index] if finishing else "initial"
            if outcome == "lightroom-error":
                raise LightroomError("optional finishing export failed")
            output_dir.mkdir(parents=True)
            exported = output_dir / "rendered.jpg"
            # Simulate the native stage safety reducing a proposed control. The
            # accepted recipe must describe actual pixels, not the proposal.
            effective = recipe.model_copy(update={"contrast": recipe.contrast - (.02 if finishing else 0)})
            state.native_states[str(original)] = effective.model_dump(mode="json")
            exported.write_text(json.dumps({"photo": str(original), "recipe": effective.model_dump()}))
            result = {"export": exported,
                      "settings": {"Contrast2012": effective.contrast * 100},
                      "develop_before": {}, "develop_after": {"Contrast2012": effective.contrast * 100},
                      "stages": [{"stage": output_dir.name}],
                      "effective_recipe": effective.model_dump()}
            if outcome == "invalid-effective":
                result["effective_recipe"] = {"exposure_ev": "not-a-number"}
            state.renders.append({"photo": original, "recipe": recipe.model_copy(deep=True), "result": result})
            return result

        def quality(before, after, **kwargs):
            finishing = after.parent.name.startswith("finishing-")
            index = int(after.parent.name.rsplit("-", 1)[1]) if finishing else -1
            accepted = not finishing or outcomes[index] != "qa-failure"
            score = 94 if not finishing else (97 + index if accepted else 42)
            issues = [] if accepted else [SimpleNamespace(code="highlight_clipping", note="云层剪切")]
            data = {"accepted": accepted, "score": score, "issues": [] if accepted else ["highlight_clipping"]}
            return SimpleNamespace(accepted=accepted, score=score, issues=issues, to_dict=lambda: data)

        def propose(settings, original_export, current_export, recipe, intent, source_kind):
            assert original_export.name == "baseline.jpg"
            state.proposals.append({"current": current_export, "recipe": recipe.model_copy(deep=True),
                                    "source_kind": source_kind})
            return FinishingProposal(source="ollama", confidence=.9,
                weaknesses=[FinishingWeakness(region="上方云层", evidence="云底明暗接近", goal="增加云层明暗区分")],
                recipe=recipe.model_copy(update={"contrast": recipe.contrast + .12}))

        def compare(settings, before, after, intent):
            state.comparisons.append((before, after))
            outcome = outcomes[int(after.parent.name.rsplit("-", 1)[1])]
            decision = outcome if outcome in {"same", "worse", "uncertain", "unavailable"} else "better"
            return FinishingDecision(decision=decision, confidence=.9, source="ollama",
                evidence=[FinishingEvidence(region="云底", before="云底与亮部明暗接近", after="明暗分开",
                                             benefit="云层更可读且地面明暗保持")])

        def review(settings, before, after, intent, stage):
            if after.parent.name.startswith("finishing-") and outcomes[int(after.parent.name.rsplit("-", 1)[1])] == "damage":
                return VisualReview(decision="worse", confidence=.95, damage_confirmed=True)
            return VisualReview(decision="same", confidence=.9)

        def checkpoint(client, *, photo, name, settings):
            state.checkpoints.append({"photo": photo, "settings": settings})
            return {"name": name, "photo": str(photo)}

        monkeypatch.setattr(pipeline, "export_lightroom_baseline", baseline)
        monkeypatch.setattr(pipeline, "apply_recipe_pass", render)
        monkeypatch.setattr(pipeline, "assess_lightroom_render", quality)
        monkeypatch.setattr(pipeline, "propose_finishing", propose)
        monkeypatch.setattr(pipeline, "compare_finishing", compare)
        monkeypatch.setattr(pipeline, "review_render", review)
        monkeypatch.setattr(pipeline, "is_distinct_candidate", lambda *args, **kwargs: (True, None))
        monkeypatch.setattr(pipeline, "create_lightroom_checkpoint", checkpoint)
        state.run = lambda: pipeline._render_variants_with_lightroom(
            settings, conn, "finishing", original, root, variants, critique=Critique())
        return state

    yield setup
    for connection in connections:
        connection.close()


def _assert_matching_delivery(state, selected_directory, expected_contrast, expected_score):
    audit = json.loads((state.root / "lightroom-audits/variant_A.json").read_text())
    sidecar = json.loads((state.root / "variant_A.json").read_text())
    stored = json.loads(get_variants(state.conn, "finishing")[0]["recipe_json"])
    selected_export = Path(audit["accepted_export"])
    assert selected_export.parent.name == selected_directory
    assert (state.root / "variant_A.jpg").read_bytes() == selected_export.read_bytes()
    rendered = json.loads(selected_export.read_text())
    assert audit["accepted_photo"] == rendered["photo"]
    assert audit["accepted_recipe"] == rendered["recipe"]
    assert audit["accepted_recipe"]["contrast"] == pytest.approx(expected_contrast)
    assert state.native_states[audit["accepted_photo"]] == rendered["recipe"]
    assert Recipe.model_validate(sidecar).model_dump(mode="json") == rendered["recipe"]
    assert Recipe.model_validate(stored).model_dump(mode="json") == rendered["recipe"]
    assert audit["accepted_score"] == sidecar["lightroom_pixel_qa"]["score"] == expected_score
    selected = audit["rounds"][audit["candidate_selection"]["selected_round"]]
    assert selected["export"] == str(selected_export)
    assert selected["settings_applied"]["Contrast2012"] == pytest.approx(expected_contrast * 100)
    assert state.checkpoints[-1]["photo"] == Path(audit["accepted_photo"])
    assert state.checkpoints[-1]["settings"]["Contrast2012"] == pytest.approx(expected_contrast * 100)
    summary = json.loads((state.root / "lightroom-qa.json").read_text())["slots"]["A"]
    assert summary["stages"] == [selected_directory]
    assert summary["score"] == expected_score
    return audit


def test_accepted_finishing_delivers_effective_pixels_recipe_photo_checkpoint_and_audit(finishing_job):
    state = finishing_job(["better"])
    state.run()
    audit = _assert_matching_delivery(state, "finishing-0", .20, 97)
    assert audit["finishing_review"][0]["selected"] is True
    assert state.proposals[0]["recipe"].contrast == .1
    assert state.renders[1]["recipe"].contrast == pytest.approx(.22)
    assert state.renders[0]["photo"] != state.renders[1]["photo"]
    assert state.comparisons[0][0].parent.name == "round-0"


@pytest.mark.parametrize("outcome", ["same", "worse", "uncertain", "unavailable", "qa-failure", "damage", "lightroom-error"])
def test_unsuccessful_finishing_preserves_complete_previously_accepted_state(finishing_job, outcome):
    state = finishing_job([outcome])
    state.run()
    audit = _assert_matching_delivery(state, "round-0", .1, 94)
    assert audit["finishing_review"][0]["selected"] is False
    if outcome == "lightroom-error":
        assert "optional finishing export failed" in audit["finishing_review"][0]["error"]
    if outcome == "qa-failure":
        assert state.comparisons == []


@pytest.mark.parametrize("second_outcome", ["worse", "uncertain", "qa-failure", "damage", "lightroom-error", "invalid-effective"])
def test_second_failure_preserves_first_success_instead_of_original_or_failed_attempt(finishing_job, second_outcome):
    state = finishing_job(["better", second_outcome])
    state.run()
    audit = _assert_matching_delivery(state, "finishing-0", .20, 97)
    assert [record["selected"] for record in audit["finishing_review"]] == [True, False]
    assert state.proposals[1]["current"].parent.name == "finishing-0"
    assert state.proposals[1]["recipe"].contrast == pytest.approx(.20)
    if second_outcome not in {"qa-failure", "lightroom-error", "invalid-effective"}:
        assert state.comparisons[1][0].parent.name == "finishing-0"


def test_invalid_effective_recipe_cannot_partially_replace_accepted_state(finishing_job):
    state = finishing_job(["invalid-effective"])
    state.run()
    audit = _assert_matching_delivery(state, "round-0", .1, 94)
    assert audit["finishing_review"][0]["selected"] is False
    assert "error" in audit["finishing_review"][0]


def test_optional_finishing_source_copy_error_preserves_delivery(finishing_job, monkeypatch):
    state = finishing_job(["better"])
    copy = pipeline.shutil.copy2

    def copy_with_optional_failure(source, destination, **kwargs):
        if "_finishing_" in Path(destination).name:
            raise OSError("cannot create optional finishing source")
        return copy(source, destination, **kwargs)

    monkeypatch.setattr(pipeline.shutil, "copy2", copy_with_optional_failure)
    state.run()
    audit = _assert_matching_delivery(state, "round-0", .1, 94)
    assert "cannot create optional finishing source" in audit["finishing_review"][0]["error"]


@pytest.mark.parametrize("suffix,expected_kind", [(".cr3", "raw"), (".jpg", "jpeg"), (".png", "png"),
                                                  (".tiff", "tiff"), (".tif", "tiff")])
def test_finishing_without_technical_sidecar_cannot_give_developed_files_raw_headroom(finishing_job, suffix, expected_kind):
    state = finishing_job(["same"], suffix=suffix)
    state.run()
    assert state.proposals[0]["source_kind"] == expected_kind
