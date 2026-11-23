"""Provenance and retained-state checks for the separate reference experiment."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from superphoto.models import EditIntent, Recipe
from superphoto.pair_review import PairEvidence, PairJudgment, PairOrder


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_reference_finish.py"
    spec = importlib.util.spec_from_file_location("reference_finish_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    original = job / "original.cr3"
    original.write_bytes(b"camera-original")
    photo = job / "native.cr3"
    shutil.copy2(original, photo)
    exported = job / "native-export.jpg"
    Image.new("RGB", (80, 60), (160, 170, 190)).save(exported)
    delivered = job / "variant_B.jpg"
    shutil.copy2(exported, delivered)
    baseline = job / "baseline.jpg"
    Image.new("RGB", (80, 60), (210, 210, 220)).save(baseline)
    audit_dir = job / "lightroom-audits"
    audit_dir.mkdir()
    recipe = Recipe(contrast=.1)
    (audit_dir / "variant_B.json").write_text(json.dumps({
        "accepted_export": str(exported), "accepted_photo": str(photo),
        "baseline_export": str(baseline), "accepted_recipe": recipe.model_dump()}))
    (job / "variant_B.json").write_text(json.dumps({**recipe.model_dump(), "intent": EditIntent().model_dump()}))
    (job / "lightroom-qa.json").write_text(json.dumps({"slots": {"B": {"status": "delivered"}}}))
    return job, delivered, exported


@pytest.mark.parametrize("fault", ["pixels", "recipe", "native-source", "undelivered"])
def test_unpaired_or_unverified_starting_result_is_rejected(experiment, tmp_path, fault):
    job, delivered, exported = _job(tmp_path)
    if fault == "pixels":
        delivered.write_bytes(b"local approximation instead of actual native export")
    elif fault == "recipe":
        (job / "variant_B.json").write_text(Recipe(contrast=.5).model_dump_json())
    elif fault == "native-source":
        (job / "native.cr3").write_bytes(b"different camera source")
    else:
        (job / "lightroom-qa.json").write_text('{"slots":{"B":{"status":"withheld"}}}')
    with pytest.raises(ValueError):
        experiment.load_verified_job(job, "B")


def test_reference_crop_is_only_in_memory_and_bounds_are_checked(experiment, tmp_path):
    reference = tmp_path / "reference.jpg"
    Image.new("RGB", (100, 70), (90, 110, 180)).save(reference)
    before = set(tmp_path.iterdir())
    pixels, provenance = experiment.reference_preview(reference, (10, 10, 90, 60))
    assert pixels and provenance["roi"] == (10, 10, 90, 60)
    assert provenance["source_size"] == [100, 70]
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(ValueError):
        experiment.reference_preview(reference, (10, 10, 110, 60))


def test_reference_round_two_failure_retains_first_effective_native_result(experiment, tmp_path, monkeypatch):
    job, _, initial_export = _job(tmp_path)
    out = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--job", str(job), "--reference", str(initial_export),
                                     "--reference-roi", "0", "0", "80", "60", "--out", str(out), "--rounds", "2"])
    monkeypatch.setattr(experiment, "probe", lambda _: {"ready": True, "resolved_model": "test-vision"})
    monkeypatch.setattr(experiment, "get_lightroom_client", lambda _: object())
    monkeypatch.setattr(experiment, "wait_for_lightroom_connection", lambda *a, **kw: {"connected": True})
    monkeypatch.setattr(experiment, "close_lightroom_client", lambda: None)
    monkeypatch.setattr(experiment, "_preview", lambda path: str(path))
    proposal = {"confidence": .9, "observations": ["参考中云层明暗更分明"],
                "weaknesses": [{"region": "上方天空", "evidence": "当前云底与亮云接近", "goal": "恢复层次"}],
                "delta": {"contrast": .1}}
    answers = iter([proposal, proposal])
    vision_calls = []
    pair_calls = []

    def vision(settings, model, prompt, images, **kwargs):
        vision_calls.append(images)
        return json.dumps(next(answers))

    def pair(settings, first, second, intent, reference_preview=None):
        pair_calls.append((first, second, reference_preview))
        evidence = PairEvidence(region="云层", image_1="云底和亮云接近",
                                image_2="明暗分开", reason="更接近参考光层次")
        order = PairOrder(image_ids=["first", "second"], preview_hashes=["p1", "p2"],
                          prompt="neutral", choice=2, confidence=.9, evidence=[evidence],
                          raw_response="forward")
        return PairJudgment(outcome="winner", winner_id="second", first_id="first", second_id="second",
                            confidence=.9, orders=[order, order.model_copy(update={"choice": 1})], source="ollama-pair")

    def render(client, *, original, recipe, output_dir, **kwargs):
        assert original.read_bytes() == b"camera-original"
        output_dir.mkdir()
        effective = recipe.model_copy(update={"contrast": recipe.contrast - .02})
        exported = output_dir / "export.jpg"
        exported.write_text(effective.model_dump_json())
        return {"export": exported, "effective_recipe": effective.model_dump(),
                "settings": {"Contrast2012": effective.contrast * 100},
                "develop_before": {}, "develop_after": {"Contrast2012": effective.contrast * 100}, "stages": []}

    def quality(before, after, **kwargs):
        accepted = after.parent.parent.name == "round-0"
        return SimpleNamespace(accepted=accepted, issues=[], to_dict=lambda: {"accepted": accepted})

    monkeypatch.setattr(experiment, "chat_vision", vision)
    monkeypatch.setattr(experiment, "compare_finished_pair", pair)
    monkeypatch.setattr(experiment, "build_pair_board", lambda *args: "Ym9hcmQ=")
    monkeypatch.setattr(experiment, "apply_recipe_pass", render)
    monkeypatch.setattr(experiment, "assess_lightroom_render", quality)
    experiment.main()
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["planner"] == "reference-guided-model" and not audit["reference_free_automatic"]
    assert [record["selected"] for record in audit["rounds"]] == [True, False]
    accepted = Path(audit["accepted_export"])
    assert accepted.parent.parent.name == "round-0"
    assert (out / "final.jpg").read_bytes() == accepted.read_bytes()
    assert audit["accepted_recipe"]["contrast"] == pytest.approx(.18)
    assert json.loads(accepted.read_text(encoding="utf-8")) == audit["accepted_recipe"]
    assert audit["rounds"][1]["retained_recipe"]["contrast"] == pytest.approx(.18)
    assert len(vision_calls) == 2 and vision_calls == ["Ym9hcmQ=", "Ym9hcmQ="]
    assert len(pair_calls) == 1 and pair_calls[0][2]
    assert (out / "round-0/proposal-raw.txt").is_file()
    assert (out / "round-0/comparison-raw.txt").is_file()
