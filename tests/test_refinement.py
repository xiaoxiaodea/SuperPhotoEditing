import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import time
from threading import Event

from fastapi.testclient import TestClient
import pytest

from superphoto import refinement as ref
from superphoto.db import get_refinement, get_job, insert_refinement, update_job, update_refinement
from superphoto.models import ColorGradingControls, MaskAdjustment, Recipe, ToneCurve
from superphoto.review import VisualReview
from test_scene_preferences import saved_job


@pytest.fixture
def ready(tmp_path, monkeypatch, scene):
    from superphoto import app as appmod
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SUPERPHOTO_LIGHTROOM_ENABLED", "1")
    appmod.reset_runtime()
    settings, conn = appmod._boot()
    saved_job(conn)
    update_job(conn, "photo", original_file="original.jpg")
    root = settings.jobs_dir / "photo"
    root.mkdir(parents=True)
    (root / "original.jpg").write_bytes(scene.read_bytes())
    (root / "variant_B.jpg").write_bytes(scene.read_bytes())
    (root / "lightroom-audits").mkdir()
    base = Recipe(exposure_ev=.30, contrast=.05, curve=ToneCurve(rgb=[(0, 0), (.5, .55), (1, 1)]),
                  color_grading=ColorGradingControls(midtone_hue=40, midtone_sat=12),
                  masks=[MaskAdjustment(target="subject", exposure_ev=.15)])
    (root / "lightroom-audits" / "variant_B.json").write_text(json.dumps({"accepted_recipe": base.model_dump()}))
    yield appmod, settings, conn, root, base
    appmod.reset_runtime()


def test_only_requested_deltas_change_and_masks_accumulate_from_accepted_state():
    base = Recipe(exposure_ev=.3, color_grading=ColorGradingControls(midtone_hue=42, midtone_sat=15),
                  masks=[MaskAdjustment(target="subject", exposure_ev=.15)])
    edits = ref.RefinementEdits(exposure_ev=.2, masks=[ref.LocalDelta(target="subject", exposure_ev=.2)])
    result = ref.apply_edits(base, edits)
    assert result.exposure_ev == pytest.approx(.5)
    assert result.masks[0].exposure_ev == pytest.approx(.35)
    assert result.color_grading == base.color_grading
    assert result.model_dump(exclude={"exposure_ev", "masks"}) == base.model_dump(exclude={"exposure_ev", "masks"})
    assert base.masks[0].exposure_ev == .15
    with pytest.raises(ValueError, match="没有产生"):
        ref.apply_edits(base, ref.RefinementEdits())
    with pytest.raises(ValueError):
        ref.RefinementEdits(exposure_ev=float("nan"))
    with pytest.raises(ValueError):
        ref.RefinementEdits(exposure_ev=3)


def test_simple_feedback_fallback_does_not_turn_negation_or_region_request_into_global_edit():
    edits = ref._simple_edits("整体亮一点，保留现在的暖色")
    assert edits.exposure_ev == .25 and edits.temperature == 0 and not edits.hsl
    local = ref._simple_edits("主体亮一点，背景暗一点")
    assert local.exposure_ev == 0
    assert {m.target: m.exposure_ev for m in local.masks} == {"subject": .25, "background": -.20}
    for prompt in ("不要提亮", "只把水面变暖", "删掉右侧的人", "主体不要提亮"):
        with pytest.raises(ValueError):
            ref._simple_edits(prompt)


def test_interpreter_cannot_add_local_relighting_or_color_changes_to_brightness_request():
    edits = ref.RefinementEdits(exposure_ev=.2, temperature=3, saturation=.1, palette_strength=.2,
                               masks=[ref.LocalDelta(target="subject", exposure_ev=.2)])
    bounded = ref.constrain_requested_scope(edits, "希望整体明亮一些，保留现在的暖色")
    assert bounded.exposure_ev == .2 and bounded.masks == []
    assert bounded.temperature == bounded.saturation == bounded.palette_strength == 0
    local = ref.constrain_requested_scope(edits, "主体提亮一些，颜色不变")
    assert len(local.masks) == 1 and local.temperature == 0
    only_local = ref.constrain_requested_scope(edits, "请把主体稍稍提亮，保留现在的暖色")
    assert only_local.exposure_ev == 0 and len(only_local.masks) == 1


def test_api_multiple_votes_and_revision_request_idempotency(ready, monkeypatch):
    appmod, settings, conn, root, base = ready
    calls = []
    started, release = Event(), Event()
    def refine(rid):
        calls.append(rid)
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(appmod, "_safe_refine", refine)
    with TestClient(appmod.app) as client:
        # like / dislike are the older names of keep / pass.
        for slot, verdict in (("A", "like"), ("B", "dislike"), ("C", "favorite")):
            response = client.put(f"/api/jobs/photo/ratings/{slot}", json={"verdict": verdict})
            assert response.status_code == 200, response.text
        assert client.get("/api/jobs/photo").json()["ratings"] == {"A": "keep", "B": "pass", "C": "favorite"}
        # One favourite per photo: the earlier one stays kept.
        assert client.put("/api/jobs/photo/ratings/A", json={"verdict": "favorite"}).json()["ratings"] == {
            "A": "favorite", "B": "pass", "C": "keep"}
        client.put("/api/jobs/photo/ratings/B", json={"verdict": None})
        assert client.get("/api/jobs/photo").json()["ratings"]["B"] is None
        assert client.put("/api/jobs/photo/ratings/D", json={"verdict": "keep"}).status_code == 404
        assert client.put("/api/jobs/photo/ratings/A", json={"verdict": "love"}).status_code == 422
        request_id = str(uuid.uuid4())
        body = {"slot": "B", "prompt": " 整体亮一点，保留现在的暖色 ", "request_id": request_id}
        one = client.post("/api/jobs/photo/refinements", json=body)
        assert one.status_code == 202, one.text
        assert client.post("/api/jobs/photo/refinements", json=body).json()["id"] == request_id
        assert started.wait(5)
        assert calls == [request_id]
        assert client.post("/api/jobs/photo/refinements", json=body | {"prompt": "整体暗一点"}).status_code == 409
        assert client.post("/api/jobs/photo/refinements", json=body | {"request_id": str(uuid.uuid4())}).status_code == 409
        assert client.get(f"/api/jobs/photo/refinements/{request_id}/image").status_code == 404
        assert client.get(f"/api/jobs/another/refinements/{request_id}").status_code == 404
        assert client.post("/api/jobs/photo/refinements", json=body | {"prompt": "   "}).status_code == 422
        assert client.post("/api/jobs/photo/refinements", json=body | {"prompt": "亮" * 1001}).status_code == 422
        assert client.get("/api/jobs/photo").json()["refinements"][0]["status"] == "running"
        release.set()


def fake_render(monkeypatch, root, *, skipped=None, error=False):
    calls = []
    monkeypatch.setattr(ref, "connect_lightroom", lambda *_: (object(), {}))
    def render(client, *, original, recipe, output_dir, **kwargs):
        calls.append((original, recipe.model_copy(deep=True)))
        assert original != root / "original.jpg"
        assert original.read_bytes() == (root / "original.jpg").read_bytes()
        if error:
            raise RuntimeError("Lightroom connection lost")
        output_dir.mkdir(parents=True)
        output = output_dir / "out.jpg"
        output.write_bytes((root / "variant_B.jpg").read_bytes())
        return {"export": str(output), "effective_recipe": recipe.model_dump(), "skipped_masks": skipped or []}
    monkeypatch.setattr(ref, "apply_recipe_pass", render)
    monkeypatch.setattr(ref, "assess_lightroom_render", lambda *a, **kw: SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True}))
    monkeypatch.setattr(ref, "render_distance", lambda *a: 1.0)
    monkeypatch.setattr(ref, "review_render", lambda *a: VisualReview(decision="same"))
    return calls


def test_lifespan_worker_runs_claimed_refinement_through_pipeline(ready, monkeypatch):
    appmod, settings, conn, root, base = ready
    calls = fake_render(monkeypatch, root)
    monkeypatch.setattr(ref, "interpret_feedback", lambda settings, image, recipe, prompt, **kwargs:
                        (ref._simple_edits(prompt), {"provider": "local"}))
    request_id = str(uuid.uuid4())
    body = {"slot": "B", "prompt": "整体亮一点", "request_id": request_id}
    with TestClient(appmod.app) as client:
        response = client.post("/api/jobs/photo/refinements", json=body)
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 5
        while True:
            record = get_refinement(conn, request_id)
            if record["status"] not in {"queued", "running"}:
                break
            assert time.monotonic() < deadline, record
            Event().wait(.01)
        assert record["status"] == "ready", record["error"]
        assert len(calls) == 1
        repeated = client.post("/api/jobs/photo/refinements", json=body)
        assert repeated.json()["id"] == request_id
        assert repeated.json()["status"] == "ready"
        assert len(calls) == 1
        assert client.get(f"/api/jobs/photo/refinements/{request_id}/image").status_code == 200


def test_native_revision_retains_original_and_supports_chained_versions(ready, monkeypatch):
    appmod, settings, conn, root, base = ready
    calls = fake_render(monkeypatch, root)
    originals = {p: p.read_bytes() for p in (root / "original.jpg", root / "variant_B.jpg")}
    previous = None
    for i in range(2):
        rid = str(uuid.uuid4())
        insert_refinement(conn, rid, "photo", "B", previous, "整体亮一点，保留现在的暖色")
        ref.run_refinement(settings, conn, rid)
        result = get_refinement(conn, rid)
        assert result["status"] == "ready", result["error"]
        recipe = Recipe.model_validate_json(result["recipe_json"])
        assert recipe.exposure_ev == pytest.approx(base.exposure_ev + .25 * (i + 1))
        assert recipe.color_grading == base.color_grading
        assert all(path.read_bytes() == content for path, content in originals.items())
        previous = rid
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    assert get_job(conn, "photo")["status"] == "ready"
    with TestClient(appmod.app) as client:
        image = client.get(f"/api/jobs/photo/refinements/{previous}/image")
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        assert len(client.get("/api/jobs/photo").json()["refinements"]) == 2
        body = {"slot": "A", "prompt": "整体亮一点", "parent_id": previous, "request_id": str(uuid.uuid4())}
        assert client.post("/api/jobs/photo/refinements", json=body).status_code == 409


@pytest.mark.parametrize("failure", ["connection", "missing_mask", "no_change", "unsupported", "quality"])
def test_failed_refinement_keeps_accepted_photo_available(ready, monkeypatch, failure):
    appmod, settings, conn, root, base = ready
    calls = fake_render(monkeypatch, root, error=failure == "connection",
                        skipped=[{"target": "sky"}] if failure == "missing_mask" else None)
    if failure == "no_change":
        monkeypatch.setattr(ref, "render_distance", lambda *a: 0)
    if failure == "quality":
        monkeypatch.setattr(ref, "assess_lightroom_render", lambda *a, **kw: SimpleNamespace(accepted=False, issues=[], to_dict=lambda: {}))
    rid = str(uuid.uuid4())
    prompt = "天空暗一点" if failure == "missing_mask" else "删除路人" if failure == "unsupported" else "整体亮一点"
    insert_refinement(conn, rid, "photo", "B", None, prompt)
    original = (root / "variant_B.jpg").read_bytes()
    ref.run_refinement(settings, conn, rid)
    assert get_refinement(conn, rid)["status"] == "error"
    assert get_job(conn, "photo")["status"] == "ready"
    assert (root / "variant_B.jpg").read_bytes() == original
    assert not (root / "refinements" / rid / "result.jpg").exists()
    if failure == "unsupported":
        assert calls == []


def test_restart_releases_abandoned_refinement_for_retry(ready):
    appmod, settings, conn, root, base = ready
    rid = str(uuid.uuid4())
    insert_refinement(conn, rid, "photo", "B", None, "整体亮一点")
    update_refinement(conn, rid, status="running", step="render")
    appmod.reset_runtime()
    _, conn = appmod._boot()
    # Recovery happens only after the lifespan worker obtains its OS lock.
    with TestClient(appmod.app):
        assert get_refinement(conn, rid)["status"] == "error"
        assert "中断" in get_refinement(conn, rid)["error"]
    insert_refinement(conn, str(uuid.uuid4()), "photo", "B", None, "整体亮一点")
