from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
import pytest


def test_parallel_workers_keep_database_results_and_transactions_separate(tmp_path, monkeypatch):
    from superphoto import app as appmod
    from superphoto.db import insert_job
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    appmod.reset_runtime()
    _, main = appmod._boot()
    insert_job(main, "concurrent", "photo.jpg")
    barrier = Barrier(4)

    def read(index):
        conn = appmod._c()
        barrier.wait(timeout=5)
        for _ in range(50):
            row = conn.execute(f"SELECT id, original_name, {index} AS worker FROM jobs").fetchone()
            assert dict(row) == {"id": "concurrent", "original_name": "photo.jpg", "worker": index}
        return conn

    with ThreadPoolExecutor(max_workers=4) as pool:
        connections = list(pool.map(read, range(4)))
    assert len({id(conn) for conn in [main, *connections]}) == 5
    main.execute("UPDATE jobs SET original_name='uncommitted.jpg'")
    assert connections[0].execute("SELECT original_name FROM jobs").fetchone()[0] == "photo.jpg"
    main.rollback()
    appmod.reset_runtime()


@pytest.mark.parametrize("path", ["/", "/index.html?ui=old", "/app.js?v=15", "/styles.css?v=13"])
def test_ui_upgrade_does_not_reuse_cached_files(tmp_path, monkeypatch, path):
    from superphoto import app as appmod
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    appmod.reset_runtime()
    with TestClient(appmod.app) as client:
        first = client.get(path)
        assert first.status_code == 200
        assert first.headers["cache-control"] == "no-store"
        conditional = client.get(path, headers={"If-None-Match": first.headers["etag"],
                                                 "If-Modified-Since": first.headers["last-modified"]})
        assert conditional.status_code == 200
        assert conditional.content == first.content
        assert conditional.headers["cache-control"] == "no-store"


def test_latest_job_restores_saved_work_without_processing(tmp_path, monkeypatch):
    from superphoto import app as appmod
    from superphoto.db import insert_job, update_job
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    appmod.reset_runtime()
    _, conn = appmod._boot()
    def unexpected_run(*args):
        pytest.fail("Restoring a result must not process a photo again")
    monkeypatch.setattr(appmod, "_safe_run", unexpected_run)
    with TestClient(appmod.app) as client:
        assert client.get("/api/jobs/latest").json() == {"id": None}
        insert_job(conn, "old", "old.jpg")
        insert_job(conn, "saved", "saved.jpg")
        update_job(conn, "saved", status="ready", step="ready")
        response = client.get("/api/jobs/latest")
        assert response.status_code == 200
        assert response.json() == {"id": "saved"}
        assert client.get("/api/jobs/saved").json()["status"] == "ready"
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_health_and_full_loop(tmp_path: Path, monkeypatch, scene: Path) -> None:
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("XAI_API_KEY", "")
    from superphoto import app as appmod

    appmod.reset_runtime()
    with TestClient(appmod.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ok"] is True
        assert body["critic"] == "local"
        assert "semantic" in body
        assert "available" in body["semantic"]
        assert "lightroom" in body
        assert "available" in body["lightroom"]

        with scene.open("rb") as fh:
            created = client.post("/api/jobs", files={"file": ("scene.jpg", fh, "image/jpeg")})
        assert created.status_code == 200, created.text
        job_id = created.json()["id"]

        job = client.get(f"/api/jobs/{job_id}")
        assert job.status_code == 200
        payload = job.json()
        assert payload["status"] == "ready"
        assert len(payload["variants"]) == 3
        assert payload["variants"][0]["moves"] == payload["variants"][0]["recipe"]["style_workflow"]["adaptation_notes"]
        assert payload["critique"]["summary"]
        assert payload["variants"][0]["recipe"]["workflow"]["repaired"]
        assert payload["variants"][0]["recipe"]["workflow"]["protected"]
        assert len(payload["variants"][0]["recipe"]["workflow"]["retouch_attributes"]) <= 3
        assert payload["variants"][0]["recipe"]["workflow"]["ethics"] in {
            "documentary",
            "portrait",
            "landscape",
            "product",
            "commercial",
        }
        assert "技术校正始终保留" in "".join(payload["variants"][0]["recipe"]["workflow"]["protected"])
        img = client.get(payload["variants"][0]["url"])
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")

        fb = client.post(
            f"/api/jobs/{job_id}/feedback",
            json={
                "chosen_slot": "B",
                "reaction": "too_strong",
            },
        )
        assert fb.status_code == 200
        prefs = client.get("/api/preferences")
        assert prefs.json()["total_picks"] == 1
        row = appmod._c().execute(
            "SELECT technical_score, style_score, restraint_score, notes FROM feedback"
        ).fetchone()
        assert tuple(row[:3]) == (3, 2, 1)
        assert "处理太重" in row["notes"]

        home = client.get("/")
        assert home.status_code == 200
        assert "SuperPhoto" in home.text


def test_retry_keeps_failed_job_and_uses_saved_original(tmp_path, monkeypatch, scene):
    from superphoto import app as appmod
    from superphoto.db import get_job, insert_job, update_job
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    appmod.reset_runtime()
    settings, conn = appmod._boot()
    original = appmod.job_dir(settings, "failed") / "original.jpg"
    original.write_bytes(scene.read_bytes())
    insert_job(conn, "failed", "test-photo.jpg")
    update_job(conn, "failed", status="error", step="error", original_file="original.jpg",
               error="Lightroom plugin not connected")
    calls = []
    def process(job_id, source):
        calls.append((job_id, source.read_bytes()))
        update_job(conn, job_id, status="ready", step="ready")
    monkeypatch.setattr(appmod, "_safe_run", process)
    with TestClient(appmod.app) as client:
        response = client.post("/api/jobs/failed/retry")
        assert response.status_code == 200
        new_id = response.json()["id"]
        assert new_id != "failed"
        assert calls == [(new_id, scene.read_bytes())]
        assert get_job(conn, new_id)["original_name"] == "test-photo.jpg"
        assert get_job(conn, "failed")["error"] == "Lightroom plugin not connected"
        assert original.read_bytes() == scene.read_bytes()
        assert client.post(f"/api/jobs/{new_id}/retry").status_code == 409
        assert client.post("/api/jobs/unknown/retry").status_code == 404
        original.unlink()
        assert client.post("/api/jobs/failed/retry").status_code == 410


@pytest.mark.parametrize("connected", [True, False])
def test_connect_endpoint_reports_real_handshake(tmp_path, monkeypatch, connected):
    from superphoto import app as appmod
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SUPERPHOTO_LIGHTROOM_ENABLED", "1")
    appmod.reset_runtime()
    def connect(_):
        if not connected:
            raise appmod.LightroomError("Lightroom 尚未准备好")
        return object(), {"connected": True, "selected": 0}
    monkeypatch.setattr(appmod, "connect_lightroom", connect)
    with TestClient(appmod.app) as client:
        response = client.post("/api/lightroom/connect")
        assert response.status_code == (200 if connected else 503)
        assert response.json() == ({"connected": True, "selected": 0} if connected else
                                   {"detail": "Lightroom 尚未准备好"})


def test_partial_delivery_exposes_only_accepted_choices(tmp_path, monkeypatch):
    import json
    from superphoto import app as appmod
    from superphoto.db import insert_job, replace_variants, update_job
    from superphoto.models import Recipe
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    appmod.reset_runtime()
    settings, conn = appmod._boot()
    insert_job(conn, "partial", "photo.jpg")
    update_job(conn, "partial", status="ready", step="ready")
    replace_variants(conn, "partial", [
        {"id": slot, "slot": slot, "look_id": "faithful", "title": slot, "why": "已保留",
         "recipe_json": Recipe().model_dump_json(), "filename": f"variant_{slot}.jpg"} for slot in "AC"])
    delivery = {"requested": 3, "delivered": 2, "omitted": [{"slot": "B", "reason": "confirmed damage"}]}
    (appmod.job_dir(settings, "partial") / "lightroom-qa.json").write_text(json.dumps({"delivery": delivery}))
    with TestClient(appmod.app) as client:
        payload = client.get("/api/jobs/partial").json()
        assert [v["slot"] for v in payload["variants"]] == ["A", "C"]
        assert payload["delivery"] == delivery
        assert client.post("/api/jobs/partial/feedback", json={"chosen_slot": "B", "reaction": "keep"}).status_code == 400
