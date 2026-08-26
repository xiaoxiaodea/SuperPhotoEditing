from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


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

        with scene.open("rb") as fh:
            created = client.post("/api/jobs", files={"file": ("scene.jpg", fh, "image/jpeg")})
        assert created.status_code == 200, created.text
        job_id = created.json()["id"]

        job = client.get(f"/api/jobs/{job_id}")
        assert job.status_code == 200
        payload = job.json()
        assert payload["status"] == "ready"
        assert len(payload["variants"]) == 3
        assert payload["critique"]["summary"]
        img = client.get(payload["variants"][0]["url"])
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")

        fb = client.post(
            f"/api/jobs/{job_id}/feedback",
            json={
                "chosen_slot": "B",
                "technical_score": 4,
                "style_score": 5,
                "restraint_score": 3,
            },
        )
        assert fb.status_code == 200
        prefs = client.get("/api/preferences")
        assert prefs.json()["total_picks"] == 1

        home = client.get("/")
        assert home.status_code == 200
        assert "SuperPhoto" in home.text
