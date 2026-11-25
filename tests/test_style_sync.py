from __future__ import annotations

import io
import json
import time
import uuid
import zipfile
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from superphoto import style_sync
from superphoto.db import insert_job, insert_refinement, replace_variants, update_job, update_refinement
from superphoto.lightcraft import discover_cli


def _make_photo(path: Path, *, luma_scale: float = 1.0, warmth: float = 0.0, size: tuple[int, int] = (240, 160)) -> Path:
    w, h = size
    y = np.linspace(0.25, 0.72, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 0.15, w, dtype=np.float32)[None, :]
    r = np.clip((y + x + warmth) * luma_scale * 255.0, 0, 255)
    g = np.clip((y * 0.95 + x * 0.5) * luma_scale * 255.0, 0, 255)
    b = np.clip((y * 0.85 + x * 0.0 - warmth) * luma_scale * 255.0, 0, 255)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=92)
    return path


def _seed_ready_anchor_job(appmod, tmp_path: Path):
    _, conn = appmod._boot()
    settings = appmod._s()
    job_id = "anchor-job"
    root = settings.jobs_dir / job_id
    root.mkdir(parents=True, exist_ok=True)
    _make_photo(root / "original.jpg", luma_scale=0.85)
    _make_photo(root / "source_preview.jpg", luma_scale=0.85)
    _make_photo(root / "variant_E.jpg", luma_scale=1.15, warmth=0.07)

    dev_dir = root / "directed" / "E" / "round-0"
    dev_dir.mkdir(parents=True, exist_ok=True)
    develop = {
        "wb": {"mode": "custom", "temp": 7150.0, "tint": 4.0},
        "light": {"exposure": 0.45},
        "curve": {"master": [[0.0, 0.08], [0.5, 0.58], [1.0, 0.94]], "refine_saturation": 50.0},
        "mixer": {"green": {"hue": -24.0, "sat": -40.0}, "orange": {"sat": 14.0}},
        "grading": {"shadows": {"hue": 220.0, "sat": 12.0}, "highlights": {"hue": 42.0, "sat": 9.0}, "blending": 100.0},
        "calibration": {"red_hue": 15.0, "blue_sat": 20.0},
        "effects": {"clarity": 8.0, "dehaze": 6.0},
        "vignette": {"amount": -10.0, "midpoint": 50.0, "feather": 70.0},
        "grain": {"amount": 12.0, "size": 20.0, "roughness": 50.0},
        "masks": [{"id": 1, "name": "sky", "visible": True, "components": [], "adjust": {"exposure": -0.5}}],
    }
    (dev_dir / "develop.json").write_text(json.dumps(develop), encoding="utf-8")
    (dev_dir / "render.jpg").write_bytes((root / "variant_E.jpg").read_bytes())

    audits = root / "lightroom-audits"
    audits.mkdir(parents=True, exist_ok=True)
    (audits / "variant_E.json").write_text(json.dumps({
        "engine": "lightcraft-directed",
        "slot": "E",
        "status": "delivered",
        "accepted_export": str(dev_dir / "render.jpg"),
        "document": develop,
    }), encoding="utf-8")

    insert_job(conn, job_id, "sunset_anchor.jpg", "original.jpg")
    update_job(conn, job_id, status="ready", step="ready")
    replace_variants(conn, job_id, [{
        "id": str(uuid.uuid4()),
        "slot": "E",
        "look_id": "directed",
        "title": "沙漠暖金",
        "why": "暖金色调与柔和哑光黑位",
        "recipe_json": json.dumps({
            "engine": "lightcraft-directed",
            "plan": {"title": "沙漠暖金", "look": {"id": "desert_gold", "strength": 0.9}},
            "style_workflow": {"direction_title": "沙漠 · 暖金", "selection_reason": "沙丘暖金"},
            "develop": develop,
        }),
        "filename": "variant_E.jpg",
    }])
    return settings, conn, job_id, root, develop


def test_extract_profile_preserves_style_dna_strips_spatial_masks_and_supports_refinements(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    from superphoto import app as appmod
    appmod.reset_runtime()
    settings, conn, job_id, root, develop = _seed_ready_anchor_job(appmod, tmp_path)

    profile = style_sync.extract_style_profile(settings, conn, job_id, "E")
    assert profile["title"] == "沙漠暖金"
    assert profile["direction_title"] == "沙漠 · 暖金"
    assert profile["wb_offset"] == {"temp": 650.0, "tint": 4.0}
    assert "masks" not in profile["style_develop"]
    assert "light" not in profile["style_develop"]  # master curve owned tone; exposure is solved per target
    assert profile["style_develop"]["mixer"] == develop["mixer"]
    assert profile["style_develop"]["grading"] == develop["grading"]
    assert profile["style_develop"]["calibration"] == develop["calibration"]
    assert profile["anchor_finished"]["anchors"]["mid"] > profile["anchor_baseline"]["anchors"]["mid"]

    # Also works from a refined version!
    ref_id = str(uuid.uuid4())
    insert_refinement(conn, ref_id, job_id, "E", None, "再暖一点")
    ref_dir = root / "refinements" / ref_id
    (ref_dir / "exports").mkdir(parents=True)
    ref_dev = {**develop, "wb": {"mode": "custom", "temp": 7400.0, "tint": 5.0}}
    (ref_dir / "exports" / "develop.json").write_text(json.dumps(ref_dev), encoding="utf-8")
    _make_photo(ref_dir / "result.jpg", luma_scale=1.18, warmth=0.10)
    update_refinement(conn, ref_id, status="ready", step="ready", summary="色调更暖", filename="result.jpg")

    ref_profile = style_sync.extract_style_profile(settings, conn, job_id, "E", version=ref_id)
    assert ref_profile["source_version"] == ref_id
    assert ref_profile["wb_offset"] == {"temp": 900.0, "tint": 5.0}
    appmod.reset_runtime()


def test_target_tone_and_wb_adaptation_unifies_differently_exposed_shots():
    profile = {
        "style_develop": {
            "mixer": {"green": {"hue": -20.0, "sat": -50.0}},
            "grading": {"shadows": {"hue": 220.0, "sat": 14.0}, "blending": 100.0},
            "grain": {"amount": 16.0, "size": 22.0, "roughness": 50.0},
        },
        "wb_offset": {"temp": 500.0, "tint": 3.0},
        "anchor_baseline": {"anchors": {"black": 0.04, "shadows": 0.14, "mid": 0.36, "highlights": 0.72, "white": 0.90},
                            "warmth": 0.02, "tint_bias": 0.0, "chroma": 0.14},
        "anchor_finished": {"anchors": {"black": 0.09, "shadows": 0.22, "mid": 0.52, "highlights": 0.80, "white": 0.94},
                            "warmth": 0.09, "tint_bias": 0.0, "chroma": 0.20},
        "tone_ev_delta": {"black": 1.0, "shadows": 0.8, "mid": 0.9, "highlights": 0.3, "white": 0.1},
    }
    dark_base = {"black": 0.02, "shadows": 0.10, "mid": 0.28, "highlights": 0.56, "white": 0.80}
    bright_base = {"black": 0.08, "shadows": 0.26, "mid": 0.64, "highlights": 0.88, "white": 0.98}

    t_dark = style_sync._target_tone_anchors(profile, dark_base, target_kind="jpeg", match_exposure=True, style_strength=1.0)
    t_bright = style_sync._target_tone_anchors(profile, bright_base, target_kind="jpeg", match_exposure=True, style_strength=1.0)
    # Exposure-matched targets pull both dark (0.28) and bright (0.64) shots tightly toward anchor mid (0.52) & matte black (0.09)
    assert abs(t_dark["mid"] - t_bright["mid"]) < 0.08
    assert t_dark["black"] >= 0.07 and t_bright["black"] >= 0.07

    # Cooler target baseline receives warm drift compensation; hues & blending remain unscaled at 0.5 strength
    doc = style_sync._compose_base_document(profile, "jpeg", {"warmth": -0.05, "tint_bias": 0.0}, match_wb=True, style_strength=0.5)
    assert doc["grading"]["shadows"]["hue"] == 220.0
    assert doc["grading"]["shadows"]["sat"] == 7.0
    assert doc["grading"]["blending"] == 100.0
    assert doc["wb"]["mode"] == "custom" and doc["wb"]["temp"] > 6500.0 + 250.0


def test_style_sync_api_batch_upload_poll_zip_and_full_export(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SUPERPHOTO_LIGHTROOM_ENABLED", "true")
    from superphoto import app as appmod
    appmod.reset_runtime()
    settings, conn, job_id, root, _ = _seed_ready_anchor_job(appmod, tmp_path)

    # Mock LightCraft connection + apply_style_to_photo + full export so the full HTTP + worker queue runs fast
    class _FakeClient:
        binary = Path("/fake/lightcraft-cli")

    monkeypatch.setattr(appmod, "connect_lightroom", lambda _bin=None: (_FakeClient(), {"connected": True}))
    monkeypatch.setattr(style_sync, "connect_lightroom", lambda _bin=None: (_FakeClient(), {"connected": True}))

    def fake_apply(s, binary, profile, source, out_dir, **kwargs):
        base = _make_photo(out_dir / "baseline" / "source_preview.jpg", luma_scale=0.8)
        res = _make_photo(out_dir / "result" / "result.jpg", luma_scale=1.15, warmth=0.07)
        return {
            "source_preview": str(base),
            "result": str(res),
            "develop": {"light": {"exposure": 0.4}, "mixer": profile["style_develop"].get("mixer", {})},
            "source_kind": "jpeg",
            "negative": None,
            "metrics": {"exposure_ev": 0.4, "before_mid": 0.35, "after_mid": 0.52, "seconds": 0.2, "notes": []},
        }

    def fake_render(binary, source, document, dest, *, edge=0, name="render.jpg", **kwargs):
        return _make_photo(Path(dest) / name, luma_scale=1.15, warmth=0.07)

    monkeypatch.setattr(style_sync, "apply_style_to_photo", fake_apply)
    monkeypatch.setattr(style_sync, "render_document", fake_render)

    p1 = _make_photo(tmp_path / "shoot_01.jpg", luma_scale=0.65)
    p2 = _make_photo(tmp_path / "shoot_02.png", luma_scale=1.15)
    p3 = _make_photo(tmp_path / "shoot_03.jpg", luma_scale=0.90)

    with TestClient(appmod.app) as client:
        files = [
            ("files", ("shoot_01.jpg", p1.read_bytes(), "image/jpeg")),
            ("files", ("shoot_02.png", p2.read_bytes(), "image/png")),
            ("files", ("shoot_03.jpg", p3.read_bytes(), "image/jpeg")),
        ]
        created = client.post(
            f"/api/jobs/{job_id}/sync-style",
            data={"slot": "E", "match_exposure": "true", "match_wb": "true", "style_strength": "0.9"},
            files=files,
        )
        assert created.status_code == 202, created.text
        batch = created.json()
        batch_id = batch["id"]
        assert batch["total_count"] == 3
        assert batch["sync_options"]["style_strength"] == 0.9

        # Wait for TaskWorker to process all 3 items
        deadline = time.monotonic() + 15
        while True:
            resp = client.get(f"/api/style-batches/{batch_id}")
            assert resp.status_code == 200
            batch = resp.json()
            if batch["status"] in {"ready", "error"}:
                break
            assert time.monotonic() < deadline, batch
            Event().wait(0.05)

        assert batch["status"] == "ready"
        assert batch["done_count"] == 3 and batch["error_count"] == 0
        assert batch["zip_url"] == f"/api/style-batches/{batch_id}/download.zip"

        # Verify listing, per-item images, full-size export, and ZIP bundle
        listed = client.get(f"/api/style-batches?job_id={job_id}").json()["batches"]
        assert any(b["id"] == batch_id for b in listed)

        first = batch["items"][0]
        assert client.get(first["result_url"]).status_code == 200
        assert client.get(first["source_url"]).status_code == 200

        full_resp = client.post(f"/api/style-batches/{batch_id}/items/{first['id']}/full", json={"format": "jpeg"})
        assert full_resp.status_code == 200, full_resp.text
        assert client.get(full_resp.json()["url"]).status_code == 200

        zip_resp = client.get(batch["zip_url"])
        assert zip_resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            names = sorted(zf.namelist())
            assert len(names) == 3
            assert all(n.endswith("-同风格.jpg") for n in names)

    appmod.reset_runtime()


@pytest.mark.skipif(discover_cli() is None, reason="lightcraft-cli not installed")
def test_real_lightcraft_series_sync_converges_exposure_and_warmth(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    from superphoto import app as appmod
    appmod.reset_runtime()
    settings, conn, job_id, _, _ = _seed_ready_anchor_job(appmod, tmp_path)
    binary = discover_cli()
    profile = style_sync.extract_style_profile(settings, conn, job_id, "E", binary=binary)

    # Two shots of the same scene at wildly different camera exposures & WB drift (~1.8 EV spread)
    under_cool = _make_photo(tmp_path / "under_cool.jpg", luma_scale=0.65, warmth=-0.05, size=(320, 220))
    over_warm = _make_photo(tmp_path / "over_warm.jpg", luma_scale=1.20, warmth=0.04, size=(320, 220))

    r1 = style_sync.apply_style_to_photo(settings, binary, profile, under_cool, tmp_path / "out1", edge=640)
    r2 = style_sync.apply_style_to_photo(settings, binary, profile, over_warm, tmp_path / "out2", edge=640)

    # Before sync their midtones were > 0.25 apart; after sync they converge tightly toward the anchor!
    before_gap = abs(r1["metrics"]["before_mid"] - r2["metrics"]["before_mid"])
    after_gap = abs(r1["metrics"]["after_mid"] - r2["metrics"]["after_mid"])
    assert before_gap > 0.22
    assert after_gap < 0.08
    assert r1["metrics"]["exposure_ev"] > r2["metrics"]["exposure_ev"]
    appmod.reset_runtime()
