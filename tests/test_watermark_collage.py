from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from superphoto import collage, watermark
from superphoto.db import (
    insert_job,
    insert_style_batch,
    insert_style_batch_item,
    refresh_style_batch_counts,
    replace_variants,
    update_job,
    update_style_batch_item,
)
from superphoto.ingest import read_exif


def _make_photo_with_exif(
    path: Path,
    *,
    size: tuple[int, int] = (900, 600),
    rgb_bias: tuple[float, float, float] = (0.55, 0.45, 0.35),
    make: str = "NIKON CORPORATION",
    model: str = "NIKON Z 7_2",
    lens: str = "NIKKOR Z 24-120mm f/4 S",
    focal: float = 120.0,
    fnumber: float = 4.0,
    exposure: float = 1 / 400,
    iso: int = 400,
    dt: str = "2024:07:04 16:35:23",
    with_exif: bool = True,
) -> Path:
    w, h = size
    yy = np.linspace(0.2, 0.85, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.15, 0.9, w, dtype=np.float32)[None, :]
    r = np.clip((yy * rgb_bias[0] + xx * 0.35) * 255.0, 0, 255)
    g = np.clip((yy * rgb_bias[1] + xx * 0.25) * 255.0, 0, 255)
    b = np.clip((yy * rgb_bias[2] + (1.0 - xx) * 0.35) * 255.0, 0, 255)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(im)
    draw.ellipse((w * 0.3, h * 0.25, w * 0.7, h * 0.75), outline=(245, 240, 225), width=4)

    path.parent.mkdir(parents=True, exist_ok=True)
    if with_exif:
        exif = Image.Exif()
        exif[271] = make
        exif[272] = model
        exif[306] = dt
        exif_ifd = exif.get_ifd(0x8769)
        exif_ifd[33434] = exposure
        exif_ifd[33437] = fnumber
        exif_ifd[34855] = iso
        exif_ifd[36867] = dt
        exif_ifd[37386] = focal
        exif_ifd[42036] = lens
        im.save(path, format="JPEG", quality=92, exif=exif.tobytes())
    else:
        im.save(path, format="JPEG", quality=92)
    return path


def test_exif_normalization_and_brand_detection(tmp_path):
    photo = _make_photo_with_exif(
        tmp_path / "nikon_shot.jpg",
        make="NIKON CORPORATION",
        model="NIKON Z 7_2",
        lens="NIKKOR Z 24-120mm f/4 S",
        focal=120.0,
        fnumber=2.8,
        exposure=1 / 1250,
        iso=800,
        dt="2025:10:12 18:22:09",
    )
    raw = read_exif(photo)
    assert raw.get("Model") == "NIKON Z 7_2"
    assert raw.get("ISOSpeedRatings") == "800"

    meta = watermark.extract_photo_metadata(photo)
    assert meta["brand"] == "nikon"
    assert meta["camera"] == "NIKON Z 7II"
    assert meta["lens"] == "NIKKOR Z 24-120mm f/4 S"
    assert meta["focal_length"] == "120mm"
    assert meta["aperture"] == "f/2.8"
    assert meta["shutter"] == "1/1250s"
    assert meta["iso"] == "ISO 800"
    assert meta["datetime"] == "2025.10.12 18:22"
    assert meta["has_exif"] is True

    # Rational formatting edge cases
    assert watermark.format_aperture("14/5") == "f/2.8"
    assert watermark.format_aperture("9/5") == "f/1.8"
    assert watermark.format_aperture("4") == "f/4"
    assert watermark.format_shutter("1/400") == "1/400s"
    assert watermark.format_shutter("10/4000") == "1/400s"
    assert watermark.format_shutter("1/2") == "1/2s"
    assert watermark.format_shutter("25/10") == "2.5s"
    assert watermark.format_shutter("30") == "30s"
    assert watermark.format_focal_length("240/10") == "24mm"
    assert watermark.format_ev_bias("-2/3") == "-0.7 EV"
    assert watermark.format_ev_bias("0/6") == ""

    # Body brand always wins over adapted third-party lens brand
    assert watermark.detect_brand("Canon", "Canon EOS R5", "NIKKOR Z 24-120mm f/4 S") == "canon"
    assert watermark.detect_brand("SONY", "ILCE-7M4", "SIGMA 35mm F1.4 DG DN") == "sony"

    gps_meta = watermark.normalize_exif({
        "Make": "Leica Camera AG",
        "Model": "LEICA Q3",
        "FNumber": "17/10",
        "ExposureTime": "1/500",
        "ISOSpeedRatings": "100",
        "FocalLength": "28",
        "GPSLatitude": "[35, 41, 2245/100]",
        "GPSLatitudeRef": "N",
        "GPSLongitude": "[139, 41, 3012/100]",
        "GPSLongitudeRef": "E",
    })
    assert gps_meta["brand"] == "leica"
    assert gps_meta["camera"] == "LEICA Q3"
    assert gps_meta["aperture"] == "f/1.7"
    assert gps_meta["gps"] == "35°41'N 139°41'E"


@pytest.mark.parametrize("style", [s["id"] for s in watermark.WATERMARK_STYLES])
def test_all_watermark_styles_render_cleanly_with_latin_and_cjk(tmp_path, style):
    photo = _make_photo_with_exif(tmp_path / f"{style}.jpg", size=(800, 540))
    meta = watermark.extract_photo_metadata(photo)
    cfg = watermark.WatermarkConfig(
        style=style,
        location="东京 · 涩谷",
        signature="摄影师测试",
        show_ev=True,
        ev_bias="+0.3 EV",
    )
    with Image.open(photo) as im:
        out = watermark.render_watermark(im, meta, cfg, max_edge=800)

    assert out.mode == "RGB"
    if style == "corner_overlay":
        assert out.size == (800, 540)
    elif style in ("leica_bar", "dark_bar"):
        assert out.width == 800 and out.height > 540
    else:
        assert out.width > 800 and out.height > 540

    # Verify JPEG encoding with sRGB ICC profile succeeds
    jpeg = watermark.encode_jpeg_bytes(out)
    assert jpeg[:2] == b"\xff\xd8" and len(jpeg) > 5000


def test_watermark_handles_missing_exif_gracefully(tmp_path):
    photo = _make_photo_with_exif(tmp_path / "no_exif.jpg", size=(600, 800), with_exif=False)
    meta = watermark.extract_photo_metadata(photo)
    assert meta["has_exif"] is False
    with Image.open(photo) as im:
        out = watermark.render_watermark(im, meta, watermark.WatermarkConfig(style="leica_bar"))
    assert out.width == 600 and out.height > 800


def test_auto_arrange_modes_and_collage_layouts_and_slicer(tmp_path):
    # Create 9 varied photos (different colors, brightness, contrast, timestamps)
    items = []
    images = []
    metas = []
    specs = [
        ((0.20, 0.30, 0.80), "2025:05:01 18:00:00", (640, 480)),
        ((0.85, 0.55, 0.25), "2025:05:01 09:10:00", (480, 640)),
        ((0.50, 0.65, 0.45), "2025:05:01 14:20:00", (600, 600)),
        ((0.90, 0.70, 0.30), "2025:05:01 10:00:00", (640, 420)),
        ((0.15, 0.20, 0.35), "2025:05:01 20:30:00", (640, 480)),
        ((0.75, 0.40, 0.35), "2025:05:01 16:40:00", (500, 650)),
        ((0.35, 0.55, 0.75), "2025:05:01 12:15:00", (640, 480)),
        ((0.65, 0.60, 0.50), "2025:05:01 11:05:00", (640, 480)),
        ((0.95, 0.85, 0.75), "2025:05:01 08:30:00", (640, 480)),
    ]
    for idx, (bias, dt, sz) in enumerate(specs):
        p = _make_photo_with_exif(tmp_path / f"p{idx}.jpg", size=sz, rgb_bias=bias, dt=dt, iso=100 * (idx + 1))
        meta = watermark.extract_photo_metadata(p)
        im = Image.open(p).convert("RGB")
        aes = collage.analyze_tile_aesthetics(im, meta)
        items.append({"id": f"id_{idx}", "metadata": meta, "aesthetics": aes})
        images.append(im)
        metas.append(meta)

    # Chronological must place 08:30 (id_8) first and 20:30 (id_4) last
    chrono_ids, reason = collage.auto_arrange_items(items, mode="chronological")
    assert chrono_ids[0] == "id_8" and chrono_ids[-1] == "id_4"
    assert "时间" in reason

    # Hero center must place highest hero_score at index 4 (center of 9-grid)
    hero_ids, _ = collage.auto_arrange_items(items, mode="hero_center")
    best_hero = max(items, key=lambda it: it["aesthetics"]["hero_score"])["id"]
    assert hero_ids[4] == best_hero
    assert set(hero_ids) == {it["id"] for it in items}

    # Color harmony & balanced preserve all items
    for mode in ("color_harmony", "balanced"):
        ordered, _ = collage.auto_arrange_items(items, mode=mode)
        assert len(ordered) == 9 and set(ordered) == {it["id"] for it in items}

    # All 9 collage layouts render without error
    for layout_info in collage.COLLAGE_LAYOUTS:
        lid = layout_info["id"]
        cfg = collage.CollageConfig(layout=lid, show_footer=True, tile_watermark=(lid == "vertical_story"))
        rendered = collage.render_collage(images, metas, cfg, target_width=1200)
        assert rendered.width >= 600 and rendered.height >= 400

    # Moments 9-grid ZIP export: multi-photo with per-tile watermark
    zip_bytes = collage.export_moments_zip_bytes(
        images,
        metas,
        collage.CollageConfig(layout="grid_9", tile_watermark=True, tile_frame_mode="padded_square"),
        tile_edge=512,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = sorted(zf.namelist())
        assert len(names) == 9
        assert names[0].endswith("-01.jpg") and names[-1].endswith("-09.jpg")
        with Image.open(io.BytesIO(zf.read(names[0]))) as tile0:
            assert tile0.size == (512, 512)

    # Single-photo 1->9 slicer ZIP export
    slice_zip = collage.export_moments_zip_bytes(
        [images[0]],
        [metas[0]],
        collage.CollageConfig(layout="slice_9", slice_offset_x=0.3, slice_offset_y=-0.2),
        tile_edge=400,
    )
    with zipfile.ZipFile(io.BytesIO(slice_zip)) as zf:
        assert len(zf.namelist()) == 9


def test_studio_api_end_to_end_upload_import_watermark_collage_and_moments(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    from superphoto import app as appmod

    appmod.reset_runtime()
    settings, conn = appmod._boot()

    # Seed one ready job + one ready style batch with real EXIF on their originals
    job_id = "exif-job-1"
    jroot = settings.jobs_dir / job_id
    jroot.mkdir(parents=True, exist_ok=True)
    _make_photo_with_exif(jroot / "original.jpg", make="SONY", model="ILCE-7M4", lens="FE 35mm F1.4 GM", focal=35.0, fnumber=1.4, exposure=1 / 1000, iso=200)
    _make_photo_with_exif(jroot / "source_preview.jpg", with_exif=False)
    _make_photo_with_exif(jroot / "variant_A.jpg", with_exif=False)
    insert_job(conn, job_id, "street_tokyo.jpg", "original.jpg")
    update_job(conn, job_id, status="ready", step="ready")
    replace_variants(conn, job_id, [{
        "id": str(uuid.uuid4()),
        "slot": "A",
        "look_id": "natural",
        "title": "自然通透",
        "why": "test",
        "recipe_json": json.dumps({}),
        "filename": "variant_A.jpg",
    }])

    batch_id = "batch-moments-1"
    insert_style_batch(conn, batch_id, job_id, "A", None, "自然通透", {"title": "自然通透"}, 2)
    for k in range(2):
        iid = f"item-{k}"
        idir = settings.data_dir / "style-batches" / batch_id / "items" / iid
        _make_photo_with_exif(idir / "original.jpg", make="FUJIFILM", model="X-T5", focal=23.0, fnumber=2.0, exposure=1 / 500, iso=320 + k * 100)
        _make_photo_with_exif(idir / "result" / "result.jpg", with_exif=False)
        insert_style_batch_item(conn, iid, batch_id, f"fuji_{k}.jpg", "original.jpg")
        update_style_batch_item(conn, iid, status="ready", step="ready", output_filename="result/result.jpg")
    refresh_style_batch_counts(conn, batch_id)

    u1 = _make_photo_with_exif(tmp_path / "u1.jpg", make="Canon", model="Canon EOS R5", focal=85.0, fnumber=1.2, exposure=1 / 800, iso=100)
    u2 = _make_photo_with_exif(tmp_path / "u2.jpg", make="Leica Camera AG", model="LEICA Q3", focal=28.0, fnumber=1.7, exposure=1 / 250, iso=400)

    with TestClient(appmod.app) as client:
        meta_resp = client.get("/api/studio/meta")
        assert meta_resp.status_code == 200
        assert len(meta_resp.json()["watermark_styles"]) == 6
        assert len(meta_resp.json()["collage_layouts"]) == 9

        # Sources lists both the job and the style batch
        sources = client.get("/api/studio/sources").json()
        assert any(j["job_id"] == job_id for j in sources["jobs"])
        assert any(b["batch_id"] == batch_id for b in sources["batches"])

        # Job EXIF endpoint extracts from original.jpg even though variant_A.jpg had no EXIF
        exif_resp = client.get(f"/api/studio/jobs/{job_id}/exif?slot=A")
        assert exif_resp.status_code == 200
        jmeta = exif_resp.json()["metadata"]
        assert jmeta["brand"] == "sony"
        assert jmeta["camera"] == "SONY ILCE-7M4"
        assert jmeta["aperture"] == "f/1.4"
        assert jmeta["shutter"] == "1/1000s"
        assert jmeta["iso"] == "ISO 200"

        # Upload local photos
        up = client.post(
            "/api/studio/upload",
            files=[
                ("files", ("u1.jpg", u1.read_bytes(), "image/jpeg")),
                ("files", ("u2.jpg", u2.read_bytes(), "image/jpeg")),
            ],
        )
        assert up.status_code == 200, up.text
        uploaded = up.json()["assets"]
        assert len(uploaded) == 2
        assert uploaded[0]["metadata"]["brand"] == "canon"
        assert uploaded[1]["metadata"]["brand"] == "leica"

        # Import style batch (anchor + 2 synced items = 3 assets, each with its own EXIF!)
        imp_b = client.post("/api/studio/import-batch", json={"batch_id": batch_id, "include_anchor": True})
        assert imp_b.status_code == 200, imp_b.text
        batch_assets = imp_b.json()["assets"]
        assert len(batch_assets) == 3
        assert batch_assets[0]["metadata"]["brand"] == "sony"
        assert batch_assets[1]["metadata"]["brand"] == "fujifilm"
        assert batch_assets[1]["metadata"]["iso"] == "ISO 320"
        assert batch_assets[2]["metadata"]["iso"] == "ISO 420"

        # Watermark preview + full export + batch ZIP
        wm_prev = client.post("/api/studio/watermark/preview", json={
            "asset_id": uploaded[0]["id"],
            "config": {"style": "blur_card", "signature": "测试签名"},
        })
        assert wm_prev.status_code == 200 and wm_prev.headers["content-type"] == "image/jpeg"

        wm_exp = client.post("/api/studio/watermark/export", json={
            "job_id": job_id,
            "slot": "A",
            "config": {"style": "leica_bar"},
        })
        assert wm_exp.status_code == 200 and "attachment" in wm_exp.headers["content-disposition"]

        wm_zip = client.post("/api/studio/watermark/batch-zip", json={
            "asset_ids": [a["id"] for a in batch_assets],
            "config": {"style": "dark_bar"},
        })
        assert wm_zip.status_code == 200
        with zipfile.ZipFile(io.BytesIO(wm_zip.content)) as zf:
            assert len(zf.namelist()) == 3

        # Auto-arrange + Moments preview + Collage preview + Moments ZIP export
        all_ids = [a["id"] for a in [*batch_assets, *uploaded]]
        arr = client.post("/api/studio/collage/auto-arrange", json={"asset_ids": all_ids, "mode": "hero_center"})
        assert arr.status_code == 200 and len(arr.json()["ordered_ids"]) == 5

        mom = client.post("/api/studio/collage/moments-preview", json={
            "asset_ids": all_ids,
            "config": {"layout": "grid_9", "tile_watermark": True},
        })
        assert mom.status_code == 200
        mdata = mom.json()
        assert mdata["count"] == 5 and mdata["cols"] == 3
        assert len(mdata["tiles"]) == 5 and mdata["tiles"][0].startswith("data:image/jpeg;base64,")
        assert mdata["suggested_caption"]

        c_prev = client.post("/api/studio/collage/preview", json={
            "asset_ids": all_ids,
            "config": {"layout": "editorial", "background": "ivory", "show_footer": True},
        })
        assert c_prev.status_code == 200 and c_prev.headers["content-type"] == "image/jpeg"

        c_zip = client.post("/api/studio/collage/export-moments-zip", json={
            "asset_ids": [uploaded[0]["id"]],
            "config": {"layout": "slice_9", "slice_offset_x": 0.1},
            "tile_edge": 400,
        })
        assert c_zip.status_code == 200
        with zipfile.ZipFile(io.BytesIO(c_zip.content)) as zf:
            assert len(zf.namelist()) == 9

    appmod.reset_runtime()
