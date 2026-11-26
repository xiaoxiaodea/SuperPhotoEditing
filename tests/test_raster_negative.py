from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from superphoto import denoise, directed, raster_negative, tone


def _make_blocky_sky_jpeg(path: Path) -> Path:
    """Create a compressed 8-bit JPEG with a smooth sky gradient + sharp dark silhouette edge."""
    h, w = 256, 256
    y = np.linspace(0.18, 0.55, h, dtype=np.float32)[:, None]
    x = np.linspace(-0.04, 0.04, w, dtype=np.float32)[None, :]
    r = np.clip((y * 0.65 + x) * 255, 0, 255)
    g = np.clip((y * 0.85 + x * 0.0) * 255, 0, 255)
    b = np.clip((y * 1.25 - x) * 255, 0, 255)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    # Sharp dark architectural silhouette in bottom third to verify edges stay sharp
    arr[170:, 80:180] = (18, 22, 26)
    Image.fromarray(arr).save(path, quality=52, subsampling=2)
    return path


def _make_gain_map_jpeg(path: Path) -> Path:
    """Construct a minimal valid ISO 21496-1 / Ultra HDR JPEG with embedded MPF gain map."""
    h, w = 128, 128
    base = np.full((h, w, 3), 180, dtype=np.uint8)
    base[:, 64:] = 255  # bright right half
    base_buf = io.BytesIO()
    Image.fromarray(base).save(base_buf, format="JPEG", quality=92)
    base_bytes = base_buf.getvalue()

    gm = np.zeros((h // 2, w // 2), dtype=np.uint8)
    gm[:, w // 4:] = 255  # +2.5 stops headroom on right half
    gm_buf = io.BytesIO()
    Image.fromarray(gm, mode="L").save(gm_buf, format="JPEG", quality=92)
    gm_bytes = gm_buf.getvalue()
    # Pad gain map so it is > 2KB (real gain maps are tens of KB)
    gm_bytes = gm_bytes[:-2] + b"\xff\xfe" + struct.pack(">H", 2400) + (b"\x00" * 2398) + b"\xff\xd9"

    xmp = (
        b"http://ns.adobe.com/xap/1.0/\x00"
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" '
        b'hdrgm:Version="1.0" hdrgm:GainMapMin="0.0" hdrgm:GainMapMax="2.5" '
        b'hdrgm:Gamma="1.0" hdrgm:OffsetSDR="0.015625" hdrgm:OffsetHDR="0.015625"/>'
        b"</rdf:RDF></x:xmpmeta>"
    )
    app1 = b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp
    # Insert APP1 right after SOI (\xff\xd8), append secondary gain map JPEG after EOI
    combined = base_bytes[:2] + app1 + base_bytes[2:] + gm_bytes
    path.write_bytes(combined)
    return path


def _read_f32_tiff_pixels(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    assert blob[:4] == b"II*\x00"
    ifd_offset = struct.unpack_from("<I", blob, 4)[0]
    count = struct.unpack_from("<H", blob, ifd_offset)[0]
    tags = {}
    for i in range(count):
        tag, typ, cnt, val = struct.unpack_from("<HHII", blob, ifd_offset + 2 + i * 12)
        tags[tag] = (typ, cnt, val)
    w = tags[256][2]
    h = tags[257][2]
    strip_off = tags[273][2]
    strip_len = tags[279][2]
    return np.frombuffer(blob[strip_off:strip_off + strip_len], dtype="<f4").reshape(h, w, 3)


def test_compressed_jpeg_negative_deblocks_and_preserves_edges(tmp_path: Path):
    src = _make_blocky_sky_jpeg(tmp_path / "sky.jpg")
    dest = tmp_path / "working_negative.tif"
    meta = raster_negative.prepare_raster_negative(src, dest)

    assert meta["enabled"] is True
    assert meta["source_kind"] == "jpeg"
    assert meta["bit_depth"] == 8
    assert meta["has_gain_map"] is False
    assert meta["chroma_radius"] >= 2
    assert meta["deband_cap_lsb"] > 0.0
    assert dest.is_file()

    pixels = _read_f32_tiff_pixels(dest)
    assert pixels.shape == (256, 256, 3)
    assert np.all(np.isfinite(pixels)) and pixels.min() >= 0.0
    # Sharp silhouette edge at row 170 must stay crisp (strong linear jump across boundary)
    above = pixels[168, 130].mean()
    below = pixels[172, 130].mean()
    assert above - below > 0.10 and above / max(below, 1e-4) > 10.0


def test_rgba_and_16bit_png_are_handled_cleanly(tmp_path: Path):
    # Transparent RGBA PNG
    rgba = np.zeros((96, 128, 4), dtype=np.uint8)
    rgba[..., 0] = 220
    rgba[..., 1] = 140
    rgba[..., 2] = 90
    rgba[..., 3] = 255
    rgba[:20, :20, 3] = 64
    rgba_path = tmp_path / "alpha.png"
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)

    meta_rgba = raster_negative.prepare_raster_negative(rgba_path, tmp_path / "alpha.tif")
    assert meta_rgba["source_kind"] == "png"
    assert meta_rgba["had_alpha"] is True
    assert meta_rgba["bit_depth"] == 8
    arr = _read_f32_tiff_pixels(tmp_path / "alpha.tif")
    assert arr.shape == (96, 128, 3)
    assert np.all(np.isfinite(arr))

    # 16-bit grayscale PNG preserves 16-bit depth without 8-bit deband cap
    g16 = np.linspace(0, 65535, 128 * 96, dtype=np.uint16).reshape(96, 128)
    p16_path = tmp_path / "grad16.png"
    Image.fromarray(g16).save(p16_path)

    meta_16 = raster_negative.prepare_raster_negative(p16_path, tmp_path / "grad16.tif")
    assert meta_16["bit_depth"] == 16
    assert meta_16["banding_risk"] == 0.0
    assert meta_16["deband_cap_lsb"] == 0.0


def test_partial_channel_highlight_reconstruction_adds_headroom_only_on_clipped_highlights():
    lin = np.zeros((64, 64, 3), dtype=np.float32)
    # Normal midtone region (must stay untouched)
    lin[:, :32] = (0.35, 0.28, 0.20)
    # Warm sunset highlight where R clipped to 1.0 while G/B remain unclipped
    lin[:, 32:] = (1.0, 0.72, 0.35)

    out, pct = raster_negative.reconstruct_partial_highlights(lin)
    assert pct > 40.0
    assert np.allclose(out[:, :32], lin[:, :32], atol=1e-6)
    assert out[:, 32:, 0].mean() > 1.25
    assert out.max() <= raster_negative.MAX_SYNTHETIC_HEADROOM + 1e-5


def test_hdr_gain_map_extraction_recovers_linear_headroom(tmp_path: Path):
    src = _make_gain_map_jpeg(tmp_path / "ultrahdr.jpg")
    dest = tmp_path / "ultrahdr.tif"
    meta = raster_negative.prepare_raster_negative(src, dest)

    assert meta["has_gain_map"] is True
    assert meta["gain_map_stops"] == pytest.approx(2.5, abs=0.05)
    assert meta["max_linear_headroom"] > 3.5
    pixels = _read_f32_tiff_pixels(dest)
    assert pixels[:, 80:, 0].mean() > 4.0
    assert pixels[:, :40, 0].mean() < 1.0


def test_hdr_gain_map_preserves_unclipped_misty_sky_without_blowout_or_crater(tmp_path: Path):
    """Smartphone HDR display-boost gain maps on unclipped skies must NOT blow out SDR tones
    or create a sharp cliff crater at bright sky corners."""
    h, w = 128, 128
    # Unclipped misty sky gradient (sRGB 190..250 -> linear 0.51..0.95) over dark forest (sRGB 50)
    base = np.full((h, w, 3), 50, dtype=np.uint8)
    sky = np.linspace(250, 190, h // 2, dtype=np.float32)[:, None]
    base[:h // 2, :, :] = np.broadcast_to(sky[..., None], (h // 2, w, 3)).astype(np.uint8)
    base_buf = io.BytesIO()
    Image.fromarray(base).save(base_buf, format="JPEG", quality=94)
    base_bytes = base_buf.getvalue()

    # Phone gain map paints +2.3 stops across the entire unclipped bright sky
    gm = np.zeros((h // 2, w // 2), dtype=np.uint8)
    gm[:h // 4, :] = 235
    gm_buf = io.BytesIO()
    Image.fromarray(gm, mode="L").save(gm_buf, format="JPEG", quality=92)
    gm_bytes = gm_buf.getvalue()
    gm_bytes = gm_bytes[:-2] + b"\xff\xfe" + struct.pack(">H", 2400) + (b"\x00" * 2398) + b"\xff\xd9"

    xmp = (
        b"http://ns.adobe.com/xap/1.0/\x00"
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" '
        b'hdrgm:Version="1.0" hdrgm:GainMapMin="0.0" hdrgm:GainMapMax="2.3" '
        b'hdrgm:Gamma="1.0" hdrgm:OffsetSDR="0.015625" hdrgm:OffsetHDR="0.015625"/>'
        b"</rdf:RDF></x:xmpmeta>"
    )
    app1 = b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp
    src = tmp_path / "misty_ultrahdr.jpg"
    src.write_bytes(base_bytes[:2] + app1 + base_bytes[2:] + gm_bytes)

    dest = tmp_path / "misty_ultrahdr.tif"
    meta = raster_negative.prepare_raster_negative(src, dest)
    assert meta["has_gain_map"] is False
    assert meta["max_linear_headroom"] <= 1.01
    pixels = _read_f32_tiff_pixels(dest)
    assert float(pixels.max()) <= 1.01
    # Top bright sky stays smoothly below 1.0 instead of blowing out to 4.7x
    assert 0.85 <= float(pixels[2, 64, 0]) <= 0.98

    # Also verify smooth sky transitioning into a small 2% clipped corner with uniform sky gain
    # yields zero discontinuous jump (excess over border is ~0).
    lin = np.full((128, 128, 3), 0.10, dtype=np.float32)
    lin[:64, :, :] = np.linspace(1.0, 0.85, 64, dtype=np.float32)[:, None, None]
    gm_uniform_sky = {
        "map": np.where(lin[..., 0] >= 0.84, 0.95, 0.0).astype(np.float32),
        "min_stops": 0.0, "max_stops": 2.3, "gamma": 1.0,
        "offset_sdr": 0.015625, "offset_hdr": 0.015625,
    }
    out_uniform = raster_negative.apply_gain_map(lin, gm_uniform_sky)
    assert float(out_uniform.max()) <= 1.01
    assert np.allclose(out_uniform, lin, atol=1e-5)


def test_raster_protection_applies_adaptive_nr_and_micro_grain():
    lifted_doc = {"light": {"exposure": 0.9, "shadows": 45}}
    meta = {"jpeg_blockiness": 0.25, "banding_risk": 0.38, "bit_depth": 8, "has_gain_map": False}

    # RAW is handled by AI Denoise, never raster_protection
    assert denoise.raster_protection(lifted_doc, "raw", meta) is None
    # Unlifted clean file needs no forced NR
    assert denoise.raster_protection({"light": {"exposure": 0.1}}, "jpeg", {"jpeg_blockiness": 0.02, "banding_risk": 0.1}) is None

    spec = denoise.raster_protection(lifted_doc, "jpeg", meta)
    assert spec is not None
    assert spec["nr_luminance"] >= 16.0
    assert spec["nr_color"] >= 22.0
    assert spec["grain"] >= 4.0

    # Never lowers a plan's existing stronger NR or grain
    existing = {**lifted_doc, "detail": {"nr_luminance": 40.0}, "grain": {"amount": 18.0, "size": 25.0}}
    merged = denoise.with_raster_protection(existing, spec)
    assert merged["detail"]["nr_luminance"] == 40.0
    assert merged["detail"]["nr_color"] == spec["nr_color"]
    assert merged["grain"]["amount"] == 18.0


def test_source_note_and_tone_limits_adapt_to_raster_negative_telemetry():
    note_hdr = directed._source_note("jpeg", {"enabled": True, "has_gain_map": True, "gain_map_stops": 2.4})
    assert "HDR Gain Map" in note_hdr and "+2.4 EV" in note_hdr

    note_png = directed._source_note("png", {"enabled": True, "bit_depth": 16, "reconstructed_pixels_pct": 1.2})
    assert "16-bit PNG" in note_png and "32-bit linear float" in note_png

    assert tone.shift_limits("jpeg", has_gain_map=True)["highlights"][1] > tone.shift_limits("jpeg")["highlights"][1]

    # Adaptive max_slope strictly bounds curve steepness
    curve, _ = tone.fit_curve({"black": 0.05, "shadows": 0.12, "mid": 0.25, "highlights": 0.70, "white": 0.92},
                              {"black": 0.02, "shadows": 0.08, "mid": 0.65, "highlights": 0.90, "white": 0.97},
                              max_slope=2.4)
    xs = np.array([p[0] for p in curve])
    ys = np.array([p[1] for p in curve])
    slopes = np.diff(ys) / np.diff(xs)
    assert float(slopes.max()) <= 2.45


def test_mask_contour_detector_and_feather_self_heal(tmp_path: Path):
    # Smooth gradient baseline vs hard stepped contour render (simulating razor-sharp luminanceRange cut)
    h, w = 128, 128
    base = np.broadcast_to(np.linspace(0.4, 0.7, w, dtype=np.float32)[None, :, None], (h, w, 3))
    base_img = Image.fromarray((base * 255).astype(np.uint8))

    broken = base.copy()
    broken[:, 48:80, :] -= 0.16  # harsh contour island in smooth skin/wall
    broken_path = tmp_path / "broken.jpg"
    Image.fromarray((np.clip(broken, 0, 1) * 255).astype(np.uint8)).save(broken_path, quality=95)

    smooth_path = tmp_path / "smooth.jpg"
    Image.fromarray((np.clip(base * 0.92, 0, 1) * 255).astype(np.uint8)).save(smooth_path, quality=95)

    assert directed._smooth_mask_contour_score(base_img, broken_path) >= directed.MASK_CONTOUR_THRESHOLD
    assert directed._smooth_mask_contour_score(base_img, smooth_path) < 0.05

    doc = {
        "masks": [{
            "id": 1, "name": "face",
            "components": [{"op": "intersect", "invert": False,
                            "shape": {"kind": "luminanceRange", "lo": 0.55, "hi": 0.80,
                                      "lo_feather": 0.02, "hi_feather": 0.025}}],
        }]
    }
    softened = directed._soften_tight_range_masks(doc)
    assert softened is not None
    sh = softened["masks"][0]["components"][0]["shape"]
    assert sh["lo_feather"] >= 0.14 and sh["hi_feather"] >= 0.14
    assert directed._soften_tight_range_masks(softened) is None

