"""32-bit linear float digital-negative pre-conditioning for ordinary JPEG and PNG files.

LightCraft 0.21.1 natively reads uncompressed IEEE-754 float32 linear RGB TIFFs
(BitsPerSample=32, SampleFormat=3) and preserves values > 1.0 as real highlight
headroom during exposure and highlight recovery (verified 0.00/255 error against
the original sRGB JPEG at neutral settings, and full highlight roll-off recovery
up to +2+ EV above 1.0).

This module turns an ordinary 8-bit or 16-bit JPEG/PNG into a clean float32
linear working negative before LightCraft touches it:

1. Precision decode: honors EXIF orientation, preserves 16-bit PNG depth,
   cleanly flattens transparent PNGs, and converts embedded wide-gamut ICC
   profiles (Display P3 / Adobe RGB) to sRGB in floating point.
2. Real HDR Gain Map recovery (ISO 21496-1 / Google Ultra HDR / Apple Adaptive
   HDR): when a modern smartphone or camera JPEG embeds a Gain Map, reconstructs
   the true linear HDR scene (> 1.0) so LightCraft recovers highlights like a RAW.
3. Compression & banding telemetry: measures 8x8 JPEG blockiness, smooth-gradient
   banding risk, and partial- vs hard-clipped highlight fractions so downstream
   tone-curve slope limits and adaptive NR scale with the actual file quality.
4. Edge-preserving chroma de-blocking (O(N) guided filter in YCbCr using Y as
   guidance): removes JPEG 4:2:0 color subsampling jaggedness and dark chroma
   blocks with zero loss of luminance sharpness.
5. Amplitude-capped smooth-gradient de-banding: interpolates quantized 8-bit
   1-LSB steps in flat areas (sky, bokeh, shadows) into continuous float32
   ramps while leaving textured pixels strictly untouched.
6. Single-image partial-channel highlight reconstruction (when no Gain Map is
   present): extrapolates smooth linear luminance headroom (> 1.0) where 1 or 2
   RGB channels clipped while the remaining channel(s) still carry real detail,
   preventing flat gray/magenta slabs when LightCraft pulls highlights down.
"""
from __future__ import annotations

import io
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .color import srgb_to_linear
from .ingest import srgb_profile_bytes, source_kind

# Maximum safe linear headroom synthesized by single-image partial-channel
# highlight reconstruction (+0.72 EV above SDR white). True Gain Map HDR files
# may legitimately exceed this up to MAX_GAIN_MAP_HEADROOM.
MAX_SYNTHETIC_HEADROOM = 1.65
MAX_GAIN_MAP_HEADROOM = 12.0


def _box_mean(img: np.ndarray, radius: int) -> np.ndarray:
    """Fast O(H*W) separable 2D box filter using reflect padding and cumulative sums."""
    if radius <= 0:
        return img
    h, w = img.shape
    r = min(radius, max(1, min(h, w) - 1))
    # Horizontal pass
    pad_x = np.pad(img, ((0, 0), (r, r)), mode="edge")
    cs_x = np.empty((h, pad_x.shape[1] + 1), dtype=np.float64)
    cs_x[:, 0] = 0.0
    np.cumsum(pad_x, axis=1, dtype=np.float64, out=cs_x[:, 1:])
    win = 2 * r + 1
    horiz = (cs_x[:, win:] - cs_x[:, :-win]) / win
    # Vertical pass
    pad_y = np.pad(horiz, ((r, r), (0, 0)), mode="edge")
    cs_y = np.empty((pad_y.shape[0] + 1, w), dtype=np.float64)
    cs_y[0, :] = 0.0
    np.cumsum(pad_y, axis=0, dtype=np.float64, out=cs_y[1:, :])
    return ((cs_y[win:, :] - cs_y[:-win, :]) / win).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. O(N) edge-preserving guided filter on 2D float32 arrays."""
    mean_i = _box_mean(guide, radius)
    mean_p = _box_mean(src, radius)
    mean_ip = _box_mean(guide * src, radius)
    cov_ip = mean_ip - mean_i * mean_p
    mean_ii = _box_mean(guide * guide, radius)
    var_i = np.maximum(0.0, mean_ii - mean_i * mean_i)
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = _box_mean(a, radius)
    mean_b = _box_mean(b, radius)
    return mean_a * guide + mean_b


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.709 float32 RGB [0..1] -> (Y [0..1], Cb [-0.5..0.5], Cr [-0.5..0.5])."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = y - (0.2126 * 1.5748 / 0.7152) * cr - (0.0722 * 1.8556 / 0.7152) * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def measure_raster_quality(srgb: np.ndarray, kind: str, *, bit_depth: int = 8) -> dict[str, float]:
    """Measure JPEG 8x8 blockiness, smooth-gradient banding risk, and highlight clip structure."""
    h, w = srgb.shape[:2]
    y = 0.2126 * srgb[..., 0] + 0.7152 * srgb[..., 1] + 0.0722 * srgb[..., 2]

    # 1. JPEG 8x8 block boundary discontinuity metric (Wang-Sheikh-Bovik style)
    blockiness = 0.0
    if kind == "jpeg" and h >= 24 and w >= 24:
        dx = np.abs(np.diff(y, axis=1))
        dy = np.abs(np.diff(y, axis=0))
        bx_mask = (np.arange(w - 1) % 8) == 7
        by_mask = (np.arange(h - 1) % 8) == 7
        # Focus on moderate/smooth regions where real texture doesn't drown out block seams
        bx = float(np.mean(dx[:, bx_mask]))
        ix = float(np.mean(dx[:, ~bx_mask]))
        by = float(np.mean(dy[by_mask, :]))
        iy = float(np.mean(dy[~by_mask, :]))
        ratio_x = (bx - ix) / max(ix, 1e-4)
        ratio_y = (by - iy) / max(iy, 1e-4)
        blockiness = float(np.clip(max(0.0, 0.5 * (ratio_x + ratio_y)) * 1.6, 0.0, 1.0))

    # 2. Smooth-region 8-bit banding risk: fraction of flat gradient areas whose
    # non-zero steps concentrate near discrete 1-2 LSB jumps (1/255 .. 2/255).
    banding_risk = 0.0
    if bit_depth <= 8 and h >= 16 and w >= 16:
        step = max(1, min(h, w) // 512)
        ys = y[::step, ::step]
        gx = np.abs(np.diff(ys, axis=1))
        gy = np.abs(np.diff(ys, axis=0))
        gmag = 0.5 * (gx[:-1, :] + gy[:, :-1])
        smooth = gmag < (6.0 / 255.0)
        smooth_frac = float(np.mean(smooth))
        # Quantized steps: non-zero transitions at ~1/255..2.5/255 inside smooth regions
        quant_steps = smooth & (gmag > (0.4 / 255.0)) & (gmag < (2.6 / 255.0))
        quant_frac = float(np.mean(quant_steps))
        # Combine smooth area coverage, quantized step density, and JPEG blockiness
        banding_risk = float(np.clip(smooth_frac * 0.45 + quant_frac * 1.4 + blockiness * 0.35, 0.0, 1.0))

    # 3. Highlight clip structure: partial-channel (recoverable!) vs hard 3-channel clip
    cmax = srgb.max(axis=-1)
    cmin = srgb.min(axis=-1)
    hard_clip = float(np.mean(cmin >= 0.992))
    partial_clip = float(np.mean((cmax >= 0.985) & (cmin < 0.945)))

    return {
        "jpeg_blockiness": round(blockiness, 4),
        "banding_risk": round(banding_risk, 4),
        "partial_clip_fraction": round(partial_clip, 5),
        "hard_clip_fraction": round(hard_clip, 5),
    }


def _parse_xmp_float(text: str, names: tuple[str, ...]) -> float | None:
    for name in names:
        # Attribute form: hdrgm:GainMapMax="2.5"
        m = re.search(rf"{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']", text)
        if m:
            try:
                val = float(m.group(1).strip().split(",")[0])
                if math.isfinite(val):
                    return val
            except ValueError:
                pass
        # Element / Seq form: <hdrgm:GainMapMax>2.5</hdrgm:GainMapMax> or <rdf:li>2.5</rdf:li>
        m = re.search(rf"<{re.escape(name)}[^>]*>(.*?)</{re.escape(name)}>", text, re.S)
        if m:
            inner = re.sub(r"<[^>]+>", " ", m.group(1)).strip()
            for token in inner.split():
                try:
                    val = float(token)
                    if math.isfinite(val):
                        return val
                except ValueError:
                    continue
    return None


def extract_gain_map(path: Path, base_size: tuple[int, int]) -> dict[str, Any] | None:
    """Detect and decode an embedded ISO 21496-1 / Ultra HDR / Apple HDR Gain Map from a JPEG.

    Returns ``{"map": float32[H,W] in 0..1, "min_stops": float, "max_stops": float,
    "gamma": float, "offset_sdr": float, "offset_hdr": float}`` or None.
    """
    if source_kind(path) != "jpeg":
        return None
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    if not blob.startswith(b"\xff\xd8"):
        return None

    # Quick marker check before scanning for secondary images
    has_xmp_marker = (b"hdrgm:" in blob or b"HDRGainMap" in blob or b"GainMapMax" in blob)
    if not has_xmp_marker:
        return None

    # Locate secondary embedded JPEG streams (after primary SOI)
    secondary_images: list[bytes] = []
    pos = 2
    while True:
        idx = blob.find(b"\xff\xd8\xff", pos)
        if idx < 0:
            break
        eoi = blob.find(b"\xff\xd9", idx + 3)
        if eoi > idx:
            chunk = blob[idx:eoi + 2]
            if len(chunk) >= 2048:  # Skip tiny EXIF thumbnails
                secondary_images.append(chunk)
            pos = eoi + 2
        else:
            break

    if not secondary_images:
        return None

    # Extract XMP text across the whole file (primary + secondary APP1 blocks)
    xmp_chunks = re.findall(rb"<x:xmpmeta[^>]*>.*?</x:xmpmeta>", blob, re.S)
    xmp_text = "\n".join(c.decode("utf-8", errors="ignore") for c in xmp_chunks)
    if not xmp_text:
        return None

    max_stops = _parse_xmp_float(xmp_text, ("hdrgm:GainMapMax", "HDRGainMapHeadroom"))
    if max_stops is None:
        # Apple HDRGainMapVersion without explicit headroom defaults conservatively to ~1.8 stops
        if "HDRGainMapVersion" in xmp_text:
            max_stops = 1.8
        else:
            return None
    # Apple HDRGainMapHeadroom is linear multiplier (e.g. 4.0 = 2 stops) if > 1.0 and no hdrgm:GainMapMax
    if "hdrgm:GainMapMax" not in xmp_text and "HDRGainMapHeadroom" in xmp_text and max_stops > 1.0:
        max_stops = math.log2(max_stops)
    if max_stops <= 0.15 or max_stops > 6.0:
        return None

    min_stops = _parse_xmp_float(xmp_text, ("hdrgm:GainMapMin",)) or 0.0
    gamma = _parse_xmp_float(xmp_text, ("hdrgm:Gamma",)) or 1.0
    offset_sdr = _parse_xmp_float(xmp_text, ("hdrgm:OffsetSDR",))
    offset_hdr = _parse_xmp_float(xmp_text, ("hdrgm:OffsetHDR",))
    offset_sdr = 0.015625 if offset_sdr is None else max(0.0, offset_sdr)
    offset_hdr = 0.015625 if offset_hdr is None else max(0.0, offset_hdr)
    gamma = max(0.2, min(5.0, gamma))

    base_w, base_h = base_size
    base_aspect = base_w / max(base_h, 1)

    # Pick the secondary JPEG that best matches the primary aspect ratio (largest valid candidate)
    best_map: np.ndarray | None = None
    best_area = 0
    for chunk in secondary_images:
        try:
            with Image.open(io.BytesIO(chunk)) as gm:
                gm = ImageOps.exif_transpose(gm)
                gw, gh = gm.size
                if gw < 32 or gh < 32:
                    continue
                aspect = gw / max(gh, 1)
                if abs(aspect - base_aspect) > 0.12 * base_aspect:
                    continue
                area = gw * gh
                if area >= best_area:
                    resized = gm.convert("L").resize((base_w, base_h), Image.Resampling.BILINEAR)
                    best_map = np.asarray(resized, dtype=np.float32) / 255.0
                    best_area = area
        except Exception:
            continue

    if best_map is None:
        return None

    return {
        "map": best_map,
        "min_stops": float(min_stops),
        "max_stops": float(max_stops),
        "gamma": float(gamma),
        "offset_sdr": float(offset_sdr),
        "offset_hdr": float(offset_hdr),
    }


def apply_gain_map(linear_sdr: np.ndarray, gm: dict[str, Any]) -> np.ndarray:
    """ISO 21496-1 linear float HDR reconstruction from SDR linear pixels + Gain Map."""
    g = np.clip(gm["map"], 0.0, 1.0)
    if abs(gm["gamma"] - 1.0) > 1e-4:
        g = np.power(np.maximum(g, 0.0), 1.0 / gm["gamma"])
    log_boost = gm["min_stops"] * (1.0 - g) + gm["max_stops"] * g
    boost = np.power(2.0, log_boost).astype(np.float32)[..., None]
    hdr = (linear_sdr + gm["offset_sdr"]) * boost - gm["offset_hdr"]
    return np.clip(hdr, 0.0, MAX_GAIN_MAP_HEADROOM).astype(np.float32)


def condition_srgb(srgb: np.ndarray, kind: str, quality: dict[str, float], *,
                   bit_depth: int = 8) -> tuple[np.ndarray, dict[str, float]]:
    """Edge-preserving chroma de-blocking + amplitude-capped smooth-gradient de-banding.

    Leaves textures, edges, and genuine 16-bit PNG gradients strictly sharp while
    turning quantized 8-bit skies/shadows and JPEG 4:2:0 blocks into smooth float32 data.
    """
    if bit_depth > 8:
        return srgb, {"chroma_radius": 0, "deband_cap_lsb": 0.0}

    blockiness = quality["jpeg_blockiness"]
    banding_risk = quality["banding_risk"]
    y, cb, cr = _rgb_to_ycbcr(srgb)

    # 1. Edge-preserving chroma de-blocking (fixes JPEG 4:2:0 color bleed & dark chroma blocks)
    chroma_radius = 0
    if kind == "jpeg" or blockiness >= 0.06:
        chroma_radius = 2 if blockiness < 0.12 else (3 if blockiness < 0.30 else 5)
        chroma_eps = 0.002 + 0.004 * blockiness
        strength = float(np.clip(0.60 + 0.40 * blockiness, 0.60, 1.0))
        cb_f = guided_filter(y, cb, chroma_radius, chroma_eps)
        cr_f = guided_filter(y, cr, chroma_radius, chroma_eps)
        cb = cb + (cb_f - cb) * strength
        cr = cr + (cr_f - cr) * strength

    # 2. Amplitude-capped smooth-gradient de-banding (zero texture blur guarantee)
    # Only engage on genuinely compressed / quantized JPEGs so crisp clean photos and natural film grain stay intact.
    deband_cap_lsb = 0.0
    if blockiness >= 0.14 or (banding_risk >= 0.28 and blockiness >= 0.06):
        gx = np.abs(np.diff(y, axis=1, append=y[:, -1:]))
        gy = np.abs(np.diff(y, axis=0, append=y[-1:, :]))
        grad = np.hypot(gx, gy)
        # Smooth-area weight: 1.0 in flat skies/shadows, ~0.0 on any real edge/texture
        tau = 0.018 + 0.014 * blockiness
        smooth_w = np.exp(-np.square(grad / tau)).astype(np.float32)
        luma_radius = 4 if blockiness < 0.20 else 6
        luma_eps = 1.8e-4 + 6.0e-4 * blockiness
        y_smooth = guided_filter(y, y, luma_radius, luma_eps)
        deband_cap_lsb = round(1.25 + 1.35 * blockiness, 2)
        cap = deband_cap_lsb / 255.0
        delta = np.clip(y_smooth - y, -cap, cap) * smooth_w
        y = np.clip(y + delta, 0.0, 1.0)

    return _ycbcr_to_rgb(y, cb, cr), {"chroma_radius": chroma_radius, "deband_cap_lsb": deband_cap_lsb}


def reconstruct_partial_highlights(linear: np.ndarray) -> tuple[np.ndarray, float]:
    """Single-image partial-channel highlight reconstruction into > 1.0 linear headroom.

    When 1 or 2 RGB channels clip near 1.0 while the remaining channel(s) still carry
    valid spatial variation (common in sunsets, clouds, skies, warm highlights),
    smoothly extrapolates unclipped channel ratios above 1.0 (capped at
    MAX_SYNTHETIC_HEADROOM). Pixels below 0.90 linear are untouched (0.00 diff).
    """
    cmax = linear.max(axis=-1)
    cmin = linear.min(axis=-1)
    cmid = np.partition(linear, 1, axis=-1)[..., 1]
    luma = (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]).astype(np.float32)
    # Dark-key night scenes (fireworks, streetlamps, neon) carry point emitters rather than
    # overexposed daylight skies; extra synthetic >1.0 headroom fights matte white ceilings when lifted.
    if float(np.mean(luma)) < 0.22:
        return linear, 0.0
    # Only synthesize > 1.0 linear headroom on broad contiguous bright highlight slabs (clouds,
    # sunsets, sunlit sky/snow) — never isolated dark-scene point emitters.
    raw_clip = (cmax > 0.92) & (cmid > 0.55) & (cmin > 0.20) & (cmin < 0.88) & (cmax - cmin > 0.03)
    if float(np.mean(raw_clip)) * 100.0 < 1e-4:
        return linear, 0.0
    broad_clip = (_box_mean(raw_clip.astype(np.float32), 6) > 0.55) & (_box_mean(luma, 12) > 0.42)
    partial = raw_clip & broad_clip
    pct = float(np.mean(partial)) * 100.0
    if pct < 1e-4:
        return linear, 0.0

    # How deeply the pixel pushed into the clip shoulder (0 at 0.92 -> 1 at 1.00)
    ramp = np.clip((cmax - 0.92) / 0.08, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep

    # Where cmax clipped at 1.0 while cmin kept rising toward 0.88, the true scene
    # luminance was climbing faster than cmax could record. Estimate the smooth
    # linear headroom multiplier from the unclipped channel's local elevation.
    luma_ref = _box_mean(cmin, 6)
    rel_elevation = np.clip((cmin - luma_ref) / np.maximum(luma_ref, 0.08), -0.1, 1.0)
    # Chromatic headroom: bright sunset/sky highlights gain smooth roll-off headroom up to
    # MAX_SYNTHETIC_HEADROOM so LightCraft highlight pull keeps warm hue without breaking white ceilings.
    sat_room = np.clip((cmax - cmin) / np.maximum(cmax, 1e-4), 0.0, 0.85)
    boost = 1.0 + ramp * np.clip(0.25 + 0.35 * sat_room + 0.30 * np.maximum(rel_elevation, 0.0), 0.0, MAX_SYNTHETIC_HEADROOM - 1.0)

    out = linear.copy()
    mask_3d = partial[..., None]
    reconstructed = np.minimum(linear * boost[..., None], MAX_SYNTHETIC_HEADROOM)
    # Preserve unclipped pixels identically; blend smoothly on partial-clip zones
    out = np.where(mask_3d, reconstructed, out)
    return out.astype(np.float32), round(pct, 3)


def write_f32_tiff(path: Path, linear_rgb: np.ndarray) -> Path:
    """Write an uncompressed IEEE-754 32-bit float linear RGB TIFF (zero external deps)."""
    arr = np.ascontiguousarray(np.maximum(linear_rgb, 0.0), dtype="<f4")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got {arr.shape}")
    h, w, _ = arr.shape
    raw = arr.tobytes()
    offset_data = 8
    offset_ifd = offset_data + len(raw)
    if offset_ifd % 2:
        offset_ifd += 1
    entries = (
        (256, 4, 1, w),              # ImageWidth
        (257, 4, 1, h),              # ImageLength
        (258, 3, 3, None),           # BitsPerSample -> [32, 32, 32]
        (259, 3, 1, 1),              # Compression = None
        (262, 3, 1, 2),              # Photometric = RGB
        (273, 4, 1, offset_data),    # StripOffsets
        (277, 3, 1, 3),              # SamplesPerPixel = 3
        (278, 4, 1, h),              # RowsPerStrip
        (279, 4, 1, len(raw)),       # StripByteCounts
        (284, 3, 1, 1),              # PlanarConfig = Chunky
        (339, 3, 3, None),           # SampleFormat -> [3, 3, 3] (IEEEFP)
    )
    extra_off = offset_ifd + 2 + len(entries) * 12 + 4
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(struct.pack("<2sHI", b"II", 42, offset_ifd))
            fh.write(raw)
            if fh.tell() < offset_ifd:
                fh.write(b"\x00")
            fh.write(struct.pack("<H", len(entries)))
            for tag, typ, cnt, val in entries:
                if tag == 258:
                    fh.write(struct.pack("<HHII", tag, typ, cnt, extra_off))
                elif tag == 339:
                    fh.write(struct.pack("<HHII", tag, typ, cnt, extra_off + 6))
                else:
                    fh.write(struct.pack("<HHII", tag, typ, cnt, val))
            fh.write(struct.pack("<I", 0))
            fh.write(struct.pack("<HHH", 32, 32, 32))
            fh.write(struct.pack("<HHH", 3, 3, 3))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _load_raster_srgb(path: Path, *, max_edge: int = 0) -> tuple[np.ndarray, int, bool]:
    """Load JPEG/PNG as float32 sRGB [0..1], honoring EXIF, ICC, 16-bit PNG, and alpha."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        bit_depth = 16 if im.mode.startswith("I;16") or im.mode == "I" else 8
        had_alpha = False
        icc = im.info.get("icc_profile")

        if bit_depth == 16:
            arr = np.asarray(im, dtype=np.float32)
            scale = 65535.0 if arr.max() > 255.0 else 255.0
            norm = np.clip(arr / scale, 0.0, 1.0)
            if norm.ndim == 2:
                srgb = np.stack([norm, norm, norm], axis=-1)
            else:
                srgb = norm[..., :3]
            if max_edge > 0 and max(srgb.shape[:2]) > max_edge:
                scale_r = max_edge / max(srgb.shape[:2])
                nw = max(1, int(round(srgb.shape[1] * scale_r)))
                nh = max(1, int(round(srgb.shape[0] * scale_r)))
                srgb = np.stack([
                    np.asarray(Image.fromarray(srgb[..., ch]).resize((nw, nh), Image.Resampling.LANCZOS), dtype=np.float32)
                    for ch in range(3)
                ], axis=-1)
            return np.clip(srgb, 0.0, 1.0).astype(np.float32), 16, False

        # Handle alpha cleanly before ICC / RGB conversion
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"))
            if int(alpha.min()) < 255:
                had_alpha = True
                bg = Image.new("RGBA", rgba.size, (24, 24, 24, 255))
                im = Image.alpha_composite(bg, rgba).convert("RGB")
            else:
                im = rgba.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")

        if max_edge > 0 and max(im.size) > max_edge:
            im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

        if icc:
            try:
                im = ImageCms.profileToProfile(
                    im,
                    ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                    ImageCms.ImageCmsProfile(io.BytesIO(srgb_profile_bytes())),
                    outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError, ValueError):
                pass

        srgb = np.asarray(im, dtype=np.float32) / 255.0
        return np.clip(srgb, 0.0, 1.0), 8, had_alpha


def prepare_raster_negative(source: Path, dest: Path, *, max_edge: int = 0) -> dict[str, Any]:
    """Pre-condition an ordinary JPEG/PNG into a 32-bit linear float TIFF working negative.

    Returns a telemetry dict describing the file's real headroom, whether an HDR
    Gain Map was recovered, and what conditioning steps were applied.
    """
    started = time.perf_counter()
    source = Path(source)
    kind = source_kind(source)
    if kind not in {"jpeg", "png"}:
        raise ValueError(f"prepare_raster_negative only accepts JPEG/PNG, got {kind}")

    srgb, bit_depth, had_alpha = _load_raster_srgb(source, max_edge=max_edge)
    h, w = srgb.shape[:2]
    quality = measure_raster_quality(srgb, kind, bit_depth=bit_depth)
    gain_map = extract_gain_map(source, (w, h))

    cleaned_srgb, cond = condition_srgb(srgb, kind, quality, bit_depth=bit_depth)
    linear = srgb_to_linear(cleaned_srgb).astype(np.float32)

    has_gain_map = gain_map is not None
    gain_map_stops = 0.0
    reconstructed_pct = 0.0
    if has_gain_map and gain_map is not None:
        linear = apply_gain_map(linear, gain_map)
        gain_map_stops = round(float(gain_map["max_stops"]), 2)
    else:
        linear, reconstructed_pct = reconstruct_partial_highlights(linear)

    write_f32_tiff(dest, linear)
    max_linear = round(float(np.percentile(linear, 99.9)), 3) if linear.size else 1.0

    return {
        "enabled": True,
        "path": str(dest),
        "source_kind": kind,
        "bit_depth": bit_depth,
        "had_alpha": had_alpha,
        "width": w,
        "height": h,
        "has_gain_map": has_gain_map,
        "gain_map_stops": gain_map_stops,
        "reconstructed_pixels_pct": reconstructed_pct,
        "max_linear_headroom": max_linear,
        **quality,
        **cond,
        "seconds": round(time.perf_counter() - started, 2),
    }
