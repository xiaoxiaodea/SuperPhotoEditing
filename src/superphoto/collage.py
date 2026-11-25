"""Automatic photo collage, smart aesthetic sorting, and WeChat Moments 9-grid engine.

Supports:
- Multi-photo grids (3x3 Moments 9-grid, 2x2 4-grid, 3x2 6-grid, cinematic 3x1 triptych)
- Aspect-aware auto editorial magazine masonry & vertical EXIF story strips
- Single-photo 9-grid / 4-grid / 6-grid slicing with adjustable focal framing
- Aesthetic auto-arrangement (hero-center symmetry, color harmony flow, chronological EXIF, balanced)
- Unified high-res collage JPEG export AND numbered WeChat Moments tiles ZIP (01.jpg .. 09.jpg)
  with optional per-tile EXIF watermark or clean square-padded framing.
"""
from __future__ import annotations

import colorsys
import math
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from .ingest import image_to_srgb
from .watermark import (
    WatermarkConfig,
    _draw_text_ycenter,
    _fit_font,
    _hex_to_rgb,
    encode_jpeg_bytes,
    format_params_line,
    render_watermark,
)

CollageLayout = Literal[
    "grid_9",
    "grid_4",
    "grid_6",
    "triptych",
    "editorial",
    "vertical_story",
    "slice_9",
    "slice_4",
    "slice_6",
]

ArrangeMode = Literal["hero_center", "color_harmony", "chronological", "balanced"]
BackgroundPreset = Literal["white", "dark", "ivory", "blur"]
TileFrameMode = Literal["cover_square", "padded_square", "original"]

COLLAGE_LAYOUTS: list[dict[str, Any]] = [
    {"id": "grid_9", "name": "朋友圈九宫格", "subtitle": "3×3 方格 · 最多 9 张", "slice": False},
    {"id": "grid_4", "name": "四宫格", "subtitle": "2×2 方格 · 4 张精选", "slice": False},
    {"id": "grid_6", "name": "六宫格", "subtitle": "3×2 网格 · 6 张组图", "slice": False},
    {"id": "triptych", "name": "电影三联画", "subtitle": "3×1 宽幅叠画 · 故事感", "slice": False},
    {"id": "editorial", "name": "智能杂志拼图", "subtitle": "自适应横竖构图 · 零浪费", "slice": False},
    {"id": "vertical_story", "name": "纵向故事长图", "subtitle": "逐图保留原比例与参数", "slice": False},
    {"id": "slice_9", "name": "单图切九宫格", "subtitle": "1 张大片切 9 格发朋友圈", "slice": True},
    {"id": "slice_4", "name": "单图切四宫格", "subtitle": "1 张照片切 2×2 四宫格", "slice": True},
    {"id": "slice_6", "name": "单图切六宫格", "subtitle": "1 张横图切 3×2 六宫格", "slice": True},
]

BG_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "dark": (18, 19, 22),
    "ivory": (246, 243, 236),
}


class CollageConfig(BaseModel):
    layout: CollageLayout = "grid_9"
    background: str = "white"  # white | dark | ivory | blur | #RRGGBB
    gap: int = Field(default=14, ge=0, le=80)
    padding: int = Field(default=28, ge=0, le=140)
    radius: int = Field(default=6, ge=0, le=60)
    tile_frame_mode: TileFrameMode = "cover_square"
    # Single-photo slice focal offset (-1 = left/top, 0 = center, +1 = right/bottom)
    slice_offset_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    slice_offset_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Optional unified header/footer strip on combined collage
    show_footer: bool = False
    footer_title: str = ""
    footer_subtitle: str = ""
    # Optional per-tile EXIF watermark (applied on Moments ZIP tiles or vertical_story / padded tiles)
    tile_watermark: bool = False
    watermark_config: WatermarkConfig = Field(default_factory=WatermarkConfig)


# ---- Aesthetic analysis & smart auto-arrangement ---------------------------


def analyze_tile_aesthetics(image: Image.Image, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract fast visual features for smart collage arrangement."""
    rgb_im = image_to_srgb(ImageOps.exif_transpose(image))
    w, h = rgb_im.size
    aspect = round(w / max(h, 1), 3)
    orientation = "square" if 0.9 <= aspect <= 1.12 else ("landscape" if aspect > 1.12 else "portrait")

    thumb = rgb_im.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32) / 255.0
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    mean_luma = float(np.mean(luma))
    contrast = float(np.std(luma))

    # Local gradient energy (edge/detail richness) to help detect visually strong hero frames
    gx = np.abs(np.diff(luma, axis=1)).mean()
    gy = np.abs(np.diff(luma, axis=0)).mean()
    detail = float(gx + gy)

    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    warmth = float(r - b)

    # Center saliency: how visually distinct the center third is from the outer border
    center = luma[28:68, 28:68]
    border = np.concatenate([luma[:20, :].ravel(), luma[-20:, :].ravel(), luma[:, :20].ravel(), luma[:, -20:].ravel()])
    center_pop = abs(float(center.mean()) - float(border.mean())) + float(center.std()) * 0.5

    hero_score = round(contrast * 2.2 + detail * 2.6 + sat * 0.45 + center_pop * 1.1, 4)
    return {
        "width": w,
        "height": h,
        "aspect": aspect,
        "orientation": orientation,
        "mean_luma": round(mean_luma, 4),
        "contrast": round(contrast, 4),
        "warmth": round(warmth, 4),
        "hue": round(hue, 4),
        "saturation": round(sat, 4),
        "hero_score": hero_score,
        "dominant_hex": f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}",
        "datetime": str((meta or {}).get("datetime") or ""),
    }


def auto_arrange_items(
    items: list[dict[str, Any]],
    mode: ArrangeMode = "hero_center",
) -> tuple[list[str], str]:
    """Return ``(ordered_ids, chinese_reason)`` for a list of studio item dicts.

    Each dict must contain ``id`` and ``aesthetics`` (from ``analyze_tile_aesthetics``)
    and optionally ``metadata`` with ``datetime``.
    """
    n = len(items)
    if n <= 1:
        return [it["id"] for it in items], "照片数量不足 2 张，保持原顺序。"

    def aes(it: dict[str, Any]) -> dict[str, Any]:
        return it.get("aesthetics") or {}

    if mode == "chronological":
        def ts_key(idx_it: tuple[int, dict[str, Any]]) -> tuple[int, str, int]:
            idx, it = idx_it
            dt = str((it.get("metadata") or {}).get("datetime") or aes(it).get("datetime") or "").strip()
            return (0 if dt else 1, dt, idx)

        ordered = [it["id"] for _, it in sorted(enumerate(items), key=ts_key)]
        with_dt = sum(1 for it in items if (it.get("metadata") or {}).get("datetime") or aes(it).get("datetime"))
        if with_dt >= 2:
            return ordered, f"已按 EXIF 拍摄时间先后排序（识别到 {with_dt}/{n} 张拍摄时间）。"
        return ordered, "大部分照片缺少 EXIF 时间信息，已按原顺序保留。"

    if mode == "color_harmony":
        # Smooth 1D perceptual color flow: sort along primary color axis (warmth + hue + brightness)
        def color_key(it: dict[str, Any]) -> tuple[float, float]:
            a = aes(it)
            sat = float(a.get("saturation", 0.0))
            warm = float(a.get("warmth", 0.0))
            luma = float(a.get("mean_luma", 0.5))
            # Blend hue/warmth with luminance so low-sat frames still transition smoothly
            tone_axis = warm * 1.6 + (float(a.get("hue", 0.0)) - 0.5) * sat * 0.4 + luma * 0.45
            return (tone_axis, luma)

        ordered = [it["id"] for it in sorted(items, key=color_key)]
        return ordered, "已按画面冷暖色相与明暗过渡顺色排列，整组视觉更连贯。"

    if mode == "balanced":
        # Interleave bright/dark and warm/cool frames so heavy corners don't clump
        by_luma = sorted(items, key=lambda it: float(aes(it).get("mean_luma", 0.5)))
        out: list[dict[str, Any]] = []
        lo, hi = 0, n - 1
        take_hi = True
        while lo <= hi:
            if take_hi:
                out.append(by_luma[hi])
                hi -= 1
            else:
                out.append(by_luma[lo])
                lo += 1
            take_hi = not take_hi
        return [it["id"] for it in out], "已按明暗节奏交错排布，避免视觉重心偏沉在某一侧。"

    # Default: "hero_center" — place strongest visual hero at the focal center, balance corners around it
    ranked = sorted(items, key=lambda it: float(aes(it).get("hero_score", 0.0)), reverse=True)
    hero = ranked[0]
    rest = ranked[1:]
    # Sort remaining by warmth/luma so symmetric pairs sit on opposite edges of the grid
    rest_sorted = sorted(
        rest,
        key=lambda it: (float(aes(it).get("warmth", 0.0)) + float(aes(it).get("mean_luma", 0.5)) * 0.5),
    )
    center_idx = n // 2 if n in (3, 5, 7, 9) else 0
    slots: list[dict[str, Any] | None] = [None] * n
    slots[center_idx] = hero
    # Fill remaining indices from outside-in symmetrically
    remaining_indices = [i for i in range(n) if i != center_idx]
    pair_order: list[int] = []
    l_ptr, r_ptr = 0, len(remaining_indices) - 1
    while l_ptr <= r_ptr:
        pair_order.append(remaining_indices[l_ptr])
        if l_ptr != r_ptr:
            pair_order.append(remaining_indices[r_ptr])
        l_ptr += 1
        r_ptr -= 1
    for slot_idx, it in zip(pair_order, rest_sorted):
        slots[slot_idx] = it
    ordered = [it["id"] for it in slots if it is not None]
    pos_text = f"正中第 {center_idx + 1} 格" if center_idx > 0 else "视觉首位"
    return ordered, f"已挑选视觉张力最强的一张置于{pos_text}，其余照片按明暗冷暖对称呼应。"


# ---- Image cropping / framing helpers --------------------------------------


def _cover_crop(im: Image.Image, tw: int, th: int, ox: float = 0.0, oy: float = 0.0) -> Image.Image:
    """Resize and crop ``im`` to fill ``(tw, th)`` with optional normalized focal offset ``(-1..1)``."""
    w, h = im.size
    target_ratio = tw / max(th, 1)
    src_ratio = w / max(h, 1)
    if src_ratio > target_ratio:
        crop_h = h
        crop_w = max(1, int(round(h * target_ratio)))
    else:
        crop_w = w
        crop_h = max(1, int(round(w / target_ratio)))
    max_x = max(0, w - crop_w)
    max_y = max(0, h - crop_h)
    x0 = int(round(max_x * (0.5 + 0.5 * max(-1.0, min(1.0, ox)))))
    y0 = int(round(max_y * (0.5 + 0.5 * max(-1.0, min(1.0, oy)))))
    cropped = im.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return cropped.resize((tw, th), Image.Resampling.LANCZOS)


def _pad_into_square(
    im: Image.Image,
    size: int,
    bg_rgb: tuple[int, int, int],
    *,
    margin_ratio: float = 0.06,
) -> Image.Image:
    """Letterbox/pillarbox ``im`` cleanly inside a ``size x size`` square preserving 100% of frame."""
    canvas = Image.new("RGB", (size, size), bg_rgb)
    margin = int(round(size * margin_ratio))
    avail = max(16, size - margin * 2)
    w, h = im.size
    scale = min(avail / max(w, 1), avail / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _paste_rounded(canvas: Image.Image, tile: Image.Image, box: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        canvas.paste(tile, box)
        return
    r = min(radius, tile.width // 2, tile.height // 2)
    mask = Image.new("L", tile.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile.width - 1, tile.height - 1), radius=r, fill=255)
    canvas.paste(tile, box, mask)


def _make_canvas(
    cw: int,
    ch: int,
    background: str,
    seed_image: Image.Image | None,
) -> tuple[Image.Image, bool]:
    """Return ``(canvas, is_dark)``."""
    bg_key = (background or "white").strip().lower()
    if bg_key == "blur" and seed_image is not None:
        sw, sh = max(48, cw // 10), max(48, ch // 10)
        bg = seed_image.resize((sw, sh), Image.Resampling.BILINEAR)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=max(6, min(sw, sh) // 8)))
        bg = bg.resize((cw, ch), Image.Resampling.BICUBIC).convert("RGBA")
        bg = Image.alpha_composite(bg, Image.new("RGBA", (cw, ch), (14, 15, 18, 128)))
        return bg.convert("RGB"), True
    if bg_key in BG_COLORS:
        rgb = BG_COLORS[bg_key]
    else:
        rgb = _hex_to_rgb(bg_key, BG_COLORS["white"])
    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return Image.new("RGB", (cw, ch), rgb), luma < 128


def _append_footer(
    canvas: Image.Image,
    cfg: CollageConfig,
    metas: list[dict[str, Any]],
    *,
    is_dark: bool,
    bg_rgb: tuple[int, int, int],
) -> Image.Image:
    if not cfg.show_footer:
        return canvas
    cw, ch = canvas.size
    bar_h = max(68, int(round(cw * 0.072)))
    out = Image.new("RGB", (cw, ch + bar_h), bg_rgb)
    out.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(out)

    title = (cfg.footer_title or "").strip()
    subtitle = (cfg.footer_subtitle or "").strip()
    if not title:
        cams = [m.get("camera", "").strip() for m in metas if m.get("camera")]
        title = cams[0] if cams else "SUPERPHOTO COLLECTION"
    if not subtitle:
        dates = [m.get("date_only", "").strip() for m in metas if m.get("date_only")]
        lenses = [m.get("lens", "").strip() for m in metas if m.get("lens")]
        bits = [b for b in (lenses[0] if lenses else "", dates[0] if dates else "", f"{len(metas)} PHOTOS") if b]
        subtitle = "  ·  ".join(bits)

    primary = (240, 241, 244) if is_dark else (28, 30, 34)
    secondary = (150, 154, 162) if is_dark else (126, 130, 138)
    line_col = (44, 46, 52) if is_dark else (232, 234, 238)
    pad_x = max(24, cfg.padding)
    draw.line((pad_x, ch, cw - pad_x, ch), fill=line_col, width=max(1, bar_h // 80))

    cy = ch + bar_h / 2.0
    max_w = max(100, cw - pad_x * 2)
    t_font, t_tw, _ = _fit_font(draw, title, "bold", max(14, int(bar_h * 0.28)), max_w)
    s_font, s_tw, _ = _fit_font(draw, subtitle, "regular", max(11, int(bar_h * 0.19)), max_w) if subtitle else (None, 0, 0)
    if subtitle:
        _draw_text_ycenter(draw, (cw - t_tw) / 2.0, cy - bar_h * 0.14, title, t_font, primary)
        _draw_text_ycenter(draw, (cw - s_tw) / 2.0, cy + bar_h * 0.17, subtitle, s_font, secondary)
    else:
        _draw_text_ycenter(draw, (cw - t_tw) / 2.0, cy, title, t_font, primary)
    return out


# ---- Single-photo slicer (1 -> 9 / 4 / 6) ----------------------------------


def slice_single_photo(
    image: Image.Image,
    layout: Literal["slice_9", "slice_4", "slice_6"],
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    tile_size: int = 1200,
) -> tuple[list[Image.Image], int, int]:
    """Slice one photograph into ``cols x rows`` crisp square tiles in Moments posting order."""
    cols, rows = {"slice_9": (3, 3), "slice_4": (2, 2), "slice_6": (3, 2)}[layout]
    rgb = image_to_srgb(ImageOps.exif_transpose(image))
    total_w = tile_size * cols
    total_h = tile_size * rows
    # Don't upscale tiny images unnecessarily when rendering previews
    src_max_tile = max(64, min(rgb.width // cols, rgb.height // rows))
    effective_tile = min(tile_size, max(src_max_tile, min(tile_size, 900)))
    total_w = effective_tile * cols
    total_h = effective_tile * rows

    board = _cover_crop(rgb, total_w, total_h, ox=offset_x, oy=offset_y)
    tiles: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * effective_tile, r * effective_tile
            tiles.append(board.crop((x0, y0, x0 + effective_tile, y0 + effective_tile)))
    return tiles, cols, rows


# ---- Layout builders -------------------------------------------------------


def _prepare_tile(
    im: Image.Image,
    meta: dict[str, Any],
    tw: int,
    th: int,
    cfg: CollageConfig,
    bg_rgb: tuple[int, int, int],
) -> Image.Image:
    work = image_to_srgb(ImageOps.exif_transpose(im))
    if cfg.tile_watermark:
        work = render_watermark(work, meta, cfg.watermark_config, max_edge=max(tw, th))
    if cfg.tile_frame_mode == "padded_square" and tw == th:
        return _pad_into_square(work, tw, bg_rgb, margin_ratio=0.04)
    return _cover_crop(work, tw, th)


def _render_uniform_grid(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    cfg: CollageConfig,
    *,
    cols: int,
    max_items: int,
    target_width: int,
) -> Image.Image:
    imgs = images[:max_items]
    mts = metas[:max_items]
    n = len(imgs)
    if n == 1:
        cols_eff, rows_eff = 1, 1
    elif n == 4 and cfg.layout == "grid_9":
        cols_eff, rows_eff = 2, 2
    else:
        cols_eff = min(cols, n)
        rows_eff = math.ceil(n / cols_eff)

    scale = target_width / 1800.0
    pad = int(round(cfg.padding * scale))
    gap = int(round(cfg.gap * scale))
    rad = int(round(cfg.radius * scale))

    tile = max(64, (target_width - pad * 2 - gap * (cols_eff - 1)) // cols_eff)
    cw = pad * 2 + tile * cols_eff + gap * (cols_eff - 1)
    ch = pad * 2 + tile * rows_eff + gap * (rows_eff - 1)
    canvas, is_dark = _make_canvas(cw, ch, cfg.background, imgs[0])
    bg_rgb = BG_COLORS.get(cfg.background.lower(), _hex_to_rgb(cfg.background, (255, 255, 255)))
    if cfg.background.lower() == "blur":
        bg_rgb = (18, 19, 22)

    for idx, (im, mt) in enumerate(zip(imgs, mts)):
        r, c = divmod(idx, cols_eff)
        x = pad + c * (tile + gap)
        y = pad + r * (tile + gap)
        tile_im = _prepare_tile(im, mt, tile, tile, cfg, bg_rgb)
        _paste_rounded(canvas, tile_im, (x, y), rad)

    return _append_footer(canvas, cfg, mts, is_dark=is_dark, bg_rgb=bg_rgb)


def _render_triptych(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    cfg: CollageConfig,
    *,
    target_width: int,
) -> Image.Image:
    imgs = images[:4]
    mts = metas[:4]
    n = len(imgs)
    scale = target_width / 1600.0
    pad = int(round(max(cfg.padding, 20) * scale))
    gap = int(round(cfg.gap * scale))
    rad = int(round(cfg.radius * scale))

    frame_w = max(120, target_width - pad * 2)
    # Cinematic 2.15:1 widescreen aspect per frame
    frame_h = max(80, int(round(frame_w / 2.15)))
    cw = pad * 2 + frame_w
    ch = pad * 2 + frame_h * n + gap * (n - 1)

    canvas, is_dark = _make_canvas(cw, ch, cfg.background, imgs[0])
    bg_rgb = BG_COLORS.get(cfg.background.lower(), _hex_to_rgb(cfg.background, (18, 19, 22)))
    if cfg.background.lower() == "blur":
        bg_rgb = (18, 19, 22)

    for idx, (im, mt) in enumerate(zip(imgs, mts)):
        y = pad + idx * (frame_h + gap)
        tile_im = _cover_crop(image_to_srgb(ImageOps.exif_transpose(im)), frame_w, frame_h)
        if cfg.tile_watermark:
            tile_im = render_watermark(tile_im, mt, cfg.watermark_config.model_copy(update={"style": "corner_overlay"}))
        _paste_rounded(canvas, tile_im, (pad, y), rad)

    return _append_footer(canvas, cfg, mts, is_dark=is_dark, bg_rgb=bg_rgb)


def _render_editorial(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    cfg: CollageConfig,
    *,
    target_width: int,
) -> Image.Image:
    """Aspect-aware justified magazine layout with zero awkward cropping."""
    imgs = [image_to_srgb(ImageOps.exif_transpose(im)) for im in images[:9]]
    mts = metas[:9]
    n = len(imgs)
    scale = target_width / 1800.0
    pad = int(round(cfg.padding * scale))
    gap = int(round(cfg.gap * scale))
    rad = int(round(cfg.radius * scale))
    inner_w = max(200, target_width - pad * 2)

    bg_rgb = BG_COLORS.get(cfg.background.lower(), _hex_to_rgb(cfg.background, (255, 255, 255)))
    if cfg.background.lower() == "blur":
        bg_rgb = (18, 19, 22)

    # Special asymmetric hero layout for 3 photos (portrait hero on left + 2 stacked right, or wide hero on top + 2 bottom)
    if n == 3:
        hero_ar = imgs[0].width / max(imgs[0].height, 1)
        if hero_ar <= 1.05:
            # Tall/square hero on left (56%), 2 companions stacked on right (44%)
            left_w = int(round((inner_w - gap) * 0.56))
            right_w = inner_w - gap - left_w
            total_h = int(round(left_w / max(hero_ar, 0.66)))
            total_h = max(int(inner_w * 0.62), min(int(inner_w * 1.15), total_h))
            top_h = (total_h - gap) // 2
            bot_h = total_h - gap - top_h
            cw, ch = pad * 2 + inner_w, pad * 2 + total_h
            canvas, is_dark = _make_canvas(cw, ch, cfg.background, imgs[0])
            _paste_rounded(canvas, _prepare_tile(imgs[0], mts[0], left_w, total_h, cfg, bg_rgb), (pad, pad), rad)
            _paste_rounded(canvas, _prepare_tile(imgs[1], mts[1], right_w, top_h, cfg, bg_rgb), (pad + left_w + gap, pad), rad)
            _paste_rounded(canvas, _prepare_tile(imgs[2], mts[2], right_w, bot_h, cfg, bg_rgb), (pad + left_w + gap, pad + top_h + gap), rad)
            return _append_footer(canvas, cfg, mts, is_dark=is_dark, bg_rgb=bg_rgb)

    # Partition photos into clean magazine rows preserving natural aspect ratios
    row_plans: dict[int, list[int]] = {
        1: [1],
        2: [2],
        3: [1, 2],
        4: [1, 3] if (imgs[0].width / max(imgs[0].height, 1)) >= 1.2 else [2, 2],
        5: [2, 3],
        6: [2, 2, 2],
        7: [2, 3, 2],
        8: [2, 3, 3],
        9: [3, 3, 3],
    }
    counts = row_plans.get(n, [3] * math.ceil(n / 3))
    rows: list[list[tuple[Image.Image, dict[str, Any]]]] = []
    cursor = 0
    for c in counts:
        if cursor >= n:
            break
        rows.append(list(zip(imgs[cursor:cursor + c], mts[cursor:cursor + c])))
        cursor += c

    # For each row, compute exact row height H = (inner_w - gap*(k-1)) / sum(clamped_aspect)
    placed_rows: list[tuple[int, list[tuple[Image.Image, dict[str, Any], int]]]] = []
    total_h = 0
    for row in rows:
        k = len(row)
        avail_w = inner_w - gap * (k - 1)
        aspects = [max(0.62, min(1.85, im.width / max(im.height, 1))) for im, _ in row]
        row_h = int(round(avail_w / sum(aspects)))
        row_h = max(int(inner_w * 0.24), min(int(inner_w * 0.78), row_h))
        widths = [max(48, int(round(row_h * ar))) for ar in aspects]
        # Snap exact right edge to inner_w
        diff = avail_w - sum(widths)
        widths[-1] = max(48, widths[-1] + diff)
        placed_rows.append((row_h, [(im, mt, w_i) for (im, mt), w_i in zip(row, widths)]))
        total_h += row_h
    total_h += gap * max(0, len(placed_rows) - 1)

    cw = pad * 2 + inner_w
    ch = pad * 2 + total_h
    canvas, is_dark = _make_canvas(cw, ch, cfg.background, imgs[0])
    y = pad
    for row_h, entries in placed_rows:
        x = pad
        for im, mt, w_i in entries:
            tile_im = _prepare_tile(im, mt, w_i, row_h, cfg, bg_rgb)
            _paste_rounded(canvas, tile_im, (x, y), rad)
            x += w_i + gap
        y += row_h + gap

    return _append_footer(canvas, cfg, mts, is_dark=is_dark, bg_rgb=bg_rgb)


def _render_vertical_story(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    cfg: CollageConfig,
    *,
    target_width: int,
) -> Image.Image:
    imgs = [image_to_srgb(ImageOps.exif_transpose(im)) for im in images[:9]]
    mts = metas[:9]
    scale = target_width / 1400.0
    pad = int(round(max(cfg.padding, 24) * scale))
    gap = int(round(max(cfg.gap, 18) * scale))
    rad = int(round(cfg.radius * scale))
    inner_w = max(180, target_width - pad * 2)

    bg_rgb = BG_COLORS.get(cfg.background.lower(), _hex_to_rgb(cfg.background, (255, 255, 255)))
    if cfg.background.lower() == "blur":
        bg_rgb = (18, 19, 22)

    blocks: list[tuple[Image.Image, str]] = []
    caption_h = int(round(36 * scale)) if cfg.tile_watermark else 0
    for im, mt in zip(imgs, mts):
        work = render_watermark(im, mt, cfg.watermark_config, max_edge=inner_w) if cfg.tile_watermark else im
        nh = max(60, int(round(inner_w * (work.height / max(work.width, 1)))))
        resized = work.resize((inner_w, nh), Image.Resampling.LANCZOS)
        cap = "" if cfg.tile_watermark else format_params_line(mt, sep="  ·  ")
        blocks.append((resized, cap))

    sub_cap_h = int(round(30 * scale)) if (not cfg.tile_watermark and any(c for _, c in blocks)) else 0
    total_h = pad * 2 + sum(im.height + (sub_cap_h if cap else 0) for im, cap in blocks) + gap * max(0, len(blocks) - 1)
    cw = pad * 2 + inner_w
    canvas, is_dark = _make_canvas(cw, total_h, cfg.background, imgs[0])
    draw = ImageDraw.Draw(canvas)
    cap_col = (165, 170, 178) if is_dark else (126, 130, 138)

    y = pad
    for im, cap in blocks:
        _paste_rounded(canvas, im, (pad, y), rad)
        y += im.height
        if cap and sub_cap_h > 0:
            font, tw, _ = _fit_font(draw, cap, "regular", max(11, int(sub_cap_h * 0.56)), inner_w)
            _draw_text_ycenter(draw, pad + (inner_w - tw) / 2.0, y + sub_cap_h * 0.52, cap, font, cap_col)
            y += sub_cap_h
        y += gap

    return _append_footer(canvas, cfg, mts, is_dark=is_dark, bg_rgb=bg_rgb)


def _render_slice_collage(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    cfg: CollageConfig,
    *,
    target_width: int,
) -> Image.Image:
    layout = cfg.layout
    assert layout in ("slice_9", "slice_4", "slice_6")
    cols = 2 if layout == "slice_4" else 3
    rows = 2 if layout in ("slice_4", "slice_6") else 3
    scale = target_width / 1800.0
    pad = int(round(cfg.padding * scale))
    gap = int(round(max(cfg.gap, 10) * scale))
    rad = int(round(cfg.radius * scale))

    tile = max(64, (target_width - pad * 2 - gap * (cols - 1)) // cols)
    tiles, cols, rows = slice_single_photo(
        images[0], layout, offset_x=cfg.slice_offset_x, offset_y=cfg.slice_offset_y, tile_size=tile
    )
    actual_tile = tiles[0].width
    cw = pad * 2 + actual_tile * cols + gap * (cols - 1)
    ch = pad * 2 + actual_tile * rows + gap * (rows - 1)
    canvas, is_dark = _make_canvas(cw, ch, cfg.background, images[0])
    bg_rgb = BG_COLORS.get(cfg.background.lower(), _hex_to_rgb(cfg.background, (255, 255, 255)))
    if cfg.background.lower() == "blur":
        bg_rgb = (18, 19, 22)

    for idx, t_im in enumerate(tiles):
        r, c = divmod(idx, cols)
        _paste_rounded(canvas, t_im, (pad + c * (actual_tile + gap), pad + r * (actual_tile + gap)), rad)

    return _append_footer(canvas, cfg, metas[:1], is_dark=is_dark, bg_rgb=bg_rgb)


def render_collage(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    config: CollageConfig | None = None,
    *,
    target_width: int = 1800,
) -> Image.Image:
    """Render a combined collage JPEG according to ``config``."""
    if not images:
        raise ValueError("请至少添加一张照片再进行拼图")
    cfg = config or CollageConfig()
    padded_metas = [metas[i] if i < len(metas) else {} for i in range(len(images))]

    if cfg.layout in ("slice_9", "slice_4", "slice_6"):
        return _render_slice_collage(images, padded_metas, cfg, target_width=target_width)
    if cfg.layout == "grid_9":
        return _render_uniform_grid(images, padded_metas, cfg, cols=3, max_items=9, target_width=target_width)
    if cfg.layout == "grid_4":
        return _render_uniform_grid(images, padded_metas, cfg, cols=2, max_items=4, target_width=target_width)
    if cfg.layout == "grid_6":
        return _render_uniform_grid(images, padded_metas, cfg, cols=3, max_items=6, target_width=target_width)
    if cfg.layout == "triptych":
        return _render_triptych(images, padded_metas, cfg, target_width=target_width)
    if cfg.layout == "editorial":
        return _render_editorial(images, padded_metas, cfg, target_width=target_width)
    if cfg.layout == "vertical_story":
        return _render_vertical_story(images, padded_metas, cfg, target_width=target_width)
    return _render_uniform_grid(images, padded_metas, cfg, cols=3, max_items=9, target_width=target_width)


def build_moments_tiles(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    config: CollageConfig | None = None,
    *,
    tile_edge: int = 1600,
) -> list[Image.Image]:
    """Generate ordered individual images ready to upload directly to WeChat Moments."""
    if not images:
        raise ValueError("请至少添加一张照片")
    cfg = config or CollageConfig()
    padded_metas = [metas[i] if i < len(metas) else {} for i in range(len(images))]

    if cfg.layout in ("slice_9", "slice_4", "slice_6"):
        tiles, _, _ = slice_single_photo(
            images[0],
            cfg.layout,
            offset_x=cfg.slice_offset_x,
            offset_y=cfg.slice_offset_y,
            tile_size=tile_edge,
        )
        return tiles

    max_count = {"grid_4": 4, "grid_6": 6, "triptych": 4}.get(cfg.layout, 9)
    bg_rgb = BG_COLORS.get(cfg.background.lower(), _hex_to_rgb(cfg.background, (255, 255, 255)))
    if cfg.background.lower() == "blur":
        bg_rgb = (18, 19, 22)

    out: list[Image.Image] = []
    for im, mt in zip(images[:max_count], padded_metas[:max_count]):
        work = image_to_srgb(ImageOps.exif_transpose(im))
        if cfg.tile_watermark:
            work = render_watermark(work, mt, cfg.watermark_config, max_edge=tile_edge)
        if cfg.tile_frame_mode == "cover_square":
            work = _cover_crop(work, tile_edge, tile_edge)
        elif cfg.tile_frame_mode == "padded_square":
            work = _pad_into_square(work, tile_edge, bg_rgb, margin_ratio=0.05)
        else:
            if max(work.size) > tile_edge:
                scale = tile_edge / max(work.size)
                work = work.resize(
                    (max(1, int(round(work.width * scale))), max(1, int(round(work.height * scale)))),
                    Image.Resampling.LANCZOS,
                )
        out.append(work)
    return out


def export_moments_zip_bytes(
    images: list[Image.Image],
    metas: list[dict[str, Any]],
    config: CollageConfig | None = None,
    *,
    tile_edge: int = 1600,
    quality: int = 94,
    prefix: str = "moments",
) -> bytes:
    """Package ordered WeChat Moments tiles (``01.jpg``..``09.jpg``) into an in-memory ZIP."""
    tiles = build_moments_tiles(images, metas, config, tile_edge=tile_edge)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, tile in enumerate(tiles, start=1):
            data = encode_jpeg_bytes(tile, quality=quality)
            zf.writestr(f"{prefix}-{idx:02d}.jpg", data)
    return buf.getvalue()
