"""Automatic EXIF metadata extraction and gallery-grade photo watermark rendering.

Extracts camera body, lens, focal length, aperture, shutter speed, ISO, exposure bias,
capture timestamp, and GPS coordinates from RAW/JPEG/PNG/TIFF photos, normalizes rational
EXIF values into crisp photographic notation (e.g. ``120mm  f/4  1/400s  ISO 400``), and
renders six curated editorial watermark templates with supersampled vector brand marks.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from pydantic import BaseModel, Field

from .ingest import image_to_srgb, read_exif, srgb_profile_bytes

WatermarkStyle = Literal[
    "leica_bar",
    "dark_bar",
    "gallery_frame",
    "blur_card",
    "film_polaroid",
    "corner_overlay",
]

WATERMARK_STYLES: list[dict[str, str]] = [
    {
        "id": "leica_bar",
        "name": "经典白栏",
        "subtitle": "白底参数条 · 经典徕卡风",
    },
    {
        "id": "dark_bar",
        "name": "暗夜黑栏",
        "subtitle": "黑底参数条 · 夜景/暗调",
    },
    {
        "id": "gallery_frame",
        "name": "画廊白框",
        "subtitle": "四周留白 · 策展装裱感",
    },
    {
        "id": "blur_card",
        "name": "氛围毛玻璃",
        "subtitle": "背景虚化悬浮 · 朋友圈热门",
    },
    {
        "id": "film_polaroid",
        "name": "胶片拍立得",
        "subtitle": "暖白相纸 · 胶片等宽字",
    },
    {
        "id": "corner_overlay",
        "name": "极简角标",
        "subtitle": "无边框悬浮 · 保留原画幅",
    },
]

BRAND_SPECS: dict[str, dict[str, str]] = {
    "leica": {"label": "LEICA", "short": "Leica", "color": "#E20612"},
    "hasselblad": {"label": "HASSELBLAD", "short": "H", "color": "#F06A1A"},
    "nikon": {"label": "NIKON", "short": "Nikon", "color": "#F2C500"},
    "sony": {"label": "SONY", "short": "α", "color": "#E66A2C"},
    "canon": {"label": "CANON", "short": "Canon", "color": "#CC1D24"},
    "fujifilm": {"label": "FUJIFILM", "short": "FUJI", "color": "#00995D"},
    "ricoh": {"label": "RICOH", "short": "GR", "color": "#D93025"},
    "panasonic": {"label": "LUMIX", "short": "LUMIX", "color": "#1F6FD6"},
    "olympus": {"label": "OM SYSTEM", "short": "OM", "color": "#1E5CA8"},
    "apple": {"label": "APPLE", "short": "iPhone", "color": "#6E737B"},
    "xiaomi": {"label": "XIAOMI", "short": "MI", "color": "#FF6900"},
    "huawei": {"label": "HUAWEI", "short": "XMAGE", "color": "#C8102E"},
    "vivo": {"label": "VIVO", "short": "ZEISS", "color": "#1F5FE0"},
    "oppo": {"label": "OPPO", "short": "OPPO", "color": "#1B8A4B"},
    "dji": {"label": "DJI", "short": "DJI", "color": "#2E74D1"},
    "sigma": {"label": "SIGMA", "short": "SIGMA", "color": "#8E939A"},
    "pentax": {"label": "PENTAX", "short": "PENTAX", "color": "#C8232C"},
    "phaseone": {"label": "PHASE ONE", "short": "P1", "color": "#2C6CB0"},
    "generic": {"label": "CAMERA", "short": "SP", "color": "#C89B54"},
}


class WatermarkConfig(BaseModel):
    style: WatermarkStyle = "leica_bar"
    show_brand_mark: bool = True
    show_lens: bool = True
    show_date: bool = True
    show_location: bool = True
    show_ev: bool = False
    border_scale: float = Field(default=1.0, ge=0.65, le=1.5)
    # Optional user overrides (empty/None falls back to auto-extracted EXIF)
    camera: str | None = None
    lens: str | None = None
    focal_length: str | None = None
    aperture: str | None = None
    shutter: str | None = None
    iso: str | None = None
    ev_bias: str | None = None
    datetime_text: str | None = None
    location: str | None = None
    signature: str | None = None
    brand: str | None = None
    accent_color: str | None = None


# ---- EXIF parsing & normalization ------------------------------------------


def _parse_fraction(raw: Any) -> Fraction | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Unwrap brackets like "[400]" or "(1, 400)"
    text = text.strip("[]() ")
    # Match explicit rational "a/b"
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            num = Fraction(m.group(1))
            den = Fraction(m.group(2))
            if den != 0:
                return num / den
        except (ValueError, ZeroDivisionError):
            return None
    # Match decimal or integer
    m = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if m:
        try:
            return Fraction(m.group(0))
        except ValueError:
            return None
    return None


def _clean_float_str(val: float, max_decimals: int = 1) -> str:
    rounded = round(val, max_decimals)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.{max_decimals}f}".rstrip("0").rstrip(".")


def format_aperture(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if text.lower().startswith("f/"):
        frac = _parse_fraction(text[2:])
        if frac is not None and frac > 0:
            return f"f/{_clean_float_str(float(frac), 1)}"
        return text
    frac = _parse_fraction(text)
    if frac is None or frac <= 0:
        return ""
    return f"f/{_clean_float_str(float(frac), 1)}"


def format_shutter(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    # Preserve clean "1/N" fractions directly
    m = re.fullmatch(r"1\s*/\s*(\d+)\s*s?", text, flags=re.IGNORECASE)
    if m and int(m.group(1)) > 0:
        return f"1/{int(m.group(1))}s"
    frac = _parse_fraction(text)
    if frac is None or frac <= 0:
        return ""
    secs = float(frac)
    if secs < 0.35:
        denom = max(1, int(round(1.0 / secs)))
        return f"1/{denom}s"
    if secs < 1.0:
        # Express clean halves/thirds if very close, else decimal
        inv = round(1.0 / secs)
        if inv in (2, 3) and abs(secs - 1.0 / inv) < 0.02:
            return f"1/{inv}s"
        return f"{_clean_float_str(secs, 1)}s"
    if secs >= 10.0:
        return f"{int(round(secs))}s"
    return f"{_clean_float_str(secs, 1)}s"


def format_focal_length(raw: Any) -> str:
    if raw is None:
        return ""
    frac = _parse_fraction(raw)
    if frac is None or frac <= 0:
        return ""
    return f"{_clean_float_str(float(frac), 1)}mm"


def format_iso(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    m = re.search(r"\d+", text)
    if not m:
        return ""
    val = int(m.group(0))
    if val <= 0:
        return ""
    return f"ISO {val}"


def format_ev_bias(raw: Any) -> str:
    if raw is None:
        return ""
    frac = _parse_fraction(raw)
    if frac is None:
        return ""
    val = float(frac)
    if abs(val) < 0.05:
        return ""
    sign = "+" if val > 0 else "-"
    return f"{sign}{abs(val):.1f} EV"


def format_datetime(raw: Any) -> tuple[str, str]:
    """Return ``(full_display, date_only)`` such as ``('2024.07.04 16:35', '2024.07.04')``."""
    if raw is None:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""
    m = re.search(r"(\d{4})[:\-/.](\d{1,2})[:\-/.](\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", text)
    if not m:
        return text, text.split(" ")[0]
    y, mo, d, hh, mm = m.groups()
    date_only = f"{y}.{int(mo):02d}.{int(d):02d}"
    if hh is not None and mm is not None:
        return f"{date_only} {int(hh):02d}:{int(mm):02d}", date_only
    return date_only, date_only


def _parse_gps_dms(raw: Any) -> tuple[int, int, float] | None:
    if raw is None:
        return None
    text = str(raw).strip().strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    vals: list[float] = []
    for p in parts[:3]:
        f = _parse_fraction(p)
        if f is None:
            return None
        vals.append(float(f))
    deg = vals[0]
    mins = vals[1] if len(vals) > 1 else 0.0
    secs = vals[2] if len(vals) > 2 else 0.0
    total = deg + mins / 60.0 + secs / 3600.0
    d_i = int(total)
    rem = (total - d_i) * 60.0
    m_i = int(rem)
    s_f = (rem - m_i) * 60.0
    return d_i, m_i, s_f


def format_gps(exif: dict[str, str]) -> str:
    lat = _parse_gps_dms(exif.get("GPSLatitude"))
    lon = _parse_gps_dms(exif.get("GPSLongitude"))
    if not lat or not lon:
        return ""
    lat_ref = (exif.get("GPSLatitudeRef") or "N").strip().upper()[:1] or "N"
    lon_ref = (exif.get("GPSLongitudeRef") or "E").strip().upper()[:1] or "E"
    return f"{lat[0]}°{lat[1]:02d}'{lat_ref} {lon[0]}°{lon[1]:02d}'{lon_ref}"


def detect_brand(make: str, model: str, lens: str = "") -> str:
    checks = (
        ("leica", ("leica", "m11", "m10", "q2", "q3", "sl2", "sl3")),
        ("hasselblad", ("hasselblad", "x2d", "x1d", "907x")),
        ("nikon", ("nikon", "nikkor")),
        ("sony", ("sony", "ilce", "ilme", "dsc-rx", "zv-e")),
        ("canon", ("canon", "eos")),
        ("fujifilm", ("fujifilm", "fuji", "gfx", "x-t", "x-h", "x100", "x-pro", "x-e")),
        ("ricoh", ("ricoh", "gr iii", "gr ii", "gr3", "gr2")),
        ("panasonic", ("panasonic", "lumix", "dc-s", "dc-g")),
        ("olympus", ("olympus", "om digital", "om-1", "om-5", "e-m1", "e-m5")),
        ("apple", ("apple", "iphone", "ipad")),
        ("xiaomi", ("xiaomi", "redmi")),
        ("huawei", ("huawei",)),
        ("vivo", ("vivo",)),
        ("oppo", ("oppo", "oneplus")),
        ("dji", ("dji", "mavic", "osmo")),
        ("sigma", ("sigma",)),
        ("pentax", ("pentax",)),
        ("phaseone", ("phase one", "phaseone", "iq4")),
    )
    # Prioritize camera body (Make + Model) so adapted third-party lenses (e.g. Sigma/Nikkor on Sony/Canon)
    # never override the actual camera brand.
    body_hay = f"{make} {model}".lower()
    for brand, needles in checks:
        if any(n in body_hay for n in needles):
            return brand
    lens_hay = (lens or "").lower()
    for brand, needles in checks:
        if any(n in lens_hay for n in needles):
            return brand
    return "generic"


def clean_camera_name(make: str, model: str, brand: str) -> str:
    make_s = re.sub(r"\s+", " ", (make or "").strip())
    model_s = re.sub(r"\s+", " ", (model or "").strip())
    if not model_s:
        return make_s
    # Clean common Nikon Z underscore conventions: "NIKON Z 7_2" -> "NIKON Z 7II"
    model_s = re.sub(r"\b(Z\s*\d+)_2\b", r"\1II", model_s, flags=re.IGNORECASE)
    model_s = re.sub(r"\b(Z\s*\d+)_3\b", r"\1III", model_s, flags=re.IGNORECASE)
    model_s = re.sub(r"\b(Z\s*f)\b", "Zf", model_s)
    # Remove noisy corporate suffixes from make before checking prefix duplication
    short_make = re.sub(
        r"\b(CORPORATION|CORP\.?|INC\.?|CO\.?,?\s*LTD\.?|IMAGING|CAMERA|AG|OPTICAL)\b",
        "",
        make_s,
        flags=re.IGNORECASE,
    ).strip(" ,.")
    if short_make and not model_s.lower().startswith(short_make.lower()):
        if brand == "apple" and model_s.lower().startswith("iphone"):
            return model_s
        if brand == "leica" and model_s.lower().startswith("leica"):
            return model_s
        if brand == "sony" and model_s.upper().startswith(("ILCE-", "ILME-", "DSC-")):
            return f"SONY {model_s}"
        return f"{short_make} {model_s}".strip()
    return model_s


def normalize_exif(
    exif: dict[str, Any] | None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    raw = {str(k): str(v) for k, v in (exif or {}).items() if v is not None and str(v).strip()}
    make = raw.get("Make", "").strip()
    model = raw.get("Model", "").strip()
    lens = raw.get("LensModel", "").strip()
    brand = detect_brand(make, model, lens)
    camera = clean_camera_name(make, model, brand)
    focal = format_focal_length(raw.get("FocalLength"))
    focal35 = format_focal_length(raw.get("FocalLengthIn35mmFilm"))
    aperture = format_aperture(raw.get("FNumber"))
    shutter = format_shutter(raw.get("ExposureTime"))
    iso = format_iso(raw.get("ISOSpeedRatings") or raw.get("PhotographicSensitivity"))
    ev_bias = format_ev_bias(raw.get("ExposureBiasValue"))
    dt_full, dt_date = format_datetime(raw.get("DateTimeOriginal"))
    gps = format_gps(raw)
    has_exif = any([camera, lens, focal, aperture, shutter, iso])
    spec = BRAND_SPECS.get(brand, BRAND_SPECS["generic"])
    return {
        "make": make,
        "model": model,
        "brand": brand,
        "brand_label": spec["label"],
        "accent_color": spec["color"],
        "camera": camera,
        "lens": lens,
        "focal_length": focal,
        "focal_length_35mm": focal35,
        "aperture": aperture,
        "shutter": shutter,
        "iso": iso,
        "ev_bias": ev_bias,
        "datetime": dt_full,
        "date_only": dt_date,
        "gps": gps,
        "width": width,
        "height": height,
        "has_exif": has_exif,
        "raw_exif": raw,
    }


def extract_photo_metadata(
    path: Path | None,
    *,
    fallback_exif: dict[str, Any] | None = None,
    image_path_for_size: Path | None = None,
) -> dict[str, Any]:
    exif: dict[str, Any] = {}
    if fallback_exif:
        exif.update(fallback_exif)
    if path is not None and path.is_file():
        fresh = read_exif(path)
        if fresh:
            exif.update(fresh)
    w = h = None
    size_src = image_path_for_size if (image_path_for_size and image_path_for_size.is_file()) else path
    if size_src is not None and size_src.is_file():
        try:
            with Image.open(size_src) as im:
                im = ImageOps.exif_transpose(im)
                w, h = im.size
        except Exception:
            pass
    return normalize_exif(exif, width=w, height=h)


def resolve_metadata(meta: dict[str, Any], cfg: WatermarkConfig) -> dict[str, Any]:
    """Merge auto-extracted metadata with explicit user overrides from ``WatermarkConfig``."""
    out = dict(meta or {})
    if cfg.camera is not None:
        out["camera"] = cfg.camera.strip()
    if cfg.lens is not None:
        out["lens"] = cfg.lens.strip()
    if cfg.focal_length is not None:
        out["focal_length"] = cfg.focal_length.strip()
    if cfg.aperture is not None:
        out["aperture"] = cfg.aperture.strip()
    if cfg.shutter is not None:
        out["shutter"] = cfg.shutter.strip()
    if cfg.iso is not None:
        out["iso"] = cfg.iso.strip()
    if cfg.ev_bias is not None:
        out["ev_bias"] = cfg.ev_bias.strip()
    if cfg.datetime_text is not None:
        out["datetime"] = cfg.datetime_text.strip()
        out["date_only"] = cfg.datetime_text.strip().split(" ")[0]
    if cfg.location is not None:
        out["location"] = cfg.location.strip()
    else:
        out.setdefault("location", out.get("gps", ""))
    out["signature"] = (cfg.signature or "").strip()
    brand = (cfg.brand or out.get("brand") or "generic").strip().lower()
    if brand not in BRAND_SPECS:
        brand = detect_brand("", out.get("camera", ""), out.get("lens", ""))
    out["brand"] = brand
    spec = BRAND_SPECS.get(brand, BRAND_SPECS["generic"])
    out["brand_label"] = spec["label"]
    accent = (cfg.accent_color or "").strip()
    out["accent_color"] = accent if re.fullmatch(r"#[0-9a-fA-F]{6}", accent) else spec["color"]
    return out


def format_params_line(meta: dict[str, Any], *, show_ev: bool = False, sep: str = "  ") -> str:
    parts = [
        (meta.get("focal_length") or "").strip(),
        (meta.get("aperture") or "").strip(),
        (meta.get("shutter") or "").strip(),
        (meta.get("iso") or "").strip(),
    ]
    if show_ev and (meta.get("ev_bias") or "").strip():
        parts.append(str(meta["ev_bias"]).strip())
    return sep.join(p for p in parts if p)


def suggest_moments_caption(metas: list[dict[str, Any]]) -> str:
    """Generate a clean, natural photographer's Moments caption from extracted EXIF."""
    cameras = []
    lenses = []
    dates = []
    for m in metas:
        cam = (m.get("camera") or "").strip()
        if cam and cam not in cameras:
            cameras.append(cam)
        lens = (m.get("lens") or "").strip()
        if lens and lens not in lenses:
            lenses.append(lens)
        d = (m.get("date_only") or "").strip()
        if d and d not in dates:
            dates.append(d)
    bits: list[str] = []
    if cameras:
        bits.append(cameras[0])
    if lenses:
        bits.append(lenses[0])
    if len(metas) == 9:
        lead = "今日九宫格，记录光影瞬间。"
    elif len(metas) >= 4:
        lead = f"一组 {len(metas)} 张日常光影。"
    else:
        lead = "记录此刻光影。"
    if bits:
        gear = " · ".join(bits)
        date_tail = f" · {dates[0]}" if dates else ""
        return f"{lead}\n📷 {gear}{date_tail}"
    return lead


# ---- Cross-platform typography helpers -------------------------------------


_LATIN_FONTS: dict[str, list[tuple[str, int]]] = {
    "bold": [
        ("/System/Library/Fonts/Avenir Next.ttc", 2),   # Demi Bold
        ("/System/Library/Fonts/Avenir Next.ttc", 0),   # Bold
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
        ("C:/Windows/Fonts/segoeuib.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ],
    "medium": [
        ("/System/Library/Fonts/Avenir Next.ttc", 5),   # Medium
        ("/System/Library/Fonts/HelveticaNeue.ttc", 10),
        ("/System/Library/Fonts/SFNS.ttf", 0),
        ("C:/Windows/Fonts/seguisb.ttf", 0),
        ("C:/Windows/Fonts/segoeui.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ],
    "regular": [
        ("/System/Library/Fonts/Avenir Next.ttc", 7),   # Regular
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/SFNS.ttf", 0),
        ("C:/Windows/Fonts/segoeui.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ],
    "mono": [
        ("/System/Library/Fonts/SFNSMono.ttf", 0),
        ("/System/Library/Fonts/Menlo.ttc", 0),
        ("/System/Library/Fonts/Courier.ttc", 0),
        ("C:/Windows/Fonts/consola.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
    ],
    "serif": [
        ("/System/Library/Fonts/NewYork.ttf", 0),
        ("/System/Library/Fonts/Palatino.ttc", 0),
        ("C:/Windows/Fonts/georgia.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0),
    ],
}

_CJK_FONTS: dict[str, list[tuple[str, int]]] = {
    "bold": [
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),  # W6
        ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
        ("C:/Windows/Fonts/msyhbd.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ],
    "medium": [
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),
        ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
        ("C:/Windows/Fonts/msyh.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 0),
    ],
    "regular": [
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),  # W3
        ("/System/Library/Fonts/STHeiti Light.ttc", 1),
        ("C:/Windows/Fonts/msyh.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ],
    "mono": [
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("C:/Windows/Fonts/msyh.ttc", 0),
    ],
    "serif": [
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("C:/Windows/Fonts/simsun.ttc", 0),
    ],
}


def _needs_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if code >= 0x2E80 or (0x0370 <= code <= 0x03FF):
            # Include CJK blocks and Greek (e.g. α) where Hiragino/CJK handles cleanly
            return True
    return False


@lru_cache(maxsize=256)
def _load_font_cached(weight: str, size: int, cjk: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(round(size)))
    table = _CJK_FONTS if cjk else _LATIN_FONTS
    candidates = [*table.get(weight, []), *table.get("regular", [])]
    if not cjk:
        candidates.extend(_CJK_FONTS.get(weight, []))
    for path, idx in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size, index=idx)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def pick_font(text: str, weight: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load_font_cached(weight, max(8, int(round(size))), _needs_cjk(text or ""))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return max(0, int(box[2] - box[0])), max(0, int(box[3] - box[1]))


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    weight: str,
    target_size: int,
    max_width: int,
    min_size: int = 10,
) -> tuple[Any, int, int]:
    size = max(min_size, int(round(target_size)))
    font = pick_font(text, weight, size)
    tw, th = _text_size(draw, text, font)
    while tw > max_width and size > min_size:
        ratio = max_width / max(tw, 1)
        next_size = max(min_size, min(size - 1, int(size * ratio)))
        size = next_size
        font = pick_font(text, weight, size)
        tw, th = _text_size(draw, text, font)
    return font, tw, th


def _draw_text_ycenter(
    draw: ImageDraw.ImageDraw,
    x: float,
    cy: float,
    text: str,
    font: Any,
    fill: tuple[int, ...],
) -> None:
    if not text:
        return
    box = draw.textbbox((0, 0), text, font=font)
    h = box[3] - box[1]
    y = cy - h / 2.0 - box[1]
    draw.text((x - box[0], y), text, font=font, fill=fill)


def _hex_to_rgb(color: str, fallback: tuple[int, int, int] = (200, 155, 84)) -> tuple[int, int, int]:
    s = (color or "").strip().lstrip("#")
    if len(s) == 6:
        try:
            return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        except ValueError:
            pass
    return fallback


def _readable_on_light(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Ensure bright brand accents (such as Nikon yellow) remain crisp when used for text on light paper."""
    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    if luma <= 145:
        return rgb
    scale = 135.0 / max(luma, 1.0)
    return (int(rgb[0] * scale), int(rgb[1] * scale), int(rgb[2] * scale))


# ---- Supersampled vector brand emblems -------------------------------------


def _draw_brand_badge(
    brand: str,
    accent_hex: str,
    diameter: int,
    *,
    dark_bg: bool = False,
) -> Image.Image:
    """Render an anti-aliased brand emblem via 4x supersampling."""
    d = max(16, int(round(diameter)))
    scale = 4
    S = d * scale
    patch = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    r, g, b = _hex_to_rgb(accent_hex)
    pad = int(S * 0.06)
    cx = cy = S / 2.0

    if brand == "leica":
        draw.ellipse((pad, pad, S - pad, S - pad), fill=(r, g, b, 255))
        font = pick_font("Leica", "serif", int(S * 0.34))
        tw, _ = _text_size(draw, "Leica", font)
        _draw_text_ycenter(draw, cx - tw / 2.0, cy, "Leica", font, (255, 255, 255, 255))
    elif brand == "hasselblad":
        ring = (238, 238, 240, 235) if dark_bg else (28, 30, 34, 240)
        draw.rounded_rectangle((pad, pad, S - pad, S - pad), radius=int(S * 0.22), outline=ring, width=max(4, S // 20))
        font = pick_font("H", "bold", int(S * 0.56))
        tw, _ = _text_size(draw, "H", font)
        _draw_text_ycenter(draw, cx - tw / 2.0, cy, "H", font, (r, g, b, 255))
    elif brand == "nikon":
        draw.rounded_rectangle((pad, pad, S - pad, S - pad), radius=int(S * 0.18), fill=(r, g, b, 255))
        # Diagonal light streaks characteristic of Nikon badge
        for idx in range(3):
            y0 = int(S * (0.24 + idx * 0.16))
            draw.line((int(S * 0.14), y0 + int(S * 0.12), int(S * 0.86), y0 - int(S * 0.06)),
                      fill=(255, 248, 210, 120), width=max(3, S // 24))
        font = pick_font("N", "bold", int(S * 0.54))
        tw, _ = _text_size(draw, "N", font)
        _draw_text_ycenter(draw, cx - tw / 2.0, cy, "N", font, (20, 20, 22, 255))
    elif brand == "sony":
        ring = (r, g, b, 255)
        draw.ellipse((pad, pad, S - pad, S - pad), outline=ring, width=max(5, S // 16))
        font = pick_font("α", "bold", int(S * 0.56))
        tw, _ = _text_size(draw, "α", font)
        _draw_text_ycenter(draw, cx - tw / 2.0, cy - S * 0.02, "α", font, ring)
    elif brand in ("ricoh", "fujifilm", "canon", "vivo", "xiaomi", "huawei", "oppo", "dji", "panasonic", "olympus"):
        short = BRAND_SPECS[brand]["short"]
        bg_fill = (r, g, b, 255)
        draw.rounded_rectangle((pad, pad, S - pad, S - pad), radius=int(S * 0.22), fill=bg_fill)
        fsize = int(S * (0.44 if len(short) <= 2 else 0.30 if len(short) <= 4 else 0.24))
        font = pick_font(short, "bold", fsize)
        tw, _ = _text_size(draw, short, font)
        _draw_text_ycenter(draw, cx - tw / 2.0, cy, short, font, (255, 255, 255, 255))
    else:
        # Minimal precision 6-blade camera aperture icon
        stroke = (232, 234, 238, 230) if dark_bg else (34, 36, 40, 230)
        accent_rgba = (r, g, b, 255)
        w_line = max(4, S // 20)
        draw.ellipse((pad, pad, S - pad, S - pad), outline=stroke, width=w_line)
        inner_r = S * 0.22
        outer_r = S * 0.42
        for k in range(6):
            a1 = math.radians(k * 60 + 18)
            a2 = math.radians((k + 1) * 60 + 48)
            x1, y1 = cx + inner_r * math.cos(a1), cy + inner_r * math.sin(a1)
            x2, y2 = cx + outer_r * math.cos(a2), cy + outer_r * math.sin(a2)
            draw.line((x1, y1, x2, y2), fill=stroke, width=max(3, S // 26))
        dot_r = S * 0.085
        draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=accent_rgba)

    return patch.resize((d, d), Image.Resampling.LANCZOS)


# ---- Core watermark renderers ----------------------------------------------


def _prepare_base_image(image: Image.Image, max_edge: int = 0) -> Image.Image:
    im = image_to_srgb(ImageOps.exif_transpose(image))
    if max_edge > 0 and max(im.size) > max_edge:
        w, h = im.size
        scale = max_edge / max(w, h)
        im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
    return im


def _secondary_right_line(meta: dict[str, Any], cfg: WatermarkConfig) -> str:
    bits: list[str] = []
    if cfg.show_date and (meta.get("datetime") or "").strip():
        bits.append(str(meta["datetime"]).strip())
    if cfg.show_location and (meta.get("location") or "").strip():
        bits.append(str(meta["location"]).strip())
    if (meta.get("signature") or "").strip():
        bits.append(str(meta["signature"]).strip())
    return "  ·  ".join(bits)


def _render_classic_bar(
    im: Image.Image,
    meta: dict[str, Any],
    cfg: WatermarkConfig,
    *,
    dark: bool,
) -> Image.Image:
    w, h = im.size
    u = max(min(w, h), 0.72 * max(w, h))
    scale = float(cfg.border_scale)
    bar_h = max(48, int(round(u * 0.118 * scale)))
    pad_x = max(20, int(round(w * 0.036)))

    bg = (16, 17, 20) if dark else (255, 255, 255)
    primary = (242, 243, 245) if dark else (24, 25, 28)
    secondary = (150, 154, 162) if dark else (124, 128, 136)
    divider = (56, 59, 66) if dark else (218, 221, 226)
    accent_rgb = _hex_to_rgb(meta.get("accent_color", "#C89B54"))

    canvas = Image.new("RGB", (w, h + bar_h), bg)
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Subtle hairline separating photo and bar
    sep_col = (34, 36, 40) if dark else (238, 239, 242)
    draw.line((0, h, w, h), fill=sep_col, width=max(1, bar_h // 90))

    camera_txt = (meta.get("camera") or "").strip() or (meta.get("signature") or "SUPERPHOTO STUDIO")
    lens_txt = (meta.get("lens") or "").strip() if cfg.show_lens else ""
    params_txt = format_params_line(meta, show_ev=cfg.show_ev, sep="  ")
    right_sub = _secondary_right_line(meta, cfg)

    # Fallback gracefully when photo has no EXIF
    if not params_txt and not right_sub:
        if meta.get("signature"):
            params_txt = str(meta["signature"])
        else:
            params_txt = f"{w} × {h}"

    cy = h + bar_h / 2.0
    title_sz = max(13, int(round(bar_h * 0.255)))
    sub_sz = max(10, int(round(bar_h * 0.168)))
    row_gap = bar_h * 0.155

    # Allocate horizontal budgets so left and right never collide
    badge_d = int(round(bar_h * 0.44)) if cfg.show_brand_mark else 0
    div_gap = max(12, int(round(bar_h * 0.19))) if cfg.show_brand_mark else 0
    usable_w = max(80, w - 2 * pad_x)
    left_budget = int(usable_w * 0.44)
    right_budget = int(usable_w * 0.50) - (badge_d + div_gap * 2 if cfg.show_brand_mark else 0)
    right_budget = max(80, right_budget)

    # Left block: Camera + Lens
    cam_font, _, _ = _fit_font(draw, camera_txt, "bold", title_sz, left_budget)
    if lens_txt:
        lens_font, _, _ = _fit_font(draw, lens_txt, "regular", sub_sz, left_budget)
        _draw_text_ycenter(draw, pad_x, cy - row_gap, camera_txt, cam_font, primary)
        _draw_text_ycenter(draw, pad_x, cy + row_gap * 1.05, lens_txt, lens_font, secondary)
    else:
        _draw_text_ycenter(draw, pad_x, cy, camera_txt, cam_font, primary)

    # Right block: [Brand Emblem] | [Params + Date/Location]
    p_font, p_tw, _ = _fit_font(draw, params_txt, "bold", int(round(title_sz * 0.96)), right_budget)
    s_font, s_tw, _ = _fit_font(draw, right_sub, "regular", sub_sz, right_budget) if right_sub else (None, 0, 0)
    right_text_w = max(p_tw, s_tw)
    right_x = w - pad_x - right_text_w

    if params_txt and right_sub:
        _draw_text_ycenter(draw, right_x, cy - row_gap, params_txt, p_font, primary)
        _draw_text_ycenter(draw, right_x, cy + row_gap * 1.05, right_sub, s_font, secondary)
    elif params_txt:
        _draw_text_ycenter(draw, right_x, cy, params_txt, p_font, primary)
    elif right_sub:
        _draw_text_ycenter(draw, right_x, cy, right_sub, s_font, secondary)

    if cfg.show_brand_mark and badge_d > 0:
        div_x = right_x - div_gap
        div_half = bar_h * 0.25
        line_w = max(1, bar_h // 55)
        draw.line((div_x, cy - div_half, div_x, cy + div_half), fill=divider, width=line_w)
        # Small brand accent tick at bottom of divider for subtle craft
        draw.line((div_x, cy + div_half * 0.35, div_x, cy + div_half), fill=accent_rgb, width=max(2, line_w))
        badge = _draw_brand_badge(meta.get("brand", "generic"), meta.get("accent_color", "#C89B54"), badge_d, dark_bg=dark)
        bx = int(round(div_x - div_gap - badge_d))
        by = int(round(cy - badge_d / 2.0))
        if bx > pad_x + 24:
            canvas.paste(badge, (bx, by), badge)

    return canvas


def _render_gallery_frame(im: Image.Image, meta: dict[str, Any], cfg: WatermarkConfig) -> Image.Image:
    w, h = im.size
    u = max(min(w, h), 0.72 * max(w, h))
    scale = float(cfg.border_scale)
    side = max(18, int(round(u * 0.048 * scale)))
    bottom = max(72, int(round(u * 0.165 * scale)))

    cw = w + side * 2
    ch = h + side + bottom
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    canvas.paste(im, (side, side))
    draw = ImageDraw.Draw(canvas)

    # Hairline museum mat border around image
    draw.rectangle((side - 1, side - 1, side + w, side + h), outline=(232, 234, 238), width=1)

    camera_txt = ((meta.get("camera") or "").strip() or "SUPERPHOTO").upper()
    lens_txt = (meta.get("lens") or "").strip() if cfg.show_lens else ""
    params_txt = format_params_line(meta, show_ev=cfg.show_ev, sep="   ")
    sub_txt = _secondary_right_line(meta, cfg)
    if lens_txt:
        sub_txt = f"{lens_txt}  ·  {sub_txt}" if sub_txt else lens_txt

    max_w = max(100, cw - side * 2)
    base_y = side + h
    cx = cw / 2.0

    title_sz = max(13, int(round(bottom * 0.175)))
    param_sz = max(11, int(round(bottom * 0.135)))
    sub_sz = max(10, int(round(bottom * 0.105)))

    primary = (26, 27, 30)
    mid = (76, 80, 88)
    muted = (138, 142, 150)
    accent = _hex_to_rgb(meta.get("accent_color", "#C89B54"))

    # Top centered title (+ optional subtle badge)
    cam_font, cam_tw, _ = _fit_font(draw, camera_txt, "bold", title_sz, max_w - (int(title_sz * 2.2) if cfg.show_brand_mark else 0))
    y1 = base_y + bottom * 0.28
    if cfg.show_brand_mark:
        bd = max(16, int(round(title_sz * 1.15)))
        gap = max(8, int(round(bd * 0.45)))
        total_w = bd + gap + cam_tw
        start_x = cx - total_w / 2.0
        badge = _draw_brand_badge(meta.get("brand", "generic"), meta.get("accent_color", "#C89B54"), bd, dark_bg=False)
        canvas.paste(badge, (int(round(start_x)), int(round(y1 - bd / 2.0))), badge)
        _draw_text_ycenter(draw, start_x + bd + gap, y1, camera_txt, cam_font, primary)
    else:
        _draw_text_ycenter(draw, cx - cam_tw / 2.0, y1, camera_txt, cam_font, primary)

    # Short centered accent hairline
    rule_w = max(24, int(round(bottom * 0.22)))
    y_rule = base_y + bottom * 0.46
    draw.line((cx - rule_w / 2.0, y_rule, cx + rule_w / 2.0, y_rule), fill=accent, width=max(2, bottom // 75))

    # Parameters line
    if params_txt:
        p_font, p_tw, _ = _fit_font(draw, params_txt, "medium", param_sz, max_w)
        _draw_text_ycenter(draw, cx - p_tw / 2.0, base_y + bottom * 0.62, params_txt, p_font, mid)

    # Lens / date / location footer
    if sub_txt:
        s_font, s_tw, _ = _fit_font(draw, sub_txt, "regular", sub_sz, max_w)
        _draw_text_ycenter(draw, cx - s_tw / 2.0, base_y + bottom * 0.80, sub_txt, s_font, muted)

    return canvas


def _render_blur_card(im: Image.Image, meta: dict[str, Any], cfg: WatermarkConfig) -> Image.Image:
    w, h = im.size
    u = max(min(w, h), 0.72 * max(w, h))
    scale = float(cfg.border_scale)
    pad_x = max(28, int(round(u * 0.085 * scale)))
    pad_top = max(28, int(round(u * 0.082 * scale)))
    pad_bot = max(92, int(round(u * 0.195 * scale)))
    radius = max(10, int(round(u * 0.022)))

    cw = w + pad_x * 2
    ch = h + pad_top + pad_bot

    # Build smooth ambient blurred backdrop from downsampled image for speed + buttery bokeh
    small_w = max(64, cw // 8)
    small_h = max(64, ch // 8)
    bg = im.resize((small_w, small_h), Image.Resampling.BILINEAR)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=max(8, min(small_w, small_h) // 8)))
    bg = bg.resize((cw, ch), Image.Resampling.BICUBIC)
    # Darken backdrop slightly + bottom scrim so white text always reads clearly
    overlay = Image.new("RGBA", (cw, ch), (12, 13, 16, 118))
    scrim_h = int(pad_bot * 1.45)
    scrim = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    s_px = scrim.load()
    for y in range(scrim_h):
        alpha = int(round(92 * ((y / max(scrim_h - 1, 1)) ** 1.3)))
        s_px[0, y] = (8, 9, 12, alpha)
    scrim = scrim.resize((cw, scrim_h), Image.Resampling.BILINEAR)
    bg_rgba = bg.convert("RGBA")
    bg_rgba = Image.alpha_composite(bg_rgba, overlay)
    bg_rgba.alpha_composite(scrim, (0, ch - scrim_h))

    # Soft floating drop shadow behind rounded foreground card
    shadow_pad = max(18, int(round(u * 0.045)))
    shadow_offset_y = max(6, int(round(u * 0.014)))
    shadow_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    sh_draw = ImageDraw.Draw(shadow_layer)
    sh_draw.rounded_rectangle(
        (pad_x, pad_top + shadow_offset_y, pad_x + w, pad_top + h + shadow_offset_y),
        radius=radius,
        fill=(0, 0, 0, 145),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_pad * 0.55))
    bg_rgba = Image.alpha_composite(bg_rgba, shadow_layer)

    # Rounded mask for crisp foreground photograph
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    card = im.convert("RGBA")
    bg_rgba.paste(card, (pad_x, pad_top), mask)

    # Subtle glass edge highlight on the card
    edge_draw = ImageDraw.Draw(bg_rgba)
    edge_draw.rounded_rectangle(
        (pad_x, pad_top, pad_x + w - 1, pad_top + h - 1),
        radius=radius,
        outline=(255, 255, 255, 42),
        width=max(1, u // 700),
    )

    canvas = bg_rgba.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    cx = cw / 2.0
    base_y = pad_top + h
    max_w = max(100, cw - pad_x * 2)

    camera_txt = (meta.get("camera") or "").strip() or "SUPERPHOTO"
    lens_txt = (meta.get("lens") or "").strip() if cfg.show_lens else ""
    params_txt = format_params_line(meta, show_ev=cfg.show_ev, sep="   ")
    sub_txt = _secondary_right_line(meta, cfg)
    if lens_txt:
        sub_txt = f"{lens_txt}  ·  {sub_txt}" if sub_txt else lens_txt

    title_sz = max(14, int(round(pad_bot * 0.18)))
    param_sz = max(11, int(round(pad_bot * 0.135)))
    sub_sz = max(10, int(round(pad_bot * 0.105)))

    y1 = base_y + pad_bot * 0.30
    cam_font, cam_tw, _ = _fit_font(draw, camera_txt, "bold", title_sz, max_w - (int(title_sz * 2.2) if cfg.show_brand_mark else 0))
    if cfg.show_brand_mark:
        bd = max(16, int(round(title_sz * 1.18)))
        gap = max(8, int(round(bd * 0.45)))
        total_w = bd + gap + cam_tw
        start_x = cx - total_w / 2.0
        badge = _draw_brand_badge(meta.get("brand", "generic"), meta.get("accent_color", "#C89B54"), bd, dark_bg=True)
        canvas.paste(badge, (int(round(start_x)), int(round(y1 - bd / 2.0))), badge)
        _draw_text_ycenter(draw, start_x + bd + gap, y1, camera_txt, cam_font, (250, 250, 252))
    else:
        _draw_text_ycenter(draw, cx - cam_tw / 2.0, y1, camera_txt, cam_font, (250, 250, 252))

    if params_txt:
        p_font, p_tw, _ = _fit_font(draw, params_txt, "medium", param_sz, max_w)
        _draw_text_ycenter(draw, cx - p_tw / 2.0, base_y + pad_bot * 0.56, params_txt, p_font, (226, 230, 236))

    if sub_txt:
        s_font, s_tw, _ = _fit_font(draw, sub_txt, "regular", sub_sz, max_w)
        _draw_text_ycenter(draw, cx - s_tw / 2.0, base_y + pad_bot * 0.77, sub_txt, s_font, (182, 188, 198))

    return canvas


def _render_film_polaroid(im: Image.Image, meta: dict[str, Any], cfg: WatermarkConfig) -> Image.Image:
    w, h = im.size
    u = max(min(w, h), 0.72 * max(w, h))
    scale = float(cfg.border_scale)
    side = max(18, int(round(u * 0.052 * scale)))
    top = max(20, int(round(u * 0.056 * scale)))
    bottom = max(88, int(round(u * 0.215 * scale)))

    cw = w + side * 2
    ch = h + top + bottom
    paper = (246, 243, 236)  # Warm archival photographic paper
    ink = (36, 34, 31)
    muted = (122, 117, 108)
    accent = _hex_to_rgb(meta.get("accent_color", "#C89B54"))

    canvas = Image.new("RGB", (cw, ch), paper)
    canvas.paste(im, (side, top))
    draw = ImageDraw.Draw(canvas)

    # Subtle recessed print edge
    draw.rectangle((side - 1, top - 1, side + w, top + h), outline=(218, 213, 203), width=max(1, u // 900))

    base_y = top + h
    inner_w = max(100, w)
    title_sz = max(13, int(round(bottom * 0.145)))
    mono_sz = max(11, int(round(bottom * 0.115)))
    sub_sz = max(10, int(round(bottom * 0.095)))

    camera_txt = (meta.get("camera") or "").strip() or "FILM ARCHIVE"
    lens_txt = (meta.get("lens") or "").strip() if cfg.show_lens else ""
    params_txt = format_params_line(meta, show_ev=cfg.show_ev, sep="  /  ")
    note_txt = _secondary_right_line(meta, cfg)

    # Top row inside chin: signature/camera on left, film frame tag on right
    y1 = base_y + bottom * 0.28
    left_label = (meta.get("signature") or "").strip() or camera_txt
    l_font, _, _ = _fit_font(draw, left_label, "serif", title_sz, int(inner_w * 0.62))
    _draw_text_ycenter(draw, side, y1, left_label, l_font, ink)

    stamp = (meta.get("date_only") or "").strip() if cfg.show_date else ""
    if not stamp:
        stamp = (meta.get("brand_label") or "35MM").upper()
    r_font, r_tw, _ = _fit_font(draw, stamp, "mono", mono_sz, int(inner_w * 0.34))
    _draw_text_ycenter(draw, side + w - r_tw, y1, stamp, r_font, _readable_on_light(accent))

    # Thin archival divider
    y_div = base_y + bottom * 0.47
    draw.line((side, y_div, side + w, y_div), fill=(224, 219, 209), width=max(1, bottom // 110))

    # Bottom technical readout in crisp mono
    y2 = base_y + bottom * 0.64
    tech_left = params_txt or f"{w}×{h}"
    tl_font, _, _ = _fit_font(draw, tech_left, "mono", mono_sz, int(inner_w * 0.72))
    _draw_text_ycenter(draw, side, y2, tech_left, tl_font, ink)

    if cfg.show_brand_mark:
        bd = max(16, int(round(bottom * 0.22)))
        badge = _draw_brand_badge(meta.get("brand", "generic"), meta.get("accent_color", "#C89B54"), bd, dark_bg=False)
        canvas.paste(badge, (side + w - bd, int(round(base_y + bottom * 0.70 - bd / 2.0))), badge)

    y3 = base_y + bottom * 0.82
    sub_parts = [p for p in (camera_txt if left_label != camera_txt else "", lens_txt, note_txt) if p]
    sub_line = "  ·  ".join(sub_parts)
    if sub_line:
        s_font, _, _ = _fit_font(draw, sub_line, "regular", sub_sz, int(inner_w * 0.82))
        _draw_text_ycenter(draw, side, y3, sub_line, s_font, muted)

    return canvas


def _render_corner_overlay(im: Image.Image, meta: dict[str, Any], cfg: WatermarkConfig) -> Image.Image:
    w, h = im.size
    u = max(min(w, h), 0.72 * max(w, h))
    scale = float(cfg.border_scale)
    band_h = max(56, int(round(u * 0.135 * scale)))
    pad_x = max(20, int(round(w * 0.035)))
    pad_b = max(14, int(round(band_h * 0.24)))

    # Smooth subtle dark gradient scrim only at the bottom 18% of the photo
    scrim_h = min(h, int(round(band_h * 1.85)))
    scrim = Image.new("RGBA", (1, scrim_h), (0, 0, 0, 0))
    px = scrim.load()
    for y in range(scrim_h):
        t = y / max(scrim_h - 1, 1)
        alpha = int(round(162 * (t ** 1.75)))
        px[0, y] = (10, 11, 14, alpha)
    scrim = scrim.resize((w, scrim_h), Image.Resampling.BILINEAR)

    base = im.convert("RGBA")
    base.alpha_composite(scrim, (0, h - scrim_h))
    draw = ImageDraw.Draw(base)

    camera_txt = (meta.get("camera") or "").strip() or (meta.get("signature") or "SUPERPHOTO")
    lens_txt = (meta.get("lens") or "").strip() if cfg.show_lens else ""
    params_txt = format_params_line(meta, show_ev=cfg.show_ev, sep="  ")
    right_sub = _secondary_right_line(meta, cfg)

    title_sz = max(13, int(round(band_h * 0.22)))
    sub_sz = max(10, int(round(band_h * 0.145)))
    row_gap = band_h * 0.135
    cy = h - pad_b - band_h * 0.26

    usable_w = max(80, w - 2 * pad_x)
    left_budget = int(usable_w * 0.45)
    right_budget = int(usable_w * 0.48)

    left_x = float(pad_x)
    if cfg.show_brand_mark:
        bd = max(16, int(round(band_h * 0.34)))
        badge = _draw_brand_badge(meta.get("brand", "generic"), meta.get("accent_color", "#C89B54"), bd, dark_bg=True)
        base.paste(badge, (pad_x, int(round(cy - bd / 2.0))), badge)
        left_x = pad_x + bd + max(10, int(round(bd * 0.42)))
        left_budget = max(60, left_budget - bd - 14)

    cam_font, _, _ = _fit_font(draw, camera_txt, "bold", title_sz, left_budget)
    if lens_txt:
        l_font, _, _ = _fit_font(draw, lens_txt, "regular", sub_sz, left_budget)
        _draw_text_ycenter(draw, left_x, cy - row_gap, camera_txt, cam_font, (250, 250, 252, 245))
        _draw_text_ycenter(draw, left_x, cy + row_gap * 1.05, lens_txt, l_font, (212, 216, 222, 215))
    else:
        _draw_text_ycenter(draw, left_x, cy, camera_txt, cam_font, (250, 250, 252, 245))

    p_font, p_tw, _ = _fit_font(draw, params_txt, "bold", int(round(title_sz * 0.94)), right_budget)
    s_font, s_tw, _ = _fit_font(draw, right_sub, "regular", sub_sz, right_budget) if right_sub else (None, 0, 0)
    if params_txt and right_sub:
        _draw_text_ycenter(draw, w - pad_x - p_tw, cy - row_gap, params_txt, p_font, (250, 250, 252, 245))
        _draw_text_ycenter(draw, w - pad_x - s_tw, cy + row_gap * 1.05, right_sub, s_font, (206, 210, 218, 210))
    elif params_txt:
        _draw_text_ycenter(draw, w - pad_x - p_tw, cy, params_txt, p_font, (250, 250, 252, 245))
    elif right_sub:
        _draw_text_ycenter(draw, w - pad_x - s_tw, cy, right_sub, s_font, (206, 210, 218, 210))

    return base.convert("RGB")


def render_watermark(
    image: Image.Image,
    meta: dict[str, Any],
    config: WatermarkConfig | None = None,
    *,
    max_edge: int = 0,
) -> Image.Image:
    """Render a watermarked sRGB PIL Image according to ``config``."""
    cfg = config or WatermarkConfig()
    resolved = resolve_metadata(meta, cfg)
    im = _prepare_base_image(image, max_edge=max_edge)

    if cfg.style == "leica_bar":
        return _render_classic_bar(im, resolved, cfg, dark=False)
    if cfg.style == "dark_bar":
        return _render_classic_bar(im, resolved, cfg, dark=True)
    if cfg.style == "gallery_frame":
        return _render_gallery_frame(im, resolved, cfg)
    if cfg.style == "blur_card":
        return _render_blur_card(im, resolved, cfg)
    if cfg.style == "film_polaroid":
        return _render_film_polaroid(im, resolved, cfg)
    if cfg.style == "corner_overlay":
        return _render_corner_overlay(im, resolved, cfg)
    return _render_classic_bar(im, resolved, cfg, dark=False)


def encode_jpeg_bytes(image: Image.Image, *, quality: int = 94) -> bytes:
    buf = BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality, subsampling=0, icc_profile=srgb_profile_bytes())
    return buf.getvalue()
