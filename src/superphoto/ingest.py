from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .color import srgb_to_linear
from .models import SourceKind

RAW_EXTS = {
    ".arw",
    ".cr2",
    ".cr3",
    ".nef",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
    ".srw",
    ".raw",
    ".nrw",
    ".rwl",
}
JPEG_EXTS = {".jpg", ".jpeg"}
PNG_EXTS = {".png"}
TIFF_EXTS = {".tif", ".tiff"}
ALLOWED_EXTS = RAW_EXTS | JPEG_EXTS | PNG_EXTS | TIFF_EXTS


class IngestError(RuntimeError):
    pass


def source_kind(path: Path) -> SourceKind:
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        return "raw"
    if ext in JPEG_EXTS:
        return "jpeg"
    if ext in PNG_EXTS:
        return "png"
    if ext in TIFF_EXTS:
        return "tiff"
    return "unknown"


def load_linear(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    kind = source_kind(path)
    if kind == "unknown":
        raise IngestError(f"unsupported file type: {path.suffix}")
    if kind == "raw":
        return _load_raw(path)
    return _load_raster(path, kind)


def _load_raster(path: Path, kind: SourceKind) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        profile_present = bool(im.info.get("icc_profile"))
        rgb = image_to_srgb(im)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = srgb_to_linear(arr)
    meta = {"source_kind": kind, "native_width": linear.shape[1], "native_height": linear.shape[0],
            "working_color_space": "sRGB", "icc_converted": profile_present}
    return linear, meta


@lru_cache(maxsize=1)
def srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def image_to_srgb(image: Image.Image) -> Image.Image:
    """Honor embedded primaries before interpreting RGB values as sRGB."""
    profile = image.info.get("icc_profile")
    if not profile:
        return image.convert("RGB")
    try:
        return ImageCms.profileToProfile(
            image, ImageCms.ImageCmsProfile(BytesIO(profile)),
            ImageCms.ImageCmsProfile(BytesIO(srgb_profile_bytes())), outputMode="RGB",
        )
    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
        raise IngestError(f"无法读取照片的 ICC 色彩配置：{exc}") from exc


def _load_raw(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import rawpy
    except ImportError as exc:
        raise IngestError(
            "RAW support requires rawpy. Install with: uv sync --extra raw"
        ) from exc
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            four_color_rgb=False,
            highlight_mode=rawpy.HighlightMode.Blend,
        )
        sizes = raw.sizes
        meta = {
            "source_kind": "raw",
            "native_width": int(sizes.width),
            "native_height": int(sizes.height),
            "raw_type": str(getattr(raw, "raw_type", "")),
        }
    linear = rgb.astype(np.float32) / 65535.0
    return linear, meta


def downscale_linear(linear: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = linear.shape[:2]
    edge = max(h, w)
    if edge <= max_edge:
        return linear
    scale = max_edge / edge
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    # Float planes preserve deep shadows and scene values above 1.0. An 8-bit
    # linear proxy erased shadow gradations before diagnosis even began.
    planes = [
        np.asarray(Image.fromarray(linear[..., ch].astype(np.float32)).resize(
            (nw, nh), Image.Resampling.LANCZOS), dtype=np.float32)
        for ch in range(linear.shape[2])
    ]
    return np.maximum(np.stack(planes, axis=-1), 0.0)


def read_exif(path: Path) -> dict[str, str]:
    try:
        import exifread
    except ImportError:
        return _pillow_exif(path)
    keys = (
        "EXIF ISOSpeedRatings",
        "EXIF PhotographicSensitivity",
        "EXIF FNumber",
        "EXIF ExposureTime",
        "EXIF FocalLength",
        "Image Model",
        "Image Make",
        "EXIF LensModel",
        "EXIF WhiteBalance",
        "EXIF ColorSpace",
        "EXIF DateTimeOriginal",
    )
    out: dict[str, str] = {}
    try:
        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        return _pillow_exif(path)
    for key in keys:
        if key in tags:
            out[key.split(" ", 1)[-1]] = str(tags[key])
    return out


def _pillow_exif(path: Path) -> dict[str, str]:
    try:
        with Image.open(path) as im:
            info = im.getexif()
            if not info:
                return {}
            named = {
                271: "Make",
                272: "Model",
                33434: "ExposureTime",
                33437: "FNumber",
                34855: "ISOSpeedRatings",
                36867: "DateTimeOriginal",
                37386: "FocalLength",
            }
            return {
                name: str(info.get(tag))
                for tag, name in named.items()
                if info.get(tag) is not None
            }
    except Exception:
        return {}
