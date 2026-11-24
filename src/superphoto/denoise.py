"""LightCraft AI Denoise on the delivered render of a noisy Bayer RAW.

When. Only a RAW source, and only the render that is delivered: the solver's probes and the look probes stay
plain, since denoise leaves tone and colour alone (0.35/255 after a blur on a 45 MP NEF, 2026-11-24). The photo's
ISO is scaled by how far the plan lifts its darks (exposure, shadows, blacks, the solved curve at a dark tone,
and the strongest lifting mask). From an effective ISO of 1600 the amount is 50, +15 per doubling, at most 90:
on a 45 MP Z 7II NEF pushed 4 stops, 50 removed part of the worm-like noise on a dark door and 100 all of it,
but left a slight mottling on the flat paint. A JPEG is left alone (the camera has denoised it already).
LightCraft denoises Bayer raws only: a linear DNG such as iPhone ProRAW renders unchanged, and the setting is
dropped from its develop.json.

How. LightCraft keeps the denoised raw data (a .lcdn file of ~150 MB for 45 MP, ~6 s on an RTX 3080 Ti) in the
library the process opened, and one library can be open in one process only. So a denoised render opens a new
library under <data>/denoise/libraries/, seeded with hard links to the products already made from the same file
content (<data>/denoise/products/<sha1 of the file>/), and a product made there is moved into that store when the
render is done. Measured on the NEF (2026-11-24): `app.export` makes a missing product by itself (12.2 s at full
size, against 6.1 s once made); a seeded library reuses the product without recomputing it and renders
pixel-identical to the library that made it. The product's name is LightCraft's own (not the file's MD5), hence
the store keyed by our own hash. Denoised renders run one at a time: one GPU, and a 45 MP RAW per process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

DENOISE_FROM_ISO = 1600.0
BASE_AMOUNT = 50.0
AMOUNT_PER_STOP = 15.0
MAX_AMOUNT = 90.0
SHADOW_STOPS = 1.5          # Shadows +100: about how far the deepest tones rise.
BLACK_STOPS = 0.5           # Blacks +100.
CURVE_TONE = 0.1            # Display value where the curve's lift is read (a deep shadow).
DISPLAY_GAMMA = 2.2
PRODUCT_SUFFIX = ".lcdn"
STALE_LIBRARY_SECONDS = 6 * 3600

Outcome = Literal["denoised", "plain"]

_RENDER_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_KEYS: dict[tuple[str, int, int], str] = {}
_NOT_APPLICABLE: set[str] = set()       # Content LightCraft could not denoise (not a Bayer raw), this process.


def enabled() -> bool:
    from .config import get_settings

    return get_settings().ai_denoise


def store_root() -> Path:
    from .config import get_settings

    return get_settings().data_dir / "denoise"


def cache_limit_bytes() -> int:
    from .config import get_settings

    return int(get_settings().ai_denoise_cache_gb * 2**30)


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def iso_of(source: Path) -> float:
    """The ISO in the file's EXIF, 0 when it cannot be read."""
    from .ingest import read_exif

    exif = read_exif(Path(source))
    for key in ("ISOSpeedRatings", "PhotographicSensitivity", "ISO"):
        raw = exif.get(key)
        if not raw:
            continue
        try:
            return float(str(raw).strip("[]").split(",")[0].split("/")[0].split()[0])
        except (ValueError, IndexError):
            continue
    return 0.0


def _lift(values: dict[str, Any]) -> float:
    return (max(0.0, _number(values.get("exposure"))) + SHADOW_STOPS * max(0.0, _number(values.get("shadows"))) / 100.0
            + BLACK_STOPS * max(0.0, _number(values.get("blacks"))) / 100.0)


def _curve_lift(points: Any) -> float:
    """Stops the master curve adds at a deep-shadow display value (0 when it darkens there)."""
    try:
        curve = sorted((float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1]))
                       for p in points)
    except (TypeError, ValueError, KeyError, IndexError):
        return 0.0
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= CURVE_TONE <= x1 and x1 > x0:
            y = y0 + (y1 - y0) * (CURVE_TONE - x0) / (x1 - x0)
            return max(0.0, DISPLAY_GAMMA * math.log2(y / CURVE_TONE)) if y > 0 else 0.0
    return 0.0


def push_stops(document: dict[str, Any]) -> float:
    """About how many stops the document lifts the photo's darks: globally, plus the strongest lifting mask."""
    stops = _lift(document.get("light") or {}) + _curve_lift((document.get("curve") or {}).get("master") or [])
    local = 0.0
    for mask in document.get("masks") or []:
        if isinstance(mask, dict) and mask.get("visible", True):
            adjust = mask.get("adjust") or {}
            share = min(100.0, max(0.0, _number(adjust.get("amount", 100.0)))) / 100.0
            local = max(local, share * _lift(adjust))
    return stops + local


def decide(source: Path, source_kind: str, document: dict[str, Any]) -> dict[str, float] | None:
    """The AI Denoise amount for the delivered render of ``document``, with why; None when it is not wanted."""
    if source_kind != "raw" or not enabled():
        return None
    iso = iso_of(source)
    if iso <= 0:
        return None
    stops = push_stops(document)
    effective = iso * 2.0 ** stops
    if effective < DENOISE_FROM_ISO:
        return None
    amount = min(MAX_AMOUNT, BASE_AMOUNT + AMOUNT_PER_STOP * math.log2(effective / DENOISE_FROM_ISO))
    return {"amount": round(amount, 1), "iso": iso, "stops": round(stops, 2), "effective_iso": round(effective)}


def with_denoise(document: dict[str, Any], amount: float) -> dict[str, Any]:
    enhance = {**(document.get("enhance") or {}), "denoise_on": True, "denoise": float(amount)}
    return {**document, "enhance": enhance}


def requested(document: dict[str, Any]) -> bool:
    enhance = document.get("enhance")
    return isinstance(enhance, dict) and bool(enhance.get("denoise_on")) and _number(enhance.get("denoise")) > 0


def without(document: dict[str, Any]) -> dict[str, Any]:
    """The document without AI Denoise (other enhance settings kept)."""
    enhance = {key: value for key, value in (document.get("enhance") or {}).items()
               if key not in ("denoise_on", "denoise")}
    stripped = {key: value for key, value in document.items() if key != "enhance"}
    if enhance:
        stripped["enhance"] = enhance
    return stripped


def recorded(folder: Path) -> bool:
    """Whether the develop.json in ``folder`` was rendered with AI Denoise."""
    try:
        return requested(json.loads((Path(folder) / "develop.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, AttributeError):
        return False


def content_key(path: Path) -> str:
    """SHA-1 of the file, cached per path, size and modification time."""
    path = Path(path).resolve()
    stat = path.stat()
    cache = (str(path), stat.st_size, stat.st_mtime_ns)
    with _STATE_LOCK:
        if cache in _KEYS:
            return _KEYS[cache]
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    with _STATE_LOCK:
        _KEYS[cache] = digest.hexdigest()
    return _KEYS[cache]


def _products(root: Path, key: str) -> list[Path]:
    folder = root / "products" / key
    return sorted(folder.glob(f"*{PRODUCT_SUFFIX}")) if folder.is_dir() else []


def _new_library(root: Path, key: str) -> Path:
    """A fresh library holding this content's products (hard links; copies where links are not possible)."""
    libraries = root / "libraries"
    libraries.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - STALE_LIBRARY_SECONDS
    for folder in libraries.iterdir():          # Left behind by a process that was stopped.
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            continue
    library = libraries / uuid.uuid4().hex
    (library / "denoise").mkdir(parents=True)
    for product in _products(root, key):
        target = library / "denoise" / product.name
        try:
            os.link(product, target)
        except OSError:
            shutil.copy2(product, target)
    return library


def _publish(root: Path, key: str, library: Path) -> bool:
    """Move products made in ``library`` into the store. True when this content has a product now."""
    folder = root / "products" / key
    before = {product.name for product in _products(root, key)}
    made = [product for product in (library / "denoise").glob(f"*{PRODUCT_SUFFIX}") if product.name not in before]
    if made:
        folder.mkdir(parents=True, exist_ok=True)
        for product in made:
            os.replace(product, folder / product.name)
        for name in before:                 # Not used this time (e.g. a new denoise model): replaced.
            (folder / name).unlink(missing_ok=True)
    if folder.is_dir():
        os.utime(folder)                    # Most recently used, for _trim.
    _trim(root, keep=key)
    return bool(_products(root, key))


def _trim(root: Path, *, keep: str) -> None:
    """Drop the least recently used products beyond the cache limit (never the one just used)."""
    products = root / "products"
    if not products.is_dir():
        return
    folders: list[tuple[float, Path, int]] = []
    for folder in products.iterdir():
        try:
            if folder.is_dir():
                size = sum(item.stat().st_size for item in folder.iterdir() if item.is_file())
                folders.append((folder.stat().st_mtime, folder, size))
        except OSError:
            continue
    total, limit = sum(size for _, _, size in folders), cache_limit_bytes()
    for _, folder, size in sorted(folders, key=lambda item: item[0]):
        if total <= limit:
            break
        if folder.name != keep:
            shutil.rmtree(folder, ignore_errors=True)
            total -= size


def render(binary: Path, source: Path, document: dict[str, Any], output: Path, *, edge: int, quality: int,
           options: dict[str, Any] | None, timeout: float) -> Outcome | None:
    """Render a document that asks for AI Denoise, in a library seeded from the store.

    "denoised": rendered with it. "plain": rendered, but LightCraft could not denoise this file (not a Bayer
    raw), so the render is the plain one. None: not rendered (no session, or this content is known not to
    denoise); render it without AI Denoise.
    """
    from .lightcraft import LightCraftError, RenderSession, _SessionError, _SessionTimeout

    root = store_root()
    with _RENDER_LOCK:
        try:
            key = content_key(source)
        except OSError:
            return None
        with _STATE_LOCK:
            if key in _NOT_APPLICABLE:
                return None
        try:
            library = _new_library(root, key)
        except OSError:
            return None
        try:
            try:
                session = RenderSession(binary, source, library=library)
            except _SessionError:
                return None
            try:
                session.render(document, output, edge=edge, quality=quality, options=options, timeout=timeout)
            except _SessionTimeout as exc:
                raise LightCraftError("LightCraft 导出超时") from exc
            except _SessionError:
                return None
            finally:
                session.close()
            try:
                made = _publish(root, key, library)
            except OSError:
                made = any((library / "denoise").glob(f"*{PRODUCT_SUFFIX}"))
        finally:
            shutil.rmtree(library, ignore_errors=True)
        if made:
            return "denoised"
        with _STATE_LOCK:
            _NOT_APPLICABLE.add(key)
        return "plain"
