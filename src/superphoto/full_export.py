"""Full-size delivery of a finished version.

Finishes are delivered at 2400 px (``lightroom_export_edge``). Every LightCraft render keeps
the exact develop document that produced it as develop.json beside it, so a full-size file is
that document rendered once more on a fresh copy of the original, with no size limit. Nothing
is planned or solved again, so tone and colour match the version the user picked. Sharpening,
texture and noise reduction now act on the native pixels, so fine detail differs a little.
Measured on a 45 MP JPEG (job aa2672ea, slot B): the 2400 px re-render matched the delivered
file exactly. The full-size render, scaled back down, differed by 0.75/255 after a blur.

A version delivered with AI Denoise keeps it: its develop.json asks for it, and render_document reuses the
denoised raw data from data/denoise while it is there (about 6 s more on a 45 MP NEF when it has to be made again).

Formats: a JPEG (quality 95, sRGB) for sharing and printing, and a 16-bit sRGB TIFF (zip) as a
master for further editing. The TIFF of a 45 MP photo is about 210 MB and takes about 11 s.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .db import get_job, get_refinement

FORMATS: dict[str, dict[str, Any]] = {
    "jpeg": {"suffix": ".jpg", "quality": 95, "options": {"colorSpace": "srgb"}, "media_type": "image/jpeg"},
    "tiff": {"suffix": ".tif", "quality": 95, "options": {"bitDepth": 16, "colorSpace": "srgb"},
             "media_type": "image/tiff"},
}
FOLDER = "full"
RENDER_TIMEOUT = 900.0
# One full-size render at a time: a 45 MP 16-bit render holds several hundred MB.
_render_lock = threading.Lock()
_NAME = re.compile(r"^(?P<target>[A-Z](?:-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?)"
                   r"(?P<suffix>\.jpg|\.tif)$")


class FullExportUnavailable(ValueError):
    """This version cannot be rendered at full size; the message says why, for the user."""


def target_name(slot: str, version: str | None) -> str:
    return f"{slot}-{version}" if version else slot


def parse_name(name: str) -> tuple[str, str] | None:
    """(target folder, format) of a full-size file name such as "B.tif"; None if not one of ours."""
    match = _NAME.match(name)
    if not match:
        return None
    fmt = next(key for key, spec in FORMATS.items() if spec["suffix"] == match["suffix"])
    return match["target"], fmt


def file_path(root: Path, name: str) -> Path | None:
    parsed = parse_name(name)
    if parsed is None:
        return None
    target, fmt = parsed
    return root / FOLDER / target / f"render{FORMATS[fmt]['suffix']}"


def download_name(slot: str, version: str | None, fmt: str) -> str:
    return f"SuperPhoto-{slot}{'-微调-' + version[:8] if version else ''}-原尺寸{FORMATS[fmt]['suffix']}"


def _in_job(root: Path, job_id: str, stored: str) -> Path:
    """A path recorded in an audit, moved under this job folder when the data directory has moved."""
    path = Path(stored)
    if path.exists():
        return path
    parts = re.split(r"[\\/]+", stored)
    if job_id in parts:
        return root.joinpath(*parts[parts.index(job_id) + 1:])
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def develop_record(conn: sqlite3.Connection, root: Path, job_id: str, slot: str, version: str | None) -> Path:
    """The develop.json that produced this version's delivered render."""
    if version:
        record = get_refinement(conn, version)
        if not record or record["job_id"] != job_id or record["slot"] != slot or record["status"] != "ready":
            raise FullExportUnavailable("找不到这个微调版本")
        folder = root / "refinements" / version
        audit = _read_json(folder / "audit.json")
        stored = (audit.get("lightroom") or {}).get("export") if isinstance(audit.get("lightroom"), dict) else None
        # A directed refinement renders straight into exports/ (directed.run_refinement, final_name "exports").
        candidates = [_in_job(root, job_id, stored).parent / "develop.json"] if stored else []
        candidates.append(folder / "exports" / "develop.json")
    else:
        audit = _read_json(root / "lightroom-audits" / f"variant_{slot}.json")
        stored = audit.get("accepted_export") if audit.get("status", "delivered") == "delivered" else None
        candidates = [_in_job(root, job_id, stored).parent / "develop.json"] if stored else []
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FullExportUnavailable("这张成片没有保存 LightCraft 显影记录（基础模式或旧版本生成），无法导出原尺寸。")


def export_full(settings: Settings, conn: sqlite3.Connection, job_id: str, slot: str, version: str | None,
                fmt: str, *, binary: Path) -> dict[str, Any]:
    """Render (or reuse) the full-size file of one version. Returns its path, size and timing."""
    from .lightcraft import render_document

    spec = FORMATS[fmt]
    root = settings.jobs_dir / job_id
    job = get_job(conn, job_id)
    if not job:
        raise FullExportUnavailable("找不到这次处理记录")
    document = _read_json(develop_record(conn, root, job_id, slot, version))
    original = root / Path(job.get("original_file") or "original.jpg").name
    target = target_name(slot, version)
    folder = root / FOLDER / target
    output = folder / f"render{spec['suffix']}"
    name = f"{target}{spec['suffix']}"
    with _render_lock:
        if document and output.is_file() and _read_json(folder / "develop.json") == document:
            meta = _read_json(folder / f"render{spec['suffix']}.json")
            return {**meta, "path": output, "name": name, "cached": True}
        if not document:
            raise FullExportUnavailable("这张成片的显影记录无法读取，无法导出原尺寸。")
        if not original.is_file():
            raise FullExportUnavailable("原片已不在本机，无法导出原尺寸。")
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / f"source{original.suffix.lower()}"
        started = time.monotonic()
        try:
            shutil.copy2(original, source)
            output.unlink(missing_ok=True)
            render_document(binary, source, document, folder, edge=0, timeout=RENDER_TIMEOUT,
                            name=output.name, quality=spec["quality"], options=spec["options"])
        finally:
            source.unlink(missing_ok=True)
        meta = {"format": fmt, "bytes": output.stat().st_size, "seconds": round(time.monotonic() - started, 1),
                **_pixel_size(output)}
        (folder / f"render{spec['suffix']}.json").write_text(json.dumps(meta), encoding="utf-8")
    return {**meta, "path": output, "name": name, "cached": False}


def _pixel_size(path: Path) -> dict[str, int]:
    """Width and height from the file header (no decode of the full image)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            return {"width": image.width, "height": image.height}
    except (OSError, ValueError):
        return {}
