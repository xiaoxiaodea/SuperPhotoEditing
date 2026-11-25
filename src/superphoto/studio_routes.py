"""FastAPI routes for the EXIF Watermark & Auto-Collage / WeChat Moments Studio."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sqlite3
import urllib.parse
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import collage, style_sync, watermark
from .color import linear_to_srgb
from .config import Settings
from .db import (
    SLOT_LETTERS,
    get_job,
    get_refinement,
    get_refinements,
    get_style_batch,
    get_style_batch_item,
    get_style_batch_items,
    get_variants,
    list_style_batches,
)
from .ingest import ALLOWED_EXTS, IngestError, image_to_srgb, load_linear, source_kind, srgb_profile_bytes
from .store import job_dir

_ASSET_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_UPLOAD_COUNT = 20
_MAX_STORED_EDGE = 3600
_THUMB_EDGE = 480


class ImportJobIn(BaseModel):
    job_id: str
    slot: str = "A"
    version: str | None = None


class ImportBatchIn(BaseModel):
    batch_id: str
    include_anchor: bool = True
    limit: int = Field(default=18, ge=1, le=50)


class WatermarkRequestIn(BaseModel):
    asset_id: str | None = None
    job_id: str | None = None
    slot: str | None = None
    version: str | None = None
    batch_id: str | None = None
    item_id: str | None = None
    config: watermark.WatermarkConfig = Field(default_factory=watermark.WatermarkConfig)
    max_edge: int = Field(default=1400, ge=0, le=6000)


class WatermarkBatchZipIn(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=30)
    config: watermark.WatermarkConfig = Field(default_factory=watermark.WatermarkConfig)
    max_edge: int = Field(default=0, ge=0, le=6000)


class AutoArrangeIn(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=30)
    mode: collage.ArrangeMode = "hero_center"


class CollageRequestIn(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=20)
    config: collage.CollageConfig = Field(default_factory=collage.CollageConfig)
    target_width: int = Field(default=1600, ge=400, le=4200)
    tile_edge: int = Field(default=1600, ge=320, le=2400)


def _studio_assets_dir(settings: Settings) -> Path:
    path = settings.data_dir / "studio" / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _asset_dir(settings: Settings, asset_id: str) -> Path:
    if not _ASSET_ID_RE.match(asset_id or ""):
        raise HTTPException(400, "无效的素材编号")
    return _studio_assets_dir(settings) / asset_id


def _content_disposition(filename: str) -> dict[str, str]:
    safe_ascii = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "export.jpg"
    quoted = urllib.parse.quote(filename)
    return {"Content-Disposition": f"attachment; filename=\"{safe_ascii}\"; filename*=UTF-8''{quoted}"}


def _open_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return image_to_srgb(ImageOps.exif_transpose(im))


def _save_jpeg(im: Image.Image, dest: Path, *, max_edge: int = 0, quality: int = 94) -> tuple[int, int]:
    work = im.convert("RGB")
    if max_edge > 0 and max(work.size) > max_edge:
        w, h = work.size
        scale = max_edge / max(w, h)
        work = work.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
    work.save(dest, format="JPEG", quality=quality, subsampling=0, icc_profile=srgb_profile_bytes())
    return work.size


def _register_asset(
    settings: Settings,
    *,
    name: str,
    source_type: str,
    image: Image.Image,
    metadata: dict[str, Any],
    source_ref: dict[str, Any] | None = None,
    existing_image_path: Path | None = None,
) -> dict[str, Any]:
    asset_id = uuid.uuid4().hex
    folder = _studio_assets_dir(settings) / asset_id
    folder.mkdir(parents=True, exist_ok=True)

    thumb_path = folder / "thumb.jpg"
    _save_jpeg(image, thumb_path, max_edge=_THUMB_EDGE, quality=88)

    if existing_image_path is not None and existing_image_path.is_file():
        image_rel = str(existing_image_path.resolve())
        w, h = image.size
    else:
        source_path = folder / "source.jpg"
        w, h = _save_jpeg(image, source_path, max_edge=_MAX_STORED_EDGE, quality=94)
        image_rel = str(source_path.resolve())

    meta = dict(metadata)
    meta["width"] = w
    meta["height"] = h
    aesthetics = collage.analyze_tile_aesthetics(image, meta)

    record = {
        "id": asset_id,
        "name": name,
        "source_type": source_type,
        "source_ref": source_ref or {},
        "image_path": image_rel,
        "thumb_path": str(thumb_path.resolve()),
        "metadata": meta,
        "aesthetics": aesthetics,
    }
    (folder / "meta.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return _public_asset(record)


def _public_asset(record: dict[str, Any]) -> dict[str, Any]:
    aid = record["id"]
    return {
        "id": aid,
        "name": record["name"],
        "source_type": record.get("source_type", "upload"),
        "source_ref": record.get("source_ref") or {},
        "url": f"/api/studio/assets/{aid}/image",
        "thumb_url": f"/api/studio/assets/{aid}/thumb",
        "metadata": record.get("metadata") or {},
        "aesthetics": record.get("aesthetics") or {},
    }


def _load_asset_record(settings: Settings, asset_id: str) -> dict[str, Any]:
    folder = _asset_dir(settings, asset_id)
    meta_file = folder / "meta.json"
    if not meta_file.is_file():
        raise HTTPException(404, "找不到这张工作坊素材，可能已被清理")
    try:
        record = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "素材元数据损坏") from exc
    img_path = Path(record.get("image_path", ""))
    data_root = settings.data_dir.resolve()
    try:
        img_path.resolve().relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(403, "非法的素材路径") from exc
    if not img_path.is_file():
        raise HTTPException(404, "素材图片文件已不在本机")
    return record


def _resolve_job_media(
    settings: Settings,
    conn: sqlite3.Connection,
    job_id: str,
    slot: str,
    version: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    job = get_job(conn, job_id)
    if not job or job["status"] != "ready":
        raise HTTPException(404, "找不到已完成的修图任务")
    slot_clean = str(slot or "A").strip().upper()
    if slot_clean not in SLOT_LETTERS:
        raise HTTPException(400, f"无效的方案槽位：{slot_clean}")
    variant = next((v for v in get_variants(conn, job_id) if v["slot"] == slot_clean), None)
    if not variant:
        raise HTTPException(404, f"找不到方案 {slot_clean}")

    root = job_dir(settings, job_id)
    if version:
        ref = get_refinement(conn, version)
        if not ref or ref["job_id"] != job_id or ref["slot"] != slot_clean or ref["status"] != "ready":
            raise HTTPException(404, "找不到该微调版本")
        image_path = root / "refinements" / version / "result.jpg"
        label = f"{Path(job.get('original_name') or 'photo').stem}-{slot_clean}-微调.jpg"
    else:
        image_path = root / variant["filename"]
        label = f"{Path(job.get('original_name') or 'photo').stem}-{slot_clean}.jpg"

    if not image_path.is_file():
        raise HTTPException(404, "成片文件已不在本机")

    orig_file = root / Path(job.get("original_file") or "original.jpg").name
    technical = json.loads(job["technical_json"]) if job.get("technical_json") else {}
    fallback_exif = technical.get("exif") if isinstance(technical, dict) else None
    meta = watermark.extract_photo_metadata(
        orig_file if orig_file.is_file() else None,
        fallback_exif=fallback_exif,
        image_path_for_size=image_path,
    )
    return image_path, meta, label


def _resolve_batch_item_media(
    settings: Settings,
    conn: sqlite3.Connection,
    batch_id: str,
    item_id: str,
) -> tuple[Path, dict[str, Any], str]:
    item = get_style_batch_item(conn, item_id)
    if not item or item["batch_id"] != batch_id or item["status"] != "ready":
        raise HTTPException(404, "找不到已完成的批量成片")
    folder = style_sync.item_dir(settings, batch_id, item_id)
    img_path = folder / (item["output_filename"] or "result/result.jpg")
    if not img_path.is_file():
        raise HTTPException(404, "批量成片文件已不在本机")
    orig_file = folder / Path(item.get("original_file") or "original.jpg").name
    meta = watermark.extract_photo_metadata(
        orig_file if orig_file.is_file() else None,
        image_path_for_size=img_path,
    )
    name = item.get("original_name") or f"{item_id[:8]}.jpg"
    return img_path, meta, name


def _resolve_watermark_source(
    settings: Settings,
    conn: sqlite3.Connection,
    body: WatermarkRequestIn,
) -> tuple[Path, dict[str, Any], str]:
    if body.asset_id:
        rec = _load_asset_record(settings, body.asset_id)
        return Path(rec["image_path"]), rec.get("metadata") or {}, rec.get("name") or "photo.jpg"
    if body.batch_id and body.item_id:
        return _resolve_batch_item_media(settings, conn, body.batch_id, body.item_id)
    if body.job_id:
        return _resolve_job_media(settings, conn, body.job_id, body.slot or "A", body.version)
    raise HTTPException(400, "请先选择一张照片（asset_id / job_id / batch_id）")


def build_router(
    settings_fn: Callable[[], Settings],
    conn_fn: Callable[[], sqlite3.Connection],
) -> APIRouter:
    router = APIRouter(prefix="/api/studio", tags=["studio"])

    @router.get("/meta")
    def studio_meta() -> dict[str, Any]:
        return {
            "watermark_styles": watermark.WATERMARK_STYLES,
            "collage_layouts": collage.COLLAGE_LAYOUTS,
            "brands": watermark.BRAND_SPECS,
        }

    @router.get("/sources")
    def list_studio_sources(limit: int = 24) -> dict[str, Any]:
        """List recent completed jobs and style batches available for 1-click import."""
        settings = settings_fn()
        conn = conn_fn()
        lim = max(1, min(60, int(limit)))
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'ready' ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (lim,),
        ).fetchall()
        jobs_out: list[dict[str, Any]] = []
        for row in rows:
            job = dict(row)
            jid = job["id"]
            variants = get_variants(conn, jid)
            if not variants:
                continue
            # Prefer favorite -> keep -> slot A
            ratings = {
                r["slot"]: r["verdict"]
                for r in conn.execute("SELECT slot, verdict FROM ratings WHERE job_id = ?", (jid,))
            }
            best = max(variants, key=lambda v: (ratings.get(v["slot"], 0), 1 if v["slot"] == "A" else 0))
            root = job_dir(settings, jid)
            if not (root / best["filename"]).is_file():
                continue
            refs = [r for r in get_refinements(conn, jid) if r["slot"] == best["slot"] and r["status"] == "ready"]
            latest_ref = refs[-1] if refs else None
            technical = json.loads(job["technical_json"]) if job.get("technical_json") else {}
            meta = watermark.normalize_exif((technical or {}).get("exif"))
            jobs_out.append({
                "job_id": jid,
                "original_name": job.get("original_name") or "photo.jpg",
                "slot": best["slot"],
                "version": latest_ref["id"] if latest_ref else None,
                "title": best["title"],
                "preview_url": (
                    f"/api/jobs/{jid}/refinements/{latest_ref['id']}/image"
                    if latest_ref else f"/media/{jid}/{best['filename']}"
                ),
                "created_at": job["created_at"],
                "metadata": meta,
            })

        batches_out: list[dict[str, Any]] = []
        for b in list_style_batches(conn, limit=15):
            if b["done_count"] <= 0:
                continue
            batches_out.append({
                "batch_id": b["id"],
                "source_job_id": b["source_job_id"],
                "title": b["title"],
                "done_count": b["done_count"],
                "total_count": b["total_count"],
                "created_at": b["created_at"],
            })
        return {"jobs": jobs_out, "batches": batches_out}

    @router.get("/jobs/{job_id}/exif")
    def job_exif(job_id: str, slot: str = "A", version: str | None = None) -> dict[str, Any]:
        settings = settings_fn()
        conn = conn_fn()
        _, meta, name = _resolve_job_media(settings, conn, job_id, slot, version)
        return {"name": name, "metadata": meta}

    @router.post("/upload")
    async def upload_studio_photos(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        settings = settings_fn()
        if not files:
            raise HTTPException(400, "请至少选择一张照片")
        if len(files) > _MAX_UPLOAD_COUNT:
            raise HTTPException(400, f"单次最多上传 {_MAX_UPLOAD_COUNT} 张照片")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        tmp_root = settings.data_dir / "studio" / ".tmp" / uuid.uuid4().hex
        tmp_root.mkdir(parents=True, exist_ok=True)
        assets: list[dict[str, Any]] = []
        try:
            for idx, upload in enumerate(files):
                name = Path(upload.filename or f"photo_{idx + 1}.jpg").name
                ext = Path(name).suffix.lower()
                if ext not in ALLOWED_EXTS:
                    raise HTTPException(400, f"不支持的文件格式：{name}（{ext or '无后缀'}）")
                tmp_file = tmp_root / f"raw_{idx}{ext}"
                size = 0
                with tmp_file.open("wb") as out:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise HTTPException(400, f"{name} 超过单张 {settings.max_upload_mb} MB 上限")
                        out.write(chunk)
                    if not size:
                        raise HTTPException(400, f"{name} 是空文件")
                await upload.close()

                kind = source_kind(tmp_file)
                meta = watermark.extract_photo_metadata(tmp_file)
                if kind == "raw":
                    try:
                        linear, _ = load_linear(tmp_file)
                    except IngestError as exc:
                        raise HTTPException(400, str(exc)) from exc
                    srgb = np.clip(linear_to_srgb(linear) * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    im = Image.fromarray(srgb, mode="RGB")
                else:
                    try:
                        im = _open_image_rgb(tmp_file)
                    except Exception as exc:
                        raise HTTPException(400, f"无法读取图片 {name}：{exc}") from exc

                assets.append(_register_asset(
                    settings,
                    name=name,
                    source_type="upload",
                    image=im,
                    metadata=meta,
                ))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
            for upload in files:
                try:
                    await upload.close()
                except Exception:
                    pass
        return {"assets": assets}

    @router.post("/import-job")
    def import_job_asset(body: ImportJobIn) -> dict[str, Any]:
        settings = settings_fn()
        conn = conn_fn()
        img_path, meta, name = _resolve_job_media(settings, conn, body.job_id, body.slot, body.version)
        im = _open_image_rgb(img_path)
        asset = _register_asset(
            settings,
            name=name,
            source_type="job",
            image=im,
            metadata=meta,
            source_ref={"job_id": body.job_id, "slot": body.slot.upper(), "version": body.version},
            existing_image_path=img_path,
        )
        return {"asset": asset}

    @router.post("/import-batch")
    def import_batch_assets(body: ImportBatchIn) -> dict[str, Any]:
        settings = settings_fn()
        conn = conn_fn()
        batch = get_style_batch(conn, body.batch_id)
        if not batch:
            raise HTTPException(404, "找不到该批量同风格任务")
        assets: list[dict[str, Any]] = []
        if body.include_anchor:
            try:
                a_path, a_meta, a_name = _resolve_job_media(
                    settings, conn, batch["source_job_id"], batch["source_slot"], batch.get("source_version")
                )
                assets.append(_register_asset(
                    settings,
                    name=a_name,
                    source_type="job",
                    image=_open_image_rgb(a_path),
                    metadata=a_meta,
                    source_ref={
                        "job_id": batch["source_job_id"],
                        "slot": batch["source_slot"],
                        "version": batch.get("source_version"),
                    },
                    existing_image_path=a_path,
                ))
            except HTTPException:
                pass

        for item in get_style_batch_items(conn, body.batch_id):
            if len(assets) >= body.limit:
                break
            if item["status"] != "ready":
                continue
            try:
                p, m, n = _resolve_batch_item_media(settings, conn, body.batch_id, item["id"])
                assets.append(_register_asset(
                    settings,
                    name=n,
                    source_type="batch_item",
                    image=_open_image_rgb(p),
                    metadata=m,
                    source_ref={"batch_id": body.batch_id, "item_id": item["id"]},
                    existing_image_path=p,
                ))
            except HTTPException:
                continue

        if not assets:
            raise HTTPException(404, "这组批量任务还没有已完成的成片可导入")
        return {"assets": assets, "batch_title": batch["title"]}

    @router.get("/assets/{asset_id}/image")
    def studio_asset_image(asset_id: str) -> FileResponse:
        rec = _load_asset_record(settings_fn(), asset_id)
        return FileResponse(Path(rec["image_path"]), media_type="image/jpeg")

    @router.get("/assets/{asset_id}/thumb")
    def studio_asset_thumb(asset_id: str) -> FileResponse:
        rec = _load_asset_record(settings_fn(), asset_id)
        thumb = Path(rec.get("thumb_path", ""))
        if not thumb.is_file():
            thumb = Path(rec["image_path"])
        return FileResponse(thumb, media_type="image/jpeg")

    @router.post("/watermark/preview")
    def watermark_preview(body: WatermarkRequestIn) -> Response:
        settings = settings_fn()
        conn = conn_fn()
        img_path, meta, _ = _resolve_watermark_source(settings, conn, body)
        im = _open_image_rgb(img_path)
        rendered = watermark.render_watermark(im, meta, body.config, max_edge=body.max_edge or 1200)
        return Response(content=watermark.encode_jpeg_bytes(rendered, quality=90), media_type="image/jpeg")

    @router.post("/watermark/export")
    def watermark_export(body: WatermarkRequestIn) -> Response:
        settings = settings_fn()
        conn = conn_fn()
        img_path, meta, name = _resolve_watermark_source(settings, conn, body)
        im = _open_image_rgb(img_path)
        rendered = watermark.render_watermark(im, meta, body.config, max_edge=body.max_edge)
        stem = Path(name).stem
        out_name = f"{stem}-水印-{body.config.style}.jpg"
        return Response(
            content=watermark.encode_jpeg_bytes(rendered, quality=95),
            media_type="image/jpeg",
            headers=_content_disposition(out_name),
        )

    @router.post("/watermark/batch-zip")
    def watermark_batch_zip(body: WatermarkBatchZipIn) -> Response:
        settings = settings_fn()
        buf = BytesIO()
        used: dict[str, int] = {}
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for idx, aid in enumerate(body.asset_ids, start=1):
                rec = _load_asset_record(settings, aid)
                im = _open_image_rgb(Path(rec["image_path"]))
                rendered = watermark.render_watermark(im, rec.get("metadata") or {}, body.config, max_edge=body.max_edge)
                stem = Path(rec.get("name") or f"photo_{idx}").stem
                n = used.get(stem, 0)
                used[stem] = n + 1
                suffix = f"-{n + 1}" if n else ""
                arc = f"{idx:02d}-{stem}{suffix}-水印.jpg"
                zf.writestr(arc, watermark.encode_jpeg_bytes(rendered, quality=94))
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers=_content_disposition(f"SuperPhoto-批量水印-{len(body.asset_ids)}张.zip"),
        )

    @router.post("/collage/auto-arrange")
    def collage_auto_arrange(body: AutoArrangeIn) -> dict[str, Any]:
        settings = settings_fn()
        items = [_load_asset_record(settings, aid) for aid in body.asset_ids]
        ordered, reason = collage.auto_arrange_items(items, mode=body.mode)
        return {"ordered_ids": ordered, "mode": body.mode, "reason": reason}

    @router.post("/collage/preview")
    def collage_preview(body: CollageRequestIn) -> Response:
        settings = settings_fn()
        records = [_load_asset_record(settings, aid) for aid in body.asset_ids]
        images = [_open_image_rgb(Path(r["image_path"])) for r in records]
        metas = [r.get("metadata") or {} for r in records]
        try:
            rendered = collage.render_collage(images, metas, body.config, target_width=min(body.target_width, 1600))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(content=watermark.encode_jpeg_bytes(rendered, quality=88), media_type="image/jpeg")

    @router.post("/collage/moments-preview")
    def collage_moments_preview(body: CollageRequestIn) -> dict[str, Any]:
        """Return exact WeChat Moments preview tiles (small JPEG data URLs) + layout metrics + caption suggestion."""
        settings = settings_fn()
        records = [_load_asset_record(settings, aid) for aid in body.asset_ids]
        images = [_open_image_rgb(Path(r["image_path"])) for r in records]
        metas = [r.get("metadata") or {} for r in records]
        tiles = collage.build_moments_tiles(images, metas, body.config, tile_edge=420)
        tile_urls = [
            "data:image/jpeg;base64," + base64.b64encode(watermark.encode_jpeg_bytes(t, quality=84)).decode("ascii")
            for t in tiles
        ]
        n = len(tiles)
        if body.config.layout == "slice_4" or (body.config.layout not in ("slice_9", "slice_6") and n == 4):
            cols = 2
        elif n == 1:
            cols = 1
        else:
            cols = 3
        caption = watermark.suggest_moments_caption(metas)
        first_loc = next((m.get("gps") for m in metas if m.get("gps")), "")
        return {
            "count": n,
            "cols": cols,
            "is_slice": body.config.layout in ("slice_9", "slice_4", "slice_6"),
            "tiles": tile_urls,
            "suggested_caption": caption,
            "suggested_location": first_loc,
        }

    @router.post("/collage/export")
    def collage_export(body: CollageRequestIn) -> Response:
        settings = settings_fn()
        records = [_load_asset_record(settings, aid) for aid in body.asset_ids]
        images = [_open_image_rgb(Path(r["image_path"])) for r in records]
        metas = [r.get("metadata") or {} for r in records]
        try:
            rendered = collage.render_collage(images, metas, body.config, target_width=max(1800, body.target_width))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        out_name = f"SuperPhoto-拼图-{body.config.layout}-{len(images)}张.jpg"
        return Response(
            content=watermark.encode_jpeg_bytes(rendered, quality=95),
            media_type="image/jpeg",
            headers=_content_disposition(out_name),
        )

    @router.post("/collage/export-moments-zip")
    def collage_export_moments_zip(body: CollageRequestIn) -> Response:
        settings = settings_fn()
        records = [_load_asset_record(settings, aid) for aid in body.asset_ids]
        images = [_open_image_rgb(Path(r["image_path"])) for r in records]
        metas = [r.get("metadata") or {} for r in records]
        try:
            data = collage.export_moments_zip_bytes(
                images,
                metas,
                body.config,
                tile_edge=body.tile_edge,
                quality=95,
                prefix="SuperPhoto-朋友圈",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        out_name = f"SuperPhoto-朋友圈九宫格-{body.config.layout}.zip"
        return Response(
            content=data,
            media_type="application/zip",
            headers=_content_disposition(out_name),
        )

    return router
