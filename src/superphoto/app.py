from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import shutil
import time
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .tasks import TaskWorker, enqueue_source
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .db import (
    connect,
    get_job,
    get_variants,
    init_db,
    insert_feedback,
    insert_job,
    look_stats,
    SLOT_LETTERS,
    VERDICTS,
    get_ratings,
    set_rating,
    get_refinement,
    get_refinements,
    insert_refinement,
    get_style_batch,
    get_style_batch_item,
    get_style_batch_items,
    insert_style_batch,
    insert_style_batch_item,
    list_style_batches,
)
from .darktable_export import darktable_cli_path
from .ingest import ALLOWED_EXTS, source_kind
from .lightroom import LightroomError, connect_lightroom, install_lightroom_plugin, lightroom_status
from . import full_export, style_sync
from .models import BatchFullExportIn, FeedbackIn, FullExportIn, RatingIn, RefinementIn
from .vision import get_vision_status
from .vault_routes import build_router as build_vault_router, publish_bridge, retract_bridge
from .pipeline import run_job
from .preference import describe_taste
from .recipes import LOOKS
from .semantic import semantic_status
from .store import job_dir
from .experience import variant_profile
from .scene_preferences import job_profile, preference_samples, variant_method
from .refinement import refinement_base, run_refinement

def _find_web_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (
        here / "web",
        here.parents[1] / "web",
        here.parents[2] / "web" if len(here.parents) > 2 else here / "web",
        Path.cwd() / "web",
    ):
        if (cand / "index.html").is_file():
            return cand
    return Path.cwd() / "web"


WEB_DIR = _find_web_dir()

_lock = threading.Lock()
_state: dict[str, Any] = {}


def reset_runtime() -> None:
    worker = _state.get("task_worker")
    if worker is not None:
        worker.stop()
    with _lock:
        for conn in _state.get("connections", {}).values():
            conn.close()
        _state.clear()


def _boot() -> tuple[Settings, sqlite3.Connection]:
    with _lock:
        if "settings" not in _state:
            settings = get_settings()
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            settings.jobs_dir.mkdir(parents=True, exist_ok=True)
            conn = connect(settings.db_path)
            init_db(conn)
            _state["settings"] = settings
            _state["connections"] = {threading.get_ident(): conn}
        # FastAPI status, feedback and processing workers run concurrently.
        # check_same_thread=False permits access, but sharing one connection
        # also shares transaction state and cached statements. Keep those
        # private to each worker so a poll cannot corrupt another result row.
        connections = _state["connections"]
        worker = threading.get_ident()
        if worker not in connections:
            connections[worker] = connect(_state["settings"].db_path)
        return _state["settings"], connections[worker]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings, _ = _boot()
    worker = TaskWorker(settings.db_path, settings.jobs_dir,
                        lambda job_id, original: _safe_run(job_id, original),
                        lambda refinement_id: _safe_refine(refinement_id),
                        run_style_item=lambda item_id: _safe_style_item(item_id))
    worker.start()
    _state["task_worker"] = worker
    publish_bridge(settings)
    try:
        yield
    finally:
        retract_bridge()
        await asyncio.to_thread(worker.stop)
        _state.pop("task_worker", None)


def _wake_worker() -> None:
    worker = _state.get("task_worker")
    if worker is not None:
        worker.wake()


app = FastAPI(title="SuperPhoto", lifespan=lifespan)

_SESSION_COOKIE = "superphoto_session"
_SESSION_SECONDS = 12 * 60 * 60
_SESSION_SECRET = secrets.token_bytes(32)


def _access_token() -> str:
    return os.environ.get("SUPERPHOTO_ACCESS_TOKEN", getattr(_s(), "access_token", "")).strip()


def _session_signature(stamp: str, token: str) -> str:
    return hmac.new(_SESSION_SECRET, f"{stamp}:{token}".encode(), hashlib.sha256).hexdigest()


def _authenticated(request: Request, token: str) -> bool:
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, supplied = authorization.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(supplied.encode(), token.encode())
    cookie = request.cookies.get(_SESSION_COOKIE, "")
    stamp, _, signature = cookie.partition(".")
    try:
        age = time.time() - int(stamp)
    except ValueError:
        return False
    return (0 <= age <= _SESSION_SECONDS and
            hmac.compare_digest(signature.encode(), _session_signature(stamp, token).encode()))


@app.middleware("http")
async def access_control(request: Request, call_next):
    token = _access_token()
    path = request.url.path
    protected = path == "/api" or path.startswith("/api/") or path == "/media" or path.startswith("/media/")
    public = path in {"/api/auth/status", "/api/auth/login"}
    if token and protected and not public:
        if not _authenticated(request, token):
            return JSONResponse({"detail": "需要登录或提供 Bearer token"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
        # Cookies must not authorize cross-origin writes. Bearer requests do
        # not rely on ambient browser credentials and do not need this check.
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not request.headers.get("authorization"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "拒绝跨站请求"}, status_code=403)
    response = await call_next(request)
    if protected:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/auth/status")
def auth_status(request: Request) -> dict[str, bool]:
    token = _access_token()
    return {"required": bool(token), "authenticated": not token or _authenticated(request, token)}


@app.post("/api/auth/login")
def auth_login(request: Request, body: dict = Body(...)) -> JSONResponse:
    token = _access_token()
    supplied = body.get("token")
    if token and (not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(), token.encode())):
        raise HTTPException(401, "访问口令不正确", headers={"WWW-Authenticate": "Bearer"})
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "拒绝跨站请求")
    response = JSONResponse({"ok": True})
    if token:
        stamp = str(int(time.time()))
        response.set_cookie(_SESSION_COOKIE, f"{stamp}.{_session_signature(stamp, token)}",
                            max_age=_SESSION_SECONDS, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", path="/")
    return response


@app.post("/api/auth/logout")
def auth_logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return response


def _s() -> Settings:
    return _boot()[0]


def _c() -> sqlite3.Connection:
    return _boot()[1]


app.include_router(build_vault_router(_s))


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = _s()
    rawpy_ok = True
    try:
        import rawpy  # noqa: F401
    except ImportError:
        rawpy_ok = False
    vision = get_vision_status(settings, check_ollama=True)
    provider = vision["provider"]
    critic = "grok" if provider == "xai" else provider
    critic_model = vision["model"]
    local_candidate = next((candidate for candidate in vision["candidates"]
                            if candidate["provider"] == "ollama"), None)
    ollama = {"ok": local_candidate is not None, "ready": local_candidate is not None,
              "model": settings.ollama_model,
              "resolved_model": local_candidate["model"] if local_candidate else None}
    semantic = semantic_status(settings)
    lr_status = lightroom_status(settings.lightroom_enabled, settings.lightcraft_bin)
    return {
        "ok": True,
        "critic": critic,
        "critic_model": critic_model,
        "vision": vision,
        "ollama": ollama,
        "grok": bool(settings.xai_api_key),
        "model": settings.xai_model if settings.xai_api_key else None,
        "rawpy": rawpy_ok,
        "darktable_cli": darktable_cli_path(),
        "semantic": semantic,
        "lightroom": lr_status,
        "taste": describe_taste(_c()),
    }


@app.post("/api/lightroom/install-plugin")
def install_lr_plugin() -> dict[str, Any]:
    settings = _s()
    try:
        return install_lightroom_plugin(settings.lightcraft_bin)
    except LightroomError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/lightroom/connect")
def connect_lr() -> dict[str, Any]:
    settings = _s()
    if not settings.lightroom_enabled:
        raise HTTPException(409, "当前未启用 LightCraft 后期")
    try:
        _client, status = connect_lightroom(settings.lightcraft_bin)
        return status
    except LightroomError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/directed-looks")
def list_directed_looks() -> dict[str, Any]:
    from .looks import catalog

    items = catalog()
    return {"count": len(items), "looks": items}


@app.get("/api/looks")
def list_looks() -> dict[str, Any]:
    return {
        "looks": [
            {
                "id": look.id,
                "title": look.title,
                "title_en": look.title_en,
                "description": look.description,
                "moves": look.moves,
            }
            for look in LOOKS.values()
        ]
    }


@app.get("/api/preferences")
def preferences() -> dict[str, Any]:
    stats = look_stats(_c())
    titled = []
    for row in stats["looks"]:
        look = LOOKS.get(row["look_id"])
        titled.append({**row, "title": look.title if look else row["look_id"]})
    samples = preference_samples(_c())
    return {"total_picks": stats["total_picks"], "looks": titled, "summary": describe_taste(_c()),
            "ratings": {"keep": sum(s["verdict"] == 1 for s in samples),
                        "favorite": sum(s["verdict"] == 2 for s in samples),
                        "pass": sum(s["verdict"] < 0 for s in samples)}}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
) -> dict[str, str]:
    settings = _s()
    conn = _c()
    name = file.filename or "upload.bin"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported file type: {ext or 'none'}")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    job_id = str(uuid.uuid4())
    folder = job_dir(settings, job_id)
    original = folder / f"original{ext}"
    try:
        size = 0
        with original.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(400, f"file larger than {settings.max_upload_mb} MB")
                output.write(chunk)
            if not size:
                raise HTTPException(400, "empty file")
            output.flush()
            os.fsync(output.fileno())
        # Publish queued only after the complete source has reached disk.
        insert_job(conn, job_id, name, original.name)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        await file.close()
    _wake_worker()
    return {"id": job_id, "kind": source_kind(original)}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict[str, str]:
    settings, conn = _boot()
    saved = get_job(conn, job_id)
    if not saved:
        raise HTTPException(404, "找不到这次处理记录")
    if saved["status"] != "error":
        raise HTTPException(409, "只能重试处理失败的照片")
    name = saved["original_name"] or "photo.jpg"
    filename = Path(saved["original_file"] or f"original{Path(name).suffix.lower()}").name
    source = settings.jobs_dir / job_id / filename
    if not source.is_file():
        raise HTTPException(410, "原片已不在本机，请重新选择照片")
    try:
        result = enqueue_source(settings, conn, source, original_name=name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _wake_worker()
    return result


def _safe_run(job_id: str, original: Path) -> None:
    settings, conn = _boot()
    run_job(settings, conn, job_id, original)


@app.get("/api/jobs/latest")
def latest_job() -> dict[str, Any]:
    row = _c().execute("SELECT id FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    return {"id": row["id"] if row else None}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    conn = _c()
    job = get_job(conn, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    variants = []
    for row in get_variants(conn, job_id):
        recipe = json.loads(row["recipe_json"])
        variants.append(
            {
                "slot": row["slot"],
                "look_id": row["look_id"],
                "title": row["title"],
                "why": row["why"],
                "moves": (recipe.get("style_workflow", {}).get("adaptation_notes", [])
                          if recipe.get("style_workflow", {}).get("mode") == "art-directed"
                          else LOOKS[row["look_id"]].moves if row["look_id"] in LOOKS else []),
                "recipe": recipe,
                "url": f"/media/{job_id}/{row['filename']}",
            }
        )
    technical = json.loads(job["technical_json"]) if job.get("technical_json") else None
    critique = json.loads(job["critique_json"]) if job.get("critique_json") else None
    decision_path = job_dir(_s(), job_id) / "style-decision.json"
    style_decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.is_file() else None
    delivery_path = job_dir(_s(), job_id) / "lightroom-qa.json"
    delivery = json.loads(delivery_path.read_text(encoding="utf-8")).get("delivery") if delivery_path.is_file() else None
    return {
        "id": job["id"],
        "status": job["status"],
        "step": job["step"],
        "error": _public_error(job.get("error")),
        "original_name": job.get("original_name"),
        "source_preview": f"/media/{job_id}/source_preview.jpg" if job["status"] in {"ready", "running"} else None,
        "technical": technical,
        "critique": critique,
        "style_decision": style_decision,
        "variants": variants,
        "delivery": delivery,
        "ratings": get_ratings(conn, job_id),
        "refinements": [_public_refinement(r) for r in get_refinements(conn, job_id)],
    }


def _ready_variant(conn, job_id: str, slot: str) -> tuple[dict, dict]:
    job = get_job(conn, job_id)
    if not job:
        raise HTTPException(404, "找不到这次处理记录")
    if job["status"] != "ready":
        raise HTTPException(409, "请等成片处理完成后再操作")
    variant = next((v for v in get_variants(conn, job_id) if v["slot"] == slot), None)
    if not variant:
        raise HTTPException(404, "找不到这张成片")
    return job, variant


@app.put("/api/jobs/{job_id}/ratings/{slot}")
def rate_variant(job_id: str, slot: str, body: RatingIn) -> dict:
    conn = _c()
    job, variant = _ready_variant(conn, job_id, slot)
    method, scene = variant_method(variant), variant_profile(variant) or job_profile(job)
    with _lock:
        verdict = {**VERDICTS, "like": VERDICTS["keep"], "dislike": VERDICTS["pass"], None: 0}[body.verdict]
        set_rating(conn, job_id, slot, verdict, method, scene)
    return {"ok": True, "ratings": get_ratings(conn, job_id), "summary": describe_taste(conn),
            "learnable": bool(method and scene)}


def _public_refinement(record: dict) -> dict:
    result = {key: record[key] for key in ("id", "job_id", "slot", "parent_id", "prompt", "status", "step",
                                         "summary", "created_at", "updated_at")}
    result["error"] = _public_error(record.get("error"))
    result["url"] = (f"/api/jobs/{record['job_id']}/refinements/{record['id']}/image"
                     if record["status"] == "ready" else None)
    return result


@app.post("/api/jobs/{job_id}/refinements", status_code=202)
def create_refinement(job_id: str, body: RefinementIn) -> dict:
    settings, conn = _boot()
    job, _ = _ready_variant(conn, job_id, body.slot)
    refinement_id = str(body.request_id)
    parent_id = str(body.parent_id) if body.parent_id else None
    with _lock:
        existing = get_refinement(conn, refinement_id)
        if existing:
            if (existing["job_id"], existing["slot"], existing["parent_id"], existing["prompt"]) != (
                    job_id, body.slot, parent_id, body.prompt):
                raise HTTPException(409, "这条微调请求已用于其他内容，请重新提交")
            return _public_refinement(existing)
        if not settings.lightroom_enabled:
            raise HTTPException(409, "需要启用 LightCraft 才能进行二次调整")
        try:
            refinement_base(settings, conn, job_id, body.slot, parent_id)
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc
        original = settings.jobs_dir / job_id / Path(job["original_file"] or "original.jpg").name
        if not original.is_file():
            raise HTTPException(410, "原片已不在本机，无法继续微调")
        try:
            insert_refinement(conn, refinement_id, job_id, body.slot, parent_id, body.prompt)
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise HTTPException(409, "这张图已有微调正在处理，请等它完成后再提交") from exc
    record = get_refinement(conn, refinement_id)
    _wake_worker()
    return _public_refinement(record)


def _safe_refine(refinement_id: str) -> None:
    settings = _s()
    conn = connect(settings.db_path)
    try:
        run_refinement(settings, conn, refinement_id)
    finally:
        conn.close()


def _safe_style_item(item_id: str) -> None:
    settings = _s()
    conn = connect(settings.db_path)
    try:
        style_sync.run_batch_item(settings, conn, item_id)
    finally:
        conn.close()


@app.get("/api/jobs/{job_id}/refinements/{refinement_id}")
def refinement_status(job_id: str, refinement_id: str) -> dict:
    record = get_refinement(_c(), refinement_id)
    if not record or record["job_id"] != job_id:
        raise HTTPException(404, "找不到这次微调")
    return _public_refinement(record)


@app.get("/api/jobs/{job_id}/refinements/{refinement_id}/image")
def refinement_image(job_id: str, refinement_id: str) -> FileResponse:
    record = get_refinement(_c(), refinement_id)
    if not record or record["job_id"] != job_id or record["status"] != "ready":
        raise HTTPException(404, "找不到微调成片")
    image = _s().jobs_dir / job_id / "refinements" / refinement_id / "result.jpg"
    if not image.is_file():
        raise HTTPException(404, "微调成片已不在本机")
    return FileResponse(image, media_type="image/jpeg", filename=f"SuperPhoto-{record['slot']}-微调-{refinement_id[:8]}.jpg")


@app.post("/api/jobs/{job_id}/full")
def prepare_full_size(job_id: str, body: FullExportIn) -> dict[str, Any]:
    """Render the version shown at full size (JPEG or 16-bit TIFF); the file is then fetched from ``url``."""
    settings, conn = _boot()
    _ready_variant(conn, job_id, body.slot)
    version = str(body.version) if body.version else None
    try:
        client, _status = connect_lightroom(settings.lightcraft_bin)
        result = full_export.export_full(settings, conn, job_id, body.slot, version, body.format,
                                         binary=Path(client.binary))
    except full_export.FullExportUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except LightroomError as exc:
        raise HTTPException(503, f"原尺寸导出失败：{exc}") from exc
    return {"url": f"/api/jobs/{job_id}/full/{result['name']}",
            "filename": full_export.download_name(body.slot, version, body.format),
            **{key: result.get(key) for key in ("format", "width", "height", "bytes", "seconds", "cached")}}


@app.get("/api/jobs/{job_id}/full/{name}")
def full_size_file(job_id: str, name: str) -> FileResponse:
    parsed = full_export.parse_name(name)
    path = full_export.file_path(_s().jobs_dir / job_id, name) if get_job(_c(), job_id) else None
    if parsed is None or path is None or not path.is_file():
        raise HTTPException(404, "原尺寸文件还没有导出")
    target, fmt = parsed
    slot, _, version = target.partition("-")
    return FileResponse(path, media_type=full_export.FORMATS[fmt]["media_type"],
                        filename=full_export.download_name(slot, version or None, fmt))


def _public_style_batch(settings: Settings, conn: sqlite3.Connection, batch: dict[str, Any],
                        *, include_items: bool = True) -> dict[str, Any]:
    profile = json.loads(batch["profile_json"])
    payload: dict[str, Any] = {
        "id": batch["id"],
        "source_job_id": batch["source_job_id"],
        "source_slot": batch["source_slot"],
        "source_version": batch["source_version"],
        "title": batch["title"],
        "direction_title": profile.get("direction_title") or batch["title"],
        "anchor_render_url": profile.get("anchor_render_url"),
        "anchor_name": profile.get("source_name") or "",
        "sync_options": profile.get("sync_options") or {},
        "wb_offset": profile.get("wb_offset") or {},
        "look": profile.get("look"),
        "status": batch["status"],
        "total_count": batch["total_count"],
        "done_count": batch["done_count"],
        "error_count": batch["error_count"],
        "created_at": batch["created_at"],
        "updated_at": batch["updated_at"],
        "zip_url": f"/api/style-batches/{batch['id']}/download.zip" if batch["done_count"] > 0 else None,
    }
    if include_items:
        items_out = []
        for item in get_style_batch_items(conn, batch["id"]):
            folder = style_sync.item_dir(settings, batch["id"], item["id"])
            has_src = (folder / "baseline" / "source_preview.jpg").is_file()
            metrics = json.loads(item["metrics_json"]) if item.get("metrics_json") else None
            items_out.append({
                "id": item["id"],
                "original_name": item["original_name"],
                "status": item["status"],
                "step": item["step"],
                "error": _public_error(item.get("error")),
                "source_url": f"/api/style-batches/{batch['id']}/items/{item['id']}/source" if has_src else None,
                "result_url": (f"/api/style-batches/{batch['id']}/items/{item['id']}/image"
                               if item["status"] == "ready" else None),
                "metrics": metrics,
            })
        payload["items"] = items_out
    return payload


@app.post("/api/jobs/{job_id}/sync-style", status_code=202)
async def create_style_batch(
    job_id: str,
    files: list[UploadFile] = File(...),
    slot: str = Form("A"),
    version: str | None = Form(None),
    match_exposure: bool = Form(True),
    match_wb: bool = Form(True),
    style_strength: float = Form(1.0),
) -> dict[str, Any]:
    """Transfer one perfected edit's style onto a batch of photos from the same shoot."""
    settings, conn = _boot()
    slot = str(slot or "").strip().upper()
    if slot not in SLOT_LETTERS:
        raise HTTPException(400, f"无效的方案槽位：{slot}")
    if not settings.lightroom_enabled:
        raise HTTPException(409, "批量同风格套用需要启用 LightCraft 渲染引擎")
    if not files:
        raise HTTPException(400, "请至少选择一张照片")
    if len(files) > 200:
        raise HTTPException(400, "单次最多批量套用 200 张照片")
    version_clean = (version or "").strip() or None
    strength = max(0.3, min(1.3, float(style_strength)))

    binary: Path | None = None
    try:
        client, _ = connect_lightroom(settings.lightcraft_bin)
        binary = Path(client.binary)
    except LightroomError:
        binary = None

    try:
        profile = style_sync.extract_style_profile(settings, conn, job_id, slot, version_clean, binary=binary)
    except (ValueError, full_export.FullExportUnavailable) as exc:
        raise HTTPException(409, str(exc)) from exc

    profile["sync_options"] = {
        "match_exposure": bool(match_exposure),
        "match_wb": bool(match_wb),
        "style_strength": round(strength, 2),
    }

    batch_id = str(uuid.uuid4())
    root = style_sync.batch_dir(settings, batch_id)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    staged: list[tuple[str, str, str]] = []
    try:
        for upload in files:
            name = Path(upload.filename or "photo.bin").name
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXTS:
                raise HTTPException(400, f"不支持的文件格式：{name}（{ext or '无后缀'}）")
            item_id = str(uuid.uuid4())
            folder = style_sync.item_dir(settings, batch_id, item_id)
            folder.mkdir(parents=True, exist_ok=True)
            original = folder / f"original{ext}"
            size = 0
            with original.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(400, f"{name} 超过单张 {settings.max_upload_mb} MB 上限")
                    output.write(chunk)
                if not size:
                    raise HTTPException(400, f"{name} 是空文件")
                output.flush()
                os.fsync(output.fileno())
            staged.append((item_id, name, original.name))
            await upload.close()
        with _lock:
            insert_style_batch(conn, batch_id, job_id, slot, version_clean, profile["title"], profile, len(staged))
            for item_id, name, orig_file in staged:
                insert_style_batch_item(conn, item_id, batch_id, name, orig_file)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        for upload in files:
            try:
                await upload.close()
            except Exception:
                pass
        raise

    _wake_worker()
    batch = get_style_batch(conn, batch_id)
    return _public_style_batch(settings, conn, batch)


@app.get("/api/style-batches")
def get_style_batches(job_id: str | None = None) -> dict[str, Any]:
    settings, conn = _boot()
    rows = list_style_batches(conn, source_job_id=job_id)
    return {"batches": [_public_style_batch(settings, conn, row, include_items=False) for row in rows]}


@app.get("/api/style-batches/{batch_id}")
def style_batch_status(batch_id: str) -> dict[str, Any]:
    settings, conn = _boot()
    batch = get_style_batch(conn, batch_id)
    if not batch:
        raise HTTPException(404, "找不到这次批量同风格任务")
    return _public_style_batch(settings, conn, batch, include_items=True)


@app.get("/api/style-batches/{batch_id}/items/{item_id}/image")
def style_batch_item_image(batch_id: str, item_id: str) -> FileResponse:
    settings, conn = _boot()
    item = get_style_batch_item(conn, item_id)
    if not item or item["batch_id"] != batch_id or item["status"] != "ready":
        raise HTTPException(404, "找不到已完成的批量成片")
    path = style_sync.item_dir(settings, batch_id, item_id) / (item["output_filename"] or "result/result.jpg")
    if not path.is_file():
        raise HTTPException(404, "批量成片已不在本机")
    stem = Path(item["original_name"] or "photo").stem
    return FileResponse(path, media_type="image/jpeg", filename=f"{stem}-同风格.jpg")


@app.get("/api/style-batches/{batch_id}/items/{item_id}/source")
def style_batch_item_source(batch_id: str, item_id: str) -> FileResponse:
    settings, conn = _boot()
    item = get_style_batch_item(conn, item_id)
    if not item or item["batch_id"] != batch_id:
        raise HTTPException(404, "找不到原片预览")
    path = style_sync.item_dir(settings, batch_id, item_id) / "baseline" / "source_preview.jpg"
    if not path.is_file():
        raise HTTPException(404, "原片预览尚未生成")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/style-batches/{batch_id}/items/{item_id}/full")
def style_batch_item_full(batch_id: str, item_id: str, body: BatchFullExportIn) -> dict[str, Any]:
    settings, conn = _boot()
    try:
        client, _ = connect_lightroom(settings.lightcraft_bin)
        result = style_sync.export_batch_item_full(settings, conn, batch_id, item_id, body.format,
                                                   binary=Path(client.binary))
    except full_export.FullExportUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except LightroomError as exc:
        raise HTTPException(503, f"原尺寸导出失败：{exc}") from exc
    return {
        "url": f"/api/style-batches/{batch_id}/items/{item_id}/full/{body.format}",
        "filename": result["filename"],
        **{k: result.get(k) for k in ("format", "width", "height", "bytes", "seconds", "cached")},
    }


@app.get("/api/style-batches/{batch_id}/items/{item_id}/full/{fmt}")
def style_batch_item_full_file(batch_id: str, item_id: str, fmt: str) -> FileResponse:
    if fmt not in full_export.FORMATS:
        raise HTTPException(404, "不支持的格式")
    settings, conn = _boot()
    item = get_style_batch_item(conn, item_id)
    if not item or item["batch_id"] != batch_id:
        raise HTTPException(404, "找不到记录")
    spec = full_export.FORMATS[fmt]
    path = style_sync.item_dir(settings, batch_id, item_id) / "full" / f"render{spec['suffix']}"
    if not path.is_file():
        raise HTTPException(404, "原尺寸文件尚未导出")
    stem = Path(item["original_name"] or "photo").stem
    return FileResponse(path, media_type=spec["media_type"], filename=f"{stem}-同风格-原尺寸{spec['suffix']}")


@app.get("/api/style-batches/{batch_id}/download.zip")
def style_batch_download_zip(batch_id: str) -> FileResponse:
    settings, conn = _boot()
    try:
        archive = style_sync.build_batch_zip(settings, conn, batch_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(archive, media_type="application/zip", filename=f"SuperPhoto-同风格批量-{batch_id[:8]}.zip")


@app.post("/api/jobs/{job_id}/feedback")
def submit_feedback(job_id: str, body: FeedbackIn) -> dict[str, Any]:
    conn = _c()
    job = get_job(conn, job_id)
    if not job or job["status"] != "ready":
        raise HTTPException(400, "job is not ready")
    variants = {v["slot"]: v for v in get_variants(conn, job_id)}
    chosen = variants.get(body.chosen_slot)
    if not chosen:
        raise HTTPException(400, "unknown slot")
    reaction_scores = {
        "keep": (5, 5, 5),
        "too_dark": (2, 3, 2),
        "too_bright": (2, 3, 2),
        "too_strong": (3, 2, 1),
        "too_flat": (3, 2, 5),
        "color_wrong": (2, 2, 3),
        "subject_weak": (3, 2, 4),
    }
    inferred = reaction_scores[body.reaction]
    labels = {
        "keep": "满意",
        "too_dark": "太暗",
        "too_bright": "太亮",
        "too_strong": "处理太重",
        "too_flat": "效果太淡",
        "color_wrong": "颜色不自然",
        "subject_weak": "主体不突出",
    }
    note = body.notes.strip()
    reaction_note = f"简单反馈：{labels[body.reaction]}"
    note = f"{reaction_note}；{note}" if note else reaction_note
    insert_feedback(
        conn,
        {
            "id": str(uuid.uuid4()),
            "job_id": job_id,
            "chosen_slot": body.chosen_slot,
            "look_id": chosen["look_id"],
            "technical_score": body.technical_score or inferred[0],
            "style_score": body.style_score or inferred[1],
            "restraint_score": body.restraint_score or inferred[2],
            "notes": note,
        },
    )
    method, scene = variant_method(chosen), variant_profile(chosen) or job_profile(job)
    if method and scene:
        verdict = 1 if (body.style_score or inferred[1]) >= 4 and (body.technical_score or inferred[0]) >= 3 else -1
        with _lock:
            set_rating(conn, job_id, body.chosen_slot, verdict, method, scene)
    return {"ok": True, "preferences": look_stats(conn), "summary": describe_taste(conn)}


@app.get("/media/{job_id}/{filename}")
def media(job_id: str, filename: str) -> FileResponse:
    settings = _s()
    folder = settings.jobs_dir / job_id
    if not folder.exists():
        raise HTTPException(404, "not found")
    allowed = {"source_preview.jpg", "critique_preview.jpg", *(f"variant_{slot}.jpg" for slot in SLOT_LETTERS)}
    if filename not in allowed:
        raise HTTPException(404, "not found")
    path = folder / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/jpeg")


def _public_error(error: str | None) -> str | None:
    if not error:
        return None
    first = error.splitlines()[0]
    if "RAW support requires rawpy" in first:
        return first
    if isinstance(error, str) and error.startswith("unsupported"):
        return first
    # Keep the first line for the UI; full traceback stays in the db.
    return first


class UIStaticFiles(StaticFiles):
    """The installed UI is replaced in place; never reuse a previous shell."""

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


if WEB_DIR.is_dir():
    app.mount("/", UIStaticFiles(directory=str(WEB_DIR), html=True), name="web")


def create_app() -> FastAPI:
    return app
