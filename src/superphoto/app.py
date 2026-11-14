from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
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
    get_ratings,
    set_rating,
    get_refinement,
    get_refinements,
    insert_refinement,
)
from .darktable_export import darktable_cli_path
from .ingest import ALLOWED_EXTS, source_kind
from .lightroom import LightroomError, connect_lightroom, install_lightroom_plugin, lightroom_status
from .models import FeedbackIn, RatingIn, RefinementIn
from .ollama import probe
from .pipeline import run_job
from .preference import describe_taste
from .recipes import LOOKS
from .semantic import semantic_status
from .store import job_dir
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
            conn.execute("""UPDATE refinements SET status='error', step='error',
                error='应用在微调完成前关闭了。调整前版本已保留，可以重新提交意见。'
                WHERE status IN ('queued', 'running')""")
            conn.commit()
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
    _boot()
    yield


app = FastAPI(title="SuperPhoto", lifespan=lifespan)


def _s() -> Settings:
    return _boot()[0]


def _c() -> sqlite3.Connection:
    return _boot()[1]


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = _s()
    rawpy_ok = True
    try:
        import rawpy  # noqa: F401
    except ImportError:
        rawpy_ok = False
    ollama = probe(settings)
    if ollama.get("ready"):
        critic = "ollama"
        critic_model = ollama.get("resolved_model")
    elif settings.xai_api_key:
        critic = "grok"
        critic_model = settings.xai_model
    else:
        critic = "local"
        critic_model = None
    semantic = semantic_status(settings)
    lr_status = lightroom_status(settings.lightroom_enabled, settings.lightcraft_bin)
    return {
        "ok": True,
        "critic": critic,
        "critic_model": critic_model,
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
            "ratings": {"like": sum(s["verdict"] > 0 for s in samples),
                        "dislike": sum(s["verdict"] < 0 for s in samples)}}


@app.post("/api/jobs")
async def create_job(
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    settings = _s()
    conn = _c()
    name = file.filename or "upload.bin"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported file type: {ext or 'none'}")
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(400, f"file larger than {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "empty file")

    job_id = str(uuid.uuid4())
    folder = job_dir(settings, job_id)
    original = folder / f"original{ext}"
    original.write_bytes(data)
    insert_job(conn, job_id, name)
    background.add_task(_safe_run, job_id, original)
    return {"id": job_id, "kind": source_kind(original)}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str, background: BackgroundTasks) -> dict[str, str]:
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
    new_id = str(uuid.uuid4())
    original = job_dir(settings, new_id) / filename
    shutil.copy2(source, original)
    insert_job(conn, new_id, name)
    background.add_task(_safe_run, new_id, original)
    return {"id": new_id, "kind": source_kind(original)}


def _safe_run(job_id: str, original: Path) -> None:
    settings, conn = _boot()
    try:
        run_job(settings, conn, job_id, original)
    except Exception:
        # run_job already records the error
        pass


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
    method, scene = variant_method(variant), job_profile(job)
    with _lock:
        set_rating(conn, job_id, slot, {"like": 1, "dislike": -1, None: 0}[body.verdict], method, scene)
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
def create_refinement(job_id: str, body: RefinementIn, background: BackgroundTasks) -> dict:
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
    background.add_task(_safe_refine, refinement_id)
    return _public_refinement(get_refinement(conn, refinement_id))


def _safe_refine(refinement_id: str) -> None:
    settings = _s()
    conn = connect(settings.db_path)
    try:
        run_refinement(settings, conn, refinement_id)
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
    method, scene = variant_method(chosen), job_profile(job)
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
    allowed = {
        "source_preview.jpg",
        "critique_preview.jpg",
        "variant_A.jpg",
        "variant_B.jpg",
        "variant_C.jpg",
    }
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
