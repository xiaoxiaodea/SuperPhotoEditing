from __future__ import annotations

import json
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
)
from .darktable_export import darktable_cli_path
from .ingest import ALLOWED_EXTS, source_kind
from .models import FeedbackIn
from .ollama import probe
from .pipeline import run_job
from .preference import describe_taste
from .recipes import LOOKS
from .store import job_dir

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
        conn = _state.get("conn")
        if conn is not None:
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
            _state["conn"] = conn
        return _state["settings"], _state["conn"]


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
    return {
        "ok": True,
        "critic": critic,
        "critic_model": critic_model,
        "ollama": ollama,
        "grok": bool(settings.xai_api_key),
        "model": settings.xai_model if settings.xai_api_key else None,
        "rawpy": rawpy_ok,
        "darktable_cli": darktable_cli_path(),
        "taste": describe_taste(_c()),
    }


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
    return {"total_picks": stats["total_picks"], "looks": titled, "summary": describe_taste(_c())}


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


def _safe_run(job_id: str, original: Path) -> None:
    settings, conn = _boot()
    try:
        run_job(settings, conn, job_id, original)
    except Exception:
        # run_job already records the error
        pass


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
                "moves": LOOKS[row["look_id"]].moves if row["look_id"] in LOOKS else [],
                "recipe": recipe,
                "url": f"/media/{job_id}/{row['filename']}",
            }
        )
    technical = json.loads(job["technical_json"]) if job.get("technical_json") else None
    critique = json.loads(job["critique_json"]) if job.get("critique_json") else None
    return {
        "id": job["id"],
        "status": job["status"],
        "step": job["step"],
        "error": _public_error(job.get("error")),
        "original_name": job.get("original_name"),
        "source_preview": f"/media/{job_id}/source_preview.jpg" if job["status"] in {"ready", "running"} else None,
        "technical": technical,
        "critique": critique,
        "variants": variants,
    }


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
    insert_feedback(
        conn,
        {
            "id": str(uuid.uuid4()),
            "job_id": job_id,
            "chosen_slot": body.chosen_slot,
            "look_id": chosen["look_id"],
            "technical_score": body.technical_score,
            "style_score": body.style_score,
            "restraint_score": body.restraint_score,
            "notes": body.notes,
        },
    )
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


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def create_app() -> FastAPI:
    return app
