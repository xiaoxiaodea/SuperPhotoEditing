from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings
from .db import connect, init_db
from .pipeline import run_job


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="superphoto")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("app", help="open the native SuperPhoto window")
    serve = sub.add_parser("serve", help="run the local web server only")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    process = sub.add_parser("process", help="process one photo on the CLI")
    process.add_argument("photo", type=Path)
    process.add_argument("--out", type=Path, default=None)

    imports = sub.add_parser("import-folder", help="submit stable photos to the running server")
    imports.add_argument("folder", type=Path)
    imports.add_argument("--watch", action="store_true")
    imports.add_argument("--url", default=None)
    imports.add_argument("--stable-seconds", type=float, default=5)
    imports.add_argument("--interval", type=float, default=5)

    sub.add_parser("install-app", help="install SuperPhoto.app to Applications and Desktop")
    sub.add_parser("install-lightcraft", help="download the pinned LightCraft CLI")
    sub.add_parser("lightcraft-status", help="check that lightcraft-cli can run")
    vision_check = sub.add_parser("vision-check", help="make one tiny real vision call and report latency")
    vision_check.add_argument("--list-models", action="store_true", help="also list the cloud provider's models")
    sub.add_parser("index-masters", help="index Hasselblad Masters for scene-matched grading")

    agent = sub.add_parser("agent", help="talk to a running SuperPhoto for the Persona assistant")
    agent_sub = agent.add_subparsers(dest="agent_cmd", required=True)
    agent_sub.add_parser("status", help="check the running server")
    edit = agent_sub.add_parser("edit", help="submit one photo and wait for the delivered finishes")
    edit.add_argument("photo", type=Path)
    edit.add_argument("--timeout", type=float, default=2700)
    job = agent_sub.add_parser("job", help="read one job and download any delivered images")
    job.add_argument("job_id")
    rate = agent_sub.add_parser("rate", help="remember a like or dislike for later photos")
    rate.add_argument("job_id")
    rate.add_argument("slot", choices=("A", "B", "C"))
    rate.add_argument("verdict", choices=("like", "dislike", "clear"))
    refine = agent_sub.add_parser("refine", help="re-render one finish from a short note")
    refine.add_argument("job_id")
    refine.add_argument("slot", choices=("A", "B", "C"))
    refine.add_argument("prompt")
    refine.add_argument("--parent", default=None)
    refine.add_argument("--timeout", type=float, default=1200)

    args = parser.parse_args(argv)
    if args.cmd in (None, "app"):
        from .desktop import main as desktop_main

        desktop_main()
    elif args.cmd == "serve":
        _serve(args.host, args.port)
    elif args.cmd == "process":
        _process(args.photo, args.out)
    elif args.cmd == "import-folder":
        if not 0 <= args.stable_seconds <= 60 or not 0 < args.interval <= 60:
            parser.error("stable-seconds must be 0..60; interval must be >0 and <=60")
        from .folder_import import import_folder
        import_folder(get_settings(), args.folder, watch=args.watch, url=args.url,
                      stable_seconds=args.stable_seconds, interval=args.interval)
    elif args.cmd == "install-app":
        from .macos_app import write_app

        path = write_app()
        print(path)
    elif args.cmd == "install-lightcraft":
        from .lightcraft import install_cli

        print(json.dumps(install_cli(), ensure_ascii=False, indent=2))
    elif args.cmd == "lightcraft-status":
        settings = get_settings()
        running = _running_app_lightroom_status(settings.host, settings.port)
        if running is not None:
            print(json.dumps(running, ensure_ascii=False, indent=2))
        else:
            from .lightroom import check_lightroom_connection

            print(json.dumps(check_lightroom_connection(settings.lightcraft_bin), ensure_ascii=False, indent=2))
    elif args.cmd == "vision-check":
        report = _vision_check(get_settings(), list_models=args.list_models)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report.get("ok"):
            raise SystemExit(1)
    elif args.cmd == "index-masters":
        from .masters import build_index

        print(build_index())
    elif args.cmd == "agent":
        from .agent import run_agent

        run_agent(args)


def _serve(host: str | None, port: int | None) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "superphoto.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
    )


def _process(photo: Path, out: Path | None) -> None:
    if not photo.is_file():
        raise SystemExit(f"not a file: {photo}")
    settings = get_settings()
    if out:
        settings.data_dir = out
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    init_db(conn)
    from .tasks import WorkerLock, enqueue_source

    lock = WorkerLock(settings.db_path.with_name(settings.db_path.name + ".worker.lock"))
    try:
        lock.acquire()
        queued = enqueue_source(settings, conn, photo)
        job_id = queued["id"]
        dest_dir = settings.jobs_dir / job_id
        dest = dest_dir / f"original{photo.suffix.lower()}"
        run_job(settings, conn, job_id, dest)
        print(json.dumps({"job_id": job_id, "dir": str(dest_dir), "status": "ready"}, ensure_ascii=False, indent=2))
    finally:
        lock.release()
        conn.close()


def _vision_check(settings, *, list_models: bool = False) -> dict[str, Any]:
    """One real request with a small solid-color image. Never prints keys or request bodies."""
    import base64
    import time
    from io import BytesIO

    from PIL import Image

    from .critique import _extract_json
    from .vision import VisionUnavailable, chat_vision, is_temporary, probe, safe_error

    selection = probe(settings)
    report: dict[str, Any] = {
        "requested_provider": settings.vision_provider, "provider": selection.get("provider"),
        "model": selection.get("resolved_model"), "ready": selection.get("ready"),
        "candidates": [{k: c.get(k) for k in ("provider", "model", "protocol")} for c in selection.get("candidates", [])],
        "unavailable": selection.get("unavailable", []), "pipeline_mode": settings.pipeline_mode, "ok": False,
        "temporary": is_temporary(selection.get("provider")),
    }
    if report["temporary"]:
        report["note"] = ("xAI/Grok is a temporary stand-in; in auto mode a running Ollama or a configured "
                          "OpenAI-compatible vision model takes over automatically")
    if list_models:
        report["models"] = _cloud_models(settings, selection.get("provider"))
    if selection.get("ready") is not True or not selection.get("resolved_model"):
        report["error"] = ("no vision model is available to this command. Models added in the client "
                           "(视觉模型 button) are unlocked only inside the client and are tested there; "
                           "otherwise start Ollama or set XAI_API_KEY etc. in .env (plain text, readable by "
                           "any program)")
        return report
    stream = BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(stream, format="JPEG", quality=90)
    image = base64.b64encode(stream.getvalue()).decode("ascii")
    audit: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        text = chat_vision(settings, selection["resolved_model"],
                           'What is the dominant color of this image? Reply with JSON only: {"color": "<one word>"}',
                           image, selection=selection, audit=audit)
    except VisionUnavailable as exc:
        report.update(error=safe_error(exc), attempts=audit.get("attempts"),
                      latency_s=round(time.perf_counter() - started, 2))
        return report
    report["latency_s"] = round(time.perf_counter() - started, 2)
    report.update(answered_by=audit.get("provider"), answered_model=audit.get("model"), usage=audit.get("usage"))
    try:
        color = str(_extract_json(text).get("color", "")).strip().lower()
    except ValueError:
        report["error"] = "the model answered, but not with JSON"
        return report
    report.update(color=color, ok=True, sees_image=color in {"red", "crimson", "scarlet", "红", "红色"})
    return report


def _cloud_models(settings, provider: str | None) -> list[str] | str:
    if provider not in {"xai", "openai"}:
        return "listing is only available for cloud providers"
    from openai import OpenAI

    try:
        with OpenAI(api_key=getattr(settings, f"{provider}_api_key"), base_url=getattr(settings, f"{provider}_base_url"),
                    timeout=30, max_retries=0) as client:
            return sorted(model.id for model in client.models.list())
    except Exception as exc:  # Never echo server bodies; they may include request details.
        return f"could not list models ({type(exc).__name__})"


def _running_app_lightroom_status(host: str, port: int) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"http://{host}:{port}/api/health", timeout=1.5, trust_env=False)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    status = payload.get("lightroom")
    return status if isinstance(status, dict) else None


if __name__ == "__main__":
    main()
