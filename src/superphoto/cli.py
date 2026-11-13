from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings
from .db import connect, init_db, insert_job
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

    sub.add_parser("install-app", help="install SuperPhoto.app to Applications and Desktop")
    sub.add_parser("install-lightroom", help="install the bundled Lightroom Classic plugin")
    sub.add_parser("lightroom-status", help="check the Lightroom Classic MCP connection")
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
    elif args.cmd == "install-app":
        from .macos_app import write_app

        path = write_app()
        print(path)
    elif args.cmd == "install-lightroom":
        from .lightroom import install_lightroom_plugin

        print(json.dumps(install_lightroom_plugin(get_settings().lightroom_mcp_server), ensure_ascii=False, indent=2))
    elif args.cmd == "lightroom-status":
        settings = get_settings()
        running = _running_app_lightroom_status(settings.host, settings.port)
        if running is not None:
            print(json.dumps(running, ensure_ascii=False, indent=2))
        else:
            from .lightroom import check_lightroom_connection

            print(json.dumps(check_lightroom_connection(settings.lightroom_mcp_server), ensure_ascii=False, indent=2))
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
    from uuid import uuid4

    job_id = str(uuid4())
    dest_dir = settings.jobs_dir / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"original{photo.suffix.lower()}"
    dest.write_bytes(photo.read_bytes())
    insert_job(conn, job_id, photo.name)
    run_job(settings, conn, job_id, dest)
    print(json.dumps({"job_id": job_id, "dir": str(dest_dir), "status": "ready"}, ensure_ascii=False, indent=2))


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
