from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    sub.add_parser("install-app", help="write SuperPhoto.app next to the project")

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


if __name__ == "__main__":
    main()
