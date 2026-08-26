"""Native window: start the local server and open SuperPhoto."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx


def project_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parents[2], here.parents[1], Path.cwd()):
        if (cand / "web" / "index.html").is_file() and (cand / "pyproject.toml").is_file():
            return cand
    return Path.cwd()


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _wait_health(url: str, seconds: float = 20.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/api/health", timeout=1.0)
            if response.status_code == 200:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def main() -> None:
    root = project_root()
    os.chdir(root)
    os.environ.setdefault("SUPERPHOTO_DATA_DIR", str(root / "data"))
    os.environ.setdefault("PATH", "/usr/local/bin:/opt/homebrew/bin:" + os.environ.get("PATH", ""))

    from .config import get_settings
    from .ollama import ensure_model, wake

    settings = get_settings()
    host = settings.host
    port = settings.port
    base = f"http://{host}:{port}"

    wake()
    ensure_model(settings.ollama_model)

    live = _port_open(host, port) and _wait_health(base, seconds=2.0)
    if not live:
        from .app import app
        import uvicorn

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True, name="superphoto-http")
        thread.start()
        if not _wait_health(base):
            print(f"SuperPhoto failed to start on {base}", file=sys.stderr)
            raise SystemExit(1)

    try:
        import webview
    except ImportError:
        import webbrowser

        webbrowser.open(base)
        print(f"Opened {base}. Install pywebview for a native window: uv sync")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    webview.create_window(
        "SuperPhoto",
        base,
        width=1440,
        height=920,
        min_size=(960, 700),
        background_color="#0c0b09",
    )
    webview.start()


if __name__ == "__main__":
    main()
