"""Native window: start the local server and open SuperPhoto."""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx


def project_root() -> Path:
    override = os.environ.get("SUPERPHOTO_ROOT")
    if override:
        cand = Path(override)
        if (cand / "web" / "index.html").is_file():
            return cand
    here = Path(__file__).resolve()
    for cand in (here.parents[2], here.parents[1], Path.cwd()):
        if (cand / "web" / "index.html").is_file() and (cand / "pyproject.toml").is_file():
            return cand
    return Path.cwd()


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def ui_url(base: str, root: Path) -> str:
    # Bust an old WebKit document cache even when upgrading from a server
    # that did not yet send Cache-Control: no-store.
    digest = hashlib.sha256()
    web = root / "web"
    if not (web / "index.html").is_file():
        web = Path(__file__).resolve().parent / "web"
    for name in ("index.html", "app.js", "styles.css"):
        digest.update((web / name).read_bytes())
    return f"{base.rstrip('/')}/?ui={digest.hexdigest()[:16]}"


def _wait_health(url: str, seconds: float = 20.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/api/auth/status", timeout=1.0, trust_env=False)
            if response.status_code == 200:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def _start_model_warmup(model: str) -> threading.Thread:
    """Warm the optional critic without delaying the local HTTP service."""

    def _warm() -> None:
        try:
            from .ollama import wake

            # Model downloads are explicit: desktop startup must not silently
            # pull a multi-gigabyte model when a cloud/local mode is sufficient.
            wake()
        except Exception as exc:
            print(f"SuperPhoto model warmup failed: {exc}", file=sys.stderr)

    thread = threading.Thread(target=_warm, daemon=True, name="superphoto-model-warmup")
    thread.start()
    return thread


def main() -> None:
    root = project_root()
    os.chdir(root)
    os.environ.setdefault("SUPERPHOTO_DATA_DIR", str(root / "data"))
    if sys.platform == "darwin":
        os.environ["PATH"] = os.pathsep.join(("/usr/local/bin", "/opt/homebrew/bin", os.environ.get("PATH", "")))

    from .config import get_settings

    settings = get_settings()
    host = settings.host
    port = settings.port
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    base = f"http://{browser_host}:{port}"

    if _port_open(host, port):
        if not _wait_health(base, seconds=20.0):
            print(f"SuperPhoto port {port} is busy but not healthy", file=sys.stderr)
            raise SystemExit(1)
    else:
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

    if settings.ollama_enabled and getattr(settings, "vision_provider", "auto") in {"auto", "ollama"}:
        _start_model_warmup(settings.ollama_model)

    try:
        import webview
    except ImportError:
        import webbrowser

        webbrowser.open(ui_url(base, root))
        print(f"Opened {base}. Install pywebview for a native window: uv sync")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    splash = """<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;height:100%;background:#141516;color:#eeeeef;
font-family:"Segoe UI","Microsoft YaHei",-apple-system,"PingFang SC",sans-serif;
display:flex;align-items:center;justify-content:center}
p{opacity:.7}
</style></head><body><p>正在打开 SuperPhoto…</p></body></html>"""
    window = webview.create_window(
        "SuperPhoto",
        html=splash,
        width=1440,
        height=920,
        min_size=(960, 700),
        background_color="#141516",
        text_select=True,
        easy_drag=False,
        transparent=False,
    )

    def _show_ui() -> None:
        time.sleep(0.15)
        window.load_url(ui_url(base, root))

    gui = "cocoa" if sys.platform == "darwin" else "edgechromium" if sys.platform == "win32" else None
    webview.start(_show_ui, gui=gui, private_mode=False, debug=False)


if __name__ == "__main__":
    main()
