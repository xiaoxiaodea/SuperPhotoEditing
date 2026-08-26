from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import httpx

from .config import Settings

VL_HINTS = ("-vl", "vl:", "vision", "llava", "minicpm-v", "gemma3")
PREFERRED = (
    "qwen3-vl:32b",
    "qwen3-vl:32b-instruct",
    "qwen3-vl:30b",
    "qwen3-vl:8b",
    "qwen3-vl:latest",
    "qwen2.5vl:7b",
    "llava:latest",
)


def probe(settings: Settings) -> dict[str, Any]:
    if not settings.ollama_enabled:
        return {
            "ok": False,
            "ready": False,
            "model": settings.ollama_model,
            "resolved_model": None,
            "models": [],
            "error": "disabled",
        }
    try:
        response = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2.0)
        response.raise_for_status()
        names = [m.get("name", "") for m in response.json().get("models", [])]
    except Exception as exc:
        return {
            "ok": False,
            "ready": False,
            "model": settings.ollama_model,
            "resolved_model": None,
            "models": [],
            "error": str(exc),
        }
    resolved = pick_model(settings.ollama_model, names)
    return {
        "ok": True,
        "ready": resolved is not None,
        "model": settings.ollama_model,
        "resolved_model": resolved,
        "models": names,
        "error": None,
    }


def pick_model(preferred: str, names: list[str]) -> str | None:
    wanted = [preferred, *PREFERRED]
    for candidate in wanted:
        for name in names:
            if _same_model(candidate, name):
                return name
    for name in names:
        lowered = name.lower()
        if any(hint in lowered for hint in VL_HINTS):
            return name
    return None


def chat_vision(settings: Settings, model: str, prompt: str, image_b64: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_ctx": 4096},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    }
    response = httpx.post(
        f"{settings.ollama_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=settings.ollama_timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data.get("message") or {}
    content = message.get("content") or message.get("thinking") or data.get("response") or ""
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def wake() -> bool:
    if _tags_ok():
        return True
    if shutil.which("open"):
        subprocess.Popen(["open", "-a", "Ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    binary = shutil.which("ollama")
    if binary:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    import time

    for _ in range(30):
        time.sleep(0.4)
        if _tags_ok():
            return True
    return False


def ensure_model(model: str) -> subprocess.Popen[bytes] | None:
    if shutil.which("ollama") is None:
        return None
    if _model_installed(model):
        return None
    return subprocess.Popen(
        ["ollama", "pull", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _tags_ok() -> bool:
    try:
        httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0).raise_for_status()
        return True
    except Exception:
        return False


def _model_installed(model: str) -> bool:
    try:
        names = [
            m.get("name", "")
            for m in httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0).json().get("models", [])
        ]
    except Exception:
        return False
    return pick_model(model, names) is not None


def _same_model(a: str, b: str) -> bool:
    def norm(x: str) -> str:
        x = x.lower()
        if ":" not in x:
            x = f"{x}:latest"
        return x

    return norm(a) == norm(b) or a.lower() == b.split(":")[0].lower() and b.endswith(":latest")
