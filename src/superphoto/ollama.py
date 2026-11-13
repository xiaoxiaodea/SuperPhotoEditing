from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

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


def _trust_proxy(url: str) -> bool:
    # A system proxy must never intercept the loopback Ollama service.
    return urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}


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
        response = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2.0,
                             trust_env=_trust_proxy(settings.ollama_url))
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


def chat_vision(settings: Settings, model: str, prompt: str, image_b64: str | list[str],
                *, response_schema: dict | None = None, audit: dict | None = None) -> str:
    """Return completed assistant text, optionally recording a caller-owned audit.

    The audit stays in memory and excludes connection details and image payloads.
    Preserve the full response before validation, including HTTP error JSON, so
    a rejected or incomplete answer can still be investigated by the caller.
    """
    payload = {
        "model": model,
        "stream": False,
        "format": response_schema if response_schema is not None else "json",
        "think": False,
        "keep_alive": "30m",
        # A 1280px VL preview + the critique prompt is ~4k tokens. num_ctx 4096
        # leaves ~200 output tokens and truncates JSON (done_reason=length).
        "options": {"temperature": 0.2, "num_ctx": 16384, "num_predict": 4096},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64] if isinstance(image_b64, str) else image_b64,
            }
        ],
    }
    if audit is not None:
        audit.update(
            request={
                "model": model,
                "prompt": prompt,
                "options": dict(payload["options"]),
                "format": payload["format"],
                "think": payload["think"],
                "stream": payload["stream"],
                "keep_alive": payload["keep_alive"],
                "images": [_image_audit(encoded, index) for index, encoded
                           in enumerate(payload["messages"][0]["images"])],
            },
            http_status=None,
            response=None,
            response_metadata={},
            response_decode_error=None,
            error=None,
        )
    try:
        response = httpx.post(
            f"{settings.ollama_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=settings.ollama_timeout,
            trust_env=_trust_proxy(settings.ollama_url),
        )
        if audit is not None:
            audit["http_status"] = response.status_code
            try:
                data = response.json()
            except Exception as exc:
                audit["response_decode_error"] = {"type": type(exc).__name__}
                # An HTML/plain-text HTTP failure should still raise its HTTP
                # error, not be disguised as a JSON decoding problem.
                response.raise_for_status()
                raise
            audit["response"] = data
            if isinstance(data, dict):
                audit["response_metadata"] = {
                    key: data[key] for key in (
                        "model", "created_at", "done", "done_reason", "prompt_eval_count",
                        "eval_count", "prompt_eval_duration", "eval_duration",
                        "load_duration", "total_duration",
                    ) if key in data
                }
            response.raise_for_status()
        else:
            response.raise_for_status()
            data = response.json()
        if data.get("done_reason") == "length":
            raise ValueError("Ollama response was truncated (done_reason=length)")
        if data.get("done") is False:
            raise ValueError("Ollama response is not complete (done=false)")
        text = _assistant_text(data)
        if not text.strip():
            raise ValueError("Ollama returned empty content")
        return text
    except Exception as exc:
        if audit is not None:
            # Exception messages can contain credential-bearing request URLs.
            audit["error"] = {"type": type(exc).__name__}
        raise


def _image_audit(encoded: str, index: int) -> dict[str, Any]:
    """Describe the exact submitted bytes without retaining their base64 payload."""
    from PIL import Image

    result: dict[str, Any] = {"index": index}
    try:
        raw = base64.b64decode(encoded, validate=True)
        result.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        with Image.open(BytesIO(raw)) as image:
            result.update(width=image.width, height=image.height)
    except Exception as exc:
        # Metadata collection must not turn an otherwise compatible transport
        # call into a local failure (including older mocks with dummy images).
        result["error"] = {"type": type(exc).__name__}
    return result


def _assistant_text(data: dict[str, Any]) -> str:
    """qwen3-vl often puts format=json in `thinking` and leaves `content` empty."""
    message = data.get("message") or {}
    parts: list[str] = []
    for raw in (message.get("content"), message.get("thinking"), data.get("response")):
        if isinstance(raw, dict):
            parts.append(json.dumps(raw, ensure_ascii=False))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif isinstance(item, str) and item.strip():
                    parts.append(item)
        elif isinstance(raw, str) and raw.strip():
            parts.append(raw)
    for part in parts:
        stripped = part.strip()
        if stripped.startswith("{") or stripped.startswith("```"):
            return part
    return parts[0] if parts else ""


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
        httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.0, trust_env=False).raise_for_status()
        return True
    except Exception:
        return False


def _model_installed(model: str) -> bool:
    try:
        names = [
            m.get("name", "")
            for m in httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0, trust_env=False).json().get("models", [])
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
