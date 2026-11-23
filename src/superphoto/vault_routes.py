"""Client endpoints for the encrypted vision-model vault.

Without the passphrase, a caller sees only whether the vault exists, whether it
is unlocked and how many models are enabled. Entering the passphrase in the
client issues a short-lived HttpOnly cookie that allows viewing and editing
profiles. API keys are write-only and are never returned, not even to that
cookie holder.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import vault
from .config import Settings

COOKIE = "superphoto_vault"
SESSION_SECONDS = 2 * 60 * 60
_sessions: dict[str, float] = {}
_sessions_lock = threading.Lock()

BRIDGE_FILE = "vision-bridge.token"
BRIDGE_HEADER = "x-superphoto-bridge"
_bridge: dict[str, Any] = {"token": None, "path": None}
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def publish_bridge(settings: Settings) -> Path | None:
    """Open the loopback bridge for this run, if enabled. Returns the token file."""
    if not settings.vision_bridge:
        return None
    token = secrets.token_urlsafe(32)
    path = Path(settings.data_dir) / BRIDGE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="ascii")
    if os.name != "nt":
        os.chmod(path, 0o600)
    _bridge.update(token=token, path=path)
    return path


def retract_bridge() -> None:
    path = _bridge.get("path")
    _bridge.update(token=None, path=None)
    if path is not None:
        Path(path).unlink(missing_ok=True)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _new_session(response: JSONResponse, request: Request) -> None:
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        now = time.time()
        for key in [k for k, expiry in _sessions.items() if expiry < now]:
            del _sessions[key]
        _sessions[_digest(token)] = now + SESSION_SECONDS
    # Session cookie: closing the client forgets it; the server also forgets it on lock.
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/api/vision",
                        secure=request.url.scheme == "https")


def _clear_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()


def _managing(request: Request) -> bool:
    token = request.cookies.get(COOKIE, "")
    if not token:
        return False
    with _sessions_lock:
        expiry = _sessions.get(_digest(token))
        return expiry is not None and expiry >= time.time()


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "拒绝跨站请求")
    if "application/json" not in request.headers.get("content-type", ""):
        raise HTTPException(415, "需要 JSON 请求")


async def _body(request: Request) -> dict[str, Any]:
    _same_origin(request)
    try:
        data = await request.json()
    except Exception:
        # Never echo the body: it may hold a key or passphrase.
        raise HTTPException(400, "请求格式无效") from None
    if not isinstance(data, dict):
        raise HTTPException(400, "请求格式无效")
    return data


def _fail(exc: vault.VaultError) -> HTTPException:
    return HTTPException(400, str(exc))


_MANUAL = "可以展开「高级设置」手动填写模型名。"


def discovery_message(found: dict[str, Any]) -> str:
    """Why no model was chosen, from error types and statuses only. Nothing was saved."""
    error, status, tried = found.get("error"), found.get("status"), found.get("tried", 0)
    code = f"（HTTP {status}）" if status else ""
    if status == 401 or error == "AuthenticationError":
        return f"API Key 无效，或者不是这个服务商的 Key{code}。没有保存。"
    if status == 402:
        return f"账户余额不足{code}。没有保存。"
    if status == 403 or error == "PermissionDeniedError":
        return f"这个 Key 没有调用模型的权限{code}。没有保存。"
    if status == 429 or error == "RateLimitError":
        return f"额度不足或请求过于频繁{code}，请稍后再试。没有保存。"
    if error in {"APIConnectionError", "ConnectError", "ConnectTimeout"}:
        return "连不上服务地址。如果这个服务需要代理，请确认系统代理已开启。没有保存。"
    if error in {"APITimeoutError", "ReadTimeout", "TimeoutException"}:
        return f"请求超时。没有保存。{_MANUAL}"
    if error == "no_image":
        return f"试了 {tried} 个模型，都有回答，但都没认出测试图的颜色，可能不支持图片输入。没有保存。{_MANUAL}"
    if error == "no_candidates":
        return f"这个服务商的模型列表里没有找到能看图的模型。没有保存。{_MANUAL}"
    if not tried:
        return f"无法获取这个服务商的模型列表（{error}{code}）。没有保存。{_MANUAL}"
    if status in {400, 404, 422}:
        return (f"试了 {tried} 个模型，服务商都拒绝了带图片的请求{code}，"
                f"它目前提供的模型可能不支持图片输入。没有保存。{_MANUAL}")
    return f"自动选择模型失败（{error}{code}）。没有保存。{_MANUAL}"


def build_router(settings: Callable[[], Settings]) -> APIRouter:
    router = APIRouter(prefix="/api/vision")

    def path():
        return vault.vault_path(settings().data_dir)

    def view(request: Request, managing: bool | None = None) -> dict[str, Any]:
        state = vault.status(path())
        managing = _managing(request) if managing is None else managing
        managing = managing and state["unlocked"]
        data: dict[str, Any] = {**state, "manage": managing, "min_passphrase": vault.MIN_PASSPHRASE,
                                "bridge": bool(_bridge["token"])}
        if managing:
            data["profiles"] = vault.management_view(path())
        return data

    def require_manage(request: Request) -> None:
        if not (_managing(request) and vault.status(path())["unlocked"]):
            raise HTTPException(403, "请先输入保险箱口令")

    @router.get("/vault")
    def vault_status(request: Request) -> dict[str, Any]:
        return view(request)

    @router.post("/vault/create")
    async def vault_create(request: Request) -> JSONResponse:
        body = await _body(request)
        try:
            await asyncio.to_thread(vault.create, path(), body.get("passphrase"))
        except vault.VaultError as exc:
            raise _fail(exc) from None
        response = JSONResponse(view(request, managing=True))
        _new_session(response, request)
        return response

    @router.post("/vault/unlock")
    async def vault_unlock(request: Request) -> JSONResponse:
        body = await _body(request)
        try:
            await asyncio.to_thread(vault.unlock, path(), body.get("passphrase"))
        except vault.VaultError as exc:
            raise _fail(exc) from None
        response = JSONResponse(view(request, managing=True))
        _new_session(response, request)
        return response

    @router.post("/vault/lock")
    async def vault_lock(request: Request) -> JSONResponse:
        _same_origin(request)
        vault.lock()
        _clear_sessions()
        response = JSONResponse(view(request, managing=False))
        response.delete_cookie(COOKIE, path="/api/vision")
        return response

    @router.post("/vault/passphrase")
    async def vault_passphrase(request: Request) -> dict[str, Any]:
        body = await _body(request)
        require_manage(request)
        try:
            await asyncio.to_thread(vault.change_passphrase, path(), body.get("old"), body.get("new"))
        except vault.VaultError as exc:
            raise _fail(exc) from None
        return view(request)

    @router.post("/vault/reset")
    async def vault_reset(request: Request) -> JSONResponse:
        body = await _body(request)
        if body.get("confirm") != "重置":
            raise HTTPException(400, "请输入“重置”确认删除全部模型配置")
        vault.reset(path())
        _clear_sessions()
        response = JSONResponse(view(request, managing=False))
        response.delete_cookie(COOKIE, path="/api/vision")
        return response

    @router.get("/presets")
    def presets() -> list[dict[str, Any]]:
        from .vision_presets import public_presets

        return public_presets()

    @router.post("/profiles")
    async def save_profile(request: Request) -> dict[str, Any]:
        body = await _body(request)
        require_manage(request)
        try:
            draft = vault.draft_profile(path(), body)
        except vault.VaultError as exc:
            raise _fail(exc) from None
        found: dict[str, Any] | None = None
        if not draft["model"]:
            from .vision import discover_model

            found = await asyncio.to_thread(discover_model, settings(), draft)
            if not found.get("model"):
                raise HTTPException(400, discovery_message(found))
            draft["model"] = found["model"]
        try:
            vault.commit_profile(path(), draft, new=not body.get("id"))
        except vault.VaultError as exc:
            raise _fail(exc) from None
        data = view(request)
        if found:
            data["discovery"] = {"model": found["model"], "tried": found["tried"], "latency_s": found.get("latency_s")}
        return data

    @router.post("/profiles/order")
    async def order_profiles(request: Request) -> dict[str, Any]:
        body = await _body(request)
        require_manage(request)
        try:
            vault.reorder(path(), body.get("ids"))
        except vault.VaultError as exc:
            raise _fail(exc) from None
        return view(request)

    @router.post("/profiles/{profile_id}/delete")
    async def delete_profile(profile_id: str, request: Request) -> dict[str, Any]:
        await _body(request)
        require_manage(request)
        try:
            vault.delete_profile(path(), profile_id)
        except vault.VaultError as exc:
            raise _fail(exc) from None
        return view(request)

    @router.post("/bridge")
    async def bridge(request: Request) -> JSONResponse:
        """One vision request through the unlocked vault, for a local process holding the run token."""
        token = _bridge["token"]
        if not token:
            raise HTTPException(404, "Not Found")
        if request.client is None or request.client.host not in _LOOPBACK:
            raise HTTPException(403, "只接受本机请求")
        if not hmac.compare_digest(request.headers.get(BRIDGE_HEADER, "").encode(), token.encode()):
            raise HTTPException(403, "令牌不正确")
        body = await _body(request)
        prompt, images, schema = body.get("prompt"), body.get("images"), body.get("schema")
        if not isinstance(prompt, str) or not prompt or len(prompt) > 400_000:
            raise HTTPException(400, "prompt 无效")
        if (not isinstance(images, list) or not 1 <= len(images) <= 8
                or not all(isinstance(i, str) and 0 < len(i) <= 16_000_000 for i in images)):
            raise HTTPException(400, "images 无效")
        if schema is not None and not isinstance(schema, dict):
            raise HTTPException(400, "schema 无效")
        timeout = body.get("timeout")
        timeout = min(max(float(timeout), 10.0), 600.0) if isinstance(timeout, (int, float)) else None
        from .vision import VisionUnavailable, safe_error, serve_bridge

        audit: dict[str, Any] = {}
        try:
            text = await asyncio.to_thread(serve_bridge, settings(), prompt, images, schema,
                                           timeout=timeout, audit=audit)
        except VisionUnavailable as exc:
            return JSONResponse({"detail": safe_error(exc), "attempts": audit.get("attempts")}, status_code=503)
        return JSONResponse({"text": text, "audit": audit})

    @router.post("/profiles/{profile_id}/test")
    async def test_profile(profile_id: str, request: Request) -> dict[str, Any]:
        await _body(request)
        require_manage(request)
        from .vision import VisionUnavailable, test_profile as run_test

        try:
            return await asyncio.to_thread(run_test, settings(), profile_id)
        except VisionUnavailable:
            raise HTTPException(404, "找不到这个模型配置") from None

    return router
