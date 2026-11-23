"""Shared visual transport. No cloud requests are made by status/probe.

Call get_vision_status(settings) for a configuration-only status suitable for a
health endpoint. check_ollama=True additionally probes the local Ollama service;
cloud readiness still means configured, never verified connectivity or credits.
Stage callers use probe() and pass its result as selection to chat_vision().
Explicit providers fail closed to the caller's local/unavailable policy. In auto,
transport failures try the remaining configured providers in order. JSON/schema
validation remains the stage's responsibility; invalid judgments never win.

Models configured in the client live in the encrypted vault (vault.py). In auto
they are tried first, in the user's order. Everything outside this module sees
them only as provider "vault" with an opaque slot label as the model; keys,
endpoints and model ids stay in memory and out of audits, errors and status.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from openai import APIStatusError, OpenAI

from . import ollama, vault
from .config import Settings

# Request logs would name the endpoint of a vault profile.
for _name in ("httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class VisionUnavailable(RuntimeError):
    """Safe to expose: contains provider/type/status, never request URLs or bodies."""


# xAI (Grok) from .env is a temporary stand-in until another vision model is
# configured, so auto tries it last and audits mark it as temporary.
AUTO_ORDER = ("ollama", "openai", "xai")
TEMPORARY_PROVIDERS = frozenset({"xai"})
VAULT = "vault"


def _vault_candidates(settings: Settings) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    path = vault.vault_path(settings.data_dir)
    state = vault.status(path)
    if not state["exists"]:
        return [], [], state
    if not state["unlocked"]:
        return [], [{"provider": VAULT, "reason": "locked"}], state
    found = [{"provider": VAULT, "profile": slot["id"], "model": slot["label"], "label": slot["label"],
              "protocol": None, "availability": "configured"} for slot in vault.slots(path)]
    return found, ([] if found else [{"provider": VAULT, "reason": "empty"}]), state


BRIDGE_LABEL = "客户端视觉模型"


def _bridge_candidate(settings: Settings) -> list[dict[str, Any]]:
    if not (settings.vision_bridge_url and settings.vision_bridge_token):
        return []
    return [{"provider": VAULT, "bridge": True, "model": BRIDGE_LABEL, "label": BRIDGE_LABEL,
             "protocol": None, "availability": "configured"}]


def serve_bridge(settings: Settings, prompt: str, images: list[str], schema: dict | None, *,
                 timeout: float | None = None, audit: dict | None = None) -> str:
    """Client side of the loopback bridge: run one request through the vault profiles only."""
    candidates, _, _ = _vault_candidates(settings)
    if not candidates:
        raise VisionUnavailable("vision bridge: the client's vault is locked or has no enabled model")
    update: dict[str, Any] = {"vision_provider": "auto"}
    if timeout:
        update["vision_timeout"] = timeout
    record: dict[str, Any] = {}
    try:
        return chat_vision(settings.model_copy(update=update), candidates[0]["model"], prompt, images,
                           response_schema=schema, audit=record,
                           selection={"candidates": candidates, "unavailable": [], "fallback": False})
    finally:
        if audit is not None:
            audit.update({key: record.get(key) for key in ("provider", "model", "usage", "attempts", "fallback")})


def _bridge_complete(settings: Settings, prompt: str, images: list[str], schema: dict | None,
                     attempt: dict) -> str:
    """Caller side: forward to the running client. Its answer is already redacted there."""
    timeout = float(settings.vision_timeout)
    response = httpx.post(f"{settings.vision_bridge_url.rstrip('/')}/api/vision/bridge",
                          json={"prompt": prompt, "images": images, "schema": schema, "timeout": timeout},
                          headers={"X-SuperPhoto-Bridge": settings.vision_bridge_token},
                          timeout=min(timeout * 4 + 30, 1800), trust_env=False)
    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    remote = data.get("audit") or {}
    attempt["bridge_attempts"] = remote.get("attempts") or data.get("attempts")
    if response.status_code != 200 or not isinstance(data.get("text"), str):
        detail = data.get("detail") if isinstance(data.get("detail"), str) else ""
        raise VisionUnavailable(f"vision bridge: HTTP {response.status_code} {detail}".strip())
    attempt["usage"] = remote.get("usage")
    if isinstance(remote.get("model"), str):
        attempt["response_model"] = remote["model"]
    return data["text"]


def is_temporary(provider: str | None) -> bool:
    return provider in TEMPORARY_PROVIDERS


def _cloud(settings: Settings, provider: str) -> dict[str, Any]:
    return {"provider": provider, "model": getattr(settings, f"{provider}_model"),
            "protocol": getattr(settings, f"{provider}_protocol")}


def get_vision_status(settings: Settings, *, check_ollama: bool = False) -> dict[str, Any]:
    """Return secret-free selection information without contacting a cloud API.

    ready is True for a configured cloud, False for local/unavailable, and None
    for unprobed Ollama. availability explicitly distinguishes those meanings.
    auto order is the client's vault profiles (user order), Ollama, generic
    OpenAI-compatible, xAI (temporary stand-in), local.
    """
    requested = settings.vision_provider
    candidates: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    providers = AUTO_ORDER if requested == "auto" else (requested,)
    vault_state = {"exists": False, "unlocked": False, "count": None}
    bridged = _bridge_candidate(settings) if requested == "auto" else []
    if requested == "auto":
        candidates, unavailable, vault_state = _vault_candidates(settings)
        candidates = candidates + bridged
    first = VAULT if vault_state["exists"] or bridged else providers[0]
    for provider in providers:
        if provider == "local":
            continue
        if provider == "ollama":
            if not settings.ollama_enabled:
                unavailable.append({"provider": provider, "reason": "disabled"})
            elif check_ollama:
                status = ollama.probe(settings)
                if status.get("ready") and status.get("resolved_model"):
                    candidates.append({"provider": provider, "model": status["resolved_model"],
                                       "protocol": "ollama", "availability": "probed"})
                else:
                    # Local service errors may include URLs/passwords. Keep a fixed reason.
                    unavailable.append({"provider": provider, "reason": "not_ready"})
            else:
                candidates.append({"provider": provider, "model": settings.ollama_model,
                                   "protocol": "ollama", "availability": "unverified"})
        elif getattr(settings, f"{provider}_api_key") and getattr(settings, f"{provider}_model").strip():
            candidates.append({**_cloud(settings, provider), "availability": "configured"})
        else:
            unavailable.append({"provider": provider, "reason": "not_configured"})
    selected = candidates[0] if candidates else {"provider": "local", "model": None,
                                               "protocol": None, "availability": "local"}
    return {"requested_provider": requested, **selected,
            "ready": (None if selected["availability"] == "unverified" else bool(candidates)),
            "resolved_model": selected["model"], "candidates": candidates,
            "temporary": is_temporary(selected["provider"]),
            "vault": vault_state,
            "unavailable": unavailable, "cloud_connectivity_checked": False,
            "fallback": (selected["provider"] == "local" and requested != "local") or
                        (requested == "auto" and bool(candidates) and selected["provider"] != first)}


def probe(settings: Settings) -> dict[str, Any]:
    """Stage selection: probes only Ollama when allowed by the provider policy."""
    return get_vision_status(settings, check_ollama=True)


def provider_from(status: dict[str, Any], audit: dict | None = None) -> str:
    """Older stage mocks expose only ready/resolved_model and mean Ollama."""
    return (audit or {}).get("provider") or status.get("provider", "ollama")


def safe_error(exc: Exception) -> str:
    if isinstance(exc, VisionUnavailable):
        return str(exc)
    return type(exc).__name__


def _redact(value: Any, settings: Settings) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, settings) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, settings) for item in value]
    if isinstance(value, str):
        for key in (settings.openai_api_key, settings.xai_api_key):
            if key:
                value = value.replace(key, "[REDACTED]")
        for secret in vault.redaction_values():
            value = value.replace(secret, "[REDACTED]")
    return value


def chat_vision(settings: Settings, model: str, prompt: str, image_b64: str | list[str], *,
                response_schema: dict | None = None, audit: dict | None = None,
                selection: dict | None = None) -> str:
    """Complete a visual request, recording actual provider/model/usage and failures.

    selection avoids a second local probe. The legacy model positional argument
    is retained for stage monkeypatches and direct Ollama callers.
    """
    status = selection if selection is not None else probe(settings)
    candidates = status.get("candidates")
    if candidates is None:  # Legacy stage mocks.
        candidates = [{"provider": status.get("provider", "ollama"), "model": model}]
    record = audit if audit is not None else {}
    record.clear()
    record.update(requested_provider=settings.vision_provider, provider=None, model=None,
                  usage=None, attempts=[], unavailable=status.get("unavailable", []),
                  fallback=False, error=None)
    images = [image_b64] if isinstance(image_b64, str) else list(image_b64)
    if not images:
        raise VisionUnavailable("vision: no images")
    for candidate in candidates:
        provider = candidate["provider"]
        selected_model = candidate["model"]
        attempt: dict[str, Any] = {"provider": provider, "model": selected_model,
                                   "protocol": candidate.get("protocol", "ollama"), "error": None}
        record["attempts"].append(attempt)
        try:
            if provider == "ollama":
                transport: dict = {}
                try:
                    text = ollama.chat_vision(settings, selected_model, prompt, images,
                                             response_schema=response_schema, audit=transport)
                finally:
                    # Preserve the existing Ollama response audit, minus configured secrets.
                    attempt["transport"] = _redact(transport, settings)
                metadata = transport.get("response_metadata", {})
                attempt["usage"] = {key: metadata[key] for key in
                                    ("prompt_eval_count", "eval_count") if key in metadata} or None
                # Keep old top-level request/response metadata for existing audit readers.
                record.update(_redact(transport, settings))
            elif provider == VAULT:
                text = _vault_complete(settings, candidate, prompt, images, response_schema, attempt)
            else:
                text = _cloud_complete(settings, provider, selected_model, prompt, images,
                                       response_schema, attempt)
            record.update(provider=provider, model=attempt.get("response_model") or selected_model,
                          usage=attempt.get("usage"), error=None, temporary_provider=is_temporary(provider),
                          fallback=len(record["attempts"]) > 1 or status.get("fallback", False))
            cleaned = _redact(record, settings)
            record.clear()
            record.update(cleaned)
            return _redact(text, settings)
        except Exception as exc:
            attempt["error"] = {"type": type(exc).__name__}
            if isinstance(exc, APIStatusError):
                attempt["error"]["status"] = exc.status_code
            if settings.vision_provider != "auto":
                break
    record["fallback"] = True
    failures = ", ".join(f"{item['model'] if item['provider'] == VAULT else item['provider']}:{item['error']['type']}"
                         for item in record["attempts"])
    record["error"] = {"type": "VisionUnavailable", "reason": failures or "not_configured"}
    cleaned = _redact(record, settings)
    record.clear()
    record.update(cleaned)
    raise VisionUnavailable(f"vision unavailable ({failures or 'local/not_configured'})") from None


def _format_rejected(exc: APIStatusError) -> bool:
    if exc.status_code not in (400, 422):
        return False
    # Inspect only to classify, never retain server messages in logs/audits.
    body = json.dumps(exc.body, ensure_ascii=False).lower()
    return any(marker in body for marker in
               ("response_format", "json_schema", "json_object", "text.format", "text_format", "schema"))


def _data_url(encoded: str) -> str:
    if encoded.startswith("data:image/"):
        return encoded
    # Native previews are JPEG; also support PNG/WebP/GIF for generic callers.
    mime = "image/jpeg"
    if encoded.startswith("iVBOR"):
        mime = "image/png"
    elif encoded.startswith("UklGR"):
        mime = "image/webp"
    elif encoded.startswith("R0lGOD"):
        mime = "image/gif"
    return f"data:{mime};base64,{encoded}"


_COLOR_PROMPT = 'What is the dominant color of this image? Reply with JSON only: {"color": "<one word>"}'


def _red_square() -> str:
    import base64
    from io import BytesIO

    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(stream, format="JPEG", quality=90)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _error_status(exc: Exception) -> int | None:
    if isinstance(exc, APIStatusError):
        return exc.status_code
    response = getattr(exc, "response", None)  # httpx.HTTPStatusError from Ollama.
    return getattr(response, "status_code", None) if isinstance(exc, httpx.HTTPStatusError) else None


def _check_profile(settings: Settings, profile: dict[str, Any]) -> dict[str, Any]:
    """Send a 64x64 red square through one profile. Returns only safe facts, never text or server bodies."""
    from .critique import _extract_json

    started = time.perf_counter()
    private: dict[str, Any] = {}
    try:
        text = _profile_complete(settings, profile, _COLOR_PROMPT, [_red_square()], None, private)
    except Exception as exc:
        return {"ok": False, "latency_s": round(time.perf_counter() - started, 2),
                "error": type(exc).__name__, "status": _error_status(exc)}
    result: dict[str, Any] = {"ok": True, "latency_s": round(time.perf_counter() - started, 2),
                              "usage": private.get("usage")}
    try:
        color = str(_extract_json(text).get("color", "")).strip().lower()
    except ValueError:
        return {**result, "json": False, "sees_image": None}
    return {**result, "json": True,
            "sees_image": "red" in color or "红" in color or color in {"crimson", "scarlet", "maroon"}}


def test_profile(settings: Settings, profile_id: str) -> dict[str, Any]:
    """One tiny real request through a single vault profile. Returns only safe facts."""
    profile = vault.resolve(profile_id)
    if profile is None:
        raise VisionUnavailable("vision profile locked or removed")
    return _check_profile(settings, profile)


# Answers that mean every model at this address will fail the same way.
_STOP_STATUS = {401, 402, 403, 429}
_STOP_ERRORS = {"APIConnectionError", "ConnectError", "ConnectTimeout", "AuthenticationError",
                "PermissionDeniedError", "RateLimitError"}


def list_models(settings: Settings, profile: dict[str, Any]) -> list[str]:
    """Model ids offered at the profile's address. Raises on any failure; callers report only the type."""
    if profile["kind"] == "ollama":
        base = profile["base_url"].rstrip("/")
        response = httpx.get(f"{base}/api/tags", timeout=5.0, trust_env=ollama._trust_proxy(base))
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", []) if isinstance(m, dict)]
    with OpenAI(api_key=profile.get("api_key") or "not-needed", base_url=profile["base_url"],
                timeout=20, max_retries=0) as client:
        # Gemini lists "models/gemini-…" but is called with the bare id.
        return [m.id.removeprefix("models/") for m in client.models.list() if isinstance(getattr(m, "id", None), str)]


def discover_model(settings: Settings, profile: dict[str, Any], *, limit: int = 3) -> dict[str, Any]:
    """Pick a model that can read images, so the user only chooses a provider and pastes a key.

    The provider's model list is ranked with the preset's hints, and the top
    few are sent the red test square until one names its color. Only safe
    facts come back: the chosen model id (shown to the passphrase holder like
    any saved model), counts, and error types or HTTP statuses.
    """
    from .vision_presets import PRESETS, rank_models

    preset = PRESETS.get(str(profile.get("preset")), PRESETS["custom"])
    started = time.perf_counter()
    result: dict[str, Any] = {"model": None, "listed": None, "tried": 0, "error": None, "status": None,
                              "answered": 0}
    try:
        names = list_models(settings, profile)
        result["listed"] = len(names)
    except Exception as exc:
        # Some providers do not list models (or not with this key format). Their usual ids still get a try.
        names = []
        result.update(list_error=type(exc).__name__, list_status=_error_status(exc))
    ranked = rank_models(names, preset["prefer"])
    if names:
        rest = [n for n in rank_models(names, [r"."]) if n not in ranked]
        candidates = ranked + rest
    else:
        candidates = list(preset["fallback"])
    candidates = candidates[:limit]
    # A cold local model can take minutes to load; cloud calls should answer fast.
    quick = settings.model_copy(update={"vision_timeout": min(settings.vision_timeout, 60.0),
                                        "vision_max_retries": 0})
    for model in candidates:
        result["tried"] += 1
        check = _check_profile(quick, {**profile, "model": model})
        if check["ok"] and check.get("sees_image"):
            result.update(model=model, error=None, status=None, latency_s=check["latency_s"])
            break
        if check["ok"]:
            result["answered"] += 1
            continue
        result.update(error=check["error"], status=check.get("status"))
        if check.get("status") in _STOP_STATUS or check["error"] in _STOP_ERRORS:
            break
    else:
        if result["answered"]:
            result.update(error="no_image", status=None)
    if not candidates:
        result["error"] = "no_candidates" if names or not result.get("list_error") else result["list_error"]
        result["status"] = result.get("list_status")
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    return result


def _profile_complete(settings: Settings, profile: dict[str, Any], prompt: str, images: list[str],
                      schema: dict | None, private: dict) -> str:
    """Call one profile's endpoint directly. private receives transport details; never persist it."""
    if profile["kind"] == "ollama":
        local = settings.model_copy(update={"ollama_url": profile["base_url"]})
        try:
            return ollama.chat_vision(local, profile["model"], prompt, images, response_schema=schema, audit=private)
        finally:
            metadata = private.get("response_metadata") or {}
            private["usage"] = {key: metadata[key] for key in ("prompt_eval_count", "eval_count")
                                if isinstance(metadata.get(key), int)} or None
    return _openai_complete(settings, api_key=profile.get("api_key") or "not-needed", base_url=profile["base_url"],
                            protocol=profile.get("protocol") or "chat/completions", model=profile["model"],
                            prompt=prompt, images=images, schema=schema, audit=private)


def _vault_complete(settings: Settings, candidate: dict[str, Any], prompt: str, images: list[str],
                    schema: dict | None, attempt: dict) -> str:
    """Call one client-configured profile. Only token counts and format modes reach the audit."""
    if candidate.get("bridge"):
        return _bridge_complete(settings, prompt, images, schema, attempt)
    profile = vault.resolve(str(candidate.get("profile") or ""))
    if profile is None:
        raise VisionUnavailable("vision profile locked or removed")
    private: dict[str, Any] = {}
    try:
        return _profile_complete(settings, profile, prompt, images, schema, private)
    finally:
        attempt["usage"] = private.get("usage")
        if profile["kind"] != "ollama":
            if private.get("schema_mode"):
                attempt["schema_mode"] = private["schema_mode"]
            attempt["format_attempts"] = [{"mode": item.get("mode"),
                                           **({"error": item["error"]} if item.get("error") else {})}
                                          for item in private.get("format_attempts", [])]


def _cloud_complete(settings: Settings, provider: str, model: str, prompt: str,
                    images: list[str], schema: dict | None, audit: dict) -> str:
    return _openai_complete(settings, api_key=getattr(settings, f"{provider}_api_key"),
                            base_url=getattr(settings, f"{provider}_base_url"),
                            protocol=getattr(settings, f"{provider}_protocol"), model=model,
                            prompt=prompt, images=images, schema=schema, audit=audit)


def _openai_complete(settings: Settings, *, api_key: str, base_url: str, protocol: str, model: str,
                     prompt: str, images: list[str], schema: dict | None, audit: dict) -> str:
    mode = settings.vision_schema_mode
    formats = (["json_schema", "json_object", "prompt"] if schema else ["json_object", "prompt"])
    if mode != "auto":
        formats = [mode if mode != "json_schema" or schema else "json_object"]
    if schema is not None:
        # Kept even after format negotiation, so fallback still has the contract.
        prompt += "\nReturn JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False)
    audit.update(protocol=protocol, usage=None, format_attempts=[])
    with OpenAI(api_key=api_key, base_url=base_url,
                timeout=settings.vision_timeout, max_retries=settings.vision_max_retries) as client:
        for index, format_mode in enumerate(formats):
            format_audit: dict[str, Any] = {"mode": format_mode}
            audit["format_attempts"].append(format_audit)
            output_format: dict | None = None
            if format_mode == "json_schema":
                output_format = {"type": "json_schema", "name": "vision_result",
                                 "schema": schema, "strict": False}
            elif format_mode == "json_object":
                output_format = {"type": "json_object"}
            try:
                if protocol == "responses":
                    kwargs = {"text": {"format": output_format}} if output_format else {}
                    response = client.responses.create(
                        model=model, store=False,
                        input=[{"role": "user", "content": [
                            {"type": "input_text", "text": prompt},
                            *({"type": "input_image", "image_url": _data_url(image), "detail": "high"}
                              for image in images)]}], **kwargs)
                    _response_metadata(response, audit)
                    if response.status != "completed":
                        raise ValueError("Vision Responses output incomplete")
                    if any(getattr(part, "type", None) == "refusal"
                           for item in response.output for part in getattr(item, "content", [])):
                        raise ValueError("Vision response refused")
                    text = response.output_text
                else:
                    if output_format and format_mode == "json_schema":
                        output_format = {"type": "json_schema", "json_schema": {
                            key: value for key, value in output_format.items() if key != "type"}}
                    kwargs = {"response_format": output_format} if output_format else {}
                    response = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            *({"type": "image_url", "image_url": {"url": _data_url(image), "detail": "high"}}
                              for image in images)]}], **kwargs)
                    _response_metadata(response, audit)
                    if not response.choices or response.choices[0].finish_reason != "stop":
                        raise ValueError("Vision Chat output incomplete")
                    message = response.choices[0].message
                    if message.refusal:
                        raise ValueError("Vision response refused")
                    text = message.content
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Vision returned empty content")
                audit["schema_mode"] = format_mode
                return text
            except APIStatusError as exc:
                format_audit.update(error={"type": type(exc).__name__, "status": exc.status_code})
                if mode == "auto" and index + 1 < len(formats) and _format_rejected(exc):
                    continue
                raise
    raise VisionUnavailable("vision: no output")


def _response_metadata(response: Any, audit: dict) -> None:
    # Explicit allowlist: never keep the response/request object or HTTP headers.
    if isinstance(getattr(response, "model", None), str):
        audit["response_model"] = response.model
    usage = getattr(response, "usage", None)
    if usage is not None:
        # Exclude arbitrary compatible-server extras that could echo credentials.
        audit["usage"] = {key: value for key in ("input_tokens", "output_tokens", "total_tokens",
                         "prompt_tokens", "completion_tokens")
                         if isinstance(value := getattr(usage, key, None), int)} or None
