"""Encrypted vision-model profiles: API keys, endpoints and model ids.

The vault file is AES-256-GCM ciphertext. Its key is derived with scrypt from
a passphrase that the user types into the client and that is never stored.
Decrypted profiles exist only in this process's memory while the vault is
unlocked. Profile values are never logged, never written to job audits and
never returned by the public API. Code outside the management view sees only
opaque slot labels such as "视觉模型 1".
"""
from __future__ import annotations

import base64
import copy
import ipaddress
import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .vision_presets import PRESETS

FORMAT = "superphoto-vision-vault"
VERSION = 1
FILENAME = "vision-vault.json"
_AAD = b"superphoto-vision-vault-v1"
# About 64 MB and a fraction of a second per derivation, so that each offline
# guess is expensive too.
_KDF = {"name": "scrypt", "n": 2 ** 16, "r": 8, "p": 1}
MIN_PASSPHRASE = 10
MAX_PROFILES = 12
KINDS = ("openai", "ollama")
PROTOCOLS = ("chat/completions", "responses")


class VaultError(RuntimeError):
    """A safe message for the user. It never contains profile values."""


class _State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.path: Path | None = None
        self.key: bytes | None = None
        self.salt: bytes = b""
        self.kdf: dict[str, Any] = dict(_KDF)
        self.profiles: list[dict[str, Any]] = []
        self.failures = 0


_state = _State()


def vault_path(data_dir: Path) -> Path:
    return Path(data_dir) / FILENAME


def _same(a: Path | None, b: Path) -> bool:
    return a is not None and Path(a).resolve() == Path(b).resolve()


def _derive(passphrase: str, salt: bytes, kdf: dict[str, Any]) -> bytes:
    return Scrypt(salt=salt, length=32, n=int(kdf["n"]), r=int(kdf["r"]), p=int(kdf["p"])).derive(
        passphrase.encode("utf-8"))


def _check_passphrase(passphrase: Any) -> str:
    if not isinstance(passphrase, str) or len(passphrase) < MIN_PASSPHRASE:
        raise VaultError(f"口令至少需要 {MIN_PASSPHRASE} 个字符")
    if len(passphrase) > 512:
        raise VaultError("口令过长")
    return passphrase


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _write(path: Path, key: bytes, salt: bytes, kdf: dict[str, Any], profiles: list[dict[str, Any]]) -> None:
    nonce = secrets.token_bytes(12)
    plain = json.dumps({"profiles": profiles, "updated": int(time.time())}, ensure_ascii=False).encode("utf-8")
    envelope = {"format": FORMAT, "version": VERSION, "kdf": {**kdf, "salt": _b64(salt)},
                "nonce": _b64(nonce), "ciphertext": _b64(AESGCM(key).encrypt(nonce, plain, _AAD))}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".vault-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle)
        if os.name != "nt":
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read(path: Path, passphrase: str) -> tuple[bytes, bytes, dict[str, Any], list[dict[str, Any]]]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("format") != FORMAT or envelope.get("version") != VERSION:
            raise ValueError("format")
        kdf = {k: envelope["kdf"][k] for k in ("name", "n", "r", "p")}
        if kdf["name"] != "scrypt":
            raise ValueError("kdf")
        salt = base64.b64decode(envelope["kdf"]["salt"])
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
    except (OSError, ValueError, KeyError, TypeError):
        raise VaultError("保险箱文件无法读取") from None
    key = _derive(passphrase, salt, kdf)
    try:
        plain = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
    except InvalidTag:
        raise VaultError("口令不正确") from None
    profiles = json.loads(plain.decode("utf-8")).get("profiles") or []
    return key, salt, kdf, [p for p in profiles if isinstance(p, dict)]


def status(path: Path) -> dict[str, Any]:
    with _state.lock:
        unlocked = _same(_state.path, path) and _state.key is not None
        return {"exists": Path(path).is_file(), "unlocked": unlocked,
                "count": sum(1 for p in _state.profiles if p.get("enabled", True)) if unlocked else None}


def create(path: Path, passphrase: str) -> None:
    passphrase = _check_passphrase(passphrase)
    with _state.lock:
        if Path(path).exists():
            raise VaultError("保险箱已经存在")
        salt = secrets.token_bytes(16)
        key = _derive(passphrase, salt, _KDF)
        _write(Path(path), key, salt, dict(_KDF), [])
        _state.path, _state.key, _state.salt, _state.kdf, _state.profiles = Path(path), key, salt, dict(_KDF), []
        _state.failures = 0


def unlock(path: Path, passphrase: Any) -> None:
    if not isinstance(passphrase, str) or not passphrase:
        raise VaultError("请输入口令")
    with _state.lock:
        if not Path(path).is_file():
            raise VaultError("还没有创建保险箱")
        # Slow repeated guesses through the client. Offline guesses already pay scrypt.
        if _state.failures:
            time.sleep(min(2 ** _state.failures, 30) * 0.25)
        try:
            key, salt, kdf, profiles = _read(Path(path), passphrase)
        except VaultError:
            _state.failures += 1
            raise
        _state.path, _state.key, _state.salt, _state.kdf, _state.profiles = Path(path), key, salt, kdf, profiles
        _state.failures = 0


def lock() -> None:
    with _state.lock:
        _state.path, _state.key, _state.salt, _state.profiles = None, None, b"", []


def reset(path: Path) -> None:
    """Forget the passphrase: delete every stored profile."""
    with _state.lock:
        if _same(_state.path, path):
            lock()
        Path(path).unlink(missing_ok=True)


def change_passphrase(path: Path, old: Any, new: Any) -> None:
    new = _check_passphrase(new)
    with _state.lock:
        _read(Path(path), old if isinstance(old, str) else "")
        _require(path)
        salt = secrets.token_bytes(16)
        key = _derive(new, salt, _KDF)
        _write(Path(path), key, salt, dict(_KDF), _state.profiles)
        _state.key, _state.salt, _state.kdf = key, salt, dict(_KDF)


def _require(path: Path) -> None:
    if not (_same(_state.path, path) and _state.key is not None):
        raise VaultError("保险箱已锁定")


def _save(path: Path) -> None:
    _write(Path(path), _state.key, _state.salt, _state.kdf, _state.profiles)


def _local_host(host: str) -> bool:
    if host in {"localhost", "host.docker.internal"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def _preset_of(profile: dict[str, Any]) -> str:
    """Profiles saved before presets existed are matched by kind and address."""
    if profile.get("preset") in PRESETS:
        return profile["preset"]
    if profile.get("kind") == "ollama":
        return "ollama"
    return next((key for key, p in PRESETS.items() if p["kind"] == profile.get("kind") and p["base_url"]
                 and p["base_url"] == profile.get("base_url")), "custom")


def _clean(raw: dict[str, Any], existing: dict[str, Any] | None, *, auto_model: bool = False) -> dict[str, Any]:
    """Validate one profile. With auto_model an empty model is allowed, for discovery to fill in."""
    preset_id = raw.get("preset")
    if preset_id is None:  # Older clients send kind/address only.
        kind = raw.get("kind") if raw.get("kind") in KINDS else None
        if kind is None:
            raise VaultError("请选择服务商")
        preset_id = _preset_of({"kind": kind, "base_url": str(raw.get("base_url") or "").strip().rstrip("/")})
    elif preset_id not in PRESETS:
        raise VaultError("请选择服务商")
    preset = PRESETS[preset_id]
    kind = preset["kind"]
    base_url = (str(raw.get("base_url") or "").strip() or preset["base_url"]).rstrip("/")
    if not base_url:
        raise VaultError("请填写服务地址")
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise VaultError("服务地址无效")
    if parts.query or parts.fragment:
        raise VaultError("服务地址不能带查询参数")
    # A key sent over plain HTTP to the internet can be read on the way.
    if parts.scheme == "http" and not _local_host(parts.hostname):
        raise VaultError("远程服务必须使用 https")
    model = str(raw.get("model") or "").strip()
    if len(model) > 200 or (not model and not auto_model):
        raise VaultError("请填写模型名")
    protocol = raw.get("protocol") if raw.get("protocol") in PROTOCOLS else (preset["protocol"] or "chat/completions")
    supplied = raw.get("api_key")
    supplied = supplied.strip() if isinstance(supplied, str) else ""
    if len(supplied) > 4096:
        raise VaultError("API Key 过长")
    api_key = supplied
    if not api_key and existing is not None:
        # A saved key only follows its own endpoint, so it cannot be redirected elsewhere.
        if existing.get("base_url") != base_url or existing.get("kind") != kind:
            if existing.get("api_key"):
                raise VaultError("更换服务地址或类型后，需要重新填写 API Key")
        else:
            api_key = existing.get("api_key", "")
    if kind == "openai" and not api_key and not _local_host(parts.hostname):
        raise VaultError("云端服务需要 API Key")
    return {"id": existing["id"] if existing else secrets.token_hex(6), "kind": kind, "preset": preset_id,
            "label": str(raw.get("label") or "").strip()[:40], "base_url": base_url, "model": model,
            "protocol": protocol if kind == "openai" else None, "api_key": api_key,
            "enabled": bool(raw.get("enabled", True))}


def _existing(raw: dict[str, Any]) -> dict[str, Any] | None:
    profile_id = raw.get("id")
    existing = next((p for p in _state.profiles if p["id"] == profile_id), None) if profile_id else None
    if profile_id and existing is None:
        raise VaultError("找不到这个模型配置")
    if existing is None and len(_state.profiles) >= MAX_PROFILES:
        raise VaultError(f"最多保存 {MAX_PROFILES} 个模型")
    return existing


def draft_profile(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """A validated, unsaved profile whose model may still be empty.

    The result holds the key (typed now, or the saved one for an unchanged
    address) so that model discovery can use it. Keep it in memory only.
    """
    with _state.lock:
        _require(path)
        return _clean(raw, _existing(raw), auto_model=True)


def commit_profile(path: Path, profile: dict[str, Any], *, new: bool) -> str:
    """Store a profile from draft_profile once its model is known."""
    with _state.lock:
        _require(path)
        existing = next((p for p in _state.profiles if p["id"] == profile.get("id")), None)
        if not new and existing is None:
            raise VaultError("找不到这个模型配置")
        if existing is None and len(_state.profiles) >= MAX_PROFILES:
            raise VaultError(f"最多保存 {MAX_PROFILES} 个模型")
        if existing is not None and (existing["base_url"], existing["kind"]) != (profile["base_url"], profile["kind"]) \
                and not profile.get("api_key") and existing.get("api_key"):
            raise VaultError("更换服务地址或类型后，需要重新填写 API Key")
        model = str(profile.get("model") or "").strip()
        if not model or len(model) > 200:
            raise VaultError("请填写模型名")
        clean = {**profile, "model": model}
        if existing is None:
            _state.profiles.append(clean)
        else:
            _state.profiles[_state.profiles.index(existing)] = clean
        _save(path)
        return clean["id"]


def save_profile(path: Path, raw: dict[str, Any]) -> str:
    with _state.lock:
        _require(path)
        existing = _existing(raw)
        return commit_profile(path, _clean(raw, existing), new=existing is None)


def delete_profile(path: Path, profile_id: str) -> None:
    with _state.lock:
        _require(path)
        kept = [p for p in _state.profiles if p["id"] != profile_id]
        if len(kept) == len(_state.profiles):
            raise VaultError("找不到这个模型配置")
        _state.profiles = kept
        _save(path)


def reorder(path: Path, ids: Any) -> None:
    with _state.lock:
        _require(path)
        current = {p["id"]: p for p in _state.profiles}
        if not isinstance(ids, list) or sorted(map(str, ids)) != sorted(current):
            raise VaultError("排序列表与已保存的模型不一致")
        _state.profiles = [current[str(i)] for i in ids]
        _save(path)


def _label(index: int) -> str:
    return f"视觉模型 {index + 1}"


def management_view(path: Path) -> list[dict[str, Any]]:
    """Everything except the key, for the passphrase-holding client only."""
    with _state.lock:
        _require(path)
        return [{"id": p["id"], "slot": _label(i), "kind": p["kind"], "preset": _preset_of(p),
                 "provider_name": PRESETS[_preset_of(p)]["name"], "label": p["label"],
                 "base_url": p["base_url"], "model": p["model"], "protocol": p["protocol"],
                 "enabled": p["enabled"], "has_key": bool(p.get("api_key"))}
                for i, p in enumerate(_state.profiles)]


def slots(path: Path) -> list[dict[str, str]]:
    """Enabled profiles in priority order, as opaque ids and labels only."""
    with _state.lock:
        if not (_same(_state.path, path) and _state.key is not None):
            return []
        return [{"id": p["id"], "label": _label(i)} for i, p in enumerate(_state.profiles) if p.get("enabled", True)]


def resolve(profile_id: str) -> dict[str, Any] | None:
    """In-process connection details for the transport. Never serialize the result."""
    with _state.lock:
        if _state.key is None:
            return None
        profile = next((p for p in _state.profiles if p["id"] == profile_id), None)
        return copy.deepcopy(profile) if profile else None


def redaction_values() -> list[str]:
    """Strings that must never appear in audits, errors or saved model text."""
    with _state.lock:
        values: list[str] = []
        for profile in _state.profiles:
            values.extend(v for v in (profile.get("api_key"), profile.get("base_url"), profile.get("model"))
                          if isinstance(v, str) and len(v) >= 3)
            host = urlsplit(profile.get("base_url") or "").hostname
            if host and len(host) >= 3:
                values.append(host)
        return sorted(set(values), key=len, reverse=True)
