"""The encrypted vision vault: nothing about a configured model leaves it."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from superphoto import vault, vision
from superphoto.config import Settings

KEY = "sk-vault-test-0123456789abcdef"
MODEL = "secret-vision-model-v9"
BASE = "https://vision.private-provider.example/v1"
PASS = "correct horse battery"


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("A vault test attempted real HTTP")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


def _profile(**changes):
    return {"kind": "openai", "label": "主力", "base_url": BASE, "model": MODEL, "protocol": "chat/completions",
            "api_key": KEY, "enabled": True, **changes}


def _leaks(text: str) -> list[str]:
    return [value for value in (KEY, MODEL, BASE, "private-provider") if value in text]


def test_vault_file_is_ciphertext_and_needs_the_passphrase(tmp_path):
    path = vault.vault_path(tmp_path)
    vault.create(path, PASS)
    vault.save_profile(path, _profile())
    assert _leaks(path.read_text(encoding="utf-8")) == []
    vault.lock()
    assert vault.status(path) == {"exists": True, "unlocked": False, "count": None}
    assert vault.resolve(vault.slots(path)[0]["id"] if vault.slots(path) else "x") is None
    with pytest.raises(vault.VaultError, match="口令不正确"):
        vault.unlock(path, "wrong passphrase!!")
    vault.unlock(path, PASS)
    [slot] = vault.slots(path)
    assert slot["label"] == "视觉模型 1" and vault.resolve(slot["id"])["api_key"] == KEY
    with pytest.raises(vault.VaultError):
        vault.create(path, PASS)
    with pytest.raises(vault.VaultError, match="至少"):
        vault.create(tmp_path / "other.json", "short")


def test_profile_rules_protect_the_key(tmp_path):
    path = vault.vault_path(tmp_path)
    vault.create(path, PASS)
    with pytest.raises(vault.VaultError, match="https"):
        vault.save_profile(path, _profile(base_url="http://vision.example.com/v1"))
    with pytest.raises(vault.VaultError, match="API Key"):
        vault.save_profile(path, _profile(api_key=""))
    vault.save_profile(path, _profile(kind="ollama", base_url="http://192.168.1.20:11434", api_key=""))
    profile_id = vault.save_profile(path, _profile())
    # Unchanged endpoint keeps the stored key; a new endpoint must not inherit it.
    vault.save_profile(path, _profile(id=profile_id, api_key="", label="改名"))
    assert vault.resolve(profile_id)["api_key"] == KEY
    with pytest.raises(vault.VaultError, match="重新填写"):
        vault.save_profile(path, _profile(id=profile_id, api_key="", base_url="https://attacker.example/v1"))
    view = vault.management_view(path)
    assert [p["slot"] for p in view] == ["视觉模型 1", "视觉模型 2"]
    assert all("api_key" not in p for p in view) and view[1]["has_key"] is True
    vault.reorder(path, [view[1]["id"], view[0]["id"]])
    assert vault.slots(path)[0]["id"] == profile_id
    vault.change_passphrase(path, PASS, "a brand new passphrase")
    vault.lock()
    vault.unlock(path, "a brand new passphrase")
    assert len(vault.slots(path)) == 2


def _cloud(monkeypatch, handler):
    seen = []

    def client(**kwargs):
        seen.append(kwargs.copy())
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False))
    monkeypatch.setattr(vision, "OpenAI", client)
    return seen


def _answer(text):
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})


def test_vault_profiles_run_first_and_audits_name_only_the_slot(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, vision_provider="auto", ollama_enabled=False, xai_api_key="env-xai-key")
    path = vault.vault_path(tmp_path)
    vault.create(path, PASS)
    vault.save_profile(path, _profile())
    seen = _cloud(monkeypatch, lambda request: _answer(f'{{"ok": true, "note": "I am {MODEL}"}}'))

    status = vision.probe(settings)
    assert [c["provider"] for c in status["candidates"]] == ["vault", "xai"]
    assert status["provider"] == "vault" and status["model"] == "视觉模型 1" and status["fallback"] is False
    assert _leaks(json.dumps(status, ensure_ascii=False)) == []

    audit = {}
    text = vision.chat_vision(settings, status["resolved_model"], "JSON", "image", selection=status, audit=audit)
    assert seen[0]["api_key"] == KEY and seen[0]["base_url"] == BASE
    assert audit["provider"] == "vault" and audit["model"] == "视觉模型 1" and audit["usage"]["total_tokens"] == 8
    assert _leaks(json.dumps(audit, ensure_ascii=False)) == [] and _leaks(text) == []


def test_failed_vault_profile_reports_slot_and_falls_through(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, vision_provider="auto", ollama_enabled=False,
                        openai_api_key="env-generic", openai_base_url="https://generic.invalid/v1")
    path = vault.vault_path(tmp_path)
    vault.create(path, PASS)
    vault.save_profile(path, _profile())
    _cloud(monkeypatch, lambda request: httpx.Response(401, json={"error": {"message": f"bad key for {MODEL}"}})
           if request.url.host != "generic.invalid" else _answer('{"ok": true}'))
    audit = {}
    vision.chat_vision(settings, "", "JSON", "image", selection=vision.probe(settings), audit=audit)
    assert audit["provider"] == "openai" and audit["attempts"][0]["error"] == {"type": "AuthenticationError",
                                                                              "status": 401}
    assert _leaks(json.dumps(audit, ensure_ascii=False)) == []


def test_locked_vault_is_skipped(tmp_path):
    path = vault.vault_path(tmp_path)
    vault.create(path, PASS)
    vault.save_profile(path, _profile())
    vault.lock()
    status = vision.probe(Settings(data_dir=tmp_path, vision_provider="auto", ollama_enabled=False))
    assert status["provider"] == "local" and {"provider": "vault", "reason": "locked"} in status["unavailable"]
    assert status["vault"] == {"exists": True, "unlocked": False, "count": None}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from superphoto import app as appmod

    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    appmod.reset_runtime()
    yield TestClient(appmod.app)
    appmod.reset_runtime()


def test_api_never_returns_keys_and_hides_profiles_without_the_passphrase(client, monkeypatch):
    assert client.get("/api/vision/vault").json()["exists"] is False
    assert client.post("/api/vision/vault/create", json={"passphrase": PASS}).json()["manage"] is True
    saved = client.post("/api/vision/profiles", json=_profile())
    assert saved.status_code == 200 and saved.json()["profiles"][0]["model"] == MODEL
    assert KEY not in saved.text

    stranger = TestClient(client.app)
    view = stranger.get("/api/vision/vault").json()
    assert view == {"exists": True, "unlocked": True, "count": 1, "manage": False, "min_passphrase": 10,
                    "bridge": False}
    assert stranger.post("/api/vision/profiles", json=_profile(api_key="other")).status_code == 403
    assert stranger.post("/api/vision/vault/unlock", json={"passphrase": "nope nope nope"}).status_code == 400
    health = stranger.get("/api/health").text
    assert _leaks(health) == [] and "视觉模型 1" in health

    # Cross-site and form posts are refused even with the session cookie.
    assert client.post("/api/vision/profiles", json=_profile(), headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/vision/profiles", content="kind=openai",
                       headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code == 415

    profile_id = saved.json()["profiles"][0]["id"]
    _cloud(monkeypatch, lambda request: _answer('{"color": "red"}'))
    tested = client.post(f"/api/vision/profiles/{profile_id}/test", json={}).json()
    assert tested["ok"] is True and tested["sees_image"] is True and _leaks(json.dumps(tested)) == []

    locked = client.post("/api/vision/vault/lock", json={}).json()
    assert locked["unlocked"] is False and "profiles" not in locked
    assert client.post("/api/vision/profiles", json=_profile()).status_code == 403
    assert client.post("/api/vision/vault/reset", json={"confirm": "no"}).status_code == 400
    assert client.post("/api/vision/vault/reset", json={"confirm": "重置"}).json()["exists"] is False


def _provider(names, sees=(), status=None):
    """A fake OpenAI-compatible provider: lists names, and only models in sees read the test image."""
    calls = []

    def handler(request):
        calls.append((request.url.host, request.url.path, request.headers.get("authorization"),
                      json.loads(request.content)["model"] if request.content else None))
        if status:
            return httpx.Response(status, json={"error": {"message": f"denied for {KEY}"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": n, "object": "model", "created": 1, "owned_by": "p"} for n in names]})
        if calls[-1][3] in sees:
            return _answer('{"color": "red"}')
        return httpx.Response(400, json={"error": {"message": "unknown variant `image_url`"}})
    return calls, handler


def test_provider_and_key_are_enough(client, monkeypatch):
    client.post("/api/vision/vault/create", json={"passphrase": PASS})
    calls, handler = _provider(["text-embedding-v3", "qwen-max", "qwen-vl-plus", "qwen3-vl-plus", "qwen-vl-max"],
                               sees={"qwen-vl-max"})
    _cloud(monkeypatch, handler)
    saved = client.post("/api/vision/profiles", json={"preset": "dashscope", "api_key": KEY})
    assert saved.status_code == 200, saved.text
    [profile] = saved.json()["profiles"]
    assert profile["model"] == "qwen-vl-max" and profile["preset"] == "dashscope"
    assert profile["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1" and profile["has_key"] is True
    assert saved.json()["discovery"]["tried"] == 2 and KEY not in saved.text
    tried = [model for _, path, _, model in calls if path.endswith("/chat/completions")]
    assert tried[:2] == ["qwen3-vl-plus", "qwen-vl-max"] and "text-embedding-v3" not in tried
    assert {host for host, *_ in calls} == {"dashscope.aliyuncs.com"}
    assert {auth for *_, auth, _ in calls} == {f"Bearer {KEY}"}

    # Re-picking on the same address reuses the stored key; a new provider needs a new key.
    calls.clear()
    again = client.post("/api/vision/profiles", json={"id": profile["id"], "preset": "dashscope", "model": ""})
    assert again.status_code == 200 and {auth for *_, auth, _ in calls} == {f"Bearer {KEY}"}
    calls.clear()
    moved = client.post("/api/vision/profiles", json={"id": profile["id"], "preset": "deepseek"})
    assert moved.status_code == 400 and "重新填写" in moved.json()["detail"] and calls == []


def test_provider_without_image_models_is_not_saved(client, monkeypatch):
    client.post("/api/vision/vault/create", json={"passphrase": PASS})
    calls, handler = _provider(["deepseek-chat", "deepseek-reasoner"])
    _cloud(monkeypatch, handler)
    refused = client.post("/api/vision/profiles", json={"preset": "deepseek", "api_key": KEY})
    assert refused.status_code == 400 and "不支持图片输入" in refused.json()["detail"]
    assert KEY not in refused.text and "deepseek-chat" not in refused.text
    assert client.get("/api/vision/vault").json()["count"] == 0
    assert {host for host, *_ in calls} == {"api.deepseek.com"}


def test_bad_key_stops_after_one_try(client, monkeypatch):
    client.post("/api/vision/vault/create", json={"passphrase": PASS})
    calls, handler = _provider([], status=401)
    _cloud(monkeypatch, handler)
    refused = client.post("/api/vision/profiles", json={"preset": "zhipu", "api_key": KEY})
    assert refused.status_code == 400 and "API Key 无效" in refused.json()["detail"] and KEY not in refused.text
    # The listing failed, so the preset's usual model was tried once and the 401 ended the search.
    assert [path.rsplit("/", 1)[-1] for _, path, *_ in calls] == ["models", "completions"]


def test_presets_and_old_profiles(client, tmp_path):
    listed = client.get("/api/vision/presets").json()
    deepseek = next(p for p in listed if p["id"] == "deepseek")
    assert deepseek["needs_key"] is True and "prefer" not in deepseek
    path = vault.vault_path(tmp_path)
    vault.create(path, PASS)
    with pytest.raises(vault.VaultError, match="服务商"):
        vault.save_profile(path, {"preset": "nope", "api_key": KEY, "model": "m"})
    vault.save_profile(path, _profile())
    vault.save_profile(path, _profile(base_url="https://api.deepseek.com/v1"))
    vault.save_profile(path, _profile(kind="ollama", base_url="http://127.0.0.1:11434", api_key=""))
    view = vault.management_view(path)
    assert [(p["preset"], p["provider_name"]) for p in view] == [
        ("custom", "其他 OpenAI 兼容服务"), ("deepseek", "DeepSeek"), ("ollama", "本机或局域网 Ollama")]


def test_bridge_lends_the_vault_to_local_processes_without_its_contents(client, tmp_path, monkeypatch):
    from superphoto import vault_routes

    body = {"prompt": "JSON", "images": ["aW1hZ2U="]}
    assert client.post("/api/vision/bridge", json=body).status_code == 404  # Off unless enabled.
    server = Settings(data_dir=tmp_path / "data", vision_bridge=True)
    token_file = vault_routes.publish_bridge(server)
    try:
        token = token_file.read_text(encoding="ascii")
        headers = {"X-SuperPhoto-Bridge": token}
        # TestClient reports host "testclient"; pretend it is loopback for the allowed calls.
        assert client.post("/api/vision/bridge", json=body, headers=headers).status_code == 403
        monkeypatch.setattr(vault_routes, "_LOOPBACK", {"testclient"})
        assert client.post("/api/vision/bridge", json=body, headers={"X-SuperPhoto-Bridge": "x"}).status_code == 403
        client.post("/api/vision/vault/create", json={"passphrase": PASS})
        client.post("/api/vision/profiles", json=_profile())
        client.post("/api/vision/vault/lock", json={})
        locked = client.post("/api/vision/bridge", json=body, headers=headers)
        assert locked.status_code == 503 and "locked" in locked.json()["detail"]
        assert client.post("/api/vision/vault/unlock", json={"passphrase": PASS}).status_code == 200
        assert client.get("/api/vision/vault").json()["bridge"] is True

        seen = _cloud(monkeypatch, lambda request: _answer(f'{{"color": "red", "by": "{MODEL}"}}'))
        caller = Settings(data_dir=tmp_path / "elsewhere", vision_provider="auto", ollama_enabled=False,
                          vision_bridge_url="http://127.0.0.1:8765", vision_bridge_token=token)
        monkeypatch.setattr(vision.httpx, "post", lambda url, **kw: client.post(
            url.replace("http://127.0.0.1:8765", ""), json=kw["json"], headers=kw["headers"]))
        status = vision.probe(caller)
        assert status["provider"] == "vault" and status["model"] == vision.BRIDGE_LABEL
        audit = {}
        text = vision.chat_vision(caller, status["resolved_model"], "JSON", "image", selection=status, audit=audit)
        assert seen[0]["api_key"] == KEY and "[REDACTED]" in text
        assert audit["provider"] == "vault" and audit["model"] == "视觉模型 1" and audit["usage"]["total_tokens"] == 8
        assert _leaks(json.dumps(audit, ensure_ascii=False)) == [] and _leaks(text) == []
    finally:
        vault_routes.retract_bridge()
    assert not token_file.exists()
    assert client.post("/api/vision/bridge", json=body, headers=headers).status_code == 404


def test_rank_models_prefers_the_preset_and_drops_non_chat_models():
    from superphoto.vision_presets import PRESETS, rank_models

    names = ["glm-4-plus", "embedding-3", "glm-4v-plus", "glm-4.5v", "glm-4.6v", "cogview-4", "glm-4v-plus-0111"]
    assert rank_models(names, PRESETS["zhipu"]["prefer"])[:3] == ["glm-4.6v", "glm-4.5v", "glm-4v-plus"]
    assert "embedding-3" not in rank_models(names, [r"."])
