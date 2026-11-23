"""Exercise the real SDK over an in-memory HTTP transport, never real credentials."""
import base64
import json

import httpx
import pytest
from openai import OpenAI

from superphoto import vision
from superphoto.config import Settings


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("A vision test attempted real HTTP")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


def completion(text='{"ok":true}', *, finish="stop", **changes):
    return {"id": "chat-test", "object": "chat.completion", "created": 1, "model": "actual-model",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}, **changes}


def response(text='{"ok":true}', *, status="completed", **changes):
    return {"id": "resp-test", "object": "response", "created_at": 1, "model": "actual-model",
            "status": status, "output": [{"id": "msg-test", "type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 21, "output_tokens": 9, "total_tokens": 30}, **changes}


@pytest.fixture
def cloud(monkeypatch):
    calls = []
    options = []

    def install(handler):
        def capture(request):
            calls.append((request.url.path, json.loads(request.content)))
            return handler(request)

        def client(**kwargs):
            options.append(kwargs.copy())
            return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(capture), trust_env=False))
        monkeypatch.setattr(vision, "OpenAI", client)
        return calls, options
    return install


def configured(provider="openai", protocol="chat/completions", **kwargs):
    values = {"vision_provider": provider, "ollama_enabled": True, "vision_max_retries": 0,
              f"{provider}_api_key": "mock-api-secret", f"{provider}_base_url": "https://vision.invalid/v1",
              f"{provider}_model": "requested-model", f"{provider}_protocol": protocol}
    return Settings(**(values | kwargs))


@pytest.mark.parametrize("provider", ["openai", "xai"])
@pytest.mark.parametrize("protocol", ["chat/completions", "responses"])
def test_protocols_multimage_usage_and_explicit_cloud_bypasses_ollama(cloud, monkeypatch, provider, protocol):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("explicit cloud probed Ollama"))
    calls, options = cloud(lambda _: httpx.Response(200, json=(response() if protocol == "responses" else completion())))
    settings = configured(provider, protocol)
    status = vision.probe(settings)
    audit = {}
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    images = [png, "second-image"]
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    assert vision.chat_vision(settings, "requested-model", "compare", images,
                              response_schema=schema, audit=audit, selection=status) == '{"ok":true}'
    path, sent = calls[0]
    assert path == "/v1/" + protocol
    content = sent["input"][0]["content"] if protocol == "responses" else sent["messages"][0]["content"]
    assert len(content) == 3
    if protocol == "responses":
        assert sent["store"] is False
        assert sent["text"]["format"]["schema"] == schema
        assert [part["image_url"] for part in content[1:]] == [f"data:image/png;base64,{png}", "data:image/jpeg;base64,second-image"]
        assert audit["usage"] == {"input_tokens": 21, "output_tokens": 9, "total_tokens": 30}
    else:
        assert sent["response_format"]["json_schema"]["schema"] == schema
        assert [part["image_url"]["url"] for part in content[1:]] == [f"data:image/png;base64,{png}", "data:image/jpeg;base64,second-image"]
        assert audit["usage"] == {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}
    assert audit["provider"] == provider and audit["model"] == "actual-model"
    assert audit["fallback"] is False
    assert options[0]["timeout"] == 90 and options[0]["max_retries"] == 0
    assert "mock-api-secret" not in json.dumps(audit)
    assert "vision.invalid" not in json.dumps(audit)
    assert "mock-api-secret" not in repr(settings)


@pytest.mark.parametrize("protocol", ["chat/completions", "responses"])
def test_schema_rejection_negotiates_only_output_format_and_keeps_contract(cloud, protocol):
    def handler(request):
        payload = json.loads(request.content)
        format = payload.get("response_format") or payload.get("text", {}).get("format")
        if format:
            return httpx.Response(400, json={"error": {"message": "unsupported response_format schema", "type": "invalid_request_error"}})
        return httpx.Response(200, json=response() if protocol == "responses" else completion())
    calls, _ = cloud(handler)
    audit = {}
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    vision.chat_vision(configured(protocol=protocol), "requested-model", "compare", ["a", "b"],
                       response_schema=schema, audit=audit)
    assert len(calls) == 3
    assert {path for path, _ in calls} == {"/v1/" + protocol}
    assert [row["mode"] for row in audit["attempts"][0]["format_attempts"]] == ["json_schema", "json_object", "prompt"]
    last = calls[-1][1]
    content = (last.get("input") or last.get("messages"))[0]["content"]
    assert json.dumps(schema) in content[0]["text"]
    assert len(content) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_failures_never_switch_protocol_or_fall_to_ollama_when_cloud_explicit(cloud, monkeypatch, caplog, status):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("cloud failure probed Ollama"))
    calls, _ = cloud(lambda _: httpx.Response(status, json={"error": {"message": "mock-api-secret at https://user:password@vision.invalid", "type": "bad"}}))
    audit = {}
    with pytest.raises(vision.VisionUnavailable) as error:
        vision.chat_vision(configured(), "requested-model", "JSON", "image", response_schema={"type": "object"}, audit=audit)
    assert len(calls) == 1
    assert audit["attempts"][0]["error"]["status"] == status
    assert audit["fallback"] is True
    for secret in ("mock-api-secret", "password", "vision.invalid"):
        assert secret not in json.dumps(audit) + str(error.value) + caplog.text


@pytest.mark.parametrize("protocol", ["chat/completions", "responses"])
@pytest.mark.parametrize("kind", ["truncated", "empty", "refusal"])
def test_incomplete_empty_and_refused_results_cannot_become_judgments(cloud, protocol, kind):
    data = response() if protocol == "responses" else completion()
    if protocol == "responses":
        if kind == "truncated":
            data["status"] = "incomplete"
        elif kind == "empty":
            data["output"] = []
        else:
            data["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
    else:
        if kind == "truncated":
            data["choices"][0]["finish_reason"] = "length"
        elif kind == "empty":
            data["choices"][0]["message"]["content"] = ""
        else:
            data["choices"][0]["message"]["refusal"] = "no"
    calls, _ = cloud(lambda _: httpx.Response(200, json=data))
    audit = {}
    with pytest.raises(vision.VisionUnavailable):
        vision.chat_vision(configured(protocol=protocol), "requested-model", "JSON", "image", audit=audit)
    assert len(calls) == 1
    assert audit["attempts"][0]["error"]["type"] == "ValueError"
    assert audit["attempts"][0]["usage"]["total_tokens"] == 30


def test_timeout_retries_are_bounded_and_sanitized(cloud, monkeypatch):
    monkeypatch.setattr("openai._base_client.time.sleep", lambda _: None)
    def timeout(request):
        raise httpx.ReadTimeout("mock-api-secret", request=request)
    calls, options = cloud(timeout)
    audit = {}
    with pytest.raises(vision.VisionUnavailable):
        vision.chat_vision(configured(vision_max_retries=1), "requested-model", "JSON", "image", audit=audit)
    assert len(calls) == 2 and options[0]["max_retries"] == 1
    assert "mock-api-secret" not in json.dumps(audit)


def test_auto_tries_ollama_then_generic_cloud_then_temporary_xai(cloud, monkeypatch):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: {"ready": True, "resolved_model": "local-vl"})
    def broken_local(*args, **kwargs):
        raise RuntimeError("secret local URL")
    monkeypatch.setattr(vision.ollama, "chat_vision", broken_local)
    calls, _ = cloud(lambda request: httpx.Response(503, json={"error": {"message": "offline"}})
                     if request.url.host == "generic.invalid" else httpx.Response(200, json=completion()))
    settings = Settings(vision_provider="auto", ollama_enabled=True, xai_api_key="fake-xai", openai_api_key="fake-openai",
                        xai_base_url="https://xai.invalid/v1", openai_base_url="https://generic.invalid/v1",
                        xai_protocol="chat/completions", vision_max_retries=0)
    audit = {}
    assert vision.chat_vision(settings, "local-vl", "JSON", "image", audit=audit) == '{"ok":true}'
    assert [attempt["provider"] for attempt in audit["attempts"]] == ["ollama", "openai", "xai"]
    assert audit["provider"] == "xai" and audit["fallback"] is True and audit["temporary_provider"] is True
    assert "secret local URL" not in json.dumps(audit)
    assert len(calls) == 2


def test_xai_is_selected_only_as_a_marked_temporary_stand_in(monkeypatch):
    settings = Settings(vision_provider="auto", ollama_enabled=True, xai_api_key="fake-xai", openai_api_key="")
    monkeypatch.setattr(vision.ollama, "probe", lambda _: {"ready": False})
    status = vision.probe(settings)
    assert status["provider"] == "xai" and status["ready"] is True and status["temporary"] is True
    monkeypatch.setattr(vision.ollama, "probe", lambda _: {"ready": True, "resolved_model": "local-vl"})
    status = vision.probe(settings)
    assert status["provider"] == "ollama" and status["temporary"] is False
    generic = settings.model_copy(update={"openai_api_key": "fake-generic", "ollama_enabled": False})
    status = vision.probe(generic)
    assert status["provider"] == "openai" and status["temporary"] is False
    assert [c["provider"] for c in status["candidates"]] == ["openai", "xai"]


def test_status_is_secret_free_and_does_not_contact_cloud_or_local_by_default(monkeypatch):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("unexpected local probe"))
    monkeypatch.setattr(vision, "OpenAI", lambda **_: pytest.fail("status contacted cloud"))
    settings = configured()
    status = vision.get_vision_status(settings)
    assert status["provider"] == "openai" and status["ready"] is True
    assert status["availability"] == "configured" and status["cloud_connectivity_checked"] is False
    assert "mock-api-secret" not in json.dumps(status)
    status = vision.get_vision_status(settings.model_copy(update={"vision_provider": "auto"}))
    assert status["provider"] == "ollama" and status["ready"] is None
    assert status["availability"] == "unverified"
    status = vision.get_vision_status(settings.model_copy(update={"vision_provider": "local"}), check_ollama=True)
    assert status["provider"] == "local" and status["ready"] is False


def test_provider_settings_aliases_validation_and_legacy_defaults(monkeypatch):
    monkeypatch.setenv("SUPERPHOTO_VISION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-generic")
    monkeypatch.setenv("SUPERPHOTO_OPENAI_API_KEY", "fake-specific")
    monkeypatch.setenv("SUPERPHOTO_OPENAI_PROTOCOL", "responses")
    settings = Settings()
    assert settings.vision_provider == "openai" and settings.openai_api_key == "fake-specific"
    assert settings.openai_protocol == "responses"
    assert Settings(vision_provider="auto").vision_provider == "auto"
    assert Settings(xai_api_key="fake-explicit").xai_api_key == "fake-explicit"
    with pytest.raises(ValueError):
        Settings(openai_protocol="identical-everywhere")
    with pytest.raises(ValueError):
        Settings(vision_max_retries=99)


def test_auto_ollama_success_is_not_labeled_fallback_because_unused_cloud_is_unconfigured(monkeypatch):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: {"ready": True, "resolved_model": "local-vl"})
    def local_chat(*args, audit, **kwargs):
        audit.update(response_metadata={"prompt_eval_count": 42, "eval_count": 7})
        return '{"ok":true}'
    monkeypatch.setattr(vision.ollama, "chat_vision", local_chat)
    audit = {}
    settings = Settings(ollama_enabled=True)
    assert vision.chat_vision(settings, "local-vl", "JSON", "image", audit=audit) == '{"ok":true}'
    assert audit["provider"] == "ollama" and audit["model"] == "local-vl"
    assert audit["usage"] == {"prompt_eval_count": 42, "eval_count": 7}
    assert audit["fallback"] is False
    assert audit["unavailable"]  # Later optional clouds are not configured.


@pytest.mark.parametrize("provider", ["openai", "xai"])
def test_missing_explicit_cloud_credentials_do_not_enable_another_provider(monkeypatch, provider):
    monkeypatch.setattr(vision.ollama, "probe", lambda _: pytest.fail("missing cloud key probed Ollama"))
    monkeypatch.setattr(vision, "OpenAI", lambda **kwargs: pytest.fail("missing key contacted another cloud"))
    other = "xai" if provider == "openai" else "openai"
    settings = Settings(vision_provider=provider, ollama_enabled=True, **{f"{other}_api_key": "fake-other"})
    status = vision.probe(settings)
    assert status["ready"] is False and status["provider"] == "local"
    assert status["unavailable"] == [{"provider": provider, "reason": "not_configured"}]
    with pytest.raises(vision.VisionUnavailable):
        vision.chat_vision(settings, "unused", "JSON", "image", selection=status)


def test_explicit_schema_mode_does_not_silently_drop_schema(cloud):
    calls, _ = cloud(lambda _: httpx.Response(400, json={"error": {"message": "unsupported json_schema"}}))
    audit = {}
    with pytest.raises(vision.VisionUnavailable):
        vision.chat_vision(configured(vision_schema_mode="json_schema"), "requested-model", "JSON", "image",
                           response_schema={"type": "object"}, audit=audit)
    assert len(calls) == 1
    assert len(audit["attempts"][0]["format_attempts"]) == 1


def test_credentials_echoed_in_success_metadata_and_text_are_redacted(cloud):
    cloud(lambda _: httpx.Response(200, json=completion('{"note":"mock-api-secret"}', model="mock-api-secret")))
    audit = {}
    result = vision.chat_vision(configured(), "requested-model", "JSON", "image", audit=audit)
    assert "mock-api-secret" not in result + json.dumps(audit)
