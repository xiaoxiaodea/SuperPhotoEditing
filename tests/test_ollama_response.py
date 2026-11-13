from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json

import httpx
from PIL import Image
import pytest

from superphoto.config import Settings
from superphoto import ollama


def _image(color: str, size: tuple[int, int]) -> tuple[str, bytes]:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    raw = stream.getvalue()
    return base64.b64encode(raw).decode("ascii"), raw


def _response(monkeypatch, data: dict, *, status: int = 200) -> list[dict]:
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(status, json=data, request=httpx.Request("POST", url))

    monkeypatch.setattr(ollama.httpx, "post", post)
    return sent


def test_audit_preserves_thinking_response_and_ordered_decoded_images(monkeypatch):
    first, first_bytes = _image("red", (7, 3))
    second, second_bytes = _image("blue", (4, 9))
    answer = '{"darker_blue_sky":"2","note":"有明确差异"}'
    data = {
        "model": "qwen3-vl:32b", "created_at": "2026-09-26T07:47:05Z",
        "message": {"role": "assistant", "content": "", "thinking": answer},
        "done": True, "done_reason": "stop", "prompt_eval_count": 2311,
        "eval_count": 114, "total_duration": 22000,
        "additional_server_metadata": {"keep": ["every", "field"]},
    }
    sent = _response(monkeypatch, data)
    audit = {}
    schema = {"type": "object", "properties": {"darker_blue_sky": {"type": "string"}}}

    result = ollama.chat_vision(Settings(), "qwen3-vl:32b", "比较第一张和第二张", [second, first],
                                response_schema=schema, audit=audit)

    assert result == answer
    assert audit["response"] == data
    assert audit["http_status"] == 200
    assert audit["error"] is None
    assert audit["response_metadata"]["done_reason"] == "stop"
    assert audit["response_metadata"]["prompt_eval_count"] == 2311
    assert audit["response_metadata"]["eval_count"] == 114
    request = audit["request"]
    assert request["model"] == "qwen3-vl:32b"
    assert request["prompt"] == "比较第一张和第二张"
    assert request["options"] == sent[0]["options"]
    assert request["format"] == schema
    assert sent[0]["messages"][0]["images"] == [second, first]
    assert request["images"] == [
        {"index": 0, "sha256": hashlib.sha256(second_bytes).hexdigest(), "bytes": len(second_bytes),
         "width": 4, "height": 9},
        {"index": 1, "sha256": hashlib.sha256(first_bytes).hexdigest(), "bytes": len(first_bytes),
         "width": 7, "height": 3},
    ]
    assert first not in json.dumps(audit)
    assert second not in json.dumps(audit)


@pytest.mark.parametrize("completion,reason", [
    ({"done": True, "done_reason": "length"}, "done_reason=length"),
    ({"done": False, "done_reason": "stop"}, "done=false"),
])
@pytest.mark.parametrize("with_audit", [False, True])
def test_incomplete_response_is_rejected_even_when_json_is_valid(monkeypatch, completion, reason, with_audit):
    data = {"message": {"content": '{"decision":"better","confidence":0.99}'},
            "eval_count": 4096, **completion}
    _response(monkeypatch, data)
    audit = {} if with_audit else None

    with pytest.raises(ValueError, match=reason):
        ollama.chat_vision(Settings(), "test-model", "compare", "unused-mock-image", audit=audit)

    if audit is not None:
        assert audit["response"] == data
        assert audit["response_metadata"]["eval_count"] == 4096
        assert audit["error"] == {"type": "ValueError"}


def test_legacy_response_without_done_remains_compatible_and_skips_audit_work(monkeypatch):
    _response(monkeypatch, {"message": {"content": '{"ok":true}'}})

    def unexpected_audit(*args):
        pytest.fail("ordinary calls must not decode images for auditing")

    monkeypatch.setattr(ollama, "_image_audit", unexpected_audit)
    assert ollama.chat_vision(Settings(), "test-model", "compare", "old-mock-image") == '{"ok":true}'


def test_http_error_preserves_request_and_full_json_without_connection_credentials(monkeypatch):
    encoded, _ = _image("green", (5, 5))
    data = {"error": "model is unavailable", "detail": {"retryable": True}}
    _response(monkeypatch, data, status=503)
    settings = Settings(ollama_url="http://test-user:private-password@127.0.0.1:11434")
    monkeypatch.setenv("AUDIT_MUST_NOT_READ_THIS", "environment-secret")
    audit = {}

    with pytest.raises(httpx.HTTPStatusError):
        ollama.chat_vision(settings, "test-model", "inspect", encoded, audit=audit)

    assert audit["request"]["prompt"] == "inspect"
    assert audit["response"] == data
    assert audit["http_status"] == 503
    assert audit["error"] == {"type": "HTTPStatusError"}
    serialized = json.dumps(audit)
    for secret in ("test-user", "private-password", "127.0.0.1", "environment-secret"):
        assert secret not in serialized


def test_non_json_http_failure_retains_http_exception_type(monkeypatch):
    def post(url, **kwargs):
        return httpx.Response(502, text="bad gateway", request=httpx.Request("POST", url))

    monkeypatch.setattr(ollama.httpx, "post", post)
    audit = {}
    with pytest.raises(httpx.HTTPStatusError):
        ollama.chat_vision(Settings(), "test-model", "inspect", "mock-image", audit=audit)
    assert audit["http_status"] == 502
    assert audit["response"] is None
    assert audit["response_decode_error"] == {"type": "JSONDecodeError"}
    assert audit["error"] == {"type": "HTTPStatusError"}


def test_transport_failure_resets_reused_response_audit_and_excludes_error_url(monkeypatch):
    def post(url, **kwargs):
        raise httpx.ConnectError("Cannot connect to http://test-user:private-password@localhost")

    monkeypatch.setattr(ollama.httpx, "post", post)
    audit = {"response": {"stale": True}, "http_status": 200,
             "response_metadata": {"done": True}, "error": {"type": "old"}}
    with pytest.raises(httpx.ConnectError):
        ollama.chat_vision(Settings(), "test-model", "inspect", "mock-image", audit=audit)
    assert audit["request"]["model"] == "test-model"
    assert audit["response"] is None
    assert audit["response_metadata"] == {}
    assert audit["http_status"] is None
    assert audit["error"] == {"type": "ConnectError"}
    assert "private-password" not in json.dumps(audit)
