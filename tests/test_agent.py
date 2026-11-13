from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from superphoto.agent import AgentError, SuperPhotoClient, file_stem
from superphoto.cli import main


def test_file_stem_keeps_a_short_readable_name() -> None:
    assert file_stem("A", "自然精修") == "A-自然精修"
    assert file_stem("B", "a/b:c") == "B-a b c"
    assert file_stem("C", "   ") == "C-C"
    assert len(file_stem("A", "很" * 50)) < 50


def _client(handler, tmp_path: Path) -> SuperPhotoClient:
    return SuperPhotoClient(
        "http://127.0.0.1:8765",
        httpx.Client(base_url="http://127.0.0.1:8765", transport=httpx.MockTransport(handler), trust_env=False),
        out_dir=tmp_path / "out",
        poll_seconds=0,
    )


def test_status_when_the_server_is_down(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler, tmp_path)
    with pytest.raises(AgentError) as caught:
        client.status()
    assert "打开 SuperPhoto" in caught.value.payload["error"]


def test_status_rejects_another_program_on_the_port(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "text/html;charset=utf-8"}, text="<!DOCTYPE HTML><html></html>")

    with pytest.raises(AgentError) as caught:
        _client(handler, tmp_path).status()
    assert "不是 SuperPhoto" in caught.value.payload["error"]


def test_status_reports_lightroom_without_the_critic_dump(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "ok": True,
            "critic": "ollama",
            "critic_model": "qwen3-vl:32b",
            "lightroom": {"connected": False, "available": True},
            "ollama": {"ready": True, "raw": "x" * 1000},
        })

    payload = _client(handler, tmp_path).status()
    assert payload["ok"] is True
    assert payload["critic"] == "ollama"
    assert payload["lightroom"]["connected"] is False
    assert "hint" in payload
    assert "ollama" not in payload


def test_edit_downloads_only_the_delivered_finishes(tmp_path: Path) -> None:
    photo = tmp_path / "boat.jpg"
    photo.write_bytes(b"original")
    polls = {"n": 0}
    steps: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/jobs":
            assert b"boat.jpg" in request.content
            return httpx.Response(200, json={"id": "job1", "kind": "jpeg"})
        if request.url.path == "/api/jobs/job1":
            polls["n"] += 1
            if polls["n"] == 1:
                return httpx.Response(200, json={"id": "job1", "status": "running", "step": "critique", "variants": []})
            return httpx.Response(200, json={
                "id": "job1",
                "status": "ready",
                "step": "ready",
                "variants": [{
                    "slot": "A",
                    "title": "自然精修",
                    "why": "保留现场的光。",
                    "url": "/media/job1/variant_A.jpg",
                    "recipe": {"secret": True},
                }],
                "ratings": {},
                "delivery": {"requested": 3, "delivered": 1, "omitted": [
                    {"slot": "B", "reason": "未通过"},
                    {"slot": "C", "reason": "未通过"},
                ]},
            })
        if request.url.path == "/media/job1/variant_A.jpg":
            return httpx.Response(200, content=b"\xff\xd8finished")
        return httpx.Response(404, json={"detail": "not found"})

    payload = _client(handler, tmp_path).edit(photo, timeout=5, on_step=steps.append)
    assert payload["ok"] is True
    assert payload["job_id"] == "job1"
    assert payload["variants"] == [{
        "slot": "A",
        "title": "自然精修",
        "why": "保留现场的光。",
        "path": str(tmp_path / "out" / "job1" / "A-自然精修.jpg"),
    }]
    assert Path(payload["variants"][0]["path"]).read_bytes() == b"\xff\xd8finished"
    assert "B" in payload["note"] and "C" in payload["note"]
    assert "recipe" not in json.dumps(payload)
    assert steps == ["critique", "ready"]


def test_edit_keeps_the_job_id_when_the_server_drops(tmp_path: Path) -> None:
    photo = tmp_path / "boat.jpg"
    photo.write_bytes(b"original")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"id": "job9"})
        raise httpx.ConnectError("gone")

    with pytest.raises(AgentError) as caught:
        _client(handler, tmp_path).edit(photo, timeout=5)
    assert caught.value.payload["job_id"] == "job9"
    assert caught.value.payload["command"] == "edit"


def test_edit_timeout_names_the_job(tmp_path: Path) -> None:
    photo = tmp_path / "boat.jpg"
    photo.write_bytes(b"original")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job2"})
        return httpx.Response(200, json={"id": "job2", "status": "running", "step": "render", "variants": []})

    with pytest.raises(AgentError) as caught:
        _client(handler, tmp_path).edit(photo, timeout=0)
    assert caught.value.payload["job_id"] == "job2"
    assert caught.value.payload["step"] == "render"


def test_rate_clear_sends_null(tmp_path: Path) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "ratings": {"A": None}, "summary": "还没有偏好", "learnable": True})

    payload = _client(handler, tmp_path).rate("job1", "A", "clear")
    assert seen["body"] == {"verdict": None}
    assert payload["verdict"] is None
    assert payload["effect"]


def test_refine_downloads_the_new_version(tmp_path: Path) -> None:
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["slot"] == "C"
            assert body["prompt"] == "整体亮一点"
            assert body["parent_id"] == "11111111-1111-1111-1111-111111111111"
            return httpx.Response(202, json={"id": "22222222-2222-2222-2222-222222222222", "status": "queued"})
        polls["n"] += 1
        if request.url.path.endswith("/image"):
            return httpx.Response(200, content=b"brighter")
        if polls["n"] == 1:
            return httpx.Response(200, json={
                "id": "22222222-2222-2222-2222-222222222222",
                "status": "running",
                "step": "render",
            })
        return httpx.Response(200, json={
            "id": "22222222-2222-2222-2222-222222222222",
            "slot": "C",
            "parent_id": "11111111-1111-1111-1111-111111111111",
            "prompt": "整体亮一点",
            "status": "ready",
            "step": "ready",
            "summary": "曝光加了一点。",
            "url": "/api/jobs/job1/refinements/22222222-2222-2222-2222-222222222222/image",
        })

    payload = _client(handler, tmp_path).refine(
        "job1",
        "C",
        "整体亮一点",
        "11111111-1111-1111-1111-111111111111",
        timeout=5,
    )
    assert payload["ok"] is True
    assert payload["summary"] == "曝光加了一点。"
    assert Path(payload["path"]).read_bytes() == b"brighter"


def test_refine_image_bytes(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "abcdef1234567890", "status": "queued", "slot": "A", "prompt": "亮一点"})
        if request.url.path.endswith("/image"):
            return httpx.Response(200, content=b"jpeg-bytes")
        return httpx.Response(200, json={
            "id": "abcdef1234567890",
            "slot": "A",
            "prompt": "亮一点",
            "status": "ready",
            "step": "ready",
            "summary": "",
            "url": "/api/jobs/job1/refinements/abcdef1234567890/image",
        })

    payload = _client(handler, tmp_path).refine("job1", "A", "亮一点", None, timeout=5)
    assert Path(payload["path"]).read_bytes() == b"jpeg-bytes"
    assert payload["path"].endswith("abcdef12.jpg")


def test_http_detail_is_the_message(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "这张图已有微调正在处理，请等它完成后再提交"})

    with pytest.raises(AgentError) as caught:
        _client(handler, tmp_path).refine("job1", "A", "亮一点", None, timeout=5)
    assert "已有微调" in caught.value.payload["error"]


def test_agent_status_cli_prints_json(tmp_path: Path, monkeypatch, capsys) -> None:
    class Fake:
        def status(self):
            return {"ok": True, "running": True}

        def close(self):
            return None

    monkeypatch.setattr("superphoto.agent.default_client", lambda: Fake())
    main(["agent", "status"])
    assert json.loads(capsys.readouterr().out)["running"] is True


def test_agent_status_cli_exits_when_stopped(monkeypatch, capsys) -> None:
    class Fake:
        def status(self):
            raise AgentError("SuperPhoto 没有在运行。先打开 SuperPhoto，再让我修这张图。")

        def close(self):
            return None

    monkeypatch.setattr("superphoto.agent.default_client", lambda: Fake())
    with pytest.raises(SystemExit) as caught:
        main(["agent", "status"])
    assert caught.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
