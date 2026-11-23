from __future__ import annotations

from superphoto.cli import _running_app_lightroom_status


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"ok": True, "lightroom": {"available": True, "connected": True, "selected": 2}}


def test_running_app_lightroom_status(monkeypatch) -> None:
    monkeypatch.setattr("superphoto.cli.httpx.get", lambda *args, **kwargs: _Response())
    assert _running_app_lightroom_status("127.0.0.1", 8765) == {
        "available": True,
        "connected": True,
        "selected": 2,
    }


def test_running_app_lightroom_status_falls_back(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("superphoto.cli.httpx.get", fail)
    assert _running_app_lightroom_status("127.0.0.1", 8765) is None


def test_vision_check_makes_one_real_shaped_call_and_never_reports_keys(monkeypatch, tmp_path) -> None:
    from superphoto.cli import _vision_check
    from superphoto.config import Settings
    from superphoto.vision import VisionUnavailable

    settings = Settings(data_dir=tmp_path, xai_api_key="xai-secret-key")
    selection = {"ready": True, "provider": "xai", "resolved_model": "grok-test",
                 "candidates": [{"provider": "xai", "model": "grok-test", "protocol": "responses"}], "unavailable": []}
    monkeypatch.setattr("superphoto.vision.probe", lambda _: selection)
    seen = {}

    def answer(settings_, model, prompt, image, *, selection=None, audit=None, **kwargs):
        seen.update(model=model, image=image)
        audit.update(provider="xai", model="grok-test-0901", usage={"input_tokens": 90})
        return '```json\n{"color": "Red"}\n```'

    monkeypatch.setattr("superphoto.vision.chat_vision", answer)
    report = _vision_check(settings)
    assert report["ok"] and report["sees_image"] and report["color"] == "red"
    assert report["answered_model"] == "grok-test-0901" and report["latency_s"] >= 0
    assert seen["model"] == "grok-test" and isinstance(seen["image"], str)
    assert report["temporary"] is True and "temporary stand-in" in report["note"]
    assert "xai-secret-key" not in repr(report)

    def offline(*args, **kwargs):
        raise VisionUnavailable("vision unavailable (xai:AuthenticationError)")

    monkeypatch.setattr("superphoto.vision.chat_vision", offline)
    failed = _vision_check(settings)
    assert not failed["ok"] and "AuthenticationError" in failed["error"]
    monkeypatch.setattr("superphoto.vision.probe", lambda _: {"ready": False, "provider": "local", "candidates": []})
    assert "XAI_API_KEY" in _vision_check(settings)["error"]
