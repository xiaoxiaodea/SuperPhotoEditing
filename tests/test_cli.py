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
