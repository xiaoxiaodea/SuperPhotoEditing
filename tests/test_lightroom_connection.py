from pathlib import Path
from types import SimpleNamespace

import pytest

from superphoto import lightroom as lr


def test_connected_session_is_not_interrupted(monkeypatch):
    client = SimpleNamespace(status=lambda: {"connected": True})
    monkeypatch.setattr(lr, "get_lightroom_client", lambda _: client)
    monkeypatch.setattr(lr, "wake_lightroom_plugin", lambda: pytest.fail("already connected"))
    monkeypatch.setattr(lr, "reconnect_lightroom_client", lambda _: pytest.fail("already connected"))
    assert lr.connect_lightroom() == (client, {"connected": True})


@pytest.mark.parametrize("needs_bridge_restart", [False, True])
def test_stopped_plugin_is_woken_before_replacing_bridge(monkeypatch, needs_bridge_restart):
    client = SimpleNamespace(status=lambda: {"connected": False})
    replacement = object()
    calls = []
    monkeypatch.setattr(lr, "get_lightroom_client", lambda _: client)
    monkeypatch.setattr(lr, "wake_lightroom_plugin", lambda: calls.append("wake"))
    def wait(active, *, timeout):
        calls.append(("wait", active, timeout))
        return {"connected": not needs_bridge_restart or active is replacement}
    monkeypatch.setattr(lr, "wait_for_lightroom_connection", wait)
    def reconnect(_):
        calls.append("replace")
        return replacement
    monkeypatch.setattr(lr, "reconnect_lightroom_client", reconnect)
    result, status = lr.connect_lightroom()
    assert status["connected"]
    assert calls[:2] == ["wake", ("wait", client, 30.0)]
    if needs_bridge_restart:
        assert result is replacement
        assert calls[2:] == ["replace", "wake", ("wait", replacement, 12.0)]
    else:
        assert result is client and len(calls) == 2


def test_connection_failure_is_actionable_and_preserves_diagnostic(monkeypatch):
    client = SimpleNamespace(status=lambda: {"connected": False})
    monkeypatch.setattr(lr, "get_lightroom_client", lambda _: client)
    monkeypatch.setattr(lr, "reconnect_lightroom_client", lambda _: client)
    monkeypatch.setattr(lr, "wake_lightroom_plugin", lambda: None)
    monkeypatch.setattr(lr, "wait_for_lightroom_connection", lambda *a, **kw: {
        "connected": False, "error": "Lightroom plugin not connected"})
    with pytest.raises(lr.LightroomError, match="重试这张照片") as failure:
        lr.connect_lightroom()
    assert str(failure.value.__cause__) == "Lightroom plugin not connected"


@pytest.mark.parametrize("has_handler", [False, True])
def test_wake_uses_only_installed_plugin_url(tmp_path, monkeypatch, has_handler):
    monkeypatch.setattr(lr.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    info = tmp_path / "Library/Application Support/Adobe/Lightroom/Modules/LightroomMCP.lrplugin/Info.lua"
    info.parent.mkdir(parents=True)
    info.write_text("URLHandler = 'PluginURLHandler.lua'" if has_handler else "return {}")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(lr.subprocess, "run", run)
    lr.wake_lightroom_plugin()
    expected = ["open", "-a", "Adobe Lightroom Classic"]
    if has_handler:
        expected.append("lightroom://com.lightroom.mcp/start-server")
    assert calls == [expected]
