from pathlib import Path
from types import SimpleNamespace

import pytest

from superphoto import lightroom as lr


def test_connected_session_is_not_interrupted(monkeypatch):
    client = SimpleNamespace(status=lambda: {"connected": True, "engine": "lightcraft"})
    monkeypatch.setattr(lr, "get_lightroom_client", lambda _: client)
    monkeypatch.setattr(lr, "reconnect_lightroom_client", lambda _: pytest.fail("already connected"))
    assert lr.connect_lightroom() == (client, {"connected": True, "engine": "lightcraft"})


def test_unresponsive_cli_is_actionable_without_restarting_anything(monkeypatch):
    client = SimpleNamespace(status=lambda: {"connected": False, "error": "lightcraft-cli did not respond"})
    monkeypatch.setattr(lr, "get_lightroom_client", lambda _: client)
    monkeypatch.setattr(lr, "reconnect_lightroom_client", lambda _: pytest.fail("no process to restart"))
    with pytest.raises(lr.LightroomError, match="install-lightcraft") as failure:
        lr.connect_lightroom()
    assert str(failure.value.__cause__) == "lightcraft-cli did not respond"


def test_missing_binary_names_the_install_command(monkeypatch):
    def missing(_=None):
        raise lr.LightroomError("未找到 lightcraft-cli。请运行 uv run superphoto install-lightcraft。")

    monkeypatch.setattr(lr, "get_lightroom_client", missing)
    with pytest.raises(lr.LightroomError, match="install-lightcraft"):
        lr.connect_lightroom()


def test_status_reports_an_executable_cli_without_opening_an_app(tmp_path, monkeypatch):
    cli = tmp_path / "lightcraft-cli"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setattr(lr, "_client", None)
    monkeypatch.setattr("superphoto.lightcraft._version", lambda path: "lightcraft-cli 0.21.1")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("SUPERPHOTO_LIGHTCRAFT_BIN", raising=False)
    monkeypatch.delenv("SUPERPHOTO_ROOT", raising=False)

    status = lr.lightroom_status(True, cli)
    assert status["available"] is True
    assert status["connected"] is True
    assert status["engine"] == "lightcraft"
    assert status["binary"] == str(cli.resolve())

    disabled = lr.lightroom_status(False, cli)
    assert disabled["enabled"] is False
    assert disabled["available"] is True
    assert disabled["connected"] is False


def test_status_without_a_binary_asks_for_install(monkeypatch, tmp_path):
    monkeypatch.setattr(lr, "_client", None)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("SUPERPHOTO_LIGHTCRAFT_BIN", raising=False)
    monkeypatch.delenv("SUPERPHOTO_ROOT", raising=False)
    monkeypatch.setattr(lr, "discover_runtime", lambda *_: None)
    from superphoto import lightcraft as lc
    monkeypatch.setattr(lc, "discover_cli", lambda *_: None)
    status = lr.lightroom_status(True, None)
    assert status["available"] is False
    assert status["connected"] is False
    assert "install-lightcraft" in status["error"]
