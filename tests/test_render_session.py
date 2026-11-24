"""One LightCraft process per photo (lightcraft.render_session), and its fall-back to cold renders."""

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from superphoto import lightcraft as lc


class FakeMcp:
    """Stands in for `lightcraft-cli mcp`: answers initialize and command_batch as 0.21.1 does."""

    def __init__(self, behaviour: list[str]):
        self.behaviour = behaviour          # per batch: "ok", "fail" (a step fails), "die", "hang"
        self.command: list[str] = []
        self.batches: list[list[dict]] = []
        self.closed = False
        self.killed = False
        self._out: queue.Queue = queue.Queue()
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close)
        self.stdout = iter(self._out.get, None)

    def __call__(self, command, **kwargs):
        self.command = command
        return self

    def _reply(self, ident, result):
        self._out.put(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}) + "\n")

    def _write(self, text):
        if self.closed or self.killed:
            raise OSError("pipe closed")
        for line in text.splitlines():
            message = json.loads(line)
            if "id" not in message:
                continue
            if message["method"] == "initialize":
                self._reply(message["id"], {"capabilities": {}})
                continue
            steps = message["params"]["arguments"]["steps"]
            self.batches.append(steps)
            what = self.behaviour.pop(0) if self.behaviour else "ok"
            if what == "die":
                self._out.put(None)
                continue
            if what == "hang":
                continue
            results = [{"ok": True, "result": {}} for _ in steps]
            if what == "fail":
                results[-1] = {"ok": False, "error": "export failed"}
            else:
                Path(steps[-1]["params"]["path"]).write_bytes(b"session")
            body = {"completed": sum(r["ok"] for r in results), "failed": 0 if what == "ok" else 1, "results": results}
            self._reply(message["id"], {"content": [{"type": "text", "text": json.dumps(body)}]})

    def _close(self):
        self.closed = True
        self._out.put(None)

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self._out.put(None)


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """A binary that exists, a source photo, cold renders recorded, and the session process faked."""
    binary = tmp_path / "lightcraft-cli.exe"
    binary.write_bytes(b"")
    source = tmp_path / "work" / "source.jpg"
    source.parent.mkdir()
    source.write_bytes(b"jpeg")
    cold: list[list[str]] = []

    def run(command, **kwargs):
        cold.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"cold")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lc.subprocess, "run", run)
    monkeypatch.delenv("SUPERPHOTO_LIGHTCRAFT_SESSIONS", raising=False)

    def fake(*behaviour):
        process = FakeMcp(list(behaviour))
        monkeypatch.setattr(lc.subprocess, "Popen", process)
        return process

    return SimpleNamespace(binary=binary, source=source, cold=cold, fake=fake, tmp=tmp_path)


def test_renders_inside_a_session_reuse_one_process_and_apply_each_document_to_reset_settings(engine):
    process = engine.fake()
    document = {"light": {"exposure": 0.4}, "masks": [{"id": 1, "name": "sky"}]}
    with lc.render_session(engine.binary, engine.source) as session:
        assert session is not None and session.alive
        first = lc.render_document(engine.binary, engine.source, document, engine.tmp / "a", edge=960)
        full = lc.render_document(engine.binary, engine.source, {}, engine.tmp / "b", name="render.tif",
                                  options={"bitDepth": 16, "colorSpace": "srgb"})
        # Another photo has no session: it renders cold.
        other = engine.tmp / "other.jpg"
        other.write_bytes(b"jpeg")
        lc.render_document(engine.binary, other, {}, engine.tmp / "c")
    assert process.command[1:] == ["mcp", "--compact", str(engine.source.resolve())]
    assert [[step["id"] for step in batch] for batch in process.batches] == [
        ["develop.reset", "develop.merge", "app.export"]] * 2
    assert process.batches[0][1]["params"]["settings"] == document
    assert process.batches[0][2]["params"] == {"path": str(first.resolve()), "longEdge": 960,
                                               "conflict": "overwrite", "quality": 92}
    # Full size is longEdge 0 (the engine's default is 3000); a TIFF has no quality.
    assert process.batches[1][2]["params"] == {"path": str(full.resolve()), "longEdge": 0, "conflict": "overwrite",
                                               "bitDepth": 16, "colorSpace": "srgb"}
    assert first.read_bytes() == b"session" and full.read_bytes() == b"session"
    assert json.loads((engine.tmp / "a" / "develop.json").read_text(encoding="utf-8")) == document
    assert len(engine.cold) == 1 and str(other) in engine.cold[0]
    # The block closed the process; afterwards renders are cold again.
    assert process.closed and lc._SESSIONS == {}
    assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "d").read_bytes() == b"cold"


def test_a_failed_session_render_is_redone_cold_and_a_dead_session_is_not_asked_again(engine):
    process = engine.fake("fail", "ok", "die")
    with lc.render_session(engine.binary, engine.source) as session:
        # A step fails: that render is redone cold; the process still answers, so it is kept.
        assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "a").read_bytes() == b"cold"
        assert session.alive
        assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "b").read_bytes() == b"session"
        # The process exits: redone cold, and the rest of the block renders cold without it.
        assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "c").read_bytes() == b"cold"
        assert not session.alive
        assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "d").read_bytes() == b"cold"
    assert len(process.batches) == 3 and len(engine.cold) == 3


def test_a_session_that_stops_answering_is_stopped_and_the_render_reports_a_timeout(engine):
    process = engine.fake("hang")
    with lc.render_session(engine.binary, engine.source) as session:
        with pytest.raises(lc.LightCraftError, match="超时"):
            lc.render_document(engine.binary, engine.source, {}, engine.tmp / "a", timeout=0.2)
        assert process.killed and not session.alive
        assert engine.cold == []                       # not tried again cold: it would only wait as long
        assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "b").read_bytes() == b"cold"


def test_no_session_when_turned_off_unavailable_or_already_open(engine, monkeypatch):
    process = engine.fake()
    monkeypatch.setenv("SUPERPHOTO_LIGHTCRAFT_SESSIONS", "0")
    with lc.render_session(engine.binary, engine.source) as session:
        assert session is None
    assert process.command == []
    monkeypatch.delenv("SUPERPHOTO_LIGHTCRAFT_SESSIONS")
    with lc.render_session(engine.tmp / "missing-cli.exe", engine.source) as session:
        assert session is None

    def cannot_start(*args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(lc.subprocess, "Popen", cannot_start)
    with lc.render_session(engine.binary, engine.source) as session:
        assert session is None
        assert lc.render_document(engine.binary, engine.source, {}, engine.tmp / "a").read_bytes() == b"cold"
    # Nested blocks for one photo share the outer session, and only the outer one closes it.
    process = engine.fake()
    with lc.render_session(engine.binary, engine.source) as outer:
        with lc.render_session(engine.binary, engine.tmp / "work" / ".." / "work" / "source.jpg") as inner:
            assert inner is outer
        assert not process.closed and outer.alive
    assert process.closed


@pytest.mark.skipif(lc.discover_cli() is None, reason="needs the LightCraft CLI")
def test_session_renders_match_cold_renders_pixel_for_pixel(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPERPHOTO_LIGHTCRAFT_SESSIONS", raising=False)
    binary = lc.discover_cli()
    y, x = np.mgrid[0:400, 0:600]
    picture = np.stack([x * 255 / 600, y * 255 / 400, 128 + 100 * np.sin(x / 23.0) * np.cos(y / 17.0)], -1)
    source = tmp_path / "source.jpg"
    Image.fromarray(picture.clip(0, 255).astype(np.uint8)).save(source, quality=95)
    documents = [
        {"light": {"exposure": 0.6, "contrast": 25, "shadows": 40}, "color": {"vibrance": 30}},
        {"wb": {"mode": "custom", "temp": 7200, "tint": 12}, "effects": {"clarity": 30},
         "grading": {"shadows": {"hue": 210, "sat": 20}, "highlights": {"hue": 40, "sat": 15}}},
        {"curve": {"master": [{"x": 0, "y": 0.05}, {"x": 0.5, "y": 0.45}, {"x": 1, "y": 1}]},
         "crop": {"geometry": {"rect": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.85}, "angle": 2.0}}},
        {},
    ]
    with lc.render_session(binary, source) as session:
        assert session is not None
        warm = [lc.render_document(binary, source, doc, tmp_path / f"s{i}", edge=300)
                for i, doc in enumerate(documents * 2)]           # each twice, interleaved: reset is complete
        assert session.alive
    for i, doc in enumerate(documents):
        cold = lc.render_document(binary, source, doc, tmp_path / f"c{i}", edge=300)
        expected = np.asarray(Image.open(cold).convert("RGB"))
        for render in (warm[i], warm[i + len(documents)]):
            assert np.array_equal(np.asarray(Image.open(render).convert("RGB")), expected), (i, render)
