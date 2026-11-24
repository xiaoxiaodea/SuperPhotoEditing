"""LightCraft AI Denoise on the delivered render of a noisy RAW (denoise.py, lightcraft.render_document)."""

import hashlib
import json
import os
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from superphoto import denoise
from superphoto import directed
from superphoto import lightcraft as lc

PRODUCT = "d0284aeed1bd5263aec989ff4d094dd1.lcdn"
DOCUMENT = {"light": {"exposure": 1.0}, "enhance": {"denoise_on": True, "denoise": 65.0}}


class FakeLibraryMcp:
    """`lightcraft-cli mcp [--library DIR] <photo>` as 0.21.1 behaves with AI Denoise on.

    Exporting a document with denoise on makes the product in the library when it is missing (a Bayer raw)
    and renders denoised; a non-Bayer file, or no library, renders plain and makes nothing.
    """

    def __init__(self, command, *, bayer: bool):
        self.command = command
        self.library = Path(command[command.index("--library") + 1]) if "--library" in command else None
        self.bayer = bayer
        self.seeded = self.library is not None and (self.library / "denoise" / PRODUCT).is_file()
        self.made = False
        self.batches: list[list[dict]] = []
        self.closed = False
        self._out: queue.Queue = queue.Queue()
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close)
        self.stdout = iter(self._out.get, None)

    def _reply(self, ident, result):
        self._out.put(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}) + "\n")

    def _write(self, text):
        for line in text.splitlines():
            message = json.loads(line)
            if "id" not in message:
                continue
            if message["method"] == "initialize":
                self._reply(message["id"], {"capabilities": {}})
                continue
            steps = message["params"]["arguments"]["steps"]
            self.batches.append(steps)
            enhance = steps[1]["params"]["settings"].get("enhance") or {}
            denoised = bool(enhance.get("denoise_on")) and self.library is not None and self.bayer
            if denoised:
                product = self.library / "denoise" / PRODUCT
                if not product.is_file():
                    product.parent.mkdir(parents=True, exist_ok=True)
                    product.write_bytes(b"denoised raw data")
                    self.made = True
            Path(steps[-1]["params"]["path"]).write_bytes(b"denoised" if denoised else b"plain")
            body = {"completed": 3, "failed": 0, "results": [{"ok": True, "result": {}}] * 3}
            self._reply(message["id"], {"content": [{"type": "text", "text": json.dumps(body)}]})

    def _close(self):
        self.closed = True
        self._out.put(None)

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._out.put(None)


@pytest.fixture
def engine(monkeypatch, tmp_path):
    binary = tmp_path / "lightcraft-cli.exe"
    binary.write_bytes(b"")
    source = tmp_path / "work" / "source.nef"
    source.parent.mkdir()
    source.write_bytes(b"raw bytes")
    started: list[FakeLibraryMcp] = []
    cold: list[list[str]] = []
    state = SimpleNamespace(bayer=True)

    def popen(command, **kwargs):
        process = FakeLibraryMcp(command, bayer=state.bayer)
        started.append(process)
        return process

    def run(command, **kwargs):
        cold.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"cold")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lc.subprocess, "Popen", popen)
    monkeypatch.setattr(lc.subprocess, "run", run)
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SUPERPHOTO_AI_DENOISE", "1")
    monkeypatch.setenv("SUPERPHOTO_AI_DENOISE_CACHE_GB", "4")
    monkeypatch.setenv("SUPERPHOTO_LIGHTCRAFT_SESSIONS", "1")
    monkeypatch.setattr(denoise, "_NOT_APPLICABLE", set())
    monkeypatch.setattr(denoise, "_KEYS", {})
    store = tmp_path / "data" / "denoise"
    return SimpleNamespace(binary=binary, source=source, started=started, cold=cold, state=state, tmp=tmp_path,
                           store=store, key=hashlib.sha1(b"raw bytes").hexdigest())


def _develop(folder: Path) -> dict:
    return json.loads((folder / "develop.json").read_text(encoding="utf-8"))


def test_the_first_denoised_render_makes_the_product_and_copies_of_the_photo_reuse_it(engine):
    out = lc.render_document(engine.binary, engine.source, DOCUMENT, engine.tmp / "a", edge=2400)
    assert out.read_bytes() == b"denoised" and _develop(engine.tmp / "a") == DOCUMENT
    first = engine.started[-1]
    library = first.library
    assert first.command[1:3] == ["mcp", "--compact"] and library.parent == engine.store / "libraries"
    assert first.made and not first.seeded and first.closed
    assert first.batches[0][2]["params"]["longEdge"] == 2400
    # The product went to the store under the file's hash; the library is gone.
    assert [p.name for p in (engine.store / "products" / engine.key).iterdir()] == [PRODUCT]
    assert not library.exists() and list((engine.store / "libraries").iterdir()) == []
    # Another slot's copy of the same file: a new library seeded with the product, nothing made again.
    copy = engine.tmp / "slot-B" / "source.nef"
    copy.parent.mkdir()
    copy.write_bytes(b"raw bytes")
    lc.render_document(engine.binary, copy, DOCUMENT, engine.tmp / "b", name="render.tif",
                       options={"bitDepth": 16})
    second = engine.started[-1]
    assert second is not first and second.seeded and not second.made
    assert second.library != library and second.batches[0][2]["params"]["bitDepth"] == 16
    assert [p.name for p in (engine.store / "products" / engine.key).iterdir()] == [PRODUCT]
    assert engine.cold == []


def test_a_file_lightcraft_cannot_denoise_keeps_its_plain_render_and_drops_the_setting(engine):
    engine.state.bayer = False                          # e.g. an iPhone ProRAW (linear DNG)
    out = lc.render_document(engine.binary, engine.source, DOCUMENT, engine.tmp / "a")
    assert out.read_bytes() == b"plain"
    assert _develop(engine.tmp / "a") == {"light": {"exposure": 1.0}}
    assert not (engine.store / "products").exists() or not any((engine.store / "products").iterdir())
    # Known now: the next render of this content does not open a library again.
    count = len(engine.started)
    assert lc.render_document(engine.binary, engine.source, DOCUMENT, engine.tmp / "b").read_bytes() == b"cold"
    assert len(engine.started) == count
    assert _develop(engine.tmp / "b") == {"light": {"exposure": 1.0}}
    assert json.loads(Path(engine.cold[-1][engine.cold[-1].index("--settings") + 1]).read_text(encoding="utf-8")) \
        == {"light": {"exposure": 1.0}}


def test_without_a_session_the_render_is_made_without_denoise_and_tried_again_next_time(engine, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(lc.subprocess, "Popen", cannot_start)
    other = {**DOCUMENT, "enhance": {"denoise_on": True, "denoise": 50.0, "raw_details": False}}
    assert lc.render_document(engine.binary, engine.source, other, engine.tmp / "a").read_bytes() == b"cold"
    assert _develop(engine.tmp / "a") == {"light": {"exposure": 1.0}, "enhance": {"raw_details": False}}
    assert denoise._NOT_APPLICABLE == set()
    assert list((engine.store / "libraries").iterdir()) == []


def test_the_store_keeps_the_most_recent_products_within_its_limit(engine, monkeypatch):
    products = engine.store / "products"
    old = time.time() - 3600
    for index, name in enumerate(("a" * 40, "b" * 40)):
        folder = products / name
        folder.mkdir(parents=True)
        (folder / "x.lcdn").write_bytes(b"x" * 1000)
        os.utime(folder, (old + index, old + index))
    stale = engine.store / "libraries" / "left-behind"
    stale.mkdir(parents=True)
    os.utime(stale, (old - 7 * 3600, old - 7 * 3600))
    monkeypatch.setenv("SUPERPHOTO_AI_DENOISE_CACHE_GB", str(1500 / 2**30))      # room for about one product
    lc.render_document(engine.binary, engine.source, DOCUMENT, engine.tmp / "a")
    assert sorted(p.name for p in products.iterdir()) == sorted(["b" * 40, engine.key])
    assert not stale.exists()


def test_decide_scales_the_amount_with_iso_and_how_far_the_darks_are_lifted(engine, monkeypatch):
    iso = {"value": 3200.0}
    monkeypatch.setattr(denoise, "iso_of", lambda source: iso["value"])
    photo = engine.source
    assert denoise.decide(photo, "jpeg", {}) is None                       # the camera denoised it already
    assert denoise.decide(photo, "raw", {}) == {"amount": 65.0, "iso": 3200.0, "stops": 0.0, "effective_iso": 3200}
    iso["value"] = 100.0
    assert denoise.decide(photo, "raw", {"light": {"exposure": 1.5, "shadows": 60}}) is None
    iso["value"] = 400.0
    lifted = {"light": {"exposure": 1.0, "shadows": 100, "highlights": -80},
              "masks": [{"visible": True, "adjust": {"amount": 50.0, "exposure": 1.0, "shadows": 40}},
                        {"visible": True, "adjust": {"amount": 100.0, "exposure": -1.0}}]}
    decision = denoise.decide(photo, "raw", lifted)
    assert decision["stops"] == pytest.approx(1.0 + 1.5 + 0.5 * (1.0 + 0.6))
    assert decision["amount"] == pytest.approx(50 + 15 * (decision["stops"] - 2), abs=0.06)
    # The solved curve lifts the darks too: 0.1 shown as 0.2 is 2.2 stops at a display gamma of 2.2.
    for curve in ([[0.0, 0.0], [0.05, 0.12], [0.15, 0.28], [1.0, 1.0]],
                  [{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.2}, {"x": 1.0, "y": 1.0}]):
        assert denoise.push_stops({"curve": {"master": curve}}) == pytest.approx(2.2)
    assert denoise.push_stops({"curve": {"master": [[0, 0], [0.1, 0.05], [1, 1]]}}) == 0.0
    iso["value"] = 51200.0
    assert denoise.decide(photo, "raw", {})["amount"] == 90.0
    monkeypatch.setenv("SUPERPHOTO_AI_DENOISE", "0")
    assert denoise.decide(photo, "raw", {}) is None


def test_iso_is_read_from_the_files_exif(tmp_path):
    photo = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x8769] = {0x8827: 3200}
    Image.new("RGB", (8, 8)).save(photo, exif=exif)
    assert denoise.iso_of(photo) == 3200.0
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(plain)
    assert denoise.iso_of(plain) == 0.0


def test_render_plan_denoises_only_the_delivered_render_of_a_noisy_raw(tmp_path, monkeypatch):
    calls: list[tuple[Path, dict]] = []

    def render(binary, source, document, destination, *, edge=0, timeout=300.0, applied=True):
        destination.mkdir(parents=True, exist_ok=True)
        kept = document if applied else denoise.without(document)
        (destination / "develop.json").write_text(json.dumps(kept), encoding="utf-8")
        calls.append((destination, document))
        y = (Image.linear_gradient("L").resize((96, 64)))
        Image.merge("RGB", (y, y, y)).save(destination / "render.jpg")
        return destination / "render.jpg"

    monkeypatch.setattr("superphoto.lightcraft.render_document", render)
    monkeypatch.setattr(denoise, "iso_of", lambda source: 6400.0)
    monkeypatch.setenv("SUPERPHOTO_AI_DENOISE", "1")
    source = tmp_path / "source.nef"
    source.write_bytes(b"raw")
    plan, _ = directed.sanitize_plan({"title": "t", "intent": "x", "develop": {"color": {"vibrance": 10}}}, slot="A")
    _, document, notes, _, output = directed._render_plan(Path("cli"), source, plan, "raw", tmp_path / "out",
                                                          edge=2400, colors=None, final_name="exports")
    final_folder, final = calls[-1]
    assert final_folder == tmp_path / "out" / "exports" and output.parent == final_folder
    assert final["enhance"]["denoise_on"] is True and final["enhance"]["denoise"] >= 80.0
    assert all("enhance" not in doc for _, doc in calls[:-1])          # the solver's probes stay plain
    assert document["enhance"] == final["enhance"] and any("AI 降噪" in note for note in notes)
    # Not denoised in the end (e.g. not a Bayer raw): the record and the notes say so.
    calls.clear()
    monkeypatch.setattr("superphoto.lightcraft.render_document",
                        lambda *args, **kwargs: render(*args, **kwargs, applied=False))
    _, document, notes, _, _ = directed._render_plan(Path("cli"), source, plan, "raw", tmp_path / "again",
                                                     edge=2400, colors=None, final_name="exports")
    assert "enhance" not in document and any("没有生效" in note for note in notes)
    # A JPEG never asks for it.
    calls.clear()
    jpeg = tmp_path / "source.jpg"
    jpeg.write_bytes(b"jpeg")
    monkeypatch.setattr("superphoto.lightcraft.render_document", render)
    directed._render_plan(Path("cli"), jpeg, plan, "jpeg", tmp_path / "jpeg", edge=2400, colors=None,
                          final_name="exports")
    assert all("enhance" not in doc for _, doc in calls)
