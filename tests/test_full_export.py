import json
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from superphoto import full_export, lightcraft
from superphoto.agent import SuperPhotoClient
from superphoto.db import insert_refinement, update_job, update_refinement
from test_scene_preferences import saved_job

DOCUMENT = {"wb": {"mode": "custom", "temp": 5600.0, "tint": 4.0}, "light": {"exposure": 0.3},
            "detail": {"sharpen_amount": 40.0}}


def test_render_document_passes_format_size_and_export_options(tmp_path, monkeypatch):
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"x")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(lightcraft.subprocess, "run", run)
    source = tmp_path / "in.jpg"
    source.write_bytes(b"jpeg")
    out = lightcraft.render_document(Path("lc"), source, DOCUMENT, tmp_path / "full", name="render.tif",
                                     options={"bitDepth": 16, "colorSpace": "srgb"})
    command = seen[-1]
    assert out == tmp_path / "full" / "render.tif"
    assert "--size" not in command and "--quality" not in command   # full size; quality is a JPEG option
    assert command[-4:] == ["--opt", "bitDepth=16", "--opt", "colorSpace=srgb"]
    assert json.loads((tmp_path / "full" / "develop.json").read_text(encoding="utf-8")) == DOCUMENT
    # The delivery default is unchanged: render.jpg at a size, quality 92, no export options.
    lightcraft.render_document(Path("lc"), source, DOCUMENT, tmp_path / "small", edge=2400)
    assert seen[-1][-4:] == ["--quality", "92", "--size", "2400"]


def test_names_round_trip_and_reject_anything_else():
    version = str(uuid.uuid4())
    assert full_export.parse_name("B.tif") == ("B", "tiff")
    assert full_export.parse_name(f"C-{version}.jpg") == (f"C-{version}", "jpeg")
    for bad in ("b.tif", "B.png", "B-../x.jpg", "../B.jpg", "B-1234.jpg", "AB.jpg"):
        assert full_export.parse_name(bad) is None
    assert full_export.download_name("B", version, "tiff") == f"SuperPhoto-B-微调-{version[:8]}-原尺寸.tif"


def test_recorded_paths_follow_a_moved_data_directory(tmp_path):
    root = tmp_path / "jobs" / "photo"
    moved = full_export._in_job(root, "photo", r"E:\old\data\jobs\photo\directed\B\round-0\render.jpg")
    assert moved == root / "directed" / "B" / "round-0" / "render.jpg"


@pytest.fixture
def ready(tmp_path, monkeypatch, scene):
    from superphoto import app as appmod

    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SUPERPHOTO_LIGHTROOM_ENABLED", "1")
    appmod.reset_runtime()
    settings, conn = appmod._boot()
    saved_job(conn)
    update_job(conn, "photo", original_file="original.jpg")
    root = settings.jobs_dir / "photo"
    (root / "lightroom-audits").mkdir(parents=True)
    (root / "original.jpg").write_bytes(scene.read_bytes())
    accepted = root / "directed" / "B" / "round-0"
    accepted.mkdir(parents=True)
    (accepted / "render.jpg").write_bytes(scene.read_bytes())
    (accepted / "develop.json").write_text(json.dumps(DOCUMENT), encoding="utf-8")
    (root / "lightroom-audits" / "variant_B.json").write_text(json.dumps(
        {"engine": "directed", "slot": "B", "status": "delivered", "accepted_export": str(accepted / "render.jpg")}))
    renders = []

    def render(binary, source, document, destination, **kwargs):
        assert source.is_file() and source.parent == destination   # a private copy, not the job's original
        assert kwargs["edge"] == 0
        renders.append({"document": document, **kwargs})
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "develop.json").write_text(json.dumps(document), encoding="utf-8")
        mode = "I;16" if kwargs["name"].endswith(".tif") else "RGB"
        Image.new(mode, (64, 48)).save(destination / kwargs["name"])
        return destination / kwargs["name"]

    monkeypatch.setattr(lightcraft, "render_document", render)
    monkeypatch.setattr(appmod, "connect_lightroom", lambda *_: (SimpleNamespace(binary="lc"), {}))
    yield appmod, settings, conn, root, renders
    appmod.reset_runtime()


def test_full_size_export_renders_the_delivered_document_once(ready):
    appmod, settings, conn, root, renders = ready
    with TestClient(appmod.app) as client:
        made = client.post("/api/jobs/photo/full", json={"slot": "B", "format": "tiff"})
        assert made.status_code == 200, made.text
        body = made.json()
        assert body["url"] == "/api/jobs/photo/full/B.tif"
        assert (body["width"], body["height"], body["cached"]) == (64, 48, False)
        assert renders[-1]["document"] == DOCUMENT
        assert renders[-1]["options"] == {"bitDepth": 16, "colorSpace": "srgb"}
        assert not (root / "full" / "B" / "source.jpg").exists()   # the copy is removed after rendering
        got = client.get(body["url"])
        assert got.status_code == 200 and got.headers["content-type"] == "image/tiff"
        assert "attachment" in got.headers["content-disposition"]
        # Same document: the file is reused. A JPEG is a separate render of the same document.
        assert client.post("/api/jobs/photo/full", json={"slot": "B", "format": "tiff"}).json()["cached"] is True
        assert client.post("/api/jobs/photo/full", json={"slot": "B"}).json()["url"] == "/api/jobs/photo/full/B.jpg"
        assert len(renders) == 2 and renders[-1]["quality"] == 95
        # A re-delivered finish (new develop document) is rendered again.
        changed = {**DOCUMENT, "light": {"exposure": 0.5}}
        (root / "directed" / "B" / "round-0" / "develop.json").write_text(json.dumps(changed), encoding="utf-8")
        assert client.post("/api/jobs/photo/full", json={"slot": "B", "format": "tiff"}).json()["cached"] is False
        assert renders[-1]["document"] == changed


def test_full_size_export_of_a_refinement_and_refusals(ready):
    appmod, settings, conn, root, renders = ready
    version = str(uuid.uuid4())
    insert_refinement(conn, version, "photo", "B", None, "整体亮一点")
    exports = root / "refinements" / version / "exports"
    exports.mkdir(parents=True)
    (exports / "develop.json").write_text(json.dumps({**DOCUMENT, "light": {"exposure": 0.6}}), encoding="utf-8")
    with TestClient(appmod.app) as client:
        body = {"slot": "B", "version": version}
        assert client.post("/api/jobs/photo/full", json=body).status_code == 409   # not ready yet
        update_refinement(conn, version, status="ready", step="ready")
        made = client.post("/api/jobs/photo/full", json=body)
        assert made.status_code == 200, made.text
        assert made.json()["url"] == f"/api/jobs/photo/full/B-{version}.jpg"
        assert renders[-1]["document"]["light"] == {"exposure": 0.6}
        assert client.post("/api/jobs/photo/full", json={"slot": "C", "version": version}).status_code == 409
        # A finish without a LightCraft develop record (base mode, older jobs) cannot be rendered again.
        refused = client.post("/api/jobs/photo/full", json={"slot": "A"})
        assert refused.status_code == 409 and "显影记录" in refused.json()["detail"]
        assert client.post("/api/jobs/photo/full", json={"slot": "B", "format": "png"}).status_code == 422
        assert client.post("/api/jobs/missing/full", json={"slot": "B"}).status_code == 404
        assert client.get("/api/jobs/photo/full/A.jpg").status_code == 404
        assert client.get("/api/jobs/photo/full/..%2Foriginal.jpg").status_code == 404
        assert client.post("/api/jobs/photo/full", json={"slot": "B", "format": "tiff"}).json()["cached"] is False
        (root / "original.jpg").unlink()
        gone = client.post("/api/jobs/photo/full", json={"slot": "B", "format": "tiff"})
        assert gone.status_code == 200 and gone.json()["cached"]   # still served; a new document needs the original
        (root / "directed" / "B" / "round-0" / "develop.json").write_text("{}", encoding="utf-8")
        assert client.post("/api/jobs/photo/full", json={"slot": "B", "format": "tiff"}).status_code == 409


def test_agent_export_downloads_the_full_size_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/jobs/job1/full":
            assert json.loads(request.content) == {"slot": "B", "format": "tiff"}
            return httpx.Response(200, json={"url": "/api/jobs/job1/full/B.tif", "filename": "SuperPhoto-B-原尺寸.tif",
                                             "width": 8256, "height": 5504, "bytes": 4})
        if request.url.path == "/api/jobs/job1/full/B.tif":
            return httpx.Response(200, content=b"tiff", headers={"content-type": "image/tiff"})
        return httpx.Response(404, json={"detail": "no"})

    client = SuperPhotoClient("http://127.0.0.1:8765", httpx.Client(
        base_url="http://127.0.0.1:8765", transport=httpx.MockTransport(handler), trust_env=False),
        out_dir=tmp_path / "out", poll_seconds=0)
    payload = client.export_full("job1", "B", None, "tiff")
    assert payload["ok"] and (payload["width"], payload["height"]) == (8256, 5504)
    assert Path(payload["path"]).read_bytes() == b"tiff"
    assert Path(payload["path"]).name == "SuperPhoto-B-原尺寸.tif"
