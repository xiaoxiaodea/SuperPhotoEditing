import json
from pathlib import Path

import httpx

from superphoto.folder_import import FolderImporter


def test_only_stable_files_are_imported_and_content_dedup_survives_restart(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "照片.jpg"
    source.write_bytes(b"sample image bytes")
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "job-1"})

    state = tmp_path / "state.json"
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        importer = FolderImporter(incoming, state, client, stable_seconds=5)
        assert importer.scan(0) == []
        assert importer.scan(4) == []
        result = importer.scan(5)
        assert result == [{"id": "job-1", "source": str(source.resolve())}]
        assert importer.scan(20) == []
        (incoming / "renamed.jpg").write_bytes(source.read_bytes())
        restarted = FolderImporter(incoming, state, client, stable_seconds=5)
        assert restarted.scan(30) == []
        assert restarted.scan(35) == []
        assert len(calls) == 1
        assert list(json.loads(state.read_text(encoding="utf-8")).values())[0]["id"] == "job-1"


def test_changed_files_wait_again_and_failed_uploads_can_retry(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    source = incoming / "photo.jpg"
    source.write_bytes(b"part1")
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"id": "job-2"})

    state = tmp_path / "state.json"
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        importer = FolderImporter(incoming, state, client, stable_seconds=5)
        assert importer.scan(0) == []
        source.write_bytes(b"part1 part2")
        assert importer.scan(5) == []
        assert importer.scan(9) == []
        assert importer.scan(10)[0]["error"] == "HTTPStatusError"
        assert not state.exists()
        assert importer.scan(11)[0]["id"] == "job-2"
        assert len(calls) == 2


def test_outputs_and_unsupported_or_oversized_files_are_not_reimported(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    output = incoming / "data"
    output.mkdir()
    (output / "variant_A.jpg").write_bytes(b"output")
    (incoming / "unsupported.txt").write_bytes(b"ignored")
    (incoming / "too-large.jpg").write_bytes(b"longer than limit")
    calls = []
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(lambda r:
            calls.append(r) or httpx.Response(200, json={"id": "unexpected"}))) as client:
        importer = FolderImporter(incoming, tmp_path / "state.json", client,
                                  stable_seconds=1, max_bytes=5, excluded=output)
        importer.scan(0)
        assert importer.scan(2) == []
        assert not calls
