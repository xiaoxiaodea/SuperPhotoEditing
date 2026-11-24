from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
import pytest

from superphoto.db import (connect, get_job, get_refinement, insert_job,
                           insert_refinement, update_job, update_refinement)
from superphoto.tasks import TaskWorker, WorkerLock, enqueue_source


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        Event().wait(.01)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from superphoto import app as appmod
    monkeypatch.setenv("SUPERPHOTO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SUPERPHOTO_ACCESS_TOKEN", raising=False)
    appmod.reset_runtime()
    settings, conn = appmod._boot()
    yield appmod, settings, conn
    appmod.reset_runtime()


def original(settings, job_id):
    folder = settings.jobs_dir / job_id
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "original.jpg"
    source.write_bytes(b"saved source bytes")
    return source


def test_submission_is_durable_without_background_tasks_and_resumes_at_lifespan(runtime, monkeypatch):
    appmod, settings, conn = runtime
    calls = []
    def process(job_id, source):
        calls.append((job_id, source.read_bytes()))
        update_job(appmod._c(), job_id, status="ready", step="ready")
    monkeypatch.setattr(appmod, "_safe_run", process)
    # A client without lifespan cannot run tasks, but submission is durable.
    client = TestClient(appmod.app)
    response = client.post("/api/jobs", files={"file": ("photo.jpg", b"saved photo", "image/jpeg")})
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    assert calls == []
    assert get_job(conn, job_id)["status"] == "queued"
    assert get_job(conn, job_id)["original_file"] == "original.jpg"
    client.close()
    appmod.reset_runtime()
    _, conn = appmod._boot()
    with TestClient(appmod.app) as client:
        wait_for(lambda: get_job(conn, job_id)["status"] == "ready")
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "ready"
    assert calls == [(job_id, b"saved photo")]


def test_restart_recovers_queued_work_and_marks_started_work_interrupted(runtime, monkeypatch):
    appmod, settings, conn = runtime
    for job_id in ("started", "queued", "parent"):
        insert_job(conn, job_id, "photo.jpg", "original.jpg")
        original(settings, job_id)
    update_job(conn, "started", status="running", step="render")
    update_job(conn, "parent", status="ready", step="ready")
    insert_refinement(conn, "started-ref", "parent", "A", None, "亮一点")
    update_refinement(conn, "started-ref", status="running", step="render")
    insert_refinement(conn, "queued-ref", "parent", "B", None, "暗一点")
    marker = settings.jobs_dir / "started" / "partial-output.jpg"
    marker.write_bytes(b"must not overwrite")
    calls = []
    def process(job_id, source):
        calls.append(job_id)
        update_job(appmod._c(), job_id, status="ready", step="ready")
    def refine(rid):
        calls.append(rid)
        update_refinement(appmod._c(), rid, status="ready", step="ready")
    monkeypatch.setattr(appmod, "_safe_run", process)
    monkeypatch.setattr(appmod, "_safe_refine", refine)
    with TestClient(appmod.app):
        wait_for(lambda: get_refinement(conn, "queued-ref")["status"] == "ready")
        assert calls == ["queued", "queued-ref"]
        for record in (get_job(conn, "started"), get_refinement(conn, "started-ref")):
            assert record["status"] == "error"
            assert "中断" in record["error"]
        assert marker.read_bytes() == b"must not overwrite"
        assert get_job(conn, "parent")["status"] == "ready"


def test_one_worker_serializes_jobs_and_refinements_and_shutdown_leaves_queue(runtime):
    _, settings, conn = runtime
    for job_id in ("first", "later", "parent"):
        insert_job(conn, job_id, "photo.jpg", "original.jpg")
        original(settings, job_id)
    update_job(conn, "parent", status="ready", step="ready")
    insert_refinement(conn, "revision", "parent", "A", None, "亮一点")
    started, release, stopped = Event(), Event(), Event()
    calls = []
    def process(job_id, source):
        calls.append(job_id)
        started.set()
        assert release.wait(5)
        own = connect(settings.db_path)
        try:
            update_job(own, job_id, status="ready", step="ready")
        finally:
            own.close()
    worker = TaskWorker(settings.db_path, settings.jobs_dir, process, lambda rid: calls.append(rid), .01)
    worker.start()
    assert started.wait(5)
    assert get_job(conn, "first")["status"] == "running"
    assert get_job(conn, "later")["status"] == "queued"
    assert get_refinement(conn, "revision")["status"] == "queued"
    def stop():
        worker.stop()
        stopped.set()
    closer = Thread(target=stop)
    closer.start()
    try:
        assert not stopped.wait(.05)
        with pytest.raises(RuntimeError, match="单进程"):
            WorkerLock(settings.db_path.with_name(settings.db_path.name + ".worker.lock")).acquire()
    finally:
        release.set()
        closer.join(timeout=5)
    assert stopped.is_set()
    assert calls == ["first"]
    assert get_job(conn, "later")["status"] == "queued"
    assert get_refinement(conn, "revision")["status"] == "queued"


def test_worker_lock_blocks_another_process_and_releases_after_exit(tmp_path):
    path = tmp_path / "worker.lock"
    lock = WorkerLock(path)
    lock.acquire()
    script = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from superphoto.tasks import WorkerLock
try:
    lock = WorkerLock(Path(sys.argv[2]))
    lock.acquire()
except RuntimeError:
    print('LOCKED')
else:
    lock.release()
    print('ACQUIRED')
"""
    args = [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "src"), str(path)]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "LOCKED"
    finally:
        lock.release()
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ACQUIRED"


def test_failed_callback_or_missing_source_cannot_leave_running(runtime, monkeypatch):
    appmod, settings, conn = runtime
    for job_id in ("missing", "crash", "unfinished"):
        insert_job(conn, job_id, "photo.jpg", "original.jpg")
    original(settings, "crash")
    original(settings, "unfinished")
    calls = []
    def process(job_id, source):
        calls.append(job_id)
        if job_id == "crash":
            raise RuntimeError("pipeline failed before its handler")
    monkeypatch.setattr(appmod, "_safe_run", process)
    with TestClient(appmod.app):
        wait_for(lambda: get_job(conn, "unfinished")["status"] == "error")
        assert calls == ["crash", "unfinished"]
        assert "原片" in get_job(conn, "missing")["error"]
        assert "pipeline failed" in get_job(conn, "crash")["error"]
        assert "未保存结果" in get_job(conn, "unfinished")["error"]


def test_bearer_and_browser_cookie_protect_api_and_media(runtime, monkeypatch):
    appmod, settings, conn = runtime
    monkeypatch.setenv("SUPERPHOTO_ACCESS_TOKEN", "a secret token")
    folder = settings.jobs_dir / "private"
    folder.mkdir()
    (folder / "variant_A.jpg").write_bytes(b"private image")
    with TestClient(appmod.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/auth/status").json() == {"required": True, "authenticated": False}
        for path in ("/api/looks", "/api/health", "/media/private/variant_A.jpg"):
            response = client.get(path)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
        assert client.get("/api/looks?token=a secret token").status_code == 401
        assert client.get("/api/looks", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/api/looks", headers={"Authorization": "Bearer a secret token"}).status_code == 200
        assert client.post("/api/auth/login", json={"token": "wrong"}).status_code == 401
        response = client.post("/api/auth/login", json={"token": "a secret token"})
        assert response.status_code == 200
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert "a secret token" not in cookie
        assert client.get("/api/auth/status").json()["authenticated"] is True
        assert client.get("/media/private/variant_A.jpg").content == b"private image"
        # Six or more versions per photo: every slot letter is served, nothing beyond them or other job files.
        (folder / "variant_F.jpg").write_bytes(b"sixth")
        (folder / "variant_J.jpg").write_bytes(b"no slot")
        assert client.get("/media/private/variant_F.jpg").content == b"sixth"
        assert client.get("/media/private/variant_J.jpg").status_code == 404
        assert client.get("/media/private/variant_F.json").status_code == 404
        assert client.post("/api/auth/logout", headers={"Origin": "http://evil.test"}).status_code == 403
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/looks").status_code == 401
        assert client.post("/api/auth/login", json={"token": "a secret token"},
                           headers={"Origin": "http://evil.test"}).status_code == 403


def test_https_cookie_is_secure_and_token_rotation_invalidates_session(runtime, monkeypatch):
    appmod, _, _ = runtime
    monkeypatch.setenv("SUPERPHOTO_ACCESS_TOKEN", "first")
    with TestClient(appmod.app, base_url="https://testserver") as client:
        response = client.post("/api/auth/login", json={"token": "first"})
        assert "Secure" in response.headers["set-cookie"]
        assert client.get("/api/looks").status_code == 200
        monkeypatch.setenv("SUPERPHOTO_ACCESS_TOKEN", "second")
        assert client.get("/api/looks").status_code == 401


def test_directory_import_enqueue_preserves_source_and_uses_fresh_job_directories(runtime, tmp_path):
    _, settings, conn = runtime
    source = tmp_path / "nas-photo.JPG"
    source.write_bytes(b"original from NAS")
    first = enqueue_source(settings, conn, source)
    second = enqueue_source(settings, conn, source, original_name="saved-name.jpg")
    assert first["kind"] == second["kind"] == "jpeg"
    assert first["id"] != second["id"]
    for result, name in ((first, source.name), (second, "saved-name.jpg")):
        record = get_job(conn, result["id"])
        assert record["status"] == "queued"
        assert record["original_name"] == name
        assert record["original_file"] == "original.jpg"
        copied = settings.jobs_dir / result["id"] / record["original_file"]
        assert copied.read_bytes() == source.read_bytes() == b"original from NAS"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        enqueue_source(settings, conn, source)
    assert len(list(settings.jobs_dir.iterdir())) == 2
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_upload_stream_limit_cleans_partial_files_and_does_not_queue(runtime, monkeypatch):
    appmod, settings, conn = runtime
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    sizes = []
    original_read = UploadFile.read
    async def read(upload, size=-1):
        sizes.append(size)
        return await original_read(upload, size)
    monkeypatch.setattr(UploadFile, "read", read)
    with TestClient(appmod.app) as client:
        response = client.post("/api/jobs", files={"file": ("big.jpg", b"x" * (1024 * 1024 + 1), "image/jpeg")})
        assert response.status_code == 400
        assert client.post("/api/jobs", files={"file": ("empty.jpg", b"", "image/jpeg")}).status_code == 400
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert list(settings.jobs_dir.iterdir()) == []
        assert sizes and all(size == 1024 * 1024 for size in sizes)
