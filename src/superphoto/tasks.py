"""Durable, single-worker scheduling for one local SQLite data directory.

Run one ASGI worker per data directory. The OS lock is held until the processing
thread exits, including graceful shutdown; a second process fails at startup.
SQLite and this lock file must live on local storage, not a network filesystem.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Callable, TYPE_CHECKING
import uuid

from .db import connect, insert_job, utcnow

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)
INTERRUPTED = "应用在处理完成前关闭了；本次任务已中断，请重试。原片和已有成片已保留。"


def enqueue_source(settings: Settings, conn: sqlite3.Connection, source: Path,
                   *, original_name: str | None = None) -> dict[str, str]:
    """Copy an existing source into a fresh job and durably queue it.

    Shared by HTTP retry and local directory import. A running TaskWorker will
    discover it on its next poll; callers may wake that worker for lower latency.
    This function never performs image processing or starts a worker itself.
    """
    from .ingest import ALLOWED_EXTS, source_kind
    from .store import job_dir

    source = Path(source)
    ext = source.suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"unsupported file type: {ext or 'none'}")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if source.stat().st_size > max_bytes:
        raise ValueError(f"file larger than {settings.max_upload_mb} MB")
    job_id = str(uuid.uuid4())
    folder = job_dir(settings, job_id)
    original = folder / f"original{ext}"
    try:
        # Bound each read and the total copied bytes, even if the source grows.
        size = 0
        with source.open("rb") as input_file, original.open("wb") as output:
            while chunk := input_file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"file larger than {settings.max_upload_mb} MB")
                output.write(chunk)
            if not size:
                raise ValueError("empty file")
            output.flush()
            os.fsync(output.fileno())
        insert_job(conn, job_id, original_name or source.name, original.name)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return {"id": job_id, "kind": source_kind(original)}


class WorkerLock:
    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("此数据目录已有后台 worker；仅支持单进程服务，请勿使用多个 ASGI workers。") from exc
        self.file = handle

    def release(self) -> None:
        if self.file is None:
            return
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()
            self.file = None
        # Never unlink: another process may already hold this file's inode.


def recover_interrupted(conn: sqlite3.Connection) -> None:
    for table in ("jobs", "refinements"):
        conn.execute(f"""UPDATE {table} SET status='error', step='error',
            error=?, updated_at=? WHERE status='running'""", (INTERRUPTED, utcnow()))
    conn.commit()


def claim_next(conn: sqlite3.Connection) -> tuple[str, dict] | None:
    """Claim before touching output directories, so a crash never replays work."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""SELECT 'jobs' AS kind, id, created_at FROM jobs WHERE status='queued'
            UNION ALL SELECT 'refinements', id, created_at FROM refinements WHERE status='queued'
            ORDER BY created_at, kind, id LIMIT 1""").fetchone()
        if row is None:
            conn.commit()
            return None
        table = row["kind"]
        conn.execute(f"UPDATE {table} SET status='running', step='starting', updated_at=? WHERE id=?",
                     (utcnow(), row["id"]))
        record = dict(conn.execute(f"SELECT * FROM {table} WHERE id=?", (row["id"],)).fetchone())
        conn.commit()
        return table, record
    except BaseException:
        conn.rollback()
        raise


class TaskWorker:
    def __init__(self, db_path: Path, jobs_dir: Path,
                 run_job: Callable[[str, Path], None], run_refinement: Callable[[str], None],
                 poll_interval: float = 0.5):
        self.db_path, self.jobs_dir = db_path, jobs_dir
        self.run_job, self.run_refinement = run_job, run_refinement
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = WorkerLock(db_path.with_name(db_path.name + ".worker.lock"))

    def start(self) -> None:
        self._lock.acquire()
        try:
            conn = connect(self.db_path)
            try:
                recover_interrupted(conn)
            finally:
                conn.close()
            self._thread = threading.Thread(target=self._run, name="superphoto-worker", daemon=True)
            self._thread.start()
        except BaseException:
            self._lock.release()
            raise

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self.wake()
        if self._thread is not None:
            # In-flight native processing cannot safely be cancelled. Drain it
            # before releasing the lock or closing application DB connections.
            self._thread.join()
        self._lock.release()

    def _run(self) -> None:
        conn = connect(self.db_path)
        try:
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    task = claim_next(conn)
                    if task is not None:
                        self._execute(conn, *task)
                        continue
                except Exception:
                    conn.rollback()
                    log.exception("Background queue poll failed")
                self._wake.wait(self.poll_interval)
        finally:
            conn.close()

    def _execute(self, conn: sqlite3.Connection, table: str, record: dict) -> None:
        error = None
        try:
            if table == "jobs":
                name = record["original_file"] or f"original{Path(record['original_name'] or '').suffix.lower()}"
                original = self.jobs_dir / record["id"] / Path(name).name
                if not original.is_file():
                    raise FileNotFoundError("原片已不在本机，请重新选择照片")
                self.run_job(record["id"], original)
            else:
                self.run_refinement(record["id"])
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            log.exception("Background task %s failed", record["id"])
        # Catch failures before a pipeline's own try/except, or a callback that
        # returns without saving a terminal state. Never leave running stranded.
        conn.execute(f"""UPDATE {table} SET status='error', step='error', error=?, updated_at=?
            WHERE id=? AND status='running'""",
            (error or "后台处理结束但未保存结果，请重试。", utcnow(), record["id"]))
        conn.commit()
