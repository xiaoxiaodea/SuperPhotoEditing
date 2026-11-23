"""Submit stable files to a running server; remember successful imports by content."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import httpx

from .ingest import ALLOWED_EXTS
from .tasks import WorkerLock


class FolderImporter:
    def __init__(self, folder: Path, state_path: Path, client: httpx.Client, *, stable_seconds: float = 5,
                 max_bytes: int = 80 * 1024 * 1024, excluded: Path | None = None):
        self.folder, self.state_path, self.client = folder.resolve(), state_path, client
        self.stable_seconds, self.max_bytes = stable_seconds, max_bytes
        self.excluded = excluded.resolve() if excluded else None
        self.observed: dict[Path, tuple[tuple[int, int], float]] = {}
        self.completed: dict[Path, tuple[int, int]] = {}
        self.imported = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    def scan(self, now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        results = []
        present = set()
        for source in sorted(self.folder.rglob("*")):
            if (source.suffix.lower() not in ALLOWED_EXTS or source.is_symlink() or not source.is_file()
                    or (self.excluded and source.resolve().is_relative_to(self.excluded))):
                continue
            present.add(source)
            try:
                stat = source.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                if self.completed.get(source) == signature:
                    continue
                previous = self.observed.get(source)
                if previous is None or previous[0] != signature:
                    self.observed[source] = signature, now
                    continue
                if now - previous[1] < self.stable_seconds or not 0 < stat.st_size <= self.max_bytes:
                    continue
                # Freeze a snapshot. Do not submit a file still being copied into the watched directory.
                with tempfile.TemporaryFile() as snapshot, source.open("rb") as original:
                    digest = hashlib.sha256()
                    copied = 0
                    while chunk := original.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > self.max_bytes:
                            break
                        digest.update(chunk)
                        snapshot.write(chunk)
                    after = source.stat()
                    if copied != stat.st_size or signature != (after.st_size, after.st_mtime_ns):
                        self.observed.pop(source, None)
                        continue
                    key = digest.hexdigest()
                    if key in self.imported:
                        self.completed[source] = signature
                        continue
                    snapshot.seek(0)
                    response = self.client.post("/api/jobs", files={"file": (source.name, snapshot)})
                    response.raise_for_status()
                    job_id = response.json()["id"]
                    self.imported[key] = {"id": job_id, "source": str(source)}
                    self._save()
                    self.completed[source] = signature
                    results.append({"id": job_id, "source": str(source)})
            except (OSError, httpx.HTTPError, ValueError, KeyError) as exc:
                # Do not expose credential-bearing request URLs or response bodies.
                results.append({"source": str(source), "error": type(exc).__name__})
        self.observed = {path: value for path, value in self.observed.items() if path in present}
        return results

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.state_path.parent, prefix="imports-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(self.imported, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            Path(name).replace(self.state_path)
        finally:
            Path(name).unlink(missing_ok=True)


def import_folder(settings, folder: Path, *, watch: bool, stable_seconds: float,
                  interval: float, url: str | None = None) -> None:
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    base = url or f"http://{host}:{settings.port}"
    state_key = hashlib.sha256((str(folder.resolve()) + "\n" + base).encode()).hexdigest()[:20]
    state_path = settings.data_dir / "imports" / f"{state_key}.json"
    lock = WorkerLock(state_path.with_suffix(".lock"))
    lock.acquire()
    try:
        headers = {"Authorization": f"Bearer {settings.access_token}"} if settings.access_token else {}
        with httpx.Client(base_url=base, headers=headers, timeout=120, trust_env=False) as client:
            client.get("/api/health").raise_for_status()
            importer = FolderImporter(folder, state_path, client, stable_seconds=stable_seconds,
                                      max_bytes=settings.max_upload_mb * 1024 * 1024, excluded=settings.data_dir)
            # Both modes observe at least twice. Unchanged size+mtime is the stability criterion.
            importer.scan()
            time.sleep(stable_seconds)
            while True:
                for result in importer.scan():
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                if not watch:
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        lock.release()
