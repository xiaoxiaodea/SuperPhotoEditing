from __future__ import annotations

from pathlib import Path

from .config import Settings
from .db import SLOT_LETTERS

SAFE_FILES = {
    "original",
    "source_preview.jpg",
    "critique_preview.jpg",
    *(f"variant_{slot}.jpg" for slot in SLOT_LETTERS),
    "technical.json",
    "critique.json",
    "plan.json",
}


def job_dir(settings: Settings, job_id: str) -> Path:
    path = settings.jobs_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def media_path(settings: Settings, job_id: str, filename: str) -> Path:
    if filename not in SAFE_FILES and not filename.startswith("original"):
        # original keeps its extension: original.ARW
        stem = filename.split(".")[0]
        if stem != "original":
            raise ValueError("illegal media filename")
    return job_dir(settings, job_id) / filename
