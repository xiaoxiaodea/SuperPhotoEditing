from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _disable_remote_critics(monkeypatch) -> None:
    monkeypatch.setenv("SUPERPHOTO_OLLAMA_ENABLED", "0")
    monkeypatch.setenv("XAI_API_KEY", "")


def make_scene(path: Path, *, underexposed: bool = True, size: tuple[int, int] = (960, 640)) -> Path:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny = yy / max(h - 1, 1)
    nx = xx / max(w - 1, 1)
    sky = np.stack(
        [
            0.25 + 0.15 * (1 - ny),
            0.38 + 0.10 * (1 - ny),
            0.62 + 0.18 * (1 - ny),
        ],
        axis=-1,
    )
    ground = np.stack(
        [
            0.18 + 0.04 * nx,
            0.28 + 0.08 * (1 - nx),
            0.12 + 0.03 * ny,
        ],
        axis=-1,
    )
    mix = smooth(ny, 0.42, 0.55)
    rgb = sky * (1 - mix[..., None]) + ground * mix[..., None]
    cx, cy, rad = 0.38, 0.58, 0.16
    dist = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2 * 1.3)
    subject = np.clip(1.0 - dist / rad, 0.0, 1.0) ** 1.4
    skin = np.array([0.62, 0.42, 0.32], dtype=np.float32)
    rgb = rgb * (1 - subject[..., None] * 0.85) + skin * subject[..., None]
    if underexposed:
        rgb *= 0.42
    rgb = np.clip(rgb + 0.015 * np.random.default_rng(0).random(rgb.shape), 0, 1)
    arr = (rgb * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, quality=92)
    return path


def smooth(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    return make_scene(tmp_path / "scene.jpg")


@pytest.fixture
def bright_scene(tmp_path: Path) -> Path:
    return make_scene(tmp_path / "bright.jpg", underexposed=False)
