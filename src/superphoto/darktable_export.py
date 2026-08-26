"""Optional darktable sidecar export.

The working renderer is the parametric engine in render.py. When darktable-cli
is installed, we can emit a JSON recipe plus a style-intent note. Full .xmp
history encoding is version-specific; this module keeps the door open without
pretending Adobe/darktable color science is identical.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .models import Recipe


def darktable_cli_path() -> str | None:
    return shutil.which("darktable-cli")


def write_recipe_sidecar(path: Path, recipe: Recipe, look_id: str) -> Path:
    payload = {
        "engine": "superphoto-parametric",
        "look_id": look_id,
        "darktable_cli": darktable_cli_path(),
        "recipe": recipe.model_dump(),
        "notes": (
            "Apply these sliders in order: white balance → exposure → "
            "highlights/shadows → contrast → HSL → color wheels → clarity → vignette."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
