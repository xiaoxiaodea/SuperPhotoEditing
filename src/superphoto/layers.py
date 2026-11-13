"""Per-pixel layer masks: subject, sky, foliage, clutter, tiny defects."""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

from .color import luma, rgb_to_hsl, smoothstep
from .models import RegionBox, TechnicalReport

if TYPE_CHECKING:
    from .semantic import SemanticMasks


@dataclass
class LayerMasks:
    subject: np.ndarray
    sky: np.ndarray
    foliage: np.ndarray
    water: np.ndarray
    skin: np.ndarray
    background: np.ndarray
    clutter: np.ndarray
    defects: np.ndarray


def build_layers(
    srgb: np.ndarray,
    technical: TechnicalReport | None = None,
    clutter_boxes: list[RegionBox] | None = None,
    semantic: "SemanticMasks | None" = None,
) -> LayerMasks:
    img = np.clip(srgb.astype(np.float32, copy=False), 0.0, 1.0)
    h, w = img.shape[:2]
    y = luma(img)
    hue, sat, lig = rgb_to_hsl(img)

    gy, gx = np.gradient(y)
    contrast = np.hypot(gx, gy).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt(((yy - cy) / max(h, 1)) ** 2 + ((xx - cx) / max(w, 1)) ** 2)

    skin = (
        (hue > 0.02) & (hue < 0.13) & (sat > 0.12) & (sat < 0.65) & (lig > 0.18) & (lig < 0.85)
    ).astype(np.float32)
    sky = (
        (yy < h * 0.48) & (hue > 0.50) & (hue < 0.72) & (sat > 0.08) & (lig > 0.28)
    ).astype(np.float32)
    foliage = ((hue > 0.22) & (hue < 0.45) & (sat > 0.15) & (lig > 0.08) & (lig < 0.8)).astype(np.float32)
    water = ((hue > 0.40) & (hue < 0.62) & (sat > 0.08) & (lig > 0.06) & (lig < 0.75)).astype(np.float32)

    sal = contrast * (0.45 + 0.55 * np.clip(y, 0.0, 1.0)) * (1.15 - np.clip(dist, 0.0, 1.0))
    sal = sal + 0.45 * skin
    if technical is not None:
        bbox = _box_mask(
            h,
            w,
            technical.subject_left,
            technical.subject_top,
            technical.subject_right,
            technical.subject_bottom,
            feather=0.08,
        )
        sal = sal * (0.30 + 0.70 * bbox)
    thr = float(np.percentile(sal, 78))
    subject = smoothstep(thr * 0.55, thr * 1.25, sal)
    if float(np.mean(subject)) > 0.72:
        subject *= smoothstep(0.35, 0.02, dist)
    subject = _feather(subject, 11)
    subject = np.clip(subject, 0.0, 1.0)

    # Semantic masks lead when the optional model is available. The heuristic
    # masks remain as low-weight detail so textured subjects do not become flat.
    if semantic is not None and semantic.confidence >= 0.20:
        subject = np.maximum(semantic.subject, subject * 0.24)
        sky = np.maximum(semantic.sky, sky * 0.18)
        foliage = np.maximum(semantic.foliage, foliage * 0.22)
        water = np.maximum(semantic.water, water * 0.20)
        # The semantic person mask protects a region; hue still identifies skin.
        skin = np.maximum(skin * semantic.skin, skin * 0.12)

    sky = _feather(sky * (1.0 - 0.65 * subject), 10)
    foliage = _feather(foliage, 6)
    water = _feather(water * (1.0 - 0.4 * subject), 8)
    skin = _feather(skin, 5)
    background = np.clip(1.0 - subject, 0.0, 1.0)

    edge = np.ones((h, w), dtype=np.float32)
    edge[int(h * 0.10) : int(h * 0.90), int(w * 0.10) : int(w * 0.90)] = 0.0
    edge = _feather(edge, 8)
    loud = np.clip((sat - 0.32) * 2.2, 0.0, 1.0) + np.clip((y - 0.78) * 3.0, 0.0, 1.0) * 0.55
    clutter = np.clip(edge * background * (1.0 - 0.7 * sky) * loud, 0.0, 1.0)
    if clutter_boxes:
        for box in clutter_boxes:
            clutter = np.maximum(clutter, _box_mask(h, w, box.left, box.top, box.right, box.bottom, 0.02) * background)

    blur_y = _blur(y, 3.5)
    hot = ((y - blur_y) > 0.22) & (y > 0.55) & (subject < 0.55)
    defects = hot.astype(np.float32)
    if float(np.mean(defects)) > 0.012:
        defects[:] = 0.0
    defects = _feather(defects, 1.6)

    return LayerMasks(
        subject=subject,
        sky=sky,
        foliage=foliage,
        water=water,
        skin=skin,
        background=background,
        clutter=np.clip(clutter, 0.0, 1.0),
        defects=np.clip(defects, 0.0, 1.0),
    )


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.2:
        return mask.astype(np.float32)
    u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(out, dtype=np.float32) / 255.0


def _blur(gray: np.ndarray, radius: float) -> np.ndarray:
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(out, dtype=np.float32) / 255.0


def _box_mask(
    h: int,
    w: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
    feather: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(w - 1, 1)
    yn = yy / max(h - 1, 1)
    inside = (xn >= left) & (xn <= right) & (yn >= top) & (yn <= bottom)
    m = inside.astype(np.float32)
    px = max(1.0, feather * max(h, w))
    return _feather(m, px * 0.35)
