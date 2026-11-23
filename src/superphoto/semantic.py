"""Optional local semantic segmentation for safer photographic masks.

The renderer must keep working without ML dependencies or a downloaded model.
When available, a small ADE20K SegFormer model supplies coarse scene masks;
heuristic masks in layers.py remain the fallback and edge-detail source.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .config import Settings

_MODEL_READY: dict[str, bool] = {}


@dataclass
class SemanticMasks:
    subject: np.ndarray
    sky: np.ndarray
    foliage: np.ndarray
    water: np.ndarray
    skin: np.ndarray
    confidence: float
    model: str


_SUBJECT_WORDS = {
    "person", "people", "man", "woman", "child", "boy", "girl", "animal",
    "bird", "cat", "dog", "horse", "cow", "sheep", "boat", "ship", "car",
    "bus", "truck", "bicycle", "motorcycle", "flower", "plant", "sculpture",
}
_SKY_WORDS = {"sky"}
_WATER_WORDS = {"water", "sea", "river", "lake", "pool", "waterfall"}
_FOLIAGE_WORDS = {"tree", "grass", "plant", "flower", "field", "palm"}
_PERSON_WORDS = {"person", "people", "man", "woman", "child", "boy", "girl"}


def semantic_status(settings: Settings) -> dict[str, Any]:
    if not settings.semantic_enabled:
        return {"enabled": False, "available": False, "model": settings.semantic_model}
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return {
            "enabled": True,
            "available": False,
            "model": settings.semantic_model,
            "reason": "安装 semantic 可选依赖后启用",
        }
    cached = _model_cached(settings.semantic_model)
    return {
        "enabled": True,
        "available": cached,
        "model": settings.semantic_model,
        "reason": None if cached else "未找到本地模型缓存，当前使用基础分区；请预先下载分割模型",
    }


def segment_scene(srgb: np.ndarray, settings: Settings) -> SemanticMasks | None:
    """Return semantic masks, or None on any optional-model failure."""
    if not settings.semantic_enabled or not _model_cached(settings.semantic_model):
        return None
    try:
        processor, model, torch = _load_model(settings.semantic_model)
        image = Image.fromarray((np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
            # Keep the class tensor at model resolution. Upsampling 150 channels
            # to 2048px before softmax consumed several GiB on small NAS hosts.
            probs = torch.softmax(logits, dim=1)[0]
            confidence, labels = torch.max(probs, dim=0)
        label_map, conf = _resize_predictions(labels.cpu().numpy(),
                                             confidence.cpu().numpy(), srgb.shape[:2])
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

        subject = _labels_mask(label_map, id2label, _SUBJECT_WORDS)
        sky = _labels_mask(label_map, id2label, _SKY_WORDS)
        water = _labels_mask(label_map, id2label, _WATER_WORDS)
        foliage = _labels_mask(label_map, id2label, _FOLIAGE_WORDS)
        person = _labels_mask(label_map, id2label, _PERSON_WORDS)

        # Suppress low-confidence boundaries, then feather for photographic edits.
        trusted = np.clip((conf - settings.semantic_min_confidence) / 0.25, 0.0, 1.0)
        subject = _feather(subject * trusted, 7.0)
        sky = _feather(sky * trusted, 5.0)
        water = _feather(water * trusted, 5.0)
        foliage = _feather(foliage * trusted, 4.0)
        # This is a person protection mask, not a literal skin parser. It prevents
        # aggressive global skin heuristics from touching unrelated orange objects.
        skin = _feather(person * trusted, 4.0)
        used = np.maximum.reduce([subject, sky, water, foliage])
        mean_conf = float(np.mean(conf[used > 0.05])) if np.any(used > 0.05) else 0.0
        return SemanticMasks(subject, sky, foliage, water, skin, mean_conf, settings.semantic_model)
    except Exception:
        # Optional AI must never make photo processing unavailable.
        return None


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    local_only = _MODEL_READY.get(model_name, False) or _model_cached(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name, local_files_only=local_only)
    _MODEL_READY[model_name] = True
    model.eval()
    return processor, model, torch


def _resize_predictions(labels: np.ndarray, confidence: np.ndarray,
                        shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    label_map = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(
        (width, height), Image.Resampling.NEAREST))
    conf = np.asarray(Image.fromarray(confidence.astype(np.float32)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return label_map, conf


def _model_cached(model_name: str) -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", home / "hub")))
    local = Path(model_name)
    cache = hub / ("models--" + model_name.replace("/", "--")) / "snapshots"
    snapshots = [local] if local.is_dir() else list(cache.iterdir()) if cache.is_dir() else []
    for snapshot in snapshots:
        if not (snapshot / "config.json").is_file() or not (snapshot / "preprocessor_config.json").is_file():
            continue
        if (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file():
            return True
        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = snapshot / filename
            if index.is_file():
                try:
                    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                    if shards and all((snapshot / shard).is_file() for shard in shards):
                        return True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
    return False


def _labels_mask(label_map: np.ndarray, id2label: dict[int, str], words: set[str]) -> np.ndarray:
    ids = [idx for idx, label in id2label.items() if any(word in label for word in words)]
    if not ids:
        return np.zeros(label_map.shape, dtype=np.float32)
    return np.isin(label_map, ids).astype(np.float32)


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(u8, "L").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(out, dtype=np.float32) / 255.0
