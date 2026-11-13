"""Hasselblad Masters as a scene-matched parametric grade.

Finished master prints are not LUTs. We index each frame's *scene*
(subject, light, environment) and the *look* (split tone, contrast, HSL).
A user photo is matched by scene, then given a scaled copy of that look.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .analysis import analyze, guess_genre, jpeg_bytes_to_linear
from .color import HUE_CENTERS, luma, rgb_to_hsl
from .desktop import project_root
from .models import Grade, HueShift, Look, Recipe, TechnicalReport

INDEX_NAME = "hasselblad_index.json"
_SCENE_KEYS = (
    "mean_luma",
    "shadow_clip",
    "top_third_blue",
    "skin_fraction",
    "green_fraction",
    "aqua_fraction",
    "orange_fraction",
    "magenta_fraction",
    "dynamic_spread",
)
_SCENE_W = {
    "green_fraction": 1.7,
    "top_third_blue": 1.5,
    "skin_fraction": 1.6,
    "aqua_fraction": 1.3,
    "orange_fraction": 1.1,
    "mean_luma": 1.2,
    "shadow_clip": 1.0,
    "magenta_fraction": 0.6,
    "dynamic_spread": 0.5,
}


def _index_path() -> Path:
    bundled = Path(__file__).with_name(INDEX_NAME)
    data = _awards_dir() / INDEX_NAME
    if data.is_file():
        return data
    return bundled


def _awards_dir() -> Path:
    return project_root() / "data" / "references" / "awards"


def _masters_dir() -> Path:
    return _awards_dir() / "hasselblad_masters"


@lru_cache(maxsize=1)
def load_index() -> dict:
    path = _index_path()
    if not path.is_file():
        return {"frames": [], "looks": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def master_look_ids() -> list[str]:
    return list(load_index().get("looks", {}).keys())


def look_for(look_id: str) -> Look | None:
    spec = load_index().get("looks", {}).get(look_id)
    if not spec:
        return None
    return Look(
        id=look_id,
        title=spec["title"],
        title_en=spec.get("title_en") or spec["title"],
        description=spec.get("description") or "",
        genre_tags=spec.get("genre_tags") or ["general"],
        moves=["hasselblad_master"],
        recipe=Recipe.model_validate(spec["recipe"]),
        source="master",
    )


def master_similarity(look_id: str, technical: TechnicalReport) -> float:
    frames = [f for f in load_index().get("frames", []) if f.get("look_id") == look_id]
    if not frames:
        return 0.0
    return max(_scene_similarity(technical, f["scene"]) for f in frames)


def ranked_master_ids(technical: TechnicalReport, limit: int = 4) -> list[str]:
    """Best-matching Hasselblad series, unique photographer, strongest scene first."""
    scored: list[tuple[float, str, str]] = []
    for frame in load_index().get("frames", []):
        sim = _scene_similarity(technical, frame["scene"])
        if sim < 0.20:
            continue
        scored.append((sim, frame["look_id"], frame.get("photographer") or frame["look_id"]))
    scored.sort(reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for sim, look_id, photographer in scored:
        if photographer in seen or look_id in out:
            continue
        seen.add(photographer)
        out.append(look_id)
        if len(out) >= limit:
            break
    return out


def match_report(technical: TechnicalReport, look_id: str) -> dict | None:
    best = None
    best_sim = -1.0
    for frame in load_index().get("frames", []):
        if frame.get("look_id") != look_id:
            continue
        sim = _scene_similarity(technical, frame["scene"])
        if sim > best_sim:
            best_sim = sim
            best = frame
    if best is None:
        return None
    return {
        "photographer": best.get("photographer") or "",
        "year": best.get("year") or "",
        "file": best.get("file") or "",
        "environment": best.get("environment") or "",
        "score": round(best_sim, 3),
        "genre": best.get("genre") or "",
    }


def recipe_for_photo(look_id: str, technical: TechnicalReport) -> Recipe:
    """Blend the series recipe with the closest frame of that master."""
    look = look_for(look_id)
    if look is None:
        return Recipe()
    best = None
    best_sim = -1.0
    for frame in load_index().get("frames", []):
        if frame.get("look_id") != look_id:
            continue
        sim = _scene_similarity(technical, frame["scene"])
        if sim > best_sim:
            best_sim = sim
            best = frame
    if best is None:
        return look.recipe
    frame_recipe = Recipe.model_validate(best["recipe"])
    return look.recipe.merged(_scale(frame_recipe, 0.55)).clamped()


def _scene_similarity(technical: TechnicalReport, scene: dict) -> float:
    acc = 0.0
    wsum = 0.0
    for key, weight in _SCENE_W.items():
        mu = float(scene.get(key, 0.0))
        val = float(getattr(technical, key))
        sd = 0.08 if key.endswith("fraction") or key == "top_third_blue" else 0.12
        z = abs(val - mu) / sd
        acc += weight * z
        wsum += weight
    dist = acc / max(wsum, 1e-6)
    # Night vs day is a hard gate.
    master_night = float(scene.get("mean_luma", 0.3)) < 0.14
    photo_night = technical.mean_luma < 0.14
    if master_night != photo_night:
        dist += 1.6
    master_skin = float(scene.get("skin_fraction", 0.0)) > 0.10
    photo_skin = technical.skin_fraction > 0.10
    if master_skin != photo_skin:
        dist += 0.7
    return float(np.exp(-dist / 2.2))


def _scale(recipe: Recipe, t: float) -> Recipe:
    data = recipe.model_dump()
    hsl = {}
    for name, shift in data.get("hsl", {}).items():
        hsl[name] = HueShift(
            hue=shift["hue"] * t,
            sat=shift["sat"] * t,
            lum=shift["lum"] * t,
        )
    g = data["grade"]

    def mix3(rgb: tuple) -> tuple[float, float, float]:
        return (rgb[0] * t, rgb[1] * t, rgb[2] * t)

    return Recipe(
        exposure_ev=data["exposure_ev"] * t,
        temperature=data["temperature"] * t,
        tint=data["tint"] * t,
        highlights=data["highlights"] * t,
        shadows=data["shadows"] * t,
        whites=data["whites"] * t,
        blacks=data["blacks"] * t,
        contrast=data["contrast"] * t,
        saturation=data["saturation"] * t,
        vibrance=data["vibrance"] * t,
        clarity=data["clarity"] * t,
        dehaze=data["dehaze"] * t,
        vignette=data["vignette"] * t,
        grain=data["grain"] * t,
        hsl=hsl,
        grade=Grade(
            shadows=mix3(tuple(g["shadows"])),
            midtones=mix3(tuple(g["midtones"])),
            highlights=mix3(tuple(g["highlights"])),
        ),
    )


def environment_label(tech: TechnicalReport) -> str:
    if tech.mean_luma < 0.12:
        return "night"
    if tech.skin_fraction > 0.12:
        return "portrait"
    if tech.green_fraction > 0.16:
        return "nature"
    if tech.aqua_fraction > 0.10 and tech.top_third_blue > 0.12:
        return "water"
    if tech.top_third_blue > 0.18:
        return "landscape"
    if tech.orange_fraction > 0.18 and tech.green_fraction < 0.06:
        return "warm-urban"
    return "architecture"


def recipe_from_print(srgb: np.ndarray, tech: TechnicalReport) -> Recipe:
    """Parametric taste of a finished print, relative to a typical developed JPEG."""
    y = luma(srgb)
    h, s, l = rgb_to_hsl(np.clip(srgb, 0, 1))
    shadow = y < 0.22
    mid = (y >= 0.22) & (y <= 0.72)
    high = y > 0.72
    sh = _mean_rgb(srgb, shadow, (0.12, 0.12, 0.12))
    mi = _mean_rgb(srgb, mid, (0.40, 0.40, 0.40))
    hi = _mean_rgb(srgb, high, (0.78, 0.78, 0.78))
    grade = Grade(
        shadows=(float(np.clip(sh[0] - sh[2], -0.12, 0.12) * 0.55), 0.0, float(np.clip(sh[2] - sh[0], -0.12, 0.12) * 0.45)),
        midtones=(float(np.clip(mi[0] - 0.40, -0.08, 0.08) * 0.35), float(np.clip(mi[1] - 0.40, -0.06, 0.06) * 0.2), float(np.clip(mi[2] - 0.40, -0.08, 0.08) * 0.3)),
        highlights=(float(np.clip(hi[0] - hi[2], -0.10, 0.10) * 0.5), float(np.clip(hi[1] - 0.78, -0.06, 0.06) * 0.2), float(np.clip(hi[2] - hi[0], -0.10, 0.10) * 0.35)),
    )
    hsl: dict[str, HueShift] = {}
    for name, deg in HUE_CENTERS.items():
        band = _hue_mask(h, s, l, deg)
        if float(band.mean()) < 0.012:
            continue
        sat_d = float(s[band].mean() - tech.saturation_mean)
        lum_d = float(l[band].mean() - float(y.mean()))
        hsl[name] = HueShift(
            sat=float(np.clip(sat_d * 1.1, -0.35, 0.28)),
            lum=float(np.clip(lum_d * 0.6, -0.22, 0.18)),
        )
    hh, ww = y.shape
    pad_h, pad_w = max(1, hh // 8), max(1, ww // 8)
    center = float(y[pad_h : hh - pad_h, pad_w : ww - pad_w].mean())
    corners = float(
        np.mean(
            [
                y[:pad_h, :pad_w].mean(),
                y[:pad_h, -pad_w:].mean(),
                y[-pad_h:, :pad_w].mean(),
                y[-pad_h:, -pad_w:].mean(),
            ]
        )
    )
    vignette = float(np.clip((center - corners) * 1.3, 0.0, 0.35))
    gy = np.abs(np.diff(y, axis=0, append=y[-1:]))
    gx = np.abs(np.diff(y, axis=1, append=y[:, -1:]))
    local = float(np.mean(gx + gy))
    return Recipe(
        temperature=float(np.clip(tech.temperature_bias * 42.0, -36, 36)),
        tint=float(np.clip((mi[1] - 0.5 * (mi[0] + mi[2])) * 80.0, -18, 18)),
        highlights=float(np.clip((0.88 - tech.highlight_p95) * 0.9, -0.35, 0.12)),
        shadows=float(np.clip((0.08 - tech.shadow_p05) * 0.9, -0.18, 0.18)),
        whites=float(np.clip((tech.highlight_p95 - 0.90) * 0.7, -0.16, 0.14)),
        blacks=float(np.clip((0.05 - tech.shadow_p05) * 1.4, -0.32, 0.10)),
        contrast=float(np.clip((tech.std_luma - 0.16) * 1.35, -0.12, 0.32)),
        saturation=float(np.clip((tech.saturation_mean - 0.27) * 0.55, -0.22, 0.16)),
        vibrance=float(np.clip((tech.saturation_mean - 0.24) * 0.35, -0.10, 0.16)),
        clarity=float(np.clip((local - 0.055) * 2.2, -0.08, 0.28)),
        dehaze=float(np.clip((tech.dynamic_spread - 0.50) * 0.25, 0.0, 0.18)),
        vignette=vignette,
        grain=0.04 if tech.mean_luma < 0.22 else 0.02,
        hsl=hsl,
        grade=grade,
    ).clamped()


def _mean_rgb(rgb: np.ndarray, mask: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    if not np.any(mask):
        return np.array(fallback, dtype=np.float32)
    return rgb[mask].mean(axis=0).astype(np.float32)


def _hue_mask(h: np.ndarray, s: np.ndarray, l: np.ndarray, degrees: float) -> np.ndarray:
    center = (degrees / 360.0) % 1.0
    dist = np.abs(h - center)
    dist = np.minimum(dist, 1.0 - dist)
    return (dist < 0.07) & (s > 0.08) & (l > 0.08) & (l < 0.92)


def build_index(limit: int | None = None) -> Path:
    catalog_path = _awards_dir() / "catalog.json"
    items = []
    if catalog_path.is_file():
        items = json.loads(catalog_path.read_text(encoding="utf-8")).get("items", [])
    by_file = {it.get("file", "").split("/")[-1]: it for it in items}
    frames: list[dict] = []
    folder = _masters_dir()
    files = sorted(folder.glob("*.jpg"))
    if limit:
        files = files[:limit]
    for path in files:
        meta = by_file.get(path.name, {})
        photographer = (meta.get("photographer") or "").strip()
        if not photographer or photographer.lower() in {"unknown", "hasselblad"}:
            continue
        year = str(meta.get("year") or "")
        look_id = "hb:" + photographer.lower().replace(" ", "-").removeprefix("master-")
        linear = jpeg_bytes_to_linear(path)
        tech = analyze(linear, path)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((768, 768))
            srgb = np.asarray(im, dtype=np.float32) / 255.0
        recipe = recipe_from_print(srgb, tech)
        env = environment_label(tech)
        frames.append(
            {
                "id": f"{look_id}:{path.stem}",
                "look_id": look_id,
                "file": f"hasselblad_masters/{path.name}",
                "photographer": photographer,
                "year": year,
                "genre": guess_genre(tech),
                "environment": env,
                "scene": {k: round(float(getattr(tech, k)), 4) for k in _SCENE_KEYS},
                "recipe": json.loads(recipe.model_dump_json()),
            }
        )
    looks: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = {}
    for frame in frames:
        grouped.setdefault(frame["look_id"], []).append(frame)
    for look_id, group in grouped.items():
        photographer = group[0]["photographer"]
        year = group[0]["year"]
        env = _mode(f["environment"] for f in group)
        genres = sorted({f["genre"] for f in group})
        recipe = _mean_recipe([Recipe.model_validate(f["recipe"]) for f in group])
        looks[look_id] = {
            "title": f"哈苏 · {photographer.title()}",
            "title_en": photographer.title(),
            "description": (
                f"Hasselblad Masters {year or ''} {photographer.title()} 组图"
                f"（{env} / {', '.join(genres)}）。按相近题材和光线做同类后期，不是套 LUT。"
            ).strip(),
            "photographer": photographer,
            "year": year,
            "genre_tags": genres or ["general"],
            "environment": env,
            "n": len(group),
            "recipe": json.loads(recipe.model_dump_json()),
        }
    payload = {"frames": frames, "looks": looks}
    dest = _awards_dir() / INDEX_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    bundled = Path(__file__).with_name(INDEX_NAME)
    bundled.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    load_index.cache_clear()
    return dest


def _mode(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else "general"


def _mean_recipe(recipes: list[Recipe]) -> Recipe:
    if not recipes:
        return Recipe()
    n = float(len(recipes))
    acc = Recipe()
    for recipe in recipes:
        acc = acc.merged(recipe)
    return _scale(acc, 1.0 / n).clamped()
