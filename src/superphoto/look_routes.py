"""Map a photo to a coloring route using 调色素材 practice images.

Each XMP look was authored against a folder of 练习图/例图/修改前. Those
pictures — not the finished 成片 — are the visual definition of "when to
use this look". Centroids below were measured from those files.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .models import TechnicalReport

_KEYS = (
    "mean_luma",
    "std_luma",
    "saturation_mean",
    "temperature_bias",
    "top_third_blue",
    "skin_fraction",
    "green_fraction",
    "aqua_fraction",
    "orange_fraction",
    "magenta_fraction",
    "shadow_clip",
    "highlight_clip",
    "dynamic_spread",
)
_WEIGHTS = {
    "mean_luma": 1.5,
    "temperature_bias": 1.2,
    "skin_fraction": 1.3,
    "green_fraction": 1.7,
    "orange_fraction": 1.5,
    "aqua_fraction": 1.3,
    "top_third_blue": 0.9,
    "saturation_mean": 0.8,
    "shadow_clip": 0.7,
    "std_luma": 0.5,
    "magenta_fraction": 0.5,
    "highlight_clip": 0.3,
    "dynamic_spread": 0.4,
}

WHEN = {
    "skam": "日光街头或开阔外景，人站在灰蓝天空下，地面偏冷",
    "tokyo_ghoul": "很暗的城市夜景：湿路、车灯、剪影，几乎没有白天",
    "romance_gold": "暖金夜景或室内钨丝灯，暗部多、橙子多",
    "zhou_chu": "偏暗的写实街景或室内，饱和不高，暖但不金",
    "dune": "沙漠、干土、沙丘：高橙、几乎没有绿叶或水体",
    "kdrama_tears": "能看清人脸的日间外景或城市，肤色可见，天空偏蓝",
    "downton": "建筑、庭院、冷蓝天空，庄园或欧式空间",
    "jp_drama": "中暗的日系街巷或室内，不要套到大面积自然山水",
    "shameless": "脏暖的城市夜晚或室内，阴影多",
    "resident_evil": "灰、对比强、饱和低的外景，末日感而不是翠绿色自然",
    "wong_kar_wai": "很暗很暖的夜街、霓虹或钨丝室内",
    "jujutsu": "中对比的城市/角色画面，偏动漫剧照",
    "natsume": "日系清淡人像或樱花，不是湖山植被风光",
    "unnatural": "冷白走廊、医院或低饱和室内",
    "hanabi": "夜空烟火或极暗高光点，几乎没有白天景物",
    "five_cm": "樱花、暖窗、日系天空，洋红或粉偏多",
    "landscape_filmic": "自然风光：绿或蓝占比高，不是沙漠、也不是夜街",
    "filmic_portrait_warm": "人脸占比高的人像或半身",
    "cinematic_teal_orange": "城市或人物，中暗，可走青橙电影感",
    "editorial_crisp": "干净光线的人像或产品",
    "moody_lowkey": "低亮度、大面积阴影的情绪片",
}

_BUILTIN = {
    "landscape_filmic": {
        "n": 0,
        "genre": "landscape",
        "mean": {
            "mean_luma": 0.38,
            "std_luma": 0.20,
            "saturation_mean": 0.36,
            "temperature_bias": 0.02,
            "top_third_blue": 0.32,
            "skin_fraction": 0.04,
            "green_fraction": 0.22,
            "aqua_fraction": 0.12,
            "orange_fraction": 0.08,
            "magenta_fraction": 0.02,
            "shadow_clip": 0.04,
            "highlight_clip": 0.01,
            "dynamic_spread": 0.62,
        },
        "std": {k: 0.12 for k in _KEYS},
    },
    "filmic_portrait_warm": {
        "n": 0,
        "genre": "portrait",
        "mean": {
            "mean_luma": 0.34,
            "std_luma": 0.18,
            "saturation_mean": 0.30,
            "temperature_bias": 0.12,
            "top_third_blue": 0.12,
            "skin_fraction": 0.28,
            "green_fraction": 0.03,
            "aqua_fraction": 0.04,
            "orange_fraction": 0.22,
            "magenta_fraction": 0.04,
            "shadow_clip": 0.08,
            "highlight_clip": 0.01,
            "dynamic_spread": 0.55,
        },
        "std": {k: 0.12 for k in _KEYS},
    },
    "cinematic_teal_orange": {
        "n": 0,
        "genre": "street",
        "mean": {
            "mean_luma": 0.20,
            "std_luma": 0.18,
            "saturation_mean": 0.34,
            "temperature_bias": 0.06,
            "top_third_blue": 0.18,
            "skin_fraction": 0.10,
            "green_fraction": 0.04,
            "aqua_fraction": 0.08,
            "orange_fraction": 0.14,
            "magenta_fraction": 0.03,
            "shadow_clip": 0.22,
            "highlight_clip": 0.01,
            "dynamic_spread": 0.58,
        },
        "std": {k: 0.12 for k in _KEYS},
    },
    "editorial_crisp": {
        "n": 0,
        "genre": "portrait",
        "mean": {
            "mean_luma": 0.42,
            "std_luma": 0.16,
            "saturation_mean": 0.24,
            "temperature_bias": -0.04,
            "top_third_blue": 0.10,
            "skin_fraction": 0.22,
            "green_fraction": 0.02,
            "aqua_fraction": 0.03,
            "orange_fraction": 0.16,
            "magenta_fraction": 0.02,
            "shadow_clip": 0.03,
            "highlight_clip": 0.01,
            "dynamic_spread": 0.50,
        },
        "std": {k: 0.12 for k in _KEYS},
    },
    "moody_lowkey": {
        "n": 0,
        "genre": "night",
        "mean": {
            "mean_luma": 0.10,
            "std_luma": 0.14,
            "saturation_mean": 0.26,
            "temperature_bias": -0.04,
            "top_third_blue": 0.16,
            "skin_fraction": 0.04,
            "green_fraction": 0.04,
            "aqua_fraction": 0.05,
            "orange_fraction": 0.06,
            "magenta_fraction": 0.02,
            "shadow_clip": 0.35,
            "highlight_clip": 0.00,
            "dynamic_spread": 0.40,
        },
        "std": {k: 0.12 for k in _KEYS},
    },
}


@lru_cache(maxsize=1)
def _measured() -> dict[str, dict]:
    path = Path(__file__).with_name("look_routes.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def routes() -> dict[str, dict]:
    data = dict(_measured())
    data.update(_BUILTIN)
    return data


def route_when(look_id: str) -> str:
    return WHEN.get(look_id, "")


def route_legend() -> str:
    return "\n".join(f"- {lid}: {WHEN[lid]}" for lid in WHEN)


def route_similarity(look_id: str, technical: TechnicalReport, *, apply_veto: bool = True) -> float | None:
    """0..1 how close this photo is to the look's practice set. None if unknown."""
    spec = routes().get(look_id)
    if spec is None:
        return None
    if apply_veto and _veto(look_id, technical):
        return 0.0
    acc = 0.0
    wsum = 0.0
    for key, weight in _WEIGHTS.items():
        mu = float(spec["mean"][key])
        sd = max(float(spec["std"][key]), 0.04)
        z = abs(float(getattr(technical, key)) - mu) / sd
        acc += weight * z
        wsum += weight
    dist = acc / max(wsum, 1e-6)
    return float(np.exp(-dist / 2.05))


def _veto(look_id: str, t: TechnicalReport) -> bool:
    if look_id == "dune" and (t.green_fraction > 0.12 or t.aqua_fraction > 0.12):
        return True
    if look_id == "hanabi" and t.mean_luma > 0.16:
        return True
    if look_id == "tokyo_ghoul" and t.mean_luma > 0.24:
        return True
    if look_id == "natsume" and (t.green_fraction > 0.14 or (t.aqua_fraction > 0.10 and t.skin_fraction < 0.08)):
        return True
    if look_id in {"kdrama_tears", "romance_gold", "filmic_portrait_warm"}:
        if t.skin_fraction < 0.05 and t.green_fraction > 0.14:
            return True
    if look_id == "jp_drama" and t.green_fraction > 0.16 and t.top_third_blue > 0.28:
        return True
    if look_id == "five_cm" and t.green_fraction > 0.20 and t.aqua_fraction > 0.12:
        return True
    if look_id == "landscape_filmic" and t.orange_fraction > 0.40 and t.green_fraction < 0.04:
        return True
    return False
