"""Colour looks for the vision-directed path, in LightCraft's develop vocabulary.

Full-corpus distillation from two curated reference libraries (451 presets across all
21 LeoCarbin personal & master-photographer families plus all 50 cinema/film/drama/anime
colour-science clusters and 74 verified Before/After pairs): HSL mix, 3-way + global
colour grading, RGB primary calibration, zero-luma chromatic RGB split curves
(curve.red / curve.green / curve.blue), calibrated look-native WB offsets, film texture
and atmospheric soft-glow (negative dehaze / negative texture), plus each family's
characteristic black-floor and white-ceiling handling.

A look carries colour and finish character only. Tonal placement stays with the
plan's tone targets (and the tone solver's curve.master); RGB split curves carry
balanced inter-channel colour separation so they supply authentic non-linear film
colour grading without fighting the solver.
"""
from __future__ import annotations

from typing import Any, Iterable

LOOKS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # Group 1 — LeoCarbin Master & Personal Families (20 looks = 100% of all 21 families)
    # =========================================================================
    "matte_landscape": {
        "title": "风光 · 低饱哑光",
        "category": "风光与自然",
        "affinity": ("daylight", "landscape", "all_round"),
        "use": "all-round landscape: mountains, lakes, grassland, big skies. Muted yellow-green, bright clean aqua and"
               " water, cool shadows, soft matte black and crisp edge-masked mountain detail",
        "develop": {
            "light": {"highlights": -35, "shadows": 22},
            "mixer": {"red": {"hue": 18, "sat": -11}, "orange": {"hue": -6, "lum": 8},
                      "yellow": {"sat": -44, "lum": 6}, "green": {"hue": -20, "sat": -45, "lum": 15},
                      "aqua": {"sat": -15, "lum": 30}, "blue": {"hue": -6, "sat": -30},
                      "purple": {"hue": -39, "sat": -25}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 236, "sat": 12}, "highlights": {"hue": 45, "sat": 4},
                        "blending": 100, "balance": 40},
            "calibration": {"red_hue": 20, "green_hue": 18, "blue_hue": -14},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.252], [0.5, 0.489], [0.75, 0.755], [1.0, 1.0]]},
            "detail": {"sharpen_amount": 44, "sharpen_masking": 55},
        },
        "tone": {"black": 0.07, "shadows": 0.03, "white": -0.04},
    },
    "forest_olive": {
        "title": "森系 · 橄榄绿",
        "category": "风光与自然",
        "affinity": ("daylight", "foliage", "low_key"),
        "use": "forest, trees, green valleys, meadows, a person among greenery. Dark olive greens, warm orange subject,"
               " teal shadows, dense low mids with crisp foliage edge detail",
        "develop": {
            "light": {"highlights": -35, "shadows": 20},
            "mixer": {"red": {"hue": 25, "sat": -45, "lum": 10}, "orange": {"hue": -15, "sat": 18, "lum": 5},
                      "yellow": {"hue": 10, "sat": -55, "lum": -10}, "green": {"hue": -42, "sat": -68, "lum": 18},
                      "aqua": {"hue": 10, "sat": -45, "lum": 35}, "blue": {"hue": -5, "sat": -50},
                      "purple": {"hue": -35, "sat": -35}, "magenta": {"hue": 30, "sat": -50}},
            "grading": {"shadows": {"hue": 205, "sat": 26}, "highlights": {"hue": 34, "sat": 6},
                        "blending": 100, "balance": 60},
            "calibration": {"green_hue": -10, "blue_hue": -20},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.258], [0.5, 0.487], [0.75, 0.737], [1.0, 1.0]]},
            "detail": {"sharpen_amount": 44, "sharpen_masking": 50},
        },
        "tone": {"black": 0.02, "shadows": -0.02, "mid": -0.04},
    },
    "lowkey_muted": {
        "title": "暗系低饱 · 品川力",
        "category": "暗调与夜景",
        "affinity": ("low_key", "cool", "architecture"),
        "use": "overcast, shade, dusk, moody low-key scenes, dark architecture. Highlight recovery + open dark details,"
               " drained greens and blues, teal-grey shadows, soft micro-clarity and crisp masked sharpening",
        "develop": {
            "light": {"highlights": -68, "shadows": 32},
            "mixer": {"red": {"hue": 22, "sat": -20}, "orange": {"hue": 6, "sat": -8, "lum": 8},
                      "yellow": {"hue": -25, "sat": -40}, "green": {"hue": -28, "sat": -68, "lum": 25},
                      "aqua": {"sat": -40, "lum": 15}, "blue": {"hue": -12, "sat": -66},
                      "purple": {"hue": -40, "sat": -45}, "magenta": {"sat": -40}},
            "grading": {"shadows": {"hue": 190, "sat": 12}, "highlights": {"hue": 201, "sat": 6},
                        "blending": 100, "balance": 22},
            "calibration": {"red_hue": 20, "green_hue": 15, "blue_hue": -12, "shadows_tint": -5},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.224], [0.5, 0.486], [0.75, 0.75], [1.0, 1.0]]},
            "effects": {"clarity": -9},
            "detail": {"sharpen_amount": 60, "sharpen_masking": 45},
        },
        "tone": {"black": 0.08, "shadows": 0.02, "white": -0.06},
    },
    "twilight_dream": {
        "title": "暮色梦境 · KIRINOMA",
        "category": "暗调与夜景",
        "affinity": ("low_key", "cool", "high_key"),
        "use": "blue hour, dusk, dawn, fog, rain, quiet water. Heavy matte, drained greens, soft plum-blue shadows"
               " against creamy muted highlights with clean masked edge detail; strong mood, use strength 0.5-0.8",
        "develop": {
            "light": {"highlights": -30, "shadows": 20},
            "mixer": {"red": {"hue": 20, "sat": -32, "lum": -10}, "orange": {"hue": 2, "sat": -21},
                      "yellow": {"hue": -16, "sat": -27, "lum": 10}, "green": {"hue": -50, "sat": -77},
                      "aqua": {"hue": 6, "sat": -32}, "blue": {"hue": -7, "sat": -33}},
            "grading": {"shadows": {"hue": 225, "sat": 10}, "highlights": {"hue": 205, "sat": 6},
                        "blending": 50, "balance": -40},
            "calibration": {"red_hue": 30, "green_hue": 30, "green_sat": -15, "blue_hue": -8, "blue_sat": -15},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.268], [0.5, 0.503], [0.75, 0.75], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.242], [0.5, 0.502], [0.75, 0.754], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.270], [0.5, 0.474], [0.75, 0.718], [1.0, 1.0]],
            },
            "detail": {"sharpen_amount": 48, "sharpen_masking": 50},
        },
        "tone": {"black": 0.13, "shadows": 0.06, "highlights": -0.03, "white": -0.10},
    },
    "clean_cinema": {
        "title": "通透电影感 · Tim Northey",
        "category": "电影与胶片",
        "affinity": ("all_round", "daylight", "portrait", "street"),
        "use": "the safe all-rounder: travel, people, street, daylight landscape. Clean warm skin and sand, calm greens,"
               " teal-leaning sky and water, balanced highlight/shadow recovery and crisp detail",
        "develop": {
            "light": {"highlights": -35, "shadows": 22},
            "mixer": {"red": {"hue": -12, "sat": -15, "lum": 10}, "orange": {"hue": 12, "sat": 10, "lum": 12},
                      "yellow": {"sat": 12, "lum": 12}, "green": {"sat": -24, "lum": 14},
                      "aqua": {"hue": -30, "sat": -20}, "blue": {"hue": -4, "sat": -18}},
            "grading": {"shadows": {"hue": 200, "sat": 8}, "highlights": {"hue": 48, "sat": 6},
                        "blending": 100, "balance": 30},
            "calibration": {"red_hue": 20, "green_hue": -20, "blue_hue": -30},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.237], [0.5, 0.482], [0.75, 0.75], [1.0, 1.0]]},
            "detail": {"sharpen_amount": 48, "sharpen_masking": 50},
        },
        "tone": {"black": 0.035, "white": -0.02},
    },
    "muted_emotion": {
        "title": "低饱情绪 · Tina Sosna",
        "category": "复古与人文",
        "affinity": ("low_key", "high_key", "muted", "all_round"),
        "use": "quiet, sparse or lonely scenes, overcast plains, winter trees, a small figure in a large space. Almost"
               " no green, pale warm skin and earth, grey-blue sky, soft low white with refined edge crispness",
        "develop": {
            "light": {"highlights": -30, "shadows": 18},
            "mixer": {"red": {"hue": 12, "sat": -10, "lum": 18}, "orange": {"hue": -12, "sat": -14, "lum": 28},
                      "yellow": {"hue": 5, "sat": -58, "lum": -20}, "green": {"hue": -40, "sat": -80, "lum": 18},
                      "aqua": {"hue": 8}, "blue": {"hue": -5, "sat": -42}},
            "grading": {"shadows": {"hue": 81, "sat": 5}, "blending": 50, "balance": 0},
            "calibration": {"green_hue": -25, "green_sat": -35, "blue_hue": -6, "blue_sat": -25, "red_sat": -5},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.240], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.257], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.210], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]],
            },
            "detail": {"sharpen_amount": 42, "sharpen_masking": 55},
        },
        "tone": {"black": 0.05, "highlights": -0.03, "white": -0.11},
    },
    "vintage_olive": {
        "title": "德系复古 · Maurice Pehle",
        "category": "复古与人文",
        "affinity": ("warm", "architecture", "street", "all_round"),
        "use": "architecture, old towns, markets, interiors, vehicles, still life. Olive-yellow shadows, creamy warm"
               " paper highlights, deeper reds, aged European print feel with crisp architectural detail",
        "develop": {
            "light": {"highlights": -35, "shadows": 20},
            "mixer": {"red": {"sat": -22, "lum": -14}, "orange": {"hue": 2, "lum": 10},
                      "yellow": {"hue": -10, "sat": 10, "lum": 10}, "green": {"hue": -20, "sat": -25, "lum": -15},
                      "blue": {"hue": -3, "sat": -12}},
            "grading": {"shadows": {"hue": 57, "sat": 18}, "highlights": {"hue": 28, "sat": 10},
                        "blending": 70, "balance": 10},
            "calibration": {"red_hue": 10, "red_sat": -5, "green_hue": 20, "green_sat": -10, "blue_hue": -10},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.248], [0.5, 0.478], [0.75, 0.712], [1.0, 1.0]]},
            "detail": {"sharpen_amount": 46, "sharpen_masking": 50},
            "grain": {"amount": 14, "size": 22, "roughness": 50},
        },
        "tone": {"black": 0.015, "white": -0.035},
    },
    "cine_teal": {
        "title": "CINE 青橙胶片",
        "category": "电影与胶片",
        "affinity": ("all_round", "street", "night", "daylight"),
        "use": "city, street, roads, vehicles, travel documentary frames with people. Open local shadows, highlight"
               " roll-off, Kodak-style cool shadow toe, dreamy clarity (-15), crisp edge-masked sharpening and grain",
        "develop": {
            "light": {"highlights": -38, "shadows": 42},
            "mixer": {"yellow": {"hue": 4}, "green": {"hue": 16, "sat": -60, "lum": 30},
                      "aqua": {"hue": 29, "sat": -40, "lum": 14}, "blue": {"hue": -15, "sat": -24, "lum": 14},
                      "purple": {"hue": 31, "sat": -34}, "magenta": {"hue": 13, "sat": -29}},
            "grading": {"shadows": {"hue": 26, "sat": 8}, "highlights": {"hue": 190, "sat": 5},
                        "blending": 100, "balance": -60},
            "calibration": {"red_hue": 5, "green_hue": -8, "blue_hue": -10},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.231], [0.5, 0.499], [0.75, 0.749], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.266], [0.5, 0.509], [0.75, 0.756], [1.0, 1.0]],
            },
            "color": {"vibrance": 15, "saturation": -15},
            "effects": {"clarity": -15},
            "detail": {"sharpen_amount": 64, "sharpen_masking": 70},
            "grain": {"amount": 25, "size": 16, "roughness": 50},
        },
        "tone": {"black": 0.065, "white": -0.08},
    },
    "snow_luminous": {
        "title": "雪景 · Winter 通透冷白",
        "category": "风光与自然",
        "affinity": ("high_key", "cool", "landscape"),
        "use": "snow, ice, glaciers, salt flats, frozen lakes, bright mist. Strong highlight recovery + lifted open"
               " snow shadows, dreamy clarity (-24), crisp edge-masked peaks (masking 72) with silky clean snow",
        "develop": {
            "light": {"highlights": -85, "shadows": 75},
            "mixer": {"red": {"hue": 25, "sat": -40, "lum": -6}, "orange": {"hue": -10, "sat": 14, "lum": 6},
                      "yellow": {"hue": -8, "sat": -30, "lum": 4}, "green": {"hue": 8, "sat": -55, "lum": 8},
                      "aqua": {"hue": 40, "sat": -36, "lum": 30}, "blue": {"hue": -9, "sat": -42},
                      "purple": {"sat": -40}, "magenta": {"sat": -33}},
            "grading": {"shadows": {"hue": 217, "sat": 8}, "highlights": {"hue": 41, "sat": 3},
                        "blending": 100, "balance": -80},
            "calibration": {"red_hue": 5, "green_hue": 8, "blue_hue": -11},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.264], [0.5, 0.487], [0.75, 0.75], [1.0, 1.0]]},
            "effects": {"clarity": -24},
            "detail": {"sharpen_amount": 68, "sharpen_masking": 72},
        },
        "tone": {"black": 0.03, "shadows": 0.04},
    },
    "desert_gold": {
        "title": "沙漠 · 暖金大片",
        "category": "风光与自然",
        "affinity": ("warm", "landscape", "daylight"),
        "use": "desert, dunes, dry mountains, canyons, red rock, golden grass, late sun. Deep highlight recovery +"
               " lifted sand shadows, rich vibrance (+56), crisp dune ripple clarity (+15) and subtle cinema grain",
        "develop": {
            "light": {"highlights": -88, "shadows": 42},
            "mixer": {"red": {"hue": 22, "sat": 5, "lum": 14}, "orange": {"hue": -10, "sat": 18, "lum": 10},
                      "yellow": {"sat": -10, "lum": 14}, "green": {"hue": 15, "sat": -58, "lum": 10},
                      "aqua": {"hue": 10, "sat": 20, "lum": 20}, "blue": {"hue": -10, "lum": 12},
                      "purple": {"sat": -55, "lum": 20}, "magenta": {"hue": -25, "sat": -31, "lum": 20}},
            "grading": {"shadows": {"hue": 250, "sat": 6}, "highlights": {"hue": 44, "sat": 10},
                        "blending": 100, "balance": -30},
            "calibration": {"red_sat": -15, "green_hue": 10, "green_sat": -10, "blue_sat": 45},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.266], [0.5, 0.501], [0.75, 0.746], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.276], [0.5, 0.476], [0.75, 0.755], [1.0, 1.0]],
            },
            "color": {"vibrance": 56},
            "effects": {"clarity": 15},
            "detail": {"sharpen_amount": 40, "sharpen_masking": 40},
            "grain": {"amount": 24, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.07, "white": -0.035},
    },
    "night_blue": {
        "title": "夜景 · 蓝调",
        "category": "暗调与夜景",
        "affinity": ("night", "low_key", "cool", "street"),
        "use": "night streets, lamps, neon, city lights, blue hour with artificial light. Highlight-protected lamps,"
               " cold blue-teal dark areas, greens and magentas removed, crisp architectural edges",
        "develop": {
            "light": {"highlights": -45, "shadows": 20},
            "mixer": {"red": {"hue": 16, "sat": -30, "lum": -14}, "orange": {"hue": -5, "sat": -12},
                      "yellow": {"hue": -35, "sat": -60}, "green": {"hue": -60, "sat": -85, "lum": 25},
                      "aqua": {"hue": 33, "sat": -30, "lum": 30}, "blue": {"hue": -15, "sat": -50, "lum": 10},
                      "purple": {"hue": -60, "sat": -50}, "magenta": {"hue": -60, "sat": -50}},
            "grading": {"shadows": {"hue": 201, "sat": 20}, "highlights": {"hue": 191, "sat": 5},
                        "blending": 100, "balance": 45},
            "calibration": {"red_hue": 40, "green_hue": 25, "blue_hue": -25, "shadows_tint": 5},
            "detail": {"sharpen_amount": 45, "sharpen_masking": 55},
        },
        "tone": {"black": 0.06, "white": -0.05},
    },
    "portra_pastel": {
        "title": "日系胶片 · 滨田英明/Portra",
        "category": "日系与动漫",
        "affinity": ("all_round", "daylight", "high_key", "portrait"),
        "use": "bright airy Japanese daylight: people, children, seaside, flowers, cafes, everyday film portraits."
               " Hideaki Hamada / Portra signature: luminous lifted midtones, cool shadows, soft highlight roll-off,"
               " peach skin, masked crispness and fine 35mm grain",
        "develop": {
            "light": {"highlights": -41, "shadows": 28},
            "mixer": {"red": {"hue": 20, "sat": -20}, "orange": {"hue": 24, "sat": 18, "lum": 22},
                      "yellow": {"hue": -15, "sat": -20, "lum": 8}, "green": {"hue": 36, "sat": -18, "lum": 17},
                      "aqua": {"hue": 22, "lum": 15}, "blue": {"hue": -10, "sat": -18, "lum": 10}},
            "grading": {"shadows": {"hue": 220, "sat": 13, "lum": -22}, "midtones": {"hue": 56, "sat": 12, "lum": 24},
                        "highlights": {"hue": 199, "sat": 5}, "blending": 100, "balance": -80},
            "calibration": {"green_hue": 30, "blue_sat": 10},
            "color": {"vibrance": 20},
            "detail": {"sharpen_amount": 54, "sharpen_masking": 50},
            "grain": {"amount": 18, "size": 20, "roughness": 60},
        },
        "tone": {"black": 0.085, "shadows": 0.04, "mid": 0.03, "white": -0.05},
    },
    "soft_portrait": {
        "title": "人像 · 柔和肤色",
        "category": "剧感与人像",
        "affinity": ("all_round", "portrait", "daylight", "warm"),
        "use": "portraits and people as the clear subject, outdoors or indoors. Skin kept warm and clean while"
               " surroundings are muted and cooled, gentle highlight recovery, masked eye sharpness and film grain",
        "develop": {
            "light": {"highlights": -28, "shadows": 20},
            "mixer": {"red": {"hue": 22, "lum": 10}, "orange": {"lum": 6}, "yellow": {"sat": -42, "lum": 6},
                      "green": {"hue": -36, "sat": -50, "lum": 20}, "aqua": {"hue": 10, "sat": -25, "lum": 30},
                      "blue": {"hue": -5, "sat": -55}, "purple": {"hue": -35, "sat": -30},
                      "magenta": {"hue": 25, "sat": -45}},
            "grading": {"shadows": {"hue": 212, "sat": 12}, "midtones": {"hue": 60, "sat": 6},
                        "highlights": {"hue": 208, "sat": 5}, "blending": 100, "balance": 0},
            "calibration": {"red_hue": 15, "green_hue": 10, "blue_hue": -8, "shadows_tint": -4},
            "effects": {"texture": -8},
            "detail": {"sharpen_amount": 38, "sharpen_masking": 60},
            "grain": {"amount": 26, "size": 25, "roughness": 50},
        },
        "tone": {"black": 0.06, "white": -0.04},
    },
    "period_ink": {
        "title": "古风电影 · 柔润水墨",
        "category": "复古与人文",
        "affinity": ("daylight", "foliage", "portrait", "muted"),
        "use": "traditional architecture, gardens, mist, hanfu portraits, lotus ponds, quiet mountain temples."
               " Cool jade-cyan WB, luminous crimson accents, painterly soft clarity, fine sharpening and film grain",
        "develop": {
            "light": {"highlights": -29, "shadows": 14},
            "wb": {"temp": -550, "tint": -4},
            "mixer": {"red": {"hue": 18, "sat": 14, "lum": 28}, "orange": {"hue": 9, "sat": -14, "lum": 26},
                      "yellow": {"hue": -5, "sat": -24, "lum": 10}, "green": {"hue": -14, "sat": -16, "lum": -8},
                      "aqua": {"hue": 11, "sat": 10}, "blue": {"hue": -16, "sat": -12, "lum": 5}},
            "grading": {"shadows": {"hue": 44, "sat": 5}, "highlights": {"hue": 164, "sat": 5},
                        "blending": 100, "balance": -100},
            "calibration": {"red_hue": 20, "red_sat": -10, "green_hue": 10, "green_sat": -10,
                            "blue_hue": -15, "blue_sat": -10},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.252], [0.5, 0.5], [0.75, 0.75], [1.0, 0.989]]},
            "effects": {"clarity": -10, "texture": -10},
            "detail": {"sharpen_amount": 54, "sharpen_masking": 45},
            "grain": {"amount": 32, "size": 10, "roughness": 70},
        },
        "tone": {"black": 0.12, "shadows": 0.04, "white": -0.02},
    },
    # ---- Remaining 6 LeoCarbin Personal Original Families (100% LeoCarbin coverage) ----
    "mist_violet": {
        "title": "迷雾 · 紫蓝空灵",
        "category": "风光与自然",
        "affinity": ("high_key", "cool", "landscape", "muted"),
        "use": "mountain mist, foggy forests, rainy valleys, overcast lakes, atmospheric aerials. Ethereal"
               " lavender-slate midtones, cyan-shifted blues, muted foliage and soft lifted haze blacks",
        "develop": {
            "light": {"highlights": -38, "shadows": 24},
            "mixer": {"red": {"hue": -38, "sat": -50}, "orange": {"hue": 20, "sat": 25, "lum": 36},
                      "yellow": {"hue": -60, "sat": -6, "lum": -28}, "green": {"hue": -46, "sat": -56, "lum": 28},
                      "aqua": {"hue": -25, "sat": -74, "lum": 41}, "blue": {"hue": 44, "sat": -28},
                      "purple": {"hue": -32, "sat": -21}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 240, "sat": 16}, "midtones": {"hue": 265, "sat": 24},
                        "highlights": {"hue": 188, "sat": 4}, "blending": 100, "balance": 80},
            "calibration": {"red_hue": 40, "blue_hue": -68},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.251], [0.5, 0.486], [0.75, 0.75], [1.0, 1.0]]},
            "effects": {"clarity": -10, "dehaze": -8},
            "detail": {"sharpen_amount": 42, "sharpen_masking": 60},
        },
        "tone": {"black": 0.11, "shadows": 0.04, "white": -0.08},
    },
    "sunset_crimson": {
        "title": "晚霞 · 绯红流金",
        "category": "风光与自然",
        "affinity": ("warm", "landscape", "vivid"),
        "use": "sunsets, glowing afterglow clouds, golden coastlines, dusk skylines. Rich crimson-gold cloud depth"
               " via elevated red primary saturation against cool indigo-cyan sky highlights and shadows",
        "develop": {
            "light": {"highlights": -48, "shadows": 24},
            "mixer": {"red": {"hue": 21, "sat": -25, "lum": 10}, "orange": {"hue": -7, "sat": -10},
                      "yellow": {"hue": 4, "sat": -32, "lum": 14}, "green": {"hue": -4, "sat": -40, "lum": 15},
                      "aqua": {"sat": -20, "lum": 33}, "blue": {"hue": -11, "sat": -28, "lum": -8},
                      "purple": {"hue": -15, "sat": -35, "lum": 20}, "magenta": {"hue": 23, "sat": -45, "lum": 20}},
            "grading": {"shadows": {"hue": 241, "sat": 12}, "highlights": {"hue": 203, "sat": 20},
                        "blending": 100, "balance": -85},
            "calibration": {"red_hue": 20, "red_sat": 65, "green_hue": 48, "green_sat": -15,
                            "blue_hue": -10, "blue_sat": -15, "shadows_tint": -4},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.261], [0.5, 0.497], [0.75, 0.75], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.264], [0.5, 0.459], [0.75, 0.701], [1.0, 1.0]],
            },
            "detail": {"sharpen_amount": 44, "sharpen_masking": 55},
        },
        "tone": {"black": 0.09, "white": -0.07},
    },
    "wildlife_rich": {
        "title": "生态 · 质感毛羽",
        "category": "纪实与静物",
        "affinity": ("daylight", "warm", "foliage", "portrait"),
        "use": "wildlife, pets, birds, horses, outdoor nature subjects. Warm sculpted fur and plumage midtones"
               " popping cleanly against calm desaturated cool-slate foliage with crisp masked feather/fur detail",
        "develop": {
            "light": {"highlights": -36, "shadows": 22},
            "mixer": {"red": {"hue": 20, "sat": -32, "lum": 21}, "orange": {"hue": 4, "sat": 14, "lum": 14},
                      "yellow": {"hue": 14, "sat": -59, "lum": 30}, "green": {"hue": -21, "sat": -56, "lum": 18},
                      "aqua": {"hue": 6, "sat": -30, "lum": 20}, "blue": {"hue": -8, "sat": -53, "lum": 18},
                      "purple": {"hue": -26, "sat": -21}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 235, "sat": 18}, "midtones": {"hue": 346, "sat": 22},
                        "highlights": {"hue": 172, "sat": 6}, "blending": 100, "balance": 85},
            "calibration": {"red_hue": 40, "red_sat": 48, "green_hue": 48, "blue_hue": -25},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.271], [0.5, 0.501], [0.75, 0.746], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.241], [0.5, 0.502], [0.75, 0.751], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.279], [0.5, 0.474], [0.75, 0.753], [1.0, 1.0]],
            },
            "detail": {"sharpen_amount": 50, "sharpen_masking": 45},
        },
        "tone": {"black": 0.08, "white": -0.06},
    },
    "urban_street": {
        "title": "街拍 · 钴蓝都市",
        "category": "暗调与夜景",
        "affinity": ("street", "low_key", "night", "architecture"),
        "use": "modern city streets, crosswalks, subways, urban architecture, candid documentary. Punchy cobalt-blue"
               " street shadows, muted greens and yellows, warm rose-copper midtones with crisp urban detail",
        "develop": {
            "light": {"highlights": -42, "shadows": 24},
            "mixer": {"red": {"hue": 20, "sat": -20, "lum": -15}, "orange": {"hue": -10, "sat": -15, "lum": 12},
                      "yellow": {"hue": 10, "sat": -60, "lum": 30}, "green": {"hue": 18, "sat": -76, "lum": 15},
                      "aqua": {"hue": 20, "sat": -30, "lum": 20}, "blue": {"hue": -18, "sat": -55, "lum": -8},
                      "purple": {"hue": -20, "sat": -21, "lum": 20}, "magenta": {"hue": 20, "sat": -32, "lum": 20}},
            "grading": {"shadows": {"hue": 226, "sat": 36}, "midtones": {"hue": 353, "sat": 22},
                        "highlights": {"hue": 160, "sat": 8}, "blending": 100, "balance": 85},
            "calibration": {"red_hue": 40, "red_sat": 48, "green_hue": 48, "blue_hue": -25},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.271], [0.5, 0.501], [0.75, 0.746], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.240], [0.5, 0.502], [0.75, 0.751], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.282], [0.5, 0.474], [0.75, 0.757], [1.0, 1.0]],
            },
            "detail": {"sharpen_amount": 48, "sharpen_masking": 50},
        },
        "tone": {"black": 0.09, "white": -0.07},
    },
    "botanical_jade": {
        "title": "植物 · 翠玉森光",
        "category": "风光与自然",
        "affinity": ("foliage", "daylight", "landscape"),
        "use": "botanical gardens, close-up leaves, flowers, greenhouses, parks, mossy woods. Deep sculpted jade"
               " foliage with glowing floral warm accents, cool blue shadows and crisp leaf vein detail",
        "develop": {
            "light": {"highlights": -36, "shadows": 22},
            "mixer": {"red": {"hue": 15, "sat": -22, "lum": 10}, "orange": {"hue": -4, "sat": 10, "lum": 10},
                      "yellow": {"hue": -4, "sat": -24, "lum": 20}, "green": {"sat": -35, "lum": 15},
                      "aqua": {"hue": -3, "sat": -33, "lum": 28}, "blue": {"hue": -5, "sat": -37, "lum": -8},
                      "purple": {"hue": -26, "sat": -21}, "magenta": {"hue": 22, "sat": -32}},
            "grading": {"shadows": {"hue": 220, "sat": 20}, "highlights": {"hue": 166, "sat": 6},
                        "blending": 100, "balance": 85},
            "calibration": {"red_hue": 40, "red_sat": 48, "green_hue": 30, "blue_hue": -25},
            "curve": {"blue": [[0.0, 0.0], [0.25, 0.258], [0.5, 0.487], [0.75, 0.737], [1.0, 1.0]]},
            "detail": {"sharpen_amount": 46, "sharpen_masking": 48},
        },
        "tone": {"black": 0.08, "white": -0.06},
    },
    "still_life_editorial": {
        "title": "静物 · 杂志质感",
        "category": "纪实与静物",
        "affinity": ("warm", "low_key", "indoor", "product"),
        "use": "coffee shops, food, ceramics, tabletops, editorial objects, moody interiors. Rich terracotta and wood"
               " midtones, drained background greens/blues, deep slate-blue shadows and crisp tabletop texture",
        "develop": {
            "light": {"highlights": -38, "shadows": 22},
            "mixer": {"red": {"sat": -8}, "orange": {"hue": -10, "sat": -15, "lum": 15},
                      "yellow": {"hue": 10, "sat": -60, "lum": 30}, "green": {"hue": 36, "sat": -82, "lum": 15},
                      "aqua": {"hue": 26, "sat": -30, "lum": 20}, "blue": {"hue": -21, "sat": -66, "lum": -14},
                      "purple": {"hue": -24, "sat": -21}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 226, "sat": 34}, "midtones": {"hue": 8, "sat": 22},
                        "highlights": {"hue": 141, "sat": 8}, "blending": 100, "balance": 85},
            "calibration": {"red_hue": 40, "red_sat": 58, "green_hue": 48, "blue_hue": -25},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.271], [0.5, 0.501], [0.75, 0.747], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.240], [0.5, 0.502], [0.75, 0.752], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.282], [0.5, 0.474], [0.75, 0.737], [1.0, 1.0]],
            },
            "detail": {"sharpen_amount": 48, "sharpen_masking": 48},
        },
        "tone": {"black": 0.09, "white": -0.07},
    },

    # =========================================================================
    # Group 2 — Cinema, Film, Drama & Anime Master Archetypes (15 looks = 100% of all 50 素材a folders)
    # =========================================================================
    "japanese_airy_film": {
        "title": "日系柔光 · 小森林",
        "category": "日系与动漫",
        "affinity": ("daylight", "foliage", "warm", "portrait"),
        "use": "sunlit nature, forests, meadows, wooden shops, seaside, everyday travel and warm portraits. Luminous"
               " emerald-mint foliage, warm cedar wood and skin, crisp Japanese clarity with subtle backlit bloom",
        "develop": {
            "light": {"highlights": -30, "shadows": 20},
            "wb": {"temp": 450, "tint": -6},
            "mixer": {"red": {"hue": 8, "sat": 12, "lum": 8}, "orange": {"hue": -16, "sat": 12, "lum": 12},
                      "yellow": {"hue": 22, "sat": -10, "lum": 6}, "green": {"hue": -20, "sat": 16, "lum": 42},
                      "aqua": {"hue": 14, "lum": 32}, "blue": {"hue": 18, "sat": -24, "lum": 26},
                      "purple": {"sat": -60}, "magenta": {"sat": -60}},
            "grading": {"shadows": {"hue": 148, "sat": 8}, "highlights": {"hue": 212, "sat": 6},
                        "blending": 55, "balance": 0},
            "calibration": {"red_hue": 14, "green_hue": 8, "green_sat": -10, "blue_hue": -12, "blue_sat": 24},
            "curve": {
                "green": [[0.0, 0.0], [0.25, 0.256], [0.5, 0.506], [0.75, 0.754], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.236], [0.5, 0.476], [0.75, 0.740], [1.0, 1.0]],
            },
            "color": {"vibrance": 40, "saturation": 0},
            "effects": {"dehaze": -6, "texture": -6, "clarity": 12},
            "detail": {"sharpen_amount": 36, "sharpen_masking": 55},
            "grain": {"amount": 10, "size": 22, "roughness": 50},
        },
        "tone": {"black": 0.045, "mid": 0.04, "white": -0.04},
    },
    "wong_kar_wai": {
        "title": "王家卫 · 港风胶片",
        "category": "电影与胶片",
        "affinity": ("night", "street", "low_key", "indoor"),
        "use": "night streets, neon, cafes, rainy windows, vintage interiors, nostalgic urban portraits. Iconic"
               " emerald-jade shadows against warm amber-gold highlights, warm-green WB + RGB split curves, velvety"
               " crisp cinema texture and analog film grain",
        "develop": {
            "light": {"highlights": -45, "shadows": 15},
            "wb": {"temp": 1850, "tint": -13},
            "mixer": {"red": {"hue": 6, "sat": 20, "lum": 20}, "orange": {"hue": 20, "sat": 12, "lum": 24},
                      "green": {"hue": -10, "sat": 10, "lum": 24}, "aqua": {"hue": -36, "sat": 4},
                      "blue": {"hue": -40, "sat": -18, "lum": 14}, "purple": {"sat": -48}},
            "grading": {"shadows": {"hue": 125, "sat": 18, "lum": -2}, "midtones": {"hue": 45, "sat": 14},
                        "highlights": {"hue": 40, "sat": 16, "lum": 8}, "global": {"hue": 74, "sat": 8},
                        "blending": 72, "balance": 0},
            "calibration": {"red_hue": -10, "red_sat": 6, "green_hue": 6, "blue_sat": 18},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.250], [0.5, 0.510], [0.75, 0.752], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.262], [0.5, 0.510], [0.75, 0.762], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.205], [0.5, 0.452], [0.75, 0.708], [1.0, 1.0]],
            },
            "color": {"vibrance": 14, "saturation": -10},
            "effects": {"clarity": 12, "texture": -16},
            "detail": {"sharpen_amount": 38, "sharpen_masking": 45},
            "grain": {"amount": 24, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.055, "white": -0.05},
    },
    "k_drama_warm": {
        "title": "韩系暖调 · 剧感柔光",
        "category": "剧感与人像",
        "affinity": ("warm", "portrait", "indoor", "night"),
        "use": "romantic portraits, couples, golden-hour city, warm cafes and indoor stories, sunset streets."
               " Sumptuous amber-gold highlights, warm WB, clean backlit micro-contrast and subtle soft bloom",
        "develop": {
            "light": {"highlights": -40, "shadows": 20},
            "wb": {"temp": 650, "tint": 2},
            "mixer": {"red": {"hue": 10, "sat": 10, "lum": 10}, "orange": {"hue": 8, "sat": 18, "lum": 18},
                      "yellow": {"hue": 4, "sat": 8, "lum": 12}, "green": {"sat": -14, "lum": 18},
                      "blue": {"sat": -20, "lum": -8}, "purple": {"sat": -60}, "magenta": {"sat": -60}},
            "grading": {"shadows": {"hue": 96, "sat": 6, "lum": 6}, "midtones": {"hue": 40, "sat": 10},
                        "highlights": {"hue": 45, "sat": 20, "lum": 16}, "global": {"hue": 40, "sat": 4},
                        "blending": 55, "balance": 0},
            "calibration": {"red_sat": 18, "green_sat": -10, "blue_sat": 24},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.256], [0.5, 0.506], [0.75, 0.752], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.236], [0.5, 0.478], [0.75, 0.728], [1.0, 1.0]],
            },
            "color": {"vibrance": 22, "saturation": -16},
            "effects": {"clarity": 10, "texture": -8, "dehaze": -5},
            "detail": {"sharpen_amount": 36, "sharpen_masking": 55},
            "grain": {"amount": 12, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.05, "mid": 0.03, "white": -0.02},
    },
    "anime_sky_blue": {
        "title": "动漫晴空 · 新海诚蓝",
        "category": "日系与动漫",
        "affinity": ("daylight", "vivid", "landscape", "night"),
        "use": "summer blue skies, towering clouds, seaside, fireworks, mecha/architecture, sunny travel landscapes."
               " Luminous cobalt skies, crisp shadow/highlight transparency, clean whites and subtle cinema grain",
        "develop": {
            "light": {"highlights": -48, "shadows": 34},
            "wb": {"temp": 120, "tint": -3},
            "mixer": {"red": {"hue": 10, "sat": 18, "lum": 10}, "orange": {"hue": 20, "sat": 22, "lum": 16},
                      "yellow": {"hue": 8, "sat": -10, "lum": 12}, "green": {"sat": -12, "lum": 22},
                      "aqua": {"sat": 8}, "blue": {"hue": 4, "sat": 8, "lum": -10},
                      "purple": {"sat": -55}, "magenta": {"sat": -50}},
            "grading": {"shadows": {"hue": 211, "sat": 6}, "highlights": {"hue": 209, "sat": 10},
                        "blending": 50, "balance": 0},
            "calibration": {"red_sat": 20, "green_sat": -12, "blue_sat": 26},
            "color": {"vibrance": 34, "saturation": -6},
            "effects": {"clarity": 12, "texture": -4, "dehaze": 2},
            "detail": {"sharpen_amount": 42, "sharpen_masking": 55},
            "grain": {"amount": 16, "size": 24, "roughness": 52},
        },
        "tone": {"black": 0.05, "mid": 0.04, "white": -0.01},
    },
    "wes_pastel": {
        "title": "复古童话 · 柔和粉彩",
        "category": "复古与人文",
        "affinity": ("high_key", "daylight", "architecture", "portrait"),
        "use": "colorful architecture, snow trees, hotels, cafes, symmetry, editorial portraits. Crisp local shadow"
               " snow/branch clarity, refined pastel reds and sculpted sky blue, delicate fine grain",
        "develop": {
            "light": {"highlights": -24, "shadows": 36},
            "mixer": {"red": {"hue": 10, "sat": 20, "lum": 12}, "orange": {"hue": 12, "sat": -8, "lum": 22},
                      "yellow": {"hue": -12, "sat": -10, "lum": 12}, "green": {"hue": 14, "sat": 18, "lum": 10},
                      "blue": {"sat": -28, "lum": -18}, "purple": {"sat": -70}, "magenta": {"sat": -70}},
            "grading": {"shadows": {"hue": 211, "sat": 8}, "highlights": {"hue": 209, "sat": 10},
                        "blending": 55, "balance": 0},
            "calibration": {"red_sat": 24, "green_sat": -13, "blue_sat": 33},
            "color": {"vibrance": 20, "saturation": -8},
            "effects": {"clarity": 16, "texture": 0, "dehaze": 8},
            "detail": {"sharpen_amount": 38, "sharpen_masking": 35},
            "grain": {"amount": 6, "size": 24, "roughness": 75},
        },
        "tone": {"black": 0.05, "white": -0.02},
    },
    "twilight_blue_drama": {
        "title": "蓝调烟火 · 暮色电影",
        "category": "暗调与夜景",
        "affinity": ("cool", "low_key", "night", "high_key"),
        "use": "blue hour, fireworks, misty forest roads, twilight dusk, rainy streets, moody winter scenes."
               " Anti-banding desaturated-then-graded cobalt-teal cinema look: drains clashing warm smoke before"
               " smoothly tinting midtones and shadows rich blue-teal with crisp atmospheric glow",
        "develop": {
            "light": {"highlights": -25, "shadows": -20},
            "wb": {"temp": -1550, "tint": -14},
            "mixer": {"red": {"hue": 14, "sat": -14}, "yellow": {"hue": -16, "sat": -18},
                      "green": {"hue": -22, "sat": -28, "lum": -14}, "aqua": {"hue": -8, "sat": -8},
                      "blue": {"hue": -10, "sat": 6}, "purple": {"hue": -20, "sat": -55},
                      "magenta": {"hue": -24, "sat": -50}},
            "grading": {"shadows": {"hue": 202, "sat": 24}, "midtones": {"hue": 200, "sat": 38, "lum": 4},
                        "highlights": {"hue": 204, "sat": 12}, "global": {"hue": 200, "sat": 24},
                        "blending": 74, "balance": 15},
            "calibration": {"red_hue": 18, "red_sat": 14, "green_hue": -10, "blue_hue": -14, "blue_sat": 24,
                            "shadows_tint": -4},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.185], [0.5, 0.405], [0.75, 0.695], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.258], [0.5, 0.514], [0.75, 0.760], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.300], [0.5, 0.542], [0.75, 0.785], [1.0, 1.0]],
            },
            "color": {"vibrance": 20, "saturation": -42},
            "effects": {"clarity": 18, "texture": -6, "dehaze": -12},
            "detail": {"sharpen_amount": 32, "sharpen_masking": 55},
            "grain": {"amount": 16, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.045, "white": -0.03},
    },
    "dune_amber": {
        "title": "史诗电影 · 琥珀沙丘",
        "category": "电影与胶片",
        "affinity": ("warm", "landscape", "daylight"),
        "use": "deserts, canyons, dry mountains, architectural monoliths, dramatic golden-hour landscapes and western"
               " roads. Warm amber-olive cinema palette via olive shadows + RGB curves, crisp depth and film grain",
        "develop": {
            "light": {"highlights": -45, "shadows": -15},
            "wb": {"temp": 300, "tint": -9},
            "mixer": {"red": {"hue": 14, "sat": 6, "lum": 10}, "orange": {"hue": 12, "sat": 8, "lum": 20},
                      "yellow": {"hue": -4, "sat": -12, "lum": 12}, "green": {"sat": -18, "lum": 20},
                      "aqua": {"sat": -75}, "blue": {"sat": -30, "lum": -18},
                      "purple": {"sat": -80}, "magenta": {"sat": -80}},
            "grading": {"shadows": {"hue": 93, "sat": 16}, "midtones": {"hue": 45, "sat": 14},
                        "highlights": {"hue": 74, "sat": 8}, "blending": 55, "balance": 0},
            "calibration": {"red_sat": 18, "green_sat": -12, "blue_sat": 28},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.248], [0.5, 0.494], [0.75, 0.724], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.258], [0.5, 0.510], [0.75, 0.766], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.246], [0.5, 0.486], [0.75, 0.744], [1.0, 1.0]],
            },
            "color": {"vibrance": 32, "saturation": -18},
            "effects": {"clarity": 12, "texture": -8, "dehaze": -2},
            "detail": {"sharpen_amount": 38, "sharpen_masking": 55},
            "grain": {"amount": 24, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.045, "white": -0.025},
    },
    "european_classic": {
        "title": "欧式油画 · 复古电影",
        "category": "复古与人文",
        "affinity": ("daylight", "warm", "architecture", "portrait"),
        "use": "historic architecture, estates, nostalgic urban portraits, autumn parks and dramatic interiors."
               " Crisp golden-olive highlight/shadow contrast (zero muddy veil), creamy sepia mids and RGB curves",
        "develop": {
            "light": {"highlights": -55, "shadows": -28},
            "wb": {"temp": 750, "tint": -4},
            "mixer": {"red": {"hue": 12, "lum": 18}, "orange": {"hue": 5, "sat": -6, "lum": 20},
                      "yellow": {"hue": -18, "sat": -6}, "green": {"hue": -28, "sat": -16, "lum": 20},
                      "aqua": {"hue": 24, "sat": -10}, "blue": {"hue": 3, "sat": -24, "lum": 24},
                      "purple": {"sat": -50}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 100, "sat": 8}, "midtones": {"hue": 42, "sat": 14},
                        "highlights": {"hue": 48, "sat": 6}, "global": {"hue": 46, "sat": 4},
                        "blending": 55, "balance": 0},
            "calibration": {"red_sat": 18, "green_sat": -10, "blue_sat": 26},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.248], [0.5, 0.502], [0.75, 0.752], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.253], [0.5, 0.503], [0.75, 0.751], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.236], [0.5, 0.480], [0.75, 0.734], [1.0, 1.0]],
            },
            "color": {"vibrance": 26, "saturation": -16},
            "effects": {"clarity": 10, "texture": -8, "dehaze": 0},
            "detail": {"sharpen_amount": 40, "sharpen_masking": 50},
            "grain": {"amount": 16, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.03, "white": -0.06},
    },
    "cyber_neon": {
        "title": "赛博霓虹 · 暗夜潮色",
        "category": "暗调与夜景",
        "affinity": ("night", "street", "vivid", "low_key"),
        "use": "night cities, neon signage, wet asphalt, futuristic architecture, moody urban street photography."
               " Cool electric cyan-violet edge, deepened blues and purples, crisp glowing neon accents",
        "develop": {
            "light": {"highlights": -40, "shadows": 15},
            "wb": {"temp": -150, "tint": -2},
            "mixer": {"red": {"hue": 14, "sat": 2, "lum": 14}, "orange": {"sat": -16, "lum": 18},
                      "yellow": {"hue": -22, "sat": 2}, "green": {"sat": -18, "lum": 18},
                      "aqua": {"hue": 14, "lum": -14}, "blue": {"sat": -12, "lum": -18},
                      "purple": {"hue": 14, "sat": -28, "lum": -20}, "magenta": {"hue": 16, "sat": -12, "lum": -14}},
            "grading": {"shadows": {"hue": 204, "sat": 10}, "midtones": {"hue": 212, "sat": 6},
                        "highlights": {"hue": 240, "sat": 6}, "blending": 55, "balance": 8},
            "calibration": {"red_sat": 24, "green_hue": 10, "green_sat": 4, "blue_hue": 4, "blue_sat": 20},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.244], [0.5, 0.496], [0.75, 0.750], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.256], [0.5, 0.504], [0.75, 0.754], [1.0, 1.0]],
            },
            "color": {"vibrance": 26, "saturation": -18},
            "effects": {"clarity": 10, "texture": -12},
            "detail": {"sharpen_amount": 36, "sharpen_masking": 50},
            "grain": {"amount": 14, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.03, "white": -0.06},
    },
    # ---- Newly Distilled 6 Cinema / Drama / Noir / Retro Archetypes (100% 素材a coverage) ----
    "j_drama_cool": {
        "title": "日剧青灰 · 清冷空气",
        "category": "日系与动漫",
        "affinity": ("cool", "daylight", "high_key", "night", "portrait"),
        "use": "Japanese drama cities, cloudy coastal towns, winter streets, train stations, introspective portraits."
               " Translucent cool blue-slate air, clean porcelain skin, crisp detail and subtle micro-grain",
        "develop": {
            "light": {"highlights": -38, "shadows": 22},
            "wb": {"temp": -900, "tint": -6},
            "mixer": {"red": {"hue": 11, "sat": 10, "lum": 11}, "orange": {"hue": 12, "sat": 8, "lum": 20},
                      "yellow": {"hue": 7, "sat": -14, "lum": 14}, "green": {"sat": -16, "lum": 24},
                      "aqua": {"hue": -24, "lum": -5}, "blue": {"hue": -4, "sat": -10, "lum": -14},
                      "purple": {"hue": -18, "sat": -70}, "magenta": {"hue": -22, "sat": -70}},
            "grading": {"shadows": {"hue": 204, "sat": 14}, "midtones": {"hue": 202, "sat": 10},
                        "highlights": {"hue": 206, "sat": 10}, "global": {"hue": 202, "sat": 8},
                        "blending": 58, "balance": 20},
            "calibration": {"red_hue": 16, "red_sat": 18, "green_hue": 6, "green_sat": -12,
                            "blue_hue": -14, "blue_sat": 24},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.228], [0.5, 0.476], [0.75, 0.736], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.274], [0.5, 0.524], [0.75, 0.770], [1.0, 1.0]],
            },
            "color": {"vibrance": 30, "saturation": -14},
            "effects": {"clarity": 10, "texture": -8, "dehaze": -4},
            "detail": {"sharpen_amount": 36, "sharpen_masking": 55},
            "grain": {"amount": 16, "size": 22, "roughness": 45},
        },
        "tone": {"black": 0.05, "white": -0.03},
    },
    "youth_film_mint": {
        "title": "青春日影 · 青绿胶片",
        "category": "剧感与人像",
        "affinity": ("daylight", "portrait", "foliage", "street"),
        "use": "coming-of-age portraits, campuses, overcast streets, seaside youth, indie Japanese/Nordic dramas."
               " Subtle Fujifilm mint-green shadow cast, crisp warm skin, gentle contrast and tactile film grain",
        "develop": {
            "light": {"highlights": -40, "shadows": 24},
            "wb": {"temp": 550, "tint": -9},
            "mixer": {"red": {"hue": 11, "sat": 8, "lum": 8}, "orange": {"hue": 6, "sat": -6, "lum": 24},
                      "yellow": {"hue": 7, "sat": -12, "lum": 14}, "green": {"sat": -10, "lum": 26},
                      "aqua": {"hue": 26, "sat": -10}, "blue": {"hue": 3, "sat": -24, "lum": 12},
                      "purple": {"sat": -36}, "magenta": {"sat": -33}},
            "grading": {"shadows": {"hue": 158, "sat": 12}, "midtones": {"hue": 92, "sat": 8},
                        "highlights": {"hue": 200, "sat": 8}, "blending": 55, "balance": 0},
            "calibration": {"red_sat": 20, "green_sat": -10, "blue_sat": 28},
            "curve": {
                "green": [[0.0, 0.0], [0.25, 0.260], [0.5, 0.510], [0.75, 0.756], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.228], [0.5, 0.482], [0.75, 0.742], [1.0, 1.0]],
            },
            "color": {"vibrance": 32, "saturation": -12},
            "effects": {"clarity": 10, "texture": -8, "dehaze": -3},
            "detail": {"sharpen_amount": 38, "sharpen_masking": 50},
            "grain": {"amount": 18, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.05, "mid": 0.03, "white": -0.02},
    },
    "summer_nostalgia": {
        "title": "夏日胶片 · 晴夏清甜",
        "category": "日系与动漫",
        "affinity": ("daylight", "vivid", "foliage", "portrait"),
        "use": "bright summer afternoons, intimate film portraits, oceanside roads, sunflower fields, childhood travel."
               " Fresh slightly cool-mint shadows, glowing sunlit skin, crisp detail and authentic 35mm analog film grain",
        "develop": {
            "light": {"highlights": -52, "shadows": 24},
            "wb": {"temp": -80, "tint": -5},
            "mixer": {"red": {"hue": 10, "sat": 14, "lum": 10}, "orange": {"hue": 12, "sat": 14, "lum": 22},
                      "yellow": {"hue": 8, "sat": -10, "lum": 14}, "green": {"sat": -8, "lum": 26},
                      "aqua": {"sat": 8, "lum": 8}, "blue": {"sat": -12, "lum": -16},
                      "purple": {"sat": -70}, "magenta": {"sat": -70}},
            "grading": {"shadows": {"hue": 196, "sat": 10}, "highlights": {"hue": 208, "sat": 10},
                        "blending": 55, "balance": 0},
            "calibration": {"red_sat": 22, "green_sat": -10, "blue_sat": 30},
            "curve": {
                "green": [[0.0, 0.0], [0.25, 0.255], [0.5, 0.506], [0.75, 0.754], [1.0, 1.0]],
            },
            "color": {"vibrance": 40, "saturation": -8},
            "effects": {"clarity": 10, "texture": 0, "dehaze": -2},
            "detail": {"sharpen_amount": 44, "sharpen_masking": 22},
            "grain": {"amount": 34, "size": 24, "roughness": 55},
        },
        "tone": {"black": 0.045, "mid": 0.04, "white": -0.015},
    },
    "crime_amber_grit": {
        "title": "罪案暗黄 · 悬疑粗粝",
        "category": "电影与胶片",
        "affinity": ("low_key", "warm", "street", "indoor"),
        "use": "gritty urban alleys, thrillers, moody cafes, raw documentary portraits, post-apocalyptic ruins."
               " Desaturated amber-olive cinema grit, crisp local contrast and palpable 35mm cinema grain",
        "develop": {
            "light": {"highlights": -45, "shadows": -18},
            "wb": {"temp": 750, "tint": -4},
            "mixer": {"red": {"hue": 11, "sat": 4, "lum": 6}, "orange": {"hue": 6, "sat": -10, "lum": 26},
                      "yellow": {"sat": -20, "lum": 14}, "green": {"hue": 6, "sat": -16, "lum": 32},
                      "aqua": {"sat": 6, "lum": 4}, "blue": {"sat": -28, "lum": 10},
                      "purple": {"sat": -36}, "magenta": {"sat": -36}},
            "grading": {"shadows": {"hue": 88, "sat": 12}, "midtones": {"hue": 42, "sat": 14},
                        "highlights": {"hue": 132, "sat": 10}, "blending": 55, "balance": 0},
            "calibration": {"red_sat": 20, "green_sat": -12, "blue_sat": 28},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.254], [0.5, 0.506], [0.75, 0.752], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.232], [0.5, 0.476], [0.75, 0.736], [1.0, 1.0]],
            },
            "color": {"vibrance": 26, "saturation": -18},
            "effects": {"clarity": 10, "texture": -12, "dehaze": -2},
            "detail": {"sharpen_amount": 40, "sharpen_masking": 48},
            "grain": {"amount": 26, "size": 22, "roughness": 55},
        },
        "tone": {"black": 0.04, "white": -0.05},
    },
    "retro_americana": {
        "title": "美式复古 · 公路胶片",
        "category": "复古与人文",
        "affinity": ("warm", "daylight", "street", "indoor"),
        "use": "1960s-80s road movies, vintage cars, diners, autumn highways, western towns, nostalgic portraits."
               " Classic Kodak amber shadow toe, rich wine-red and denim-blue separation, crisp detail and retro grain",
        "develop": {
            "light": {"highlights": -40, "shadows": 20},
            "wb": {"temp": 650, "tint": -5},
            "mixer": {"red": {"hue": 12, "lum": 11}, "orange": {"hue": 9, "sat": -6, "lum": 24},
                      "yellow": {"hue": 7, "sat": -12}, "green": {"sat": -14, "lum": 24},
                      "aqua": {"hue": 20, "sat": -4}, "blue": {"sat": -24, "lum": 4},
                      "purple": {"hue": 18, "sat": -72}, "magenta": {"hue": 18, "sat": -80}},
            "grading": {"shadows": {"hue": 56, "sat": 12}, "midtones": {"hue": 40, "sat": 12},
                        "highlights": {"hue": 210, "sat": 10}, "blending": 55, "balance": 15},
            "calibration": {"red_hue": 18, "red_sat": 22, "green_hue": 14, "green_sat": -10,
                            "blue_hue": 6, "blue_sat": 28},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.254], [0.5, 0.502], [0.75, 0.750], [1.0, 1.0]],
                "green": [[0.0, 0.0], [0.25, 0.254], [0.5, 0.504], [0.75, 0.752], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.228], [0.5, 0.480], [0.75, 0.742], [1.0, 1.0]],
            },
            "color": {"vibrance": 32, "saturation": -10},
            "effects": {"clarity": 8, "texture": -8, "dehaze": -2},
            "detail": {"sharpen_amount": 38, "sharpen_masking": 50},
            "grain": {"amount": 22, "size": 24, "roughness": 50},
        },
        "tone": {"black": 0.045, "white": -0.04},
    },
    "noir_steel": {
        "title": "暗黑冷钢 · 哥特纪实",
        "category": "暗调与夜景",
        "affinity": ("low_key", "night", "architecture", "muted"),
        "use": "dark gothic cities, brutalist concrete, rain-soaked steel, sci-fi ruins, brooding dark portraits."
               " Restrained desaturated gunmetal slate, crisp micro-contrast and tactile 35mm dark-film grain",
        "develop": {
            "light": {"highlights": -45, "shadows": -20},
            "wb": {"temp": 380, "tint": -3},
            "mixer": {"red": {"hue": 11, "sat": 6, "lum": 6}, "orange": {"hue": 6, "sat": -12, "lum": 24},
                      "yellow": {"hue": 7, "sat": -16, "lum": 14}, "green": {"sat": 4, "lum": 24},
                      "aqua": {"sat": 2}, "blue": {"sat": -24, "lum": -6},
                      "purple": {"sat": -32}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 208, "sat": 10}, "midtones": {"hue": 48, "sat": 8},
                        "highlights": {"hue": 210, "sat": 8}, "blending": 55, "balance": 0},
            "calibration": {"red_sat": 20, "green_sat": -12, "blue_sat": 28},
            "curve": {
                "red": [[0.0, 0.0], [0.25, 0.248], [0.5, 0.502], [0.75, 0.750], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.25, 0.242], [0.5, 0.492], [0.75, 0.748], [1.0, 1.0]],
            },
            "color": {"vibrance": 24, "saturation": -22},
            "effects": {"clarity": 12, "texture": -14, "dehaze": 6},
            "detail": {"sharpen_amount": 40, "sharpen_masking": 48},
            "grain": {"amount": 26, "size": 24, "roughness": 55},
        },
        "tone": {"black": 0.03, "white": -0.05},
    },
}

# Always keep a balanced core spine when shortlisting for VLM planning.
_CORE_SPINE = ("clean_cinema", "matte_landscape", "cine_teal", "soft_portrait", "portra_pastel")

# Stylization intensity tiers:
# - "light" (22 looks): restrained, everyday-deliverable polish with believable skin & clean whites.
# - "strong" (13 looks): unmistakable dramatic colour cast, heavy matte, or bold cinema grading
#   (capped to at most 1 candidate per multi-version set so users get more versatile light options).
STRONG_LOOK_IDS: frozenset[str] = frozenset({
    "wong_kar_wai",
    "twilight_blue_drama",
    "twilight_dream",
    "cyber_neon",
    "dune_amber",
    "crime_amber_grit",
    "noir_steel",
    "lowkey_muted",
    "muted_emotion",
    "period_ink",
    "mist_violet",
    "sunset_crimson",
    "urban_street",
})


def look_intensity(look_id: str | None) -> str:
    """Return ``'strong'`` for bold/heavy stylizations and ``'light'`` for restrained everyday looks."""
    if not look_id:
        return "light"
    spec = LOOKS.get(look_id)
    if spec and spec.get("intensity") in ("light", "strong"):
        return str(spec["intensity"])
    return "strong" if look_id in STRONG_LOOK_IDS else "light"


def is_strong_look(look_id: str | None) -> bool:
    return look_intensity(look_id) == "strong"


def shortlist_for_scene(baseline: Any | None = None, *, technical: Any | None = None,
                        limit: int = 22, max_strong: int = 5) -> list[str]:
    """Select the most scene-relevant subset of ``LOOKS`` for VLM probing & planning.

    Biases the candidate pool toward restrained everyday ``light`` looks (at least ~75% of the
    shortlist) while capping ``strong`` stylizations to ``max_strong`` so the planner naturally
    delivers more subtle, usable options and fewer heavy-handed looks.
    """
    all_ids = list(LOOKS.keys())
    if limit <= 0 or len(all_ids) <= limit:
        return all_ids

    # Measure simple scene cues (luminance key, warmth, colourfulness) if a preview/stats is supplied.
    tags: set[str] = {"all_round"}
    mid = None
    warmth = None
    chroma = None
    if isinstance(baseline, dict):
        mid = baseline.get("mid")
        warmth = baseline.get("warmth")
        chroma = baseline.get("chroma")
    elif baseline is not None:
        try:
            import numpy as np
            from . import tone

            rgb = tone.load_small(baseline, 256) if not isinstance(baseline, np.ndarray) else baseline
            anc = tone.measure(rgb)
            mid = float(anc.get("mid", 0.4))
            warmth = float(np.mean(rgb[..., 0] - rgb[..., 2]))
            chroma = float(tone.chroma_mean(rgb))
        except Exception:
            pass
    if mid is None and technical is not None:
        mid = getattr(technical, "mean_luma", None)
    if chroma is None and technical is not None:
        chroma = getattr(technical, "mean_saturation", None)

    if mid is not None:
        if mid < 0.24:
            tags.update(("night", "low_key", "street"))
        elif mid > 0.56:
            tags.update(("high_key", "daylight", "landscape"))
        else:
            tags.update(("daylight", "portrait", "street", "landscape", "foliage"))
    else:
        tags.update(("daylight", "portrait", "landscape", "street"))

    if warmth is not None:
        if warmth > 0.06:
            tags.update(("warm", "indoor"))
        elif warmth < -0.04:
            tags.update(("cool", "architecture"))
        else:
            tags.update(("warm", "cool", "foliage"))

    if chroma is not None:
        if chroma < 0.08:
            tags.add("muted")
        elif chroma > 0.18:
            tags.add("vivid")

    def _score(lid: str) -> tuple[int, int]:
        aff = set(LOOKS[lid].get("affinity") or ())
        spine_bonus = 3 if lid in _CORE_SPINE else 0
        overlap = len(aff & tags)
        return (spine_bonus + overlap, -all_ids.index(lid))

    ranked = sorted(all_ids, key=_score, reverse=True)
    chosen: list[str] = []
    strong_used = 0
    for lid in ranked:
        if len(chosen) >= limit:
            break
        if is_strong_look(lid):
            if strong_used >= max_strong:
                continue
            strong_used += 1
        chosen.append(lid)
    # Fill any remaining slots if light pool was smaller than limit - max_strong
    if len(chosen) < limit:
        for lid in ranked:
            if lid not in chosen:
                chosen.append(lid)
                if len(chosen) >= limit:
                    break
    chosen_set = set(chosen)
    return [lid for lid in all_ids if lid in chosen_set]


def catalog() -> list[dict[str, Any]]:
    """Structured metadata for all distilled looks (exposed via GET /api/directed-looks)."""
    items = []
    for lid, spec in LOOKS.items():
        dev = spec.get("develop") or {}
        intensity = look_intensity(lid)
        items.append({
            "id": lid,
            "title": spec["title"],
            "category": spec.get("category", "电影与胶片"),
            "intensity": intensity,
            "recommended_strength": 0.65 if intensity == "light" else 0.75,
            "use": spec["use"],
            "has_rgb_curves": bool(dev.get("curve")),
            "has_wb": bool(dev.get("wb")),
            "has_global_grading": bool((dev.get("grading") or {}).get("global")),
            "has_local_light": bool(dev.get("light")),
            "tone": spec.get("tone") or {},
        })
    return items


# LightCraft's built-in colour profiles (`profiles.list`, 0.21.1), offered as "profile": {"id", "amount"} under a
# version's own edit. In the develop document a profile is one key, {"profile": {"id", "amount" 0..200}}; a cold
# render, develop.merge and develop.profile give the same pixels (probe 2026-11-24). Portrait is left out (it
# barely changes a picture: 0.9 at amount 100), as are the monochrome and B&W ones.
# "use" is what each did on its own to ten unedited photos (the three JPEG dev photos and the seven reference
# RAWs, .task-cache/probe-looks-2026-11-24/out/effects.log), at amount 100: Lab shift, colourfulness ratio and
# the black point it lifts. The solver still places the tones, so a matte profile's black goes where the plan's
# target says.
PROFILES: dict[str, dict[str, str]] = {
    "lc.vivid": {"title": "鲜艳", "use": "richer colour everywhere (x1.25), blues a little deeper; no hue cast"},
    "lc.landscape": {"title": "风光", "use": "richer blue sky and green foliage (x1.14), a slightly cooler frame"},
    "lc.neutral": {"title": "中性", "use": "flatter, a little less colour (x0.90), softer whites; a quiet base"},
    "lc.film.warm-print": {"title": "胶片 · 暖调相纸", "use": "a warm yellow print cast, slightly less colour (x0.90),"
                           " a trace of lifted black"},
    "lc.film.cool-fade": {"title": "胶片 · 冷调褪色", "use": "faded cool film: lifted black, held white, green-cyan"
                          " lean, much less colour (x0.77)"},
    "lc.film.golden-hour": {"title": "胶片 · 金色时刻", "use": "strong golden warmth over the whole frame, richer"
                            " colour (x1.26)"},
    "lc.film.faded-slide": {"title": "胶片 · 褪色反转片", "use": "a blue slide-film cast, colour a little richer (x1.15)"},
    "lc.cine.teal-amber": {"title": "电影 · 青橙", "use": "amber warm tones against teal cool ones, colourfulness about"
                           " unchanged (x0.95)"},
    "lc.cine.night-blue": {"title": "电影 · 夜蓝", "use": "a cold blue cast, slightly darker, less colour (x0.86)"},
    "lc.cine.desert-heat": {"title": "电影 · 沙漠热浪", "use": "the strongest: an orange-red heat over the whole frame,"
                            " richer (x1.17); changes a picture about twice as much as the others"},
    "lc.cine.neon-dusk": {"title": "电影 · 霓虹黄昏", "use": "a magenta-violet lean, richer (x1.14)"},
    "lc.muted.matte-soft": {"title": "低饱 · 柔和哑光", "use": "a soft matte: lifted black, much less colour (x0.75)"},
    "lc.muted.bleached": {"title": "低饱 · 漂白", "use": "bleach bypass: colour drained (x0.62), crisp whites"},
    "lc.muted.pastel-haze": {"title": "低饱 · 粉彩薄雾", "use": "a bright haze: lifted black and mids, less colour"
                             " (x0.76)"},
    "lc.muted.quiet-green": {"title": "低饱 · 静谧绿", "use": "lighter and a little muted (x0.86), a small change"},
}


def profile_lines(notes: dict[str, str] | None = None) -> list[str]:
    """One line per profile for a planning prompt, followed by ``notes[profile_id]`` where given."""
    notes = notes or {}
    return [f"{profile_id} ({spec['title']}): {spec['use']}" + (f" {notes[profile_id]}" if profile_id in notes else "")
            for profile_id, spec in PROFILES.items()]


def _tone_hint(shifts: dict[str, float]) -> str:
    """The family's black and white handling as a suggestion; the plan's tone targets decide."""
    parts = [f"{name} {value:+.2f}" for name, value in shifts.items() if abs(value) >= 0.005]
    return f" [presets at full strength: {', '.join(parts)} on the tone targets]" if parts else ""


def look_lines(notes: dict[str, str] | None = None, look_ids: Iterable[str] | None = None) -> list[str]:
    """One line per look for a planning prompt: id, Chinese title, stylization tier, when to use it and
    its tone handling, followed by ``notes[look_id]`` where given (what the look measured on this photo)."""
    notes = notes or {}
    ids = list(look_ids) if look_ids is not None else list(LOOKS.keys())
    out = []
    for look_id in ids:
        if look_id not in LOOKS:
            continue
        spec = LOOKS[look_id]
        tier = "[强风格 · 全组最多选1套]" if is_strong_look(look_id) else "[轻量风格 · 建议强度0.45-0.75]"
        out.append(
            f"{look_id} ({spec['title']}) {tier}: {spec['use']}{_tone_hint(spec.get('tone') or {})}"
            + (f" {notes[look_id]}" if look_id in notes else "")
        )
    return out
