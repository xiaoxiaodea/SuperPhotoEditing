"""Colour looks for the vision-directed path, in LightCraft's develop vocabulary.

Each look is the palette of one family of the reference preset library
(resources/LeoCarbin-双系列预设, 298 look presets in 21 families): the HSL mix,
split toning and calibration of the preset nearest to its family's median,
plus what the family does to the ends of the tone curve (most of them lift
black into a matte and hold white below paper white). Family statistics are in
.task-cache/resource-study-2026-11-13/preset-inventory.json.

A look carries colour only. Tonal placement stays with the plan's tone targets;
"tone" here is what the family does to them (display luma, 0..1), shown to the
planner as a suggestion it may write into its targets. It is not added to them:
the targets describe the finished frame (see directed.tone_targets). Exposure,
contrast, highlights and shadows of the presets are left out: they were written
for one particular raw file and are what the solver replaces.

directed._merged_develop adds a look underneath the model's own values, scaled
by the strength the model chose.
"""
from __future__ import annotations

from typing import Any

LOOKS: dict[str, dict[str, Any]] = {
    "matte_landscape": {
        "title": "风光 · 低饱哑光",
        "use": "all-round landscape: mountains, lakes, grassland, big skies. Muted yellow-green, bright clean aqua and"
               " water, cool shadows, soft matte black",
        "develop": {
            "mixer": {"red": {"hue": 18, "sat": -11}, "orange": {"hue": -6, "lum": 8},
                      "yellow": {"sat": -44, "lum": 6}, "green": {"hue": -20, "sat": -45, "lum": 15},
                      "aqua": {"sat": -15, "lum": 30}, "blue": {"hue": -6, "sat": -30},
                      "purple": {"hue": -39, "sat": -25}, "magenta": {"sat": -32}},
            "grading": {"shadows": {"hue": 236, "sat": 12}, "highlights": {"hue": 45, "sat": 4},
                        "blending": 100, "balance": 40},
            "calibration": {"red_hue": 20, "green_hue": 18, "blue_hue": -14},
        },
        "tone": {"black": 0.07, "shadows": 0.03, "white": -0.04},
    },
    "forest_olive": {
        "title": "森系 · 橄榄绿",
        "use": "forest, trees, green valleys, meadows, a person among greenery. Dark olive greens, warm orange subject,"
               " teal shadows, dense low mids",
        "develop": {
            "mixer": {"red": {"hue": 25, "sat": -45, "lum": 10}, "orange": {"hue": -15, "sat": 18, "lum": 5},
                      "yellow": {"hue": 10, "sat": -55, "lum": -10}, "green": {"hue": -42, "sat": -68, "lum": 18},
                      "aqua": {"hue": 10, "sat": -45, "lum": 35}, "blue": {"hue": -5, "sat": -50},
                      "purple": {"hue": -35, "sat": -35}, "magenta": {"hue": 30, "sat": -50}},
            "grading": {"shadows": {"hue": 205, "sat": 26}, "highlights": {"hue": 34, "sat": 6},
                        "blending": 100, "balance": 60},
            "calibration": {"green_hue": -10, "blue_hue": -20},
        },
        "tone": {"black": 0.02, "shadows": -0.02, "mid": -0.04},
    },
    "lowkey_muted": {
        "title": "暗系低饱",
        "use": "overcast, shade, dusk, moody low-key scenes, dark architecture. Greens and blues nearly drained,"
               " teal-grey shadows and highlights, matte black, held-back white",
        "develop": {
            "mixer": {"red": {"hue": 22, "sat": -20}, "orange": {"hue": 6, "sat": -8, "lum": 8},
                      "yellow": {"hue": -25, "sat": -40}, "green": {"hue": -28, "sat": -68, "lum": 25},
                      "aqua": {"sat": -40, "lum": 15}, "blue": {"hue": -12, "sat": -66},
                      "purple": {"hue": -40, "sat": -45}, "magenta": {"sat": -40}},
            "grading": {"shadows": {"hue": 190, "sat": 12}, "highlights": {"hue": 201, "sat": 6},
                        "blending": 100, "balance": 22},
            "calibration": {"red_hue": 20, "green_hue": 15, "blue_hue": -12, "shadows_tint": -5},
        },
        "tone": {"black": 0.08, "shadows": 0.02, "white": -0.06},
    },
    "twilight_dream": {
        "title": "暮色梦境",
        "use": "blue hour, dusk, dawn, fog, rain, quiet water. Heavy matte, drained greens, soft blue-grey air; the"
               " strongest mood of the set, use strength 0.4-0.7",
        "develop": {
            "mixer": {"red": {"hue": 20, "sat": -32, "lum": -10}, "orange": {"hue": 2, "sat": -21},
                      "yellow": {"hue": -16, "sat": -27, "lum": 10}, "green": {"hue": -50, "sat": -77},
                      "aqua": {"hue": 6, "sat": -32}, "blue": {"hue": -7, "sat": -33}},
            "grading": {"shadows": {"hue": 225, "sat": 10}, "highlights": {"hue": 205, "sat": 6},
                        "blending": 50, "balance": -40},
            "calibration": {"red_hue": 30, "green_hue": 30, "green_sat": -15, "blue_hue": -8, "blue_sat": -15},
        },
        "tone": {"black": 0.13, "shadows": 0.06, "highlights": -0.03, "white": -0.10},
    },
    "clean_cinema": {
        "title": "通透电影感",
        "use": "the safe all-rounder: travel, people, street, daylight landscape. Clean warm skin and sand, calm greens,"
               " teal-leaning sky and water, only a trace of matte",
        "develop": {
            "mixer": {"red": {"hue": -12, "sat": -15, "lum": 10}, "orange": {"hue": 12, "sat": 10, "lum": 12},
                      "yellow": {"sat": 12, "lum": 12}, "green": {"sat": -24, "lum": 14},
                      "aqua": {"hue": -30, "sat": -20}, "blue": {"hue": -4, "sat": -18}},
            "grading": {"shadows": {"hue": 200, "sat": 8}, "highlights": {"hue": 48, "sat": 6},
                        "blending": 100, "balance": 30},
            "calibration": {"red_hue": 20, "green_hue": -20, "blue_hue": -30},
        },
        "tone": {"black": 0.035, "white": -0.02},
    },
    "muted_emotion": {
        "title": "低饱情绪",
        "use": "quiet, sparse or lonely scenes, overcast plains, winter trees, a small figure in a large space. Almost"
               " no green, pale warm skin and earth, grey-blue sky, soft low white",
        "develop": {
            "mixer": {"red": {"hue": 12, "sat": -10, "lum": 18}, "orange": {"hue": -12, "sat": -14, "lum": 28},
                      "yellow": {"hue": 5, "sat": -58, "lum": -20}, "green": {"hue": -40, "sat": -80, "lum": 18},
                      "aqua": {"hue": 8}, "blue": {"hue": -5, "sat": -42}},
            "grading": {"shadows": {"hue": 81, "sat": 5}, "blending": 50, "balance": 0},
            "calibration": {"green_hue": -25, "green_sat": -35, "blue_hue": -6, "blue_sat": -25, "red_sat": -5},
        },
        "tone": {"black": 0.05, "highlights": -0.03, "white": -0.11},
    },
    "vintage_olive": {
        "title": "德系复古",
        "use": "architecture, old towns, markets, interiors, vehicles, still life. Olive-yellow shadows, warm"
               " highlights, deeper reds, an aged print feel without a matte",
        "develop": {
            "mixer": {"red": {"sat": -22, "lum": -14}, "orange": {"hue": 2, "lum": 10},
                      "yellow": {"hue": -10, "sat": 10, "lum": 10}, "green": {"hue": -20, "sat": -25, "lum": -15},
                      "blue": {"hue": -3, "sat": -12}},
            "grading": {"shadows": {"hue": 57, "sat": 18}, "highlights": {"hue": 28, "sat": 10},
                        "blending": 70, "balance": 10},
            "calibration": {"red_hue": 10, "red_sat": -5, "green_hue": 20, "green_sat": -10, "blue_hue": -10},
            "grain": {"amount": 14, "size": 22, "roughness": 50},
        },
        "tone": {"black": 0.015, "white": -0.035},
    },
    "cine_teal": {
        "title": "CINE 青橙",
        "use": "city, street, roads, vehicles, travel documentary frames with people. Pale teal greens and blues,"
               " warm shadows against cyan highlights, soft clarity, film grain, clear matte",
        "develop": {
            "mixer": {"yellow": {"hue": 4}, "green": {"hue": 16, "sat": -60, "lum": 30},
                      "aqua": {"hue": 29, "sat": -40, "lum": 14}, "blue": {"hue": -15, "sat": -24, "lum": 14},
                      "purple": {"hue": 31, "sat": -34}, "magenta": {"hue": 13, "sat": -29}},
            "grading": {"shadows": {"hue": 26, "sat": 8}, "highlights": {"hue": 190, "sat": 5},
                        "blending": 100, "balance": -60},
            "calibration": {"red_hue": 5, "green_hue": -8, "blue_hue": -10},
            "color": {"vibrance": 15, "saturation": -15},
            "effects": {"clarity": -12},
            "grain": {"amount": 22, "size": 16, "roughness": 50},
        },
        "tone": {"black": 0.065, "white": -0.08},
    },
    "snow_luminous": {
        "title": "雪景 · 通透冷白",
        "use": "snow, ice, glaciers, salt flats, frozen lakes, bright mist. Whites stay white and luminous, blues turn"
               " pale and icy, cool shadows with a trace of warmth in the highlights, soft micro-contrast",
        "develop": {
            "mixer": {"red": {"hue": 25, "sat": -40, "lum": -6}, "orange": {"hue": -10, "sat": 14, "lum": 6},
                      "yellow": {"hue": -8, "sat": -30, "lum": 4}, "green": {"hue": 8, "sat": -55, "lum": 8},
                      "aqua": {"hue": 40, "sat": -36, "lum": 30}, "blue": {"hue": -9, "sat": -42},
                      "purple": {"sat": -40}, "magenta": {"sat": -33}},
            "grading": {"shadows": {"hue": 217, "sat": 8}, "highlights": {"hue": 41, "sat": 3},
                        "blending": 100, "balance": -80},
            "calibration": {"red_hue": 5, "green_hue": 8, "blue_hue": -11},
            "effects": {"clarity": -12},
        },
        "tone": {"black": 0.03, "shadows": 0.04},
    },
    "desert_gold": {
        "title": "沙漠 · 暖金",
        "use": "desert, dunes, dry mountains, canyons, red rock, golden grass, late sun. Glowing orange-gold earth,"
               " deep clean blue sky, no green, lifted matte shadows with crisp structure",
        "develop": {
            "mixer": {"red": {"hue": 22, "sat": 5, "lum": 14}, "orange": {"hue": -10, "sat": 18, "lum": 10},
                      "yellow": {"sat": -10, "lum": 14}, "green": {"hue": 15, "sat": -58, "lum": 10},
                      "aqua": {"hue": 10, "sat": 20, "lum": 20}, "blue": {"hue": -10, "lum": 12},
                      "purple": {"sat": -55, "lum": 20}, "magenta": {"hue": -25, "sat": -31, "lum": 20}},
            "grading": {"shadows": {"hue": 250, "sat": 6}, "highlights": {"hue": 44, "sat": 10},
                        "blending": 100, "balance": -30},
            "calibration": {"red_sat": -15, "green_hue": 10, "green_sat": -10, "blue_sat": 45},
            "color": {"vibrance": 18},
            "effects": {"clarity": 10},
        },
        "tone": {"black": 0.07, "white": -0.035},
    },
    "night_blue": {
        "title": "夜景 · 蓝调",
        "use": "night streets, lamps, neon, city lights, blue hour with artificial light. Cold blue-teal dark areas,"
               " lamps kept warm, greens and magentas removed, matte black",
        "develop": {
            "mixer": {"red": {"hue": 16, "sat": -30, "lum": -14}, "orange": {"hue": -5, "sat": -12},
                      "yellow": {"hue": -35, "sat": -60}, "green": {"hue": -60, "sat": -85, "lum": 25},
                      "aqua": {"hue": 33, "sat": -30, "lum": 30}, "blue": {"hue": -15, "sat": -50, "lum": 10},
                      "purple": {"hue": -60, "sat": -50}, "magenta": {"hue": -60, "sat": -50}},
            "grading": {"shadows": {"hue": 201, "sat": 20}, "highlights": {"hue": 191, "sat": 5},
                        "blending": 100, "balance": 45},
            "calibration": {"red_hue": 40, "green_hue": 25, "blue_hue": -25, "shadows_tint": 5},
        },
        "tone": {"black": 0.06, "white": -0.05},
    },
    "portra_pastel": {
        "title": "日系胶片 · Portra",
        "use": "bright airy daylight: people, children, seaside, flowers, cafes, light-toned travel frames. Peach skin,"
               " yellow-green foliage, pale cyan sky, open mids, creamy matte",
        "develop": {
            "mixer": {"red": {"hue": 20, "sat": -20}, "orange": {"hue": 20, "sat": 16, "lum": 18},
                      "yellow": {"lum": 8}, "green": {"hue": 40}, "aqua": {"hue": 22, "lum": 15},
                      "blue": {"hue": -10, "lum": 10}},
            "grading": {"shadows": {"hue": 220, "sat": 13}, "midtones": {"hue": 53, "sat": 12, "lum": 10},
                        "highlights": {"hue": 208, "sat": 5}, "blending": 100, "balance": -80},
            "calibration": {"green_hue": 30},
        },
        "tone": {"black": 0.085, "shadows": 0.04, "mid": 0.03, "white": -0.05},
    },
    "soft_portrait": {
        "title": "人像 · 柔和肤色",
        "use": "portraits and people as the clear subject, outdoors or indoors. Skin kept warm and clean while the"
               " surroundings (foliage, sky, blue or purple clothing) are muted and cooled, light grain",
        "develop": {
            "mixer": {"red": {"hue": 22, "lum": 10}, "orange": {"lum": 6}, "yellow": {"sat": -42, "lum": 6},
                      "green": {"hue": -36, "sat": -50, "lum": 20}, "aqua": {"hue": 10, "sat": -25, "lum": 30},
                      "blue": {"hue": -5, "sat": -55}, "purple": {"hue": -35, "sat": -30},
                      "magenta": {"hue": 25, "sat": -45}},
            "grading": {"shadows": {"hue": 212, "sat": 12}, "midtones": {"hue": 60, "sat": 6},
                        "highlights": {"hue": 208, "sat": 5}, "blending": 100, "balance": 0},
            "calibration": {"red_hue": 15, "green_hue": 10, "blue_hue": -8, "shadows_tint": -4},
            "grain": {"amount": 16, "size": 25, "roughness": 50},
        },
        "tone": {"black": 0.06, "white": -0.04},
    },
}


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


def look_lines(notes: dict[str, str] | None = None) -> list[str]:
    """One line per look for a planning prompt: id, Chinese title, when to use it and its tone handling,
    followed by ``notes[look_id]`` where given (what the look measured on this photo)."""
    notes = notes or {}
    return [f"{look_id} ({spec['title']}): {spec['use']}{_tone_hint(spec.get('tone') or {})}"
            + (f" {notes[look_id]}" if look_id in notes else "")
            for look_id, spec in LOOKS.items()]
