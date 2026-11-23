"""Vision-directed finishing on LightCraft.

A vision model (any provider configured in vision.py) reads the unedited
LightCraft baseline together with measured numbers (tone percentiles and an
8x8 table of cell brightness) and writes three complete versions:

- tone targets: where the finished picture's black, shadows, midtones,
  highlights and white must sit. A solver measures small renders and sets the
  global exposure and LightCraft's master curve so that they are met (tone.py).
  The model says what the picture should look like; it does not have to guess
  which slider values produce it.
- global character and colour in LightCraft's own develop vocabulary, with an
  optional look distilled from the reference preset library (looks.py).
- local masks placed in normalized coordinates: linear, radial, luminance range
  and colour range, combined by intersection and subtraction so local work
  follows real edges. A mask may carry a measured target for its region.

Each version is rendered, checked for pixel damage and reviewed by the model
against a professional finish with an itemised 0-100 score. The review may
return a complete revised plan; the best scoring acceptable round is delivered.

Background: the hand-directed probe of 2026-11-16 and the benchmark of
2026-11-17 (.task-cache/bench-2026-11-17), where slider-only plans came out
timid, darker than the baseline and nearly identical across versions.
Brightness ratio and distance from the baseline are aesthetic questions and go
to the visual review as information. The pixel gate rejects only damage.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import shutil
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import tone
from .config import Settings
from .lightcraft import RELATIVE_WB_MODE
from .looks import LOOKS, look_lines
from .vision import VisionUnavailable, chat_vision, is_temporary, probe, provider_from, safe_error

ENGINE = "lightcraft-directed"
SLOTS = ("A", "B", "C")
DEFAULT_TITLES = {"A": "自然完成", "B": "更有力度", "C": "风格化"}
ROLES = {
    "A": "Natural finish: faithful colour, the tonal range this scene deserves and a clear light hierarchy. The"
         " standard edit a careful professional would deliver; clearly better than the baseline.",
    "B": "Dramatic light: a visibly different TONAL structure from A. Deeper shadows and denser sky or"
         " surroundings, a brighter and more separated subject, stronger local dodge and burn. Its tone targets"
         " differ from A's. Not merely more saturation. The deeper shadows are the surroundings, never the subject:"
         " when the subject or a face sits in the darkest 10-20% (see the cell table), darken around it with masks"
         " and keep black and shadows near image 1. In a night or low-key frame whose darks are already near"
         " black there are no deeper shadows to find: B keeps the frame near image 1's darkness (median within"
         " about +0.5 EV of image 1, clearly below A's) and lifts only the subject's own lit surface, under a"
         " luminance-limited mask, so it reads as an island of light in the dark, not as a lifted frame. The"
         " ground, path, pavement and walls stay at image 1's brightness or darker (burn them a little with"
         " masks); do not lift a pool on the ground: lifting the lit ground is what A does, and a B that lifts it"
         " too looks like A on a board however much darker its mid is (a pool, if any, is small and right under"
         " the lamp). The frame highlights (p90) there are that lit ground: keep them near image 1 and at least"
         " 0.7 EV below A's. A natural A lifts a night less than a stop, so a B left near image 1 is still close"
         " to A: the subject's lit surface has to end clearly brighter than A leaves it (about +1 EV over image"
         " 1), with its box on the side the lamp actually hits, not its shaded side."
         " In a bright scene (sunlit sand, snow, salt, white walls) the sunlit surface stays luminous in B too"
         " (frame mid within about -0.3 EV of image 1): the depth comes from what is already in shade or"
         " colour (cliffs, shadow sides, sky), burned with masks, not from pulling the whole frame down. A"
         " subject darker than its bright surroundings (camels or people on sand, a figure against snow or sky)"
         " separates by staying darker against that luminous ground, with clarity or contrast on it; do not"
         " lift it towards the ground's brightness, which on a uniform surface makes a visible pool of light.",
    "C": "Colour style: A's light design carried into a distinct, tasteful PALETTE, normally built on one of the"
         " reference looks. Skin and white objects stay believable. Next to A it must read as another palette at"
         " a glance: change the colour of the LARGE areas (sky, water, ground, the overall white balance, the"
         " colour of the mid-tones), not only a split-tone in the darks or a small shift of one hue, which on a"
         " board looks like A again. With A's tone targets that takes the look at 0.8-1.0, or white balance,"
         " calibration, grading and colour targets on the large areas that visibly recolour them (mixer hue"
         " cannot: see mixer); a matte look also needs its lifted black and softened highlights written into the"
         " tone targets, or it reads as A slightly brighter.",
}

# Safety ranges in LightCraft units (exposure in EV, everything else on the
# -100..100 slider scale). The visual review and the damage gate decide, not a
# file-type rule.
GLOBAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "light": {"exposure": (-1.5, 1.5), "contrast": (-40, 50), "highlights": (-90, 30),
              "shadows": (-40, 70), "whites": (-50, 40), "blacks": (-50, 30)},
    "color": {"vibrance": (-40, 40), "saturation": (-40, 25)},
    "effects": {"texture": (-30, 35), "clarity": (-30, 35), "dehaze": (-20, 40)},
    "curve": {"highlights": (-40, 40), "lights": (-40, 40), "darks": (-40, 40), "shadows": (-40, 40)},
    "vignette": {"amount": (-40, 20), "midpoint": (0, 100), "feather": (0, 100)},
    "detail": {"sharpen_amount": (0, 80), "nr_luminance": (0, 40), "nr_color": (0, 60)},
    "calibration": {"red_hue": (-40, 60), "red_sat": (-40, 80), "green_hue": (-60, 60), "green_sat": (-60, 40),
                    "blue_hue": (-70, 40), "blue_sat": (-60, 80), "shadows_tint": (-20, 20)},
    "grain": {"amount": (0, 45), "size": (0, 60), "roughness": (0, 100)},
}
# Kelvin offset from as-shot. A JPEG renders identically at custom 6500 K / 0.
WB_RANGES = {"temp": (-1500.0, 1500.0), "tint": (-30.0, 30.0)}
JPEG_NEUTRAL_KELVIN = 6500.0
MIXER_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
MIXER_RANGES = {"hue": (-30.0, 30.0), "sat": (-50.0, 40.0), "lum": (-40.0, 40.0)}
GRADING_WHEELS = ("shadows", "midtones", "highlights")
WHEEL_RANGES = {"hue": (0.0, 360.0), "sat": (0.0, 30.0), "lum": (-30.0, 30.0)}
GRADING_RANGES = {"blending": (0.0, 100.0), "balance": (-100.0, 100.0)}
LOCAL_RANGES = {"exposure": (-1.2, 1.2), "contrast": (-40, 40), "highlights": (-80, 40), "shadows": (-40, 60),
                "whites": (-50, 40), "blacks": (-40, 30), "temp": (-30, 30), "tint": (-25, 25),
                "saturation": (-50, 35), "clarity": (-30, 35), "texture": (-30, 35), "dehaze": (-20, 45)}
# What a look and the model's own values may add up to.
MERGED_MIXER_RANGES = {"hue": (-60.0, 60.0), "sat": (-85.0, 60.0), "lum": (-45.0, 45.0)}
MERGED_COLOR_RANGES = {"vibrance": (-60.0, 60.0), "saturation": (-60.0, 40.0)}
MERGED_WHEEL_RANGES = {"hue": (0.0, 360.0), "sat": (0.0, 45.0), "lum": (-30.0, 30.0)}
# The solver may move global exposure further than the model is allowed to ask for directly.
SOLVER_EXPOSURE_RANGE = (-2.5, 2.5)
# Exposure the solver may put on a mask. A mask with a pixel-accurate part (luminance or colour, as its
# shape or its limit) follows the thing it is on; a bare gradient or ellipse shows as a blob or a halo
# when it is pushed, so it gets less and the report says how to make it precise.
SOLVER_LOCAL_RANGE = (-1.25, 1.25)
SOLVER_PLAIN_RANGE = (-0.7, 0.7)
# Global saturation the solver may end on for a frame colourfulness target (look included).
SOLVER_SATURATION_RANGE = (-40.0, 30.0)
# A RAW's negative is grey (desert DSC05752 frame colourfulness .051; the photographer's finished edit .318): with
# vibrance and a warm white balance, saturation +60 against +30 still added a sixth (.186 -> .217).
RAW_SOLVER_SATURATION_RANGE = (-40.0, 55.0)
# With a look, the look is the version's colour: the frame colourfulness target only nudges global
# saturation this far from where the look puts it. Held to the target in p4-r1, desert_gold lost half its
# colour and matte_landscape its matte, and both C versions came out within 5 of A.
LOOK_SATURATION_ROOM = 10.0
# Upward room on a RAW negative: the looks were built on developed pictures and barely colour the grey negative
# (ref-r3: desert_gold x1.02-1.06 on DSC05752/05746, his finished dunes x5-6), so C stayed at the negative's colour.
# The chroma target there is the development; the look keeps its hues and the downward guard stays.
RAW_LOOK_SATURATION_UP = 45.0
# Before planning, every look is rendered once at full strength on the photo and its frame colourfulness is
# given to the planner. Without it the plans wrote a "natural" chroma under a look that adds or takes far more:
# p18-r1 DSC_1441 C0-C2 (desert_gold 0.75-0.8, which gives 1.31x image 1 at strength 1.0) wrote 10.1-12.8
# against 12.2, the solver hit -10 in all three and the frames stayed at 15.9-19.0; on DSC_2121 matte_landscape
# (0.79x) and cine_teal (0.65x) sat under plans that asked 1.1-1.3x. Measured on the three dev photos (13 looks),
# colourfulness moves in proportion to strength (strength 0.6 lands within 0.2 of the straight line on the
# daylight frames, 0.4 on the night at 2.6) and global saturation +/-10 moves it about +/-10%, so one render per
# look is enough.
# The same probes give each look's mean Lab distance from the unedited render, the measure that calls sibling
# versions nearly identical below 5: on DSC_2620 (frosty plain) every look is 1.7-2.8 at strength 1.0, on the
# old night/desert/lake set mostly 1-4.5, on DSC_1515/1963 4.5-9.3; C versions under a look below 5 came out
# 4.0 from A (p20-r1 2620 C0 snow_luminous, -6 finish; p19-r1 2121 C0 desert_gold at 2.0, 4.0 from A, -6).
# Distance grows about in proportion to strength (0.6 gives 0.58-0.77 of it on the three old dev photos).
LOOK_PROBE_STRENGTH = 1.0
LOOK_PROBE_WORKERS = 4
# The darkest vignette a planned version gets (a user's refinement is not held). In p4-r1 every round with
# -15 or more lost points for it: uneven light on uniform sand, dead corners on a dark lower half.
VIGNETTE_FLOOR = {"A": -10.0, "B": -14.0, "C": -14.0}
# Global controls that are left out of a plan with at least TONE_OWNS_FROM percentile targets: the
# solver places those tones with exposure and the master curve, and a second set of tone controls
# under it only has to be undone by a steeper curve.
# Replaying 27 benchmark plans with and without them: dropped, the curve was less steep (plans with a
# slope of 3.0 or more went from 18 to 10), bright sand kept its whites and a face in shadow stayed readable.
TONE_OWNED: dict[str, tuple[str, ...]] = {
    "light": ("exposure", "contrast", "highlights", "shadows", "whites", "blacks"),
    "curve": ("highlights", "lights", "darks", "shadows"),
}
TONE_OWNS_FROM = 3
DEFAULT_CURVE_SATURATION = 50.0
# LightCraft renders a 640 px frame differently from a delivery-sized one (the same document at 640 against
# 800-1600: p12-r1 DSC_2121 B black 0.046 against 0.078-0.082, DSC_1441 A 0.130 against 0.144, white 0.004-0.014
# higher), so a solver probing at 640 met targets the delivered frame missed. From 800 up the frame reads
# within 0.006 of the delivered one, at the same render time (about 1.4 s).
PROBE_EDGE = 960
MAX_MASKS = 6
SHAPE_KINDS = ("linear", "radial", "luminance", "color")
GENRES = {"landscape", "portrait", "street", "night", "indoor", "product", "general"}
DELIVERY_SCORE = 95.0
# Seconds to wait before asking again when the vision call itself fails (no answer at all, unlike a bad answer).
# p16-r1: 8 of 21 reviews came back "vision unavailable" in between successful ones while three photos ran at
# once; each one ended that version's revisions on the spot (DSC_2121 B never got a review).
VISION_RETRY_WAITS = (20.0, 60.0)

# Legacy gate codes that judge brightness, amount of change or HSV saturation,
# not damage. HSV saturation reads dark teal water or a deepened sky as
# "oversaturated"; strong colour is checked on chroma below.
ADVISORY_CODES = {"too_dark", "too_bright", "excessive_distance", "exposure_unresolved",
                  "exposure_lift_unresolved", "oversaturation", "green_oversaturation"}
# Hard bounds on LINEAR mean luma versus the baseline (about -1.75..+2.3 EV).
# Only catastrophes: brightness is for the reviewer and the tone targets.
LUMA_RATIO_BOUNDS = (0.3, 5.0)
# A RAW's baseline is its flat, dark undeveloped render (see tone.RAW_SHIFT_LIMITS): the photographer's finished
# dune DSC05752 is x4.72 its linear mean, at the edge of the JPEG bound.
RAW_LUMA_RATIO_BOUNDS = (0.3, 8.0)

# LightCraft's luminance-range scale against display luma, measured on 0.21.1
# (.task-cache/bench-2026-11-17/tools/lumscale.py up to 0.8; above that .task-cache/bench-2026-11-18/tools/lumcal.py,
# which darkens instead of lifting so bright pixels do not clip). The engine's scale is logarithmic in light (about
# 1.2 EV per 0.1) and display white sits near 0.875, not 1.0: the old top entry (display 0.9 = engine 0.9) put a
# limit written at display .86 at engine .871 = display .99 (p17-r1 DSC_1441 B2: the far-dune mask, lo .86, reached
# 2% of sand at .90), and a camel's hi .80 at engine .829 = display .84, feathered to .91 (the halo on the sand).
_LUMA_DISPLAY = (0.0, 0.01, 0.045, 0.09, 0.145, 0.225, 0.355, 0.525, 0.63, 0.76, 0.82, 0.875, 0.915, 0.95, 0.99, 1.0)
_LUMA_ENGINE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87, 0.875)
_KIND_ALIASES = {"luminancerange": "luminance", "luma": "luminance", "brightness": "luminance",
                 "colorrange": "color", "colour": "color", "colourrange": "color", "gradient": "linear"}
_KIND_LABELS = {"linear": "线性渐变", "radial": "径向", "luminance": "亮度范围", "color": "颜色范围"}


class DirectedUnavailable(RuntimeError):
    """The directed path cannot produce a delivery; callers may fall back."""


# --------------------------------------------------------------------------
# Plan sanitizing


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamped(raw: Any, ranges: dict[str, tuple[float, float]], path: str, notes: list[str], *,
             wrap_hue: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        notes.append(f"{path}: 不是对象，已忽略")
        return out
    for key, value in raw.items():
        if key not in ranges:
            notes.append(f"{path}.{key}: 不支持，已忽略")
            continue
        number = _number(value)
        if number is None:
            notes.append(f"{path}.{key}: 不是数值，已忽略")
            continue
        if wrap_hue and key == "hue":
            number = number % 360.0
        lo, hi = ranges[key]
        clipped = min(hi, max(lo, number))
        if abs(clipped - number) > 1e-9:
            notes.append(f"{path}.{key}: {number:g} 超出范围，按 {clipped:g} 执行")
        out[key] = round(clipped, 4)
    return out


def _point(raw: Any, lo: float, hi: float) -> dict[str, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        x, y = _number(raw[0]), _number(raw[1])
    elif isinstance(raw, dict):
        x, y = _number(raw.get("x")), _number(raw.get("y"))
    else:
        return None
    if x is None or y is None:
        return None
    return {"x": round(min(hi, max(lo, x)), 4), "y": round(min(hi, max(lo, y)), 4)}


def _unit(value: float | None, default: float) -> float:
    """A 0..1 value that a model may also have written as 0..100."""
    if value is None:
        return default
    return value / 100.0 if value > 1.0 else value


def _colourfulness(value: float) -> float:
    """Colourfulness as 0..1; the tables show it x100, so anything above 0.6 is read on that scale."""
    return round(min(0.6, max(0.0, value / 100.0 if value > 0.6 else value)), 4)


def _shown(plan: dict[str, Any]) -> dict[str, Any]:
    """A plan as the prompts show it to the model: chroma on the x100 scale the prompt asks for and the measurements
    use (``sanitize_plan`` keeps it 0..1). Since p28 (change 40): p27-r1 DSC_1963 B2 color -3 read the plan's 0.05
    next to the measured "target 5" as "前景砾石的色度目标写成 chroma 5.0，而同一字段在天空遮罩里是 0.36（即 x100 的
    36），尺度写错了一个数量级". A value shown under 0.7 would be read back as 0..1, so a trace of colour shows as 0.7."""
    def x100(value: Any) -> Any:
        if not isinstance(value, (int, float)) or value <= 0:
            return value
        return max(0.7, round(value * 100.0, 1))

    out = copy.deepcopy(plan)
    if isinstance(out.get("tone"), dict) and "chroma" in out["tone"]:
        out["tone"]["chroma"] = x100(out["tone"]["chroma"])
    for mask in out.get("masks") or []:
        target = mask.get("target") if isinstance(mask, dict) else None
        if isinstance(target, dict) and "chroma" in target:
            target["chroma"] = x100(target["chroma"])
    return out


def _box(raw: Any) -> list[float] | None:
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4 and all(_number(v) is not None for v in raw)):
        return None
    left, top, right, bottom = (min(1.0, max(0.0, float(v))) for v in raw)
    if right - left < 0.02 or bottom - top < 0.02:
        return None
    return [round(left, 4), round(top, 4), round(right, 4), round(bottom, 4)]


def _shape(raw: Any, path: str, notes: list[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        notes.append(f"{path}: 缺少形状")
        return None
    kind = str(raw.get("kind") or "").strip()
    kind = _KIND_ALIASES.get(kind.lower().replace("_", "").replace("-", ""), kind)
    if kind == "linear":
        start, end = _point(raw.get("start"), -0.5, 1.5), _point(raw.get("end"), -0.5, 1.5)
        if start is None or end is None or math.dist((start["x"], start["y"]), (end["x"], end["y"])) < 0.02:
            notes.append(f"{path}: 线性渐变缺少有效的起止点")
            return None
        return {"kind": "linear", "start": start, "end": end}
    if kind == "radial":
        center = _point(raw.get("center"), -0.25, 1.25)
        rx, ry = _number(raw.get("rx")), _number(raw.get("ry"))
        if center is None or rx is None or ry is None or rx <= 0 or ry <= 0:
            notes.append(f"{path}: 径向蒙版缺少中心或半径")
            return None
        angle = _number(raw.get("angle"))
        feather = _number(raw.get("feather"))
        return {"kind": "radial", "center": center,
                "rx": round(min(1.2, max(0.01, rx)), 4), "ry": round(min(1.2, max(0.01, ry)), 4),
                "angle": round(min(180.0, max(-180.0, angle or 0.0)), 2),
                "feather": round(min(100.0, max(0.0, 70.0 if feather is None else feather)), 2),
                "invert": raw.get("invert") is True}
    if kind == "luminance":
        lo, hi = _number(raw.get("lo")), _number(raw.get("hi"))
        if max(lo or 0.0, hi or 0.0) > 1.0:  # Written on the 0..100 scale of the tables.
            lo, hi = (None if lo is None else lo / 100.0), (None if hi is None else hi / 100.0)
        lo = min(1.0, max(0.0, 0.0 if lo is None else lo))
        hi = min(1.0, max(0.0, 1.0 if hi is None else hi))
        if hi - lo < 0.03 or (lo <= 0.0 and hi >= 1.0):
            notes.append(f"{path}: 亮度范围无效（需要 lo < hi，且不能覆盖全部亮度）")
            return None
        feather = min(0.3, max(0.0, _unit(_number(raw.get("feather")), 0.06)))
        return {"kind": "luminance", "lo": round(lo, 4), "hi": round(hi, 4), "feather": round(feather, 4),
                "invert": raw.get("invert") is True}
    if kind == "color":
        items = raw.get("points") if isinstance(raw.get("points"), list) else [raw.get("point")]
        points = [point for point in (_point(item, 0.005, 0.995) for item in items) if point is not None][:5]
        if not points:
            notes.append(f"{path}: 颜色范围缺少取样点")
            return None
        spread = _number(raw.get("range"))
        return {"kind": "color", "points": points,
                "range": round(min(100.0, max(0.0, 50.0 if spread is None else spread)), 2),
                "invert": raw.get("invert") is True}
    notes.append(f"{path}: 不支持的蒙版类型 {kind!r}")
    return None


def _target(raw: Any, path: str, notes: list[str]) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        notes.append(f"{path}.target: 不是对象，已忽略")
        return None
    box = _box(raw.get("box"))
    median = _number(raw.get("median") if raw.get("median") is not None else raw.get("luma"))
    hue = _number(raw.get("hue"))
    chroma = _number(raw.get("chroma") if raw.get("chroma") is not None else raw.get("colourfulness"))
    if box is None or (median is None and chroma is None):
        notes.append(f"{path}.target: 需要 box，以及 median 或 hue 加 chroma，已忽略")
        return None
    out: dict[str, Any] = {"box": box}
    if median is not None:
        out["median"] = round(min(0.97, max(0.02, _unit(median, 0.5))), 4)
    if chroma is not None:
        amount = _colourfulness(chroma)
        if hue is None and amount > 0.06:   # Only a near-neutral target can do without a hue.
            notes.append(f"{path}.target: 有 chroma 没有 hue，颜色目标已忽略")
        else:
            out["hue"], out["chroma"] = round((hue or 0.0) % 360.0, 1), amount
    elif hue is not None:
        notes.append(f"{path}.target: 有 hue 没有 chroma，颜色目标已忽略")
    return out if len(out) > 1 else None


def _masks(raw: Any, path: str, notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    items = raw if isinstance(raw, list) else []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        where = f"{path}[{index}]"
        shape = _shape(item.get("shape"), where, notes)
        adjust = _clamped(item.get("adjust"), LOCAL_RANGES, f"{where}.adjust", notes)
        if shape is None:
            continue
        target = _target(item.get("target"), where, notes)
        if target is None and not any(abs(value) > 1e-6 for value in adjust.values()):
            notes.append(f"{where}: 没有任何调整，已忽略")
            continue
        name = str(item.get("name") or "").strip()[:40] or f"mask{index}"
        if name in names:
            name = f"{name}-{index}"
        names.add(name)
        mask: dict[str, Any] = {"name": name, "purpose": str(item.get("purpose") or "").strip()[:160],
                                "shape": shape, "adjust": adjust}
        for key in ("limit", "exclude"):
            if item.get(key) is not None:
                extra = _shape(item[key], f"{where}.{key}", notes)
                if extra is not None:
                    mask[key] = extra
        edge = _number(item.get("edge"))
        if edge is not None and edge > 0:
            mask["edge"] = round(min(100.0, edge), 2)
        if target is not None:
            mask["target"] = target
        out.append(mask)
        if len(out) == MAX_MASKS:
            if len(items) > index:
                notes.append(f"{path}: 最多执行 {MAX_MASKS} 个蒙版，其余已忽略")
            break
    return out


def _look(raw: Any, path: str, notes: list[str]) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        notes.append(f"{path}.look: 不是对象，已忽略")
        return None
    look_id = str(raw.get("id") or "").strip()
    if look_id in {"", "none", "null"}:
        return None
    if look_id not in LOOKS:
        notes.append(f"{path}.look: 没有名为 {look_id!r} 的风格，已忽略")
        return None
    strength = _number(raw.get("strength"))
    strength = 0.8 if strength is None else (strength / 100.0 if strength > 1.5 else strength)
    return {"id": look_id, "strength": round(min(1.0, max(0.2, strength)), 3)}


def sanitize_plan(raw: Any, *, slot: str) -> tuple[dict[str, Any], list[str]]:
    """Return a plan restricted to supported controls and safe ranges, plus notes."""
    if not isinstance(raw, dict):
        raise ValueError(f"{slot}: plan is not an object")
    notes: list[str] = []
    develop_raw = raw.get("develop") if isinstance(raw.get("develop"), dict) else {}
    develop: dict[str, Any] = {}
    for group, ranges in GLOBAL_RANGES.items():
        values = _clamped(develop_raw.get(group), ranges, f"{slot}.{group}", notes)
        if values:
            develop[group] = values
    wb = _clamped(develop_raw.get("wb"), WB_RANGES, f"{slot}.wb", notes)
    if wb:
        develop["wb"] = wb
    mixer: dict[str, dict[str, float]] = {}
    raw_mixer = develop_raw.get("mixer")
    for band, values in (raw_mixer.items() if isinstance(raw_mixer, dict) else []):
        if band not in MIXER_BANDS:
            notes.append(f"{slot}.mixer.{band}: 不支持，已忽略")
            continue
        clean = _clamped(values, MIXER_RANGES, f"{slot}.mixer.{band}", notes)
        if clean:
            mixer[band] = clean
    if mixer:
        develop["mixer"] = mixer
    grading: dict[str, Any] = {}
    raw_grading = develop_raw.get("grading")
    for key, values in (raw_grading.items() if isinstance(raw_grading, dict) else []):
        if key in GRADING_WHEELS:
            clean = _clamped(values, WHEEL_RANGES, f"{slot}.grading.{key}", notes, wrap_hue=True)
            if clean:
                grading[key] = clean
        elif key in GRADING_RANGES:
            grading.update(_clamped({key: values}, GRADING_RANGES, f"{slot}.grading", notes))
        else:
            notes.append(f"{slot}.grading.{key}: 不支持，已忽略")
    if any(key in grading for key in GRADING_WHEELS):
        develop["grading"] = grading
    known = {*GLOBAL_RANGES, "wb", "mixer", "grading"}
    notes.extend(f"{slot}.{group}: 不支持，已忽略" for group in develop_raw if group not in known)
    plan: dict[str, Any] = {
        "title": str(raw.get("title") or DEFAULT_TITLES.get(slot, slot)).strip()[:30],
        "intent": str(raw.get("intent") or "").strip()[:300],
        "develop": develop,
        "masks": _masks(raw.get("masks"), f"{slot}.masks", notes),
    }
    tone_raw = raw.get("tone")
    targets: dict[str, float] = tone.clean_targets(tone_raw, notes, f"{slot}.tone")
    if targets and isinstance(tone_raw, dict):
        saturation = _number(tone_raw.get("saturation"))
        if saturation is not None:
            targets["saturation"] = round(min(100.0, max(0.0, saturation)), 2)
    if isinstance(tone_raw, dict):
        chroma = _number(tone_raw.get("chroma") if tone_raw.get("chroma") is not None
                         else tone_raw.get("colourfulness"))
        if chroma is not None and chroma > 0:
            targets["chroma"] = _colourfulness(chroma)
    if targets:
        plan["tone"] = targets
    if sum(name in tone.ANCHORS for name in targets) >= TONE_OWNS_FROM:
        for group, names in TONE_OWNED.items():
            values = develop.get(group) or {}
            dropped = [name for name in names if abs(values.pop(name, 0.0)) > 1e-6]
            if group in develop and not values:
                del develop[group]
            if dropped:
                notes.append(f"{slot}.{group}: {'、'.join(dropped)} 由影调目标接管（求解器用曝光和主曲线落点），未执行")
    vignette = develop.get("vignette") or {}
    floor = VIGNETTE_FLOOR.get(slot)
    if floor is not None and vignette.get("amount", 0.0) < floor:
        notes.append(f"{slot}.vignette.amount: {vignette['amount']:+.0f} 过重（在均匀的沙、雪、天空上读作打光不均，"
                     f"在本来就暗的角落里堵成死黑），按 {floor:+.0f} 执行")
        vignette["amount"] = floor
    look = _look(raw.get("look"), slot, notes)
    if look is not None:
        plan["look"] = look
    return plan, notes


def is_executable(plan: dict[str, Any]) -> bool:
    return bool(plan.get("develop") or plan.get("masks") or plan.get("tone") or plan.get("look"))


# --------------------------------------------------------------------------
# Compilation to a LightCraft develop document


def _merged_develop(plan: dict[str, Any]) -> dict[str, Any]:
    """The model's develop values with the chosen look added underneath."""
    develop = copy.deepcopy(plan.get("develop") or {})
    look = plan.get("look") or {}
    spec = LOOKS.get(look.get("id") or "")
    if spec is None:
        return develop
    strength = float(look.get("strength", 0.8))
    for group, values in (spec.get("develop") or {}).items():
        if group == "mixer":
            mixer = develop.setdefault("mixer", {})
            for band, hsl in values.items():
                target = mixer.setdefault(band, {})
                for key, value in hsl.items():
                    lo, hi = MERGED_MIXER_RANGES[key]
                    target[key] = round(min(hi, max(lo, target.get(key, 0.0) + value * strength)), 2)
        elif group == "grading":
            grading = develop.setdefault("grading", {})
            for key, value in values.items():
                if key in GRADING_WHEELS:
                    if key not in grading:  # The model's own wheel replaces the look's.
                        grading[key] = {"hue": float(value.get("hue", 0.0)),
                                        "sat": round(min(MERGED_WHEEL_RANGES["sat"][1], value.get("sat", 0.0) * strength), 2),
                                        "lum": round(value.get("lum", 0.0) * strength, 2)}
                elif key not in grading:
                    grading[key] = float(value)
        else:
            ranges = MERGED_COLOR_RANGES if group == "color" else GLOBAL_RANGES.get(group, {})
            target = develop.setdefault(group, {})
            for key, value in values.items():
                lo, hi = ranges.get(key, (-100.0, 100.0))
                target[key] = round(min(hi, max(lo, target.get(key, 0.0) + value * strength)), 2)
    return develop


def tone_targets(plan: dict[str, Any]) -> dict[str, float]:
    """Anchor targets of a plan as written.

    A look's black and white handling is offered to the planner as numbers (``look_lines``), not added to
    what it wrote: "tone" describes the finished frame, and the reviewer and the next revision read it that
    way. Added silently, a matte look moved every C version's black 0.05-0.06 above its written target;
    the review called the result grey fog, and each revision copied the measured black back into the plan,
    which the look then lifted again (p5-r1: written black 0.07 -> 0.19, executed 0.12 -> 0.25).
    """
    targets = {name: value for name, value in (plan.get("tone") or {}).items() if name in tone.ANCHORS}
    return tone._increasing({name: targets[name] for name in tone.ORDER if name in targets}) if targets else {}


def _engine_luma(value: float, exposure: float) -> float:
    """Display luma of the unedited picture as LightCraft's luminance-range value.

    The engine evaluates range masks after global exposure, so a range read off
    the baseline has to move with it.
    """
    shifted = tone.to_display(min(1.0, tone.to_linear(value) * 2.0 ** exposure))
    return float(np.interp(shifted, _LUMA_DISPLAY, _LUMA_ENGINE))


# The planning prompt (and mask_overlay) gives a radial's rx as a fraction of the picture's width and ry of its
# height; LightCraft measures both on the long edge (tools/geom_probe.py: rx = ry = .3 on a flat 1200x800 frame
# reached 360 px both ways, on 800x1200 too). Until p25 radials went to the engine as written, so on a 3:2 landscape
# each was 1.5 times as tall as planned: the p24 DSC_1963 B0 face ellipse (centre y .56, ry .26) reached from y .17
# to .95 instead of .30 to .82, into the sky where it left a glow. Within the ellipse the weight is full out to
# 1 - feather/100 of the radius, then falls as 1 - smoothstep to 0 at the edge.
def _engine_shape(shape: dict[str, Any], exposure: float,
                  colors: Callable[[list[dict[str, float]]], list[list[float]]] | None,
                  aspect: float | None = None) -> tuple[dict[str, Any], bool] | None:
    """LightCraft shape and component-level invert, or None when it cannot be built.

    ``aspect`` (width / height of the picture) turns a radial's rx (a fraction of the width) and ry (of the
    height) into LightCraft's, which measures both on the long edge."""
    kind = shape["kind"]
    if kind == "linear":
        return {"kind": "linear", "start": dict(shape["start"]), "end": dict(shape["end"])}, False
    if kind == "radial":
        engine = copy.deepcopy(shape)
        if aspect and aspect > 0:
            engine["rx"] = round(float(shape["rx"]) * min(1.0, aspect), 4)
            engine["ry"] = round(float(shape["ry"]) * min(1.0, 1.0 / aspect), 4)
        return engine, False
    if kind == "luminance":
        lo = 0.0 if shape["lo"] <= 0.0 else _engine_luma(shape["lo"], exposure)
        hi = 1.0 if shape["hi"] >= 1.0 else _engine_luma(shape["hi"], exposure)
        feather = shape.get("feather", 0.06)
        lo_feather = 0.0 if shape["lo"] <= 0.0 else max(0.0, lo - _engine_luma(max(0.0, shape["lo"] - feather), exposure))
        hi_feather = 0.0 if shape["hi"] >= 1.0 else max(0.0, _engine_luma(min(1.0, shape["hi"] + feather), exposure) - hi)
        if hi - lo < 0.01:
            return None
        return {"kind": "luminanceRange", "lo": round(lo, 4), "hi": round(hi, 4),
                "lo_feather": round(lo_feather, 4), "hi_feather": round(hi_feather, 4)}, bool(shape.get("invert"))
    if kind == "color":
        if colors is None:
            return None
        try:
            samples = colors(shape["points"])
        except Exception:  # Sampling is a separate engine run; the mask is dropped with a note.
            return None
        if not samples:
            return None
        # Oklab-like values scale with the cube root of linear light.
        scale = 2.0 ** (exposure / 3.0)
        return {"kind": "colorRange", "refine": float(shape.get("range", 50.0)),
                "samples": [[round(float(v) * scale, 6) for v in sample] for sample in samples]}, bool(shape.get("invert"))
    return None


def compile_document(plan: dict[str, Any], source_kind: str, *,
                     colors: Callable[[list[dict[str, float]]], list[list[float]]] | None = None,
                     curve: list[list[float]] | None = None,
                     aspect: float | None = None) -> tuple[dict[str, Any], list[str]]:
    """Translate a sanitized plan into a LightCraft develop document.

    ``colors`` returns engine colour samples for normalized points (needed by
    colour-range masks); ``curve`` is the solved master curve; ``aspect`` (width /
    height) sizes radials as the plan means them (see _engine_shape).
    """
    notes: list[str] = []
    develop = _merged_develop(plan)
    document: dict[str, Any] = {}
    for group in ("light", "color", "effects", "curve", "mixer", "grading", "vignette", "detail", "calibration",
                  "grain"):
        if develop.get(group):
            document[group] = copy.deepcopy(develop[group])
    if curve:
        section = document.setdefault("curve", {})
        section["master"] = [[float(x), float(y)] for x, y in curve]
        section["refine_saturation"] = float((plan.get("tone") or {}).get("saturation", DEFAULT_CURVE_SATURATION))
    wb = develop.get("wb") or {}
    if abs(wb.get("temp", 0.0)) > 1e-6 or abs(wb.get("tint", 0.0)) > 1e-6:
        if source_kind in {"jpeg", "png", "tiff"}:
            document["wb"] = {"mode": "custom", "temp": round(JPEG_NEUTRAL_KELVIN + wb.get("temp", 0.0), 1),
                              "tint": round(wb.get("tint", 0.0), 2)}
        else:   # Made absolute from the RAW's as-shot reading when rendered (lightcraft.resolve_wb).
            document["wb"] = {"mode": RELATIVE_WB_MODE, "temp": round(wb.get("temp", 0.0), 1),
                              "tint": round(wb.get("tint", 0.0), 2)}
    exposure = float((develop.get("light") or {}).get("exposure", 0.0))
    masks = []
    for mask in plan.get("masks") or []:
        main = _engine_shape(mask["shape"], exposure, colors, aspect)
        if main is None:
            notes.append(f"蒙版「{mask['name']}」的{_KIND_LABELS.get(mask['shape']['kind'], '形状')}无法建立，未执行")
            continue
        components = [{"op": "add", "invert": main[1], "shape": main[0]}]
        for key, op in (("limit", "intersect"), ("exclude", "subtract")):
            if mask.get(key):
                extra = _engine_shape(mask[key], exposure, colors, aspect)
                if extra is None:
                    notes.append(f"蒙版「{mask['name']}」的 {key} 无法建立，按不带该条件执行")
                    continue
                components.append({"op": op, "invert": extra[1], "shape": extra[0]})
        compiled = {"id": len(masks) + 1, "name": mask["name"], "visible": True, "invert": False, "on_photo": True,
                    "components": components, "adjust": {"amount": 100.0, **mask["adjust"]}}
        if mask.get("edge"):
            compiled["refine"] = float(mask["edge"])
        masks.append(compiled)
    if masks:
        document["masks"] = masks
    return document, notes


def _look_saturation(plan: dict[str, Any]) -> float:
    """Global saturation that the plan's look adds under the plan's own value."""
    look = plan.get("look") or {}
    spec = LOOKS.get(look.get("id") or "")
    if spec is None:
        return 0.0
    return float(((spec.get("develop") or {}).get("color") or {}).get("saturation", 0.0)) * float(look.get("strength", 0.8))


def is_precise(mask: dict[str, Any]) -> bool:
    """True when a mask follows the picture (a luminance or colour range as its shape or its limit)."""
    return any((mask.get(key) or {}).get("kind") in {"luminance", "color"} for key in ("shape", "limit"))


# A RAW's unedited render is the negative: its light near-neutrals (grey sand, a cast sky or snow) are not the
# scene's, and developing it moves them. Against the negative, the photographer's own finished edits of the
# reference RAWs reach 1.00-2.24 times the natural-colour gate (ref-r7 regions: DSC05746 2.24, DSC05752 1.15,
# DSC02270 1.08, DSC02290 1.00), and the guard held version A's dunes under 0.9 of it (DSC05746 A kept 0% of the
# solver's colour, 0.61 of his). On a RAW the gate and the guard take this multiple of it; a cast past it
# (ref-r7 DSC05752 C: 4.1) still fails.
RAW_NEUTRAL_GATE = 2.5


def neutral_nearness(cast: dict[str, Any], gate: float = 1.0) -> float:
    """How near a ``neutral_cast`` measurement is to the natural-colour gate times ``gate`` (1.0 at it)."""
    from .lightroom_qa import NEUTRAL_INTRODUCED, NEUTRAL_SHIFT

    if "median_shift" not in cast:
        return 0.0
    return min(cast["median_shift"] / NEUTRAL_SHIFT, cast["introduced_chroma"] / NEUTRAL_INTRODUCED) / gate


def neutral_guard(baseline_render: Path, gate: float = 1.0) -> Callable[[np.ndarray], float]:
    """How near a solver preview's near-neutrals are to the natural-colour gate times ``gate`` (1.0 at it).

    Measured against the unedited render as the gate measures the delivery render, on the preview's size.
    """
    from .lightroom_qa import neutral_cast

    with Image.open(baseline_render) as opened:
        reference = ImageOps.exif_transpose(opened).convert("RGB")
    sized: dict[tuple[int, int], np.ndarray] = {}

    def nearness(rgb: np.ndarray) -> float:
        size = (rgb.shape[1], rgb.shape[0])
        if size not in sized:
            sized[size] = np.asarray(reference.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return neutral_nearness(neutral_cast(sized[size], rgb), gate)

    return nearness


# A color-shape mask selects its own hue, so a temp/tint turn stays on that colour instead of casting the
# whole area a linear or radial covers: it gets a wider range for a colour target. Saturated water needs
# tint -60..-100 to turn 9-10 degrees (tools/hue_probe.py on p13 DSC_2121; mixer hue +/-100 turns ~4).
COLOUR_MASK_LIMITS: dict[str, tuple[float, float]] | None = {"temp": (-60.0, 60.0), "tint": (-60.0, 60.0)}
# Taking colour out never casts: on a RAW a mask with a colour target may desaturate further than the solver's
# usual -40. The negative's blue sky is far more colourful than the finished one (DSC02262: .22 in image 1, his
# sky .05); ref-r9 A held the sky mask at -40 with global saturation at -40 too and the sky kept .13. A probe of
# that render: mask saturation -40 -> sky chroma 12.2, -70 -> 5.2, -100 -> 2.4 (x100).
RAW_MASK_SATURATION_FLOOR = -70.0
# The photographer colours nothing locally: his 19 RAW edits (resources/) hold 76 mask corrections and every one has
# local temperature, tint and saturation 0; his colour is wb, vibrance, HSL and grading. On a RAW the wide color-mask
# turn above painted the colour instead (ref-r13: DSC05752 B lit sand tint +47, hue 18 for 33, salmon, where tint 0
# renders hue 34; DSC05749 C shade tint +49 at saturation -68, mauve; DSC05746 A shade temp -56 over a third of the
# frame, a grey foreground). A RAW mask keeps temp and tint to a nudge on every shape.
RAW_MASK_COLOUR_LIMITS: dict[str, tuple[float, float]] = {"temp": (-24.0, 24.0), "tint": (-10.0, 10.0)}

# A lifting radial whose luminance limit reaches past its subject also lifts what is brighter than the subject in
# its ellipse, and for one EV a bright pixel rises far more on the display than a dark one. p24-r1 DSC_1963 B0
# (face box at .32 in image 1, limit hi .68 feather .05, ellipse ry .26 reaching into the sky): rendered as
# solved, the sky above the head rose +.20 and the face box +.09, "画面 x≈0.49、y≈0.30-0.38 出现一块明显偏白偏亮的雾状
# 光斑" (subject -5, technical -4). Over p20-p24, 39 of the 72 reviews of a lifting radial with a luminance limit
# deducted for a halo or recorded halo damage. A halo shows on smooth ground (sky, haze), so before solving each
# such mask (with a lift target, or a planned exposure of SPILL_MIN_EV or more) is stepped +1 EV on a probe:
# where pixels outside its box that image 1 has brighter than the box median + SPILL_ABOVE and smooth (3x3 std
# under SPILL_SMOOTH) rise more than the box does (and more than SPILL_FLOOR) over at least SPILL_AREA of the
# frame, the limit's hi is brought under them, then a feather lower at a time (never under the box median) while
# such pixels still rise over SPILL_FAINT per EV on SPILL_AREA, at most SPILL_STEPS probes. Rendered as solved,
# 1963 B0 at hi .575 kept a faint patch (.9% of smooth sky up +.02, under .03 per EV on the probe); at .535, where
# the guard now stops on all three p24 versions, the sky stayed flat and the face box rose +.07 instead of +.09.
# Textured brighter pixels are left alone: on DSC_2620 B1 (dark foal on frost) the frost round the legs is
# textured, and narrowing that limit halved the foal's lift. The white-yak band (1963 B, lo .45, exposure +.18, no
# lift target) rose its surroundings at most +.02, so it is no candidate.
SPILL_ABOVE = 0.10
SPILL_FLOOR = 0.03
SPILL_FAINT = 0.015
SPILL_AREA = 0.002
SPILL_FEATHER = 0.04
SPILL_SMOOTH = 0.012
SPILL_MIN_EV = 0.3
SPILL_STEPS = 3


def guard_spill(plan: dict[str, Any], source_kind: str, render: Callable[[dict[str, Any]], Path],
                image: Image.Image | Path, *, colors: Callable[[list[dict[str, float]]], list[list[float]]] | None = None
                ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Narrow the luminance limit of a lifting radial mask that lifts its brighter surroundings more than its box.

    ``render(document)`` renders a small preview; ``image`` is image 1, whose luma the limit is written in.
    Returns the plan (a copy when a limit changed), notes for the reviewer and one record per mask probed."""
    plane = None
    candidates = []
    for index, mask in enumerate(plan.get("masks") or []):
        shape, limit, target = mask.get("shape") or {}, mask.get("limit") or {}, mask.get("target") or {}
        if (shape.get("kind") != "radial" or shape.get("invert") or limit.get("kind") != "luminance"
                or limit.get("invert") or float(limit.get("hi", 1.0)) >= 1.0 or not target.get("box")):
            continue
        if plane is None:
            plane = tone.luma_of(tone.load_small(image))
        y0, y1, x0, x1 = tone._span(*plane.shape, target["box"])
        subject = float(np.median(plane[y0:y1, x0:x1]))
        wanted = _number(target.get("median"))
        if ((wanted is not None and wanted > subject + 0.02)
                or float(mask["adjust"].get("exposure", 0.0)) >= SPILL_MIN_EV):
            candidates.append((index, subject, (y0, y1, x0, x1)))
    if not candidates:
        return plan, [], []
    height, width = plane.shape
    padded = np.pad(plane, 1, mode="edge")
    smooth = np.stack([padded[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)]).std(axis=0) < SPILL_SMOOTH

    def luma(work: dict[str, Any]) -> np.ndarray:
        document = compile_document(work, source_kind, colors=colors, aspect=width / height)[0]
        found = tone.luma_of(tone.load_small(render(document)))
        if found.shape != plane.shape:
            found = np.asarray(Image.fromarray(found.astype(np.float32), mode="F").resize(
                (width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        return found

    def measure(work: dict[str, Any], index: int, subject: float, span: tuple[int, int, int, int],
                base: np.ndarray) -> dict[str, Any]:
        stepped = copy.deepcopy(work)
        adjust = stepped["masks"][index]["adjust"]
        adjust["exposure"] = float(adjust.get("exposure", 0.0)) + 1.0
        rise = luma(stepped) - base
        inside = np.zeros(plane.shape, dtype=bool)
        y0, y1, x0, x1 = span
        inside[y0:y1, x0:x1] = True
        box_rise = float(np.median(rise[inside]))
        brighter = ~inside & (plane > subject + SPILL_ABOVE)
        lifted = brighter & (rise > max(box_rise, SPILL_FLOOR))
        spilled = lifted & smooth
        return {"box_rise": round(box_rise, 3), "area": round(float(spilled.mean()), 4),
                "lifted_area": round(float(lifted.mean()), 4),
                "faint": round(float((brighter & smooth & (rise > SPILL_FAINT)).mean()), 4),
                "spill_rise": round(float(np.median(rise[spilled])), 3) if spilled.any() else 0.0,
                "edge": float(np.percentile(plane[spilled], 5)) if spilled.any() else None}

    out = plan
    base = luma(out)
    notes: list[str] = []
    records: list[dict[str, Any]] = []
    for index, subject, span in candidates:
        mask = out["masks"][index]
        found = measure(out, index, subject, span, base)
        record = {"name": mask["name"], "subject": round(subject, 3), "hi": mask["limit"]["hi"], **found}
        records.append(record)
        if found["box_rise"] < 0.01 or found["area"] < SPILL_AREA or found["edge"] is None:
            continue
        feather = min(float(mask["limit"].get("feather", 0.06)), SPILL_FEATHER)
        floor = round(subject + feather, 3)
        new_hi = round(max(floor, found["edge"] - feather), 3)
        chosen = None
        trials = []
        while len(trials) < SPILL_STEPS and new_hi <= float(mask["limit"]["hi"]) - 0.02:
            trial = copy.deepcopy(out)
            trial["masks"][index]["limit"] = {**mask["limit"], "hi": new_hi, "feather": feather}
            trial_base = luma(trial)
            after = measure(trial, index, subject, span, trial_base)
            trials.append({"hi": new_hi, "area": after["area"], "faint": after["faint"], "box_rise": after["box_rise"]})
            if after["area"] <= found["area"] / 2:
                chosen = (new_hi, after, trial, trial_base)
            if after["faint"] < SPILL_AREA or new_hi <= floor:
                break
            new_hi = round(max(floor, new_hi - feather), 3)
        if trials:
            record["trials"] = trials
        if chosen is None:
            continue
        new_hi, after, out, base = chosen
        record.update(new_hi=new_hi, after_area=after["area"], after_faint=after["faint"],
                      after_box_rise=after["box_rise"], kept=True)
        notes.append(f"蒙版「{mask['name']}」提亮时，框外比主体亮的像素（image 1 亮度 {found['edge']:.2f} 起，约"
                     f" {found['area'] * 100:.1f}% 画面）每 EV 抬 {found['spill_rise']:+.2f}，比框内的"
                     f" {found['box_rise']:+.2f} 还多，会成一块光斑：亮度上限 {mask['limit']['hi']:.2f} 收到 {new_hi:.2f}"
                     f"（羽化 {feather:.2f}）")
    for record in records:
        record["edge"] = None if record["edge"] is None else round(record["edge"], 3)
    return out, notes, records


def solve_plan(plan: dict[str, Any], source_kind: str, render: Callable[[dict[str, Any]], Path], *,
               colors: Callable[[list[dict[str, float]]], list[list[float]]] | None = None,
               baseline: dict[str, float] | None = None,
               neutral: Callable[[np.ndarray], float] | None = None,
               aspect: float | None = None
               ) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any] | None]:
    """Meet a plan's tone, region and colour targets by measurement.

    ``render(document)`` renders a small preview; ``aspect`` (width / height)
    goes to compile_document. ``baseline`` holds the
    percentiles of the unedited photo; given, tone targets are kept within
    ``tone.SHIFT_LIMITS`` of them. ``neutral`` (see ``neutral_guard``) holds a
    natural-colour version's colour moves off the near-neutrals' cast gate.
    Returns the plan as executed (global
    exposure and saturation and the values of targeted masks replaced by the
    solved ones), its document, compile notes and the solver report (None when
    the plan has no targets, in which case nothing is rendered).
    """
    targets = tone_targets(plan)
    bound_notes: list[str] = []
    limits = tone.shift_limits(source_kind)
    if targets and baseline:
        targets, bound_notes = tone.bound_shift(targets, baseline, limits)
    chroma = (plan.get("tone") or {}).get("chroma")
    regions = []
    for index, mask in enumerate(plan.get("masks") or []):
        target = mask.get("target")
        if not target:
            continue
        start = float(mask["adjust"].get("exposure", 0.0))
        lo, hi = SOLVER_LOCAL_RANGE if is_precise(mask) else SOLVER_PLAIN_RANGE
        # A plan may start a mask outside the solver's range; the solver then only brings it back in.
        region: dict[str, Any] = {"key": index, "box": target["box"], "range": (min(lo, start), max(hi, start)),
                                  "exposure": start, "name": str(mask.get("name") or index)}
        if "median" in target:
            region["median"] = target["median"]
        if "chroma" in target:
            region["colour"] = (target.get("hue", 0.0), target["chroma"])
            region["start"] = {control: float(mask["adjust"].get(control, 0.0)) for control in tone.COLOUR_CONTROLS}
            if COLOUR_MASK_LIMITS and mask["shape"]["kind"] == "color":
                region["colour_limits"] = dict(COLOUR_MASK_LIMITS)
            if source_kind == "raw":
                region["colour_limits"] = {**(region.get("colour_limits") or {}), **RAW_MASK_COLOUR_LIMITS,
                                           "saturation": (RAW_MASK_SATURATION_FLOOR, tone.COLOUR_LIMITS["saturation"][1])}
        regions.append(region)
    if not targets and not regions and chroma is None:
        document, notes = compile_document(plan, source_kind, colors=colors, aspect=aspect)
        return plan, document, notes, None
    work = copy.deepcopy(plan)
    from_look = _look_saturation(plan)

    def apply(exposure: float, local: dict[Any, dict[str, float]], saturation: float) -> None:
        light = work["develop"].setdefault("light", {})
        if abs(exposure) > 1e-6 or "exposure" in light:
            light["exposure"] = round(float(exposure), 3)
        for index, owned in local.items():
            adjust, written = work["masks"][index]["adjust"], plan["masks"][index]["adjust"]
            for control, value in owned.items():
                if control == "exposure" and control not in written and abs(value) < 1e-9:
                    adjust.pop(control, None)   # The solver carries an exposure for every targeted mask.
                else:
                    adjust[control] = round(float(value), 3)
        if chroma is not None:
            work["develop"].setdefault("color", {})["saturation"] = round(float(saturation) - from_look, 2)

    def shoot(curve: list[list[float]], exposure: float, local: dict[Any, dict[str, float]], saturation: float) -> Path:
        apply(exposure, local, saturation)
        document, _ = compile_document(work, source_kind, colors=colors, curve=curve, aspect=aspect)
        return render(document)

    develop = plan.get("develop") or {}
    start_saturation = float((_merged_develop(plan).get("color") or {}).get("saturation", 0.0))
    saturation_range = RAW_SOLVER_SATURATION_RANGE if source_kind == "raw" else SOLVER_SATURATION_RANGE
    with_look = (plan.get("look") or {}).get("id") in LOOKS
    look_up = RAW_LOOK_SATURATION_UP if source_kind == "raw" else LOOK_SATURATION_ROOM
    if with_look:
        saturation_range = (min(start_saturation, max(saturation_range[0], start_saturation - LOOK_SATURATION_ROOM)),
                            max(start_saturation, min(saturation_range[1], start_saturation + look_up)))
    report = tone.solve_scene(
        shoot, targets, regions, exposure=float((develop.get("light") or {}).get("exposure", 0.0)),
        exposure_range=SOLVER_EXPOSURE_RANGE, chroma=chroma, saturation=start_saturation,
        saturation_range=saturation_range, neutral=neutral, unedited=baseline, shift_limits=limits)
    if with_look and report.get("chroma"):
        report["chroma"] = {**report["chroma"], "look_room": look_up}
    apply(report["exposure"], report["local"], report["saturation"])
    if not work["develop"].get("light"):
        work["develop"].pop("light", None)
    document, notes = compile_document(work, source_kind, colors=colors, curve=report["curve"], aspect=aspect)
    report = {**report, "targets": targets,
              "regions": [{**region, "name": work["masks"][region["key"]]["name"],
                           "plain": not is_precise(work["masks"][region["key"]])} for region in report["regions"]]}
    report.pop("local", None)
    missing = [f"蒙版「{region['name']}」没有覆盖它的目标框 {region['box']}，{what}目标未执行"
               for region in report["regions"]
               for what, missed in (("亮度", region.get("missed")), ("颜色", (region.get("colour") or {}).get("missed")))
               if missed]
    report["notes"] = bound_notes + list(report["notes"])
    return work, document, notes + list(report["notes"]) + missing, report


def describe_plan(plan: dict[str, Any]) -> list[str]:
    """Short Chinese lines for the result page."""
    lines = []
    light = (plan.get("develop") or {}).get("light") or {}
    names = {"exposure": "曝光", "contrast": "对比", "highlights": "高光", "shadows": "阴影",
             "whites": "白色", "blacks": "黑色"}
    moves = [f"{names[key]} {value:+.2f} EV" if key == "exposure" else f"{names[key]} {value:+.0f}"
             for key, value in light.items() if key in names and abs(value) > 1e-6]
    if moves:
        lines.append("全局：" + "、".join(moves))
    targets = plan.get("tone") or {}
    labels = {"black": "黑", "shadows": "暗部", "mid": "中间调", "highlights": "亮部", "white": "白"}
    placed = [f"{labels[name]} {targets[name] * 100:.0f}" for name in tone.ORDER if name in targets]
    if placed:
        lines.append("影调落点（0–100）：" + " · ".join(placed))
    look = plan.get("look") or {}
    if look.get("id") in LOOKS:
        lines.append(f"风格：{LOOKS[look['id']]['title']}（强度 {look.get('strength', 0.8) * 100:.0f}%）")
    for mask in plan.get("masks") or []:
        kinds = [_KIND_LABELS.get(mask["shape"]["kind"], "蒙版")]
        if mask.get("limit"):
            kinds.append("限定" + _KIND_LABELS.get(mask["limit"]["kind"], ""))
        if mask.get("exclude"):
            kinds.append("排除" + _KIND_LABELS.get(mask["exclude"]["kind"], ""))
        lines.append(f"{mask['name']}（{'，'.join(kinds)}）：{mask['purpose'] or '局部调整'}")
    return lines


# --------------------------------------------------------------------------
# Images


def load_preview(path: Path, edge: int = 1280) -> Image.Image:
    from .ingest import image_to_srgb

    with Image.open(path) as opened:
        image = image_to_srgb(ImageOps.exif_transpose(opened)).convert("RGB")
    image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    return image


def _b64(image: Image.Image, quality: int = 88) -> str:
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older Pillow.
        return ImageFont.load_default()


def grid_overlay(image: Image.Image) -> Image.Image:
    """Copy with 0.1-spaced normalized coordinate lines and edge labels."""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = base.size
    font = _font(max(12, min(width, height) // 40))
    for step in range(1, 10):
        x, y = round(width * step / 10), round(height * step / 10)
        for offset, color in ((1, (0, 0, 0, 90)), (0, (255, 255, 255, 120))):
            draw.line([(x + offset, 0), (x + offset, height)], fill=color, width=1)
            draw.line([(0, y + offset), (width, y + offset)], fill=color, width=1)
        label = f"{step / 10:.1f}"
        for text_xy, anchor in (((x, 4), "mt"), ((4, y), "lm")):
            draw.text((text_xy[0] + 1, text_xy[1] + 1), label, font=font, fill=(0, 0, 0, 200), anchor=anchor)
            draw.text(text_xy, label, font=font, fill=(255, 255, 0, 230), anchor=anchor)
    return Image.alpha_composite(base, layer).convert("RGB")


_MASK_COLOURS = ((255, 64, 64), (64, 200, 255), (80, 255, 120), (255, 160, 0), (220, 90, 255), (255, 255, 255),
                 (255, 120, 190), (160, 255, 255))


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict) and _number(value.get("x")) is not None and _number(value.get("y")) is not None:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) is not None for v in value):
        return float(value[0]), float(value[1])
    return None


def _dashed(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill, width: int,
            dash: float = 10.0) -> None:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(1, int(length // dash))
    for index in range(0, steps, 2):
        a, b = index / steps, min(1.0, (index + 1) / steps)
        draw.line([(start[0] + (end[0] - start[0]) * a, start[1] + (end[1] - start[1]) * a),
                   (start[0] + (end[0] - start[0]) * b, start[1] + (end[1] - start[1]) * b)], fill=fill, width=width)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: dict[str, Any], size: tuple[int, int], colour, width: int) -> None:
    """Outline of a radial (its ellipse) or a linear (solid where full, dashed where it ends) shape."""
    w, h = size
    kind = shape.get("kind")
    if kind == "radial":
        centre = _xy(shape.get("center"))
        rx, ry = _number(shape.get("rx")), _number(shape.get("ry"))
        if centre is None or not rx or not ry:
            return
        angle = math.radians(_number(shape.get("angle")) or 0.0)
        points = []
        for step in range(73):
            t = 2 * math.pi * step / 72
            dx, dy = rx * w * math.cos(t), ry * h * math.sin(t)
            points.append((centre[0] * w + dx * math.cos(angle) - dy * math.sin(angle),
                           centre[1] * h + dx * math.sin(angle) + dy * math.cos(angle)))
        if shape.get("invert"):
            for a, b in zip(points[::2], points[1::2]):
                draw.line([a, b], fill=colour, width=width)
        else:
            draw.line(points, fill=colour, width=width)
    elif kind == "linear":
        start, end = _xy(shape.get("start")), _xy(shape.get("end"))
        if start is None or end is None:
            return
        sx, sy, ex, ey = start[0] * w, start[1] * h, end[0] * w, end[1] * h
        length = math.hypot(ex - sx, ey - sy)
        if length < 1:
            return
        px, py = -(ey - sy) / length * (w + h), (ex - sx) / length * (w + h)
        draw.line([(sx - px, sy - py), (sx + px, sy + py)], fill=colour, width=width)
        _dashed(draw, (ex - px, ey - py), (ex + px, ey + py), colour, width)
        draw.line([(sx, sy), (ex, ey)], fill=colour, width=max(1, width - 1))


def mask_overlay(image: Image.Image, plan: dict[str, Any]) -> Image.Image:
    """Copy with every mask's geometry and target box drawn in its own colour and numbered from 1.

    A radial is drawn as its ellipse (dotted when inverted), a linear as a solid line where its effect
    is full and a dashed line where it has faded out; a geometric limit or exclude is drawn thinner.
    Target boxes are rectangles with the mask's number. Luminance and colour ranges cannot be drawn.
    """
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = base.size
    font = _font(max(14, min(size) // 30))
    for index, mask in enumerate(plan.get("masks") or []):
        colour = (*_MASK_COLOURS[index % len(_MASK_COLOURS)], 235)
        label = str(index + 1)
        for key, width in (("shape", 3), ("limit", 1), ("exclude", 1)):
            part = mask.get(key)
            if isinstance(part, dict):
                _draw_shape(draw, part, size, colour, width)
        shape = mask.get("shape") or {}
        anchor = _xy(shape.get("center")) or _xy(shape.get("start"))
        if anchor is not None:
            xy = (min(size[0] - 12, max(12, anchor[0] * size[0])), min(size[1] - 12, max(12, anchor[1] * size[1])))
            draw.text((xy[0] + 1, xy[1] + 1), label, font=font, fill=(0, 0, 0, 230), anchor="mm")
            draw.text(xy, label, font=font, fill=colour, anchor="mm")
        box = _box((mask.get("target") or {}).get("box"))
        if box:
            rect = [box[0] * size[0], box[1] * size[1], box[2] * size[0], box[3] * size[1]]
            draw.rectangle(rect, outline=(0, 0, 0, 200), width=4)
            draw.rectangle(rect, outline=colour, width=2)
            draw.text((rect[0] + 3, rect[1] + 2), label, font=font, fill=colour, anchor="lt")
    return Image.alpha_composite(base, layer).convert("RGB")


def _board(before: Image.Image, after: Image.Image) -> str:
    from .pair_review import build_pair_board

    return build_pair_board(_b64(before, 90), _b64(after, 90))


def picture_stats(image: Image.Image | Path) -> dict[str, Any]:
    """Tone percentiles and the 8x8 cell tables of one picture."""
    rgb = tone.load_small(image)
    return {"anchors": tone.measure(rgb), "colourfulness": round(tone.chroma_mean(rgb), 4), **tone.grid(rgb)}


# Region median targets are checked against image 1 before the first render. p20-r1 DSC_1963: all three first
# plans asked the rider's face (image 1 .22 in its box) for .55, daylight skin, where a mask gives at most about
# .33; the masks ended at their stop in every round and every review took 2-5 points for it (subject, and tone
# for the mask against the curve). The 8x8 cell table cannot show a face a twentieth of the frame wide.
# A target within REACH_SLACK of what its mask reaches is left alone (feather and the curve's fit move it that much).
REACH_SLACK = 0.03
# A non-inverted radial smaller than this share of the frame holds none of the frame's percentiles, so the curve
# takes its region with the frame and only the reach from the curve's estimate counts. p22-r1 DSC_1963 B0: a face
# radial (rx .05, ry .055) at .37 in image 1, which the tone targets (shadows .43 -> .29) take to .21; +1.25 EV
# reached .56 from image 1, so it was not flagged; the solver measured .23 before the mask and ended at .20 at
# its stop ("面部目标中位 0.540 在原片 0.229 的区域上不可达").
REACH_SMALL_AREA = 0.05
# A color mask's points are checked against its target box: the share of the box within REACH_POINTS_DE (Lab) of
# any point's colour in image 1. Below REACH_POINTS_SHARE the mask selects other pixels than the box shows, and
# the solver measures and moves only the selected ones. p22-r1 DSC_1963 A0/B0/C0 yak_whites: points at luma .50,
# .47 and .11 (a horn) for white fur at .83, 18% of the box near them; the solver read .08 on the selected pixels
# and each review took 3 on subject, reading the box as lying on the dark horns. B2 added a point on the fur (.82):
# 99%, measured .81.
REACH_POINTS_DE = 15.0
REACH_POINTS_SHARE = 0.4
# A target box whose image-1 luma spans more than this from its 10th to its 90th percentile holds its region and
# the surroundings: its median is neither, and the solver measures only the part its mask moves most (a lift skips
# what is already bright). Face boxes on DSC_1963 p20-p22 spanned .74-.80 (dark face, bright hat and sky); in
# p20 and p22 the box median was .34/.37 and the solver read .24/.23. The yak box with a horn in it: .77. Every
# other target box p20-p22: .51 at most.
REACH_SPREAD = 0.6
# A color mask whose region holds a frame percentile moves that percentile with it. Estimated on image 1: the curve
# fitted to the frame tone targets, then each color mask with a target moves the pixels near its points' colours
# (weight 1 within SHARED_NEAR[0] Lab, 0 from SHARED_NEAR[1]) by the EV that takes its box there. When one mask
# holds at least tone.OWN_FROM of the pixels at a shadows/mid/highlights percentile and the percentile then lands
# more than SHARED_MISS from its target, the two targets contradict each other. DSC_1515 (sky = the frame's mid,
# 83-100% of its band): p20 A0/B0/C0, p22 B0/C0, p23 A0/B0/C0 estimated -.06 to -.17 and measured -.055 to -.165
# (p23 A0 -.090/-.078, B0 -.169/-.165, C0 -.111/-.107); the p23 A0/B0/C0 reviews took 3/3/5 on tone naming it
# (B0: "天空蒙版把整幅中位数一并拉了下去"). Against the probes' coverage this proxy matches the engine's colour
# masks on blue sky (IoU .80-.97); on near-grey frost (DSC_2620) it selects more than the engine, but its estimates
# there stay under .015. Masks
# with misplaced points are left out (their selection is not the box's).
# Since p27 (change 39) every mask with a median target is weighed so: a radial or linear by its geometry as planned
# (rx of the width, ry of the height, as the engine renders it since change 38), a luminance shape by its range,
# times its limit and less its exclude (each weighed the same way). p25-r1 DSC_1515 C0 暗谷回温, a radial with a
# luminance limit, held the frame shadows and took it +.045 off (tools/shares_check.py), which the colour-only
# check never saw. And the miss counts from .03, as the reviewer reads a region .03 off its target as missed:
# p25-r1 DSC_1515 A0/B0 sky masks estimated -.035/-.040, measured -.035/-.049, and the reviews took 3-4 on tone for
# them (B0 "中位目标0.560与天空框目标0.470描述同一批像素"). tools/conflict_retro.py over the 54 first plans of
# p20-p25: a miss over .03 flagged 21 holding 52 conflict points against 33 points on the 33 others; over .05, 10
# holding 34.
SHARED_NEAR = (6.0, 15.0)
SHARED_MISS = 0.03
# A region target the frame tone targets move one way and its mask has to move back (kind "against"): the curve
# fitted to the frame targets takes the region (the box's pixels the mask holds at least half) by tone.AGAINST_EV or
# more one way, and its target then needs the mask tone.AGAINST_EV or more the other way. The reviews read that as
# one undoing the other: p26-r1 DSC_1963 A2 face "遮罩要用 +1.19EV 去抵消全局曲线带来的 -0.5EV，属于硬掰" (subject -4);
# p25-r1 DSC_1515 A0 右谷落影提亮, curve -0.81 EV on its box and the mask +0.91 EV, where the solver let the shadows
# give way .20 -> .27 (tone -5). tools/conflict_retro.py over the 54 first plans of p20-p25: 21 flagged, holding 43
# conflict points against 42 on the 33 others. The line names the frame percentile nearest the region and the
# value of it that ends the conflict (refitting the curve), found in AGAINST_STEPS steps towards image 1's value.
AGAINST_STEPS = 10


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _geometry_weight(shape: dict[str, Any], height: int, width: int) -> np.ndarray | None:
    """A radial's or linear's weight on a height x width grid as planned (see _engine_shape for the engine's)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
    if shape.get("kind") == "radial":
        dx = xx - float(shape["center"]["x"]) * width
        dy = yy - float(shape["center"]["y"]) * height
        turn = math.radians(float(shape.get("angle") or 0.0))
        along, across = dx * math.cos(turn) + dy * math.sin(turn), dy * math.cos(turn) - dx * math.sin(turn)
        radius = np.hypot(along / max(float(shape["rx"]) * width, 1e-6), across / max(float(shape["ry"]) * height, 1e-6))
        full = 1.0 - min(100.0, max(0.0, float(shape.get("feather", 70.0)))) / 100.0
        weight = 1.0 - _smoothstep((radius - full) / max(1.0 - full, 1e-6))
        return 1.0 - weight if shape.get("invert") else weight
    if shape.get("kind") == "linear":
        sx, sy = float(shape["start"]["x"]) * width, float(shape["start"]["y"]) * height
        ex, ey = float(shape["end"]["x"]) * width, float(shape["end"]["y"]) * height
        length = (ex - sx) ** 2 + (ey - sy) ** 2
        if length < 1e-9:
            return None
        return 1.0 - _smoothstep(((xx - sx) * (ex - sx) + (yy - sy) * (ey - sy)) / length)
    return None


def _part_weight(shape: dict[str, Any], plane: np.ndarray, lab: Callable[[], np.ndarray]) -> np.ndarray | None:
    """Estimated weight of one mask part on image 1 (``lab`` gives image 1 in Lab), or None for none known."""
    kind = shape.get("kind")
    if kind in ("radial", "linear"):
        return _geometry_weight(shape, *plane.shape)
    if kind == "luminance":
        lo, hi = float(shape.get("lo", 0.0)), float(shape.get("hi", 1.0))
        feather = max(float(shape.get("feather", 0.06)), 1e-3)
        weight = np.ones_like(plane, dtype=np.float64)
        if hi < 1.0:
            weight = weight * np.clip(1.0 - (plane - hi) / feather, 0.0, 1.0)
        if lo > 0.0:
            weight = weight * np.clip(1.0 - (lo - plane) / feather, 0.0, 1.0)
    elif kind == "color" and shape.get("points"):
        near, _ = _point_distance(lab(), plane, shape["points"])
        weight = np.clip((SHARED_NEAR[1] - near) / (SHARED_NEAR[1] - SHARED_NEAR[0]), 0.0, 1.0)
    else:
        return None
    return 1.0 - weight if shape.get("invert") else weight


def _point_distance(lab: np.ndarray, plane: np.ndarray, points: list[dict[str, Any]]) -> tuple[np.ndarray, list]:
    """Lab distance of every pixel to the nearest point's colour (median of a 3% patch), and each point's luma."""
    height, width = plane.shape
    near, sampled = np.full((height, width), np.inf), []
    for point in points:
        px, py = float(point["x"]), float(point["y"])
        y0, y1, x0, x1 = tone._span(height, width, [px - .015, py - .015, px + .015, py + .015])
        colour = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        near = np.minimum(near, np.linalg.norm(lab - colour, axis=2))
        sampled.append({"x": round(px, 3), "y": round(py, 3), "luma": round(float(np.median(plane[y0:y1, x0:x1])), 3)})
    return near, sampled


def _display(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _shared_percentiles(plane: np.ndarray, anchors: dict[str, float], targets: dict[str, float],
                        curve: list[list[float]], moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frame percentiles a mask's region holds and its target pulls off their own (see SHARED_NEAR)."""
    xs, ys = np.asarray(curve, dtype=np.float64).T
    curved = np.interp(plane, xs, ys)
    for move in moves:
        y0, y1, x0, x1 = move["span"]
        picked = curved[y0:y1, x0:x1][move["weight"][y0:y1, x0:x1] >= 0.5]
        if picked.size < 12:
            move["ev"] = 0.0
            continue
        move["by_curve"] = float(np.median(picked))
        ev = tone.ev_between(move["by_curve"], move["target"])
        move["ev"] = min(max(ev, move["range"][0]), move["range"][1])
    moves = [move for move in moves if move["ev"]]
    if not moves:
        return []
    after = _display(tone._linear(curved) * 2.0 ** sum(move["ev"] * move["weight"] for move in moves))
    out = []
    for anchor in tone.OWNED_ANCHORS:
        if anchor not in targets or anchor not in anchors:
            continue
        band = np.abs(plane - anchors[anchor]) <= tone.OWN_BAND
        if int(band.sum()) < 40:
            continue
        shares = [float(np.mean(move["weight"][band] >= 0.5)) for move in moves]
        owner = moves[int(np.argmax(shares))]
        share = max(shares)
        simulated = float(np.percentile(after, tone.ANCHORS[anchor]))
        miss = simulated - targets[anchor]
        if share < tone.OWN_FROM or abs(miss) <= SHARED_MISS or miss * owner["ev"] <= 0:
            continue
        out.append({"kind": "shared", "name": owner["name"], "box": owner["box"], "target": round(owner["target"], 3),
                    "anchor": anchor, "image": round(float(anchors[anchor]), 3), "written": round(targets[anchor], 3),
                    "simulated": round(simulated, 3), "share": round(share, 2), "ev": round(owner["ev"], 2),
                    "by_curve": round(owner["by_curve"], 3), "shape": owner.get("shape", "color")})
    return out


def _against(plane: np.ndarray, anchors: dict[str, float], targets: dict[str, float], curve: list[list[float]],
             weight: np.ndarray, span: tuple[int, int, int, int], wanted: float) -> dict[str, Any] | None:
    """The curve's and the mask's moves on a region when they oppose each other (see AGAINST_STEPS), else None."""
    y0, y1, x0, x1 = span
    picked = plane[y0:y1, x0:x1][weight[y0:y1, x0:x1] >= 0.5]
    if picked.size < 12:
        return None
    region = float(np.median(picked))

    def moves(fitted: list[list[float]]) -> tuple[float, float, float]:
        by_curve = tone.curve_value(fitted, region)
        return by_curve, tone.ev_between(region, by_curve), tone.ev_between(by_curve, wanted)

    def opposed(curve_ev: float, need: float) -> bool:
        return curve_ev * need < 0 and min(abs(curve_ev), abs(need)) >= tone.AGAINST_EV

    by_curve, curve_ev, need = moves(curve)
    if not opposed(curve_ev, need):
        return None
    # Each written percentile alone, nearest the region first (one written at image 1's value cannot end the
    # conflict), then the two around the region together.
    written = [name for name in tone.ORDER if name in targets and name in anchors]
    nearest = sorted(written, key=lambda name: abs(anchors[name] - region))
    options = [(name,) for name in nearest if abs(targets[name] - anchors[name]) >= 0.005]
    below = [name for name in written if anchors[name] <= region]
    above = [name for name in written if anchors[name] > region]
    if below and above:
        options.append((below[-1], above[0]))
    fix = None
    for names in options:
        for step in range(1, AGAINST_STEPS + 1):
            values = {name: targets[name] + (anchors[name] - targets[name]) * step / AGAINST_STEPS for name in names}
            fitted = tone.fit_curve(anchors, {**targets, **values})[0]
            if fitted and not opposed(*moves(fitted)[1:]):
                fix = {name: round(value, 3) for name, value in values.items()}
                break
        if fix is not None:
            break
    anchor = nearest[0]
    edge = tone.to_display(tone.to_linear(by_curve) * 2.0 ** math.copysign(tone.AGAINST_EV, need))
    return {"image": round(region, 3), "by_curve": round(by_curve, 3), "curve_ev": round(curve_ev, 2),
            "need": round(need, 2), "edge": round(edge, 3), "anchor": anchor,
            "anchor_image": round(float(anchors[anchor]), 3), "anchor_written": round(float(targets[anchor]), 3),
            "anchor_fix": fix}


def plan_reach(plan: dict[str, Any], image: Image.Image | Path) -> list[dict[str, Any]]:
    """Region median targets in ``plan`` that cannot be met as written, measured on image 1.

    Out of reach (kind "reach"): the mask's exposure range is taken from image 1's median in the box, and again
    after a curve fitted from image 1's percentiles to the frame tone targets (the solver's own fit, without the
    plan's global sliders, look or the percentiles that follow a large mask). A target is flagged only when
    neither reaches it, since that curve misjudges mid tones under a large mask (p21-r1 DSC_1515 B0: a lit wall at
    .585 put at .37, the solver's curve kept it at .60); under a small radial only the curve's counts. Each entry
    gives the mask's name, box, the target, image 1's median in the box, what the curve makes of it and the range
    the mask reaches from each.
    Misplaced points (kind "points"): a color mask whose points' colours cover too little of its box; the entry
    gives each point's luma in image 1 and the share of the box near them.
    Mixed box (kind "spread"): a box whose 10th and 90th percentiles lie more than REACH_SPREAD apart.
    Shared percentile (kind "shared"): a mask's region holds a frame percentile that its target then takes
    off the percentile's own target; the entry gives the percentile, its image-1 value and target, what it is
    estimated to measure, the share of its pixels the mask holds, the mask's EV and the region by the curve.
    Against the curve (kind "against"): the frame tone targets move the region one way and its target needs the
    mask the other way (see AGAINST_STEPS); the entry gives both moves, the frame percentile nearest the region and
    the value of it that ends the conflict (None when none between its target and image 1's value does)."""
    from .lightroom_qa import _rgb_to_lab

    rgb = tone.load_small(image)
    plane = tone.luma_of(rgb)
    cached: list[np.ndarray] = []

    def lab() -> np.ndarray:
        if not cached:
            cached.append(_rgb_to_lab(rgb))
        return cached[0]

    anchors = tone.measure(rgb)
    targets = {name: float(value) for name, value in (plan.get("tone") or {}).items()
               if name in tone.ORDER and _number(value) is not None}
    curve = tone.fit_curve(anchors, targets)[0] if len(targets) >= 3 else []
    out: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    for mask in plan.get("masks") or []:
        target = mask.get("target") or {}
        if _number(target.get("median")) is None or not target.get("box"):
            continue
        name, box = str(mask.get("name") or ""), [round(float(v), 3) for v in target["box"]]
        y0, y1, x0, x1 = tone._span(plane.shape[0], plane.shape[1], target["box"])
        low, measured, high = (float(v) for v in np.percentile(plane[y0:y1, x0:x1], [10.0, 50.0, 90.0]))
        wanted = float(target["median"])
        shape = mask.get("shape") or {}
        if high - low > REACH_SPREAD:
            out.append({"kind": "spread", "name": name, "box": box, "target": round(wanted, 3),
                        "image": round(measured, 3), "p10": round(low, 3), "p90": round(high, 3)})
        lo, hi = SOLVER_LOCAL_RANGE if is_precise(mask) else SOLVER_PLAIN_RANGE
        start = _number((mask.get("adjust") or {}).get("exposure")) or 0.0
        lo, hi = min(lo, start), max(hi, start)
        weight: np.ndarray | None = None
        if shape.get("kind") == "color" and shape.get("points"):
            near, points = _point_distance(lab(), plane, shape["points"])
            share = float(np.mean(near[y0:y1, x0:x1] < REACH_POINTS_DE))
            if share < REACH_POINTS_SHARE:
                out.append({"kind": "points", "name": name, "box": box, "target": round(wanted, 3),
                            "image": round(measured, 3), "points": points, "share": round(share, 2)})
            else:
                weight = np.clip((SHARED_NEAR[1] - near) / (SHARED_NEAR[1] - SHARED_NEAR[0]), 0.0, 1.0)
                weight = 1.0 - weight if shape.get("invert") else weight
        else:
            weight = _part_weight(shape, plane, lab)
        for key, keep in (("limit", True), ("exclude", False)):
            part = _part_weight(mask[key], plane, lab) if weight is not None and mask.get(key) else None
            if part is not None:
                weight = weight * (part if keep else 1.0 - part)
        if weight is not None and curve:
            moves.append({"name": name, "box": box, "target": wanted, "span": (y0, y1, x0, x1),
                          "weight": weight, "range": (lo, hi), "shape": shape.get("kind")})
            found = _against(plane, anchors, targets, curve, weight, (y0, y1, x0, x1), wanted)
            if found:
                out.append({"kind": "against", "name": name, "box": box, "target": round(wanted, 3), **found})
        by_curve = tone.curve_value(curve, measured)
        reach = {key: (tone.to_display(tone.to_linear(value) * 2.0 ** lo),
                       tone.to_display(tone.to_linear(value) * 2.0 ** hi))
                 for key, value in (("image", measured), ("curve", by_curve))}
        small = (shape.get("kind") == "radial" and not shape.get("invert") and bool(curve)
                 and math.pi * (_number(shape.get("rx")) or 0.0) * (_number(shape.get("ry")) or 0.0) < REACH_SMALL_AREA)
        judged = [reach["curve"]] if small else list(reach.values())
        if min(low for low, _ in judged) - REACH_SLACK <= wanted <= max(high for _, high in judged) + REACH_SLACK:
            continue
        out.append({"kind": "reach", "name": name, "box": box, "target": round(wanted, 3), "image": round(measured, 3),
                    "by_curve": round(by_curve, 3), "reach": [round(v, 3) for v in reach["image"]],
                    "reach_curve": [round(v, 3) for v in reach["curve"]], "ev": [lo, hi],
                    "precise": is_precise(mask), "small": small})
    if curve and moves:
        out.extend(_shared_percentiles(plane, anchors, targets, curve, moves))
    return out


def measure_looks(binary: Path, original: Path, folder: Path, source_kind: str) -> dict[str, dict[str, float]]:
    """Every look at LOOK_PROBE_STRENGTH on this photo, against the unedited render: "chroma", its frame
    colourfulness as a ratio (left out on a frame with no colour to speak of), and "distance", the mean Lab
    difference (``lightroom_qa.render_distance``, the measure that calls sibling versions nearly identical).

    Rendered on a copy of the photo in ``folder``; a look whose render fails is left out, and an empty dict
    means no numbers (planning goes on without them).
    """
    from .lightcraft import LightCraftError, render_document
    from .lightroom_qa import render_distance

    folder.mkdir(parents=True, exist_ok=True)
    source = folder / f"source{original.suffix.lower()}"
    shutil.copy2(original, source)

    def one(name: str, plan: dict[str, Any]) -> tuple[str, Path | None]:
        document, _ = compile_document(plan, source_kind)
        try:
            return name, render_document(binary, source, document, folder / name, edge=PROBE_EDGE)
        except (LightCraftError, OSError):
            return name, None

    jobs = [("unedited", {})] + [(look_id, {"look": {"id": look_id, "strength": LOOK_PROBE_STRENGTH}})
                                 for look_id in LOOKS]
    with ThreadPoolExecutor(max_workers=LOOK_PROBE_WORKERS) as pool:
        renders = dict(pool.map(lambda item: one(*item), jobs))
    plain = renders.pop("unedited")
    try:
        base = _frame_chroma(picture_stats(plain)) if plain else None
    except (OSError, ValueError):
        base = None
    if base is None:
        return {}
    measured: dict[str, dict[str, float]] = {}
    for look_id, path in renders.items():
        if path is None:
            continue
        try:
            entry = {"distance": round(render_distance(plain, path), 2)}
            if base >= 0.5:   # x100; a grey frame gives no ratio worth reading
                entry["chroma"] = round(_frame_chroma(picture_stats(path)) / base, 3)
        except (OSError, ValueError):
            continue
        measured[look_id] = entry
    return measured


# --------------------------------------------------------------------------
# Prompts

_TONE = """TONE TARGETS are the most important decision. "tone" says where luminance percentiles of the FINISHED frame
sit, on the 0..1 display scale of the measurements: black (1st percentile), shadows (10th), mid (median),
highlights (90th), white (99th). A solver measures the render and sets global exposure and a master curve until
they are met, so do not try to reach them with exposure or contrast sliders. Give all five.
"chroma" is the mean colourfulness of the finished frame on the x100 scale of the colourfulness table (the
baseline's frame mean is given with the measurements), read AT IMAGE 1'S BRIGHTNESS: a lifted frame is more
colourful at the same saturation and a darkened one less, so the solver scales your number with the brightness
change of your tone targets (a night lifted 1.5 EV about doubles it); write the colour you want as it would read
at the original brightness. What the hue/chroma targets of masks add to or take from the frame comes on top:
chroma is for the rest of the frame. The solver sets global saturation to reach it. Leave color.saturation out. With a look the look sets the colour and chroma only nudges it (saturation within 10 of
the look's own): choose the look strength for the amount of colour change you want.
A natural finish usually sits at 1.0-1.3 times the baseline's mean; a flat, hazy file that is
stretched hard up to 1.5 times; a muted or matte look at 0.7-0.95 times. Well above 1.5 times reads as
oversaturated (a developed JPEG; a RAW's grey negative takes more, see the source note). Give it in every version.
Reference placements (adapt them to the scene; never force one scene into another's numbers):
- normal daylight: black .02-.04, shadows .10-.18, mid .38-.50, highlights .76-.88, white .93-.97
- high key (snow, salt, bright sand, mist): black .06-.14, shadows .30-.45, mid .58-.72, highlights .86-.93,
  white .96-.985
- low key, dusk, heavy overcast: black .01-.03, shadows .05-.10, mid .20-.32, highlights .55-.75, white .90-.96
- night with lamps: black .008-.02, shadows .02-.05, mid .05-.14, highlights .25-.55, white .80-.95
- backlit subject on a bright background: keep the background bright and lift the subject with a mask target.
The targets are for the whole frame after all masks. A stretch from a flat baseline (white minus black under .55)
to a full range is what makes a hazy or dull file look finished; a high-contrast baseline needs little change.
The ranges are where a flat or dull file should go, not a place to drag a file that already has its light: move
each percentile from image 1 only as far as the picture needs. Bright sand at mid .89 keeps mid about .78-.86 and
gets its density from masks on the rock or the shadows; pulled down to .60 it turns into grey cardboard.
LIMITS FROM IMAGE 1: in linear light the solver keeps shadows and mid within -1.5..+1.6 EV of image 1's
percentiles, highlights and white within -1.5..+2.0 EV; black is free. A target past that is executed at the
limit and reported. In display values +1.6 EV is about x2.6 for a value near .03, x1.75 near .2, x1.65 near .5.
More changes what the scene is (a night reads as dusk, lifted shadows turn to grey fog and noise) and the frame
fails the delivery check.
THE CURVE MOVES EVERY PIXEL OF ONE BRIGHTNESS TOGETHER. A face or subject that sits among the darkest 10-20% of
the frame goes down with black and shadows, and a mask lifts it back by at most 1.25 EV. Before you push black or
shadows down, look up the subject's luma in the cell table: when it is that dark, keep black and shadows near
image 1 or above, or accept a dark subject. When a mask cannot reach its target the solver names the frame target
to move and by how much."""

_UNITS = """MASKS. Coordinates are normalized to the image (x 0 left..1 right, y 0 top..1 bottom). Each mask is
{"name","purpose","shape":{..},"limit":{..}?,"exclude":{..}?,"edge":0..100?,"adjust":{..},"target":{..}?}
and covers shape INTERSECTED with limit, MINUS exclude. Any shape kind may be used as shape, limit or exclude:
- linear: {"kind":"linear","start":{"x":..,"y":..},"end":{"x":..,"y":..}}. Full effect at start, fading to
  zero at end. Sky from the top: start y=0, end just below the horizon. Foreground: start y=1, end y=0.75.
- radial: {"kind":"radial","center":{"x":..,"y":..},"rx":..,"ry":..,"angle":0,"feather":70,"invert":false}.
  rx is a fraction of image WIDTH, ry a fraction of image HEIGHT. invert:true affects everything OUTSIDE.
- luminance: {"kind":"luminance","lo":0.55,"hi":1.0,"feather":0.06}. Pixels whose luma in image 1 (see the
  cell table, divided by 100) lies between lo and hi. Pixel-accurate.
- color: {"kind":"color","points":[{"x":..,"y":..},{"x":..,"y":..}]}. Everything that has the colour found at
  1-4 sample points you place on the region in image 1: a lake, blue sky, grass, sand, a wall, a coat.
  Pixel-accurate edges. Put the points well inside the region, on its typical colour.
A geometric shape alone shows as a soft blob on a uniform surface. Combine shapes so local work is invisible:
- sky: linear from the top with limit = luminance above the land's brightness, or color sampled in the sky
- a dark subject or face on a brighter background: radial around it with limit = luminance lo 0, hi just above
  the subject's brightness, so the background is untouched
- a luminance limit must INCLUDE the thing it is for: lo below its darkest part, hi above its brightest (read
  them from the cell table). A face at .12 under a hat brim needs lo 0; a camel at .79 on sand at .90 needs hi
  about .82. A limit that leaves the subject out leaves it untouched.
- the range fades out over "feather" BEYOND lo and hi: hi .86 with feather .06 still lifts sand at .88 by two
  thirds of the mask's exposure, a glowing ring around the subject. Keep hi + feather below what must stay
  untouched (camel .79 on sand .88: hi .82, feather .04).
- water, foliage, sand, a wall: color with 2-3 points; add a radial or linear limit to keep it to one area
- a bright distraction: radial with limit = luminance above its surroundings
- "edge": 40-70 snaps a linear or radial mask to nearby object edges.
"target": {"box":[left,top,right,bottom],"median":0.45,"hue":35,"chroma":18} states what a tight box well inside
the masked thing must MEASURE in the finished picture; the solver then finds this mask's values. Either part may
be used alone:
- "median": median luma inside the box. The solver moves this mask's exposure, so leave adjust.exposure out. Use
  it for what must land exactly: a face (daylight skin .45-.62, shade or backlit .38-.50, night .25-.40), the main
  subject, a sky (deep blue .38-.52, pale blue .55-.68, bright overcast .72-.84), a foreground that must not sink.
  FIRST READ THE FACE'S BOX IN IMAGE 1: a target is what a mask can do from there, at most about 1.25 EV up
  (x1.9 near .12, x1.75 near .2, x1.65 near .5). The skin values above are for a face the light reaches. A face
  deep in a hat brim's or hood's shadow (image 1 .10-.18) stays a face in shade and aims .18-.28: that reads
  natural and keeps the brim's shadow. Give that face's mask adjust.shadows +40..+60 too: exposure alone stops
  at its limit there, and shadows lifts the face's darkest part about a third of a stop further while it still
  reads as skin in shade. Sunlit skin there (.40+) leaves the mask at its limit and makes the
  solver lift the frame's black towards it (grey frame), and where reached it reads as a pasted-on patch.
- "hue" and "chroma": the colour inside the box. hue is in colour-wheel degrees as in the hue table (0 red, 30
  orange, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta), chroma on the x100 scale of the colourfulness
  table. The solver moves this mask's temp, tint and saturation, so leave those out of adjust. Use it for a
  surface whose colour must be right: sand that must stay warm gold (hue 30-45), a blue sky (205-225), water,
  foliage, skin (hue 15-30), or snow, cloud and white clothes that must be neutral (chroma 2-5 with the hue
  measured there). Stay within 0.7-1.4 times the colourfulness measured in the box (on a developed picture; a
  RAW negative's colour is developed further, see the source note) and within about 20 degrees
  of its hue: the solver moves a mask's temp at most +/-24, tint +/-18 and saturation -40..+30 (-70..+30 on a
  RAW: taking out a negative's cast never casts), because more
  reads as a colour cast. A color-shape mask selects its own colour, so there it goes to temp/tint +/-60: a
  colour target on a color mask is the way to turn one large area (teal water from blue, about 10 degrees on
  saturated water); not on a RAW, where every mask keeps temp +/-24 and tint +/-10 (see the source note). A palette change of the whole frame belongs to white balance, calibration, grading or a
  look (mixer hue barely moves here). A surface that is darkened loses colour, so a burned sky or foreground normally needs a
  colour target to stay alive. Boxes darker than .08 carry no hue; the solver skips their colour.
The box must lie INSIDE the area the mask covers and ON the thing: read its edges off image 2's grid lines. A box
0.05 beside a small subject measures the background, and the solver then works on the wrong pixels. The solver
reports a mask that does not move its box, or reaches it only beside it.
adjust keys (only those you need): exposure (EV), contrast, highlights, shadows, whites, blacks, temp, tint,
saturation, clarity, texture, dehaze (slider scale -100..100 except exposure). Typical local magnitudes:
exposure +/-0.15..0.60, highlights -15..-50, shadows +10..+35, dehaze +6..+25, clarity/texture -30..+18 (down
to soften bright or soft areas, up only on texture that should read),
temp +/-4..12, saturation +/-5..20. The solver moves a bare linear or radial mask at most +/-0.7 EV (more shows
as a blob or a halo) and a mask with a luminance or color shape or limit at most +/-1.25 EV.
GLOBAL develop groups (omit what you do not change; values are absolute, 0 = no change). Tonal placement comes
from "tone"; use these for character:
- light: exposure (EV -1.5..1.5), contrast, highlights, shadows, whites, blacks. With three or more tone targets
  these and the parametric curve are NOT executed: the solver places every tone with exposure and the master
  curve, and a second set of tone controls under it only made the curve steeper. Recover a bright area or open
  a dark one with a mask (highlights or shadows on a luminance-limited mask), not globally.
- wb: temp (Kelvin OFFSET from as-shot, -1500..1500, positive = warmer), tint (-30..30)
- color: vibrance (overall intensity comes from tone.chroma);  effects: texture, clarity, dehaze (global dehaze
  above +12 shifts colour; the tone targets already clear haze, so prefer local dehaze on the hazy layer)
- curve (parametric): highlights, lights, darks, shadows (only without tone targets, see light)
- mixer: {band: {hue, sat, lum}}, bands red orange yellow green aqua blue purple magenta. In this engine sat
  and lum work at full strength but hue barely turns a colour: even +/-100 moves its band only about 4 degrees
  (measured on sand, sky and water), so a mixer hue never changes a palette
- grading: {shadows|midtones|highlights: {hue 0..360, sat 0..30, lum}, blending 0..100, balance -100..100}
- calibration: {red_hue, red_sat, green_hue, green_sat, blue_hue, blue_sat, shadows_tint}. The strongest global
  hue tool here: red_hue -100..+100 turns reds and oranges about -18..+6 degrees, blue_hue -60 turns a blue sky
  about -8 degrees (towards cyan)
- vignette: amount (A down to -10, B and C down to -14), midpoint, feather. On uniform sand, snow or sky, or
  where the corners are already dark, use -6 at most or none: it reads as uneven light or dead corners
- grain: amount (0..45), size, roughness
- detail: sharpen_amount (30-50 gives a crisp finish), nr_luminance (10-30 when dark areas are lifted or ISO
  is high), nr_color"""

_LESSONS = """Lessons from the reference tutorials and from earlier tests on this system:
- Order of work: tonal placement first, then local light, then colour, then finish (sharpening, noise, vignette).
- The photographer's own method, from his narrated tutorials (learn the reasons, not numbers: every photo differs):
  * Tone before colour. Dull or grey colour is a flat-tone problem first: place the tones, then judge the colour.
  * Lift the black end so nothing is dead black and the shadows can carry colour, and hold the white end below
    paper white so the brights do not glare; even his deepest-black looks pull both ends in a little. Keep the
    darks clearly below the mids (a lifted black is not a grey frame): that step between shadows and mid is what
    makes a picture clear (通透). Lift the mids for that clarity; open the shadows only a little (seeing
    everything has no mood).
  * Exposure is the control he sets per photo, and nearly always up: his first versions come out too dark. In
    daylight, between two placements the brighter, airier one is usually right.
  * Lifting the light deepens colour by itself, so a large lift of a developed picture needs little or no added
    colour (tone.chroma near the baseline's, no vibrance on top; a RAW negative is developed, see the source
    note). Richness comes from colour depth (white balance that is right for
    the light, calibration, grading of the tonal zones), not from saturation: calm colour, never candy colour.
    The mixer mostly takes stray colour out so the subject stands out; a band whose saturation goes down gets
    its luminance up, or it turns dirty.
  * Atmosphere comes from toning the tonal zones, not from pulling the white balance off the light: cool
    cyan-blue shadows (hue 200-225) against warm highlights (hue 35-60), at low grading saturation (shadows
    about 8-20, highlights at most about half that); more bleeds into a cast. The brightest whites stay clean:
    a cold white is turned cream, never yellow.
  * Clarity and texture usually go slightly down (-5..-15 on snow, mist, water, sky, skin). High contrast reads
    dirty unless the bright zone is softened: a luminance mask on the brights (lo about .65) with clarity
    -15..-30 and texture -10..-20 leaves the darks crisp. Rugged surfaces (rock, bark, rippled sand) keep a
    little clarity.
  * Many of his frames are lighter at the top and denser at the bottom: a linear mask from the bottom edge
    (start y=1, end y about .6) at -0.2..-0.5 EV with shadows down, unless the bottom holds the subject.
- A full histogram is not the goal; the light relationship between subject and surroundings is. The eye should
  land on the subject: lift it, quieten bright edges and corners, and keep the brightest tone for what matters.
- Bright scenes (snow, salt flats, sand, white walls) must stay bright and luminous. Do not darken them
  globally into grey; recover highlights and add density with local masks.
- Dark scenes: keep night as night. The median goes at most about 1.5 EV above image 1 (a median of .03 to about
  .07, not .12), the lamps stay the brightest thing in the frame, and the darkest quarter stays near black. Lift
  what matters (subject, path, faces) with masks, and keep lamps from clipping with a small radial or a luminance
  mask (highlights down). Do not lift the whole frame into grey fog.
- Haze and fog are atmosphere: keep them in the far distance; give the near and middle layers contrast and
  colour (tone targets plus local dehaze or clarity on those layers). Never one strong global dehaze.
- Avoid cyan, green or magenta casts on ice, snow, fog, cloud and skin. Whites stay neutral unless the light
  itself is coloured. Overcast skies stay grey-white, not brown or purple.
- Do not make a version darker than the baseline unless the baseline is overexposed.
- A tiny edit is not a finished edit. Every version must be visibly better than the baseline, and the three
  versions must be visibly different from each other: B in its light structure (what is bright and what is
  dark), C in its palette (a look at strength 0.6 or more, white balance, calibration, grading, colour targets on
  the large areas; mixer only for a band's saturation or luminance). Versions that differ only in overall
  brightness read as the same picture.
- Fewer, better masks: every mask is another edge to hide and another target that can pull against the curve.
  Two to four precise masks beat six loose ones."""

_PLAN_SHAPE = """{"title": "...", "intent": "...", "tone": {"black": .., "shadows": .., "mid": .., "highlights": ..,
 "white": .., "chroma": ..}, "develop": {"light": {...}, ...}, "look": {"id": "...", "strength": 0.8} or null,
 "masks": [{"name": "sky", "purpose": "...", "shape": {...}, "limit": {...}, "adjust": {...}, "target": {...}}]}"""


def _look_note(entry: dict[str, float], image: float | None) -> str:
    """What one look measured on this photo, in the planner's chroma and sibling-distance terms."""
    parts = []
    ratio = entry.get("chroma")
    if ratio is not None and image:
        parts.append(f"frame colourfulness {ratio * image:.1f} = x{ratio:.2f} image 1; at strength s {image:.1f}"
                     f" {'+' if ratio >= 1.0 else '-'} s x {abs(ratio - 1.0) * image:.1f}")
    if entry.get("distance") is not None:
        parts.append(f"changes the picture by {entry['distance']:.1f}")
    return f"[on this photo at strength 1.0: {'; '.join(parts)}]" if parts else ""


def _looks_block(measured: dict[str, dict[str, float]] | None = None, image_chroma: float | None = None,
                 min_distance: float = 5.0, source_kind: str = "jpeg") -> str:
    """The looks for a prompt; ``measured`` (from ``measure_looks``) and image 1's frame colourfulness x100 add
    what each look does to this photo's colour and how far it moves the picture."""
    notes = {look_id: note for look_id, entry in (measured or {}).items() if (note := _look_note(entry, image_chroma))}
    lines = look_lines(notes)
    if not lines:
        return ""
    text = ("LOOKS: palettes distilled from the professional reference preset library (HSL mix, split toning,"
            " calibration, black and white handling). Version C should normally use one: \"look\": {\"id\": ..,"
            " \"strength\": 0.3..1.0}. Pick the one that fits this scene and light, then add your own develop values"
            " and masks on top. A and B normally use no look. A look changes colour only: your tone targets are"
            " executed as written. The bracketed numbers are what the family's presets do to black and white"
            " (scale them by your strength); write them into your targets only where a matte or a held-back white"
            " suits this picture.")
    if notes:
        text += (" \"on this photo\" was MEASURED: each look rendered on this photo at strength 1.0, read at image"
                 " 1's brightness like the chroma target; the colourfulness moves in proportion to strength.")
        if source_kind == "raw":
            text += (" On this RAW negative the looks were built for developed pictures and add little colour: the"
                     f" solver may raise global saturation up to {RAW_LOOK_SATURATION_UP:g} over the look's own (the"
                     f" development) and lower it at most {LOOK_SATURATION_ROOM:g}, so write C's chroma for the"
                     " developed picture (see the source note) and use the look for its hues and handling.")
        else:
            text += (f" With a look the solver moves global saturation at most {LOOK_SATURATION_ROOM:g} from the look's"
                     " own, about +/-10% of the frame's colourfulness, so write C's chroma from this number at your"
                     " strength: a chroma the look does not allow stays unreached.")
        text += (" Many looks take colour away; for a warmer or"
                 " richer palette pick one whose number is above image 1, or use one that takes colour away at a"
                 " lower strength. \"changes the picture by\" is the mean colour difference of the look alone from"
                 f" image 1, the measure by which two versions closer than {min_distance:g} read as nearly identical;"
                 " it grows about in proportion to strength (0.6 gives 0.6-0.75 of it). C's tone usually sits near"
                 f" A's, so a look below {min_distance:g} does not by itself make C visibly different from A: then"
                 " give C its own white balance, grading, calibration or colour targets on the large areas as well,"
                 " or choose a look with a larger number that suits the scene.")
    return text + "\n" + "\n".join(f"- {line}" for line in lines)


def _source_note(source_kind: str) -> str:
    if source_kind == "raw":
        # Placements from the photographer's own finished edits of his RAWs (resources/, 11 before/after pairs;
        # reference benchmark 2026-11-21): ref-r1 kept image 1's grey (snow mid .45 against his .75, dune white
        # .69-.72 against .94-.96, dune colour x0.2-0.45 of his).
        return ("RAW source: image 1 is the file's undeveloped render, flat, grey and about 1 EV darker than a raw"
                " converter shows it. It is the negative, not a picture whose light is already set: develop it to"
                " where the finished scene belongs, and read 'move each percentile only as far as the picture"
                " needs' against the finished scene, not against image 1. The photographer's own finished edits of"
                " RAWs like these (reference library) moved far from the negative: snow and mist over water: mid"
                " .69-.80, highlights .84-.92, white .88-.96, black .09-.26 (a soft, lifted black), shadows"
                " .25-.70 (a grey snowfield or lake in image 1 is underexposed, not a dark scene); inside such a frame"
                " the mist is the brightest soft layer (.80-.88) and a lake or water surface stays light (.58-.70, it"
                " mirrors a bright sky: ref-r4 wrote water .46-.50 and mist .70, his .59-.70 and .85 in the same boxes,"
                " and those frames came out 15-18 under his mid), and a clean cool blue: he takes out the negative's"
                " cast but never drains the colour, and his three finished snow frames all measure 9-12 frame"
                " colourfulness at hue about 211 whatever image 1 showed (the brightest snow and mist 3-5, sky 5-8,"
                " open snow, shade and water 9-14). tone.chroma is read at image 1's brightness and the lift"
                " multiplies it, so aim at that 9-12 after the lift: a grey frame lifted a stop or more gains about"
                " 1.5 (write 6-8, about 0.4-0.5 times image 1 where image 1 reads blue at 12 or more; ref-r8 wrote"
                " 9.5 on a frame reading 15.7 and delivered 15 against his 10), a frame already bright gains about 1"
                " (write 10-13; ref-r6 read 11-12 of snow blue on a frame whose mean was 7 as a cast and wrote 4: it"
                " kept a third of his 12); snow and water mask"
                " colour targets keep that blue rather than white; sunlit"
                " sand dunes:"
                " white .93-.98 where sky, haze or a brightly lit crest is in the frame but .69-.86 in a frame of"
                " sand alone, black .10-.23, and the picture is the split between the lit face and the shade side,"
                " which he widens far past image 1's. Measure it there first, in image 1, in the two boxes you will"
                " target, and let that one ratio decide, not how soft the light looks or how sharp the ridge is (the"
                " negative flattens every light alike: ref-r7 measured 1.43, called it hazy and wrote a lit face of"
                " .55 where his is .73; ref-r10 measured 1.2, called it clear because the ridge line was crisp, and"
                " burned the shade side to .23-.25 where his is .31-.38). 1.3 times or more: the light was clear"
                " side light (his three such frames measure 1.4-6.5 in image 1), and he makes the shade side dense"
                " and deep, median .11-.17 in its box (shadows .10-.14) with little colour, under a lit face of"
                " .60-.76, .70-.76 where the lit face holds the frame's middle (see below; his crest frame .73: five rounds"
                " wrote .53-.64 there and came out 15-20 under his mid; ref-r6 wrote lit .54 and opened the shade"
                " side to .28-.44 on two such dunes; ref-r10"
                " wrote shade sides of .20-.24 on three where his measure .11-.17 in the same boxes). Under 1.3,"
                " however crisp the ridge: the light was soft or hazy, and he keeps the two close: lit face"
                " .55-.60, shade side .30-.38 (shadows .28-.32), the shade side keeping its colour: its mask"
                " chroma 20-26, near the lit face's 30 (ref-r11 wrote 7-10 there: on a soft frame that mask holds"
                " most of the picture, and the frame kept half his colour). On such a"
                " frame both faces have one colour and nearly one brightness in image 1, so a color or luminance"
                " mask drawn for one face holds the other too and most of the frame's mid (ref-r10: a shade-side"
                " color mask held half the mid's pixels, the plan gave up the mid for it and delivered .35-.41"
                " against his .50): keep tone.mid at .47-.50 there and let the curve's stretch make the split."
                " The frame's mid is not a goal of its own: it lies on whatever holds the frame's middle pixel, and"
                " image 1 shows what, not the areas you estimate: compare image 1's mid with image 1's lit face in"
                " its box. Within about .03 of each other (his crest frame: mid .33, lit face .33) the mid lies on"
                " the lit face, so write tone.mid at the lit face's own value (his crest frame measures .71; ref-r8"
                " wrote the lit face .69 with mid .46, the pre-render check reported the contradiction and the plan"
                " gave up the lit face, back to .48). Where image 1's mid lies well between the shade side and the"
                " lit face, the middle pixel is on neither face but on the half-lit ground between them, and it"
                " stays there: mid .44-.45 in his two such frames (his aerial frame reads shade .12, mid .53, lit"
                " face .80 in image 1; ref-r12 judged by area that its lit face held the middle, wrote mid .62-.68"
                " and came out 16-20 over his). Under clear light, do not lift shadows to open the shade"
                " side: give it a mask with its median target. Colour: deep warm gold sand, his finished dune frames"
                " measure 25-37 frame colourfulness (about 18 seen from the air with much of the frame in shade); on"
                " a RAW that is development, not oversaturation. Where image 1's sand is grey (frame mean under"
                " about 8) that is 4-5 times image 1's, and as the lift multiplies what you write by 1.5-2, write"
                " tone.chroma at 15-20 (ref-r8 wrote 13 on a frame reading 5 and kept 0.44 of his colour); where"
                " image 1 already carries colour (12 or"
                " more, as a drone file can) write only about 1.5 times it (ref-r6 wrote 28 on a frame reading 12.4:"
                " neon yellow against his 18.5, and the review cut it to 14.5). Warm the white balance (wb temp +800..+1500: saturation alone stops near 2.5"
                " times on this negative, the warm white balance gives the rest), vibrance +40..+70 and a warm"
                " grading of highlights and midtones. Mask colour targets on a RAW are read against the finished"
                " scene too, not image 1's colour: his lit sand measures hue 27-35, chroma 30-42 on the x100 scale"
                " in every dune frame, soft, hazy or seen from the air, so write the lit-sand mask's chroma there, not at a"
                " cautious 17-26 (ref-r4 wrote 17-26 on three dunes and they kept half his colour; in the same boxes"
                " his sand is 36-40), the shade side hue 14-26, chroma 2-10 under clear light (dense and nearly"
                " neutral where deepest: his aerial frame's long shadow measures .11 at chroma 2) and 20-26 under"
                " soft light; a sand target at"
                " image 1's own 5-13 makes the solver take out the colour the white balance gave (ref-r3: sand"
                " target 10, saturation -40, his 41). His masks carry no colour at all: in his 19 RAW edits not one"
                " of 76 mask corrections moves temperature, tint or saturation; every hue is global (wb, vibrance,"
                " HSL, grading). So on a RAW the solver keeps a mask's temp within +/-24 and tint within +/-10 (a"
                " larger turn painted lit sand salmon and shade mauve in ref-r13): set the hues with the white"
                " balance, calibration and grading, and let a mask colour target say how much colour, not turn a hue. In all of these edits, snow, mist and dunes alike, the sky stays"
                " light and pale: the top of the frame .82-.95 with little colour (chroma 6-10, a pale blue-grey or"
                " white-blue), never burned to a deep or saturated blue, so a sky mask on a RAW targets median .82-.95"
                " and chroma 6-10 even where image 1 shows 17-24 of blue, and keeps that in a revision (ref-r3"
                " DSC02262: sky written median .60, chroma 20-30, delivered 31 against his 6, the frame's largest"
                " miss). The bottom of the frame follows what is there, not a habit of darkening the bottom edge"
                " (his 11 finished pairs): open water running to the bottom edge he darkens hard toward it, from"
                " .80-.88 under the mist through .64-.70 mid-water to .26-.31 in the bottom tenth, .4-.5 under the"
                " frame's mid, though image 1 shows that water nearly flat (.52-.62; his two such frames), so give"
                " it a linear from the bottom edge (start y=1, end about y=.7) limited by the water's color, target"
                " box on the bottom tenth at median .28-.35, shadows -30..-50, beside the water's own mask"
                " targeted mid-water (ref-r12 gave such a lake one color mask at .70 and kept its bottom at .58"
                " against his .27); a shade side already dark at the bottom in image 1 stays dense (see dunes);"
                " lit sand or snow at the bottom edge he does not burn: in his six such frames the bottom tenth"
                " lies within -.11..+.09 of the frame's mid and keeps the frame's richest colour, so a RAW plan"
                " draws no darkening linear over lit ground there (ref-r12 burned a dune foreground to .32 at"
                " chroma 15 where his is .47 at chroma 44). On dunes weight comes from the shade side, not from a"
                " dark sky or a burned foreground; on snow nothing is burned for weight: his shaded snow and"
                " foreground snow stay light, .62-.74 in the boxes plans drew there (ref-r10 burned DSC02262's"
                " foreground snow to .28-.41 and its shade side to .39-.52 where his measure .65-.70, and came out"
                " 11-27 under his mid)."
                " Wider limits for a RAW: shadows"
                " and mid up to +3.0 EV over image 1, highlights and white +2.6 EV. Highlights recover strongly."
                " The global wb temp and tint offset is applied to the RAW's own as-shot white balance.")
    return ("Already-developed JPEG: pure white areas hold no detail (do not try to recover them), and strongly"
            " lifted shadows reveal noise and banding (add nr_luminance when a dark file is lifted).")


def _measurements(technical: Any) -> dict[str, Any]:
    keys = ("source_kind", "width", "height", "mean_luma", "median_luma", "highlight_clip", "black_clip",
            "white_clip_fraction", "saturation_mean", "temperature_bias", "noise", "sharpness",
            "top_third_blue", "skin_fraction", "green_fraction")
    data = technical.model_dump() if hasattr(technical, "model_dump") else dict(technical or {})
    out = {key: (round(value, 4) if isinstance(value, float) else value) for key in keys if (value := data.get(key)) is not None}
    iso = (data.get("exif") or {}).get("EXIF ISOSpeedRatings")
    if iso:
        out["iso"] = iso
    return out


def _anchor_text(anchors: dict[str, float]) -> str:
    return ", ".join(f"{name} {anchors[name]:.3f}" for name in tone.ORDER if name in anchors)


def _frame_chroma(stats: dict[str, Any]) -> float:
    """Mean colourfulness of a frame x100, from its own measurement or from the cell table."""
    if stats.get("colourfulness") is not None:
        return float(stats["colourfulness"]) * 100.0
    table = stats.get("chroma") or [[0]]
    return sum(sum(row) for row in table) / max(1, sum(len(row) for row in table))


def _stats_block(stats: dict[str, Any] | None) -> str:
    if not stats:
        return ""
    stats = {"hue": [], **stats}
    return f"""Measured on image 1 (display luma, 0 = black, 1 = white):
- percentiles: {_anchor_text(stats['anchors'])}
- median luma per cell x100 (8x8 cells; x and y are cell centres):
{tone.grid_text(stats['luma'])}
- colourfulness per cell x100 (0 grey, 15-35 natural colour, 55+ very strong); frame mean {_frame_chroma(stats):.1f}:
{tone.grid_text(stats['chroma'])}
- hue per cell in colour-wheel degrees (0 red, 30 orange, 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta;
  -- where a cell has no colour to speak of):
{tone.grid_text(stats['hue'])}
"""


def plan_prompt(technical: Any, size: tuple[int, int], stats: dict[str, Any] | None = None,
                looks: dict[str, dict[str, float]] | None = None, min_distance: float = 5.0) -> str:
    measurements = _measurements(technical)
    source_kind = str(measurements.get("source_kind") or "jpeg")
    width, height = size
    raw = source_kind == "raw"
    # On a RAW the note used to be one bullet at the end of a 33k-character prompt, and the reading (same call) set
    # the goals first: ref-r5 DSC02262 read "preserve the deep blue sky, the sky lacks density, the foreground snow
    # is too bright" and planned a dense sky at .56 (his .85) where ref-r4 had planned it pale.
    first = (f"RAW DEVELOPMENT, read this before anything else; where the general guidance below differs, this wins"
             f" (it is measured from the photographer's own finished edits of RAWs like this one):\n"
             f"{_source_note(source_kind)}\nJudge what is weak and what to preserve against that finished picture,"
             " not against image 1: the negative's deep blue sky, grey snow or dull sand are not the scene's mood.\n"
             if raw else "")
    local_work = ("lift the subject or face, give the shade side or the bottom of the frame its density, burn a bright"
                  " distracting edge, separate distant layers, protect a light source (the sky of these RAWs stays"
                  " light, see above)" if raw else
                  "densify the sky, lift the subject or face, burn a bright distracting edge, separate distant layers,"
                  " protect a light source")
    return f"""You are a senior photo retoucher directing LightCraft, a Lightroom-like raw developer.
Image 1 is the neutral LightCraft render of the photograph (no edits). Image 2 is the same picture with a
coordinate grid: x runs 0 (left) to 1 (right), y runs 0 (top) to 1 (bottom); lines every 0.1.
Source: {source_kind.upper()}, aspect {width / max(height, 1):.3f} (width/height).
File measurements (not artistic targets): {json.dumps(measurements, ensure_ascii=False)}
{_stats_block(stats)}
{first}Work like a professional:
1. Read the picture: scene, focal subject and where it is, light direction and quality, what is weak
   (ordered by severity), and what must be preserved (mood, atmosphere, coloured light).
2. Decide where the tones of the finished picture should sit and write that as tone targets.
3. Design the local work a retoucher would do by hand: {local_work}. Use 2-4 masks per version (5 at
   most), each with a name and a purpose, built so that their edges cannot be seen.
4. Design three finished versions:
   A: {ROLES['A']}
   B: {ROLES['B']}
   C: {ROLES['C']}

{_TONE}

{_UNITS}

{_looks_block(looks, _frame_chroma(stats) if stats else None, min_distance, source_kind)}

{_LESSONS}
- {"The RAW development note at the top applies to everything above." if raw else _source_note(source_kind)}
- Never invent or remove content. No crop.

Return JSON only. All human-readable text in Chinese:
{{"reading": {{"scene": "...", "genre": "landscape|portrait|street|night|indoor|product|general",
  "subject": "...", "subject_box": [left, top, right, bottom] or null, "subject_confidence": 0.0,
  "light": "...", "problems": [{{"issue": "...", "region": "...", "severity": 0.0}}], "preserve": ["..."]}},
 "plans": {{"A": {_PLAN_SHAPE}, "B": {{...}}, "C": {{...}}}}}}"""


_MISSES = ("THE MASK DOES NOT COVER THIS BOX: the box did not respond, so the solver left the mask as planned."
           " Move the mask onto the thing, or put the box inside the mask")
# Where a global white-balance push points on the hue circle of the measurements (0 red, 60 yellow, 120 green,
# 240 blue): tint + adds magenta, temp + adds orange. A small push toward hue p turns a colour at hue h by sin(p - h).
WB_HUE_PUSH = {"tint": 300.0, "temp": 30.0}


def _hue_turn(colour: dict[str, Any], luma: float | None) -> str:
    """Which global control turns a mask's missed hue toward its target ("" when the hue is near or too grey).

    A RAW mask keeps temp and tint to a nudge, so a hue it cannot reach is the palette's: the offline replay of
    ref-r13 DSC05752 B under RAW_MASK_COLOUR_LIMITS held the lit crest at hue 41-46 for 33 with the mask at tint +10."""
    miss = (colour["target_hue"] - colour["after_hue"] + 180.0) % 360.0 - 180.0
    if abs(miss) <= 8.0 or colour["target_chroma"] < 0.06 or colour["after_chroma"] < 0.05:
        return ""
    turns = {control: math.sin(math.radians(push - colour["after_hue"])) for control, push in WB_HUE_PUSH.items()}
    control = max(turns, key=lambda name: abs(turns[name]))
    sign = "+" if turns[control] * miss > 0 else "-"
    zone = ("highlights" if luma is not None and luma >= 0.6 else
            "shadows" if luma is not None and luma < 0.3 else "midtones")
    return (f". The hue sits {abs(miss):.0f} degrees off: on this colour white balance {control} {sign} turns it toward"
            f" {colour['target_hue']:.0f} (it moves the whole frame, so weigh the other regions), and so does the"
            f" {zone} grading wheel set near hue {colour['target_hue']:.0f}")


def _region_lines(region: dict[str, Any]) -> list[str]:
    """What the solver reached for one mask target, in the reviewer's terms."""
    lines = []
    head = f"- region target \"{region['name']}\" box {region['box']}: "
    if region.get("target") is not None:
        text = (f"median {region['before']:.3f} -> {region['after']:.3f} (target {region['target']:.3f}, mask exposure "
                f"{region['exposure']:+.2f} EV)")
        off = abs(region["after"] - region["target"]) > 0.03
        if region.get("kept_lift"):
            kept = region["kept_lift"]
            text += (f". KEPT HIGHER: you wrote {kept['target']:.3f} for it, and your frame tone targets already lift it to"
                     f" about {kept['by_curve']:.3f}, the same way you meant to move it. The solver aimed it at"
                     f" {kept['kept']:.3f} rather than pulling it back with its mask against the curve (that flattens the"
                     " subject) or lowering the frame target at its brightness (that dulls the frame's top). If it must"
                     " sit lower, write a lower frame tone target at its brightness")
        if region.get("missed"):
            text += f". {_MISSES}"
        elif region.get("spilled"):
            held = region["spilled"]
            text += (f". HELD BACK: this mask reaches most of the frame, not just this region. At {held['from']:+.2f} EV it"
                     f" took the whole frame's median more than {held.get('limit_ev', tone.SPILL_LIMIT):g} EV from image 1's"
                     f" {held['unedited']:.3f}"
                     f" (a night reads as dusk), so the solver held it at {held['exposure']:+.2f} EV (frame median"
                     f" {held['median']:.3f}). To light this region, tighten the mask onto it (a radial or brush on the lit"
                     " surface, a narrower color or luminance range), not the frame tone targets")
        elif region.get("limited") and off and region.get("plain"):
            text += (f". NOT REACHED: a bare linear or radial mask is held to {SOLVER_PLAIN_RANGE[1]:g} EV either way,"
                     " because more shows as a blob or a halo. Give the mask a luminance or color limit so it follows"
                     f" the thing (then up to {SOLVER_LOCAL_RANGE[1]:g} EV), or write frame tone targets that already"
                     " bring this region close")
        elif region.get("limited") and off:
            text += (". NOT REACHED: the mask exposure is at its limit. The frame tone targets leave this region too"
                     " far from where you want it for one mask; move the tone targets towards it")
        fix = region.get("frame_fix")
        if fix and off and not region.get("missed"):
            between = [name for name in fix.get("between") or [] if name]
            where = (f"between your frame {between[0]} and {between[1]} targets" if len(set(between)) == 2
                     else f"at your frame {between[0]} target" if between else "")
            text += (f". With the mask at its limit this region goes into the curve at {fix['input']:.3f}{', ' + where if where else ''},"
                     f" and the curve puts it at {fix['by_curve']:.3f}. FIX: write tone.{fix['anchor']} about"
                     f" {fix['needed']:.3f} (now {fix['now']:.3f}) and keep the targets around it in order, or ask"
                     " less of this region. Which one matters more is your call: a face or the subject usually wins"
                     " over a percentile")
        if (not region.get("missed") and region.get("response", 1.0) < 0.6
                and not (region.get("limited") and off)):
            text += (f". The mask is weak here (the measured part moves {region['response']:.1f} EV per EV), so its"
                     " middle moved more than this box")
        if not region.get("missed"):
            if region.get("moved"):
                text += (". THE MASK DOES NOT REACH INTO THIS BOX. It lies just beside it, and the numbers here are"
                         " measured there: check in the picture that the mask is on the thing you meant, then put the"
                         " box on the mask")
            elif region.get("coverage", 1.0) < 0.6:
                text += f". The mask covers {region['coverage']:.0%} of this box; the numbers are for that part"
        if region.get("overrides"):
            text += (". This mask covers most of the frame at that brightness and cannot carry this region alone,"
                     f" so the frame target {', '.join(region['overrides'])} was moved towards it by the rest: the"
                     " frame target and this region target describe the same pixels, write them to agree")
        if region.get("gave_way"):
            text += (f". Your frame {', '.join(region['gave_way'])} target sits at this region's brightness and would"
                     " have taken it the other way, so that target gave way instead of the mask undoing the curve:"
                     " a subject wins over a percentile. To get that percentile, write tone targets that leave this"
                     " region where you want it, move the rest of the frame with masks, or ask less of this region")
        if region.get("shares"):
            text += (f". Most of the frame at the {', '.join(region['shares'])} percentile is this region, so that"
                     " frame target reads what this mask does and missed: the two targets describe the same pixels,"
                     " write them to agree")
        if region.get("against_curve") is not None:
            text += (f". THE FRAME TONE TARGETS MOVE THIS BOX {region['against_curve']:+.1f} EV AND THE MASK HAS TO UNDO IT:"
                     " the tone targets and this region target contradict each other, which steepens the curve and"
                     " makes the picture harsh. Write tone targets that already put this region near where you want it")
        lines.append(head + text)
    colour = region.get("colour")
    if colour:
        text = (f"hue {colour['before_hue']:.0f} -> {colour['after_hue']:.0f} (target {colour['target_hue']:.0f}), "
                f"colourfulness x100 {colour['before_chroma'] * 100:.0f} -> {colour['after_chroma'] * 100:.0f} "
                f"(target {colour['target_chroma'] * 100:.0f}); mask temp {colour['temp']:+.0f}, tint "
                f"{colour['tint']:+.0f}, saturation {colour['saturation']:+.0f}")
        if colour.get("missed"):
            text += f". {_MISSES}"
        elif colour.get("dark"):
            text += (". NOT EXECUTED: this region is too dark to carry a hue (median luma below"
                     f" {tone.COLOUR_MIN_LUMA:g}); pushing colour into it only tints noise. Lift it first or drop the"
                     " colour target")
        elif (abs(colour["after_chroma"] - colour["target_chroma"]) > max(0.03, 0.25 * colour["target_chroma"])
              or (colour["target_chroma"] >= 0.06 and colour["after_chroma"] >= 0.05
                  and abs((colour["after_hue"] - colour["target_hue"] + 180.0) % 360.0 - 180.0) > 8.0)):
            if colour.get("limits"):        # The range this mask was solved in (a RAW keeps temp and tint narrow).
                limits, wide = {control: tuple(span) for control, span in colour["limits"].items()}, ""
            else:
                limits = tone.COLOUR_LIMITS
                wide = (f" (a color-shape mask +/-{COLOUR_MASK_LIMITS['tint'][1]:g})" if COLOUR_MASK_LIMITS else "")
            text += (f". NOT REACHED: on this mask the solver stays within temp +/-{limits['temp'][1]:g}, tint"
                     f" +/-{limits['tint'][1]:g}{wide} and saturation {limits['saturation'][0]:g}..+{limits['saturation'][1]:g},"
                     " because more reads as a colour cast, and it does not buy colourfulness with a temp or tint"
                     " shift. The target is too far from the file for a local mask: ask for less here and use white"
                     " balance, calibration, grading or a look for a palette change (mixer hue barely moves here)")
            text += _hue_turn(colour, region.get("after"))
        lines.append(head + text)
    return lines


def _look_fix(chroma: dict[str, Any], look: dict[str, Any] | None, looks: dict[str, dict[str, float]] | None,
              image: float) -> str:
    """For a frame held at the look's saturation room and short of its chroma: what the look gives on this photo
    and the strength that fits the written chroma ("" when the look is not why it missed)."""
    room = chroma.get("look_room")
    ratio = ((looks or {}).get((look or {}).get("id") or "") or {}).get("chroma")
    if not room or ratio is None or image <= 0:
        return ""
    pinned = abs(abs(chroma["saturation"] - chroma.get("saturation_start", 0.0)) - room) < 0.5
    if not pinned or abs(chroma["after"] - chroma["target"]) <= 0.01:
        return ""
    strength = float(look.get("strength", 0.8))
    gives, full, written = image * (1.0 + strength * (ratio - 1.0)), image * ratio, chroma.get("written", chroma["target"]) * 100
    if 0.9 * gives <= written <= 1.1 * gives:
        return ""
    text = (f"- the look {look['id']} measured on this photo at image 1's brightness: {gives:.1f} at strength {strength:g}"
            f" (image 1 {image:.1f}, {full:.1f} at strength 1.0), about {0.9 * gives:.1f}-{1.1 * gives:.1f} with the"
            f" saturation room; the chroma written is {written:.1f}, outside what this look gives. FIX: ")
    needed = (written - image) / (full - image) if abs(full - image) >= 0.05 * image else None
    if needed is not None and 0.2 <= needed <= 1.0:
        return text + f"strength {needed:.2f} gives the written chroma with this look, or write chroma {gives:.1f}"
    return text + (f"this look does not give {written:.1f} at any strength; write chroma near {gives:.1f}, or choose a"
                   f" look whose number is {'above' if written > image else 'below'} image 1")


def measured_block(solve: dict[str, Any] | None, baseline: dict[str, Any] | None, render: dict[str, Any] | None,
                   siblings: dict[str, float] | None = None, min_distance: float = 5.0,
                   sibling_stats: dict[str, dict[str, Any]] | None = None, *, look: dict[str, Any] | None = None,
                   looks: dict[str, dict[str, float]] | None = None) -> str:
    """Numbers for the reviewer: what the solver reached and how the frame changed.

    ``sibling_stats`` (picture_stats of the siblings' first renders) show, for a sibling this version
    is nearly identical to, where the two are alike and which way they differ. ``look`` (the plan's) and
    ``looks`` (from ``measure_looks``) explain a colourfulness target the look kept out of reach.
    """
    lines: list[str] = []
    if baseline and render:
        targets = (solve or {}).get("targets") or {}
        rows = []
        for name in tone.ORDER:
            row = f"{name} {baseline['anchors'][name]:.3f} -> {render['anchors'][name]:.3f}"
            rows.append(row + (f" (target {targets[name]:.3f})" if name in targets else ""))
        lines.append("- tone percentiles, baseline -> this version: " + "; ".join(rows))
        shift = tone.ev_between(baseline["anchors"]["mid"], render["anchors"]["mid"])
        if abs(shift) >= 0.3:
            lines.append(f"- the median moved {shift:+.1f} EV in linear light against the baseline"
                         + (" (more than 1.5 EV either way changes what the scene is: a night reads as dusk, lifted"
                            " shadows show noise and blocking)" if abs(shift) > 1.5 else ""))
    if solve:
        if abs(solve.get("exposure", 0.0)) > 1e-6:
            lines.append(f"- solver: global exposure {solve['exposure']:+.2f} EV; largest remaining miss "
                         f"{solve.get('error', 0.0):.3f}")
        chroma = solve.get("chroma")
        if chroma:
            written, gain = chroma.get("written", chroma["target"]), chroma.get("tone_gain", 1.0)
            scaled = (f" (written {written * 100:.1f} at image 1's brightness; the brightness change alone made the frame"
                      f" x{gain:.2f} as colourful, so that is what it is held to)" if abs(gain - 1.0) >= 0.05 else "")
            added = chroma.get("regions_added", 0.0)
            if abs(added) >= 0.005:
                scaled += f" ({added * 100:+.1f} of it from the region colour targets)"
            lines.append(f"- frame colourfulness x100: target {chroma['target'] * 100:.1f}{scaled}, "
                         f"{chroma['after'] * 100:.1f} with global saturation {chroma['saturation']:+.0f} on the small"
                         " solver preview"
                         + (f"; with a look the solver moves saturation at most {chroma['look_room']:g} from where the"
                            " look puts it, because the look is this version's colour: for more or less colour change"
                            " the look strength" if chroma.get("look_room") else ""))
            fix = _look_fix(chroma, look, looks, _frame_chroma(baseline)) if baseline else ""
            if fix:
                lines.append(fix)
        for region in solve.get("regions") or []:
            lines.extend(_region_lines(region))
        for note in solve.get("notes") or []:
            lines.append(f"- solver note: {note}")
    if baseline and render:
        delta = [[after - before for after, before in zip(row_after, row_before)]
                 for row_after, row_before in zip(render["luma"], baseline["luma"])]
        lines.append("- median luma per cell x100 of this version (8x8; x and y are cell centres):\n"
                     + tone.grid_text(render["luma"]))
        lines.append("- change against the baseline per cell (this version minus baseline, x100):\n"
                     + tone.grid_text(delta, signed=True))
        lines.append(f"- colourfulness per cell x100 of this version (frame mean {_frame_chroma(baseline):.1f} -> "
                     f"{_frame_chroma(render):.1f}; 0 grey, 15-35 natural, 55+ very strong):\n"
                     + tone.grid_text(render["chroma"]))
        if render.get("hue"):
            lines.append("- hue per cell of this version in colour-wheel degrees (0 red, 30 orange, 60 yellow, 120"
                         " green, 180 cyan, 240 blue, 300 magenta; -- no colour):\n" + tone.grid_text(render["hue"]))
    for other, distance in sorted((siblings or {}).items()):
        verdict = "they look nearly identical; this version must differ visibly" if distance < min_distance else "distinct"
        lines.append(f"- difference of this render from version {other}'s first render: {distance:.1f} (below"
                     f" {min_distance:.0f} = nearly identical): {verdict}")
        sibling = (sibling_stats or {}).get(other)
        if distance < min_distance and sibling and render:
            # p11-r1 DSC_1276 B: told only "4.2, nearly identical", three revisions kept the night near image 1
            # while A had lifted it 0.8 EV; nothing said where the two were alike or which way to move.
            ones = "; ".join(f"{name} {sibling['anchors'][name]:.3f} / {render['anchors'][name]:.3f}"
                             for name in tone.ORDER)
            alike = [[mine - theirs for mine, theirs in zip(row_mine, row_theirs)]
                     for row_mine, row_theirs in zip(render["luma"], sibling["luma"])]
            lines.append(f"- version {other}'s first render against this one: tone percentiles {other} / this: {ones};"
                         f" frame colourfulness x100 {_frame_chroma(sibling):.1f} / {_frame_chroma(render):.1f}."
                         f" Luma per cell, this version minus version {other} (x100; near 0 is where the two look the"
                         " same, so that is where this version has to become different):\n"
                         + tone.grid_text(alike, signed=True))
    return "\n".join(lines)


def review_prompt(slot: str, plan: dict[str, Any], reading: dict[str, Any], gate: dict[str, Any],
                  source_kind: str, measured: str = "", *, looks: dict[str, dict[str, float]] | None = None,
                  image_chroma: float | None = None, min_distance: float = 5.0) -> str:
    gate_text = "no pixel damage found."
    if gate.get("hard"):
        gate_text = ("pixel damage that BLOCKS delivery: " + "；".join(gate["hard"])
                     + ". Your revision must remove the cause.")
    if gate.get("advisory"):
        # These rules compare brightness, amount of change and HSV saturation with the baseline. They were
        # written for a pipeline that had to stay close to the file and are not findings about this version.
        gate_text += (" Remarks from rules that only compare brightness, amount of change and HSV saturation with"
                      " the baseline (they also fire on a deliberate, well made edit; judge these points by eye): "
                      + "；".join(gate["advisory"]))
    shown = _shown(plan)
    executed = {key: shown.get(key) for key in ("tone", "develop", "look", "masks") if shown.get(key)}
    return f"""You are reviewing version {slot} ("{plan.get('title')}") of a retouch, as a demanding senior retoucher.
Image 1 is a board: panel 1 (left) is the unedited LightCraft render, panel 2 (right) is this version.
Image 2 is this version alone with a coordinate grid (x 0..1 left to right, y 0..1 top to bottom) and the masks
drawn over it, each in its own colour and numbered in plan order (1 = first mask): a radial as its ellipse
(dotted when inverted), a linear as a solid line where its effect is full and a dashed line where it has faded
out, a geometric limit thinner, and the target box as a rectangle with the mask's number. Luminance and color
ranges are not drawn. Use it to check that every box sits on the thing it names and every mask on its area;
judge the picture itself on panel 2 of image 1, which has no drawing on it.
Photo reading: {json.dumps(reading, ensure_ascii=False)}
Role of this version: {ROLES.get(slot, '')}
Intent: {plan.get('intent')}
Plan as executed (the solver has already set global exposure and saturation, the exposure, temp, tint and
saturation of masks with a target, and the master curve): {json.dumps(executed, ensure_ascii=False)}
Measured:
{measured or '- not available'}
Automatic pixel check: {gate_text}

Judge panel 2 against what a top professional would deliver from this same file with non-generative tools.
Score = 100 minus deductions in five areas:
- tone (25): tonal placement and range, light hierarchy, depth; not flat, not muddy, not crushed or blown
- color (20): white balance, believable skin, whites and neutrals, palette harmony, saturation
- subject (20): the subject reads at once; local work is well placed and its edges cannot be seen
- technical (20): no clipping, halos, banding, noise, colour casts or lost texture
- finish (15): polish, and whether the version fulfils its role and differs visibly from its siblings
95 or more: nothing a top retoucher would still change. 90-94: one or two small things. 80-89: clearly
improvable. Below 80: visible faults, or a timid, unfinished edit. Most first renders score below 90.
Name every deduction with its points and a concrete, visible reason. Check each mask purpose and the global
intent against what is VISIBLE: achieved, partial, missed or overdone. Look for damage: clipped highlights,
crushed shadows, halos or visible mask edges, colour casts (cyan, green, magenta), banding in skies, lost texture.
Decide whether panel 2 is better than panel 1 as a finished photograph.
If the score is below 95 and a change would raise it, set revise=true and return the COMPLETE revised plan in the
same format (tone, develop, look, masks). Corrections may go either way: move the tone targets, strengthen what is
not visible, weaken what is overdone, move or resize masks that miss, make a visible mask precise with a limit.
Change only what your deductions name, and keep everything else exactly as it is (the same masks with the same
shapes, boxes and targets, the same look and develop values): in earlier tests, revisions that rewrote parts that
worked lost more than they gained. Where the measurements say NOT REACHED or give a FIX, act on it; a target the
solver could not reach does not get closer by being written again. If nothing would improve it, revise=false and
plan=null.

{_TONE}

{_UNITS}

{_looks_block(looks, image_chroma, min_distance, source_kind)}

{_LESSONS}
- {_source_note(source_kind)}

Return JSON only, Chinese text:
{{"deductions": [{{"area": "tone|color|subject|technical|finish", "points": 0, "why": "..."}}], "score": 0,
 "verdict": "better|same|worse", "confidence": 0.0,
 "checks": [{{"item": "mask name or 全局", "status": "achieved|partial|missed|overdone", "note": "..."}}],
 "damage": [{{"kind": "clipping|shadow_crush|halo|mask_edge|color_cast|banding|detail_loss", "region": "...", "severity": 0.0}}],
 "notes": ["..."], "revise": false, "plan": {_PLAN_SHAPE} or null}}"""


def refine_prompt(instruction: str, plan: dict[str, Any], source_kind: str) -> str:
    shown = _shown(plan)
    current = {key: shown.get(key) for key in ("title", "intent", "tone", "develop", "look", "masks") if shown.get(key)}
    return f"""A user wants to change a finished retouch. Image 1 is the current result with a coordinate grid
(x 0..1 left to right, y 0..1 top to bottom). The plan that produced it (tone targets, LightCraft develop, masks):
{json.dumps(current, ensure_ascii=False)}
User request (Chinese, authoritative): {instruction}

Change only what the request asks for and keep everything else, including the tone targets unless the request is
about overall brightness or contrast. Prefer a local mask when the request names a region. Make the change clearly
visible but free of damage. Luminance and colour shapes refer to the ORIGINAL unedited picture, which you cannot
see here: use them only when the plan already does, and otherwise place linear or radial shapes on image 1.

{_TONE}

{_UNITS}
- {_source_note(source_kind)}

Return JSON only, Chinese text:
{{"summary": "一句话说明改了什么", "plan": {_PLAN_SHAPE}}}"""


# --------------------------------------------------------------------------
# Vision calls


def _vision_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"vision_timeout": max(settings.vision_timeout,
                                                             settings.directed_vision_timeout)})


def _ask(settings: Settings, selection: dict[str, Any], prompt: str, images: list[str],
         audit: dict[str, Any]) -> dict[str, Any]:
    """One vision request; retries once when the answer is not a JSON object.

    A call that gets no answer at all is asked again after each of VISION_RETRY_WAITS before it gives up.
    """
    from .critique import _extract_json

    last: Exception | None = None
    for attempt in range(2):
        asked = prompt if attempt == 0 else prompt + "\nYour previous answer was not valid JSON. Reply with the JSON object only."
        for wait in (*VISION_RETRY_WAITS, None):
            transport: dict[str, Any] = {}
            try:
                text = chat_vision(_vision_settings(settings), selection.get("resolved_model") or "", asked,
                                   images, selection=selection, audit=transport)
                break
            except VisionUnavailable:
                audit.setdefault("calls", []).append(transport)
                if wait is None:
                    raise
                time.sleep(wait)
        audit.setdefault("calls", []).append(transport)
        try:
            return _extract_json(text)
        except ValueError as exc:
            last = exc
    raise ValueError(f"vision answer was not JSON: {last}")


def _reading(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    subject_box = _box(data.get("subject_box"))
    problems = []
    for item in data.get("problems") or []:
        if isinstance(item, dict) and str(item.get("issue") or "").strip():
            severity = _number(item.get("severity"))
            problems.append({"issue": str(item["issue"]).strip()[:160], "region": str(item.get("region") or "").strip()[:60],
                             "severity": round(min(1.0, max(0.0, 0.4 if severity is None else severity)), 2)})
    genre = str(data.get("genre") or "general").strip().lower()
    confidence = _number(data.get("subject_confidence"))
    return {"scene": str(data.get("scene") or "").strip()[:300], "genre": genre if genre in GENRES else "general",
            "subject": str(data.get("subject") or "").strip()[:160], "subject_box": subject_box,
            "subject_confidence": round(min(1.0, max(0.0, confidence or 0.0)), 2) if subject_box else 0.0,
            "light": str(data.get("light") or "").strip()[:200],
            "problems": sorted(problems, key=lambda p: -p["severity"])[:8],
            "preserve": [str(x).strip()[:80] for x in (data.get("preserve") or []) if str(x).strip()][:6]}


_PLANS_AGAIN = ('\nYour previous answer had no usable "plans" object. Reply with the complete JSON object: "reading" and'
                ' "plans" with the keys "A", "B" and "C", each a full plan.')


_PLANS_REST = ('\nYour previous answer was cut off after the plan(s) {done}. Reply with a JSON object holding only "plans"'
               ' with the key(s) {missing}, each a full plan in the same format and each clearly different from the plan(s)'
               ' already written (the reading stays as it was). Already written:\n{written}')


def _plan_rest(settings: Settings, selection: dict[str, Any], prompt: str, images: list[str], audit: dict[str, Any],
               plans: dict[str, dict[str, Any]], notes: dict[str, list[str]], missing: list[str]) -> None:
    done = [slot for slot in SLOTS if slot in plans]
    written = json.dumps({slot: _shown(plans[slot]) for slot in done}, ensure_ascii=False)
    try:
        data = _ask(settings, selection, prompt + _PLANS_REST.format(done=", ".join(done), missing=", ".join(missing),
                                                                     written=written), images, audit)
    except (VisionUnavailable, ValueError):
        return
    raw = data.get("plans") if isinstance(data.get("plans"), dict) else data
    for slot in missing:
        if not isinstance(raw.get(slot), dict):
            continue
        try:
            plan, note = sanitize_plan(raw[slot], slot=slot)
        except ValueError as exc:
            notes.setdefault(slot, []).append(str(exc))
            continue
        if is_executable(plan):
            plans[slot], notes[slot] = plan, note
        else:
            notes[slot] = note + ["方案没有任何可执行的调整"]


_PLANS_REACH = ('\nMeasured on image 1 before anything was rendered, these region targets cannot be met as written,'
                ' or contradict the frame\'s tone targets (the 8x8 cell table cannot show a small region\'s own'
                ' brightness):\n{lines}\nA mask held at its stop leaves the region short of what the plan describes,'
                ' a mask pushing back against the curve reads as a fight, and the review counts both against the version.'
                ' Reply with a JSON object holding only "plans" with the key(s) {slots}, each the full plan in the same'
                ' format, changed only where these numbers call for it: aim each region out of reach at what its mask'
                ' reaches from there (a face in shade stays a face in shade and reads natural; a dark animal keeps its'
                ' dark coat), and {frame_rule}. Where the subject must'
                ' still stand out, do it around it: darken its surroundings with their own masks, or separate it by'
                ' colour. Put a color mask\'s points on what its box holds, and on nothing that must not move with it.'
                ' Keep each version\'s role, and keep the versions as different from each other as they were.'
                ' Your plans were:\n{written}')
_REACH_FRAME_RULE = {
    "jpeg": ("keep the frame's tone targets at that brightness near image 1 so the curve does not take the region"
             " away first (not above image 1: that turns the frame grey)"),
    "raw": ("keep the frame's tone targets where the finished scene belongs: image 1 is the RAW's undeveloped"
            " negative, so do not pull the frame back towards it to make a region target fit; aim the region with"
            " the developed frame instead"),
}


def _points_line(slot: str, item: dict[str, Any]) -> str:
    sampled = ", ".join(f"{point['luma']:.2f} at ({point['x']:.2f}, {point['y']:.2f})" for point in item["points"])
    return (f"- plan {slot}, color mask \"{item['name']}\", box {item['box']}: image 1 measures {item['image']:.2f} in the"
            f" box, but the points sample luma {sampled}; only {item['share'] * 100:.0f}% of the box has a colour near"
            " a point, so the mask selects other pixels than the box shows, and the solver measures and moves only"
            " the pixels it selects. The points are misplaced for this box.")


def _reach_line(slot: str, item: dict[str, Any], source_kind: str = "jpeg") -> str:
    if item.get("kind") == "points":
        return _points_line(slot, item)
    if item.get("kind") == "against":
        lift, back = item["curve_ev"] > 0, item["need"] > 0
        if lift and source_kind == "raw":
            # A RAW's lift is its development: pulling the frame back towards the negative to keep a dark sky
            # left ref-r2 DSC02262 at mid .45 (his .75) and DSC05752 at white .79 (his .96).
            return (f"- plan {slot}, mask \"{item['name']}\", box {item['box']}: image 1 measures {item['image']:.2f}"
                    f" on the pixels this mask holds there. Your frame tone targets develop that brightness"
                    f" {item['curve_ev']:+.2f} EV, to about {item['by_curve']:.2f}, and the target"
                    f" {item['target']:.2f} then needs the mask to {'lift' if back else 'darken'} it"
                    f" {item['need']:+.2f} EV: the curve and the mask push the same pixels opposite ways. The frame's"
                    " lift is the RAW's development, so keep the frame targets and aim the region at about"
                    f" {item['edge']:.2f}, with the developed picture.")
        text = (f"- plan {slot}, mask \"{item['name']}\", box {item['box']}: image 1 measures {item['image']:.2f} on the"
                f" pixels this mask holds there. Your frame tone targets {'lift' if lift else 'darken'} that brightness"
                f" {item['curve_ev']:+.2f} EV, to about {item['by_curve']:.2f} (the nearest frame percentile is the"
                f" {item['anchor']}, {item['anchor_image']:.2f} in image 1, written {item['anchor_written']:.2f}), and the"
                f" target {item['target']:.2f} then needs the mask to {'lift' if back else 'darken'} it {item['need']:+.2f}"
                " EV: the curve and the mask push the same pixels opposite ways, which reads as one undoing the other."
                f" Make them agree: aim the region at about {item['edge']:.2f} (the curve's {item['by_curve']:.2f} moved"
                f" less than {tone.AGAINST_EV:g} EV by the mask)")
        if item.get("anchor_fix"):
            text += (", or write " + " and ".join(f"the {name} at about {value:.2f}"
                                                  for name, value in item["anchor_fix"].items())
                     + f" so the curve {'lifts' if lift else 'darkens'} this region less")
        return text + ", whichever this version's role calls for."
    if item.get("kind") == "shared":
        anchor = item["anchor"]
        return (f"- plan {slot}, {item.get('shape') or 'color'} mask \"{item['name']}\", box {item['box']}: its region holds"
                f" {item['share'] * 100:.0f}% of the pixels at the frame's {anchor} (image 1 {item['image']:.2f}), so the"
                f" frame {anchor} is measured on this region. Your tone targets take the region to about"
                f" {item['by_curve']:.2f} and its target {item['target']:.2f} moves it {item['ev']:+.2f} EV from there,"
                f" so the frame {anchor} will measure about {item['simulated']:.2f}, not the {item['written']:.2f}"
                f" written: the two targets contradict each other. Make them agree by changing ONE of the two: keep"
                f" the target {item['target']:.2f} and write the {anchor} at about {item['simulated']:.2f}, which is what"
                f" the frame will measure with it; only where this version's role needs the {anchor} at"
                f" {item['written']:.2f}, keep that and aim the region near {item['by_curve']:.2f} instead. Moving both"
                " moves the contradiction, it does not end it.")
    if item.get("kind") == "spread":
        return (f"- plan {slot}, mask \"{item['name']}\", box {item['box']}: image 1 runs from {item['p10']:.2f} to"
                f" {item['p90']:.2f} in the box (10th to 90th percentile), so the box holds the region and its"
                f" surroundings; its median {item['image']:.2f} is neither, and the solver measures only the part the"
                " mask moves most. Draw the box on the region alone (a face: the face, not the hat or sky around it)"
                " and aim the target at what that region measures.")
    up = item["target"] > item["image"]
    ev = item["ev"][1] if up else item["ev"][0]
    edge = 1 if up else 0
    text = (f"- plan {slot}, mask \"{item['name']}\", box {item['box']}: image 1 measures {item['image']:.2f} in the box;"
            f" this mask {'lifts' if up else 'darkens'} it at most {ev:+.2f} EV, to about {item['reach'][edge]:.2f}")
    if abs(item["by_curve"] - item["image"]) >= 0.02:
        text += (f". Your frame tone targets first take that brightness to about {item['by_curve']:.2f}, which leaves"
                 f" {item['reach_curve'][edge]:.2f}")
        if item.get("small"):
            text += " (a mask this small holds none of the frame's tones, so the curve moves the region with the frame)"
    if not item["precise"]:
        text += (f" (a bare linear or radial mask is held to {SOLVER_PLAIN_RANGE[1]:g} EV; with a luminance or color"
                 f" limit it goes to {SOLVER_LOCAL_RANGE[1]:g} EV)")
    return text + f". The target {item['target']:.2f} is out of reach."


def _plan_reach_fix(settings: Settings, selection: dict[str, Any], prompt: str, images: list[str],
                    audit: dict[str, Any], plans: dict[str, dict[str, Any]], notes: dict[str, list[str]],
                    reach: dict[str, list[dict[str, Any]]], baseline: Image.Image | Path, source_kind: str = "jpeg"
                    ) -> tuple[list[str], dict[str, list[dict[str, Any]]], list[str]]:
    """Ask once for the plans whose region targets image 1 shows out of reach or in conflict: the slots replaced,
    what each replacement still flags, and the slots whose answer was refused for flagging more than its plan did
    (that plan is kept; p24-r1 DSC_1963 B 1 flag -> 2, round 0 82; p27-r1 DSC_1515 B 1 -> 2, round 0 77)."""
    slots = [slot for slot in SLOTS if slot in reach]
    lines = "\n".join(_reach_line(slot, item, source_kind) for slot in slots for item in reach[slot])
    written = json.dumps({slot: _shown(plans[slot]) for slot in slots}, ensure_ascii=False)
    frame_rule = _REACH_FRAME_RULE["raw" if source_kind == "raw" else "jpeg"]
    try:
        data = _ask(settings, selection, prompt + _PLANS_REACH.format(lines=lines, slots=", ".join(slots),
                                                                      frame_rule=frame_rule, written=written),
                    images, audit)
    except (VisionUnavailable, ValueError):
        return [], {}, []
    raw = data.get("plans") if isinstance(data.get("plans"), dict) else data
    replaced: list[str] = []
    after: dict[str, list[dict[str, Any]]] = {}
    refused: list[str] = []
    for slot in slots:
        if not isinstance(raw.get(slot), dict):
            continue
        try:
            plan, note = sanitize_plan(raw[slot], slot=slot)
        except ValueError:
            continue
        if not is_executable(plan):
            continue
        found = plan_reach(plan, baseline)
        if len(found) > len(reach[slot]):
            refused.append(slot)
            continue
        plans[slot], notes[slot] = plan, note
        replaced.append(slot)
        after[slot] = found
    return replaced, after, refused


def plan_versions(settings: Settings, selection: dict[str, Any], baseline: Image.Image,
                  technical: Any, looks: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Ask the vision model to read the baseline and write three plans.

    ``looks`` (from ``measure_looks``) gives what each look does to this photo.
    """
    audit: dict[str, Any] = {}
    stats = picture_stats(baseline)
    prompt = plan_prompt(technical, baseline.size, stats, looks, settings.lightroom_candidate_min_distance)
    images = [_b64(baseline), _b64(grid_overlay(baseline))]
    plans: dict[str, dict[str, Any]] = {}
    notes: dict[str, list[str]] = {}
    data: dict[str, Any] = {}
    shape = ""
    for attempt in range(2):   # An answer cut short or shaped differently is asked for once more.
        try:
            data = _ask(settings, selection, prompt if attempt == 0 else prompt + _PLANS_AGAIN, images, audit)
        except (VisionUnavailable, ValueError) as exc:
            raise DirectedUnavailable(f"视觉模型没有给出可用的修图方案（{safe_error(exc)}）") from exc
        plans_raw = data.get("plans") if isinstance(data.get("plans"), dict) else {}
        if not plans_raw and isinstance(data.get("A"), dict):   # The versions written at the top level.
            plans_raw = {slot: data[slot] for slot in SLOTS if isinstance(data.get(slot), dict)}
        plans, notes = {}, {}
        for slot in SLOTS:
            if slot not in plans_raw:
                continue
            try:
                plans[slot], notes[slot] = sanitize_plan(plans_raw[slot], slot=slot)
            except ValueError as exc:
                notes[slot] = [str(exc)]
                continue
            if not is_executable(plans[slot]):
                notes[slot].append("方案没有任何可执行的调整")
                del plans[slot]
        if "A" in plans:
            break
        shape = f"顶层字段 {sorted(str(key) for key in data)[:8]}，plans 字段 {sorted(str(key) for key in plans_raw)[:6]}"
    if "A" not in plans:
        raise DirectedUnavailable(f"视觉模型的方案里没有可执行的 A 版本（{shape}）")
    missing = [slot for slot in SLOTS if slot not in plans]
    if missing:
        # An answer cut short after A (p13-r1 DSC_1441: twice cut, the repaired JSON held A alone, so B and C were
        # never made). The missing versions are asked for once on their own, next to what is already written.
        _plan_rest(settings, selection, prompt, images, audit, plans, notes, missing)
    # Region targets out of their mask's reach from image 1 are asked for once more with the numbers, and the
    # replaced plans that still have some once again: the re-asked plans redraw their boxes, which the first
    # numbers did not measure. tools/reach_after.py p24-p26: 6 of 13 re-asked slots were still flagged, all of
    # them DSC_1963 faces moved to a tight box (image 1 .17) and aimed .30-.36, beyond the .14-.26 a mask reaches
    # there after the curve (p26-r1 A0/C0 subject -5/-4, the mask at +1.25 EV). Since p28 (change 40) an answer
    # that flags more than the plan it would replace is refused and that plan kept (reach "refused"); a refused slot
    # is not asked again.
    reach: dict[str, Any] = {}
    source_kind = str(_measurements(technical).get("source_kind") or "jpeg")
    flagged = {slot: found for slot in SLOTS if slot in plans and (found := plan_reach(plans[slot], baseline))}
    if flagged:
        replaced, after, refused = _plan_reach_fix(settings, selection, prompt, images, audit, plans, notes,
                                                   flagged, baseline, source_kind)
        reach = {"flagged": flagged, "replaced": replaced, "after": after, "refused": refused}
        still = {slot: found for slot, found in after.items() if found}
        if still:
            again, after, refused = _plan_reach_fix(settings, selection, prompt, images, audit, plans, notes,
                                                    still, baseline, source_kind)
            reach["again"] = {"flagged": still, "replaced": again, "after": after, "refused": refused}
    provider = provider_from(selection, (audit.get("calls") or [{}])[-1])
    return {"reading": _reading(data.get("reading")), "plans": plans, "notes": notes, "baseline_stats": stats,
            "looks": dict(looks or {}), "reach": reach, "provider": provider,
            "temporary_provider": is_temporary(provider), "audit": audit, "prompt_chars": len(prompt)}


# Since p29 (change 41) a version whose first render is under the sibling distance from A's is planned once more
# before its first review, with both renders and the cells where the two look the same; the new plan is kept only
# when its render is farther from A's, it flags no more in plan_reach and it adds no blocking damage. p28-r1: 8
# deductions of 4-6 points (about 41) named a version under 5 from A (finish, and tone for a B without its own light
# structure); revisions, held by the review's "change only what your deductions name", moved DSC_2620 B 3.9 -> 4.3
# -> 5.8 (burns -.36/-.45 written first, -.80/-.85 by round 2) and DSC_1963 B to 4.8/4.9. Estimated before rendering
# (tools/apart_sim.py over p20-p27) the distance does not separate them: B corr .60 with the measured one, an
# estimate under 5 caught 10 of the 21 under 5 and named 9 that were not. So it is measured on the renders.
APART_FROM = "A"

_PLANS_APART = ('\nThe first plans have been rendered. Image 3 is version A\'s render; {images}. Measured on the renders'
                ' (mean Lab difference from A; below {limit:g} two versions read as nearly identical on a board, and the'
                ' review takes 5-8 points for it):\n{lines}\nReply with a JSON object holding only "plans" with the'
                ' key(s) {slots}, each the full plan in the same format, rewritten so that next to A it reads as its role'
                ' at a glance. Where the cell tables show near 0 the two look the same: that is where the version has to'
                ' become different, at about twice the strength of what it does there now. B: another light structure'
                ' (the surroundings burned with their own limited masks strongly enough to see, the subject separated,'
                ' tone targets apart from A\'s as its role says); C: another palette on the large areas (a colour target'
                ' on each large surface it recolours, the look at 0.8-1.0, white balance or calibration). Keep what'
                ' already differs, keep every region target within what its mask reaches from image 1, and keep the'
                ' version\'s role; version A stays as written. The plans were:\n{written}')


def _apart_line(slot: str, distance: float, mine: dict[str, Any], theirs: dict[str, Any]) -> str:
    ones = "; ".join(f"{name} {theirs['anchors'][name]:.3f} / {mine['anchors'][name]:.3f}" for name in tone.ORDER)
    luma = [[m - t for m, t in zip(row_m, row_t)] for row_m, row_t in zip(mine["luma"], theirs["luma"])]
    chroma = [[m - t for m, t in zip(row_m, row_t)] for row_m, row_t in zip(mine["chroma"], theirs["chroma"])]
    return (f"- plan {slot} renders {distance:.1f} from A. Tone percentiles A / {slot}: {ones}; frame colourfulness"
            f" x100 {_frame_chroma(theirs):.1f} / {_frame_chroma(mine):.1f}. Luma per cell, {slot} minus A (x100):\n"
            + tone.grid_text(luma, signed=True) + f"\n  Colourfulness per cell, {slot} minus A (x100):\n"
            + tone.grid_text(chroma, signed=True))


def _apart_fix(ctx: _Context, technical: Any, planning: dict[str, Any], firsts: list[dict[str, Any]]) -> dict[str, Any]:
    """Plan once more the versions whose first render is nearly identical to A's (see APART_FROM); ``firsts`` and
    ``planning`` are updated in place for each one kept. Returns what was measured and decided."""
    from .lightroom_qa import render_distance

    limit = ctx.settings.lightroom_candidate_min_distance
    by_slot = {item["slot"]: item for item in firsts}
    anchor = by_slot.get(APART_FROM)
    if anchor is None or anchor["round"] is None:
        return {}
    close = {slot: round(render_distance(by_slot[slot]["round"].render, anchor["round"].render), 2)
             for slot in SLOTS if slot != APART_FROM and slot in by_slot and by_slot[slot]["round"] is not None}
    close = {slot: distance for slot, distance in close.items() if distance < limit}
    if not close:
        return {}
    report: dict[str, Any] = {"close": close, "replaced": [], "refused": {}, "after": {}}
    renders = {slot: load_preview(by_slot[slot]["round"].render) for slot in [APART_FROM, *close]}
    stats = {slot: picture_stats(image) for slot, image in renders.items()}
    lines = "\n".join(_apart_line(slot, close[slot], stats[slot], stats[APART_FROM]) for slot in close)
    described = ", ".join(f"image {4 + index} is version {slot}'s" for index, slot in enumerate(close))
    written = json.dumps({slot: _shown(planning["plans"][slot]) for slot in SLOTS if slot in planning["plans"]},
                         ensure_ascii=False)
    prompt = plan_prompt(technical, ctx.baseline_image.size, planning.get("baseline_stats"), ctx.looks, limit)
    images = [_b64(ctx.baseline_image), _b64(grid_overlay(ctx.baseline_image)), *(_b64(renders[slot])
                                                                              for slot in [APART_FROM, *close])]
    try:
        data = _ask(ctx.settings, ctx.selection, prompt + _PLANS_APART.format(
            images=described, limit=limit, lines=lines, slots=", ".join(close), written=written), images,
            planning.setdefault("audit", {}))
    except (VisionUnavailable, ValueError):
        return report
    raw = data.get("plans") if isinstance(data.get("plans"), dict) else data
    candidates: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for slot in close:
        if not isinstance(raw.get(slot), dict):
            continue
        try:
            plan, note = sanitize_plan(raw[slot], slot=slot)
        except ValueError:
            continue
        if not is_executable(plan):
            continue
        before = len(plan_reach(planning["plans"][slot], ctx.baseline_image))
        found = plan_reach(plan, ctx.baseline_image)
        if len(found) > before:
            report["refused"][slot] = f"plan_reach {before} -> {len(found)}"
            continue
        candidates[slot] = (plan, note)

    def attempt(slot: str) -> tuple[str, dict[str, Any] | None, float | None]:
        folder = ctx.root / "directed" / slot
        parked = folder / "round-0-close"
        (folder / "round-0").rename(parked)
        item: dict[str, Any] = {"round": None}
        distance = None
        try:
            item = _first_round(ctx, slot, *candidates[slot])
            if item["round"] is not None:
                distance = round(render_distance(item["round"].render, anchor["round"].render), 2)
        except Exception:   # Whatever stops the new render, the first one stays.
            distance = None
        kept = (distance is not None and distance > close[slot]
                and (item["round"].gate["passed"] or not by_slot[slot]["round"].gate["passed"]))
        if not kept:
            if (folder / "round-0").exists():
                (folder / "round-0").rename(folder / "round-0-apart")
            parked.rename(folder / "round-0")
        return slot, item if kept else None, distance

    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        results = list(pool.map(attempt, candidates))
    for slot, item, distance in results:
        report["after"][slot] = distance
        if item is None:
            report["refused"][slot] = f"render {close[slot]} -> {distance}"
            continue
        firsts[firsts.index(by_slot[slot])] = item
        planning["plans"][slot], planning["notes"][slot] = candidates[slot]
        report["replaced"].append(slot)
    return report


_CUT_OFF = ("\nYour previous answer was cut off before the score. Keep each \"why\" under 80 characters and return"
            " the whole JSON object, score and plan included.")


def review_version(settings: Settings, selection: dict[str, Any], slot: str, plan: dict[str, Any],
                   reading: dict[str, Any], baseline: Image.Image, render: Image.Image, gate: dict[str, Any],
                   source_kind: str, measured: str = "", *, looks: dict[str, dict[str, float]] | None = None,
                   image_chroma: float | None = None) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    prompt = review_prompt(slot, plan, reading, gate, source_kind, measured, looks=looks, image_chroma=image_chroma,
                           min_distance=settings.lightroom_candidate_min_distance)
    images = [_board(baseline, render), _b64(mask_overlay(grid_overlay(render), plan))]
    try:
        data = _ask(settings, selection, prompt, images, audit)
        if _number(data.get("score")) is None and not data.get("verdict"):
            # A cut-off answer is closed after its last complete value and loses the score and the plan
            # that come after the deductions; ask once more rather than end the slot on half a review.
            try:
                again = _ask(settings, selection, prompt + _CUT_OFF, images, audit)
            except (VisionUnavailable, ValueError):
                again = {}
            if _number(again.get("score")) is not None or again.get("verdict"):
                data = again
    except (VisionUnavailable, ValueError) as exc:
        return {"verdict": "unavailable", "confidence": 0.0, "score": None, "deductions": [], "checks": [],
                "damage": [], "notes": [], "revise": False, "plan": None, "plan_notes": [],
                "error": safe_error(exc), "audit": audit}
    verdict = str(data.get("verdict") or "").strip().lower()
    confidence = _number(data.get("confidence"))
    damage = []
    for item in data.get("damage") or []:
        if isinstance(item, dict) and str(item.get("region") or "").strip():
            severity = _number(item.get("severity"))
            damage.append({"kind": str(item.get("kind") or "detail_loss")[:20], "region": str(item["region"])[:80],
                           "severity": round(min(1.0, max(0.0, 0.5 if severity is None else severity)), 2)})
    checks = [{"item": str(c.get("item") or "")[:40], "status": str(c.get("status") or "")[:12],
               "note": str(c.get("note") or "")[:160]}
              for c in (data.get("checks") or []) if isinstance(c, dict)][:10]
    deductions = []
    for item in data.get("deductions") or []:
        points = _number(item.get("points")) if isinstance(item, dict) else None
        if points is not None and points > 0:
            deductions.append({"area": str(item.get("area") or "")[:12], "points": round(min(40.0, points), 1),
                               "why": str(item.get("why") or "")[:200]})
    score = _number(data.get("score"))
    if score is not None:
        # The itemised deductions are the reasoning; a headline above them is not accepted.
        score = min(100.0, max(0.0, score))
        if deductions:
            score = min(score, max(0.0, 100.0 - sum(d["points"] for d in deductions)))
        score = round(score, 1)
    revised, plan_notes = None, []
    if data.get("revise") is True and isinstance(data.get("plan"), dict):
        try:
            revised, plan_notes = sanitize_plan(data["plan"], slot=slot)
        except ValueError as exc:
            plan_notes = [str(exc)]
        if revised is not None and not is_executable(revised):
            revised = None
    return {"verdict": verdict if verdict in {"better", "same", "worse"} else "same",
            "confidence": round(min(1.0, max(0.0, confidence or 0.0)), 2), "score": score,
            "deductions": deductions[:10], "checks": checks, "damage": damage[:6],
            "notes": [str(n)[:200] for n in (data.get("notes") or []) if str(n).strip()][:6],
            "revise": revised is not None, "plan": revised, "plan_notes": plan_notes, "error": None,
            "provider": provider_from(selection, (audit.get("calls") or [{}])[-1]), "audit": audit}


def revise_for_instruction(settings: Settings, selection: dict[str, Any], plan: dict[str, Any],
                           instruction: str, current: Image.Image, source_kind: str) -> tuple[dict[str, Any], list[str], str, dict]:
    audit: dict[str, Any] = {}
    data = _ask(settings, selection, refine_prompt(instruction, plan, source_kind),
                [_b64(grid_overlay(current))], audit)
    revised, notes = sanitize_plan(data.get("plan"), slot="R")
    revised["title"] = plan.get("title") or revised["title"]
    return revised, notes, str(data.get("summary") or "").strip()[:200], audit


# --------------------------------------------------------------------------
# Damage gate


def _colour_blowout(baseline: Path, render: Path) -> str | None:
    """Strong colour over a much larger area than in the baseline (max-min chroma, not HSV)."""
    if not (Path(baseline).is_file() and Path(render).is_file()):
        return None
    strong = [float(np.mean((rgb.max(axis=-1) - rgb.min(axis=-1)) > 0.62))
              for rgb in (tone.load_small(baseline, 480), tone.load_small(render, 480))]
    if strong[1] > max(strong[0] + 0.12, strong[0] * 2.0 + 0.04):
        return f"color_blowout: 强烈色彩的面积从 {strong[0]:.1%} 增加到 {strong[1]:.1%}"
    return None


def safety_gate(baseline: Path, render: Path, slot: str, source_kind: str = "jpeg") -> dict[str, Any]:
    from .lightroom_qa import assess_lightroom_render

    report = assess_lightroom_render(baseline, render, slot=slot, creative_palette=slot != "A",
                                     natural_color=slot == "A")
    hard, advisory = [], []
    # On a RAW the near-neutrals are the negative's: a move up to RAW_NEUTRAL_GATE times the gate is development,
    # and the reviewer is told so rather than "weaken the colour".
    cast = getattr(report, "neutral_assessment", None) or {}
    raw_neutral = source_kind == "raw" and neutral_nearness(cast, RAW_NEUTRAL_GATE) <= 1.0
    for issue in report.issues:
        if issue.code == "neutral_color_cast" and raw_neutral:
            advisory.append(f"{issue.code}: RAW 底片里近中性的亮部在显影中移动了，为自然色门槛的"
                            f" {neutral_nearness(cast):.1f} 倍（摄影师本人对这类 RAW 的成片为 1.0-2.2 倍）：这是显影，"
                            "不是偏色，判断颜色以场景本身为准")
        elif issue.advisory or not issue.hard or issue.code in ADVISORY_CODES:
            advisory.append(f"{issue.code}: {issue.note}")
        else:
            hard.append(f"{issue.code}: {issue.note}")
    ratio = report.metrics.mean_luma / max(report.baseline.mean_luma, 1e-4)
    lo, hi = RAW_LUMA_RATIO_BOUNDS if source_kind == "raw" else LUMA_RATIO_BOUNDS
    if not lo <= ratio <= hi:
        hard.append(f"extreme_brightness: 整体线性亮度为基线的 {ratio:.0%}（{math.log2(max(ratio, 1e-6)):+.1f} EV），"
                    f"超出 {lo:.0%}–{hi:.0%}")
    blowout = _colour_blowout(baseline, render)
    if blowout:
        hard.append(blowout)
    return {"passed": not hard, "hard": hard, "advisory": advisory, "luma_ratio": round(ratio, 3),
            "distance": round(report.metrics.perceptual_distance, 2), "report": report.to_dict()}


# --------------------------------------------------------------------------
# Candidate loop


@dataclass
class _Round:
    index: int
    plan: dict[str, Any]
    plan_notes: list[str]
    document: dict[str, Any]
    compile_notes: list[str]
    render: Path
    gate: dict[str, Any]
    solve: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None
    review: dict[str, Any] = field(default_factory=dict)

    def acceptable(self) -> bool:
        if not self.gate["passed"]:
            return False
        if self.review.get("verdict") == "worse" and self.review.get("confidence", 0) >= 0.7:
            return False
        return max((d["severity"] for d in self.review.get("damage", [])), default=0.0) < 0.65

    def score(self) -> float:
        """0..100. The reviewer's itemised score when it gave one, otherwise the verdict."""
        damage = max((d["severity"] for d in self.review.get("damage", [])), default=0.0)
        value = self.review.get("score")
        if value is None:
            points = {"better": 2.0, "same": 1.0, "worse": 0.0}.get(self.review.get("verdict"), 1.0)
            checks = self.review.get("checks") or []
            achieved = sum(c["status"] == "achieved" for c in checks) / len(checks) if checks else 0.0
            value = 40.0 + 20.0 * points + 10.0 * achieved
        return float(value) - 25.0 * damage

    def summary(self) -> dict[str, Any]:
        review = {key: value for key, value in self.review.items() if key not in {"audit", "plan"}}
        return {"round": self.index, "plan": self.plan, "plan_notes": self.plan_notes, "document": self.document,
                "compile_notes": self.compile_notes, "render": str(self.render), "gate": self.gate,
                "solve": self.solve, "review": review, "revised_plan": self.review.get("plan")}


@dataclass
class _Context:
    settings: Settings
    selection: dict[str, Any]
    binary: Path
    root: Path
    original: Path
    baseline_export: Path
    baseline_image: Image.Image
    reading: dict[str, Any]
    source_kind: str
    baseline_stats: dict[str, Any] | None = None
    colors: Callable[[list[dict[str, float]]], list[list[float]]] | None = None
    looks: dict[str, dict[str, float]] = field(default_factory=dict)   # measure_looks: what each look does here


def color_sampler(binary: Path, source: Path) -> Callable[[list[dict[str, float]]], list[list[float]]]:
    """Engine colour samples at normalized points of the unedited photo, cached per point."""
    import threading

    from .lightcraft import sample_colors

    cache: dict[tuple[float, float], list[float]] = {}
    lock = threading.Lock()

    def colors(points: list[dict[str, float]]) -> list[list[float]]:
        keys = [(round(point["x"], 3), round(point["y"], 3)) for point in points]
        with lock:
            missing = [key for key in dict.fromkeys(keys) if key not in cache]
            if missing:
                for key, sample in zip(missing, sample_colors(binary, source, missing)):
                    cache[key] = sample
            return [cache[key] for key in keys]

    return colors


def _aspect_of(image: Image.Image | Path) -> float | None:
    """Width / height of a picture as shown (EXIF orientation applied), or None when it cannot be read."""
    try:
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            with Image.open(image) as opened:
                width, height = ImageOps.exif_transpose(opened).size
    except (OSError, ValueError):
        return None
    return width / height if width > 0 and height > 0 else None


def _render_plan(binary: Path, source: Path, plan: dict[str, Any], source_kind: str, out_dir: Path, *,
                 edge: int, colors: Callable | None, final_name: str | None = None,
                 baseline: dict[str, float] | None = None, neutral_reference: Path | None = None,
                 image: Image.Image | Path | None = None, aspect: float | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any] | None, Path]:
    """Solve a plan with small probe renders, then render it at delivery size.

    ``baseline`` (percentiles of the unedited photo) bounds how far the tone targets may move.
    ``neutral_reference`` (the unedited render, for a natural-colour version) keeps the colour moves
    off the near-neutral cast gate. ``image`` (image 1), given, lets ``guard_spill`` narrow a lifting
    mask's luminance limit first. ``aspect`` (width / height) sizes radials; it defaults to image 1's.
    """
    from .lightcraft import render_document

    if aspect is None and image is not None:
        aspect = _aspect_of(image)
    count = {"probes": 0, "spill": 0}

    def renderer(folder: str) -> Callable[[dict[str, Any]], Path]:
        def probe_render(document: dict[str, Any]) -> Path:
            count[folder] += 1
            return render_document(binary, source, document, out_dir / folder / f"{count[folder]:02d}", edge=PROBE_EDGE)
        return probe_render

    spill_notes: list[str] = []
    spill: list[dict[str, Any]] = []
    if image is not None:
        plan, spill_notes, spill = guard_spill(plan, source_kind, renderer("spill"), image, colors=colors)
    neutral = (neutral_guard(neutral_reference, RAW_NEUTRAL_GATE if source_kind == "raw" else 1.0)
               if neutral_reference is not None and neutral_reference.is_file() else None)
    executed, document, notes, solve = solve_plan(plan, source_kind, renderer("probes"), colors=colors,
                                                  baseline=baseline, neutral=neutral, aspect=aspect)
    if spill and solve is not None:
        solve["spill"] = spill
    render = render_document(binary, source, document, out_dir / final_name if final_name else out_dir, edge=edge)
    return executed, document, spill_notes + notes, solve, render


def _first_round(ctx: _Context, slot: str, plan: dict[str, Any], plan_notes: list[str]) -> dict[str, Any]:
    """Render a slot's first plan. Returns the round, or the error that stopped it."""
    from .lightcraft import LightCraftError

    folder = ctx.root / "directed" / slot
    source = folder / f"source{ctx.original.suffix.lower()}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ctx.original, source)
    try:
        current = _make_round(ctx, slot, 0, plan, plan_notes, source)
    except LightCraftError as exc:
        return {"slot": slot, "source": source, "round": None, "error": str(exc)}
    return {"slot": slot, "source": source, "round": current, "error": None}


def _make_round(ctx: _Context, slot: str, index: int, plan: dict[str, Any], plan_notes: list[str],
                source: Path) -> _Round:
    out_dir = ctx.root / "directed" / slot / f"round-{index}"
    executed, document, notes, solve, render = _render_plan(
        ctx.binary, source, plan, ctx.source_kind, out_dir, edge=ctx.settings.lightroom_export_edge, colors=ctx.colors,
        baseline=(ctx.baseline_stats or {}).get("anchors"),
        neutral_reference=ctx.baseline_export if slot == "A" else None,   # The gate's natural_color versions.
        image=ctx.baseline_image)
    gate = safety_gate(ctx.baseline_export, render, slot, ctx.source_kind)
    return _Round(index, executed, plan_notes, document, notes, render, gate, solve=solve)


def _run_slot(ctx: _Context, first: dict[str, Any], siblings: dict[str, Path]) -> dict[str, Any]:
    """Review, revise and re-render one slot, starting from its first render.

    ``siblings`` are the first renders of the other versions; every round of this slot is measured
    against them, so a revision that moves towards or away from a sibling shows in the numbers.
    """
    from .lightcraft import LightCraftError
    from .lightroom_qa import render_distance

    slot, source = first["slot"], first["source"]
    rounds: list[_Round] = []
    error = first["error"]
    current: _Round | None = first["round"]
    sibling_stats = ({other: picture_stats(load_preview(path)) for other, path in siblings.items()}
                     if slot != "A" else None)
    while current is not None:
        render_image = load_preview(current.render)
        current.stats = picture_stats(render_image)
        distances = ({other: round(render_distance(current.render, path), 2) for other, path in siblings.items()}
                     if slot != "A" else None)
        measured = measured_block(current.solve, ctx.baseline_stats, current.stats,
                                  distances, ctx.settings.lightroom_candidate_min_distance, sibling_stats,
                                  look=current.plan.get("look"), looks=ctx.looks)
        current.review = review_version(ctx.settings, ctx.selection, slot, current.plan, ctx.reading,
                                        ctx.baseline_image, render_image, current.gate, ctx.source_kind, measured,
                                        looks=ctx.looks, image_chroma=_frame_chroma(ctx.baseline_stats)
                                        if ctx.baseline_stats else None)
        (current.render.parent / "review.json").write_text(
            json.dumps(current.summary() | {"audit": current.review.get("audit"), "measured": measured},
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        rounds.append(current)
        revised = current.review.get("plan")
        finished = (current.gate["passed"] and current.acceptable()
                    and (current.review.get("score") or 0.0) >= DELIVERY_SCORE)
        if (current.index >= ctx.settings.directed_max_revisions or revised is None or revised == current.plan
                or finished):
            break
        try:
            current = _make_round(ctx, slot, current.index + 1, revised, current.review.get("plan_notes") or [], source)
        except LightCraftError as exc:
            error = str(exc)
            break
    acceptable = [r for r in rounds if r.acceptable()]
    if not acceptable:
        reasons = error or "；".join(dict.fromkeys(
            [*(h for r in rounds for h in r.gate["hard"]),
             *(f"{d['region']}：{d['kind']}" for r in rounds for d in r.review.get("damage", []) if d["severity"] >= .65),
             *("视觉复核判定变差" for r in rounds if r.review.get("verdict") == "worse")])) or "未通过成片检查"
        return {"slot": slot, "status": "withheld", "reason": reasons, "rounds": [r.summary() for r in rounds]}
    best = max(acceptable, key=lambda r: (r.score(), r.index))
    return {"slot": slot, "status": "delivered", "best": best, "rounds": [r.summary() for r in rounds],
            "source": source, "error": error}


def directed_selection(settings: Settings) -> dict[str, Any] | None:
    """Vision selection for the directed path, or None when it cannot run."""
    if settings.pipeline_mode == "legacy" or not settings.lightroom_enabled:
        return None
    status = probe(settings)
    if status.get("ready") is not True or status.get("provider") == "local" or not status.get("resolved_model"):
        return None
    return status


def _critique(planning: dict[str, Any], selection: dict[str, Any]):
    from .models import Critique, EditIntent, Issue, RegionBox

    reading = planning["reading"]
    provider = planning.get("provider") or selection.get("provider") or "local"
    source = provider if provider in {"ollama", "openai", "xai", "vault"} else "local"
    box = reading.get("subject_box")
    intent = EditIntent(goal=planning["plans"]["A"].get("intent") or reading.get("scene") or EditIntent().goal,
                        preserve=reading.get("preserve", [])[:6],
                        subject_box=RegionBox(left=box[0], top=box[1], right=box[2], bottom=box[3], kind="subject",
                                              note=reading.get("subject", "")) if box else None,
                        subject_confidence=reading.get("subject_confidence", 0.0))
    issues = [Issue(dim="visual", severity=p["severity"], note=(f"{p['region']}：" if p["region"] else "") + p["issue"])
              for p in reading.get("problems", [])[:6]]
    summary = "；".join(x for x in (reading.get("scene"), reading.get("light")) if x)
    return Critique(genre=reading.get("genre", "general"), summary=summary, issues=issues, source=source,
                    vision_provider=provider, transport_audit={"pipeline": ENGINE}, subject_note=reading.get("subject", ""),
                    intent=intent, looks=[])


def run_directed_job(settings: Settings, conn: sqlite3.Connection, job_id: str, original: Path, root: Path, *,
                     technical: Any, baseline_export: Path, selection: dict[str, Any]) -> dict[str, Any]:
    """Plan, render, review and deliver A/B/C. Raises DirectedUnavailable when nothing can be delivered."""
    from .db import replace_variants, update_job
    from .lightroom import connect_lightroom
    from .lightroom_qa import render_distance
    from .models import EditIntent, RegionBox

    update_job(conn, job_id, step="critique")
    client, _status = connect_lightroom(settings.lightcraft_bin)
    binary = Path(client.binary)
    source_kind = str(getattr(technical, "source_kind", "jpeg"))
    baseline_image = load_preview(baseline_export)
    directed_root = root / "directed"
    looks = measure_looks(binary, original, directed_root / "looks", source_kind)
    planning = plan_versions(settings, selection, baseline_image, technical, looks)
    directed_root.mkdir(parents=True, exist_ok=True)
    (directed_root / "plan.json").write_text(json.dumps(planning, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    critique = _critique(planning, selection)
    (root / "critique.json").write_text(json.dumps(critique.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    update_job(conn, job_id, step="lightroom", critique_json=critique.model_dump_json())

    ctx = _Context(settings, selection, binary, root, original, baseline_export, baseline_image,
                   planning["reading"], source_kind, baseline_stats=planning.get("baseline_stats"),
                   colors=color_sampler(binary, original), looks=looks)
    slots = [slot for slot in SLOTS if slot in planning["plans"]]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        # First renders of all versions, so each reviewer knows how far its version is from the others.
        firsts = list(pool.map(lambda slot: _first_round(ctx, slot, planning["plans"][slot],
                                                         planning["notes"].get(slot, [])), slots))
        apart = _apart_fix(ctx, technical, planning, firsts)
        if apart:
            planning["apart"] = apart
            (directed_root / "plan.json").write_text(json.dumps(planning, ensure_ascii=False, indent=2, default=str),
                                                     encoding="utf-8")
        rendered ={item["slot"]: item["round"].render for item in firsts if item["round"] is not None}
        results = list(pool.map(lambda item: _run_slot(
            ctx, item, {other: path for other, path in rendered.items() if other != item["slot"]}), firsts))

    audit_root = root / "lightroom-audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    rows, omitted, summary_slots, finals = [], [], {}, {}
    reading = planning["reading"]
    for result in results:
        slot = result["slot"]
        if result["status"] != "delivered":
            omitted.append({"slot": slot, "reason": result["reason"]})
            summary_slots[slot] = {"status": "withheld", "reason": result["reason"], "rounds": len(result["rounds"])}
            (audit_root / f"variant_{slot}.json").write_text(json.dumps(
                {"engine": ENGINE, "slot": slot, "status": "failed", "reason": result["reason"],
                 "rounds": result["rounds"]}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            continue
        finals[slot] = result
    for slot, result in finals.items():
        best: _Round = result["best"]
        others = [finals[other]["best"].render for other in finals if other != slot]
        nearest = min((render_distance(best.render, other) for other in others), default=None)
        final_path = root / f"variant_{slot}.jpg"
        shutil.copy2(best.render, final_path)
        box = reading.get("subject_box")
        intent = EditIntent(goal=best.plan.get("intent") or reading.get("scene") or EditIntent().goal,
                            preserve=reading.get("preserve", [])[:6], natural_color=slot == "A",
                            subject_box=RegionBox(left=box[0], top=box[1], right=box[2], bottom=box[3], kind="subject")
                            if box else None, subject_confidence=reading.get("subject_confidence", 0.0))
        review = best.review
        moves = describe_plan(best.plan)
        selection_info = {"distinct": nearest is None or nearest >= settings.lightroom_candidate_min_distance,
                          "nearest_distance": nearest, "target_distance": settings.lightroom_candidate_min_distance,
                          "selected_round": best.index}
        sidecar = {
            "engine": ENGINE, "source_kind": source_kind, "plan": best.plan, "develop": best.document,
            "intent": intent.model_dump(),
            "style_workflow": {"method_id": "", "preset_id": (best.plan.get("look") or {}).get("id"),
                               "mode": "art-directed",
                               "route_label": best.plan["title"], "selection_reason": best.plan.get("intent", ""),
                               "adaptation_notes": moves, "manual_gaps": []},
            "lightroom_pixel_qa": best.gate["report"],
            "visual_review": {"decision": review.get("verdict", "unavailable"), "confidence": review.get("confidence", 0.0),
                              "score": review.get("score"), "deductions": review.get("deductions", []),
                              "notes": review.get("notes", []), "source": review.get("provider") or planning.get("provider"),
                              "checks": review.get("checks", []), "damage": review.get("damage", [])},
            "directed": {"selected_round": best.index, "rounds": len(result["rounds"]),
                         "compile_notes": best.compile_notes, "plan_notes": best.plan_notes,
                         "gate_advisory": best.gate["advisory"], "tone": best.solve,
                         "vision_provider": planning.get("provider"),
                         "temporary_provider": planning.get("temporary_provider", False)},
            "candidate_selection": selection_info,
        }
        (root / f"variant_{slot}.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        (audit_root / f"variant_{slot}.json").write_text(json.dumps(
            {"engine": ENGINE, "slot": slot, "status": "delivered", "source_kind": source_kind,
             "baseline_export": str(baseline_export), "accepted_export": str(best.render),
             "accepted_photo": str(result["source"]), "plan": best.plan, "document": best.document,
             "selected_round": best.index, "candidate_selection": selection_info, "rounds": result["rounds"]},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        verdict = {"better": "视觉复核：有改善", "same": "视觉复核：变化有限", "worse": "视觉复核：有保留意见"}.get(
            review.get("verdict"), "视觉复核不可用，仅完成技术检查")
        why = f"{best.plan.get('intent') or best.plan['title']} {verdict}。"
        if not selection_info["distinct"]:
            why += " 与其他方案的差别较细微。"
        rows.append({"id": str(uuid.uuid4()), "slot": slot, "look_id": "directed", "title": best.plan["title"],
                     "why": why.strip(), "recipe_json": json.dumps(sidecar, ensure_ascii=False),
                     "filename": final_path.name})
        summary_slots[slot] = {"status": "delivered", "score": best.gate["report"]["score"], "rounds": len(result["rounds"]),
                               "selected_round": best.index, "issues": best.gate["hard"] + best.gate["advisory"],
                               "review": review.get("verdict"), "review_score": review.get("score"),
                               "candidate_selection": selection_info, "final": str(final_path)}
    if not rows:
        raise DirectedUnavailable("视觉定向的三个版本都没有通过检查：" + "；".join(o["reason"] for o in omitted))
    rows.sort(key=lambda row: row["slot"])
    replace_variants(conn, job_id, rows)
    similar = [slot for slot, item in summary_slots.items()
               if item.get("status") == "delivered" and not item["candidate_selection"]["distinct"]]
    delivery = {"requested": len(slots), "delivered": len(rows), "omitted": omitted,
                "distinct_delivered": len(rows) - len(similar), "similar_slots": similar}
    temporary = planning.get("temporary_provider", False)
    summary = {"pipeline": ENGINE, "baseline": str(baseline_export), "vision_provider": planning.get("provider"),
               "vision_model": selection.get("resolved_model"), "temporary_provider": temporary,
               "max_revisions": settings.directed_max_revisions,
               "slots": summary_slots, "delivery": delivery}
    (root / "lightroom-qa.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    workflow = {"pipeline": ENGINE, "vision_provider": planning.get("provider"),
                "vision_model": selection.get("resolved_model"), "temporary_provider": temporary,
                "axis_order": ["baseline", "measure", "vision_read", "plan", "tone_solve", "render", "damage_gate",
                               "vision_review", "deliver"]}
    (root / "workflow.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# User refinement of a directed finish


def refinement_source(settings: Settings, conn: sqlite3.Connection, job_id: str, slot: str,
                      parent_id: str | None) -> dict[str, Any] | None:
    """Return the image and plan to refine when the finish came from this path."""
    from .db import get_refinement, get_variants

    root = settings.jobs_dir / job_id
    if parent_id:
        parent = get_refinement(conn, parent_id)
        if not parent or parent["job_id"] != job_id or parent["slot"] != slot or parent["status"] != "ready":
            return None
        try:
            data = json.loads(parent["recipe_json"] or "{}")
        except json.JSONDecodeError:
            return None
        if data.get("engine") != ENGINE:
            return None
        return {"image": root / "refinements" / parent_id / "result.jpg", "plan": data["plan"],
                "source_kind": data.get("source_kind", "jpeg")}
    audit_path = root / "lightroom-audits" / f"variant_{slot}.json"
    if not audit_path.is_file():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("engine") != ENGINE or audit.get("status") != "delivered":
        return None
    variant = next((v for v in get_variants(conn, job_id) if v["slot"] == slot), None)
    if variant is None:
        return None
    return {"image": root / variant["filename"], "plan": audit["plan"], "source_kind": audit.get("source_kind", "jpeg")}


def run_refinement(settings: Settings, conn: sqlite3.Connection, refinement_id: str, record: dict[str, Any],
                   folder: Path, base: dict[str, Any], audit: dict[str, Any]) -> None:
    """Revise a directed plan from a user note and render it on a pristine copy."""
    from .db import get_job, update_refinement
    from .lightroom import LightroomError, connect_lightroom
    from .lightroom_qa import render_distance

    before = Path(base["image"])
    if not before.is_file():
        raise ValueError("调整前的成片已不在本机")
    selection = probe(settings)
    if selection.get("ready") is not True or not selection.get("resolved_model"):
        raise LightroomError("这张成片由视觉定向流程生成，二次微调需要视觉模型；当前视觉模型不可用。")
    update_refinement(conn, refinement_id, step="interpret")
    try:
        plan, notes, summary, vision_audit = revise_for_instruction(
            settings, selection, base["plan"], record["prompt"], load_preview(before), base["source_kind"])
    except (VisionUnavailable, ValueError) as exc:
        raise LightroomError(f"视觉模型没有给出可执行的微调方案（{safe_error(exc)}），调整前版本已保留。") from exc
    if not is_executable(plan):
        raise LightroomError("视觉模型返回的微调方案没有任何可执行的调整，调整前版本已保留。")
    update_refinement(conn, refinement_id, step="lightroom", summary=summary or "按要求修改")
    job = get_job(conn, record["job_id"])
    original = settings.jobs_dir / record["job_id"] / Path(job["original_file"] or "original.jpg").name
    if not original.is_file():
        raise ValueError("原片已不在本机，无法继续微调。")
    source = folder / f"source{original.suffix.lower()}"
    shutil.copy2(original, source)
    client, _status = connect_lightroom(settings.lightcraft_bin)
    binary = Path(client.binary)
    plan, document, compile_notes, solve, output = _render_plan(
        binary, source, plan, base["source_kind"], folder, edge=settings.lightroom_export_edge,
        colors=color_sampler(binary, source), final_name="exports", aspect=_aspect_of(before))
    audit.update(engine=ENGINE, base_image=str(before), base_plan=base["plan"], plan=plan, plan_notes=notes,
                 compile_notes=compile_notes, document=document, tone=solve, vision=vision_audit)
    update_refinement(conn, refinement_id, step="qa")
    gate = safety_gate(before, output, "B")
    audit["quality"] = gate
    if not gate["passed"]:
        raise LightroomError("微调版未通过画质检查，调整前版本已保留。" + "；".join(gate["hard"]))
    if render_distance(before, output) < .12:
        raise LightroomError("输出没有可见变化，调整前版本已保留。请把希望改变的地方说得更具体。")
    shutil.copy2(output, folder / "result.jpg")
    update_refinement(conn, refinement_id, status="ready", step="ready", error=None, summary=summary or "按要求修改",
                      recipe_json=json.dumps({"engine": ENGINE, "plan": plan, "develop": document,
                                              "source_kind": base["source_kind"]}, ensure_ascii=False),
                      filename="result.jpg")
