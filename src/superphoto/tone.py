"""Measured tone placement for the vision-directed path.

A vision model is good at saying where tones should sit ("blacks near 0.03,
midtones around 0.45") and poor at guessing which slider values produce that
in a particular engine. This module measures a render and builds LightCraft's
master point curve so that chosen luminance percentiles land on chosen values.

All values are display-referred sRGB luma in 0..1, measured on a small copy.
The curve is applied by the engine after sliders and masks (checked on
LightCraft 0.21.1), so targets describe the finished picture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageOps

# Name -> luminance percentile. Black and white ignore the outer 1 %, so a few
# specular pixels or a dark frame edge do not define the range.
ANCHORS: dict[str, float] = {"black": 1.0, "shadows": 10.0, "mid": 50.0, "highlights": 90.0, "white": 99.0}
ORDER = tuple(ANCHORS)
# A point curve steeper than this turns 8-bit gradients into bands and lifts noise;
# flatter than the minimum leaves a tonal range without separation.
MAX_SLOPE = 3.2
MIN_SLOPE = 0.22


def load_small(source: Path | Image.Image, edge: int = 640) -> np.ndarray:
    """RGB float array in 0..1, long edge at most ``edge``."""
    if isinstance(source, Image.Image):
        image = source.convert("RGB")
    else:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((edge, edge), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def luma_of(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def measure(source: Path | Image.Image | np.ndarray) -> dict[str, float]:
    """Luma at each anchor percentile."""
    rgb = source if isinstance(source, np.ndarray) else load_small(source)
    values = np.percentile(luma_of(rgb), [ANCHORS[name] for name in ORDER])
    return {name: round(float(value), 4) for name, value in zip(ORDER, values)}


def _span(height: int, width: int, box: Sequence[float]) -> tuple[int, int, int, int]:
    """Pixel rows and columns (y0, y1, x0, x1) of a normalized box, at least one pixel each way."""
    left, top, right, bottom = (min(1.0, max(0.0, float(v))) for v in box)
    x0 = min(width - 1, int(left * width))
    y0 = min(height - 1, int(top * height))
    return y0, max(y0 + 1, int(round(bottom * height))), x0, max(x0 + 1, int(round(right * width)))


def _crop(rgb: np.ndarray, box: Sequence[float] | None) -> np.ndarray:
    if not box:
        return rgb
    y0, y1, x0, x1 = _span(rgb.shape[0], rgb.shape[1], box)
    return rgb[y0:y1, x0:x1]


def region_stats(source: Path | Image.Image | np.ndarray, box: list[float] | None = None) -> dict[str, float]:
    """Median luma, mean chroma and clipping inside a normalized box (whole frame without one)."""
    rgb = _crop(source if isinstance(source, np.ndarray) else load_small(source), box)
    luma = luma_of(rgb)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    return {"median": round(float(np.median(luma)), 4), "p10": round(float(np.percentile(luma, 10)), 4),
            "p90": round(float(np.percentile(luma, 90)), 4), "chroma": round(float(np.mean(chroma)), 4),
            "white_clip": round(float(np.mean(rgb.min(axis=-1) > 0.985)), 5),
            "black_clip": round(float(np.mean(rgb.max(axis=-1) < 0.012)), 5)}


# Colour as a vector on the colour wheel: hue in degrees the way a grading wheel counts it (0 red,
# 60 yellow, 120 green, 180 cyan, 240 blue, 300 magenta) and a length that is close to
# max(R,G,B) - min(R,G,B). Both components are linear in RGB, which is what a solver needs.


def colour_vector(rgb: np.ndarray) -> np.ndarray:
    """Mean colour of pixels as (alpha, beta); its angle is the hue, its length the colourfulness."""
    mean = np.asarray(rgb, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    return np.array([mean[0] - (mean[1] + mean[2]) / 2.0, (3.0 ** 0.5 / 2.0) * (mean[1] - mean[2])])


def colour_polar(vector: np.ndarray) -> tuple[float, float]:
    """(hue in degrees 0..360, colourfulness 0..1) of a colour vector."""
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0), float(np.hypot(vector[0], vector[1]))


def colour_target(hue: float, chroma: float) -> np.ndarray:
    angle = np.radians(float(hue) % 360.0)
    return float(chroma) * np.array([np.cos(angle), np.sin(angle)])


def region_colour(source: Path | Image.Image | np.ndarray, box: list[float] | None = None) -> np.ndarray:
    return colour_vector(_crop(source if isinstance(source, np.ndarray) else load_small(source), box))


def clean_targets(raw: Any, notes: list[str] | None = None, path: str = "tone") -> dict[str, float]:
    """Keep known anchors with finite values in 0..1 and an increasing order."""
    notes = notes if notes is not None else []
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        notes.append(f"{path}: 不是对象，已忽略")
        return {}
    out: dict[str, float] = {}
    for name, value in raw.items():
        if name not in ANCHORS:
            if name not in {"saturation", "chroma", "colourfulness"}:   # Read by the caller.
                notes.append(f"{path}.{name}: 不支持，已忽略")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            notes.append(f"{path}.{name}: 不是数值，已忽略")
            continue
        if not np.isfinite(number) or isinstance(value, bool):
            notes.append(f"{path}.{name}: 不是数值，已忽略")
            continue
        if number > 1.0:  # A 0..255 or percent answer.
            number = number / 255.0 if number > 100.0 else number / 100.0
        out[name] = round(min(0.995, max(0.0, number)), 4)
    previous = -1.0
    for name in ORDER:
        if name not in out:
            continue
        if out[name] <= previous + 0.004:
            notes.append(f"{path}.{name}: {out[name]:g} 不高于更暗一档的目标，已忽略")
            del out[name]
            continue
        previous = out[name]
    return out


# Knots of the fitted curve: denser in the shadows, where night scenes keep three anchors below 0.05.
KNOTS = 24
_KNOT_X = np.linspace(0.0, 1.0, KNOTS + 1) ** 1.6
# How strongly each anchor pulls. The median sets how bright the picture reads. Black was 0.8 until
# p5-r1, where the fit left it 0.01-0.04 above target in 20 of 27 plans and every review read that as
# a missing dark anchor; refitting those plans at 1.4 halved its mean miss (0.011 -> 0.005).
_ANCHOR_WEIGHT = {"black": 1.4, "shadows": 1.0, "mid": 1.4, "highlights": 1.0, "white": 1.0}
# Penalty on a change of slope between neighbouring knots. Targets that a smooth
# curve can reach are met within about 0.01; targets that would need a kink are
# shared out between their neighbours. Lowering it to 1.5e-4 (p6-r1) met every anchor more closely on
# paper but let the top of a bright-sand curve bend up to the white pin: the 99th percentile landed at
# 0.996 instead of 0.955 and the version failed the clipping gate.
SMOOTHNESS = 4e-4
LIFTED_BLACK = 0.25   # cost of an output black above zero (a matte)
END_AT_WHITE = 6.0    # how firmly input white stays output white
REPORT_MISS = 0.025


def _bounded_lstsq(matrix: np.ndarray, rhs: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   iterations: int = 12000) -> np.ndarray:
    """Least squares with box bounds (accelerated projected gradient with restarts)."""
    hessian = matrix.T @ matrix
    gradient0 = matrix.T @ rhs
    step = 1.0 / (float(np.linalg.eigvalsh(hessian)[-1]) + 1e-12)
    value = np.clip(np.linalg.lstsq(matrix, rhs, rcond=None)[0], lo, hi)
    ahead, momentum = value.copy(), 1.0
    for _ in range(iterations):
        nxt = np.clip(ahead - step * (hessian @ ahead - gradient0), lo, hi)
        moved = nxt - value
        if float(np.max(np.abs(moved))) < 1e-10:
            value = nxt
            break
        if float(np.dot(ahead - nxt, moved)) > 0.0:   # Overshoot: drop the momentum.
            ahead, momentum = nxt, 1.0
        else:
            following = (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5) / 2.0
            ahead = nxt + ((momentum - 1.0) / following) * moved
            momentum = following
        value = nxt
    return value


def fit_curve(measured: dict[str, float], targets: dict[str, float], *,
              max_slope: float = MAX_SLOPE) -> tuple[list[list[float]], dict[str, float]]:
    """Smooth master curve that takes the measured anchors as close to their targets as it cleanly can.

    The curve is piecewise linear on fixed knots with every slope inside
    MIN_SLOPE..MAX_SLOPE and a penalty on slope changes, so it cannot kink or
    posterize however the targets are written. Returns the curve and the value
    it gives each anchor ("executed"), which is what the render will show.
    """
    names = [name for name in ORDER if name in targets and name in measured]
    if not names:
        return [], {}
    steps = np.diff(_KNOT_X)
    size = KNOTS + 1                       # y0 and one rise per interval
    cumulative = np.tril(np.ones((size, size)))

    def at(value: float) -> np.ndarray:
        x = min(1.0, max(0.0, float(value)))
        index = int(np.clip(np.searchsorted(_KNOT_X, x, side="right") - 1, 0, KNOTS - 1))
        share = (x - _KNOT_X[index]) / steps[index]
        basis = np.zeros(size)
        basis[index], basis[index + 1] = 1.0 - share, share
        return basis @ cumulative

    rows, rhs = [], []
    for name in names:
        weight = _ANCHOR_WEIGHT.get(name, 1.0)
        rows.append(weight * at(measured[name]))
        rhs.append(weight * float(targets[name]))
    for index in range(KNOTS - 1):         # slope[index + 1] - slope[index]
        row = np.zeros(size)
        row[index + 1], row[index + 2] = -1.0 / steps[index], 1.0 / steps[index + 1]
        rows.append(SMOOTHNESS ** 0.5 * row)
        rhs.append(0.0)
    black = np.zeros(size)
    black[0] = LIFTED_BLACK ** 0.5
    rows.append(black)
    rhs.append(0.0)
    rows.append(END_AT_WHITE ** 0.5 * cumulative[-1])
    rhs.append(END_AT_WHITE ** 0.5)
    # Separation only matters where the picture has pixels: below the black anchor and above the
    # white anchor the curve may run nearly flat, as a levels adjustment does.
    populated = (_KNOT_X[1:] > float(measured[names[0]])) & (_KNOT_X[:-1] < float(measured[names[-1]]))
    lo = np.concatenate([[0.0], np.where(populated, MIN_SLOPE, 0.02) * steps])
    hi = np.concatenate([[0.35], float(max_slope) * steps])
    solution = _bounded_lstsq(np.asarray(rows), np.asarray(rhs), lo, hi)
    values = np.clip(cumulative @ solution, 0.0, 1.0)
    curve = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(_KNOT_X, values)]
    executed = {name: round(curve_value(curve, float(measured[name])), 4) for name in names}
    return curve, executed


def build_curve(measured: dict[str, float], targets: dict[str, float]) -> tuple[list[list[float]], list[str]]:
    """Master point curve for the targets, with notes on targets it could not reach cleanly."""
    curve, executed = fit_curve(measured, targets)
    notes = [f"tone.{name}: 目标 {targets[name]:.2f} 与相邻影调冲突，平滑曲线按 {executed[name]:.2f} 执行"
             for name in executed if abs(executed[name] - targets[name]) > REPORT_MISS]
    return curve, notes


def _increasing(values: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    previous = -1.0
    for name in ORDER:
        if name in values:
            out[name] = max(values[name], previous + 0.005)
            previous = out[name]
    return out


# --------------------------------------------------------------------------
# Numbers a vision model can read, and a closed loop over exposure, local
# exposures and the master curve.


def to_linear(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def to_display(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def ev_between(measured: float, wanted: float) -> float:
    """Exposure change in EV that takes one display luma to another (linear light)."""
    return float(np.log2(max(to_linear(wanted), 1e-4) / max(to_linear(measured), 1e-4)))


# How far (EV, linear light) a percentile target may move from the unedited photo. Past this a scene
# changes what it is: a night reads as dusk, lifted shadows show noise and blocking, and the frame
# goes past the brightness the delivery gate allows. Black is free: a crushed black may be lifted.
SHIFT_LIMITS: dict[str, tuple[float, float]] = {"shadows": (-3.0, 1.6), "mid": (-1.5, 1.6),
                                                "highlights": (-1.5, 2.0), "white": (-1.5, 2.0)}
# A RAW's unedited render is an undeveloped negative, not a finished picture: LightCraft renders it 0.7-1.2 EV
# under Adobe's default, flat and grey. The photographer's own finished edits of his RAWs (resources/, the
# reference benchmark of 2026-11-21) sit up to +2.0 EV (snowfield DSC02262) and +3.1 EV (dune DSC05752, mid .26 ->
# .71) above that render, and the solver held them at +1.6 EV: snow came out at mid .45 against his .75.
RAW_SHIFT_LIMITS: dict[str, tuple[float, float]] = {"shadows": (-3.0, 3.0), "mid": (-1.5, 3.0),
                                                    "highlights": (-1.5, 2.6), "white": (-1.5, 2.6)}


GAIN_MAP_SHIFT_LIMITS: dict[str, tuple[float, float]] = {"shadows": (-3.0, 1.8), "mid": (-1.5, 1.8),
                                                         "highlights": (-2.0, 2.4), "white": (-2.0, 2.4)}


def shift_limits(source_kind: str, *, has_gain_map: bool = False) -> dict[str, tuple[float, float]]:
    if source_kind == "raw":
        return RAW_SHIFT_LIMITS
    if has_gain_map:
        return GAIN_MAP_SHIFT_LIMITS
    return SHIFT_LIMITS
# The least gap kept between neighbouring targets after a bound moved one of them.
TARGET_GAP = 0.004


def bound_shift(targets: dict[str, float], baseline: dict[str, float],
                limits: dict[str, tuple[float, float]] | None = None) -> tuple[dict[str, float], list[str]]:
    """Targets held within ``limits`` (SHIFT_LIMITS) of the unedited photo's percentiles, and a note for each
    one moved.

    A target that is brought back pulls the darker targets down (or the lighter ones up) with it so
    the order holds.
    """
    bounded = {name: float(value) for name, value in targets.items()}
    notes: list[str] = []
    for name, (lo, hi) in (limits or SHIFT_LIMITS).items():
        if name not in bounded or name not in baseline:
            continue
        shift = ev_between(baseline[name], bounded[name])
        if lo <= shift <= hi:
            continue
        limit = hi if shift > hi else lo
        value = round(to_display(to_linear(baseline[name]) * 2.0 ** limit), 4)
        notes.append(f"tone.{name}: 目标 {bounded[name]:.3f} 比原片 {baseline[name]:.3f} {shift:+.1f} EV，超出"
                     f" {lo:+.1f}..{hi:+.1f} EV（再多就改变了场景本身，夜景会变成黄昏），按 {value:.3f} 执行")
        bounded[name] = value
    names = [name for name in ORDER if name in bounded]
    for upper, lower in zip(names[::-1], names[::-1][1:]):      # Darker targets stay below lighter ones.
        bounded[lower] = round(min(bounded[lower], bounded[upper] - TARGET_GAP), 4)
    for lower, upper in zip(names, names[1:]):
        bounded[upper] = round(max(bounded[upper], bounded[lower] + TARGET_GAP), 4)
    return {name: value for name, value in bounded.items() if name in targets}, notes


def grid(source: Path | Image.Image | np.ndarray, cells: int = 8) -> dict[str, list[list[int]]]:
    """Median luma, mean colourfulness and hue per cell.

    Row 0 is the top of the picture, column 0 the left. Luma and colourfulness
    are 0..100 integers; colourfulness is max(R,G,B) - min(R,G,B): 0 for grey,
    about 15-35 for natural colour and above 55 for very strong colour. Hue is
    the colour-wheel angle of the cell's mean colour in degrees (0 red, 60
    yellow, 120 green, 180 cyan, 240 blue, 300 magenta), or -1 for a cell with
    too little colour to have one.
    """
    rgb = source if isinstance(source, np.ndarray) else load_small(source)
    height, width = rgb.shape[:2]
    luma = luma_of(rgb)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    ys = np.linspace(0, height, cells + 1).astype(int)
    xs = np.linspace(0, width, cells + 1).astype(int)
    out_luma, out_chroma, out_hue = [], [], []
    for row in range(cells):
        luma_row, chroma_row, hue_row = [], [], []
        for col in range(cells):
            block = (slice(ys[row], max(ys[row] + 1, ys[row + 1])), slice(xs[col], max(xs[col] + 1, xs[col + 1])))
            luma_row.append(int(round(float(np.median(luma[block])) * 100)))
            chroma_row.append(int(round(float(np.mean(chroma[block])) * 100)))
            hue, strength = colour_polar(colour_vector(rgb[block]))
            hue_row.append(int(round(hue)) % 360 if strength >= 0.025 else -1)
        out_luma.append(luma_row)
        out_chroma.append(chroma_row)
        out_hue.append(hue_row)
    return {"luma": out_luma, "chroma": out_chroma, "hue": out_hue}


def grid_text(table: list[list[int]], signed: bool = False) -> str:
    """Fixed-width rows for a prompt, with column and row positions as fractions."""
    cells = len(table)
    header = "      " + " ".join(f"x{(col + 0.5) / cells:.2f}"[:5].rjust(5) for col in range(cells))
    rows = [header]
    for index, row in enumerate(table):
        body = " ".join(("--" if value < 0 and not signed else f"{value:+d}" if signed else f"{value:d}").rjust(5)
                        for value in row)
        rows.append(f"y{(index + 0.5) / cells:.2f} {body}")
    return "\n".join(rows)


def curve_value(curve: list[list[float]], x: float) -> float:
    if not curve:
        return float(x)
    xs, ys = np.asarray(curve, dtype=np.float64).T
    return float(np.interp(x, xs, ys))


def curve_inverse(curve: list[list[float]], y: float) -> float:
    """Input value that the curve maps to ``y`` (the curve never decreases)."""
    if not curve:
        return float(y)
    xs, ys = np.asarray(curve, dtype=np.float64).T
    ys = np.maximum.accumulate(ys) + np.arange(len(ys)) * 1e-9
    return float(np.interp(y, ys, xs))


def chroma_mean(rgb: np.ndarray) -> float:
    """Mean colourfulness of a frame: max(R,G,B) - min(R,G,B), 0 for grey."""
    return float(np.mean(rgb.max(axis=-1) - rgb.min(axis=-1)))


def brightness_gain(before: np.ndarray, after: np.ndarray) -> float:
    """How much more colourful ``after`` is than ``before`` from brightness alone.

    max - min is saturation (HSV) times value: this holds every pixel at the saturation it had in
    ``before`` and takes the value of ``after``. A lift counts; the colour a steeper curve adds to
    the pixels it stretches does not (that is colour the frame target is meant to cover, not a
    change of the scale it was written on). Values under 0.02 give no reliable saturation.
    """
    if before.shape != after.shape:
        return chroma_mean(after) / max(chroma_mean(before), 1e-3)
    value = before.max(axis=-1)
    saturation = (value - before.min(axis=-1)) / np.maximum(value, 0.02)
    return float(np.sum(saturation * after.max(axis=-1)) / max(float(np.sum(saturation * value)), 1e-6))


def _shifted(anchors: dict[str, float], ev: float) -> dict[str, float]:
    """Anchors after an exposure change, as linear-light scaling predicts them."""
    return {name: to_display(min(1.0, to_linear(value) * 2.0 ** ev)) for name, value in anchors.items()}


def _worst_miss(anchors: dict[str, float], targets: dict[str, float]) -> float:
    _, executed = fit_curve(anchors, targets)
    return max((abs(executed[name] - targets[name]) for name in executed), default=0.0)


def plan_exposure(anchors: dict[str, float], targets: dict[str, float], exposure: float,
                  exposure_range: tuple[float, float]) -> float:
    """Smallest exposure change after which a smooth curve can reach the targets.

    Exposure is only a helper for the curve: a bright scene whose midtones move
    down keeps its exposure (the curve does that without greying the
    highlights), a dark scene whose midtones must rise by more than the curve's
    slope allows gets the difference as exposure.
    """
    best_shift, best = 0.0, _worst_miss(anchors, targets)
    if best <= REPORT_MISS:
        return exposure
    common = [name for name in ORDER if name in targets and name in anchors]
    lean = sum(ev_between(anchors[name], targets[name]) for name in common) / max(1, len(common))
    direction = 1.0 if lean >= 0 else -1.0
    for step in range(1, 13):
        shift = direction * 0.25 * step
        if not exposure_range[0] <= exposure + shift <= exposure_range[1]:
            break
        miss = _worst_miss(_shifted(anchors, shift), targets)
        if miss < best - 0.004:
            best_shift, best = shift, miss
        if miss <= REPORT_MISS:
            break
    return round(exposure + best_shift, 3)


def _within(wanted: float, limits: tuple[float, float]) -> float:
    return round(min(limits[1], max(limits[0], wanted)), 3)


# Controls the solver may move on a mask to reach a colour target, with the probe used to measure
# their effect, the largest move per step and the range they may end in. The range is narrower than
# what a plan may write directly: past it a mask reads as a colour cast, and a target that needs more
# is reported as out of reach.
COLOUR_CONTROLS = ("temp", "tint", "saturation")
COLOUR_PROBE = {"temp": 12.0, "tint": 10.0, "saturation": 15.0}
COLOUR_STEP = {"temp": 16.0, "tint": 12.0, "saturation": 20.0}
COLOUR_LIMITS: dict[str, tuple[float, float]] = {"temp": (-24.0, 24.0), "tint": (-18.0, 18.0),
                                                 "saturation": (-40.0, 30.0)}
COLOUR_TOLERANCE = 0.012
# Whether that tolerance shrinks with the mean-colour ratio of the target's surface, as the wanted vector does.
COLOUR_TOLERANCE_SCALED = True
# Cost of holding a control one full step away from where the plan put it, on the scale of a squared
# colour miss: a full step has to buy about 0.003 of colour (0.01 for the control that is the wrong
# tool for the job) to be taken in full. Effort per control (temp, tint, saturation), squared into
# that cost: on a near-neutral target saturation is the wrong tool. On a coloured one temp and tint
# used to be (x3, at 1e-4): a tint step on saturated water buys only ~0.007, so the regularised step
# stopped ~5% of the way and no hue on a coloured surface was ever turned (p13: tint never past 3.9
# in 50 targets; DSC_2121 C lake 196-198 for 186-190, "nearly identical to A"). They turn the hue.
COLOUR_EFFORT = 1e-5
COLOUR_EFFORT_NEUTRAL = (1.0, 1.0, 3.0)
COLOUR_EFFORT_COLOURED = (1.0, 1.0, 1.0)
# There temp and tint are kept to turning the hue: the colourfulness they change is weighed as a miss.
COLOUR_RADIAL_WEIGHT = 1.0
# Below this median luma a box has no hue worth chasing: the controls would run to their stops on noise.
COLOUR_MIN_LUMA = 0.08
# A colour target more than this many degrees round the wheel from the surface's colour is turned with temp and
# tint alone, without the radial term: taking the colour out comes nearest in a/b, and the surface ends grey. p21-r1
# DSC_2620 C1: frost at hue 33 asked for cyan 200 ended at saturation -40, temp -0.8, hue 43, reported reached;
# "霜原被 saturation -40 抽成中性灰" (-6 colour). p17-r1 DSC_1276 A0/C0: a path at 213-229 asked for 36-40, -20..-37.
COLOUR_FAR_TURN = 60.0
# Saturation scales a colour about grey, so what a step of it does turns and grows with the colour. Once the colour
# has turned this many degrees from where saturation was last measured, its column is turned and scaled to follow:
# measured on warm frost it still pointed warm after temp had turned the frost to cyan 186, so the solver tried
# saturation -20 there, the step did not help and colour work stopped at colourfulness .035 for .065.
COLOUR_SATURATION_FOLLOW = 20.0
# Colour passes after the closing tone passes (see the end of solve_scene): one pass moves a control at most one
# COLOUR_STEP, so this takes back a stale tint of about 4 steps.
FINAL_COLOUR_PASSES = 4
# A box that moves by less than this fraction of what its mask was asked to do is not under the mask.
MISS_RESPONSE = 0.2
# The least of a box (and never fewer than a dozen pixels) a mask must reach to be measured there.
BOX_SHARE = 0.08
# A percentile target whose pixels lie under one mask follows that mask's target: from OWN_FROM to
# OWN_FULL of the pixels within OWN_BAND of the percentile. Black and white stay with the frame.
OWN_FROM, OWN_FULL = 0.4, 0.8
OWN_BAND = 0.015
OWNED_ANCHORS = ("shadows", "mid", "highlights")
# How far such a percentile follows. "excess": only by what its mask cannot carry within its range, so the
# mask moves its region and the curve keeps the frame as planned for everything else at that brightness (a
# face in the lake's band, the pavement in the trunk's). "full": all the way to the region target, so the
# curve carries the region and moves everything at that brightness with it. Replays of p4-r1 (27 plans):
# see .task-cache/bench-2026-11-18/rp6-*.
FOLLOW = "excess"
# Share of a mask's range the solver may plan to use before a percentile has to help; the rest is slack
# for a feathered edge and for what the engine does differently from the probe.
FOLLOW_ROOM = 0.85
# A mask and the curve moving one box this far (EV) in opposite directions are working against each other.
AGAINST_EV = 0.5
# How far (EV) from a region's brightness the frame target that gives way for it may lie (see give_way).
# p7-r1/p8-r1, DSC_2121: a face under a hat brim (box 0.16) sat at the black percentile (0.155) of a hazy
# frame; black targets of 0.03-0.06 took it down 1.1-3.5 EV. Held at its range the face mask left the face
# at 0.13-0.25 (reviews -3..-6); allowed 2.5 EV it reached 0.28-0.40 but read as pasted on, and the same
# allowance gave the camels of DSC_1441 a glowing halo on the sand (C 72-73). In p5-r1, where a looser black
# fit left the black near 0.12, the face sat at 0.32-0.38 with its mask inside the range: 93 and 94.
GIVE_WAY_REACH = 1.0
# The share of what the mask cannot carry that the moved target must bring the region (see give_way).
GIVE_WAY_CARRY = 0.5
# Black or white give way only for a region within this many EV of them (p8-r1 DSC_1441 C: black 0.12 -> 0.21
# for camels 0.64 EV above black and 0.76 EV below shadows; the camels gained, every deep tone went grey).
GIVE_WAY_EDGE = 0.5
# Whether that frame target also gives way where the mask could carry the region: the curve lifts a subject
# and its mask pulls it back (or the other way). p14-r1: 15 of 84 regions (p12: 0 of 79) were moved
# 0.5-1.8 EV by the curve against their mask, and 19 deductions (65 points) named that lift-then-pull: the
# trunk of DSC_1276 lifted +1.3..+1.8 EV and pulled back to below the pavement, the white jacket of DSC_2121
# lifted +1.1..+1.35 EV and held down while the clouds took the frame's top.
GIVE_WAY_UNDO = True
# The targets that still give way only by what the mask cannot carry: they hold the frame's range (rp23 DSC_2121
# A0: white 0.935 -> 0.79 for the jacket's mask, the clouds went flat and the sky milky).
GIVE_WAY_ENDS = ("black", "white")
# How far (EV) past its own target a region may stay where the frame tone targets lift it further than its target
# lifts it (both upwards, the curve by AGAINST_EV or more past the target). The region then keeps the curve's lift
# up to this much, instead of its mask pulling it back against the curve or the percentile at its brightness giving
# way. DSC_2121's white jacket (box ~0.53, written 0.64-0.79, highlights 0.74-0.86 put it at ~0.80-0.83): pulled
# back -0.5..-0.8 EV in p14/p15 ("lift-then-pull", jacket flattened, -1..-3 subject/tone in 12 of 18 reviews);
# in p16-r1 the highlights gave way 0.80 -> 0.63 instead ("亮部整体偏低", clouds and snow dull, -3/-4 tone in all
# three versions). Where it simply sat high (p14 A2 0.80, B0 0.83) the reviews accepted it ("正好达标").
LIFT_KEEP = 0.75
# Only for a region target that is a lift of at least this many EV (the jacket targets lift it 0.46-0.62 EV): a
# target near the region's own brightness means "keep it where it is", and the percentile gives way as before.
LIFT_MIN = 0.25
# And only for a region at the frame's top: its box no more than this many EV under the frame's highlights
# percentile (the jacket's boxes -0.07..+0.17 EV). There the frame's highlights target already asks for that
# brightness. Lower down, a kept lift lets a mid-tone surface outshine the subject: DSC_1276's pavement (-0.9..-1.0
# EV, written +1.0 EV, the curve +1.48) only just missed AGAINST_EV in rp29, and "the pavement is brighter than the
# trunk" is the deduction the reviews of that photo repeat most.
LIFT_TOP = 0.35
# How far (EV, linear light) the masks may take the frame's median from the unedited photo, beyond where the curve
# puts it: past this the scene changes what it is (the reviewer is told the same). A mask that reaches most of the
# frame (a colour-range mask with a wide luminance limit on a lamp-lit night) is a global lift: p15-r1 DSC_1276, the
# trunk's colour masks at +1.25 lifted the median of all nine versions 0.029 -> 0.10-0.12 (+2.1..+2.5 EV, the
# curve alone gave 0.05-0.09), every review -6..-11 tone ("整帧被抬 ... 夜读成黄昏").
SPILL_LIMIT = 1.5
# A mask is held back for that only where its own exposure moves the frame median this many EV per EV.
SPILL_MIN = 0.1
SPILL_PASSES = 3
SPILL_SLACK = 0.05
# The frame colourfulness target is read at the photo's own brightness: lifting a frame makes it more
# colourful at the same saturation (max - min grows with the values), darkening it less so. The solver
# scales the target by what the tone change alone did, within this factor either way.
CHROMA_TONE_CAP = 2.0
# How near the near-neutrals of a natural-colour version may come to the colour-cast gate (1.0) on the
# small solver preview before the colour moves are scaled back, and the halvings spent finding how far.
# The margin covers the preview against the delivery render (rp8 of p4-r1, DSC_1441 A: 1.05 on the
# preview failed at 1.03, 0.94 passed). When the colour moves are not the cause, global saturation
# goes down by NEUTRAL_DESATURATE a step, at most NEUTRAL_STEPS steps.
NEUTRAL_ROOM = 0.9
NEUTRAL_STEPS = 3
NEUTRAL_DESATURATE = 8.0
# Where the neutrals stay near the gate without the colour moves too, whether keeping the moves is tried as well.
NEUTRAL_KEEP_MOVES = True
# Closing pass: the curve is fitted to the frame as planned, and everything after it (masks the solver moved,
# colour work, saturation, the neutral guard) moves the percentiles again; recheck refits only after a large
# saturation change. p12-r1 DSC_1441: mid/highlights/white finished 0.02-0.03 under the curve's own values
# after colour work, B0/B1 shadows and mid 0.05-0.14 under where a cliff mask darkened their band, and every
# review read those as targets not reached. The frame as it finally stands is measured against what the curve
# fitted to the aims gives the frame as planned ("executed"), and the curve refitted with each target moved by
# that drift (at most CLOSE_STEP a pass), the boxes put back on target; a pass is kept only when tones and
# boxes together end nearer (each counted past CLOSE_SLACK) and no mask ends pinned or against the curve.
CLOSE_PASSES = 2
CLOSE_STEP = 0.08
CLOSE_SLACK = 0.01
# Whether a percentile whose pixels lie mostly under a mask the solver moved is read with the masks at their start
# (what that mask did to it left to its region target) instead of on the frame as it stands. Off: held, a big sky
# mask that darkened its half of the frame left the frame mid under its target and the closing pass never chased
# it. Replay of all 27 p21-r1 plans (rp32 held, rp33 off): mean |mid miss| .0216 -> .0167, none worse in mid;
# 1515 A0 mid -.055 -> -.044, C0 -.074 -> -.060, 1963 A2 -.042 -> -.011, C2 -.056 -> -.027; black/shadows within
# .001 except 1963 C2 shadows -.028 -> -.036; sky masks a little deeper (1515 A0 .64 -> .85 EV).
CLOSE_HOLD_MOVED = False


def _linear(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _soft(plane: np.ndarray) -> np.ndarray:
    """3x3 mean with repeated edges: an 8-bit render is too coarse in the shadows to compare pixel by pixel."""
    height, width = plane.shape
    padded = np.pad(plane, 1, mode="edge")
    return sum(padded[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)) / 9.0


def coverage(reference: np.ndarray, shot: np.ndarray, delta: float) -> np.ndarray:
    """Share (0..1) of an exposure step of ``delta`` EV on one mask that reached each pixel.

    ``reference`` and ``shot`` are luma planes before and after the step. Pixels that are black in
    both read as not reached: nothing can be measured there.
    """
    before, after = _soft(_linear(reference)), _soft(_linear(shot))
    change = np.log2(np.maximum(after, 1e-3) / np.maximum(before, 1e-3))
    return np.clip(change / float(delta), 0.0, 1.0)


def covered_pixels(share: np.ndarray, box: Sequence[float]) -> tuple[np.ndarray | None, bool, float]:
    """The pixels of a target box that its mask reaches.

    Returns the selection over the frame (None when the mask is not there), whether it had to be
    looked for beside the box, and the share of the box as written that is selected. A box often
    straddles the edge of its mask; measured whole, the uncovered part hides what the mask does and
    the mask is pushed too far. When the box itself holds too little of the mask the search widens by
    half the box each way, which catches a box drawn a little off a small subject.
    """
    height, width = share.shape
    y0, y1, x0, x1 = _span(height, width, box)
    need = max(12.0, BOX_SHARE * (y1 - y0) * (x1 - x0))
    for grow in (0.0, 0.5):
        gy, gx = int(round(grow * (y1 - y0))), int(round(grow * (x1 - x0)))
        window = (slice(max(0, y0 - gy), min(height, y1 + gy)), slice(max(0, x0 - gx), min(width, x1 + gx)))
        inside = share[window]
        reached = inside[inside >= MISS_RESPONSE]
        if reached.size < need:
            continue
        # From half the mask's own level here, so a feathered or thin mask still counts as one area.
        strong = inside >= max(MISS_RESPONSE, 0.5 * float(np.percentile(reached, 90.0)))
        if int(strong.sum()) < need:
            continue
        picked = np.zeros(share.shape, dtype=bool)
        picked[window] = strong
        return picked, grow > 0.0, float(picked[y0:y1, x0:x1].mean())
    return None, False, float(np.mean(share[y0:y1, x0:x1] >= MISS_RESPONSE))


def owner_of(plane: np.ndarray, value: float, covers: dict[int, np.ndarray]) -> tuple[int | None, float]:
    """The mask that covers the pixels at one brightness and how fully it owns them (0..1).

    ``plane`` is the luma of the frame, ``covers`` the coverage of each candidate mask.
    """
    band = np.abs(plane - float(value)) <= OWN_BAND
    if not covers or int(band.sum()) < 40:
        return None, 0.0
    shares = {index: float(np.mean(cover[band])) for index, cover in covers.items()}
    index = max(shares, key=lambda item: shares[item])
    return index, min(1.0, max(0.0, (shares[index] - OWN_FROM) / (OWN_FULL - OWN_FROM)))


def frame_fix(reference: dict[str, float], executed: dict[str, float], curve: list[list[float]],
              targets: dict[str, float], value: float, gap: float) -> dict[str, Any] | None:
    """The frame target that would let a region a mask cannot carry reach its own target.

    ``value`` is the region's median going into the curve (its mask at its limit), ``gap`` how much
    it still misses by after the curve. The region sits between two anchors of the frame as planned
    (``reference``); the anchor whose output carries it most is the one to move. Returns that anchor,
    its current target, the value it would need, and the anchors around the region.
    """
    names = [name for name in ORDER if name in reference and name in executed]
    if not names or abs(gap) < 1e-6:
        return None
    below = [name for name in names if reference[name] <= value]
    above = [name for name in names if reference[name] >= value]
    lower, upper = (below[-1] if below else None), (above[0] if above else None)
    if lower is None or upper is None or lower == upper:
        anchor, weight = (lower or upper), 1.0
    else:
        span = max(1e-6, reference[upper] - reference[lower])
        share = (value - reference[lower]) / span
        anchor, weight = (lower, 1.0 - share) if share <= 0.5 else (upper, share)
    now = float(targets.get(anchor, executed[anchor]))
    needed = min(0.99, max(0.0, float(executed[anchor]) + gap / max(weight, 0.5)))
    return {"anchor": anchor, "now": round(now, 3), "needed": round(needed, 3), "between": [lower, upper],
            "input": round(float(value), 3), "by_curve": round(curve_value(curve, value), 3)}


def _owned_steps(reach: np.ndarray, need: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exposure steps that bring every box to its target, each mask answering for its own box.

    ``reach[i, j]`` is what mask j does to box i. A mask that runs into a bound stays there and
    its target is given up for this step: the other masks account for what it does to their
    boxes, but they do not push further to make up for it.
    """
    step = np.zeros(len(need))
    free = list(range(len(need)))
    for _ in range(len(need) + 1):
        if not free:
            break
        held = [j for j in range(len(need)) if j not in free]
        rhs = need[free] - reach[np.ix_(free, held)] @ step[held]
        block = reach[np.ix_(free, free)]
        solution = np.linalg.solve(block.T @ block + 1e-3 * np.eye(len(free)), block.T @ rhs)
        excess = [max(lo[j] - value, value - hi[j]) for j, value in zip(free, solution)]
        worst = int(np.argmax(excess))
        if excess[worst] <= 1e-9:
            step[free] = solution
            break
        j = free[worst]
        step[j] = float(np.clip(solution[worst], lo[j], hi[j]))
        free.remove(j)
    return step


def solve_scene(render: Callable[..., Path], targets: dict[str, float],
                regions: list[dict[str, Any]] | None = None, *, exposure: float = 0.0,
                exposure_range: tuple[float, float] = (-2.5, 2.5), rounds: int = 6,
                tolerance: float = 0.015, chroma: float | None = None, saturation: float = 0.0,
                saturation_range: tuple[float, float] = (-60.0, 40.0),
                neutral: Callable[[np.ndarray], float] | None = None,
                unedited: dict[str, float] | None = None,
                shift_limits: dict[str, tuple[float, float]] | None = None,
                max_slope: float = MAX_SLOPE) -> dict[str, Any]:
    """Place tones, region brightness, region colour and colour intensity by measurement.

    ``render(curve, exposure, local, saturation)`` renders a small preview and
    returns its path. ``local`` maps a region key to the adjustments the solver
    owns for that mask: its ``exposure`` (moved for a brightness target, and
    stepped once on every targeted mask to see where the mask lies) and
    ``{"temp", "tint", "saturation"}`` for a colour target. ``saturation`` is
    the global saturation slider.

    Each region is ``{"key", "box", "median"?, "exposure", "range": (lo, hi),
    "colour"?: (hue degrees, colourfulness 0..1), "start"?: {control: value},
    "colour_limits"?: {control: (lo, hi)}}``. A target is measured on the pixels
    of its box that the mask reaches. The mask's exposure is moved until their
    median luma is on target; its temp, tint and saturation are moved towards
    the wanted hue and colourfulness (colourfulness as in ``region_stats``: the
    mean of max - min), as far as that is worth the distance from the plan.
    ``chroma`` is the wanted mean colourfulness of the whole frame, reached with
    global saturation.

    A percentile target whose pixels lie under one mask follows that mask's
    target (returned as ``aims``): the curve then carries the region instead of
    working against its mask. A mask that does not reach its box is reported as
    missed and left where it started.

    Tone steps work on values with the current curve removed, which stays
    correct while exposure and the local exposures change.

    ``neutral(rgb)``, given, says how near a frame's near-neutrals are to a
    colour-cast gate (1.0 at the gate). Above ``NEUTRAL_ROOM`` the solver's
    colour moves are scaled back towards the plan's own until they are not.

    ``unedited``, given, holds the percentiles of the unedited photo: a target
    that gives way for a region goes back at most to where the photo was.
    """
    regions = [dict(region) for region in (regions or [])]
    toned = [index for index, region in enumerate(regions) if region.get("median") is not None]
    tinted = [index for index, region in enumerate(regions) if region.get("colour") is not None]
    local: dict[Any, dict[str, float]] = {}
    for index, region in enumerate(regions):
        owned: dict[str, float] = {"exposure": float(region.get("exposure", 0.0))}
        if index in tinted:
            start = region.get("start") or {}
            for control in COLOUR_CONTROLS:
                owned[control] = float(start.get(control, 0.0))
        local[region["key"]] = owned
    probes = [0]
    frame: dict[str, Any] = {}
    picks: dict[int, np.ndarray | None] = {}

    def picked(values: np.ndarray, index: int) -> np.ndarray:
        """The pixels a target is measured on: those its mask reaches, the whole box until that is known."""
        pick = picks.get(index)
        if pick is not None and pick.shape == values.shape[:2]:
            return values[pick]
        cut = _crop(values, regions[index]["box"])
        return cut.reshape(-1, cut.shape[-1]) if cut.ndim == 3 else cut.reshape(-1)

    def medians_of(plane: np.ndarray) -> list[float]:
        return [round(float(np.median(picked(plane, index))), 4) for index in range(len(regions))]

    def colours_of(rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Mean colour vector and colourfulness of every colour target."""
        out = []
        for index in tinted:
            pixels = picked(rgb, index)
            out.append((colour_vector(pixels), float(np.mean(pixels.max(axis=-1) - pixels.min(axis=-1)))))
        return out

    def shoot(curve: list[list[float]]) -> tuple[dict[str, float], list[float]]:
        probes[0] += 1
        frame["current"] = False
        rgb = load_small(render(curve, exposure, {key: dict(value) for key, value in local.items()}, saturation))
        frame["rgb"], frame["chroma"] = rgb, chroma_mean(rgb)
        return measure(rgb), medians_of(luma_of(rgb))

    anchors, medians = shoot([])
    frame["current"] = True        # False while the last render is a probe, not the settings as they stand.
    first = frame["rgb"]
    before, before_chroma = dict(anchors), frame["chroma"]
    notes: list[str] = []
    starts = {index: local[regions[index]["key"]]["exposure"] for index in toned}
    # "reference" is the frame as planned, before the solver moves a mask, with the curve removed. The
    # curve is fitted to it and to nothing else: fitted to the frame the masks have moved, curve and
    # masks chase each other (masks lift, the curve pushes down and steepens, the masks lift again).
    state: dict[str, Any] = {"reference": anchors, "reference_medians": medians, "widest": {},
                             "anchors": anchors, "medians": medians, "after": anchors, "after_regions": medians,
                             "curve": [], "executed": {}, "aims": dict(targets)}
    # reach[i, j]: EV the pixels of target i move per EV of exposure on the mask of target j. The
    # diagonal is near 1 for a mask that covers its target and less under a feathered edge; the rest
    # is one mask reaching into another's target (a face inside a figure).
    reach = np.eye(len(toned))
    response = {index: 1.0 for index in toned}
    missed: set[int] = set()
    pinned: set[int] = set()
    colour_missed: set[int] = set()
    beside: set[int] = set()            # Targets measured beside their box: the mask was not in it.
    covered: dict[int, float] = {}      # Share of each box its mask reaches.
    covers: dict[int, np.ndarray] = {}
    owners: dict[str, int] = {}         # Percentile targets that follow a region target.
    sharers: dict[str, int] = {}        # Percentile targets whose pixels lie mostly under one mask.
    yielded: dict[str, int] = {}        # Percentile targets that gave way for a region at their brightness.
    undone: set[str] = set()            # Of those, the ones that took back the curve's whole move (GIVE_WAY_UNDO).
    spreads: dict[int, float] = {}      # EV the frame median moves per EV of each mask's exposure.
    spilled: dict[int, dict[str, float]] = {}   # Masks held back because they moved the whole frame (SPILL_LIMIT).
    lifted: dict[int, dict[str, float]] = {}    # Regions left where the curve lifts them, past their target (LIFT_KEEP).
    # A RAW negative is developed further than a finished picture may be moved (RAW_SHIFT_LIMITS): the masks may
    # lift its median by as much more as the curve may (ref-r4 DSC02262: the foreground snow's lift was held at the
    # JPEG limit, median .63 from the negative's .385, his .75).
    spill_up = SPILL_LIMIT + max(0.0, (shift_limits or SHIFT_LIMITS)["mid"][1] - SHIFT_LIMITS["mid"][1])

    def read(curve: list[list[float]]) -> None:
        state["after"], state["after_regions"] = shoot(curve)
        state["curve"], frame["current"] = curve, True
        # The same frame with the curve removed.
        state["anchors"] = {name: curve_inverse(curve, value) for name, value in state["after"].items()}
        state["medians"] = [curve_inverse(curve, value) for value in state["after_regions"]]

    def drift() -> float:
        return max([abs(state["after_regions"][index] - float(regions[index]["median"]))
                    for index in toned if index not in missed and index not in pinned] + [0.0])

    def probe_masks() -> None:
        """One render per targeted mask, a stop away: which pixels does it reach, and whose boxes do they move?"""
        kept_frame = (frame["rgb"], frame["chroma"])
        plane = luma_of(frame["rgb"])
        shots: dict[int, tuple[np.ndarray, float]] = {}
        for index, region in enumerate(regions):
            owned = local[region["key"]]
            # Away from the nearer end of the scale, so that clipped pixels are not read as out of reach.
            bright = to_linear(float(np.percentile(_crop(plane, region["box"]), 90.0))) > 0.4
            delta = -1.0 if bright else 1.0
            kept = owned["exposure"]
            owned["exposure"] = kept + delta
            shot_anchors, _ = shoot([])
            owned["exposure"] = kept
            shot = luma_of(frame["rgb"])
            shots[index] = (shot, delta)
            if "mid" in shot_anchors and "mid" in state["reference"]:
                spreads[index] = ev_between(state["reference"]["mid"], shot_anchors["mid"]) / delta
            share = coverage(plane, shot, delta)
            picks[index], moved, covered[index] = covered_pixels(share, region["box"])
            if moved:
                beside.add(index)
            if index in toned:
                covers[index] = share
            if picks[index] is None:
                if index in toned:
                    missed.add(index)
                if index in tinted:
                    colour_missed.add(index)
        frame["rgb"], frame["chroma"] = kept_frame
        frame["current"] = True
        base = medians_of(plane)
        for column, index in enumerate(toned):
            shot, delta = shots[index]
            for row, other in enumerate(toned):
                reach[row, column] = ev_between(base[other], float(np.median(picked(shot, other)))) / delta
        for position, index in enumerate(toned):
            response[index] = float(min(1.5, max(0.0, reach[position, position])))
            if reach[position, position] < MISS_RESPONSE:
                missed.add(index)
        state["reference_medians"] = state["medians"] = state["after_regions"] = base

    def reaches(name: str, value: float) -> bool:
        """May the frame target ``name`` give way for a region at ``value`` (curve removed)?"""
        distance = abs(ev_between(state["reference"][name], value))
        if name == "black" or name == "white":
            # Below black (above white) the curve scales the region with that target alone: a face darker
            # than the frame's 1st percentile (p9-r1 DSC_2121 A: box 0.067, black 0.158 -> 0.05, face taken
            # down 5.5 EV) goes wherever black goes, however far below it lies.
            outside = value <= state["reference"][name] if name == "black" else value >= state["reference"][name]
            return outside or distance <= GIVE_WAY_EDGE
        return distance <= GIVE_WAY_REACH

    def give_way(trial: list[list[float]], aims: dict[str, float], named: list[str]) -> None:
        """Let the frame target at a region's brightness give way where the curve takes the region against its mask.

        A small region (a face, camels) owns no percentile, but its pixels may sit at one: the curve the
        targets ask for then moves it one way, its target wants the other, and its mask cannot give back
        more than its range. The percentile nearest the region's brightness moves by what the mask cannot
        carry, never by more than the curve took.
        """
        for index in toned:
            if index in missed or index in owners.values():
                continue
            planned_box = state["reference_medians"][index]
            planned = curve_value(trial, planned_box)
            by_curve = ev_between(planned_box, planned)
            need = ev_between(planned, float(regions[index]["median"]))
            if by_curve * need >= 0 or min(abs(by_curve), abs(need)) < AGAINST_EV:
                continue
            lo, hi = regions[index].get("range", (-1.5, 1.5))
            gain = max(response[index], 0.3)
            start = starts[index]
            # Black and white only for a region that is that tone (a face that is the frame's darkest pixels):
            # moved for a region between them and the next target, they lift every deep tone into fog.
            near = [name for name in named if name in state["reference"] and name not in owners
                    and name not in yielded and reaches(name, planned_box)]
            if not near:
                continue
            name = min(near, key=lambda item: abs(ev_between(state["reference"][item], planned_box)))
            # With GIVE_WAY_UNDO a percentile between black and white first tries to take back what the curve did
            # against the region, as reconcile does for a region that owns its percentile, so the mask carries
            # only the rest; where that cannot pass (the photo's own value or the order stops the target short of
            # carrying the region) it gives way by what the mask cannot carry, as before (rp23 DSC_1441 C1: the
            # whole undo failed the carry check at shadows 0.427 = image 1 and the camel mask ended pinned).
            # Black and white only ever do the latter: they hold the frame's range, and taking one back for a
            # mask flattens that whole end (rp23 DSC_2121 A0: white 0.935 -> 0.79 for the jacket, the clouds
            # went flat and the sky milky).
            tries = (True, False) if GIVE_WAY_UNDO and name not in GIVE_WAY_ENDS else (False,)
            for undoing in tries:
                undo = float(np.sign(need)) * min(abs(need), abs(by_curve)) if undoing else 0.0
                rest = need - undo
                carried = gain * min(max(rest / gain, FOLLOW_ROOM * (lo - start)), FOLLOW_ROOM * (hi - start))
                excess = float(np.sign(need)) * min(abs(undo + rest - carried), abs(by_curve))
                if abs(excess) < 0.1:
                    continue
                follows = to_display(min(1.0, to_linear(aims[name]) * 2.0 ** excess))
                # Giving way is doing less to the photo, never more: no further than the photo's own value.
                # p10-r1 DSC_2121: the plan's own lift put the planned black at 0.19-0.22 (image 1: 0.158) and
                # black went all the way there for the face; every review called the frame grey, without an anchor.
                if unedited and name in unedited:
                    own = float(unedited[name])
                    follows = min(follows, max(aims[name], own)) if excess > 0 else max(follows, min(aims[name], own))
                place_in_order = named.index(name)
                if place_in_order > 0:
                    follows = max(follows, aims[named[place_in_order - 1]] + 0.02)
                if place_in_order + 1 < len(named):
                    follows = min(follows, aims[named[place_in_order + 1]] - 0.02)
                if abs(follows - aims[name]) <= REPORT_MISS:
                    continue
                # Only where that target carries the region: one between two targets moves with both, and moving
                # the one below alone lifts the darkest tones without bringing the region along (p8-r1 DSC_1441 C:
                # black 0.12 -> 0.21 for camels between black and shadows, camels unmoved).
                moved = dict(aims, **{name: round(follows, 4)})
                gained = ev_between(planned, curve_value(fit_curve(state["reference"], moved, max_slope=max_slope)[0], planned_box))
                if gained * excess <= 0 or abs(gained) < GIVE_WAY_CARRY * abs(excess):
                    continue
                aims[name] = round(follows, 4)
                yielded[name] = index
                if undoing:
                    undone.add(name)
                break

    def keep_lifts(trial: list[list[float]]) -> None:
        """Leave a region the curve lifts past its own lift target where the curve puts it (up to LIFT_KEEP).

        The tone targets and the region target both brighten it; the curve just does more. Pulling it back
        with its mask flattens it against the curve, and lowering the percentile at its brightness dulls the
        frame's top; keeping the extra lift does neither.
        """
        for index in toned:
            if index in missed or index in lifted:
                continue
            planned_box = state["reference_medians"][index]
            planned = curve_value(trial, planned_box)
            target = float(regions[index]["median"])
            over = ev_between(target, planned)
            if ev_between(planned_box, target) < LIFT_MIN or over < AGAINST_EV:
                continue
            top = state["reference"].get("highlights")
            if top is None or ev_between(top, planned_box) < -LIFT_TOP:
                continue
            kept = round(to_display(min(1.0, to_linear(target) * 2.0 ** min(over, LIFT_KEEP))), 4)
            regions[index]["median"] = kept
            lifted[index] = {"target": round(target, 4), "kept": kept, "by_curve": round(planned, 4)}

    def reconcile() -> None:
        """Let a percentile target follow the region target of the mask that covers its pixels.

        With FOLLOW "excess" it follows only by what that mask cannot carry: the curve as the targets
        ask for it puts the region somewhere, the mask can move it from there within its range, and only
        the rest of the way is left to the percentile.
        """
        aims = dict(targets)
        plane = luma_of(frame["rgb"])
        candidates = {index: covers[index] for index in toned if index not in missed}
        named = [name for name in ORDER if name in targets]
        trial = fit_curve(state["reference"], targets, max_slope=max_slope)[0] if FOLLOW == "excess" else None
        if trial is not None and LIFT_KEEP > 0:
            keep_lifts(trial)
        for name in OWNED_ANCHORS:
            if name not in targets or name not in state["reference"]:
                continue
            index, taken = owner_of(plane, state["reference"][name], candidates)
            if index is None or taken <= 0.0:
                continue
            sharers[name] = index
            if trial is not None:
                planned_box = state["reference_medians"][index]
                planned = curve_value(trial, planned_box)
                by_curve = ev_between(planned_box, planned)
                need = ev_between(planned, float(regions[index]["median"]))
                # Where the curve would move this band half a stop or more one way and the region wants the
                # other, the curve gives way: a mask undoing the curve is what makes a picture harsh.
                opposed = by_curve * need < 0 and min(abs(by_curve), abs(need)) >= AGAINST_EV
                undo = float(np.sign(need)) * min(abs(need), abs(by_curve)) if opposed else 0.0
                lo, hi = regions[index].get("range", (-1.5, 1.5))
                gain = max(response[index], 0.3)
                start = starts[index]
                rest = need - undo
                carried = gain * min(max(rest / gain, FOLLOW_ROOM * (lo - start)), FOLLOW_ROOM * (hi - start))
                excess = undo + rest - carried
                if abs(excess) < 0.1:
                    continue
                follows = to_display(min(1.0, to_linear(targets[name]) * 2.0 ** excess))
            else:
                shift = ev_between(state["reference_medians"][index], float(regions[index]["median"]))
                follows = to_display(min(1.0, to_linear(state["reference"][name]) * 2.0 ** shift))
            # A percentile that follows a region still keeps the scene: no further from the unedited photo than a
            # written target may go (p15-r1 DSC_1276 B0: mid .034 followed the trunk to .11, +2.3 EV over image 1).
            if unedited and name in unedited and name in (shift_limits or SHIFT_LIMITS):
                low_ev, high_ev = (shift_limits or SHIFT_LIMITS)[name]
                own = to_linear(float(unedited[name]))
                follows = min(max(follows, min(targets[name], to_display(own * 2.0 ** low_ev))),
                              max(targets[name], to_display(own * 2.0 ** high_ev)))
            # Never past the targets on either side: the frame keeps its order, the mask does the rest.
            place_in_order = named.index(name)
            if place_in_order > 0:
                follows = max(follows, targets[named[place_in_order - 1]] + 0.02)
            if place_in_order + 1 < len(named):
                follows = min(follows, targets[named[place_in_order + 1]] - 0.02)
            aims[name] = round((1.0 - taken) * targets[name] + taken * follows, 4)
            if abs(aims[name] - targets[name]) > REPORT_MISS:
                owners[name] = index
        if trial is not None:
            give_way(trial, aims, named)
        state["aims"] ={name: round(min(0.995, value), 4)
                         for name, value in _increasing({name: aims[name] for name in ORDER if name in aims}).items()}

    def fit() -> None:
        measured = dict(state["reference"])
        for name, pick in (("black", min), ("white", max)):       # Room for what the masks pushed outwards.
            if name in measured and name in state["widest"]:
                measured[name] = pick(measured[name], state["widest"][name])
        # "fit_to" is set only by the closing pass: the aims moved by what the finished frame missed them by.
        state["curve"], state["executed"] = fit_curve(measured, state.get("fit_to") or state["aims"], max_slope=max_slope)

    def settle_curve(passes: int) -> None:
        """Fit the curve to the frame as planned and correct for what the engine does differently."""
        for _ in range(passes):
            fit()
            read(state["curve"])
            if max([abs(state["after"][name] - state["executed"][name]) for name in state["executed"]] + [0.0]) <= tolerance:
                break
            state["reference"], state["reference_medians"] = state["anchors"], state["medians"]

    def widened() -> bool:
        """True when local work pushed the darkest or brightest tones past what the curve was fitted for."""
        grown = False
        for name, sign in (("black", -1.0), ("white", 1.0)):
            if name in state["executed"] and sign * (state["after"][name] - state["executed"][name]) > tolerance:
                state["widest"][name] = state["anchors"][name]
                grown = True
        return grown

    def place(passes: int) -> None:
        """Move each mask until its target sits where it should under the curve."""
        live = [position for position, index in enumerate(toned) if index not in missed]
        for _ in range(max(1, passes)):
            if not live:
                break
            owned = [local[regions[toned[position]]["key"]] for position in live]
            limits = [tuple(regions[toned[position]].get("range", (-1.5, 1.5))) for position in live]
            need = np.array([ev_between(state["medians"][toned[position]],
                                        curve_inverse(state["curve"], float(regions[toned[position]]["median"])))
                             for position in live])
            now = np.array([item["exposure"] for item in owned])
            # All masks together: a mask that also lights another's box is counted there.
            step = _owned_steps(reach[np.ix_(live, live)], need,
                                np.maximum(np.array([limit[0] for limit in limits]) - now, -1.25),
                                np.minimum(np.array([limit[1] for limit in limits]) - now, 1.25))
            pinned.clear()
            for position, item, limit, value, wanted in zip(live, owned, limits, now + step, need):
                item["exposure"] = _within(float(value), limit)
                if (item["exposure"] >= limit[1] and wanted > 0) or (item["exposure"] <= limit[0] and wanted < 0):
                    pinned.add(toned[position])     # At its stop with further to go.
            step = np.array([item["exposure"] for item in owned]) - now
            previous = [state["medians"][toned[position]] for position in live]
            read(state["curve"])
            size = float(step @ step)
            if size > 0.01:   # Broyden: bend the measured responses towards what this step did.
                got = np.array([ev_between(before_step, state["medians"][toned[position]])
                                for before_step, position in zip(previous, live)])
                block = reach[np.ix_(live, live)]
                reach[np.ix_(live, live)] = block + np.outer(got - block @ step, step) / size
            if targets and widened():
                fit()
                read(state["curve"])
            if drift() <= tolerance:
                break

    def hold_spill() -> None:
        """Hold back the masks that take the frame's median past SPILL_LIMIT from the unedited photo.

        Where the curve's own aim puts the median further (a written target within SHIFT_LIMITS), that is the
        limit: the masks add nothing past it. A mask that moves the median is one that reaches most of the frame;
        it is held at the exposure that keeps the median there, and reported: more of it lifts the scene, not
        its region.
        """
        if not unedited or "mid" not in unedited or "mid" not in state["after"]:
            return
        own = to_linear(float(unedited["mid"]))
        ceiling, floor = to_display(own * 2.0 ** spill_up), to_display(own * 2.0 ** -SPILL_LIMIT)
        aimed = state["aims"].get("mid") if targets else None
        if aimed is not None:
            ceiling, floor = max(ceiling, float(aimed)), min(floor, float(aimed))
        for _ in range(SPILL_PASSES):
            if not frame["current"]:
                read(state["curve"])
            now = state["after"]["mid"]
            over = ev_between(ceiling, now) if now > ceiling else ev_between(floor, now) if now < floor else 0.0
            if abs(over) < SPILL_SLACK:
                break
            movers = [index for index in toned if index not in missed and spreads.get(index, 0.0) >= SPILL_MIN
                      and local[regions[index]["key"]]["exposure"] * over > 0]
            if not movers:
                break
            total = sum(spreads[index] * local[regions[index]["key"]]["exposure"] for index in movers)
            share = min(1.0, over / total)
            for index in movers:
                owned = local[regions[index]["key"]]
                was = owned["exposure"]
                held = round(was * (1.0 - share), 3)
                lo, hi = regions[index].get("range", (-1.5, 1.5))
                regions[index]["range"] = (lo, max(lo, min(hi, held))) if was > 0 else (min(hi, max(lo, held)), hi)
                owned["exposure"] = _within(held, regions[index]["range"])
                spilled.setdefault(index, {"from": round(was, 3)})
            read(state["curve"])
            if drift() > tolerance:
                place(2)
        for index, record in spilled.items():
            record.update(exposure=local[regions[index]["key"]]["exposure"], median=round(state["after"]["mid"], 4),
                          unedited=round(float(unedited["mid"]), 4), limit=round(ceiling if record["from"] > 0 else floor, 4),
                          limit_ev=round(spill_up if record["from"] > 0 else SPILL_LIMIT, 2))

    if regions:
        probe_masks()
    before_regions, before_colours = medians_of(luma_of(first)), colours_of(first)
    if targets:
        reconcile()
        moved = plan_exposure(state["reference"], state["aims"], exposure, exposure_range)
        if abs(moved - exposure) > 0.05:
            notes.append(f"tone: 平滑曲线单独够不到这组影调，全局曝光从 {exposure:+.2f} 调到 {moved:+.2f} EV")
            exposure = moved
            state["anchors"], state["medians"] = shoot([])
            state["after"], state["after_regions"] = state["anchors"], state["medians"]
            state["reference"], state["reference_medians"] = state["anchors"], state["medians"]
            frame["current"] = True
        settle_curve(3)
    if toned:
        place(rounds)
        hold_spill()
    if not frame["current"]:
        read(state["curve"])

    def recheck() -> None:
        """After colour work: is the curve still right for the frame as planned, are the boxes still on target?"""
        if targets and abs(saturation - saturation_start) >= 5.0:     # Saturation moves luma with it.
            kept = {index: local[regions[index]["key"]]["exposure"] for index in toned}
            if any(abs(kept[index] - starts[index]) > 1e-6 for index in toned):
                for index in toned:
                    local[regions[index]["key"]]["exposure"] = starts[index]
                planned, boxes = shoot(state["curve"])
                for index in toned:
                    local[regions[index]["key"]]["exposure"] = kept[index]
            else:
                planned, boxes = state["after"], state["after_regions"]
            if max([abs(planned[name] - state["executed"][name]) for name in state["executed"]] + [0.0]) > tolerance:
                state["reference"] = {name: curve_inverse(state["curve"], value) for name, value in planned.items()}
                state["reference_medians"] = [curve_inverse(state["curve"], value) for value in boxes]
                fit()
                frame["current"] = False
        if not frame["current"]:
            read(state["curve"])
        if toned and drift() > tolerance:
            place(2)

    saturation_start = saturation

    def settle_chroma(passes: int = 3) -> None:
        nonlocal saturation
        for _ in range(passes):
            if abs(frame["chroma"] - chroma) <= max(0.006, 0.05 * chroma):
                break
            gain = (1.0 + saturation / 100.0) * (chroma / max(frame["chroma"], 1e-3)) ** 0.9
            wanted = min(saturation_range[1], max(saturation_range[0], 100.0 * gain - 100.0))
            if abs(wanted - saturation) < 1.0:
                break
            saturation = round(wanted, 1)
            read(state["curve"])

    keys = [regions[index]["key"] for index in tinted]
    limits: list[dict[str, tuple[float, float]]] = []
    for index in tinted:
        allowed = {**COLOUR_LIMITS, **(regions[index].get("colour_limits") or {})}
        owned = local[regions[index]["key"]]
        # A plan may start a control outside the solver's range; the solver then only brings it back in.
        limits.append({control: (min(allowed[control][0], owned[control]), max(allowed[control][1], owned[control]))
                       for control in COLOUR_CONTROLS})
    scale = np.array([COLOUR_STEP[control] for control in COLOUR_CONTROLS])
    begin = [np.array([local[key][control] for control in COLOUR_CONTROLS]) for key in keys]
    jacobian = [np.zeros((2, len(COLOUR_CONTROLS))) for _ in tinted]
    wanted_vectors: list[np.ndarray] = []
    saturation_at: list[np.ndarray] = []    # The colour where each saturation column was last measured.
    ratios: list[float] = []        # Mean-colour length per colourfulness on each target's surface.
    dark: set[int] = set()
    # Of the many ways to reach a colour, the one a retoucher would take: saturation for more or
    # less of the colour that is there, temp and tint to turn a hue or to clean a neutral.
    effort = [np.diag(COLOUR_EFFORT_NEUTRAL if regions[index]["colour"][1] < 0.06 else COLOUR_EFFORT_COLOURED) ** 2
              for index in tinted]

    def probe_colours() -> None:
        base = colours_of(frame["rgb"])
        for position, index in enumerate(tinted):
            hue, amount = regions[index]["colour"]
            vector, pixel = base[position]
            # The target is written as colourfulness (mean of max - min); the solver works on the
            # mean colour, which is shorter by this ratio on textured or mixed surfaces.
            ratio = min(1.0, max(0.3, float(np.hypot(*vector)) / max(pixel, 1e-3)))
            wanted_vectors.append(colour_target(hue, float(amount) * ratio))
            saturation_at.append(np.array(vector, dtype=float))
            ratios.append(ratio)
            if index not in colour_missed and state["after_regions"][index] < COLOUR_MIN_LUMA:
                dark.add(index)
        live = [position for position, index in enumerate(tinted) if index not in colour_missed and index not in dark]
        if not live:
            return
        for axis, control in enumerate(COLOUR_CONTROLS):      # One probe per control, all masks at once.
            stepped: dict[int, tuple[float, float]] = {}
            for position in live:
                key, delta = keys[position], COLOUR_PROBE[control]
                if local[key][control] + delta > limits[position][control][1]:
                    delta = -delta
                stepped[position] = (local[key][control], delta)
                local[key][control] += delta
            shoot(state["curve"])
            probed = colours_of(frame["rgb"])
            for position in live:
                kept, delta = stepped[position]
                jacobian[position][:, axis] = (probed[position][0] - base[position][0]) / delta
                local[keys[position]][control] = kept
        for position in live:
            if float(np.linalg.norm(jacobian[position] * scale)) < 0.008:   # A full step moves nothing.
                colour_missed.add(tinted[position])
        read(state["curve"])

    def follow_saturation(position: int, vector: np.ndarray) -> None:
        was = saturation_at[position]
        size, was_size = float(np.hypot(*vector)), float(np.hypot(*was))
        if size < 1e-6 or was_size < 1e-6:
            return
        turn = float(np.arctan2(vector[1], vector[0]) - np.arctan2(was[1], was[0]))
        if abs((np.degrees(turn) + 180.0) % 360.0 - 180.0) <= COLOUR_SATURATION_FOLLOW:
            return
        cos, sin = np.cos(turn), np.sin(turn)
        axis = COLOUR_CONTROLS.index("saturation")
        jacobian[position][:, axis] = (np.array([[cos, -sin], [sin, cos]]) @ jacobian[position][:, axis]
                                       * (size / was_size))
        saturation_at[position] = np.array(vector, dtype=float)

    def settle_colours(passes: int) -> None:
        current = [vector for vector, _ in colours_of(frame["rgb"])]
        miss = [float(np.hypot(*(wanted_vectors[position] - current[position]))) for position in range(len(keys))]
        # Close enough: a hue or colourfulness this near its target is not worth another move. The tolerance is in
        # the target's own units, scaled like the wanted vector: held absolute, a textured near-neutral (mean colour a
        # third of its colourfulness) read as reached at 0.047 for 0.08 and was never moved (p15-r1 DSC_2121 C0-C2
        # ridge_gold: temp, tint and saturation all 0, hue 28 for 34-38; "褐色荒山仍是灰褐", C "nearly identical").
        near = [max(COLOUR_TOLERANCE * (ratios[position] if COLOUR_TOLERANCE_SCALED else 1.0),
                    0.09 * float(np.hypot(*wanted_vectors[position]))) for position in range(len(keys))]
        active = [position for position, index in enumerate(tinted)
                  if index not in colour_missed and index not in dark and miss[position] > near[position]]
        for _ in range(passes):
            if not active:
                break
            taken: dict[int, tuple[np.ndarray, dict[str, float]]] = {}
            for position in active:
                key = keys[position]
                follow_saturation(position, current[position])
                scaled = jacobian[position] * scale
                # Least squares on the colour miss plus the cost of the distance from the plan's own
                # values. Damping the step instead would only slow it down, not change where it ends.
                away = (np.array([local[key][control] for control in COLOUR_CONTROLS]) - begin[position]) / scale
                cost = COLOUR_EFFORT * effort[position]
                length = float(np.hypot(*current[position]))
                wanted_length = float(np.hypot(*wanted_vectors[position]))
                far = (length > 1e-6 and wanted_length > 1e-6 and float(current[position] @ wanted_vectors[position])
                       < np.cos(np.radians(COLOUR_FAR_TURN)) * length * wanted_length)
                if far:     # A turn round the wheel: temp and tint only, through grey if need be.
                    scaled[:, COLOUR_CONTROLS.index("saturation")] = 0.0
                lhs = scaled.T @ scaled + cost
                if regions[tinted[position]]["colour"][1] >= 0.06 and length > 1e-6 and not far:
                    # On a coloured surface temp and tint turn the hue; what they add to or take from its
                    # colourfulness counts as a miss of its own, so saturation does that part (sand made
                    # richer with saturation, not by warming it).
                    radial = np.zeros(len(COLOUR_CONTROLS))
                    radial[:2] = (current[position] / length) @ scaled[:, :2]
                    lhs = lhs + COLOUR_RADIAL_WEIGHT * np.outer(radial, radial)
                step = np.linalg.solve(lhs, scaled.T @ (wanted_vectors[position] - current[position]) - cost @ away)
                # Shortened as a whole, not control by control: clipping only the control that has the far way
                # to go keeps the others' compensation for all of it (p13 DSC_2121 C0 lake: tint cut to -12 of
                # the -60 asked, saturation still -7 for the whole turn, so the step bought nothing and was
                # taken back for good).
                step = step / max(1.0, float(np.max(np.abs(step)))) * scale
                old = {control: local[key][control] for control in COLOUR_CONTROLS}
                for axis, control in enumerate(COLOUR_CONTROLS):
                    local[key][control] = _within(old[control] + float(step[axis]), limits[position][control])
                taken[position] = (np.array([local[key][control] - old[control] for control in COLOUR_CONTROLS]), old)
            read(state["curve"])
            now = colours_of(frame["rgb"])
            taken_back = False
            for position in list(active):
                moved_by, old = taken[position]
                change = now[position][0] - current[position]
                size = float(moved_by @ moved_by)
                if size > 1e-9:   # Broyden: bend the measured response towards what this step did.
                    jacobian[position] += np.outer(change - jacobian[position] @ moved_by, moved_by) / size
                left = float(np.hypot(*(wanted_vectors[position] - now[position][0])))
                if left > miss[position] * 0.97 or size <= 1e-9:
                    # The step did not help (a clipped channel, a control at its stop, or the best
                    # trade against the distance from the plan is already reached): take it back.
                    local[keys[position]].update(old)
                    active.remove(position)
                    taken_back = True
                    continue
                current[position], miss[position] = now[position][0], left
                if left <= near[position]:
                    active.remove(position)
            if taken_back:
                read(state["curve"])
                current = [vector for vector, _ in colours_of(frame["rgb"])]

    written, tone_gain, region_added = chroma, 1.0, 0.0
    if chroma is not None:
        # The frame as it stands has the tones placed and the colour as planned: what the tone change
        # did to the colourfulness, held between 1 and what its brightness change alone does (at the
        # same saturation, ``brightness_gain``). Holding a lifted
        # night to the old number would wash it out. Letting the colour a contrast curve adds raise
        # the target pushed saturation onto the neutrals of daylight frames (rp6 of p4-r1: DSC_1441 A,
        # gain 1.17 at 0.88x luma, +15-30 saturation, a neutral cast); scaling a darkened frame's
        # target down by its brightness washed its colour out (p5-r1 DSC_1441 B: 0.9, saturation -40,
        # the sand "淡黄灰").
        measured = frame["chroma"] / max(before_chroma, 1e-3)
        bright = brightness_gain(first, frame["rgb"])
        tone_gain = min(max(measured, min(1.0, bright)), max(1.0, bright))
        tone_gain = min(CHROMA_TONE_CAP, max(1.0 / CHROMA_TONE_CAP, tone_gain))
        chroma = float(chroma) * tone_gain
        settle_chroma()
    if tinted:
        settled = frame["chroma"]
        probe_colours()
        settle_colours(6)
        if chroma is not None:        # Region colours moved the frame mean, and the other way round.
            # The colour the region targets add (or take away) is what the plan asked of them; the
            # frame target is for the rest. Held to the written number, global saturation took back
            # what a mask gave its region and washed out what no target covered (p5-r1 DSC_1441 B:
            # the cliff +30, the sand under -40).
            region_added = frame["chroma"] - settled
            chroma = max(0.0, chroma + region_added)
            before_second = saturation
            settle_chroma()
            if abs(saturation - before_second) >= 2.0:
                settle_colours(2)
    if (chroma is not None or tinted) and (targets or toned):
        recheck()
        if chroma is not None:        # The last curve fit and mask moves shifted the colourfulness once more.
            settle_chroma(2)

    neutral_report = None
    if neutral is not None and (chroma is not None or tinted):
        solved_saturation = saturation
        solved = {key: {control: local[key][control] for control in COLOUR_CONTROLS} for key in keys}

        def keep_share(share: float) -> float:
            """Keep this share of the solver's colour moves (mask temp, tint and saturation from where
            the plan put them, and global saturation above the plan's; a lower one only helps the
            neutrals and stays); returns how near the neutrals are to the gate."""
            nonlocal saturation
            added = solved_saturation - saturation_start
            saturation = round(saturation_start + (added * share if added > 0 else added), 1)
            for position, key in enumerate(keys):
                for axis, control in enumerate(COLOUR_CONTROLS):
                    start_value = float(begin[position][axis])
                    local[key][control] = round(start_value + (solved[key][control] - start_value) * share, 3)
            read(state["curve"])
            return neutral(frame["rgb"])

        def desaturate(now: float) -> tuple[float, float]:
            """What is left comes from the tones or the plan's own colour (a steeper curve makes warm near-neutrals
            warmer): take colour out of the whole frame, as far as that helps. Returns how near the neutrals end
            and the saturation that leaves them there; the frame is rendered at it."""
            nonlocal saturation
            held_here, best = saturation, (now, saturation)
            if now <= NEUTRAL_ROOM:
                return best
            for step in range(1, NEUTRAL_STEPS + 1):
                saturation = round(max(saturation_range[0], held_here - NEUTRAL_DESATURATE * step), 1)
                read(state["curve"])
                reached = neutral(frame["rgb"])
                if reached < best[0]:
                    best = (reached, saturation)
                if reached <= NEUTRAL_ROOM or saturation <= saturation_range[0]:
                    break
            if saturation != best[1]:
                saturation = best[1]
                read(state["curve"])
            return best

        full = neutral(frame["rgb"])
        if full > NEUTRAL_ROOM:
            kept, now, settled = 1.0, full, False
            bare = keep_share(0.0)
            if bare >= full:             # The colour moves are not what casts the neutrals: keep them.
                now = keep_share(1.0)
            elif bare > NEUTRAL_ROOM:     # Even without them the neutrals are near the gate.
                # The whole frame loses colour either way. Where the colour moves are not much of the cast, dropping
                # them buys nothing (p15-r1 DSC_1441 A1/A2: 0.94 with them, 0.92 without, 0.74 after one step of
                # desaturation; the cliff's and camels' colour targets were dropped all the same): both are tried,
                # and the one with the most colour left under the room is kept, the moves where that is a tie.
                options = [(0.0, saturation, *desaturate(bare))]
                if NEUTRAL_KEEP_MOVES:
                    with_moves = keep_share(1.0)
                    options.append((1.0, saturation, *desaturate(with_moves)))

                def rank(option: tuple[float, float, float, float]) -> tuple[bool, float, float]:
                    share, _, reached, value = option
                    under = reached <= NEUTRAL_ROOM
                    return under, (value if under else -reached), share

                kept, held, now, chosen = max(options, key=rank)
                if kept != options[-1][0]:       # The frame is rendered as the last option tried left it.
                    keep_share(kept)
                    saturation = chosen
                    read(state["curve"])
                settled = True
            else:
                low, high, now = 0.0, 1.0, bare
                for _ in range(NEUTRAL_STEPS):
                    middle = (low + high) / 2.0
                    reached = keep_share(middle)
                    if reached > NEUTRAL_ROOM:
                        high = middle
                    else:
                        low, now = middle, reached
                kept = low
                if kept != middle:
                    now = keep_share(kept)
            if not settled:
                held = saturation
                now, _ = desaturate(now)
            neutral_report = {"solved": round(full, 2), "after": round(now, 2), "kept": round(kept, 2),
                              "desaturated": round(held - saturation, 1)}
            if kept < 1.0 or saturation < held:
                notes.append(f"tone.neutral: 原片近中性的亮部偏色（达到自然色门槛的 {full:.0%}）：求解器的加色调整只保留了 "
                             f"{kept:.0%}" + (f"，全局饱和度再降 {held - saturation:.0f}" if saturation < held else "")
                             + f"，现为门槛的 {now:.0%}；要更浓的颜色，只加在有色的区域上，不要加在近中性的亮部上")

    def snapshot() -> dict[str, Any]:
        return {"state": {key: dict(value) if isinstance(value, dict) else value for key, value in state.items()},
                "local": {key: dict(value) for key, value in local.items()}, "frame": dict(frame),
                "reach": reach.copy(), "pinned": set(pinned)}

    def restore(kept: dict[str, Any]) -> None:
        state.clear()
        state.update(kept["state"])
        for key, values in kept["local"].items():
            local[key].clear()
            local[key].update(values)
        frame.clear()
        frame.update(kept["frame"])
        reach[...] = kept["reach"]
        pinned.clear()
        pinned.update(kept["pinned"])

    def close_tones() -> dict[str, Any] | None:
        """Measure the frame as it now stands against the aims; refit the curve for what moved since it was fitted.

        Percentile targets that follow or gave way for a region are left alone: their pixels are that region's.
        One whose pixels lie mostly under a mask the solver moved is read on the frame with the masks where the
        plan put them: what that mask did to it is its region target's, and the curve chasing it back has the
        mask and the curve pull against each other (a sand mask at -1.6 under a curve lifting the sand).
        """
        plain = dict(state["executed"])
        free = [name for name in ORDER if name in plain and name in state["after"]
                and name not in owners and name not in yielded]
        live = [index for index in toned if index not in missed]
        if not free:
            return None
        if not frame["current"]:
            read(state["curve"])
        moved = {index for index in live if abs(local[regions[index]["key"]]["exposure"] - starts[index]) >= 0.1}
        carried = [name for name in free if sharers.get(name) in moved] if CLOSE_HOLD_MOVED else []

        # Measured from what the curve fitted to the aims gives, not from the aims: where a smooth curve could
        # not reach a target the fit shared it out, and chasing that again only steepens the curve (rp19 of
        # p12-r1 DSC_2121: black chased to 0.0, steepest slope 2.6 -> 3.2, the black no nearer).
        def misses() -> dict[str, float]:
            out = {name: state["after"][name] - plain[name] for name in free if name not in carried}
            if carried:
                kept = {index: local[regions[index]["key"]]["exposure"] for index in toned}
                for index in toned:
                    local[regions[index]["key"]]["exposure"] = starts[index]
                planned, _ = shoot(state["curve"])
                for index in toned:
                    local[regions[index]["key"]]["exposure"] = kept[index]
                out.update({name: planned[name] - plain[name] for name in carried})
            return out

        def cost() -> float:
            tones = sum(max(0.0, abs(state["after"][name] - plain[name]) - CLOSE_SLACK)
                        for name in free if name not in carried)
            boxes = sum(max(0.0, abs(state["after_regions"][index] - float(regions[index]["median"])) - CLOSE_SLACK)
                        for index in live)
            return tones + boxes

        def strain() -> int:
            """Masks at their stop with further to go, and masks moved against what the curve does to their box."""
            against = 0
            for index in live:
                planned_box = state["reference_medians"][index]
                by_curve = ev_between(planned_box, curve_value(state["curve"], planned_box))
                shifted = local[regions[index]["key"]]["exposure"] - starts[index]
                if by_curve * shifted < 0 and min(abs(by_curve), abs(shifted)) >= AGAINST_EV:
                    against += 1
            return against + len(pinned)

        guard = neutral is not None and (chroma is not None or bool(tinted))
        gate = neutral(frame["rgb"]) if guard else 0.0
        start = best = cost()
        held = strain()
        fitted, passes = dict(state["aims"]), 0
        for _ in range(CLOSE_PASSES):
            miss = misses()
            if max(abs(value) for value in miss.values()) <= tolerance:
                break
            trial = dict(fitted)
            for name, value in miss.items():
                if abs(value) > tolerance:
                    trial[name] = min(0.995, max(0.0, fitted[name] - max(-CLOSE_STEP, min(CLOSE_STEP, value))))
            kept = snapshot()
            state["fit_to"] = _increasing(trial)
            fit()
            read(state["curve"])
            if live and drift() > tolerance:
                place(2)
            now = cost()
            # A pass that pins a mask or sets one against the curve trades a few hundredths of tone for
            # the harshness the reviews punish most (rp19 DSC_2121 B0, DSC_1441 B2).
            if (now >= best - 1e-4 or strain() > held
                    or (guard and neutral(frame["rgb"]) > max(NEUTRAL_ROOM, gate))):
                restore(kept)
                break
            best, fitted, passes = now, dict(state["fit_to"]), passes + 1
        state.pop("fit_to", None)
        if not frame["current"]:          # The last render was the frame as planned, or a pass taken back.
            read(state["curve"])
        # "executed" stays what the curve fitted to the aims does to the frame as planned: the notes compare
        # it with the aims to tell a target the curve cannot reach from one that moved afterwards.
        state["executed"] = plain
        if not passes:
            return None
        return {"fitted": {name: round(value, 4) for name, value in fitted.items()},
                "cost": [round(start, 4), round(best, 4)], "passes": passes}

    if toned and SPILL_PASSES:
        hold_spill()        # Colour work and the recheck moved the masks again.
    closing = close_tones() if targets else None
    if tinted and FINAL_COLOUR_PASSES and (targets or toned):
        # The recheck, hold_spill and the closing passes refit the curve and move the masks after the colour work,
        # and a tone curve turns a saturated colour's hue: ref-r13 DSC05752 B lit sand reached hue 32.7 for 33 at
        # tint +47 under the curve of the colour work; the recheck's refit (mask +.93 -> +1.04) turned it to 17.8,
        # salmon, and the tint stayed (tint 0 renders 34.3 under the final curve). Settle once more on the frame as
        # it stands; a step that does not bring a colour nearer is taken back, as in the colour work.
        if not frame["current"]:
            read(state["curve"])
        held_colours = {key: {control: local[key][control] for control in COLOUR_CONTROLS} for key in keys}
        guard_before = neutral(frame["rgb"]) if neutral is not None else 0.0
        settle_colours(FINAL_COLOUR_PASSES)
        if neutral is not None and neutral(frame["rgb"]) > max(NEUTRAL_ROOM, guard_before):
            for key, values in held_colours.items():     # The neutral guard's share stands.
                local[key].update(values)
            read(state["curve"])

    chroma_report = None
    if chroma is not None:
        chroma_report = {"target": round(float(chroma), 4), "written": round(float(written), 4),
                         "tone_gain": round(tone_gain, 3), "regions_added": round(region_added, 4),
                         "before": round(before_chroma, 4),
                         "after": round(frame["chroma"], 4), "saturation": saturation,
                         "saturation_start": saturation_start}
        if abs(frame["chroma"] - chroma) > max(0.015, 0.12 * chroma):
            notes.append(f"tone.chroma: 色彩浓度目标 {written * 100:.0f}（按影调变化折算为 {chroma * 100:.0f}），"
                         f"全局饱和度已调到 {saturation:+.0f}，成片为 {frame['chroma'] * 100:.0f}")
    after_colours = colours_of(frame["rgb"])
    after, after_regions = state["after"], state["after_regions"]
    planned, aims = state["executed"], state["aims"]
    for name in ORDER:
        if name not in targets or name not in after:
            continue
        if name in owners:
            notes.append(f"tone.{name}: 目标 {targets[name]:.2f} 与一个区域目标冲突（那个蒙版覆盖了这一档亮度的大部分画面），"
                         f"{'蒙版够不到的部分由这一档补上' if FOLLOW == 'excess' else '按区域目标执行'}，成片为 {after[name]:.2f}")
        elif name in yielded:
            notes.append(f"tone.{name}: 目标 {targets[name]:.2f} 会把蒙版「{regions[yielded[name]].get('name', regions[yielded[name]]['key'])}」的目标区域一起往反方向推，"
                         f"{'' if name in undone else '超出蒙版能补回的量，'}这一档让到 {aims[name]:.2f}，成片为 {after[name]:.2f}")
        elif abs(planned.get(name, after[name]) - aims[name]) > REPORT_MISS:
            notes.append(f"tone.{name}: 目标 {targets[name]:.2f} 与相邻影调冲突，成片为 {after[name]:.2f}")
        elif abs(after[name] - aims[name]) > REPORT_MISS + 0.015:
            notes.append(f"tone.{name}: 曲线按目标 {targets[name]:.2f} 执行，局部蒙版之后成片为 {after[name]:.2f}")
    error = max([abs(planned.get(name, after[name]) - aims[name]) for name in aims if name in after]
                + [abs(after_regions[index] - float(regions[index]["median"])) for index in toned] + [0.0])
    report = []
    for index, region in enumerate(regions):
        owned = local[region["key"]]
        item: dict[str, Any] = {"key": region["key"], "box": region["box"], "before": before_regions[index],
                                "after": after_regions[index]}
        if index in covered:
            item["coverage"] = round(covered[index], 2)
        if index in beside:
            item["moved"] = True           # Measured beside the box: the mask does not reach into it.
        if index in toned:
            item.update(target=float(region["median"]), exposure=owned["exposure"],
                        limited=owned["exposure"] in tuple(region.get("range", (-1.5, 1.5))),
                        response=round(response[index], 2), missed=index in missed)
            if index in lifted:
                item["kept_lift"] = dict(lifted[index])
                notes.append(f"tone.lift: 影调目标已把「{region.get('name', region['key'])}」提到约 {lifted[index]['by_curve']:.2f}，"
                             f"比它自己的目标 {lifted[index]['target']:.2f} 更亮（同为提亮），留在 {lifted[index]['kept']:.2f}，"
                             "不再用蒙版反向压回，也不让这一档影调目标让路")
            # What the curve alone does to this target, in EV. A mask that moves it the other way by
            # half a stop or more is undoing the tone targets: the two were written against each other.
            planned_box = state["reference_medians"][index]
            by_curve = ev_between(planned_box, curve_value(state["curve"], planned_box))
            shifted = owned["exposure"] - starts[index]
            if index not in missed and by_curve * shifted < 0 and min(abs(by_curve), abs(shifted)) >= AGAINST_EV:
                item["against_curve"] = round(by_curve, 2)
            gave = [name for name in ORDER if yielded.get(name) == index]
            if gave:
                item["gave_way"] = gave            # Percentile targets at this region's brightness that gave way.
            followed = [name for name in ORDER if owners.get(name) == index]
            if followed:
                item["overrides"] = followed       # Percentile targets that were set aside for this one.
            # Percentile targets this mask moved off: their pixels are mostly this mask's.
            shared = [name for name in ORDER if sharers.get(name) == index and name not in owners and name in after
                      and abs(after[name] - aims[name]) > REPORT_MISS + 0.015]
            if shared:
                item["shares"] = shared
            if index in spilled:
                # Held back because it moved the whole frame's median (hold_spill); no frame fix: moving the frame
                # targets towards this region is the lift the hold stopped.
                item["spilled"] = dict(spilled[index])
                held = spilled[index]
                notes.append(f"tone.spill: 蒙版「{region.get('name', region['key'])}」覆盖了画面大部分中间调，曝光 {held['from']:+.2f} EV "
                             f"会让整帧中位数离原片 {held['unedited']:.3f} 超过 {held['limit_ev']:g} EV，已收到 {held['exposure']:+.2f} EV"
                             f"（成片中位数 {held['median']:.3f}）")
            elif (index not in missed and item["limited"] and abs(item["after"] - item["target"]) > 0.03
                    and planned_box > 0.0):
                fix = frame_fix(state["reference"], state["executed"], state["curve"], targets,
                                to_display(min(1.0, to_linear(planned_box) * 2.0 ** shifted)),
                                item["target"] - item["after"])
                if fix:
                    item["frame_fix"] = fix
        if index in tinted:
            position = tinted.index(index)
            hue_before, _ = colour_polar(before_colours[position][0])
            hue_after, _ = colour_polar(after_colours[position][0])
            item["colour"] = {"target_hue": float(region["colour"][0]), "target_chroma": float(region["colour"][1]),
                              "before_hue": round(hue_before, 1), "before_chroma": round(before_colours[position][1], 4),
                              "after_hue": round(hue_after, 1), "after_chroma": round(after_colours[position][1], 4),
                              **{control: owned[control] for control in COLOUR_CONTROLS},
                              "missed": index in colour_missed,
                              "limits": {control: list(limits[position][control]) for control in COLOUR_CONTROLS}}
            if index in dark:
                item["colour"]["dark"] = True
        report.append(item)
    return {"curve": state["curve"], "exposure": exposure, "local": local, "saturation": saturation,
            "before": before, "after": after, "executed": state["executed"], "aims": aims, "regions": report,
            "chroma": chroma_report, "notes": notes, "error": round(float(error), 4), "probes": probes[0],
            **({"neutral": neutral_report} if neutral_report else {}), **({"closing": closing} if closing else {})}
