"""Quality gate for JPEGs rendered by LightCraft.

The local renderer is intentionally not used here: this module re-opens the
exported pixels, compares them with the unedited baseline, and produces
bounded corrective recipes for one or two refinement passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .color import srgb_to_linear
from .models import HueShift, Recipe, TechnicalReport
from .ingest import image_to_srgb


@dataclass
class LightroomPixelMetrics:
    mean_luma: float
    luma_std: float
    shadow_clip: float
    highlight_clip: float
    saturation_mean: float
    high_saturation: float
    green_fraction: float
    green_saturation: float
    skin_fraction: float
    skin_saturation: float
    sky_fraction: float
    sky_saturation: float
    red_fraction: float
    perceptual_distance: float = 0.0
    low_tone_fraction: float = 0.0


@dataclass
class LightroomQualityIssue:
    code: str
    severity: float
    note: str
    hard: bool = True
    advisory: bool = False


@dataclass
class LightroomQualityReport:
    score: float
    accepted: bool
    metrics: LightroomPixelMetrics
    baseline: LightroomPixelMetrics
    issues: list[LightroomQualityIssue] = field(default_factory=list)
    region_source: str = "color-heuristic"
    exposure_assessment: dict[str, float] = field(default_factory=dict)
    neutral_assessment: dict[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _PixelSample:
    rgb: np.ndarray
    luma: np.ndarray
    hue: np.ndarray
    saturation: np.ndarray
    value: np.ndarray


def render_distance(first_path: Path, second_path: Path, *, max_edge: int = 960) -> float:
    """Return the mean Lab distance between two final Lightroom candidates."""
    first = _read_pixels(first_path, max_edge=max_edge)
    second = _read_pixels(second_path, size=(first.rgb.shape[1], first.rgb.shape[0]))
    return _perceptual_distance(first.rgb, second.rgb)


def is_distinct_candidate(
    rendered_path: Path,
    accepted_paths: list[Path],
    *,
    min_distance: float = 2.8,
) -> tuple[bool, float | None]:
    """Require a candidate to add a visibly different choice, not a duplicate."""
    if not accepted_paths:
        return True, None
    nearest = min(render_distance(rendered_path, other) for other in accepted_paths)
    return nearest >= min_distance, nearest


def assess_lightroom_render(
    baseline_path: Path,
    rendered_path: Path,
    *,
    slot: str,
    min_score: float = 82.0,
    max_edge: int = 960,
    region_masks: dict[str, np.ndarray] | None = None,
    creative_palette: bool = False,
    previous_accepted_path: Path | None = None,
    exposure_recovery: bool = False,
    exposure_lift_ev: float = 0.0,
    natural_color: bool = False,
    render_recipe: Recipe | None = None,
) -> LightroomQualityReport:
    """Compare an actual Lightroom export against its Lightroom baseline.

    A caller may supply an already verified render of the same photograph and
    framing for a subsequent finishing pass. It affects only the too-dark gate;
    clipping, color and distance checks still use the original baseline.
    """
    baseline = _read_pixels(baseline_path, max_edge=max_edge)
    rendered = _read_pixels(rendered_path, size=(baseline.rgb.shape[1], baseline.rgb.shape[0]))
    masks = _content_masks(baseline)
    trusted = set()
    for name, region in (region_masks or {}).items():
        if name not in {"sky", "skin"}:
            continue
        region = np.asarray(region)
        if region.ndim != 2 or not np.all(np.isfinite(region)):
            continue
        mask_image = Image.fromarray((np.clip(region, 0, 1) * 255).astype(np.uint8))
        region = np.asarray(mask_image.resize((baseline.rgb.shape[1], baseline.rgb.shape[0]), Image.Resampling.NEAREST)) > 127
        # The optional model supplies a person region, not literal skin pixels.
        masks[name] = (masks[name] & region) if name == "skin" else region
        trusted.add(name)
    base_metrics = _metrics(baseline, masks)
    out_metrics = _metrics(rendered, masks)
    out_metrics.perceptual_distance = _perceptual_distance(baseline.rgb, rendered.rgb)

    issues: list[LightroomQualityIssue] = []

    shadow_limit = max(0.08, base_metrics.shadow_clip + 0.035)
    if out_metrics.shadow_clip > shadow_limit:
        severity = _severity(out_metrics.shadow_clip, shadow_limit, 0.12)
        issues.append(
            LightroomQualityIssue(
                "shadow_clipping",
                severity,
                f"暗部裁切 {out_metrics.shadow_clip:.1%}，高于当前照片安全线 {shadow_limit:.1%}",
            )
        )

    highlight_limit = max(0.025, base_metrics.highlight_clip + 0.012)
    if out_metrics.highlight_clip > highlight_limit:
        severity = _severity(out_metrics.highlight_clip, highlight_limit, 0.08)
        issues.append(
            LightroomQualityIssue(
                "highlight_clipping",
                severity,
                f"高光裁切 {out_metrics.highlight_clip:.1%}，高于当前照片安全线 {highlight_limit:.1%}",
            )
        )

    # Deliberate B/C grading can color previously neutral water or walls while
    # keeping almost all pixels moderately saturated. Distinguish this from
    # excessive high-saturation coverage; retain stricter A and skin checks.
    styled = creative_palette and slot in {"B", "C"}
    saturation_limit = max(base_metrics.saturation_mean + (0.14 if styled else 0.075),
                           base_metrics.saturation_mean * (1.50 if styled else 1.20))
    high_sat_limit = max(base_metrics.high_saturation + 0.105, base_metrics.high_saturation * 1.38)
    if out_metrics.saturation_mean > saturation_limit or out_metrics.high_saturation > high_sat_limit:
        severity = max(
            _severity(out_metrics.saturation_mean, saturation_limit, 0.18),
            _severity(out_metrics.high_saturation, high_sat_limit, 0.24),
        )
        issues.append(
            LightroomQualityIssue(
                "oversaturation",
                severity,
                f"平均饱和度 {out_metrics.saturation_mean:.3f}（上限 {saturation_limit:.3f}）；"
                f"高饱和区域 {out_metrics.high_saturation:.1%}（上限 {high_sat_limit:.1%}）",
            )
        )

    if base_metrics.green_fraction >= 0.08:
        # Recovering light can reveal ordinary foliage color. A relative gain
        # from a washed-out source alone is not evidence of neon greens.
        green_limit = max(.55, base_metrics.green_saturation + 0.055, base_metrics.green_saturation * 1.09)
        if out_metrics.green_saturation > green_limit:
            issues.append(
                LightroomQualityIssue(
                    "green_oversaturation",
                    _severity(out_metrics.green_saturation, green_limit, 0.18),
                    f"大面积绿色饱和度 {out_metrics.green_saturation:.3f}，高于保护线 {green_limit:.3f}",
                )
            )

    if base_metrics.skin_fraction >= 0.018:
        skin_hi = max(base_metrics.skin_saturation + 0.075, base_metrics.skin_saturation * 1.18)
        skin_lo = max(0.08, base_metrics.skin_saturation * 0.68)
        if out_metrics.skin_saturation > skin_hi:
            issues.append(
                LightroomQualityIssue(
                    "skin_oversaturation" if "skin" in trusted else "warm_palette_saturation",
                    _severity(out_metrics.skin_saturation, skin_hi, 0.16),
                    "人物内暖色区域饱和度过高" if "skin" in trusted else "暖橙色区域饱和度增加；未确认该区域为肤色",
                    hard="skin" in trusted, advisory="skin" not in trusted,
                )
            )
        elif out_metrics.skin_saturation < skin_lo:
            issues.append(
                LightroomQualityIssue(
                    "skin_washed_out" if "skin" in trusted else "warm_palette_desaturation",
                    _severity(skin_lo, out_metrics.skin_saturation, 0.16),
                    "人物内暖色区域饱和度损失过多" if "skin" in trusted else "暖橙色区域饱和度减少；未确认该区域为肤色",
                    hard="skin" in trusted, advisory="skin" not in trusted,
                )
            )

    if base_metrics.sky_fraction >= 0.06:
        sky_limit = max(base_metrics.sky_saturation + 0.09, base_metrics.sky_saturation * 1.22)
        if out_metrics.sky_saturation > sky_limit:
            issues.append(
                LightroomQualityIssue(
                    "sky_oversaturation" if "sky" in trusted else "blue_palette_saturation",
                    _severity(out_metrics.sky_saturation, sky_limit, 0.18),
                    "天空区域饱和度过高" if "sky" in trusted else "蓝青色区域饱和度增加；可能属于山体、水面或天空",
                    hard="sky" in trusted, advisory="sky" not in trusted,
                )
            )

    luma_ratio = out_metrics.mean_luma / max(base_metrics.mean_luma, 1e-4)
    bright = baseline.luma > .6
    lower = (baseline.luma > .015) & (baseline.luma < .3)
    highlight_recovery = (
        np.mean(bright) > .25 and np.mean(lower) > .15
        and np.mean(rendered.luma[lower]) >= np.mean(baseline.luma[lower]) * .65
        and out_metrics.shadow_clip <= base_metrics.shadow_clip + .01
        and out_metrics.highlight_clip <= base_metrics.highlight_clip + .01
    )
    recovery_note = ""
    if previous_accepted_path is not None and luma_ratio < 0.62:
        previous = _read_pixels(previous_accepted_path, size=(baseline.rgb.shape[1], baseline.rgb.shape[0]))
        # Use the stricter incremental check on later passes. OR-ing it with
        # the original rule could excuse uniform darkening of an accepted edit.
        highlight_recovery, recovery_note = _continued_highlight_recovery(baseline, previous, rendered)
    if luma_ratio < 0.62:
        issues.append(
            LightroomQualityIssue(
                "too_dark",
                _severity(0.62, luma_ratio, 0.28),
                f"整体亮度只有 LightCraft 基线的 {luma_ratio:.0%}" + recovery_note,
                hard=not highlight_recovery, advisory=bool(highlight_recovery),
            )
        )
    elif luma_ratio > 1.62:
        issues.append(
            LightroomQualityIssue(
                "too_bright",
                _severity(luma_ratio, 1.62, 0.55),
                f"整体亮度达到 LightCraft 基线的 {luma_ratio:.0%}",
            )
        )

    exposure_assessment = {}
    if exposure_recovery:
        # Fixed baseline coordinates: moving clipping below the threshold must
        # never masquerade as recovery of detail that was absent in a JPEG.
        usable = (baseline.luma > .03) & (baseline.luma < .85) & (np.max(baseline.rgb, axis=-1) < .99)
        white = np.min(baseline.rgb, axis=-1) > .99
        exposure_assessment = {"usable_fraction": float(np.mean(usable)), "original_white_fraction": float(np.mean(white))}
        if np.mean(usable) > .35:
            before_mid = float(np.median(baseline.luma[usable]))
            after_mid = float(np.median(rendered.luma[usable]))
            exposure_assessment.update(usable_median_before=before_mid, usable_median_after=after_mid)
            if before_mid > .30 and after_mid > before_mid * .87:
                issues.append(LightroomQualityIssue("exposure_unresolved", 1.0,
                    "有效区域的过亮中间调未得到校正；不能仅凭减少高光裁切判定恢复成功"))
        if np.any(white):
            exposure_assessment["original_white_median_after"] = float(np.median(rendered.luma[white]))
        if np.mean(white) >= .03 and float(np.median(rendered.luma[white])) < .85:
            issues.append(LightroomQualityIssue("clipped_whites_greyed", 1.0,
                "原片无纹理的纯白区域被压灰；降低亮度没有恢复原本不存在的细节"))

    if exposure_lift_ev > .15:
        # A damage-free export can still leave the diagnosed dark midtones
        # untouched. Check the same original coordinates, independently of look
        # distance or parameter values. Never equate this with aesthetic quality.
        if render_recipe is not None and _changes_framing(render_recipe):
            exposure_assessment["lift_skipped"] = "geometry_changes"
        else:
            midtones = (baseline.luma > .04) & (baseline.luma < .40)
            exposure_assessment["lift_region_fraction"] = float(np.mean(midtones))
            if np.mean(midtones) >= .15:
                before_mid = float(np.median(baseline.luma[midtones]))
                after_mid = float(np.median(rendered.luma[midtones]))
                ratio = after_mid / max(before_mid, 1e-6)
                minimum = float(2 ** (min(exposure_lift_ev, .55) * .5))
                exposure_assessment.update(lift_median_before=before_mid,
                    lift_median_after=after_mid, lift_ratio=ratio, lift_min_ratio=minimum)
                if ratio < minimum:
                    issues.append(LightroomQualityIssue("exposure_lift_unresolved", 1.0,
                        f"已诊断偏暗的中间调亮度只达到基线 {ratio:.0%}，未达本次最低改善目标 {minimum:.0%}；不能以无损伤代替完成校正"))

    neutral_assessment = {}
    if natural_color and render_recipe is not None and _changes_framing(render_recipe):
        neutral_assessment = {"skipped": "geometry_changes"}
    elif natural_color:
        neutral_assessment = neutral_cast(baseline.rgb, rendered.rgb)
        shift = neutral_assessment.get("median_shift", 0.0)
        if shift > NEUTRAL_SHIFT and neutral_assessment["introduced_chroma"] > NEUTRAL_INTRODUCED:
            issues.append(LightroomQualityIssue("neutral_color_cast", _severity(shift, NEUTRAL_SHIFT, 5),
                f"原片近中性亮部占 {neutral_assessment['fraction']:.1%}，新增色度偏移 {shift:.1f}；"
                "与保留自然颜色的目标不符，需减弱调色而非补偿曝光"))

    # A should be restrained. B/C may move farther, but extreme display-referred
    # changes usually indicate that contrast and colour moves compounded.
    distance_limit = 17.0 if slot == "A" else 29.0
    if out_metrics.perceptual_distance > distance_limit:
        issues.append(
            LightroomQualityIssue(
                "excessive_distance",
                _severity(out_metrics.perceptual_distance, distance_limit, 18.0),
                f"与 LightCraft 基线的平均感知距离 {out_metrics.perceptual_distance:.1f} 过大",
                hard=slot == "A" or out_metrics.perceptual_distance > 36.0,
            )
        )

    penalty = sum((18.0 if issue.hard else 8.0) * max(0.25, issue.severity) for issue in issues if not issue.advisory)
    score = max(0.0, min(100.0, 100.0 - penalty))
    hard_fail = any(issue.hard for issue in issues)
    return LightroomQualityReport(
        score=round(score, 2),
        accepted=(not hard_fail and score >= min_score),
        metrics=out_metrics,
        baseline=base_metrics,
        issues=issues,
        region_source="semantic-regions+color" if trusted else "color-heuristic",
        exposure_assessment=exposure_assessment,
        neutral_assessment=neutral_assessment,
    )


def _continued_highlight_recovery(
    baseline: _PixelSample, previous: _PixelSample, rendered: _PixelSample,
) -> tuple[bool, str]:
    """Verify that a later darkening preserves the accepted non-bright tones."""
    bright = baseline.luma > .6
    lower = (baseline.luma > .015) & (baseline.luma < .3)
    nonbright = ~bright
    bright_area, lower_area = float(np.mean(bright)), float(np.mean(lower))
    note = f"；按原始基线分区，亮区占比 {bright_area:.1%}、中暗区占比 {lower_area:.1%}"
    if bright_area <= .25 or lower_area <= .15:
        return False, note + "，不足以确认后续亮区恢复"

    def mean_ratio(region: np.ndarray) -> float:
        return float(np.mean(rendered.luma[region]) / max(float(np.mean(previous.luma[region])), 1e-6))

    def p20_ratio(region: np.ndarray) -> float:
        return float(np.percentile(rendered.luma[region], 20)
                     / max(float(np.percentile(previous.luma[region], 20)), 1e-6))

    lower_mean, lower_p20 = mean_ratio(lower), p20_ratio(lower)
    nonbright_mean, nonbright_p20 = mean_ratio(nonbright), p20_ratio(nonbright)
    bright_mean = mean_ratio(bright)
    # Count newly clipped pixels, not just net coverage: restoring one region
    # must not conceal new clipping elsewhere.
    added_black = float(np.mean((np.max(rendered.rgb, axis=-1) <= 5.0 / 255.0)
                                & (np.max(previous.rgb, axis=-1) > 5.0 / 255.0)))
    added_highlight = float(np.mean((rendered.luma > .98) & (previous.luma <= .98)))
    preserved = (
        min(lower_mean, lower_p20, nonbright_mean, nonbright_p20) >= .90
        and bright_mean < .8 * min(1.0, lower_mean, nonbright_mean)
        and added_black <= .005 and added_highlight <= .005
    )
    note += (f"；相对已接受版本：中暗区均值/P20 {lower_mean:.1%}/{lower_p20:.1%}，"
             f"非亮区均值/P20 {nonbright_mean:.1%}/{nonbright_p20:.1%}，亮区均值 {bright_mean:.1%}，"
             f"新增暗部裁切 {added_black:.2%}、新增高光裁切 {added_highlight:.2%}；"
             + ("符合仅继续压缩亮区的条件" if preserved else "未满足后续亮区恢复的保护条件"))
    return preserved, note


def correction_for_report(report: LightroomQualityReport, *, strength: float = 1.0) -> Recipe:
    """Create a bounded delta recipe that addresses only observed regressions."""
    strength = max(0.35, min(1.0, strength))
    recipe = Recipe()
    hsl: dict[str, HueShift] = {}

    def hue(name: str, *, sat: float = 0.0, lum: float = 0.0) -> None:
        current = hsl.get(name, HueShift())
        hsl[name] = HueShift(sat=current.sat + sat * strength, lum=current.lum + lum * strength)

    for issue in report.issues:
        if issue.advisory:
            continue
        scale = strength * max(0.55, min(1.25, issue.severity))
        if issue.code == "shadow_clipping":
            recipe.shadows += 0.07 * scale
            recipe.blacks += 0.10 * scale
            recipe.contrast -= 0.04 * scale
        elif issue.code == "highlight_clipping":
            recipe.highlights -= 0.12 * scale
            recipe.whites -= 0.08 * scale
            recipe.exposure_ev -= 0.05 * scale
        elif issue.code == "oversaturation":
            recipe.vibrance -= 0.12 * scale
            recipe.saturation -= 0.045 * scale
        elif issue.code == "green_oversaturation":
            hue("green", sat=-0.16 * scale, lum=-0.025 * scale)
            hue("yellow", sat=-0.07 * scale)
            recipe.vibrance -= 0.035 * scale
        elif issue.code == "skin_oversaturation":
            hue("orange", sat=-0.10 * scale, lum=0.025 * scale)
            hue("red", sat=-0.055 * scale)
        elif issue.code == "skin_washed_out":
            hue("orange", sat=0.07 * scale, lum=0.02 * scale)
            hue("red", sat=0.035 * scale)
        elif issue.code == "sky_oversaturation":
            hue("blue", sat=-0.11 * scale, lum=0.025 * scale)
            hue("aqua", sat=-0.08 * scale)
        elif issue.code == "exposure_lift_unresolved":
            recipe.exposure_ev += 0.12 * scale
            recipe.shadows += 0.04 * scale
        elif issue.code == "too_dark":
            recipe.exposure_ev += 0.10 * scale
            recipe.shadows += 0.06 * scale
        elif issue.code == "too_bright":
            recipe.exposure_ev -= 0.10 * scale
            recipe.highlights -= 0.06 * scale
        elif issue.code == "excessive_distance":
            recipe.contrast -= 0.035 * scale
            recipe.vibrance -= 0.035 * scale
            recipe.saturation -= 0.02 * scale

    recipe.hsl = hsl
    return recipe.clamped()


def _midtone_support(points: list[tuple[float, float]], width: float) -> list[tuple[float, float]]:
    """Give sparse curves a midpoint to move, within the native point budget."""
    px, py = np.array(points, dtype=float).T
    xs = np.unique(np.r_[px, .5 - width, .5, .5 + width])
    if len(xs) > 32:
        return points  # Dense curves already occupy the renderer's point budget.
    return list(zip(xs.tolist(), np.interp(xs, px, py).tolist()))


def separate_candidate_recipe(recipe: Recipe, *, slot: str, attempt: int, exposure_locked: bool = False,
                              method: str | None = None, technical: TechnicalReport | None = None,
                              primary: Recipe | None = None) -> Recipe:
    """Develop a safe candidate's own light design; never borrow another's curve.

    Work from the effective recipe after stage rollbacks. Keep black/white
    endpoints and native color intact, vary midtones, and compensate subject
    exposure so that a deeper B does not simply darken a person's face.
    """
    step = min(max(attempt + 1, 1), 3)
    if method is not None and technical is not None:
        result = recipe.model_copy(deep=True)
        # Candidate distance is not a reason to drain more blue/green or add
        # a stronger cast. Develop the existing midtone design, preserving
        # black/white anchors, palette and local exposure.
        direction = 1 if slot == "C" or technical.mean_luma < .12 else -1
        points = result.curve.rgb or [(0, 0), (.18, .18), (.5, .5), (.8, .8), (1, 1)]
        if method.startswith("preset:"):
            midpoint = float(np.interp(.5, *np.array(points).T))
            direction = 1 if midpoint >= .5 or technical.mean_luma < .12 else -1
        recovery = primary is not None and not primary.curve.is_identity()
        if recovery:
            # A corrected photo has a midpoint below identity by design. Measure
            # this look's density relative to its corrected base, not to .5.
            base_mid = float(np.interp(.5, *np.array(primary.curve.rgb).T))
            direction = 1 if float(np.interp(.5, *np.array(points).T)) >= base_mid else -1
        amplitude = .025 if recovery else .012
        points = _midtone_support(points, .3)
        shaped = [(x, float(np.clip(y + direction * amplitude * step * max(0, 1-abs(x-.5)/.3), 0, 1)))
                  for x, y in points]
        ys = np.maximum.accumulate([y for _, y in shaped])
        result.curve.rgb = [(x, float(y)) for (x, _), y in zip(shaped, ys)]
        return result.clamped()
    direction = -1 if slot == "B" else 1
    exposure = 0.0 if exposure_locked else direction * .08 * step
    points = recipe.curve.rgb or [(0, 0), (.18, .18), (.5, .5), (.78, .78), (1, 1)]
    points = _midtone_support(points, .35)
    shaped = []
    for x, y in points:
        midtone = max(0.0, 1.0 - abs(x - .5) / .35)
        shaped.append((x, float(np.clip(y + direction * .02 * step * midtone, 0, 1))))
    ys = np.maximum.accumulate([y for _, y in shaped])
    curve = recipe.curve.model_copy(update={"rgb": [(x, float(y)) for (x, _), y in zip(shaped, ys)]})
    masks = [m.model_copy(deep=True) for m in recipe.masks]
    if slot == "B":
        for mask in masks:
            if mask.target == "subject":
                mask.exposure_ev = min(.75, mask.exposure_ev - exposure)
    return recipe.model_copy(update={"exposure_ev": recipe.exposure_ev + exposure,
                                     "curve": curve, "masks": masks}).clamped()


def blend_recipes(safe: Recipe, styled: Recipe, style_weight: float = 0.22) -> Recipe:
    """Build a replacement candidate: safe baseline plus a restrained style trace."""
    w = max(0.0, min(0.70, style_weight))
    data: dict[str, object] = {}
    scalar_fields = (
        "exposure_ev",
        "temperature",
        "tint",
        "highlights",
        "shadows",
        "whites",
        "blacks",
        "contrast",
        "saturation",
        "vibrance",
        "texture",
        "clarity",
        "dehaze",
        "vignette",
        "grain",
    )
    for name in scalar_fields:
        data[name] = getattr(safe, name) * (1.0 - w) + getattr(styled, name) * w

    hsl: dict[str, HueShift] = {}
    for name in set(safe.hsl) | set(styled.hsl):
        a = safe.hsl.get(name, HueShift())
        b = styled.hsl.get(name, HueShift())
        hsl[name] = HueShift(
            hue=a.hue * (1.0 - w) + b.hue * w,
            sat=a.sat * (1.0 - w) + b.sat * w,
            lum=a.lum * (1.0 - w) + b.lum * w,
        )
    data["hsl"] = hsl
    # Use the proven-safe curve. Scalar/HSL moves retain a visible style trace
    # without inheriting a curve that may have caused clipping.
    data["curve"] = safe.curve.model_copy(deep=True)
    data["parametric_tone"] = safe.parametric_tone.model_copy(deep=True)
    data["detail"] = safe.detail.model_copy(deep=True)
    data["optics"] = safe.optics.model_copy(deep=True)
    data["finish"] = safe.finish.model_copy(deep=True)
    data["crop"] = safe.crop.model_copy(deep=True)
    data["grade"] = safe.grade.model_copy(deep=True)
    wheels = safe.color_grading.model_copy(deep=True)
    for band in ("shadow", "midtone", "highlight"):
        a, b = safe.color_grading, styled.color_grading
        a_sat, b_sat = getattr(a, band + "_sat"), getattr(b, band + "_sat")
        if a_sat is not None or b_sat is not None:
            a_angle = np.deg2rad(getattr(a, band + "_hue") or 0)
            b_angle = np.deg2rad(getattr(b, band + "_hue") or 0)
            x = (a_sat or 0) * (1-w) * np.cos(a_angle) + (b_sat or 0) * w * np.cos(b_angle)
            y = (a_sat or 0) * (1-w) * np.sin(a_angle) + (b_sat or 0) * w * np.sin(b_angle)
            setattr(wheels, band + "_sat", float(np.hypot(x, y)))
            if x or y:
                setattr(wheels, band + "_hue", float(np.rad2deg(np.arctan2(y, x)) % 360))
        a_lum, b_lum = getattr(a, band + "_lum"), getattr(b, band + "_lum")
        if a_lum is not None or b_lum is not None:
            setattr(wheels, band + "_lum", (a_lum or 0) * (1-w) + (b_lum or 0) * w)
    for key, default in (("blending", 50), ("balance", 0)):
        a, b = getattr(safe.color_grading, key), getattr(styled.color_grading, key)
        if a is not None or b is not None:
            setattr(wheels, key, (default if a is None else a) * (1-w) + (default if b is None else b) * w)
    data["color_grading"] = safe.color_grading.model_copy(deep=True) if not w else wheels
    data["local"] = safe.local.model_copy(deep=True)
    # A failed local stage cannot return at full strength just because the
    # safe version has no mask. Preserve only selections proven in the safe pass.
    data["masks"] = [m.model_copy(deep=True) for m in safe.masks]
    return Recipe.model_validate(data).clamped()


def _changes_framing(recipe: Recipe) -> bool:
    """Fixed-pixel neutral comparisons require matching crop and perspective."""
    crop_defaults = {"top": 0, "left": 0, "bottom": 1, "right": 1, "angle": 0}
    optics_defaults = {"manual_distortion": 0, "perspective_vertical": 0,
                       "perspective_horizontal": 0, "perspective_rotate": 0,
                       "perspective_scale": 100, "perspective_aspect": 0, "perspective_upright": 0}
    return any(value is not None and abs(value-default) > 1e-6
               for controls, defaults in ((recipe.crop, crop_defaults), (recipe.optics, optics_defaults))
               for key, default in defaults.items() for value in [getattr(controls, key)])


def needs_color_repair(report: LightroomQualityReport) -> bool:
    """Use color-only repair once other technical damage has been resolved."""
    return (any(issue.code == "neutral_color_cast" for issue in report.issues)
            and not any(issue.hard and not issue.advisory and issue.code != "neutral_color_cast"
                        for issue in report.issues))


def reduce_color_cast(recipe: Recipe, primary: Recipe, amount: float = .5) -> Recipe:
    """Reduce chromatic styling while retaining this candidate's light design.

    Channel curves move toward their shared density, preserving the base
    correction's channel differences. This is a fresh-source absolute recipe.
    """
    if recipe == primary:
        return recipe.model_copy(deep=True)
    amount = float(np.clip(amount, 0, .70))
    result = recipe.model_copy(deep=True)
    colors = blend_recipes(primary, recipe, style_weight=amount)
    for key in ("temperature", "tint", "saturation", "vibrance"):
        setattr(result, key, getattr(colors, key))
    for name, shift in result.hsl.items():
        mixed = colors.hsl[name]
        shift.hue, shift.sat = mixed.hue, mixed.sat
    for band in ("shadows", "midtones", "highlights"):
        a, b = getattr(primary.grade, band), getattr(recipe.grade, band)
        setattr(result.grade, band, tuple(x + amount * (y-x) for x, y in zip(a, b)))
    for band in ("shadow", "midtone", "highlight"):
        for field_name in ("hue", "sat"):
            key = band + "_" + field_name
            setattr(result.color_grading, key, getattr(colors.color_grading, key))
    base_masks = {mask.target: mask for mask in primary.masks}
    for mask in result.masks:
        base = base_masks.get(mask.target)
        for key in ("temperature", "tint", "saturation"):
            a = getattr(base, key) if base is not None else 0
            setattr(mask, key, a + amount * (getattr(mask, key)-a))

    if any(getattr(recipe.curve, c) or getattr(primary.curve, c) for c in "rgb"):
        points = lambda curve, c: np.array(getattr(curve, c) or [(0, 0), (1, 1)], dtype=float).T
        knots = np.unique(np.r_[0., 1., *[points(curve, c)[0] for curve in (recipe.curve, primary.curve) for c in "rgb"]])
        if len(knots) > 32:
            knots = np.linspace(0, 1, 32)
        full = np.array([np.interp(knots, *points(recipe.curve, c)) for c in "rgb"])
        base = np.array([np.interp(knots, *points(primary.curve, c)) for c in "rgb"])
        weights = np.array([.2126, .7152, .0722])
        common, base_common = weights @ full, weights @ base
        for i, channel in enumerate("rgb"):
            target = common + base[i] - base_common
            values = np.clip(target + amount * (full[i]-target), 0, 1)
            # Each curve is a convex blend of monotone channel functions when
            # the base is neutral. Keep imported unusual bases monotone too.
            values = np.maximum.accumulate(values)
            setattr(result.curve, channel, list(zip(knots.tolist(), values.tolist())))
    return result.clamped()


def _read_pixels(path: Path, *, max_edge: int = 960, size: tuple[int, int] | None = None) -> _PixelSample:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = image_to_srgb(ImageOps.exif_transpose(image))
        if size is None:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        elif image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    linear = srgb_to_linear(rgb)
    luma = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hue, saturation, value = _rgb_to_hsv(rgb)
    return _PixelSample(rgb=rgb, luma=luma, hue=hue, saturation=saturation, value=value)


def _content_masks(sample: _PixelSample) -> dict[str, np.ndarray]:
    hue = sample.hue * 360.0
    sat = sample.saturation
    val = sample.value
    h, w = hue.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    green = (hue >= 48.0) & (hue <= 172.0) & (sat >= 0.16) & (val >= 0.07)
    skin = (hue >= 5.0) & (hue <= 48.0) & (sat >= 0.14) & (sat <= 0.82) & (val >= 0.16)
    sky = (yy <= 0.62) & (hue >= 168.0) & (hue <= 265.0) & (sat >= 0.14) & (val >= 0.13)
    red = ((hue <= 22.0) | (hue >= 342.0)) & (sat >= 0.32) & (val >= 0.12)
    return {"green": green, "skin": skin, "sky": sky, "red": red}


def _metrics(sample: _PixelSample, masks: dict[str, np.ndarray]) -> LightroomPixelMetrics:
    def area(name: str) -> float:
        return float(np.mean(masks[name]))

    def masked_sat(name: str) -> float:
        mask = masks[name]
        return float(np.mean(sample.saturation[mask])) if np.any(mask) else 0.0

    return LightroomPixelMetrics(
        mean_luma=float(np.mean(sample.luma)),
        luma_std=float(np.std(sample.luma)),
        # linear 0.02 is approximately sRGB 39/255: a normal dark tone,
        # not lost detail. Require all output channels to be near black.
        shadow_clip=float(np.mean(np.max(sample.rgb, axis=-1) <= 5.0 / 255.0)),
        highlight_clip=float(np.mean(sample.luma > 0.98)),
        saturation_mean=float(np.mean(sample.saturation)),
        high_saturation=float(np.mean(sample.saturation > 0.65)),
        green_fraction=area("green"),
        green_saturation=masked_sat("green"),
        skin_fraction=area("skin"),
        skin_saturation=masked_sat("skin"),
        sky_fraction=area("sky"),
        sky_saturation=masked_sat("sky"),
        red_fraction=area("red"),
        low_tone_fraction=float(np.mean(sample.luma < 0.02)),
    )


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    delta = maximum - minimum
    saturation = np.where(maximum > 1e-6, delta / np.maximum(maximum, 1e-6), 0.0)
    hue = np.zeros_like(maximum)
    chromatic = delta > 1e-6
    red = (maximum == rgb[..., 0]) & chromatic
    green = (maximum == rgb[..., 1]) & chromatic
    blue = (maximum == rgb[..., 2]) & chromatic
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6.0
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
    return hue / 6.0, saturation, maximum


def _perceptual_distance(a: np.ndarray, b: np.ndarray) -> float:
    lab_a = _rgb_to_lab(a)
    lab_b = _rgb_to_lab(b)
    return float(np.mean(np.linalg.norm(lab_a - lab_b, axis=-1)))


# A natural-colour version fails when the near-neutrals of the original move this far (median a*b*)
# and this much of that is new colour rather than less of the colour they had.
NEUTRAL_SHIFT, NEUTRAL_INTRODUCED = 3.0, 2.5


def neutral_cast(before_rgb: np.ndarray, after_rgb: np.ndarray) -> dict[str, float]:
    """How far the near-neutral light pixels of ``before_rgb`` moved in colour in ``after_rgb``.

    Both are sRGB 0..1 arrays of one size. Fixed source coordinates identify measured near-neutrals,
    not snow, clothing or any other material. Existing colored light is not a target.
    """
    before_lab, after_lab = _rgb_to_lab(before_rgb), _rgb_to_lab(after_rgb)
    neutral = ((before_lab[..., 0] > 55) & (before_lab[..., 0] < 98)
               & (np.linalg.norm(before_lab[..., 1:], axis=-1) < 6))
    fraction = float(np.mean(neutral))
    out: dict[str, float] = {"fraction": fraction}
    if fraction < .005:
        return out
    before_ab, after_ab = before_lab[..., 1:][neutral], after_lab[..., 1:][neutral]
    delta = after_ab - before_ab
    # Distance from the original color-to-neutral segment allows pure
    # desaturation, but catches a blue-to-green swap at equal chroma.
    projection = np.clip(np.sum(after_ab * before_ab, axis=-1)
                         / np.maximum(np.sum(before_ab ** 2, axis=-1), 1e-6), 0, 1)
    out.update(median_shift=float(np.linalg.norm(np.median(delta, axis=0))),
               chroma_before=float(np.median(np.linalg.norm(before_ab, axis=-1))),
               chroma_after=float(np.median(np.linalg.norm(after_ab, axis=-1))),
               introduced_chroma=float(np.median(np.linalg.norm(after_ab - before_ab * projection[:, None], axis=-1))),
               delta_a=float(np.median(delta[:, 0])), delta_b=float(np.median(delta[:, 1])))
    return out


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def _severity(value: float, limit: float, span: float) -> float:
    return float(max(0.2, min(1.5, (value - limit) / max(span, 1e-6) + 0.35)))
