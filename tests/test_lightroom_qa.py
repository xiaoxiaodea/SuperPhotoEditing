from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from superphoto.color import linear_to_srgb
from superphoto.lightroom_qa import (
    assess_lightroom_render,
    blend_recipes,
    correction_for_report,
    is_distinct_candidate,
    separate_candidate_recipe,
)
from superphoto.models import ColorGradingControls, HueShift, MaskAdjustment, Recipe, ToneCurve


def _save_scene(path: Path, *, green_scale: float = 1.0, shadow_scale: float = 1.0) -> Path:
    h, w = 180, 260
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 0.18 + 0.30 * xx / w
    rgb[..., 1] = (0.25 + 0.48 * yy / h) * green_scale
    rgb[..., 2] = 0.10 + 0.14 * xx / w
    rgb[yy < 42] = np.array([0.42, 0.57, 0.76], dtype=np.float32)
    rgb[(xx - 120) ** 2 + (yy - 105) ** 2 < 900] = np.array([0.58, 0.38, 0.29], dtype=np.float32)
    rgb[:55, :60] *= shadow_scale
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path, quality=95)
    return path


@pytest.mark.parametrize("exposure_locked", [False, True])
def test_separation_uses_effective_curve_and_preserves_subject_exposure_and_color(exposure_locked):
    base = Recipe(masks=[MaskAdjustment(target="subject", exposure_ev=.2),
                         MaskAdjustment(target="background", exposure_ev=-.25)],
                  color_grading=ColorGradingControls(shadow_hue=220, shadow_sat=6))
    changed = separate_candidate_recipe(base, slot="B", attempt=2, exposure_locked=exposure_locked)
    assert changed.curve.rgb[0] == (0, 0) and changed.curve.rgb[-1] == (1, 1)
    assert changed.curve.rgb[2][1] < .5
    assert changed.exposure_ev + changed.masks[0].exposure_ev == pytest.approx(.2)
    if exposure_locked:
        assert changed.exposure_ev == 0
        assert changed.masks[0].exposure_ev == .2
    assert changed.color_grading == base.color_grading
    assert changed.masks[1] == base.masks[1]
    assert base.curve.rgb == [] and base.masks[0].exposure_ev == .2


def test_lightroom_pixel_gate_accepts_restrained_render(tmp_path: Path) -> None:
    baseline = _save_scene(tmp_path / "baseline.jpg")
    rendered = _save_scene(tmp_path / "rendered.jpg", green_scale=0.98, shadow_scale=1.0)

    report = assess_lightroom_render(baseline, rendered, slot="A")

    assert report.accepted is True
    assert report.score >= 82
    assert report.issues == []


def test_low_key_tone_shaping_is_not_misreported_as_black_clipping(tmp_path):
    before = np.broadcast_to(np.linspace(42, 50, 120, dtype=np.uint8)[None, :, None], (80, 120, 3)).copy()
    after = before - 7
    source, output = tmp_path / "low-key.png", tmp_path / "shaped.png"
    Image.fromarray(before).save(source)
    Image.fromarray(after).save(output)
    report = assess_lightroom_render(source, output, slot="B")
    assert report.metrics.low_tone_fraction > .4
    assert report.metrics.shadow_clip == 0
    assert report.accepted


def _save_luma(path: Path, luma: np.ndarray) -> Path:
    gray = np.rint(np.clip(linear_to_srgb(luma), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(np.repeat(gray[..., None], 3, axis=-1)).save(path)
    return path


@pytest.fixture
def highlight_recovery_scene(tmp_path):
    baseline = np.empty((120, 160), dtype=np.float32)
    baseline[:50] = np.linspace(.90, .96, 160)
    baseline[50:] = np.linspace(.04, .16, 160)
    bright = baseline > .6
    previous = baseline.copy()
    previous[bright] *= .65
    previous[~bright] *= .60
    baseline_path = _save_luma(tmp_path / "baseline.png", baseline)
    previous_path = _save_luma(tmp_path / "accepted.png", previous)
    assert assess_lightroom_render(baseline_path, previous_path, slot="B").accepted
    return baseline_path, previous_path, previous, bright


def test_followup_highlight_recovery_preserves_previously_accepted_dark_ground(tmp_path, highlight_recovery_scene):
    baseline, previous_path, previous, bright = highlight_recovery_scene
    candidate = previous * .96
    candidate[bright] = previous[bright] * .55
    output = _save_luma(tmp_path / "highlight-recovery.png", candidate)

    report = assess_lightroom_render(baseline, output, slot="B", previous_accepted_path=previous_path)
    issue = next(issue for issue in report.issues if issue.code == "too_dark")
    assert report.accepted and report.score == 100
    assert issue.advisory and not issue.hard
    assert "相对已接受版本" in issue.note and "非亮区均值/P20" in issue.note
    assert "亮区均值" in issue.note and "新增暗部裁切" in issue.note and "新增高光裁切" in issue.note
    assert correction_for_report(report) == Recipe()


def test_followup_uniform_dimming_is_not_highlight_recovery(tmp_path, highlight_recovery_scene):
    baseline, previous_path, previous, _ = highlight_recovery_scene
    output = _save_luma(tmp_path / "uniform.png", previous * .55)
    report = assess_lightroom_render(baseline, output, slot="B", previous_accepted_path=previous_path)
    assert not report.accepted
    assert any(issue.code == "too_dark" and issue.hard and not issue.advisory for issue in report.issues)


def test_followup_ground_crush_cannot_hide_behind_preserved_mean(tmp_path, highlight_recovery_scene):
    baseline, previous_path, previous, bright = highlight_recovery_scene
    candidate = previous.copy()
    candidate[bright] *= .55
    crushed = (~bright) & (np.indices(previous.shape)[1] < 40)
    lifted = (~bright) & ~crushed
    lost_light = float(np.sum(candidate[crushed]) * .4)
    candidate[crushed] *= .6
    candidate[lifted] += lost_light / np.count_nonzero(lifted)
    assert np.mean(candidate[~bright]) == pytest.approx(np.mean(previous[~bright]))
    assert np.percentile(candidate[~bright], 20) < np.percentile(previous[~bright], 20) * .9

    output = _save_luma(tmp_path / "crushed-ground.png", candidate)
    report = assess_lightroom_render(baseline, output, slot="B", previous_accepted_path=previous_path)
    assert not report.accepted
    assert any(issue.code == "too_dark" and issue.hard for issue in report.issues)


def test_no_previous_accepted_render_keeps_original_darkness_rule(tmp_path, highlight_recovery_scene):
    baseline, _, previous, bright = highlight_recovery_scene
    candidate = previous * .96
    candidate[bright] = previous[bright] * .55
    output = _save_luma(tmp_path / "no-predecessor.png", candidate)
    report = assess_lightroom_render(baseline, output, slot="B")
    explicit_none = assess_lightroom_render(baseline, output, slot="B", previous_accepted_path=None)
    assert report.to_dict() == explicit_none.to_dict()
    assert not report.accepted
    assert any(issue.code == "too_dark" and issue.hard and not issue.advisory for issue in report.issues)


@pytest.mark.parametrize("clipping", ["black", "highlight"])
def test_followup_recovery_cannot_add_clipping_below_original_global_limits(tmp_path, highlight_recovery_scene, clipping):
    baseline, previous_path, previous, bright = highlight_recovery_scene
    candidate = previous.copy()
    candidate[bright] *= .55
    if clipping == "black":
        candidate[50:56, :32] = 0
    else:
        candidate[:6, :32] = 1
    output = _save_luma(tmp_path / f"added-{clipping}.png", candidate)
    report = assess_lightroom_render(baseline, output, slot="B", previous_accepted_path=previous_path)
    assert not report.accepted
    assert not any(issue.code in {"shadow_clipping", "highlight_clipping"} for issue in report.issues)
    assert any(issue.code == "too_dark" and issue.hard for issue in report.issues)


def test_followup_uniform_dimming_cannot_use_legacy_highlight_exception(tmp_path):
    baseline = np.empty((120, 160), dtype=np.float32)
    baseline[:50], baseline[50:] = .92, .12
    previous = baseline.copy()
    previous[:50] *= .5
    previous[50:] *= 1.1
    before = _save_luma(tmp_path / "bright-baseline.png", baseline)
    accepted = _save_luma(tmp_path / "relit-ground.png", previous)
    output = _save_luma(tmp_path / "uniform-five-percent.png", previous * .95)
    assert assess_lightroom_render(before, accepted, slot="B").accepted
    assert assess_lightroom_render(before, output, slot="B").accepted
    report = assess_lightroom_render(before, output, slot="B", previous_accepted_path=accepted)
    assert not report.accepted
    assert any(issue.code == "too_dark" and issue.hard for issue in report.issues)


def test_dark_source_separation_preserves_palette_and_shadow_anchors():
    from test_celebrity_styles import _tech
    base = Recipe(curve=ToneCurve(rgb=[(0, 0), (.18, .18), (.5, .52), (1, 1)]),
                  hsl={"aqua": HueShift(sat=-.62)}, masks=[MaskAdjustment(target="subject", exposure_ev=.1)])
    changed = separate_candidate_recipe(base, slot="B", attempt=1, method="graphic",
                                        technical=_tech(mean_luma=.08, shadow_clip=.22, aqua_fraction=.27))
    assert changed.exposure_ev == base.exposure_ev
    assert changed.curve.rgb[0] == base.curve.rgb[0] and changed.curve.rgb[1] == base.curve.rgb[1]
    assert np.interp(.5, *np.array(changed.curve.rgb).T) > np.interp(.5, *np.array(base.curve.rgb).T)
    assert changed.masks == base.masks
    assert changed.hsl == base.hsl and changed.color_grading == base.color_grading


def test_styled_palette_allows_moderate_chroma_but_keeps_saturation_and_skin_limits(tmp_path):
    import colorsys
    paths = []
    for name, saturation in (("source", .18), ("warm", .28), ("neon", .85)):
        path = tmp_path / f"{name}.png"
        rgb = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(.10, saturation, .65))
        Image.new("RGB", (100, 100), rgb).save(path)
        paths.append(path)
    source, warm, neon = paths
    assert not assess_lightroom_render(source, warm, slot="A", creative_palette=True).accepted
    assert assess_lightroom_render(source, warm, slot="C", creative_palette=True).accepted
    assert not assess_lightroom_render(source, neon, slot="C", creative_palette=True).accepted
    skin = assess_lightroom_render(source, warm, slot="C", creative_palette=True,
                                   region_masks={"skin": np.ones((100, 100))})
    assert not skin.accepted
    assert any(issue.code == "skin_oversaturation" and issue.hard for issue in skin.issues)


def test_lightroom_pixel_gate_flags_green_and_clipping(tmp_path: Path) -> None:
    baseline = _save_scene(tmp_path / "baseline.jpg")
    # Saturate greens and crush one region to simulate a Lightroom regression.
    with Image.open(baseline) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr[..., 1] = np.clip(arr[..., 1] * 1.45, 0, 1)
    arr[:90, :130] *= 0.02
    rendered = tmp_path / "rendered.jpg"
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB").save(rendered, quality=95)

    report = assess_lightroom_render(baseline, rendered, slot="C")
    codes = {issue.code for issue in report.issues}
    correction = correction_for_report(report)

    assert report.accepted is False
    assert "shadow_clipping" in codes
    assert "green_oversaturation" in codes or "oversaturation" in codes
    assert correction.shadows > 0 or correction.blacks > 0
    assert correction.vibrance < 0 or correction.saturation < 0 or correction.hsl["green"].sat < 0


def test_candidate_distinctness_rejects_duplicate_and_accepts_visible_branch(tmp_path: Path) -> None:
    first = _save_scene(tmp_path / "first.jpg")
    duplicate = _save_scene(tmp_path / "duplicate.jpg", green_scale=0.99)
    branch = _save_scene(tmp_path / "branch.jpg", green_scale=1.35, shadow_scale=0.72)

    duplicate_ok, duplicate_distance = is_distinct_candidate(duplicate, [first], min_distance=2.8)
    branch_ok, branch_distance = is_distinct_candidate(branch, [first], min_distance=2.8)

    assert duplicate_ok is False
    assert duplicate_distance is not None and duplicate_distance < 2.8
    assert branch_ok is True
    assert branch_distance is not None and branch_distance >= 2.8


def test_blue_mountain_is_not_confirmed_sky_damage_without_semantics(tmp_path):
    baseline = np.full((100, 150, 3), 180, dtype=np.uint8)
    baseline[30:60, 20:120] = [80, 120, 160]
    rendered = baseline.copy()
    rendered[30:60, 20:120] = [60, 120, 160]
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    Image.fromarray(baseline).save(before)
    Image.fromarray(rendered).save(after)
    report = assess_lightroom_render(before, after, slot="C")
    assert report.accepted
    assert any(i.code == "blue_palette_saturation" and i.advisory for i in report.issues)
    assert correction_for_report(report).hsl == {}
    true_sky = np.zeros((100, 150), dtype=np.float32)
    true_sky[:20] = 1
    report = assess_lightroom_render(before, after, slot="C", region_masks={"sky": true_sky})
    assert report.accepted and not report.issues
    true_sky[:] = 0
    true_sky[30:60, 20:120] = 1
    report = assess_lightroom_render(before, after, slot="C", region_masks={"sky": true_sky})
    assert not report.accepted and any(i.code == "sky_oversaturation" and i.hard for i in report.issues)


def test_blend_recipes_keeps_safe_curve_and_restrained_style() -> None:
    safe = Recipe(
        exposure_ev=0.1,
        vibrance=-0.04,
        hsl={"green": HueShift(sat=-0.08)},
        curve=ToneCurve(rgb=[(0.0, 0.0), (1.0, 1.0)]),
    )
    styled = Recipe(
        exposure_ev=0.5,
        vibrance=0.4,
        hsl={"green": HueShift(hue=-20, sat=0.5)},
        curve=ToneCurve(rgb=[(0.0, 0.1), (0.5, 0.7), (1.0, 0.9)]),
    )

    result = blend_recipes(safe, styled, style_weight=0.2)

    assert result.exposure_ev == pytest.approx(0.18, abs=0.01)
    assert result.vibrance == pytest.approx(0.048, abs=0.01)
    assert result.hsl["green"].hue == -4.0
    assert result.hsl["green"].sat == pytest.approx(0.036)
    assert result.curve == safe.curve
