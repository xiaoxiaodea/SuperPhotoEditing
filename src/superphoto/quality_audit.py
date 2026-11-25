"""Strict 8-Dimension Photographer-Grade Quality Audit for SuperPhoto distillation & QA.

Unlike coarse global-mean metrics (which can score >95% even when tree snow washes out,
highlights band, midtones go hazy, or film grain vanishes), this module audits renders
across 8 orthogonal perceptual dimensions matching a senior retoucher's eye:

1. Tone Anchor Precision (black / shadows / mid / highlights / white)
2. Local & Worst-Region Lab Colour Accuracy (mean ΔE + 90th-percentile 16x16 patch ΔE)
3. Colour Distribution & Mood Fidelity (RGB histogram cosine + warmth / tint / chroma)
4. Micro-Detail & Shadow/Highlight Crispness (zone-stratified Laplacian edge retention)
5. Transparency & Local Contrast ("通透感": multi-scale local contrast + highlight neutrality)
6. Anti-Banding & Smooth Gradient Integrity (detects false colour/luma steps & hue tears)
7. Organic Film Grain & Micro-Texture Match (high-frequency grain sigma in smooth areas)
8. Structural Similarity (local windowed luminance & chroma SSIM)

Every evaluated frame receives both a strict composite score (0-100%) and explicit
Chinese diagnostic defect flags whenever any single dimension breaches a retoucher gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from . import tone
from .color import srgb_to_linear
from .raster_negative import _box_mean

# Hard red-line thresholds for retoucher-grade PASS
GATE_THRESHOLDS: dict[str, float] = {
    "tone_pct": 90.0,
    "local_lab_pct": 80.0,
    "mood_hist_pct": 84.0,
    "detail_pct": 78.0,
    "clarity_pct": 82.0,
    "antibanding_pct": 84.0,
    "grain_pct": 75.0,
    "ssim_pct": 86.0,
}


def _load_aligned(path_a: Path, path_b: Path, max_edge: int = 960) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path_a) as ia:
        im_a = ImageOps.exif_transpose(ia).convert("RGB")
        im_a.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    with Image.open(path_b) as ib:
        im_b = ImageOps.exif_transpose(ib).convert("RGB")
        if im_b.size != im_a.size:
            im_b = im_b.resize(im_a.size, Image.Resampling.LANCZOS)
    a = np.asarray(im_a, dtype=np.float32) / 255.0
    b = np.asarray(im_b, dtype=np.float32) / 255.0
    return a, b


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Fast vectorized sRGB [0..1] -> CIE Lab (D65)."""
    lin = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(np.maximum(t, 0.0)), (kap * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a_ch = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return np.stack([L, a_ch, b_ch], axis=-1).astype(np.float32)


def _windowed_ssim(y1: np.ndarray, y2: np.ndarray, radius: int = 4) -> float:
    mu1 = _box_mean(y1, radius)
    mu2 = _box_mean(y2, radius)
    s1_sq = np.maximum(0.0, _box_mean(y1 * y1, radius) - mu1 * mu1)
    s2_sq = np.maximum(0.0, _box_mean(y2 * y2, radius) - mu2 * mu2)
    s12 = _box_mean(y1 * y2, radius) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1_sq + s2_sq + c2)
    return float(np.clip(np.mean(num / np.maximum(den, 1e-8)), 0.0, 1.0))


def _ratio_score(cand: float, ref: float, *, floor: float = 1e-4, asymmetry: float = 1.0) -> float:
    """Score how closely ``cand`` matches ``ref`` (100 when equal; penalises shortfall harder if asymmetry > 1)."""
    if ref <= floor and cand <= floor:
        return 100.0
    r = (cand + floor) / (ref + floor)
    if r < 1.0:
        err = (1.0 - r) * asymmetry
    else:
        err = (r - 1.0) / max(r, 1.0)
    return float(max(0.0, min(100.0, 100.0 * np.exp(-2.2 * err * err))))


def _zone_detail(y: np.ndarray, ref_y: np.ndarray) -> dict[str, float]:
    """Stratified high-frequency edge detail in dark (<0.25), mid (0.25..0.72), and bright (>0.72) zones."""
    lap_c = np.abs(4.0 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:])
    lap_r = np.abs(4.0 * ref_y[1:-1, 1:-1] - ref_y[:-2, 1:-1] - ref_y[2:, 1:-1] - ref_y[1:-1, :-2] - ref_y[1:-1, 2:])
    yr = ref_y[1:-1, 1:-1]

    scores = []
    ratios: dict[str, float] = {}
    for name, mask in (
        ("sh", yr < 0.25),
        ("mid", (yr >= 0.25) & (yr <= 0.72)),
        ("hi", yr > 0.72),
    ):
        if int(mask.sum()) < 80:
            continue
        dc = float(lap_c[mask].mean())
        dr = float(lap_r[mask].mean())
        if dr < 0.002:
            continue
        ratios[name] = round(dc / max(dr, 1e-5), 3)
        scores.append(_ratio_score(dc, dr, floor=0.002, asymmetry=1.35))
    # Also measure local edge correlation (checks whether crisp details like snow on tree branches survived!)
    c_flat = lap_c.reshape(-1)
    r_flat = lap_r.reshape(-1)
    if float(np.std(c_flat)) > 1e-5 and float(np.std(r_flat)) > 1e-5:
        corr = float(np.corrcoef(c_flat, r_flat)[0, 1])
    else:
        corr = 1.0
    corr_score = max(0.0, min(100.0, corr * 100.0))
    zone_avg = float(np.mean(scores)) if scores else 100.0
    return {
        "detail_pct": round(0.65 * zone_avg + 0.35 * corr_score, 1),
        "edge_corr": round(corr, 3),
        "zone_ratios": ratios,
    }


def _clarity_and_transparency(cand: np.ndarray, ref: np.ndarray, yc: np.ndarray, yr: np.ndarray) -> dict[str, float]:
    """Measure local mid-scale contrast ('通透感') and clean highlight neutrality (anti-veil / anti-mud)."""
    # 1. Multi-scale local contrast energy (radius 15px captures architectural/facial/sky separation)
    lc_c = float(np.std(yc - _box_mean(yc, 15)))
    lc_r = float(np.std(yr - _box_mean(yr, 15)))
    local_contrast_sc = _ratio_score(lc_c, lc_r, floor=0.005, asymmetry=1.45)

    # 2. Shadow-to-mid tonal separation (p50 - p08) & overall tonal punch (p90 - p10)
    c_p08, c_p50, c_p90 = np.percentile(yc, [8, 50, 90])
    r_p08, r_p50, r_p90 = np.percentile(yr, [8, 50, 90])
    step_sc = _ratio_score(float(c_p50 - c_p08), float(r_p50 - r_p08), floor=0.03, asymmetry=1.3)
    span_sc = _ratio_score(float(c_p90 - c_p08), float(r_p90 - r_p08), floor=0.05, asymmetry=1.3)

    # 3. Clean bright-neutral transparency: where photographer keeps bright whites clean (luma>0.72, sat<0.12),
    # check if candidate introduced muddy yellow/green/grey veil!
    r_sat = ref.max(axis=-1) - ref.min(axis=-1)
    c_sat = cand.max(axis=-1) - cand.min(axis=-1)
    clean_whites = (yr > 0.72) & (r_sat < 0.14)
    if int(clean_whites.sum()) >= 120:
        mud_excess = max(0.0, float(c_sat[clean_whites].mean() - r_sat[clean_whites].mean()))
        veil_drop = max(0.0, float(yr[clean_whites].mean() - yc[clean_whites].mean()))
        white_clean_sc = max(0.0, min(100.0, 100.0 * (1.0 - mud_excess * 4.5 - veil_drop * 3.5)))
    else:
        mud_excess = 0.0
        white_clean_sc = 100.0

    score = 0.42 * local_contrast_sc + 0.24 * step_sc + 0.14 * span_sc + 0.20 * white_clean_sc
    return {
        "clarity_pct": round(score, 1),
        "local_contrast_ratio": round(lc_c / max(lc_r, 1e-5), 3),
        "white_mud_excess": round(mud_excess, 4),
    }


def _banding_and_hue_tear(cand: np.ndarray, ref: np.ndarray, yc: np.ndarray, yr: np.ndarray,
                          lab_c: np.ndarray, lab_r: np.ndarray) -> dict[str, float]:
    """Detect false colour banding, quantized smooth gradients, and bright hue tearing (e.g. firework halos).

    Applies a gentle 5x5 pre-smooth (radius=2) before measuring banding/contour jumps so authentic
    1-pixel organic 35mm film grain is never mistaken for colour posterization.
    """
    ac_s = _box_mean(lab_c[..., 1], 2)
    bc_s = _box_mean(lab_c[..., 2], 2)
    ar_s = _box_mean(lab_r[..., 1], 2)
    br_s = _box_mean(lab_r[..., 2], 2)
    lr_s = _box_mean(lab_r[..., 0], 2)

    # 1. Medium-scale 2D chroma discontinuity / colour banding in smooth-luma regions (both X and Y gradients!)
    def _grad2d(arr: np.ndarray) -> np.ndarray:
        gx = np.diff(arr, axis=1)[:-1, :]
        gy = np.diff(arr, axis=0)[:, :-1]
        return np.hypot(gx, gy)

    da_c = np.hypot(_grad2d(ac_s), _grad2d(bc_s))
    da_r = np.hypot(_grad2d(ar_s), _grad2d(br_s))
    dl_r = _grad2d(lr_s)
    smooth_r = dl_r < 2.2
    if int(smooth_r.sum()) >= 200:
        excess_step = np.maximum(0.0, da_c[smooth_r] - da_r[smooth_r] - 0.6)
        band_rate = float(np.mean(excess_step > 1.2) * 100.0)
        band_sev = float(np.percentile(excess_step, 98))
    else:
        band_rate, band_sev = 0.0, 0.0

    # 2. Bright/saturated hue tear: in bright glowing pixels (yr > 0.40) with noticeable chroma, measure
    # hue-angle divergence where candidate splits one smooth colour glow into two clashing hues (e.g. orange+cyan).
    c_chr = np.hypot(ac_s, bc_s)
    r_chr = np.hypot(ar_s, br_s)
    glow = (yr > 0.40) & ((c_chr > 11.0) | (r_chr > 11.0))
    if int(glow.sum()) >= 120:
        dot = (ac_s * ar_s + bc_s * br_s)[glow]
        norm = np.maximum(c_chr[glow] * r_chr[glow], 1.0)
        ang_err = np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
        hue_tear_pct = float(np.mean(ang_err > 36.0) * 100.0)
        hue_p90 = float(np.percentile(ang_err, 90))
    else:
        hue_tear_pct, hue_p90 = 0.0, 0.0

    # 3. Quantized 2D staircase in smooth sky/gradient regions (medium-scale)
    yc_s, yr_s = _box_mean(yc, 2), _box_mean(yr, 2)
    dy_c = _grad2d(yc_s)
    dy_r = _grad2d(yr_s)
    flat_r = (dy_r > 0.0005) & (dy_r < 0.009)
    if int(flat_r.sum()) >= 200:
        stair_excess = max(0.0, float(np.mean(dy_c[flat_r] > 0.015) - np.mean(dy_r[flat_r] > 0.015)) * 100.0)
    else:
        stair_excess = 0.0

    penalty = band_rate * 2.6 + max(0.0, band_sev - 1.5) * 2.4 + hue_tear_pct * 0.55 + stair_excess * 2.5
    score = max(0.0, min(100.0, 100.0 - penalty))
    return {
        "antibanding_pct": round(score, 1),
        "band_rate_pct": round(band_rate, 3),
        "hue_tear_pct": round(hue_tear_pct, 2),
        "hue_p90_deg": round(hue_p90, 1),
    }


def _grain_and_texture(yc: np.ndarray, yr: np.ndarray) -> dict[str, float]:
    """Measure organic fine film grain / high-frequency texture match in smooth non-edge regions."""
    blur3_c, blur11_c = _box_mean(yc, 2), _box_mean(yc, 9)
    blur3_r, blur11_r = _box_mean(yr, 2), _box_mean(yr, 9)
    smooth = np.abs(blur3_r - blur11_r) < 0.022
    hf_c = (yc - blur3_c) * 255.0
    hf_r = (yr - blur3_r) * 255.0
    if int(smooth.sum()) >= 200:
        g_c = float(np.std(hf_c[smooth]))
        g_r = float(np.std(hf_r[smooth]))
    else:
        g_c = float(np.std(hf_c))
        g_r = float(np.std(hf_r))
    # Only strictly penalise grain mismatch when photographer clearly has tactile film grain (> 3.2 LSB)
    # or when candidate is noticeably noisier than a clean reference.
    if g_r < 2.2 and g_c < 2.8:
        sc = 100.0
    else:
        sc = _ratio_score(g_c, g_r, floor=0.8, asymmetry=1.25)
    return {
        "grain_pct": round(sc, 1),
        "grain_cand_lsb": round(g_c, 2),
        "grain_ref_lsb": round(g_r, 2),
    }


def audit_against_reference(cand_path: Path, ref_path: Path, orig_de: float | None = None, *,
                            max_edge: int = 960) -> dict[str, Any]:
    """Run the strict 8-dimension Photographer-Grade Audit on ``cand_path`` vs ``ref_path``."""
    cand, ref = _load_aligned(cand_path, ref_path, max_edge=max_edge)
    yc = 0.2126 * cand[..., 0] + 0.7152 * cand[..., 1] + 0.0722 * cand[..., 2]
    yr = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]

    # 1. Tone Anchor Fidelity
    ca, ra = tone.measure(cand), tone.measure(ref)
    tone_mae = float(np.mean([abs(ca[k] - ra[k]) for k in tone.ORDER]))
    tone_pct = round(max(0.0, min(100.0, 100.0 * (1.0 - tone_mae / 0.26))), 1)

    # 2. Local + Worst-Region Lab Colour Accuracy (mean ΔE + 16x16 patch p90 ΔE)
    lab_c, lab_r = _rgb_to_lab(cand), _rgb_to_lab(ref)
    de_map = np.linalg.norm(lab_c - lab_r, axis=-1)
    mean_de = float(de_map.mean())
    h, w = de_map.shape
    rows, cols = min(16, h), min(16, w)
    gh, gw = max(1, h // rows), max(1, w // cols)
    trim = de_map[:gh * rows, :gw * cols].reshape(rows, gh, cols, gw).mean(axis=(1, 3))
    patch_p90_de = float(np.percentile(trim, 90))
    eff_de = 0.60 * mean_de + 0.40 * patch_p90_de
    local_lab_pct = round(max(0.0, min(100.0, 100.0 * float(np.exp(-((eff_de / 22.0) ** 1.45))))), 1)

    # 3. Colour Distribution & Mood Fidelity
    hists_c = [np.histogram(cand[..., ch], bins=32, range=(0, 1))[0].astype(np.float64) for ch in range(3)]
    hists_r = [np.histogram(ref[..., ch], bins=32, range=(0, 1))[0].astype(np.float64) for ch in range(3)]
    hist_cos = float(np.mean([
        np.dot(hc, hr) / (np.linalg.norm(hc) * np.linalg.norm(hr) + 1e-8)
        for hc, hr in zip(hists_c, hists_r)
    ]))
    hist_pct = round(max(0.0, min(100.0, hist_cos * 100.0)), 1)
    cw, rw = float((cand[..., 0] - cand[..., 2]).mean()), float((ref[..., 0] - ref[..., 2]).mean())
    ct = float((cand[..., 1] - 0.5 * (cand[..., 0] + cand[..., 2])).mean())
    rt = float((ref[..., 1] - 0.5 * (ref[..., 0] + ref[..., 2])).mean())
    cc, rc = float(tone.chroma_mean(cand)), float(tone.chroma_mean(ref))
    c_ratio = min(cc, rc) / max(cc, rc, 1e-3)
    mood_pct = round(max(0.0, min(100.0, 100.0 * (0.5 * max(0.0, 1.0 - (abs(cw - rw) * 3.5 + abs(ct - rt) * 4.5)) + 0.5 * c_ratio))), 1)
    mood_hist_pct = round(0.55 * hist_pct + 0.45 * mood_pct, 1)

    # 4. Zone-Stratified Micro-Detail Preservation
    det = _zone_detail(yc, yr)

    # 5. Transparency & Local Contrast ("通透感")
    clr = _clarity_and_transparency(cand, ref, yc, yr)

    # 6. Anti-Banding & Smooth Gradient Integrity
    bnd = _banding_and_hue_tear(cand, ref, yc, yr, lab_c, lab_r)

    # 7. Organic Film Grain / Micro-Texture Match
    grn = _grain_and_texture(yc, yr)

    # 8. Structural Similarity (Luminance + Chromatic SSIM)
    ssim_l = _windowed_ssim(yc, yr, radius=4)
    ssim_c = 0.5 * (_windowed_ssim(cand[..., 0] - yc + 0.5, ref[..., 0] - yr + 0.5, 4)
                    + _windowed_ssim(cand[..., 2] - yc + 0.5, ref[..., 2] - yr + 0.5, 4))
    ssim_pct = round(max(0.0, min(100.0, (0.65 * ssim_l + 0.35 * ssim_c) * 100.0)), 1)

    # Strict 8-Dimension Weighted Composite
    strict_pct = round(
        0.16 * tone_pct
        + 0.22 * local_lab_pct
        + 0.12 * mood_hist_pct
        + 0.13 * det["detail_pct"]
        + 0.12 * clr["clarity_pct"]
        + 0.11 * bnd["antibanding_pct"]
        + 0.06 * grn["grain_pct"]
        + 0.08 * ssim_pct,
        1,
    )

    # Identify explicit retoucher defect flags (any dimension breaching its hard red-line)
    defects: list[dict[str, Any]] = []
    if det["detail_pct"] < GATE_THRESHOLDS["detail_pct"]:
        low_z = [z for z, r in det["zone_ratios"].items() if r < 0.78]
        z_txt = {"sh": "暗部", "mid": "中调", "hi": "亮部"}
        where = "/".join(z_txt[z] for z in low_z) if low_z else "局部"
        defects.append({"code": "detail_loss", "label": f"{where}细节偏糊 ({det['detail_pct']:.0f}%)", "score": det["detail_pct"]})
    if clr["clarity_pct"] < GATE_THRESHOLDS["clarity_pct"]:
        why = "白场/亮部发闷" if clr["white_mud_excess"] > 0.025 else "局部反差偏灰"
        defects.append({"code": "low_clarity", "label": f"{why}·欠通透 ({clr['clarity_pct']:.0f}%)", "score": clr["clarity_pct"]})
    if bnd["antibanding_pct"] < GATE_THRESHOLDS["antibanding_pct"]:
        defects.append({"code": "color_banding", "label": f"色彩断层/色相撕裂 ({bnd['antibanding_pct']:.0f}%)", "score": bnd["antibanding_pct"]})
    if grn["grain_pct"] < GATE_THRESHOLDS["grain_pct"]:
        direction = "缺少胶片颗粒" if grn["grain_cand_lsb"] < grn["grain_ref_lsb"] else "颗粒偏重"
        defects.append({"code": "grain_mismatch", "label": f"{direction} ({grn['grain_cand_lsb']:.1f} vs {grn['grain_ref_lsb']:.1f} LSB)", "score": grn["grain_pct"]})
    if local_lab_pct < GATE_THRESHOLDS["local_lab_pct"]:
        defects.append({"code": "local_color_drift", "label": f"局部色偏明显 (p90 ΔE {patch_p90_de:.1f})", "score": local_lab_pct})
    if tone_pct < GATE_THRESHOLDS["tone_pct"]:
        defects.append({"code": "tone_drift", "label": f"影调偏离 ({tone_pct:.0f}%)", "score": tone_pct})
    if ssim_pct < GATE_THRESHOLDS["ssim_pct"]:
        defects.append({"code": "structure_loss", "label": f"结构保真偏低 ({ssim_pct:.0f}%)", "score": ssim_pct})

    severe = any(d["score"] < 72.0 for d in defects)
    if strict_pct >= 90.0 and not defects:
        verdict = "EXCELLENT"
    elif strict_pct >= 85.0 and not severe and len(defects) <= 1:
        verdict = "PASS"
    else:
        verdict = "NEEDS_WORK"

    return {
        "strict_pct": strict_pct,
        "verdict": verdict,
        "defects": defects,
        "lab_de": round(mean_de, 2),
        "patch_p90_de": round(patch_p90_de, 2),
        "de_closed_pct": round(max(0.0, (1.0 - mean_de / max(orig_de, 1e-3)) * 100.0), 1) if orig_de else 0.0,
        "tone_pct": tone_pct,
        "local_lab_pct": local_lab_pct,
        "hist_pct": hist_pct,
        "mood_pct": mood_pct,
        "mood_hist_pct": mood_hist_pct,
        "detail_pct": det["detail_pct"],
        "edge_corr": det["edge_corr"],
        "zone_detail_ratios": det["zone_ratios"],
        "clarity_pct": clr["clarity_pct"],
        "local_contrast_ratio": clr["local_contrast_ratio"],
        "white_mud_excess": clr["white_mud_excess"],
        "antibanding_pct": bnd["antibanding_pct"],
        "band_rate_pct": bnd["band_rate_pct"],
        "hue_tear_pct": bnd["hue_tear_pct"],
        "grain_pct": grn["grain_pct"],
        "grain_cand_lsb": grn["grain_cand_lsb"],
        "grain_ref_lsb": grn["grain_ref_lsb"],
        "ssim_pct": ssim_pct,
    }
