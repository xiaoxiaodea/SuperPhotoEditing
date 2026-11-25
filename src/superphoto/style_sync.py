"""Series Style Sync: transfer one perfected edit's look onto a batch of photos from the same shoot.

Unlike naive slider copy-paste (which blows out brighter shots, crushes darker shots,
and drops floating spatial masks onto the wrong subjects), Style Sync separates:

1. **Style DNA (kept consistent across the entire shoot)**:
   - Colour science: HSL mixer, 3-way colour grading wheels, RGB calibration, vibrance,
     effects (clarity / texture / dehaze), vignette, film grain, detail, and relative WB shift.
   - Perceptual tone & colour fingerprint measured on the user's finished anchor photo
     (black matte floor, shadows, midtones, highlights, white ceiling, colourfulness, warmth).

2. **Per-Photo Adaptation (solved independently per target photo via LightCraft RenderSession)**:
   - Ordinary JPEG/PNG targets are first upgraded through ``raster_negative.prepare_raster_negative``
     so they enjoy the same 32-bit linear float headroom and banding resistance as RAWs.
   - Spatial local masks from the anchor are never pasted blindly onto different compositions.
   - Each target photo's unedited baseline is measured; ``tone.solve_scene`` solves that photo's
     own global exposure, smooth 24-knot master curve, and fine saturation trim inside a single
     fast ``RenderSession`` (~2-5 seconds per photo, zero VLM calls!) so brightness, matte black,
     highlight roll-off, and colour density match the anchor cohesively without clipping or blowing
     out differently exposed frames.
   - Optional automatic WB compensation gently aligns camera Auto-WB drift across shots.
   - Automatic Bayer RAW AI Denoise or JPEG/PNG shadow/anti-banding protection runs on delivery,
     and every synced photo retains its exact ``develop.json`` for on-demand full-size JPEG / 16-bit TIFF export.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import denoise, directed, full_export, tone
from .config import Settings
from .db import (
    get_job,
    get_refinement,
    get_style_batch,
    get_style_batch_item,
    get_style_batch_items,
    get_variants,
    refresh_style_batch_counts,
    update_style_batch_item,
)
from .ingest import source_kind as detect_source_kind
from .lightcraft import (
    LightCraftError,
    RELATIVE_WB_MODE,
    as_shot_wb,
    render_document,
    render_session,
)
from .lightroom import connect_lightroom
from .store import job_dir

# Colour and character groups that define a shoot's visual identity (copied across the series).
STYLE_GROUPS = ("mixer", "grading", "calibration", "effects", "vignette", "grain", "detail")
# Neutral JPEG reference Kelvin used by LightCraft when wb.mode == "custom".
_JPEG_NEUTRAL_K = directed.JPEG_NEUTRAL_KELVIN
# Maximum automatic inter-shot WB compensation (prevents turning a sunset neutral or a shade shot orange).
_AUTO_WB_TEMP_LIMIT = 450.0
_AUTO_WB_TINT_LIMIT = 8.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _measure_picture(path: Path) -> dict[str, Any]:
    """Perceptual tone anchors, colourfulness, warmth (R-B) and tint (G - (R+B)/2)."""
    rgb = tone.load_small(path, 480)
    anchors = tone.measure(rgb)
    chroma = round(float(tone.chroma_mean(rgb)), 4)
    # Use midtone non-clipped pixels for reliable colour temperature / tint comparison.
    luma = tone.luma_of(rgb)
    mid = (luma > 0.12) & (luma < 0.88)
    sample = rgb[mid] if int(np.sum(mid)) >= 64 else rgb.reshape(-1, 3)
    mean_r = float(np.mean(sample[:, 0]))
    mean_g = float(np.mean(sample[:, 1]))
    mean_b = float(np.mean(sample[:, 2]))
    warmth = round(mean_r - mean_b, 4)
    tint_bias = round(mean_g - 0.5 * (mean_r + mean_b), 4)
    return {
        "anchors": anchors,
        "chroma": chroma,
        "warmth": warmth,
        "tint_bias": tint_bias,
    }


def _scale_nested(value: Any, factor: float) -> Any:
    """Scale numeric style parameters proportionally by ``factor`` (preserving hues/blending/midpoint/size)."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in {"hue", "blending", "balance", "midpoint", "roundness", "feather", "size", "roughness",
                     "sharpen_radius", "sharpen_detail", "sharpen_masking", "nr_detail", "nr_contrast",
                     "nr_color_detail", "nr_color_smoothness"}:
                out[k] = v
            else:
                out[k] = _scale_nested(v, factor)
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value) * factor, 3)
    return copy.deepcopy(value)


def _extract_wb_offset(document: dict[str, Any], source_kind: str, *,
                       plan: dict[str, Any] | None = None,
                       binary: Path | None = None,
                       original: Path | None = None) -> dict[str, float]:
    """Recover the anchor's relative (temp_K, tint) white balance shift so it transfers cleanly across cameras."""
    wb = document.get("wb")
    if not isinstance(wb, dict):
        return {"temp": 0.0, "tint": 0.0}
    mode = wb.get("mode")
    if mode == RELATIVE_WB_MODE:
        return {"temp": round(float(wb.get("temp", 0.0)), 1), "tint": round(float(wb.get("tint", 0.0)), 2)}
    if mode == "custom":
        temp = float(wb.get("temp", _JPEG_NEUTRAL_K))
        tint = float(wb.get("tint", 0.0))
        if source_kind in {"jpeg", "png", "tiff"}:
            return {"temp": round(temp - _JPEG_NEUTRAL_K, 1), "tint": round(tint, 2)}
        # RAW custom WB: subtract its own as-shot WB if readable; else fall back to plan's wb offset.
        if binary is not None and original is not None and original.is_file():
            shot = as_shot_wb(binary, original)
            if shot is not None:
                return {"temp": round(temp - shot[0], 1), "tint": round(tint - shot[1], 2)}
        plan_wb = ((plan or {}).get("develop") or {}).get("wb") or {}
        if isinstance(plan_wb, dict):
            return {"temp": round(float(plan_wb.get("temp", 0.0)), 1),
                    "tint": round(float(plan_wb.get("tint", 0.0)), 2)}
    return {"temp": 0.0, "tint": 0.0}


def extract_style_profile(settings: Settings, conn: sqlite3.Connection, job_id: str, slot: str,
                          version: str | None = None, *, binary: Path | None = None) -> dict[str, Any]:
    """Extract a portable Series Style Profile from any finished variant or refinement."""
    job = get_job(conn, job_id)
    if not job or job["status"] != "ready":
        raise ValueError("请先等基准照片处理完成后再批量套用风格")
    variant = next((v for v in get_variants(conn, job_id) if v["slot"] == slot), None)
    if not variant:
        raise ValueError(f"找不到方案 {slot}")

    root = job_dir(settings, job_id)
    dev_path = full_export.develop_record(conn, root, job_id, slot, version)
    document = _read_json(dev_path)
    if not document:
        raise ValueError("基准成片缺少 LightCraft 显影记录，无法提取风格")

    if version:
        ref = get_refinement(conn, version)
        if not ref or ref["status"] != "ready":
            raise ValueError("找不到该微调版本")
        anchor_render = root / "refinements" / version / "result.jpg"
        ref_data = _read_json(root / "refinements" / version / "audit.json")
        plan = ref_data.get("plan") if isinstance(ref_data.get("plan"), dict) else None
        title = f"{variant['title']} · 微调版"
        direction_title = (json.loads(variant["recipe_json"]).get("style_workflow") or {}).get("direction_title") or variant["title"]
        subtitle = ref.get("summary") or ref.get("prompt") or ""
    else:
        anchor_render = root / variant["filename"]
        sidecar = json.loads(variant["recipe_json"])
        plan = sidecar.get("plan") if isinstance(sidecar.get("plan"), dict) else None
        title = variant["title"]
        direction_title = (sidecar.get("style_workflow") or {}).get("direction_title") or variant["title"]
        subtitle = (sidecar.get("style_workflow") or {}).get("selection_reason") or variant.get("why") or ""

    if not anchor_render.is_file():
        raise ValueError("基准成片图片文件已不在本机")

    baseline_path = root / "lightroom-exports" / "baseline" / "render.jpg"
    if not baseline_path.is_file():
        baseline_path = root / "source_preview.jpg"
    if not baseline_path.is_file():
        raise ValueError("基准原片预览不存在")

    original = root / Path(job.get("original_file") or "original.jpg").name
    src_kind = detect_source_kind(original) if original.is_file() else "jpeg"

    style_develop: dict[str, Any] = {}
    for group in STYLE_GROUPS:
        if document.get(group):
            style_develop[group] = copy.deepcopy(document[group])
    # Preserve RGB split channel curves (curve.red / green / blue) alongside colour science;
    # curve.master stays excluded because it is solved per target photo by tone.solve_scene.
    curve = document.get("curve") or {}
    if isinstance(curve, dict):
        rgb_curves = {ch: copy.deepcopy(curve[ch]) for ch in ("red", "green", "blue") if isinstance(curve.get(ch), list)}
        if rgb_curves:
            style_develop["rgb_curves"] = rgb_curves
    # Preserve vibrance & look-level saturation baseline; solver refines saturation per target photo.
    color = document.get("color") or {}
    if isinstance(color, dict) and color:
        style_develop["color"] = {k: float(v) for k, v in color.items() if k in ("vibrance", "saturation")}
    # Carry edge-aware local tone recovery (highlights / shadows) even alongside a solved master curve,
    # since a 1D master curve cannot recover local snow/highlight/shadow micro-contrast. Carry global
    # contrast/whites/blacks only when no master curve was solved.
    light = document.get("light") or {}
    has_master_curve = bool((document.get("curve") or {}).get("master"))
    if isinstance(light, dict):
        keep_keys = ("highlights", "shadows") if has_master_curve else tuple(k for k in light if k != "exposure")
        non_exp = {k: float(light[k]) for k in keep_keys if k in light and abs(float(light[k])) > 1e-4}
        if non_exp:
            style_develop["light"] = non_exp

    wb_offset = _extract_wb_offset(document, src_kind, plan=plan, binary=binary, original=original)
    anchor_finished = _measure_picture(anchor_render)
    anchor_baseline = _measure_picture(baseline_path)

    # Record EV tone deltas of anchor (finished vs baseline) for scene-preserving blending
    tone_ev_delta = {
        k: round(tone.ev_between(anchor_baseline["anchors"][k], anchor_finished["anchors"][k]), 3)
        for k in tone.ORDER
        if k in anchor_baseline["anchors"] and k in anchor_finished["anchors"]
    }
    chroma_ratio = round(
        anchor_finished["chroma"] / max(anchor_baseline["chroma"], 0.015), 3
    )

    look_info = (plan or {}).get("look") if isinstance(plan, dict) else None
    return {
        "version": 1,
        "source_job_id": job_id,
        "source_slot": slot,
        "source_version": version,
        "source_name": job.get("original_name") or "",
        "source_kind": src_kind,
        "title": title,
        "direction_title": direction_title,
        "subtitle": str(subtitle)[:200],
        "look": look_info,
        "style_develop": style_develop,
        "wb_offset": wb_offset,
        "curve_saturation": float((document.get("curve") or {}).get("refine_saturation", directed.DEFAULT_CURVE_SATURATION)),
        "anchor_finished": anchor_finished,
        "anchor_baseline": anchor_baseline,
        "tone_ev_delta": tone_ev_delta,
        "chroma_ratio": chroma_ratio,
        "anchor_render_url": (f"/api/jobs/{job_id}/refinements/{version}/image"
                              if version else f"/media/{job_id}/{variant['filename']}"),
    }


def _target_tone_anchors(profile: dict[str, Any], target_base: dict[str, float], *,
                         target_kind: str, match_exposure: bool, style_strength: float,
                         has_gain_map: bool = False) -> dict[str, float]:
    """Compute cohesive per-photo tone targets blending absolute anchor look with target scene mood.

    - ``match_exposure=True``: pulls differently exposed shots in the same shoot toward the anchor's
      brightness, matte black floor, and highlight ceiling (75% absolute consistency + 25% relative
      delta) so a gallery of 20 photos reads as one unified edit.
    - ``match_exposure=False``: keeps each photo's relative brightness while still transferring the
      anchor's exact matte black floor, contrast character, and highlight roll-off.
    """
    fin = profile["anchor_finished"]["anchors"]
    deltas = profile.get("tone_ev_delta") or {}
    strength = max(0.2, min(1.3, float(style_strength)))

    # When match_exposure=True, pull both darker and brighter shots directly toward the anchor's
    # finished levels (85% anchor + 15% own baseline) without double-applying the anchor's EV lift
    # onto already-bright shots. When match_exposure=False, preserve per-photo exposure via relative delta.
    abs_w = (
        {"black": 0.90, "shadows": 0.80, "mid": 0.85, "highlights": 0.80, "white": 0.85}
        if match_exposure else
        {"black": 0.75, "shadows": 0.25, "mid": 0.15, "highlights": 0.25, "white": 0.65}
    )

    raw_targets: dict[str, float] = {}
    for k in tone.ORDER:
        if k not in target_base or k not in fin:
            continue
        tb = float(target_base[k])
        af = float(fin[k])
        w = abs_w[k]
        if match_exposure:
            blended = w * af + (1.0 - w) * tb
        else:
            ev_d = float(deltas.get(k, 0.0)) * strength
            rel = tone.to_display(min(0.995, max(0.0, tone.to_linear(tb) * (2.0 ** ev_d))))
            blended = w * af + (1.0 - w) * rel
        # Scale movement from target baseline by style_strength
        val = tb + (blended - tb) * min(1.0, strength)
        raw_targets[k] = round(min(0.992, max(0.005, val)), 4)

    # Enforce monotonic ordering & safe EV shift bounds so a night shot never blows out to noon
    limits = tone.shift_limits(target_kind, has_gain_map=has_gain_map)
    if match_exposure and target_kind != "raw":
        limits = {**limits, "mid": (-2.0, 2.0), "shadows": (-3.0, 2.0)}
    bounded, _ = tone.bound_shift(raw_targets, target_base, limits)
    return tone._increasing(bounded)


def _compose_base_document(profile: dict[str, Any], target_kind: str, target_base_stats: dict[str, Any], *,
                           match_wb: bool, style_strength: float) -> dict[str, Any]:
    """Build the initial LightCraft develop document carrying the anchor's colour science + WB."""
    strength = max(0.2, min(1.3, float(style_strength)))
    raw_style = {k: v for k, v in (profile.get("style_develop") or {}).items() if k != "rgb_curves"}
    doc = _scale_nested(raw_style, strength)
    rgb_curves = (profile.get("style_develop") or {}).get("rgb_curves") or {}
    if isinstance(rgb_curves, dict) and rgb_curves:
        csec = doc.setdefault("curve", {})
        for ch, pts in rgb_curves.items():
            if ch in ("red", "green", "blue") and isinstance(pts, list):
                csec[ch] = [
                    [round(float(x), 4), round(min(1.0, max(0.0, float(x) + (float(y) - float(x)) * strength)), 4)]
                    for x, y in pts
                ]

    # White balance: anchor's relative artistic shift + optional inter-shot Auto-WB drift compensation
    wb_off = profile.get("wb_offset") or {}
    temp = float(wb_off.get("temp", 0.0)) * strength
    tint = float(wb_off.get("tint", 0.0)) * strength

    if match_wb:
        anchor_base_w = float((profile.get("anchor_baseline") or {}).get("warmth", 0.0))
        target_base_w = float(target_base_stats.get("warmth", 0.0))
        # Positive (anchor_base_w - target_base_w) means target baseline is cooler than anchor -> warm it up
        drift_temp = max(-_AUTO_WB_TEMP_LIMIT, min(_AUTO_WB_TEMP_LIMIT, (anchor_base_w - target_base_w) * 2800.0))
        anchor_base_t = float((profile.get("anchor_baseline") or {}).get("tint_bias", 0.0))
        target_base_t = float(target_base_stats.get("tint_bias", 0.0))
        # Positive tint_bias means greener; LightCraft positive tint pushes magenta to neutralize green
        drift_tint = max(-_AUTO_WB_TINT_LIMIT, min(_AUTO_WB_TINT_LIMIT, (target_base_t - anchor_base_t) * 120.0))
        temp += drift_temp
        tint += drift_tint

    temp = max(-3200.0, min(3800.0, temp))
    tint = max(-45.0, min(45.0, tint))
    if abs(temp) > 5.0 or abs(tint) > 0.3:
        if target_kind in {"jpeg", "png", "tiff"}:
            doc["wb"] = {"mode": "custom", "temp": round(min(15000.0, max(2000.0, _JPEG_NEUTRAL_K + temp)), 1),
                         "tint": round(tint, 2)}
        else:
            doc["wb"] = {"mode": RELATIVE_WB_MODE, "temp": round(temp, 1), "tint": round(tint, 2)}
    return doc


def apply_style_to_photo(settings: Settings, binary: Path, profile: dict[str, Any], source: Path,
                         out_dir: Path, *, match_exposure: bool = True, match_wb: bool = True,
                         style_strength: float = 1.0, edge: int | None = None) -> dict[str, Any]:
    """Adapt and render one target photo in the exact visual style of ``profile`` (zero VLM calls!)."""
    started = time.perf_counter()
    source = Path(source).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_edge = int(edge if edge is not None else settings.lightroom_export_edge)
    target_kind = detect_source_kind(source)

    # Upgrade ordinary JPEG/PNG inputs through our 32-bit float digital negative engine
    render_source = source
    negative_meta: dict[str, Any] | None = None
    if settings.raster_negative_enabled and target_kind in {"jpeg", "png"}:
        try:
            from .raster_negative import prepare_raster_negative

            neg_path = out_dir / "working_negative.tif"
            negative_meta = prepare_raster_negative(source, neg_path, max_edge=max(export_edge, 3200))
            (out_dir / "negative.json").write_text(json.dumps(negative_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            render_source = neg_path
        except Exception:
            negative_meta = None
            render_source = source

    probe_count = [0]
    with render_session(binary, render_source):
        # 1. Neutral baseline render for measurements and Before/After UI preview
        baseline_img = render_document(binary, render_source, {}, out_dir / "baseline",
                                       edge=min(export_edge, 1600), name="source_preview.jpg")
        base_stats = _measure_picture(baseline_img)

        # 2. Compose style-carrying base document + target tone & chroma goals
        base_doc = _compose_base_document(profile, target_kind, base_stats,
                                          match_wb=match_wb, style_strength=style_strength)
        has_gain_map = bool((negative_meta or {}).get("has_gain_map"))
        targets = _target_tone_anchors(profile, base_stats["anchors"], target_kind=target_kind,
                                       match_exposure=match_exposure, style_strength=style_strength,
                                       has_gain_map=has_gain_map)

        # Target colourfulness: blend anchor's finished chroma and anchor's relative chroma gain
        anchor_fin_c = float((profile.get("anchor_finished") or {}).get("chroma") or base_stats["chroma"])
        rel_c = base_stats["chroma"] * float(profile.get("chroma_ratio") or 1.0)
        want_chroma = round(max(0.02, min(0.45, 0.55 * anchor_fin_c + 0.45 * rel_c)), 4)

        start_sat = float((base_doc.get("color") or {}).get("saturation", 0.0))
        sat_lo = max(-50.0, start_sat - 22.0)
        sat_hi = min(55.0 if target_kind == "raw" else 35.0, start_sat + (35.0 if target_kind == "raw" else 20.0))
        curve_sat = float(profile.get("curve_saturation", directed.DEFAULT_CURVE_SATURATION))
        banding_risk = float((negative_meta or {}).get("banding_risk") or 0.0)
        max_slope = (2.45 if target_kind in {"jpeg", "png"} and not has_gain_map and banding_risk >= 0.25
                     else tone.MAX_SLOPE)

        def _build_doc(curve: list[list[float]], exposure: float, _local: dict, saturation: float) -> dict[str, Any]:
            doc = copy.deepcopy(base_doc)
            if abs(exposure) > 1e-4 or "light" in doc:
                doc.setdefault("light", {})["exposure"] = round(float(exposure), 3)
            if curve:
                csec = doc.setdefault("curve", {})
                csec["master"] = [[float(x), float(y)] for x, y in curve]
                csec["refine_saturation"] = curve_sat
            if abs(saturation) > 1e-2 or "color" in doc:
                doc.setdefault("color", {})["saturation"] = round(float(saturation), 2)
            return doc

        def _shoot(curve: list[list[float]], exposure: float, local: dict, saturation: float) -> Path:
            probe_count[0] += 1
            return render_document(binary, render_source, _build_doc(curve, exposure, local, saturation),
                                   out_dir / "probes" / f"{probe_count[0]:02d}", edge=directed.PROBE_EDGE)

        # 3. Solve per-photo exposure + smooth 24-knot curve + saturation inside the open session
        solved = tone.solve_scene(
            _shoot, targets, regions=[], exposure=0.0, exposure_range=(-2.4, 2.4),
            rounds=4, tolerance=0.018, chroma=want_chroma, saturation=start_sat,
            saturation_range=(sat_lo, sat_hi), unedited=base_stats["anchors"],
            shift_limits=tone.shift_limits(target_kind, has_gain_map=has_gain_map),
            max_slope=max_slope,
        )
        final_doc = _build_doc(solved["curve"], solved["exposure"], {}, solved["saturation"])

        # 4. Automatic RAW AI Denoise or JPEG/PNG shadow/anti-banding protection
        notes: list[str] = []
        decision = denoise.decide(source, target_kind, final_doc)
        if decision:
            final_doc = denoise.with_denoise(final_doc, decision["amount"])
        elif target_kind in {"jpeg", "png"}:
            prot = denoise.raster_protection(final_doc, target_kind, negative_meta)
            if prot:
                final_doc = denoise.with_raster_protection(final_doc, prot)
                notes.append(f"JPG/PNG 暗部保护（降噪 {prot['nr_luminance']:g}/{prot['nr_color']:g}）")

        # 5. Final delivery render (2400px) with develop.json preserved alongside
        final_render = render_document(binary, render_source, final_doc, out_dir / "result",
                                       edge=export_edge, name="result.jpg")
        if decision and denoise.recorded(final_render.parent):
            notes.append(f"AI 降噪 {decision['amount']:g}")

    after_stats = _measure_picture(final_render)
    # Remove intermediate solver probe images to keep disk usage lean
    shutil.rmtree(out_dir / "probes", ignore_errors=True)

    return {
        "source_preview": str(baseline_img),
        "result": str(final_render),
        "develop": final_doc,
        "source_kind": target_kind,
        "negative": negative_meta,
        "metrics": {
            "exposure_ev": round(float(solved["exposure"]), 2),
            "saturation": round(float(solved["saturation"]), 1),
            "before_mid": base_stats["anchors"].get("mid"),
            "after_mid": after_stats["anchors"].get("mid"),
            "anchor_mid": (profile.get("anchor_finished") or {}).get("anchors", {}).get("mid"),
            "before_chroma": base_stats["chroma"],
            "after_chroma": after_stats["chroma"],
            "probes": probe_count[0],
            "notes": notes,
            "seconds": round(time.perf_counter() - started, 2),
        },
    }


def batch_dir(settings: Settings, batch_id: str) -> Path:
    return settings.data_dir / "style-batches" / batch_id


def item_dir(settings: Settings, batch_id: str, item_id: str) -> Path:
    return batch_dir(settings, batch_id) / "items" / item_id


def run_batch_item(settings: Settings, conn: sqlite3.Connection, item_id: str) -> None:
    """Background worker entrypoint: style-sync a single queued photo inside a style batch."""
    item = get_style_batch_item(conn, item_id)
    if not item or item["status"] not in {"queued", "running"}:
        return
    batch = get_style_batch(conn, item["batch_id"])
    if not batch:
        update_style_batch_item(conn, item_id, status="error", step="error", error="找不到风格批量任务")
        return

    folder = item_dir(settings, item["batch_id"], item_id)
    original = folder / Path(item["original_file"]).name
    if not original.is_file():
        update_style_batch_item(conn, item_id, status="error", step="error", error="源片已不在本机")
        refresh_style_batch_counts(conn, item["batch_id"])
        return

    try:
        update_style_batch_item(conn, item_id, status="running", step="connect")
        client, _status = connect_lightroom(settings.lightcraft_bin)
        binary = Path(client.binary)
        profile = json.loads(batch["profile_json"])
        options = profile.get("sync_options") or {}
        update_style_batch_item(conn, item_id, step="sync")
        outcome = apply_style_to_photo(
            settings,
            binary,
            profile,
            original,
            folder,
            match_exposure=bool(options.get("match_exposure", True)),
            match_wb=bool(options.get("match_wb", True)),
            style_strength=float(options.get("style_strength", 1.0)),
        )
        update_style_batch_item(
            conn,
            item_id,
            status="ready",
            step="ready",
            error=None,
            output_filename="result/result.jpg",
            develop_json=json.dumps(outcome["develop"], ensure_ascii=False),
            metrics_json=json.dumps(outcome["metrics"], ensure_ascii=False),
        )
    except Exception as exc:
        update_style_batch_item(conn, item_id, status="error", step="error", error=str(exc) or type(exc).__name__)
    finally:
        refresh_style_batch_counts(conn, item["batch_id"])


def export_batch_item_full(settings: Settings, conn: sqlite3.Connection, batch_id: str, item_id: str,
                           fmt: str, *, binary: Path) -> dict[str, Any]:
    """Render (or reuse) full-resolution JPEG or 16-bit TIFF for one style-synced batch photo."""
    spec = full_export.FORMATS[fmt]
    item = get_style_batch_item(conn, item_id)
    if not item or item["batch_id"] != batch_id or item["status"] != "ready":
        raise full_export.FullExportUnavailable("这张批量照片尚未完成")
    folder = item_dir(settings, batch_id, item_id)
    original = folder / Path(item["original_file"]).name
    if not original.is_file():
        raise full_export.FullExportUnavailable("源片已不在本机")
    document = json.loads(item["develop_json"] or "{}")
    if not document:
        raise full_export.FullExportUnavailable("缺少显影记录")

    full_dir = folder / "full"
    output = full_dir / f"render{spec['suffix']}"
    stem = Path(item["original_name"] or "photo").stem
    dl_name = f"{stem}-同风格-原尺寸{spec['suffix']}"
    with full_export._render_lock:
        if output.is_file() and _read_json(full_dir / "develop.json") == document:
            meta = _read_json(full_dir / f"render{spec['suffix']}.json")
            return {**meta, "path": output, "filename": dl_name, "cached": True}
        full_dir.mkdir(parents=True, exist_ok=True)
        use_neg = (settings.raster_negative_enabled and (folder / "negative.json").is_file()
                   and original.suffix.lower() in {".jpg", ".jpeg", ".png"})
        src_copy = full_dir / ("source.tif" if use_neg else f"source{original.suffix.lower()}")
        started = time.monotonic()
        try:
            if use_neg:
                from .raster_negative import prepare_raster_negative

                prepare_raster_negative(original, src_copy, max_edge=0)
            else:
                shutil.copy2(original, src_copy)
            output.unlink(missing_ok=True)
            render_document(binary, src_copy, document, full_dir, edge=0, timeout=full_export.RENDER_TIMEOUT,
                            name=output.name, quality=spec["quality"], options=spec["options"])
        finally:
            src_copy.unlink(missing_ok=True)
        meta = {"format": fmt, "bytes": output.stat().st_size, "seconds": round(time.monotonic() - started, 1),
                **full_export._pixel_size(output)}
        (full_dir / f"render{spec['suffix']}.json").write_text(json.dumps(meta), encoding="utf-8")
    return {**meta, "path": output, "filename": dl_name, "cached": False}


def build_batch_zip(settings: Settings, conn: sqlite3.Connection, batch_id: str) -> Path:
    """Package all ready photos of a style batch into one clean downloadable ZIP archive."""
    batch = get_style_batch(conn, batch_id)
    if not batch:
        raise ValueError("找不到批量任务")
    items = [item for item in get_style_batch_items(conn, batch_id) if item["status"] == "ready"]
    if not items:
        raise ValueError("还没有已完成的成片可供打包下载")
    archive = batch_dir(settings, batch_id) / "synced-results.zip"
    tmp = archive.with_suffix(".zip.tmp")
    used_names: dict[str, int] = {}
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for item in items:
                img = item_dir(settings, batch_id, item["id"]) / (item["output_filename"] or "result/result.jpg")
                if not img.is_file():
                    continue
                stem = Path(item["original_name"] or item["id"]).stem
                count = used_names.get(stem, 0)
                used_names[stem] = count + 1
                suffix = f"-{count + 1}" if count else ""
                arcname = f"{stem}{suffix}-同风格.jpg"
                zf.write(img, arcname=arcname)
        tmp.replace(archive)
    finally:
        tmp.unlink(missing_ok=True)
    return archive
