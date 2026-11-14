from __future__ import annotations

import json
import shutil
import sqlite3
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import analyze
from .color import linear_to_srgb
from .composition import apply_geometry
from .config import Settings
from .critique import critique_photo
from .darktable_export import write_recipe_sidecar
from .db import replace_variants, update_job
from .ingest import downscale_linear, load_linear
from .layers import build_layers
from .lightroom import (
    LightroomError,
    apply_recipe_pass,
    create_lightroom_checkpoint,
    connect_lightroom,
    export_lightroom_baseline,
)
from .lightroom_qa import (
    LightroomQualityReport,
    assess_lightroom_render,
    blend_recipes,
    correction_for_report,
    is_distinct_candidate,
    separate_candidate_recipe,
)
from .models import Critique, CropControls, DetailControls, EditIntent, OpticsControls, Recipe
from .preference import build_plans, honor_locked_controls
from .celebrity_styles import build_style_decision
from .qa import refine_plan, score_grade, target_for
from .render import export_srgb_jpeg
from .semantic import segment_scene
from .review import VisualReview, repair_visual_recipe, review_render
from .finishing_review import compare_finishing, propose_finishing
from .store import job_dir
from .processing import serialized_processing
from .workflow import (
    apply_restoration,
    apply_retouch,
    plan_restoration,
    score_restoration,
    simple_explanation,
)


@serialized_processing
def run_job(settings: Settings, conn: sqlite3.Connection, job_id: str, original: Path) -> None:
    root = job_dir(settings, job_id).resolve()
    original = original.resolve()
    try:
        update_job(conn, job_id, status="running", step="ingest", original_file=original.name)
        baseline_result = None
        if settings.lightroom_enabled:
            update_job(conn, job_id, step="connect")
            baseline_result = _prepare_lightroom_baseline(settings, original, root)
            update_job(conn, job_id, step="ingest")
        analysis_source = Path(baseline_result["export"]) if baseline_result else original
        linear, _meta = load_linear(analysis_source)
        working = downscale_linear(linear, settings.working_edge)
        del linear

        update_job(conn, job_id, step="analyze")
        technical = analyze(working, original)
        (root / "technical.json").write_text(
            json.dumps(technical.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        export_srgb_jpeg(working, root / "source_preview.jpg", quality=88, max_edge=settings.preview_edge)
        export_srgb_jpeg(working, root / "critique_preview.jpg", quality=82, max_edge=settings.critique_edge)

        # Axis A: technical restoration only. Color and global tone are frozen,
        # then a dedicated gate decides whether the stage is safe to keep.
        update_job(conn, job_id, step="restore")
        restoration = plan_restoration(technical)
        restored = apply_restoration(working, restoration)
        restoration_gate = score_restoration(working, restored)
        if not restoration_gate.accepted:
            restored = working
        restoration_report = {
            "actions": restoration.actions,
            "protections": restoration.protections,
            "accepted": restoration_gate.accepted,
            "score": round(restoration_gate.score, 1),
            "notes": restoration_gate.notes,
        }
        (root / "workflow.json").write_text(
            json.dumps({"restoration": restoration_report}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        update_job(conn, job_id, step="critique", technical_json=technical.model_dump_json())
        critique = critique_photo(settings, root / "critique_preview.jpg", technical)
        from .master_studies import study_context
        references = study_context(critique, technical)
        critique.intent.reference_ids = [r["id"] for r in references["references"]]
        critique.intent.reference_checks = references["review_checks"]
        (root / "master-study-guidance.json").write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
        from .directions import design_directions
        from .scene_preferences import preset_feedback
        preference = preset_feedback(conn, critique, technical)
        style_decision = build_style_decision(technical, critique, feedback=preference["evidence"])
        preference["selected"] = style_decision.look_ids
        (root / "preference-selection.json").write_text(json.dumps(preference, ensure_ascii=False, indent=2), encoding="utf-8")
        critique.directions = design_directions(settings, root / "critique_preview.jpg", critique, technical,
                                               audit_path=root / "directions-audit.json", look_ids=style_decision.look_ids)
        (root / "critique.json").write_text(
            json.dumps(critique.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        (root / "style-decision.json").write_text(
            json.dumps(style_decision.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        plans = build_plans(style_decision.look_ids, critique, technical)
        match_by_id = {item.look_id: item for item in style_decision.candidates}
        route_labels = {plan.slot: plan.title for plan in plans}
        plans = [
            plan.model_copy(
                update={
                    "recipe": _professionalize_recipe(plan.recipe, technical, plan.geometry),
                    "route_label": route_labels[plan.slot],
                    "match_score": match_by_id[plan.look_id].match_score,
                    "adaptation_notes": plan.moves if critique.directions else _adaptation_notes(plan.recipe, technical),
                    "manual_gaps": style_decision.manual_gaps,
                }
            )
            for plan in plans
        ]
        (root / "plan.json").write_text(
            json.dumps([p.model_dump() for p in plans], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        update_job(
            conn,
            job_id,
            step="render",
            critique_json=critique.model_dump_json(),
            plan_json=json.dumps([p.model_dump() for p in plans], ensure_ascii=False),
        )

        variant_rows = []
        qa_summaries = []
        semantic_cache: dict[tuple[float, float, float, float, float], object | None] = {}
        semantic_report = {"enabled": settings.semantic_enabled, "used": False, "model": settings.semantic_model}
        for plan in plans:
            frame = restored if plan.geometry.is_noop() else apply_geometry(restored, plan.geometry)
            frame_srgb = np.clip(linear_to_srgb(np.maximum(frame, 0.0)), 0.0, 1.0)
            geom = plan.geometry
            semantic_key = (
                round(geom.rotate_deg, 3),
                round(geom.crop_left, 4),
                round(geom.crop_top, 4),
                round(geom.crop_right, 4),
                round(geom.crop_bottom, 4),
            )
            if semantic_key not in semantic_cache:
                semantic_cache[semantic_key] = segment_scene(frame_srgb, settings)
            semantic = semantic_cache[semantic_key]
            if semantic is not None:
                semantic_report.update({"used": True, "confidence": round(semantic.confidence, 3)})
            layers = build_layers(frame_srgb, technical, critique.clutter, semantic=semantic)
            best_img = None
            best_qa = None
            best_plan = plan
            best_attributes: list[str] = []
            best_recipe = plan.recipe
            current = plan
            # A local approximation is useful as a fallback preview; it cannot
            # select or refine a recipe that a different engine will deliver.
            for attempt in range(1 if settings.lightroom_enabled else 3):
                # Retain every purposeful adjustment; local pixels are only a
                # preview when Lightroom is the delivery engine.
                rendered, retouch_recipe, attributes = apply_retouch(
                    frame,
                    current,
                    layers=layers,
                    technical=technical,
                    clutter_boxes=critique.clutter,
                )
                qa = score_grade(frame, rendered, technical, current.slot)
                if best_qa is None or qa.score > best_qa.score:
                    best_img, best_qa, best_plan = rendered, qa, current
                    best_attributes, best_recipe = attributes, retouch_recipe
                if qa.score >= target_for(current.slot) and not qa.too_strong:
                    break
                if attempt == 2:
                    break
                if qa.too_strong or qa.too_weak:
                    current = refine_plan(current, qa, critique, technical, attempt)
                    continue
                break

            update_job(conn, job_id, step="qa")
            out_name = f"variant_{plan.slot}.jpg"
            export_srgb_jpeg(best_img, root / out_name, quality=92, max_edge=settings.preview_edge)
            sidecar = best_recipe.model_dump()
            sidecar["lut_mix"] = best_plan.lut_mix
            sidecar["geometry"] = best_plan.geometry.model_dump()
            sidecar["qa_score"] = round(best_qa.score, 1)
            sidecar["qa_notes"] = best_qa.notes
            sidecar["intent"] = best_plan.intent.model_dump()
            sidecar["preview_only_qa"] = settings.lightroom_enabled
            sidecar["workflow"] = simple_explanation(
                restoration, best_attributes, semantic is not None, ethics=best_plan.ethics
            )
            sidecar["workflow"]["restoration_gate"] = restoration_report
            sidecar["workflow"]["retouch_attributes"] = best_attributes
            sidecar["workflow"]["ethics"] = best_plan.ethics
            sidecar["style_workflow"] = {
                "method_id": best_plan.method_id,
                "preset_id": best_plan.look_id if best_plan.method_id.startswith("preset:") else None,
                "preset_amount": best_plan.lut_mix,
                "reference_ids": best_plan.intent.reference_ids,
                "reference_checks": best_plan.intent.reference_checks,
                "mode": "art-directed" if critique.directions else "reference-look",
                "route_label": best_plan.route_label,
                "match_score": round(best_plan.match_score, 3),
                "selection_reason": match_by_id[best_plan.look_id].reason,
                "adaptation_notes": best_plan.adaptation_notes,
                "manual_gaps": best_plan.manual_gaps,
            }
            write_recipe_sidecar(root / f"variant_{plan.slot}.json", best_recipe, best_plan.look_id)
            why = best_plan.why
            mark = f"技术自检 {best_qa.score:.0f} 分"
            if not settings.lightroom_enabled and mark not in why:
                why = f"{why} {mark}。"
            qa_summaries.append(f"{best_plan.slot}:{best_qa.score:.0f}")
            variant_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "slot": plan.slot,
                    "look_id": best_plan.look_id,
                    "title": best_plan.title,
                    "why": why,
                    "recipe_json": json.dumps(sidecar, ensure_ascii=False),
                    "filename": out_name,
                }
            )
        ethics = plans[0].ethics if plans else "commercial"
        workflow_payload = {
            "restoration": restoration_report,
            "semantic": semantic_report,
            "axis_order": ["read_card", "defects", "route", "style", "restoration", "retouch", "lightroom"],
            "style_decision": style_decision.model_dump(),
            "ethics": ethics,
            "retouch_rule": "保留完整编辑意图；局部光影与整体色调分别执行；锐化降噪按原片决定",
            "analysis_source": "lightroom-baseline" if baseline_result else "color-managed-source",
        }
        (root / "workflow.json").write_text(
            json.dumps(workflow_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "qa.json").write_text(
            json.dumps({"slots": qa_summaries, "semantic": semantic_report, "workflow": workflow_payload}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        replace_variants(conn, job_id, variant_rows)
        if settings.lightroom_enabled:
            _render_variants_with_lightroom(settings, conn, job_id, original, root, variant_rows,
                                           baseline_result=baseline_result, critique=critique)
        update_job(conn, job_id, status="ready", step="ready", error=None)
    except Exception as exc:
        update_job(
            conn,
            job_id,
            status="error",
            step="error",
            error=f"{exc}\n{traceback.format_exc()}",
        )
        raise


def _professionalize_recipe(recipe: Recipe, technical, geometry) -> Recipe:
    """Add conservative per-frame detail, optics and crop controls for Lightroom."""
    developed = technical.source_kind in {"jpeg", "png"}
    iso = _technical_iso(technical.exif)
    noisy = technical.noise > 0.065 and (not developed or iso >= 1600)
    soft = technical.sharpness < 0.045 and not noisy

    luminance_nr = None
    if noisy:
        luminance_nr = float(np.clip(12.0 + max(0.0, technical.noise - 0.06) * 360.0 + max(0.0, iso - 800.0) / 180.0, 12.0, 45.0))
    sharpness = None
    if not developed:
        sharpness = 55.0 if soft else (42.0 if noisy else 48.0)

    detail = DetailControls(
        sharpness=recipe.detail.sharpness if recipe.detail.sharpness is not None else sharpness,
        sharpen_radius=recipe.detail.sharpen_radius if recipe.detail.sharpen_radius is not None else (1.0 if sharpness is not None else None),
        sharpen_detail=recipe.detail.sharpen_detail if recipe.detail.sharpen_detail is not None else (22.0 if sharpness is not None else None),
        sharpen_masking=recipe.detail.sharpen_masking if recipe.detail.sharpen_masking is not None else (55.0 if noisy else (35.0 if sharpness is not None else None)),
        luminance_nr=recipe.detail.luminance_nr if recipe.detail.luminance_nr is not None else luminance_nr,
        luminance_nr_detail=recipe.detail.luminance_nr_detail if recipe.detail.luminance_nr_detail is not None else (45.0 if luminance_nr is not None else None),
        luminance_nr_contrast=recipe.detail.luminance_nr_contrast if recipe.detail.luminance_nr_contrast is not None else (8.0 if luminance_nr is not None else None),
        color_nr=recipe.detail.color_nr if recipe.detail.color_nr is not None else (28.0 if noisy else (18.0 if not developed else None)),
        color_nr_detail=recipe.detail.color_nr_detail if recipe.detail.color_nr_detail is not None else (50.0 if noisy or not developed else None),
        color_nr_smoothness=recipe.detail.color_nr_smoothness if recipe.detail.color_nr_smoothness is not None else (50.0 if noisy or not developed else None),
    )
    optics = OpticsControls(lens_profile_enable=True)
    crop = CropControls()
    if geometry is not None and not geometry.is_noop():
        crop = CropControls(
            top=geometry.crop_top,
            left=geometry.crop_left,
            bottom=geometry.crop_bottom,
            right=geometry.crop_right,
            angle=-geometry.rotate_deg,
        )
    return recipe.model_copy(update={"detail": detail, "optics": optics, "crop": crop}).clamped()


def _technical_iso(exif: dict[str, str]) -> float:
    for key in ("ISOSpeedRatings", "PhotographicSensitivity", "ISO"):
        raw = exif.get(key)
        if not raw:
            continue
        try:
            return float(str(raw).split("/")[0].split()[0])
        except ValueError:
            continue
    return 0.0


def _adaptation_notes(recipe: Recipe, technical) -> list[str]:
    notes: list[str] = ["没有照搬示范数值；先按当前直方图限制曝光、黑位和高光。"]
    if technical.source_kind in {"jpeg", "png"}:
        notes.append("输入已经显影，降低了曝光、清晰度、去雾和整体风格强度，避免二次 HDR。")
    if technical.highlight_clip > 0.01 or technical.highlight_p95 > 0.92:
        notes.append("高光接近上限，因此进一步压低高光/白色色阶。")
    if technical.saturation_mean > 0.42:
        notes.append("原片饱和不低，先减容易脏的颜色，再保留自然饱和度。")
    if technical.mean_luma < 0.14:
        notes.append("保留低调意图，只整理暗部密度，不把画面强行提成白天。")
    if not recipe.curve.is_identity():
        notes.append("曲线先于基本滑杆定调，并控制为少锚点平滑过渡。")
    return notes


def _connected_lightroom(settings: Settings):
    return connect_lightroom(settings.lightcraft_bin)


def _prepare_lightroom_baseline(settings: Settings, original: Path, root: Path) -> dict:
    client, _status = _connected_lightroom(settings)
    inputs = root / "lightroom-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    baseline_copy = inputs / f"baseline{original.suffix.lower()}"
    shutil.copy2(original, baseline_copy)
    result = export_lightroom_baseline(client, original=baseline_copy,
        output_dir=root / "lightroom-exports" / "baseline", export_edge=settings.lightroom_export_edge)
    (root / "baseline.json").write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@dataclass
class _ValidCandidate:
    export: Path
    photo: Path
    recipe: Recipe
    report: LightroomQualityReport
    result: dict
    review: VisualReview
    distance: float | None
    round_index: int
    replaced: bool = False

def _render_variants_with_lightroom(
    settings: Settings,
    conn: sqlite3.Connection,
    job_id: str,
    original: Path,
    root: Path,
    variants: list[dict[str, object]],
    *,
    baseline_result: dict | None = None,
    critique: Critique | None = None,
) -> None:
    """Render every candidate in Lightroom before exposing the choice UI.

    Each variant gets a distinct file path, so Lightroom stores an independent
    Develop state and recipes cannot accumulate across A/B/C.
    """
    update_job(conn, job_id, step="lightroom")
    client, status = _connected_lightroom(settings)
    supported_tools = set(status.get("tools") or [])
    inputs = root / "lightroom-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    audit_root = root / "lightroom-audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    baseline_copy = inputs / f"baseline{original.suffix.lower()}"
    baseline_dir = root / "lightroom-exports" / "baseline"
    if baseline_result is None:
        if baseline_dir.exists():
            raise LightroomError(f"LightCraft 基线输出目录已存在：{baseline_dir}")
        shutil.copy2(original, baseline_copy)
        baseline_result = export_lightroom_baseline(client, original=baseline_copy,
            output_dir=baseline_dir, export_edge=settings.lightroom_export_edge)
    baseline_export = Path(baseline_result["export"])
    quality_regions = None
    try:
        from .semantic import _model_cached, segment_scene
        from .lightroom_qa import _read_pixels
        if settings.semantic_enabled and _model_cached(settings.semantic_model):
            regions = segment_scene(_read_pixels(baseline_export).rgb, settings)
            if regions is not None and regions.confidence >= .65:
                quality_regions = {"sky": regions.sky, "skin": regions.skin}
    except Exception:
        # Optional segmentation cannot make an otherwise usable photo fail.
        pass
    default_intent = critique.intent if critique else EditIntent()
    direction_technical = None
    direction_methods = {}
    if critique is not None and (root / "technical.json").is_file():
        from .models import TechnicalReport
        from .finishing_knowledge import method_ids
        direction_technical = TechnicalReport.model_validate_json((root / "technical.json").read_text())
        direction_methods = dict(zip("ABC", method_ids(critique, direction_technical)))
        direction_methods.update({d.slot: d.method_id for d in critique.directions if d.method_id})
    for row in variants:
        method = json.loads(str(row["recipe_json"])).get("style_workflow", {}).get("method_id")
        if method:
            direction_methods[str(row["slot"])] = method
    safe_recipe: Recipe | None = None
    accepted_candidates: list[Path] = []
    delivered_rows: list[dict[str, object]] = []
    omitted: list[dict[str, str]] = []
    lightroom_summary: dict[str, object] = {
        "baseline": str(baseline_export),
        "min_score": settings.lightroom_qa_min_score,
        "max_refinements": settings.lightroom_qa_max_refinements,
        "candidate_min_distance": settings.lightroom_candidate_min_distance,
        "slots": {},
    }
    for row in variants:
        slot = str(row["slot"])
        sidecar_intent = json.loads(str(row["recipe_json"])).get("intent")
        intent = EditIntent.model_validate(sidecar_intent) if sidecar_intent else default_intent
        preset_mode = direction_methods.get(slot, "").startswith("preset:")
        creative_palette = preset_mode or (slot in {"B", "C"} and bool(critique and critique.directions))
        review_kwargs = {}
        if critique is not None:
            def stage_safety(stage, before, after):
                checked = assess_lightroom_render(before or baseline_export, after, slot=slot,
                                                   min_score=settings.lightroom_qa_min_score, region_masks=quality_regions,
                                                   creative_palette=creative_palette)
                return {"decision": "same" if checked.accepted else "worse", "source": "pixel-check",
                        "rejects": not checked.accepted, "confidence": 1.0,
                        "issue_codes": [issue.code for issue in checked.issues if not issue.advisory],
                        "notes": [issue.note for issue in checked.issues]}
            review_kwargs["stage_reviewer"] = stage_safety
        suffix = original.suffix.lower()
        source_copy = inputs / f"variant_{slot}{suffix}"
        shutil.copy2(original, source_copy)
        recipe = Recipe.model_validate_json(str(row["recipe_json"]))
        original_recipe = recipe.model_copy(deep=True)
        cumulative_recipe = Recipe()
        rounds: list[dict[str, object]] = []
        accepted_export: Path | None = None
        accepted_photo: Path | None = None
        accepted_recipe: Recipe | None = None
        best_valid: _ValidCandidate | None = None
        used_similarity_fallback = False
        selected_round: int | None = None
        last_report = None
        last_pass = None
        visual_review = VisualReview()
        render_error = None
        nearest_candidate_distance: float | None = None
        total_rounds = settings.lightroom_qa_max_refinements + 1
        for attempt in range(total_rounds):
            export_dir = root / "lightroom-exports" / slot / f"round-{attempt}"
            if export_dir.exists():
                raise LightroomError(f"LightCraft 候选输出目录已存在：{export_dir}")
            try:
                pass_result = apply_recipe_pass(
                    client,
                    original=source_copy,
                    recipe=recipe,
                    output_dir=export_dir,
                    export_edge=settings.lightroom_export_edge,
                    **review_kwargs,
                )
            except LightroomError as exc:
                render_error = str(exc)
                rounds.append({"round": attempt, "mode": "initial" if attempt == 0 else "refine",
                               "recipe": recipe.model_dump(), "error": render_error})
                break
            effective = Recipe.model_validate(pass_result.get("effective_recipe", recipe.model_dump()))
            cumulative_recipe = cumulative_recipe.merged(effective).clamped()
            report = assess_lightroom_render(
                baseline_export,
                Path(pass_result["export"]),
                slot=slot,
                min_score=settings.lightroom_qa_min_score,
                region_masks=quality_regions,
                creative_palette=creative_palette,
            )
            candidate_distinct = True
            visual_review = review_render(settings, baseline_export, Path(pass_result["export"]), intent, "final")
            candidate_distance = None
            if report.accepted:
                candidate_distinct, candidate_distance = is_distinct_candidate(
                    Path(pass_result["export"]),
                    accepted_candidates,
                    min_distance=settings.lightroom_candidate_min_distance,
                )
            rounds.append(
                {
                    "round": attempt,
                    "mode": "initial" if attempt == 0 else "refine",
                    "recipe": recipe.model_dump(),
                    "settings_applied": pass_result["settings"],
                    "professional_stages": pass_result.get("stages", []),
                    "develop_before": pass_result["develop_before"],
                    "develop_after": pass_result["develop_after"],
                    "export": str(pass_result["export"]),
                    "pixel_qa": report.to_dict(),
                    "visual_review": visual_review.model_dump(),
                    "candidate_distinct": candidate_distinct,
                    "nearest_candidate_distance": candidate_distance,
                }
            )
            last_report = report
            last_pass = pass_result
            nearest_candidate_distance = candidate_distance
            if report.accepted and not visual_review.rejects:
                best_valid = _ValidCandidate(Path(pass_result["export"]), source_copy,
                    cumulative_recipe.model_copy(deep=True), report, pass_result, visual_review,
                    candidate_distance, len(rounds) - 1)
            if report.accepted and not visual_review.rejects and (candidate_distinct or not settings.lightroom_require_distinct_candidates):
                accepted_export = Path(pass_result["export"])
                accepted_photo = source_copy
                accepted_recipe = cumulative_recipe.model_copy(deep=True)
                selected_round = len(rounds) - 1
                break
            if report.accepted and visual_review.rejects and attempt < settings.lightroom_qa_max_refinements:
                delta = honor_locked_controls(visual_review.correction.as_recipe(), intent)
                if delta != Recipe():
                    recipe = delta
                    continue
            if report.accepted and (not candidate_distinct or visual_review.rejects):
                break
            if attempt < settings.lightroom_qa_max_refinements:
                recipe = honor_locked_controls(correction_for_report(report, strength=1.0 - attempt * 0.18), intent)

        if accepted_export is None and not preset_mode and slot == "A" and last_report is not None and last_report.accepted and visual_review.rejects:
            accepted_export, accepted_photo, accepted_recipe = baseline_export, baseline_copy, Recipe()
            last_report = assess_lightroom_render(baseline_export, baseline_export, slot=slot,
                                                  min_score=settings.lightroom_qa_min_score, region_masks=quality_regions)
            last_pass = {"settings": {}, "stages": []}
            visual_review = VisualReview(decision="same", confidence=1.0, source="retained-baseline",
                                         notes=["复核未发现可靠改善，交付保留的自然基线"])
            row["title"] = "保留自然基线"
            row["why"] = "当前调整未改善画面，保留 LightCraft 自然基线。"

        replaced = False
        if accepted_export is None and (preset_mode or (slot != "A" and safe_recipe is not None)) and render_error is None:
            # A quality failure may need a safer curve. A similar but valid
            # candidate needs its own light design, not A's curve again.
            base_weight = .60 if preset_mode else (0.40 if slot == "B" else 0.34)
            weight_step = 0.12
            max_replacement_attempts = 3
            separation_base = best_valid.recipe if best_valid else None
            separation_attempt = 0
            for rep_attempt in range(max_replacement_attempts):
                weight = None
                if separation_base is not None:
                    mode = "distinctness-replacement"
                    replacement = separate_candidate_recipe(separation_base, slot=slot, attempt=separation_attempt,
                                                            exposure_locked="exposure_ev" in intent.locked_controls,
                                                            method=direction_methods.get(slot), technical=direction_technical)
                    separation_attempt += 1
                elif visual_review.rejects:
                    mode = "visual-damage-repair"
                    replacement = repair_visual_recipe(original_recipe, visual_review, rep_attempt)
                else:
                    mode = "preset-strength-replacement" if preset_mode else "safe-style-replacement"
                    weight = float(np.clip(base_weight - rep_attempt * weight_step, 0.10, 0.68))
                    if preset_mode:
                        from .preference import reduce_style_strength, sanitize_correction
                        primary = (sanitize_correction(critique.correction, direction_technical).as_recipe()
                                   if critique is not None and direction_technical is not None else Recipe())
                        replacement = reduce_style_strength(original_recipe, primary, weight)
                    else:
                        replacement = blend_recipes(safe_recipe, original_recipe, style_weight=weight)
                replacement = honor_locked_controls(replacement, intent)
                rep_suffix = f"_rep{rep_attempt}" if rep_attempt > 0 else ""
                replacement_copy = inputs / f"variant_{slot}_replacement{rep_suffix}{suffix}"
                shutil.copy2(original, replacement_copy)
                replacement_dir = root / "lightroom-exports" / slot / f"replacement-{rep_attempt}"
                try:
                    replacement_result = apply_recipe_pass(
                        client,
                        original=replacement_copy,
                        recipe=replacement,
                        output_dir=replacement_dir,
                        export_edge=settings.lightroom_export_edge,
                        **review_kwargs,
                    )
                except LightroomError as exc:
                    render_error = str(exc)
                    rounds.append({"round": len(rounds), "mode": mode, "error": str(exc),
                                   "recipe": replacement.model_dump()})
                    break
                replacement_report = assess_lightroom_render(
                    baseline_export,
                    Path(replacement_result["export"]),
                    slot=slot,
                    min_score=settings.lightroom_qa_min_score,
                    region_masks=quality_regions,
                    creative_palette=creative_palette,
                )
                replacement_review = review_render(settings, baseline_export, Path(replacement_result["export"]), intent, "final")
                replacement_distinct = False
                replacement_distance = None
                if replacement_report.accepted:
                    replacement_distinct, replacement_distance = is_distinct_candidate(
                        Path(replacement_result["export"]),
                        accepted_candidates,
                        min_distance=settings.lightroom_candidate_min_distance,
                    )
                rounds.append(
                    {
                        "round": len(rounds),
                        "mode": mode,
                        "recipe": replacement.model_dump(),
                        "style_weight": weight,
                        "settings_applied": replacement_result["settings"],
                        "professional_stages": replacement_result.get("stages", []),
                        "develop_before": replacement_result["develop_before"],
                        "develop_after": replacement_result["develop_after"],
                        "export": str(replacement_result["export"]),
                        "pixel_qa": replacement_report.to_dict(),
                        "visual_review": replacement_review.model_dump(),
                        "candidate_distinct": replacement_distinct,
                        "nearest_candidate_distance": replacement_distance,
                    }
                )
                last_report = replacement_report
                last_pass = replacement_result
                visual_review = replacement_review
                nearest_candidate_distance = replacement_distance
                if replacement_report.accepted and not replacement_review.rejects:
                    candidate = _ValidCandidate(Path(replacement_result["export"]), replacement_copy,
                        Recipe.model_validate(replacement_result.get("effective_recipe", replacement.model_dump())),
                        replacement_report, replacement_result, replacement_review, replacement_distance,
                        len(rounds) - 1, replaced=True)
                    # Technical acceptance is a gate. A larger distance from
                    # another slot, or a higher passing QA score, is not visual
                    # evidence that this replacement is a better photograph.
                    # Retain the first valid design if the diversity search
                    # never finds an eligible distinct candidate.
                    if best_valid is None:
                        best_valid = candidate
                    if separation_base is None:
                        separation_base = candidate.recipe
                if replacement_report.accepted and not replacement_review.rejects and (replacement_distinct or not settings.lightroom_require_distinct_candidates):
                    accepted_export = Path(replacement_result["export"])
                    accepted_photo = replacement_copy
                    accepted_recipe = Recipe.model_validate(replacement_result.get("effective_recipe", replacement.model_dump()))
                    visual_review = replacement_review
                    replaced = True
                    selected_round = len(rounds) - 1
                    break

        # Diversity is a selection target, not a pixel defect. Exhausting the
        # bounded search must not discard already valid work or skip C. Restore
        # the matching pixels, recipe, Lightroom photo and audit from the first
        # valid attempt, even when later attempts changed more or failed QA.
        if accepted_export is None and best_valid is not None:
            used_similarity_fallback = True
            accepted_export, accepted_photo, accepted_recipe = best_valid.export, best_valid.photo, best_valid.recipe
            last_report, last_pass, visual_review = best_valid.report, best_valid.result, best_valid.review
            nearest_candidate_distance, selected_round = best_valid.distance, best_valid.round_index
            replaced = best_valid.replaced

        if accepted_export is None or accepted_photo is None or accepted_recipe is None or last_report is None or last_pass is None:
            issue_text = render_error or "、".join(issue.note for issue in (last_report.issues if last_report else []))
            if visual_review.rejects and not render_error:
                issue_text = "；".join(visual_review.notes)
            (audit_root / f"variant_{slot}.json").write_text(json.dumps({
                "slot": slot, "status": "failed", "reason": issue_text, "rounds": rounds,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            # A rejected optional direction cannot discard accepted photos or
            # prevent the next direction from being rendered. It stays out of
            # the delivered rows; do not relabel damaged pixels as accepted.
            omitted.append({"slot": slot, "reason": issue_text or "未通过成片检查"})
            lightroom_summary["slots"][slot] = {"status": "withheld", "reason": issue_text, "rounds": len(rounds)}
            continue

        # A safe first render is a retained candidate, not proof that the
        # intended light and color are finished. Refine from its EFFECTIVE
        # recipe on an independent source so a rejected attempt cannot mutate
        # the photo, recipe or export that will be delivered.
        finishing_records: list[dict[str, object]] = []
        if settings.finishing_review_enabled and critique is not None:
            from .ingest import source_kind as detect_source_kind
            source_kind = direction_technical.source_kind if direction_technical else detect_source_kind(original)
            for finishing_attempt in range(settings.finishing_max_rounds):
                proposal = propose_finishing(settings, baseline_export, accepted_export,
                                             accepted_recipe, intent, source_kind)
                finishing_record = {"attempt": finishing_attempt, "proposal": proposal.model_dump(),
                                    "retained_export": str(accepted_export), "selected": False}
                finishing_records.append(finishing_record)
                if proposal.recipe is None:
                    break
                finishing_copy = inputs / f"variant_{slot}_finishing_{finishing_attempt}{suffix}"
                finishing_dir = root / "lightroom-exports" / slot / f"finishing-{finishing_attempt}"
                try:
                    shutil.copy2(original, finishing_copy)
                    finishing_result = apply_recipe_pass(client, original=finishing_copy,
                        recipe=proposal.recipe, output_dir=finishing_dir,
                        export_edge=settings.lightroom_export_edge, **review_kwargs)
                    finishing_export = Path(finishing_result["export"])
                    finishing_effective = Recipe.model_validate(finishing_result.get("effective_recipe", proposal.recipe.model_dump()))
                    finishing_report = assess_lightroom_render(baseline_export, finishing_export,
                        slot=slot, min_score=settings.lightroom_qa_min_score,
                        region_masks=quality_regions, creative_palette=creative_palette,
                        previous_accepted_path=accepted_export)
                    finishing_record.update(export=str(finishing_export), pixel_qa=finishing_report.to_dict())
                    finishing_comparison = None
                    finishing_review = VisualReview()
                    if finishing_report.accepted:
                        finishing_comparison = compare_finishing(settings, accepted_export, finishing_export, intent)
                        finishing_record["comparison"] = finishing_comparison.model_dump()
                        if finishing_comparison.accepts:
                            finishing_review = review_render(settings, baseline_export, finishing_export, intent, "final")
                    finishing_distinct, finishing_distance = False, None
                    if finishing_report.accepted:
                        finishing_distinct, finishing_distance = is_distinct_candidate(finishing_export,
                            accepted_candidates, min_distance=settings.lightroom_candidate_min_distance)
                    selected = bool(finishing_report.accepted and finishing_comparison
                                    and finishing_comparison.accepts and not finishing_review.rejects)
                    finishing_record["selected"] = selected
                    rounds.append({"round": len(rounds), "mode": "finishing", "recipe": proposal.recipe.model_dump(),
                        "settings_applied": finishing_result["settings"],
                        "professional_stages": finishing_result.get("stages", []),
                        "develop_before": finishing_result["develop_before"],
                        "develop_after": finishing_result["develop_after"],
                        "export": str(finishing_export), "pixel_qa": finishing_report.to_dict(),
                        "visual_review": finishing_review.model_dump(),
                        "finishing_comparison": finishing_comparison.model_dump() if finishing_comparison else None,
                        "selected": selected, "candidate_distinct": finishing_distinct,
                        "nearest_candidate_distance": finishing_distance})
                    if not selected:
                        break
                    accepted_export, accepted_photo = finishing_export, finishing_copy
                    accepted_recipe = finishing_effective
                    last_report, last_pass, visual_review = finishing_report, finishing_result, finishing_review
                    nearest_candidate_distance, selected_round = finishing_distance, len(rounds) - 1
                except (LightroomError, OSError, ValueError) as exc:
                    finishing_record["error"] = str(exc)
                    break
        if slot == "A":
            safe_recipe = accepted_recipe.model_copy(deep=True)
        accepted_candidates.append(accepted_export)

        checkpoint = None
        checkpoint_error = None
        if "create_develop_preset" in supported_tools:
            try:
                checkpoint = create_lightroom_checkpoint(
                    client,
                    photo=accepted_photo,
                    name=f"SuperPhoto-{job_id[:8]}-{slot}-accepted-{uuid.uuid4().hex[:8]}",
                    settings=last_pass["settings"],
                )
            except LightroomError as exc:
                checkpoint_error = str(exc)

        final_path = root / str(row["filename"])
        shutil.copy2(accepted_export, final_path)
        sidecar_payload = json.loads(str(row["recipe_json"]))
        sidecar_payload.update(accepted_recipe.model_dump())
        if visual_review.source == "retained-baseline":
            workflow = sidecar_payload.setdefault("style_workflow", {})
            workflow.update(requested_method_id=workflow.get("method_id"), method_id="", preset_id=None,
                            preset_amount=0, delivery="baseline", route_label="保留自然基线", selection_reason="")
            row["look_id"] = "faithful"
        sidecar_payload["lightroom_pixel_qa"] = last_report.to_dict()
        sidecar_payload["lightroom_refined"] = len(rounds) > 1
        sidecar_payload["lightroom_replaced"] = replaced
        sidecar_payload["visual_review"] = visual_review.model_dump()
        sidecar_payload["finishing_review"] = finishing_records
        selection = {
            "distinct": nearest_candidate_distance is None or nearest_candidate_distance >= settings.lightroom_candidate_min_distance,
            "nearest_distance": nearest_candidate_distance,
            "target_distance": settings.lightroom_candidate_min_distance,
            "selected_round": selected_round,
            "fallback_policy": "first_valid_design" if used_similarity_fallback else None,
        }
        sidecar_payload["candidate_selection"] = selection
        row["recipe_json"] = json.dumps(sidecar_payload, ensure_ascii=False)
        quality_mark = "LightCraft 技术检查通过"
        row["why"] = f"{str(row.get('why') or '').rstrip()} {quality_mark}。".strip()
        if not selection["distinct"]:
            row["why"] += " 与其他方案的差别较细微。"
        audit = {
            "slot": slot,
            "source_copy": str(source_copy),
            "baseline_export": str(baseline_export),
            "accepted_export": str(accepted_export),
            "accepted_photo": str(accepted_photo),
            "accepted_recipe": accepted_recipe.model_dump(),
            "accepted_score": last_report.score,
            "visual_review": visual_review.model_dump(),
            "finishing_review": finishing_records,
            "accepted_checkpoint": checkpoint,
            "checkpoint_error": checkpoint_error,
            "nearest_candidate_distance": nearest_candidate_distance,
            "refined": len(rounds) > 1,
            "replaced": replaced,
            "candidate_selection": selection,
            "rounds": rounds,
        }
        (audit_root / f"variant_{slot}.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (root / f"variant_{slot}.json").write_text(json.dumps(sidecar_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        lightroom_summary["slots"][slot] = {
            "status": "delivered",
            "score": last_report.score,
            "rounds": len(rounds),
            "stages": [stage["stage"] for stage in last_pass.get("stages", [])],
            "stage_count": len(last_pass.get("stages", [])),
            "checkpoint_capable": "create_develop_preset" in supported_tools,
            "checkpoint_created": checkpoint is not None,
            "checkpoint_error": checkpoint_error,
            "replaced": replaced,
            "nearest_candidate_distance": nearest_candidate_distance,
            "candidate_selection": selection,
            "issues": [issue.code for issue in last_report.issues],
            "final": str(final_path),
        }
        delivered_rows.append(row)
    replace_variants(conn, job_id, delivered_rows)
    lightroom_summary["delivery"] = {"requested": len(variants), "delivered": len(delivered_rows), "omitted": omitted}
    (root / "lightroom-qa.json").write_text(
        json.dumps(lightroom_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not delivered_rows:
        raise LightroomError("本次所有方案均未通过成片检查，未交付有问题的图片。原片已保留，可以重试。")
