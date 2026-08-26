from __future__ import annotations

import json
import sqlite3
import traceback
import uuid
from pathlib import Path

import numpy as np

from .analysis import analyze
from .color import linear_to_srgb
from .composition import apply_geometry
from .config import Settings
from .critique import compose_look_ids, critique_photo
from .darktable_export import write_recipe_sidecar
from .db import replace_variants, update_job
from .ingest import downscale_linear, load_linear
from .layers import build_layers
from .preference import build_plans, preferred_look_ids
from .qa import refine_plan, score_grade, target_for
from .render import apply_recipe, export_srgb_jpeg
from .store import job_dir


def run_job(settings: Settings, conn: sqlite3.Connection, job_id: str, original: Path) -> None:
    root = job_dir(settings, job_id)
    try:
        update_job(conn, job_id, status="running", step="ingest", original_file=original.name)
        linear, _meta = load_linear(original)
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

        update_job(conn, job_id, step="critique", technical_json=technical.model_dump_json())
        critique = critique_photo(settings, root / "critique_preview.jpg", technical)
        (root / "critique.json").write_text(
            json.dumps(critique.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        look_ids = compose_look_ids(critique, preferred_look_ids(conn), technical)
        plans = build_plans(look_ids, critique, technical)
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
        for plan in plans:
            frame = working if plan.geometry.is_noop() else apply_geometry(working, plan.geometry)
            frame_srgb = np.clip(linear_to_srgb(np.maximum(frame, 0.0)), 0.0, 1.0)
            layers = build_layers(frame_srgb, technical, critique.clutter)
            best_img = None
            best_qa = None
            best_plan = plan
            current = plan
            for attempt in range(3):
                rendered = apply_recipe(
                    frame,
                    current.recipe,
                    layers=layers,
                    technical=technical,
                    clutter_boxes=critique.clutter,
                )
                qa = score_grade(frame, rendered, technical, current.slot)
                if best_qa is None or qa.score > best_qa.score:
                    best_img, best_qa, best_plan = rendered, qa, current
                if qa.score >= target_for(current.slot) and not qa.too_strong:
                    break
                if current.slot == "A" and qa.score >= 70 and not qa.too_strong:
                    break
                if attempt == 2:
                    break
                current = refine_plan(current, qa, critique, technical, attempt)

            update_job(conn, job_id, step="qa")
            out_name = f"variant_{plan.slot}.jpg"
            export_srgb_jpeg(best_img, root / out_name, quality=92, max_edge=settings.preview_edge)
            sidecar = best_plan.recipe.model_dump()
            sidecar["lut_mix"] = best_plan.lut_mix
            sidecar["geometry"] = best_plan.geometry.model_dump()
            sidecar["qa_score"] = round(best_qa.score, 1)
            sidecar["qa_notes"] = best_qa.notes
            write_recipe_sidecar(root / f"variant_{plan.slot}.json", best_plan.recipe, best_plan.look_id)
            why = best_plan.why
            mark = f"自检 {best_qa.score:.0f} 分"
            if mark not in why:
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
        (root / "qa.json").write_text(
            json.dumps({"slots": qa_summaries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        replace_variants(conn, job_id, variant_rows)
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
