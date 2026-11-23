"""Re-render a saved job's exact recipes in a fresh job, keeping its history.

By default this skips replanning for an exact recipe regression. --redesign
runs the current complete pipeline on the same original in a fresh job.
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import os
import traceback
from uuid import UUID, uuid4

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=UUID)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--redesign", action="store_true", help="run the current analysis and planning pipeline")
    parser.add_argument("--restart", action="store_true", help="restart Lightroom normally before validation")
    parser.add_argument("--frozen-source", type=Path, help="use an archived superphoto package for a reproducible baseline")
    parser.add_argument("--recover-sidecars", action="store_true", help="recover failed-job recipes from saved sidecars when no variants were delivered")
    parser.add_argument("--slots", default="ABC", choices=["A", "B", "C", "ABC"], help="render selected saved slots for a focused regression")
    parser.add_argument("--initial-recipes", action="store_true", help="retest each audit's original plan before QA corrections")
    parser.add_argument("--rebuild-plans", action="store_true", help="apply current plan composition to the saved model directions, without another model call")
    parser.add_argument("--skip-unconfirmed-subject", action="store_true", help="omit an impossible subject mask only when the saved critique explicitly reports no subject")
    args = parser.parse_args()
    if args.frozen_source:
        if not (args.frozen_source / "superphoto" / "pipeline.py").is_file():
            raise ValueError("Frozen source must contain a superphoto package")
        os.environ["SUPERPHOTO_ROOT"] = str(Path(__file__).resolve().parents[1])
        sys.path.insert(0, str(args.frozen_source.resolve()))
    from superphoto.config import Settings
    from superphoto.db import connect, get_job, get_variants, init_db, insert_job, update_job
    from superphoto.lightroom import close_lightroom_client
    from superphoto.models import Critique
    from superphoto.pipeline import _render_variants_with_lightroom, run_job
    source_data = args.data.resolve()
    with sqlite3.connect(f"file:{source_data / 'superphoto.db'}?mode=ro", uri=True) as source_conn:
        source_conn.row_factory = sqlite3.Row
        saved = get_job(source_conn, str(args.job_id))
        variants = get_variants(source_conn, str(args.job_id))
    source = source_data / "jobs" / str(args.job_id)
    if args.recover_sidecars and saved and saved["status"] == "error" and not variants:
        for plan in json.loads((source / "plan.json").read_text(encoding="utf-8")):
            sidecar = json.loads((source / f"variant_{plan['slot']}.json").read_text(encoding="utf-8"))
            recipe = sidecar["recipe"]
            recipe.update(intent=plan["intent"], style_workflow={"method_id": plan.get("method_id")})
            variants.append({"slot": plan["slot"], "look_id": sidecar["look_id"],
                             "title": plan["title"], "why": plan["why"],
                             "recipe_json": json.dumps(recipe, ensure_ascii=False),
                             "filename": f"variant_{plan['slot']}.jpg"})
    if not saved or saved["status"] not in {"ready", "error"} or not variants:
        raise ValueError("Replay requires a finished job with saved recipes")
    if not set(args.slots).issubset({r["slot"] for r in variants}):
        raise ValueError("The saved job does not contain every requested slot")
    omitted_masks = []
    variants = [row for row in variants if row["slot"] in args.slots]
    if args.initial_recipes:
        for row in variants:
            audit = json.loads((source / "lightroom-audits" / f"variant_{row['slot']}.json").read_text(encoding="utf-8"))
            initial = audit["rounds"][0]["recipe"]
            payload = json.loads(row["recipe_json"])
            payload.update(initial)
            row["recipe_json"] = json.dumps(payload, ensure_ascii=False)
    rebuilt = None
    if args.rebuild_plans:
        from superphoto.models import TechnicalReport
        from superphoto.preference import build_plans
        from superphoto.pipeline import _professionalize_recipe
        critique = Critique.model_validate_json((source / "critique.json").read_text(encoding="utf-8"))
        technical = TechnicalReport.model_validate_json((source / "technical.json").read_text(encoding="utf-8"))
        old_plans = json.loads((source / "plan.json").read_text(encoding="utf-8"))
        rebuilt = build_plans([p["look_id"] for p in old_plans], critique, technical)
        for plan in rebuilt:
            plan.recipe = _professionalize_recipe(plan.recipe, technical, plan.geometry)
        for row in variants:
            plan = next(p for p in rebuilt if p.slot == row["slot"])
            payload = json.loads(row["recipe_json"])
            payload.update(plan.recipe.model_dump(), intent=plan.intent.model_dump())
            row["recipe_json"] = json.dumps(payload, ensure_ascii=False)
    if args.skip_unconfirmed_subject:
        intent = json.loads((source / "critique.json").read_text(encoding="utf-8"))["intent"]
        if intent.get("subject_confidence") != 0 or intent.get("subject_box") is not None:
            raise ValueError("Saved critique does not explicitly report an unconfirmed subject")
        for row in variants:
            recipe = json.loads(row["recipe_json"])
            omitted_masks.extend({"slot": row["slot"], **m} for m in recipe.get("masks", []) if m["target"] == "subject")
            recipe["masks"] = [m for m in recipe.get("masks", []) if m["target"] != "subject"]
            row["recipe_json"] = json.dumps(recipe, ensure_ascii=False)
    original_name = Path(saved.get("original_file") or "original.jpg").name
    if not (source / original_name).is_file():
        raise FileNotFoundError(source / original_name)
    settings = Settings(data_dir=(args.out or source_data).resolve(), lightroom_enabled=True)
    conn = connect(settings.db_path)
    init_db(conn)
    new_id = str(uuid4())
    root = settings.jobs_dir / new_id
    root.mkdir(parents=True)
    original = root / original_name
    shutil.copy2(source / original_name, original)
    for name in ("technical.json", "critique.json", "plan.json", "style-decision.json", "source_preview.jpg",
                 "critique_preview.jpg", "directions-audit.json", "workflow.json", "qa.json", "restoration.json"):
        if not args.redesign and (source / name).is_file():
            shutil.copy2(source / name, root / name)
    (root / "replay.json").write_text(json.dumps({"source_job": str(args.job_id), "replanned": args.redesign,
        "recovered_sidecars": args.recover_sidecars, "frozen_source": str(args.frozen_source) if args.frozen_source else None,
        "omitted_unconfirmed_subject_masks": omitted_masks, "slots": args.slots,
        "initial_recipes": args.initial_recipes, "rebuild_plans": args.rebuild_plans}, indent=2))
    if rebuilt is not None:
        (root / "plan.json").write_text(json.dumps([p.model_dump() for p in rebuilt], ensure_ascii=False, indent=2))
    for row in variants:
        row["id"], row["job_id"] = str(uuid4()), new_id
    insert_job(conn, new_id, saved["original_name"])
    update_job(conn, new_id, status="running", step="lightroom", original_file=original.name,
               technical_json=saved["technical_json"], critique_json=saved["critique_json"], plan_json=saved["plan_json"])
    print(json.dumps({"job_id": new_id, "replay_of": str(args.job_id)}, ensure_ascii=False), flush=True)
    try:
        if args.restart:
            from verify_native_masks import restart_lightroom
            restart_lightroom()
        if args.redesign:
            run_job(settings, conn, new_id, original)
        else:
            critique = Critique.model_validate_json((root / "critique.json").read_text(encoding="utf-8"))
            _render_variants_with_lightroom(settings, conn, new_id, original, root, variants, critique=critique)
        update_job(conn, new_id, status="ready", step="ready", error=None)
        print(json.dumps({"job_id": new_id, "status": "ready", "dir": str(root)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_job(conn, new_id, status="error", step="error", error=f"{exc}\n{traceback.format_exc()}")
        raise
    finally:
        close_lightroom_client()
        conn.close()


if __name__ == "__main__":
    main()
