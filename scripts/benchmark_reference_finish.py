"""Reference-guided model experiment on a verified, delivered Lightroom result.

This is explicitly different from the application's reference-free planner. A
reference ROI is used only as in-memory model input; all candidate photographs
are exported by Lightroom from independent copies of the original source.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

from superphoto.config import Settings
from superphoto.critique import _extract_json
from superphoto.finishing_review import (
    _ProposalResponse, _apply_delta, _intent_context, decision_from_pair,
)
from superphoto.pair_review import build_pair_board, compare_finished_pair
from superphoto.ingest import ALLOWED_EXTS, image_to_srgb, source_kind, srgb_profile_bytes
from superphoto.lightroom import (
    apply_recipe_pass, close_lightroom_client, get_lightroom_client, wait_for_lightroom_connection,
)
from superphoto.lightroom_qa import assess_lightroom_render
from superphoto.models import EditIntent, Recipe
from superphoto.ollama import chat_vision, probe
from superphoto.review import _preview


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _log(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def load_verified_job(job: Path, slot: str) -> dict:
    """Reject local previews, stale sidecars and unpaired native exports."""
    job = job.resolve()
    audit_path = job / "lightroom-audits" / f"variant_{slot}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") == "failed":
        raise ValueError("The requested slot did not deliver a native Lightroom result")
    delivered = job / f"variant_{slot}.jpg"
    exported = Path(audit["accepted_export"]).resolve()
    native_photo = Path(audit["accepted_photo"]).resolve()
    baseline = Path(audit["baseline_export"]).resolve()
    sidecar_path = job / f"variant_{slot}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    recipe = Recipe.model_validate(audit["accepted_recipe"])
    if recipe != Recipe.model_validate(sidecar):
        raise ValueError("The delivered recipe differs from the accepted native audit")
    if _hash(delivered) != _hash(exported):
        raise ValueError("Delivered bytes differ from the accepted native Lightroom export")
    originals = [path for path in job.iterdir()
                 if path.is_file() and path.stem == "original" and path.suffix.lower() in ALLOWED_EXTS]
    if len(originals) != 1:
        raise ValueError("Expected exactly one original source photograph in the completed job")
    original = originals[0]
    if _hash(original) != _hash(native_photo):
        raise ValueError("The accepted native photograph does not match the job original")
    if not baseline.is_file():
        raise ValueError("The original native baseline export is missing")
    summary_path = job / "lightroom-qa.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("slots", {}).get(slot, {}).get("status") != "delivered":
        raise ValueError("The job summary does not confirm delivery of this slot")
    if sidecar.get("intent"):
        intent = EditIntent.model_validate(sidecar["intent"])
    elif (job / "critique.json").is_file():
        intent = EditIntent.model_validate(json.loads((job / "critique.json").read_text(encoding="utf-8"))["intent"])
    else:
        raise ValueError("The delivered result has no recorded photographic intent")
    return {"job": job, "slot": slot, "original": original, "baseline": baseline,
            "export": exported, "photo": native_photo, "recipe": recipe, "intent": intent,
            "provenance": {"audit": str(audit_path), "audit_sha256": _hash(audit_path),
                "sidecar": str(sidecar_path), "sidecar_sha256": _hash(sidecar_path),
                "original": str(original), "original_sha256": _hash(original),
                "baseline": str(baseline), "baseline_sha256": _hash(baseline),
                "delivered": str(delivered), "delivered_sha256": _hash(delivered),
                "accepted_export": str(exported), "accepted_export_sha256": _hash(exported),
                "accepted_photo": str(native_photo), "accepted_photo_sha256": _hash(native_photo)}}


def reference_preview(path: Path, roi: tuple[int, int, int, int]) -> tuple[str, dict]:
    """Crop a screenshot for model inspection, without writing an edited photo."""
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        width, height = oriented.size
        left, top, right, bottom = roi
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Reference ROI {roi} is outside the {width}x{height} source")
        image = image_to_srgb(oriented).crop(roi)
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=90, icc_profile=srgb_profile_bytes())
    return base64.b64encode(stream.getvalue()).decode("ascii"), {
        "path": str(path.resolve()), "sha256": _hash(path), "roi": roi,
        "source_size": [width, height], "usage": "in-memory sRGB model reference only"}


def _proposal_prompt(recipe: Recipe, intent: EditIntent, kind: str) -> str:
    return f"""这是明确使用同图成片作目标的修图实验。单张画布左侧编号1是当前 Lightroom 成片，右侧编号2是参考。
比较同一场景的明暗组织、色彩深度、光带与主体关系，先写至少一条具体区域 observations。
指出当前版相对参考还有哪些可见不足，最多三处 weaknesses，每处含 region、evidence、goal。
参考是网页图片，可能有裁切或压缩，不要机械追逐其像素、锐化光晕或整体亮度。
只调整摄影光色，不合成内容、不改变几何；没有明确可改善处可以不改，不能为了差异瞎调。
只返回 JSON：confidence（0到1）、observations（字符串数组）、weaknesses（上述对象数组）、
delta（仅写变化字段），可选 notes（字符串数组）。delta 是叠加在当前配方上的增量。
支持全局 exposure_ev ±0.15 EV；highlights/shadows/whites/blacks/contrast ±0.20；texture ±0.12。
hsl 使用 red/orange/yellow/green/aqua/blue/purple/magenta，hue ±3度，sat/lum ±0.20。
masks 只能使用现有 target，每个一次，exposure_ev ±{'.65' if kind == 'raw' else '.20'} EV；
contrast/whites/highlights ±0.20，其余仅小调整。禁新增蒙版、改 reason、色轮、曲线、全局白平衡。
遵守锁定参数、自然色、黑位与氛围保护，参考与保护冲突时不强行模仿。
意图：{_intent_context(intent)}
源类型：{kind}。当前非默认参数：{recipe.model_dump_json(exclude_defaults=True, exclude_none=True)}
/no_think"""




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--slot", choices=["A", "B", "C"], default="B")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-roi", type=int, nargs=4, required=True,
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), help="Pixel coordinates in the oriented reference image")
    parser.add_argument("--out", type=Path, required=True, help="New experiment directory; never overwritten")
    parser.add_argument("--rounds", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    saved = load_verified_job(args.job, args.slot)
    reference, reference_info = reference_preview(args.reference, tuple(args.reference_roi))
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = Settings(lightroom_enabled=True)
    original, baseline = saved["original"], saved["baseline"]
    accepted_export, accepted_photo, recipe = saved["export"], saved["photo"], saved["recipe"]
    intent, kind = saved["intent"], source_kind(original)
    audit = {"planner": "reference-guided-model", "reference_free_automatic": False,
             "status": "running", "slot": args.slot, "round_limit": args.rounds,
             "pixel_qa_settings": {"min_score": settings.lightroom_qa_min_score, "creative_palette": True},
             "source": saved["provenance"], "reference": reference_info, "rounds": [],
             "note": "作者参考引导的模型精修实验；不是无参考自动后期结果。"}
    _write(root / "audit.json", audit)
    _log(root=root, planner=audit["planner"], verified_export=accepted_export)
    try:
        status = probe(settings)
        if not status.get("ready") or not status.get("resolved_model"):
            raise RuntimeError("Vision model unavailable; retaining verified delivered result")
        audit["model"] = status["resolved_model"]
        client = None
        for attempt in range(args.rounds):
            record = {"attempt": attempt, "retained_export": str(accepted_export),
                      "retained_recipe": recipe.model_dump(), "selected": False}
            audit["rounds"].append(record)
            round_dir = root / f"round-{attempt}"
            round_dir.mkdir()
            try:
                _log(attempt=attempt, phase="model-proposal", current=accepted_export)
                prompt = _proposal_prompt(recipe, intent, kind)
                transport_audit = {}
                board = build_pair_board(_preview(accepted_export), reference)
                (round_dir / "proposal-board.jpg").write_bytes(base64.b64decode(board))
                record.update(proposal_prompt=prompt, proposal_transport=transport_audit)
                raw = chat_vision(settings, status["resolved_model"], prompt, board, audit=transport_audit)
                (round_dir / "proposal-raw.txt").write_text(raw, encoding="utf-8")
                record.update(proposal_prompt=prompt, proposal_raw=raw, proposal_transport=transport_audit)
                _write(root / "audit.json", audit)
                proposal = _ProposalResponse.model_validate(_extract_json(raw))
                record["proposal"] = proposal.model_dump()
                if proposal.confidence < .70 or not proposal.weaknesses:
                    record["retained_reason"] = "没有足够明确的参考差距与调整依据"
                    _log(attempt=attempt, phase="retained", reason=record["retained_reason"])
                    break
                candidate = _apply_delta(recipe, proposal.delta, intent, kind)
                if candidate is None:
                    record["retained_reason"] = "边界约束后没有有效变化"
                    _log(attempt=attempt, phase="retained", reason=record["retained_reason"])
                    break
                record["proposed_recipe"] = candidate.model_dump()
                _write(round_dir / "proposed-recipe.json", candidate.model_dump())
                source_copy = round_dir / ("source" + original.suffix.lower())
                shutil.copy2(original, source_copy)
                if _hash(source_copy) != saved["provenance"]["original_sha256"]:
                    raise ValueError("Independent source copy no longer matches the verified original")
                if client is None:
                    client = get_lightroom_client(settings.lightroom_mcp_server)
                    if not wait_for_lightroom_connection(client, timeout=45).get("connected"):
                        raise RuntimeError("Lightroom did not connect")

                def stage_safety(stage, before, after):
                    quality = assess_lightroom_render(before or baseline, after, slot=args.slot,
                        min_score=settings.lightroom_qa_min_score, creative_palette=True)
                    return {"decision": "same" if quality.accepted else "worse", "source": "pixel-check",
                            "rejects": not quality.accepted, "confidence": 1.0,
                            "issue_codes": [issue.code for issue in quality.issues if not issue.advisory],
                            "notes": [issue.note for issue in quality.issues]}

                _log(attempt=attempt, phase="native-render", source=source_copy,
                     proposed_recipe=round_dir / "proposed-recipe.json")
                rendered = apply_recipe_pass(client, original=source_copy, recipe=candidate,
                    output_dir=round_dir / "lightroom", export_edge=settings.lightroom_export_edge,
                    stage_reviewer=stage_safety)
                record["rendered"] = rendered
                _write(round_dir / "rendered.json", rendered)
                exported = Path(rendered["export"])
                effective = Recipe.model_validate(rendered["effective_recipe"])
                quality = assess_lightroom_render(baseline, exported, slot=args.slot,
                    min_score=settings.lightroom_qa_min_score, creative_palette=True,
                    previous_accepted_path=accepted_export)
                record.update(effective_recipe=effective.model_dump(), pixel_qa=quality.to_dict(),
                              export_sha256=_hash(exported))
                _write(root / "audit.json", audit)
                _log(attempt=attempt, phase="native-export", export=exported,
                     pixel_accepted=quality.accepted, issues=[issue.code for issue in quality.issues])
                if not quality.accepted:
                    record["retained_reason"] = "候选未通过像素质量检查"
                    break
                pair = compare_finished_pair(settings, accepted_export, exported, intent,
                                             reference_preview=reference)
                decision = decision_from_pair(pair)
                _write(round_dir / "comparison.json", pair.model_dump())
                (round_dir / "comparison-raw.txt").write_text(decision.raw_response or "", encoding="utf-8")
                record["comparison"] = decision.model_dump()
                if decision.accepts:
                    accepted_export, accepted_photo, recipe = exported, source_copy, effective
                    record["selected"] = True
                _log(attempt=attempt, phase="comparison", decision=decision.decision,
                     confidence=decision.confidence, selected=record["selected"], retained=accepted_export)
                if not record["selected"]:
                    break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                _log(attempt=attempt, phase="retained-after-error", error=record["error"], retained=accepted_export)
                break
            finally:
                _write(root / "audit.json", audit)
        audit["status"] = "complete"
    except Exception as exc:
        audit.update(status="retained-after-error", error=f"{type(exc).__name__}: {exc}")
        _log(phase="retained-after-error", error=audit["error"])
    finally:
        close_lightroom_client()
        final = root / "final.jpg"
        shutil.copy2(accepted_export, final)
        _write(root / "accepted-recipe.json", recipe.model_dump())
        audit.update(accepted_export=str(accepted_export), accepted_photo=str(accepted_photo),
                     accepted_recipe=recipe.model_dump(), final=str(final), final_sha256=_hash(final),
                     improved=any(record["selected"] for record in audit["rounds"]))
        _write(root / "audit.json", audit)
        _log(root=root, final=final, improved=audit["improved"], status=audit["status"])


if __name__ == "__main__":
    main()
