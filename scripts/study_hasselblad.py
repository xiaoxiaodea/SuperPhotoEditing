"""Study every local Hasselblad image with an auditable, resumable vision pass.

Reads the supplied ph-main archive and the older project collection. Source
images are never edited. Each file retains a coverage entry, even if identical.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import httpx
from PIL import Image, ImageOps
import numpy as np
from pydantic import Field

from superphoto.config import get_settings
from superphoto.critique import _extract_json
from superphoto.ingest import image_to_srgb
from superphoto.master_studies import Principle, PRINCIPLE_GUIDANCE, StudyAnalysis
from superphoto.ollama import _assistant_text, probe
from superphoto.analysis import analyze
from superphoto.color import srgb_to_linear


def inventory(primary: Path, legacy: Path) -> list[dict]:
    manifest = [json.loads(line) for line in (primary / "manifest.jsonl").read_text().splitlines() if line.strip()]
    main_meta = {row["path"]: row for row in manifest}
    catalog_path = legacy.parent / "catalog.json"
    old_meta = {}
    if catalog_path.is_file():
        old_meta = {Path(row["file"]).name: row for row in json.loads(catalog_path.read_text())["items"]
                    if row.get("award") == "hasselblad_masters"}
    records = {}
    for corpus, root in (("ph-main", primary), ("legacy", legacy)):
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
                continue
            relative = path.relative_to(root).as_posix()
            meta = main_meta.get(relative, {}) if corpus == "ph-main" else old_meta.get(path.name, {})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source = {"corpus": corpus, "path": str(path.resolve()), "relative_path": relative,
                      "photographer": meta.get("photographer", "未标注"), "year": str(meta.get("year", "")),
                      "category": meta.get("category", ""), "title": meta.get("title", ""),
                      "page": meta.get("page", ""), "source_url": meta.get("url", meta.get("source_url", "")),
                      "role": "jury_or_program" if "_jury" in path.parts else "portfolio"}
            if digest not in records:
                records[digest] = {"id": "hb-" + digest[:20], "sha256": digest, "sources": []}
            records[digest]["sources"].append(source)
    return list(records.values())


def image_data(record: dict) -> str:
    path = Path(record["sources"][0]["path"])
    with Image.open(path) as image:
        image = image_to_srgb(ImageOps.exif_transpose(image))
        record["dimensions"] = list(image.size)
        image.thumbnail((1152, 1152), Image.Resampling.LANCZOS)
        tech = analyze(srgb_to_linear(np.asarray(image, dtype=np.float32) / 255), path)
        record["features"] = {k: float(getattr(tech, k)) for k in ("mean_luma", "shadow_clip", "highlight_clip",
            "saturation_mean", "temperature_bias", "green_fraction", "aqua_fraction", "skin_fraction", "dynamic_spread")}
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=88)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class VisualObservations(StudyAnalysis):
    # The visual pass observes individual pictures and chooses grounded
    # principles. Lightroom operations are compiled from reviewed guidance,
    # instead of spending inference on speculative slider instructions.
    lessons: list[Principle] = Field(min_length=2, max_length=3)


def analyze_batch(settings, model: str, batch: list[dict], *, audit_dir: Path) -> tuple[dict, dict]:
    schema = VisualObservations.model_json_schema()
    definitions = schema.pop("$defs", {})
    result_schema = {"type": "object", "properties": {r["id"]: schema for r in batch},
                     "required": [r["id"] for r in batch], "additionalProperties": False, "$defs": definitions}
    images = [image_data(r) for r in batch]
    image_order = [{"image": i + 1, "id": r["id"], "role": r["sources"][0]["role"],
                    "dimensions": r["dimensions"], "measured_luma": round(r["features"]["mean_luma"], 3)}
                   for i, r in enumerate(batch)]
    prompt = f"""逐张研读附带的 {len(batch)} 张摄影学习图，每张都要独立分析，中文输出。
图片顺序与 JSON 对象键严格对应：{json.dumps(image_order, ensure_ascii=False)}。
任务是从每张成片的可见证据提炼摄影后期的指导价值。不要用作者名气代替判断，
不要复制通用的“构图好、色彩高级、层次丰富”评语。每条 strengths 必须指明具体区域、
物体、明暗或颜色关系，解释它怎样引导视线或服务情绪。caption 要具体到能区分这张图。
观察构图留白、主次、光线方向、影调密度、主辅色、远近空气、材质和叙事；选最有价值的 3–4 点。
每张选择 2–3 个最有证据支持的后期原则，lessons 仅返回原则 ID 字符串列表。
subject_separation=主体与环境的主次；atmospheric_depth=真实雾气/空气散射的远近反差，
没有 fog 就不能选 atmospheric_depth，塑料膜、玻璃、浅景深和构图纵深都不是空气透视；
negative_space=留白；color_hierarchy=主辅配色；temperature_layers=现场冷暖分层；
highlight_rolloff=高光过渡；black_depth=暗部密度；material_detail=材质纹理；
skin_integrity=自然肤色纹理；neutral_material=中性建筑/产品材质；
monochrome_structure=黑白影调结构；graphic_geometry=几何线条；environment_story=环境叙事。
原则的适用条件、Lightroom 操作与复核标准由程序提供，不需要猜测或生成大师参数。
preserve/avoid 必须尊重当前用户原片已有的光线和自然层次。不能建议所有图都提亮主体、
压暗背景、加清晰度或去雾。黑白图可借鉴影调结构，但不能要求彩色照片自动变黑白。
capture_dependencies 单列依赖拍摄的内容（位置、时机、天气、布光、造型、构图或合成），
明确这些不能靠调色凭空生成。成片无法证明拍摄设备、原片状态、真正后期步骤或作者意图，
uncertainty 要说明该图哪些只是视觉判断。如果是书页拼图/封面/评委肖像，如实分类，
仍分析可见的布局或光色价值，不把评委照标成获奖作品。不要识别人物身份或推断敏感属性。
所有分析只基于各自图片，不混用另外一张图的主体、颜色或构图。句子具体简洁。
只返回符合 schema 的 JSON 对象，每张一个键，无 Markdown、无额外说明。
scene_tags 只填实际可见元素，不重复；城市里没有雾、雪、水面就不要填 fog、snow、water。
light 的 hard_directional 表示清晰投影的硬光，soft_diffuse 表示没有硬边投影的柔光。
tone 的 high_key 表示大部分画面处于明亮调子，不能因为是晴天就标 high_key。
紫色和蓝色同属偏冷配色，不能把它们的区别写成冷暖对比。
如果几张是同一照片的不同裁切，仍分别分析留白、主体大小、画面边缘和视线关系的差异。
capture_dependencies 用完整中文句子说明哪些现场条件不能靠后期生成。
特别检查：浅景深造成的背景模糊不是雾；柔光不等于有雾。白色棉絮、云朵道具不是天气。
画面若明显在摆弄拍摄道具、灯架或展示制作过程，image_kind=behind_scenes，仍分析但不当成获奖成片。
左右都有物体不等于左右对称；蓝色与灰色不是互补色；中性灰没有可单独调整的色相。
HSL 同时影响所有相同颜色，不能声称它只改变某栋楼而不影响同色巴士。局部指导仅用主体、
天空、背景三类现有蒙版；远近深度、棉絮道具或指定物体的精细蒙版属于未自动支持的能力。
不要通过压暗远景去模拟空气透视。保留自然远近反差与散射，只有真实雾气才谈雾气保护。
capture_dependencies 描述这张作品依赖的现场关系，不把晴天、对称或某种道具变成所有照片必须遵守的规则。
输出结构、字段含义与可选枚举如下，请先读懂再填写：
{json.dumps(result_schema, ensure_ascii=False)}
/no_think"""
    payload = {"model": model, "stream": False, "format": result_schema, "think": False, "keep_alive": "30m",
               "options": {"temperature": .1, "num_ctx": 16384, "num_predict": 7200},
               "messages": [{"role": "user", "content": prompt, "images": images}]}
    start = time.monotonic()
    response = httpx.post(settings.ollama_url.rstrip("/") + "/api/chat", json=payload,
                          timeout=max(settings.ollama_timeout, 600), trust_env=False)
    response.raise_for_status()
    raw = response.json()
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / (batch[0]["id"] + ".json")
    write_json(audit_path, {"image_order": image_order, "response": _assistant_text(raw)})
    parsed = _extract_json(_assistant_text(raw))
    if set(parsed) != {r["id"] for r in batch}:
        raise ValueError("Returned image IDs do not match the batch")
    analyses = {}
    for key, value in parsed.items():
        observation = VisualObservations.model_validate(value).model_dump()
        lessons = []
        for principle in dict.fromkeys(observation["lessons"]):
            when, action, avoid, review = PRINCIPLE_GUIDANCE[principle]
            lessons.append({"principle": principle, "applies_when": when, "action": action,
                            "avoid": avoid, "review": review})
        observation["lessons"] = lessons
        analyses[key] = StudyAnalysis.model_validate(observation).model_dump()
    metrics = {"seconds": round(time.monotonic() - start, 2), "model": model,
               "eval_count": raw.get("eval_count"), "done_reason": raw.get("done_reason"),
               "analysis_mode": "visual_observations_with_compiled_lessons"}
    write_json(audit_path, {"image_order": image_order, "response": _assistant_text(raw), "metrics": metrics})
    metrics["response_audit"] = str(audit_path)
    return analyses, metrics


def write_json(path: Path, data: dict | list) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/Users/dextermosby/Downloads/ph-main/hasselblad"))
    parser.add_argument("--legacy", type=Path, default=Path("data/references/awards/hasselblad_masters"))
    parser.add_argument("--out", type=Path, default=Path("data/references/hasselblad-study"))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", action="store_true", help="Take an evenly spread pilot sample")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--model")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        settings = get_settings()
        status = probe(settings)
        capabilities = {}
        for name in status.get("models", []):
            response = httpx.post(settings.ollama_url.rstrip("/") + "/api/show", json={"model": name}, trust_env=False, timeout=15)
            if response.is_success:
                capabilities[name] = response.json().get("capabilities", [])
        print(json.dumps({**status, "capabilities": capabilities}, ensure_ascii=False), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    records = inventory(args.source, args.legacy)
    write_json(args.out / "inventory.json", records)
    print(json.dumps({"files": sum(len(r["sources"]) for r in records), "unique_bytes": len(records),
                      "corpus": dict(Counter(s["corpus"] for r in records for s in r["sources"]))}), flush=True)
    if args.inventory_only:
        return
    settings = get_settings()
    status = probe(settings)
    model = args.model or status.get("resolved_model")
    if not model:
        raise RuntimeError(f"Local vision model unavailable: {status}")
    output = args.out / "records"
    output.mkdir(exist_ok=True)
    pending = []
    for record in records:
        path = output / (record["id"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("sha256") == record["sha256"] and saved.get("schema_version") == 3 and saved.get("status") == "complete":
                try:
                    StudyAnalysis.model_validate(saved["analysis"])
                    continue
                except ValueError:
                    pass
        pending.append(record)
    pending = pending[args.offset:]
    if args.limit is not None:
        pending = ([pending[i * len(pending) // args.limit] for i in range(min(args.limit, len(pending)))]
                   if args.sample else pending[:args.limit])
    failures = []
    done = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        try:
            analyses, metrics = analyze_batch(settings, model, batch, audit_dir=args.out / "batches")
        except Exception as exc:
            print(json.dumps({"batch_failed": [r["id"] for r in batch], "error": str(exc)}, ensure_ascii=False), flush=True)
            # Retry each image independently; no missing photo can disappear in
            # a partial multi-image response or be replaced with generic prose.
            analyses, metrics = {}, {"retry": "individual", "model": model}
            for record in batch:
                try:
                    result, detail = analyze_batch(settings, model, [record], audit_dir=args.out / "batches")
                    analyses.update(result)
                    metrics[record["id"]] = detail
                except Exception as retry:
                    failures.append({"id": record["id"], "error": str(retry)})
        for record in batch:
            if record["id"] not in analyses:
                continue
            record.update(schema_version=3, status="complete", analysis=analyses[record["id"]],
                          model=model, analyzed_at=datetime.now(timezone.utc).isoformat(), metrics=metrics)
            write_json(output / (record["id"] + ".json"), record)
            done += 1
        progress = {"completed_this_run": done, "requested_this_run": len(pending), "total_unique": len(records),
                    "saved_records": len(list(output.glob("*.json"))), "last_batch": metrics, "failures": failures}
        write_json(args.out / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
