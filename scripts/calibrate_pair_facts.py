#!/usr/bin/env python3
"""Freeze and run five anonymous, single-board regional-fact checks.

This standalone experiment does not use Lightroom or update application code.
Each output directory permits one attempt per planned call, including failures.
Use a new --output-dir for a deliberately separate replication; rerunning the
same directory only runs calls without an attempt marker. No expected answers,
source identities, prior reviews, original images, or reference art are sent.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superphoto.config import Settings  # noqa: E402
from superphoto.ollama import _assistant_text, _trust_proxy  # noqa: E402
from superphoto.pair_review import build_pair_board  # noqa: E402
from superphoto.review import _preview  # noqa: E402


DEFAULT_DIR = ROOT / ".analysis/critic-calibration-2026-09-26/holdout-facts"
DEFAULT_CASES = DEFAULT_DIR.parent / "cases.json"
OPTIONS = {"temperature": 0, "seed": 146, "num_ctx": 16384, "num_predict": 2000}
RELATION = ["1", "2", "same", "uncertain"]
PRESENCE = ["both", "1", "2", "neither", "uncertain"]

# Frozen before the first inference. These questions intentionally contain no
# expected answer, chronological direction, author name, or aesthetic winner.
QUESTIONS = {
    "monument": [
        ("sky_darker", "relation", "比较上方大块蓝色天空，哪张更暗？"),
        ("shaft_brighter", "relation", "比较画面中央竖直纪念碑的碑身本身，哪张更亮？依据碑身本身，不以周围天空的明暗替代。"),
        ("reflection_brighter", "relation", "比较中央碑身下方水中的竖直倒影，两图倒影亮度是否明显不同？若有，选明显更亮的一张；若没有明显差异，选same。"),
        ("trees_near_silhouette", "presence", "检查两侧岸边成群树木：哪些图中的树群整体接近剪影，即树冠很暗、内部细节很少？"),
    ],
    "hiker": [
        ("mountain_blue_cyan_stronger", "relation", "比较左侧至中央的中景山体，哪张山体的蓝青色更浓？"),
        ("sky_brighter", "relation", "比较上方淡色天空，哪张更亮？"),
        ("face_lifted", "relation", "比较右下方人物的面部，两图之间是否存在明确的大幅面部提亮？若有，选面部明显更亮的一张；若只有轻微变化或没有明确大幅提亮，选same。"),
        ("distant_mountains_lower_contrast", "presence", "分别检查每张图内部的远近山体：哪些图中，靠近天空的远山比下方较近山体对比更低？"),
    ],
    "landscape": [
        ("sky_darker", "relation", "比较上部蓝色天空，哪张更暗？"),
        ("cloud_contrast_stronger", "relation", "比较中央云层，哪张云层内部的明暗起伏更明显？"),
        ("lit_ridge_brighter", "relation", "比较画面中部偏右被光照到的草坡山脊，哪张更亮？"),
        ("lower_grass_brighter", "relation", "比较下半部草坡，哪张整体更亮？"),
    ],
}
SCHEDULE = [
    ("01_monument_AB", "henson_monument_reflection_holdout", "monument", "AB", "holdout"),
    ("02_monument_BA", "henson_monument_reflection_holdout", "monument", "BA", "holdout"),
    ("03_hiker_AB", "hiker_palette_and_existing_marks_holdout", "hiker", "AB", "holdout"),
    ("04_hiker_BA", "hiker_palette_and_existing_marks_holdout", "hiker", "BA", "holdout"),
    ("05_landscape_AA", "kelby_sky_vs_land_development", "landscape", "AA", "development_null_control"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode_json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def save_json(path: Path, value, *, exclusive: bool = False) -> None:
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(encode_json(value))


def schema_for(group: str) -> dict:
    properties = {}
    for name, kind, _ in QUESTIONS[group]:
        properties[name] = {
            "type": "object", "additionalProperties": False,
            "required": ["choice", "image_1", "image_2"],
            "properties": {
                "choice": {"type": "string", "enum": PRESENCE if kind == "presence" else RELATION},
                "image_1": {"type": "string", "minLength": 1},
                "image_2": {"type": "string", "minLength": 1},
            },
        }
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def prompt_for(group: str) -> str:
    lines = [
        "只依据这张比较板的可见像素回答区域事实，不评价审美、自然程度或哪张更好。",
        "比较板中标号1和2分别指对应照片，灰色边框及数字条不是照片内容。",
        "逐项独立观察。每项返回choice以及image_1、image_2，分别描述图1和图2该区域实际可见的表现。",
        "差异比较题：choice只能为字符串1、2、same、uncertain。1或2表示问题指定属性更明显的图；没有明确可见差异用same；看不清或证据不足用uncertain。",
        "存在性题：choice只能为both、1、2、neither、uncertain；分别表示两图均成立、仅图1成立、仅图2成立、两图均不成立、无法确认。",
        "不要因整体更暗推断每个局部都更暗，不要把色彩浓淡当作亮度，也不要猜测拍摄或处理过程。",
        "仅返回JSON对象，键严格使用下列题目id，不添加总评分或总胜者：",
    ]
    lines.extend(f"{name}（{'存在性题' if kind == 'presence' else '差异比较题'}）：{question}"
                 for name, kind, question in QUESTIONS[group])
    return "\n".join(lines) + "\n/no_think"


def prepare(cases_path: Path, output: Path, model: str) -> dict:
    plan_path = output / "frozen-plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Output has files but no frozen plan; inspect it and use a new output directory.")
    cases_bytes = cases_path.read_bytes()
    cases = {case["id"]: case for case in json.loads(cases_bytes)["cases"]}
    preview_cache: dict[str, str] = {}
    calls = []
    for call_id, case_id, group, order, split in SCHEDULE:
        case = cases[case_id]
        sources = [Path(case[f"image_{letter.lower()}_path"]).resolve() for letter in order]
        # Existence and genuine-export provenance were checked in cases.json.
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            if str(source) not in preview_cache:
                preview_cache[str(source)] = _preview(source)
        call_dir = output / call_id
        call_dir.mkdir()
        board_b64 = build_pair_board(*(preview_cache[str(path)] for path in sources))
        board_raw = base64.b64decode(board_b64, validate=True)
        (call_dir / "board.jpg").write_bytes(board_raw)
        prompt = prompt_for(group)
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        payload = {"model": model, "stream": False, "think": False, "keep_alive": "30m",
                   "format": schema_for(group), "options": dict(OPTIONS),
                   "messages": [{"role": "user", "content": prompt, "images": [board_b64]}]}
        request_raw = encode_json(payload)
        (call_dir / "request.json").write_bytes(request_raw)
        calls.append({"id": call_id, "case_id": case_id, "group": group, "order": order,
                      "split": split, "panel_mapping": list(order),
                      "source_paths": [str(path) for path in sources],
                      "source_sha256": [sha(path.read_bytes()) for path in sources],
                      "preview_sha256": [sha(base64.b64decode(preview_cache[str(path)])) for path in sources],
                      "board_sha256": sha(board_raw), "board_bytes": len(board_raw),
                      "prompt_sha256": sha(prompt.encode()), "request_sha256": sha(request_raw),
                      "expected_response_schema": schema_for(group)})
    plan = {"version": 1, "frozen_at": now(), "model": model, "options": dict(OPTIONS),
            "script_path": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__).read_bytes()),
            "cases_path": str(cases_path), "cases_sha256": sha(cases_bytes),
            "board_helper": "superphoto.pair_review.build_pair_board",
            "board_helper_sha256": sha((ROOT / "src/superphoto/pair_review.py").read_bytes()),
            "preview_helper": "superphoto.review._preview",
            "preview_helper_sha256": sha((ROOT / "src/superphoto/review.py").read_bytes()),
            "questions": QUESTIONS, "calls": calls,
            "protocol": ["exactly five independent requests; one user message and one board per request",
                         "no expected answers, source identities, original images, references or old reviews in requests",
                         "AB/BA questions identical; choices mapped back to physical files only after response",
                         "one attempt per call including failure; no automatic retries or prompt tuning",
                         "same-input AA is forced through the model, without the production identical-image shortcut",
                         "questions report regional facts, never an overall aesthetic score",
                         "two holdout scenes are held out for this critic calibration, not previously unseen by all app development"]}
    save_json(plan_path, plan, exclusive=True)
    (output / "frozen-plan.sha256").write_text(sha(plan_path.read_bytes()) + "\n")
    return plan


def validate_answer(data: dict, schema: dict) -> dict:
    text = _assistant_text(data)
    if data.get("done_reason") == "length" or data.get("done") is not True:
        raise ValueError("Incomplete or truncated model response")
    # Do not repair malformed responses or treat failure as a neutral answer.
    answer = json.loads(text)
    if not isinstance(answer, dict) or set(answer) != set(schema["required"]):
        raise ValueError("Answer keys do not match the frozen questions")
    for name, item in answer.items():
        item_schema = schema["properties"][name]
        if not isinstance(item, dict) or set(item) != set(item_schema["required"]):
            raise ValueError(f"Malformed regional answer: {name}")
        if item["choice"] not in item_schema["properties"]["choice"]["enum"]:
            raise ValueError(f"Invalid choice: {name}")
        if not all(isinstance(item[key], str) and item[key].strip() for key in ("image_1", "image_2")):
            raise ValueError(f"Missing per-image observation: {name}")
    return answer


def verify_frozen(output: Path, plan: dict) -> None:
    if sha((output / "frozen-plan.json").read_bytes()) != (output / "frozen-plan.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("Frozen plan hash changed")
    for call in plan["calls"]:
        folder = output / call["id"]
        for filename, field in (("request.json", "request_sha256"), ("board.jpg", "board_sha256"),
                                ("prompt.txt", "prompt_sha256")):
            if sha((folder / filename).read_bytes()) != call[field]:
                raise RuntimeError(f"Frozen input changed: {call['id']}/{filename}")


def run(output: Path, plan: dict, settings: Settings, timeout: float) -> None:
    verify_frozen(output, plan)
    pending = [call for call in plan["calls"] if not (output / call["id"] / "attempt.json").exists()]
    if not pending:
        print("All five calls already attempted; no model calls made.", flush=True)
        return
    base_url = settings.ollama_url.rstrip("/")
    # Read-only preflight occurs before attempt markers: a sandbox/network block
    # here never consumes an inference attempt or silently creates five failures.
    with httpx.Client(timeout=timeout, trust_env=_trust_proxy(base_url)) as client:
        tags = client.get(base_url + "/api/tags", timeout=15)
        tags.raise_for_status()
        models = tags.json().get("models", [])
        if plan["model"] not in [item.get("name") for item in models]:
            raise RuntimeError(f"Frozen model {plan['model']} is not installed; no inference was sent")
        if not (output / "model-preflight.json").exists():
            save_json(output / "model-preflight.json", {"time": now(), "models": models}, exclusive=True)
        for call in pending:
            folder = output / call["id"]
            request_raw = (folder / "request.json").read_bytes()
            started = time.monotonic()
            record = {"id": call["id"], "started_at": now(), "request_sha256": sha(request_raw),
                      "board_sha256": call["board_sha256"], "status": "attempt_started",
                      "retry_policy": "never automatically retry, including interrupted/failed attempts"}
            save_json(folder / "attempt.json", record, exclusive=True)
            print(f"RUNNING {call['id']} ({len(plan['calls'])} total planned)", flush=True)
            try:
                response = client.post(base_url + "/api/chat", content=request_raw,
                                       headers={"Content-Type": "application/json"})
                (folder / "response.raw.json").write_bytes(response.content)
                record.update(http_status=response.status_code, response_sha256=sha(response.content),
                              response_bytes=len(response.content))
                data = response.json()
                save_json(folder / "response.json", data)
                if isinstance(data, dict):
                    record["response_metadata"] = {key: data[key] for key in (
                        "model", "created_at", "done", "done_reason", "prompt_eval_count", "eval_count",
                        "prompt_eval_duration", "eval_duration", "load_duration", "total_duration") if key in data}
                    (folder / "assistant.txt").write_text(_assistant_text(data), encoding="utf-8")
                response.raise_for_status()
                answer = validate_answer(data, call["expected_response_schema"])
                save_json(folder / "answer.json", answer)
                record["status"] = "completed_valid"
            except Exception as exc:
                record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            finally:
                record.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 3))
                save_json(folder / "result.json", record)
                print(f"FINISHED {call['id']}: {record['status']}; {record['elapsed_seconds']}s", flush=True)


def summarize(output: Path, plan: dict) -> dict:
    rows = []
    normalized = {}
    for call in plan["calls"]:
        folder = output / call["id"]
        result_file = folder / "result.json"
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {
            "status": "attempt_interrupted_or_unknown" if (folder / "attempt.json").exists() else "not_attempted"}
        row = {"id": call["id"], "order": call["order"], "split": call["split"],
               "status": result["status"], "metadata": result.get("response_metadata", {}),
               "error": result.get("error")}
        answer_file = folder / "answer.json"
        if result["status"] == "completed_valid" and answer_file.exists():
            answers = json.loads(answer_file.read_text(encoding="utf-8"))
            mapped = {key: call["panel_mapping"][int(item["choice"]) - 1]
                      if item["choice"] in ("1", "2") else item["choice"] for key, item in answers.items()}
            row.update(answers=answers, physical_choices=mapped)
            normalized[call["id"]] = mapped
        rows.append(row)
    consistency = []
    for first, second in ((plan["calls"][0], plan["calls"][1]), (plan["calls"][2], plan["calls"][3])):
        for key in first["expected_response_schema"]["required"]:
            a, b = normalized.get(first["id"], {}).get(key), normalized.get(second["id"], {}).get(key)
            state = ("unavailable" if a is None or b is None else "uncertain" if "uncertain" in (a, b)
                     else "consistent" if a == b else "inconsistent")
            consistency.append({"case_id": first["case_id"], "question": key,
                                "AB_physical_choice": a, "BA_physical_choice": b, "status": state})
    control = normalized.get(plan["calls"][4]["id"], {})
    summary = {"generated_at": now(), "frozen_plan_sha256": sha((output / "frozen-plan.json").read_bytes()),
               "calls_planned": len(plan["calls"]),
               "calls_attempted": sum((output / c["id"] / "attempt.json").exists() for c in plan["calls"]),
               "calls_valid": sum(row["status"] == "completed_valid" for row in rows),
               "calls": rows, "order_consistency": consistency,
               "AA_control": {key: {"choice": value, "status": "same_reported" if value == "same" else
                                    "uncertain" if value == "uncertain" else "false_difference"}
                              for key, value in control.items()},
               "interpretation_limits": ["Order consistency is not correctness; compare retained answers with independent visual records separately.",
                                         "Unknown, malformed, incomplete and failed answers are retained and not counted as correct.",
                                         "Only two holdout scenes and one development AA control; no overall aesthetic claim.",
                                         "No inference is retried and prompts are not revised after responses."]}
    save_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--model", help="Freeze a specific installed Ollama model; default is application setting")
    parser.add_argument("--timeout", type=float, default=600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Freeze prompts/boards/requests; make no network calls")
    mode.add_argument("--summarize-only", action="store_true", help="Read existing responses only; make no network calls")
    args = parser.parse_args()
    settings = Settings()
    output = args.output_dir.resolve()
    if args.summarize_only and not (output / "frozen-plan.json").exists():
        parser.error("--summarize-only requires an existing frozen plan")
    plan = prepare(args.cases.resolve(), output, args.model or settings.ollama_model)
    verify_frozen(output, plan)
    if not args.prepare_only and not args.summarize_only:
        run(output, plan, settings, args.timeout)
    summary = summarize(output, plan)
    print(json.dumps({"output_dir": str(output), "planned": summary["calls_planned"],
                      "attempted": summary["calls_attempted"], "valid": summary["calls_valid"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
