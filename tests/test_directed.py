from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from superphoto import directed
from superphoto import refinement as ref
from superphoto import tone
from superphoto.config import Settings
from superphoto.db import (
    connect,
    get_job,
    get_refinement,
    get_variants,
    init_db,
    insert_job,
    insert_refinement,
    replace_variants,
    update_job,
)
from superphoto.vision import VisionUnavailable

SELECTION = {"ready": True, "provider": "xai", "resolved_model": "grok-test",
             "candidates": [{"provider": "xai", "model": "grok-test", "protocol": "responses"}]}


@pytest.fixture(autouse=True)
def _no_vision_waits(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(directed.time, "sleep", waits.append)
    return waits


def _plan(exposure: float, *, title: str, grading: bool = False) -> dict:
    plan = {
        "title": title, "intent": f"{title}：压天空、提主体",
        "develop": {"light": {"exposure": exposure, "highlights": -35, "shadows": 20},
                    "color": {"vibrance": 8}},
        "masks": [
            {"name": "天空", "purpose": "压暗天空增加密度",
             "shape": {"kind": "linear", "start": {"x": 0.5, "y": 0.0}, "end": {"x": 0.5, "y": 0.45}},
             "adjust": {"exposure": -0.25, "highlights": -30}},
            {"name": "主体", "purpose": "提亮主体",
             "shape": {"kind": "radial", "center": {"x": 0.38, "y": 0.58}, "rx": 0.18, "ry": 0.25},
             "adjust": {"exposure": 0.3, "shadows": 15}},
        ],
    }
    if grading:
        plan["develop"]["grading"] = {"shadows": {"hue": 210, "sat": 12}, "highlights": {"hue": 40, "sat": 10},
                                      "balance": 10}
    return plan


PLANNING = {
    "reading": {"scene": "河边黄昏", "genre": "landscape", "subject": "小船", "subject_box": [0.25, 0.4, 0.5, 0.8],
                "subject_confidence": 0.8, "light": "逆光", "problems": [{"issue": "主体偏暗", "region": "中下",
                                                                         "severity": 0.7}],
                "preserve": ["黄昏暖色"]},
    "plans": {"A": _plan(0.2, title="自然"), "B": _plan(0.35, title="浓郁"),
              "C": _plan(0.2, title="电影", grading=True)},
}


def _jpeg(path: Path, scale: float = 1.0, tint: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    yy, xx = np.mgrid[0:240, 0:360].astype(np.float32)
    base = np.stack([60 + 80 * xx / 360, 70 + 60 * yy / 240, 90 + 40 * (1 - yy / 240)], axis=-1)
    pixels = np.clip(base * scale * np.array(tint, dtype=np.float32), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, quality=92)
    return path


def _fake_render(calls: list):
    lock = threading.Lock()

    def render(binary, source, document, destination, *, edge=0, timeout=300.0):
        assert source.is_file() and source.name.startswith("source")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "develop.json").write_text(json.dumps(document), encoding="utf-8")
        exposure = document.get("light", {}).get("exposure", 0.0)
        tint = (1.08, 1.0, 0.9) if document.get("grading") or document.get("profile") else (1.0, 1.0, 1.0)
        output = _jpeg(destination / "render.jpg", scale=2 ** exposure, tint=tint)
        with lock:
            calls.append((destination, document))
        return output

    return render


def _gate(failing: set[str] = frozenset()):
    def gate(baseline, render, slot, source_kind="jpeg"):
        hard = ["highlight_clipping: 高光裁切"] if slot in failing else []
        return {"passed": not hard, "hard": hard, "advisory": ["too_bright: 比基线亮"], "luma_ratio": 1.2,
                "distance": 6.0, "report": {"score": 70.0 if hard else 94.0, "issues": []}}

    return gate


ELEMENTS = {"elements": ["water", "boat", "sky", "sunset"], "light_kind": "golden", "genre": "landscape"}
PLAN_ASK = re.compile(r"Design (three|\d+) finished versions")
ELEMENTS_ASK = "name what is in it"
DAMAGE_ASK = "Check each version for DAMAGE only"


class FakeVision:
    """Thread-safe scripted vision model keyed by prompt type and slot."""

    def __init__(self, reviews: dict[str, list[dict]], planning: dict | str = PLANNING, *,
                 elements: dict | None = None, damage: dict | None = None):
        self.reviews = {slot: list(items) for slot, items in reviews.items()}
        self.planning = planning
        self.elements = ELEMENTS if elements is None else elements
        self.damage = damage or {}
        self.prompts: list[str] = []
        self.images: list[int] = []
        self.lock = threading.Lock()

    def __call__(self, settings, model, prompt, images, *, selection=None, audit=None, response_schema=None):
        assert model == "grok-test" and selection is SELECTION
        assert isinstance(images, list) and all(isinstance(i, str) for i in images)
        assert settings.vision_timeout >= settings.directed_vision_timeout
        if audit is not None:
            audit.update(provider="xai", model="grok-test")
        with self.lock:
            self.prompts.append(prompt)
            self.images.append(len(images))
            if ELEMENTS_ASK in prompt:
                assert len(images) == 1
                return json.dumps(self.elements)
            if DAMAGE_ASK in prompt:
                return json.dumps({"versions": self.damage}, ensure_ascii=False)
            if "The first plans have been rendered" in prompt:
                return json.dumps({"plans": {}})
            assert len(images) == 2
            if PLAN_ASK.search(prompt):
                planning = self.planning
                return planning if isinstance(planning, str) else json.dumps(planning, ensure_ascii=False)
            slot = re.search(r"reviewing version ([A-I])", prompt).group(1)
            return json.dumps(self.reviews[slot].pop(0), ensure_ascii=False)


def _review(verdict="better", *, revise_plan=None, checks="achieved", damage=None):
    return {"verdict": verdict, "confidence": 0.8,
            "checks": [{"item": "天空", "status": checks, "note": ""}, {"item": "主体", "status": checks, "note": ""}],
            "damage": damage or [], "notes": ["测试"], "revise": revise_plan is not None, "plan": revise_plan}


@pytest.fixture
def job(tmp_path, monkeypatch):
    # The classic trio with per-version review; the chosen-directions flow is set per test.
    settings = Settings(data_dir=tmp_path / "data", lightroom_enabled=True, directed_versions=3,
                        directed_max_revisions=1)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job", "photo.jpg")
    update_job(conn, "job", original_file="original.jpg")
    root = settings.jobs_dir / "job"
    original = _jpeg(root / "original.jpg")
    baseline = _jpeg(root / "lightroom-exports" / "baseline" / "baseline.jpg")
    technical = SimpleNamespace(source_kind="jpeg", model_dump=lambda: {
        "source_kind": "jpeg", "width": 360, "height": 240, "mean_luma": 0.3, "exif": {"EXIF ISOSpeedRatings": "200"}})
    calls: list = []
    monkeypatch.setattr("superphoto.lightroom.connect_lightroom",
                        lambda *_: (SimpleNamespace(binary=Path("lightcraft-cli")), {}))
    monkeypatch.setattr("superphoto.lightcraft.render_document", _fake_render(calls))
    monkeypatch.setattr(directed, "safety_gate", _gate())
    yield SimpleNamespace(settings=settings, conn=conn, root=root, original=original, baseline=baseline,
                          technical=technical, renders=calls)
    conn.close()


def _run(job):
    return directed.run_directed_job(job.settings, job.conn, "job", job.original, job.root, technical=job.technical,
                                     baseline_export=job.baseline, selection=SELECTION)


# --------------------------------------------------------------------------
# Plan sanitizing and compilation


def test_sanitize_plan_clamps_drops_unknown_and_wraps_hue():
    raw = {
        "title": "测试", "intent": "x",
        "develop": {
            "light": {"exposure": 3.0, "contrast": "12", "highlights": float("nan"), "bogus": 4, "shadows": True},
            "wb": {"temp": 4000, "tint": -3},
            "mixer": {"blue": {"hue": -50, "sat": -10}, "teal": {"hue": 3}},
            "grading": {"shadows": {"hue": -30, "sat": 50}, "blending": 140, "spin": 3},
            "lens": {"distortion": 3},
        },
        "masks": [{"name": "m", "shape": {"kind": "radial", "center": [0.5, 0.5], "rx": 5, "ry": 0.2},
                   "adjust": {"exposure": 2.0, "glow": 3}}]
                 + [{"name": "dup", "shape": {"kind": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 0, "y": 0.5}},
                     "adjust": {"exposure": -0.1}} for _ in range(8)]
                 + [{"name": "flat", "shape": {"kind": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 0, "y": 0.5}},
                     "adjust": {}}],
    }
    plan, notes = directed.sanitize_plan(raw, slot="A")
    light = plan["develop"]["light"]
    assert light == {"exposure": 1.5, "contrast": 12.0}
    assert plan["develop"]["wb"] == {"temp": 1500.0, "tint": -3.0}
    assert plan["develop"]["mixer"] == {"blue": {"hue": -30.0, "sat": -10.0}}
    assert plan["develop"]["grading"]["shadows"] == {"hue": 330.0, "sat": 30.0}
    assert plan["develop"]["grading"]["blending"] == 100.0 and "spin" not in plan["develop"]["grading"]
    assert "lens" not in plan["develop"]
    assert len(plan["masks"]) == directed.MAX_MASKS
    first = plan["masks"][0]
    assert first["shape"]["rx"] == 1.2 and first["shape"]["feather"] == 70.0 and first["adjust"] == {"exposure": 1.2}
    assert len({mask["name"] for mask in plan["masks"]}) == directed.MAX_MASKS
    joined = "\n".join(notes)
    for fragment in ("A.light.exposure: 3", "A.light.bogus", "A.mixer.teal", "A.lens", "A.masks[1].adjust.glow",
                     "最多执行"):
        assert fragment in joined


def test_planned_vignette_is_held_but_a_users_refinement_is_not():
    raw = {"title": "t", "develop": {"vignette": {"amount": -24, "midpoint": 40}}}
    for slot in "ABC":
        plan, notes = directed.sanitize_plan(raw, slot=slot)
        assert plan["develop"]["vignette"] == {"amount": directed.VIGNETTE_FLOOR[slot], "midpoint": 40.0}
        assert any(f"{slot}.vignette.amount" in note for note in notes)
    assert directed.sanitize_plan(raw, slot="R")[0]["develop"]["vignette"]["amount"] == -24.0
    assert directed.sanitize_plan({"title": "t", "develop": {"vignette": {"amount": -6}}}, slot="A")[1] == []


def test_sanitize_plan_rejects_non_object_and_degenerate_shapes():
    with pytest.raises(ValueError):
        directed.sanitize_plan(["not", "a", "plan"], slot="B")
    plan, notes = directed.sanitize_plan({"masks": [
        {"shape": {"kind": "linear", "start": [0.5, 0.5], "end": [0.5, 0.505]}, "adjust": {"exposure": 0.2}},
        {"shape": {"kind": "brush"}, "adjust": {"exposure": 0.2}},
        {"shape": {"kind": "radial", "center": [0.5, 0.5], "rx": 0, "ry": 0.2}, "adjust": {"exposure": 0.2}},
    ]}, slot="B")
    assert plan["masks"] == [] and plan["develop"] == {}
    assert plan["title"] == directed.DEFAULT_TITLES["B"]
    assert len(notes) == 3


def test_measurements_carry_the_iso_as_read_exif_stores_it():
    # read_exif drops exifread's "EXIF " prefix; the planner used to look for the prefixed key and never saw an ISO.
    base = {"source_kind": "raw", "width": 6000, "height": 4000}
    assert directed._measurements({**base, "exif": {"ISOSpeedRatings": "3200"}})["iso"] == 3200
    assert directed._measurements({**base, "exif": {"ISOSpeedRatings": "[800, 0]"}})["iso"] == 800
    assert directed._measurements({**base, "exif": {"PhotographicSensitivity": "100/1"}})["iso"] == 100
    assert "iso" not in directed._measurements({**base, "exif": {"ISOSpeedRatings": "0"}})
    assert "iso" not in directed._measurements(base)
    technical = SimpleNamespace(model_dump=lambda: {**base, "exif": {"ISOSpeedRatings": "6400"}})
    prompt = directed.plan_prompt(technical, (6000, 4000))
    assert '"iso": 6400' in prompt


def test_compile_document_shifts_white_balance_from_the_files_own_reference():
    plan, _ = directed.sanitize_plan(_plan(0.2, title="t") | {"develop": {"light": {"exposure": 0.2},
                                                                         "wb": {"temp": 800, "tint": 4}}}, slot="A")
    jpeg, jpeg_notes = directed.compile_document(plan, "jpeg")
    assert jpeg["wb"] == {"mode": "custom", "temp": 7300.0, "tint": 4.0} and not jpeg_notes
    assert jpeg["light"] == {"exposure": 0.2}
    mask = jpeg["masks"][0]
    assert mask["components"][0]["shape"]["kind"] == "linear" and mask["adjust"]["amount"] == 100.0
    assert mask["adjust"]["exposure"] == -0.25 and mask["visible"] and mask["on_photo"]
    assert [m["id"] for m in jpeg["masks"]] == [1, 2]
    # A RAW keeps the offset; the renderer adds it to the file's as-shot reading (lightcraft.resolve_wb).
    raw, raw_notes = directed.compile_document(plan, "raw")
    assert raw["wb"] == {"mode": "asShotOffset", "temp": 800.0, "tint": 4.0} and not raw_notes
    assert raw["masks"] == jpeg["masks"]
    lines = directed.describe_plan(plan)
    assert lines[0].startswith("全局：曝光 +0.20 EV") and any("线性渐变" in line for line in lines)


def test_safety_gate_keeps_brightness_advisory_and_damage_hard(monkeypatch, tmp_path):
    issue = lambda code, hard=True, advisory=False: SimpleNamespace(code=code, note=code, hard=hard, advisory=advisory)
    ratio = {"value": 0.6}

    def assess(baseline, render, **kwargs):
        assert kwargs["slot"] == "A" and kwargs["natural_color"] and not kwargs["creative_palette"]
        return SimpleNamespace(
            issues=[issue("too_dark"), issue("excessive_distance"), issue("highlight_clipping"),
                    issue("soft_hint", hard=False), issue("neutral_color_cast", advisory=True)],
            metrics=SimpleNamespace(mean_luma=ratio["value"] * 0.4, perceptual_distance=9.0),
            baseline=SimpleNamespace(mean_luma=0.4), to_dict=lambda: {"score": 80})

    monkeypatch.setattr("superphoto.lightroom_qa.assess_lightroom_render", assess)
    gate = directed.safety_gate(tmp_path / "b.jpg", tmp_path / "r.jpg", "A")
    assert [h.split(":")[0] for h in gate["hard"]] == ["highlight_clipping"]
    assert {a.split(":")[0] for a in gate["advisory"]} == {"too_dark", "excessive_distance", "soft_hint",
                                                           "neutral_color_cast"}
    assert not gate["passed"] and gate["luma_ratio"] == 0.6
    monkeypatch.setattr("superphoto.lightroom_qa.assess_lightroom_render",
                        lambda *a, **k: SimpleNamespace(issues=[], metrics=SimpleNamespace(mean_luma=0.04,
                                                        perceptual_distance=30.0),
                                                        baseline=SimpleNamespace(mean_luma=0.4), to_dict=dict))
    dark = directed.safety_gate(tmp_path / "b.jpg", tmp_path / "r.jpg", "C")
    assert not dark["passed"] and dark["hard"][0].startswith("extreme_brightness")
    # A RAW's baseline is its dark, flat negative: x6 its linear mean is a developed RAW, not a catastrophe.
    monkeypatch.setattr("superphoto.lightroom_qa.assess_lightroom_render",
                        lambda *a, **k: SimpleNamespace(issues=[], metrics=SimpleNamespace(mean_luma=0.6,
                                                        perceptual_distance=30.0),
                                                        baseline=SimpleNamespace(mean_luma=0.1), to_dict=dict))
    assert not directed.safety_gate(tmp_path / "b.jpg", tmp_path / "r.jpg", "B")["passed"]
    assert directed.safety_gate(tmp_path / "b.jpg", tmp_path / "r.jpg", "B", "raw")["passed"]


def test_a_raw_negatives_near_neutrals_may_move_further_before_the_cast_gate(monkeypatch, tmp_path):
    from superphoto.lightroom_qa import NEUTRAL_INTRODUCED, NEUTRAL_SHIFT

    issue = SimpleNamespace(code="neutral_color_cast", note="cast", hard=True, advisory=False)
    times = {"value": 2.0}

    def assess(baseline, render, **kwargs):
        cast = {"fraction": 0.3, "median_shift": NEUTRAL_SHIFT * times["value"],
                "introduced_chroma": NEUTRAL_INTRODUCED * times["value"]}
        return SimpleNamespace(issues=[issue], neutral_assessment=cast,
                               metrics=SimpleNamespace(mean_luma=0.4, perceptual_distance=9.0),
                               baseline=SimpleNamespace(mean_luma=0.4), to_dict=dict)

    monkeypatch.setattr("superphoto.lightroom_qa.assess_lightroom_render", assess)
    gate = lambda kind: directed.safety_gate(tmp_path / "b.jpg", tmp_path / "r.jpg", "A", kind)
    # His own finished edits reach 1.0-2.24 times the gate against the negative: development, not a cast.
    assert not gate("jpeg")["passed"]
    assert gate("raw")["passed"] and gate("raw")["advisory"][0].startswith("neutral_color_cast")
    assert "2.0 倍" in gate("raw")["advisory"][0] and "显影" in gate("raw")["advisory"][0]
    times["value"] = 3.0
    assert not gate("raw")["passed"] and gate("raw")["hard"][0].startswith("neutral_color_cast")


def test_the_neutral_guard_measures_against_a_multiple_of_the_gate(tmp_path):
    grey = np.full((24, 24, 3), 0.75, dtype=np.float32)
    Image.fromarray((grey * 255).astype(np.uint8)).save(tmp_path / "base.png")
    warm = grey.copy()
    warm[..., 0] += 0.06
    warm[..., 2] -= 0.06
    plain = directed.neutral_guard(tmp_path / "base.png")(warm)
    assert plain > 1.0
    assert directed.neutral_guard(tmp_path / "base.png", directed.RAW_NEUTRAL_GATE)(warm) == pytest.approx(
        plain / directed.RAW_NEUTRAL_GATE)


def test_render_resolves_a_raw_white_balance_offset_from_its_as_shot_reading(monkeypatch, tmp_path):
    from superphoto import lightcraft as lc
    source = tmp_path / "x.dng"
    source.write_bytes(b"raw")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "develop.wb" in command:
            out = '{"command":"develop.wb","ms":0.3,"ok":true,"result":{"temp":5551.0,"tint":12.0}}\n'
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        Path(command[command.index("-o") + 1]).write_bytes(b"jpg")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lc.subprocess, "run", run)
    document = {"light": {"exposure": 1.0}, "wb": {"mode": lc.RELATIVE_WB_MODE, "temp": 800.0, "tint": -2.0}}
    lc.render_document(tmp_path / "cli.exe", source, document, tmp_path / "a")
    lc.render_document(tmp_path / "cli.exe", source, document, tmp_path / "b")
    written = json.loads((tmp_path / "a" / "develop.json").read_text(encoding="utf-8"))
    assert written["wb"] == {"mode": "custom", "temp": 6351.0, "tint": 10.0}
    assert sum("develop.wb" in c for c in calls) == 1                # read once per file
    # No reading: the offset is dropped rather than guessed.
    other = tmp_path / "y.dng"
    other.write_bytes(b"raw")
    monkeypatch.setattr(lc, "as_shot_wb", lambda *a, **k: None)
    assert "wb" not in lc.resolve_wb(tmp_path / "cli.exe", other, document)
    assert lc.resolve_wb(tmp_path / "cli.exe", other, {"wb": {"mode": "custom", "temp": 5000}})["wb"]["temp"] == 5000


def test_mask_overlay_draws_shapes_and_boxes_and_skips_what_it_cannot_draw(tmp_path):
    image = Image.open(_jpeg(tmp_path / "x.jpg")).convert("RGB")
    plan = _plan(0.2, title="t")
    plan["masks"][1]["target"] = {"box": [0.3, 0.5, 0.45, 0.7], "median": 0.4}
    plan["masks"].append({"name": "亮度", "shape": {"kind": "luminance", "lo": 0.5, "hi": 1.0}, "adjust": {}})
    plan["masks"].append({"name": "坏", "shape": {"kind": "radial", "center": "x", "rx": None}, "adjust": {},
                          "target": {"box": "nope"}})
    drawn = directed.mask_overlay(image, plan)
    assert drawn.size == image.size and drawn.tobytes() != image.tobytes()
    pixels = np.asarray(drawn).astype(int) - np.asarray(image).astype(int)
    box_edge = pixels[int(0.5 * 240), int(0.3 * 360) + 1:int(0.45 * 360) - 1]
    assert np.abs(box_edge).sum() > 0                                 # The target box is outlined.
    assert directed.mask_overlay(image, {"masks": []}).tobytes() == image.tobytes()


def test_luminance_limits_follow_the_engines_measured_scale_at_the_top():
    # Measured on LightCraft 0.21.1 (bench-2026-11-18/tools/lumcal.py): engine .80 = display .76, .85 = .915, and
    # display white is about .875. Sand at .90 must stay inside a limit written as lo .86 (p17-r1 DSC_1441 B2).
    assert directed._engine_luma(0.76, 0.0) == pytest.approx(0.80)
    assert directed._engine_luma(0.915, 0.0) == pytest.approx(0.85)
    assert directed._engine_luma(1.0, 0.0) == pytest.approx(0.875)
    shape, inverted = directed._engine_shape({"kind": "luminance", "lo": 0.86, "hi": 1.0, "feather": 0.03}, 0.0, None)
    assert not inverted and shape["hi"] == 1.0
    assert shape["lo"] == pytest.approx(0.8345, abs=0.002)
    assert shape["lo"] < directed._engine_luma(0.90, 0.0)
    assert 0.0 < shape["lo_feather"] < 0.02
    # A camel's hi .80 with feather .04 stays below sand at .86 (it used to reach display .91).
    camel, _ = directed._engine_shape({"kind": "luminance", "lo": 0.0, "hi": 0.80, "feather": 0.04}, 0.0, None)
    assert camel["hi"] + camel["hi_feather"] <= directed._engine_luma(0.845, 0.0) < directed._engine_luma(0.86, 0.0)
    values = [directed._engine_luma(step / 100, 0.0) for step in range(101)]
    assert values == sorted(values)
    assert directed._engine_luma(0.9, -1.0) < 0.8          # the engine reads the range after global exposure


def test_grid_overlay_and_prompts_are_self_describing(tmp_path):
    image = Image.open(_jpeg(tmp_path / "x.jpg")).convert("RGB")
    grid = directed.grid_overlay(image)
    assert grid.size == image.size and grid.tobytes() != image.tobytes()
    technical = SimpleNamespace(model_dump=lambda: {"source_kind": "raw", "width": 6000, "height": 4000})
    prompt = directed.plan_prompt(technical, (360, 240))
    assert "RAW source" in prompt and "aspect 1.500" in prompt and "rx is a fraction of image WIDTH" in prompt
    # The RAW note comes before the reading and the plan, once, and the local-work examples do not darken its sky.
    assert prompt.index("RAW DEVELOPMENT") < prompt.index("RAW source") < prompt.index("Work like a professional")
    assert prompt.count("RAW source") == 1 and "densify the sky" not in prompt
    jpeg = directed.plan_prompt(SimpleNamespace(model_dump=lambda: {"source_kind": "jpeg"}), (360, 240))
    assert "RAW DEVELOPMENT" not in jpeg and "densify the sky" in jpeg and "Already-developed JPEG" in jpeg
    review = directed.review_prompt("B", {"title": "t", "intent": "i", "develop": {}, "masks": []}, {}, {
        "hard": ["highlight_clipping: x"], "advisory": []}, "jpeg")
    assert "BLOCKS delivery" in review and "Already-developed JPEG" in review
    assert "on this photo" not in prompt and "on this photo" not in review   # no measured looks, no numbers


def test_plans_shown_to_the_model_give_chroma_on_the_x100_scale_the_prompt_asks_for():
    # p27-r1 DSC_1963 B2: the plan held 0.05 and 0.36 while the prompt and the measurements speak x100.
    raw = dict(_plan(0.0, title="自然"), tone={"black": 0.03, "shadows": 0.12, "mid": 0.45, "highlights": 0.82,
                                              "white": 0.95, "chroma": 22})
    raw["masks"][1]["target"] = {"box": [0.3, 0.5, 0.45, 0.65], "hue": 35, "chroma": 5}
    raw["masks"][0]["target"] = {"box": [0.1, 0.05, 0.9, 0.2], "hue": 217, "chroma": 0.36}
    plan, _ = directed.sanitize_plan(raw, slot="A")
    assert plan["masks"][1]["target"]["chroma"] == 0.05 and plan["tone"]["chroma"] == 0.22
    shown = directed._shown(plan)
    assert shown["tone"]["chroma"] == 22.0 and shown["masks"][1]["target"]["chroma"] == 5.0
    assert shown["masks"][0]["target"]["chroma"] == 36.0 and plan["masks"][0]["target"]["chroma"] == 0.36
    review = directed.review_prompt("A", plan, {}, {}, "jpeg")
    executed = review.split("the master curve): ")[1].split("\nMeasured:")[0]
    assert '"chroma": 5.0' in executed and '"chroma": 36.0' in executed and '"chroma": 0.05' not in executed
    # Read back, as a revision copies it, the plan is the same; a trace of colour does not come back as 0..1.
    assert directed.sanitize_plan(shown, slot="A")[0] == plan
    faint = directed._shown({"masks": [{"target": {"chroma": 0.003}}]})
    assert directed._colourfulness(faint["masks"][0]["target"]["chroma"]) == 0.007


def test_looks_are_measured_on_the_photo_and_given_to_planner_and_reviewer(job, monkeypatch):
    looks = directed.measure_looks(Path("lightcraft-cli"), job.original, job.root / "directed" / "looks", "jpeg")
    profiles = {key: entry for key, entry in looks.items() if key.startswith(directed.PROFILE_KEY)}
    assert set(looks) - set(profiles) == set(directed.LOOKS)
    assert {key.removeprefix(directed.PROFILE_KEY) for key in profiles} == set(directed.PROFILES)
    assert all(entry["chroma"] > 0 and entry["distance"] > 0 for entry in looks.values())
    probes = {destination.name: document for destination, document in job.renders}
    assert probes["unedited"] == {} and probes["desert_gold"]["grading"]   # strength 1.0 of each look
    assert probes["profile-lc.film.cool-fade"] == {"profile": {"id": "lc.film.cool-fade", "amount": 50.0}}
    # The fake renderer warms every graded document and every profile alike: one ratio, away from 1.
    assert len({entry["chroma"] for entry in looks.values()}) == 1 and looks["desert_gold"]["chroma"] != 1.0
    # A profile's cast is measured too (the fake one is warm: yellow, b > 0); a look's is not.
    assert all(entry["b"] > 0.5 and "a" in entry for entry in profiles.values()) and "b" not in looks["desert_gold"]
    assert (job.root / "directed" / "looks" / "source.jpg").is_file()   # rendered on a copy

    block = directed._looks_block({"desert_gold": {"chroma": 1.31, "distance": 3.8},
                                   "matte_landscape": {"chroma": 0.79, "distance": 2.3}}, 12.2)
    line = next(line for line in block.splitlines() if line.startswith("- desert_gold"))
    assert ("[on this photo at strength 1.0: frame colourfulness 16.0 = x1.31 image 1; at strength s 12.2 + s x 3.8;"
            " changes the picture by 3.8]") in line
    assert "12.2 - s x 2.6" in next(line for line in block.splitlines() if line.startswith("- matte_landscape"))
    assert "was MEASURED" in block and "closer than 5 read as nearly identical" in block
    assert "on this photo" not in directed._looks_block()
    # A grey frame gives no colourfulness ratio; the distance is still given.
    grey = directed._looks_block({"cine_teal": {"distance": 1.7}}, 0.3)
    assert "[on this photo at strength 1.0: changes the picture by 1.7]" in grey

    vision = FakeVision({slot: [_review("better")] for slot in "ABC"})
    monkeypatch.setattr(directed, "chat_vision", vision)
    _run(job)
    plan_prompt = next(p for p in vision.prompts if "Design three finished versions" in p)
    reviews = [p for p in vision.prompts if "reviewing version" in p]
    assert "on this photo at strength 1.0" in plan_prompt and all("on this photo at strength 1.0" in p for p in reviews)
    saved = json.loads((job.root / "directed" / "plan.json").read_text(encoding="utf-8"))
    assert set(saved["looks"]) == set(looks)
    assert "PROFILES:" in plan_prompt and "[on this photo at amount 50:" in plan_prompt


def test_a_profile_sits_under_any_version_but_the_natural_a():
    raw = {"title": "t", "intent": "x", "profile": {"id": "lc.cine.desert-heat", "amount": 0.45}}
    plan, notes = directed.sanitize_plan(raw, slot="C")
    assert plan["profile"] == {"id": "lc.cine.desert-heat", "amount": 45.0} and directed.is_executable(plan)
    document, _ = directed.compile_document(plan, "raw")
    assert document == {"profile": {"id": "lc.cine.desert-heat", "amount": 45.0}}   # one key in LightCraft's document
    assert "底片风格：电影 · 沙漠热浪（45%）" in directed.describe_plan(plan)
    # Amounts are held to 20..100 (50 when none is given); a bare id is accepted.
    assert directed.sanitize_plan({**raw, "profile": {"id": "lc.vivid", "amount": 180}}, slot="B")[0]["profile"] \
        == {"id": "lc.vivid", "amount": 100.0}
    assert directed.sanitize_plan({**raw, "profile": "lc.vivid"}, slot="B")[0]["profile"] == {"id": "lc.vivid", "amount": 50.0}
    # A is the natural version; an unknown profile, the default colour profile or null give none.
    natural, notes = directed.sanitize_plan(raw, slot="A")
    assert "profile" not in natural and any("A 是自然版本" in note for note in notes)
    unknown, notes = directed.sanitize_plan({**raw, "profile": {"id": "lut:mine"}}, slot="B")
    assert "profile" not in unknown and any("lut:mine" in note for note in notes)
    for value in (None, "lc.color", {"id": "none"}):
        assert "profile" not in directed.sanitize_plan({**raw, "profile": value}, slot="B")[0]
    plain, _ = directed.sanitize_plan({"title": "t", "develop": {"color": {"vibrance": 5}}}, slot="B")
    assert "profile" not in directed.compile_document(plain, "jpeg")[0]
    # The refinement of a delivered version ("R") may take one.
    assert directed.sanitize_plan(raw, slot="R")[0]["profile"]["id"] == "lc.cine.desert-heat"

    block = directed._looks_block({"profile:lc.cine.desert-heat": {"chroma": 1.02, "distance": 4.2, "a": 1.0, "b": 3.9},
                                   "profile:lc.film.cool-fade": {"distance": 2.3, "a": -0.45, "b": -0.1}}, 12.0)
    line = next(line for line in block.splitlines() if line.startswith("- lc.cine.desert-heat"))
    assert ("[on this photo at amount 50: frame colourfulness 12.2 = x1.02 image 1; cast yellow 3.9, magenta 1.0"
            " (mean Lab); changes the picture by 4.2]") in line
    cool = next(line for line in block.splitlines() if line.startswith("- lc.film.cool-fade"))
    assert "[on this photo at amount 50: cast green 0.5 (mean Lab); changes the picture by 2.3]" in cool
    assert "PROFILES:" in block and "Never on A" in block and "at most 10 either way" in block
    assert "45 up on this RAW" in directed._looks_block(None, None, 5.0, "raw")
    assert all(f"- {profile_id} (" in block for profile_id in directed.PROFILES)
    assert '"profile": {"id": "...", "amount": 50} or null' in directed._PLAN_SHAPE


def test_a_look_that_keeps_the_frame_off_its_chroma_gets_a_strength():
    look = {"id": "desert_gold", "strength": 0.8}
    looks = {"desert_gold": {"chroma": 1.31, "distance": 3.8}}
    # p18-r1 DSC_1441 C0: written 12.0 under desert_gold 0.8, saturation held at -10, frame 15.9.
    chroma = {"target": 0.12, "written": 0.12, "after": 0.159, "saturation": -10.0, "saturation_start": 0.0,
              "look_room": 10.0}
    text = directed._look_fix(chroma, look, looks, 12.2)
    assert "15.2 at strength 0.8" in text and "does not give 12.0 at any strength" in text and "below image 1" in text
    assert directed._look_fix({**chroma, "written": 0.13}, look, looks, 12.2).endswith(
        "strength 0.21 gives the written chroma with this look, or write chroma 15.2")
    # Not held at the room, within the look's reach, or no measurement: nothing to add.
    assert directed._look_fix({**chroma, "saturation": -6.0}, look, looks, 12.2) == ""
    assert directed._look_fix({**chroma, "written": 0.145}, look, looks, 12.2) == ""
    assert directed._look_fix(chroma, look, {}, 12.2) == ""
    assert directed._look_fix({**chroma, "look_room": None}, look, looks, 12.2) == ""
    stats = {"anchors": {name: 0.5 for name in directed.tone.ORDER}, "colourfulness": 0.122,
             "luma": [[50]], "chroma": [[12]], "hue": [[30]]}
    solve = {"chroma": chroma, "regions": [], "notes": []}
    assert "FIX: this look does not give" in directed.measured_block(solve, stats, stats, look=look, looks=looks)
    assert "FIX: this look" not in directed.measured_block(solve, stats, stats)


# --------------------------------------------------------------------------
# End to end with a scripted vision model and a fake renderer


def test_a_version_rendered_nearly_identical_to_a_is_planned_once_more_before_its_review(job, monkeypatch):
    # p28-r1: 8 deductions (about 41 points) named B or C under 5 from A; revisions did not move them apart.
    close = dict(PLANNING, plans=dict(PLANNING["plans"], B=_plan(0.22, title="浓郁")))
    asked: list[tuple[str, int]] = []

    def vision(answer_b):
        def answer(settings, model, prompt, images, **_):
            asked.append((prompt, len(images)))
            if "The first plans have been rendered" in prompt:
                if answer_b is None:
                    raise directed.VisionUnavailable("offline")
                return json.dumps({"plans": {"B": answer_b}}, ensure_ascii=False)
            if "Design three finished versions" in prompt:
                return json.dumps(close, ensure_ascii=False)
            return json.dumps(_review("better"), ensure_ascii=False)
        return answer

    monkeypatch.setattr(directed, "chat_vision", vision(_plan(0.9, title="浓郁（拉开）")))
    _run(job)
    [(prompt, count)] = [(p, n) for p, n in asked if "The first plans have been rendered" in p]
    assert count == 4 and "image 4 is version B's" in prompt and "- plan B renders" in prompt
    assert "Luma per cell, B minus A" in prompt and '"浓郁"' in prompt.split("The plans were:")[1]
    apart = json.loads((job.root / "directed" / "plan.json").read_text(encoding="utf-8"))["apart"]
    assert set(apart["close"]) == {"B"} and apart["replaced"] == ["B"] and apart["after"]["B"] > apart["close"]["B"]
    review_b = next(p for p, _ in asked if "reviewing version B" in p)
    assert "浓郁（拉开）" in review_b
    folder = job.root / "directed" / "B"
    assert (folder / "round-0-close").is_dir() and (folder / "round-0" / "review.json").is_file()
    first_b = json.loads((folder / "round-0" / "review.json").read_text(encoding="utf-8"))
    assert first_b["plan"]["title"] == "浓郁（拉开）" and first_b["document"]["light"]["exposure"] == 0.9

    # An answer that renders no farther from A is refused and the first render reviewed as it was.
    for name in ("directed", "lightroom-audits"):
        shutil.rmtree(job.root / name, ignore_errors=True)
    asked.clear()
    monkeypatch.setattr(directed, "chat_vision", vision(_plan(0.2, title="浓郁（照旧）")))
    _run(job)
    apart = json.loads((job.root / "directed" / "plan.json").read_text(encoding="utf-8"))["apart"]
    assert apart["replaced"] == [] and apart["refused"]["B"].startswith("render")
    assert "浓郁（照旧）" not in next(p for p, _ in asked if "reviewing version B" in p)
    assert (folder / "round-0-apart").is_dir() and (folder / "round-0" / "review.json").is_file()
    assert not (folder / "round-0-close").exists()

    # No answer: the first plans go on to their reviews.
    for name in ("directed", "lightroom-audits"):
        shutil.rmtree(job.root / name, ignore_errors=True)
    asked.clear()
    monkeypatch.setattr(directed, "chat_vision", vision(None))
    _run(job)
    apart = json.loads((job.root / "directed" / "plan.json").read_text(encoding="utf-8"))["apart"]
    assert apart["replaced"] == [] and set(apart["close"]) == {"B"}
    assert sum("reviewing version" in p for p, _ in asked) >= 3


def test_directed_job_plans_reviews_revises_and_withholds(job, monkeypatch):
    monkeypatch.setattr(directed, "safety_gate", _gate({"C"}))
    revised_a = _plan(0.3, title="自然")
    vision = FakeVision({
        "A": [_review("same", revise_plan=revised_a, checks="partial"), _review("better")],
        "B": [_review("better")],
        "C": [_review("same", revise_plan=_plan(0.1, title="电影", grading=True)), _review("worse")],
    })
    monkeypatch.setattr(directed, "chat_vision", vision)
    summary = _run(job)

    rows = {row["slot"]: row for row in get_variants(job.conn, "job")}
    assert set(rows) == {"A", "B"}
    assert summary["delivery"]["delivered"] == 2 and summary["delivery"]["omitted"][0]["slot"] == "C"
    assert "highlight_clipping" in summary["delivery"]["omitted"][0]["reason"]
    sidecar = json.loads(rows["A"]["recipe_json"])
    assert sidecar["engine"] == directed.ENGINE and sidecar["directed"]["selected_round"] == 1
    assert sidecar["plan"]["develop"]["light"]["exposure"] == 0.3
    assert sidecar["style_workflow"]["mode"] == "art-directed" and sidecar["style_workflow"]["adaptation_notes"]
    assert sidecar["visual_review"]["decision"] == "better" and sidecar["lightroom_pixel_qa"]["score"] == 94.0
    assert sidecar["intent"]["natural_color"] is True
    assert (job.root / "variant_A.jpg").is_file() and (job.root / "variant_B.jpg").is_file()
    assert not (job.root / "variant_C.jpg").exists()
    # Round 1 of A rendered the reviewer's revision, not the first plan.
    documents = {destination.relative_to(job.root).as_posix(): doc for destination, doc in job.renders}
    assert documents["directed/A/round-0"]["light"]["exposure"] == 0.2
    assert documents["directed/A/round-1"]["light"]["exposure"] == 0.3
    assert "directed/B/round-1" not in documents  # Finished after one review.
    audit = json.loads((job.root / "lightroom-audits" / "variant_A.json").read_text(encoding="utf-8"))
    assert audit["engine"] == directed.ENGINE and audit["status"] == "delivered" and len(audit["rounds"]) == 2
    failed = json.loads((job.root / "lightroom-audits" / "variant_C.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    qa = json.loads((job.root / "lightroom-qa.json").read_text(encoding="utf-8"))
    assert qa["delivery"]["requested"] == 3 and qa["vision_model"] == "grok-test"
    # Grok is recorded as a temporary stand-in everywhere the provider is named.
    assert qa["temporary_provider"] is True and sidecar["directed"]["temporary_provider"] is True
    plan_audit = json.loads((job.root / "directed" / "plan.json").read_text(encoding="utf-8"))
    workflow = json.loads((job.root / "workflow.json").read_text(encoding="utf-8"))
    assert plan_audit["temporary_provider"] is True and workflow["temporary_provider"] is True
    stored = get_job(job.conn, "job")
    critique = json.loads(stored["critique_json"])
    assert critique["source"] == "xai" and critique["intent"]["subject_box"]["left"] == 0.25
    assert critique["issues"][0]["note"] == "中下：主体偏暗"
    assert (job.root / "directed" / "A" / "round-1" / "review.json").is_file()
    # Every reviewer saw the photo reading and the gate notes.
    reviews = [p for p in vision.prompts if "reviewing version" in p]
    assert len(reviews) == 5 and all("河边黄昏" in p for p in reviews)
    assert any("BLOCKS delivery" in p for p in reviews if "version C" in p)


def test_a_revision_is_measured_against_its_siblings_again(job, monkeypatch):
    vision = FakeVision({
        "A": [_review("better")],
        "B": [_review("better")],
        "C": [_review("same", revise_plan=_plan(1.0, title="电影", grading=True)), _review("better")],
    })
    monkeypatch.setattr(directed, "chat_vision", vision)
    _run(job)
    pattern = r"difference of this render from version A's first render: ([0-9.]+)"
    first, second = [float(re.search(pattern, p).group(1)) for p in vision.prompts if "reviewing version C" in p]
    assert first != second
    assert not any("difference of this render" in p for p in vision.prompts if "reviewing version A" in p)


def test_directed_job_raises_when_nothing_is_deliverable(job, monkeypatch):
    monkeypatch.setattr(directed, "safety_gate", _gate({"A", "B", "C"}))
    monkeypatch.setattr(directed, "chat_vision", FakeVision({slot: [_review("same"), _review("same")]
                                                            for slot in "ABC"}))
    with pytest.raises(directed.DirectedUnavailable, match="都没有通过"):
        _run(job)
    assert get_variants(job.conn, "job") == []


def test_reviewer_damage_or_confident_worse_blocks_a_round(job, monkeypatch):
    monkeypatch.setattr(directed, "chat_vision", FakeVision({
        "A": [_review("better", damage=[{"kind": "halo", "region": "主体边缘", "severity": 0.8}])],
        "B": [_review("worse")],
        "C": [_review("better")],
    }))
    summary = _run(job)
    assert [row["slot"] for row in get_variants(job.conn, "job")] == ["C"]
    reasons = {item["slot"]: item["reason"] for item in summary["delivery"]["omitted"]}
    assert "halo" in reasons["A"] and "变差" in reasons["B"]


# --------------------------------------------------------------------------
# Six versions: elements first, directions chosen with local experience, one damage check


LOOK_ID = next(iter(directed.LOOKS))
NEW_ROLE = "Cool blue-hour mood: the water and sky cooled to a deep blue, the boat kept warm as the one warm note."


def _six_planning(directions: dict | None = None) -> dict:
    return {"reading": PLANNING["reading"],
            "directions": directions if directions is not None else {
                "B": {"id": "dramatic", "why": "逆光下的小船适合压暗四周"},
                "C": {"id": f"look:{LOOK_ID}", "why": "暖色黄昏适合这个色调"},
                "D": {"id": "new", "title": "蓝调水面", "role": NEW_ROLE, "look": None, "why": "水面占画面一半"},
                "E": {"id": "airy", "why": "画面通透"},
                "F": {"id": "bogus", "why": "?"}},
            "plans": {"A": _plan(0.2, title="自然"), "B": _plan(0.6, title="光影"),
                      "C": _plan(0.2, title="电影", grading=True), "D": _plan(-0.3, title="蓝调"),
                      "E": _plan(0.95, title="通透"), "F": _plan(-0.7, title="浓郁")}}


def _six(job, **values):
    job.settings = job.settings.model_copy(update={"directed_versions": 6, "directed_max_revisions": 0, **values})


def test_six_versions_read_elements_choose_directions_and_share_one_damage_check(job, monkeypatch):
    _six(job)
    vision = FakeVision({}, _six_planning(), damage={
        "B": [{"kind": "banding", "region": "天空", "severity": 0.3}],
        "E": [{"kind": "halo", "region": "船身边缘", "severity": 0.8}]})
    monkeypatch.setattr(directed, "chat_vision", vision)
    summary = _run(job)

    # The elements are read first, on one small image, and the planner sees them with the direction library.
    assert ELEMENTS_ASK in vision.prompts[0] and vision.images[0] == 1
    plan_prompt = next(p for p in vision.prompts if PLAN_ASK.search(p))
    assert "Design 6 finished versions, A, B, C, D, E, F" in plan_prompt and "DIRECTION LIBRARY" in plan_prompt
    assert "A first look at the picture found: water, vehicle, sky; light golden" in plan_prompt
    assert "nothing has been rated yet" in plan_prompt and '"directions"' in plan_prompt
    # No per-version review: one damage check over every rendered version, image 1 first.
    assert not any("reviewing version" in p for p in vision.prompts)
    [damage_at] = [i for i, p in enumerate(vision.prompts) if DAMAGE_ASK in p]
    assert vision.images[damage_at] == 7 and "image 7 is version F" in vision.prompts[damage_at]

    rows = {row["slot"]: row for row in get_variants(job.conn, "job")}
    assert set(rows) == {"A", "B", "C", "D", "F"}
    assert summary["delivery"]["omitted"] == [{"slot": "E", "reason": "损坏检查：船身边缘：光晕"}]
    sidecars = {slot: json.loads(row["recipe_json"]) for slot, row in rows.items()}
    methods = {slot: s["style_workflow"]["method_id"] for slot, s in sidecars.items()}
    own = "own:" + __import__("hashlib").sha1("蓝调水面".encode()).hexdigest()[:10]
    # F asked for an unknown direction: it falls back to the first unused one (dramatic and airy are taken).
    assert methods == {"A": "direction:natural", "B": "direction:dramatic", "C": f"direction:look:{LOOK_ID}",
                       "D": f"direction:{own}", "F": "direction:palette"}
    # A look direction's plan carries its look; the planner's new direction is remembered for later photos.
    assert sidecars["C"]["plan"]["look"]["id"] == LOOK_ID
    assert [(d["id"], d["title"], d["origin_job"]) for d in directed.experience.get_directions(job.conn)] == [
        (own, "蓝调水面", "job")]
    assert "没有可用的方向选择（bogus）" in " ".join(sidecars["F"]["directed"]["plan_notes"])
    # What the experience learns from: the picture's profile, the direction and why, the measured picture.
    block = sidecars["B"]["experience"]
    assert block["profile"]["elements"] == ["water", "vehicle", "sky"] and block["profile"]["light"] == "golden"
    assert block["direction"] == {"id": "dramatic", "title": "戏剧光影", "why": "逆光下的小船适合压暗四周"}
    assert set(block["picture"]) >= {"black", "mid", "white", "colourfulness", "warmth"} and block["weight"] == 1.0
    assert directed.experience.variant_profile(rows["B"]) == block["profile"]
    assert sidecars["B"]["visual_review"]["decision"] == "checked"
    assert sidecars["B"]["visual_review"]["damage"][0]["kind"] == "banding"
    assert rows["B"]["why"].startswith("戏剧光影：逆光下的小船适合压暗四周。") and "天空：断层" in rows["B"]["why"]
    assert "损坏检查" not in rows["A"]["why"] and sidecars["A"]["intent"]["natural_color"] is True
    assert summary["delivery"]["order"] == ["A", "B", "C", "D", "F"]
    workflow = json.loads((job.root / "workflow.json").read_text(encoding="utf-8"))
    assert {"vision_elements", "experience", "plan_directions", "damage_check"} <= set(workflow["axis_order"])
    assert "vision_review" not in workflow["axis_order"]
    saved = json.loads((job.root / "directed" / "plan.json").read_text(encoding="utf-8"))
    assert "_prompt" not in saved and saved["experience"]["used"] is False
    assert saved["directions"]["D"]["new"] is True and saved["elements"]["light_kind"] == "golden"
    assert (job.root / "directed" / "damage.json").is_file()


def test_picks_on_a_similar_photo_reach_the_planner_and_order_the_versions(job, monkeypatch):
    _six(job)
    monkeypatch.setattr(directed, "chat_vision", FakeVision({}, _six_planning()))
    _run(job)
    rows = {row["slot"]: row for row in get_variants(job.conn, "job")}
    from superphoto.db import set_rating
    from superphoto.scene_preferences import variant_method
    for slot, verdict in (("B", 2), ("C", -1), ("E", 1)):
        set_rating(job.conn, "job", slot, verdict, variant_method(rows[slot]),
                   directed.experience.variant_profile(rows[slot]))

    # The next photo shows the same things in the same light.
    insert_job(job.conn, "next", "next.jpg")
    root = job.settings.jobs_dir / "next"
    original = _jpeg(root / "original.jpg")
    baseline = _jpeg(root / "lightroom-exports" / "baseline" / "baseline.jpg")
    vision = FakeVision({}, _six_planning())
    monkeypatch.setattr(directed, "chat_vision", vision)
    summary = directed.run_directed_job(job.settings, job.conn, "next", original, root, technical=job.technical,
                                        baseline_export=baseline, selection=SELECTION)
    plan_prompt = next(p for p in vision.prompts if PLAN_ASK.search(p))
    assert "LOCAL EXPERIENCE: this user's own picks" in plan_prompt
    assert "戏剧光影 (dramatic): favorite 1" in plan_prompt and "pass 1" in plan_prompt
    assert "not a ban" in plan_prompt
    # The favourite's direction first, the passed one last; the rest keep their order.
    order = summary["delivery"]["order"]
    assert order[0] == "B" and order[1] == "E" and order[-1] == "C"
    sidecar = json.loads({r["slot"]: r for r in get_variants(job.conn, "next")}["B"]["recipe_json"])
    assert sidecar["experience"]["weight"] > 1.0 and sidecar["experience"]["rank"] == 0
    assert json.loads((root / "directed" / "plan.json").read_text(encoding="utf-8"))["experience"]["used"] is True


def test_chosen_directions_fall_back_and_never_repeat():
    library = directed.experience.library()
    plans = {slot: _plan(0.2, title=slot) for slot in "ABCDE"}
    plans["E"]["look"] = {"id": LOOK_ID, "strength": 0.9}
    chosen, notes = directed._chosen_directions({
        "A": {"id": "dramatic"},                                   # A is always the natural finish.
        "B": {"id": "dramatic", "why": "w"},
        "C": {"id": "dramatic"},                                   # Repeated: falls back.
        "D": {"id": "natural"},                                    # Natural belongs to A.
        "E": {"id": "new", "title": "", "role": "x"},              # A new direction without a title: its plan's.
    }, plans, library)
    assert {slot: item["id"] for slot, item in chosen.items()} == {
        "A": "natural", "B": "dramatic", "C": "airy", "D": "palette", "E": directed.experience.own_id("E")}
    assert chosen["E"]["new"] and chosen["E"]["title"] == "E" and chosen["E"]["look"] == LOOK_ID
    assert set(notes) == {"C", "D", "E"} and chosen["B"]["why"] == "w"
    # No answer at all still gives every version a direction.
    chosen, _ = directed._chosen_directions(None, {slot: _plan(0.2, title=slot) for slot in "ABC"}, library)
    assert [chosen[slot]["id"] for slot in "ABC"] == ["natural", "dramatic", "airy"]


def test_an_incomplete_new_direction_is_named_after_its_plan_not_a_taken_one():
    # Seen with a real model: F asked for a new direction without a role after B-E had used the built-in ones,
    # and became a second "palette", so its rating would have been learned as C's.
    library = directed.experience.library()
    plans = {slot: _plan(0.2, title=slot) for slot in "ABCDEFG"}
    plans["E"]["look"] = {"id": LOOK_ID, "strength": 0.9}
    plans["F"]["title"] = "高原冷冽·深青蓝（新方向）"
    plans["G"]["title"] = "冷暖分区"
    chosen, notes = directed._chosen_directions({
        "B": {"id": "dramatic"}, "C": {"id": "palette"}, "D": {"id": "airy"}, "E": {"id": f"look:{LOOK_ID}"},
        "F": {"id": "new", "title": "", "why": "缺一个冷冽高原的方向"},
        "G": {"id": "palette"},                                    # Taken, and no built-in left: the plan's title.
    }, plans, library)
    assert chosen["F"]["id"] == directed.experience.own_id("高原冷冽·深青蓝")
    assert chosen["F"]["title"] == "高原冷冽·深青蓝" and chosen["F"]["role"] == "缺一个冷冽高原的方向"
    assert chosen["G"]["id"] == directed.experience.own_id("冷暖分区") and chosen["G"]["role"] == "G：压天空、提主体"
    assert chosen["F"]["new"] and chosen["G"]["new"] and set(notes) == {"F", "G"}
    assert len({item["id"] for item in chosen.values()}) == len(chosen)


def test_without_revisions_an_unavailable_damage_check_still_delivers(job, monkeypatch):
    _six(job, directed_versions=3)
    vision = FakeVision({})

    def offline_damage(settings, model, prompt, images, **kwargs):
        if DAMAGE_ASK in prompt:
            raise VisionUnavailable("offline")
        return vision(settings, model, prompt, images, **kwargs)

    monkeypatch.setattr(directed, "chat_vision", offline_damage)
    summary = _run(job)
    rows = {row["slot"]: row for row in get_variants(job.conn, "job")}
    assert set(rows) == {"A", "B", "C"} and summary["damage_check"]["error"]
    assert all("损坏检查不可用" in row["why"] for row in rows.values())
    # The trio still records its fixed directions, so its picks are learned too.
    methods = {slot: json.loads(row["recipe_json"])["style_workflow"]["method_id"] for slot, row in rows.items()}
    assert methods == {"A": "direction:natural", "B": "direction:dramatic", "C": "direction:palette"}
    assert not any("LOCAL EXPERIENCE" in p or "DIRECTION LIBRARY" in p for p in vision.prompts)


def test_planner_failures_raise_directed_unavailable(job, monkeypatch):
    def offline(*args, **kwargs):
        raise VisionUnavailable("vision unavailable (xai:APITimeoutError)")

    monkeypatch.setattr(directed, "chat_vision", offline)
    with pytest.raises(directed.DirectedUnavailable, match="APITimeoutError"):
        _run(job)
    monkeypatch.setattr(directed, "chat_vision", FakeVision({}, planning={"reading": {}, "plans": {
        "B": _plan(0.2, title="b")}}))
    with pytest.raises(directed.DirectedUnavailable, match="A 版本"):
        _run(job)
    # Only the look probes, which come before planning; no version was rendered.
    assert job.renders and {destination.parent.name for destination, _ in job.renders} == {"looks"}


def test_non_json_answer_is_retried_once(job, monkeypatch):
    answers = iter(["好的，我来分析这张照片。", json.dumps(PLANNING, ensure_ascii=False)])
    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: next(answers))
    image = directed.load_preview(job.baseline)
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert set(planning["plans"]) == {"A", "B", "C"} and len(planning["audit"]["calls"]) == 2
    assert planning["reading"]["genre"] == "landscape"


def test_a_vision_call_that_gets_no_answer_is_asked_again_after_a_wait(job, monkeypatch, _no_vision_waits):
    full = dict(_review("better"), score=91)
    calls: list[int] = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise VisionUnavailable("vision unavailable (客户端视觉模型:VisionUnavailable)")
        return json.dumps(full, ensure_ascii=False)

    monkeypatch.setattr(directed, "chat_vision", flaky)
    image = directed.load_preview(job.baseline)
    plan, _ = directed.sanitize_plan(_plan(0.2, title="t"), slot="C")
    gate = {"passed": True, "hard": [], "advisory": []}
    review = directed.review_version(job.settings, SELECTION, "C", plan, {}, image, image, gate, "jpeg")
    assert review["score"] == 91 and len(calls) == 3 and _no_vision_waits == list(directed.VISION_RETRY_WAITS)
    assert len(review["audit"]["calls"]) == 3
    # Out of waits: the review is unavailable, as before, after one call per wait and one more.
    calls.clear()

    def offline(*args, **kwargs):
        calls.append(1)
        raise VisionUnavailable("vision unavailable (x:APITimeoutError)")

    monkeypatch.setattr(directed, "chat_vision", offline)
    review =directed.review_version(job.settings, SELECTION, "C", plan, {}, image, image, gate, "jpeg")
    assert review["verdict"] == "unavailable" and "APITimeoutError" in review["error"]
    assert len(calls) == len(directed.VISION_RETRY_WAITS) + 1


def test_plans_cut_off_after_a_ask_for_the_missing_versions(job, monkeypatch):
    whole = json.dumps(PLANNING, ensure_ascii=False)
    cut = whole[:whole.index('"B"') + 40]   # A complete, B cut mid-plan
    rest = json.dumps({"plans": {"B": PLANNING["plans"]["B"], "C": PLANNING["plans"]["C"]}}, ensure_ascii=False)
    prompts: list[str] = []

    def answer(settings, model, prompt, images, **_):
        prompts.append(prompt)
        return cut if len(prompts) == 1 else rest

    monkeypatch.setattr(directed, "chat_vision", answer)
    image = directed.load_preview(job.baseline)
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert set(planning["plans"]) == {"A", "B", "C"} and len(prompts) == 2
    assert "cut off after the plan(s) A" in prompts[1] and '"title"' in prompts[1].split("Already written:")[1]
    assert planning["plans"]["B"]["title"] == PLANNING["plans"]["B"]["title"]
    # Asked once: a second short answer leaves the versions that were written.
    prompts.clear()
    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: prompts.append(1) or cut)
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert set(planning["plans"]) == {"A"} and len(prompts) == 2


def _subject_target(median: float, *, limit: bool = True, title: str = "自然") -> dict:
    plan = _plan(0.2, title=title)
    subject = plan["masks"][1]
    subject["adjust"].pop("exposure")
    subject["target"] = {"box": [0.3, 0.5, 0.45, 0.65], "median": median}
    if limit:
        subject["limit"] = {"kind": "luminance", "lo": 0.0, "hi": 0.8, "feather": 0.05}
    return plan


def test_region_targets_out_of_reach_in_image_1_are_found_before_rendering(job):
    image = directed.load_preview(job.baseline)
    far, _ = directed.sanitize_plan(_subject_target(0.9), slot="A")
    [item] = directed.plan_reach(far, image)
    # The box measures about .40 in image 1; +1.25 EV in linear light takes that to about .59.
    assert item["name"] == "主体" and item["target"] == 0.9 and item["precise"]
    assert abs(item["image"] - 0.40) < 0.03 and abs(item["reach"][1] - 0.59) < 0.03 and item["ev"] == [-1.25, 1.25]
    near, _ = directed.sanitize_plan(_subject_target(0.55), slot="A")
    assert directed.plan_reach(near, image) == []
    # A bare radial is held to 0.7 EV, so the same .55 is out of its reach, and the line says how to get further.
    bare, _ = directed.sanitize_plan(_subject_target(0.55, limit=False), slot="A")
    [item] = directed.plan_reach(bare, image)
    assert not item["precise"] and item["reach"][1] < 0.53
    line = directed._reach_line("A", item)
    assert 'plan A, mask "主体"' in line and "lifts it at most +0.70 EV" in line and "with a luminance or color" in line
    assert line.endswith("The target 0.55 is out of reach.")
    # A darkening target below what the mask takes it to.
    deep, _ = directed.sanitize_plan(_subject_target(0.05), slot="A")
    [item] = directed.plan_reach(deep, image)
    assert "darkens it at most -1.25 EV" in directed._reach_line("A", item)


def _patched_picture() -> Image.Image:
    """A lit ground with a face in shade, a patch of white fur and a dark horn beside it (p22-r1 DSC_1963)."""
    yy, xx = np.mgrid[0:240, 0:360].astype(np.float32)
    grey = 200 + 40 * xx / 360                          # sunlit ground, display luma about .82-.94
    pixels = np.stack([grey, grey * 0.97, grey * 0.92], axis=-1)
    pixels[96:132, 144:180] = (52, 44, 38)              # face: x .40-.50, y .40-.55
    pixels[120:168, 216:288] = (236, 236, 232)          # fur: x .60-.80, y .50-.70
    pixels[108:120, 295:310] = (30, 28, 26)             # horn: x .82-.86, y .45-.50
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def _region_plan(*masks: dict, tone: dict | None = None) -> dict:
    plan = dict(_plan(0.0, title="自然"), masks=list(masks))
    if tone:
        plan["tone"] = tone
    return directed.sanitize_plan(plan, slot="A")[0]


def test_misplaced_points_mixed_boxes_and_small_masks_are_found_before_rendering():
    image = _patched_picture()
    face = {"kind": "radial", "center": {"x": 0.45, "y": 0.475}, "rx": 0.04, "ry": 0.05}

    def mask(name, shape, box, median, **extra):
        return {"name": name, "purpose": "提亮", "shape": shape, "adjust": {"shadows": 10},
                "target": {"box": box, "median": median}, **extra}

    # A box drawn round the face (luma .18) takes in the lit ground: its median is neither, so it is sent back.
    found = directed.plan_reach(_region_plan(mask("脸", face, [0.35, 0.35, 0.58, 0.62], 0.2)), image)
    [item] = [entry for entry in found if entry["kind"] == "spread"]
    assert item["p10"] < 0.2 and item["p90"] - item["p10"] > directed.REACH_SPREAD
    line = directed._reach_line("A", item)
    assert 'plan A, mask "脸"' in line and "holds the region and its surroundings" in line and "Draw the box" in line
    # On the face alone .2 is within a bare radial's 0.7 EV.
    tight = [0.41, 0.41, 0.49, 0.54]
    assert directed.plan_reach(_region_plan(mask("脸", face, tight, 0.2)), image) == []
    # Colour points on the horn for a box on the fur: the mask would select the horn's pixels, not the fur.
    fur, horn = [0.62, 0.52, 0.78, 0.68], {"kind": "color", "points": [{"x": 0.84, "y": 0.475}], "range": 50}
    [item] = directed.plan_reach(_region_plan(mask("白毛", horn, fur, 0.85)), image)
    assert item["kind"] == "points" and item["share"] < 0.1 and item["points"][0]["luma"] < 0.2 < 0.8 < item["image"]
    line = directed._reach_line("A", item)
    assert 'color mask "白毛"' in line and "only 0% of the box" in line and "points are misplaced" in line
    on_fur = {"kind": "color", "points": [{"x": 0.7, "y": 0.6}, {"x": 0.84, "y": 0.475}], "range": 50}
    assert directed.plan_reach(_region_plan(mask("白毛", on_fur, fur, 0.85)), image) == []
    # Tone targets that sink the shadows take the face down with them; a small radial holds none of the frame's
    # tones, so it is judged from there, while a large one may still carry its region's percentiles.
    sunk = {"black": 0.02, "shadows": 0.12, "mid": 0.55, "highlights": 0.78, "white": 0.92}
    found = directed.plan_reach(_region_plan(mask("脸", face, tight, 0.2), tone=sunk), image)
    [item] = [entry for entry in found if entry["kind"] == "reach"]
    assert item["small"] and {entry["kind"] for entry in found} == {"reach", "against"}
    assert item["reach"][1] + directed.REACH_SLACK >= 0.2 > item["reach_curve"][1] + directed.REACH_SLACK
    assert "a mask this small holds none of the frame's tones" in directed._reach_line("A", item)
    # A large one is not out of reach, though the curve still takes the face down that its mask lifts back.
    wide = dict(face, rx=0.18, ry=0.25)
    found = directed.plan_reach(_region_plan(mask("脸", wide, tight, 0.2), tone=sunk), image)
    assert [entry["kind"] for entry in found] == ["against"]


def test_a_region_target_its_mask_must_take_back_from_the_curve_is_found_before_rendering():
    image = _patched_picture()
    anchors = tone.measure(tone.load_small(image))
    face = {"kind": "radial", "center": {"x": 0.45, "y": 0.475}, "rx": 0.04, "ry": 0.05}
    tight = [0.41, 0.41, 0.49, 0.54]

    def plan(tone_targets, median=0.2):
        return _region_plan({"name": "脸", "purpose": "提亮", "shape": face, "adjust": {"shadows": 10},
                             "limit": {"kind": "luminance", "lo": 0.0, "hi": 0.5, "feather": 0.05},
                             "target": {"box": tight, "median": median}}, tone=tone_targets)

    # The face (.18) is the frame's black: sinking the black and shadows takes it far down, its target lifts it back.
    sunk = {"black": 0.02, "shadows": 0.12, "mid": 0.55, "highlights": 0.78, "white": 0.92}
    [item] = [entry for entry in directed.plan_reach(plan(sunk), image) if entry["kind"] == "against"]
    assert item["curve_ev"] <= -tone.AGAINST_EV and item["need"] >= tone.AGAINST_EV
    assert abs(item["image"] - 0.18) < 0.02 and item["by_curve"] < 0.05 and item["anchor"] == "black"
    # Neither percentile alone keeps the face; the two around it together do, at values between written and image 1.
    fix = item["anchor_fix"]
    assert set(fix) == {"black", "shadows"}
    assert all(min(sunk[name], anchors[name]) <= value <= max(sunk[name], anchors[name]) for name, value in fix.items())
    line = directed._reach_line("A", item)
    assert 'plan A, mask "脸"' in line and "darken that brightness" in line and "lift it +" in line
    assert "push the same pixels opposite ways" in line and f"aim the region at about {item['edge']:.2f}" in line
    assert f"write the black at about {fix['black']:.2f} and the shadows at about {fix['shadows']:.2f}" in line
    assert line.endswith("whichever this version's role calls for.")
    # Written so, or with the tone targets left at image 1's, the curve no longer takes the face away.
    assert not [entry for entry in directed.plan_reach(plan(dict(sunk, **fix)), image) if entry["kind"] == "against"]
    kept = {name: round(value, 3) for name, value in anchors.items()}
    assert directed.plan_reach(plan(kept), image) == []


def test_a_raw_frame_lift_is_kept_when_a_region_pushes_against_it():
    # ref-r2 DSC02262: a deep sky against the developed snowfield; the frame was pulled back to the negative.
    item = {"kind": "against", "name": "天空", "box": [0.15, 0.02, 0.85, 0.18], "target": 0.38, "image": 0.42,
            "by_curve": 0.6, "curve_ev": 1.12, "need": -1.42, "edge": 0.512, "anchor": "mid", "anchor_image": 0.385,
            "anchor_written": 0.5, "anchor_fix": {"highlights": 0.479}}
    jpeg = directed._reach_line("A", item)
    assert "write the highlights at about 0.48" in jpeg
    raw = directed._reach_line("A", item, "raw")
    assert "highlights at about" not in raw and "RAW's development" in raw and "about 0.51" in raw
    # A RAW frame that darkens the region keeps the usual choice.
    darker = dict(item, curve_ev=-1.1, need=1.3)
    assert directed._reach_line("A", darker, "raw") == directed._reach_line("A", darker)
    for kind, rule in directed._REACH_FRAME_RULE.items():
        text = directed._PLANS_REACH.format(lines="-", slots="A", frame_rule=rule, written="{}")
        assert ("near image 1" in text) == (kind == "jpeg") and ("negative" in text) == (kind == "raw")


def _sky_picture() -> Image.Image:
    """Blue sky (luma .58-.62) over red rock (.22-.85): the sky's few tones are the frame's mid (p23-r1 DSC_1515)."""
    yy, xx = np.mgrid[0:240, 0:360].astype(np.float32)
    rock = (0.25 + 0.72 * xx / 359)[..., None] * np.array([255.0, 217.0, 178.0])
    sky = (0.96 + 0.08 * xx / 359)[..., None] * np.array([115.0, 158.0, 217.0])
    pixels = np.where((yy < 96)[..., None], sky, rock)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def test_a_region_target_that_takes_a_frame_percentile_off_its_own_is_found_before_rendering():
    image = _sky_picture()
    anchors = tone.measure(tone.load_small(image))
    kept = {name: round(value, 3) for name, value in anchors.items()}
    points = [{"x": 0.2, "y": 0.1}, {"x": 0.8, "y": 0.3}]

    def sky(median):
        return {"name": "天空", "purpose": "压深天空", "shape": {"kind": "color", "points": points, "range": 50},
                "adjust": {"dehaze": 10}, "target": {"box": [0.6, 0.1, 0.9, 0.3], "median": median}}

    # The frame mid kept where image 1 has it while the sky, which is that mid, goes .6 -> .48.
    [item] = directed.plan_reach(_region_plan(sky(0.48), tone=kept), image)
    assert item["kind"] == "shared" and item["anchor"] == "mid" and item["share"] > 0.8
    assert item["ev"] < -0.4 and item["simulated"] < kept["mid"] - directed.SHARED_MISS
    assert abs(item["simulated"] - 0.48) < 0.03 and abs(item["by_curve"] - 0.6) < 0.02
    line = directed._reach_line("A", item)
    assert 'color mask "天空"' in line and "pixels at the frame's mid" in line and "contradict each other" in line
    assert f"will measure about {item['simulated']:.2f}, not the {kept['mid']:.2f} written" in line
    # One change is asked for, the region's target kept first (p27-r1 DSC_1515 B moved both and the sky mask lifted).
    assert "changing ONE of the two" in line and "Moving both moves the contradiction" in line
    assert f"keep the target 0.48 and write the mid at about {item['simulated']:.2f}" in line
    # Either agreeing pair passes: the mid written where the sky takes it, or the sky left near the mid's.
    agreed = dict(kept, mid=item["simulated"])
    assert directed.plan_reach(_region_plan(sky(0.48), tone=agreed), image) == []
    assert directed.plan_reach(_region_plan(sky(item["by_curve"] - 0.02), tone=kept), image) == []
    # A linear burn over the sky holds the same pixels by its geometry; a luminance limit under the sky lets it go.
    burn = {"name": "天空", "purpose": "压深天空", "shape": {"kind": "linear", "start": {"x": 0.5, "y": 0.0},
                                                          "end": {"x": 0.5, "y": 0.6}},
            "adjust": {"dehaze": 10}, "target": {"box": [0.6, 0.05, 0.9, 0.25], "median": 0.48}}
    [item] = directed.plan_reach(_region_plan(burn, tone=kept), image)
    assert item["kind"] == "shared" and item["shape"] == "linear" and item["anchor"] == "mid"
    assert item["simulated"] < kept["mid"] - directed.SHARED_MISS
    assert 'linear mask "天空"' in directed._reach_line("A", item)
    under = dict(burn, limit={"kind": "luminance", "lo": 0.0, "hi": 0.3, "feather": 0.04})
    assert not [entry for entry in directed.plan_reach(_region_plan(under, tone=kept), image) if entry["kind"] == "shared"]


def test_plans_with_targets_out_of_reach_are_asked_for_once_more(job, monkeypatch):
    first = dict(PLANNING, plans=dict(PLANNING["plans"], A=_subject_target(0.9)))
    fixed = {"plans": {"A": _subject_target(0.52, title="自然（主体按原图可达）")}}
    prompts: list[str] = []

    def answer(settings, model, prompt, images, **_):
        prompts.append(prompt)
        return json.dumps(first if len(prompts) == 1 else fixed, ensure_ascii=False)

    monkeypatch.setattr(directed, "chat_vision", answer)
    image = directed.load_preview(job.baseline)
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert len(prompts) == 2 and "cannot be met as written" in prompts[1]
    asked = prompts[1].split("cannot be met as written")[1]
    assert 'plan A, mask "主体"' in asked and "The target 0.90 is out of reach." in asked
    assert 'only "plans" with the key(s) A,' in asked and '"median": 0.9' in asked.split("Your plans were:")[1]
    assert planning["plans"]["A"]["title"] == "自然（主体按原图可达）"
    assert planning["plans"]["B"]["title"] == PLANNING["plans"]["B"]["title"]
    assert set(planning["reach"]["flagged"]) == {"A"} and planning["reach"]["replaced"] == ["A"]
    assert planning["reach"]["after"] == {"A": []} and "again" not in planning["reach"]
    # A replaced plan still out of reach is asked for once again with its own numbers, and no more after that.
    still = {"plans": {"A": _subject_target(0.85, title="自然（仍不可达）")}}
    prompts.clear()
    monkeypatch.setattr(directed, "chat_vision", lambda settings, model, prompt, images, **_: prompts.append(prompt)
                        or json.dumps(first if len(prompts) == 1 else still, ensure_ascii=False))
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert len(prompts) == 3 and "The target 0.85 is out of reach." in prompts[2].split("Your plans were:")[0]
    assert '"median": 0.85' in prompts[2].split("Your plans were:")[1]
    again = planning["reach"]["again"]
    assert set(again["flagged"]) == {"A"} and again["replaced"] == ["A"] and again["after"]["A"]
    assert again["refused"] == [] and planning["reach"]["refused"] == []
    # An answer that flags more than the plan it would replace is refused and that plan kept, not asked again.
    worse = {"plans": {"A": _subject_target(0.95, title="自然（更多冲突）")}}
    real = directed.plan_reach

    def reach(plan, image):
        found = real(plan, image)
        return found * 2 if found and plan["title"] == "自然（更多冲突）" else found

    monkeypatch.setattr(directed, "plan_reach", reach)
    prompts.clear()
    monkeypatch.setattr(directed, "chat_vision", lambda settings, model, prompt, images, **_: prompts.append(prompt)
                        or json.dumps(first if len(prompts) == 1 else worse, ensure_ascii=False))
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert len(prompts) == 2 and planning["reach"]["refused"] == ["A"] and planning["reach"]["replaced"] == []
    assert planning["plans"]["A"]["masks"][1]["target"]["median"] == 0.9 and "again" not in planning["reach"]
    monkeypatch.setattr(directed, "plan_reach", real)
    # Asked once: an answer without plans keeps what was written.
    prompts.clear()
    monkeypatch.setattr(directed, "chat_vision",
                        lambda *a, **k: prompts.append(1) or json.dumps(first if len(prompts) == 1 else {"x": 1}))
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert len(prompts) == 2 and planning["reach"]["replaced"] == []
    assert planning["plans"]["A"]["masks"][1]["target"]["median"] == 0.9
    # Nothing out of reach: no second call and no reach report.
    prompts.clear()
    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: prompts.append(1) or json.dumps(PLANNING))
    planning = directed.plan_versions(job.settings, SELECTION, image, job.technical)
    assert len(prompts) == 1 and planning["reach"] == {}


def test_a_review_cut_off_before_its_score_is_asked_again(job, monkeypatch):
    full = dict(_review("better"), deductions=[{"area": "tone", "points": 4, "why": "偏暗"}], score=96)
    cut = '{"deductions": [{"area": "tone", "points": 3, "why": "高光未到位"}, {"area": "color", "points": 3, "why": "调色只'
    prompts: list[str] = []

    def answer(settings, model, prompt, images, **_):
        prompts.append(prompt)
        return cut if len(prompts) == 1 else json.dumps(full, ensure_ascii=False)

    monkeypatch.setattr(directed, "chat_vision", answer)
    image = directed.load_preview(job.baseline)
    plan, _ = directed.sanitize_plan(_plan(0.2, title="t"), slot="C")
    review = directed.review_version(job.settings, SELECTION, "C", plan, {}, image, image,
                                     {"passed": True, "hard": [], "advisory": []}, "jpeg")
    assert review["score"] == 96 and review["verdict"] == "better" and len(prompts) == 2
    assert "cut off before the score" in prompts[1] and len(review["audit"]["calls"]) == 2
    # A second cut-off answer keeps what the first one said.
    prompts.clear()
    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: prompts.append(1) or cut)
    review = directed.review_version(job.settings, SELECTION, "C", plan, {}, image, image,
                                     {"passed": True, "hard": [], "advisory": []}, "jpeg")
    assert review["score"] is None and [d["points"] for d in review["deductions"]] == [3.0, 3.0] and len(prompts) == 2


def test_selection_requires_lightcraft_and_a_ready_remote_or_ollama_model(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path, lightroom_enabled=True)
    monkeypatch.setattr(directed, "probe", lambda _: {"ready": True, "provider": "xai", "resolved_model": "g"})
    assert directed.directed_selection(settings)["resolved_model"] == "g"
    assert directed.directed_selection(settings.model_copy(update={"pipeline_mode": "legacy"})) is None
    assert directed.directed_selection(settings.model_copy(update={"lightroom_enabled": False})) is None
    monkeypatch.setattr(directed, "probe", lambda _: {"ready": False, "provider": "local", "resolved_model": None})
    assert directed.directed_selection(settings) is None


# --------------------------------------------------------------------------
# Pipeline integration


def test_pipeline_falls_back_to_legacy_in_auto_and_fails_in_directed_mode(tmp_path, monkeypatch):
    from superphoto import pipeline

    settings = Settings(data_dir=tmp_path / "data", lightroom_enabled=True)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job", "photo.jpg")
    root = settings.jobs_dir / "job"
    root.mkdir(parents=True)
    technical = SimpleNamespace(model_dump_json=lambda: "{}")
    args = (conn, "job", root / "original.jpg", root, technical, root / "baseline.jpg")

    monkeypatch.setattr(directed, "directed_selection", lambda _: None)
    assert pipeline._run_directed(settings, *args) is False
    fallback = json.loads((root / "directed-fallback.json").read_text(encoding="utf-8"))
    assert fallback["fallback"] == "legacy" and "视觉模型" in fallback["reason"]

    monkeypatch.setattr(directed, "directed_selection", lambda _: SELECTION)

    def unavailable(*a, **k):
        replace_variants(conn, "job", [{"id": "x", "slot": "A", "look_id": "directed", "title": "t",
                                        "recipe_json": "{}", "filename": "variant_A.jpg"}])
        (root / "lightroom-audits").mkdir(exist_ok=True)
        raise directed.DirectedUnavailable("三个版本都没有通过")

    monkeypatch.setattr(directed, "run_directed_job", unavailable)
    assert pipeline._run_directed(settings, *args) is False
    assert get_variants(conn, "job") == [] and not (root / "lightroom-audits").exists()
    assert "都没有通过" in json.loads((root / "directed-fallback.json").read_text(encoding="utf-8"))["reason"]
    strict = settings.model_copy(update={"pipeline_mode": "directed"})
    with pytest.raises(directed.DirectedUnavailable):
        pipeline._run_directed(strict, *args)

    def broken(*a, **k):
        raise KeyError("bug")

    monkeypatch.setattr(directed, "run_directed_job", broken)
    assert pipeline._run_directed(settings, *args) is False
    assert "KeyError" in (root / "directed-error.txt").read_text(encoding="utf-8")
    with pytest.raises(KeyError):
        pipeline._run_directed(strict, *args)

    monkeypatch.setattr(directed, "run_directed_job", lambda *a, **k: {"delivery": {}})
    assert pipeline._run_directed(settings, *args) is True
    legacy = settings.model_copy(update={"pipeline_mode": "legacy"})
    monkeypatch.setattr(directed, "directed_selection", lambda _: pytest.fail("legacy never probes"))
    assert pipeline._run_directed(legacy, *args) is False
    conn.close()


def test_run_job_uses_directed_result_and_skips_the_slider_pipeline(tmp_path, monkeypatch, scene):
    from superphoto import pipeline

    settings = Settings(data_dir=tmp_path / "data", lightroom_enabled=True)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job", "photo.jpg")
    root = settings.jobs_dir / "job"
    root.mkdir(parents=True)
    original = root / "original.jpg"
    shutil.copy2(scene, original)
    baseline = root / "baseline.jpg"
    shutil.copy2(scene, baseline)
    monkeypatch.setattr(pipeline, "_prepare_lightroom_baseline", lambda *a: {"export": baseline})
    seen = {}

    def run_directed(settings_, conn_, job_id, original_, root_, technical, baseline_export):
        seen.update(job_id=job_id, baseline=baseline_export, kind=technical.source_kind)
        assert (root_ / "critique_preview.jpg").is_file() and (root_ / "technical.json").is_file()
        return True

    monkeypatch.setattr(pipeline, "_run_directed", run_directed)
    monkeypatch.setattr(pipeline, "critique_photo", lambda *a, **k: pytest.fail("legacy critique must not run"))
    pipeline.run_job(settings, conn, "job", original)
    assert seen == {"job_id": "job", "baseline": baseline, "kind": "jpeg"}
    assert get_job(conn, "job")["status"] == "ready"
    conn.close()


# --------------------------------------------------------------------------
# Refinement of a directed finish


@pytest.fixture
def delivered(job, monkeypatch):
    monkeypatch.setattr(directed, "chat_vision", FakeVision({slot: [_review("better")] for slot in "ABC"}))
    _run(job)
    update_job(job.conn, "job", status="ready", step="ready")
    monkeypatch.setattr(ref, "connect_lightroom", lambda *_: (object(), {}))
    monkeypatch.setattr(directed, "probe", lambda _: SELECTION)
    monkeypatch.setattr("superphoto.lightroom_qa.render_distance", lambda *a: 3.0)
    job.renders.clear()
    return job


def _refine_answer(exposure: float):
    revised = _plan(0.2, title="自然")
    revised["masks"].append({"name": "右侧压暗", "purpose": "压暗右侧亮边",
                             "shape": {"kind": "linear", "start": {"x": 1.0, "y": 0.5}, "end": {"x": 0.8, "y": 0.5}},
                             "adjust": {"exposure": exposure}})
    return {"summary": "压暗右侧亮边", "plan": revised}


def test_refinement_revises_the_directed_plan_on_a_pristine_copy(delivered, monkeypatch):
    job = delivered
    image, base = ref.refinement_base(job.settings, job.conn, "job", "A", None)
    assert image == job.root / "variant_A.jpg" and base["plan"]["title"] == "自然"
    prompts = []

    def vision(settings, model, prompt, images, **kwargs):
        prompts.append(prompt)
        assert len(images) == 1
        return json.dumps(_refine_answer(-0.3), ensure_ascii=False)

    monkeypatch.setattr(directed, "chat_vision", vision)
    insert_refinement(job.conn, "r1", "job", "A", None, "右边太亮了，压暗一点")
    ref.run_refinement(job.settings, job.conn, "r1")
    record = get_refinement(job.conn, "r1")
    assert record["status"] == "ready", record["error"]
    assert record["summary"] == "压暗右侧亮边"
    saved = json.loads(record["recipe_json"])
    assert saved["engine"] == directed.ENGINE and saved["plan"]["masks"][-1]["name"] == "右侧压暗"
    assert "右边太亮了" in prompts[0] and "自然" in prompts[0]
    folder = job.root / "refinements" / "r1"
    assert (folder / "result.jpg").is_file() and (folder / "source.jpg").is_file()
    destination, document = job.renders[-1]
    assert destination == folder / "exports" and document["masks"][-1]["adjust"]["exposure"] == -0.3
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    assert audit["engine"] == directed.ENGINE and audit["base_plan"]["title"] == "自然"

    # A follow-up builds on the refined plan, not the original delivery.
    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: json.dumps(_refine_answer(-0.5), ensure_ascii=False))
    image, base = ref.refinement_base(job.settings, job.conn, "job", "A", "r1")
    assert image == folder / "result.jpg" and base["plan"]["masks"][-1]["adjust"]["exposure"] == -0.3
    insert_refinement(job.conn, "r2", "job", "A", "r1", "再暗一点")
    ref.run_refinement(job.settings, job.conn, "r2")
    assert get_refinement(job.conn, "r2")["status"] == "ready"


def test_refinement_keeps_previous_version_when_vision_or_gate_fails(delivered, monkeypatch):
    job = delivered
    monkeypatch.setattr(directed, "probe", lambda _: {"ready": False, "provider": "local"})
    insert_refinement(job.conn, "r1", "job", "B", None, "天空再蓝一点")
    ref.run_refinement(job.settings, job.conn, "r1")
    record = get_refinement(job.conn, "r1")
    assert record["status"] == "error" and "视觉模型" in record["error"]

    monkeypatch.setattr(directed, "probe", lambda _: SELECTION)
    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: "我无法处理")
    insert_refinement(job.conn, "r2", "job", "B", None, "天空再蓝一点")
    ref.run_refinement(job.settings, job.conn, "r2")
    assert "调整前版本已保留" in get_refinement(job.conn, "r2")["error"]

    monkeypatch.setattr(directed, "chat_vision", lambda *a, **k: json.dumps(_refine_answer(-0.3)))
    monkeypatch.setattr(directed, "safety_gate", _gate({"B"}))
    insert_refinement(job.conn, "r3", "job", "B", None, "天空再蓝一点")
    ref.run_refinement(job.settings, job.conn, "r3")
    record = get_refinement(job.conn, "r3")
    assert record["status"] == "error" and "画质检查" in record["error"]
    assert not (job.root / "refinements" / "r3" / "result.jpg").exists()


def test_legacy_audits_still_use_the_recipe_refinement(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", lightroom_enabled=True)
    conn = connect(settings.db_path)
    init_db(conn)
    insert_job(conn, "job", "photo.jpg")
    assert directed.refinement_source(settings, conn, "job", "A", None) is None
    audits = settings.jobs_dir / "job" / "lightroom-audits"
    audits.mkdir(parents=True)
    (audits / "variant_A.json").write_text(json.dumps({"accepted_recipe": {}}), encoding="utf-8")
    assert directed.refinement_source(settings, conn, "job", "A", None) is None
    conn.close()


def _spill_scene(sky: str = "smooth") -> Image.Image:
    """Image 1 for guard_spill: sky over the top 40% (flat .62, or textured), a dark figure, textured ground."""
    rng = np.random.default_rng(7)
    height, width = 120, 180
    plane = np.clip(0.5 + rng.normal(0.0, 0.08, (height, width)), 0.0, 1.0)
    top = int(0.4 * height)
    if sky == "textured":
        plane[:top] = np.clip(0.62 + rng.normal(0.0, 0.08, (top, width)), 0.0, 1.0)
    else:                                       # Flat, or darkening from .62 at the horizon to .52 at the top.
        plane[:top] = 0.62 if sky == "smooth" else np.linspace(0.52, 0.62, top)[:, None]
    plane[int(0.45 * height):int(0.62 * height), int(0.45 * width):int(0.55 * width)] = 0.2
    return Image.fromarray((np.repeat(plane[..., None], 3, axis=2) * 255).astype(np.uint8))


def _spill_render(image: Image.Image, folder: Path):
    """``render(document)``: each radial x luminance-range mask scales display luma by 2**(exposure*weight)."""
    plane = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    engine = np.interp(plane, directed._LUMA_DISPLAY, directed._LUMA_ENGINE)
    height, width = plane.shape
    yy, xx = np.mgrid[0:height, 0:width]
    count = [0]

    def render(document):
        out = plane.copy()
        for mask in document.get("masks") or []:
            weight = np.ones_like(plane)
            for part in mask["components"]:
                shape = part["shape"]
                if shape["kind"] == "radial":            # LightCraft measures both radii on the long edge.
                    long = max(width, height)
                    r = np.hypot((xx - shape["center"]["x"] * width) / (shape["rx"] * long),
                                 (yy - shape["center"]["y"] * height) / (shape["ry"] * long))
                    weight = weight * np.clip((1.0 - r) / 0.3, 0.0, 1.0)
                elif shape["kind"] == "luminanceRange":
                    over = (engine - shape["hi"]) / max(shape["hi_feather"], 1e-4)
                    weight = weight * np.clip(1.0 - over, 0.0, 1.0) * (engine >= shape["lo"])
            out = out * 2.0 ** (float(mask["adjust"].get("exposure", 0.0)) * weight)
        count[0] += 1
        path = folder / f"{count[0]:02d}.png"
        Image.fromarray((np.clip(out, 0.0, 1.0) * 255).round().astype(np.uint8)).save(path)
        return path

    return render


def _spill_plan(ry: float = 0.3, **target) -> dict:
    plan, _ = directed.sanitize_plan({
        "title": "t", "tone": {"black": 0.03, "mid": 0.5, "white": 0.95},
        "masks": [{"name": "人物提亮", "purpose": "提亮人物", "adjust": {"exposure": 0.2},
                   "shape": {"kind": "radial", "center": {"x": 0.5, "y": 0.53}, "rx": 0.12, "ry": ry},
                   "limit": {"kind": "luminance", "lo": 0.0, "hi": 0.7, "feather": 0.05},
                   "target": {"box": [0.46, 0.47, 0.54, 0.6], **(target or {"median": 0.4})}}]}, slot="A")
    return plan


@pytest.mark.parametrize("sky", ["smooth", "gradient"])
def test_guard_spill_brings_the_limit_under_smooth_sky_its_radial_lifts_more_than_the_subject(tmp_path, sky):
    image = _spill_scene(sky)
    plan = _spill_plan()
    out, notes, records = directed.guard_spill(plan, "jpeg", _spill_render(image, tmp_path), image)
    assert plan["masks"][0]["limit"]["hi"] == 0.7                     # The plan passed in is left as written.
    limit = out["masks"][0]["limit"]
    (record,) = records
    assert limit["hi"] <= record["edge"] - limit["feather"] + 0.005 and limit["hi"] > 0.24
    assert len(record["trials"]) == (1 if sky == "smooth" else 2)     # A darker sky edge takes a feather more.
    assert record["area"] >= directed.SPILL_AREA and record["kept"] and record["after_faint"] < directed.SPILL_AREA
    assert record["after_box_rise"] > 0.05                             # The figure still lifts.
    assert len(notes) == 1 and "人物提亮" in notes[0] and "光斑" in notes[0]


@pytest.mark.parametrize("case", ["short", "textured", "no_lift"])
def test_guard_spill_leaves_masks_that_lift_no_smooth_brighter_ground(tmp_path, case):
    image = _spill_scene("textured" if case == "textured" else "smooth")
    plan = _spill_plan(ry=0.1 if case == "short" else 0.3, **({"hue": 30.0, "chroma": 0.05} if case == "no_lift" else {}))
    out, notes, records = directed.guard_spill(plan, "jpeg", _spill_render(image, tmp_path), image)
    assert out["masks"][0]["limit"] == plan["masks"][0]["limit"] and not notes
    assert (records == []) == (case == "no_lift")
    assert all("new_hi" not in record for record in records)


@pytest.mark.parametrize("aspect, rx, ry", [(1.5, 0.12, 0.2), (2 / 3, 0.08, 0.3), (None, 0.12, 0.3)])
def test_compile_document_sizes_radials_on_the_long_edge_as_the_engine_does(aspect, rx, ry):
    plan, _ = directed.sanitize_plan({
        "title": "t", "tone": {"black": 0.03, "mid": 0.5, "white": 0.95},
        "masks": [{"name": "人物", "purpose": "提亮", "adjust": {"exposure": 0.3},
                   "shape": {"kind": "radial", "center": {"x": 0.5, "y": 0.5}, "rx": 0.12, "ry": 0.3},
                   "limit": {"kind": "luminance", "lo": 0.0, "hi": 0.7},
                   "exclude": {"kind": "radial", "center": {"x": 0.5, "y": 0.2}, "rx": 0.12, "ry": 0.3}}]}, slot="A")
    document, _ = directed.compile_document(plan, "jpeg", aspect=aspect)
    shapes = [part["shape"] for part in document["masks"][0]["components"] if part["shape"]["kind"] == "radial"]
    assert len(shapes) == 2 and all((shape["rx"], shape["ry"]) == pytest.approx((rx, ry)) for shape in shapes)
    assert plan["masks"][0]["shape"]["ry"] == 0.3                    # The plan keeps the model's units.
