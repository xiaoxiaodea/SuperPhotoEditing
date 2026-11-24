"""Explicit photo votes influence compatible finishes for similar source scenes."""
from __future__ import annotations

import json
import math
import random
import re
import sqlite3

from .finishing_knowledge import METHODS, method_ids
from .models import Critique, TechnicalReport


def scene_profile(critique: Critique, technical: TechnicalReport) -> dict:
    primary = re.split(r"及|与|对岸|背景|background| with | and |[,，;；。]",
                       critique.subject_note.lower(), maxsplit=1)[0]
    defaults = method_ids(critique, technical)
    family = critique.genre
    if family != "night":
        if "graphic" in defaults:
            family = "architecture"
        elif "forest" in defaults:
            family = "vegetation"
        elif any(word in primary for word in ("船", "河", "湖", "海", "boat", "river", "water", "lake", "sea")):
            family = "water"
    return {"version": 1, "family": family,
            "features": {key: float(getattr(technical, key)) for key in (
                "mean_luma", "temperature_bias", "saturation_mean", "green_fraction", "aqua_fraction", "skin_fraction")}}


def scene_similarity(a: dict, b: dict) -> float:
    if a.get("version") != b.get("version") or a.get("family") != b.get("family"):
        return 0.0
    scales = {"mean_luma": .18, "temperature_bias": .16, "saturation_mean": .22,
              "green_fraction": .25, "aqua_fraction": .25, "skin_fraction": .20}
    try:
        distance = sum(((a["features"][k] - b["features"][k]) / scale) ** 2 for k, scale in scales.items())
        similarity = math.exp(-distance / 2)
        return similarity if similarity >= .35 else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def variant_method(variant: dict) -> str:
    recipe = json.loads(variant["recipe_json"])
    method = recipe.get("style_workflow", {}).get("method_id", "")
    if method in METHODS or method.startswith("direction:"):    # direction: a directed version (experience.py)
        return method
    from .recipes import LOOKS
    if method.startswith("preset:") and method.removeprefix("preset:") in LOOKS:
        return method
    # Previously delivered modern candidates used these exact canonical titles.
    # Never infer a finish from its A/B/C position or the obsolete LUT reference.
    return next((key for key, item in METHODS.items() if variant["title"] == item["title"]), "")


def job_profile(job: dict) -> dict:
    try:
        return scene_profile(Critique.model_validate_json(job["critique_json"]),
                             TechnicalReport.model_validate_json(job["technical_json"]))
    except (ValueError, TypeError):
        return {}


def preference_samples(conn: sqlite3.Connection) -> list[dict]:
    samples = [{"method_id": r["method_id"], "verdict": r["verdict"], "scene": json.loads(r["scene_json"])}
               for r in conn.execute("SELECT * FROM ratings WHERE verdict != 0")]
    # One latest legacy reaction per delivered photo, unless explicitly replaced
    # or undone. Low expert scores must never become positive selections.
    rows = conn.execute("""SELECT f.*, j.technical_json, j.critique_json, v.title, v.recipe_json
        FROM feedback f JOIN jobs j ON j.id=f.job_id
        JOIN variants v ON v.job_id=f.job_id AND v.slot=f.chosen_slot
        WHERE f.rowid IN (SELECT MAX(rowid) FROM feedback GROUP BY job_id, chosen_slot)
        AND NOT EXISTS (SELECT 1 FROM ratings r WHERE r.job_id=f.job_id AND r.slot=f.chosen_slot)""")
    for row in rows:
        item = dict(row)
        method = variant_method(item)
        if method:
            samples.append({"method_id": method, "scene": job_profile(item),
                            "verdict": 1 if item["style_score"] >= 4 and item["technical_score"] >= 3 else -1})
    return samples


def select_methods(conn: sqlite3.Connection, critique: Critique, technical: TechnicalReport, *, seed: str) -> dict:
    defaults = method_ids(critique, technical)
    profile = scene_profile(critique, technical)
    pool = list(dict.fromkeys([*defaults, "natural", "sculpted", "soft_warm", "soft_neutral"]))
    if profile["family"] in {"architecture", "water", "landscape", "general"}:
        pool = list(dict.fromkeys([*pool, "graphic"]))
    evidence = {key: {"like": 0.0, "dislike": 0.0} for key in pool}
    matched = 0
    for sample in preference_samples(conn):
        method = sample["method_id"]
        similarity = scene_similarity(profile, sample["scene"])
        if method not in evidence or similarity <= 0:
            continue
        matched += 1
        evidence[method]["like" if sample["verdict"] > 0 else "dislike"] += similarity
    weights = {key: (3.0 if key in defaults else 1.0) * math.exp(max(-3, min(2,
               .8 * values["like"] - 1.6 * values["dislike"]))) for key, values in evidence.items()}
    selected = list(defaults)
    if matched:
        rng = random.Random(seed)
        selected = sorted(pool, key=lambda key: -math.log(max(rng.random(), 1e-12)) / weights[key])[:3]
        # Keep comparison order familiar; frequency is determined by sampling,
        # never by fixed mandatory slots that would make a dislike ineffective.
        order = {key: i for i, key in enumerate(("natural", "graphic", "forest", "sculpted", "soft_neutral", "soft_warm", "documentary"))}
        selected.sort(key=order.__getitem__)
    return {"version": 1, "scene": profile, "matched_feedback": matched, "selected": selected,
            "weights": weights, "evidence": evidence}


def preset_feedback(conn: sqlite3.Connection, critique: Critique, technical: TechnicalReport) -> dict:
    """Explicit votes apply to the preset actually delivered, in similar scenes."""
    from .recipes import LOOKS
    from .style_selection import FAMILIES
    profile = scene_profile(critique, technical)
    evidence = {lid: {"like": 0.0, "dislike": 0.0} for lid in LOOKS}
    legacy_families = {"natural": {"natural", "clean"}, "sculpted": {"cinema", "clean"},
        "soft_warm": {"warm_portrait", "pastel"}, "soft_neutral": {"pastel", "nordic"},
        "forest": {"olive"}, "graphic": {"editorial"}, "documentary": {"warm_night", "lowkey"}}
    matched = 0
    for sample in preference_samples(conn):
        similarity = scene_similarity(profile, sample["scene"])
        if not similarity:
            continue
        method = sample["method_id"]
        if method.startswith("preset:"):
            ids = [method.removeprefix("preset:")]
        else:
            # Old methods have no reliable preset identity. Transfer only a
            # weak family preference, never pretend their old look_id ran.
            ids = [lid for lid in evidence if FAMILIES.get(lid) in legacy_families.get(method, set())]
            similarity *= .3
        if any(lid in evidence for lid in ids):
            matched += 1
        for lid in ids:
            if lid in evidence:
                evidence[lid]["like" if sample["verdict"] > 0 else "dislike"] += similarity
    return {"version": 2, "scene": profile, "matched_feedback": matched, "evidence": evidence}


def taste_summary(conn: sqlite3.Connection) -> str:
    samples = preference_samples(conn)
    if not samples:
        return "给每个版本选 pass、留下或最爱，系统会记住你在类似画面下喜欢的方向。"
    favourites = sum(s["verdict"] == 2 for s in samples)
    keeps = sum(s["verdict"] == 1 for s in samples)
    passes = sum(s["verdict"] < 0 for s in samples)
    return f"已记下 {favourites} 个最爱、{keeps} 个留下、{passes} 个 pass；相似画面会参考这些选择来定方向。"
