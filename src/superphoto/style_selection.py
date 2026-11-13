"""Rank executable presets by this photograph, before designing its edits.

Practice-image distances are evidence of compatibility, not aesthetic scores.
Finished-master studies remain references: their inferred histogram recipes
are not original author presets and do not compete with imported XMPs here.
"""
from __future__ import annotations

import re

import numpy as np

from .look_routes import route_similarity, route_when
from .models import Critique, StyleMatch, TechnicalReport


FAMILIES = {
    "faithful": "natural", "celebrity_rescue": "natural",
    "filmic_portrait_warm": "warm_portrait", "kdrama_tears": "warm_portrait",
    "romance_gold": "gold", "cinematic_teal_orange": "cinema", "jujutsu": "cinema",
    "celebrity_tim_northey": "clean", "editorial_crisp": "editorial",
    "landscape_filmic": "landscape", "celebrity_extreme_dehaze": "depth",
    "celebrity_forest_michael_kagerer": "olive", "celebrity_german_maurice": "vintage",
    "celebrity_portra_hamada": "pastel", "natsume": "pastel", "five_cm": "pastel",
    "moody_lowkey": "lowkey", "celebrity_low_sat_shinagawa": "lowkey",
    "celebrity_twilight_kirinoma": "twilight", "tokyo_ghoul": "twilight",
    "skam": "nordic", "zhou_chu": "muted_city", "jp_drama": "airy_city",
    "dune": "desert", "downton": "period", "shameless": "warm_city",
    "resident_evil": "bleak", "wong_kar_wai": "warm_night",
    "unnatural": "cool_indoor", "hanabi": "fireworks",
}


def scene_context(critique: Critique, t: TechnicalReport) -> dict[str, bool]:
    # Use the main subject for subject-specific styles. Incidental buildings
    # behind a boat do not turn the picture into architectural photography.
    subject = critique.subject_note.lower()
    primary = re.split(r"及|与|对岸|背景|background| with | and |[,，;；。]", subject, maxsplit=1)[0]
    has = lambda words: any(word in primary for word in words)
    night = critique.genre == "night" or has(("夜景", "夜晚", "夜街", "night", "霓虹"))
    portrait = critique.genre in {"portrait", "family", "fashion"} or has(("人像", "肖像", "portrait", "面部", "半身", "人物", "登山者", "hiker", "person", "女孩", "男孩"))
    architecture = critique.genre == "architecture" or has(("建筑", "教堂", "楼宇", "architecture", "church", "monument", "纪念碑"))
    landscape = critique.genre in {"landscape", "nature"}
    return {
        "night": night, "portrait": portrait, "architecture": architecture,
        "landscape": landscape,
        "urban": critique.genre in {"street", "urban", "documentary"} or architecture,
        "indoor": critique.genre == "indoor" or has(("室内", "interior", "indoor")),
        "forest": landscape and (t.green_fraction > .10 or any(w in subject for w in
            ("森林", "绿叶", "荷叶", "绿植", "绿色", "草坡", "草地", "forest", "foliage", "grass"))),
        "desert": has(("沙漠", "沙丘", "干土", "desert", "dune")) or
                  (landscape and not architecture and not portrait and not subject
                   and t.orange_fraction > .35 and t.green_fraction < .05 and t.aqua_fraction < .08),
        "fireworks": has(("烟花", "烟火", "firework")),
        "twilight": night or has(("暮色", "黄昏", "黎明", "蓝调", "dawn", "dusk", "twilight")),
        "pastel": portrait or t.magenta_fraction > .035,
    }


def incompatibility(look_id: str, c: dict, t: TechnicalReport, critique: Critique) -> str:
    if look_id == "dune" and not c["desert"]:
        return "缺少沙漠或干燥暖土色场景；不能仅凭暖色像素套沙丘。"
    if look_id == "hanabi" and not c["fireworks"]:
        return "未识别到烟火，夜景不能一律套烟火预设。"
    if look_id in {"tokyo_ghoul", "wong_kar_wai"} and not (c["night"] or c["indoor"]):
        return "场景不是夜街或低照度室内。"
    if look_id in {"filmic_portrait_warm", "kdrama_tears", "celebrity_portra_hamada"} and not c["portrait"]:
        return "主体不是人像；暖色岩石或建筑不能当作肤色。"
    if look_id == "celebrity_forest_michael_kagerer" and not c["forest"]:
        return "没有足够的植被场景支持森林配色。"
    if look_id == "celebrity_extreme_dehaze" and (not c["landscape"] or t.std_luma > .13 or critique.intent.preserve_atmosphere):
        return "远景并非需要强整理的灰片，或明确需要保留空气感。"
    if look_id == "celebrity_twilight_kirinoma" and not c["twilight"]:
        return "缺少暮色或夜间光线依据。"
    if look_id in {"natsume", "five_cm"} and not c["pastel"]:
        return "缺少清淡人像或粉色场景的依据。"
    if look_id == "unnatural" and not c["indoor"]:
        return "冷白室内预设与当前场景不符。"
    if look_id in {"jp_drama", "shameless", "zhou_chu", "jujutsu", "resident_evil"} and not (c["urban"] or c["indoor"]):
        return "城市叙事配色不适合当前自然主体。"
    if look_id == "romance_gold" and not ((c["indoor"] or c["night"]) and t.temperature_bias > .06):
        return "缺少暖灯光，避免把日光场景整体染金。"
    if look_id == "downton" and not (c["architecture"] or c["indoor"] or critique.genre == "portrait"):
        return "缺少建筑、庭院或室内人像的空间关系。"
    if look_id == "skam" and not (c["urban"] or c["portrait"] or (c["landscape"] and not c["night"] and t.top_third_blue > .15)):
        return "缺少日光人物、街头或灰蓝天空下的开阔外景。"
    if look_id == "cinematic_teal_orange" and not (c["urban"] or c["portrait"] or c["night"] or
            (t.orange_fraction > .08 and t.aqua_fraction + t.top_third_blue > .18)):
        return "缺少城市、人物或已有冷暖色块的关系，不强造青橙对冲。"
    if look_id == "editorial_crisp" and not (c["portrait"] or c["architecture"] or critique.genre in {"product", "fashion"}):
        return "当前主体不需要杂志式硬朗材质。"
    if look_id in {"moody_lowkey", "celebrity_low_sat_shinagawa"} and t.mean_luma > .26:
        return "原图中间调明亮，不为风格强行压成暗调。"
    return ""


def rank_presets(t: TechnicalReport, critique: Critique, *, feedback: dict | None = None,
                 preferred: list[str] | None = None) -> tuple[list[StyleMatch], dict[str, str]]:
    from .celebrity_styles import _style_scores
    from .recipes import LOOKS

    context = scene_context(critique, t)
    celebrity = {s.look_id: s for s in _style_scores(t, critique)}
    recommended = set(critique.looks) if critique.source != "local" else set()
    ranked, excluded = [], {}
    for lid, look in LOOKS.items():
        if lid in {"faithful", "celebrity_rescue"}:
            continue  # Correction is shared beneath every look, not a compulsory slot.
        veto = incompatibility(lid, context, t, critique)
        if veto:
            excluded[lid] = veto
            continue
        # The old route veto treats cyan pixels as water and warm pixels as
        # skin. Semantic exclusions above own eligibility; statistics rank it.
        measured = route_similarity(lid, t, apply_veto=False)
        genre_fit = critique.genre in look.genre_tags
        if context["architecture"]:
            genre_fit = "architecture" in look.genre_tags or lid in {"editorial_crisp", "celebrity_tim_northey"}
        if lid in celebrity:
            fit = celebrity[lid].score
        else:
            fit = .20 + .60 * (measured if measured is not None else .35) + (.12 if genre_fit else -.06)
        # Context has priority over a similar histogram or a generic VLM name.
        semantic = 0.0
        if (lid == "dune" and context["desert"]) or (lid == "hanabi" and context["fireworks"]):
            semantic = .25
        elif lid == "downton" and context["architecture"]:
            semantic = .15
        elif lid == "landscape_filmic" and context["landscape"] and not context["architecture"] and not context["desert"]:
            semantic = .12
        elif lid == "celebrity_portra_hamada" and context["portrait"]:
            semantic = .12
        vote = (feedback or {}).get(lid, {})
        preference = float(np.clip(.07 * vote.get("like", 0) - .20 * vote.get("dislike", 0), -.45, .15))
        # Legacy API callers can still supply explicit preferences; pipeline
        # uses scene-specific votes instead of global historical wins.
        if feedback is None and lid in (preferred or []):
            preference = .04
        vision = .08 if lid in recommended else 0.0
        score = float(np.clip(fit + semantic + vision + preference, 0, 1))
        when = route_when(lid) or (celebrity[lid].reason if lid in celebrity else look.description)
        reason = f"{critique.subject_note or critique.genre}；{when}"
        if vision:
            reason += " 读图建议与场景条件相符。"
        if preference:
            reason += " 已参考相似画面的满意度反馈。"
        ranked.append(StyleMatch(look_id=lid, title=look.title, match_score=round(score, 4),
            family=FAMILIES.get(lid, lid), source="study_recipe" if lid in celebrity else look.source,
            reason=reason, evidence={"scene_fit": round(fit, 4), "semantic": semantic,
                "vision": vision, "preference": round(preference, 4),
                **({"practice_similarity": round(measured, 4)} if measured is not None else {})}))
    ranked.sort(key=lambda item: (-item.match_score, item.look_id))
    return ranked, excluded


def select_diverse_presets(ranked: list[StyleMatch], critique: Critique, t: TechnicalReport) -> list[StyleMatch]:
    """Prefer compatible, genuinely different recipes; never randomize by job ID."""
    from .directions import _signature, _signature_distance, preset_direction
    from .preference import honor_locked_controls
    from .workflow import apply_ethics, ethics_regime_for, guard_hues

    selected, signatures = [], []
    remaining = list(ranked)
    while remaining and len(selected) < 3:
        scored = []
        for match in remaining:
            recipe = preset_direction("ABC"[len(selected)], match.look_id, critique, t).recipe
            ethics = ethics_regime_for(critique.genre)
            recipe = honor_locked_controls(apply_ethics(guard_hues(recipe, t, ethics), ethics), critique.intent)
            sig = _signature(recipe, full=True)
            distance = min((_signature_distance(sig, old) for old in signatures), default=1.0)
            # Do not force a poor scene match solely for a stronger appearance.
            penalty = (.12 if any(m.family == match.family for m in selected) else 0)
            penalty += max(0, .035 - distance) * 3
            scored.append((match.match_score - penalty, match, sig))
        _, choice, signature = max(scored, key=lambda item: item[0])
        selected.append(choice)
        signatures.append(signature)
        remaining.remove(choice)
    return selected


def preset_legend() -> str:
    from .recipes import LOOKS
    return "\n".join(f"- {lid}: {look.description} 使用条件：{route_when(lid) or '根据主体、光线和原有色彩判断'}"
                     for lid, look in LOOKS.items() if lid not in {"faithful", "celebrity_rescue"})
