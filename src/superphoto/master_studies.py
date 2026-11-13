"""Per-photograph observations and conditional lessons from local study images.

An observed finished image cannot reveal its original exposure, actual editing
recipe, or lighting setup. Lessons describe visible relationships to adapt.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Principle = Literal["subject_separation", "atmospheric_depth", "negative_space", "color_hierarchy",
    "temperature_layers", "highlight_rolloff", "black_depth", "material_detail", "skin_integrity",
    "neutral_material", "monochrome_structure", "graphic_geometry", "environment_story"]

# A model's study prose is evidence to inspect, never an executable command.
# These reviewed operations retain the individual image's selected principles
# while preventing unsupported masking or destructive slider suggestions.
PRINCIPLE_GUIDANCE = {
    "subject_separation": (
        "当前主体与环境缺少清楚的明暗或配色关系，且主体区域可可靠选择",
        "先利用已有光线与颜色区分主次；只在需要的可靠蒙版上小幅修光或收敛竞争色，保留明确的主体和背景编辑意图",
        "不一律提亮主体、压暗背景；剪影和承担叙事的环境应保持原有关系",
        "主体是否更容易被看见，同时保留原来的光线方向与环境信息？"),
    "atmospheric_depth": (
        "当前画面确实存在由雾气或空气散射形成的远近低反差关系",
        "控制全局去雾与清晰度，保留远景较低的反差和较柔的颜色；局部只使用已有可靠的主体、天空或背景区域",
        "不把浅景深模糊当成雾，不压暗远景模拟空气透视，不声称已经生成精细远近蒙版",
        "远景是否仍比近景柔和，空气层次是否自然，是否出现边缘光晕？"),
    "negative_space": (
        "当前主体周围的空白、天空或低细节区域承担视觉呼吸与构图作用",
        "保留留白的亮度与简洁；必要时适度收敛竞争色，避免给低细节区域增加无意义的纹理和局部对比",
        "不照搬参考图裁切，也不把留白当成必须填满或提亮的缺陷",
        "视线是否更集中，留白是否仍有空间感，边缘是否增加了抢眼元素？"),
    "color_hierarchy": (
        "当前画面存在可辨认的主色、辅助色与竞争色",
        "先降低与主题无关的竞争色，再小幅调整主色的饱和度或明度；共享同一种颜色的主背景不能只靠全局 HSL 分离",
        "不将灰色当成独立色相，不把所有颜色同时加浓，不复制参考图的固定配色",
        "重点色是否更清楚，辅助色是否仍有层次，同色的其他区域是否被误伤？"),
    "temperature_layers": (
        "当前原片已经存在不同光源或亮度区域之间的冷暖关系",
        "保留现场冷暖方向，按现有光线在相应亮度段小幅使用色轮；用自然的色彩过渡区分亮部与阴影",
        "不以全局白平衡代替局部光色，不强制所有照片变成冷阴影暖高光",
        "冷暖是否服务原有光线，肤色与中性材质是否出现不必要的染色？"),
    "highlight_rolloff": (
        "当前亮部过渡生硬或本次处理正在增加高光裁切",
        "小幅整理高光与白色亮度，必要时柔化曲线肩部，保留反光的光泽和自然强度",
        "不把所有高光压成灰色；已经裁切的 JPEG 细节不能靠后期凭空恢复",
        "亮部是否有更自然的过渡，同时仍保留光源、反光和材质的亮度感？"),
    "black_depth": (
        "当前暗部承担画面结构、剪影或夜色氛围，同时仍需保留必要细节",
        "保留黑位密度，区别结构性深黑与需要看清的阴影；只修正被本次调整额外压死的区域",
        "不统一抬黑制造灰雾，也不为了风格差异进一步压死原本已裁切的暗部",
        "暗部是否仍有密度与层次，画面是否因过度提阴影失去原来的光比？"),
    "material_detail": (
        "当前材质纹理确实是画面重点，且原片有足够细节与信噪比",
        "根据原片清晰度和噪声小幅整理纹理与锐化；材质细节应服从真实光线及远近关系",
        "不全图加清晰度，不把噪声或压缩边缘当成质感，不声称从模糊图恢复不存在的细节",
        "重要材质是否更可读，同时没有噪声放大、描边或远景过硬？"),
    "skin_integrity": (
        "当前确实有人像，且需要维护自然肤色和皮肤纹理",
        "仅在肤色已经偏色或过饱和时小幅整理红橙色与局部光线，保留自然血色和皮肤纹理",
        "不把肤色饱和度降到最低，不磨成无纹理皮肤，不用全图橙色区域代替可靠的人物判断",
        "肤色是否自然、明暗过渡是否柔和，是否丢失血色或真实纹理？"),
    "neutral_material": (
        "当前建筑、白墙或产品的中性材质需要保持可信的颜色",
        "优先维护中性材质的自然色温与明暗面，让其和画面的点缀色形成清楚关系",
        "不将白墙染成参考图的蓝色或黄色，不用中性灰的所谓色相滑块进行无效处理",
        "中性材质是否仍可信，受光面与阴影面是否保留立体关系？"),
    "monochrome_structure": (
        "当前已是黑白照片，或用户明确要求黑白处理；彩色图仅可借鉴其影调思想",
        "观察大块明暗、轮廓与中间调的分工，保留重要层次和黑白之间的过渡",
        "不因参考图是黑白就自动去掉用户照片的颜色",
        "主要轮廓与中间调是否各有作用，有没有只剩死黑和纯白？"),
    "graphic_geometry": (
        "当前建筑或抽象形态的线条、面与重复节奏承担构图作用",
        "整理相邻亮暗面的层次与杂色，保持有意的透视和线条关系；不要用过强锐化替代几何秩序",
        "不因参考图对称就强制裁切或拉正，也不把左右有物体称为镜像对称",
        "线条节奏与材质是否更明确，同时保留原构图和自然透视？"),
    "environment_story": (
        "当前主体与周围环境共同承担故事和现场感",
        "保留关键环境线索的可读性，优先收敛无关的抢眼亮点和杂色，让主体与环境共同组织视线",
        "不一律模糊或压黑背景，不自动移除有叙事作用的人物和结构",
        "主体与环境的联系是否更清楚，原有故事线索是否仍然可读？"),
}


def workflow_lessons(analysis: dict) -> list[dict]:
    result = []
    seen = set()
    tags = set(analysis.get("scene_tags", []))
    for proposed in analysis.get("lessons", []):
        principle = proposed.get("principle")
        if principle not in PRINCIPLE_GUIDANCE or principle in seen:
            continue
        # Depth through glass, framing or shallow focus is not atmospheric
        # scattering. A lesson must be grounded in the source image too.
        if principle == "atmospheric_depth" and "scene_tags" in analysis:
            observed = " ".join([analysis.get("caption", ""),
                                 *(s.get("evidence", "") for s in analysis.get("strengths", []))])
            if "fog" not in tags or not re.search(r"雾|空气透视|大气透视|散射|远景.{0,8}朦胧|fog|mist|haze|atmospheric", observed, re.I):
                continue
        if principle == "skin_integrity" and "scene_tags" in analysis and "people" not in tags:
            continue
        if analysis.get("palette") == "monochrome" and principle in {"color_hierarchy", "temperature_layers", "skin_integrity"}:
            continue
        seen.add(principle)
        when, action, avoid, review = PRINCIPLE_GUIDANCE[principle]
        result.append({"principle": principle, "applies_when": when, "action": action, "avoid": avoid, "review": review})
    return result


class Strength(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: Literal["composition", "light", "tone", "color", "depth", "detail", "story"]
    evidence: str = Field(min_length=12, max_length=160)
    benefit: str = Field(min_length=8, max_length=120)


class StudyLesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principle: Principle
    applies_when: str = Field(min_length=8, max_length=140)
    action: str = Field(min_length=12, max_length=180)
    avoid: str = Field(min_length=8, max_length=140)
    review: str = Field(min_length=8, max_length=140)


class StudyAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_kind: Literal["photograph", "behind_scenes", "portrait_reference", "layout", "nonphotographic"]
    caption: str = Field(min_length=12, max_length=160)
    genre: Literal["architecture", "landscape", "portrait", "street", "nature", "night", "product", "abstract", "other"]
    scene_tags: list[Literal["people", "water", "vegetation", "sky", "architecture", "fog", "snow", "artificial_light", "studio", "interior", "animals"]] = Field(max_length=6)
    light: Literal["hard_directional", "soft_diffuse", "backlit", "mixed", "flat", "unknown"]
    tone: Literal["low_key", "high_key", "balanced"]
    palette: Literal["monochrome", "warm", "cool", "neutral", "mixed"]
    strengths: list[Strength] = Field(min_length=3, max_length=4)
    lessons: list[StudyLesson] = Field(min_length=2, max_length=3)
    capture_dependencies: list[str] = Field(min_length=1, max_length=3)
    uncertainty: str = Field(min_length=8, max_length=160)

    @field_validator("scene_tags")
    @classmethod
    def unique_tags(cls, tags):
        if len(tags) != len(set(tags)):
            raise ValueError("Scene tags must be distinct visible elements")
        return tags


@lru_cache(maxsize=1)
def load_studies() -> dict:
    path = Path(__file__).with_name("master_studies.json")
    if not path.is_file():
        return {"records": [], "coverage": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"records": [], "coverage": {"status": "unavailable"}}


def scene_description(critique, technical) -> dict:
    from .finishing_knowledge import method_ids
    primary = re.split(r"及|与|对岸|背景|background| with | and |[,，;；。]",
                       critique.subject_note.lower(), maxsplit=1)[0]
    # The critic often describes air perspective in its preservation goal,
    # while the short summary only mentions buildings and water. Read the
    # original intent too, but never feed our generated review checks back
    # into scene detection.
    words = " ".join([critique.subject_note, critique.summary, critique.intent.goal,
                      *critique.intent.preserve]).lower()
    genre = critique.genre
    if "graphic" in method_ids(critique, technical):
        genre = "architecture"
    tags = set()
    terms = {"people": r"人像|人物|行人|人群|portrait|person|people", "water": r"河|湖|海|水面|船|river|lake|sea|boat|water",
             "vegetation": r"森林|树林|绿叶|植被|forest|tree|foliage", "sky": r"天空|蓝天|云层|sky|cloud",
             "architecture": r"建筑|楼宇|教堂|大楼|城市|building|architecture|city",
             "fog": r"雾|朦胧|空气透视|大气透视|haze|mist|fog|atmospheric perspective", "snow": r"雪|冰川|snow|glacier",
             "artificial_light": r"夜景|路灯|霓虹|night|neon", "studio": r"棚拍|摄影棚|studio",
             "interior": r"室内|室内空间|interior|indoor", "animals": r"动物|鸟|鲸|animal|bird|whale"}
    for tag, pattern in terms.items():
        if re.search(pattern, words):
            tags.add(tag)
    if re.search(r"无(?:明显)?雾|没有(?:明显)?雾|no (?:visible )?(?:fog|haze)|without (?:fog|haze)", words):
        tags.discard("fog")
    if genre == "portrait":
        tags.add("people")
    if genre == "architecture":
        tags.add("architecture")
    if genre == "indoor":
        tags.add("interior")
        # The critic uses "indoor" for both portraits and interior spaces;
        # the study archive describes their photographic subject instead.
        genre = "portrait" if "people" in tags else "architecture"
        if genre == "architecture":
            tags.add("architecture")
    if technical.green_fraction > .25 and genre in {"landscape", "nature"}:
        tags.add("vegetation")
    return {"genre": genre, "primary": primary, "tags": sorted(tags)}


def select_studies(critique, technical, *, limit: int = 3) -> list[dict]:
    scene = scene_description(critique, technical)
    target_tags = set(scene["tags"])
    families = [{"landscape", "nature"}, {"architecture", "street"}]
    scored = []
    for record in load_studies().get("records", []):
        analysis = record.get("analysis", {})
        if record.get("schema_version") != 3 or record.get("status") != "complete" or analysis.get("image_kind") != "photograph":
            continue
        sources = record.get("sources", [])
        if not sources or all(s.get("role") == "jury_or_program" for s in sources):
            continue
        genre = analysis.get("genre")
        same_genre = genre == scene["genre"]
        compatible = same_genre or any({genre, scene["genre"]} <= family for family in families)
        tags = set(analysis.get("scene_tags", []))
        depth_transfer = not compatible and "fog" in target_tags & tags
        if not compatible and not depth_transfer and scene["genre"] not in {"general", "other"}:
            continue
        overlap = len(target_tags & tags) / max(1, len(target_tags | tags))
        if not same_genre and not (target_tags & tags):
            continue
        features = record.get("features", {})
        distance = sum(((float(features.get(key, getattr(technical, key))) - getattr(technical, key)) / scale) ** 2
                       for key, scale in {"mean_luma": .22, "green_fraction": .28, "aqua_fraction": .28,
                                          "dynamic_spread": .3}.items())
        score = (.52 if same_genre else .30) + .28 * overlap + .20 * math.exp(-distance / 2)
        if "interior" in tags and "interior" not in target_tags:
            score -= .12
        if "people" in tags and "people" not in target_tags:
            score -= .10
        caption = analysis.get("caption", "")
        source_night = genre == "night" or bool(re.search(r"夜景|夜空|星空|霓虹|night|starry", caption, re.I))
        if source_night != (scene["genre"] == "night" or "artificial_light" in target_tags):
            score -= .18
        if score < .52:
            continue
        lessons = []
        for lesson in workflow_lessons(analysis):
            principle = lesson["principle"]
            if depth_transfer and principle != "atmospheric_depth":
                continue
            if principle == "atmospheric_depth" and "fog" not in target_tags:
                continue
            if principle == "skin_integrity" and scene["genre"] != "portrait":
                continue
            if principle == "neutral_material" and scene["genre"] not in {"architecture", "product"}:
                continue
            if principle == "graphic_geometry" and "architecture" not in target_tags and scene["genre"] not in {"architecture", "abstract", "product"}:
                continue
            if principle == "monochrome_structure":
                # Tone structure may transfer, conversion to B&W is never an
                # inferred user request. Keep it out of automatic color design.
                continue
            lessons.append(lesson)
        if lessons:
            scored.append((score, record, lessons))
    scored.sort(key=lambda row: (-row[0], row[1]["id"]))
    # One relevant atmosphere lesson is more useful for a misty city than
    # three dry architectural close-ups. Only the depth principle crosses
    # genres here, never the source's lighting or palette.
    if "fog" in target_tags:
        depth = next((row for row in scored if any(l["principle"] == "atmospheric_depth" for l in row[2])), None)
        if depth:
            scored.remove(depth)
            scored.insert(0, depth)
    selected, authors = [], set()
    for score, record, lessons in scored:
        source = next((s for s in record["sources"] if s.get("role") != "jury_or_program"), record["sources"][0])
        author = source.get("photographer")
        # The two supplied archives differ in capitalization, accents and a
        # legacy "master " prefix. These are still the same photographer.
        normalized = unicodedata.normalize("NFKD", author or "未标注").casefold()
        normalized = re.sub(r"^master\s+", "", normalized)
        author_key = "".join(ch for ch in normalized if ch.isalnum())
        if author_key in authors:
            continue
        authors.add(author_key)
        selected.append({"id": record["id"], "photographer": author, "year": source.get("year"),
                         "source_image": source["relative_path"], "source_page": source.get("page"),
                         "match_score": round(score, 3), "caption": record["analysis"]["caption"],
                         "observed_strengths": record["analysis"]["strengths"][:2], "lessons": lessons[:2],
                         "capture_dependencies": record["analysis"]["capture_dependencies"],
                         "uncertainty": record["analysis"]["uncertainty"]})
        if len(selected) >= limit:
            break
    return selected


def study_context(critique, technical) -> dict:
    selected = select_studies(critique, technical)
    checks = []
    for item in selected:
        for lesson in item["lessons"]:
            if lesson["review"] not in checks:
                checks.append(lesson["review"])
    return {"scene": scene_description(critique, technical), "references": selected, "review_checks": checks[:4],
            "rule": "借鉴可见的光色关系；必须重新观察用户照片是否满足适用条件，不复制大师成片的曝光、色相或构图。"}


def apply_study_guards(recipe, critique, technical):
    """Conservative executable protections, activated only by matched lessons."""
    selected = select_studies(critique, technical)
    principles = {lesson["principle"] for item in selected for lesson in item["lessons"]}
    result = recipe.model_copy(deep=True)
    if "atmospheric_depth" in principles and "fog" in scene_description(critique, technical)["tags"]:
        result.dehaze = min(result.dehaze, .02)
        result.clarity = min(result.clarity, .08)
    if "highlight_rolloff" in principles and technical.highlight_clip > .02:
        result.highlights = min(result.highlights, -.08)
        result.whites = min(result.whites, 0)
    return result
