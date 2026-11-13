"""Curated, executable lessons from the user's local photography material.

The source reports describe finished photographs, not before/after training
pairs. These are scene-conditioned methods and reference hints, never inferred
camera settings or a claim that a model was trained on the archive.
"""
from __future__ import annotations

from functools import lru_cache
import re

from .catalog import load_xmp_looks
from .look_routes import route_similarity, route_when
from .models import ColorGradingControls, Critique, HueShift, Recipe, TechnicalReport, ToneCurve


SOURCES = {
    "workflow": "名人仿色调色.md · 通用工作流与场景风格模块",
    "hasselblad": "ph-main/Hasselblad-REPORT.md · 大气透视、光比与冷暖、单一色彩重点",
    "leica": "ph-main/Leica-REPORT.md · 人与环境、克制配色、保留现场关系",
    "practice": "调色素材 · 已导入的 16 个 XMP、129 张练习图的场景统计",
}

METHODS = {
    "soft_neutral": {
        "title": "柔和淡彩", "goal": "舒展中间调、柔化反差，让颜色清淡而保留现场色温",
        "moves": ["提亮中间调，柔和亮暗交界", "收敛杂色，保留现场冷暖", "保留主体细节与黑位"],
        "lesson": "粉彩研究：柔和来自影调和颜色的组织，不能依靠模糊细节或整幅染色。",
        "sources": ["workflow", "hasselblad"],
    },
    "natural": {
        "title": "自然精修", "goal": "整理明暗和杂色，让主体清楚、现场空气感保留",
        "moves": ["保留现场光色和远近层次", "整理抢眼颜色，让主体更清楚", "保留明亮处的细腻过渡"],
        "lesson": "通用工作流：先判断哪里需要修光；已有好看的雾、逆光、深色应保留。自然精修也要有可见收益。",
        "sources": ["workflow", "hasselblad"],
    },
    "sculpted": {
        "title": "冷调电影", "goal": "通过已有亮暗面和主体环境关系增强立体感，保留现场配色",
        "moves": ["组织亮暗面，突出已有光线", "整理确实抢眼的颜色，保留画面主色", "保留主体质感和亮部光泽"],
        "lesson": "Tim Northey 与哈苏光线研究：先建立中间调和局部光比；颜色确实干扰主体时再整理，蓝青不是默认削弱对象，主体也不是一律把脸提到最亮。",
        "sources": ["workflow", "hasselblad"],
    },
    "soft_warm": {
        "title": "暖调胶片", "goal": "舒展中间调与柔和高光，保留现场暖意和自然主色",
        "moves": ["让中间调舒展，保留黑位密度", "让颜色服务亮暗关系，保留自然主色", "保留中间调和亮部已有的暖意"],
        "lesson": "日系胶片与哈苏粉彩研究：中间调舒展，仅整理当前画面中实际竞争的颜色；保留苍白主色的色深，不强制制造冷暖对冲，不把整幅染黄或全局磨糊。",
        "sources": ["workflow", "hasselblad"],
    },
    "forest": {
        "title": "森林层次", "goal": "把绿叶收成统一的橄榄绿，让光照处与主体更突出",
        "moves": ["统一绿叶颜色，保留叶片层次", "适度压实近处，保留远处空气", "用温暖亮部集中视线"],
        "lesson": "Michael Kagerer 与哈苏雾林：绿转橄榄、蓝绿降饱和、主体保持自然暖色；不把整片森林去雾，不同步压死远景。",
        "sources": ["workflow", "hasselblad"],
    },
    "graphic": {
        "title": "银灰质感", "goal": "把建筑的蓝青杂色收成银灰色，保留暖色光点和材质层次",
        "moves": ["明显收敛蓝青，让建筑呈现银灰质感", "保留暖色光点作为画面重点", "整理亮暗面，保留暗部细节"],
        "lesson": "哈苏建筑研究：材质、几何、阴影都是主体，天空是负空间；不能用统一变暗代替体块关系。避免白墙染蓝或染黄。",
        "sources": ["hasselblad", "workflow"],
    },
    "documentary": {
        "title": "暖色夜景", "goal": "以温暖灯光和克制的环境颜色突出夜间故事，保留夜色密度",
        "moves": ["减少杂色，保留画面中的重点色", "保持夜色和暗部密度", "让暖色线索连接人物与环境"],
        "lesson": "Leica 研究与低饱暗系：系列配色要统一，明暗服务内容；夜晚仍是夜晚，不为出片把它改成白天，不清空环境。",
        "sources": ["leica", "workflow"],
    },
}


def method_ids(critique: Critique, technical: TechnicalReport) -> tuple[str, str, str]:
    # "Boat and buildings on the opposite bank" is a boat photograph. Match
    # the leading subject, not an incidental building anywhere in the scene.
    primary_subject = re.split(r"及|与|对岸|背景|background| with | and |[,，;；。]",
                               critique.subject_note.lower(), maxsplit=1)[0]
    scene = f"{critique.genre} {primary_subject}".lower()
    architecture = any(word in scene for word in ("architecture", "建筑", "教堂", "楼宇"))
    forest = technical.green_fraction > .23 and critique.genre in {"landscape", "nature"}
    # A low-key daylight photograph is not a night scene. Treating it as one
    # made both B and C dark, restrained versions of the same source palette.
    night = critique.genre == "night"
    return "natural", "graphic" if architecture else ("forest" if forest else "sculpted"), "documentary" if night else "soft_warm"


def method_recipe(method: str) -> Recipe:
    """Conservative JPEG starting points; vision adapts these to each scene."""
    curves = {
        "soft_neutral": [(0, .008), (.18, .22), (.5, .56), (.8, .81), (1, .985)],
        "natural": [(0, 0), (.18, .17), (.5, .515), (.8, .815), (1, 1)],
        "sculpted": [(0, 0), (.18, .155), (.5, .47), (.8, .805), (1, .99)],
        "soft_warm": [(0, .008), (.18, .205), (.5, .57), (.8, .84), (1, .98)],
        "forest": [(0, .004), (.18, .15), (.5, .48), (.8, .81), (1, .99)],
        "graphic": [(0, 0), (.18, .15), (.5, .49), (.8, .825), (1, 1)],
        "documentary": [(0, .004), (.18, .17), (.5, .51), (.8, .80), (1, .975)],
    }
    recipe = Recipe(curve=ToneCurve(rgb=curves[method]))
    if method == "soft_neutral":
        return recipe.merged(Recipe(contrast=-.08, saturation=-.12, highlights=-.10,
                                   texture=.04, hsl={"aqua": HueShift(sat=-.25), "blue": HueShift(sat=-.25)}))
    if method == "natural":
        return recipe.merged(Recipe(contrast=.08, texture=.07, saturation=-.025))
    if method in {"sculpted", "graphic", "forest"}:
        recipe = recipe.merged(Recipe(contrast=.10, texture=.07, blacks=-.025,
            hsl={"blue": HueShift(sat=-.15), "aqua": HueShift(sat=-.12), "green": HueShift(sat=-.10)}))
        if method == "graphic":
            recipe.saturation = -.04
            recipe.hsl = {"blue": HueShift(sat=-.25), "aqua": HueShift(sat=-.30),
                          "green": HueShift(sat=-.15)}
            return recipe
        recipe.color_grading = ColorGradingControls(shadow_hue=218, shadow_sat=6,
            midtone_hue=222, midtone_sat=0, highlight_hue=43, highlight_sat=6, blending=55, balance=-15)
        if method == "forest":
            recipe.hsl["green"] = HueShift(hue=-8, sat=-.20, lum=.04)
            recipe.color_grading.shadow_hue = 58
            recipe.color_grading.shadow_sat = 3
        return recipe
    if method == "soft_warm":
        return recipe.merged(Recipe(contrast=-.04,
            hsl={"blue": HueShift(sat=-.20, lum=.04), "aqua": HueShift(sat=-.18, lum=.03),
                 "green": HueShift(sat=-.12)},
            color_grading=ColorGradingControls(shadow_sat=0, midtone_hue=38, midtone_sat=6,
                highlight_hue=48, highlight_sat=8, blending=55, balance=25)))
    return recipe.merged(Recipe(contrast=.02, saturation=-.10,
        hsl={"blue": HueShift(sat=-.50), "aqua": HueShift(sat=-.50),
             "green": HueShift(sat=-.30), "yellow": HueShift(sat=-.12)},
        color_grading=ColorGradingControls(shadow_sat=0, midtone_hue=38, midtone_sat=18,
            highlight_hue=40, highlight_sat=16, blending=65, balance=15)))


def retain_palette_design(recipe: Recipe, method: str, technical: TechnicalReport) -> Recipe:
    """Preserve the scene design, including explicit zero color controls.

    Missing controls already inherit method_recipe in the planner. Reapplying
    that recipe here erased deliberate choices and turned cyan photographs
    grey/brown. Palette area is not evidence that a color needs removing.
    """
    return recipe.model_copy(deep=True)


@lru_cache(maxsize=1)
def _practice_looks():
    return load_xmp_looks()


def finishing_context(critique: Critique, technical: TechnicalReport, *, methods=None) -> dict:
    from .master_studies import study_context
    scores = sorted(((route_similarity(lid, technical) or 0, lid) for lid in _practice_looks()), reverse=True)
    hints = []
    for score, lid in scores[:2]:
        if score < .35:
            continue
        look = _practice_looks()[lid]
        hints.append({"id": lid, "when": route_when(lid), "similarity": round(score, 3),
                      "palette": look.description,
                      "hsl_example": {k: v.model_dump() for k, v in look.recipe.hsl.items()}})
    return {
        "version": 2, "sources": SOURCES,
        "verified_workflows": [
            {"author": "Scott Kelby", "source": "https://lightroomkillertips.com/seven-clicks-to-done-lightroom-before-and-after-step-by-step/",
             "evidence": "公开 RAW 与步骤截图；天空局部曝光 -1.58、对比 +36、白色 +72。",
             "lesson": "偏白天空可以压曝光建立密度，再抬白色保留云的明亮；不要把天空整块压灰，也不要把山体一起去饱和。",
             "applies_when": "RAW 有可恢复的明亮天空和云；JPEG 或夜景不能照搬数值。"},
            {"author": "郭城安", "source": "https://www.bilibili.com/video/BV1tX4y1o7gS/",
             "evidence": "公开章节及时间轴预览帧；先镜头/裁切、光线分析和曲线，再处理具体颜色。",
             "lesson": "先安排主体、阴影与亮面的关系；颜色干净不等于降低所有颜色的饱和度。"},
            {"author": "Thomas看看世界", "source": "https://chuweimin.com/tutorials/a/photo-depth-3-tips/",
             "evidence": "作者图文与公开视频时间轴。",
             "lesson": "近中远景分别处理明暗与反差，保留远处空气；一两个主色加少量重点色。"},
            {"author": "青山摄影", "source": "https://www.bilibili.com/video/BV1H5411y71y/",
             "evidence": "公开章节和预览帧显示皮肤、眼睛、头发、服饰分开处理。",
             "lesson": "质感按材质决定；不能对皮肤、衣服和背景同时增加清晰度。现有 subject/sky/background 不能声称实现皮肤或眼睛专用蒙版。"},
        ],
        "per_image_studies": study_context(critique, technical),
        "methods": {slot: {"id": method, **METHODS[method]} for slot, method in zip("ABC", methods or method_ids(critique, technical))},
        "practice_hints": hints,
        "rules": [
            "先读主体、现场光线和主色；决定保留什么，再设计影调。",
            "先构图和局部光比，再整理确实干扰主体的颜色；蓝天、蓝衣服和蓝建筑不能仅凭色相当作同一个区域。",
            "明亮中间调、深色锚点与高光应各有位置；干净不等于全局去饱和，暖调不等于给所有亮度区间染黄。",
            "各方案通过光线位置、明暗结构或材质形成有意义的区别，可以保留同一套自然主辅色。",
            "保留雾气、夜色、自然肤色；资料中的拍摄布光、道具或主体位置不能用后期凭空重造。",
            "JPEG 不照搬 RAW 示范的大曝光、大校准数值。当前执行器没有的控制不能写成已经实现。",
            "向用户解释照片哪里更好，用普通中文，不要求用户懂摄影师名字、LUT 或参数。",
        ],
    }
