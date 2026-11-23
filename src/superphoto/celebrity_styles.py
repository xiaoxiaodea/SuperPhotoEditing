from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import (
    Critique,
    DetailControls,
    FinishControls,
    HueShift,
    Look,
    ParametricTone,
    Recipe,
    StyleDecision,
    StyleMatch,
    TechnicalReport,
    ToneCurve,
)


CELEBRITY_STYLE_IDS = (
    "celebrity_rescue",
    "celebrity_forest_michael_kagerer",
    "celebrity_low_sat_shinagawa",
    "celebrity_portra_hamada",
    "celebrity_tim_northey",
    "celebrity_twilight_kirinoma",
    "celebrity_german_maurice",
    "celebrity_extreme_dehaze",
)


CELEBRITY_LOOKS: dict[str, Look] = {
    "celebrity_rescue": Look(
        id="celebrity_rescue",
        title="废片拯救",
        title_en="Rescue baseline",
        description="先用曲线重建影调，再压高光、提阴影和整理脏色；不是原片对照，而是一张稳妥可交付的 Lightroom 成片。",
        genre_tags=["general", "portrait", "landscape", "street", "indoor", "product"],
        moves=["曲线定调", "高光保护", "阴影恢复", "HSL 先降脏色"],
        recipe=Recipe(
            highlights=-0.26,
            shadows=0.18,
            whites=0.05,
            blacks=-0.03,
            contrast=0.18,
            saturation=-0.05,
            vibrance=0.08,
            texture=0.04,
            clarity=0.03,
            parametric_tone=ParametricTone(shadows=-0.04, darks=-0.03, lights=0.04, highlights=-0.03),
            finish=FinishControls(vignette_midpoint=48, vignette_roundness=0, vignette_feather=72, vignette_style=1),
            hsl={
                "yellow": HueShift(sat=-0.12, lum=0.12),
                "orange": HueShift(sat=-0.10, lum=-0.04),
                "green": HueShift(sat=-0.08),
                "blue": HueShift(sat=-0.08),
            },
            curve=ToneCurve(rgb=[(0.0, 0.025), (0.22, 0.18), (0.52, 0.55), (0.80, 0.86), (1.0, 0.985)]),
        ),
    ),
    "celebrity_forest_michael_kagerer": Look(
        id="celebrity_forest_michael_kagerer",
        title="森系 · Michael Kagerer",
        title_en="Forest editorial",
        description="暗、低饱和、橄榄绿和暖橙主体，黑点保留呼吸感。",
        genre_tags=["landscape", "nature", "portrait", "street"],
        moves=["奶黑曲线", "绿转橄榄", "橙色主体分离", "蓝紫降饱和"],
        recipe=Recipe(
            exposure_ev=-0.08,
            highlights=-0.22,
            shadows=-0.05,
            blacks=0.12,
            contrast=0.12,
            saturation=-0.10,
            vibrance=0.22,
            hsl={
                "green": HueShift(hue=-24, sat=-0.36, lum=0.12),
                "yellow": HueShift(hue=-6, sat=-0.18, lum=0.06),
                "orange": HueShift(sat=-0.08, lum=0.12),
                "blue": HueShift(sat=-0.34),
                "purple": HueShift(sat=-0.32),
            },
            curve=ToneCurve(
                rgb=[(0.0, 0.055), (0.18, 0.13), (0.48, 0.47), (0.82, 0.84), (1.0, 0.97)],
                b=[(0.0, 0.025), (0.55, 0.53), (1.0, 0.95)],
            ),
        ),
    ),
    "celebrity_low_sat_shinagawa": Look(
        id="celebrity_low_sat_shinagawa",
        title="低饱暗系 · 品川力",
        title_en="Low-saturation cinema",
        description="低调、褪色电影感，重点压蓝绿并保留暖色线索。",
        genre_tags=["street", "night", "indoor", "documentary"],
        moves=["褪色曲线", "蓝绿强降饱和", "暖中间调", "克制清晰度"],
        recipe=Recipe(
            highlights=-0.55,
            shadows=0.08,
            whites=-0.48,
            blacks=0.48,
            contrast=-0.08,
            saturation=-0.20,
            clarity=-0.10,
            temperature=5,
            hsl={
                "blue": HueShift(sat=-0.68),
                "aqua": HueShift(hue=10, sat=-0.30, lum=0.22),
                "green": HueShift(hue=-30, sat=-0.66),
                "yellow": HueShift(hue=-4, sat=-0.38, lum=-0.10),
                "orange": HueShift(sat=-0.14, lum=-0.08),
                "red": HueShift(hue=5, sat=-0.16),
            },
            curve=ToneCurve(
                rgb=[(0.0, 0.075), (0.24, 0.20), (0.52, 0.50), (0.82, 0.78), (1.0, 0.93)],
                r=[(0.0, 0.0), (0.55, 0.51), (1.0, 0.94)],
            ),
        ),
    ),
    "celebrity_portra_hamada": Look(
        id="celebrity_portra_hamada",
        title="日系胶片 · 滨田英明",
        title_en="Portra high-key",
        description="高调、通透、奶黑，暖黄玫瑰肤色与橄榄绿植。",
        genre_tags=["portrait", "family", "street", "nature"],
        moves=["高调奶黑曲线", "肤色暖黄", "绿植橄榄", "细颗粒"],
        recipe=Recipe(
            exposure_ev=0.48,
            highlights=-0.28,
            shadows=-0.42,
            whites=-0.15,
            blacks=0.48,
            contrast=-0.10,
            tint=12,
            texture=-0.18,
            clarity=-0.08,
            grain=0.20,
            detail=DetailControls(luminance_nr=18, color_nr=25),
            finish=FinishControls(grain_size=25, grain_roughness=50, vignette_midpoint=52, vignette_roundness=8, vignette_feather=78, vignette_style=1),
            hsl={
                "red": HueShift(sat=-0.10),
                "orange": HueShift(hue=-2, sat=0.06, lum=0.16),
                "green": HueShift(hue=13, sat=-0.15),
                "blue": HueShift(hue=-4, sat=-0.14, lum=0.08),
            },
            curve=ToneCurve(rgb=[(0.0, 0.08), (0.24, 0.29), (0.52, 0.60), (0.82, 0.87), (1.0, 0.965)]),
        ),
    ),
    "celebrity_tim_northey": Look(
        id="celebrity_tim_northey",
        title="全能出片 · Tim Northey",
        title_en="Clean cinematic",
        description="干净高对比、暖亮部与冷暗部，暗部扎实而高光柔和。",
        genre_tags=["general", "portrait", "street", "landscape", "product"],
        moves=["轻 S 曲线", "压高光抬阴影", "橙色偏黄", "青蓝降饱和"],
        recipe=Recipe(
            exposure_ev=0.24,
            highlights=-0.62,
            shadows=0.24,
            whites=-0.20,
            blacks=-0.07,
            contrast=0.18,
            vibrance=0.10,
            texture=0.06,
            clarity=-0.02,
            parametric_tone=ParametricTone(shadows=-0.08, darks=-0.05, lights=0.08, highlights=-0.10),
            finish=FinishControls(vignette_midpoint=45, vignette_roundness=-5, vignette_feather=80, vignette_style=1),
            hsl={
                "orange": HueShift(hue=3, sat=-0.05, lum=0.06),
                "aqua": HueShift(sat=-0.18, lum=0.10),
                "blue": HueShift(sat=-0.20, lum=0.08),
            },
            curve=ToneCurve(
                rgb=[(0.0, 0.025), (0.22, 0.16), (0.52, 0.55), (0.80, 0.87), (1.0, 0.975)],
                r=[(0.0, 0.0), (0.35, 0.32), (1.0, 1.0)],
                b=[(0.0, 0.025), (0.55, 0.50), (1.0, 0.94)],
            ),
        ),
    ),
    "celebrity_twilight_kirinoma": Look(
        id="celebrity_twilight_kirinoma",
        title="暮色梦境 · KIRINOMA",
        title_en="Twilight dream",
        description="低中调梦核，暗部平滑、蓝紫阴影和红黄高光。",
        genre_tags=["night", "street", "portrait", "landscape"],
        moves=["暗部像素平移", "蓝紫阴影", "红黄高光", "大颗粒"],
        recipe=Recipe(
            exposure_ev=0.34,
            highlights=-0.35,
            shadows=0.48,
            whites=-0.20,
            blacks=0.42,
            contrast=-0.16,
            texture=-0.10,
            clarity=-0.12,
            saturation=-0.12,
            grain=0.14,
            detail=DetailControls(sharpness=120, sharpen_radius=1.4, sharpen_detail=27, sharpen_masking=41),
            finish=FinishControls(grain_size=56, grain_roughness=60, vignette_midpoint=38, vignette_roundness=-20, vignette_feather=82, vignette_style=1),
            hsl={
                "red": HueShift(hue=9, sat=-0.28),
                "orange": HueShift(hue=-3, sat=-0.20),
                "yellow": HueShift(hue=-16, sat=-0.14),
                "green": HueShift(hue=-15, sat=-0.66),
                "blue": HueShift(sat=-0.30),
            },
            curve=ToneCurve(
                rgb=[(0.0, 0.09), (0.16, 0.10), (0.42, 0.39), (0.74, 0.78), (1.0, 0.98)],
                r=[(0.0, 0.035), (0.55, 0.53), (1.0, 1.0)],
                b=[(0.0, 0.07), (0.55, 0.54), (1.0, 0.91)],
            ),
        ),
    ),
    "celebrity_german_maurice": Look(
        id="celebrity_german_maurice",
        title="德系复古 · Maurice Pehle",
        title_en="German vintage",
        description="沉、克制、生活气，橄榄绿与偏青蓝，高光保留暖光晕。",
        genre_tags=["street", "documentary", "indoor", "portrait", "landscape"],
        moves=["压实黑位", "橄榄绿", "蓝偏青", "高光留余量"],
        recipe=Recipe(
            exposure_ev=-0.22,
            contrast=-0.14,
            highlights=-0.45,
            shadows=-0.16,
            whites=-0.42,
            blacks=0.12,
            saturation=-0.14,
            texture=-0.04,
            clarity=-0.05,
            grain=0.18,
            detail=DetailControls(sharpness=0, color_nr=50, color_nr_detail=50, color_nr_smoothness=50),
            finish=FinishControls(grain_size=45, grain_roughness=41, vignette_midpoint=42, vignette_roundness=-10, vignette_feather=70, vignette_style=1),
            hsl={
                "orange": HueShift(hue=-4, sat=-0.20),
                "yellow": HueShift(sat=-0.16),
                "green": HueShift(hue=-24, sat=-0.24),
                "blue": HueShift(hue=-3, sat=-0.12),
            },
            curve=ToneCurve(
                rgb=[(0.0, 0.0), (0.10, 0.035), (0.48, 0.48), (0.82, 0.84), (1.0, 0.93)],
                r=[(0.0, 0.055), (0.55, 0.54), (1.0, 1.0)],
                b=[(0.0, 0.0), (0.55, 0.53), (1.0, 0.84)],
            ),
        ),
    ),
    "celebrity_extreme_dehaze": Look(
        id="celebrity_extreme_dehaze",
        title="极致去雾",
        title_en="Deep dehaze",
        description="用曲线和色深而不是暴力去雾滑杆，适合灰雾、长焦和低对比远景。",
        genre_tags=["landscape", "nature", "architecture"],
        moves=["RGB 同步 S 曲线", "全色先降饱和", "对比释放颜色", "清晰度克制"],
        recipe=Recipe(
            highlights=-0.20,
            shadows=0.12,
            whites=0.10,
            blacks=-0.14,
            contrast=0.30,
            saturation=-0.12,
            vibrance=0.18,
            clarity=-0.03,
            dehaze=0.08,
            hsl={name: HueShift(sat=-0.10) for name in ("red", "orange", "yellow", "green", "aqua", "blue")},
            curve=ToneCurve(
                rgb=[(0.0, 0.0), (0.20, 0.14), (0.50, 0.53), (0.80, 0.88), (1.0, 1.0)],
                r=[(0.0, 0.0), (0.20, 0.14), (0.50, 0.53), (0.80, 0.88), (1.0, 1.0)],
                g=[(0.0, 0.025), (0.20, 0.15), (0.50, 0.53), (0.80, 0.88), (1.0, 1.0)],
                b=[(0.0, 0.035), (0.20, 0.16), (0.50, 0.53), (0.80, 0.88), (1.0, 1.0)],
            ),
        ),
    ),
}


STYLE_MANUAL_GAPS: list[str] = []


@dataclass(frozen=True)
class _Score:
    look_id: str
    score: float
    reason: str


def build_style_decision(
    technical: TechnicalReport,
    critique: Critique,
    preferred: list[str] | None = None,
    *, feedback: dict | None = None,
) -> StyleDecision:
    from .style_selection import rank_presets, select_diverse_presets

    ranked, excluded = rank_presets(technical, critique, feedback=feedback, preferred=preferred)
    matches = select_diverse_presets(ranked, critique, technical)
    top = matches[0]
    return StyleDecision(
        tone=_tone_card(technical), color=_color_card(technical), purpose=_purpose_card(technical, critique),
        route_id=top.look_id, route_title=top.title, route_reason=top.reason,
        look_ids=[item.look_id for item in matches], candidates=matches,
        ranked_candidates=ranked, excluded=excluded,
        workflow_steps=["读图卡", "缺陷优先级", "按场景筛选预设", "比较适用风格", "技术校正",
                        "曲线定调", "局部光影适配", "色彩校正", "细节降噪与镜头校正", "Lightroom 分阶段导出"],
        manual_gaps=list(STYLE_MANUAL_GAPS),
    )


def _style_scores(t: TechnicalReport, critique: Critique) -> list[_Score]:
    low = 1.0 - float(np.clip((t.mean_luma - 0.08) / 0.45, 0.0, 1.0))
    green = float(np.clip(t.green_fraction / 0.28, 0.0, 1.0))
    skin = float(np.clip(t.skin_fraction / 0.24, 0.0, 1.0))
    blue = float(np.clip((t.top_third_blue + t.aqua_fraction) / 0.45, 0.0, 1.0))
    warm = float(np.clip((t.temperature_bias + 0.15) / 0.55, 0.0, 1.0))
    flat = float(np.clip((0.18 - t.std_luma) / 0.14, 0.0, 1.0))
    muted = float(np.clip((0.36 - t.saturation_mean) / 0.30, 0.0, 1.0))
    night = 1.0 if critique.genre == "night" else 0.0
    landscape = 1.0 if critique.genre in {"landscape", "nature"} else 0.0
    portrait = 1.0 if critique.genre == "portrait" else 0.0
    return [
        _Score("celebrity_forest_michael_kagerer", 0.24 + 0.38 * green + 0.14 * warm + 0.10 * low + 0.08 * landscape, "绿植与暖色主体适合压成橄榄森系，且当前影调能承受偏暗处理。"),
        _Score("celebrity_low_sat_shinagawa", 0.22 + 0.28 * low + 0.22 * muted + 0.12 * night + 0.08 * (1.0 - green), "低亮度和低饱和结构适合褪色暗调，蓝绿可以主动收敛。"),
        _Score("celebrity_portra_hamada", 0.22 + 0.36 * skin + 0.14 * (1.0 - low) + 0.10 * green + 0.08 * portrait, "肤色或明亮生活场景适合高调奶黑和 Portra 式暖肤色。"),
        _Score("celebrity_tim_northey", 0.48 + 0.10 * skin + 0.08 * blue + 0.06 * (1.0 - low), "题材与色块较均衡，适合走干净高对比的全能电影路线。"),
        _Score("celebrity_twilight_kirinoma", 0.20 + 0.30 * low + 0.18 * blue + 0.12 * night + 0.08 * muted, "低中调和冷色线索适合暗部平移、蓝紫阴影与柔亮高光。"),
        _Score("celebrity_german_maurice", 0.25 + 0.20 * low + 0.16 * green + 0.14 * muted + 0.08 * warm, "克制的暖色、绿植或生活纪实题材适合沉稳德系复古。"),
        _Score("celebrity_extreme_dehaze", 0.18 + 0.38 * flat + 0.18 * landscape + 0.10 * blue + 0.06 * muted, "对比和动态展开不足，优先用曲线与色深重建远景通透度。"),
    ]


def _tone_card(t: TechnicalReport) -> str:
    level = "低调" if t.mean_luma < 0.16 else ("高调" if t.mean_luma > 0.52 else "中调")
    contrast = "短调偏平" if t.std_luma < 0.12 else ("长调高反差" if t.dynamic_spread > 0.72 else "中等反差")
    black = "黑位已有裁切" if t.black_clip > 0.10 else ("黑位有呼吸" if t.shadow_p05 > 0.025 else "黑点接近纯黑")
    highlight = "高光需优先保护" if t.highlight_clip > 0.01 or t.highlight_p95 > 0.93 else "高光仍有余量"
    return f"{level}、{contrast}；{black}，{highlight}。"


def _color_card(t: TechnicalReport) -> str:
    blocks = sorted(
        [
            (t.green_fraction, "绿色"),
            (t.aqua_fraction + t.top_third_blue * 0.5, "青蓝"),
            (max(t.orange_fraction, t.skin_fraction), "橙暖色"),
            (t.magenta_fraction, "洋红"),
        ],
        reverse=True,
    )
    dominant = "、".join(name for value, name in blocks[:2] if value > 0.025) or "中性色"
    sat = "高饱和" if t.saturation_mean > 0.45 else ("低饱和" if t.saturation_mean < 0.20 else "中等饱和")
    cast = "偏暖" if t.temperature_bias > 0.18 else ("偏冷" if t.temperature_bias < -0.12 else "冷暖较平衡")
    return f"主色块为{dominant}，{sat}，整体{cast}。"


def _purpose_card(t: TechnicalReport, critique: Critique) -> str:
    if t.std_luma < 0.12 or any(issue.dim == "contrast" and issue.severity >= 0.35 for issue in critique.issues):
        return "先重建影调和对比，再决定色彩风格；避免用全局饱和硬救灰片。"
    if critique.genre == "portrait":
        return "先保护肤色和高光，再在胶片通透与电影对冲之间选择。"
    if t.mean_luma < 0.16:
        return "保留低调意图，优先整理黑位与冷暖关系，不把夜景强行提成白天。"
    return "先完成必要校正，再按本图的主体、光线与配色筛选相容的成片风格。"


def _defect_pressure(t: TechnicalReport, critique: Critique) -> float:
    issue = max((item.severity for item in critique.issues), default=0.0)
    technical = min(1.0, t.highlight_clip * 8 + t.black_clip * 1.5 + max(0.0, 0.13 - t.std_luma) * 3)
    return float(np.clip(max(issue, technical), 0.0, 1.0))


def _style_family(look_id: str) -> str:
    if look_id in {"celebrity_forest_michael_kagerer", "celebrity_german_maurice"}:
        return "olive"
    if look_id in {"celebrity_low_sat_shinagawa", "celebrity_twilight_kirinoma"}:
        return "moody"
    if look_id == "celebrity_portra_hamada":
        return "film"
    if look_id == "celebrity_extreme_dehaze":
        return "technical"
    return "clean"
