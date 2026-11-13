from __future__ import annotations

from superphoto.celebrity_styles import CELEBRITY_LOOKS, build_style_decision
from superphoto.models import Critique, EditIntent, Issue, TechnicalReport


def _tech(**updates) -> TechnicalReport:
    base = dict(
        width=1200,
        height=800,
        source_kind="raw",
        mean_luma=0.30,
        median_luma=0.28,
        std_luma=0.18,
        highlight_clip=0.0,
        shadow_clip=0.03,
        dynamic_spread=0.62,
        sharpness=0.08,
        noise=0.03,
        saturation_mean=0.30,
        temperature_bias=0.03,
        top_third_blue=0.16,
        skin_fraction=0.04,
        green_fraction=0.08,
        histogram=[0.0] * 32,
        shadow_p05=0.02,
        highlight_p95=0.88,
        aqua_fraction=0.05,
        orange_fraction=0.08,
        magenta_fraction=0.02,
    )
    base.update(updates)
    return TechnicalReport(**base)


def test_style_decision_ranks_compatible_presets_without_mandatory_rescue() -> None:
    tech = _tech(green_fraction=0.28, orange_fraction=0.18, mean_luma=0.22)
    critique = Critique(
        genre="landscape",
        summary="植被场景偏平",
        issues=[Issue(dim="contrast", severity=0.6, note="影调偏平")],
    )
    decision = build_style_decision(tech, critique)
    assert "celebrity_rescue" not in decision.look_ids
    assert len(decision.look_ids) == 3
    assert len(set(decision.look_ids)) == 3
    assert "celebrity_forest_michael_kagerer" in decision.look_ids
    assert decision.route_id == decision.look_ids[0]
    assert len({item.family for item in decision.candidates}) == 3
    assert decision.ranked_candidates
    assert "曲线定调" in decision.workflow_steps
    assert decision.manual_gaps == []


def test_flat_landscape_can_choose_dehaze_route() -> None:
    tech = _tech(
        mean_luma=0.34,
        std_luma=0.055,
        dynamic_spread=0.34,
        saturation_mean=0.18,
        top_third_blue=0.25,
        aqua_fraction=0.13,
        green_fraction=0.05,
    )
    critique = Critique(genre="landscape", intent=EditIntent(preserve_atmosphere=False),
                        issues=[Issue(dim="contrast", severity=0.7, note="远景灰雾")])
    decision = build_style_decision(tech, critique)
    assert "celebrity_extreme_dehaze" in decision.look_ids
    assert "重建" in next(m.reason for m in decision.candidates if m.look_id == "celebrity_extreme_dehaze")


def test_dark_tone_area_does_not_claim_clipping_or_change_scene_ranking() -> None:
    technical = _tech(mean_luma=.08, shadow_clip=.03, black_clip=0)
    reference = build_style_decision(technical, Critique())
    low_key = build_style_decision(technical.model_copy(update={"shadow_clip": .75}), Critique())
    clipped = build_style_decision(technical.model_copy(update={"shadow_clip": .75, "black_clip": .20}), Critique())

    assert "低调" in low_key.tone
    assert "黑位已有裁切" not in low_key.tone
    assert "celebrity_rescue" not in low_key.look_ids
    assert low_key.purpose == reference.purpose
    assert "黑位已有裁切" in clipped.tone


def test_celebrity_recipes_all_have_curve_first_structure() -> None:
    assert set(CELEBRITY_LOOKS) >= {
        "celebrity_rescue",
        "celebrity_forest_michael_kagerer",
        "celebrity_low_sat_shinagawa",
        "celebrity_portra_hamada",
        "celebrity_tim_northey",
        "celebrity_twilight_kirinoma",
        "celebrity_german_maurice",
        "celebrity_extreme_dehaze",
    }
    assert all(not look.recipe.curve.is_identity() for look in CELEBRITY_LOOKS.values())
