"""Choose a different eligible preset when rendered candidates remain similar."""
from __future__ import annotations

from .models import Critique, Recipe, StyleMatch, TechnicalReport, VariantPlan


def alternative_preset_plan(
    slot: str, ranked: list[StyleMatch], critique: Critique, technical: TechnicalReport,
    *, excluded_ids: set[str], accepted_recipes: list[Recipe],
) -> VariantPlan | None:
    """Reuse this job's scene/feedback ranking, then compare executable designs.

    Eligibility is checked again; diversity cannot resurrect an excluded snow,
    night or portrait preset. No random A/B/C roles or cross-slot recipe blends.
    """
    from .directions import preset_direction, _signature, _signature_distance
    from .preference import build_plans
    from .recipes import LOOKS
    from .style_selection import incompatibility, scene_context

    context = scene_context(critique, technical)
    signatures = [_signature(recipe, full=True) for recipe in accepted_recipes]
    options = []
    for match in ranked:
        if (match.look_id in excluded_ids or match.look_id not in LOOKS
                or match.match_score < .25
                or incompatibility(match.look_id, context, technical, critique)):
            continue
        direction = preset_direction(slot, match.look_id, critique, technical)
        adapted = critique.model_copy(update={'directions': [direction]})
        plan = build_plans([match.look_id], adapted, technical, slots=(slot,))[0]
        sig = _signature(plan.recipe, full=True)
        distance = min((_signature_distance(sig, previous) for previous in signatures), default=0)
        # Fit and previous feedback remain primary; use surviving design
        # differences to break ties among compatible alternatives.
        score = match.match_score + min(distance, .2) * .5
        plan.match_score = match.match_score
        plan.adaptation_notes = ['实际导出与已有方案过近，改选本次场景准入名单中的另一预设。']
        options.append((score, match.look_id, plan))
    return max(options, key=lambda item: (item[0], item[1]))[2] if options else None
