import pytest

from superphoto.models import ArtDirection, ColorGradingControls, Critique, EditIntent, MaskAdjustment, Recipe
from superphoto.preference import _scale_recipe, build_plans
from superphoto.lightroom_qa import blend_recipes
from superphoto.refinement import LocalDelta, RefinementEdits, apply_edits
from superphoto.review import VisualDamage, VisualReview, repair_visual_recipe
from test_celebrity_styles import _tech


@pytest.mark.parametrize("directed", [False, True])
def test_final_jpeg_plan_bounds_masks_inherited_from_critique(directed):
    mask = MaskAdjustment(target="sky", exposure_ev=-1.58, contrast=.36, whites=.72, blacks=-.4)
    critique = Critique(intent=EditIntent(masks=[mask]))
    if directed:
        critique.directions = [ArtDirection(slot="A", title="明暗", goal="保留云层", recipe=Recipe())]
    plan = build_plans(["faithful"], critique, _tech(source_kind="jpeg"))[0]
    actual = plan.recipe.masks[0]
    assert (actual.exposure_ev, actual.contrast, actual.whites, actual.blacks) == (-.75, .30, .30, -.25)
    assert critique.intent.masks[0] == mask


def test_raw_plan_retains_diagnosed_sky_recovery_and_does_not_raise_near_white_global_endpoint():
    from superphoto.models import CorrectionHint
    recovery = MaskAdjustment(target="sky", exposure_ev=-.7, contrast=.2, whites=.3, highlights=-.4)
    critique = Critique(correction=CorrectionHint(whites=.2), intent=EditIntent(masks=[recovery]),
        directions=[ArtDirection(slot="A", title="云层", goal="自然", recipe=Recipe(whites=.1,
            masks=[MaskAdjustment(target="sky", exposure_ev=-.3, temperature=2)]))])
    plan = build_plans(["faithful"], critique, _tech(source_kind="raw", highlight_p95=.98))[0]
    mask = plan.recipe.masks[0]
    assert (mask.exposure_ev, mask.contrast, mask.whites, mask.highlights) == (-.7, .2, .3, -.4)
    assert mask.temperature == 2
    assert plan.recipe.whites == .02


def test_scaling_and_safe_blend_do_not_reintroduce_unsafe_local_tone():
    mask = MaskAdjustment(target="sky", exposure_ev=-1.58, contrast=.36, whites=.72, blacks=-.4)
    recipe = Recipe(masks=[mask])
    half = _scale_recipe(recipe, .5).masks[0]
    assert (half.exposure_ev, half.contrast, half.whites, half.blacks) == (-.79, .18, .36, -.2)
    assert _scale_recipe(recipe, 0).masks[0].is_noop()
    assert blend_recipes(Recipe(), recipe, style_weight=0).masks == []
    safe = Recipe(masks=[MaskAdjustment(target="sky", highlights=-.1)])
    assert blend_recipes(safe, recipe, style_weight=.2).masks == safe.masks


def test_safe_blend_reduces_color_cast_with_the_tone_instead_of_retaining_full_tint():
    styled = Recipe(color_grading=ColorGradingControls(midtone_hue=40, midtone_sat=20, midtone_lum=8))
    assert blend_recipes(Recipe(), styled, style_weight=0).color_grading == Recipe().color_grading
    wheels = blend_recipes(Recipe(), styled, style_weight=.2).color_grading
    assert wheels.midtone_sat == pytest.approx(4)
    assert wheels.midtone_hue == pytest.approx(40)
    assert wheels.midtone_lum == pytest.approx(1.6)


def test_confirmed_local_clipping_repair_reduces_all_tone_controls():
    recipe = Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=-1, contrast=.4, whites=.7, blacks=-.3)])
    review = VisualReview(damage=[VisualDamage(kind="clipping", region="sky", severity=.9, evidence="云层纯白")])
    repaired = repair_visual_recipe(recipe, review, 2)
    assert repaired.masks[0].is_noop()


def test_small_feedback_preserves_existing_raw_recovery_and_jpeg_limit():
    raw = Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=-1.58, whites=.6)])
    actual = apply_edits(raw, RefinementEdits(masks=[LocalDelta(target="sky", saturation=-.05)]))
    assert actual.masks[0].exposure_ev == -1.58 and actual.masks[0].whites == .6
    actual = apply_edits(raw, RefinementEdits(masks=[LocalDelta(target="sky", exposure_ev=.2)]))
    assert actual.masks[0].exposure_ev == pytest.approx(-1.38)
    jpeg = Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=.7)])
    actual = apply_edits(jpeg, RefinementEdits(masks=[LocalDelta(target="sky", exposure_ev=.2)]))
    assert actual.masks[0].exposure_ev == .75


@pytest.mark.parametrize("crush_ground", [False, True])
def test_bright_sky_recovery_checks_remaining_ground_instead_of_global_mean(tmp_path, crush_ground):
    import numpy as np
    from PIL import Image
    from superphoto.lightroom_qa import assess_lightroom_render
    before = np.full((100, 100, 3), 64, dtype=np.uint8)
    before[:50] = 242
    after = before.copy()
    after[:50] = 150
    after[50:] = 10 if crush_ground else 60
    source, output = tmp_path / "source.png", tmp_path / "output.png"
    Image.fromarray(before).save(source)
    Image.fromarray(after).save(output)
    report = assess_lightroom_render(source, output, slot="B")
    assert report.accepted is not crush_ground
    dark = next(i for i in report.issues if i.code == "too_dark")
    assert dark.hard is crush_ground


@pytest.mark.parametrize("always_reject", [False, True])
def test_local_white_recovery_keeps_exposure_and_records_native_readback(tmp_path, always_reject):
    from test_postprocessing import _Client
    from superphoto.lightroom import apply_professional_recipe
    client = _Client()
    source = tmp_path / "copy.cr3"
    source.write_bytes(b"source")
    def review(stage, before, after):
        values = [a["settings"] for name, a in client.calls if name == "apply_ai_mask" and not a.get("reset")][-1]
        rejected = always_reject or values["local_Whites"] > 40
        return {"rejects": rejected, "issue_codes": ["highlight_clipping"] if rejected else []}
    recipe = Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=-1.25, contrast=.28, whites=.72)])
    result = apply_professional_recipe(client, original=source, recipe=recipe, output_dir=tmp_path / "out", stage_reviewer=review)
    effective = Recipe.model_validate(result["effective_recipe"])
    assert result["stages"][0]["rolled_back"] is always_reject
    if always_reject:
        assert effective.masks == [] and result["masks"] == []
    else:
        assert effective.masks[0].exposure_ev == -1.25
        assert effective.masks[0].whites == .36
        assert result["masks"][0]["settings"]["local_Whites"] == 36
        assert result["stages"][0]["attempts"][-1]["white_strength"] == .5
