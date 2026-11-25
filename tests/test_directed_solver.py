"""Directed plans with measured targets, solved against the synthetic renderer of test_tone."""
from __future__ import annotations

import json

from superphoto import directed, tone
from test_tone import _renderer, _weights

FIGURE_BOX = [0.24, 0.56, 0.36, 0.84]
SAND_BOX = [0.55, 0.6, 0.95, 0.95]


def _document_render(masks: dict, seen: list | None = None):
    """``render(document)`` for solve_plan: reads back what compile_document wrote."""
    synthetic = _renderer(masks)

    def render(document):
        local = {mask["name"]: {key: value for key, value in mask["adjust"].items() if key != "amount"}
                 for mask in document.get("masks", [])}
        if seen is not None:
            seen.append(document)
        return synthetic((document.get("curve") or {}).get("master") or [],
                         (document.get("light") or {}).get("exposure", 0.0), local,
                         (document.get("color") or {}).get("saturation", 0.0))

    return render


def _mask(name: str, target: dict, precise: bool = False, **adjust) -> dict:
    mask = {"name": name, "purpose": "测试", "adjust": adjust, "target": target,
            "shape": {"kind": "radial", "center": {"x": 0.5, "y": 0.5}, "rx": 0.2, "ry": 0.2}}
    if precise:
        mask["limit"] = {"kind": "luminance", "lo": 0.0, "hi": 0.4}
    return mask


def test_sanitize_reads_colour_targets_on_the_table_scale():
    plan, notes = directed.sanitize_plan({
        "title": "t", "tone": {"black": 0.03, "mid": 0.45, "white": 0.95, "chroma": 16},
        "masks": [_mask("沙", {"box": SAND_BOX, "hue": 398, "chroma": 18}),
                  _mask("雪", {"box": SAND_BOX, "chroma": 3}),
                  _mask("脸", {"box": FIGURE_BOX, "median": 46, "hue": 25, "chroma": 0.22}),
                  _mask("缺色相", {"box": SAND_BOX, "chroma": 30}),
                  _mask("只有色相", {"box": SAND_BOX, "hue": 30}, exposure=0.2)]}, slot="A")
    assert plan["tone"]["chroma"] == 0.16 and "saturation" not in plan["tone"]
    targets = {mask["name"]: mask.get("target") for mask in plan["masks"]}
    assert targets["沙"] == {"box": SAND_BOX, "hue": 38.0, "chroma": 0.18}
    assert targets["雪"] == {"box": SAND_BOX, "hue": 0.0, "chroma": 0.03}
    assert targets["脸"] == {"box": FIGURE_BOX, "median": 0.46, "hue": 25.0, "chroma": 0.22}
    assert "缺色相" not in targets and targets["只有色相"] is None
    assert any("有 chroma 没有 hue" in note for note in notes) and any("需要 box" in note for note in notes)
    assert not any("tone.chroma" in note for note in notes)


def test_solve_plan_writes_solved_values_into_the_plan_and_reports_them():
    base = tone.load_small(_renderer({})([], 0.0, {}, 0.0))
    sand, frame = tone.region_stats(base, SAND_BOX)["chroma"], tone.chroma_mean(base)
    plan, _ = directed.sanitize_plan({
        "title": "t", "tone": {"black": 0.03, "shadows": 0.3, "mid": 0.52, "highlights": 0.82, "white": 0.93,
                               "chroma": frame * 1.15},
        "develop": {"color": {"vibrance": 5}},
        "masks": [_mask("人物", {"box": FIGURE_BOX, "median": 0.33}, precise=True, shadows=10),
                  _mask("沙地", {"box": SAND_BOX, "hue": 38, "chroma": sand * 1.25}),
                  _mask("脱靶", {"box": FIGURE_BOX, "median": 0.5}, exposure=0.1)]}, slot="A")
    seen: list = []
    masks = {"人物": _weights([0.2, 0.5, 0.4, 0.9]), "沙地": _weights([0.5, 0.5, 1.0, 1.0]),
             "脱靶": _weights([0.6, 0.0, 0.9, 0.3])}
    executed, document, notes, report = directed.solve_plan(plan, "jpeg", _document_render(masks, seen))
    assert plan["masks"][0]["adjust"] == {"shadows": 10.0}          # The model's plan is left as written.
    adjust = {mask["name"]: mask["adjust"] for mask in executed["masks"]}
    assert adjust["人物"]["exposure"] > 0.3 and adjust["人物"]["shadows"] == 10.0
    assert any(abs(adjust["沙地"][control]) > 1.0 for control in tone.COLOUR_CONTROLS)
    assert adjust["脱靶"]["exposure"] == 0.1
    assert executed["develop"]["color"]["saturation"] == report["saturation"] and document["curve"]["master"]
    assert document == seen[-1] or document["masks"] == seen[-1]["masks"]
    regions = {region["name"]: region for region in report["regions"]}
    assert abs(regions["人物"]["after"] - 0.33) < 0.035 and not regions["人物"]["missed"]
    assert not regions["人物"]["plain"] and regions["沙地"]["plain"]
    assert regions["脱靶"]["missed"] and any("脱靶" in note and "没有覆盖" in note for note in notes)
    assert abs(regions["沙地"]["colour"]["after_chroma"] - sand * 1.25) < 0.025
    assert "exposure" not in adjust["沙地"]                           # Carried by the solver, not written by it.
    # The frame target is read at the photo's own brightness and scaled by what the tones alone did.
    chroma = report["chroma"]
    # These targets stretch the frame more than they lift it: the colour the stretch adds does not scale the target.
    assert chroma["written"] == round(frame * 1.15, 4) and 0.98 < chroma["tone_gain"] < 1.05
    # The sand mask's colour target comes on top of the frame target.
    assert abs(chroma["target"] - (frame * 1.15 * chroma["tone_gain"] + chroma["regions_added"])) < 0.002
    assert abs(chroma["after"] - chroma["target"]) < 0.02
    json.dumps(report, ensure_ascii=False)                          # The report is stored beside the render.

    text = directed.measured_block(report, None, None)
    assert "THE MASK DOES NOT COVER THIS BOX" in text and 'region target "沙地"' in text
    assert "frame colourfulness x100" in text and "hue" in text


def test_bare_geometric_mask_is_held_back_and_the_report_says_how_to_go_further():
    plan, _ = directed.sanitize_plan({
        "title": "t", "tone": {"black": 0.03, "shadows": 0.3, "mid": 0.52, "highlights": 0.82, "white": 0.93},
        "masks": [_mask("人物", {"box": FIGURE_BOX, "median": 0.42})]}, slot="A")
    executed, _, _, report = directed.solve_plan(plan, "jpeg", _document_render({"人物": _weights([0.2, 0.5, 0.4, 0.9])}))
    region = report["regions"][0]
    assert executed["masks"][0]["adjust"]["exposure"] == directed.SOLVER_PLAIN_RANGE[1] and region["limited"]
    text = directed.measured_block(report, None, None)
    assert "bare linear or radial mask" in text and "luminance or color limit" in text
    # The same mask made precise may go further.
    plan["masks"][0]["limit"] = {"kind": "luminance", "lo": 0.0, "hi": 0.4}
    executed, _, _, report = directed.solve_plan(plan, "jpeg", _document_render({"人物": _weights([0.2, 0.5, 0.4, 0.9])}))
    assert executed["masks"][0]["adjust"]["exposure"] > 1.0 and "bare linear" not in directed.measured_block(report, None, None)


def test_tone_owned_controls_are_left_out_of_a_plan_with_tone_targets(monkeypatch):
    monkeypatch.setattr(directed, "TONE_OWNED", {"light": ("contrast", "whites"), "curve": ("lights", "darks")})
    raw = {"title": "t", "tone": {"black": 0.03, "mid": 0.45, "white": 0.95},
           "develop": {"light": {"contrast": 20, "highlights": -40, "whites": 10}, "curve": {"lights": 12}}}
    plan, notes = directed.sanitize_plan(raw, slot="A")
    assert plan["develop"] == {"light": {"highlights": -40.0}}
    assert any("A.light" in note and "contrast" in note for note in notes) and any("A.curve" in note for note in notes)
    # Two targets do not place the tones: the plan's own controls stay.
    raw["tone"] = {"black": 0.03, "white": 0.95}
    assert directed.sanitize_plan(raw, slot="A")[0]["develop"]["light"]["contrast"] == 20.0


def test_look_saturation_is_kept_under_the_solved_value(monkeypatch):
    monkeypatch.setitem(directed.LOOKS, "test_look", {"title": "t", "develop": {"color": {"saturation": -20}}})
    base = tone.chroma_mean(tone.load_small(_renderer({})([], 0.0, {}, 0.0)))
    plan, _ = directed.sanitize_plan({"title": "t", "tone": {"chroma": base}, "look": {"id": "test_look",
                                                                                         "strength": 1.0}}, slot="C")
    executed, document, _, report = directed.solve_plan(plan, "jpeg", _document_render({}))
    # The look desaturates and the frame target asks for the colour the photo has: the look is this
    # version's colour, so the target only nudges the slider back by the look's room.
    room = directed.LOOK_SATURATION_ROOM
    assert abs(document["color"]["saturation"] - (-20.0 + room)) < 1.0
    assert abs(executed["develop"]["color"]["saturation"] - room) < 1.0
    assert report["chroma"]["after"] < base - 0.005 and report["chroma"]["look_room"] == room
    assert "with a look the solver moves saturation at most" in directed.measured_block(report, None, None)
    # Without the look the same target is met.
    plain, _ = directed.sanitize_plan({"title": "t", "tone": {"chroma": base}, "develop": {"color": {"saturation": -20}}},
                                      slot="C")
    _, _, _, free = directed.solve_plan(plain, "jpeg", _document_render({}))
    assert abs(free["chroma"]["after"] - base) < 0.01 and "look_room" not in free["chroma"]


def test_a_profile_keeps_its_colour_like_a_look():
    base = tone.chroma_mean(tone.load_small(_renderer({})([], 0.0, {}, 0.0)))
    # A muted profile under a target that asks for the photo's own colour: the solver would cancel the profile
    # with saturation; it only nudges within the look room, and the profile stays in the document.
    plan, _ = directed.sanitize_plan({"title": "t", "tone": {"chroma": base * 1.5},
                                      "profile": {"id": "lc.muted.bleached", "amount": 60}}, slot="B")
    _, document, _, report = directed.solve_plan(plan, "jpeg", _document_render({}))
    assert document["profile"] == {"id": "lc.muted.bleached", "amount": 60.0}
    assert document["color"]["saturation"] <= directed.LOOK_SATURATION_ROOM + 0.5
    assert report["chroma"]["look_room"] == directed.LOOK_SATURATION_ROOM
    free, _ = directed.sanitize_plan({"title": "t", "tone": {"chroma": base * 1.5}}, slot="B")
    _, free_document, _, free_report = directed.solve_plan(free, "jpeg", _document_render({}))
    assert free_document["color"]["saturation"] > directed.LOOK_SATURATION_ROOM + 5.0
    assert "look_room" not in free_report["chroma"]


def test_a_raw_negative_under_a_look_may_still_be_developed_in_colour(monkeypatch):
    monkeypatch.setitem(directed.LOOKS, "test_look", {"title": "t", "use": "u",
                                                      "develop": {"color": {"saturation": -20}}})
    base = tone.chroma_mean(tone.load_small(_renderer({})([], 0.0, {}, 0.0)))
    plan, _ = directed.sanitize_plan({"title": "t", "tone": {"chroma": base * 1.6}, "look": {"id": "test_look",
                                                                                             "strength": 1.0}}, slot="C")
    _, jpeg, _, jpeg_report = directed.solve_plan(plan, "jpeg", _document_render({}))
    _, raw, _, raw_report = directed.solve_plan(plan, "raw", _document_render({}))
    # A developed JPEG keeps the look's colour within its room; the RAW negative is developed past it.
    assert jpeg["color"]["saturation"] <= -20.0 + directed.LOOK_SATURATION_ROOM + 0.5
    assert raw["color"]["saturation"] > -20.0 + directed.LOOK_SATURATION_ROOM + 5.0
    assert raw_report["chroma"]["after"] > jpeg_report["chroma"]["after"] + 0.01
    assert raw_report["chroma"]["look_room"] == directed.RAW_LOOK_SATURATION_UP
    raw_text = directed._looks_block({"test_look": {"chroma": 1.0, "distance": 2.0}}, 6.0, 5.0, "raw")
    assert "RAW negative" in raw_text and "stays unreached" not in raw_text
    assert "stays unreached" in directed._looks_block({"test_look": {"chroma": 1.0, "distance": 2.0}}, 6.0)


def test_a_raw_mask_may_take_out_more_colour_than_a_jpeg_one(monkeypatch):
    seen: list = []
    solve = tone.solve_scene
    monkeypatch.setattr(tone, "solve_scene", lambda render, targets, regions=None, **kw: (
        seen.append([dict(region) for region in regions or []]) or solve(render, targets, regions, **kw)))
    plan, _ = directed.sanitize_plan({"title": "t", "tone": {"mid": 0.5},
                                      "masks": [_mask("sky", {"box": SAND_BOX, "hue": 210, "chroma": 0.06})]}, slot="A")
    weights = {"sky": _weights([0.2, 0.5, 0.4, 0.9])}
    directed.solve_plan(plan, "jpeg", _document_render(weights))
    directed.solve_plan(plan, "raw", _document_render(weights))
    # A negative's blue sky needs more than -40 of mask saturation to reach his pale one; adding colour does not widen.
    assert "colour_limits" not in seen[0][0]
    assert seen[1][0]["colour_limits"]["saturation"] == (directed.RAW_MASK_SATURATION_FLOOR,
                                                         tone.COLOUR_LIMITS["saturation"][1])


def test_a_raw_color_mask_keeps_temp_and_tint_to_a_nudge(monkeypatch):
    seen: list = []
    solve = tone.solve_scene
    monkeypatch.setattr(tone, "solve_scene", lambda render, targets, regions=None, **kw: (
        seen.append([dict(region) for region in regions or []]) or solve(render, targets, regions, **kw)))
    mask = _mask("sand", {"box": SAND_BOX, "hue": 33, "chroma": 0.3})
    mask["shape"] = {"kind": "color", "points": [{"x": 0.3, "y": 0.6}]}
    plan, _ = directed.sanitize_plan({"title": "t", "tone": {"mid": 0.5}, "masks": [mask]}, slot="A")
    weights = {"sand": _weights([0.2, 0.5, 0.4, 0.9])}
    _, _, _, jpeg = directed.solve_plan(plan, "jpeg", _document_render(weights))
    _, _, _, raw = directed.solve_plan(plan, "raw", _document_render(weights))
    # His RAW masks never colour (76 of 76 at temp, tint and saturation 0): the wide color-mask turn is a JPEG's.
    assert seen[0][0]["colour_limits"]["tint"] == directed.COLOUR_MASK_LIMITS["tint"]
    assert seen[1][0]["colour_limits"]["tint"] == directed.RAW_MASK_COLOUR_LIMITS["tint"]
    assert seen[1][0]["colour_limits"]["temp"] == directed.RAW_MASK_COLOUR_LIMITS["temp"]
    colour = raw["regions"][0]["colour"]
    assert abs(colour["tint"]) <= 10.0 + 1e-6 and colour["limits"]["tint"] == [-10.0, 10.0]


def test_a_missed_hue_is_told_which_global_control_turns_it():
    def lines(after_hue, target_hue, after=0.76):
        colour = {"target_hue": target_hue, "target_chroma": 0.36, "before_hue": 48.0, "before_chroma": 0.1,
                  "after_hue": after_hue, "after_chroma": 0.38, "temp": 24.0, "tint": 10.0, "saturation": -30.0,
                  "missed": False, "limits": {"temp": [-24.0, 24.0], "tint": [-10.0, 10.0], "saturation": [-70.0, 30.0]}}
        return "\n".join(directed._region_lines({"name": "crest", "box": SAND_BOX, "before": 0.3, "after": after,
                                                 "colour": colour}))
    # Lit sand too yellow: magenta (tint +) turns it toward red, and the region is in the highlights.
    text = lines(46.0, 33.0)
    assert "NOT REACHED" in text and "tint +/-10" in text
    assert "white balance tint + turns it toward 33" in text and "highlights grading wheel" in text
    # Sand too red asks for the other way; a blue sky too purple asks for green, in the midtones.
    assert "white balance tint - turns it" in lines(18.0, 33.0)
    assert "white balance tint - turns it toward 220" in lines(235.0, 220.0, after=0.5) and "midtones" in lines(235.0, 220.0, after=0.5)
    # A hue within reach gets no turn.
    assert "degrees off" not in lines(38.0, 33.0)


def test_a_look_does_not_move_the_written_tone_targets(monkeypatch):
    monkeypatch.setitem(directed.LOOKS, "test_matte", {"title": "t", "use": "u", "develop": {},
                                                       "tone": {"black": 0.06, "white": -0.05}})
    written = {"black": 0.02, "shadows": 0.04, "mid": 0.067, "highlights": 0.22, "white": 0.455}
    plan, _ = directed.sanitize_plan({"title": "t", "tone": dict(written), "look": {"id": "test_matte",
                                                                                     "strength": 0.7}}, slot="C")
    # The planner wrote the finished frame; a matte it wants is in its numbers, not added on top of them.
    assert directed.tone_targets(plan) == written
    # The family's handling is offered to the planner instead.
    assert any(line.startswith("test_matte") and "black +0.06" in line and "white -0.05" in line
               for line in directed.look_lines())


def test_tone_targets_stay_within_reach_of_the_unedited_photo():
    night = {"black": 0.009, "shadows": 0.013, "mid": 0.029, "highlights": 0.144, "white": 0.322}
    bounded, notes = tone.bound_shift({"black": 0.016, "shadows": 0.045, "mid": 0.12, "highlights": 0.42,
                                       "white": 0.70}, night)
    for name, (lo, hi) in tone.SHIFT_LIMITS.items():
        assert lo - 0.02 <= tone.ev_between(night[name], bounded[name]) <= hi + 0.02, name
    assert bounded["mid"] < 0.08 and bounded["black"] == 0.016          # Black is free.
    assert [bounded[name] for name in tone.ORDER] == sorted(bounded[name] for name in tone.ORDER)
    assert any(note.startswith("tone.mid") for note in notes) and not any("tone.black" in note for note in notes)
    # A target within reach is left as written, and so is a target with no baseline to compare with.
    assert tone.bound_shift({"mid": 0.05, "white": 0.4}, night) == ({"mid": 0.05, "white": 0.4}, [])
    assert tone.bound_shift({"mid": 0.5}, {}) == ({"mid": 0.5}, [])
    # Bright high-key frames may not be dragged down more than the limit either.
    bright = {"mid": 0.895, "white": 0.952}
    held, _ = tone.bound_shift({"mid": 0.4, "white": 0.95}, bright)
    assert abs(tone.ev_between(0.895, held["mid"]) - tone.SHIFT_LIMITS["mid"][0]) < 0.02


def test_a_raw_negative_may_be_developed_further_than_a_finished_jpeg():
    # The photographer's snowfield DSC02262: the flat RAW render's mid .37 -> his finished .75 (+2.2 EV).
    flat = {"black": 0.16, "shadows": 0.31, "mid": 0.37, "highlights": 0.49, "white": 0.65}
    wanted = {"black": 0.15, "shadows": 0.55, "mid": 0.75, "highlights": 0.90, "white": 0.95}
    jpeg, jpeg_notes = tone.bound_shift(wanted, flat, tone.shift_limits("jpeg"))
    raw, raw_notes = tone.bound_shift(wanted, flat, tone.shift_limits("raw"))
    assert jpeg["mid"] < 0.70 and any(note.startswith("tone.mid") for note in jpeg_notes)
    assert raw == wanted and not raw_notes
    assert tone.shift_limits("png") is tone.SHIFT_LIMITS


def test_solve_plan_bounds_its_targets_by_the_baseline_and_says_so():
    measured = tone.measure(tone.load_small(_renderer({})([], 0.0, {}, 0.0)))
    plan, _ = directed.sanitize_plan({"title": "t", "tone": {"mid": measured["mid"]}}, slot="A")
    wanted = plan["tone"]["mid"]
    baseline = {"mid": tone.to_display(tone.to_linear(wanted) / 8.0)}     # The plan asks for +3 EV.
    _, _, notes, report = directed.solve_plan(plan, "jpeg", _document_render({}), baseline=baseline)
    assert report["targets"]["mid"] < wanted and any(note.startswith("tone.mid") for note in report["notes"])
    assert any(note.startswith("tone.mid") for note in notes)
    _, _, _, free = directed.solve_plan(plan, "jpeg", _document_render({}))
    assert free["targets"]["mid"] == wanted


def test_a_region_the_curve_buries_is_told_which_frame_target_to_move():
    reference = {"black": 0.16, "shadows": 0.23, "mid": 0.38, "highlights": 0.51, "white": 0.62}
    executed = {"black": 0.045, "shadows": 0.125, "mid": 0.40, "highlights": 0.79, "white": 0.935}
    curve, _ = tone.fit_curve(reference, executed)
    fix = tone.frame_fix(reference, executed, curve, {"shadows": 0.125, "mid": 0.40}, 0.25, 0.16)
    assert fix["anchor"] == "shadows" and fix["between"] == ["shadows", "mid"] and fix["now"] == 0.125
    assert abs(fix["needed"] - (0.125 + 0.16 / (1 - 0.02 / 0.15))) < 0.002
    assert tone.frame_fix(reference, executed, curve, {}, 0.05, 0.1)["between"] == [None, "black"]
    assert tone.frame_fix(reference, executed, curve, {}, 0.3, 0.0) is None
    region = {"name": "脸", "box": [0.6, 0.38, 0.67, 0.42], "before": 0.157, "after": 0.12, "target": 0.30,
              "exposure": 1.25, "limited": True, "plain": False, "response": 0.4, "missed": False, "frame_fix": fix}
    text = "\n".join(directed._region_lines(region))
    assert "FIX: write tone.shadows about 0.310 (now 0.125)" in text and "between your frame shadows and mid" in text
    assert "The mask is weak here" not in text


def test_a_region_kept_above_its_target_is_told_why():
    region = {"name": "白羽绒服", "box": [0.58, 0.42, 0.68, 0.56], "before": 0.527, "after": 0.81, "target": 0.81,
              "exposure": -0.05, "limited": False, "plain": False, "response": 0.95, "missed": False,
              "kept_lift": {"target": 0.64, "kept": 0.81, "by_curve": 0.82}}
    text = "\n".join(directed._region_lines(region))
    assert "KEPT HIGHER: you wrote 0.640" in text and "lift it to about 0.820" in text and "at 0.810" in text


def test_plan_without_targets_is_compiled_without_rendering():
    plan, _ = directed.sanitize_plan({"title": "t", "develop": {"light": {"contrast": 10}}}, slot="A")

    def render(document):
        raise AssertionError("nothing to solve")

    executed, document, notes, report = directed.solve_plan(plan, "jpeg", render)
    assert executed is plan and report is None and document["light"]["contrast"] == 10.0


def test_measured_block_shows_where_a_nearly_identical_sibling_is_alike():
    from PIL import Image
    import numpy as np

    dark = Image.fromarray(np.full((64, 96, 3), 30, dtype=np.uint8), mode="RGB")
    lit = np.full((64, 96, 3), 30, dtype=np.uint8)
    lit[:, 48:] = 90                                       # Only the right half is brighter in this version.
    mine, theirs = directed.picture_stats(Image.fromarray(lit, mode="RGB")), directed.picture_stats(dark)
    text = directed.measured_block(None, theirs, mine, {"A": 3.2}, 5.0, {"A": theirs})
    assert "nearly identical" in text and "this version minus version A" in text
    grid = text.split("this version minus version A")[1]
    assert grid.count("+0") == 32 and grid.count("+23") == 32     # Alike on the left, apart on the right.
    # A distinct sibling gets the verdict only, no table.
    assert "minus version A" not in directed.measured_block(None, theirs, mine, {"A": 7.0}, 5.0, {"A": theirs})
