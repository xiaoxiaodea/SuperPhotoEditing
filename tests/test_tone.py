"""The measuring solver, driven by a small synthetic renderer instead of LightCraft."""
from __future__ import annotations

import numpy as np
from PIL import Image

from superphoto import tone

HEIGHT, WIDTH = 96, 144


def _scene() -> np.ndarray:
    """Linear RGB: a graded blue-grey sky, graded warm sand, dark rocks and a dark figure."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    fall = 1.0 - 0.55 * yy / (HEIGHT * 0.45)
    sky = np.stack([0.50 * fall, 0.58 * fall, 0.72 * fall], axis=-1)
    rise = 0.35 + 0.65 * xx / WIDTH
    sand = np.stack([0.42 * rise, 0.34 * rise, 0.24 * rise], axis=-1)
    picture = np.where((yy < HEIGHT * 0.45)[..., None], sky, sand)
    picture[(yy > HEIGHT * 0.93) & (xx < WIDTH * 0.5)] = np.array([0.006, 0.006, 0.007])      # rocks
    figure = (abs(xx - WIDTH * 0.3) < WIDTH * 0.08) & (abs(yy - HEIGHT * 0.7) < HEIGHT * 0.18)
    picture[figure] = np.array([0.07, 0.06, 0.055])
    return np.clip(picture * (1.0 + 0.04 * rng.standard_normal(picture.shape[:2])[..., None]), 0.0, 1.0)


def _weights(box: list[float]) -> np.ndarray:
    weights = np.zeros((HEIGHT, WIDTH))
    weights[int(box[1] * HEIGHT):int(box[3] * HEIGHT), int(box[0] * WIDTH):int(box[2] * WIDTH)] = 1.0
    return weights


def _display(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _saturated(linear: np.ndarray, amount: np.ndarray | float) -> np.ndarray:
    grey = tone.luma_of(linear)[..., None]
    return grey + (linear - grey) * (1.0 + np.asarray(amount)[..., None] / 100.0)


def _renderer(masks: dict, calls: list | None = None, colour_darkens: float = 0.0, base: np.ndarray | None = None):
    """``colour_darkens``: EV a mask's saturation takes off the pixels it covers, per point (an engine whose
    colour moves shift brightness, as LightCraft's do). ``base``: the linear scene, ``_scene()`` if not given."""
    base = _scene() if base is None else base

    def render(curve, exposure, local, saturation):
        linear = base * 2.0 ** exposure
        for key, owned in local.items():
            weights = masks[key]
            shift = owned.get("exposure", 0.0) - colour_darkens * owned.get("saturation", 0.0)
            linear = linear * 2.0 ** (shift * weights)[..., None]
            temp, tint = owned.get("temp", 0.0) * weights, owned.get("tint", 0.0) * weights
            linear = linear * np.stack([1 + 0.004 * temp + 0.0015 * tint, 1 - 0.003 * tint, 1 - 0.004 * temp], axis=-1)
            linear = _saturated(linear, owned.get("saturation", 0.0) * weights)
        linear = _saturated(linear, np.full(linear.shape[:2], float(saturation)))
        shown = _display(linear)
        if curve:
            xs, ys = np.asarray(curve).T
            shown = np.interp(shown, xs, ys)
        if calls is not None:
            calls.append({"exposure": exposure, "local": local, "saturation": saturation})
        return Image.fromarray((np.clip(shown, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    return render


def test_fit_curve_meets_reachable_targets_and_stays_smooth():
    measured = {"black": 0.06, "shadows": 0.2, "mid": 0.42, "highlights": 0.66, "white": 0.8}
    targets = {"black": 0.03, "shadows": 0.14, "mid": 0.45, "highlights": 0.82, "white": 0.95}
    curve, executed = tone.fit_curve(measured, targets)
    assert all(abs(executed[name] - targets[name]) < 0.02 for name in targets)
    xs, ys = np.asarray(curve).T
    slopes = np.diff(ys) / np.diff(xs)
    assert np.all(np.diff(ys) >= 0) and slopes.max() <= tone.MAX_SLOPE + 0.05


def test_fit_curve_shares_out_a_target_that_would_need_a_kink():
    measured = {"black": 0.05, "shadows": 0.2, "mid": 0.22, "highlights": 0.6, "white": 0.9}
    targets = {"black": 0.03, "shadows": 0.1, "mid": 0.5, "highlights": 0.8, "white": 0.95}
    curve, executed = tone.fit_curve(measured, targets)
    _, notes = tone.build_curve(measured, targets)
    xs, ys = np.asarray(curve).T
    assert (np.diff(ys) / np.diff(xs)).max() <= tone.MAX_SLOPE + 0.05
    assert abs(executed["mid"] - targets["mid"]) > tone.REPORT_MISS and any("tone.mid" in note for note in notes)


def test_solver_places_tones_and_a_region_under_its_mask():
    box = [0.24, 0.56, 0.36, 0.84]
    render = _renderer({"figure": _weights([0.2, 0.5, 0.4, 0.9])})
    targets = {"black": 0.03, "shadows": 0.3, "mid": 0.52, "highlights": 0.82, "white": 0.93}
    regions = [{"key": "figure", "box": box, "median": 0.36, "exposure": 0.0, "range": (-1.6, 1.6)}]
    report = tone.solve_scene(render, targets, regions)
    region = report["regions"][0]
    assert abs(region["after"] - 0.36) < 0.03 and region["before"] < 0.3
    assert not region["missed"] and region["exposure"] > 0.3 and region["response"] > 0.6
    assert all(abs(report["after"][name] - targets[name]) < 0.035 for name in targets)
    assert report["curve"] and report["probes"] <= 9


def test_solver_withdraws_a_mask_that_does_not_cover_its_box():
    calls: list = []
    # The mask sits on the sky; its target box is on the figure.
    render = _renderer({"face": _weights([0.6, 0.05, 0.9, 0.3])}, calls)
    regions = [{"key": "face", "box": [0.24, 0.56, 0.36, 0.84], "median": 0.4, "exposure": 0.1, "range": (-1.6, 1.6)}]
    report = tone.solve_scene(render, {}, regions)
    region = report["regions"][0]
    assert region["missed"] and region["exposure"] == 0.1 and region["response"] < tone.MISS_RESPONSE
    assert report["local"]["face"]["exposure"] == 0.1         # The delivered state is the withdrawn one,
    assert abs(region["after"] - region["before"]) < 0.02 and report["error"] > 0.1    # and so is what was measured.
    assert [call["local"]["face"]["exposure"] for call in calls] == [0.1, 1.1]         # One look for the mask, no more.


def test_solver_measures_a_target_on_the_part_of_its_box_the_mask_covers():
    # The right half of the box lies on bright sand outside the mask.
    render = _renderer({"figure": _weights([0.2, 0.5, 0.4, 0.9])})
    regions = [{"key": "figure", "box": [0.30, 0.56, 0.50, 0.84], "median": 0.4, "exposure": 0.0, "range": (-1.6, 1.6)}]
    region = tone.solve_scene(render, {}, regions)["regions"][0]
    assert 0.35 < region["coverage"] < 0.65 and not region["missed"] and "moved" not in region
    assert region["before"] < 0.32 and abs(region["after"] - 0.4) < 0.03      # The figure, not figure and sand mixed.
    assert region["response"] > 0.8 and 0.9 < region["exposure"] < 1.3     # About 1.1 EV takes 0.27 to 0.40.


def test_solver_finds_a_mask_that_lies_just_beside_its_box():
    render = _renderer({"figure": _weights([0.2, 0.5, 0.4, 0.9])})
    regions = [{"key": "figure", "box": [0.41, 0.6, 0.47, 0.8], "median": 0.5, "exposure": 0.0, "range": (-1.6, 1.6)}]
    region = tone.solve_scene(render, {}, regions)["regions"][0]
    assert region["moved"] and region["coverage"] == 0.0 and not region["missed"]


def test_percentile_target_follows_the_region_that_covers_its_pixels(monkeypatch):
    # The darkest sand is the frame's 10th percentile, and one mask covers all of it.
    render = _renderer({"sand": _weights([0.0, 0.45, 0.2, 0.93])})
    targets = {"black": 0.03, "shadows": 0.3, "mid": 0.56, "highlights": 0.82, "white": 0.93}
    regions = [{"key": "sand", "box": [0.02, 0.5, 0.12, 0.9], "median": 0.5, "exposure": 0.0, "range": (-1.6, 1.6)}]
    monkeypatch.setattr(tone, "FOLLOW", "full")
    full = tone.solve_scene(render, targets, regions)
    region = full["regions"][0]
    # The percentile target would darken that sand, the region target lifts it: the region is the specific one.
    assert region["overrides"] == ["shadows"] and full["aims"]["shadows"] > 0.42
    assert any("tone.shadows" in note and "区域目标" in note for note in full["notes"])
    assert abs(region["after"] - 0.5) < 0.03 and "against_curve" not in region
    assert abs(region["exposure"]) < 0.6                  # The curve carries it; the mask is not fighting the curve.
    xs, ys = np.asarray(full["curve"]).T
    assert (np.diff(ys) / np.diff(xs)).max() <= tone.MAX_SLOPE + 0.05
    free = tone.solve_scene(render, targets, [])
    assert free["aims"] == targets and abs(free["after"]["shadows"] - 0.3) < 0.03
    # By default the curve only stops pushing the sand down; the mask lifts it the rest of the way, so the
    # other pixels at that brightness (a face in the lake's band) keep the frame target.
    monkeypatch.setattr(tone, "FOLLOW", "excess")
    report = tone.solve_scene(render, targets, regions)
    region = report["regions"][0]
    assert region["overrides"] == ["shadows"] and 0.33 < report["aims"]["shadows"] < full["aims"]["shadows"] - 0.05
    assert abs(region["after"] - 0.5) < 0.03 and "against_curve" not in region
    assert region["exposure"] > full["regions"][0]["exposure"] + 0.3
    # A mask with room to spare and a curve that already moves its way: the frame target stays as written.
    lifted = dict(targets, shadows=0.45, mid=0.6)
    calm = tone.solve_scene(render, lifted, [dict(regions[0], median=0.55)])
    assert calm["aims"]["shadows"] == 0.45 and "overrides" not in calm["regions"][0]
    assert abs(calm["regions"][0]["after"] - 0.55) < 0.03
    # The mask carries its region below the frame target on its own; the frame percentile, which is that
    # region, reads it and misses, and the report says the two targets describe the same pixels.
    shared = tone.solve_scene(render, targets, [dict(regions[0], median=0.22)])
    assert shared["aims"]["shadows"] == 0.3 and shared["regions"][0]["shares"] == ["shadows"]
    assert shared["after"]["shadows"] < 0.27 and abs(shared["regions"][0]["after"] - 0.22) < 0.03


def test_frame_target_at_a_small_regions_brightness_gives_way_for_it(monkeypatch):
    # A mask on a fifth of the rocks, which are the frame's black: too few pixels to own the percentile.
    render = _renderer({"rocks": _weights([0.0, 0.9, 0.12, 1.0])})
    targets = {"black": 0.02, "shadows": 0.3, "mid": 0.52, "highlights": 0.82, "white": 0.93}
    regions = [{"key": "rocks", "name": "石头", "box": [0.01, 0.94, 0.11, 0.99], "median": 0.14, "exposure": 0.0,
                "range": (-1.0, 1.0)}]
    report = tone.solve_scene(render, targets, regions)
    region = report["regions"][0]
    # The black target would take the rocks down two stops and the mask can give back less than one: black
    # gives way by the rest, never past what the curve took, and the other targets keep what was written.
    assert region["gave_way"] == ["black"] and 0.05 < report["aims"]["black"] <= 0.075
    assert all(report["aims"][name] == targets[name] for name in ("shadows", "mid", "highlights", "white"))
    assert any(note.startswith("tone.black") and "石头" in note for note in report["notes"])
    assert -1.0 <= region["exposure"] <= 1.0
    # Where the unedited photo was darker than the plan's frame, black goes back no further than the photo.
    capped = tone.solve_scene(render, targets, regions, unedited=dict(targets, black=0.06))
    assert capped["regions"][0]["gave_way"] == ["black"] and capped["aims"]["black"] == 0.06
    monkeypatch.setattr(tone, "GIVE_WAY_CARRY", 10.0)        # No target can carry that much: none gives way.
    held = tone.solve_scene(render, targets, regions)
    assert "gave_way" not in held["regions"][0] and held["aims"]["black"] == 0.02
    assert region["after"] > held["regions"][0]["after"] + 0.03


def test_percentile_gives_way_instead_of_a_mask_undoing_the_curve(monkeypatch):
    # A small cloud at the frame's highlights: the targets lift that band over half a stop, the region target
    # keeps the cloud where it is. Its mask could pull it back, but a mask undoing the curve makes a harsh
    # picture: highlights gives way and the mask is left little to do.
    render = _renderer({"cloud": _weights([0.35, 0.08, 0.55, 0.17])})
    targets = {"black": 0.07, "shadows": 0.39, "mid": 0.6, "highlights": 0.88, "white": 0.95}
    regions = [{"key": "cloud", "name": "云", "box": [0.4, 0.1, 0.5, 0.15], "median": 0.72, "exposure": 0.0,
                "range": (-1.5, 1.5)}]
    report = tone.solve_scene(render, targets, regions)
    region = report["regions"][0]
    assert region["gave_way"] == ["highlights"] and report["aims"]["highlights"] < 0.8
    assert abs(region["exposure"]) < 0.15 and abs(region["after"] - 0.72) < 0.02
    assert report["aims"]["white"] == 0.95 and not any("超出" in note for note in report["notes"])
    # Where the photo's own value stops highlights short of carrying the cloud, it still gives way by what a
    # narrow mask cannot carry, as it did before.
    narrow = [dict(regions[0], range=(-0.3, 0.3))]
    capped = tone.solve_scene(render, targets, narrow, unedited=dict(targets, highlights=0.8))
    assert capped["regions"][0]["gave_way"] == ["highlights"] and capped["aims"]["highlights"] == 0.8
    assert any(note.startswith("tone.highlights") and "超出" in note for note in capped["notes"])
    monkeypatch.setattr(tone, "GIVE_WAY_UNDO", False)        # Only what the mask cannot carry: none here.
    held = tone.solve_scene(render, targets, regions)["regions"][0]
    assert "gave_way" not in held and held["exposure"] < -0.3


def test_white_does_not_give_way_for_a_mask_that_can_carry_its_region(monkeypatch):
    # The same at the top of the sky, nearest the white target: white holds the frame's range and stays.
    render = _renderer({"cloud": _weights([0.35, 0.0, 0.55, 0.06])})
    targets = {"black": 0.07, "shadows": 0.39, "mid": 0.6, "highlights": 0.82, "white": 0.98}
    regions = [{"key": "cloud", "name": "云", "box": [0.4, 0.0, 0.5, 0.04], "median": 0.76, "exposure": 0.0,
                "range": (-1.5, 1.5)}]
    report = tone.solve_scene(render, targets, regions)
    assert "gave_way" not in report["regions"][0] and report["aims"]["white"] == 0.98
    assert report["regions"][0]["exposure"] < -0.3
    monkeypatch.setattr(tone, "GIVE_WAY_ENDS", ())           # Were white treated as the others, it would give way.
    loose = tone.solve_scene(render, targets, regions)
    assert loose["regions"][0]["gave_way"] == ["white"] and loose["aims"]["white"] < 0.95


def test_a_region_the_curve_lifts_past_its_own_lift_target_keeps_some_of_that_lift(monkeypatch):
    # A cloud at the highlights of a darker frame: the tone targets lift it about 1.4 EV, its own target lifts it
    # half a stop. Neither pulling it back with its mask nor lowering highlights: it keeps up to LIFT_KEEP more.
    render = _renderer({"cloud": _weights([0.35, 0.08, 0.55, 0.17])}, base=_scene() * 0.3)
    targets = {"black": 0.07, "shadows": 0.39, "mid": 0.6, "highlights": 0.8, "white": 0.94}
    regions = [{"key": "cloud", "name": "云", "box": [0.4, 0.1, 0.5, 0.15], "median": 0.56, "exposure": 0.0,
                "range": (-1.5, 1.5)}]
    report = tone.solve_scene(render, targets, regions)
    region = report["regions"][0]
    kept = region["kept_lift"]
    assert kept["target"] == 0.56 and kept["by_curve"] > 0.78
    assert abs(kept["kept"] - tone.to_display(tone.to_linear(0.56) * 2.0 ** tone.LIFT_KEEP)) < 0.002
    assert region["target"] == kept["kept"] and abs(region["after"] - kept["kept"]) < 0.02
    assert report["aims"]["highlights"] == 0.8 and "gave_way" not in region and region["exposure"] > -0.4
    assert any(note.startswith("tone.lift") and "云" in note for note in report["notes"])
    # A lower target: highlights stays too, where it used to give way.
    lower = tone.solve_scene(render, targets, [dict(regions[0], median=0.52)])
    assert lower["aims"]["highlights"] == 0.8 and "gave_way" not in lower["regions"][0]
    # Dark sand, a mid-tone 1.9 EV under the frame's highlights, keeps its own target (the pavement of a night
    # street must not outshine the subject); with the LIFT_TOP gate open it would have kept the lift too.
    sand = {"sand": _weights([0.0, 0.45, 0.2, 0.93])}
    low = [{"key": "sand", "name": "沙", "box": [0.02, 0.5, 0.12, 0.9], "median": 0.275, "exposure": 0.0,
            "range": (-1.5, 1.5)}]
    held = tone.solve_scene(_renderer(sand, base=_scene() * 0.3), targets, low)["regions"][0]
    assert "kept_lift" not in held and abs(held["after"] - 0.275) < 0.02
    monkeypatch.setattr(tone, "LIFT_TOP", 10.0)
    assert tone.solve_scene(_renderer(sand, base=_scene() * 0.3), targets, low)["regions"][0]["kept_lift"]["kept"] > 0.33
    monkeypatch.setattr(tone, "LIFT_KEEP", 0.0)
    pulled = tone.solve_scene(render, targets, regions)["regions"][0]
    assert "kept_lift" not in pulled and pulled["exposure"] < -0.8 and abs(pulled["after"] - 0.56) < 0.02
    gave = tone.solve_scene(render, targets, [dict(regions[0], median=0.52)])
    assert gave["regions"][0]["gave_way"] == ["highlights"] and gave["aims"]["highlights"] < 0.78


def test_a_mask_that_moves_the_whole_frame_is_held_at_the_scene_limit(monkeypatch):
    # The sand mask reaches the whole lower frame, which holds the median: lifting it for a box on the sand
    # lifts the scene. Past SPILL_LIMIT from the unedited photo the solver holds it back and says so.
    render = _renderer({"sand": _weights([0.0, 0.45, 1.0, 1.0]), "figure": _weights([0.2, 0.5, 0.4, 0.9])})
    unedited = tone.measure(tone.load_small(render([], 0.0, {}, 0.0)))
    monkeypatch.setattr(tone, "SPILL_LIMIT", 0.3)
    regions = [{"key": "sand", "name": "沙", "box": [0.6, 0.6, 0.9, 0.9], "median": 0.8, "exposure": 0.0,
                "range": (-1.5, 1.5)}]
    held = tone.solve_scene(render, {}, regions, unedited=unedited)
    region = held["regions"][0]
    assert region["spilled"]["from"] > region["exposure"] + 0.3 and region["exposure"] > 0.2
    assert tone.ev_between(unedited["mid"], held["after"]["mid"]) < 0.3 + tone.SPILL_SLACK
    assert any(note.startswith("tone.spill") and "沙" in note for note in held["notes"])
    assert "frame_fix" not in region and region["spilled"]["limit_ev"] == 0.3
    # A RAW negative is developed further: its masks get the RAW's extra room on the median as well.
    raw = tone.solve_scene(render, {}, regions, unedited=unedited, shift_limits=tone.RAW_SHIFT_LIMITS)
    extra = tone.RAW_SHIFT_LIMITS["mid"][1] - tone.SHIFT_LIMITS["mid"][1]
    assert tone.ev_between(unedited["mid"], raw["after"]["mid"]) > 0.5
    assert (raw["regions"][0].get("spilled") or {}).get("limit_ev", round(0.3 + extra, 2)) == round(0.3 + extra, 2)
    monkeypatch.setattr(tone, "SPILL_PASSES", 0)
    free = tone.solve_scene(render, {}, regions, unedited=unedited)
    assert "spilled" not in free["regions"][0] and abs(free["regions"][0]["after"] - 0.8) < 0.03
    assert tone.ev_between(unedited["mid"], free["after"]["mid"]) > 0.5
    # A mask on a small figure barely moves the median: lifted to its range, it is not held.
    monkeypatch.setattr(tone, "SPILL_PASSES", 3)
    figure = [{"key": "figure", "box": [0.24, 0.56, 0.36, 0.84], "median": 0.5, "exposure": 0.0, "range": (-1.6, 1.6)}]
    lit = tone.solve_scene(render, {}, figure, unedited=unedited)
    assert "spilled" not in lit["regions"][0] and lit["regions"][0]["exposure"] == 1.6


def test_a_followed_percentile_stays_within_the_scene_limits(monkeypatch):
    # As in the follow test above, with the photo's own shadows near the written target and a tight limit:
    # the shadows follow the sand no further from the photo than a written target may go.
    render = _renderer({"sand": _weights([0.0, 0.45, 0.2, 0.93])})
    targets = {"black": 0.03, "shadows": 0.3, "mid": 0.56, "highlights": 0.82, "white": 0.93}
    regions = [{"key": "sand", "box": [0.02, 0.5, 0.12, 0.9], "median": 0.5, "exposure": 0.0, "range": (-1.6, 1.6)}]
    loose = tone.solve_scene(render, targets, regions)
    assert loose["aims"]["shadows"] > 0.33
    monkeypatch.setattr(tone, "SHIFT_LIMITS", dict(tone.SHIFT_LIMITS, shadows=(-3.0, 0.1)))
    held = tone.solve_scene(render, targets, regions, unedited=dict(targets))
    cap = tone.to_display(tone.to_linear(0.3) * 2.0 ** 0.1)
    assert held["aims"]["shadows"] <= cap + 1e-3 and held["aims"]["shadows"] < loose["aims"]["shadows"] - 0.02


def test_a_textured_near_neutral_is_moved_towards_its_colour_target(monkeypatch):
    # Warm and cool stripes, as on a brown hillside: the mean colour is a third of the colourfulness. The
    # target asks almost twice the colour the patch has; read against an absolute tolerance it looked reached.
    scene = _scene()
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    patch = (yy > HEIGHT * 0.55) & (yy < HEIGHT * 0.9) & (xx > WIDTH * 0.6) & (xx < WIDTH * 0.95)
    scene[patch & (xx % 2 == 0)] = np.array([0.28, 0.245, 0.215])
    scene[patch & (xx % 2 == 1)] = np.array([0.225, 0.24, 0.25])
    render = _renderer({"ridge": _weights([0.6, 0.55, 0.95, 0.9])}, base=scene)
    regions = [{"key": "ridge", "box": [0.65, 0.6, 0.9, 0.85], "colour": (38.0, 0.08), "exposure": 0.0}]
    colour = tone.solve_scene(render, {}, regions)["regions"][0]["colour"]
    assert colour["saturation"] > 10 and colour["after_chroma"] > colour["before_chroma"] + 0.007
    assert abs(colour["after_hue"] - 38.0) < 5
    monkeypatch.setattr(tone, "COLOUR_TOLERANCE_SCALED", False)
    still = tone.solve_scene(render, {}, regions)["regions"][0]["colour"]
    assert still["saturation"] == 0.0 and still["after_chroma"] == still["before_chroma"]


def test_frame_colour_target_is_read_at_the_photos_own_brightness(monkeypatch):
    render = _renderer({})
    base = tone.load_small(render([], 0.0, {}, 0.0))
    start, anchors = tone.chroma_mean(base), tone.measure(base)
    # Lift every percentile about a stop and ask for the colour the photo has: saturation stays where it is.
    lifted = {name: tone.to_display(min(1.0, tone.to_linear(value) * 2.0)) for name, value in anchors.items()}
    report = tone.solve_scene(render, lifted, chroma=start)
    chroma = report["chroma"]
    assert chroma["tone_gain"] > 1.08 and chroma["written"] == round(start, 4)
    assert abs(report["saturation"]) < 3
    # Read as an absolute number the same target would have washed the lifted frame out.
    monkeypatch.setattr(tone, "CHROMA_TONE_CAP", 1.0)
    held = tone.solve_scene(render, lifted, chroma=start)
    assert held["saturation"] < -7


def test_region_colour_targets_come_on_top_of_the_frame_target():
    box = [0.55, 0.6, 0.95, 0.95]
    render = _renderer({"sand": _weights([0.5, 0.5, 1.0, 1.0])})
    base = tone.load_small(render([], 0.0, {}, 0.0))
    start, sand = tone.chroma_mean(base), tone.region_stats(base, box)["chroma"]
    regions = [{"key": "sand", "box": box, "colour": (38.0, sand * 1.4)}]
    report = tone.solve_scene(render, {}, regions, chroma=start)
    chroma = report["chroma"]
    # The mask's colour raises the frame mean; global saturation does not take it back.
    assert chroma["regions_added"] > 0.004 and abs(report["saturation"]) < 4
    assert abs(chroma["target"] - (start + chroma["regions_added"])) < 0.002
    assert abs(report["regions"][0]["colour"]["after_chroma"] - sand * 1.4) < 0.025


def test_colour_moves_are_scaled_back_off_the_neutral_gate():
    render = _renderer({})
    start = tone.chroma_mean(tone.load_small(render([], 0.0, {}, 0.0)))
    free = tone.solve_scene(render, {}, chroma=start * 1.4)
    assert free["saturation"] > 15 and "neutral" not in free
    # A gate the full move would cross: 1.0 at 1.3 times the photo's colour, the photo itself well inside.
    held = tone.solve_scene(render, {}, chroma=start * 1.4, neutral=lambda rgb: tone.chroma_mean(rgb) / (start * 1.3))
    report = held["neutral"]
    assert report["solved"] > 1.0 and report["after"] <= tone.NEUTRAL_ROOM and 0.0 < report["kept"] < 1.0
    assert 0.0 < held["saturation"] < free["saturation"]
    assert any(note.startswith("tone.neutral") for note in held["notes"])
    # The photo's own colour is already past the gate: with the solver's colour taken back, global
    # saturation goes below the plan's until the neutrals are clear.
    tight = tone.solve_scene(render, {}, chroma=start * 1.4, neutral=lambda rgb: tone.chroma_mean(rgb) / (start * 0.95))
    report = tight["neutral"]
    assert report["kept"] == 0.0 and report["desaturated"] > 0 and report["after"] <= tone.NEUTRAL_ROOM
    assert tight["saturation"] < 0
    # Nothing near the gate: the guard measures and leaves the colour alone.
    calm = tone.solve_scene(render, {}, chroma=start * 1.4, neutral=lambda rgb: 0.2)
    assert calm["saturation"] == free["saturation"] and "neutral" not in calm


def test_colour_moves_that_barely_cast_the_neutrals_stay_while_the_frame_is_desaturated(monkeypatch):
    # The gate reads mostly the sky, which the sand's colour target does not touch: the frame is past it either
    # way, global saturation comes down one step either way, and the sand keeps its colour.
    box, sky = [0.55, 0.6, 0.95, 0.95], [0.0, 0.0, 1.0, 0.4]
    render = _renderer({"sand": _weights([0.5, 0.5, 1.0, 1.0])})
    base = tone.load_small(render([], 0.0, {}, 0.0))
    regions = [{"key": "sand", "box": box, "colour": (38.0, tone.region_stats(base, box)["chroma"] * 1.3)}]

    def gate(weight: float):
        def cast(rgb):
            return tone.region_stats(rgb, sky)["chroma"] + weight * tone.region_stats(rgb, box)["chroma"]
        own = cast(base)
        return lambda rgb: cast(rgb) / (own * 1.05)

    kept = tone.solve_scene(render, {}, regions, neutral=gate(0.1))
    assert kept["neutral"]["kept"] == 1.0 and kept["neutral"]["desaturated"] == 8.0
    assert kept["neutral"]["after"] <= tone.NEUTRAL_ROOM and kept["regions"][0]["colour"]["saturation"] > 10
    # Where keeping them costs the whole frame another step of colour, they go.
    costly = tone.solve_scene(render, {}, regions, neutral=gate(0.2))
    assert costly["neutral"]["kept"] == 0.0 and costly["neutral"]["desaturated"] == 8.0
    monkeypatch.setattr(tone, "NEUTRAL_KEEP_MOVES", False)
    dropped = tone.solve_scene(render, {}, regions, neutral=gate(0.1))
    assert dropped["neutral"]["kept"] == 0.0 and dropped["regions"][0]["colour"]["saturation"] == 0.0


def test_brightness_gain_counts_a_lift_but_not_the_colour_contrast_adds():
    rng = np.random.default_rng(3)
    before = rng.uniform(0.2, 0.6, (40, 40, 1)) * np.array([1.0, 0.85, 0.7])
    # A stop and a half brighter at the same saturation: colourfulness and the gain grow alike.
    lifted = before * 1.5
    assert abs(tone.brightness_gain(before, lifted) - 1.5) < 0.01
    # Stretched around its own middle: much more colourful, no brighter, so the frame target keeps its scale.
    middle = float(np.mean(before.max(axis=-1)))
    stretched = np.clip(middle + (before - middle) * 1.6, 0.0, 1.0)
    assert tone.chroma_mean(stretched) > 1.4 * tone.chroma_mean(before)
    assert abs(tone.brightness_gain(before, stretched) - 1.0) < 0.05


def test_solver_reaches_a_region_colour_and_reports_it():
    box = [0.55, 0.6, 0.95, 0.95]
    render = _renderer({"sand": _weights([0.5, 0.5, 1.0, 1.0])})
    start = tone.region_stats(tone.load_small(render([], 0.0, {}, 0.0)), box)["chroma"]
    regions = [{"key": "sand", "box": box, "colour": (38.0, start * 1.3)}]
    report = tone.solve_scene(render, {}, regions)
    colour = report["regions"][0]["colour"]
    assert not colour["missed"]
    assert abs(colour["after_chroma"] - start * 1.3) < 0.02 and abs(colour["after_hue"] - 38.0) < 6.0
    assert colour["saturation"] > 10 and abs(colour["temp"]) < 8 and abs(colour["tint"]) < 8


def test_solver_turns_the_hue_of_a_coloured_surface_with_temp_and_tint():
    # Same colourfulness, the hue 10 degrees round: a turn, which saturation cannot make. With temp and tint
    # costing three times saturation the step stopped a few percent of the way (p13 DSC_2121 C lake).
    box = [0.55, 0.6, 0.95, 0.95]
    render = _renderer({"sand": _weights([0.5, 0.5, 1.0, 1.0])})
    rgb = tone.load_small(render([], 0.0, {}, 0.0))
    start = tone.region_stats(rgb, box)["chroma"]
    hue, _ = tone.colour_polar(tone.region_colour(rgb, box))
    colour = tone.solve_scene(render, {}, [{"key": "sand", "box": box, "colour": (hue - 10.0, start)}])["regions"][0]["colour"]
    # Within the solver's "near enough" (9% of the colour vector, about 5 degrees).
    assert abs(colour["after_hue"] - (hue - 10.0)) < 6.0 and abs(colour["after_chroma"] - start) < 0.15 * start
    assert max(abs(colour["temp"]), abs(colour["tint"])) > 4.0
    # Wider limits (a color-shape mask) are used only when given.
    for control in ("temp", "tint"):
        assert tone.COLOUR_LIMITS[control][0] <= colour[control] <= tone.COLOUR_LIMITS[control][1]


def test_a_hue_on_the_far_side_is_turned_not_washed_out():
    # p21-r1 DSC_2620 C1: frost at hue 33, colourfulness .056, asked for cold cyan (200, .065). Taking the colour
    # out came nearest in a/b, so the solver ended at saturation -40 with temp -0.8: grey frost, reported reached.
    base = _scene()
    base[int(HEIGHT * 0.5):, :] = np.array([0.30, 0.275, 0.235])        # Warm near-neutral ground.
    box = [0.1, 0.6, 0.9, 0.95]
    render = _renderer({"frost": _weights([0.0, 0.5, 1.0, 1.0])}, base=base)
    rgb = tone.load_small(render([], 0.0, {}, 0.0))
    hue, _ = tone.colour_polar(tone.region_colour(rgb, box))
    assert 30.0 < hue < 45.0 and 0.055 < tone.region_stats(rgb, box)["chroma"] < 0.07
    colour = tone.solve_scene(render, {}, [{"key": "frost", "box": box, "colour": (200.0, 0.065)}])["regions"][0]["colour"]
    assert colour["saturation"] >= 0.0                          # Not desaturated towards the far hue (was -40).
    assert colour["temp"] < -10.0                                # Cooled instead (was -1).
    # Within a color-shape mask's wider limits the turn is made, and saturation, measured on the warm frost, follows
    # the colour round: the colourfulness comes back on the cyan side (was saturation -40 at hue 53 here too).
    wide = {"temp": (-60.0, 60.0), "tint": (-60.0, 60.0)}
    region = {"key": "frost", "box": box, "colour": (200.0, 0.065), "colour_limits": wide}
    colour = tone.solve_scene(render, {}, [region])["regions"][0]["colour"]
    assert abs((colour["after_hue"] - 200.0 + 180.0) % 360.0 - 180.0) < 12.0
    assert colour["saturation"] > 0.0 and abs(colour["after_chroma"] - 0.065) < 0.012


def test_closing_pass_takes_back_what_colour_work_did_to_the_tones(monkeypatch):
    # The sand mask's saturation also darkens the sand, half the frame: the curve was fitted before it moved.
    box = [0.55, 0.6, 0.95, 0.95]
    render = _renderer({"sand": _weights([0.0, 0.45, 1.0, 1.0])}, colour_darkens=0.02)
    start = tone.region_stats(tone.load_small(render([], 0.0, {}, 0.0)), box)["chroma"]
    targets = {"black": 0.03, "shadows": 0.2, "mid": 0.5, "highlights": 0.8, "white": 0.93}
    regions = [{"key": "sand", "box": box, "colour": (38.0, start * 1.5)}]
    monkeypatch.setattr(tone, "CLOSE_PASSES", 0)
    open_loop = tone.solve_scene(render, targets, regions)
    assert open_loop["regions"][0]["colour"]["saturation"] > 8
    assert abs(open_loop["after"]["mid"] - targets["mid"]) > 0.04          # About 0.45 for 0.50.
    assert any(note.startswith("tone.mid") for note in open_loop["notes"])
    monkeypatch.setattr(tone, "CLOSE_PASSES", 2)
    closed = tone.solve_scene(render, targets, regions)
    # Back to what the curve fitted to the targets gives (the fit itself leaves white about 0.02 short).
    assert max(abs(closed["after"][name] - open_loop["executed"][name]) for name in targets) < 0.02
    assert abs(closed["after"]["mid"] - targets["mid"]) < 0.015
    assert closed["closing"]["passes"] >= 1 and closed["closing"]["cost"][1] < closed["closing"]["cost"][0]
    assert closed["closing"]["fitted"]["mid"] > targets["mid"] + 0.03     # The curve now aims past the miss.
    # The report's "executed" is still the curve fitted to the aims, not to the moved targets.
    assert closed["executed"] == open_loop["executed"]
    assert not any(note.startswith("tone.") for note in closed["notes"])


def test_solver_does_not_chase_a_far_colour_past_its_limits():
    box = [0.55, 0.6, 0.95, 0.95]
    render = _renderer({"sand": _weights([0.5, 0.5, 1.0, 1.0])})
    start = tone.region_stats(tone.load_small(render([], 0.0, {}, 0.0)), box)["chroma"]
    # Two and a half times the colour and a hue 24 degrees away: more than a local mask should do.
    colour = tone.solve_scene(render, {}, [{"key": "sand", "box": box, "colour": (60.0, start * 2.5)}])["regions"][0]["colour"]
    for control, (lo, hi) in tone.COLOUR_LIMITS.items():
        assert lo <= colour[control] <= hi
    assert start * 1.1 < colour["after_chroma"] < start * 1.6 and colour["after_hue"] > colour["before_hue"]
    # A plan that starts outside the range is brought towards it, never pushed further out.
    held = tone.solve_scene(render, {}, [{"key": "sand", "box": box, "colour": (60.0, start * 2.5),
                                          "start": {"saturation": 34.0}}])["regions"][0]["colour"]
    assert held["saturation"] <= 34.0


def test_solver_leaves_colour_alone_where_the_box_is_too_dark_to_have_a_hue():
    render = _renderer({"rocks": _weights([0.0, 0.93, 0.5, 1.0])})
    regions = [{"key": "rocks", "box": [0.05, 0.95, 0.45, 0.99], "colour": (30.0, 0.1)}]
    colour = tone.solve_scene(render, {}, regions)["regions"][0]["colour"]
    assert colour["dark"] and not colour["missed"] and all(colour[control] == 0.0 for control in tone.COLOUR_CONTROLS)


def test_solver_leaves_colour_controls_alone_when_the_mask_misses():
    render = _renderer({"sand": _weights([0.0, 0.0, 0.3, 0.3])})
    regions = [{"key": "sand", "box": [0.55, 0.6, 0.95, 0.95], "colour": (38.0, 0.2)}]
    colour = tone.solve_scene(render, {}, regions)["regions"][0]["colour"]
    assert colour["missed"] and all(colour[control] == 0.0 for control in tone.COLOUR_CONTROLS)


def test_solver_sets_global_saturation_for_a_frame_colourfulness():
    render = _renderer({})
    start = tone.chroma_mean(tone.load_small(render([], 0.0, {}, 0.0)))
    report = tone.solve_scene(render, {}, [], chroma=start * 1.3)
    assert report["saturation"] > 10 and abs(report["chroma"]["after"] - start * 1.3) < 0.012
    lower = tone.solve_scene(render, {}, [], chroma=start * 0.7)
    assert lower["saturation"] < -10 and abs(lower["chroma"]["after"] - start * 0.7) < 0.012


def test_coverage_reads_where_a_mask_lies_and_ignores_what_is_black_in_both():
    plane = np.full((40, 60), 0.4)
    plane[:, :6] = 0.0                                     # Black: nothing to measure.
    shot = plane.copy()
    shot[:, :30] = tone.to_display(tone.to_linear(0.4) * 2.0)   # One stop on the left half.
    shot[:, :6] = 0.0
    share = tone.coverage(plane, shot, 1.0)
    assert share[20, 15] > 0.97 and share[20, 45] < 0.03 and share[20, 2] == 0.0
    picked, moved, part = tone.covered_pixels(share, [0.3, 0.2, 0.7, 0.8])
    assert picked is not None and not moved and 0.4 < part < 0.6 and picked[20, 25] and not picked[20, 35]
    assert tone.covered_pixels(share, [0.8, 0.2, 0.95, 0.8])[0] is None
    index, taken = tone.owner_of(plane, 0.4, {7: share})
    assert index == 7 and 0.0 < taken < 0.3                # Half of the pixels at that brightness: not the owner.


def test_grid_reports_hue_only_where_there_is_colour():
    pixels = np.zeros((80, 80, 3), dtype=np.float32)
    pixels[:, :40] = (0.2, 0.3, 0.8)      # blue
    pixels[:, 40:] = (0.5, 0.5, 0.5)      # grey
    table = tone.grid(pixels, cells=2)
    assert 215 <= table["hue"][0][0] <= 245 and table["hue"][0][1] == -1
    assert "--" in tone.grid_text(table["hue"])
