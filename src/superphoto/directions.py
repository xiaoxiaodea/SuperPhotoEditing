"""Design three different photographs from the same scene, before rendering."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .config import Settings
from .models import ArtDirection, ColorGradingControls, Critique, HueShift, MaskAdjustment, Recipe, TechnicalReport, ToneCurve
from .finishing_knowledge import METHODS, finishing_context, method_ids, method_recipe, retain_palette_design
from .ollama import chat_vision, probe


def scene_directions(critique: Critique, technical: TechnicalReport, *, methods=None) -> list[ArtDirection]:
    """Use the study methods even when the vision model is unavailable."""
    result = []
    has_sky = technical.top_third_blue > .12
    for slot, method in zip("ABC", methods or method_ids(critique, technical)):
        lesson = METHODS[method]
        recipe = method_recipe(method)
        masks = {m.target: m.model_copy(deep=True) for m in critique.intent.masks}
        if has_sky and "sky" not in masks:
            masks["sky"] = MaskAdjustment(target="sky", exposure_ev={"A": -.08, "B": -.16, "C": -.04}[slot],
                highlights={"A": -.10, "B": -.16, "C": -.08}[slot], reason="保留天空过渡，让亮度服务画面重点")
        if "subject" in masks and masks["subject"].exposure_ev > 0:
            subject = masks["subject"]
            masks["subject"] = subject.model_copy(update={
                "exposure_ev": float(np.clip(subject.exposure_ev + {"A": .04, "B": .16, "C": .04}[slot], -.75, .65))})
            if slot == "B":
                masks.setdefault("background", MaskAdjustment(target="background", exposure_ev=-.12,
                    saturation=-.05, reason="轻收环境光色，让主体与环境分开"))
        recipe.masks = list(masks.values())
        result.append(ArtDirection(slot=slot, method_id=method, title=lesson["title"], goal=lesson["goal"],
                                   moves=lesson["moves"], recipe=recipe))
    return result


def preset_direction(slot: str, look_id: str, critique: Critique, technical: TechnicalReport) -> ArtDirection:
    """The selected executable preset is the starting recipe, not a label."""
    from .lightroom import native_color_recipe
    from .preference import _fit_lut_to_scene, _scale_recipe, look_match, lut_amount
    from .recipes import get_look

    look = get_look(look_id)
    # Strength belongs to this photo and preset, never to a fixed A/B/C role.
    amount = lut_amount(look, technical, "B", look_match(look_id, technical, critique.genre), critique.lut_mix)
    base = look.recipe
    if look.source == "master":
        from .masters import recipe_for_photo
        base = recipe_for_photo(look_id, technical)
    recipe = _fit_lut_to_scene(_scale_recipe(native_color_recipe(base), amount), look_id, technical)
    # Presets do not know this source's exposure, crop, sharpening or masks.
    recipe.exposure_ev = 0
    recipe.masks = [m.model_copy(deep=True) for m in critique.intent.masks]
    for key in ("highlights", "shadows", "whites", "blacks", "contrast"):
        correction = getattr(critique.correction, key)
        if abs(correction) > .05 and correction * getattr(recipe, key) > 0:
            setattr(recipe, key, getattr(recipe, key) * .5)
    direction = ArtDirection(slot=slot, look_id=look_id, method_id="preset:" + look_id,
        preset_amount=amount, title=look.title, goal=f"{critique.intent.goal}；参考{look.title}的影调与配色",
        moves=[look.description], recipe=recipe)
    return bound_direction(direction, critique, technical)


def bound_direction(direction: ArtDirection, critique: Critique, technical: TechnicalReport) -> ArtDirection:
    """Keep deliberate tone/color design, constrain damage-prone extremes."""
    data = direction.recipe.clamped().model_dump()
    natural = direction.method_id == "natural" or (not direction.method_id and direction.slot == "A")
    bounds = {"exposure_ev": (-.40, .28), "temperature": (-8, 8), "tint": (-6, 6),
              "contrast": (-.20, .28), "highlights": (-.35, .10), "shadows": (-.15, .22),
              "whites": (-.22, .06), "blacks": (-.18, .04), "texture": (-.15, .18),
              "clarity": (-.12, .15), "dehaze": (-.05, .08), "saturation": (-.16, .08),
              "vibrance": (-.10, .16), "grain": (0, .08), "vignette": (0, .15)}
    if technical.source_kind == "raw":
        bounds.update(exposure_ev=(-.8, .6), highlights=(-.85, .15), shadows=(-.20, .35),
                      whites=(-.35, .12), blacks=(-.4, .08))
    if technical.highlight_p95 > .93:
        bounds["whites"] = (bounds["whites"][0], .02)
    if technical.mean_luma < .12:
        bounds["exposure_ev"] = (-.20, .10)
    if critique.intent.preserve_atmosphere:
        bounds["dehaze"] = (-.03, .04)
    for key, (lo, hi) in bounds.items():
        data[key] = float(np.clip(data[key], lo, hi))
    # Use the luminance wheels, not a global tint or inherited channel curves.
    if critique.intent.natural_color:
        data["temperature"] = data["tint"] = 0.0
        if not direction.look_id:
            data["curve"].update(r=[], g=[], b=[])
        else:
            # Imported channel curves carry the preset palette. Bound their
            # chromatic displacement instead of deleting every preset's color.
            for channel in ("r", "g", "b"):
                points = data["curve"][channel]
                ys = np.maximum.accumulate([float(np.clip(y, max(0, x-.035), min(1, x+.035))) for x, y in points])
                data["curve"][channel] = [(x, float(y)) for (x, _), y in zip(points, ys)]
        for name, value in data["color_grading"].items():
            if value is not None and name.endswith("_sat"):
                data["color_grading"][name] = min(value, 18.0 if natural else 24.0)
    for name, value in data["color_grading"].items():
        if value is not None and name.endswith("_lum"):
            data["color_grading"][name] = float(np.clip(value, -12, 12))
    points = data["curve"]["rgb"]
    if points:
        # Bound deviation from identity and require a monotone curve.
        points = sorted((float(np.clip(x, 0, 1)), float(np.clip(y, max(0, x-.10), min(1, x+.08)))) for x, y in points)
        if len({x for x, _ in points}) != len(points) or any(a[1] > b[1] for a, b in zip(points, points[1:])):
            points = []
        elif points[0][0] != 0 or points[-1][0] != 1:
            points = []
        if points and critique.intent.preserve_black_depth:
            points[0] = (0, min(points[0][1], .02))
        data["curve"]["rgb"] = points
    data.update(detail={}, optics={}, crop={}, local={}, grade={})
    for name, shift in data["hsl"].items():
        shift["hue"] = float(np.clip(shift["hue"], -12, 12))
        shift["sat"] = float(np.clip(shift["sat"], -.30 if natural else -.75, .12))
        shift["lum"] = float(np.clip(shift["lum"], -.18, .15))
    if (critique.intent.subject_confidence == 0 and critique.intent.subject_box is None
            and not any(mask.target == "subject" for mask in critique.intent.masks)):
        # Native background masks invert a subject selection; both need a
        # subject supported by the critique or an explicit editing intent.
        data["masks"] = [mask for mask in data["masks"] if mask["target"] == "sky"]
    for mask in data["masks"]:
        raw_sky = technical.source_kind == "raw" and mask["target"] == "sky"
        mask["exposure_ev"] = float(np.clip(mask["exposure_ev"], -2 if raw_sky else -.75, .75))
        mask["contrast"] = float(np.clip(mask["contrast"], -.50 if raw_sky else -.30, .50 if raw_sky else .30))
        mask["whites"] = float(np.clip(mask["whites"], -.75 if raw_sky else -.30, .75 if raw_sky else .30))
        mask["blacks"] = float(np.clip(mask["blacks"], -.50 if raw_sky else -.25, .50 if raw_sky else .25))
    if technical.black_clip > .10 and (direction.look_id or direction.method_id in {"sculpted", "graphic", "forest"} or direction.slot == "B"):
        # Low-key sources have no room for the old repeated darkening. Shape
        # their light planes and palette without consuming more shadow detail.
        data["exposure_ev"] = max(0.0, data["exposure_ev"])
        data["blacks"] = max(0.0, data["blacks"])
        data["contrast"] = min(0.0, data["contrast"])
        data["curve"]["rgb"] = [(x, max(x, y) if x <= .5 else y) for x, y in data["curve"]["rgb"]]
    from .master_studies import apply_study_guards
    return direction.model_copy(update={"recipe": apply_study_guards(Recipe.model_validate(data), critique, technical)})


def _signature(recipe: Recipe, *, full: bool = False) -> np.ndarray:
    xs = np.array([.15, .3, .5, .7, .85])
    curve = recipe.curve.rgb or [(0, 0), (1, 1)]
    ys = np.interp(xs, *np.array(curve).T)
    tone = ys + recipe.exposure_ev * .16 + recipe.contrast * (xs - .5) * .45
    wheels = recipe.color_grading
    # Hue and spatial light matter: equal saturation is not an equal palette.
    palette = []
    for band in ("shadow", "midtone", "highlight"):
        hue = np.deg2rad(getattr(wheels, band + "_hue") or 0)
        sat = (getattr(wheels, band + "_sat") or 0) * .012
        palette.extend((np.cos(hue) * sat, np.sin(hue) * sat))
    colors = [recipe.hsl.get(hue, HueShift()) for hue in ("orange", "green", "aqua", "blue")]
    hsl = [v for c in colors for v in (c.hue * .008, c.sat * .6, c.lum * .6)]
    masks = {m.target: m for m in recipe.masks}
    # Cloud density and black/white separation can change without changing EV.
    local = [getattr(masks[target], control) * weight if target in masks else 0
             for target in ("subject", "sky", "background")
             for control, weight in (("exposure_ev", .25), ("contrast", .5), ("whites", .5), ("blacks", .5))]
    signature = np.r_[tone * 2, palette, hsl, local, recipe.saturation * .6]
    if full:
        channels = [np.interp(xs, *np.array(getattr(recipe.curve, name) or [(0, 0), (1, 1)]).T) - xs
                    for name in ("r", "g", "b")]
        luminance = [(getattr(wheels, band + "_lum") or 0) * .012 for band in ("shadow", "midtone", "highlight")]
        parametric = [getattr(recipe.parametric_tone, name) * .5 for name in ("shadows", "darks", "lights", "highlights")]
        signature = np.r_[signature, np.array(channels).ravel() * 3, luminance, parametric]
    return signature


def _signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    # A localized light/palette change should not be diluted by unused controls.
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _response_schema(*, preset: bool = False) -> dict:
    # Constrain the actual decoder: a prose example alone allowed B/C to be
    # returned as placeholder strings instead of executable plans.
    number = {"type": "number"}
    recipe = {name: number for name in ("exposure_ev", "contrast", "highlights", "shadows", "whites",
              "blacks", "texture", "clarity", "dehaze", "saturation", "vibrance")}
    recipe["curve"] = {"type": "object", "properties": {"rgb": {"type": "array", "minItems": 5,
        "maxItems": 5, "items": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                                 "minItems": 2, "maxItems": 2}}}, "required": ["rgb"], "additionalProperties": False}
    recipe["hsl"] = {"type": "object", "properties": {
        hue: HueShift.model_json_schema() for hue in ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")},
        "additionalProperties": False}
    recipe["color_grading"] = ColorGradingControls.model_json_schema()
    recipe["masks"] = {"type": "array", "items": MaskAdjustment.model_json_schema(), "maxItems": 3}
    direction = {"type": "object", "properties": {
        "title": {"type": "string"}, "goal": {"type": "string"},
        "moves": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "recipe": {"type": "object", "properties": recipe, "required": [] if preset else ["curve", "hsl"], "additionalProperties": False}},
        "required": ["title", "goal", "moves", "recipe"], "additionalProperties": False}
    return {"type": "object", "$defs": {"direction": direction},
            "properties": {slot: {"$ref": "#/$defs/direction"} for slot in "ABC"},
            "required": list("ABC"), "additionalProperties": False}


def design_directions(settings: Settings, preview: Path, critique: Critique, technical: TechnicalReport,
                      *, audit_path: Path | None = None, methods=None, look_ids: list[str] | None = None) -> list[ArtDirection]:
    defaults = ([preset_direction(slot, lid, critique, technical) for slot, lid in zip("ABC", look_ids)]
                if look_ids else scene_directions(critique, technical, methods=methods))
    by_slot = {d.slot: d for d in defaults}
    proposed = {}
    knowledge = finishing_context(critique, technical, methods=methods)
    if look_ids:
        from .recipes import get_look
        from .look_routes import route_when
        knowledge["methods"] = {d.slot: {"id": d.method_id, "look_id": d.look_id,
            "title": d.title, "palette": get_look(d.look_id).description,
            "applies_when": route_when(d.look_id), "preset_amount": d.preset_amount,
            "starting_recipe": d.recipe.model_dump(exclude_defaults=True)} for d in defaults}
    audit = {"model_response": None, "issues": [], "selected": [], "knowledge": knowledge,
             "preset_ids": look_ids or [], "starting_directions": [d.model_dump() for d in defaults]}
    status = probe(settings)
    if status.get("ready"):
        prompt = f"""Design THREE distinct professional photographic finishes for THIS photo.
Scene: {critique.subject_note}. Intent: {critique.intent.goal}.
Shared technical correction already planned: {critique.correction.model_dump_json()}.
Shared local correction already planned: {json.dumps([m.model_dump() for m in critique.intent.masks], ensure_ascii=False)}.
Retain these local recovery relationships across the looks; do not erase a paired exposure/contrast/whites adjustment.
Preserve: {json.dumps(critique.intent.preserve, ensure_ascii=False)}.
Source file: {technical.source_kind}; the attached JPEG is only its viewing preview.
Measurements: mean linear luma={technical.mean_luma:.3f}, clipping={technical.highlight_clip:.3f},
near-black clipping={technical.black_clip:.3f}, low-tone area={technical.shadow_clip:.3f} (NOT clipping), saturation={technical.saturation_mean:.3f},
cyan/aqua coverage={technical.aqua_fraction:.3f}, orange coverage={technical.orange_fraction:.3f},
green coverage={technical.green_fraction:.3f}. These color areas are not semantic sky/skin masks.
Study methods selected for THIS scene, with provenance and compatible practice examples:
{json.dumps(knowledge, ensure_ascii=False)}
Use each slot's selected method as a light/tonal starting point. Method names and reference palettes
do not require recoloring this photograph; adapt their lessons to the colors and light actually present.
The per_image_studies contain individually observed references. Use their region-specific evidence
to reason about light hierarchy, tonal transitions and color competition in THIS photo. Apply a lesson
only if this photo satisfies applies_when; honor avoid and capture_dependencies. These are finished
reference images, not before/after pairs or known recipes. Never import a reference's exposure or turn
a color photo monochrome because a study was black-and-white. Localize the benefit to the current scene.
The methods above already account for this user's feedback on similar scenes. Keep their assigned method ids.
For soft_warm, LIFT midtones (curve output at x=.5 at least .52), retain existing warmth where appropriate,
and quiet only visibly competing colors; extra tint is optional. For documentary retain night density.
The three directions may share the same natural palette when light placement, tonal separation or material
has meaningful differences. Do not invent palette differences just to separate A/B/C or match a method name.
Each must have a visible photographic benefit: focal hierarchy, readable light planes, material or spatial depth.
Choose per-picture moves with that benefit; changing every color or only renaming a recipe does not achieve it.
Avoid gratuitous haze removal, neon hues, halos, clipped shadows and fake texture. Subject/sky masks are native AI selections.
You may use a 5-point RGB curve, exposure -0.4..0.28 EV, contrast -0.2..0.28, highlights -0.35..0.1,
shadows -0.15..0.22, texture -0.15..0.18, clarity -0.12..0.15, saturation -0.16..0.08, vibrance -0.1..0.16.
RAW global recovery can use exposure -0.8..0.6, highlights -0.85..0.15, shadows -0.20..0.35,
whites -0.35..0.12 and blacks -0.4..0.08. Avoid stacking these with the diagnosis correction unnecessarily.
Color grading uses actual Lightroom wheels: hue 0..360, sat 0..24 (NOT fractional), lum -12..12, blending 0..100, balance -30..30.
HSL: hue -12..12 in our units, sat -0.75..0.12 (-0.30..0.12 for natural), lum -0.18..0.15.
Design local light BEFORE color styling. Keep one or two main colors and an accent; reduce only a competing color.
Dominant cyan is not automatically dirty. Preserve blue sky, blue clothing and neutral surfaces where appropriate.
Global HSL affects every region sharing that hue, including the sky, clothing, buildings and foreground.
"background blue/green" is not a mask: naming a region does not restrict a global color adjustment to it.
For each substantial desaturation, name the visible competing region and explain in moves why its current
color intensity distracts from the picture. Color coverage, a preset or a cinematic label is not that evidence.
Preserve or recover a pale main color instead of muting it further. Check the effect on other regions with
the same hue before choosing global HSL; keep their useful color relationships intact.
Do not compensate a weak local-light plan with three color wheels or extreme HSL desaturation.
If near-black clipping exceeds 10%, do not darken B globally or crush its low curve to create a difference.
Local masks: target subject/sky/background, exposure_ev normally -0.75..0.75, contrast -0.3..0.3,
whites -0.3..0.3, blacks -0.25..0.25, highlights -0.4..0.2, shadows -0.25..0.35,
texture -0.2..0.2, saturation -0.25..0.15.
For a recoverable pale RAW sky ONLY, exposure may reach -2 EV, contrast +0.5, whites +0.75:
lower sky exposure to reveal color and use positive whites/contrast to keep clouds luminous, following the verified RAW example.
Judge the needed strength from this photo; do not leave an obviously washed-out sky essentially unchanged.
When the RAW sky is nearly white, -0.1 or -0.2 EV is often insufficient; consider -0.7..-1.6 EV
paired with contrast +.15..+.35 and whites +.2..+.6, checking cloud highlights instead of flattening the whole sky.
Do not apply this RAW recovery recipe to a JPEG, white building, snow, water or an intentional high-key sky.
Do not add a subject mask if no reliable discrete subject is selectable; a continuous hillside need not be a subject mask.
Add a sky mask only when sky is actually visible; a bright boat, water or pale building is not sky.
Preserve descriptions are inferred observations: inspect the photo and correct any color description
that contradicts the actual reference. Warm reflections can coexist with a cool city.
Return a JSON object with keys A, B, C. EACH value is a COMPLETE object with title, goal, moves and recipe.
Every move must map to controls actually present in its recipe and a region those controls can reach.
Global tone and HSL remain global. Available local selections are subject, sky and background; background is
the inverse of a subject selection, not a free selection of hills or foreground. A subject mask covers the
whole subject, not separate skin or eyes. Gradients, luminosity intersections and brush selections are unavailable.
Do not promise local hillside, skin, eye or foreground edits without a supported mask that actually isolates
the named region. If unavailable, describe the feasible global tone/sky balance, or omit the unsupported move.
Unspecified controls inherit the selected method's starting point. Explicit zeros disable a control.
Explicitly set inherited HSL/wheel controls to zero when the scene does not justify those color changes.
Titles, goals and moves in Chinese. 1..4 concise moves per direction. Omit unused settings.
Every recipe must contain its own 5-point curve.rgb, endpoints x=0 and x=1, increasing x/y.
Example of ONE recipe (design three different curves, do not copy this):
{{"contrast":0.12,"blacks":-0.06,"curve":{{"rgb":[[0,0],[0.2,0.16],[0.5,0.49],[0.8,0.83],[1,1]]}},
"color_grading":{{"shadow_hue":215,"shadow_sat":5,"highlight_hue":42,"highlight_sat":8}},
"masks":[]}}
/no_think"""
        try:
            from .critique import _extract_json
            if look_ids:
                prompt = prompt.replace("Every recipe must contain its own 5-point curve.rgb, endpoints x=0 and x=1, increasing x/y.",
                    "Return SPARSE overrides to the provided starting_recipe. Keep its curve/HSL/wheels unless a visible conflict justifies changing them.")
                prompt = prompt.replace("Example of ONE recipe (design three different curves, do not copy this):",
                    "A curve override is optional; do not replace the three selected presets with generic natural/cool/warm curves. Example syntax:")
                prompt += "\nSlots are three image-matched presets, NOT fixed brightness roles. Keep preset identity; adapt strength, light and competing colors to THIS image. Describe only the changes you actually make.\n"
            audit["prompt"] = prompt
            raw = chat_vision(settings, status["resolved_model"], prompt, base64.b64encode(preview.read_bytes()).decode("ascii"),
                              response_schema=_response_schema(preset=bool(look_ids)))
            audit["model_response"] = raw
            data = _extract_json(raw)
            items = data.get("directions", data)
            if isinstance(items, dict):
                items = [dict(value, slot=key) for key, value in items.items() if key in "ABC" and isinstance(value, dict)]
            for item in items:
                try:
                    method = knowledge["methods"][item["slot"]]["id"]
                    # Preserve the executable lesson when the model only supplies
                    # a partial recipe. Missing HSL cannot silently remove the
                    # palette design that the text promises to the user.
                    default = by_slot[item["slot"]]
                    starting = default.recipe.model_dump() if look_ids else method_recipe(method).model_dump()
                    for key, value in item.get("recipe", {}).items():
                        if key == "hsl" and isinstance(value, dict):
                            for color, controls in value.items():
                                starting["hsl"].setdefault(color, {}).update(controls)
                        elif key in {"color_grading", "curve"} and isinstance(value, dict):
                            starting[key].update(value)
                        else:
                            starting[key] = value
                    item = dict(item, recipe=starting, method_id=method,
                                look_id=default.look_id, preset_amount=default.preset_amount)
                    if look_ids:
                        item["title"] = default.title
                    direction = ArtDirection.model_validate(item)
                    direction.source = "ollama"
                    direction = bound_direction(direction, critique, technical)
                    if method == "natural" and direction.recipe.curve.rgb:
                        midpoint = float(np.interp(.5, *np.array(direction.recipe.curve.rgb).T))
                        if not .47 <= midpoint <= .56:
                            audit["issues"].append(f"{direction.slot}: natural finish had extreme midtone shift; used method curve")
                            direction.recipe.curve = method_recipe(method).curve
                    if method == "soft_warm" and direction.recipe.curve.rgb:
                        midpoint = float(np.interp(.5, *np.array(direction.recipe.curve.rgb).T))
                        if midpoint < .515:
                            audit["issues"].append(f"{direction.slot}: luminous goal had darkening curve; used method curve")
                            direction.recipe.curve = method_recipe(method).curve
                    if method == "soft_warm":
                        wheels = direction.recipe.color_grading
                        if 140 <= (wheels.shadow_hue or 0) <= 285 and (wheels.shadow_sat or 0) > 0:
                            wheels.shadow_sat = 0
                            audit["issues"].append(f"{direction.slot}: removed duplicate cool shadow tint from warm method")
                    subject = next((m for m in critique.intent.masks if m.target == "subject"), None)
                    if not look_ids and direction.slot == "B" and subject is not None and subject.exposure_ev > 0:
                        masks = {m.target: m for m in direction.recipe.masks}
                        focal = masks.get("subject", subject.model_copy(deep=True))
                        focal.exposure_ev = max(focal.exposure_ev, min(.65, subject.exposure_ev + .08))
                        masks["subject"] = focal
                        direction.recipe.masks = list(masks.values())
                    proposed[direction.slot] = direction
                except (ValueError, TypeError, KeyError) as exc:
                    audit["issues"].append(str(exc))
                    continue
        except Exception as exc:
            audit["issues"].append(f"{type(exc).__name__}: {exc}")
    result = []
    identity = _signature(Recipe())
    for default in defaults:
        candidate = proposed.get(default.slot, default)
        method = knowledge["methods"][default.slot]["id"]
        candidate = candidate.model_copy(update={
            "method_id": method,
            "recipe": retain_palette_design(candidate.recipe, method, technical),
        })
        signature = _signature(candidate.recipe)
        # Catch three differently named no-ops before spending time exporting.
        if _signature_distance(signature, identity) < .012 or any(
                _signature_distance(signature, _signature(previous.recipe)) < .018 for previous in result):
            audit["issues"].append(f"{default.slot}: near identity or duplicate; used scene direction")
            candidate = default
        result.append(bound_direction(candidate, critique, technical))
    if audit_path is not None:
        audit["selected"] = [d.model_dump() for d in result]
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
