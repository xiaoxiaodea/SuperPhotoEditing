"""Local experience: which directions this user keeps, favourites and passes, by what is in the picture.

The planner first reads the picture's elements (sky, snow, sand, a figure...) and its light. With them this module
finds what was learned on similar pictures and writes it for the planner, which then chooses the directions of the
versions itself (slot A is always the natural finish). Everything stays in the local database under data/:

- the direction library: built-in directions, one palette direction per reference look, and the directions the
  planner proposed for earlier photos (table ``directions``);
- direction evidence: how often a direction was favourited, kept or passed on similar pictures. A pass says the
  treatment did not suit that picture; it is weak and scene-local, never a ban;
- taste: how the kept and favourite versions differ from the same photo's natural version A (brighter mids, a more
  matte black, warmer, more colour), over similar pictures;
- exemplars: the favourite versions of the most similar pictures, as title, intent and tone targets.

A rating is stored in ``ratings`` with method_id ``direction:<id>`` and scene_json = the picture's profile (version
2). The version's measured picture and plan come from its sidecar in ``variants.recipe_json`` (block
``experience``). Why: the per-photo vision review's score did not track quality (ref-r12/13: correlation -0.01
with closeness to the photographer's edits, no veto in 117 rounds), so taste is learned from the user's picks.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from . import tone
from .db import VERDICT_NAMES, get_directions, insert_direction
from .looks import LOOKS

PROFILE_VERSION = 2
METHOD_PREFIX = "direction:"

# What the planner may name. A closed list keeps the evidence of "snow" on one photo comparable with the next.
ELEMENTS: dict[str, str] = {
    "sky": "天空", "clouds": "云", "sun": "太阳或光源", "stars": "星空", "mountain": "山", "rock": "岩石",
    "sand": "沙丘沙地", "snow": "雪", "ice": "冰", "water": "水面", "sea": "海", "reflection": "倒影",
    "mist": "雾霭", "forest": "树林", "grass": "草地", "field": "田野", "flowers": "花", "person": "人物",
    "face": "人脸", "animal": "动物", "building": "建筑", "street": "街道", "interior": "室内", "vehicle": "车船",
    "city_lights": "城市灯光", "food": "食物", "object": "静物",
}
_ELEMENT_ALIASES = {
    "people": "person", "figure": "person", "figures": "person", "portrait": "face", "skin": "face",
    "lake": "water", "river": "water", "pond": "water", "ocean": "sea", "beach": "sea", "dune": "sand",
    "dunes": "sand", "desert": "sand", "trees": "forest", "tree": "forest", "woods": "forest", "meadow": "grass",
    "fog": "mist", "haze": "mist", "cloud": "clouds", "camel": "animal", "camels": "animal", "horse": "animal",
    "architecture": "building", "buildings": "building", "city": "building", "car": "vehicle", "boat": "vehicle",
    "lights": "city_lights", "neon": "city_lights", "mountains": "mountain", "cliff": "rock", "rocks": "rock",
}
LIGHTS: dict[str, str] = {
    "golden": "日出日落暖光", "blue_hour": "蓝调时刻", "hard_sun": "晴天硬光", "soft_daylight": "柔和日光",
    "overcast": "阴天", "backlit": "逆光", "fog": "雾天散射光", "night": "夜景灯光", "indoor": "室内光",
}


def clean_elements(raw: Any) -> list[str]:
    found: list[str] = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        key = str(item or "").strip().lower().replace(" ", "_").replace("-", "_")
        key = _ELEMENT_ALIASES.get(key, key)
        if key in ELEMENTS and key not in found:
            found.append(key)
    return found[:8]


def clean_light(raw: Any) -> str:
    key = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    return key if key in LIGHTS else ""


FEATURES = {"mean_luma": .18, "saturation_mean": .22, "temperature_bias": .16}


def profile(reading: dict[str, Any], technical: Any = None) -> dict[str, Any]:
    """What a picture is, for finding similar ones: its elements and light (read by the planner), genre, source."""
    features = {}
    for key in FEATURES:
        value = getattr(technical, key, None)
        if isinstance(value, (int, float)) and math.isfinite(value):
            features[key] = round(float(value), 4)
    return {"version": PROFILE_VERSION, "elements": clean_elements(reading.get("elements")),
            "light": clean_light(reading.get("light_kind")), "genre": str(reading.get("genre") or "general"),
            "source_kind": str(getattr(technical, "source_kind", "jpeg") or "jpeg"), "features": features}


SIMILAR_FROM = 0.2


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """0..1; 0 below SIMILAR_FROM. Shared elements count most, then the same light, genre, source and exposure."""
    if a.get("version") != PROFILE_VERSION or b.get("version") != PROFILE_VERSION:
        return 0.0
    mine, theirs = set(a.get("elements") or ()), set(b.get("elements") or ())
    shared = len(mine & theirs) / len(mine | theirs) if mine | theirs else 0.0
    if not shared and a.get("genre") != b.get("genre"):
        return 0.0
    light = (1.0 if a.get("light") and a.get("light") == b.get("light") else
             0.6 if not a.get("light") or not b.get("light") else 0.35)
    genre = 1.0 if a.get("genre") == b.get("genre") else 0.6
    source = 1.0 if a.get("source_kind") == b.get("source_kind") else 0.8
    fa, fb = a.get("features") or {}, b.get("features") or {}
    keys = [key for key in FEATURES if key in fa and key in fb]
    try:
        distance = sum(((fa[key] - fb[key]) / FEATURES[key]) ** 2 for key in keys)
    except (TypeError, ValueError):
        keys, distance = [], 0.0
    features = math.exp(-distance / 2) if keys else 0.7
    score = (0.25 + 0.75 * shared) * light * genre * source * (0.5 + 0.5 * features)
    return round(score, 4) if score >= SIMILAR_FROM else 0.0


# --------------------------------------------------------------------------
# Direction library


@dataclass(frozen=True)
class Direction:
    id: str
    title: str
    role: str                     # what the planner is told this version is (English, like the rest of the prompt)
    look: str | None = None       # a reference look the version is built on
    natural: bool = False         # faithful colour: the neutral guard and the natural-colour gate apply
    vignette_floor: float = -14.0
    origin: str = "builtin"       # builtin | look | planner


NATURAL_ROLE = (
    "Natural finish: faithful colour, the tonal range this scene deserves and a clear light hierarchy. The standard"
    " edit a careful professional would deliver; clearly better than the baseline.")
DRAMATIC_ROLE = (
    "Dramatic light: a visibly different TONAL structure from A. Deeper shadows and denser sky or"
    " surroundings, a brighter and more separated subject, stronger local dodge and burn. Its tone targets"
    " differ from A's. Not merely more saturation. The deeper shadows are the surroundings, never the subject:"
    " when the subject or a face sits in the darkest 10-20% (see the cell table), darken around it with masks"
    " and keep black and shadows near image 1. In a night or low-key frame whose darks are already near"
    " black there are no deeper shadows to find: this version keeps the frame near image 1's darkness (median"
    " within about +0.5 EV of image 1, clearly below A's) and lifts only the subject's own lit surface, under a"
    " luminance-limited mask, so it reads as an island of light in the dark, not as a lifted frame. The"
    " ground, path, pavement and walls stay at image 1's brightness or darker (burn them a little with"
    " masks); do not lift a pool on the ground: lifting the lit ground is what A does, and a version that lifts"
    " it too looks like A on a board however much darker its mid is (a pool, if any, is small and right under"
    " the lamp). The frame highlights (p90) there are that lit ground: keep them near image 1 and at least"
    " 0.7 EV below A's. A natural A lifts a night less than a stop, so a version left near image 1 is still"
    " close to A: the subject's lit surface has to end clearly brighter than A leaves it (about +1 EV over image"
    " 1), with its box on the side the lamp actually hits, not its shaded side."
    " In a bright scene (sunlit sand, snow, salt, white walls) the sunlit surface stays luminous here too"
    " (frame mid within about -0.3 EV of image 1): the depth comes from what is already in shade or"
    " colour (cliffs, shadow sides, sky), burned with masks, not from pulling the whole frame down. A"
    " subject darker than its bright surroundings (camels or people on sand, a figure against snow or sky)"
    " separates by staying darker against that luminous ground, with clarity or contrast on it; do not"
    " lift it towards the ground's brightness, which on a uniform surface makes a visible pool of light.")
PALETTE_ROLE = (
    "Colour style: A's light design carried into a distinct, tasteful PALETTE, normally built on one of the"
    " reference looks. Skin and white objects stay believable. Next to A it must read as another palette at"
    " a glance: change the colour of the LARGE areas (sky, water, ground, the overall white balance, the"
    " colour of the mid-tones), not only a split-tone in the darks or a small shift of one hue, which on a"
    " board looks like A again. With A's tone targets that takes the look at 0.8-1.0, or white balance,"
    " calibration, grading and colour targets on the large areas that visibly recolour them (mixer hue"
    " cannot: see mixer); a matte look also needs its lifted black and softened highlights written into the"
    " tone targets, or it reads as A slightly brighter.")
# The photographer's own finish in resources/ (his verified edits: p1 about .14, p50 about .70, p99 about .93,
# "亮部柔光" = clarity -40 on the brights).
AIRY_ROLE = (
    "Airy and luminous (通透明亮), the reference photographer's own finish: a lifted matte black (p1 about"
    " .12-.16), bright open mids (p50 about .62-.72), white held below paper (p99 about .92-.95), soft brights"
    " (clarity about -20 to -40 on the brightest areas through a luminance mask) and gentle, slightly lowered"
    " saturation. It must read lighter and softer than A at a glance, never washed out: the subject keeps its"
    " separation through local light, not through contrast. Not for a night or a low-key frame whose mood is its"
    " darkness.")

NATURAL = Direction("natural", "自然完成", NATURAL_ROLE, natural=True, vignette_floor=-10.0)
BUILTIN: tuple[Direction, ...] = (
    NATURAL,
    Direction("dramatic", "戏剧光影", DRAMATIC_ROLE),
    Direction("airy", "通透明亮", AIRY_ROLE),
    Direction("palette", "风格化色彩", PALETTE_ROLE),
)


def _look_direction(look_id: str, look: dict[str, Any]) -> Direction:
    return Direction(f"look:{look_id}", look["title"],
                     f"Colour style on the reference look {look_id} ({look['title']}; made for: {look['use']}). "
                     + PALETTE_ROLE.removeprefix("Colour style: "), look=look_id, origin="look")


def library(conn: sqlite3.Connection | None = None) -> dict[str, Direction]:
    items = {d.id: d for d in BUILTIN}
    items.update({f"look:{lid}": _look_direction(lid, look) for lid, look in LOOKS.items()})
    if conn is not None:
        for row in get_directions(conn):
            items.setdefault(row["id"], Direction(row["id"], row["title"], row["role"],
                                                  look=row["look_id"] if row["look_id"] in LOOKS else None,
                                                  origin="planner"))
    return items


def own_id(title: str) -> str:
    """The stable id of a planner-proposed direction: the same title on a later photo is the same direction."""
    return "own:" + hashlib.sha1((title.strip()[:24] or "新方向").encode("utf-8")).hexdigest()[:10]


def remember_direction(conn: sqlite3.Connection, title: str, role: str, look: str | None = None,
                       job_id: str | None = None) -> Direction:
    """A direction the planner proposed: kept under a stable id, so its ratings add up on later photos."""
    title = title.strip()[:24] or "新方向"
    direction_id = own_id(title)
    look = look if look in LOOKS else None
    insert_direction(conn, direction_id, title, role.strip()[:1200], look, job_id)
    return library(conn)[direction_id]


# --------------------------------------------------------------------------
# What a delivered version leaves behind


def picture(image: Image.Image | Path) -> dict[str, float]:
    """Tone anchors, colourfulness and warmth (mean R - B) of a rendered version."""
    rgb = tone.load_small(image, 320)
    stats: dict[str, float] = dict(tone.measure(rgb))
    stats["colourfulness"] = round(tone.chroma_mean(rgb), 4)
    stats["warmth"] = round(float(np.mean(rgb[..., 0] - rgb[..., 2])), 4)
    return stats


def sidecar_block(scene: dict[str, Any], direction: Direction | dict[str, Any], why: str, render: Path, *,
                  weight: float | None = None, rank: int | None = None) -> dict[str, Any]:
    """The sidecar's ``experience``: what the picture was, which direction this version took and why, and its
    measured picture (for taste). ``weight`` is the direction's learned weight here, ``rank`` its display place."""
    if isinstance(direction, Direction):
        direction = {"id": direction.id, "title": direction.title}
    block = {"profile": scene, "direction": {"id": direction["id"], "title": direction["title"],
                                             "why": why.strip()[:200]}, "picture": picture(render)}
    if weight is not None:
        block["weight"] = round(float(weight), 3)
    if rank is not None:
        block["rank"] = rank
    return block


def variant_profile(variant: dict[str, Any]) -> dict[str, Any]:
    """The picture profile a directed version was made for (its sidecar's style_workflow.method_id names the
    direction, ``direction:<id>``); {} for older versions."""
    try:
        scene = json.loads(variant.get("recipe_json") or "{}").get("experience", {}).get("profile")
    except (TypeError, ValueError, AttributeError):
        return {}
    return scene if isinstance(scene, dict) and scene.get("version") == PROFILE_VERSION else {}


# --------------------------------------------------------------------------
# Learning from the picks

# A favourite is the version the user would deliver; a pass only says this treatment did not suit that picture.
VERDICT_VALUE = {2: 2.0, 1: 1.0, -1: -0.5}
WEIGHT_SCALE = 0.6
WEIGHT_LIMIT = 1.5


def _sidecar(recipe_json: str | None) -> dict[str, Any]:
    try:
        data = json.loads(recipe_json or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def samples(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("""SELECT r.job_id, r.slot, r.verdict, r.method_id, r.scene_json,
            v.recipe_json, a.recipe_json AS anchor_json
        FROM ratings r
        LEFT JOIN variants v ON v.job_id = r.job_id AND v.slot = r.slot
        LEFT JOIN variants a ON a.job_id = r.job_id AND a.slot = 'A'
        WHERE r.verdict != 0 AND r.method_id LIKE 'direction:%'""")
    found = []
    for row in rows:
        try:
            scene = json.loads(row["scene_json"])
        except (TypeError, ValueError):
            continue
        sidecar, anchor = _sidecar(row["recipe_json"]), _sidecar(row["anchor_json"])
        plan = sidecar.get("plan") or {}
        found.append({
            "job_id": row["job_id"], "slot": row["slot"], "verdict": row["verdict"],
            "direction": row["method_id"].removeprefix(METHOD_PREFIX), "scene": scene,
            "picture": (sidecar.get("experience") or {}).get("picture"),
            "anchor": (anchor.get("experience") or {}).get("picture"),
            "plan": {"title": plan.get("title"), "intent": plan.get("intent"), "tone": plan.get("tone") or {},
                     "look": plan.get("look"), "profile": plan.get("profile")}})
    return found


def direction_evidence(conn: sqlite3.Connection, current: dict[str, Any],
                       found: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Per direction: favourite / keep / pass counts on similar pictures, their similarity-weighted score and weight."""
    evidence: dict[str, dict[str, Any]] = {}
    for sample in samples(conn) if found is None else found:
        near = similarity(current, sample["scene"])
        if not near:
            continue
        item = evidence.setdefault(sample["direction"], {"favorite": 0, "keep": 0, "pass": 0, "score": 0.0,
                                                         "shared": set()})
        item[VERDICT_NAMES[sample["verdict"]]] += 1
        item["score"] += near * VERDICT_VALUE[sample["verdict"]]
        item["shared"] |= set(current.get("elements") or ()) & set(sample["scene"].get("elements") or ())
    for item in evidence.values():
        item["weight"] = round(math.exp(max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, WEIGHT_SCALE * item["score"]))), 3)
        item["score"] = round(item["score"], 3)
        item["shared"] = sorted(item["shared"])
    return evidence


def weights(conn: sqlite3.Connection, current: dict[str, Any], directions: Iterable[str]) -> dict[str, float]:
    """Learned weight per direction for this picture; 1.0 where nothing similar was rated."""
    evidence = direction_evidence(conn, current)
    return {d: evidence.get(d, {}).get("weight", 1.0) for d in directions}


def display_order(slot_weights: dict[str, float]) -> list[str]:
    """Versions the user is likelier to keep come first; ties keep the slot order (A first)."""
    return sorted(slot_weights, key=lambda slot: (-slot_weights[slot], slot))


TASTE_KEYS = ("black", "mid", "white", "warmth")
TASTE_FROM = 1.5      # similarity-weighted count of rated versions before taste is reported


def taste(conn: sqlite3.Connection, current: dict[str, Any],
          found: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """How the kept and favourite (and passed) versions sat against the same photo's natural version A."""
    groups: dict[str, dict[str, Any]] = {}
    for sample in samples(conn) if found is None else found:
        near = similarity(current, sample["scene"])
        mine, anchor = sample.get("picture"), sample.get("anchor")
        if not near or not isinstance(mine, dict) or not isinstance(anchor, dict):
            continue
        try:
            offset = {key: float(mine[key]) - float(anchor[key]) for key in TASTE_KEYS}
            offset["colour"] = math.log(max(float(mine["colourfulness"]), 1e-3) / max(float(anchor["colourfulness"]), 1e-3))
        except (KeyError, TypeError, ValueError):
            continue
        group = groups.setdefault("kept" if sample["verdict"] > 0 else "passed", {"n": 0.0, "sum": {}})
        weight = near * (2.0 if sample["verdict"] == 2 else 1.0)
        group["n"] += weight
        for key, value in offset.items():
            group["sum"][key] = group["sum"].get(key, 0.0) + weight * value
    result = {}
    for name, group in groups.items():
        mean = {key: value / group["n"] for key, value in group["sum"].items()}
        result[name] = {"n": round(group["n"], 2), **{key: round(mean[key], 4) for key in TASTE_KEYS},
                        "colour": round(math.exp(mean["colour"]), 3)}
    return result


def _leaning(offset: dict[str, Any]) -> list[str]:
    words = []
    if offset["mid"] >= 0.03:
        words.append("brighter mids")
    elif offset["mid"] <= -0.03:
        words.append("darker mids")
    if offset["black"] >= 0.02:
        words.append("a more lifted, matte black")
    elif offset["black"] <= -0.02:
        words.append("a deeper black")
    if offset["white"] <= -0.03:
        words.append("whites held lower")
    if offset["colour"] >= 1.12:
        words.append("more colour")
    elif offset["colour"] <= 0.88:
        words.append("less colour")
    if offset["warmth"] >= 0.012:
        words.append("warmer")
    elif offset["warmth"] <= -0.012:
        words.append("cooler")
    return words


def _offset_text(offset: dict[str, Any]) -> str:
    return (f"mid {offset['mid']:+.2f}, black {offset['black']:+.2f}, white {offset['white']:+.2f},"
            f" colour x{offset['colour']:.2f}, warmth {offset['warmth']:+.3f}")


def exemplars(current: dict[str, Any], found: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    """The favourite versions of the most similar pictures."""
    favourites = [(similarity(current, s["scene"]), s) for s in found if s["verdict"] == 2]
    favourites = [(near, s) for near, s in favourites if near > 0]
    favourites.sort(key=lambda item: -item[0])
    return [{"similarity": near, **s} for near, s in favourites[:limit]]


def _tone_text(targets: dict[str, Any]) -> str:
    parts = [f"{key} {targets[key]:.2f}" for key in ("black", "mid", "white")
             if isinstance(targets.get(key), (int, float))]
    return ", ".join(parts)


def experience_block(conn: sqlite3.Connection, current: dict[str, Any],
                     directions: dict[str, Direction] | None = None) -> str:
    """Planner text: what this user picked on similar pictures. Empty when nothing similar was rated."""
    found = samples(conn)
    evidence = direction_evidence(conn, current, found)
    if not evidence:
        return ""
    directions = directions or library(conn)
    lines = ["LOCAL EXPERIENCE: this user's own picks on earlier, similar photos (kept on this machine)."]
    lines.append("Directions on similar pictures:")
    for direction_id, item in sorted(evidence.items(), key=lambda kv: -kv[1]["weight"]):
        title = directions[direction_id].title if direction_id in directions else direction_id
        counts = ", ".join(f"{name} {item[name]}" for name in ("favorite", "keep", "pass") if item[name])
        shared = f"; shared elements: {', '.join(item['shared'])}" if item["shared"] else ""
        lines.append(f"- {title} ({direction_id}): {counts}{shared}; weight {item['weight']:.2f}")
    leaning = taste(conn, current, found)
    kept = leaning.get("kept")
    if kept and kept["n"] >= TASTE_FROM:
        words = _leaning(kept)
        lines.append(f"Kept and favourite versions sat against the same photo's natural A at {_offset_text(kept)}"
                     + (f": this user leans to {', '.join(words)}." if words else "."))
    passed = leaning.get("passed")
    if passed and passed["n"] >= TASTE_FROM:
        words = _leaning(passed)
        lines.append(f"Passed versions sat at {_offset_text(passed)} against A"
                     + (f" ({', '.join(words)} did not suit these pictures)." if words else "."))
    best = exemplars(current, found)
    if best:
        lines.append("Favourites on the most similar pictures:")
        for item in best:
            plan = item["plan"]
            look = plan.get("look") or {}
            look_text = f", look {look.get('id')} {look.get('strength', '')}".rstrip() if look.get("id") else ""
            profile = plan.get("profile") or {}
            if profile.get("id"):
                look_text += f", profile {profile['id']} {profile.get('amount', '')}".rstrip()
            lines.append(f"- [{item['similarity']:.2f}] {plan.get('title') or item['direction']} ({item['direction']}):"
                         f" {plan.get('intent') or ''} (tone {_tone_text(plan.get('tone') or {})}{look_text})")
    lines.append("Use this when choosing the directions: give a favoured direction a version when this picture's"
                 " elements allow it; choose a direction passed on similar pictures again only for a reason in this"
                 " picture. A pass is about that picture, not a ban. Keep at least one direction this user has not"
                 " rated on pictures like this, so their taste can still be found.")
    return "\n".join(lines)
