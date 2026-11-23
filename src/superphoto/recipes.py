from __future__ import annotations

from .catalog import load_xmp_looks
from .models import Grade, HueShift, Look, Recipe

# Moves distilled from published master workflows, with enough amplitude
# to read as a finished grade (not an untouched JPEG).
# - Thomas Heaton: Highlights → contrast; no global sat blast.
# - Ted Gore: darker/quieter greens; sat where light already is.
# - Colorists: warm lights, cool shadows.
# - Portrait: skin first; keep most of the shadow.

MOVES: dict[str, Recipe] = {
    "protect_highlights": Recipe(highlights=-0.20),
    "heaton_tone": Recipe(highlights=-0.18, contrast=0.16, whites=0.05, blacks=-0.06),
    "gore_greens": Recipe(
        hsl={
            "green": HueShift(hue=8, sat=-0.18, lum=-0.12),
            "yellow": HueShift(hue=3, sat=-0.08),
            "aqua": HueShift(sat=-0.10, lum=-0.06),
        }
    ),
    "warm_light_cool_shadow": Recipe(
        contrast=0.10,
        dehaze=0.08,
        vibrance=0.10,
        grade=Grade(
            shadows=(-0.040, 0.012, 0.058),
            midtones=(0.016, 0.006, -0.008),
            highlights=(0.048, 0.018, -0.022),
        ),
    ),
    "sky_settle": Recipe(
        highlights=-0.16,
        hsl={"blue": HueShift(sat=0.14, lum=-0.12), "aqua": HueShift(sat=-0.10, lum=-0.05)},
    ),
    "portra_skin": Recipe(
        temperature=12,
        vibrance=0.10,
        contrast=0.12,
        shadows=0.10,
        blacks=-0.04,
        grade=Grade(
            shadows=(0.016, 0.006, -0.012),
            midtones=(0.022, 0.008, -0.010),
            highlights=(0.048, 0.018, -0.020),
        ),
        hsl={
            "orange": HueShift(sat=-0.10, lum=0.06, hue=-3),
            "red": HueShift(sat=-0.08),
            "aqua": HueShift(sat=-0.14),
            "yellow": HueShift(sat=-0.04),
        },
    ),
    "teal_orange": Recipe(
        contrast=0.14,
        blacks=-0.12,
        saturation=-0.04,
        grade=Grade(
            shadows=(-0.048, 0.016, 0.072),
            midtones=(-0.010, 0.004, 0.016),
            highlights=(0.055, 0.018, -0.032),
        ),
    ),
    "editorial_snap": Recipe(
        contrast=0.20,
        clarity=0.16,
        whites=0.08,
        blacks=-0.10,
        vibrance=0.08,
        dehaze=0.04,
    ),
    "cool_clean": Recipe(temperature=-8, saturation=-0.05),
    "moody": Recipe(
        contrast=0.16,
        blacks=-0.20,
        shadows=-0.08,
        saturation=-0.12,
        vignette=0.22,
        dehaze=0.04,
    ),
    "fine_grain": Recipe(grain=0.10),
    "soft_vignette": Recipe(vignette=0.16),
    "presence": Recipe(clarity=0.16, dehaze=0.08, contrast=0.06),
}


LOOKS: dict[str, Look] = {
    "faithful": Look(
        id="faithful",
        title="忠实校正",
        title_en="Faithful",
        description="只做技术校正，不加味道。已经完成的 JPEG 应接近原片。",
        genre_tags=["general", "documentary", "portrait", "landscape"],
        moves=["protect_highlights"],
        recipe=Recipe(),
    ),
    "filmic_portrait_warm": Look(
        id="filmic_portrait_warm",
        title="暖调人像",
        title_en="Filmic portrait",
        description="皮肤略降饱和，高光明显偏暖，阴影基本不动。",
        genre_tags=["portrait", "street", "indoor"],
        moves=["portra_skin", "protect_highlights", "fine_grain"],
        recipe=MOVES["portra_skin"]
        .merged(MOVES["protect_highlights"])
        .merged(MOVES["fine_grain"]),
    ),
    "cinematic_teal_orange": Look(
        id="cinematic_teal_orange",
        title="电影青橙",
        title_en="Cinematic",
        description="阴影青、高光橙，压黑，带暗角。一眼能看出调色。",
        genre_tags=["street", "night", "portrait", "urban"],
        moves=["teal_orange", "soft_vignette"],
        recipe=MOVES["teal_orange"].merged(MOVES["soft_vignette"]),
    ),
    "landscape_filmic": Look(
        id="landscape_filmic",
        title="风光层次",
        title_en="Landscape depth",
        description="Gore：绿改橄榄、暖光冷影、收天空，通透和对比要够。",
        genre_tags=["landscape", "nature", "architecture"],
        moves=["gore_greens", "warm_light_cool_shadow", "sky_settle", "presence"],
        recipe=MOVES["gore_greens"]
        .merged(MOVES["warm_light_cool_shadow"])
        .merged(MOVES["sky_settle"])
        .merged(MOVES["presence"]),
    ),
    "editorial_crisp": Look(
        id="editorial_crisp",
        title="杂志硬朗",
        title_en="Editorial",
        description="中灰对比和清晰度明显抬高，略冷，杂志感。",
        genre_tags=["portrait", "product", "fashion"],
        moves=["editorial_snap", "cool_clean"],
        recipe=MOVES["editorial_snap"].merged(MOVES["cool_clean"]),
    ),
    "moody_lowkey": Look(
        id="moody_lowkey",
        title="低键情绪",
        title_en="Moody low-key",
        description="更深的黑、更收的饱和、明显暗角，不提亮。",
        genre_tags=["night", "documentary", "street", "indoor", "landscape"],
        moves=["moody", "fine_grain"],
        recipe=MOVES["moody"].merged(MOVES["fine_grain"]),
    ),
}

DEFAULT_LOOK_IDS = ("faithful", "filmic_portrait_warm", "landscape_filmic")

LOOKS = {**LOOKS, **load_xmp_looks()}

# The celebrity-imitation workflow is kept in a separate module so its
# Lightroom-oriented recipes and routing rules can evolve independently.
from .celebrity_styles import CELEBRITY_LOOKS  # noqa: E402

from .resource_knowledge import RESOURCE_LOOKS  # noqa: E402

LOOKS = {**LOOKS, **CELEBRITY_LOOKS, **RESOURCE_LOOKS}


def get_look(look_id: str) -> Look:
    if look_id in LOOKS:
        return LOOKS[look_id]
    from .masters import look_for

    look = look_for(look_id)
    if look is None:
        raise KeyError(look_id)
    return look


def allowed_look_ids() -> list[str]:
    from .masters import master_look_ids

    return list(LOOKS.keys()) + master_look_ids()


def xmp_look_ids() -> set[str]:
    return {lid for lid, look in LOOKS.items() if look.source == "xmp"}
