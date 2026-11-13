"""Cinematic looks sourced from the Lightroom XMP packs in 调色素材."""

from __future__ import annotations

from .models import Look
from .xmp import parse_xmp, preset_dir

XMP_LOOKS: list[dict] = [
    {
        "id": "skam",
        "title": "SKAM 北欧",
        "title_en": "SKAM Nordic",
        "description": "冷青分割、胶片颗粒、压白拉黑，北欧青春剧。",
        "file": "skam.xmp",
        "genre_tags": ["street", "portrait", "urban", "general"],
        "moves": ["xmp-curve", "split-cyan", "grain"],
    },
    {
        "id": "tokyo_ghoul",
        "title": "东京喰种 夜",
        "title_en": "Tokyo Ghoul night",
        "description": "夜景发雾、低饱和、色相拧一下，暗部发青。",
        "file": "tokyo_ghoul.xmp",
        "genre_tags": ["night", "street", "urban"],
        "moves": ["xmp-curve", "haze", "hsl-twist"],
    },
    {
        "id": "romance_gold",
        "title": "暖金爱情",
        "title_en": "Romance gold",
        "description": "橙子拉饱和，高光金、阴影偏橄榄，韩剧暖爱。",
        "file": "romance_gold.xmp",
        "genre_tags": ["portrait", "indoor", "street"],
        "moves": ["orange-sat", "gold-highlights"],
    },
    {
        "id": "zhou_chu",
        "title": "周处除三害",
        "title_en": "Zhou Chu",
        "description": "青阴影、黄绿高光、绿叶提亮，台湾电影感。",
        "file": "zhou_chu.xmp",
        "genre_tags": ["street", "urban", "landscape"],
        "moves": ["xmp-curve", "green-lift", "split"],
    },
    {
        "id": "dune",
        "title": "沙丘胶片",
        "title_en": "Dune film",
        "description": "死青、沙黄、通道曲线压肩，沙漠电影。",
        "file": "dune.xmp",
        "genre_tags": ["landscape", "nature", "general"],
        "moves": ["rgb-curve", "kill-aqua", "khaki"],
    },
    {
        "id": "kdrama_tears",
        "title": "眼泪女王 韩系",
        "title_en": "K-drama",
        "description": "肤色橙子饱和，高光奶油金，韩剧人像。",
        "file": "kdrama_tears.xmp",
        "genre_tags": ["portrait", "indoor", "street"],
        "moves": ["orange-sat", "cream-curve"],
    },
    {
        "id": "downton",
        "title": "唐顿庄园",
        "title_en": "Downton",
        "description": "欧式暖中灰、高光略青、通道曲线，年代剧。",
        "file": "downton.xmp",
        "genre_tags": ["portrait", "indoor", "architecture", "landscape"],
        "moves": ["rgb-curve", "period-warm"],
    },
    {
        "id": "jp_drama",
        "title": "怪物 日剧",
        "title_en": "Japanese drama",
        "description": "绿转青、蓝提亮、轻颗粒，日剧空气感。",
        "file": "jp_drama.xmp",
        "genre_tags": ["street", "indoor", "portrait", "urban"],
        "moves": ["xmp-curve", "green-to-teal"],
    },
    {
        "id": "shameless",
        "title": "无耻之徒",
        "title_en": "Shameless",
        "description": "脏暖阴影、略抬黑、亮角，美剧街头。",
        "file": "shameless.xmp",
        "genre_tags": ["street", "urban", "portrait", "indoor"],
        "moves": ["warm-shadows", "edge-lift"],
    },
    {
        "id": "resident_evil",
        "title": "生化危机 末世",
        "title_en": "Resident Evil",
        "description": "压暗部、沙绿中灰、高光青，末世电影。",
        "file": "resident_evil.xmp",
        "genre_tags": ["night", "street", "urban", "landscape"],
        "moves": ["rgb-curve", "crush-shadows"],
    },
    {
        "id": "wong_kar_wai",
        "title": "繁花 王家卫",
        "title_en": "Wong Kar-wai",
        "description": "压肩、暖中灰、青高光，沪上电影。",
        "file": "wong_kar_wai.xmp",
        "genre_tags": ["street", "indoor", "night", "urban", "portrait"],
        "moves": ["rgb-curve", "period-warm"],
    },
    {
        "id": "jujutsu",
        "title": "咒术回战",
        "title_en": "Jujutsu Kaisen",
        "description": "青阴影、压紫、橙子提亮，动漫剧照感。",
        "file": "jujutsu.xmp",
        "genre_tags": ["street", "night", "urban", "general"],
        "moves": ["xmp-curve", "split-cyan"],
    },
    {
        "id": "natsume",
        "title": "夏目友人帐",
        "title_en": "Natsume",
        "description": "洋红抽干、浅灰抬黑、奶油高光，日式清淡。",
        "file": "natsume.xmp",
        "genre_tags": ["portrait", "general"],
        "moves": ["kill-magenta", "matte-curve"],
    },
    {
        "id": "unnatural",
        "title": "非自然死亡",
        "title_en": "Unnatural",
        "description": "橙子拉起来、洋红抽掉，日剧室内。",
        "file": "unnatural.xmp",
        "genre_tags": ["indoor", "portrait", "night"],
        "moves": ["orange-sat", "kill-magenta"],
    },
    {
        "id": "hanabi",
        "title": "蓝调烟火",
        "title_en": "Blue fireworks",
        "description": "中灰强青、蓝通道提亮、抽红抽绿，夜空烟火。",
        "file": "hanabi.xmp",
        "genre_tags": ["night", "urban"],
        "moves": ["blue-curve", "teal-mids"],
    },
    {
        "id": "five_cm",
        "title": "秒速五厘米",
        "title_en": "5 cm/s",
        "description": "蓝降饱和、浅抬黑、青分割，新海诚天空。",
        "file": "five_cm.xmp",
        "genre_tags": ["landscape", "portrait", "general", "street"],
        "moves": ["xmp-curve", "quiet-blue"],
    },
]


def load_xmp_looks() -> dict[str, Look]:
    root = preset_dir()
    looks: dict[str, Look] = {}
    for spec in XMP_LOOKS:
        path = root / spec["file"]
        if not path.exists():
            continue
        recipe = parse_xmp(path, apply_exposure=False)
        looks[spec["id"]] = Look(
            id=spec["id"],
            title=spec["title"],
            title_en=spec["title_en"],
            description=spec["description"],
            genre_tags=list(spec["genre_tags"]),
            moves=list(spec["moves"]),
            recipe=recipe,
            source="xmp",
        )
    return looks
