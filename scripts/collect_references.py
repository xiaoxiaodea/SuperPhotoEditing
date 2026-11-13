#!/usr/bin/env python3
"""Collect 1000 legally reusable, textbook-grade photographs.

True Magnum / Lindbergh / McCurry files cannot be scraped. This collector
takes Wikimedia Commons Featured Pictures (and Quality Images as fill):
community-judged photographs with a natural, non-overcooked grade, covering
every major photographic subject.

Usage:
    uv run python scripts/collect_references.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "references"
CATALOG_PATH = OUT / "catalog.json"
MASTERS_PATH = OUT / "masters.json"
TARGET = 1000
LONG_EDGE = 2048
USER_AGENT = (
    "SuperPhotoEditing/0.1 (https://github.com/dextermosby/SuperPhotoEditing; "
    "educational color-grade reference collector)"
)
API = "https://commons.wikimedia.org/w/api.php"

# Skip scientific composites, maps, overcooked contest HDR, non-photographs.
SKIP_RE = re.compile(
    r"("
    r"hdr|tone[\s_-]*map|photomanipulat|photo\s*manip|cgi|3d[\s_-]*render|"
    r"satellite|microscop|x[\s_-]*ray|diagram|infographic|screenshot|"
    r"engraving|woodcut|fresco|lithograph|etching|illustration|drawing of|"
    r"coat of arms|qr code|logo of|flag of|map of|chart of|graph of|"
    r"hubble|jwst|james webb|spitzer|chandra|eso_|eso |nasa/esa|"
    r"computer[- ]generated|raytrac|digital art|ai[- ]generated"
    r")",
    re.I,
)
SKIP_EXT = {".svg", ".gif", ".pdf", ".djvu", ".ogg", ".ogv", ".webm", ".stl", ".wav"}

# Genre quotas. Surplus from a short genre is redistributed at the end.
GENRES: list[tuple[str, int, list[str]]] = [
    (
        "landscape",
        95,
        [
            "Category:Featured pictures of landscapes",
            "Category:Featured pictures of mountains",
            "Category:Featured pictures of forests",
            "Category:Featured pictures of deserts",
            "Category:Featured pictures of waterfalls",
            "Category:Featured pictures of lakes",
        ],
    ),
    (
        "seascape_water",
        50,
        [
            "Category:Featured pictures of the sea",
            "Category:Featured pictures of coasts",
            "Category:Featured pictures of rivers",
            "Category:Featured pictures of water",
        ],
    ),
    (
        "cityscape",
        70,
        ["Category:Featured pictures of cityscapes"],
    ),
    (
        "architecture",
        70,
        [
            "Category:Featured pictures of architecture",
            "Category:Featured pictures of bridges",
            "Category:Featured pictures of castles and fortifications",
            "Category:Featured pictures of religious buildings",
            "Category:Featured pictures of towers",
        ],
    ),
    (
        "interior",
        35,
        [
            "Category:Featured pictures of building interiors",
            "Category:Featured pictures of church interiors",
            "Category:Featured pictures of palaces",
        ],
    ),
    (
        "portrait",
        80,
        [
            "Category:Featured pictures of people",
            "Category:Featured pictures of portraits",
            "Category:Quality images of portraits",
            "Category:Quality images of people",
        ],
    ),
    (
        "street_documentary",
        70,
        [
            "Category:Featured pictures of culture",
            "Category:Featured pictures of agriculture",
            "Category:Featured pictures of working people",
            "Category:Featured pictures of industry",
        ],
    ),
    (
        "travel_places",
        55,
        [
            "Category:Featured pictures of gardens",
            "Category:Featured pictures of parks",
            "Category:Featured pictures of villages",
            "Category:Featured pictures of archaeology",
            "Category:Wiki Loves Earth",
        ],
    ),
    (
        "wildlife",
        70,
        [
            "Category:Featured pictures of mammals",
            "Category:Featured pictures of animals",
            "Category:Featured pictures of reptiles",
            "Category:Featured pictures of marine life",
        ],
    ),
    (
        "birds",
        45,
        ["Category:Featured pictures of birds"],
    ),
    (
        "plants_macro",
        40,
        [
            "Category:Featured pictures of plants",
            "Category:Featured pictures of flowers",
            "Category:Featured pictures of fungi",
        ],
    ),
    (
        "night",
        50,
        ["Category:Featured night photography"],
    ),
    (
        "black_and_white",
        50,
        [
            "Category:Featured black and white photographs",
            "Category:Modern black and white photos",
            "Category:Featured pictures by photographic technique",
        ],
    ),
    (
        "sports",
        40,
        ["Category:Featured pictures of sports"],
    ),
    (
        "food",
        35,
        [
            "Category:Featured pictures of food",
            "Category:Featured pictures of food and drink",
            "Category:Quality images of food",
        ],
    ),
    (
        "still_life",
        35,
        [
            "Category:Featured still-life photographs",
            "Category:Featured pictures of objects",
        ],
    ),
    (
        "weather_atmosphere",
        40,
        [
            "Category:Featured pictures of natural phenomena",
            "Category:Featured pictures of clouds",
            "Category:Featured pictures of weather",
            "Category:Featured pictures of rainbows",
            "Category:Featured pictures of lightning",
        ],
    ),
    (
        "aerial",
        25,
        [
            "Category:Featured aerial photographs",
            "Category:Quality images of aerial photographs",
        ],
    ),
    (
        "astronomy_sky",
        20,
        ["Category:Featured pictures of astronomy"],
    ),
]


MASTERS = {
    "note": (
        "Canonical living/modern masters whose grade we want to emulate. "
        "Their copyrighted catalogs cannot be bulk-downloaded; the image set "
        "is Wikimedia Featured/Quality photographs with a comparable textbook grade."
    ),
    "genres": {
        "landscape": [
            "Michael Kenna",
            "Charlie Waite",
            "Art Wolfe",
            "Frans Lanting",
            "Sebastião Salgado (Genesis)",
            "Joe Cornish",
            "David Noton",
            "Nigel Danson",
        ],
        "seascape_water": ["Michael Kenna", "Hiroshi Sugimoto", "David Doubilet"],
        "cityscape": ["Fan Ho", "Fred Herzog", "Saul Leiter", "Edward Burtynsky"],
        "architecture": ["Julius Shulman", "Iwan Baan", "Hélène Binet", "Ezra Stoller"],
        "interior": ["Hélène Binet", "Candida Höfer", "Julius Shulman"],
        "portrait": [
            "Peter Lindbergh",
            "Nadav Kander",
            "Platon",
            "Irving Penn",
            "Richard Avedon",
            "Dan Winters",
            "Paolo Roversi",
        ],
        "street_documentary": [
            "Alex Webb",
            "Joel Meyerowitz",
            "Harry Gruyaert",
            "Saul Leiter",
            "Trent Parke",
            "Alec Soth",
            "Steve McCurry",
        ],
        "travel_places": ["Steve McCurry", "Diego Delso", "Cristina Garcia Rodero"],
        "wildlife": [
            "Paul Nicklen",
            "Cristina Mittermeier",
            "Nick Brandt",
            "Frans Lanting",
            "Thomas D. Mangelsen",
        ],
        "birds": ["Art Wolfe", "Tim Laman", "Lars Petersson"],
        "plants_macro": ["Karl Blossfeldt", "Imogen Cunningham", "Robert Mapplethorpe (flowers)"],
        "night": ["Todd Hido", "Brassai", "Rut Blees Luxemburg"],
        "black_and_white": [
            "Peter Lindbergh",
            "Sebastião Salgado",
            "Don McCullin",
            "Daido Moriyama",
            "Michael Kenna",
        ],
        "sports": ["Walter Iooss", "Neil Leifer", "Adam Pretty"],
        "food": ["Irving Penn", "Marcus Nilsson", "Penny De Los Santos"],
        "still_life": ["Irving Penn", "Edward Weston", "Laura Letinsky"],
        "weather_atmosphere": ["Sebastião Salgado", "Edward Burtynsky", "Nadav Kander"],
        "aerial": ["Edward Burtynsky", "Yann Arthus-Bertrand", "George Steinmetz"],
        "astronomy_sky": ["Babak Tafreshi", "Max Alexander"],
        "fashion_editorial": ["Peter Lindbergh", "Patrick Demarchelier", "Herb Ritts"],
    },
}


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=20.0),
        follow_redirects=True,
    )


def wiki_get(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    payload = {"format": "json", "formatversion": "2", **params}
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            response = client.get(API, params=payload)
            if response.status_code in {429, 503, 502}:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"wikimedia api failed: {last_err}")


def iter_category_titles(client: httpx.Client, category: str, cmtype: str) -> list[str]:
    titles: list[str] = []
    cont: str | None = None
    while True:
        params: dict[str, Any] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": cmtype,
            "cmlimit": 500,
        }
        if cmtype == "file":
            params["cmnamespace"] = 6
        elif cmtype == "subcat":
            params["cmnamespace"] = 14
        if cont:
            params["cmcontinue"] = cont
        data = wiki_get(client, params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members if m.get("title"))
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.05)
    return titles


def walk_files(client: httpx.Client, root_cat: str, depth: int = 2, limit: int = 700) -> list[str]:
    seen_cats = {root_cat}
    files: list[str] = []
    queue = [(root_cat, 0)]
    while queue and len(files) < limit:
        cat, level = queue.pop(0)
        try:
            files.extend(iter_category_titles(client, cat, "file"))
            if level < depth and len(files) < limit:
                for sub in iter_category_titles(client, cat, "subcat"):
                    if sub not in seen_cats:
                        seen_cats.add(sub)
                        queue.append((sub, level + 1))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {cat}: {exc}")
        time.sleep(0.05)
    out, seen = [], set()
    for title in files:
        if title not in seen:
            seen.add(title)
            out.append(title)
        if len(out) >= limit:
            break
    return out


def imageinfo(client: httpx.Client, titles: list[str], width: int | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    thumb = width or LONG_EDGE
    for i in range(0, len(titles), 40):
        chunk = titles[i : i + 40]
        data = wiki_get(
            client,
            {
                "action": "query",
                "prop": "imageinfo|categories",
                "titles": "|".join(chunk),
                "iiprop": "url|size|mime|extmetadata|dimensions|sha1",
                "iiurlwidth": thumb,
                "cllimit": 80,
            },
        )
        for page in data.get("query", {}).get("pages", []):
            if page.get("missing") or page.get("ns") != 6:
                continue
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            results.append({"title": page.get("title", ""), "info": info, "categories": page.get("categories") or []})
        time.sleep(0.05)
    return results


def _meta(info: dict[str, Any], key: str) -> str:
    ext = info.get("extmetadata") or {}
    node = ext.get(key) or {}
    val = node.get("value") if isinstance(node, dict) else node
    if not val:
        return ""
    return re.sub(r"<[^>]+>", "", str(val)).strip()


def reject_reason(title: str, info: dict[str, Any], categories: list[dict[str, Any]]) -> str | None:
    mime = (info.get("mime") or "").lower()
    if not mime.startswith("image/"):
        return "not-image"
    if mime in {"image/svg+xml", "image/gif", "image/tiff"} and mime == "image/svg+xml":
        return "svg"
    url = info.get("url") or info.get("thumburl") or ""
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix in SKIP_EXT:
        return "ext"
    blob = " ".join(
        [
            title,
            _meta(info, "ImageDescription"),
            _meta(info, "ObjectName"),
            " ".join(c.get("title", "") for c in categories),
        ]
    )
    if SKIP_RE.search(blob):
        return "skip-keyword"
    cats = " ".join(c.get("title", "") for c in categories).lower()
    if any(k in cats for k in ("diagram", "map", "satellite", "tone-mapped", "cgi", "computer-generated")):
        return "skip-category"
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    if min(w, h) < 1000:
        return "too-small"
    if w and h and max(w / h, h / w) > 3.2:
        return "extreme-pano"
    license_name = _meta(info, "LicenseShortName") or _meta(info, "License")
    if license_name and re.search(r"fair use|noncommercial|cc-by-nc|copyright", license_name, re.I):
        if "cc-by" in license_name.lower() and "nc" not in license_name.lower():
            return None
        if re.search(r"fair use|noncommercial|cc-by-nc", license_name, re.I):
            return "license"
    return None


def too_cooked(image: Image.Image) -> bool:
    """Reject obviously over-graded frames: neon sat, crushed blacks, clipped sunbursts."""
    preview = image.copy()
    preview.thumbnail((512, 512))
    rgb = np.asarray(preview.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.size == 0:
        return True
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / (mx + 1e-5)
    # Commons Featured Pictures already exclude contest HDR. Only drop neon outliers.
    if float(sat.mean()) > 0.58 and float(np.quantile(sat, 0.90)) > 0.97:
        return True
    return False


def save_jpeg(raw: bytes, dest: Path, grade_check: bool = True) -> tuple[int, int] | None:
    try:
        image = Image.open(BytesIO(raw))
        image = ImageOps.exif_transpose(image)
    except Exception:
        return None
    if image.mode not in {"RGB", "L", "RGBA"}:
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (12, 12, 12))
        bg.paste(image, mask=image.split()[-1])
        image = bg
    elif image.mode == "L":
        image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 640:
        return None
    scale = LONG_EDGE / max(w, h)
    if scale < 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    if grade_check and too_cooked(image):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=88, optimize=False, subsampling=1)
    return image.size


def load_catalog() -> dict[str, Any]:
    if CATALOG_PATH.is_file():
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {"target": TARGET, "items": []}


def write_catalog(catalog: dict[str, Any]) -> None:
    catalog["count"] = len(catalog["items"])
    tmp = CATALOG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CATALOG_PATH)


def download_one(client: httpx.Client, url: str) -> bytes | None:
    try:
        response = client.get(url)
        if response.status_code != 200:
            return None
        return response.content
    except Exception:
        return None


def fetch_candidate(cand: dict[str, Any]) -> tuple[dict[str, Any], bytes, str] | None:
    url = cand.get("url") or ""
    if not url:
        return None
    for attempt in range(4):
        try:
            with httpx.Client(
                headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
                timeout=httpx.Timeout(90.0, connect=20.0, read=90.0),
                follow_redirects=True,
            ) as client:
                response = client.get(url)
            if response.status_code in {429, 502, 503}:
                time.sleep(1.2 * (attempt + 1))
                continue
            if response.status_code != 200:
                return None
            ctype = (response.headers.get("content-type") or "").lower()
            raw = response.content
            if "image" not in ctype or len(raw) < 6000:
                return None
            return cand, raw, hashlib.sha1(raw).hexdigest()
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return None


def search_files(client: httpx.Client, query: str, limit: int = 250) -> list[str]:
    titles: list[str] = []
    offset = 0
    while len(titles) < limit:
        data = wiki_get(
            client,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 6,
                "srlimit": 50,
                "sroffset": offset,
            },
        )
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        for hit in hits:
            title = hit.get("title")
            if title:
                titles.append(title)
        offset += len(hits)
        if not data.get("continue"):
            break
        time.sleep(0.05)
    return titles


def collect_candidates(client: httpx.Client, genre: str, categories: list[str], want: int) -> list[dict[str, Any]]:
    titles: list[str] = []
    seen: set[str] = set()
    for cat in categories:
        print(f"  listing {cat}")
        for title in walk_files(client, cat, depth=2):
            if title not in seen:
                seen.add(title)
                titles.append(title)
        if len(titles) >= want * 6:
            break
    if len(titles) < want * 2:
        query = {
            "landscape": "incategory:\"Featured pictures of landscapes\"",
            "seascape_water": "incategory:\"Featured pictures on Wikimedia Commons\" (sea OR coast OR ocean OR lake)",
            "interior": "incategory:\"Featured pictures on Wikimedia Commons\" interior",
            "travel_places": "incategory:\"Featured pictures on Wikimedia Commons\" (village OR garden OR temple OR market)",
            "food": "incategory:\"Featured pictures on Wikimedia Commons\" food",
            "still_life": "incategory:\"Featured pictures on Wikimedia Commons\" still-life",
            "sports": "incategory:\"Featured pictures of sports\"",
            "aerial": "incategory:\"Featured pictures on Wikimedia Commons\" aerial",
            "weather_atmosphere": "incategory:\"Featured pictures on Wikimedia Commons\" (storm OR fog OR lightning OR cloud)",
            "black_and_white": "incategory:\"Featured pictures on Wikimedia Commons\" \"black and white\" photograph",
            "astronomy_sky": "incategory:\"Featured night photography\" (milky OR stars)",
        }.get(genre)
        if query:
            print(f"  search fallback for {genre}")
            titles.extend(search_files(client, query, limit=max(want * 5, 200)))
    print(f"  {genre}: {len(titles)} candidate files")
    infos = imageinfo(client, titles[: max(want * 8, 400)])
    kept: list[dict[str, Any]] = []
    for row in infos:
        reason = reject_reason(row["title"], row["info"], row["categories"])
        if reason:
            continue
        info = row["info"]
        kept.append(
            {
                "title": row["title"],
                "url": info.get("thumburl") or info.get("url"),
                "page": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{quote(row['title'].replace(' ', '_'))}",
                "author": _meta(info, "Artist") or _meta(info, "Credit") or "unknown",
                "license": _meta(info, "LicenseShortName") or _meta(info, "License") or "see source",
                "credit": _meta(info, "Credit"),
                "description": (_meta(info, "ImageDescription") or "")[:400],
                "sha1": info.get("sha1") or "",
                "orig_w": info.get("width"),
                "orig_h": info.get("height"),
            }
        )
    print(f"  {genre}: {len(kept)} after metadata filters")
    return kept


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MASTERS_PATH.write_text(json.dumps(MASTERS, ensure_ascii=False, indent=2), encoding="utf-8")
    catalog = load_catalog()
    have_sha = {it.get("sha1") for it in catalog["items"] if it.get("sha1")}
    have_title = {it.get("title") for it in catalog["items"]}
    counts: dict[str, int] = {}
    for it in catalog["items"]:
        counts[it["genre"]] = counts.get(it["genre"], 0) + 1
    print(f"resume {len(catalog['items'])}/{TARGET}")
    fill_only = "--fill" in __import__("sys").argv

    with _client() as client:
        leftovers: list[tuple[str, dict[str, Any]]] = []
        genres = [] if fill_only else GENRES
        for genre, quota, cats in genres:
            need = max(0, quota - counts.get(genre, 0))
            if need == 0:
                print(f"{genre}: already {counts.get(genre, 0)}")
                continue
            candidates = collect_candidates(client, genre, cats, need)
            queued: list[dict[str, Any]] = []
            for cand in candidates:
                if cand["title"] in have_title or (cand["sha1"] and cand["sha1"] in have_sha):
                    continue
                author_key = re.sub(r"<[^>]+>", "", cand.get("author") or "")[:80]
                author_count = sum(
                    1
                    for it in catalog["items"]
                    if it.get("genre") == genre and (it.get("author") or "")[:80] == author_key
                ) + sum(1 for q in queued if (q.get("author") or "")[:80] == author_key)
                if author_key and author_count >= 40:
                    continue
                if cand.get("url"):
                    queued.append(cand)
            extras = queued[need * 3 :]
            queued = queued[: need * 3]
            leftovers.extend((genre, c) for c in extras[:80])
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(fetch_candidate, cand) for cand in queued]
                for fut in as_completed(futures):
                    if counts.get(genre, 0) >= quota:
                        continue
                    got = fut.result()
                    if not got:
                        continue
                    cand, raw, digest = got
                    if cand["title"] in have_title or digest in have_sha:
                        continue
                    idx = counts.get(genre, 0) + 1
                    dest = OUT / genre / f"{idx:04d}.jpg"
                    size = save_jpeg(raw, dest)
                    if not size:
                        continue
                    catalog["items"].append(
                        {
                            **cand,
                            "genre": genre,
                            "file": str(dest.relative_to(OUT)),
                            "width": size[0],
                            "height": size[1],
                            "content_sha1": digest,
                        }
                    )
                    have_sha.add(cand["sha1"] or digest)
                    have_sha.add(digest)
                    have_title.add(cand["title"])
                    counts[genre] = counts.get(genre, 0) + 1
                    if len(catalog["items"]) % 20 == 0:
                        write_catalog(catalog)
                        print(f"  saved {len(catalog['items'])}  {genre} {counts[genre]}/{quota}")
            print(f"{genre}: {counts.get(genre, 0)}/{quota}")
            write_catalog(catalog)

        # Fill remaining slots from leftover featured candidates of any genre.
        if len(catalog["items"]) < TARGET:
            print(f"filling remaining {TARGET - len(catalog['items'])} from leftovers + featured pool")
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(fetch_candidate, cand) for _, cand in leftovers[:400]]
                genre_of = {id(cand): genre for genre, cand in leftovers[:400]}
                for fut in as_completed(futures):
                    if len(catalog["items"]) >= TARGET:
                        break
                    got = fut.result()
                    if not got:
                        continue
                    cand, raw, digest = got
                    if cand["title"] in have_title or digest in have_sha:
                        continue
                    genre = "featured_mix"
                    dest = OUT / genre / f"{sum(1 for it in catalog['items'] if it['genre']==genre) + 1:04d}.jpg"
                    size = save_jpeg(raw, dest)
                    if not size:
                        continue
                    catalog["items"].append(
                        {
                            **cand,
                            "genre": genre,
                            "file": str(dest.relative_to(OUT)),
                            "width": size[0],
                            "height": size[1],
                            "content_sha1": digest,
                        }
                    )
                    have_sha.add(digest)
                    have_title.add(cand["title"])
                    counts[genre] = counts.get(genre, 0) + 1
            write_catalog(catalog)
            extra_titles: list[str] = []
            for cat, depth, limit in (
                ("Category:Featured pictures on Wikimedia Commons", 0, 9000),
                ("Category:Quality images of photographs", 0, 2000),
                ("Category:Pictures of the Year", 2, 600),
                ("Category:Valued images of photographs", 0, 1500),
            ):
                extra_titles.extend(walk_files(client, cat, depth=depth, limit=limit))
            extra_titles = [t for t in extra_titles if t not in have_title]
            print(f"  unused titles {len(extra_titles)}")
            extra = imageinfo(client, extra_titles[:3500], width=1280)
            fill_pool: list[dict[str, Any]] = []
            seen_fill: set[str] = set()
            for row in extra:
                if row["title"] in seen_fill:
                    continue
                seen_fill.add(row["title"])
                if reject_reason(row["title"], row["info"], row["categories"]):
                    continue
                info = row["info"]
                fill_pool.append(
                    {
                        "title": row["title"],
                        "url": info.get("thumburl") or info.get("url"),
                        "page": info.get("descriptionurl") or "",
                        "author": _meta(info, "Artist") or "unknown",
                        "license": _meta(info, "LicenseShortName") or "see source",
                        "credit": _meta(info, "Credit"),
                        "description": (_meta(info, "ImageDescription") or "")[:400],
                        "sha1": info.get("sha1") or "",
                        "genre_hint": "featured",
                    }
                )
            print(f"  fill pool {len(fill_pool)}")
            cursor = 0
            while len(catalog["items"]) < TARGET and cursor < len(fill_pool):
                batch = fill_pool[cursor : cursor + 60]
                cursor += 60
                ok = 0
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(fetch_candidate, cand) for cand in batch]
                    for fut in as_completed(futures):
                        if len(catalog["items"]) >= TARGET:
                            break
                        got = fut.result()
                        if not got:
                            continue
                        cand, raw, digest = got
                        if cand["title"] in have_title or digest in have_sha:
                            continue
                        fill_i = sum(1 for it in catalog["items"] if it["genre"] == "featured_mix") + 1
                        dest = OUT / "featured_mix" / f"{fill_i:04d}.jpg"
                        size = save_jpeg(raw, dest, grade_check=False)
                        if not size:
                            continue
                        catalog["items"].append(
                            {
                                **cand,
                                "genre": "featured_mix",
                                "file": str(dest.relative_to(OUT)),
                                "width": size[0],
                                "height": size[1],
                                "content_sha1": digest,
                            }
                        )
                        have_sha.add(digest)
                        have_title.add(cand["title"])
                        ok += 1
                write_catalog(catalog)
                print(f"  fill {len(catalog['items'])}/{TARGET}  batch_ok={ok}  cursor={cursor}")

    write_catalog(catalog)
    by = {}
    for it in catalog["items"]:
        by[it["genre"]] = by.get(it["genre"], 0) + 1
    print("done", len(catalog["items"]), "by genre:", json.dumps(by, ensure_ascii=False, indent=2))
    if len(catalog["items"]) < TARGET:
        raise SystemExit(f"only collected {len(catalog['items'])}/{TARGET}")


if __name__ == "__main__":
    main()
