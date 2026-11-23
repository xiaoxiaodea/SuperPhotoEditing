#!/usr/bin/env python3
"""Download Hasselblad Masters and other major award-winning photographs.

For local study of composition, color and light. Files are stored under
data/references/awards/ with photographer / award / year metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "references" / "awards"
CATALOG = OUT / "catalog.json"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SuperPhotoEditing/0.1 "
    "(educational photography-study collector)"
)
LONG_EDGE = 2048

SKIP_NAME = re.compile(
    r"(logo|icon|sprite|avatar|banner|listing|timeline|cookie|buy-now|"
    r"placeholder|default\.png|1x1|pixel|favicon|share-image)",
    re.I,
)

HASSELBLAD_YEARS = {
    "2008": [
        "andrej-kopac",
        "august-bradley",
        "benjamin-antony-monn",
        "bronek-kozka",
        "gregor-halenda",
        "hans-strand",
        "louie-palu",
        "master-julia-fullerton-batten",
        "master-kevin-then",
        "morfi-jimenez-mercado",
    ],
    "2010": [
        "bang-peng",
        "claudio-napolitano",
        "dirk-rees",
        "joao-carlos",
        "lyle-owerko",
        "mark-holthusen",
        "mark-zibert",
        "nina-berman",
        "stephan-zirwes",
    ],
    "2012": [
        "denis-rouvre",
        "eric-wong",
        "frank-meyl",
        "joachim-schmeisser",
        "joe-felzman",
        "jon-lowenstein",
        "ken-hermann",
        "milosz-wozaczynski",
        "olivier-valsecchi",
        "tom-d-jones",
    ],
    "2014": [
        "antonio-pedrosa",
        "bara-prasilova",
        "bryn-griffiths",
        "chris-straley",
        "dmitry-ageev",
        "joseph-goh-meng-huat",
        "martin-schubert",
        "master-hengki-koentjoro",
        "paul-gisbrecht",
        "rafael-rojas",
        "rafal-maleszyk",
        "roman-jehanno",
    ],
    "2016": [
        "ali-rajabi",
        "david-peskens",
        "giorgio-cravero",
        "jake-reeder",
        "john-paul-evans",
        "katerina-belkina",
        "lars-van-de-goor",
        "natalia-evelyn-bencicova",
        "roy-rossovich",
        "swee-oh",
    ],
    "2018": [
        "ben-thomas",
        "benjamin-everett",
        "jorge-de-la-torriente",
        "kamilla-hanapova",
        "karim-iliya",
        "marcin-gizycki",
        "maria-svarbova",
        "michal-baran",
        "nabil-rosman",
        "tina-signesdottir-hult",
        "victor-hamke",
    ],
    "2021": [
        "albrecht-voss",
        "honghua-shi",
        "marcus-moller-bitsch",
        "marek-wuerfl",
        "mati-machner",
        "paul-fuentes",
        "ramon-vaquero",
        "florian-ledoux",
        "gavin-goodman",
        "yihao-wang",
        "nikolay-schegolev",
        "alice-zilberberg",
    ],
    "2023": [
        "weimin-chu",
        "tiina-itkonen",
        "panji-indra-permana",
        "tom-pitts",
        "jan-pypers",
        "efraim-baaijens",
        "efraim-baaijens-tiny-titans",
    ],
    "2026": [
        "rohan-reilly",
        "kevin-boyle",
        "svetlana-jovanovic",
        "gosse-bouma",
        "alfred-minnaar",
        "yudha-kusuma-putera",
        "panitbhand-paribatra",
        "panitbhand-paribatra-na-ayudhya",
    ],
}

GALLERY_PAGES = [
    ("hasselblad_masters", "2026", "https://www.hasselblad.com/inspiration/masters/2026/"),
    ("hasselblad_masters", "2023", "https://www.hasselblad.com/inspiration/masters/2023/"),
    ("hasselblad_masters", "2021", "https://www.hasselblad.com/inspiration/masters/2021/"),
    ("hasselblad_masters", "2026", "https://petapixel.com/2026/06/30/the-7-spectacular-winners-of-the-hasselblad-masters-2026-photo-contest/"),
    ("hasselblad_masters", "2023", "https://www.photographynews.co.uk/news/hasselblad-masters-2023-winners-gallery/"),
    ("hasselblad_masters", "2023", "https://www.all-about-photo.com/photo-articles/photo-article/1605/hasselblad-masters-2023-winners"),
    ("sony_world_photography", "2025", "https://petapixel.com/2025/04/16/the-wonderful-winners-of-sony-world-photography-awards-2025/"),
    ("sony_world_photography", "2025", "https://www.theguardian.com/artanddesign/gallery/2025/mar/13/sony-world-photography-open-award-winners-in-pictures"),
    ("wildlife_photographer_of_the_year", "2025", "https://www.nhm.ac.uk/discover/news/2025/october/wildlife-photographer-of-the-year-2025-winning-images.html"),
    ("wildlife_photographer_of_the_year", "2025", "https://www.theatlantic.com/photography/archive/2025/10/winners-wildlife-photographer-year-2025/684599/"),
    ("wildlife_photographer_of_the_year", "2025", "https://www.theguardian.com/environment/gallery/2025/oct/15/wildlife-photographer-of-the-year-awards-in-pictures"),
    ("wildlife_photographer_of_the_year", "2024", "https://www.nhm.ac.uk/discover/news/2024/october/wildlife-photographer-of-the-year-2024-winning-images.html"),
    ("wildlife_photographer_of_the_year", "2024", "https://www.theguardian.com/environment/gallery/2024/oct/09/wildlife-photographer-of-the-year-2024-winners-in-pictures"),
    ("wildlife_photographer_of_the_year", "2023", "https://www.theguardian.com/environment/gallery/2023/oct/11/wildlife-photographer-of-the-year-2023-winners-in-pictures"),
    ("world_press_photo", "2025", "https://www.theguardian.com/artanddesign/gallery/2025/apr/17/world-press-photo-contest-2025-winners-in-pictures"),
    ("world_press_photo", "2026", "https://www.theguardian.com/world/gallery/2026/apr/09/world-press-photo-2026-winners-in-pictures"),
]


def fetch(url: str, timeout: int = 40) -> bytes:
    req = Request(url, headers={"User-Agent": UA, "Accept": "*/*", "Referer": url})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_text(url: str) -> str:
    return fetch(url).decode("utf-8", "replace")


def load_catalog() -> dict:
    if CATALOG.is_file():
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    return {"items": [], "source": "award_winners"}


def save_catalog(cat: dict) -> None:
    cat["count"] = len(cat["items"])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CATALOG)


def save_jpeg(raw: bytes, dest: Path) -> tuple[int, int] | None:
    try:
        image = Image.open(BytesIO(raw))
        image = ImageOps.exif_transpose(image)
    except Exception:
        return None
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 700:
        return None
    scale = LONG_EDGE / max(w, h)
    if scale < 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=90, optimize=False, subsampling=1)
    return image.size


def looks_like_photo_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if SKIP_NAME.search(path):
        return False
    if not re.search(r"\.(jpe?g|webp)$", path.split("?")[0]):
        return False
    return True


def upscale_hasselblad(url: str) -> str:
    # cdn.hasselblad.com/f/77891/942x942/hash/file.jpg -> try 2000 on long edge
    m = re.search(r"(https://cdn\.hasselblad\.com/f/77891/)(\d+)x(\d+)(/[0-9a-f]+/.+)$", url)
    if not m:
        return url
    w, h = int(m.group(2)), int(m.group(3))
    if max(w, h) >= 1800:
        return url
    if w >= h:
        nw, nh = 2000, max(1, int(2000 * h / w))
    else:
        nh, nw = 2000, max(1, int(2000 * w / h))
    return f"{m.group(1)}{nw}x{nh}{m.group(4)}"


def extract_urls(html: str, base: str) -> list[str]:
    found: list[str] = []
    for pat in (
        r'srcset="([^"]+)"',
        r'data-src="([^"]+)"',
        r'data-srcset="([^"]+)"',
        r'src="([^"]+)"',
        r'content="(https://[^"]+\.(?:jpe?g|webp)[^"]*)"',
        r'(https://cdn\.hasselblad\.com/[^"\'\\\s]+)',
        r'(https://[^"\'\\\s]+\.(?:jpe?g|webp)(?:\?[^"\'\\\s]*)?)',
    ):
        for m in re.findall(pat, html, re.I):
            chunk = m
            # srcset: pick last (usually largest)
            if " " in chunk and "," in chunk:
                parts = [p.strip().split(" ")[0] for p in chunk.split(",") if p.strip()]
                chunk = parts[-1] if parts else chunk
            url = urljoin(base, chunk.strip())
            url = url.replace("\\u002F", "/").replace("\\/", "/")
            if looks_like_photo_url(url):
                found.append(url)
    # unique preserve order
    out, seen = [], set()
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def hasselblad_photographer_images(year: str, slug: str) -> list[str]:
    urls: list[str] = []
    for path in (
        f"https://www.hasselblad.com/page-data/inspiration/masters/{year}/{slug}/page-data.json",
        f"https://www.hasselblad.com/inspiration/masters/{year}/{slug}/",
    ):
        try:
            text = fetch_text(path)
        except Exception:
            continue
        urls.extend(extract_urls(text, path))
        time.sleep(0.15)
    return urls


def add_item(cat: dict, have: set[str], award: str, year: str, photographer: str, url: str, page: str) -> bool:
    key = url.split("?")[0]
    if key in have:
        return False
    try:
        raw = fetch(upscale_hasselblad(url) if "cdn.hasselblad.com" in url else url)
    except Exception:
        try:
            raw = fetch(url)
        except Exception:
            return False
    if len(raw) < 12000:
        return False
    digest = hashlib.sha1(raw).hexdigest()
    if digest in have:
        return False
    n = sum(1 for it in cat["items"] if it.get("award") == award) + 1
    dest = OUT / award / f"{n:04d}.jpg"
    size = save_jpeg(raw, dest)
    if not size:
        return False
    cat["items"].append(
        {
            "award": award,
            "year": year,
            "photographer": photographer,
            "source_url": url,
            "page": page,
            "file": str(dest.relative_to(OUT.parent)),
            "width": size[0],
            "height": size[1],
            "content_sha1": digest,
            "note": "award-winning photograph, local study copy",
        }
    )
    have.add(key)
    have.add(digest)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = load_catalog()
    have = {it.get("source_url", "").split("?")[0] for it in cat["items"]}
    have |= {it.get("content_sha1") for it in cat["items"] if it.get("content_sha1")}
    added = 0

    print("Hasselblad Masters photographer pages")
    for year, slugs in HASSELBLAD_YEARS.items():
        for slug in slugs:
            page = f"https://www.hasselblad.com/inspiration/masters/{year}/{slug}/"
            urls = hasselblad_photographer_images(year, slug)
            photo_urls = [
                u
                for u in urls
                if "cdn.hasselblad.com" in u
                and not SKIP_NAME.search(u)
                and "banner" not in u.lower()
                and "listing" not in u.lower()
            ]
            print(f"  {year}/{slug}: {len(photo_urls)} urls")
            for url in photo_urls:
                if add_item(cat, have, "hasselblad_masters", year, slug.replace("-", " "), url, page):
                    added += 1
                    if added % 10 == 0:
                        save_catalog(cat)
                        print(f"    saved {len(cat['items'])}")
            time.sleep(0.2)

    print("press / magazine galleries")
    for award, year, page in GALLERY_PAGES:
        try:
            html = fetch_text(page)
        except Exception as e:
            print(f"  skip {page}: {e}")
            continue
        urls = extract_urls(html, page)
        # keep likely photos, drop chrome
        urls = [
            u
            for u in urls
            if not SKIP_NAME.search(u)
            and "logo" not in u.lower()
            and "sprite" not in u.lower()
        ]
        print(f"  {award} {year}: {len(urls)} urls  {page}")
        for url in urls[:80]:
            if add_item(cat, have, award, year, "", url, page):
                added += 1
                if added % 10 == 0:
                    save_catalog(cat)
                    print(f"    saved {len(cat['items'])}")
        time.sleep(0.3)

    save_catalog(cat)
    by = {}
    for it in cat["items"]:
        by[it["award"]] = by.get(it["award"], 0) + 1
    print("done", len(cat["items"]), by)


if __name__ == "__main__":
    main()
