#!/usr/bin/env python3
"""Keep only photographs that can serve as composition / color-grade masters.

Wikimedia Featured status is technical, not taste. This pass drops paintings,
museum specimens, product shots, historical scans and flat snapshots, then
refills from stronger photographic categories using a composition/color/light
score.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "references"
CATALOG = OUT / "catalog.json"
SCORES = OUT / "scores.json"
TARGET = 1000
KEEP_SCORE = 0.54
AUTHOR_CAP = 16

_spec = importlib.util.spec_from_file_location("collect_references", ROOT / "scripts" / "collect_references.py")
collect = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(collect)

VETO_RE = re.compile(
    r"("
    r"painting|fresco|raffaello|raphael|sanzio|tapestry|engraving|woodcut|lithograph|"
    r"illumination|manuscript|jahangir|freer gallery|cleveland museum|"
    r"anatomical man|ada lovelace portrait|"
    r"MHNT|specimen|larva|dors|ventr|CUT 20|Aiolopus|Enyo gorgon|"
    r"eggshell|without Eggshell|Schrauben|Muttern|Unterlegscheiben|"
    r"Schachfiguren|Spiralbinder|Dominostein|asparagus soup|coffee beans|"
    r"B-17|Flying Fortress|NARA|page 1\.png|"
    r"coat of arms|stained glass|x[\s_-]*ray|microscop|fossil|cranium|"
    r"skull|coin |stamp |flag of|diagram|infographic|screenshot|"
    r"aerogel|pipe installation|Porsche 911|MiG-29|F-A-18|Omaha Beach|"
    r"Pentagon|wikimania|autostadt|aktstudien|fused edit|"
    r"Hatch Cooper|myjnia|car wash|dandelion clock|szyszki|"
    r"grasshopper|butterfly CUT|Colias phicomone|"
    r"by-RalfR|by RalfR|fotofluege|RalfR-|"
    r"hubble|jwst|james webb|tone[\s_-]*map|hdr|DXR"
    r")",
    re.I,
)

GENRE_SOURCES: list[tuple[str, int, list[str]]] = [
    ("landscape", 110, ["Category:Featured pictures of landscapes", "Category:Featured pictures of mountains"]),
    ("seascape_water", 55, ["Category:Featured pictures of the sea", "Category:Featured pictures of coasts"]),
    ("cityscape", 70, ["Category:Featured pictures of cityscapes"]),
    ("architecture", 80, ["Category:Featured pictures of architecture", "Category:Featured pictures of bridges"]),
    ("interior", 40, ["Category:Featured pictures of building interiors"]),
    ("portrait", 80, ["Category:Featured pictures of people", "Category:Featured pictures of portraits"]),
    ("street_documentary", 70, ["Category:Featured pictures of culture", "Category:Featured pictures of agriculture"]),
    ("travel_places", 50, ["Category:Featured pictures of gardens", "Category:Featured pictures of villages"]),
    ("wildlife", 55, ["Category:Featured pictures of mammals"]),
    ("birds", 40, ["Category:Featured pictures of birds"]),
    ("night", 40, ["Category:Featured night photography"]),
    ("black_and_white", 45, ["Category:Featured black and white photographs", "Category:Modern black and white photos"]),
    ("plants_macro", 30, ["Category:Featured pictures of plants", "Category:Featured pictures of flowers"]),
    ("sports", 25, ["Category:Featured pictures of sports"]),
    ("food", 25, ["Category:Featured pictures of food"]),
    ("still_life", 20, ["Category:Featured still-life photographs"]),
    ("weather_atmosphere", 35, ["Category:Featured pictures of natural phenomena", "Category:Featured pictures of clouds"]),
    ("aerial", 20, ["Category:Featured aerial photographs"]),
    ("astronomy_sky", 20, ["Category:Featured pictures of astronomy"]),
]


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def save_catalog(cat: dict) -> None:
    cat["count"] = len(cat["items"])
    tmp = CATALOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CATALOG)


def blob(item: dict) -> str:
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("description") or ""),
            str(item.get("file") or ""),
            str(item.get("author") or ""),
        ]
    )


def veto_reason(item: dict, image: Image.Image | None = None) -> str | None:
    text = blob(item)
    if VETO_RE.search(text):
        return "keyword"
    title = item.get("title") or ""
    if re.search(r"\b(painting|fresco|oil on|canvas)\b", title, re.I):
        return "artwork"
    if image is None:
        return None
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    white = float((y > 0.92).mean())
    black = float((y < 0.04).mean())
    chroma = arr.std(axis=2).mean()
    # White-cyc product / egg-on-paper.
    if white > 0.42 and float(y.std()) < 0.18:
        return "white-product"
    # Museum specimen on pure black, often a pair of views.
    if black > 0.48 and white < 0.02:
        return "black-studio"
    # Historical albumen / scan: low sharpness + paper yellow + low local contrast.
    gy = np.abs(np.diff(y, axis=0)).mean()
    colorful_probe = float(
        np.sqrt((arr[:, :, 0] - arr[:, :, 1]).std() ** 2 + (0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]).std() ** 2)
    )
    if chroma < 0.022 and colorful_probe < 0.05 and gy < 0.022 and float(y.mean()) > 0.40:
        return "scan"
    w, h = image.size
    if max(w / h, h / w) > 3.0:
        return "extreme-pano"
    return None


def _interest(arr: np.ndarray) -> np.ndarray:
    y = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    gx = np.abs(np.diff(y, axis=1, prepend=y[:, :1]))
    gy = np.abs(np.diff(y, axis=0, prepend=y[:1]))
    edge = gx + gy
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-5)
    mid = 1.0 - np.abs(y - 0.5) * 1.4
    mid = np.clip(mid, 0.15, 1.0)
    return edge * (0.45 + sat) * mid


def _grid_subject(interest: np.ndarray) -> tuple[float, float, float, float]:
    """Return (area, cx, cy, thirds) from the hottest 6x6 cells."""
    gh, gw = 6, 6
    hh, ww = interest.shape
    energy = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            energy[i, j] = interest[i * hh // gh : (i + 1) * hh // gh, j * ww // gw : (j + 1) * ww // gw].mean()
    thr = float(np.quantile(energy, 0.62))
    hot = energy >= max(thr, energy.mean())
    if not hot.any():
        return 1.0, 0.5, 0.5, 0.3
    rows, cols = np.where(hot)
    area = float((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) / (gh * gw))
    cy = float((rows.astype(np.float32) + 0.5).mean() / gh)
    cx = float((cols.astype(np.float32) + 0.5).mean() / gw)
    tx = min(abs(cx - 1 / 3), abs(cx - 2 / 3), abs(cx - 0.5) * 1.2)
    ty = min(abs(cy - 1 / 3), abs(cy - 2 / 3), abs(cy - 0.5) * 1.2)
    thirds = 1.0 - float(min(1.0, tx + ty))
    return area, cx, cy, thirds


def score_image(image: Image.Image) -> dict[str, float]:
    preview = image.copy()
    preview.thumbnail((512, 512))
    arr = np.asarray(preview.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-5)
    p5, p95 = np.percentile(y, [5, 95])
    spread = float(p95 - p5)
    std_y = float(y.std())
    mean_y = float(y.mean())
    mean_sat = float(sat.mean())
    p90_sat = float(np.quantile(sat, 0.90))

    rg = arr[:, :, 0] - arr[:, :, 1]
    yb = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    gray = float(arr.std(axis=2).mean()) < 0.03

    interest = _interest(arr)
    sub, cx, cy, thirds = _grid_subject(interest)
    center_dist = float(np.hypot(cx - 0.5, cy - 0.5))

    shadow = float((y < 0.12).mean())
    highlight = float((y > 0.85).mean())
    light = 0.0
    if 0.03 <= shadow <= 0.42:
        light += 0.45
    if 0.008 <= highlight <= 0.22:
        light += 0.25
    if 0.28 <= spread <= 0.88:
        light += 0.30
    light = float(np.clip(light, 0, 1))

    # Postcard / midday tourist: loud color, little atmosphere.
    snapshot = 0.0
    if colorful > 0.20 and mean_sat > 0.24 and mean_y > 0.38 and shadow < 0.06:
        snapshot = 0.75
    if colorful > 0.26 and p90_sat > 0.7:
        snapshot = max(snapshot, 0.65)
    if mean_y > 0.55 and std_y < 0.13:
        snapshot = max(snapshot, 0.7)

    contrast = float(np.clip((std_y - 0.07) / 0.17, 0, 1))

    # Restrained grade sits around colorfulness 0.08–0.18. Neon tourist is higher.
    if gray:
        color = 0.70 + 0.20 * contrast
    else:
        color = float(np.exp(-((colorful - 0.13) ** 2) / (2 * 0.07**2)))
        if 0.10 <= mean_sat <= 0.30:
            color = 0.55 * color + 0.45
        elif mean_sat > 0.36:
            color *= 0.55
        elif mean_sat < 0.07:
            color *= 0.75
        if p90_sat > 0.90 and mean_sat > 0.38:
            color *= 0.4
    color = float(np.clip(color, 0, 1))

    comp = 0.40 * thirds
    if 0.16 <= sub <= 0.72:
        comp += 0.40
    elif 0.10 <= sub < 0.16 or 0.72 < sub <= 0.88:
        comp += 0.20
    if 0.10 <= center_dist <= 0.34:
        comp += 0.20
    elif center_dist < 0.08:
        comp += 0.04
    else:
        comp += 0.08
    comp = float(np.clip(comp, 0, 1))

    lap = float(
        np.abs(y[1:-1, 1:-1] * 4 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]).mean()
    )
    sharp = float(np.clip(lap / 0.12, 0, 1))

    total = 0.32 * comp + 0.30 * color + 0.26 * light + 0.12 * contrast
    total *= 0.82 + 0.18 * sharp  # atmosphere (fog, dusk) must still pass
    total *= 1.0 - 0.72 * snapshot
    total = float(np.clip(total, 0, 1))
    return {
        "score": total,
        "comp": comp,
        "color": color,
        "light": light,
        "contrast": contrast,
        "sharp": sharp,
        "snapshot": snapshot,
        "spread": spread,
        "colorful": colorful,
        "sub": float(sub),
    }


def score_item(item: dict) -> dict:
    path = OUT / item["file"]
    row = {"file": item["file"], "genre": item.get("genre"), "title": item.get("title"), "author": item.get("author")}
    if not path.is_file():
        row.update(score=0.0, veto="missing")
        return row
    try:
        image = Image.open(path)
        image.load()
    except Exception:
        row.update(score=0.0, veto="unreadable")
        return row
    reason = veto_reason(item, image)
    metrics = score_image(image)
    row.update(metrics)
    if reason:
        row["score"] = 0.0
        row["veto"] = reason
    else:
        row["veto"] = ""
    return row


def score_all(items: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(items, 1):
        out.append(score_item(item))
        if i % 100 == 0:
            print(f"  scored {i}/{len(items)}")
    SCORES.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def keep_best(items: list[dict], scored: list[dict]) -> list[dict]:
    by_file = {s["file"]: s for s in scored}
    kept: list[dict] = []
    authors: Counter[str] = Counter()
    ranked = sorted(items, key=lambda it: by_file.get(it["file"], {}).get("score", 0), reverse=True)
    genre_count: Counter[str] = Counter()
    quotas = {g: q for g, q, _ in GENRE_SOURCES}
    quotas["featured_mix"] = 40
    for item in ranked:
        s = by_file.get(item["file"], {})
        if s.get("score", 0) < KEEP_SCORE or s.get("veto"):
            continue
        genre = item.get("genre") or "featured_mix"
        if genre_count[genre] >= quotas.get(genre, 40):
            continue
        author = re.sub(r"<[^>]+>", "", item.get("author") or "")[:80]
        if author and authors[author] >= AUTHOR_CAP:
            continue
        item = dict(item)
        item["photo_score"] = round(float(s["score"]), 3)
        item["score_detail"] = {k: round(float(s[k]), 3) for k in ("comp", "color", "light", "contrast") if k in s}
        kept.append(item)
        genre_count[genre] += 1
        if author:
            authors[author] += 1
        if len(kept) >= TARGET:
            break
    return kept


def rewrite_library(kept: list[dict]) -> list[dict]:
    staging = OUT / "_kept"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    new_items = []
    genre_idx: Counter[str] = Counter()
    for item in kept:
        genre = item.get("genre") or "featured_mix"
        genre_idx[genre] += 1
        dest_rel = f"{genre}/{genre_idx[genre]:04d}.jpg"
        dest = staging / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = OUT / item["file"]
        if src.is_file():
            shutil.copy2(src, dest)
            item = dict(item)
            item["file"] = dest_rel
            new_items.append(item)
    # Replace genre folders (keep catalog/scores/masters).
    for child in OUT.iterdir():
        if child.is_dir() and child.name != "_kept":
            shutil.rmtree(child)
    for child in staging.iterdir():
        shutil.move(str(child), str(OUT / child.name))
    shutil.rmtree(staging, ignore_errors=True)
    return new_items


def refill(kept: list[dict]) -> list[dict]:
    have_title = {it.get("title") for it in kept}
    have_sha = {it.get("sha1") or it.get("content_sha1") for it in kept}
    counts = Counter(it.get("genre") for it in kept)
    authors = Counter(re.sub(r"<[^>]+>", "", it.get("author") or "")[:80] for it in kept)

    with collect._client() as client:
        for genre, quota, cats in GENRE_SOURCES:
            need = quota - counts[genre]
            if need <= 0:
                continue
            print(f"refill {genre} need {need}")
            titles: list[str] = []
            seen: set[str] = set()
            for cat in cats:
                for title in collect.walk_files(client, cat, depth=1, limit=500):
                    if title not in seen and title not in have_title:
                        seen.add(title)
                        titles.append(title)
            rows = collect.imageinfo(client, titles[: min(len(titles), 420)], width=1600)
            cands = []
            for row in rows:
                if collect.reject_reason(row["title"], row["info"], row["categories"]):
                    continue
                info = row["info"]
                cand = {
                    "title": row["title"],
                    "url": info.get("thumburl") or info.get("url"),
                    "page": info.get("descriptionurl") or "",
                    "author": collect._meta(info, "Artist") or collect._meta(info, "Credit") or "unknown",
                    "license": collect._meta(info, "LicenseShortName") or "see source",
                    "credit": collect._meta(info, "Credit"),
                    "description": (collect._meta(info, "ImageDescription") or "")[:400],
                    "sha1": info.get("sha1") or "",
                    "genre": genre,
                }
                if VETO_RE.search(blob(cand)):
                    continue
                cands.append(cand)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            added = 0
            cursor = 0
            while added < need and cursor < len(cands):
                batch = cands[cursor : cursor + 48]
                cursor += 48
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futs = [pool.submit(collect.fetch_candidate, c) for c in batch]
                    for fut in as_completed(futs):
                        if added >= need:
                            break
                        got = fut.result()
                        if not got:
                            continue
                        cand, raw, digest = got
                        if cand["title"] in have_title or digest in have_sha:
                            continue
                        author = re.sub(r"<[^>]+>", "", cand.get("author") or "")[:80]
                        if author and authors[author] >= AUTHOR_CAP:
                            continue
                        idx = counts[genre] + 1
                        dest = OUT / genre / f"{idx:04d}.jpg"
                        size = collect.save_jpeg(raw, dest, grade_check=False)
                        if not size:
                            continue
                        try:
                            im = Image.open(dest)
                            im.load()
                        except Exception:
                            dest.unlink(missing_ok=True)
                            continue
                        if veto_reason(cand, im):
                            dest.unlink(missing_ok=True)
                            continue
                        metrics = score_image(im)
                        if metrics["score"] < KEEP_SCORE:
                            dest.unlink(missing_ok=True)
                            continue
                        item = {
                            **cand,
                            "file": f"{genre}/{idx:04d}.jpg",
                            "width": size[0],
                            "height": size[1],
                            "content_sha1": digest,
                            "photo_score": round(metrics["score"], 3),
                            "score_detail": {k: round(metrics[k], 3) for k in ("comp", "color", "light", "contrast")},
                        }
                        kept.append(item)
                        have_title.add(cand["title"])
                        have_sha.add(digest)
                        counts[genre] += 1
                        authors[author] += 1
                        added += 1
                print(f"  {genre} {counts[genre]}/{quota}  (+{added})")
    return kept


def top_up(kept: list[dict]) -> list[dict]:
    """Fill leftover slots from extra master-grade photographic categories."""
    have_title = {it.get("title") for it in kept}
    have_sha = {it.get("sha1") or it.get("content_sha1") for it in kept}
    authors = Counter(re.sub(r"<[^>]+>", "", it.get("author") or "")[:80] for it in kept)
    mix_n = sum(1 for it in kept if it.get("genre") == "master_mix")
    extra_cats = [
        "Category:Featured pictures of landscapes",
        "Category:Featured pictures of cityscapes",
        "Category:Featured pictures of architecture",
        "Category:Featured pictures of building interiors",
        "Category:Featured night photography",
        "Category:Featured pictures of mountains",
        "Category:Featured pictures of forests",
        "Category:Photographs by Diego Delso",
        "Category:Photographs by Luc Viatour",
    ]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with collect._client() as client:
        titles: list[str] = []
        seen: set[str] = set()
        for cat in extra_cats:
            for title in collect.walk_files(client, cat, depth=1, limit=400):
                if title not in seen and title not in have_title:
                    seen.add(title)
                    titles.append(title)
        rows = collect.imageinfo(client, titles[:1200], width=1600)
        cands = []
        for row in rows:
            if collect.reject_reason(row["title"], row["info"], row["categories"]):
                continue
            info = row["info"]
            cand = {
                "title": row["title"],
                "url": info.get("thumburl") or info.get("url"),
                "page": info.get("descriptionurl") or "",
                "author": collect._meta(info, "Artist") or "unknown",
                "license": collect._meta(info, "LicenseShortName") or "see source",
                "description": (collect._meta(info, "ImageDescription") or "")[:400],
                "sha1": info.get("sha1") or "",
                "genre": "master_mix",
            }
            if not VETO_RE.search(blob(cand)):
                cands.append(cand)
        cursor = 0
        while len(kept) < TARGET and cursor < len(cands):
            batch = cands[cursor : cursor + 48]
            cursor += 48
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(collect.fetch_candidate, c) for c in batch]
                for fut in as_completed(futs):
                    if len(kept) >= TARGET:
                        break
                    got = fut.result()
                    if not got:
                        continue
                    cand, raw, digest = got
                    if cand["title"] in have_title or digest in have_sha:
                        continue
                    author = re.sub(r"<[^>]+>", "", cand.get("author") or "")[:80]
                    if author and authors[author] >= AUTHOR_CAP + 8:
                        continue
                    mix_n += 1
                    dest = OUT / "master_mix" / f"{mix_n:04d}.jpg"
                    size = collect.save_jpeg(raw, dest, grade_check=False)
                    if not size:
                        mix_n -= 1
                        continue
                    try:
                        im = Image.open(dest)
                        im.load()
                    except Exception:
                        dest.unlink(missing_ok=True)
                        mix_n -= 1
                        continue
                    if veto_reason(cand, im):
                        dest.unlink(missing_ok=True)
                        mix_n -= 1
                        continue
                    metrics = score_image(im)
                    if metrics["score"] < KEEP_SCORE:
                        dest.unlink(missing_ok=True)
                        mix_n -= 1
                        continue
                    kept.append(
                        {
                            **cand,
                            "file": f"master_mix/{mix_n:04d}.jpg",
                            "width": size[0],
                            "height": size[1],
                            "content_sha1": digest,
                            "photo_score": round(metrics["score"], 3),
                        }
                    )
                    have_title.add(cand["title"])
                    have_sha.add(digest)
                    authors[author] += 1
            print(f"  top-up {len(kept)}/{TARGET}")
    return kept


def main() -> None:
    cat = load_catalog()
    items = cat["items"]
    if "--top-up" in sys.argv:
        print(f"top-up from {len(items)}")
        kept = top_up(items)
        save_catalog({**cat, "items": kept, "curated": True, "target": TARGET})
        print("done", len(kept), dict(Counter(it.get("genre") for it in kept)))
        if len(kept) < TARGET:
            raise SystemExit(f"only {len(kept)}/{TARGET} after top-up")
        return
    print(f"scoring {len(items)} images")
    scored = score_all(items)
    scores = [s["score"] for s in scored]
    vetoed = sum(1 for s in scored if s.get("veto"))
    print(
        f"score min {min(scores):.2f} p25 {np.percentile(scores,25):.2f} "
        f"median {np.median(scores):.2f} p75 {np.percentile(scores,75):.2f} "
        f"max {max(scores):.2f} veto {vetoed}"
    )
    if "--score-only" in sys.argv:
        low = sorted(scored, key=lambda s: s["score"])[:12]
        high = sorted(scored, key=lambda s: s["score"], reverse=True)[:12]
        print("LOW")
        for s in low:
            print(f"  {s['score']:.2f} {s.get('veto') or ''} {s['file']} {str(s.get('title'))[:70]}")
        print("HIGH")
        for s in high:
            print(f"  {s['score']:.2f} {s['file']} {str(s.get('title'))[:70]}")
        return

    kept = keep_best(items, scored)
    print(f"kept {len(kept)} after score/veto/author cap")
    kept = rewrite_library(kept)
    print(f"rewrote library {len(kept)}")
    save_catalog({**cat, "items": kept, "curated": True})
    if len(kept) < TARGET:
        kept = refill(kept)
        save_catalog({**cat, "items": kept, "curated": True, "target": TARGET})
    print("done", len(kept), dict(Counter(it.get("genre") for it in kept)))
    if len(kept) < TARGET:
        print(f"top-up {TARGET - len(kept)} from extra photographic categories")
        kept = top_up(kept)
        save_catalog({**cat, "items": kept, "curated": True, "target": TARGET})
        print("after top-up", len(kept), dict(Counter(it.get("genre") for it in kept)))
    if len(kept) < TARGET:
        raise SystemExit(f"only {len(kept)}/{TARGET} after curation")


if __name__ == "__main__":
    main()
