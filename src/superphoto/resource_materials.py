"""Offline inventory of the user's reference files; archive contents remain data.

Nothing is registered as an executable look by this module. Full preset values
are written only to the explicitly requested local study report, never bundled
with the app. Reviewed lessons are maintained separately in resource_knowledge.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

CRS = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
MAX_XMP_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_XMP_BYTES = 32 * 1024 * 1024
FAMILY_WORDS = (
    ("CINE", "cine"), ("Winter", "snow"), ("雪景", "snow"), ("沙漠", "desert"),
    ("古风", "period"), ("Michael", "forest"), ("Tim", "clean_cinema"),
    ("KIRINOMA", "twilight"), ("Maurice", "vintage"), ("Tina", "muted_emotion"),
    ("品川", "lowkey"), ("滨田", "pastel_portrait"), ("迷雾", "fog"),
    ("晚霞", "sunset"), ("风光", "landscape"), ("街拍", "street"),
    ("人像", "portrait"), ("植物", "foliage"), ("动物", "animal"),
    ("夜景", "night"), ("静物", "still_life"), ("颜色分级", "grading"),
    ("颗粒", "grain"), ("蒙版", "mask"),
)


def material_family(source: str) -> str:
    return next((family for word, family in FAMILY_WORDS if word.casefold() in source.casefold()), "workflow")


def inspect_xmp(payload: bytes, source: str, filename: str) -> dict:
    if len(payload) > MAX_XMP_BYTES:
        raise ValueError("XMP exceeds the study size limit")
    # We have no use for DTDs, external references or entity expansion here.
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper() or b"\x00" in payload:
        raise ValueError("Only ordinary UTF-8 XMP without DTD/entity declarations is supported")
    root = ET.fromstring(payload)
    description = root.find(".//" + RDF + "Description")
    if description is None:
        raise ValueError("XMP has no RDF description")
    raw = {key[len(CRS):]: value for key, value in description.attrib.items() if key.startswith(CRS)}
    for child in description:
        if child.tag.startswith(CRS) and child.text and child.text.strip():
            raw[child.tag[len(CRS):]] = child.text.strip()
    values = {}
    for key, value in raw.items():
        try:
            number = float(value)
            if math.isfinite(number):
                values[key] = number
        except ValueError:
            continue
    name = description.find(CRS + "Name")
    title = " ".join(text.strip() for text in name.itertext() if text.strip()) if name is not None else filename
    local_masks = any("Mask" in element.tag or "Correction" in element.tag for element in description.iter())
    role = "look"
    if any(word in title for word in ("说明", "先使用", "再使用", "最后使用")):
        role = "instructions"
    elif "清除" in title:
        role = "reset"
    elif local_masks or "蒙版" in source:
        role = "mask"
    elif any(word in title for word in ("基础", "基调")):
        role = "baseline"
    elif "颜色分级" in source or "颜色分级" in title:
        role = "grading"
    elif "颗粒" in source or "颗粒" in title:
        role = "grain"
    curves = {}
    for key in ("ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"):
        element = description.find(CRS + key)
        if element is not None:
            curves[key] = [item.text.strip() for item in element.iter(RDF + "li") if item.text]
    unsupported = []
    if any(abs(values.get(key, 0)) > 0 for key in
           ("RedHue", "RedSaturation", "GreenHue", "GreenSaturation", "BlueHue", "BlueSaturation", "ShadowTint")):
        unsupported.append("camera_calibration")
    if description.find(CRS + "Look") is not None or any(key in raw for key in ("CameraProfile", "CameraProfileDigest")):
        unsupported.append("camera_or_look_profile")
    if local_masks:
        unsupported.append("embedded_local_masks")
    if any(abs(values.get(key, 0)) > 0 for key in ("ColorGradeGlobalSat", "ColorGradeGlobalLum")):
        unsupported.append("global_color_wheel")
    signature = hashlib.sha256(json.dumps({"values": values, "curves": curves}, sort_keys=True).encode()).hexdigest()
    return {"source": source, "filename": filename, "title": title,
            "sha256": hashlib.sha256(payload).hexdigest(), "settings_signature": signature,
            "family": material_family(source), "role": role, "values": values,
            "curves": curves, "unsupported": unsupported,
            "auto_registered": False}


def inventory_resources(root: Path) -> dict:
    """Hash files and inspect bounded XMP members in memory, without extraction."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Resource root must be a directory")
    files, presets, errors = [], [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
            continue
        source = path.relative_to(root).as_posix()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": source, "bytes": path.stat().st_size, "sha256": digest})
        try:
            if path.suffix.casefold() == ".xmp":
                with path.open("rb") as stream:
                    presets.append(inspect_xmp(stream.read(MAX_XMP_BYTES + 1), source, path.name))
            elif path.suffix.casefold() == ".zip":
                budget = 0
                with ZipFile(path) as archive:
                    for member in archive.infolist():
                        inner = PurePosixPath(member.filename.replace("\\", "/"))
                        if member.is_dir() or inner.suffix.casefold() != ".xmp" or "__MACOSX" in inner.parts or inner.name.startswith("._"):
                            continue
                        locator = source + "!" + member.filename
                        if inner.is_absolute() or ".." in inner.parts or ":" in inner.parts[0]:
                            errors.append({"source": locator, "error": "unsafe_member_path"})
                            continue
                        budget += member.file_size
                        if member.file_size > MAX_XMP_BYTES or budget > MAX_ARCHIVE_XMP_BYTES:
                            errors.append({"source": locator, "error": "archive_study_size_limit"})
                            continue
                        try:
                            with archive.open(member) as stream:
                                payload = stream.read(MAX_XMP_BYTES + 1)
                            presets.append(inspect_xmp(payload, locator, inner.name))
                        except (ValueError, ET.ParseError) as exc:
                            errors.append({"source": locator, "error": type(exc).__name__})
        except (ValueError, OSError, ET.ParseError, BadZipFile, RuntimeError, NotImplementedError) as exc:
            errors.append({"source": source, "error": type(exc).__name__})
    unique = {entry["sha256"]: entry for entry in presets}
    groups = {}
    for family in sorted({entry["family"] for entry in unique.values()}):
        records = [entry for entry in unique.values() if entry["family"] == family]
        stats = {}
        for key in sorted({key for entry in records for key in entry["values"]}):
            numbers = [entry["values"][key] for entry in records if key in entry["values"]]
            if any(number != 0 for number in numbers):
                stats[key] = {"n": len(numbers), "min": min(numbers), "median": statistics.median(numbers), "max": max(numbers)}
        groups[family] = {"unique": len(records), "roles": dict(Counter(entry["role"] for entry in records)),
                          "unsupported": dict(Counter(key for entry in records for key in entry["unsupported"])),
                          "statistics": stats}
    summary = {"files": len(files), "file_types": dict(Counter(Path(entry["path"]).suffix.casefold() for entry in files)),
               "xmp_entries": len(presets), "unique_xmp_bytes": len(unique),
               "unique_setting_signatures": len({entry["settings_signature"] for entry in presets}),
               "duplicate_xmp_entries": len(presets) - len(unique),
               "roles": dict(Counter(entry["role"] for entry in unique.values())), "groups": groups,
               "auto_registered": False}
    return {"schema_version": 1, "summary": summary, "files": files, "presets": presets, "errors": errors}
