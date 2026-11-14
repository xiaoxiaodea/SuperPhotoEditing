"""Headless LightCraft finishing.

SuperPhoto still plans slider recipes. Rendering no longer opens Lightroom
Classic. Each export is one `lightcraft-cli render` of a private copy, with a
develop-settings JSON file translated from those sliders.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PINNED_VERSION = "0.21.1"
CLI_ZIP = f"lightcraft-cli-{PINNED_VERSION}-macos-universal.zip"
RELEASE = f"https://github.com/storytold/lightcraft/releases/download/v{PINNED_VERSION}"
TOOLS = (
    "get_photo_metadata",
    "set_develop_settings",
    "export_photos",
    "import_photos",
    "apply_ai_mask",
    "create_develop_preset",
)

_LIGHT = {
    "Exposure2012": "exposure",
    "Contrast2012": "contrast",
    "Highlights2012": "highlights",
    "Shadows2012": "shadows",
    "Whites2012": "whites",
    "Blacks2012": "blacks",
}
_CURVE = {
    "ParametricHighlights": "highlights",
    "ParametricLights": "lights",
    "ParametricDarks": "darks",
    "ParametricShadows": "shadows",
    "ParametricShadowSplit": "split_shadows",
    "ParametricMidtoneSplit": "split_mid",
    "ParametricHighlightSplit": "split_highlights",
}
_POINT_CURVES = {
    "ToneCurvePV2012": "master",
    "ToneCurvePV2012Red": "red",
    "ToneCurvePV2012Green": "green",
    "ToneCurvePV2012Blue": "blue",
}
_MIXER = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_GRADING = {
    "shadows": ("SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum"),
    "midtones": ("ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum"),
    "highlights": ("SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum"),
}
_DETAIL = {
    "Sharpness": "sharpen_amount",
    "SharpenRadius": "sharpen_radius",
    "SharpenDetail": "sharpen_detail",
    "SharpenEdgeMasking": "sharpen_masking",
    "LuminanceSmoothing": "nr_luminance",
    "LuminanceNoiseReductionDetail": "nr_detail",
    "LuminanceNoiseReductionContrast": "nr_contrast",
    "ColorNoiseReduction": "nr_color",
    "ColorNoiseReductionDetail": "nr_color_detail",
    "ColorNoiseReductionSmoothness": "nr_color_smoothness",
}
_GEOMETRY = {
    "PerspectiveVertical": "vertical",
    "PerspectiveHorizontal": "horizontal",
    "PerspectiveRotate": "rotate",
    "PerspectiveScale": "scale",
    "PerspectiveAspect": "aspect",
}
# Adobe PerspectiveUpright: 0 off, 1 auto, 2 level, 3 vertical, 4 full, 5 guided.
_UPRIGHT = {0: "off", 1: "auto", 2: "level", 3: "vertical", 4: "full", 5: "guided"}
_LOCAL = {
    "local_Exposure": "exposure",
    "local_Contrast": "contrast",
    "local_Highlights": "highlights",
    "local_Shadows": "shadows",
    "local_Whites": "whites",
    "local_Blacks": "blacks",
    "local_Temperature": "temp",
    "local_Tint": "tint",
    "local_Saturation": "saturation",
    "local_Clarity": "clarity",
    "local_Texture": "texture",
    "local_Dehaze": "dehaze",
}


class LightCraftError(RuntimeError):
    """A user-actionable LightCraft failure."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bundled_cli_path() -> Path:
    return project_root() / "vendor" / "lightcraft" / "lightcraft-cli"


def discover_cli(explicit: Path | None = None) -> Path | None:
    """Return an executable lightcraft-cli, ignoring leftover Lightroom MCP paths."""
    candidates: list[Path] = []
    if explicit is not None and explicit.suffix != ".js":
        candidates.append(explicit)
    env = os.environ.get("SUPERPHOTO_LIGHTCRAFT_BIN")
    if env:
        candidates.append(Path(env))
    found = shutil.which("lightcraft-cli")
    if found:
        candidates.append(Path(found))
    root = os.environ.get("SUPERPHOTO_ROOT")
    if root:
        candidates.append(Path(root) / "vendor" / "lightcraft" / "lightcraft-cli")
    candidates.append(bundled_cli_path())
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def install_cli(destination: Path | None = None) -> dict[str, Any]:
    """Download the pinned macOS universal CLI into vendor/lightcraft."""
    if sys_platform() != "darwin":
        raise LightCraftError(
            f"自动安装只覆盖 macOS。请从 {RELEASE}/{CLI_ZIP} 下载对应平台的 lightcraft-cli，"
            "放到 vendor/lightcraft/lightcraft-cli，或设置 SUPERPHOTO_LIGHTCRAFT_BIN。"
        )
    target = destination or bundled_cli_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lightcraft-") as tmp:
        archive = Path(tmp) / CLI_ZIP
        _download(f"{RELEASE}/{CLI_ZIP}", archive)
        _verify_archive(archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        binary = _find_binary(Path(tmp))
        if binary is None:
            raise LightCraftError(f"{CLI_ZIP} 里没有可执行的 lightcraft-cli。")
        shutil.copy2(binary, target)
    target.chmod(target.stat().st_mode | 0o755)
    version = _version(target)
    return {
        "ok": True,
        "binary": str(target),
        "version": version,
        "pinned": PINNED_VERSION,
        "message": f"LightCraft CLI 已安装：{target}",
    }


def sys_platform() -> str:
    import sys
    return sys.platform


def adobe_to_lightcraft(adobe: dict[str, Any], masks: list[dict[str, Any]]) -> dict[str, Any]:
    """Translate the slider dict the recipe planner already produces."""
    doc: dict[str, Any] = {}
    if adobe.get("WhiteBalance") == "Custom":
        doc["wb"] = {
            "mode": "custom",
            "temp": adobe.get("Temperature", 6500),
            "tint": adobe.get("Tint", 0),
        }
    light = {name: adobe[key] for key, name in _LIGHT.items() if key in adobe}
    if light:
        doc["light"] = light
    curve = {name: adobe[key] for key, name in _CURVE.items() if key in adobe}
    for key, name in _POINT_CURVES.items():
        points = _curve_points(adobe.get(key))
        if points:
            curve[name] = points
    if curve:
        doc["curve"] = curve
    mixer: dict[str, dict[str, float]] = {}
    for suffix in _MIXER:
        band = suffix.lower()
        hsl = {}
        for component, short in (("Hue", "hue"), ("Saturation", "sat"), ("Luminance", "lum")):
            key = f"{component}Adjustment{suffix}"
            if key in adobe:
                hsl[short] = adobe[key]
        if hsl:
            mixer[band] = hsl
    if mixer:
        doc["mixer"] = mixer
    color = {}
    if "Vibrance" in adobe:
        color["vibrance"] = adobe["Vibrance"]
    if "Saturation" in adobe:
        color["saturation"] = adobe["Saturation"]
    if color:
        doc["color"] = color
    grading: dict[str, Any] = {}
    for wheel, (hue_key, sat_key, lum_key) in _GRADING.items():
        wheel_values = {}
        if hue_key in adobe:
            wheel_values["hue"] = adobe[hue_key]
        if sat_key in adobe:
            wheel_values["sat"] = adobe[sat_key]
        if lum_key in adobe:
            wheel_values["lum"] = adobe[lum_key]
        if wheel_values:
            grading[wheel] = wheel_values
    if "ColorGradeBlending" in adobe:
        grading["blending"] = adobe["ColorGradeBlending"]
    if "SplitToningBalance" in adobe:
        grading["balance"] = adobe["SplitToningBalance"]
    if grading:
        doc["grading"] = grading
    effects = {}
    for key, name in (("Texture", "texture"), ("Clarity2012", "clarity"), ("Dehaze", "dehaze")):
        if key in adobe:
            effects[name] = adobe[key]
    if effects:
        doc["effects"] = effects
    vignette = {}
    for key, name in (
        ("PostCropVignetteAmount", "amount"),
        ("PostCropVignetteMidpoint", "midpoint"),
        ("PostCropVignetteRoundness", "roundness"),
        ("PostCropVignetteFeather", "feather"),
    ):
        if key in adobe:
            vignette[name] = adobe[key]
    if vignette:
        doc["vignette"] = vignette
    grain = {}
    for key, name in (("GrainAmount", "amount"), ("GrainSize", "size"), ("GrainFrequency", "roughness")):
        if key in adobe:
            grain[name] = adobe[key]
    if grain:
        doc["grain"] = grain
    detail = {name: adobe[key] for key, name in _DETAIL.items() if key in adobe}
    if detail:
        doc["detail"] = detail
    optics: dict[str, Any] = {}
    if "LensManualDistortionAmount" in adobe:
        optics["distortion"] = adobe["LensManualDistortionAmount"]
    if adobe.get("LensProfileEnable") is True:
        optics["lens_profile"] = True
        optics["profile_distortion"] = 100
        optics["profile_vignetting"] = 100
    elif adobe.get("LensProfileEnable") is False:
        optics["lens_profile"] = False
        optics["profile_distortion"] = 0
        optics["profile_vignetting"] = 0
    if optics:
        doc["optics"] = optics
    geometry = {name: adobe[key] for key, name in _GEOMETRY.items() if key in adobe}
    upright = adobe.get("PerspectiveUpright")
    if upright is not None:
        geometry["upright"] = _UPRIGHT.get(int(upright), "off")
    if geometry:
        doc["geometry"] = geometry
    crop_keys = ("CropLeft", "CropTop", "CropRight", "CropBottom", "CropAngle")
    if any(key in adobe for key in crop_keys):
        doc["crop"] = {
            "geometry": {
                "rect": {
                    "x0": adobe.get("CropLeft", 0.0),
                    "y0": adobe.get("CropTop", 0.0),
                    "x1": adobe.get("CropRight", 1.0),
                    "y1": adobe.get("CropBottom", 1.0),
                },
                "angle": adobe.get("CropAngle", 0.0),
            }
        }
    rendered_masks = [_mask_document(index, mask) for index, mask in enumerate(masks, start=1)]
    if rendered_masks:
        doc["masks"] = rendered_masks
    return doc


def _mask_document(index: int, mask: dict[str, Any]) -> dict[str, Any]:
    target = str(mask["target"])
    settings = mask.get("settings") or {}
    adjust = {"amount": 100.0}
    adjust.update({name: settings[key] for key, name in _LOCAL.items() if key in settings})
    return {
        "id": index,
        "name": target,
        "visible": True,
        "invert": False,
        "on_photo": True,
        "components": [{"op": "add", "invert": False, "shape": {"kind": target}}],
        "adjust": adjust,
    }


def _curve_points(flat: Any) -> list[dict[str, float]]:
    if not isinstance(flat, list) or len(flat) < 4:
        return []
    points = []
    for index in range(0, len(flat) - 1, 2):
        points.append({"x": round(float(flat[index]) / 255.0, 6), "y": round(float(flat[index + 1]) / 255.0, 6)})
    return points


@dataclass
class _PhotoState:
    adobe: dict[str, Any] = field(default_factory=lambda: {
        "WhiteBalance": "As Shot",
        "Temperature": 6500.0,
        "Tint": 0.0,
    })
    masks: list[dict[str, Any]] = field(default_factory=list)


class LightCraftClient:
    """Same tool calls the staged finisher already uses, executed by the CLI."""

    def __init__(self, binary: Path):
        self.binary = binary
        self.tools = set(TOOLS)
        self._photos: dict[str, _PhotoState] = {}

    def status(self) -> dict[str, Any]:
        version = _version(self.binary)
        connected = bool(version)
        return {
            "available": True,
            "connected": connected,
            "engine": "lightcraft",
            "binary": str(self.binary),
            "version": version,
            "error": None if connected else "lightcraft-cli did not respond",
            "tools": sorted(self.tools),
        }

    def close(self) -> None:
        self._photos.clear()

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        if name not in self.tools:
            raise LightCraftError(f"LightCraft 不支持工具：{name}")
        args = arguments or {}
        if name == "import_photos":
            self._state(str(args.get("source_path") or ""))
            return {"ok": True}
        if name == "get_photo_metadata":
            photo = self._state(str(args.get("photo_id") or ""))
            return {"developSettings": dict(photo.adobe)}
        if name == "set_develop_settings":
            photo = self._state(str(args.get("photo_id") or ""))
            photo.adobe.update(args.get("settings") or {})
            return {"ok": True}
        if name == "apply_ai_mask":
            return self._apply_mask(args)
        if name == "create_develop_preset":
            return {"success": True, "name": args.get("preset_name"), "engine": "lightcraft"}
        if name == "export_photos":
            return self._export(args, timeout=timeout or 300.0)
        raise LightCraftError(f"LightCraft 工具 {name} 未实现")

    def _state(self, photo_id: str) -> _PhotoState:
        return self._photos.setdefault(photo_id, _PhotoState())

    def _apply_mask(self, args: dict[str, Any]) -> dict[str, Any]:
        photo = self._state(str(args.get("photo_id") or ""))
        target = str(args.get("target") or "")
        if target not in {"subject", "sky", "background"}:
            raise LightCraftError(f"LightCraft 不能确认 {target or '未知'} 蒙版")
        if args.get("reset"):
            photo.masks = [mask for mask in photo.masks if mask["target"] != target]
            return {"success": True, "target": target, "reset": True}
        mask_id = f"lc-{target}"
        record = {"target": target, "mask_id": mask_id, "settings": dict(args.get("settings") or {})}
        photo.masks = [mask for mask in photo.masks if mask["target"] != target]
        photo.masks.append(record)
        return {"success": True, "mask_id": mask_id, "target": target, "settings": record["settings"]}

    def _export(self, args: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        ids = args.get("photo_ids") or []
        if not ids:
            raise LightCraftError("LightCraft 导出没有照片")
        photo_id = str(ids[0])
        source = Path(photo_id)
        if not source.is_file():
            raise LightCraftError(f"找不到作业底片：{source}")
        destination = Path(str(args.get("destination") or ""))
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "render.jpg"
        photo = self._state(photo_id)
        document = adobe_to_lightcraft(photo.adobe, photo.masks)
        settings_path = destination / "develop.json"
        settings_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        edge = int(args.get("width") or args.get("height") or 0)
        command = [str(self.binary), "render", str(source), "-o", str(output), "--settings", str(settings_path), "--quality", "92"]
        if edge > 0:
            command.extend(["--size", str(edge)])
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise LightCraftError("LightCraft 导出超时") from exc
        except OSError as exc:
            raise LightCraftError(f"无法运行 lightcraft-cli：{exc}") from exc
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout or "").strip()
            raise LightCraftError("LightCraft 导出失败" + (f"：{detail[-800:]}" if detail else ""))
        return {"count": 1, "path": str(output), "engine": "lightcraft"}


def _download(url: str, destination: Path) -> None:
    try:
        urllib.request.urlretrieve(url, destination)
    except OSError as exc:
        raise LightCraftError(f"下载 LightCraft 失败：{exc}") from exc


def _verify_archive(archive: Path) -> None:
    try:
        with urllib.request.urlopen(f"{RELEASE}/SHA256SUMS.txt", timeout=60) as response:
            sums = response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise LightCraftError(f"无法核对 LightCraft 安装包：{exc}") from exc
    expected = ""
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1] == CLI_ZIP:
            expected = parts[0]
            break
    if not expected:
        raise LightCraftError(f"校验清单里没有 {CLI_ZIP}")
    import hashlib
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        raise LightCraftError("LightCraft 安装包校验不一致，已停止安装。")


def _find_binary(root: Path) -> Path | None:
    direct = root / "lightcraft-cli"
    if direct.is_file():
        return direct
    for path in root.rglob("lightcraft-cli"):
        if path.is_file() and not path.is_symlink():
            return path
    return None


def _version(binary: Path) -> str:
    try:
        completed = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or completed.stderr or "").strip()
