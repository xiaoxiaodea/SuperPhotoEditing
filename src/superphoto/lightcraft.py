"""Headless LightCraft finishing.

SuperPhoto still plans slider recipes. Rendering no longer opens Lightroom
Classic. Each export is one `lightcraft-cli render` of a private copy, with a
develop-settings JSON file translated from those sliders.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import tarfile
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
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


def cli_name() -> str:
    return "lightcraft-cli.exe" if sys_platform() == "win32" else "lightcraft-cli"


def release_asset() -> str:
    """Official portable bundles contain the CLI alongside the desktop binary."""
    system, machine = sys_platform(), platform.machine().lower()
    if system == "darwin":
        return CLI_ZIP
    if system == "win32" and machine in {"amd64", "x86_64", "arm64", "aarch64"}:
        arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
        return f"lightcraft-{PINNED_VERSION}-windows-{arch}-portable.zip"
    if system.startswith("linux") and machine in {"amd64", "x86_64", "arm64", "aarch64"}:
        arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
        return f"lightcraft-{PINNED_VERSION}-linux-{arch}.tar.gz"
    raise LightCraftError(f"未提供 {system}/{machine} 的自动安装包；请设置 SUPERPHOTO_LIGHTCRAFT_BIN。")


def bundled_cli_path() -> Path:
    return project_root() / "vendor" / "lightcraft" / cli_name()


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
        candidates.append(Path(root) / "vendor" / "lightcraft" / cli_name())
    candidates.append(bundled_cli_path())
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def install_cli(destination: Path | None = None) -> dict[str, Any]:
    """Install a checksum-verified native CLI without extracting unrelated files."""
    asset = release_asset()
    target = destination or bundled_cli_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lightcraft-download-") as tmp:
        archive = Path(tmp) / asset
        _download(f"{RELEASE}/{asset}", archive)
        _verify_archive(archive)
        with tempfile.TemporaryDirectory(prefix="lightcraft-extract-") as extracted:
            binary = Path(extracted) / cli_name()
            _extract_cli(archive, binary)
            binary.chmod(binary.stat().st_mode | 0o755)
            version = _version(binary)
            if not version:
                raise LightCraftError("下载的 LightCraft CLI 无法在本机运行，原有安装未替换。")
            # Stage beside the destination so replacement is atomic on its volume.
            fd, staged_name = tempfile.mkstemp(prefix="lightcraft-", suffix=target.suffix, dir=target.parent)
            os.close(fd)
            staged = Path(staged_name)
            try:
                shutil.copy2(binary, staged)
                staged.replace(target)
            finally:
                staged.unlink(missing_ok=True)
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
        import math

        temperature, tint = adobe.get("Temperature"), adobe.get("Tint")
        if (isinstance(temperature, bool) or isinstance(tint, bool)
                or not isinstance(temperature, (int, float)) or not isinstance(tint, (int, float))
                or not math.isfinite(temperature) or not math.isfinite(tint)
                or not 2000 <= temperature <= 50000 or not -150 <= tint <= 150):
            raise LightCraftError("自定义白平衡需要有效的色温和色调；不能使用猜测的默认值。")
        doc["wb"] = {"mode": "custom", "temp": temperature, "tint": tint}
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
    # This CLI adapter has no source WB metadata. Omit unknown numeric bases so
    # relative edits preserve As Shot instead of converting from an invented Kelvin.
    adobe: dict[str, Any] = field(default_factory=lambda: {"WhiteBalance": "As Shot"})
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
            return {"developSettings": dict(photo.adobe), "settings_source": "planned",
                    "source_metadata_available": False}
        if name == "set_develop_settings":
            photo = self._state(str(args.get("photo_id") or ""))
            photo.adobe.update(args.get("settings") or {})
            return {"ok": True}
        if name == "apply_ai_mask":
            return self._apply_mask(args)
        if name == "create_develop_preset":
            return self._save_checkpoint(args)
        if name == "export_photos":
            return self._export(args, timeout=timeout or 300.0)
        raise LightCraftError(f"LightCraft 工具 {name} 未实现")

    def _state(self, photo_id: str) -> _PhotoState:
        return self._photos.setdefault(photo_id, _PhotoState())

    def _save_checkpoint(self, args: dict[str, Any]) -> dict[str, Any]:
        import hashlib

        photo_id = str(args.get("photo_id") or "")
        if not photo_id:
            raise LightCraftError("保存配方缺少照片路径")
        name = str(args.get("preset_name") or "accepted")
        state = self._state(photo_id)
        document = adobe_to_lightcraft(state.adobe, state.masks)
        folder = Path(photo_id).parent / "lightcraft-checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (hashlib.sha256(name.encode("utf-8")).hexdigest()[:24] + ".json")
        fd, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(document, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            Path(temporary).replace(target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"success": True, "name": name, "engine": "lightcraft", "path": str(target),
                "format": "lightcraft-develop-json", "settings_source": "planned"}

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
        return {"success": True, "mask_id": mask_id, "target": target, "settings": record["settings"],
                "status": "planned", "verified": False}

    def _export(self, args: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        ids = args.get("photo_ids") or []
        if not ids:
            raise LightCraftError("LightCraft 导出没有照片")
        photo_id = str(ids[0])
        source = Path(photo_id)
        if not source.is_file():
            raise LightCraftError(f"找不到作业底片：{source}")
        destination = Path(str(args.get("destination") or ""))
        photo = self._state(photo_id)
        document = adobe_to_lightcraft(photo.adobe, photo.masks)
        edge = int(args.get("width") or args.get("height") or 0)
        output = render_document(self.binary, source, document, destination, edge=edge, timeout=timeout)
        return {"count": 1, "path": str(output), "engine": "lightcraft"}


# A develop document's "wb" in this mode holds an offset (Kelvin, tint) from the file's as-shot white balance;
# render_document turns it into the engine's custom mode with the RAW's own as-shot reading. The planner's global
# white balance is an offset, and before this a RAW (whose as-shot value we did not know) dropped it altogether,
# so a RAW could not be warmed globally: on the flat desert RAW DSC05752 (as shot 5551 K / +12) a custom 7000 K
# took frame colourfulness from .186 to .283 at the same tone (his finished edit: .318), more than saturation
# +30 more (.217). LightCraft reports the as-shot reading through `run --import <file> develop.wb`.
RELATIVE_WB_MODE = "asShotOffset"
_AS_SHOT: dict[str, tuple[float, float] | None] = {}
_AS_SHOT_LOCK = threading.Lock()


def as_shot_wb(binary: Path, source: Path, *, timeout: float = 120.0) -> tuple[float, float] | None:
    """The (Kelvin, tint) white balance LightCraft reads from a RAW as shot, cached per file; None if unknown."""
    path = Path(source).resolve()
    try:
        key = f"{path}|{path.stat().st_mtime_ns}"
    except OSError:
        return None
    with _AS_SHOT_LOCK:
        if key in _AS_SHOT:
            return _AS_SHOT[key]
    value: tuple[float, float] | None = None
    try:
        completed = subprocess.run([str(binary), "run", "--import", str(path), "develop.wb"], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=timeout)
        for line in (completed.stdout or "").splitlines():
            try:
                answer = json.loads(line)
            except ValueError:
                continue
            result = answer.get("result") if isinstance(answer, dict) and answer.get("ok") else None
            if isinstance(result, dict) and answer.get("command") == "develop.wb":
                temp, tint = float(result["temp"]), float(result.get("tint", 0.0))
                if 2000.0 <= temp <= 50000.0 and -150.0 <= tint <= 150.0:
                    value = (temp, tint)
    except (OSError, subprocess.TimeoutExpired, KeyError, TypeError, ValueError):
        value = None
    with _AS_SHOT_LOCK:
        _AS_SHOT[key] = value
    return value


def resolve_wb(binary: Path, source: Path, document: dict[str, Any]) -> dict[str, Any]:
    """The document with an as-shot-relative white balance made absolute; dropped when the reading fails."""
    wb = document.get("wb")
    if not isinstance(wb, dict) or wb.get("mode") != RELATIVE_WB_MODE:
        return document
    resolved = {key: value for key, value in document.items() if key != "wb"}
    shot = as_shot_wb(binary, source)
    if shot is not None:
        resolved["wb"] = {"mode": "custom",
                          "temp": round(min(50000.0, max(2000.0, shot[0] + float(wb.get("temp", 0.0)))), 1),
                          "tint": round(min(150.0, max(-150.0, shot[1] + float(wb.get("tint", 0.0)))), 2)}
    return resolved


_QUALITY_FORMATS = {".jpg", ".jpeg", ".avif", ".jxl"}


class _SessionError(LightCraftError):
    """The session could not render; a cold render may still work."""


class _SessionTimeout(_SessionError):
    """The session did not answer in time (it has been stopped)."""


class RenderSession:
    """One `lightcraft-cli mcp` process with one photo loaded, rendering develop documents as they come.

    A cold `render` starts the engine and decodes the photo for every file. A plan's solver renders about a
    dozen probes of the same photo one after another, so the decode dominates: on the 45 MP DSC_1963
    (2026-11-24) a cold 960px render took 2.4 s, a session render 0.27 s after a first one of 2.2 s
    (2400px: 3.0 s vs 1.2 s per render; a 45 MP DNG at 960px: 2.7 s vs 1.0 s). Every session render was
    pixel-identical to the cold one, masks and RAW included, with each document applied to reset settings
    (``develop.reset``, ``develop.merge``, ``app.export`` in one ``command_batch``). (`run --script -` cannot
    be used: it reads all of stdin before it answers.)

    Not for several threads at once: renders queue on the session's lock.

    ``library`` opens the photo in that LightCraft library (one process per library), which AI Denoise needs
    for its product (see denoise.py); without one nothing is written to disk.
    """

    START_TIMEOUT = 30.0

    def __init__(self, binary: Path, source: Path, *, library: Path | None = None):
        self.source = Path(source).resolve()
        self._lock = threading.Lock()
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._counter = 0
        self.alive = False
        command = [str(binary), "mcp", "--compact"]
        if library is not None:
            command.extend(["--library", str(Path(library).resolve())])
        try:
            self._proc = subprocess.Popen(
                [*command, str(self.source)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as exc:
            raise _SessionError(f"无法启动 LightCraft 会话：{exc}") from exc
        threading.Thread(target=self._read, name="lightcraft-session", daemon=True).start()
        self.alive = True
        try:
            self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "superphoto", "version": "1"}}, self.START_TIMEOUT)
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except _SessionError:
            self.close()
            raise

    def _read(self) -> None:
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def _send(self, message: dict[str, Any]) -> None:
        try:
            self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self._stop()
            raise _SessionError("LightCraft 会话已中断") from exc

    def _rpc(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not self.alive:
            raise _SessionError("LightCraft 会话已关闭")
        self._counter += 1
        ident = self._counter
        self._send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self._stop()
                raise _SessionTimeout("LightCraft 会话超时") from None
            if line is None:
                self._stop()
                raise _SessionError("LightCraft 会话意外退出")
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if not isinstance(reply, dict) or reply.get("id") != ident:
                continue
            if "error" in reply or not isinstance(reply.get("result"), dict):
                raise _SessionError(f"LightCraft 会话出错：{json.dumps(reply.get('error'), ensure_ascii=False)[:300]}")
            return reply["result"]

    def render(self, document: dict[str, Any], output: Path, *, edge: int = 0, quality: int = 92,
               options: dict[str, Any] | None = None, timeout: float = 300.0) -> Path:
        """Render ``document`` (complete settings, applied to reset ones) to ``output``, as `render` would."""
        output = Path(output).resolve()
        export = {"path": str(output), "longEdge": max(0, int(edge)), "conflict": "overwrite", **(options or {})}
        if output.suffix.lower() in _QUALITY_FORMATS:
            export["quality"] = quality
        steps = [{"id": "develop.reset", "params": {}}, {"id": "develop.merge", "params": {"settings": document}},
                 {"id": "app.export", "params": export}]
        with self._lock:
            result = self._rpc("tools/call", {"name": "command_batch",
                                              "arguments": {"steps": steps, "stop_on_error": True}}, timeout)
        text = "".join(part.get("text", "") for part in result.get("content") or [] if isinstance(part, dict))
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        results = body.get("results") if isinstance(body, dict) else None
        if (result.get("isError") or not isinstance(results, list) or len(results) != len(steps)
                or not all(isinstance(r, dict) and r.get("ok") for r in results)):
            failed = next((r.get("error") for r in results or [] if isinstance(r, dict) and not r.get("ok")), None)
            raise _SessionError(f"LightCraft 会话导出失败：{str(failed or text)[:300]}")
        if not output.is_file():
            raise _SessionError("LightCraft 会话没有写出文件")
        return output

    def _stop(self) -> None:
        self.alive = False
        try:
            self._proc.kill()
        except OSError:
            pass

    def close(self) -> None:
        """End the process (it exits when its input closes); stopped if it does not within a few seconds."""
        self.alive = False
        try:
            self._proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)


_SESSIONS: dict[Path, RenderSession] = {}
_SESSIONS_LOCK = threading.Lock()


def sessions_enabled() -> bool:
    """SUPERPHOTO_LIGHTCRAFT_SESSIONS=0 (environment or .env) renders every file with its own cold `render`."""
    from .config import get_settings

    return get_settings().lightcraft_sessions


@contextmanager
def render_session(binary: Path, source: Path) -> Iterator[RenderSession | None]:
    """Inside this block, render_document renders ``source`` through one LightCraft process.

    Nested blocks for the same file reuse the open session. When none can start (or sessions are turned
    off), it yields None and every render stays a cold `render`; a session that fails mid-way is dropped
    the same way, and that render is redone cold.
    """
    key = Path(source).resolve()
    with _SESSIONS_LOCK:
        existing = _SESSIONS.get(key)
    if existing is not None or not sessions_enabled() or not Path(binary).is_file():
        yield existing
        return
    try:
        session = RenderSession(binary, key)
    except _SessionError:
        yield None
        return
    with _SESSIONS_LOCK:
        _SESSIONS[key] = session
    try:
        yield session
    finally:
        with _SESSIONS_LOCK:
            if _SESSIONS.get(key) is session:
                del _SESSIONS[key]
        session.close()


def _session_for(source: Path) -> RenderSession | None:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(Path(source).resolve())
    return session if session is not None and session.alive else None


def render_document(binary: Path, source: Path, document: dict[str, Any], destination: Path, *,
                    edge: int = 0, timeout: float = 300.0, name: str = "render.jpg", quality: int = 92,
                    options: dict[str, Any] | None = None) -> Path:
    """Render one LightCraft develop document to destination/<name> (render.jpg by default).

    The document is written beside the render as develop.json, so every
    delivered file keeps the exact native settings that produced it (an
    as-shot-relative white balance already made absolute, see resolve_wb).
    ``edge`` 0 renders at full size (cropped). The file format follows the extension of ``name``;
    ``options`` are passed as ``app.export`` options, e.g. {"bitDepth": 16, "colorSpace": "srgb"}.
    Inside ``render_session`` for this source the render goes through that session.
    A document with AI Denoise on renders in a library of its own (denoise.render); when the file cannot be
    denoised, the setting is dropped from develop.json and the render is the plain one.
    """
    from . import denoise

    document = resolve_wb(binary, source, document)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / name
    settings_path = destination / "develop.json"
    if denoise.requested(document):
        outcome = denoise.render(binary, source, document, output, edge=edge, quality=quality, options=options,
                                 timeout=timeout)
        if outcome != "denoised":
            document = denoise.without(document)
        if outcome is not None:
            settings_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            return output
    settings_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    session = _session_for(source)
    if session is not None:
        try:
            return session.render(document, output, edge=edge, quality=quality, options=options, timeout=timeout)
        except _SessionTimeout as exc:
            raise LightCraftError("LightCraft 导出超时") from exc
        except _SessionError:
            pass   # Redone cold: the engine's own error if the document is at fault, else the render.
    command = [str(binary), "render", str(source), "-o", str(output), "--settings", str(settings_path)]
    if output.suffix.lower() in _QUALITY_FORMATS:
        command.extend(["--quality", str(quality)])
    if edge > 0:
        command.extend(["--size", str(edge)])
    for key, value in (options or {}).items():
        command.extend(["--opt", f"{key}={json.dumps(value) if not isinstance(value, str) else value}"])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise LightCraftError("LightCraft 导出超时") from exc
    except OSError as exc:
        raise LightCraftError(f"无法运行 lightcraft-cli：{exc}") from exc
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise LightCraftError("LightCraft 导出失败" + (f"：{detail[-800:]}" if detail else ""))
    return output


def sample_colors(binary: Path, source: Path, points: list[tuple[float, float]], *,
                  timeout: float = 120.0) -> list[list[float]]:
    """Colour of the unedited photo at normalized points, in the engine's own colour-range space.

    A colour-range mask needs samples in LightCraft's internal (Oklab-like)
    values; asking the engine avoids any mismatch with our own conversion.
    """
    if not points:
        return []
    lines = []
    for x, y in points:
        lines.append(json.dumps({"command": "mask.add", "params": {"kind": "colorRange", "name": "sample"}}))
        lines.append(json.dumps({"command": "mask.sampleColor", "params": {"x": float(x), "y": float(y)}}))
    fd, script = tempfile.mkstemp(suffix=".jsonl", prefix="lc-sample-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        command = [str(binary), "run", "--keep-going", "--import", str(source), "--script", script]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise LightCraftError("LightCraft 取色超时") from exc
        except OSError as exc:
            raise LightCraftError(f"无法运行 lightcraft-cli：{exc}") from exc
    finally:
        Path(script).unlink(missing_ok=True)
    samples: list[list[float]] = []
    for line in (completed.stdout or "").splitlines():
        if '"mask.sampleColor"' not in line:
            continue
        try:
            record = json.loads(line)
            values = record["result"]["samples"][-1] if record.get("ok") is True else None
        except (ValueError, KeyError, IndexError, TypeError):
            values = None
        if not (isinstance(values, list) and len(values) == 3):
            raise LightCraftError("LightCraft 取色失败")
        samples.append([float(v) for v in values])
    if len(samples) != len(points):
        raise LightCraftError("LightCraft 取色结果不完整")
    return samples


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
        if len(parts) >= 2 and parts[-1].lstrip("*") == archive.name:
            expected = parts[0]
            break
    if not expected:
        raise LightCraftError(f"校验清单里没有 {archive.name}")
    import hashlib
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected:
        raise LightCraftError("LightCraft 安装包校验不一致，已停止安装。")


def _extract_cli(archive: Path, target: Path) -> None:
    """Copy only a regular CLI file to an explicit path; never unpack archive paths."""
    wanted = cli_name()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            matches = [entry for entry in bundle.infolist()
                       if Path(entry.filename).name == wanted and not entry.is_dir()
                       and (entry.external_attr >> 16) & 0o170000 != 0o120000]
            if len(matches) != 1:
                raise LightCraftError(f"{archive.name} 里没有唯一的 {wanted}。")
            with bundle.open(matches[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            matches = [entry for entry in bundle.getmembers()
                       if Path(entry.name).name == wanted and entry.isfile()]
            if len(matches) != 1:
                raise LightCraftError(f"{archive.name} 里没有唯一的 {wanted}。")
            with bundle.extractfile(matches[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


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
        completed = subprocess.run([str(binary), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or completed.stderr or "").strip()
