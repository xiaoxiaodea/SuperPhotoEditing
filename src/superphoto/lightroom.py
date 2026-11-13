"""Local MCP bridge and Lightroom Classic finishing workflow.

The bridge speaks newline-delimited JSON-RPC to the vendored Lightroom MCP
server. SuperPhoto imports its private job copy, never the user's master file,
then applies one selected recipe as bounded Lightroom Develop settings.
"""

from __future__ import annotations

import atexit
import colorsys
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ColorGradingControls, Grade, Recipe

_PROTOCOL_VERSION = "2025-06-18"
_HUES = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
_PROFESSIONAL_STAGES = (
    "color-foundation",
    "technical-correction",
    "tonal-shape",
    "color-correction",
    "creative-look",
    "detail-output",
)


class LightroomError(RuntimeError):
    """A user-actionable Lightroom bridge failure."""


@dataclass
class LightroomRuntime:
    command: list[str]
    cwd: Path


def project_root() -> Path:
    configured = os.environ.get("SUPERPHOTO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _discover_node() -> Path | None:
    configured = os.environ.get("SUPERPHOTO_NODE")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path.home() / "Library" / "Application Support" / "SuperPhoto" / "runtime" / "node",
            Path("/opt/homebrew/bin/node"),
            Path("/usr/local/bin/node"),
            Path("/usr/bin/node"),
        ]
    )
    found = shutil.which("node")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


def discover_runtime(server_path: Path | None = None) -> LightroomRuntime | None:
    root = project_root()
    candidates: list[Path] = []
    if server_path:
        candidates.append(server_path.expanduser())
    configured = os.environ.get("SUPERPHOTO_LIGHTROOM_MCP_SERVER")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            root / "vendor" / "lightroom-mcp" / "server" / "dist" / "index.js",
            root / "vendor" / "lightroom-mcp" / "server" / "src" / "index.ts",
        ]
    )
    node = _discover_node()
    if not node:
        return None
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.suffix == ".js":
            return LightroomRuntime([str(node), str(candidate), "stdio"], candidate.parent)
    return None


class LightroomMcpClient:
    def __init__(self, runtime: LightroomRuntime, timeout: float = 35.0):
        self.runtime = runtime
        self.timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr: list[str] = []
        self._id = 0
        self._lock = threading.RLock()
        self._tools: set[str] = set()

    def start(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                return
            env = os.environ.copy()
            self._process = subprocess.Popen(
                self.runtime.command,
                cwd=self.runtime.cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            self._request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "SuperPhoto", "version": "0.1.0"},
                },
                timeout=10.0,
            )
            self._notify("notifications/initialized", {})
            listed = self._request("tools/list", {}, timeout=10.0)
            self._tools = {tool["name"] for tool in listed.get("tools", []) if isinstance(tool, dict) and tool.get("name")}

    def close(self) -> None:
        with self._lock:
            proc = self._process
            self._process = None
            self._tools.clear()
            if not proc:
                return
            if proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

    @property
    def tools(self) -> set[str]:
        self.start()
        return set(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        self.start()
        if name not in self._tools:
            raise LightroomError(f"Lightroom MCP 不支持工具：{name}")
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout or self.timeout,
        )
        if result.get("isError"):
            text = _content_text(result)
            raise LightroomError(text or f"Lightroom 工具 {name} 执行失败")
        text = _content_text(result)
        if not text:
            return result
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def status(self) -> dict[str, Any]:
        try:
            self.start()
        except Exception as exc:
            return {"available": True, "connected": False, "error": str(exc), "tools": []}
        try:
            selection = self.call_tool("get_selected_photos", {"limit": 1, "offset": 0}, timeout=5.0)
            return {
                "available": True,
                "connected": True,
                "selected": int(selection.get("count", 0)) if isinstance(selection, dict) else 0,
                "tools": sorted(self._tools),
            }
        except Exception as exc:
            return {"available": True, "connected": False, "error": str(exc), "tools": sorted(self._tools)}

    def _request(self, method: str, params: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        with self._lock:
            proc = self._require_process()
            self._id += 1
            request_id = self._id
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            assert proc.stdin is not None
            try:
                proc.stdin.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise LightroomError(f"Lightroom MCP 进程已退出：{exc}") from exc
            deadline = time.monotonic() + timeout
            deferred: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                try:
                    message = self._responses.get(timeout=max(0.05, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if message.get("id") != request_id:
                    deferred.append(message)
                    continue
                for item in deferred:
                    self._responses.put(item)
                if "error" in message:
                    error = message["error"]
                    detail = error.get("message") if isinstance(error, dict) else str(error)
                    raise LightroomError(detail or f"MCP 请求失败：{method}")
                return message.get("result") or {}
            for item in deferred:
                self._responses.put(item)
            tail = "；".join(self._stderr[-3:])
            raise LightroomError(f"Lightroom MCP 请求超时：{method}" + (f"（{tail}）" if tail else ""))

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        proc = self._require_process()
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    def _require_process(self) -> subprocess.Popen[str]:
        if not self._process or self._process.poll() is not None:
            code = self._process.poll() if self._process else None
            tail = "；".join(self._stderr[-4:])
            raise LightroomError(f"Lightroom MCP 无法启动（退出码 {code}）" + (f"：{tail}" if tail else ""))
        return self._process

    def _read_stdout(self) -> None:
        proc = self._process
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and "id" in message:
                self._responses.put(message)

    def _read_stderr(self) -> None:
        proc = self._process
        if not proc or not proc.stderr:
            return
        for line in proc.stderr:
            text = line.strip()
            if text:
                self._stderr.append(text)
                del self._stderr[:-30]


def _content_text(result: dict[str, Any]) -> str:
    chunks = result.get("content") or []
    return "\n".join(str(item.get("text", "")) for item in chunks if isinstance(item, dict) and item.get("type") == "text")


_client_lock = threading.Lock()
_client: LightroomMcpClient | None = None


def get_lightroom_client(server_path: Path | None = None) -> LightroomMcpClient:
    global _client
    with _client_lock:
        if _client is None:
            runtime = discover_runtime(server_path)
            if runtime is None:
                raise LightroomError("未找到可运行的 Lightroom MCP（需要 Node.js 和已构建的随附服务）")
            _client = LightroomMcpClient(runtime)
        return _client


def wait_for_lightroom_connection(
    client: LightroomMcpClient,
    *,
    timeout: float = 12.0,
    interval: float = 0.5,
) -> dict[str, Any]:
    """Wait for both plugin sockets to settle instead of failing a cold start."""
    deadline = time.monotonic() + timeout
    status: dict[str, Any] = {"available": True, "connected": False, "tools": []}
    while True:
        status = client.status()
        if status.get("connected"):
            return status
        if time.monotonic() >= deadline:
            return status
        time.sleep(interval)


def reconnect_lightroom_client(server_path: Path | None = None) -> LightroomMcpClient:
    """Replace a stale single-client bridge after the plugin becomes available."""
    close_lightroom_client()
    return get_lightroom_client(server_path)


def wake_lightroom_plugin() -> None:
    """Ask Lightroom to start its plugin through its SDK URL handler.

    Opening the app also handles a cold launch. Older installed plugins do
    not have the handler; avoid sending an unknown URL to those versions.
    This never quits Lightroom or depends on Accessibility permissions.
    """
    if sys.platform != "darwin":
        return
    command = ["open", "-a", "Adobe Lightroom Classic"]
    info = (Path.home() / "Library/Application Support/Adobe/Lightroom/Modules"
            / "LightroomMCP.lrplugin/Info.lua")
    if info.is_file() and "PluginURLHandler.lua" in info.read_text(encoding="utf-8"):
        command.append("lightroom://com.lightroom.mcp/start-server")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LightroomError("无法打开 Lightroom Classic，请打开它后重试这张照片。") from exc
    if result.returncode:
        raise LightroomError("无法打开 Lightroom Classic，请确认已安装并能正常打开，再重试这张照片。")


_connection_lock = threading.Lock()


def connect_lightroom(server_path: Path | None = None) -> tuple[LightroomMcpClient, dict[str, Any]]:
    """Recover the plugin as well as the bridge before processing a photo."""
    with _connection_lock:
        client = get_lightroom_client(server_path)
        status = client.status()
        if status.get("connected"):
            return client, status
        wake_lightroom_plugin()
        status = wait_for_lightroom_connection(client, timeout=30.0)
        if not status.get("connected"):
            client = reconnect_lightroom_client(server_path)
            # A URL delivered while Lightroom is still launching may be
            # ignored. Send it once more after the cold-start waiting period.
            wake_lightroom_plugin()
            status = wait_for_lightroom_connection(client, timeout=12.0)
        if not status.get("connected"):
            detail = str(status.get("error") or "plugin did not respond")
            raise LightroomError(
                "Lightroom 尚未准备好。请在 Lightroom 完成启动或关闭弹出的对话框，"
                "再点“重试这张照片”。如果仍未连接，请在“文件 → 增效工具管理器”中"
                "启用 Lightroom MCP，并点击 Start Server。"
            ) from LightroomError(detail)
        return client, status


def close_lightroom_client() -> None:
    global _client
    with _client_lock:
        if _client:
            _client.close()
        _client = None


atexit.register(close_lightroom_client)


def lightroom_status(enabled: bool = True, server_path: Path | None = None) -> dict[str, Any]:
    runtime = discover_runtime(server_path)
    if not enabled:
        return {"enabled": False, "available": runtime is not None, "connected": False, "tools": []}
    if runtime is None:
        return {"enabled": True, "available": False, "connected": False, "error": "未找到随附 Lightroom MCP", "tools": []}
    with _client_lock:
        active = _client
    if active is None:
        return {
            "enabled": True,
            "available": True,
            "connected": False,
            "error": "将在首次执行自动后期时连接 Lightroom",
            "tools": [],
        }
    status = active.status()
    status["enabled"] = True
    return status


def check_lightroom_connection(server_path: Path | None = None) -> dict[str, Any]:
    runtime = discover_runtime(server_path)
    if runtime is None:
        return {"available": False, "connected": False, "error": "未找到随附 Lightroom MCP", "tools": []}
    return get_lightroom_client(server_path).status()


def install_lightroom_plugin(server_path: Path | None = None) -> dict[str, Any]:
    runtime = discover_runtime(server_path)
    if runtime is None:
        raise LightroomError("未找到可运行的 Lightroom MCP")
    source = _bundled_plugin_dir(runtime)
    destination = Path.home() / "Library" / "Application Support" / "Adobe" / "Lightroom" / "Modules" / "LightroomMCP.lrplugin"
    if source is None:
        command = [*runtime.command[:-1], "install-plugin"]
        completed = subprocess.run(command, cwd=runtime.cwd, text=True, capture_output=True, timeout=30)
        detail = (completed.stderr or completed.stdout).strip()
        if completed.returncode != 0:
            raise LightroomError(detail or "Lightroom 插件安装失败")
        return {"ok": True, "message": detail or "Lightroom 插件已安装", "restart_required": True}

    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f"{destination.name}.backup-{uuid.uuid4().hex[:8]}")
    try:
        if destination.exists():
            destination.rename(backup)
        shutil.copytree(source, destination)
    except Exception as exc:
        shutil.rmtree(destination, ignore_errors=True)
        if backup.exists():
            backup.rename(destination)
        raise LightroomError(f"Lightroom 插件更新失败，已恢复旧版本：{exc}") from exc
    else:
        shutil.rmtree(backup, ignore_errors=True)
    return {
        "ok": True,
        "message": f"Lightroom 插件已更新：{destination}",
        "restart_required": True,
        "destination": str(destination),
    }


def _bundled_plugin_dir(runtime: LightroomRuntime) -> Path | None:
    for base in (runtime.cwd, *runtime.cwd.parents):
        for candidate in (
            base / "plugin" / "LightroomMCP.lrplugin",
            base / "LightroomMCP.lrplugin",
        ):
            if candidate.is_dir() and (candidate / "Info.lua").is_file():
                return candidate
    return None


def native_color_recipe(recipe: Recipe, *, audit: dict[str, Any] | None = None) -> Recipe:
    """Copy a recipe with legacy RGB grading translated into native wheels.

    RGB additions and Lightroom color grading have different tonal weights;
    this is a conservative directional approximation, not a color match.
    The RGB span sets saturation (100 * span, at most 25), its chromatic
    direction sets hue, and weighted RGB luminance sets wheel luminance
    (100 * luminance, at most +/-15). An explicitly supplied native control,
    including zero, owns its entire wheel and supersedes the old RGB offsets.
    Clearing the legacy representation makes this normalization idempotent
    and prevents renderers or later merges from applying the color twice.
    """
    result = recipe.model_copy(deep=True)
    controls = result.color_grading.model_dump()
    converted: dict[str, dict[str, float]] = {}
    overridden: list[str] = []
    for legacy_name, wheel, limit in (
        ("shadows", "shadow", .25),
        ("midtones", "midtone", .20),
        ("highlights", "highlight", .25),
    ):
        offsets = getattr(recipe.grade, legacy_name)
        if not all(math.isfinite(value) for value in offsets):
            raise ValueError(f"Non-finite legacy color grade: {legacy_name}")
        if all(abs(value) < 1e-4 for value in offsets):
            continue
        fields = [f"{wheel}_{component}" for component in ("hue", "sat", "lum")]
        if any(controls[field] is not None for field in fields):
            overridden.append(legacy_name)
            continue
        rgb = tuple(max(-limit, min(limit, value)) for value in offsets)
        low, high = min(rgb), max(rgb)
        span = high - low
        hue = colorsys.rgb_to_hsv(*(value - low for value in rgb))[0] * 360 if span > 1e-8 else 0.0
        luminance = sum(value * weight for value, weight in zip(rgb, (.2126, .7152, .0722), strict=True))
        values = {
            f"{wheel}_hue": round(hue, 4),
            f"{wheel}_sat": round(min(25.0, span * 100), 4),
            f"{wheel}_lum": round(max(-15.0, min(15.0, luminance * 100)), 4),
        }
        controls.update(values)
        converted[legacy_name] = values
    result.grade = Grade()
    result.color_grading = ColorGradingControls(**controls)
    if audit is not None and (converted or overridden):
        audit["color_grading_conversion"] = {
            "method": "legacy-rgb-to-native-v1",
            "approximate": True,
            "legacy_grade": recipe.grade.model_dump(),
            "converted": converted,
            "native_overrides": overridden,
            "color_grading": result.color_grading.model_dump(exclude_none=True),
        }
    return result


def recipe_to_develop_settings(
    recipe: Recipe,
    current: dict[str, Any] | None = None,
    *,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert to bounded SDK settings; basic moves add, native wheels replace."""
    recipe = native_color_recipe(recipe, audit=audit)
    current = _flatten_develop_settings(current or {})
    out: dict[str, Any] = {}

    def add(key: str, delta: float, lo: float, hi: float, *, include_zero: bool = False) -> None:
        if abs(delta) < 1e-4 and not include_zero:
            return
        base = _number(current.get(key), 0.0)
        out[key] = round(max(lo, min(hi, base + delta)), 4)

    add("Exposure2012", recipe.exposure_ev, -5.0, 5.0)
    add("Contrast2012", recipe.contrast * 100.0, -100.0, 100.0)
    add("Highlights2012", recipe.highlights * 100.0, -100.0, 100.0)
    add("Shadows2012", recipe.shadows * 100.0, -100.0, 100.0)
    add("Whites2012", recipe.whites * 100.0, -100.0, 100.0)
    add("Blacks2012", recipe.blacks * 100.0, -100.0, 100.0)
    add("Texture", recipe.texture * 100.0, -100.0, 100.0)
    add("Clarity2012", recipe.clarity * 100.0, -100.0, 100.0)
    add("Dehaze", recipe.dehaze * 100.0, -100.0, 100.0)
    add("Vibrance", recipe.vibrance * 100.0, -100.0, 100.0)
    add("Saturation", recipe.saturation * 100.0, -100.0, 100.0)
    add("PostCropVignetteAmount", -recipe.vignette * 100.0, -100.0, 100.0)
    add("GrainAmount", recipe.grain * 100.0, 0.0, 100.0)

    for field, key, low, high in (
        ("shadow_hue", "SplitToningShadowHue", 0, 360),
        ("shadow_sat", "SplitToningShadowSaturation", 0, 100),
        ("shadow_lum", "ColorGradeShadowLum", -100, 100),
        ("midtone_hue", "ColorGradeMidtoneHue", 0, 360),
        ("midtone_sat", "ColorGradeMidtoneSat", 0, 100),
        ("midtone_lum", "ColorGradeMidtoneLum", -100, 100),
        ("highlight_hue", "SplitToningHighlightHue", 0, 360),
        ("highlight_sat", "SplitToningHighlightSaturation", 0, 100),
        ("highlight_lum", "ColorGradeHighlightLum", -100, 100),
        ("blending", "ColorGradeBlending", 0, 100),
        ("balance", "SplitToningBalance", -100, 100),
    ):
        _set_absolute(out, key, getattr(recipe.color_grading, field), low, high)

    parametric = recipe.parametric_tone
    add("ParametricShadows", parametric.shadows * 100.0, -100.0, 100.0)
    add("ParametricDarks", parametric.darks * 100.0, -100.0, 100.0)
    add("ParametricLights", parametric.lights * 100.0, -100.0, 100.0)
    add("ParametricHighlights", parametric.highlights * 100.0, -100.0, 100.0)
    _set_absolute(out, "ParametricShadowSplit", parametric.shadow_split, 0.0, 100.0)
    _set_absolute(out, "ParametricMidtoneSplit", parametric.midtone_split, 0.0, 100.0)
    _set_absolute(out, "ParametricHighlightSplit", parametric.highlight_split, 0.0, 100.0)

    detail = recipe.detail
    for key, value, lo, hi in (
        ("Sharpness", detail.sharpness, 0.0, 150.0),
        ("SharpenRadius", detail.sharpen_radius, 0.5, 3.0),
        ("SharpenDetail", detail.sharpen_detail, 0.0, 100.0),
        ("SharpenEdgeMasking", detail.sharpen_masking, 0.0, 100.0),
        ("LuminanceSmoothing", detail.luminance_nr, 0.0, 100.0),
        ("LuminanceNoiseReductionDetail", detail.luminance_nr_detail, 0.0, 100.0),
        ("LuminanceNoiseReductionContrast", detail.luminance_nr_contrast, 0.0, 100.0),
        ("ColorNoiseReduction", detail.color_nr, 0.0, 100.0),
        ("ColorNoiseReductionDetail", detail.color_nr_detail, 0.0, 100.0),
        ("ColorNoiseReductionSmoothness", detail.color_nr_smoothness, 0.0, 100.0),
    ):
        _set_absolute(out, key, value, lo, hi)

    optics = recipe.optics
    if optics.lens_profile_enable is not None:
        out["LensProfileEnable"] = bool(optics.lens_profile_enable)
    for key, value, lo, hi in (
        ("LensManualDistortionAmount", optics.manual_distortion, -100.0, 100.0),
        ("PerspectiveVertical", optics.perspective_vertical, -100.0, 100.0),
        ("PerspectiveHorizontal", optics.perspective_horizontal, -100.0, 100.0),
        ("PerspectiveRotate", optics.perspective_rotate, -10.0, 10.0),
        ("PerspectiveScale", optics.perspective_scale, 50.0, 150.0),
        ("PerspectiveAspect", optics.perspective_aspect, -100.0, 100.0),
        ("PerspectiveUpright", optics.perspective_upright, 0.0, 5.0),
    ):
        _set_absolute(out, key, value, lo, hi, integer=key == "PerspectiveUpright")

    finish = recipe.finish
    for key, value, lo, hi in (
        ("PostCropVignetteMidpoint", finish.vignette_midpoint, 0.0, 100.0),
        ("PostCropVignetteRoundness", finish.vignette_roundness, -100.0, 100.0),
        ("PostCropVignetteFeather", finish.vignette_feather, 0.0, 100.0),
        ("PostCropVignetteStyle", finish.vignette_style, 1.0, 3.0),
        ("GrainSize", finish.grain_size, 0.0, 100.0),
        ("GrainFrequency", finish.grain_roughness, 0.0, 100.0),
    ):
        _set_absolute(out, key, value, lo, hi, integer=key == "PostCropVignetteStyle")

    crop = recipe.crop
    for key, value, lo, hi in (
        ("CropTop", crop.top, 0.0, 1.0),
        ("CropLeft", crop.left, 0.0, 1.0),
        ("CropBottom", crop.bottom, 0.0, 1.0),
        ("CropRight", crop.right, 0.0, 1.0),
        ("CropAngle", crop.angle, -45.0, 45.0),
    ):
        _set_absolute(out, key, value, lo, hi)

    if abs(recipe.temperature) >= 1e-4 or abs(recipe.tint) >= 1e-4:
        # As Shot can omit BOTH numeric values from getDevelopSettings().
        # Switching it to Custom with a made-up Kelvin or tint base changes
        # the photograph in the opposite direction to the relative request.
        # The companion control must also be known before changing WB mode.
        base_temp = current.get("Temperature")
        base_tint = current.get("Tint")
        missing = [key for key, value, low, high in (
            ("Temperature", base_temp, 2000, 50000),
            ("Tint", base_tint, -150, 150),
        ) if (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or not low <= value <= high)]
        wb = {
            "requested": {"temperature": recipe.temperature, "tint": recipe.tint},
            "base": {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                     for key in ("WhiteBalance", "Temperature", "Tint") if (value := current.get(key)) is not None},
            "status": "preserved" if missing else "applied",
        }
        if missing:
            wb.update(reason="unavailable_white_balance_base", missing_or_invalid=missing,
                      note="未取得完整的真实白平衡基值，保留当前白平衡；不使用默认色温或色调。")
        else:
            out.update({
                "WhiteBalance": "Custom",
                "Temperature": round(max(2000.0, min(50000.0, base_temp + recipe.temperature * 45.0)), 1),
                "Tint": round(max(-150.0, min(150.0, base_tint + recipe.tint)), 4),
            })
            wb["settings"] = {key: out[key] for key in ("WhiteBalance", "Temperature", "Tint")}
        if audit is not None:
            audit["white_balance"] = wb

    for hue, shift in recipe.hsl.items():
        if hue not in _HUES:
            continue
        suffix = hue.capitalize()
        add(f"HueAdjustment{suffix}", shift.hue / 0.30, -100.0, 100.0)
        add(f"SaturationAdjustment{suffix}", shift.sat * 100.0, -100.0, 100.0)
        add(f"LuminanceAdjustment{suffix}", shift.lum * 100.0, -100.0, 100.0)

    curve_keys = {
        "ToneCurvePV2012": recipe.curve.rgb,
        "ToneCurvePV2012Red": recipe.curve.r,
        "ToneCurvePV2012Green": recipe.curve.g,
        "ToneCurvePV2012Blue": recipe.curve.b,
    }
    for key, points in curve_keys.items():
        if points:
            out[key] = _curve_to_lightroom(points)
            out["ToneCurveName2012"] = "Custom"
    return out


def professional_recipe_stages(recipe: Recipe) -> list[tuple[str, Recipe]]:
    """Split a finished recipe into Lightroom passes that mirror a pro workflow."""
    recipe = native_color_recipe(recipe)
    foundation = Recipe(hsl={key: value.model_copy(deep=True) for key, value in recipe.hsl.items()},
                        saturation=min(0, recipe.saturation))
    technical = Recipe(
        exposure_ev=recipe.exposure_ev,
        temperature=recipe.temperature,
        tint=recipe.tint,
        highlights=recipe.highlights,
        shadows=recipe.shadows,
        whites=recipe.whites,
        blacks=recipe.blacks,
    )
    tonal = Recipe(
        contrast=recipe.contrast,
        texture=recipe.texture,
        curve=recipe.curve.model_copy(deep=True),
        parametric_tone=recipe.parametric_tone.model_copy(deep=True),
    )
    color = Recipe(color_grading=recipe.color_grading.model_copy(deep=True))
    creative = Recipe(
        saturation=max(0, recipe.saturation),
        vibrance=recipe.vibrance,
        clarity=recipe.clarity,
        dehaze=recipe.dehaze,
        vignette=recipe.vignette,
        grain=recipe.grain,
        finish=recipe.finish.model_copy(deep=True),
    )
    detail = Recipe(
        detail=recipe.detail.model_copy(deep=True),
        optics=recipe.optics.model_copy(deep=True),
        crop=recipe.crop.model_copy(deep=True),
    )
    stages = list(zip(_PROFESSIONAL_STAGES, (foundation, technical, tonal, color, creative, detail), strict=True))
    if native_mask_settings(recipe):
        stages.insert(2, ("local-light", Recipe(local=recipe.local, masks=recipe.masks)))
    return [(name, stage) for name, stage in stages
            if name == "local-light" or abs(stage.temperature) >= 1e-4 or abs(stage.tint) >= 1e-4
            or recipe_to_develop_settings(stage)]


def native_mask_settings(recipe: Recipe) -> list[tuple[str, dict[str, float]]]:
    """Explicit semantic edits take precedence over legacy local sliders."""
    from .models import MaskAdjustment

    local = recipe.local
    masks = {}
    if local.subject_lift or local.subject_vibrance or local.subject_clarity:
        masks["subject"] = MaskAdjustment(
            target="subject", exposure_ev=max(-0.4, min(0.6, local.subject_lift * 4)),
            saturation=max(-0.2, min(0.15, local.subject_vibrance * 0.5)),
            clarity=min(0.15, local.subject_clarity),
        )
    if local.background_darken or local.background_desat:
        masks["background"] = MaskAdjustment(
            target="background", exposure_ev=-min(0.4, local.background_darken * 4),
            saturation=-min(0.25, local.background_desat),
        )
    if local.sky_dehaze:
        masks["sky"] = MaskAdjustment(target="sky", dehaze=min(0.15, local.sky_dehaze))
    masks.update({m.target: m for m in recipe.masks})
    aliases = {"exposure_ev": "Exposure", "contrast": "Contrast", "whites": "Whites", "blacks": "Blacks",
               "highlights": "Highlights", "shadows": "Shadows",
               "temperature": "Temperature", "tint": "Tint", "saturation": "Saturation",
               "clarity": "Clarity", "texture": "Texture", "dehaze": "Dehaze"}
    result = []
    for target, mask in masks.items():
        if mask.is_noop():
            continue
        # Include zeros: updates are absolute and must also clear superseded moves.
        values = {f"local_{name}": round(getattr(mask, key) *
                  (1 if key in {"exposure_ev", "temperature", "tint"} else 100), 4)
                  for key, name in aliases.items()}
        result.append((target, values))
    return result


def checkpoint_fields(settings: dict[str, Any]) -> list[str]:
    """Return the explicit setting whitelist that can safely be captured."""
    return sorted(settings)


def create_lightroom_checkpoint(
    client: LightroomMcpClient,
    *,
    photo: Path,
    name: str,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Capture an accepted Lightroom state when the enhanced MCP exposes presets."""
    if "create_develop_preset" not in client.tools:
        return None
    fields = checkpoint_fields(settings)
    if not fields:
        return None
    return client.call_tool(
        "create_develop_preset",
        {"photo_id": str(photo), "preset_name": name, "settings": fields},
        timeout=60.0,
    )


def export_lightroom_baseline(
    client: LightroomMcpClient,
    *,
    original: Path,
    output_dir: Path,
    export_edge: int = 2400,
) -> dict[str, Any]:
    """Import a pristine job copy and export Lightroom's unedited baseline."""
    metadata = _ensure_photo(client, original)
    exported = _export_photo(client, original, output_dir, export_edge=export_edge)
    return {"ok": True, "metadata": metadata, "export": exported}


def _attenuate_tonal_recipe(recipe: Recipe, before: dict, strength: float) -> Recipe:
    """Back off toward the actual earlier Lightroom curve, including on retries."""
    import numpy as np
    from .preference import _scale_recipe

    reduced = _scale_recipe(recipe, strength)
    for channel, key in (("rgb", "ToneCurvePV2012"), ("r", "ToneCurvePV2012Red"),
                         ("g", "ToneCurvePV2012Green"), ("b", "ToneCurvePV2012Blue")):
        target = getattr(recipe.curve, channel)
        if not target:
            continue
        original = before.get(key) or [0, 0, 255, 255]
        prior = np.asarray(original, dtype=float).reshape(-1, 2) / 255
        desired = np.asarray(target, dtype=float)
        xs = np.unique(np.r_[prior[:, 0], desired[:, 0]])
        old_y = np.interp(xs, prior[:, 0], prior[:, 1])
        new_y = np.interp(xs, desired[:, 0], desired[:, 1])
        setattr(reduced.curve, channel, list(zip(xs.tolist(), (old_y + (new_y - old_y) * strength).tolist())))
    return reduced


def apply_professional_recipe(
    client: LightroomMcpClient,
    *,
    original: Path,
    recipe: Recipe,
    output_dir: Path,
    export_edge: int = 2400,
    checkpoint_prefix: str | None = None,
    stage_reviewer=None,
) -> dict[str, Any]:
    """Apply and render each non-empty professional stage on one imported copy."""
    color_audit: dict[str, Any] = {}
    recipe = native_color_recipe(recipe, audit=color_audit)
    metadata = _ensure_photo(client, original)
    current = _flatten_develop_settings(metadata.get("developSettings") or {})
    develop_before = metadata.get("developSettings")
    stage_records: list[dict[str, Any]] = []
    all_settings: dict[str, Any] = {}
    mask_records = []
    skipped_masks = []
    effective_overrides = {}
    white_balance_records: list[dict[str, Any]] = []
    last_export: Path | None = None
    stages = professional_recipe_stages(recipe)
    for index, (stage_name, stage_recipe) in enumerate(stages, start=1):
        if stage_name == "local-light":
            if "apply_ai_mask" not in client.tools:
                raise LightroomError("当前 Lightroom 插件缺少原生局部蒙版接口，请安装工程内更新后的插件并重新加载。")
            before_local = last_export
            for target, values in native_mask_settings(stage_recipe):
                result = client.call_tool("apply_ai_mask", {"photo_id": str(original), "target": target, "settings": values}, timeout=100.0)
                if (isinstance(result, dict) and result.get("skipped") is True
                        and result.get("success") is False and result.get("target") == target
                        and result.get("reason") == "no_mask_created" and not result.get("mask_id")):
                    skipped_masks.append(result)
                    continue
                if not isinstance(result, dict) or not result.get("success") or not result.get("mask_id"):
                    raise LightroomError(f"Lightroom {target} 蒙版未确认完成")
                mask_records.append(result)
            stage_dir = output_dir / f"{index:02d}-local-light"
            last_export = _render_stage_feedback(client, original, stage_dir, export_edge=export_edge)
            review = stage_reviewer(stage_name, before_local, last_export) if stage_reviewer else None
            rolled_back = False
            attempts = [{"white_strength": 1.0, "export": str(last_export), "review": review}]
            if (review and review.get("rejects") is True
                    and set(review.get("issue_codes") or []) == {"highlight_clipping"}
                    and any(m.whites > 0 for m in stage_recipe.masks)):
                # Preserve sky density when only the white endpoint is too
                # bright. Undoing the entire mask discarded useful recovery.
                for strength in (.5, 0.0):
                    reduced = stage_recipe.model_copy(deep=True)
                    for mask in reduced.masks:
                        if mask.whites > 0:
                            mask.whites *= strength
                    owned_masks = {record["target"]: record["mask_id"] for record in mask_records}
                    revised = []
                    for target, values in native_mask_settings(reduced):
                        if target not in owned_masks:
                            continue
                        result = client.call_tool("apply_ai_mask", {"photo_id": str(original), "target": target,
                            "settings": values}, timeout=100.0)
                        if not isinstance(result, dict) or not result.get("success") or result.get("mask_id") != owned_masks[target]:
                            raise LightroomError(f"Lightroom {target} 蒙版高光修正未确认完成")
                        revised.append(result)
                    mask_records = revised
                    last_export = _render_stage_feedback(client, original, stage_dir / f"whites-{strength:.2f}", export_edge=export_edge)
                    review = stage_reviewer(stage_name, before_local, last_export)
                    attempts.append({"white_strength": strength, "export": str(last_export), "review": review})
                    if review.get("rejects") is not True:
                        effective_overrides["masks"] = [m.model_dump() for m in reduced.masks]
                        break
            if review and review.get("rejects") is True:
                for record in mask_records:
                    client.call_tool("apply_ai_mask", {"photo_id": str(original), "target": record["target"], "settings": {}, "reset": True})
                last_export = _render_stage_feedback(client, original, stage_dir / "rollback", export_edge=export_edge)
                rolled_back = True
            stage_records.append({"index": index, "stage": stage_name, "settings_applied": {},
                                  "masks": list(mask_records), "export": str(last_export),
                                  "skipped_masks": list(skipped_masks),
                                  "develop_after": current, "checkpoint": None,
                                  "attempts": attempts,
                                  "visual_review": review, "rolled_back": rolled_back})
            if rolled_back:
                mask_records = []
            continue
        before_stage = dict(current)
        conversion_audit: dict[str, Any] = {}
        settings = recipe_to_develop_settings(stage_recipe, current, audit=conversion_audit)
        white_balance = conversion_audit.get("white_balance")
        if white_balance is not None:
            white_balance_records.append(white_balance)
            if white_balance["status"] == "preserved":
                effective_overrides.update(temperature=0.0, tint=0.0)
        if not settings:
            if white_balance is not None:
                stage_records.append({
                    "index": index, "stage": stage_name, "settings_applied": {},
                    "develop_after": dict(current), "export": None, "checkpoint": None,
                    "skipped": True, "white_balance": white_balance,
                    "visual_review": None, "rolled_back": False,
                    "accepted_strength": 0.0, "attempts": [],
                })
            continue
        client.call_tool("set_develop_settings", {"photo_id": str(original), "settings": settings})
        metadata_after = client.call_tool("get_photo_metadata", {"photo_id": str(original)})
        current = _flatten_develop_settings(metadata_after.get("developSettings") or {})
        if stage_name == "color-correction":
            for key, expected in settings.items():
                if key.startswith(("ColorGrade", "SplitToning")):
                    actual = current.get(key)
                    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                            or not math.isfinite(actual) or abs(actual - expected) > .51):
                        raise LightroomError(f"Lightroom 颜色分级未正确回读：{key}，请求 {expected}，实际 {actual}")
        stage_dir = output_dir / f"{index:02d}-{stage_name}"
        exported = _render_stage_feedback(client, original, stage_dir, export_edge=export_edge)
        review = None
        rolled_back = False
        attempts = []
        accepted_strength = 1.0
        if stage_reviewer is not None and stage_name in {"color-foundation", "tonal-shape", "color-correction", "creative-look"}:
            review = stage_reviewer(stage_name, last_export, exported)
            attempts.append({"strength": 1.0, "export": str(exported), "review": review})
            if review.get("rejects") is True:
                # Restore the exact settings touched by this stage, never an
                # inverse curve or a guessed negative slider delta.
                rollback = {key: before_stage[key] for key in settings if key in before_stage}
                missing = set(settings) - set(rollback)
                if not missing:
                    # A curve that is too strong need not erase the entire tone
                    # design. Try bounded smaller moves from the exact prior state.
                    if stage_name == "tonal-shape":
                        for strength in (.65, .35):
                            client.call_tool("set_develop_settings", {"photo_id": str(original), "settings": rollback})
                            reduced = _attenuate_tonal_recipe(stage_recipe, before_stage, strength)
                            settings = rollback | recipe_to_develop_settings(reduced, before_stage)
                            client.call_tool("set_develop_settings", {"photo_id": str(original), "settings": settings})
                            metadata_after = client.call_tool("get_photo_metadata", {"photo_id": str(original)})
                            current = _flatten_develop_settings(metadata_after.get("developSettings") or {})
                            exported = _render_stage_feedback(client, original, stage_dir / f"strength-{strength:.2f}", export_edge=export_edge)
                            review = stage_reviewer(stage_name, last_export, exported)
                            attempts.append({"strength": strength, "export": str(exported), "review": review})
                            if review.get("rejects") is not True:
                                accepted_strength = strength
                                effective_overrides.update(reduced.model_dump(include={"contrast", "texture", "curve", "parametric_tone"}))
                                break
                    if review.get("rejects") is True:
                        client.call_tool("set_develop_settings", {"photo_id": str(original), "settings": rollback})
                        settings = rollback
                        metadata_after = client.call_tool("get_photo_metadata", {"photo_id": str(original)})
                        current = _flatten_develop_settings(metadata_after.get("developSettings") or {})
                        exported = _render_stage_feedback(client, original, stage_dir / "rollback", export_edge=export_edge)
                        rolled_back = True
                        accepted_strength = 0.0
                else:
                    review["rollback_unavailable"] = sorted(missing)
        checkpoint = None
        if checkpoint_prefix:
            checkpoint = create_lightroom_checkpoint(
                client,
                photo=original,
                name=f"{checkpoint_prefix}-{index:02d}-{stage_name}-{uuid.uuid4().hex[:8]}",
                settings=all_settings | settings,
            )
        all_settings.update(settings)
        stage_records.append(
            {
                "index": index,
                "stage": stage_name,
                "settings_applied": settings,
                "develop_after": metadata_after.get("developSettings"),
                "export": str(exported),
                "checkpoint": checkpoint,
                "visual_review": review,
                "rolled_back": rolled_back,
                "accepted_strength": accepted_strength,
                "attempts": attempts,
                **({"white_balance": white_balance} if white_balance is not None else {}),
            }
        )
        last_export = exported
    final_export = _export_photo(client, original, output_dir / "final", export_edge=export_edge)
    effective = recipe.model_dump() | effective_overrides
    skipped_targets = {item["target"] for item in skipped_masks}
    effective["masks"] = [mask for mask in effective["masks"] if mask["target"] not in skipped_targets]
    for target, keys in {"sky": ("sky_dehaze",),
                         "subject": ("subject_clarity", "subject_sharpen", "subject_lift", "subject_vibrance"),
                         "background": ("background_darken", "background_desat")}.items():
        if target in skipped_targets:
            for key in keys:
                effective["local"][key] = 0.0
    neutral = Recipe().model_dump()
    stage_fields = {
        "color-foundation": ("hsl",),
        "local-light": ("masks", "local"),
        "tonal-shape": ("contrast", "texture", "curve", "parametric_tone"),
        "color-correction": ("color_grading",),
        "creative-look": ("vibrance", "clarity", "dehaze", "vignette", "grain", "finish"),
    }
    for record in stage_records:
        if record.get("rolled_back"):
            for key in stage_fields.get(record["stage"], ()):
                effective[key] = neutral[key]
            if (record["stage"] == "color-foundation" and recipe.saturation < 0) or (
                    record["stage"] == "creative-look" and recipe.saturation > 0):
                effective["saturation"] = 0.0
    return {
        "ok": True,
        "settings": all_settings,
        "develop_before": develop_before,
        "develop_after": stage_records[-1]["develop_after"] if stage_records else develop_before,
        "export": final_export,
        "stages": stage_records,
        "masks": mask_records,
        "skipped_masks": skipped_masks,
        "effective_recipe": effective,
        "white_balance": white_balance_records,
        **color_audit,
    }


def apply_recipe_pass(
    client: LightroomMcpClient,
    *,
    original: Path,
    recipe: Recipe,
    output_dir: Path,
    export_edge: int = 2400,
    stage_reviewer=None,
) -> dict[str, Any]:
    """Apply a recipe through staged Lightroom passes and return the final render."""
    return apply_professional_recipe(
        client,
        original=original,
        recipe=recipe,
        output_dir=output_dir,
        export_edge=export_edge,
        stage_reviewer=stage_reviewer,
    )


def finish_job_in_lightroom(
    client: LightroomMcpClient,
    *,
    original: Path,
    recipe: Recipe,
    output_dir: Path,
    audit_path: Path,
    slot: str,
    export_edge: int = 2400,
) -> dict[str, Any]:
    if not original.is_file():
        raise LightroomError(f"找不到作业底片：{original}")
    result = apply_recipe_pass(
        client,
        original=original,
        recipe=recipe,
        output_dir=output_dir,
        export_edge=export_edge,
    )
    metadata_before = result["develop_before"]
    metadata_after = result["develop_after"]
    audit = {
        "slot": slot,
        "source_copy": str(original),
        "settings_applied": result["settings"],
        "develop_before": metadata_before,
        "develop_after": metadata_after,
        "export": str(result["export"]),
        "white_balance": result.get("white_balance", []),
        **({"color_grading_conversion": result["color_grading_conversion"]}
           if "color_grading_conversion" in result else {}),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**result, "audit": audit_path}


def _ensure_photo(client: LightroomMcpClient, original: Path) -> dict[str, Any]:
    if not original.is_file():
        raise LightroomError(f"找不到作业底片：{original}")
    try:
        return client.call_tool("get_photo_metadata", {"photo_id": str(original)})
    except LightroomError as exc:
        if "not found" not in str(exc).lower():
            raise
        client.call_tool("import_photos", {"source_path": str(original)}, timeout=300.0)
        return client.call_tool("get_photo_metadata", {"photo_id": str(original)})


def _render_stage_feedback(
    client: LightroomMcpClient,
    original: Path,
    output_dir: Path,
    *,
    export_edge: int,
) -> Path:
    """Export fresh pixels; Library thumbnails can lag behind Develop edits."""
    return _export_photo(client, original, output_dir, export_edge=min(1600, export_edge))


def _export_photo(
    client: LightroomMcpClient,
    original: Path,
    output_dir: Path,
    *,
    export_edge: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    client.call_tool(
        "export_photos",
        {
            "photo_ids": [str(original)],
            "destination": str(output_dir),
            "format": "jpeg",
            "quality": 92,
            "width": export_edge,
            "height": export_edge,
        },
        timeout=300.0,
    )
    exported = sorted(
        (p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not exported:
        raise LightroomError("Lightroom 报告导出成功，但输出目录没有 JPEG")
    return exported[0]


def _flatten_develop_settings(settings: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "whiteBalance": "WhiteBalance",
        "temperature": "Temperature",
        "tint": "Tint",
        "exposure": "Exposure2012",
        "contrast": "Contrast2012",
        "highlights": "Highlights2012",
        "shadows": "Shadows2012",
        "whites": "Whites2012",
        "blacks": "Blacks2012",
        "texture": "Texture",
        "clarity": "Clarity2012",
        "dehaze": "Dehaze",
        "vibrance": "Vibrance",
        "saturation": "Saturation",
        "toneCurveName": "ToneCurveName2012", "toneCurve": "ToneCurvePV2012",
        "toneCurveRed": "ToneCurvePV2012Red", "toneCurveGreen": "ToneCurvePV2012Green",
        "toneCurveBlue": "ToneCurvePV2012Blue",
        "parametricShadows": "ParametricShadows", "parametricDarks": "ParametricDarks",
        "parametricLights": "ParametricLights", "parametricHighlights": "ParametricHighlights",
    }
    out = {target: value for source, target in aliases.items() if (value := settings.get(source)) is not None}
    hsl = settings.get("hsl")
    if isinstance(hsl, dict):
        out.update(hsl)
    if isinstance(settings.get("sdk"), dict):
        out.update({key: value for key, value in settings["sdk"].items() if value is not None})
    out.update({key: value for key, value in settings.items() if key[:1].isupper() and value is not None})
    return out


def _curve_to_lightroom(points: list[tuple[float, float]]) -> list[int]:
    cleaned = sorted({round(max(0.0, min(1.0, x)), 6): max(0.0, min(1.0, y)) for x, y in points}.items())
    if not cleaned or cleaned[0][0] > 0.0:
        cleaned.insert(0, (0.0, 0.0))
    if cleaned[-1][0] < 1.0:
        cleaned.append((1.0, 1.0))
    flat: list[int] = []
    last_x = -1
    for x, y in cleaned[:32]:
        xi = int(round(x * 255.0))
        yi = int(round(y * 255.0))
        if xi <= last_x:
            continue
        flat.extend([xi, yi])
        last_x = xi
    if flat[-2] != 255:
        flat.extend([255, 255])
    return flat


def _set_absolute(
    out: dict[str, Any],
    key: str,
    value: float | int | None,
    lo: float,
    hi: float,
    *,
    integer: bool = False,
) -> None:
    if value is None:
        return
    bounded = max(lo, min(hi, float(value)))
    out[key] = int(round(bounded)) if integer else round(bounded, 4)


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unique_review_dir(root: Path, slot: str) -> Path:
    return root / "lightroom" / f"{slot}_{uuid.uuid4().hex[:10]}"
