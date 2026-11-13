from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from superphoto.lightroom import (
    LightroomError,
    LightroomMcpClient,
    LightroomRuntime,
    apply_professional_recipe,
    create_lightroom_checkpoint,
    discover_runtime,
    install_lightroom_plugin,
    professional_recipe_stages,
    recipe_to_develop_settings,
    wait_for_lightroom_connection,
)
from superphoto.models import (
    CropControls,
    DetailControls,
    FinishControls,
    HueShift,
    OpticsControls,
    ParametricTone,
    Recipe,
    ToneCurve,
)


class _SettlingClient:
    def __init__(self) -> None:
        self.calls = 0

    def status(self) -> dict:
        self.calls += 1
        if self.calls < 3:
            return {"available": True, "connected": False, "error": "Lightroom plugin not connected", "tools": []}
        return {"available": True, "connected": True, "tools": ["get_selected_photos"]}


def test_wait_for_lightroom_connection_allows_cold_socket_settle() -> None:
    client = _SettlingClient()
    status = wait_for_lightroom_connection(client, timeout=1.0, interval=0.0)
    assert status["connected"] is True
    assert client.calls == 3


def test_discover_runtime_uses_explicit_node_with_empty_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    node.chmod(0o755)
    server = tmp_path / "index.js"
    server.write_text("", encoding="utf-8")
    monkeypatch.setenv("SUPERPHOTO_NODE", str(node))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    runtime = discover_runtime(server)

    assert runtime is not None
    assert runtime.command == [str(node.resolve()), str(server.resolve()), "stdio"]


def test_recipe_to_develop_settings_is_additive_and_bounded() -> None:
    recipe = Recipe(
        exposure_ev=0.4,
        temperature=10,
        tint=-4,
        highlights=-0.3,
        saturation=0.2,
        vignette=0.15,
        grain=0.12,
        hsl={"orange": HueShift(hue=3.0, sat=-0.2, lum=0.1)},
        curve=ToneCurve(rgb=[(0.0, 0.0), (0.5, 0.55), (1.0, 1.0)]),
    )
    current = {
        "Temperature": 5200,
        "Tint": 8,
        "Exposure2012": 0.2,
        "Highlights2012": -10,
        "Saturation": 90,
        "HueAdjustmentOrange": 95,
    }

    settings = recipe_to_develop_settings(recipe, current)

    assert settings["Exposure2012"] == 0.6
    assert settings["Temperature"] == 5650
    assert settings["Tint"] == 4
    assert settings["Highlights2012"] == -40
    assert settings["Saturation"] == 100
    assert settings["HueAdjustmentOrange"] == 100
    assert settings["SaturationAdjustmentOrange"] == -20
    assert settings["PostCropVignetteAmount"] == -15
    assert settings["GrainAmount"] == 12
    assert settings["ToneCurvePV2012"] == [0, 0, 128, 140, 255, 255]
    assert settings["ToneCurveName2012"] == "Custom"


def test_professional_controls_map_to_enhanced_mcp_schema() -> None:
    recipe = Recipe(
        texture=-0.18,
        parametric_tone=ParametricTone(shadows=-0.12, lights=0.08, shadow_split=22),
        detail=DetailControls(
            sharpness=58,
            sharpen_radius=1.2,
            sharpen_detail=26,
            sharpen_masking=48,
            luminance_nr=22,
            color_nr=30,
        ),
        optics=OpticsControls(lens_profile_enable=True, perspective_vertical=4, perspective_scale=102),
        finish=FinishControls(
            vignette_midpoint=42,
            vignette_roundness=-10,
            vignette_feather=76,
            vignette_style=1,
            grain_size=56,
            grain_roughness=60,
        ),
        crop=CropControls(top=0.02, left=0.03, bottom=0.98, right=0.97, angle=-1.2),
    )

    settings = recipe_to_develop_settings(recipe)

    assert settings["Texture"] == -18
    assert settings["ParametricShadows"] == -12
    assert settings["ParametricLights"] == 8
    assert settings["ParametricShadowSplit"] == 22
    assert settings["Sharpness"] == 58
    assert settings["SharpenRadius"] == 1.2
    assert settings["LuminanceSmoothing"] == 22
    assert settings["ColorNoiseReduction"] == 30
    assert settings["LensProfileEnable"] is True
    assert settings["PerspectiveVertical"] == 4
    assert settings["PerspectiveScale"] == 102
    assert settings["PostCropVignetteFeather"] == 76
    assert settings["GrainFrequency"] == 60
    assert settings["CropAngle"] == -1.2


def test_professional_recipe_splits_settings_into_five_ordered_stages() -> None:
    recipe = Recipe(
        exposure_ev=0.25,
        contrast=0.12,
        texture=0.08,
        hsl={"green": HueShift(sat=-0.2)},
        vibrance=0.16,
        grain=0.1,
        detail=DetailControls(sharpness=45),
        optics=OpticsControls(lens_profile_enable=True),
    )

    stages = professional_recipe_stages(recipe)

    assert [name for name, _ in stages] == [
        "color-foundation",
        "technical-correction",
        "tonal-shape",
        "creative-look",
        "detail-output",
    ]
    assert "SaturationAdjustmentGreen" in recipe_to_develop_settings(stages[0][1])
    assert set(recipe_to_develop_settings(stages[1][1])) == {"Exposure2012"}
    assert {"Contrast2012", "Texture"} <= set(recipe_to_develop_settings(stages[2][1]))
    assert {"Vibrance", "GrainAmount"} <= set(recipe_to_develop_settings(stages[3][1]))
    assert {"Sharpness", "LensProfileEnable"} <= set(recipe_to_develop_settings(stages[4][1]))


def test_apply_professional_recipe_exports_each_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"raw")

    class Client:
        tools = {"get_photo_metadata", "set_develop_settings", "export_photos"}

        def __init__(self) -> None:
            self.settings: dict[str, object] = {}
            self.calls: list[str] = []

        def call_tool(self, name, arguments=None, *, timeout=None):
            args = arguments or {}
            self.calls.append(name)
            if name == "get_photo_metadata":
                return {"developSettings": dict(self.settings)}
            if name == "set_develop_settings":
                self.settings.update(args["settings"])
                return {"ok": True}
            if name == "export_photos":
                destination = Path(args["destination"])
                (destination / "render.jpg").write_bytes(b"jpeg")
                return {"count": 1}
            raise AssertionError(name)

    client = Client()
    result = apply_professional_recipe(
        client,  # type: ignore[arg-type]
        original=photo,
        recipe=Recipe(
            exposure_ev=0.2,
            contrast=0.1,
            hsl={"blue": HueShift(sat=-0.1)},
            grain=0.1,
            detail=DetailControls(sharpness=40),
        ),
        output_dir=tmp_path / "exports",
        export_edge=1600,
    )

    assert [stage["stage"] for stage in result["stages"]] == [
        "color-foundation",
        "technical-correction",
        "tonal-shape",
        "creative-look",
        "detail-output",
    ]
    assert client.calls.count("set_develop_settings") == 5
    assert client.calls.count("export_photos") == 6
    assert Path(result["export"]).read_bytes() == b"jpeg"


def test_apply_professional_recipe_exports_fresh_intermediate_pixels(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"raw")

    class Client:
        tools = {"get_photo_metadata", "set_develop_settings", "get_photo_preview", "export_photos"}

        def __init__(self) -> None:
            self.settings: dict[str, object] = {}
            self.calls: list[str] = []

        def call_tool(self, name, arguments=None, *, timeout=None):
            args = arguments or {}
            self.calls.append(name)
            if name == "get_photo_metadata":
                return {"developSettings": dict(self.settings)}
            if name == "set_develop_settings":
                self.settings.update(args["settings"])
                return {"ok": True}
            if name == "get_photo_preview":
                target = Path(args["destination_dir"]) / args["filename"]
                target.write_bytes(b"preview")
                return {"saved_to": str(target)}
            if name == "export_photos":
                destination = Path(args["destination"])
                (destination / "final.jpg").write_bytes(b"final")
                return {"count": 1}
            raise AssertionError(name)

    client = Client()
    result = apply_professional_recipe(
        client,  # type: ignore[arg-type]
        original=photo,
        recipe=Recipe(exposure_ev=0.2, contrast=0.1, grain=0.1),
        output_dir=tmp_path / "exports",
    )

    assert client.calls.count("get_photo_preview") == 0
    assert client.calls.count("export_photos") == 4
    assert Path(result["export"]).read_bytes() == b"final"
    assert all(Path(stage["export"]).read_bytes() == b"final" for stage in result["stages"])


def test_checkpoint_uses_explicit_sorted_setting_allowlist(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"raw")

    class Client:
        tools = {"create_develop_preset"}

        def __init__(self) -> None:
            self.arguments = None

        def call_tool(self, name, arguments=None, *, timeout=None):
            assert name == "create_develop_preset"
            self.arguments = arguments
            return {"success": True, "name": arguments["preset_name"]}

    client = Client()
    result = create_lightroom_checkpoint(
        client,  # type: ignore[arg-type]
        photo=photo,
        name="SuperPhoto-test-A-accepted-v1",
        settings={"Texture": 5, "Exposure2012": 0.2},
    )

    assert result and result["success"] is True
    assert client.arguments["settings"] == ["Exposure2012", "Texture"]


def test_install_lightroom_plugin_replaces_existing_bundle_with_rollback_safe_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "vendor" / "lightroom-mcp"
    server = root / "server" / "dist" / "index.js"
    server.parent.mkdir(parents=True)
    server.write_text("", encoding="utf-8")
    plugin = root / "plugin" / "LightroomMCP.lrplugin"
    plugin.mkdir(parents=True)
    (plugin / "Info.lua").write_text("return {}", encoding="utf-8")
    (plugin / "HandlerPreview.lua").write_text("preview", encoding="utf-8")
    fake_home = tmp_path / "home"
    installed = fake_home / "Library" / "Application Support" / "Adobe" / "Lightroom" / "Modules" / "LightroomMCP.lrplugin"
    installed.mkdir(parents=True)
    (installed / "Info.lua").write_text("old", encoding="utf-8")
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    node.chmod(0o755)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("SUPERPHOTO_NODE", str(node))

    result = install_lightroom_plugin(server)

    assert result["ok"] is True
    assert result["restart_required"] is True
    assert (installed / "HandlerPreview.lua").read_text(encoding="utf-8") == "preview"
    assert (installed / "Info.lua").read_text(encoding="utf-8") == "return {}"


def test_apply_professional_recipe_falls_back_when_preview_handler_is_not_loaded(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"raw")

    class Client:
        tools = {"get_photo_metadata", "set_develop_settings", "get_photo_preview", "export_photos"}

        def __init__(self) -> None:
            self.settings: dict[str, object] = {}
            self.calls: list[str] = []

        def call_tool(self, name, arguments=None, *, timeout=None):
            args = arguments or {}
            self.calls.append(name)
            if name == "get_photo_metadata":
                return {"developSettings": dict(self.settings)}
            if name == "set_develop_settings":
                self.settings.update(args["settings"])
                return {"ok": True}
            if name == "get_photo_preview":
                raise LightroomError("Unknown action: get_photo_preview")
            if name == "export_photos":
                destination = Path(args["destination"])
                (destination / "render.jpg").write_bytes(b"jpeg")
                return {"count": 1}
            raise AssertionError(name)

    client = Client()
    result = apply_professional_recipe(
        client,  # type: ignore[arg-type]
        original=photo,
        recipe=Recipe(exposure_ev=0.2, contrast=0.1),
        output_dir=tmp_path / "exports",
    )

    assert client.calls.count("get_photo_preview") == 0
    assert client.calls.count("export_photos") == 3
    assert Path(result["export"]).read_bytes() == b"jpeg"


def test_mcp_client_handshake_and_tool_call(tmp_path: Path) -> None:
    fake = tmp_path / "fake_mcp.py"
    fake.write_text(
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' not in msg:
        continue
    if msg['method'] == 'initialize':
        result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fake', 'version': '1'}}
    elif msg['method'] == 'tools/list':
        result = {'tools': [{'name': 'get_selected_photos', 'inputSchema': {'type': 'object'}}]}
    elif msg['method'] == 'tools/call':
        result = {'content': [{'type': 'text', 'text': json.dumps({'count': 1, 'photos': [{'id': 'p1'}]})}]}
    else:
        result = {}
    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)
""".strip(),
        encoding="utf-8",
    )
    client = LightroomMcpClient(LightroomRuntime([sys.executable, str(fake)], tmp_path), timeout=2)
    try:
        result = client.call_tool("get_selected_photos", {"limit": 1})
        assert result["count"] == 1
        assert client.tools == {"get_selected_photos"}
        with pytest.raises(LightroomError, match="不支持工具"):
            client.call_tool("set_develop_settings", {})
    finally:
        client.close()


def test_mcp_tool_error_is_user_actionable(tmp_path: Path) -> None:
    fake = tmp_path / "fake_mcp_error.py"
    fake.write_text(
        """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' not in msg:
        continue
    if msg['method'] == 'initialize':
        result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fake', 'version': '1'}}
    elif msg['method'] == 'tools/list':
        result = {'tools': [{'name': 'get_selected_photos', 'inputSchema': {'type': 'object'}}]}
    else:
        result = {'content': [{'type': 'text', 'text': 'Lightroom plugin not connected.'}], 'isError': True}
    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)
""".strip(),
        encoding="utf-8",
    )
    client = LightroomMcpClient(LightroomRuntime([sys.executable, str(fake)], tmp_path), timeout=2)
    try:
        with pytest.raises(LightroomError, match="plugin not connected"):
            client.call_tool("get_selected_photos")
    finally:
        client.close()
