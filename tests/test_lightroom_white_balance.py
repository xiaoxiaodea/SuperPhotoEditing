from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from superphoto.lightroom import (
    apply_professional_recipe,
    finish_job_in_lightroom,
    professional_recipe_stages,
    recipe_to_develop_settings,
)
from superphoto.models import Recipe


def test_as_shot_warming_uses_real_kelvin_and_preserves_tint() -> None:
    audit: dict = {}
    settings = recipe_to_develop_settings(
        Recipe(temperature=0.5),
        {"whiteBalance": "As Shot", "sdk": {"Temperature": 5950, "Tint": 18}},
        audit=audit,
    )
    assert settings == {"WhiteBalance": "Custom", "Temperature": 5972.5, "Tint": 18}
    assert audit["white_balance"]["status"] == "applied"
    assert audit["white_balance"]["base"]["Temperature"] == 5950


def test_tint_adjustment_preserves_known_as_shot_temperature() -> None:
    settings = recipe_to_develop_settings(
        Recipe(tint=-3), {"WhiteBalance": "As Shot", "Temperature": 5950, "Tint": 18}
    )
    assert settings == {"WhiteBalance": "Custom", "Temperature": 5950, "Tint": 15}


def test_null_sdk_values_do_not_discard_numeric_readback_aliases() -> None:
    settings = recipe_to_develop_settings(
        Recipe(temperature=2),
        {"whiteBalance": "As Shot", "temperature": 5950, "tint": 18,
         "sdk": {"WhiteBalance": "As Shot", "Temperature": None, "Tint": None}},
    )
    assert settings["Temperature"] == 6040
    assert settings["Tint"] == 18


@pytest.mark.parametrize("base", [
    {},
    {"WhiteBalance": "As Shot"},
    {"WhiteBalance": "As Shot", "Temperature": 5950},
    {"WhiteBalance": "Custom", "Tint": 18},
    {"Temperature": "5950", "Tint": 18},
    {"Temperature": float("nan"), "Tint": 18},
    {"Temperature": 5950, "Tint": float("inf")},
    {"Temperature": True, "Tint": False},
    {"Temperature": 0, "Tint": 0},
])
def test_unknown_or_non_kelvin_base_preserves_wb_but_keeps_other_edits(base: dict) -> None:
    audit: dict = {}
    settings = recipe_to_develop_settings(Recipe(temperature=0.5, tint=2, contrast=0.1), base, audit=audit)
    assert settings == {"Contrast2012": 10}
    assert audit["white_balance"]["status"] == "preserved"
    assert audit["white_balance"]["reason"] == "unavailable_white_balance_base"
    json.dumps(audit, allow_nan=False)


def test_neutral_wb_does_not_change_as_shot_mode() -> None:
    assert recipe_to_develop_settings(
        Recipe(), {"WhiteBalance": "As Shot", "Temperature": 5950, "Tint": 18}
    ) == {}


def test_white_balance_only_recipe_is_still_scheduled_without_a_base() -> None:
    assert [name for name, _ in professional_recipe_stages(Recipe(temperature=0.5))] == [
        "technical-correction"
    ]


class _WhiteBalanceClient:
    tools = {"get_photo_metadata", "set_develop_settings", "export_photos"}

    def __init__(self, settings: dict) -> None:
        self.settings = dict(settings)
        self.writes: list[dict] = []

    def call_tool(self, name: str, arguments: dict | None = None, *, timeout=None):
        args = arguments or {}
        if name == "get_photo_metadata":
            return {"developSettings": {"sdk": dict(self.settings)}}
        if name == "set_develop_settings":
            self.writes.append(dict(args["settings"]))
            self.settings.update(args["settings"])
            return {"success": True}
        if name == "export_photos":
            (Path(args["destination"]) / "result.jpg").write_bytes(b"mock-export")
            return {"success": True}
        raise AssertionError(name)


@pytest.mark.parametrize("has_other_correction", [False, True])
def test_unresolved_wb_is_audited_and_not_claimed_in_effective_recipe(
    tmp_path: Path, has_other_correction: bool,
) -> None:
    source = tmp_path / "source.CR3"
    source.write_bytes(b"mock-raw")
    client = _WhiteBalanceClient({"WhiteBalance": "As Shot"})
    result = apply_professional_recipe(
        client, original=source, output_dir=tmp_path / "exports",
        recipe=Recipe(temperature=0.5, tint=1, highlights=-0.3 if has_other_correction else 0),
    )
    assert client.settings["WhiteBalance"] == "As Shot"
    assert not any({"WhiteBalance", "Temperature", "Tint"} & write.keys() for write in client.writes)
    assert result["effective_recipe"]["temperature"] == 0
    assert result["effective_recipe"]["tint"] == 0
    assert result["stages"][0]["white_balance"]["status"] == "preserved"
    assert result["white_balance"][0]["missing_or_invalid"] == ["Temperature", "Tint"]
    if has_other_correction:
        assert client.settings["Highlights2012"] == -30
    else:
        assert result["stages"][0]["skipped"] is True
        assert not client.writes
    assert result["export"].is_file()


def test_complete_wb_readback_reaches_the_native_apply_and_saved_audit(tmp_path: Path) -> None:
    source = tmp_path / "source.CR3"
    source.write_bytes(b"mock-raw")
    audit_path = tmp_path / "audit.json"
    client = _WhiteBalanceClient({"WhiteBalance": "As Shot", "Temperature": 5950, "Tint": 18})
    result = finish_job_in_lightroom(
        client, original=source, output_dir=tmp_path / "exports", audit_path=audit_path,
        recipe=Recipe(temperature=0.5), slot="B",
    )
    assert client.settings == {"WhiteBalance": "Custom", "Temperature": 5972.5, "Tint": 18}
    assert result["effective_recipe"]["temperature"] == 0.5
    assert json.loads(audit_path.read_text())["white_balance"][0]["base"]["Temperature"] == 5950


def test_native_white_balance_readback_guards() -> None:
    lua = shutil.which("lua")
    if lua is None:
        pytest.skip("Lua interpreter unavailable")
    root = Path(__file__).resolve().parents[1] / "vendor" / "lightroom-mcp"
    result = subprocess.run(
        [lua, "plugin/spec/white_balance_smoke.lua"], cwd=root, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
