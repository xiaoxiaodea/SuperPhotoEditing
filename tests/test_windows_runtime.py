from pathlib import Path
from subprocess import CompletedProcess
import hashlib
import io
import zipfile

import numpy as np
import pytest

from superphoto import lightcraft as lc
from superphoto import semantic
from superphoto.lightroom import recipe_to_develop_settings
from superphoto.models import Recipe


@pytest.mark.parametrize("system,machine,asset", [
    ("win32", "AMD64", "lightcraft-0.21.1-windows-x64-portable.zip"),
    ("win32", "ARM64", "lightcraft-0.21.1-windows-arm64-portable.zip"),
    ("linux", "x86_64", "lightcraft-0.21.1-linux-x86_64.tar.gz"),
    ("linux", "aarch64", "lightcraft-0.21.1-linux-aarch64.tar.gz"),
    ("darwin", "arm64", "lightcraft-cli-0.21.1-macos-universal.zip"),
])
def test_platform_specific_installer(monkeypatch, system, machine, asset):
    monkeypatch.setattr(lc, "sys_platform", lambda: system)
    monkeypatch.setattr(lc.platform, "machine", lambda: machine)
    assert lc.release_asset() == asset
    assert lc.bundled_cli_path().suffix == (".exe" if system == "win32" else "")


def test_installer_verifies_platform_bundle_and_keeps_existing_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "sys_platform", lambda: "win32")
    monkeypatch.setattr(lc.platform, "machine", lambda: "AMD64")
    contents = io.BytesIO()
    with zipfile.ZipFile(contents, "w") as bundle:
        bundle.writestr("portable/lightcraft-cli.exe", b"test native binary")
        bundle.writestr("../planted.py", b"should never be extracted")
    payload = contents.getvalue()
    asset = lc.release_asset()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(lc, "_download", lambda url, path: path.write_bytes(payload))
    monkeypatch.setattr(lc.urllib.request, "urlopen", lambda *args, **kwargs:
                        io.BytesIO(f"{digest}  {asset}\n".encode()))
    monkeypatch.setattr(lc, "_version", lambda binary: "lightcraft-cli 0.21.1")
    target = tmp_path / "vendor" / "lightcraft-cli.exe"
    result = lc.install_cli(target)
    assert result["ok"]
    assert target.read_bytes() == b"test native binary"
    assert not (tmp_path / "planted.py").exists()
    monkeypatch.setattr(lc, "_version", lambda binary: "")
    target.write_bytes(b"existing installation")
    with pytest.raises(lc.LightCraftError, match="无法在本机运行"):
        lc.install_cli(target)
    assert target.read_bytes() == b"existing installation"


def test_cli_white_balance_does_not_invent_source_metadata(tmp_path):
    client = lc.LightCraftClient(tmp_path / "unused.exe")
    state = client.call_tool("get_photo_metadata", {"photo_id": str(tmp_path / "raw.dng")})
    current = state["developSettings"]
    assert current == {"WhiteBalance": "As Shot"}
    assert state["settings_source"] == "planned"
    assert state["source_metadata_available"] is False
    audit = {}
    settings = recipe_to_develop_settings(Recipe(temperature=3, tint=2), current, audit=audit)
    assert "Temperature" not in settings
    assert "WhiteBalance" not in settings
    assert audit["white_balance"]["status"] == "preserved"
    assert "wb" not in lc.adobe_to_lightcraft(current | settings, [])
    with pytest.raises(lc.LightCraftError, match="白平衡"):
        lc.adobe_to_lightcraft({"WhiteBalance": "Custom", "Temperature": 6500}, [])


def test_native_checkpoint_persists_complete_replayable_controls(tmp_path):
    import json

    source = tmp_path / "photo.jpg"
    source.write_bytes(b"source")
    client = lc.LightCraftClient(tmp_path / "unused.exe")
    client.call_tool("set_develop_settings", {"photo_id": str(source), "settings": {"Exposure2012": .3}})
    mask = client.call_tool("apply_ai_mask", {"photo_id": str(source), "target": "sky",
                                             "settings": {"local_Exposure": -.2}})
    assert mask["status"] == "planned" and mask["verified"] is False
    result = client.call_tool("create_develop_preset", {"photo_id": str(source), "preset_name": "accepted:/A"})
    saved = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert saved["light"]["exposure"] == .3
    assert saved["masks"][0]["adjust"]["exposure"] == -.2
    assert result["settings_source"] == "planned"
    assert result["format"] == "lightcraft-develop-json"


def test_status_checks_binary_instead_of_only_file_existence(tmp_path, monkeypatch):
    from superphoto import lightroom
    cli = tmp_path / "lightcraft-cli.exe"
    cli.write_bytes(b"wrong executable")
    monkeypatch.setattr(lightroom, "_client", None)
    monkeypatch.setattr(lc.subprocess, "run", lambda args, **kwargs: CompletedProcess(args, 1, "", "bad image"))
    status = lightroom.lightroom_status(True, cli)
    assert status["available"] is True
    assert status["connected"] is False


def test_semantic_resizes_only_predictions_without_creating_new_labels():
    labels = np.array([[0, 2], [2, 9]], dtype=np.int64)
    confidence = np.array([[.2, .6], [.7, 1]], dtype=np.float32)
    result, conf = semantic._resize_predictions(labels, confidence, (640, 960))
    assert result.shape == conf.shape == (640, 960)
    assert set(np.unique(result)) == {0, 2, 9}
    assert conf.dtype == np.float32
    assert conf.min() >= .2 and conf.max() <= 1


def test_semantic_respects_hf_home_and_requires_all_shards(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshot = tmp_path / "hub" / "models--test--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for filename in ("config.json", "preprocessor_config.json"):
        (snapshot / filename).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer1":"a.safetensors","layer2":"b.safetensors"}}', encoding="utf-8")
    (snapshot / "a.safetensors").write_bytes(b"a")
    assert not semantic._model_cached("test/model")
    (snapshot / "b.safetensors").write_bytes(b"b")
    assert semantic._model_cached("test/model")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    assert not semantic._model_cached("test/model")
