from superphoto.macos_app import _launcher_script
from pathlib import Path
import pytest


@pytest.mark.parametrize("changed", ["index.html", "app.js", "styles.css"])
def test_desktop_entry_url_changes_with_any_ui_file(tmp_path, changed):
    from superphoto.desktop import ui_url
    web = tmp_path / "web"
    web.mkdir()
    for name in ("index.html", "app.js", "styles.css"):
        (web / name).write_text(name)
    initial = ui_url("http://127.0.0.1:8765", tmp_path)
    assert initial == ui_url("http://127.0.0.1:8765/", tmp_path)
    (web / changed).write_text("updated " + changed)
    assert ui_url("http://127.0.0.1:8765", tmp_path) != initial


def test_launcher_does_not_read_desktop_project():
    script = _launcher_script(Path("/tmp/python3"), Path("/tmp/site-packages"), Path("/tmp/node"))
    assert "Desktop/Projects" not in script
    assert "uv run --project" not in script
    assert "SUPERPHOTO_ROOT" in script
    assert f'SUPERPHOTO_NODE="{Path("/tmp/node")}"' in script
    assert "Application Support/SuperPhoto" in script.replace("\\", "/")
