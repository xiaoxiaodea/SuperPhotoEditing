from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from .desktop import project_root

BUNDLE_ID = "studio.superphoto.app"


def write_app(dest: Path | None = None) -> Path:
    root = project_root()
    dest = dest or (root / "SuperPhoto.app")
    if dest.exists():
        shutil.rmtree(dest)
    contents = dest / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        python = Path(shutil.which("python3") or "/usr/bin/python3")

    launcher = macos / "SuperPhoto"
    launcher.write_text(_launcher_script(root, python), encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    (contents / "Info.plist").write_text(_info_plist(), encoding="utf-8")
    _write_icon(resources / "AppIcon.icns")

    for extra in (_home() / "Applications", _home() / "Desktop"):
        extra.mkdir(parents=True, exist_ok=True)
        alias = extra / "SuperPhoto.app"
        if alias.exists() or alias.is_symlink():
            if alias.resolve() == dest.resolve():
                continue
            shutil.rmtree(alias, ignore_errors=True)
            try:
                alias.unlink()
            except FileNotFoundError:
                pass
        try:
            if extra == _home() / "Applications":
                if alias.exists():
                    shutil.rmtree(alias)
                shutil.copytree(dest, alias)
            else:
                os.symlink(dest, alias)
        except OSError:
            pass
    return dest


def _home() -> Path:
    return Path.home()


def _launcher_script(root: Path, python: Path) -> str:
    cfg = root / ".venv" / "pyvenv.cfg"
    base_python = python
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.startswith("home = "):
                home = Path(line.split("=", 1)[1].strip())
                for name in ("python3.12", "python3", "python"):
                    cand = home / name
                    if cand.is_file():
                        base_python = cand
                        break
    site = root / ".venv" / "lib" / "python3.12" / "site-packages"
    return f"""#!/bin/bash
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
export HOME="${{HOME}}"
export SUPERPHOTO_DATA_DIR="{root / "data"}"
export PYTHONPATH="{root / "src"}:{site}"
cd "{root}" || exit 1
mkdir -p "$HOME/Library/Logs"
ls "{root}/web/index.html" >/dev/null 2>&1 || true
if [ -x "/usr/bin/open" ]; then
  open -a Ollama >/dev/null 2>&1 || true
fi
exec "{base_python}" -m superphoto.desktop >> "$HOME/Library/Logs/SuperPhoto.log" 2>&1
"""


def _info_plist() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>zh_CN</string>
  <key>CFBundleDisplayName</key><string>SuperPhoto</string>
  <key>CFBundleExecutable</key><string>SuperPhoto</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>{escape(BUNDLE_ID)}</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>SuperPhoto</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSDesktopFolderUsageDescription</key>
  <string>SuperPhoto 需要读取桌面上的照片和项目文件。</string>
  <key>NSDocumentsFolderUsageDescription</key>
  <string>SuperPhoto 需要读取文稿中的照片。</string>
  <key>NSDownloadsFolderUsageDescription</key>
  <string>SuperPhoto 需要读取下载文件夹中的照片。</string>
  <key>NSRemovableVolumesUsageDescription</key>
  <string>SuperPhoto 需要读取存储卡上的 RAW 照片。</string>
</dict>
</plist>
"""


def _write_icon(icns: Path) -> None:
    from PIL import Image, ImageDraw

    master = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    draw = ImageDraw.Draw(master)
    draw.rounded_rectangle((40, 40, 984, 984), radius=220, fill=(20, 19, 16, 255))
    cx, cy, r = 512, 512, 290
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(212, 165, 116, 255), width=36)
    draw.ellipse((cx - 90, cy - 90, cx + 90, cy + 90), fill=(212, 165, 116, 255))
    for angle in range(0, 360, 45):
        import math

        rad = math.radians(angle)
        x1 = cx + int(math.cos(rad) * 210)
        y1 = cy + int(math.sin(rad) * 210)
        x2 = cx + int(math.cos(rad) * 330)
        y2 = cy + int(math.sin(rad) * 330)
        draw.line((x1, y1, x2, y2), fill=(212, 165, 116, 255), width=28)

    iconset = icns.parent / "AppIcon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    for size, retina in ((16, False), (16, True), (32, False), (32, True), (128, False), (128, True), (256, False), (256, True), (512, False), (512, True)):
        px = size * (2 if retina else 1)
        name = f"icon_{size}x{size}{'@2x' if retina else ''}.png"
        master.resize((px, px), Image.Resampling.LANCZOS).save(iconset / name)
    subprocess.run(["iconutil", "-c", "icns", "-o", str(icns), str(iconset)], check=False)
    shutil.rmtree(iconset, ignore_errors=True)
    if not icns.exists():
        master.resize((256, 256), Image.Resampling.LANCZOS).save(icns.with_suffix(".png"))
