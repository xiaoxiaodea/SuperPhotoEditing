from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from .desktop import project_root

BUNDLE_ID = "studio.superphoto.app"
LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
    "LaunchServices.framework/Support/lsregister"
)


def write_app(dest: Path | None = None) -> Path:
    """Install a double-clickable SuperPhoto.app.

    Finder cannot launch a *file* named SuperPhoto.app (symlink or alias).
    A signed .app that still reads ~/Desktop/Projects is also blocked by
    TCC. Copy the UI into the bundle and the Python env into
    ~/Library/Application Support/SuperPhoto, then write real app bundles
    to ~/Applications and the Desktop.
    """
    root = project_root()
    support = _support_dir()
    _sync_runtime(root, support)
    node = _bundle_node(support)
    icon = support / "AppIcon.icns"
    if not icon.is_file():
        _write_icon(icon)

    apps_dir = _home() / "Applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    dest = dest or (apps_dir / "SuperPhoto.app")
    bundles: list[Path] = [dest, root / "SuperPhoto.app", _home() / "Desktop" / "SuperPhoto.app"]
    seen: set[str] = set()
    python = _base_python(root)
    site = support / "site-packages"
    for bundle in bundles:
        key = str(bundle.resolve()) if bundle.exists() else str(bundle)
        if key in seen:
            continue
        seen.add(key)
        _write_bundle(root, bundle, icon, python, site, node)
        _codesign(bundle)
        _clear_quarantine(bundle)
        _register_bundle(bundle)
    return dest


def _write_bundle(root: Path, dest: Path, icon: Path, python: Path, site: Path, node: Path | None) -> None:
    _remove_path(dest)
    contents = dest / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    src_dest = resources / "src"
    web_dest = resources / "web"
    shutil.copytree(root / "src", src_dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "web", web_dest, ignore=shutil.ignore_patterns("__pycache__"))
    vendor_src = root / "vendor" / "lightroom-mcp"
    if vendor_src.is_dir():
        shutil.copytree(
            vendor_src,
            resources / "vendor" / "lightroom-mcp",
            ignore=shutil.ignore_patterns("tests", "*.map", ".DS_Store"),
        )
    for name in ("pyproject.toml", "uv.lock", ".env"):
        src = root / name
        if src.is_file():
            shutil.copy2(src, resources / name)

    script = resources / "launch.sh"
    script.write_text(_launcher_script(python, site, node), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _compile_stub(macos / "SuperPhoto")
    (contents / "Info.plist").write_text(_info_plist(), encoding="utf-8")
    if icon.is_file():
        shutil.copy2(icon, resources / "AppIcon.icns")
    else:
        _write_icon(resources / "AppIcon.icns")


def _sync_runtime(root: Path, support: Path) -> None:
    support.mkdir(parents=True, exist_ok=True)
    site_src = _venv_site(root / ".venv")
    site_dst = support / "site-packages"
    if site_src is not None:
        if not (site_dst / "numpy").exists():
            _clone_or_copy(site_src, site_dst)
        _neutralize_editable(site_dst, root)
    data_src = root / "data"
    data_dst = support / "data"
    if data_src.is_dir() and not (data_dst / "jobs").exists():
        _clone_or_copy(data_src, data_dst)
    else:
        data_dst.mkdir(parents=True, exist_ok=True)


def _bundle_node(support: Path) -> Path | None:
    """Keep a stable Node executable for Finder-launched app bundles."""
    found = shutil.which("node")
    if not found:
        for candidate in (Path("/opt/homebrew/bin/node"), Path("/usr/local/bin/node"), Path("/usr/bin/node")):
            if candidate.is_file():
                found = str(candidate)
                break
    if not found:
        return None
    source = Path(found).resolve()
    if not source.is_file():
        return None
    runtime = support / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    target = runtime / "node"
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return target


def _venv_site(venv: Path) -> Path | None:
    lib = venv / "lib"
    if not lib.is_dir():
        return None
    matches = sorted(lib.glob("python*/site-packages"))
    return matches[-1] if matches else None


def _neutralize_editable(site: Path, root: Path) -> None:
    marker = str(root)
    for pth in site.glob("*.pth"):
        try:
            text = pth.read_text(encoding="utf-8")
        except OSError:
            continue
        if "editable" in pth.name.lower() or marker in text:
            try:
                pth.unlink()
            except OSError:
                pass


def _clone_or_copy(src: Path, dst: Path) -> None:
    _remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cloned = subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True)
    if cloned.returncode != 0 or not dst.exists():
        _remove_path(dst)
        subprocess.run(["ditto", str(src), str(dst)], check=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _codesign(dest: Path) -> None:
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(dest)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _clear_quarantine(dest: Path) -> None:
    subprocess.run(["xattr", "-cr", str(dest)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _register_bundle(dest: Path) -> None:
    if Path(LSREGISTER).is_file():
        subprocess.run([LSREGISTER, "-f", str(dest)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _home() -> Path:
    return Path.home()


def _support_dir() -> Path:
    return _home() / "Library" / "Application Support" / "SuperPhoto"


_STUB_C = r"""
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    char exe[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0) {
        return 1;
    }
    char resolved[PATH_MAX];
    if (realpath(exe, resolved) == NULL) {
        strncpy(resolved, exe, sizeof(resolved) - 1);
        resolved[sizeof(resolved) - 1] = '\0';
    }
    char *macos = strstr(resolved, "/Contents/MacOS/");
    char script[PATH_MAX];
    if (macos != NULL) {
        size_t prefix = (size_t)(macos - resolved);
        snprintf(script, sizeof(script), "%.*s/Contents/Resources/launch.sh", (int)prefix, resolved);
    } else {
        snprintf(script, sizeof(script), "%s/../Resources/launch.sh", resolved);
    }
    char *args[] = {"/bin/bash", script, NULL};
    execv("/bin/bash", args);
    perror("SuperPhoto: exec /bin/bash");
    return 127;
}
"""


def _compile_stub(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["clang", "-Os", "-arch", "arm64", "-Wl,-no_adhoc_codesign", "-x", "c", "-", "-o", str(dest)],
        input=_STUB_C.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["clang", "-Os", "-arch", "arm64", "-x", "c", "-", "-o", str(dest)],
            input=_STUB_C.encode("utf-8"),
            capture_output=True,
        )
    if result.returncode != 0 or not dest.is_file():
        raise RuntimeError(
            "clang failed to build SuperPhoto launcher:\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _base_python(root: Path) -> Path:
    cfg = root / ".venv" / "pyvenv.cfg"
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.startswith("home = "):
                home = Path(line.split("=", 1)[1].strip())
                for name in ("python3.12", "python3", "python"):
                    cand = home / name
                    if cand.is_file():
                        return cand
    found = shutil.which("python3")
    return Path(found) if found else Path("/usr/bin/python3")


def _launcher_script(python: Path, site: Path, node: Path | None = None) -> str:
    support = _support_dir()
    data = support / "data"
    node_export = f'export SUPERPHOTO_NODE="{node}"' if node else ""
    return f"""#!/bin/bash
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export HOME="${{HOME}}"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
HERE="$(cd "$(dirname "$0")" && pwd)"
RESOURCES="$(cd "$HERE/../Resources" && pwd)"
LOG="$HOME/Library/Logs/SuperPhoto.log"
mkdir -p "$HOME/Library/Logs" "{data}"

export SUPERPHOTO_ROOT="$RESOURCES"
export SUPERPHOTO_DATA_DIR="{data}"
{node_export}
export PYTHONPATH="$RESOURCES/src:{site}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$RESOURCES" || exit 1
/usr/bin/open -a Ollama >/dev/null 2>&1 || true

PY="{python}"
if [ ! -x "$PY" ]; then
  /usr/bin/osascript -e 'display dialog "找不到 SuperPhoto 的 Python 运行环境。" buttons {{"好"}} default button 1 with title "SuperPhoto"' >/dev/null 2>&1 || true
  echo "missing python $PY" >> "$LOG"
  exit 1
fi
if [ ! -d "{site}" ]; then
  /usr/bin/osascript -e 'display dialog "SuperPhoto 的依赖还没装好。请在终端运行：\\nuv run superphoto install-app" buttons {{"好"}} default button 1 with title "SuperPhoto"' >/dev/null 2>&1 || true
  echo "missing site-packages {site}" >> "$LOG"
  exit 1
fi

echo "$(date) start PY=$PY ROOT=$RESOURCES" >> "$LOG"
exec "$PY" -u -m superphoto.desktop >> "$LOG" 2>&1
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
  <key>LSEnvironment</key>
  <dict>
    <key>PATH</key>
    <string>{escape(str(_home() / ".local" / "bin"))}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSAppleEventsUsageDescription</key>
  <string>SuperPhoto 需要打开本机的 Ollama。</string>
  <key>NSDesktopFolderUsageDescription</key>
  <string>SuperPhoto 需要读取桌面上的照片和项目文件。</string>
  <key>NSDocumentsFolderUsageDescription</key>
  <string>SuperPhoto 需要读取文稿中的照片。</string>
  <key>NSDownloadsFolderUsageDescription</key>
  <string>SuperPhoto 需要读取下载文件夹中的照片。</string>
  <key>NSRemovableVolumesUsageDescription</key>
  <string>SuperPhoto 需要读取存储卡上的 RAW 照片。</string>
  <key>NSLocalNetworkUsageDescription</key>
  <string>SuperPhoto 需要连接本机引擎来显示后期界面。</string>
  <key>NSAppTransportSecurity</key>
  <dict>
    <key>NSAllowsLocalNetworking</key><true/>
    <key>NSAllowsArbitraryLoads</key><true/>
  </dict>
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
