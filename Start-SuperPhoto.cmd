@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)
uv sync --python 3.12 --extra raw
if errorlevel 1 (
  pause
  exit /b 1
)
uv run --no-sync superphoto app
if errorlevel 1 pause
