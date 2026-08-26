# SuperPhoto

Diagnose a photograph, render three parametric looks, and learn from the one you keep.

Correction (exposure, white balance, clipping) is separated from look (Portra, cinematic, landscape filmic). Pixel generation is not the editor. Variants are slider recipes you can inspect.

## Run

Double-click **SuperPhoto** on the Desktop (or in `~/Applications`). That starts the local engine and opens a window.

First time, or from the repo:

```bash
cd SuperPhotoEditing
uv sync --extra dev --extra raw
uv run superphoto install-app
open SuperPhoto.app
```

Ollama should be running. The app wakes it and uses `qwen3-vl:32b` for diagnosis. If that model is missing it starts a pull. If Ollama is down it falls back to Grok (when `XAI_API_KEY` is set) then to the local technical critic.

CLI still works:

```bash
uv run superphoto          # same as the app window
uv run superphoto serve    # browser at http://127.0.0.1:8765
```

CLI for a single file:

```bash
uv run superphoto process path/to/photo.jpg
```

## What is local vs API

| Piece | Where |
|---|---|
| Decode, EXIF, histogram, clipping, render | Local |
| Recipe library + preference counts | Local (`data/`) |
| Photographic critique | SpaceXAI `grok-4.6` if `XAI_API_KEY` is set, otherwise local |
| RAW | Optional `rawpy` |
| darktable-cli | Detected if installed; not required for v0.1 |

RAW files stay on disk. The cloud critic only receives a JPEG preview.
