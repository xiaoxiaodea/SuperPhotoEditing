"""Exercise approved native LightCraft with repository/synthetic inputs only."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import httpx
import numpy as np
from PIL import Image

from superphoto.lightcraft import LightCraftClient, discover_cli


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / ".task-cache" / "windows-engine-verification"
    output.mkdir(parents=True, exist_ok=True)
    binary = discover_cli()
    if not binary:
        raise RuntimeError("Install the approved official CLI first")
    source = output / "input.jpg"
    shutil.copy2(root / "samples" / "landscape.jpg", source)
    client = LightCraftClient(binary)
    for name in ("baseline", "sky"):
        if name == "sky":
            client.call_tool("apply_ai_mask", {"photo_id": str(source), "target": "sky",
                                               "settings": {"local_Exposure": -.5}})
        client.call_tool("export_photos", {"photo_ids": [str(source)],
                                          "destination": str(output / name), "width": 640, "height": 640})
    before = np.asarray(Image.open(output / "baseline" / "render.jpg").convert("RGB"), dtype=np.float32)
    after = np.asarray(Image.open(output / "sky" / "render.jpg").convert("RGB"), dtype=np.float32)
    delta = np.mean(np.abs(before - after), axis=2)
    assert float(delta.mean()) > .05, "sky mask had no pixel effect"
    checkpoint = client.call_tool("create_develop_preset", {"photo_id": str(source), "preset_name": "sky-accepted"})
    replay = output / "checkpoint-replayed.jpg"
    completed = subprocess.run([str(binary), "render", str(source), "-o", str(replay),
                                "--settings", checkpoint["path"], "--size", "640"],
                               capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    assert completed.returncode == 0, completed.stderr
    replayed = np.asarray(Image.open(replay).convert("RGB"), dtype=np.float32)
    assert np.allclose(after, replayed, atol=1), "saved checkpoint did not reproduce the accepted render"
    report = {"checkpoint_replayed": True, "sky_mask": {"mean_absolute_pixel_change": float(delta.mean()),
                            "upper_half_change": float(delta[:delta.shape[0] // 2].mean()),
                            "lower_half_change": float(delta[delta.shape[0] // 2:].mean())},
              "refinement_downloads": []}
    report_path = root / ".task-cache" / "windows-lightcraft-verification" / "report.json"
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    with httpx.Client(base_url=previous["base"], trust_env=False, timeout=30) as http:
        for job in previous["jobs"]:
            refinement = job["refinement"]
            response = http.get(f"/api/jobs/{job['id']}/refinements/{refinement['id']}/image")
            response.raise_for_status()
            target = output / f"{Path(job['sample']).stem}-refined.jpg"
            target.write_bytes(response.content)
            with Image.open(target) as image:
                image.load()
                size = image.size
            assert response.content != Path(job["variants"][0]["path"]).read_bytes()
            report["refinement_downloads"].append({"job": job["id"], "bytes": len(response.content),
                                                    "size": size, "path": str(target)})
    synthetic = output / "synthetic-raw"
    if not synthetic.exists():
        result = subprocess.run([str(binary), "synth-merge", "hdr", "-o", str(synthetic)],
                                capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr)
    sources = list(synthetic.rglob("*.dng"))
    assert sources, "synthetic RAW fixture was not created"
    raw_output = output / "synthetic-raw.jpg"
    result = subprocess.run([str(binary), "render", str(sources[0]), "-o", str(raw_output), "--size", "640"],
                            capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    assert result.returncode == 0, result.stderr
    with Image.open(raw_output) as image:
        image.load()
        report["synthetic_raw"] = {"source": str(sources[0]), "size": image.size,
                                    "note": "procedural DNG; camera-specific RAW compatibility untested"}
    target = output / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
