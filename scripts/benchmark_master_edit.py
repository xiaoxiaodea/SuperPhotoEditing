"""Render a recorded, human-directed study recipe through the real Lightroom engine.

This is a capability experiment, not an automatic-planner result. The input is
copied, parameters/readback and an independent pixel assessment are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from superphoto.config import Settings
from superphoto.lightroom import (apply_professional_recipe, close_lightroom_client,
    export_lightroom_baseline, get_lightroom_client, wait_for_lightroom_connection)
from superphoto.lightroom_qa import assess_lightroom_render
from superphoto.models import Recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.recipe.read_text())
    recipe = Recipe.model_validate(payload["recipe"])
    root = (args.out / uuid4().hex[:8]).resolve()
    root.mkdir(parents=True)
    original = root / ("study-copy" + args.photo.suffix.lower())
    shutil.copy2(args.photo, original)
    settings = Settings()
    print(json.dumps({"root": str(root), "mode": "human-directed-capability-study"}), flush=True)
    try:
        client = get_lightroom_client()
        if not wait_for_lightroom_connection(client, timeout=45).get("connected"):
            raise RuntimeError("Lightroom did not connect")
        baseline = export_lightroom_baseline(client, original=original, output_dir=root / "baseline", export_edge=2400)
        rendered = apply_professional_recipe(client, original=original, recipe=recipe,
            output_dir=root / "edited", export_edge=2400)
        report = assess_lightroom_render(Path(baseline["export"]), Path(rendered["export"]),
            slot="B", creative_palette=True).to_dict()
        audit = {"mode": "human-directed-capability-study", "planner_automatic": False,
            "study": payload, "input_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "baseline": baseline, "rendered": rendered, "pixel_assessment": report,
            "note": "Recorded study output; pixel warnings are reported, not silently treated as an automatic delivery."}
        (root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
        print(json.dumps({"root": str(root), "export": str(rendered["export"]),
            "pixel_accepted": report["accepted"], "issues": report["issues"]}, ensure_ascii=False), flush=True)
    finally:
        close_lightroom_client()


if __name__ == "__main__":
    main()
