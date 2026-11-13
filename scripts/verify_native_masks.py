"""Exercise real Lightroom masks on a new copy, preserving the input file.

Run with the project Python: scripts/verify_native_masks.py PHOTO [--restart].
Requires Lightroom Classic and the updated bundled plugin to be installed.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import time
from uuid import uuid4

from superphoto.lightroom import (
    apply_professional_recipe, close_lightroom_client, export_lightroom_baseline,
    get_lightroom_client, wait_for_lightroom_connection,
)
from superphoto.models import ColorGradingControls, MaskAdjustment, Recipe


def restart_lightroom():
    print("Reloading Lightroom for the validation connection", flush=True)
    subprocess.run(["osascript", "-e", 'tell application "Adobe Lightroom Classic" to quit'], check=True, timeout=90)
    for _ in range(45):
        status = subprocess.run(["osascript", "-e", 'application "Adobe Lightroom Classic" is running'],
                                capture_output=True, text=True, check=True)
        if status.stdout.strip() == "false":
            break
        time.sleep(1)
    else:
        raise RuntimeError("Lightroom has not quit; no process was force-terminated")
    subprocess.run(["open", "-a", "Adobe Lightroom Classic"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--color-grading", action="store_true", help="also verify native color wheel readback")
    parser.add_argument("--sky-recovery", action="store_true", help="verify RAW sky exposure, contrast, whites and blacks without a subject mask")
    args = parser.parse_args()
    if args.restart:
        restart_lightroom()
    root = (Path(".task-cache/native-validation") / uuid4().hex[:8]).resolve()
    root.mkdir(parents=True)
    copy = root / ("test-copy" + args.photo.suffix.lower())
    shutil.copy2(args.photo, copy)
    print("validation_dir", root, flush=True)
    try:
        client = get_lightroom_client()
        status = wait_for_lightroom_connection(client, timeout=45)
        print("connected", status.get("connected"), flush=True)
        if not status.get("connected"):
            raise RuntimeError(str(status))
        baseline = export_lightroom_baseline(client, original=copy, output_dir=root / "baseline", export_edge=1600)
        print("baseline", baseline["export"], flush=True)
        recipe = Recipe(masks=[MaskAdjustment(target="subject", exposure_ev=0.3, shadows=0.1),
                               MaskAdjustment(target="sky", highlights=-0.18)])
        if args.sky_recovery:
            recipe = Recipe(masks=[MaskAdjustment(target="sky", exposure_ev=-1.58, contrast=.36, whites=.72, blacks=-.05)])
        if args.color_grading:
            recipe.color_grading = ColorGradingControls(shadow_hue=218, shadow_sat=9,
                midtone_hue=37, midtone_sat=3, highlight_hue=43, highlight_sat=7, blending=55, balance=10)
        first = apply_professional_recipe(client, original=copy, output_dir=root / "adjusted", export_edge=1600, recipe=recipe)
        print("masks", json.dumps(first["masks"], ensure_ascii=False), flush=True)
        second = apply_professional_recipe(client, original=copy, output_dir=root / "reapplied", export_edge=1600, recipe=recipe)
        assert [m["mask_id"] for m in first["masks"]] == [m["mask_id"] for m in second["masks"]]
        assert all(not m["created"] for m in second["masks"]), "reapplication must reuse owned masks"
        (root / "result.json").write_text(json.dumps({"baseline": baseline, "first": first, "reapplied": second},
                                                     ensure_ascii=False, indent=2, default=str))
        print("final", second["export"], flush=True)
    finally:
        close_lightroom_client()


if __name__ == "__main__":
    main()
