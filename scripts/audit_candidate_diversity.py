"""Read saved Lightroom exports and decompose candidate differences.

No Lightroom connection, model inference or job mutation is involved. Values
describe pixel differences, not beauty or perceptual percentages. Normalized
mask locations are diagnostic observations, not semantic ground truth.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from superphoto.lightroom_qa import _content_masks, _read_pixels, _rgb_to_lab


def audit_job(root: Path) -> dict:
    baseline = _read_pixels(root / "lightroom-exports/baseline/baseline.jpg")
    size = (baseline.rgb.shape[1], baseline.rgb.shape[0])
    audits = {slot: json.loads((root / "lightroom-audits" / f"variant_{slot}.json").read_text())
              for slot in "ABC" if (root / "lightroom-audits" / f"variant_{slot}.json").is_file()}
    delivered = {slot: audit for slot, audit in audits.items() if audit.get("accepted_export")}
    images = {slot: _rgb_to_lab(_read_pixels(root / f"variant_{slot}.jpg", size=size).rgb)
              for slot in delivered}
    pairs = {}
    for first, second in combinations(images, 2):
        a, b = images[first], images[second]
        delta = a - b
        lightness_energy = np.mean(delta[..., 0] ** 2)
        total_energy = np.mean(np.sum(delta ** 2, axis=-1))
        # Measure what remains after fitting one global affine L* mapping.
        design = np.column_stack((a[..., 0].ravel(), np.ones(a.shape[0] * a.shape[1])))
        target = b[..., 0].ravel()
        fit = np.linalg.lstsq(design, target, rcond=None)[0]
        pairs[first + second] = {
            "mean_delta_e76": float(np.mean(np.linalg.norm(delta, axis=-1))),
            "mean_abs_delta_L": float(np.mean(np.abs(delta[..., 0]))),
            "mean_delta_ab": float(np.mean(np.linalg.norm(delta[..., 1:], axis=-1))),
            "lightness_squared_difference_share": float(lightness_energy / max(total_energy, 1e-12)),
            "mean_abs_L_residual_after_global_affine_fit": float(np.mean(np.abs(target - design @ fit))),
        }
    candidates = {}
    for slot, audit in delivered.items():
        candidates[slot] = {
            "sha256": sha256((root / f"variant_{slot}.jpg").read_bytes()).hexdigest(),
            "accepted_recipe": audit["accepted_recipe"],
            "candidate_selection": audit.get("candidate_selection"),
            "rolled_back_stages": [
                {"round": attempt["round"], "stage": stage["stage"],
                 "notes": (stage.get("visual_review") or {}).get("notes", [])}
                for attempt in audit["rounds"] for stage in attempt.get("professional_stages", [])
                if stage.get("rolled_back")
            ],
        }
    masks = _content_masks(baseline)
    yy = np.linspace(0, 1, baseline.rgb.shape[0])[:, None]
    mask_locations = {}
    for name in ("sky", "skin"):
        mask = masks[name]
        count = max(int(mask.sum()), 1)
        mask_locations[name] = {
            "image_fraction": float(mask.mean()),
            "fraction_below_y025": float((mask & (yy > .25)).sum() / count),
            "fraction_in_bottom_20pct": float((mask & (yy > .8)).sum() / count),
        }
    return {"job": root.name, "root": str(root), "pairs": pairs,
            "candidates": candidates,
            "withheld": {slot: audit.get("reason") for slot, audit in audits.items() if slot not in delivered},
            "baseline_heuristic_mask_locations": mask_locations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=UUID, nargs="+")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jobs = [audit_job(args.data.resolve() / "jobs" / str(job)) for job in args.jobs]
    result = {
        "method": "sRGB, max edge 960, aligned LANCZOS resize, D65 Lab, mean Euclidean delta E 76",
        "limitation": "L* energy share is a numerical decomposition, not a share of perceived difference. "
                      "Similarity and quality need separate visual assessment; heuristic masks are not semantic labels.",
        "jobs": jobs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for job in jobs:
        print(job["job"], json.dumps(job["pairs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
