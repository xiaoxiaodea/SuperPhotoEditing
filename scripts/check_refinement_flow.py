"""Run one real refinement through the installed app, without recording taste votes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from uuid import UUID, uuid4

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=UUID, required=True)
    parser.add_argument("--slot", choices=list("ABC"), default="B")
    parser.add_argument("--prompt", default="整体亮一点，保留现在的暖色")
    parser.add_argument("--parent", type=UUID)
    args = parser.parse_args()
    with httpx.Client(base_url="http://127.0.0.1:8765", trust_env=False, timeout=30) as client:
        start_deadline = time.monotonic() + 30
        while True:
            try:
                client.get("/", timeout=2).raise_for_status()
                break
            except httpx.ConnectError:
                if time.monotonic() >= start_deadline:
                    raise
                time.sleep(.3)
        response = client.get(f"/api/jobs/{args.job}")
        response.raise_for_status()
        job = response.json()
        assert job["status"] == "ready"
        images = {v["url"]: hashlib.sha256(client.get(v["url"]).content).hexdigest() for v in job["variants"]}
        ratings = job.get("ratings", {})
        response = client.post(f"/api/jobs/{args.job}/refinements", json={
            "slot": args.slot, "prompt": args.prompt, "request_id": str(uuid4()),
            "parent_id": str(args.parent) if args.parent else None})
        response.raise_for_status()
        record = response.json()
        rid = record["id"]
        print(json.dumps({"refinement_id": rid, "job_id": str(args.job), "prompt": args.prompt}, ensure_ascii=False), flush=True)
        previous = None
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            response = client.get(f"/api/jobs/{args.job}/refinements/{rid}")
            response.raise_for_status()
            record = response.json()
            progress = (record["status"], record["step"])
            if progress != previous:
                print(json.dumps(record, ensure_ascii=False), flush=True)
                previous = progress
            if record["status"] in {"ready", "error"}:
                break
            time.sleep(2)
        assert record["status"] == "ready", record
        final = client.get(f"/api/jobs/{args.job}").json()
        assert final["status"] == "ready" and final.get("ratings", {}) == ratings
        assert all(hashlib.sha256(client.get(url).content).hexdigest() == value for url, value in images.items())
        rendered = client.get(record["url"])
        rendered.raise_for_status()
        data = Path.home() / "Library/Application Support/SuperPhoto/data"
        folder = data / "jobs" / str(args.job) / "refinements" / rid
        audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        assert rendered.content == Path(audit["lightroom"]["export"]).read_bytes()
        from superphoto.lightroom_qa import render_distance
        from superphoto.refinement import change_summary
        from superphoto.models import Recipe
        before = Recipe.model_validate(audit["base_recipe"])
        after = Recipe.model_validate(audit["lightroom"]["effective_recipe"])
        print(json.dumps({"verified_native_export": True, "original_variants_unchanged": True,
            "ratings_unchanged": True, "interpreter": audit["interpreter"], "changes": change_summary(before, after),
            "base_exposure": before.exposure_ev, "new_exposure": after.exposure_ev,
            "color_grading_preserved": before.color_grading == after.color_grading,
            "pixel_distance": render_distance(Path(audit["base_image"]), folder / "result.jpg"),
            "quality": audit["quality"]["score"], "audit": str(folder / "audit.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
