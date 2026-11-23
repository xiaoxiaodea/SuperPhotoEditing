"""Drive a running SuperPhoto over HTTP and optionally in an installed browser.

Uses repository sample images only. No server or external renderer is started.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import time
import uuid

import httpx
from PIL import Image


def wait_job(client, job_id, seconds=180):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["step"] != last:
            print(f"{job_id}: {job['step']}", flush=True)
            last = job["step"]
        if job["status"] == "error":
            raise RuntimeError(job.get("error") or "processing failed")
        if job["status"] == "ready":
            return job
        time.sleep(.25)
    raise TimeoutError(job_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8876")
    parser.add_argument("--out", type=Path, default=Path(".task-cache/windows-verification"))
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--existing-job", action="append", default=[],
                        help="reuse an existing sample job by sample order instead of uploading again")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SUPERPHOTO_ACCESS_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    report = {"base": args.base, "jobs": [], "browser": None}
    with httpx.Client(base_url=args.base, headers=headers, timeout=120, trust_env=False) as client:
        health_response = client.get("/api/health")
        health_response.raise_for_status()
        report["health"] = health_response.json()
        for index, name in enumerate(("landscape.jpg", "portrait.jpg")):
            if index < len(args.existing_job):
                job_id = args.existing_job[index]
            else:
                with (root / "samples" / name).open("rb") as source:
                    response = client.post("/api/jobs", files={"file": (name, source, "image/jpeg")})
                response.raise_for_status()
                job_id = response.json()["id"]
            job = wait_job(client, job_id, args.timeout)
            variants = job["variants"]
            assert variants, "no delivered variants"
            item = {"id": job_id, "sample": name, "variants": []}
            for variant in variants:
                response = client.get(variant["url"])
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image.load()
                assert image.width > 0 and image.height > 0
                assert image.format == "JPEG"
                target = args.out / f"{Path(name).stem}-{variant['slot']}.jpg"
                target.write_bytes(response.content)
                item["variants"].append({"slot": variant["slot"], "width": image.width,
                                         "height": image.height, "bytes": len(response.content),
                                         "path": str(target)})
            slot = variants[0]["slot"]
            response = client.put(f"/api/jobs/{job_id}/ratings/{slot}", json={"verdict": "like"})
            response.raise_for_status()
            saved = client.get(f"/api/jobs/{job_id}").json()
            item["rating_saved"] = saved.get("ratings", {}).get(slot) == "like"
            assert item["rating_saved"], "rating did not persist"
            if health_response.json()["lightroom"]["enabled"]:
                response = client.post(f"/api/jobs/{job_id}/refinements", json={
                    "slot": slot, "prompt": "整体亮一点，保留现在的暖色", "request_id": str(uuid.uuid4())})
                response.raise_for_status()
                refinement_id = response.json()["id"]
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    refinement = client.get(f"/api/jobs/{job_id}/refinements/{refinement_id}").json()
                    if refinement["status"] in {"ready", "error"}:
                        break
                    time.sleep(.25)
                assert refinement["status"] == "ready", refinement
                item["refinement"] = {"id": refinement_id, "status": refinement["status"]}
            else:
                item["refinement"] = "not_available_in_basic_renderer"
            report["jobs"].append(item)
        if args.browser:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=str(args.browser), headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(args.base, wait_until="networkidle")
                if token:
                    page.locator("#auth-token").fill(token)
                    page.locator("#auth-form button").click()
                    page.locator("#auth-stage").wait_for(state="hidden")
                page.locator("#file").set_input_files(str(root / "samples" / "landscape.jpg"))
                page.locator("#result-stage").wait_for(state="visible", timeout=args.timeout * 1000)
                assert page.locator("#variants img").count() > 0
                page.screenshot(path=str(args.out / "windows-browser.png"), full_page=True)
                report["browser"] = {"page_errors": errors, "screenshot": str(args.out / "windows-browser.png")}
                assert not errors, errors
                browser.close()
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
