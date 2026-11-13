"""Validate recovery through the running app's single Lightroom connection.

Optionally reload the installed plugin with a normal Lightroom restart, then
retry a saved failed photo through the same endpoint as the UI.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from uuid import UUID

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--retry-job", type=UUID)
    parser.add_argument("--check-preference-reads", action="store_true", help="Check concurrent read-only preference/status polling")
    args = parser.parse_args()
    if args.restart:
        from verify_native_masks import restart_lightroom
        restart_lightroom()
    with httpx.Client(base_url="http://127.0.0.1:8765", trust_env=False, timeout=65) as client:
        deadline = time.monotonic() + 30
        while True:
            try:
                client.get("/", timeout=2).raise_for_status()
                break
            except httpx.TransportError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.3)
        response = client.post("/api/lightroom/connect")
        response.raise_for_status()
        status = response.json()
        assert status.get("connected"), status
        print(json.dumps({"connected": True, "tools": len(status.get("tools", []))}), flush=True)
        if args.check_preference_reads:
            def read(index):
                path = "/api/preferences" if index % 2 == 0 else "/api/jobs/latest"
                response = client.get(path)
                response.raise_for_status()
                body = response.json()
                assert ("ratings" if index % 2 == 0 else "id") in body, body
                return response.status_code
            with ThreadPoolExecutor(max_workers=6) as pool:
                outcomes = list(pool.map(read, range(36)))
            print(json.dumps({"concurrent_preference_reads": len(outcomes), "all_ok": True}), flush=True)
        if not args.retry_job:
            return
        response = client.post(f"/api/jobs/{args.retry_job}/retry")
        response.raise_for_status()
        job_id = response.json()["id"]
        print(json.dumps({"job_id": job_id, "retry_of": str(args.retry_job)}), flush=True)
        deadline, previous = time.monotonic() + 1200, None
        while time.monotonic() < deadline:
            response = client.get(f"/api/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            progress = {"job_id": job_id, "status": job["status"], "step": job["step"]}
            if progress != previous:
                print(json.dumps(progress), flush=True)
                previous = progress
            if job["status"] == "error":
                raise RuntimeError(job["error"])
            if job["status"] == "ready":
                assert 1 <= len(job["variants"]) <= 3
                print(json.dumps({"job_id": job_id, "variants": [v["slot"] for v in job["variants"]],
                                  "delivery": job.get("delivery")}, ensure_ascii=False), flush=True)
                return
            time.sleep(3)
        raise TimeoutError(f"Job {job_id} is still running; check the app before retrying")


if __name__ == "__main__":
    main()
