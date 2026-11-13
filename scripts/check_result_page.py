"""Exercise the result UI in macOS WebKit with current and pre-upgrade HTML.

Reads one completed job through the running app. The fixture only forwards GET;
photo processing and feedback writes are never called. Uses an ephemeral port
so its browser state is separate from the user's application.
"""
import argparse
import json
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import webview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=UUID)
    parser.add_argument("--study-gallery", type=Path)
    parser.add_argument("--feedback", action="store_true", help="Exercise votes and refinement UI against isolated in-memory fixtures")
    args = parser.parse_args()
    if args.study_gallery:
        check_study_gallery(args.study_gallery)
        return
    if not args.job:
        parser.error("--job or --study-gallery is required")
    with httpx.Client(base_url="http://127.0.0.1:8765", trust_env=False, timeout=15) as client:
        response = client.get(f"/api/jobs/{args.job}")
        response.raise_for_status()
        job = response.json()
        assert job["status"] == "ready" and job["variants"]
        payload = json.dumps(job, ensure_ascii=False)
        requests = []
        fixture_job = copy.deepcopy(job)
        fixture_job.update(ratings={}, refinements=[])
        complete_refinements = threading.Event()

        class Fixture(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                requests.append(("POST", self.path))
                if args.feedback and self.path == f"/api/jobs/{args.job}/refinements":
                    data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    record = {"id": data["request_id"], "slot": data["slot"], "parent_id": data["parent_id"],
                              "job_id": str(args.job), "prompt": data["prompt"], "status": "running", "step": "lightroom",
                              "summary": "整体提亮", "error": None, "url": None}
                    fixture_job["refinements"].append(record)
                    self.json_response(record, 202)
                    return
                self.send_error(405)

            def json_response(self, payload, status=200):
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_PUT(self):
                requests.append(("PUT", self.path))
                prefix = f"/api/jobs/{args.job}/ratings/"
                if not args.feedback or not self.path.startswith(prefix):
                    self.send_error(405)
                    return
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture_job["ratings"][self.path.removeprefix(prefix)] = data["verdict"]
                self.json_response({"ok": True, "ratings": fixture_job["ratings"], "learnable": True, "summary": "界面测试中的偏好"})

            def do_GET(self):
                requests.append(("GET", self.path))
                path = urlsplit(self.path).path
                if args.feedback and path == f"/api/jobs/{args.job}":
                    self.json_response(fixture_job)
                    return
                if args.feedback and path.startswith(f"/api/jobs/{args.job}/refinements/"):
                    record = fixture_job["refinements"][-1]
                    if complete_refinements.is_set():
                        record.update(status="ready", step="ready", url=job["variants"][-1]["url"])
                    self.json_response(record)
                    return
                if path == "/api/jobs/latest":
                    body = json.dumps({"id": str(args.job)}).encode()
                    status, content_type = 200, "application/json"
                else:
                    result = client.get(self.path)
                    status = result.status_code
                    body = result.content
                    content_type = result.headers.get("content-type", "application/octet-stream")
                    if path == "/" and "legacy=1" in self.path:
                        body = body.replace(b' id="result-count"', b'')
                        body = body.replace(b'<p class="compare-hint hidden" id="delivery-note"></p>', b'')
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        window = webview.create_window("SuperPhoto · 结果页验证", base + "/?legacy=1", width=1200, height=850)
        errors = []

        def wait_result():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                result = window.evaluate_js("""(() => ({
                    ready: typeof state !== 'undefined' && state.job?.status === 'ready',
                    visible: !document.getElementById('result-stage')?.classList.contains('hidden'),
                    count: document.getElementById('result-count')?.textContent,
                    cards: document.querySelectorAll('.variant').length,
                    images: [...document.querySelectorAll('.variant img')].every(i => i.complete && i.naturalWidth > 0)
                }))()""")
                if result and result.get("ready") and result.get("visible") and result.get("images"):
                    assert result["cards"] == len(job["variants"]), result
                    assert result["count"] == f'{len(job["variants"])} 套后期成片', result
                    return result
                time.sleep(.2)
            raise AssertionError(f"Result did not display: {result}")

        def check():
            try:
                print("legacy HTML + installed script:", wait_result(), flush=True)
                partial = dict(job, variants=[job["variants"][-1]],
                               delivery={"omitted": [{"slot": v["slot"]} for v in job["variants"][:-1]]})
                window.evaluate_js(f"renderResult({json.dumps(partial, ensure_ascii=False)})")
                selected = partial["variants"][0]["slot"]
                absent = next(slot for slot in "ABC" if slot != selected)
                result = window.evaluate_js(f"""(() => {{
                    selectSlot('{absent}');
                    return {{selected:state.selected, count:$('result-count').textContent,
                             note:$('delivery-note').textContent, cards:$('variants').children.length}};
                }})()""")
                assert result["selected"] == selected and result["cards"] == 1, result
                assert result["count"] == "1 套后期成片" and "已撤下" in result["note"], result
                print("partial delivery:", result, flush=True)
                window.evaluate_js(f"renderResult({payload}); fail('simulated view failure')")
                result = window.evaluate_js("({title:$('error-title').textContent, button:$('error-retry').textContent})")
                assert result["title"] == "成片已完成，页面未能显示", result
                assert result["button"] == "重新打开成片", result
                window.evaluate_js("retryJob()")
                time.sleep(.5)
                print("reopen completed job:", wait_result(), flush=True)
                window.load_url(base + "/")
                time.sleep(.5)
                print("current HTML + remembered result:", wait_result(), flush=True)
                if args.feedback:
                    check_feedback()
                else:
                    for record in job.get("refinements", []):
                        if record["status"] != "ready":
                            continue
                        window.evaluate_js(f"state.versions['{record['slot']}'] = '{record['id']}'; selectSlot('{record['slot']}')")
                        wait_js("$('hero-edit').complete && $('hero-edit').naturalWidth > 0 && $('hero-orig').complete && $('hero-orig').naturalWidth > 0")
                        assert window.evaluate_js("$('hero-before-label').textContent === '微调前'")
                    print("Saved native refinement images displayed with their corresponding before versions.", flush=True)
                window.evaluate_js("reset()")
                window.load_url(base + "/?reset-check=1")
                time.sleep(1)
                assert window.evaluate_js("!$('drop-stage').classList.contains('hidden') && state.jobId === null")
                if not args.feedback:
                    assert not any(method == "POST" for method, _ in requests), requests
                print("PASS: WebKit displayed real exports and restored saved work. All simulated votes/refinements stayed inside the fixture.", flush=True)
            except Exception:
                errors.append(traceback.format_exc())
                print(errors[-1], flush=True)
            finally:
                window.destroy()

        def wait_js(expression):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if window.evaluate_js(expression):
                    return
                time.sleep(.15)
            raise AssertionError(f"WebKit condition did not complete: {expression}")

        def check_feedback():
            assert len(job["variants"]) == 3, "Feedback smoke test expects three delivered choices"
            for slot, verdict in (("A", "like"), ("B", "dislike"), ("C", "like")):
                window.evaluate_js(f"document.querySelector('.variant[data-slot=\"{slot}\"] [data-verdict=\"{verdict}\"]').click()")
                wait_js(f"state.job.ratings?.['{slot}'] === '{verdict}'")
            assert fixture_job["ratings"] == {"A": "like", "B": "dislike", "C": "like"}
            window.evaluate_js("document.querySelector('.variant[data-slot=\"B\"] [data-verdict=\"dislike\"]').click()")
            wait_js("state.job.ratings?.B === null")
            window.load_url(base + "/")
            time.sleep(.3)
            wait_result()
            assert window.evaluate_js("document.querySelectorAll('.rating[aria-pressed=\"true\"]').length") == 2
            window.evaluate_js("""document.querySelector('.variant[data-slot="B"] .variant-refine').click();
                $('notes').value = '整体亮一点，保留现在的暖色';
                $('notes').dispatchEvent(new Event('input', {bubbles:true}));
                window.__space = new KeyboardEvent('keydown', {key:' ', code:'Space', bubbles:true, cancelable:true});
                $('notes').dispatchEvent(window.__space);
                $('notes').dispatchEvent(new KeyboardEvent('keydown', {key:'1', bubbles:true}));""")
            assert window.evaluate_js("state.selected === 'B' && !window.__space.defaultPrevented && !$('refine-submit').disabled")
            window.evaluate_js("$('refine-submit').click(); $('refine-submit').click()")
            wait_js("state.job.refinements?.length === 1")
            assert len(fixture_job["refinements"]) == 1
            assert window.evaluate_js("$('refine-submit').disabled")
            window.load_url(base + "/")
            time.sleep(.3)
            wait_result()
            window.evaluate_js("selectSlot('B')")
            assert window.evaluate_js("$('refine-submit').disabled && $('refine-status').textContent.includes('Lightroom')")
            complete_refinements.set()
            wait_js("selectedVersion()?.status === 'ready'")
            assert window.evaluate_js("$('hero-before-label').textContent === '微调前' && $('version-download').href === $('hero-edit').src")
            window.evaluate_js("$('refine-version').value = ''; $('refine-version').dispatchEvent(new Event('change'))")
            assert window.evaluate_js("selectedVersion() == null && $('hero-before-label').textContent === '原片'")
            print("PASS: independent votes, undo, reload, text entry, duplicate-submit guard, progress recovery, before/after and version switching.", flush=True)

        try:
            webview.start(check, gui="cocoa", private_mode=False)
        finally:
            server.shutdown()
        if errors:
            raise SystemExit(1)


def check_study_gallery(path: Path):
    window = webview.create_window("SuperPhoto · 逐图报告验证", path.resolve().as_uri(), width=1200, height=850)
    errors = []
    def check():
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                ready = window.evaluate_js("typeof data !== 'undefined' && document.querySelectorAll('.card').length === data.items.length")
                if ready:
                    break
                time.sleep(.2)
            assert ready
            result = window.evaluate_js("({images:data.items.length,coverage:data.coverage,rendered:document.querySelectorAll('.card').length})")
            assert result["images"] == 851 and result["rendered"] == 851
            window.evaluate_js("document.getElementById('kind').value='pending'; render()")
            assert window.evaluate_js("document.querySelectorAll('.card').length === data.coverage.pending")
            window.evaluate_js("document.getElementById('kind').value=''; document.getElementById('search').value='Ali Rajabi'; render()")
            assert window.evaluate_js("document.querySelectorAll('.card').length >= 19")
            window.evaluate_js("document.getElementById('search').value='a-nonexistent-photographer'; render()")
            assert window.evaluate_js("document.querySelectorAll('.card').length === 0")
            print("PASS: native WebKit loaded all study cards and filtered by completion and photographer.", flush=True)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception:
            errors.append(traceback.format_exc())
            print(errors[-1], flush=True)
        finally:
            window.destroy()
    webview.start(check, gui="cocoa", private_mode=False)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
