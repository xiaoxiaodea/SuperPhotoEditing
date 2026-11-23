"""Open the actual desktop window, verify its loaded UI, then close it."""
import json
from pathlib import Path
import time

import webview
from superphoto import desktop


result = {}
real_start = webview.start


def instrumented_start(func, **kwargs):
    def drive():
        window = webview.windows[0]
        try:
            func()
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                try:
                    loaded = window.evaluate_js("document.querySelector('#pick') !== null")
                    if loaded:
                        result.update(backend=kwargs.get("gui"),
                                      title=window.evaluate_js("document.title"),
                                      url=window.evaluate_js("window.location.href"),
                                      upload_button=bool(loaded))
                        break
                except Exception:
                    pass
                time.sleep(.2)
            if not result:
                result["error"] = "desktop UI did not load"
        except Exception as exc:
            result["error"] = type(exc).__name__
        finally:
            window.destroy()
    real_start(drive, **kwargs)


if __name__ == "__main__":
    webview.start = instrumented_start
    desktop.main()
    output = Path(".task-cache/windows-verification/desktop.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result.get("error") or not result.get("upload_button"):
        raise SystemExit(1)
