"""Commands for the Persona assistant.

Talks only to a SuperPhoto server that is already running. A second engine would
fight the same Lightroom lock and database. Finished JPEGs are downloaded over
HTTP and written under ~/Library/Application Support/SuperPhoto/agent, so the
assistant can show them without knowing the server's data directory.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import get_settings

StepCallback = Callable[[str], None]
_TERMINAL = {"ready", "error"}
_UNSAFE = set('\\/:*?"<>|\n\r\t')


class AgentError(Exception):
    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.payload = {"ok": False, "error": message, **extra}


def output_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "SuperPhoto" / "agent"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SuperPhoto" / "agent"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "superphoto" / "agent"


def file_stem(slot: str, title: str) -> str:
    cleaned = "".join(" " if char in _UNSAFE else char for char in title).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        cleaned = slot
    if len(cleaned) > 40:
        cleaned = cleaned[:40].rstrip()
    return f"{slot}-{cleaned}"


class SuperPhotoClient:
    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        out_dir: Path | None = None,
        poll_seconds: float = 2.0,
        access_token: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.out_dir = out_dir or output_root()
        self.poll_seconds = poll_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            trust_env=False,
            headers={"Authorization": f"Bearer {access_token}"} if access_token else {},
            timeout=httpx.Timeout(120.0, connect=3.0),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def status(self) -> dict[str, Any]:
        health = self._get("/api/health")
        lightroom = health.get("lightroom") if isinstance(health.get("lightroom"), dict) else {}
        payload: dict[str, Any] = {
            "ok": True,
            "running": True,
            "critic": health.get("critic"),
            "critic_model": health.get("critic_model"),
            "lightroom": lightroom,
        }
        if not lightroom.get("connected"):
            payload["hint"] = "LightCraft 还没就绪。修图时会再试一次；若失败，运行 uv run superphoto install-lightcraft。"
        return payload

    def edit(self, photo: Path, timeout: float, on_step: StepCallback | None = None) -> dict[str, Any]:
        if not photo.is_file():
            raise AgentError(f"找不到这张照片：{photo}")
        created = self._send(
            "POST",
            "/api/jobs",
            files={"file": (photo.name, photo.read_bytes())},
        )
        job_id = str(created.get("id") or "")
        if not job_id:
            raise AgentError("SuperPhoto 没有返回任务编号")
        try:
            job = self._wait(lambda: self._get(f"/api/jobs/{job_id}"), timeout, on_step)
        except AgentError as exc:
            exc.payload.setdefault("command", "edit")
            exc.payload.setdefault("job_id", job_id)
            raise
        return self._job_payload(job, command="edit")

    def job(self, job_id: str, on_step: StepCallback | None = None) -> dict[str, Any]:
        job = self._get(f"/api/jobs/{job_id}")
        if on_step and job.get("step"):
            on_step(str(job["step"]))
        return self._job_payload(job, command="job")

    def rate(self, job_id: str, slot: str, verdict: str) -> dict[str, Any]:
        body = {"verdict": None if verdict == "clear" else verdict}
        result = self._send("PUT", f"/api/jobs/{job_id}/ratings/{slot}", json=body)
        return {
            "ok": True,
            "command": "rate",
            "job_id": job_id,
            "slot": slot,
            "verdict": body["verdict"],
            "ratings": result.get("ratings") or {},
            "summary": result.get("summary") or "",
            "learnable": bool(result.get("learnable")),
            "effect": "这次评价只影响以后相似的照片，不会改当前成片。",
        }

    def refine(
        self,
        job_id: str,
        slot: str,
        prompt: str,
        parent_id: str | None,
        timeout: float,
        on_step: StepCallback | None = None,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        body: dict[str, Any] = {"slot": slot, "prompt": prompt, "request_id": request_id}
        if parent_id:
            body["parent_id"] = parent_id
        created = self._send("POST", f"/api/jobs/{job_id}/refinements", json=body)
        refinement_id = str(created.get("id") or request_id)
        try:
            record = self._wait(
                lambda: self._get(f"/api/jobs/{job_id}/refinements/{refinement_id}"),
                timeout,
                on_step,
            )
        except AgentError as exc:
            exc.payload.setdefault("command", "refine")
            exc.payload.setdefault("job_id", job_id)
            exc.payload.setdefault("refinement_id", refinement_id)
            raise
        return self._refinement_payload(job_id, record)

    def _job_payload(self, job: dict[str, Any], command: str) -> dict[str, Any]:
        job_id = str(job.get("id") or "")
        status = str(job.get("status") or "")
        if job.get("_timed_out"):
            raise AgentError(
                "这张照片还在处理。稍后再查同一个任务。",
                command=command,
                job_id=job_id,
                status=status,
                step=job.get("step"),
            )
        if status == "error":
            raise AgentError(
                str(job.get("error") or "处理失败"),
                command=command,
                job_id=job_id,
                status="error",
            )
        if status != "ready":
            return {
                "ok": True,
                "command": command,
                "job_id": job_id,
                "status": status,
                "step": job.get("step"),
                "variants": [],
            }
        variants = [self._save_variant(job_id, variant) for variant in job.get("variants") or []]
        refinements = [
            self._save_refinement(job_id, record)
            for record in job.get("refinements") or []
            if record.get("status") == "ready" and record.get("url")
        ]
        payload: dict[str, Any] = {
            "ok": True,
            "command": command,
            "job_id": job_id,
            "status": "ready",
            "variants": variants,
            "ratings": job.get("ratings") or {},
        }
        if refinements:
            payload["refinements"] = refinements
        note = _delivery_note(job.get("delivery"))
        if note:
            payload["note"] = note
        if not variants:
            payload["note"] = "处理完成了，但没有可交付的成片。"
        return payload

    def _refinement_payload(self, job_id: str, record: dict[str, Any]) -> dict[str, Any]:
        refinement_id = str(record.get("id") or "")
        if record.get("_timed_out"):
            raise AgentError(
                "微调还在进行。稍后再查这个任务。",
                command="refine",
                job_id=job_id,
                slot=record.get("slot"),
                refinement_id=refinement_id,
                status=record.get("status"),
                step=record.get("step"),
            )
        if record.get("status") == "error":
            raise AgentError(
                str(record.get("error") or "微调失败，原来的成片还在。"),
                command="refine",
                job_id=job_id,
                slot=record.get("slot"),
                refinement_id=refinement_id,
                status="error",
            )
        saved = self._save_refinement(job_id, record)
        return {
            "ok": True,
            "command": "refine",
            "job_id": job_id,
            "slot": record.get("slot"),
            "refinement_id": refinement_id,
            "parent_id": record.get("parent_id"),
            "prompt": record.get("prompt") or "",
            "summary": record.get("summary") or "",
            "path": saved["path"],
            "effect": "这是在选中的成片上重新渲染的一版，原来的成片还在。",
        }

    def _save_variant(self, job_id: str, variant: dict[str, Any]) -> dict[str, str]:
        slot = str(variant.get("slot") or "")
        title = str(variant.get("title") or slot)
        path = self.out_dir / job_id / f"{file_stem(slot, title)}.jpg"
        self._download(str(variant.get("url") or ""), path)
        why = str(variant.get("why") or "").strip()
        if len(why) > 500:
            why = why[:500].rstrip() + "…"
        return {"slot": slot, "title": title, "why": why, "path": str(path)}

    def _save_refinement(self, job_id: str, record: dict[str, Any]) -> dict[str, str]:
        slot = str(record.get("slot") or "")
        prompt = str(record.get("prompt") or "微调")
        suffix = str(record.get("id") or "")[:8]
        path = self.out_dir / job_id / f"{file_stem(slot, '微调 ' + prompt)}-{suffix}.jpg"
        self._download(str(record.get("url") or ""), path)
        return {"id": str(record.get("id") or ""), "slot": slot, "prompt": prompt, "path": str(path)}

    def _wait(self, fetch: Callable[[], dict[str, Any]], timeout: float, on_step: StepCallback | None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_step: str | None = None
        while True:
            payload = fetch()
            step = str(payload.get("step") or "")
            if step and step != last_step:
                last_step = step
                if on_step:
                    on_step(step)
            if payload.get("status") in _TERMINAL:
                return payload
            if time.monotonic() >= deadline:
                return {**payload, "_timed_out": True}
            time.sleep(self.poll_seconds)

    def _download(self, url_path: str, dest: Path) -> None:
        if not url_path.startswith("/"):
            raise AgentError("成片地址无效")
        response = self._request("GET", url_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(dest)

    def _get(self, path: str) -> dict[str, Any]:
        body = self._request("GET", path).json()
        if not isinstance(body, dict):
            raise AgentError("SuperPhoto 返回了无法识别的内容")
        return body

    def _send(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        body = self._request(method, path, **kwargs).json()
        if not isinstance(body, dict):
            raise AgentError("SuperPhoto 返回了无法识别的内容")
        return body

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise AgentError(
                "SuperPhoto 没有在运行。先打开 SuperPhoto，再让我修这张图。",
            ) from exc
        if response.status_code >= 400:
            raise AgentError(_error_detail(response))
        if _looks_like_html(response):
            raise AgentError(_port_taken_message())
        return response


def default_client() -> SuperPhotoClient:
    settings = get_settings()
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    return SuperPhotoClient(f"http://{host}:{settings.port}", access_token=settings.access_token)


def run_agent(args: Any) -> None:
    client = default_client()
    try:
        payload = _dispatch(client, args, _print_step)
    except AgentError as exc:
        payload = exc.payload
    finally:
        client.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload.get("ok"):
        raise SystemExit(1)


def _dispatch(client: SuperPhotoClient, args: Any, on_step: StepCallback) -> dict[str, Any]:
    command = args.agent_cmd
    if command == "status":
        return client.status()
    if command == "edit":
        return client.edit(args.photo, args.timeout, on_step)
    if command == "job":
        return client.job(args.job_id, on_step)
    if command == "rate":
        return client.rate(args.job_id, args.slot, args.verdict)
    if command == "refine":
        return client.refine(args.job_id, args.slot, args.prompt, args.parent, args.timeout, on_step)
    raise AgentError(f"未知命令：{command}")


def _print_step(step: str) -> None:
    print(f"step: {step}", file=sys.stderr, flush=True)


def _delivery_note(delivery: Any) -> str:
    if not isinstance(delivery, dict):
        return ""
    omitted = delivery.get("omitted")
    if not isinstance(omitted, list) or not omitted:
        return ""
    slots = "、".join(str(item.get("slot")) for item in omitted if isinstance(item, dict) and item.get("slot"))
    if not slots:
        return ""
    return f"方案 {slots} 未通过检查，没有交付。"


def _error_detail(response: httpx.Response) -> str:
    if _looks_like_html(response):
        return _port_taken_message()
    try:
        body = response.json()
    except json.JSONDecodeError:
        body = None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    text = response.text.strip()
    if text:
        return text.splitlines()[0][:300]
    return f"SuperPhoto 返回了错误（{response.status_code}）"


def _looks_like_html(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        return True
    return response.text.lstrip().startswith("<!")


def _port_taken_message() -> str:
    return "8765 端口上不是 SuperPhoto。先关掉占用这个端口的程序，再打开 SuperPhoto。"
