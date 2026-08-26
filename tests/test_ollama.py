from __future__ import annotations

from superphoto.ollama import pick_model


def test_pick_prefers_qwen3_vl_32b() -> None:
    names = ["llama3:latest", "qwen3-vl:8b", "qwen3-vl:32b"]
    assert pick_model("qwen3-vl:32b", names) == "qwen3-vl:32b"


def test_pick_falls_back_to_any_vl() -> None:
    names = ["llama3:latest", "llava:13b"]
    assert pick_model("qwen3-vl:32b", names) == "llava:13b"


def test_pick_none_when_text_only() -> None:
    names = ["llama3:latest", "deepseek-r1:70b"]
    assert pick_model("qwen3-vl:32b", names) is None