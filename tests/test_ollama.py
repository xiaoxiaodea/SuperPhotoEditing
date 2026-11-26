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


def test_compact_local_prompt_and_vision_budget_fit_uniform_4k_ctx() -> None:
    from types import SimpleNamespace
    from superphoto import directed, experience, ollama

    tech = SimpleNamespace(model_dump=lambda: {"source_kind": "jpeg", "width": 1800, "height": 3200, "mean_luma": 0.45})
    full_prompt = directed.plan_prompt(tech, (1800, 3200), library=experience.library(), compact=False)
    compact_prompt = directed.plan_prompt(tech, (1800, 3200), library=experience.library(), compact=True)
    # Ultra-compact local prompt cuts size >9x while keeping all rules, looks, profiles & delta schema
    assert len(compact_prompt) < 4200 and len(compact_prompt) < len(full_prompt) * 0.15
    assert "TONE TARGETS" in compact_prompt and "MASKS" in compact_prompt and "LOOKS:" in compact_prompt
    assert "PROFILES" in compact_prompt and '"v":{' in compact_prompt

    # Single border-ruled 896px preview + compact prompt + 1280 output fits cleanly inside 4096 num_ctx!
    vis_plan = ollama._estimate_vision_tokens(["a"], [{"width": 553, "height": 944}])
    assert 600 <= vis_plan <= 1000
    assert int(len(compact_prompt) * 0.32) + 200 + vis_plan + 1280 <= 3900

    # Single 1260x1176 contact sheet for damage_check stays tiny
    vis_dmg = ollama._estimate_vision_tokens(["x"], [{"width": 1260, "height": 1176}])
    assert vis_dmg < 2200