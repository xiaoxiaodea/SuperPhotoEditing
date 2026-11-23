"""Vision providers offered in the client.

A preset fixes the endpoint and protocol, so the user only picks a provider and
pastes a key. The model is chosen automatically: the provider's model list is
ranked with these preferences and the top few are tried with a tiny test image
until one reads it. The patterns are hints, not truth. A provider can rename
models, and that only means a later candidate (or the user's own choice under
advanced settings) is used.
"""
from __future__ import annotations

import re
from typing import Any

# Name parts of models that cannot answer an image prompt with text.
_EXCLUDE = re.compile(r"embed|tts|whisper|audio|transcri|dall-?e|gpt-image|imagen|image-gen|-image\b|imagine|"
                      r"moderation|realtime|rerank|search|speech|guard|text-only", re.I)
# Generic hints for image-capable chat models, used after a preset's own preferences.
VISION_HINT = (r"vl\b|-vl|vl-|vision|omni|gpt-4o|gpt-4\.1|gpt-5|gemini|claude|grok-4|pixtral|llava|gemma-?3|"
               r"minicpm-v|qvq|glm-4\.\d+v|glm-4v|step-1v|doubao-seed")

PRESETS: dict[str, dict[str, Any]] = {
    "openai": {"name": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
               "protocol": "responses", "prefer": [r"^gpt-5(\.\d+)?$", r"^gpt-4\.1$", r"^gpt-4o$",
                                                   r"^gpt-5(\.\d+)?-mini$"], "fallback": ["gpt-4.1", "gpt-4o"]},
    "xai": {"name": "xAI Grok", "kind": "openai", "base_url": "https://api.x.ai/v1", "protocol": "responses",
            "prefer": [r"^grok-4(\.\d+)?$", r"^grok-4", r"^grok-\d.*vision"], "fallback": ["grok-4"]},
    "gemini": {"name": "Google Gemini", "kind": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "protocol": "chat/completions",
               "prefer": [r"gemini-[\d.]+-pro$", r"gemini-[\d.]+-flash$", r"gemini-[\d.]+-pro", r"gemini-[\d.]+-flash"],
               "fallback": ["gemini-2.5-pro", "gemini-2.5-flash"]},
    "anthropic": {"name": "Anthropic Claude", "kind": "openai", "base_url": "https://api.anthropic.com/v1",
                  "protocol": "chat/completions", "prefer": [r"^claude-sonnet-\d", r"^claude-opus-\d", r"^claude"],
                  "fallback": ["claude-sonnet-5-5", "claude-opus-5-5"]},
    "deepseek": {"name": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com/v1",
                 "protocol": "chat/completions", "prefer": [r"vl", r"vision", r"^deepseek-chat$"],
                 "fallback": ["deepseek-chat"]},
    "dashscope": {"name": "阿里云百炼 · 通义千问", "kind": "openai",
                  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "protocol": "chat/completions",
                  "prefer": [r"^qwen3-vl-plus", r"^qwen-vl-max", r"^qwen3?-vl", r"^qvq"],
                  "fallback": ["qwen-vl-max", "qwen-vl-plus"]},
    "ark": {"name": "火山方舟 · 豆包", "kind": "openai", "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "protocol": "chat/completions", "prefer": [r"doubao-seed", r"doubao.*vision"], "fallback": []},
    "zhipu": {"name": "智谱 GLM", "kind": "openai", "base_url": "https://open.bigmodel.cn/api/paas/v4",
              "protocol": "chat/completions", "prefer": [r"^glm-4\.\d+v", r"^glm-4v"],
              "fallback": ["glm-4.5v", "glm-4v-plus"]},
    "moonshot": {"name": "月之暗面 Kimi", "kind": "openai", "base_url": "https://api.moonshot.cn/v1",
                 "protocol": "chat/completions", "prefer": [r"vision", r"^kimi.*(vl|vision)"],
                 "fallback": ["moonshot-v1-8k-vision-preview"]},
    "siliconflow": {"name": "硅基流动", "kind": "openai", "base_url": "https://api.siliconflow.cn/v1",
                    "protocol": "chat/completions",
                    "prefer": [r"qwen3-vl", r"qwen2\.5-vl-72b", r"qwen.*vl", r"deepseek-vl", r"glm-4\.\d+v"],
                    "fallback": ["Qwen/Qwen2.5-VL-72B-Instruct"]},
    "hunyuan": {"name": "腾讯混元", "kind": "openai", "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
                "protocol": "chat/completions", "prefer": [r"vision"], "fallback": ["hunyuan-vision"]},
    "stepfun": {"name": "阶跃星辰", "kind": "openai", "base_url": "https://api.stepfun.com/v1",
                "protocol": "chat/completions", "prefer": [r"^step-\d+o", r"^step-1v"], "fallback": ["step-1v-8k"]},
    "qianfan": {"name": "百度千帆", "kind": "openai", "base_url": "https://qianfan.baidubce.com/v2",
                "protocol": "chat/completions", "prefer": [r"vl", r"vision"], "fallback": []},
    "mistral": {"name": "Mistral", "kind": "openai", "base_url": "https://api.mistral.ai/v1",
                "protocol": "chat/completions", "prefer": [r"pixtral-large", r"^mistral-medium", r"pixtral"],
                "fallback": ["pixtral-large-latest"]},
    "openrouter": {"name": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "protocol": "chat/completions",
                   "prefer": [r"^google/gemini-[\d.]+-pro", r"^openai/gpt-5", r"^anthropic/claude-sonnet", r"^qwen/.*vl"],
                   "fallback": []},
    "ollama": {"name": "本机或局域网 Ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434", "protocol": None,
               "needs_key": False,
               "prefer": [r"qwen3-vl", r"qwen2\.5-?vl", r"gemma3", r"llama3\.2-vision", r"minicpm-v", r"llava"],
               "fallback": []},
    "local-openai": {"name": "本机 OpenAI 兼容服务（LM Studio / vLLM）", "kind": "openai",
                     "base_url": "http://127.0.0.1:1234/v1", "protocol": "chat/completions", "needs_key": False,
                     "prefer": [], "fallback": []},
    "custom": {"name": "其他 OpenAI 兼容服务", "kind": "openai", "base_url": "", "protocol": "chat/completions",
               "custom_url": True, "prefer": [], "fallback": []},
}


def public_presets() -> list[dict[str, Any]]:
    """What the client needs to draw the form. No model preferences."""
    return [{"id": key, "name": p["name"], "kind": p["kind"], "base_url": p["base_url"],
             "protocol": p["protocol"], "needs_key": p.get("needs_key", True),
             "custom_url": p.get("custom_url", False)} for key, p in PRESETS.items()]


def _sort_key(name: str) -> tuple:
    snapshot = bool(re.search(r"\d{6,8}|\d{4}-\d{2}-\d{2}|preview|exp\b|-\d{4}$", name, re.I))
    parts = tuple((0, -int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", name) if t)
    return (snapshot, parts)


def rank_models(names: list[str], prefer: list[str]) -> list[str]:
    """Preferred patterns first, then generic vision hints; newer, stable names first in each group."""
    pool = [n for n in dict.fromkeys(names) if isinstance(n, str) and n.strip() and not _EXCLUDE.search(n)]
    ranked: list[str] = []
    for pattern in [*prefer, VISION_HINT]:
        rx = re.compile(pattern, re.I)
        ranked.extend(sorted((n for n in pool if rx.search(n) and n not in ranked), key=_sort_key))
    return ranked
