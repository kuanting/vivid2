"""Local / self-hosted VLMs behind an OpenAI-compatible endpoint.

Works with vLLM (``vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001``; the
default port 8000 collides with the UE 5.8 editor's MCP server), Ollama
(``http://localhost:11434/v1``), LM Studio, etc. These servers differ in how
well they enforce JSON schemas, so the reply is parsed defensively.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vivid2.stack.base import UsageStats
from vivid2.vlm.base import VLMError, VLMProvider, encode_jpeg


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise VLMError(f"no JSON object in reply: {text[:200]!r}")
        return json.loads(m.group(0))


class OpenAICompatVLM(VLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        max_tokens: int = 1024,
        max_image_side: int = 1024,
        enforce_schema: bool = True,
        timeout_s: float = 120.0,
    ):
        from openai import OpenAI  # optional dependency: pip install vivid2[local-vlm]

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.max_tokens = max_tokens
        self.max_image_side = max_image_side
        self.enforce_schema = enforce_schema

    def query(
        self,
        images: List[np.ndarray],
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], UsageStats]:
        parts: List[Dict[str, Any]] = []
        for img in images:
            data, _ = encode_jpeg(img, self.max_image_side)
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        parts.append({"type": "text", "text": prompt + "\n\nReply with a single JSON object matching this schema:\n" + json.dumps(schema)})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": parts}]

        kwargs: Dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=0.0)
        if self.enforce_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "assessment", "schema": schema}}
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # server-specific error types vary widely
            raise VLMError(f"{type(e).__name__}: {e}") from e
        latency = time.perf_counter() - t0
        u = resp.usage
        usage = UsageStats(
            latency_s=latency,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            model=self.model,
        )
        return _extract_json(resp.choices[0].message.content or ""), usage
