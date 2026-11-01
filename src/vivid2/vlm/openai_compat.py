"""Local / self-hosted VLMs behind an OpenAI-compatible endpoint.

Works with vLLM (``vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001``; the
default port 8000 collides with the UE 5.8 editor's MCP server), Ollama
(``http://localhost:11434/v1``), LM Studio, etc. These servers differ in how
well they enforce JSON schemas, so the reply is parsed defensively.

The prompt asks for the target point on a 0-1000 grid. Some models (Qwen-VL
in particular) answer in pixels of the image they were sent instead; with
``point_format="pixels"`` the reply is rescaled to 0-1000 so grounding is
scored fairly. The original point is kept as ``point_raw``.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from vivid2.stack.base import UsageStats
from vivid2.vlm.base import ImageInput, VLMError, VLMProvider, encode_jpeg


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse the reply as JSON, falling back to the outermost {...} span."""
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise VLMError(f"no JSON object in reply: {text[:200]!r}")
        try:
            out = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise VLMError(f"invalid JSON in reply: {text[:200]!r}") from e
    if not isinstance(out, dict):
        raise VLMError(f"reply is not a JSON object: {text[:200]!r}")
    return out


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
        point_format: str = "norm1000",
    ):
        from openai import OpenAI  # optional dependency: pip install vivid2[local-vlm]

        if point_format not in ("norm1000", "pixels"):
            raise ValueError("point_format must be 'norm1000' or 'pixels'")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.max_tokens = max_tokens
        self.max_image_side = max_image_side
        self.enforce_schema = enforce_schema
        self.point_format = point_format

    def fingerprint(self) -> Dict[str, Any]:
        return {**super().fingerprint(), "max_tokens": self.max_tokens, "max_image_side": self.max_image_side,
                "enforce_schema": self.enforce_schema, "point_format": self.point_format}

    def query(
        self,
        images: List[ImageInput],
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], UsageStats]:
        parts: List[Dict[str, Any]] = []
        sent_size = None
        for img in images:
            data, sent_size = encode_jpeg(img, self.max_image_side)
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
        if not getattr(resp, "choices", None):
            raise VLMError("empty reply (no choices)")
        u = resp.usage
        usage = UsageStats(
            latency_s=latency,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            model=self.model,
        )
        out = _extract_json(resp.choices[0].message.content or "")
        if self.point_format == "pixels" and sent_size:
            out = pixels_to_norm1000(out, sent_size)
        return out, usage


def pixels_to_norm1000(out: Dict[str, Any], size: Tuple[int, int]) -> Dict[str, Any]:
    """Rescale a reply's pixel ``point`` (in the sent image) to the 0-1000 grid."""
    pt = out.get("point")
    if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
        return out
    try:
        u, v = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return out
    if u < 0 or v < 0:
        return out  # "no point" stays as is
    w, h = size
    return {**out, "point_raw": list(pt), "point": [1000.0 * u / w, 1000.0 * v / h]}
