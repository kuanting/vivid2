"""Claude (Anthropic API) VLM provider.

Credentials resolve the standard SDK way (ANTHROPIC_API_KEY, or an
``ant auth login`` profile). Structured output uses ``output_config.format``
so the reply is guaranteed-valid JSON for the given schema. Server-side
refusal fallbacks are enabled by default so a policy decline on one model is
retried on another inside the same call.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vivid2.stack.base import UsageStats
from vivid2.vlm.base import VLMError, VLMProvider, encode_jpeg

# USD per 1M tokens (input, output) at Anthropic first-party rates; used only
# for the cost column of reports. Haiku 5.5 is the <=100K-token prompt rate.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicVLM(VLMProvider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_tokens: int = 16000,  # current models always think; a low cap truncates replies
        max_image_side: int = 1024,
        fallbacks: bool = True,
        timeout_s: float = 60.0,
    ):
        import anthropic  # optional dependency: pip install vivid2[claude]

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout_s)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_image_side = max_image_side
        self.fallbacks = fallbacks

    def query(
        self,
        images: List[np.ndarray],
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], UsageStats]:
        content: List[Dict[str, Any]] = []
        for img in images:
            data, _ = encode_jpeg(img, self.max_image_side)
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        content.append({"type": "text", "text": prompt})

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        try:
            if self.fallbacks:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as e:
            raise VLMError(f"rate limited: {e}") from e
        except self._anthropic.APIStatusError as e:
            raise VLMError(f"API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise VLMError(f"connection error: {e}") from e
        latency = time.perf_counter() - t0

        # A refusal fallback can answer from a different model inside the same
        # call: report and price the model that actually served the reply.
        served = getattr(resp, "model", None) or self.model
        fallback = any(getattr(e, "type", None) == "fallback_message"
                       for e in (getattr(resp.usage, "iterations", None) or []))
        usage = UsageStats(
            latency_s=latency,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=served,
            fallback=fallback,
        )
        pin, pout = PRICES.get(served, PRICES.get(self.model, (0.0, 0.0)))
        usage.cost_usd = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6

        if resp.stop_reason == "refusal":
            raise VLMError(f"refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise VLMError("response truncated at max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise VLMError("no text block in response")
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as e:
            raise VLMError(f"invalid JSON in reply: {text[:200]!r}") from e
