"""Record/replay cache for VLM calls.

``mode="record"`` forwards to the inner provider and appends every exchange to
a JSONL cache; ``mode="replay"`` answers from the cache only and never calls the
model. The key covers the provider's fingerprint (model and settings), the
prompt, the schema and the SHA-256 of the exact JPEG bytes sent, so a cache
shared between models, efforts or frames never returns the wrong answer.

Replay is exact on saved frames (``vivid2 eval-frames``). Closed-loop runs
re-render every frame, and renders are rarely byte-identical, so a closed-loop
replay mostly misses; a miss is an error, scored as an abstention.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vivid2.stack.base import UsageStats
from vivid2.vlm.base import ImageInput, VLMError, VLMProvider, image_bytes

KEY_VERSION = 2


def cache_key(fingerprint: Dict[str, Any], images: List[ImageInput], prompt: str, schema: Dict[str, Any],
              system: Optional[str], max_side: int = 1024) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"v": KEY_VERSION, "provider": fingerprint, "prompt": prompt, "schema": schema,
                         "system": system or ""}, sort_keys=True).encode())
    for img in images:
        h.update(hashlib.sha256(image_bytes(img, max_side)).digest())
    return h.hexdigest()


class ReplayVLM(VLMProvider):
    def __init__(self, cache_path: str, inner: VLMProvider, mode: str = "record"):
        """``inner`` is needed in both modes: its fingerprint is part of the key."""
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        self.path = Path(cache_path)
        self.mode = mode
        self.inner = inner
        self.model = inner.model
        self._lock = threading.Lock()
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def fingerprint(self) -> Dict[str, Any]:
        return self.inner.fingerprint()

    def query(self, images, prompt, schema, system=None) -> Tuple[Dict[str, Any], UsageStats]:
        key = cache_key(self.fingerprint(), images, prompt, schema, system,
                        getattr(self.inner, "max_image_side", 1024))
        rec = self.cache.get(key)
        if rec is not None:
            # The recorded usage, so replayed metrics match the original run exactly.
            return dict(rec["output"]), UsageStats(
                latency_s=rec["latency_s"], input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
                cost_usd=rec.get("cost_usd", 0.0), model=rec.get("served_model", self.model),
                fallback=rec.get("fallback", False))
        if self.mode == "replay":
            raise VLMError(f"replay cache miss {key[:12]} (frame or settings not in {self.path.name})")
        out, usage = self.inner.query(images, prompt, schema, system)
        rec = {"key": key, "output": out, "latency_s": usage.latency_s, "input_tokens": usage.input_tokens,
               "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd, "served_model": usage.model,
               "fallback": usage.fallback}
        with self._lock:
            self.cache[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return out, usage
