"""Record/replay cache for VLM calls, for reproducible and free CI reruns.

``mode="record"`` forwards to an inner provider and appends every exchange to
a JSONL cache; ``mode="replay"`` answers from the cache only. Keys hash the
prompt, schema and image bytes, so a changed frame or prompt is a cache miss.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vivid2.stack.base import UsageStats
from vivid2.vlm.base import VLMError, VLMProvider


def cache_key(images: List[np.ndarray], prompt: str, schema: Dict[str, Any], system: Optional[str]) -> str:
    h = hashlib.sha1()
    h.update(prompt.encode())
    h.update(json.dumps(schema, sort_keys=True).encode())
    h.update((system or "").encode())
    for img in images:
        h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


class ReplayVLM(VLMProvider):
    def __init__(self, cache_path: str, mode: str = "replay", inner: Optional[VLMProvider] = None, miss: Optional[Dict[str, Any]] = None):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        if mode == "record" and inner is None:
            raise ValueError("record mode needs an inner provider")
        self.path = Path(cache_path)
        self.mode = mode
        self.inner = inner
        self.miss = miss
        self.model = f"replay:{inner.model}" if inner else "replay"
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def query(self, images, prompt, schema, system=None) -> Tuple[Dict[str, Any], UsageStats]:
        key = cache_key(images, prompt, schema, system)
        if key in self.cache:
            rec = self.cache[key]
            return rec["output"], UsageStats(latency_s=rec["latency_s"], input_tokens=rec["input_tokens"],
                                             output_tokens=rec["output_tokens"], cost_usd=0.0, model=rec["model"])
        if self.mode == "replay":
            if self.miss is not None:
                return dict(self.miss), UsageStats(model="replay-miss")
            raise VLMError(f"replay cache miss {key}")
        out, usage = self.inner.query(images, prompt, schema, system)
        rec = {"key": key, "output": out, "latency_s": usage.latency_s, "input_tokens": usage.input_tokens,
               "output_tokens": usage.output_tokens, "model": usage.model}
        self.cache[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return out, usage
