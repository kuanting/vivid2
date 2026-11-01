"""Run recorder and log loader.

Each run directory holds one JSONL file per stream plus ``meta.json`` and,
once scored, ``result.json``. Metrics are computed offline from these files,
so every score can be recomputed or audited without the simulator.

Streams:
    trajectory  {t, pos[3], q[4]}
    collisions  {t, object_name, impact_point[3]}
    frames      {t, frame_id, cam_pos[3], cam_q[4], width, height, fov_deg, gt[{object_id, bbox}]}
    detections  {t, frame_id, layer, dets[{label, bbox, score}]}
    decisions   {t, frame_id, layer, output{...}, latency_s, input_tokens, output_tokens, cost_usd, model}
    reports     {t, frame_id, ned[3], confidence, source, note}
    events      {t, kind, ...}
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

STREAMS = ("trajectory", "collisions", "frames", "detections", "decisions", "reports", "events")


def _default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


class Recorder:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._files = {s: open(self.run_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in STREAMS}
        self._lock = threading.Lock()  # sim topic callbacks arrive on client threads

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=_default)
        with self._lock:
            self._files[stream].write(line + "\n")

    def write_json(self, name: str, obj: Dict[str, Any]) -> None:
        with open(self.run_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_default)

    def close(self) -> None:
        with self._lock:
            for f in self._files.values():
                f.close()


@dataclass
class RunLog:
    run_dir: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    streams: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: Path) -> "RunLog":
        run_dir = Path(run_dir)
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        streams = {}
        for s in STREAMS:
            p = run_dir / f"{s}.jsonl"
            streams[s] = (
                [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                if p.exists()
                else []
            )
        return cls(run_dir=run_dir, meta=meta, streams=streams)

    def __getitem__(self, stream: str) -> List[Dict[str, Any]]:
        return self.streams.get(stream, [])
