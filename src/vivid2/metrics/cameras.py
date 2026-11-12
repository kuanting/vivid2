"""Which recorded photos are scored: the search camera's (see vivid2.cameras).

Navigation cameras ("Front") are recorded for audit but have no ground truth and are never scored.
Logs from before cameras were recorded have no ``camera`` field: all their photos are the search
camera's.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Set

DEFAULT_CAMERA = "Payload"


def frame_camera(rec: Dict[str, Any]) -> str:
    return rec.get("camera", DEFAULT_CAMERA)


def search_cameras(meta: Dict[str, Any]) -> Set[str]:
    cams = meta.get("cameras") or {}
    found = {c for c, v in cams.items() if (v or {}).get("role") == "search"}
    return found or {DEFAULT_CAMERA}


def search_frames(frames: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The scored photos among ``frames``."""
    cams = search_cameras(meta)
    return [f for f in frames if frame_camera(f) in cams]


def frames_by_camera(frames: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(frame_camera(f) for f in frames).items()))
