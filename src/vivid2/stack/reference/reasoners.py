"""Reference reasoning layers: a VLM reasoner and a no-VLM threshold baseline."""
from __future__ import annotations

from importlib import resources
from typing import Any, Dict, List, Tuple

import numpy as np

from vivid2.stack.base import Assessment, Detection, Frame, MissionSpec, Reasoner, UsageStats
from vivid2.vlm.base import VLMError, VLMProvider

ASSESSMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_found": {"type": "boolean"},
        "confidence": {"type": "number"},
        "point": {"type": "array", "items": {"type": "number"}},
        "description": {"type": "string"},
        "hazards": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string", "enum": ["continue", "inspect", "skip_area", "return_home"]},
    },
    "required": ["target_found", "confidence", "point", "description", "hazards", "next_action"],
    "additionalProperties": False,
}


def load_prompt(mission: str) -> str:
    return resources.files("vivid2.vlm.prompts").joinpath(f"{mission}.md").read_text(encoding="utf-8")


class VLMReasoner(Reasoner):
    """Asks a VLM whether the frame shows a target matching the mission brief.

    If a perception layer runs upstream, its detections are listed in the
    prompt as hints (hybrid stack); otherwise the VLM works from pixels alone.
    """

    name = "vlm_reasoner"

    def __init__(self, provider: VLMProvider, system: str = None):
        self.provider = provider
        self.system = system
        self._templates: Dict[str, str] = {}

    def _prompt(self, frame: Frame, detections: List[Detection], mission: MissionSpec) -> str:
        tpl = self._templates.setdefault(mission.mission, load_prompt(mission.mission))
        if detections:
            lines = [
                f"- {d.label} score {d.score:.2f} at normalised box "
                f"[{1000 * d.bbox[0] / frame.width:.0f}, {1000 * d.bbox[1] / frame.height:.0f}, "
                f"{1000 * d.bbox[2] / frame.width:.0f}, {1000 * d.bbox[3] / frame.height:.0f}]"
                for d in detections
            ]
            det_block = "An onboard detector proposed these candidates (may be wrong):\n" + "\n".join(lines) + "\n"
        else:
            det_block = ""
        return tpl.format(altitude_m=mission.altitude_m, brief=mission.brief, detections_block=det_block)

    def assess(self, frame: Frame, detections: List[Detection], mission: MissionSpec) -> Tuple[Assessment, UsageStats]:
        try:
            image = frame.jpeg if frame.jpeg is not None else frame.rgb
            out, usage = self.provider.query([image], self._prompt(frame, detections, mission), ASSESSMENT_SCHEMA, self.system)
        except VLMError as e:
            return Assessment(False, 0.0, raw={"error": str(e)}), UsageStats(model=self.provider.model)
        try:
            return parse_assessment(out, frame.width, frame.height), usage
        except ValueError as e:
            # Unusable reply: an abstention (scored as an error), not a "no target" answer.
            return Assessment(False, 0.0, raw={"error": f"invalid reply: {e}", "reply": out}), usage


NEXT_ACTIONS = set(ASSESSMENT_SCHEMA["properties"]["next_action"]["enum"])


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1", "false", "no", "0"):
        return v.strip().lower() in ("true", "yes", "1")
    raise ValueError(f"target_found is not a boolean: {v!r}")


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def parse_assessment(out: Any, width: int, height: int) -> Assessment:
    """Turn a VLM reply into an Assessment, tolerating loosely typed fields.

    Raises ValueError only when the decision itself (``target_found``) is
    missing or unreadable; other bad fields fall back to safe defaults.
    """
    if not isinstance(out, dict) or "target_found" not in out:
        raise ValueError("reply has no target_found field")
    found = _as_bool(out["target_found"])
    pixel = None
    pt = out.get("point")
    if found and isinstance(pt, (list, tuple)) and len(pt) == 2:
        u, v = _as_float(pt[0], -1.0), _as_float(pt[1], -1.0)
        if u >= 0 and v >= 0:
            pixel = (u / 1000.0 * width, v / 1000.0 * height)
    hazards = out.get("hazards")
    action = out.get("next_action")
    return Assessment(
        target_found=found,
        confidence=min(max(_as_float(out.get("confidence")), 0.0), 1.0),
        pixel=pixel,
        description=str(out.get("description") or ""),
        hazards=[str(h) for h in hazards] if isinstance(hazards, list) else [],
        next_action=action if action in NEXT_ACTIONS else "continue",
        raw=out,
    )


class DetectionThresholdReasoner(Reasoner):
    """No-VLM baseline: declare a find when the best detection clears a threshold."""

    name = "threshold_reasoner"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def assess(self, frame: Frame, detections: List[Detection], mission: MissionSpec) -> Tuple[Assessment, UsageStats]:
        best = max(detections, key=lambda d: d.score, default=None)
        if best is None or best.score < self.threshold:
            return Assessment(False, best.score if best else 0.0), UsageStats(model="threshold")
        return Assessment(True, best.score, pixel=best.center, description=best.label), UsageStats(model="threshold")
