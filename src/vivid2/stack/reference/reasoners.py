"""Reference reasoning layers: a VLM reasoner and a no-VLM threshold baseline."""
from __future__ import annotations

from importlib import resources
from typing import Any, Dict, List, Tuple

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
            out, usage = self.provider.query([frame.rgb], self._prompt(frame, detections, mission), ASSESSMENT_SCHEMA, self.system)
        except VLMError as e:
            return Assessment(False, 0.0, raw={"error": str(e)}), UsageStats(model=self.provider.model)
        pt = out.get("point") or [-1, -1]
        pixel = None
        if out.get("target_found") and len(pt) == 2 and min(pt) >= 0:
            pixel = (pt[0] / 1000.0 * frame.width, pt[1] / 1000.0 * frame.height)
        return (
            Assessment(
                target_found=bool(out.get("target_found")),
                confidence=float(out.get("confidence", 0.0)),
                pixel=pixel,
                description=str(out.get("description", "")),
                hazards=list(out.get("hazards", [])),
                next_action=str(out.get("next_action", "continue")),
                raw=out,
            ),
            usage,
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
