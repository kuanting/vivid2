"""Reference reasoning layers: a VLM reasoner and a no-VLM threshold baseline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from typing import Any, Dict, List, Tuple

import numpy as np

from vivid2.stack.base import Assessment, Detection, Frame, MissionSpec, Reasoner, UsageStats
from vivid2.vlm.base import View, VLMError, VLMProvider, parse_tiles, vlm_views

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

    def __init__(self, provider: VLMProvider, system: str = None, tiles: str = "1x1"):
        parse_tiles(tiles)
        self.provider = provider
        self.system = system
        self.tiles = tiles
        self._templates: Dict[str, str] = {}

    def _prompt(self, view: View, n_views: int, index: int, detections: List[Detection], mission: MissionSpec) -> str:
        tpl = self._templates.setdefault(mission.mission, load_prompt(mission.mission))
        lines = []
        for d in detections:  # detector boxes are in frame pixels; give them in this view's 0-1000 coordinates
            u0, v0 = (d.bbox[0] - view.x0) / view.sx, (d.bbox[1] - view.y0) / view.sy
            u1, v1 = (d.bbox[2] - view.x0) / view.sx, (d.bbox[3] - view.y0) / view.sy
            if u1 < 0 or v1 < 0 or u0 > view.width or v0 > view.height:
                continue  # in another tile
            lines.append(f"- {d.label} score {d.score:.2f} at normalised box "
                         f"[{1000 * u0 / view.width:.0f}, {1000 * v0 / view.height:.0f}, "
                         f"{1000 * u1 / view.width:.0f}, {1000 * v1 / view.height:.0f}]")
        det_block = ("An onboard detector proposed these candidates (may be wrong):\n" + "\n".join(lines) + "\n"
                     if lines else "")
        prompt = tpl.format(altitude_m=mission.altitude_m, brief=mission.brief, detections_block=det_block)
        if n_views > 1:
            prompt += (f"\n\nThis image is tile {index + 1} of {n_views} of one camera frame (a {self.tiles} grid, "
                       "shown at full camera resolution). Answer for this tile only; a person may be cut off at its edge.")
        return prompt

    def _ask(self, view: View, n_views: int, index: int, detections, mission) -> Tuple[Assessment, UsageStats]:
        try:
            out, usage = self.provider.query([view.jpeg], self._prompt(view, n_views, index, detections, mission),
                                             ASSESSMENT_SCHEMA, self.system)
        except VLMError as e:
            return Assessment(False, 0.0, raw={"error": str(e)}), UsageStats(model=self.provider.model)
        try:
            a = parse_assessment(out, view.width, view.height)
        except ValueError as e:
            # Unusable reply: an abstention (scored as an error), not a "no target" answer.
            return Assessment(False, 0.0, raw={"error": f"invalid reply: {e}", "reply": out}), usage
        if a.pixel is not None:
            a.pixel = view.to_frame(*a.pixel)
        return a, usage

    def assess(self, frame: Frame, detections: List[Detection], mission: MissionSpec) -> Tuple[Assessment, UsageStats]:
        views = vlm_views(frame.jpeg if frame.jpeg is not None else frame.rgb, self.tiles)
        if len(views) == 1:
            return self._ask(views[0], 1, 0, detections, mission)
        with ThreadPoolExecutor(max_workers=len(views)) as pool:
            answers = list(pool.map(lambda iv: self._ask(iv[1], len(views), iv[0], detections, mission),
                                    enumerate(views)))
        return merge_tile_assessments(answers)


def merge_tile_assessments(answers: List[Tuple[Assessment, UsageStats]]) -> Tuple[Assessment, UsageStats]:
    """One decision for a tiled frame: the most confident 'found' tile, else the most confident tile.

    Tiles that errored are ignored unless every tile errored (then the frame is
    an abstention, like a single failed call).
    """
    usage = UsageStats(
        latency_s=max(u.latency_s for _, u in answers),  # tiles are asked in parallel
        input_tokens=sum(u.input_tokens for _, u in answers),
        output_tokens=sum(u.output_tokens for _, u in answers),
        cost_usd=sum(u.cost_usd for _, u in answers),
        model=next((u.model for _, u in answers if u.model), ""),
        fallback=any(u.fallback for _, u in answers),
    )
    ok = [(i, a) for i, (a, _) in enumerate(answers) if "error" not in a.raw]
    if not ok:
        errors = "; ".join(str(a.raw.get("error")) for a, _ in answers)
        return Assessment(False, 0.0, raw={"error": f"all {len(answers)} tiles failed: {errors}"}), usage
    found = [(i, a) for i, a in ok if a.target_found]
    i, best = max(found or ok, key=lambda ia: ia[1].confidence)
    hazards = sorted({h for _, a in ok for h in a.hazards})
    raw = dict(best.raw)
    raw.update({"tile": i, "tiles_answered": len(ok), "tiles_found": [j for j, _ in found]})
    return Assessment(best.target_found, best.confidence, best.pixel, best.description, hazards,
                      best.next_action, raw), usage


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
