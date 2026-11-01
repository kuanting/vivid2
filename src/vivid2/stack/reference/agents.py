"""Reference agents: a modular search agent (Perception -> Reasoner -> Planner -> Controller) and a survey agent."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from vivid2.geo import pixel_ray_to_ground, pixel_to_world
from vivid2.stack.base import Agent, Assessment, Frame, MissionContext, Perception, Planner, Reasoner, TargetReport, UsageStats

DEPTH_SATURATION_M = 65.0  # 16-bit millimetre depth saturates at 65.535 m
NO_FRAME_RETRY_S = 0.1


def geolocate(frame: Frame, u: float, v: float, ground_down_m: float) -> Optional[np.ndarray]:
    """Pixel -> NED using median depth around the pixel, else a flat-ground ray."""
    if frame.depth is not None:
        x, y = int(round(u)), int(round(v))
        patch = frame.depth[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
        valid = patch[(patch > 0.1) & (patch < DEPTH_SATURATION_M)]
        if valid.size:
            return pixel_to_world(u, v, float(np.median(valid)), frame.cam_pos, frame.cam_q,
                                  frame.width, frame.height, frame.fov_deg)
    return pixel_ray_to_ground(u, v, frame.cam_pos, frame.cam_q, frame.width, frame.height,
                               frame.fov_deg, ground_down_m)


def _scalar_params(obj: Any) -> Optional[Dict[str, Any]]:
    """Public scalar attributes of a layer object (its tunable settings)."""
    if obj is None:
        return None
    return {k: v for k, v in sorted(getattr(obj, "__dict__", {}).items())
            if not k.startswith("_") and isinstance(v, (bool, int, float, str))}


class SearchAgent(Agent):
    """Fly a coverage pattern; assess frames; report geolocated targets.

    ``perception`` may be None (pure-VLM stack); ``reasoner`` decides finds.
    """

    def __init__(
        self,
        name: str,
        planner: Planner,
        reasoner: Reasoner,
        perception: Optional[Perception] = None,
        min_confidence: float = 0.6,
        frame_period_s: float = 1.0,
        dedup_radius_m: float = 15.0,
    ):
        self.name = name
        self.planner = planner
        self.reasoner = reasoner
        self.perception = perception
        self.min_confidence = min_confidence
        self.frame_period_s = frame_period_s
        self.dedup_radius_m = dedup_radius_m

    def describe(self) -> Dict[str, Any]:
        provider = getattr(self.reasoner, "provider", None)
        return {
            "agent": self.name,
            "perception": self.perception.name if self.perception else None,
            "reasoner": self.reasoner.name,
            "reasoner_model": getattr(provider, "model", None),
            "planner": self.planner.name,
            "min_confidence": self.min_confidence,
            "frame_period_s": self.frame_period_s,
            "dedup_radius_m": self.dedup_radius_m,
            # Layer settings (sidelap, thresholds, VLM effort...) so a config change forces a re-run.
            "params": {
                "planner": _scalar_params(self.planner),
                "reasoner": _scalar_params(self.reasoner),
                "provider": provider.fingerprint() if hasattr(provider, "fingerprint") else _scalar_params(provider),
                "perception": _scalar_params(self.perception),
            },
        }

    async def run(self, ctx: MissionContext) -> None:
        m = ctx.mission
        probe = await ctx.get_frame()  # learn camera geometry for planning
        wh = (probe.width, probe.height) if probe else (640, 480)
        fov = probe.fov_deg if probe else 90.0
        path = self.planner.plan(m, fov, wh)

        await ctx.controller.takeoff(m.ground_down_m - m.altitude_m)
        await ctx.controller.fly_path(path, m.max_speed_mps)

        reported: List[np.ndarray] = []
        while not ctx.should_stop() and not ctx.controller.path_done():
            t_loop = asyncio.get_running_loop().time()
            frame = await ctx.get_frame()
            if frame is None:
                await asyncio.sleep(NO_FRAME_RETRY_S)  # don't hammer the sim while no image is available
                continue

            dets = []
            if self.perception is not None:
                dets = await asyncio.to_thread(self.perception.detect, frame)
                ctx.log("detections", {
                    "t": frame.t_ns, "frame_id": frame.frame_id, "layer": "perception",
                    "dets": [{"label": d.label, "bbox": list(d.bbox), "score": d.score} for d in dets],
                })

            assessment, usage = await asyncio.to_thread(self.reasoner.assess, frame, dets, m)
            ctx.log("decisions", decision_record(frame, assessment, usage))

            if assessment.target_found and assessment.pixel and assessment.confidence >= self.min_confidence:
                ned = geolocate(frame, assessment.pixel[0], assessment.pixel[1], m.ground_down_m)
                if ned is not None and all(np.linalg.norm(ned[:2] - r[:2]) > self.dedup_radius_m for r in reported):
                    reported.append(ned)
                    ctx.report_target(TargetReport(ned=ned, confidence=assessment.confidence, source=self.reasoner.name,
                                                   frame_id=frame.frame_id, t_ns=frame.t_ns, note=assessment.description))
                    if len(reported) >= m.expected_targets:
                        break
            if assessment.next_action == "return_home":
                ctx.log("events", {"t": frame.t_ns, "kind": "agent_abort", "reason": assessment.description})
                break

            wait = self.frame_period_s - (asyncio.get_running_loop().time() - t_loop)
            if wait > 0:
                await asyncio.sleep(wait)

        await ctx.controller.return_and_land(m.start_ned, m.max_speed_mps)


def decision_record(frame: Frame, assessment: Assessment, usage: UsageStats) -> Dict[str, Any]:
    """The ``decisions`` stream record for one reasoner call (closed or open loop)."""
    raw = assessment.raw or {}
    out: Dict[str, Any] = {
        "target_found": assessment.target_found, "confidence": assessment.confidence,
        "pixel": list(assessment.pixel) if assessment.pixel else None,
        "description": assessment.description, "hazards": assessment.hazards,
        "next_action": assessment.next_action,
    }
    for k in ("error", "point_raw"):
        if k in raw:
            out[k] = raw[k]
    rec = {
        "t": frame.t_ns, "frame_id": frame.frame_id, "layer": "reasoner", "output": out,
        "latency_s": usage.latency_s, "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd, "model": usage.model,
    }
    if usage.fallback:
        rec["fallback"] = True
    return rec


class SurveyAgent(Agent):
    """Fly the coverage pattern and record every frame; makes no decisions.

    Used with ``--frame-bank`` to record the frames that VLMs are then compared
    on offline (``vivid2 eval-frames``), so every model sees identical images.
    """

    def __init__(self, name: str, planner: Planner, frame_period_s: float = 1.0):
        self.name = name
        self.planner = planner
        self.frame_period_s = frame_period_s

    def describe(self) -> Dict[str, Any]:
        return {"agent": self.name, "planner": self.planner.name, "frame_period_s": self.frame_period_s,
                "params": {"planner": _scalar_params(self.planner)}}

    async def run(self, ctx: MissionContext) -> None:
        m = ctx.mission
        probe = await ctx.get_frame()
        wh = (probe.width, probe.height) if probe else (640, 480)
        path = self.planner.plan(m, probe.fov_deg if probe else 90.0, wh)
        await ctx.controller.takeoff(m.ground_down_m - m.altitude_m)
        await ctx.controller.fly_path(path, m.max_speed_mps)
        while not ctx.should_stop() and not ctx.controller.path_done():
            t_loop = asyncio.get_running_loop().time()
            if await ctx.get_frame() is None:
                await asyncio.sleep(NO_FRAME_RETRY_S)
                continue
            wait = self.frame_period_s - (asyncio.get_running_loop().time() - t_loop)
            if wait > 0:
                await asyncio.sleep(wait)
        await ctx.controller.return_and_land(m.start_ned, m.max_speed_mps)
