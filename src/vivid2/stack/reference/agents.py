"""Reference agents: a modular search agent (Perception -> Reasoner -> Planner -> Controller) and a survey agent."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from vivid2.geo import pixel_ray_to_ground, pixel_to_world, world_to_pixel
from vivid2.stack.base import Agent, Assessment, Frame, MissionContext, Perception, Planner, Reasoner, TargetReport, UsageStats

DEPTH_SATURATION_M = 65.0  # 16-bit millimetre depth saturates at 65.535 m
NO_FRAME_RETRY_S = 0.1

# Closer look ("verify") before reporting, as search crews do
VERIFY_CLEARANCE_M = 10.0   # stay this far above the highest obstacle around the candidate
VERIFY_MIN_EXTRA_M = 5.0    # ... and this far above the mission's minimum height above ground
VERIFY_RADIUS_M = 8.0       # ground radius checked for obstacles, and how far a confirmation may be from the candidate
VERIFY_DESCENT_MPS = 3.0


def geolocate(frame: Frame, u: float, v: float, ground_down_m: float) -> Optional[np.ndarray]:
    """Pixel -> NED using median depth around the pixel, else a flat-ground ray."""
    if frame.depth is not None:
        x, y = int(round(u)), int(round(v))
        patch = frame.depth[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
        valid = patch[(patch > 0.1) & (patch < DEPTH_SATURATION_M)]
        if valid.size:
            # 75th percentile: a person partly under a crown shows leaves in the patch; lean on the ground
            return pixel_to_world(u, v, float(np.percentile(valid, 75)), frame.cam_pos, frame.cam_q,
                                  frame.width, frame.height, frame.fov_deg)
    return pixel_ray_to_ground(u, v, frame.cam_pos, frame.cam_q, frame.width, frame.height,
                               frame.fov_deg, ground_down_m)


def highest_obstacle_down(frame: Frame, u: float, v: float, radius_m: float = VERIFY_RADIUS_M) -> Optional[float]:
    """NED down of the highest surface within ``radius_m`` of pixel (u, v), from a nadir depth image."""
    if frame.depth is None:
        return None
    d_center = frame.depth[int(np.clip(round(v), 0, frame.height - 1)), int(np.clip(round(u), 0, frame.width - 1))]
    rng = float(d_center) if 0.1 < d_center < DEPTH_SATURATION_M else 50.0
    f = frame.width / 2.0 / np.tan(np.radians(frame.fov_deg) / 2.0)
    r_px = int(max(3, radius_m * f / max(rng, 1.0)))
    x, y = int(round(u)), int(round(v))
    patch = frame.depth[max(y - r_px, 0):y + r_px + 1, max(x - r_px, 0):x + r_px + 1]
    valid = patch[(patch > 0.1) & (patch < DEPTH_SATURATION_M)]
    if not valid.size:
        return None
    return float(frame.cam_pos[2] + valid.min())  # planar depth of a nadir camera ~ vertical distance


def remaining_path(path: List[List[float]], pos_ne) -> List[List[float]]:
    """Waypoints still to fly after an interruption at ``pos_ne``: those after the nearest path segment."""
    if len(path) < 2:
        return list(path)
    p = np.asarray(pos_ne[:2], float)
    best_i, best_d = 0, np.inf
    for i in range(len(path) - 1):
        a, b = np.asarray(path[i][:2], float), np.asarray(path[i + 1][:2], float)
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        d = float(np.linalg.norm(a + t * ab - p))
        if d < best_d:
            best_i, best_d = i, d
    return [list(w) for w in path[best_i + 1:]]


def _scalar_params(obj: Any) -> Optional[Dict[str, Any]]:
    """Public scalar attributes of a layer object (its tunable settings)."""
    if obj is None:
        return None
    return {k: v for k, v in sorted(getattr(obj, "__dict__", {}).items())
            if not k.startswith("_") and isinstance(v, (bool, int, float, str))}


class SearchAgent(Agent):
    """Fly a coverage pattern; assess frames; report geolocated targets.

    ``perception`` may be None (pure-VLM stack); ``reasoner`` decides finds.

    With ``verify`` on, a candidate (a find at ``candidate_confidence`` or more)
    is not reported straight away. The agent:
    1. leaves the survey path and flies over the candidate;
    2. descends to about VERIFY_CLEARANCE_M above the highest obstacle around it,
       read from the depth image, and never below the mission's minimum height
       + VERIFY_MIN_EXTRA_M;
    3. takes up to ``verify_frames`` close-ups, the VLM looking at a full-resolution
       crop around the candidate, and reports the target only if a majority confirm
       it at ``min_confidence`` (a single noisy "yes" is not enough), located as the
       mean of the confirming close-ups;
    4. climbs back and resumes the path.

    Dismissed spots are not checked again. ``max_verifications`` bounds the
    detours; after that, finds at ``min_confidence`` are reported directly.
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
        verify: bool = False,
        candidate_confidence: float = 0.3,
        verify_frames: int = 3,
        max_verifications: int = 8,
    ):
        self.name = name
        self.planner = planner
        self.reasoner = reasoner
        self.perception = perception
        self.min_confidence = min_confidence
        self.frame_period_s = frame_period_s
        self.dedup_radius_m = dedup_radius_m
        self.verify = verify
        self.candidate_confidence = candidate_confidence
        self.verify_frames = verify_frames
        self.max_verifications = max_verifications
        self._path: List[List[float]] = []

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
            "verify": ({"candidate_confidence": self.candidate_confidence, "frames": self.verify_frames,
                        "max": self.max_verifications} if self.verify else False),
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
        survey_down = m.ground_down_m - m.altitude_m

        await ctx.controller.takeoff(survey_down)
        here, _ = ctx.pose()
        self._path = [[float(v) for v in here]] + [list(w) for w in path]  # where the flight starts counts as the first point
        await ctx.controller.fly_path(path, m.max_speed_mps)

        reported: List[np.ndarray] = []
        dismissed: List[np.ndarray] = []
        verifications = 0
        while not ctx.should_stop() and not ctx.controller.path_done():
            t_loop = asyncio.get_running_loop().time()
            frame = await ctx.get_frame()
            if frame is None:
                await asyncio.sleep(NO_FRAME_RETRY_S)  # don't hammer the sim while no image is available
                continue

            assessment, _ = await self._assess(ctx, frame)
            checking = self.verify and verifications < self.max_verifications
            threshold = self.candidate_confidence if checking else self.min_confidence
            if assessment.target_found and assessment.pixel and assessment.confidence >= threshold:
                ned = geolocate(frame, assessment.pixel[0], assessment.pixel[1], m.ground_down_m)
                if ned is not None and self._is_new(ned, reported + dismissed):
                    if checking:
                        verifications += 1
                        report = await self._verify(ctx, frame, assessment, ned, survey_down)
                        if report is not None:
                            reported.append(report.ned)
                            ctx.report_target(report)
                        else:
                            dismissed.append(ned)
                        if len(reported) >= m.expected_targets or ctx.should_stop():
                            break
                        await self._resume(ctx, survey_down)
                        continue
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

    def _is_new(self, ned: np.ndarray, known: List[np.ndarray]) -> bool:
        return all(np.linalg.norm(ned[:2] - k[:2]) > self.dedup_radius_m for k in known)

    async def _assess(self, ctx: MissionContext, frame: Frame, phase: str = "survey"):
        dets = []
        if self.perception is not None:
            dets = await asyncio.to_thread(self.perception.detect, frame)
            ctx.log("detections", {
                "t": frame.t_ns, "frame_id": frame.frame_id, "layer": "perception", "phase": phase,
                "dets": [{"label": d.label, "bbox": list(d.bbox), "score": d.score} for d in dets],
            })
        assessment, usage = await asyncio.to_thread(self.reasoner.assess, frame, dets, ctx.mission)
        ctx.log("decisions", {**decision_record(frame, assessment, usage), "phase": phase})
        return assessment, dets

    async def _verify(self, ctx: MissionContext, frame: Frame, assessment: Assessment, ned: np.ndarray,
                      survey_down: float) -> Optional[TargetReport]:
        """Fly over a candidate and descend for a closer look; a confirmed report, or None."""
        m = ctx.mission
        ground = float(ned[2])  # the candidate sits on the ground
        top = highest_obstacle_down(frame, assessment.pixel[0], assessment.pixel[1])
        height = m.min_agl_m + VERIFY_MIN_EXTRA_M
        if top is not None:
            height = max(height, ground - top + VERIFY_CLEARANCE_M)
        look_down = max(ground - height, survey_down)  # never climb above the survey altitude to look
        pos, _ = ctx.pose()
        ctx.log("events", {"t": frame.t_ns, "kind": "verify_start", "candidate": [float(v) for v in ned],
                           "confidence": assessment.confidence, "height_m": ground - look_down,
                           "frame_id": frame.frame_id})
        await ctx.controller.goto([float(ned[0]), float(ned[1]), float(pos[2])], m.max_speed_mps)
        await ctx.controller.goto([float(ned[0]), float(ned[1]), look_down], VERIFY_DESCENT_MPS)
        need = self.verify_frames // 2 + 1  # majority of the close-ups
        yes: List[Any] = []  # (ned, assessment, frame) of confirming close-ups
        looks = 0
        best_conf = 0.0
        last_t = frame.t_ns
        while looks < self.verify_frames and len(yes) < need and (looks - len(yes)) <= self.verify_frames - need:
            if ctx.should_stop():
                break
            close = await ctx.get_frame()
            if close is None:
                continue
            looks += 1
            last_t = close.t_ns
            px = world_to_pixel(ned, close.cam_pos, close.cam_q, close.width, close.height, close.fov_deg)
            inside = px is not None and 0 <= px[0] < close.width and 0 <= px[1] < close.height
            close.focus = (px[0], px[1]) if inside else (close.width / 2.0, close.height / 2.0)
            a, _ = await self._assess(ctx, close, phase="verify")
            best_conf = max(best_conf, a.confidence if a.target_found else 0.0)
            if a.target_found and a.pixel and a.confidence >= self.min_confidence:
                at = geolocate(close, a.pixel[0], a.pixel[1], m.ground_down_m)
                if at is not None and np.linalg.norm(at[:2] - ned[:2]) <= 2 * VERIFY_RADIUS_M + self.dedup_radius_m:
                    yes.append((at, a, close))
        found: Optional[TargetReport] = None
        if len(yes) >= need:
            at = np.mean([y[0] for y in yes], axis=0)
            last = yes[-1]
            found = TargetReport(ned=at, confidence=float(np.mean([y[1].confidence for y in yes])),
                                 source=self.reasoner.name, frame_id=last[2].frame_id, t_ns=last[2].t_ns,
                                 note=last[1].description)
        ctx.log("events", {"t": last_t, "kind": "verify_end", "confirmed": found is not None,
                           "confidence": best_conf, "looks": looks, "votes": len(yes)})
        return found

    async def _resume(self, ctx: MissionContext, survey_down: float) -> None:
        """Climb back to the survey altitude and fly the rest of the path."""
        pos, _ = ctx.pose()
        back = [float(pos[0]), float(pos[1]), survey_down]
        await ctx.controller.goto(back, VERIFY_DESCENT_MPS)
        rest = remaining_path(self._path, pos)
        self._path = [back] + rest
        if rest:
            await ctx.controller.fly_path(rest, ctx.mission.max_speed_mps)


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
