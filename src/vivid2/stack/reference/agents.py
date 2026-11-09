"""Reference agents: a modular search agent (Perception -> Reasoner -> Planner -> Controller) and a survey agent."""
from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from vivid2.geo import nadir_footprint, pixel_ray_to_ground, pixel_to_world, world_to_pixel
from vivid2.stack.base import Agent, Assessment, Frame, MissionContext, Perception, Planner, Reasoner, TargetReport, UsageStats
from vivid2.stack.reference.controllers import DESCENT_MPS
from vivid2.stack.reference.lowflight import CanopyMap, Clearing, DipSettings
from vivid2.terrain import TerrainModel

DEPTH_SATURATION_M = 65.0  # 16-bit millimetre depth saturates at 65.535 m
NO_FRAME_RETRY_S = 0.1

# Closer look ("verify") before reporting, as search crews do
VERIFY_CLEARANCE_M = 10.0   # stay this far above the highest obstacle around the candidate
# The reference stacks' own floor for closer looks and dips: lower looks cost more time than they
# gain. A stack under test may fly as low as the scenario allows (limits.min_agl_m, 0.5 m).
LOOK_MIN_AGL_M = 8.0
VERIFY_RADIUS_M = 8.0       # ground radius checked for obstacles, and how far a confirmation may be from the candidate
VERIFY_DESCENT_MPS = 3.0

# Dips into clearings (see lowflight.py)
DIP_DESCENT_MPS = 3.0
DIP_CLIMB_MPS = 3.0
DIP_STEP_M = 10.0           # descend in steps, re-checking the clearing with each closer depth image
DIP_STEP_OVERHEAD_S = 1.5   # per step: settling, one frame
DIP_COLUMN_M = 4.0          # radius around the descent line that must stay clear
DIP_CLEARANCE_M = 3.0       # ... of anything higher than this below the next stop
DIP_KEEP_RADIUS = 0.75      # stop descending if the clearing shrinks below this share of min_radius_m
LANDING_OVERHEAD_S = 40.0   # final slow landing and disarm, in the time budget
MAX_GUARD_CLIMBS = 6
HOME_SKIP_M = 25.0          # no dips into the launch clearing


GROUND_TOLERANCE_M = 3.0  # a depth point this far above the coarse terrain is not on the ground


async def camera_geometry(ctx: MissionContext):
    """Payload camera ((width, height), fov_deg) for planning the survey before the first frame.

    From the mission spec; a context that doesn't describe its camera costs one frame, taken
    where the drone stands.
    """
    cam = ctx.mission.camera
    if cam:
        return (int(cam["width"]), int(cam["height"])), float(cam["fov_deg"])
    probe = await ctx.get_frame()
    return ((probe.width, probe.height), probe.fov_deg) if probe is not None else ((640, 480), 90.0)


async def pace(ctx: MissionContext, started_s: float, period_s: float) -> None:
    """Wait until ``period_s`` of mission time has passed since ``started_s`` (``ctx.elapsed_s()``).

    Mission time, not wall time: the simulator is paused while VIVID 2 records a
    frame, so wall-clock pacing would take fewer frames on a slower machine.
    """
    wait = period_s - (ctx.elapsed_s() - started_s)
    if wait > 0:
        await ctx.sleep(wait)


def ray_to_terrain(frame: Frame, u: float, v: float, terrain: TerrainModel, iterations: int = 4) -> Optional[np.ndarray]:
    """Where the pixel's ray meets the (smooth, coarse) terrain: a few flat-ground steps."""
    down = float(terrain.down_at(frame.cam_pos[0], frame.cam_pos[1]))
    p = None
    for _ in range(iterations):
        p = pixel_ray_to_ground(u, v, frame.cam_pos, frame.cam_q, frame.width, frame.height, frame.fov_deg, down)
        if p is None:
            return None
        down = float(terrain.down_at(p[0], p[1]))
    return p


def geolocate(frame: Frame, u: float, v: float, ground_down_m: float,
              terrain: Optional[TerrainModel] = None) -> Optional[np.ndarray]:
    """Pixel -> NED of a person on the ground.

    Uses the depth around the pixel. When that lands well above the terrain (the
    patch caught the crown a person is partly under), or there is no depth, the
    pixel's ray is intersected with the terrain instead (flat ``ground_down_m``
    without a terrain model).
    """
    terrain = terrain if terrain is not None else TerrainModel.flat(ground_down_m)
    if frame.depth is not None:
        x, y = int(round(u)), int(round(v))
        patch = frame.depth[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
        valid = patch[(patch > 0.1) & (patch < DEPTH_SATURATION_M)]
        if valid.size:
            # 75th percentile: a person partly under a crown shows leaves in the patch; lean on the ground
            p = pixel_to_world(u, v, float(np.percentile(valid, 75)), frame.cam_pos, frame.cam_q,
                               frame.width, frame.height, frame.fov_deg)
            if p[2] >= float(terrain.down_at(p[0], p[1])) - GROUND_TOLERANCE_M:
                return p
    return ray_to_terrain(frame, u, v, terrain)


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


class LowFlightMixin:
    """Canopy map, survey guard and the dip tour, shared by the search and survey agents.

    Subclasses set ``self.low`` (DipSettings or None), ``self.frame_period_s``,
    ``self._path`` and ``self._survey_down``, call ``_lf_start`` before takeoff and
    ``_lf_observe`` on every payload frame.
    """

    low: Optional[DipSettings] = None
    frame_period_s: float = 1.0

    def _lf_start(self, mission, wh, fov: float) -> None:
        self._canopy: Optional[CanopyMap] = None
        self._dips = 0
        self._climbs = 0
        self._odo_m = 0.0
        self._last_pos: Optional[np.ndarray] = None
        self._t_ns = 0  # sim time of the newest frame
        self._assess_s = 0.0  # running mean mission time of one look (perception + reasoner)
        if self.low is None:
            return
        terrain = mission.terrain if mission.terrain is not None else TerrainModel.flat(mission.ground_down_m)
        self._canopy = CanopyMap(mission.search_polygon_ne, terrain, self.low.cell_m, self.low.open_height_m)
        foot = min(nadir_footprint(self._dip_height(mission), wh[0], wh[1], fov))
        self._dip_spacing = max(2.0 * self.low.min_radius_m, 0.8 * foot)

    def _lf_observe(self, frame: Frame) -> None:
        self._t_ns = frame.t_ns
        p = np.asarray(frame.cam_pos, float)
        if self._last_pos is not None:
            self._odo_m += float(np.linalg.norm(p - self._last_pos))
        self._last_pos = p
        if self._canopy is not None:
            self._canopy.add(frame)

    def _lf_event(self, ctx: MissionContext, kind: str, **kw) -> None:
        ctx.log("events", {"t": self._t_ns, "kind": kind, **kw})

    def _dip_height(self, mission) -> float:
        return max(self.low.dip_agl_m, LOOK_MIN_AGL_M)

    def _all_found(self, ctx: MissionContext) -> bool:
        return False

    async def _resume(self, ctx: MissionContext, survey_down: float) -> None:
        """Climb back to the survey altitude and fly the rest of the path."""
        pos, _ = ctx.pose()
        back = [float(pos[0]), float(pos[1]), survey_down]
        await ctx.controller.goto(back, VERIFY_DESCENT_MPS)
        rest = remaining_path(self._path, pos)
        self._path = [back] + rest
        if rest:
            await ctx.controller.fly_path(rest, ctx.mission.max_speed_mps)

    async def _lf_guard(self, ctx: MissionContext) -> bool:
        """Climb if the canopy or ground ahead comes within the guard clearance; True if it climbed."""
        if self._canopy is None or self._climbs >= MAX_GUARD_CLIMBS:
            return False
        pos, _ = ctx.pose()
        rest = remaining_path(self._path, pos)
        if not rest:
            return False
        ahead = np.asarray(rest[0][:2], float) - pos[:2]
        dist = float(np.linalg.norm(ahead))
        if dist < 1e-3:
            return False
        end = pos[:2] + ahead / dist * min(dist, self.low.guard_lookahead_m)
        top = self._canopy.highest_in_corridor(pos[:2], end, self.low.guard_half_width_m)
        if top is None or top - pos[2] >= self.low.guard_clearance_m:
            return False
        climb = self.low.guard_clearance_m - (top - pos[2]) + 2.0
        self._climbs += 1
        self._survey_down -= climb
        self._path = [[w[0], w[1], min(w[2], self._survey_down)] for w in self._path]
        self._lf_event(ctx, "guard_climb", clearance_m=float(top - pos[2]), climb_m=float(climb))
        await self._resume(ctx, self._survey_down)
        return True

    def _transit_down(self, a, b) -> float:
        """Survey altitude, or higher if the mapped canopy along a-b would come too close."""
        top = self._canopy.highest_in_corridor(a[:2], b[:2], self.low.guard_half_width_m)
        if top is None:
            return self._survey_down
        return min(self._survey_down, top - self.low.guard_clearance_m)

    def _dip_cost(self, ctx: MissionContext, pos, spot: Clearing) -> Dict[str, float]:
        """Rough time (s) and path length (m) of a dip, and of the flight home after it."""
        m = ctx.mission
        drop = max(0.0, float(m.terrain.down_at(spot.n, spot.e)) - self._dip_height(m) - self._survey_down) \
            if m.terrain is not None else 0.0
        travel = math.hypot(spot.n - pos[0], spot.e - pos[1])
        home = math.hypot(spot.n - m.start_ned[0], spot.e - m.start_ned[1])
        look_s = self.low.dip_frames * max(self.frame_period_s, self._assess_s)
        dip_s = (travel / m.max_speed_mps + drop / DIP_DESCENT_MPS + drop / DIP_CLIMB_MPS
                 + math.ceil(drop / DIP_STEP_M) * DIP_STEP_OVERHEAD_S + look_s)
        home_s = home / m.max_speed_mps + max(0.0, m.start_ned[2] - self._survey_down) / DESCENT_MPS + LANDING_OVERHEAD_S
        return {"dip_s": dip_s, "home_s": home_s, "dip_m": travel + 2 * drop,
                "home_m": home + max(0.0, m.start_ned[2] - self._survey_down)}

    def _dip_affordable(self, ctx: MissionContext, pos, spot: Clearing) -> bool:
        m = ctx.mission
        c = self._dip_cost(ctx, pos, spot)
        if ctx.elapsed_s() + c["dip_s"] + c["home_s"] > self.low.time_share * m.timeout_s:
            return False
        if m.energy_budget_m and self._odo_m + c["dip_m"] + c["home_m"] > self.low.time_share * m.energy_budget_m:
            return False
        return True

    async def _lf_tour(self, ctx: MissionContext,
                       look: Callable[[MissionContext, Clearing], Awaitable[None]]) -> None:
        """Dip into the clearings found, nearest first, while time and energy allow.

        The launch clearing is skipped: the drone saw it up close at takeoff.
        """
        home = ctx.mission.start_ned
        visited: List[tuple] = []

        def candidates() -> List[Clearing]:
            return [c for c in self._canopy.clearings(self.low.min_radius_m, self._dip_spacing, exclude=visited)
                    if math.hypot(c.n - home[0], c.e - home[1]) > HOME_SKIP_M and self._may_descend_at(c.n, c.e)]

        planned = candidates()
        self._lf_event(ctx, "dip_plan", spots=[[round(c.n, 1), round(c.e, 1), round(c.radius_m, 1)] for c in planned])
        while self._dips < self.low.max_dips and not ctx.should_stop() and not self._all_found(ctx):
            pos, _ = ctx.pose()
            spots = candidates()
            if not spots:
                break
            spot = min(spots, key=lambda c: math.hypot(c.n - pos[0], c.e - pos[1]))
            if not self._dip_affordable(ctx, pos, spot):
                self._lf_event(ctx, "dip_budget", dips=self._dips, elapsed_s=ctx.elapsed_s(), odo_m=self._odo_m)
                break
            visited.append((spot.n, spot.e))
            self._dips += 1
            await self._dip(ctx, spot, look)
        self._lf_event(ctx, "dip_tour_end", dips=self._dips, clearings=len(planned))

    async def _dip(self, ctx: MissionContext, spot: Clearing,
                   look: Callable[[MissionContext, Clearing], Awaitable[None]]) -> None:
        """Fly over a clearing, descend into it in checked steps, look, climb back."""
        m = ctx.mission
        pos, _ = ctx.pose()
        transit = self._transit_down(pos, (spot.n, spot.e))
        self._lf_event(ctx, "dip_start", at=[spot.n, spot.e], radius_m=spot.radius_m)
        await ctx.controller.goto([spot.n, spot.e, transit], m.max_speed_mps)
        z = transit
        stopped = None
        bottom = None
        while not ctx.should_stop():
            col = self._canopy.highest_within(spot.n, spot.e, DIP_COLUMN_M)    # highest thing in the descent line
            floor = self._canopy.lowest_within(spot.n, spot.e, DIP_COLUMN_M)   # the clearing floor seen
            ground = float(m.terrain.down_at(spot.n, spot.e)) if m.terrain is not None else m.ground_down_m
            if floor is not None:
                ground = min(ground, floor)  # never trust the coarse DEM below the floor the camera sees
            bottom = ground - self._dip_height(m)
            if z >= bottom - 0.5:
                break
            nz = min(z + DIP_STEP_M, bottom)
            if col is not None and col - nz < DIP_CLEARANCE_M:
                stopped = "obstacle"
                break
            if self._canopy.radius_at(spot.n, spot.e) < DIP_KEEP_RADIUS * self.low.min_radius_m:
                stopped = "clearing_shrank"
                break
            await ctx.controller.goto([spot.n, spot.e, nz], DIP_DESCENT_MPS)
            z = nz
            frame = await ctx.get_frame()
            if frame is not None:
                self._lf_observe(frame)
        height = (bottom + self._dip_height(m) - z) if bottom is not None else None
        self._lf_event(ctx, "dip_bottom", at=[spot.n, spot.e], height_m=height, stopped=stopped)
        await look(ctx, spot)
        await ctx.controller.goto([spot.n, spot.e, transit], DIP_CLIMB_MPS)
        self._lf_event(ctx, "dip_end", at=[spot.n, spot.e])

    def _lf_describe(self) -> Any:
        return self.low.describe() if self.low is not None else False

    def _may_descend_at(self, n: float, e: float) -> bool:
        """Whether this drone may leave its survey height over (n, e) for a closer look or a dip."""
        return True


def _scalar_params(obj: Any) -> Optional[Dict[str, Any]]:
    """Public scalar attributes of a layer object (its tunable settings)."""
    if obj is None:
        return None
    return {k: v for k, v in sorted(getattr(obj, "__dict__", {}).items())
            if not k.startswith("_") and isinstance(v, (bool, int, float, str))}


class SearchAgent(LowFlightMixin, Agent):
    """Fly a coverage pattern; assess frames; report geolocated targets.

    ``perception`` may be None (pure-VLM stack); ``reasoner`` decides finds.

    With ``verify`` on, a candidate (a find at ``candidate_confidence`` or more)
    is not reported straight away. The agent:
    1. leaves the survey path and flies to a point on the line of sight from the
       candidate to the camera that saw it (so the view stays as unobstructed);
    2. descends along that line to about VERIFY_CLEARANCE_M above the highest
       obstacle around the candidate and under the look point, read from the
       depth image, and never below LOOK_MIN_AGL_M;
    3. takes up to ``verify_frames`` close-ups, the VLM looking at a full-resolution
       crop around the candidate, and reports the target only if a majority confirm
       it at ``min_confidence`` (a single noisy "yes" is not enough), located as the
       mean of the confirming close-ups;
    4. climbs back and resumes the path.

    Dismissed spots are not checked again. ``max_verifications`` bounds the
    detours; after that, finds at ``min_confidence`` are reported directly.

    With ``dip`` (DipSettings), the agent then dips into the clearings it mapped
    during the survey (lowflight.py) and looks ``dip_frames`` times at the
    bottom of each; a majority of those looks must agree to report a find.
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
        dip: Optional[DipSettings] = None,
    ):
        self.name = name
        self.low = dip
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
            "geolocate": "depth_or_terrain_ray",
            "verify": ({"candidate_confidence": self.candidate_confidence, "frames": self.verify_frames,
                        "max": self.max_verifications, "approach": "line_of_sight",
                        "min_height_m": LOOK_MIN_AGL_M} if self.verify else False),
            **({"dip": self._lf_describe()} if self.low is not None else {}),  # old run hashes stay valid
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
        wh, fov = await camera_geometry(ctx)
        self._survey_down = m.ground_down_m - m.altitude_m
        path = self._plan(m, fov, wh)
        self._reported: List[np.ndarray] = []
        self._dismissed: List[np.ndarray] = []
        self._verifications = 0
        self._lf_start(m, wh, fov)

        await self._depart(ctx, path)
        here, _ = ctx.pose()
        self._path = [[float(v) for v in here]] + [list(w) for w in path]  # where the flight starts counts as the first point
        await ctx.controller.fly_path(path, m.max_speed_mps)

        finished = False  # everyone found, or the reasoner asked to go home
        while not ctx.should_stop() and not ctx.controller.path_done():
            t_loop = ctx.elapsed_s()
            frame = await ctx.get_frame()
            if frame is None:
                await ctx.sleep(NO_FRAME_RETRY_S)  # don't hammer the sim while no image is available
                continue
            self._lf_observe(frame)
            if await self._lf_guard(ctx):
                continue
            step = await self._survey_step(ctx, frame)
            if step == "stop":
                finished = True
                break
            if step == "resumed":
                continue
            await self._pace(ctx, t_loop)

        if self.low is not None and not finished and not ctx.should_stop():
            await self._lf_tour(ctx, self._dip_look)
        await self._go_home(ctx)

    # -- hooks for a drone flying with others (stack/reference/team.py) -------------------
    def _plan(self, mission, fov: float, wh) -> List[List[float]]:
        """The survey path."""
        return self.planner.plan(mission, fov, wh)

    async def _depart(self, ctx: MissionContext, path: List[List[float]]) -> None:
        """Take off to the survey height; the survey path starts from there."""
        await ctx.controller.takeoff(self._survey_down)

    async def _go_home(self, ctx: MissionContext) -> None:
        await ctx.controller.return_and_land(ctx.mission.start_ned, ctx.mission.max_speed_mps)

    async def _survey_step(self, ctx: MissionContext, frame: Frame) -> str:
        """Assess one survey frame: "continue", "resumed" (after a closer look) or "stop"."""
        m = ctx.mission
        assessment, _ = await self._assess(ctx, frame)
        checking = self.verify and self._verifications < self.max_verifications
        threshold = self.candidate_confidence if checking else self.min_confidence
        if assessment.target_found and assessment.pixel and assessment.confidence >= threshold:
            ned = geolocate(frame, assessment.pixel[0], assessment.pixel[1], m.ground_down_m, m.terrain)
            if ned is not None and self._is_new(ned, self._reported + self._dismissed):
                if checking:
                    if not self._may_descend_at(float(ned[0]), float(ned[1])):
                        return "continue"  # not this drone's to look at closer (a teammate's area)
                    self._verifications += 1
                    report = await self._verify(ctx, frame, assessment, ned, self._survey_down)
                    if report is not None:
                        self._reported.append(report.ned)
                        ctx.report_target(report)
                    else:
                        self._dismissed.append(ned)
                    if self._all_found(ctx) or ctx.should_stop():
                        return "stop"
                    await self._resume(ctx, self._survey_down)
                    return "resumed"
                self._reported.append(ned)
                ctx.report_target(TargetReport(ned=ned, confidence=assessment.confidence, source=self.reasoner.name,
                                               frame_id=frame.frame_id, t_ns=frame.t_ns, note=assessment.description))
                if self._all_found(ctx):
                    return "stop"
        if assessment.next_action == "return_home":
            ctx.log("events", {"t": frame.t_ns, "kind": "agent_abort", "reason": assessment.description})
            return "stop"
        return "continue"

    def _all_found(self, ctx: MissionContext) -> bool:
        return len(self._reported) >= ctx.mission.expected_targets

    async def _pace(self, ctx: MissionContext, t_loop: float) -> None:
        """Wait for the next photo: ``frame_period_s`` after this one was asked for."""
        await pace(ctx, t_loop, self.frame_period_s)

    async def _dip_look(self, ctx: MissionContext, spot: Clearing) -> None:
        """Look ``dip_frames`` times from the bottom of a dip; report a find a majority agree on."""
        m = ctx.mission
        need = self.low.dip_frames // 2 + 1
        yes: List[Any] = []  # (ned, assessment, frame)
        for _ in range(self.low.dip_frames):
            if ctx.should_stop():
                break
            t_loop = ctx.elapsed_s()
            frame = await ctx.get_frame()
            if frame is None:
                await ctx.sleep(NO_FRAME_RETRY_S)
                continue
            self._lf_observe(frame)
            a, _ = await self._assess(ctx, frame, phase="dip")
            if a.target_found and a.pixel and a.confidence >= self.min_confidence:
                at = geolocate(frame, a.pixel[0], a.pixel[1], m.ground_down_m, m.terrain)
                if at is not None:
                    yes.append((at, a, frame))
            await self._pace(ctx, t_loop)
        # The largest group of looks that agree on where the person is
        best: List[Any] = []
        for at, _, _ in yes:
            group = [y for y in yes if np.linalg.norm(y[0][:2] - at[:2]) <= self.dedup_radius_m]
            if len(group) > len(best):
                best = group
        confirmed = len(best) >= need
        if confirmed:
            ned = np.mean([y[0] for y in best], axis=0)
            if self._is_new(ned, self._reported):
                last = best[-1]
                self._reported.append(ned)
                ctx.report_target(TargetReport(ned=ned, confidence=float(np.mean([y[1].confidence for y in best])),
                                               source=self.reasoner.name, frame_id=last[2].frame_id, t_ns=last[2].t_ns,
                                               note=last[1].description))
        self._lf_event(ctx, "dip_look", at=[spot.n, spot.e], votes=len(best), confirmed=confirmed)

    def _is_new(self, ned: np.ndarray, known: List[np.ndarray]) -> bool:
        return all(np.linalg.norm(ned[:2] - k[:2]) > self.dedup_radius_m for k in known)

    async def _assess(self, ctx: MissionContext, frame: Frame, phase: str = "survey"):
        # The agent's thinking takes mission time: the drone flies on (or hovers) meanwhile.
        t0 = ctx.elapsed_s()
        dets = []
        if self.perception is not None:
            dets = await ctx.think(self.perception.detect, frame)
            ctx.log("detections", {
                "t": frame.t_ns, "frame_id": frame.frame_id, "layer": "perception", "phase": phase,
                "dets": [{"label": d.label, "bbox": list(d.bbox), "score": d.score} for d in dets],
            })
        assessment, usage = await ctx.think(self.reasoner.assess, frame, dets, ctx.mission)
        took = ctx.elapsed_s() - t0
        ctx.log("decisions", {**decision_record(frame, assessment, usage), "phase": phase, "response_s": round(took, 3)})
        self._assess_s = took if not getattr(self, "_assess_s", 0.0) else 0.8 * self._assess_s + 0.2 * took
        return assessment, dets

    async def _verify(self, ctx: MissionContext, frame: Frame, assessment: Assessment, ned: np.ndarray,
                      survey_down: float) -> Optional[TargetReport]:
        """Come closer along the line of sight and look again; a confirmed report, or None.

        The look point lies on the line from the candidate to the camera that saw it,
        so the candidate stays where it was in the image, unobstructed as it was, only
        larger. (Straight overhead, a person seen at an angle under a crown's edge is
        hidden by the crown.) Its height keeps VERIFY_CLEARANCE_M above the highest
        obstacle around the candidate and under the look point, and LOOK_MIN_AGL_M.
        """
        m = ctx.mission
        ground = float(ned[2])  # the candidate sits on the ground
        top = highest_obstacle_down(frame, assessment.pixel[0], assessment.pixel[1])
        height = LOOK_MIN_AGL_M
        if top is not None:
            height = max(height, ground - top + VERIFY_CLEARANCE_M)
        cam = np.asarray(frame.cam_pos, float)
        seen_from = ground - float(cam[2])  # height of the detecting camera above the candidate
        frac = float(np.clip(height / seen_from, 0.0, 1.0)) if seen_from > 0 else 1.0
        look = ned + frac * (cam - ned)
        # Keep clear of whatever lies under the look point too (it is between the candidate and the camera's nadir).
        below = world_to_pixel([look[0], look[1], ground], cam, frame.cam_q, frame.width, frame.height, frame.fov_deg)
        if below is not None and 0 <= below[0] < frame.width and 0 <= below[1] < frame.height:
            top_look = highest_obstacle_down(frame, below[0], below[1])
            if top_look is not None and look[2] > top_look - VERIFY_CLEARANCE_M:
                frac = float(np.clip((ground - (top_look - VERIFY_CLEARANCE_M)) / seen_from, frac, 1.0))
                look = ned + frac * (cam - ned)
        look_down = max(float(look[2]), survey_down)  # never climb above the survey altitude to look
        pos, _ = ctx.pose()
        ctx.log("events", {"t": frame.t_ns, "kind": "verify_start", "candidate": [float(v) for v in ned],
                           "confidence": assessment.confidence, "height_m": ground - look_down,
                           "look_at": [float(look[0]), float(look[1]), look_down], "frame_id": frame.frame_id})
        await ctx.controller.goto([float(look[0]), float(look[1]), float(pos[2])], m.max_speed_mps)
        await ctx.controller.goto([float(look[0]), float(look[1]), look_down], VERIFY_DESCENT_MPS)
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
                await ctx.sleep(NO_FRAME_RETRY_S)
                continue
            self._lf_observe(close)
            looks += 1
            last_t = close.t_ns
            px = world_to_pixel(ned, close.cam_pos, close.cam_q, close.width, close.height, close.fov_deg)
            inside = px is not None and 0 <= px[0] < close.width and 0 <= px[1] < close.height
            close.focus = (px[0], px[1]) if inside else (close.width / 2.0, close.height / 2.0)
            a, _ = await self._assess(ctx, close, phase="verify")
            best_conf = max(best_conf, a.confidence if a.target_found else 0.0)
            if a.target_found and a.pixel and a.confidence >= self.min_confidence:
                at = geolocate(close, a.pixel[0], a.pixel[1], m.ground_down_m, m.terrain)
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


class SurveyAgent(LowFlightMixin, Agent):
    """Fly the coverage pattern and record every frame; makes no decisions.

    Used with ``--frame-bank`` to record the frames that VLMs are then compared
    on offline (``vivid2 eval-frames``), so every model sees identical images.
    With ``dip``, it also dips into the clearings it mapped and records
    ``dip_frames`` frames from the bottom of each.
    """

    def __init__(self, name: str, planner: Planner, frame_period_s: float = 1.0, dip: Optional[DipSettings] = None):
        self.name = name
        self.planner = planner
        self.frame_period_s = frame_period_s
        self.low = dip

    def describe(self) -> Dict[str, Any]:
        d = {"agent": self.name, "planner": self.planner.name, "frame_period_s": self.frame_period_s,
             "params": {"planner": _scalar_params(self.planner)}}
        if self.low is not None:
            d["dip"] = self._lf_describe()
        return d

    async def run(self, ctx: MissionContext) -> None:
        m = ctx.mission
        wh, fov = await camera_geometry(ctx)
        path = self.planner.plan(m, fov, wh)
        self._survey_down = m.ground_down_m - m.altitude_m
        self._lf_start(m, wh, fov)
        await ctx.controller.takeoff(self._survey_down)
        here, _ = ctx.pose()
        self._path = [[float(v) for v in here]] + [list(w) for w in path]
        await ctx.controller.fly_path(path, m.max_speed_mps)
        while not ctx.should_stop() and not ctx.controller.path_done():
            t_loop = ctx.elapsed_s()
            frame = await ctx.get_frame()
            if frame is None:
                await ctx.sleep(NO_FRAME_RETRY_S)
                continue
            self._lf_observe(frame)
            if await self._lf_guard(ctx):
                continue
            await pace(ctx, t_loop, self.frame_period_s)
        if self.low is not None and not ctx.should_stop():
            await self._lf_tour(ctx, self._dip_record)
        await ctx.controller.return_and_land(m.start_ned, m.max_speed_mps)

    async def _dip_record(self, ctx: MissionContext, spot: Clearing) -> None:
        """Record ``dip_frames`` frames from the bottom of a dip."""
        for _ in range(self.low.dip_frames):
            if ctx.should_stop():
                break
            t_loop = ctx.elapsed_s()
            frame = await ctx.get_frame()
            if frame is not None:
                self._lf_observe(frame)
            await pace(ctx, t_loop, self.frame_period_s)
