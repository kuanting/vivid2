"""Reference team stack: each drone searches its own strip and shares its finds.

The simplest team that is safe and does not waste effort, as a floor for team algorithms:

* **Strips.** The search area is cut into equal east-west strips, one per drone
  (``StripPlanner``). Each drone flies the reference lawnmower over its own strip **at the
  mission's survey height**, so its photos are the same as a lone drone's.
* **Transit heights.** To reach its strip and to come home, a drone flies one layer
  (``min_separation_m`` + 2 m) or more above the survey height: the drone whose strip starts
  farthest from the pads flies highest. So a drone passing over a nearer strip is always above
  the drone climbing or descending there, and two drones in transit never share a height.
* **The pads' keep-out circle.** Drones climb and land vertically over their own pads (at least
  10 m apart). No drone flies across the circle around the pads: each enters and leaves it
  straight along its own pad's direction from the centre, and survey lanes go around it. So a
  drone climbing or landing never has another one passing over it.
* **Closer looks and dips** only in the drone's own strip (a candidate in a teammate's strip is
  left to that teammate), and only once no teammate heard from recently is within
  ``clear_radius_m`` horizontally. The same check comes before every climb or descent between
  survey and transit height.
* **Sharing.** Over the radio (``ctx.radio``) a drone tells its teammates every candidate it
  goes to look at closer (a claim), every person it reports or dismisses, and where it is, once a
  second. A spot a teammate claimed or reported is left alone; the search ends when the team has
  found everyone.
* **One beat.** Every drone takes its photos on the same beat of mission time, so the simulator
  takes the team's photos together, in one lockstep batch.

The same agents run in both team modes. Distributed: one per drone, over the scenario's radio.
Central (``TeamOf``): one FleetAgent runs them all over an ideal in-memory radio. Dynamic
re-allocation, probability maps or comms-aware planning are left to the stacks being tested.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vivid2.radio import Radio, RadioProfile
from vivid2.stack.base import Agent, FleetAgent, FleetContext, MissionContext
from vivid2.stack.reference.agents import VERIFY_DESCENT_MPS, SearchAgent
from vivid2.stack.reference.planners import StripPlanner

CLEAR_RADIUS_M = 30.0  # no teammate this close (horizontally) before a drone changes height
CLEAR_WAIT_MAX_S = 60.0
POSITION_PERIOD_S = 1.0
POSITION_FRESH_S = 5.0
BEAT_TOLERANCE_S = 0.1  # a photo this close to the beat counts as on it
KEEP_OUT_MARGIN_M = 15.0  # the pads' keep-out circle reaches this far past the outermost pad
# Layers are this much more than the separation limit apart: real drones hold a height to about
# half a metre, and a drone circling the pads crosses the others' lines in, a layer above or below.
LAYER_MARGIN_M = 2.0
ARC_STEP_DEG = 30.0  # waypoints along the keep-out circle
WAIT_STEP_S = 0.5


def layer_m(mission) -> float:
    """Height between two transit layers: the separation limit plus LAYER_MARGIN_M."""
    return float(mission.min_separation_m if mission.min_separation_m is not None else 5.0) + LAYER_MARGIN_M


def _angle(c: np.ndarray, p) -> float:
    return math.atan2(float(p[1]) - c[1], float(p[0]) - c[0])


def _on_circle(c: np.ndarray, r: float, a: float, down: float) -> List[float]:
    return [float(c[0] + r * math.cos(a)), float(c[1] + r * math.sin(a)), float(down)]


def arc(c: np.ndarray, r: float, a0: float, a1: float, down: float) -> List[List[float]]:
    """Waypoints along the circle from angle a0 to a1 (the shorter way, a1 included)."""
    d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    n = max(1, math.ceil(abs(d) / math.radians(ARC_STEP_DEG)))
    return [_on_circle(c, r, a0 + d * k / n, down) for k in range(1, n + 1)]


def around(path: List[List[float]], c: np.ndarray, r: float) -> List[List[float]]:
    """The path with every leg that would cross the circle routed around it, at the leg's height."""
    out: List[List[float]] = []
    for w in path:
        p = np.asarray(w[:2], float)
        if np.linalg.norm(p - c) < r:  # a waypoint inside: move it out to the circle
            w = _on_circle(c, r, _angle(c, p), w[2])
        if out:
            a, b = np.asarray(out[-1][:2], float), np.asarray(w[:2], float)
            ab = b - a
            L2 = float(ab @ ab)
            if L2 > 1e-9:
                # where the leg a + t*ab meets the circle
                f = a - c
                bq, cq = 2 * float(f @ ab), float(f @ f) - r * r
                disc = bq * bq - 4 * L2 * cq
                if disc > 0:
                    s = math.sqrt(disc)
                    t0, t1 = (-bq - s) / (2 * L2), (-bq + s) / (2 * L2)
                    if 0 < t0 < 1 and 0 < t1 < 1:
                        p0, p1 = a + t0 * ab, a + t1 * ab
                        z = out[-1][2]
                        out.append([float(p0[0]), float(p0[1]), z])
                        out.extend(arc(c, r, _angle(c, p0), _angle(c, p1), z))
        out.append([float(v) for v in w])
    return out


class TeamSearchAgent(SearchAgent):
    """The reference SearchAgent as one drone of a team (see the module docstring)."""

    def __init__(self, *args, index: int = 0, size: int = 1, clear_radius_m: float = CLEAR_RADIUS_M, **kw):
        super().__init__(*args, **kw)
        self.index = index
        self.size = size
        self.clear_radius_m = clear_radius_m

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["team"] = {"strategy": "strips+transit_layers", "size": self.size, "clear_radius_m": self.clear_radius_m,
                     "keep_out_margin_m": KEEP_OUT_MARGIN_M, "layer_margin_m": LAYER_MARGIN_M, "shares": ["claim", "found", "dismissed", "position"]}
        return d

    async def run(self, ctx: MissionContext) -> None:
        self._heard: Dict[str, tuple] = {}  # teammate -> (mission time, NED)
        self._from_team: List[np.ndarray] = []
        self._last_told_s = -1e9
        self._waited_s = 0.0
        self._geo: Optional[Dict[str, Any]] = None
        await super().run(ctx)

    # -- where this drone flies ----------------------------------------------------------------
    def _plan(self, mission, fov: float, wh) -> List[List[float]]:
        path = super()._plan(mission, fov, wh)
        team = mission.team or []
        if self.size <= 1 or len(team) != self.size:
            return path  # flying alone: the single-drone flight
        pads = np.array([t["start_ned"][:2] for t in team], float)
        c = pads.mean(axis=0)
        r = float(np.linalg.norm(pads - c, axis=1).max()) + KEEP_OUT_MARGIN_M
        sidelap = getattr(self.planner, "sidelap", 0.2)
        inset = getattr(self.planner, "inset", 2.0)
        starts = []
        for j in range(self.size):
            pj = path if j == self.index else StripPlanner(j, self.size, sidelap, inset).plan(mission, fov, wh)
            pj = around(pj, c, r) if pj else pj
            starts.append(np.asarray(pj[0][:2], float) if pj else c)
        order = sorted(range(self.size), key=lambda j: (float(np.linalg.norm(starts[j] - c)), j))
        rank = order.index(self.index)
        poly = np.asarray(mission.search_polygon_ne, float)
        width = (poly[:, 1].max() - poly[:, 1].min()) / self.size
        lo = poly[:, 1].min() + self.index * width
        self._geo = {
            "centre": c, "radius": r, "pad": np.asarray(mission.start_ned, float),
            "transit_down": self._survey_down - (rank + 1) * layer_m(mission), "rank": rank,
            "strip_e": (lo, lo + width),
        }
        return around(path, c, r)

    def _radial(self) -> List[float]:
        g = self._geo
        return _on_circle(g["centre"], g["radius"], _angle(g["centre"], g["pad"]), g["transit_down"])

    async def _fly(self, ctx, waypoints: List[List[float]]) -> None:
        """Fly a route, keeping in touch with the team, until it is flown."""
        await ctx.controller.fly_path(waypoints, ctx.mission.max_speed_mps)
        while not ctx.controller.path_done() and not ctx.should_stop():
            self._listen(ctx)
            self._tell_position(ctx)
            await ctx.sleep(WAIT_STEP_S)

    async def _depart(self, ctx, path: List[List[float]]) -> None:
        if self._geo is None:
            return await super()._depart(ctx, path)
        g = self._geo
        await ctx.controller.takeoff(g["transit_down"])  # vertically over the pad
        self._tell_position(ctx)
        start = path[0]
        route = [self._radial()]  # out of the keep-out circle along this pad's own direction
        route += arc(g["centre"], g["radius"], _angle(g["centre"], g["pad"]), _angle(g["centre"], start),
                     g["transit_down"])
        route.append([start[0], start[1], g["transit_down"]])
        await self._fly(ctx, route)
        await self._clear_below(ctx, start[:2])
        await ctx.controller.goto(start, VERIFY_DESCENT_MPS)  # down to the survey height

    async def _go_home(self, ctx) -> None:
        if self._geo is None:
            return await super()._go_home(ctx)
        g = self._geo
        pos, _ = ctx.pose()
        await self._clear_below(ctx, pos[:2])
        if pos[2] > g["transit_down"]:
            await ctx.controller.goto([pos[0], pos[1], g["transit_down"]], VERIFY_DESCENT_MPS)
        a = _angle(g["centre"], pos)
        route = []
        if np.linalg.norm(pos[:2] - g["centre"]) > g["radius"]:
            route.append(_on_circle(g["centre"], g["radius"], a, g["transit_down"]))
        route += arc(g["centre"], g["radius"], a, _angle(g["centre"], g["pad"]), g["transit_down"])
        route.append(self._radial())
        await self._fly(ctx, route)
        await super()._go_home(ctx)  # straight in along the pad's own direction, then down

    def _may_descend_at(self, n: float, e: float) -> bool:
        g = self._geo
        if g is None:
            return True
        lo, hi = g["strip_e"]
        inside = lo <= e < hi or (self.index == self.size - 1 and e == hi)
        return inside and math.hypot(n - g["centre"][0], e - g["centre"][1]) > g["radius"]

    # -- the radio -------------------------------------------------------------------------
    def _listen(self, ctx) -> None:
        radio = getattr(ctx, "radio", None)
        if radio is None:
            return
        for msg in radio.poll():
            p = msg.payload if isinstance(msg.payload, dict) else {}
            ned = np.asarray(p.get("ned", ()), float)
            if p.get("kind") == "found" and ned.size == 3:
                if self._is_new(ned, self._reported):  # two drones may report the same person
                    self._from_team.append(ned)
                    self._reported.append(ned)
            elif p.get("kind") in ("dismissed", "claim") and ned.size == 3:
                # a teammate is taking (or took) a closer look there: leave it to them
                self._from_team.append(ned)
                self._dismissed.append(ned)
            elif p.get("kind") == "position" and ned.size == 3:
                self._heard[msg.sender] = (ctx.elapsed_s(), ned)

    def _own(self, items: List[np.ndarray], since: int) -> List[np.ndarray]:
        return [x for x in items[since:] if not any(x is y for y in self._from_team)]

    def _tell(self, ctx, kind: str, items: List[np.ndarray]) -> None:
        radio = getattr(ctx, "radio", None)
        if radio is not None:
            for ned in items:
                radio.broadcast({"kind": kind, "ned": [round(float(v), 2) for v in ned]})

    def _tell_position(self, ctx) -> None:
        radio = getattr(ctx, "radio", None)
        if radio is None or ctx.elapsed_s() - self._last_told_s < POSITION_PERIOD_S:
            return
        self._last_told_s = ctx.elapsed_s()
        pos, _ = ctx.pose()
        radio.broadcast({"kind": "position", "ned": [round(float(v), 1) for v in pos]})

    # -- hooks into the single-drone search ----------------------------------------------
    async def _pace(self, ctx, t_loop: float) -> None:
        """The next photo on the team's shared beat (multiples of ``frame_period_s`` of mission time).

        Drones whose photos fall due together share one lockstep batch: about 2.5 times less wall
        time for four drones. Mission time is the same either way.
        """
        p = self.frame_period_s
        if p <= 0:
            return
        beat = math.ceil((t_loop + p - BEAT_TOLERANCE_S) / p) * p
        wait = beat - ctx.elapsed_s()
        if wait > 0:
            await ctx.sleep(wait)

    def _all_found(self, ctx) -> bool:
        self._listen(ctx)
        return super()._all_found(ctx)

    async def _survey_step(self, ctx, frame) -> str:
        self._listen(ctx)
        if self._all_found(ctx):
            return "stop"
        self._tell_position(ctx)
        nr, nd = len(self._reported), len(self._dismissed)
        step = await super()._survey_step(ctx, frame)
        self._tell(ctx, "found", self._own(self._reported, nr))
        self._tell(ctx, "dismissed", self._own(self._dismissed, nd))
        return step

    async def _dip_look(self, ctx, spot) -> None:
        self._listen(ctx)
        nr = len(self._reported)
        await super()._dip_look(ctx, spot)
        self._tell(ctx, "found", self._own(self._reported, nr))

    async def _verify(self, ctx, frame, assessment, ned, survey_down):
        self._tell(ctx, "claim", [ned])  # so teammates don't fly to the same candidate
        await self._clear_below(ctx, ned)
        return await super()._verify(ctx, frame, assessment, ned, survey_down)

    async def _dip(self, ctx, spot, look) -> None:
        await self._clear_below(ctx, (spot.n, spot.e))
        await super()._dip(ctx, spot, look)

    async def _clear_below(self, ctx, ne) -> None:
        """Wait until no teammate heard from recently is within ``clear_radius_m`` of the spot."""
        if getattr(ctx, "radio", None) is None:
            return
        waited = 0.0
        while not ctx.should_stop() and waited < CLEAR_WAIT_MAX_S:
            self._listen(ctx)
            self._tell_position(ctx)
            now = ctx.elapsed_s()
            near = [i for i, (t, p) in self._heard.items()
                    if now - t <= POSITION_FRESH_S and np.hypot(p[0] - ne[0], p[1] - ne[1]) < self.clear_radius_m]
            if not near:
                break
            if waited == 0.0:
                ctx.log("events", {"t": self._t_ns, "kind": "wait_clear", "teammates": near})
            await ctx.sleep(1.0)
            waited += 1.0
        self._waited_s += waited


async def _run_all(coros: Dict[str, Any]) -> None:
    """Run every drone's agent; the first error cancels the others and is raised."""
    tasks = [asyncio.ensure_future(c) for c in coros.values()]
    try:
        for done in asyncio.as_completed(tasks):
            await done
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t


class TeamOf(FleetAgent):
    """A central team made of per-drone agents talking over an ideal in-memory radio."""

    team_mode = "central"

    def __init__(self, name: str, agents: Sequence[Agent]):
        self.name = name
        self.agents = list(agents)

    @property
    def uses_ground_truth(self) -> bool:
        return any(getattr(a, "uses_ground_truth", False) for a in self.agents)

    def describe(self) -> Dict[str, Any]:
        return {**self.agents[0].describe(), "agent": self.name, "team_mode": self.team_mode}

    async def run(self, fleet: FleetContext) -> None:
        ids = list(fleet.drones)
        if len(ids) != len(self.agents):
            raise ValueError(f"{self.name}: {len(self.agents)} agents for a team of {len(ids)}")
        radio = Radio(RadioProfile.ideal(), ids, clock=fleet.elapsed_s, t_ns=fleet.t_ns, log=fleet.log)
        for i in ids:
            fleet.drones[i].radio = radio.port(i, fleet.drones[i].sleep)
        await _run_all({i: a.run(fleet.drones[i]) for i, a in zip(ids, self.agents)})
