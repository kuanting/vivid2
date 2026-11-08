"""Reference team stack: each drone searches its own strip at its own height and shares its finds.

The simplest team that is safe and does not waste effort, as a floor for team algorithms:

* **Strips.** The search area is cut into equal east-west strips, one per drone
  (``StripPlanner``); each drone flies the reference lawnmower over its own strip.
* **Height layers.** Drone *i* flies ``i x (min_separation_m + 1)`` above the survey altitude
  (60/66/72/78 m for four drones at 60 m), so drones in transit never meet at the same height.
  Each climbs and lands vertically over its own pad; pads are at least 10 m apart.
* **Sharing.** Over the radio (``ctx.radio``) a drone tells its teammates every candidate it
  goes to look at closer (a claim), every person it reports or dismisses, and where it is, once a
  second. A spot a teammate claimed or reported is left alone; the search ends when the team has
  found everyone.
* **One beat.** Every drone takes its photos on the same beat of mission time, so the simulator
  takes the team's photos together, in one lockstep batch.
* **Clear below.** Before descending for a closer look or a dip, a drone waits (up to a minute)
  until no teammate it has heard from in the last 5 s is within 30 m horizontally.

The same agents run in both team modes. Distributed: one per drone, over the scenario's radio.
Central (``TeamOf``): one FleetAgent runs them all over an ideal in-memory radio. Dynamic
re-allocation, probability maps or comms-aware planning are left to the stacks being tested.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vivid2.radio import Radio, RadioProfile
from vivid2.stack.base import Agent, FleetAgent, FleetContext, MissionContext
from vivid2.stack.reference.agents import SearchAgent

CLEAR_RADIUS_M = 30.0  # no teammate this close (horizontally) before a drone descends
CLEAR_WAIT_MAX_S = 60.0
POSITION_PERIOD_S = 1.0
POSITION_FRESH_S = 5.0
BEAT_TOLERANCE_S = 0.1  # a photo this close to the beat counts as on it


def layer_m(mission) -> float:
    """Height between two drones' layers: the separation limit plus a metre."""
    return float(mission.min_separation_m if mission.min_separation_m is not None else 5.0) + 1.0


class _AtLayer:
    """A drone's context with the mission's survey altitude raised to this drone's layer."""

    def __init__(self, ctx: MissionContext, mission):
        self._ctx = ctx
        self.mission = mission

    def __getattr__(self, name):
        return getattr(self._ctx, name)


class TeamSearchAgent(SearchAgent):
    """The reference SearchAgent as one drone of a team (see the module docstring)."""

    def __init__(self, *args, index: int = 0, size: int = 1, clear_radius_m: float = CLEAR_RADIUS_M, **kw):
        super().__init__(*args, **kw)
        self.index = index
        self.size = size
        self.clear_radius_m = clear_radius_m

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["team"] = {"strategy": "strips+layers", "size": self.size, "clear_radius_m": self.clear_radius_m,
                     "shares": ["claim", "found", "dismissed", "position"]}
        return d

    async def run(self, ctx: MissionContext) -> None:
        m = ctx.mission
        self._heard: Dict[str, tuple] = {}  # teammate -> (mission time, NED)
        self._from_team: List[np.ndarray] = []
        self._last_told_s = -1e9
        self._waited_s = 0.0
        mine = dataclasses.replace(m, altitude_m=m.altitude_m + self.index * layer_m(m))
        await super().run(_AtLayer(ctx, mine))

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
