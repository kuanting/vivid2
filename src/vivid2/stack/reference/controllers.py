"""Controller layer backed by the Project AirSim Drone API.

Works unchanged with both the built-in simple_flight controller and PX4
(``controller.type: px4-api`` in the robot config): Project AirSim forwards
these move commands to PX4 over MAVLink. Turning the drone (``heading``,
``yaw_deg``, ``turn_to``) is verified on simple_flight only.

Headings are kept within +-174 degrees: Project AirSim's yaw handling does not wrap at +-180,
so a turn to due south never registers as done (and its "face the direction of travel" mode
stalls on a southbound leg). "travel" therefore flies each leg pointed along its own bearing.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vivid2.cameras import wrap_deg
from vivid2.crash import UPRIGHT_DEG
from vivid2.geo import tilt_deg_of, yaw_deg_of
from vivid2.stack.base import Controller

DESCENT_MPS = 3.0
LAND_FROM_M = 2.0  # height above the start point where the final, slow landing begins
VERTICAL_LEG_M = 1.0  # a path leg shorter than this horizontally is flown holding the heading
TURN_LIMIT_DEG = 174.0  # Project AirSim's yaw handling does not wrap at +-180 degrees
TURN_MARGIN_DEG = 5.0
TURN_TIMEOUT_S = 20.0
CLIMB_MPS = 3.0  # the climb to the requested height after takeoff
CLIMB_MARGIN_S = 30.0  # the climb's time limit: twice its time at CLIMB_MPS plus this (simulator time)
AIRBORNE_TOL_M = 5.0  # after the climb the drone must be this close to the requested height, and upright


class TakeoffError(RuntimeError):
    """The drone did not reach its height after takeoff: it crashed (e.g. rolled over and fell back).

    The run ends as an error at once (running the campaign again re-runs it) rather than waiting out the
    hard time limit.
    """


def tilt_deg(q: Dict[str, float]) -> float:
    """Tilt of a pose's rotation dict (see vivid2.geo.tilt_deg_of): 0 level, 180 upside down."""
    return tilt_deg_of((q["w"], q["x"], q["y"], q["z"]))


def safe_heading(deg: float) -> float:
    """A heading in degrees, wrapped and kept within +-TURN_LIMIT_DEG."""
    return max(-TURN_LIMIT_DEG, min(TURN_LIMIT_DEG, wrap_deg(float(deg))))


def yaw_kwargs(heading: Optional[float]) -> Dict[str, Any]:
    """The client's yaw arguments for an absolute heading in degrees (None: hold, as before)."""
    if heading is None:
        return {}
    from projectairsim.drone import YawControlMode

    return {"yaw_control_mode": YawControlMode.MaxDegreeOfFreedom, "yaw_is_rate": False,
            "yaw": math.radians(safe_heading(heading))}


def travel_legs(start: Sequence[float], waypoints: List[List[float]]) -> List[Tuple[List[float], Optional[float]]]:
    """Each waypoint with the heading to fly to it: the leg's bearing, or None (hold) for a vertical leg.

    A near-vertical leg has no direction to face (its bearing is that of a tiny horizontal step).
    """
    legs = []
    prev = list(start)
    for w in waypoints:
        dn, de = w[0] - prev[0], w[1] - prev[1]
        legs.append((list(w), None if math.hypot(dn, de) < VERTICAL_LEG_M else math.degrees(math.atan2(de, dn))))
        prev = w
    return legs


class AirSimController(Controller):
    name = "airsim_api"

    def __init__(self, drone, clock_ns: Callable[[], int], log: Callable[[str, Dict[str, Any]], None]):
        self._drone = drone
        self._clock = clock_ns
        self._log = log
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self.took_off = False  # the takeoff climb is done (a crash before then is the simulator's: vivid2.crash)

    def _event(self, kind: str, **kw) -> None:
        self._log("events", {"t": self._clock(), "kind": kind, **kw})

    async def _cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._stopped = True  # a path flown in pieces stops after the current one
            self._drone.cancel_last_task()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        self._task = None

    async def takeoff(self, altitude_down_m: float) -> None:
        """Take off, then climb straight up to ``altitude_down_m``; raises TakeoffError if that failed.

        Project AirSim's takeoff (2 m) has a time limit of its own, and the climb gets one here, both in
        simulator time. A drone that crashed on the way up never finishes its climb, so without the limit
        the run would wait out its hard time limit (over an hour for a 30-minute scenario).
        """
        self._drone.enable_api_control()
        self._drone.arm()
        self._event("takeoff")
        await (await self._drone.takeoff_async())
        pose = self._drone.get_ground_truth_pose()
        t = pose["translation"]
        limit_s = 2.0 * abs(t["z"] - altitude_down_m) / CLIMB_MPS + CLIMB_MARGIN_S
        await (await self._drone.move_to_position_async(t["x"], t["y"], altitude_down_m, velocity=CLIMB_MPS,
                                                         timeout_sec=limit_s))
        pose = self._drone.get_ground_truth_pose()
        off = abs(pose["translation"]["z"] - altitude_down_m)
        tilt = tilt_deg(pose["rotation"])
        if off > AIRBORNE_TOL_M or tilt > UPRIGHT_DEG:
            self._event("takeoff_failed", down=round(pose["translation"]["z"], 2), target_down=round(altitude_down_m, 2),
                        tilt_deg=round(tilt, 1))
            raise TakeoffError(f"takeoff failed: after the climb ({limit_s:.0f} s limit) the drone is {off:.1f} m from "
                               f"its height and tilted {tilt:.0f} degrees; it probably crashed")
        self.took_off = True

    async def fly_path(self, waypoints: List[List[float]], speed_mps: float, heading: Any = "hold") -> None:
        if not (heading in ("hold", "travel") or isinstance(heading, (int, float))):
            raise ValueError(f"heading must be 'hold', 'travel' or degrees, not {heading!r}")
        await self._cancel()
        here = self._drone.get_ground_truth_pose()["translation"]
        start = [here["x"], here["y"], here["z"]]
        self._event("path_start", waypoints=waypoints, speed=speed_mps, **{"from": start},
                    **({"heading": heading} if heading != "hold" else {}))
        if heading == "travel":
            self._stopped = False
            self._task = asyncio.ensure_future(self._fly_legs(travel_legs(start, waypoints), speed_mps))
            return
        self._task = await self._drone.move_on_path_async(
            path=waypoints, velocity=speed_mps, **yaw_kwargs(None if heading == "hold" else heading))

    async def _fly_legs(self, legs: List[Tuple[List[float], Optional[float]]], speed_mps: float) -> None:
        """One leg at a time, each pointed along its own bearing (vertical legs hold the heading)."""
        for w, bearing in legs:
            if self._stopped:
                return
            task = await self._drone.move_to_position_async(w[0], w[1], w[2], velocity=speed_mps,
                                                             **yaw_kwargs(bearing))
            await task

    def path_done(self) -> bool:
        return self._task is None or self._task.done()

    async def goto(self, ned: Sequence[float], speed_mps: float, yaw_deg: Optional[float] = None) -> None:
        await self._cancel()
        self._event("goto", target=list(ned), speed=speed_mps,
                    **({"yaw_deg": safe_heading(yaw_deg)} if yaw_deg is not None else {}))
        await (await self._drone.move_to_position_async(ned[0], ned[1], ned[2], velocity=speed_mps,
                                                         **yaw_kwargs(yaw_deg)))

    async def turn_to(self, yaw_deg: float, rate_dps: Optional[float] = None) -> None:
        await self._cancel()
        want = wrap_deg(yaw_deg)
        eff = safe_heading(want)
        self._event("turn", yaw_deg=eff, **({"commanded_deg": want} if eff != want else {}),
                    **({"rate_dps": rate_dps} if rate_dps else {}))
        await (await self._drone.rotate_to_yaw_async(math.radians(eff), timeout_sec=TURN_TIMEOUT_S,
                                                     margin=math.radians(TURN_MARGIN_DEG),
                                                     yaw_rate=math.radians(rate_dps) if rate_dps else 0.0))
        q = self._drone.get_ground_truth_pose()["rotation"]
        self._event("turn_end", yaw_deg=round(yaw_deg_of((q["w"], q["x"], q["y"], q["z"])), 1))

    async def hover(self) -> None:
        await self._cancel()
        self._event("hover")
        await (await self._drone.hover_async())

    async def return_and_land(self, home_ned: Sequence[float], speed_mps: float) -> None:
        await self._cancel()
        pose = self._drone.get_ground_truth_pose()["translation"]
        self._event("rtl", home=list(home_ned))
        await (await self._drone.move_to_position_async(home_ned[0], home_ned[1], pose["z"], velocity=speed_mps))
        # land_async descends slowly (~0.2 m/s); come down at a moderate rate first.
        hover_z = home_ned[2] - LAND_FROM_M
        if pose["z"] < hover_z:
            await (await self._drone.move_to_position_async(home_ned[0], home_ned[1], hover_z, velocity=DESCENT_MPS))
        await (await self._drone.land_async())
        self._drone.disarm()
        self._drone.disable_api_control()
        self._event("landed")
