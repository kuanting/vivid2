"""Controller layer backed by the Project AirSim Drone API.

Works unchanged with both the built-in simple_flight controller and PX4
(``controller.type: px4-api`` in the robot config): Project AirSim forwards
these move commands to PX4 over MAVLink.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence

from vivid2.stack.base import Controller

DESCENT_MPS = 3.0
LAND_FROM_M = 2.0  # height above the start point where the final, slow landing begins


class AirSimController(Controller):
    name = "airsim_api"

    def __init__(self, drone, clock_ns: Callable[[], int], log: Callable[[str, Dict[str, Any]], None]):
        self.drone = drone
        self._clock = clock_ns
        self._log = log
        self._task: Optional[asyncio.Task] = None

    def _event(self, kind: str, **kw) -> None:
        self._log("events", {"t": self._clock(), "kind": kind, **kw})

    async def _cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self.drone.cancel_last_task()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        self._task = None

    async def takeoff(self, altitude_down_m: float) -> None:
        self.drone.enable_api_control()
        self.drone.arm()
        self._event("takeoff")
        await (await self.drone.takeoff_async())
        pose = self.drone.get_ground_truth_pose()
        t = pose["translation"]
        await (await self.drone.move_to_position_async(t["x"], t["y"], altitude_down_m, velocity=3.0))

    async def fly_path(self, waypoints: List[List[float]], speed_mps: float) -> None:
        await self._cancel()
        here = self.drone.get_ground_truth_pose()["translation"]
        self._event("path_start", waypoints=waypoints, speed=speed_mps, **{"from": [here["x"], here["y"], here["z"]]})
        self._task = await self.drone.move_on_path_async(path=waypoints, velocity=speed_mps)

    def path_done(self) -> bool:
        return self._task is None or self._task.done()

    async def goto(self, ned: Sequence[float], speed_mps: float) -> None:
        await self._cancel()
        self._event("goto", target=list(ned), speed=speed_mps)
        await (await self.drone.move_to_position_async(ned[0], ned[1], ned[2], velocity=speed_mps))

    async def hover(self) -> None:
        await self._cancel()
        self._event("hover")
        await (await self.drone.hover_async())

    async def return_and_land(self, home_ned: Sequence[float], speed_mps: float) -> None:
        await self._cancel()
        pose = self.drone.get_ground_truth_pose()["translation"]
        self._event("rtl", home=list(home_ned))
        await (await self.drone.move_to_position_async(home_ned[0], home_ned[1], pose["z"], velocity=speed_mps))
        # land_async descends slowly (~0.2 m/s); come down at a moderate rate first.
        hover_z = home_ned[2] - LAND_FROM_M
        if pose["z"] < hover_z:
            await (await self.drone.move_to_position_async(home_ned[0], home_ned[1], hover_z, velocity=DESCENT_MPS))
        await (await self.drone.land_async())
        self.drone.disarm()
        self.drone.disable_api_control()
        self._event("landed")
