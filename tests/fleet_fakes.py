"""A sim-free team of drones on one discrete-event clock, for offline team-agent tests.

Every drone's agent runs as its own coroutine. Whenever all of them wait on the clock (a photo,
a sleep, a move in progress), time jumps to the earliest wake-up and every drone flies on along
its commanded path, so the drones share one mission time as they do in the simulator.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from conftest import NADIR_Q
from vivid2.geo import world_to_pixel
from vivid2.radio import Radio, RadioProfile
from vivid2.recorder import Recorder
from vivid2.runner import mission_spec, score_run_dir
from vivid2.scenario import Run
from vivid2.stack.base import (Controller, DistributedTeam, FleetAgent, FleetContext, Frame, GTAnnotation,
                               MissionContext, TargetReport)

SETTLE_TURNS = 60  # event-loop turns for every drone to reach its next wait on the clock
SAMPLE_S = 0.5  # trajectory samples (and kinematics steps) at most this far apart
CAPTURE_S = 0.25  # a photo's mission time, as in lockstep
CLIMB_MPS = 3.0


class _Drone:
    def __init__(self, ned):
        self.pos = np.asarray(ned, float)
        self.path: List[np.ndarray] = []
        self.speed = 5.0

    def fly(self, dt: float) -> None:
        remaining = self.speed * dt
        while self.path and remaining > 0:
            d = self.path[0] - self.pos
            n = float(np.linalg.norm(d))
            if n <= remaining:
                self.pos = self.path.pop(0).copy()
                remaining -= n
            else:
                self.pos = self.pos + d / n * remaining
                remaining = 0


class FakeTeamController(Controller):
    def __init__(self, ctx: "FakeDroneContext"):
        self.ctx = ctx
        self.d = ctx.drone

    async def _arrive(self) -> None:
        while self.d.path:
            await self.ctx.fleet.wait(SAMPLE_S)

    async def takeoff(self, altitude_down_m: float) -> None:
        self.ctx.log("events", {"t": self.ctx.fleet.t_ns, "kind": "takeoff"})
        self.d.path, self.d.speed = [np.array([self.d.pos[0], self.d.pos[1], altitude_down_m])], CLIMB_MPS
        await self._arrive()

    async def fly_path(self, waypoints, speed_mps, heading="hold") -> None:
        self.ctx.log("events", {"t": self.ctx.fleet.t_ns, "kind": "path_start", "waypoints": waypoints,
                                "speed": speed_mps, "from": self.d.pos.tolist()})
        self.d.path, self.d.speed = [np.asarray(p, float) for p in waypoints], speed_mps

    def path_done(self) -> bool:
        return not self.d.path

    async def goto(self, ned, speed_mps, yaw_deg=None) -> None:
        self.ctx.log("events", {"t": self.ctx.fleet.t_ns, "kind": "goto", "target": list(map(float, ned))})
        self.d.path, self.d.speed = [np.asarray(ned, float)], speed_mps
        await self._arrive()

    async def hover(self) -> None:
        self.d.path = []

    async def return_and_land(self, home_ned, speed_mps) -> None:
        self.ctx.log("events", {"t": self.ctx.fleet.t_ns, "kind": "rtl", "home": list(home_ned)})
        self.d.path, self.d.speed = [np.array([home_ned[0], home_ned[1], self.d.pos[2]])], speed_mps
        await self._arrive()
        self.d.path, self.d.speed = [np.asarray(home_ned, float)], CLIMB_MPS
        await self._arrive()
        self.ctx.log("events", {"t": self.ctx.fleet.t_ns, "kind": "landed"})


class FakeDroneContext(MissionContext):
    def __init__(self, fleet: "FakeFleet", drone_id: str, pad, mission):
        self.fleet = fleet
        self.drone_id = drone_id
        self.drone = _Drone(pad)
        self.mission = mission
        self.controller = FakeTeamController(self)
        self.radio = None

    async def get_frame(self, camera: Optional[str] = None) -> Optional[Frame]:
        if camera not in (None, "Payload"):
            raise ValueError(f"the fake fleet's drones have no camera {camera!r}")
        f = self.fleet
        await f.wait(CAPTURE_S)
        f.frame_id += 1
        f.frame_owner[f.frame_id] = self.drone_id
        gt = []
        for t in f.targets:
            proj = world_to_pixel(np.asarray(t["ned"]) + [0, 0, -0.85], self.drone.pos, NADIR_Q, f.W, f.H, f.fov)
            if proj and 0 <= proj[0] < f.W and 0 <= proj[1] < f.H:
                u, v, _ = proj
                gt.append(GTAnnotation(t["id"], (u - f.box, v - f.box, u + f.box, v + f.box)))
        depth = np.full((f.H, f.W), self.mission.ground_down_m - self.drone.pos[2], np.float32)
        frame = Frame(f.t_ns, f.frame_id, np.zeros((f.H, f.W, 3), np.uint8), self.drone.pos.copy(), NADIR_Q, f.fov,
                      depth=depth, gt=gt)
        f.rec.log("frames", {"t": f.t_ns, "frame_id": f.frame_id, "drone": self.drone_id, "cam_pos": self.drone.pos.copy(),
                             "cam_q": list(NADIR_Q), "width": f.W, "height": f.H, "fov_deg": f.fov,
                             "gt": [{"object_id": g.object_id, "bbox": list(g.bbox)} for g in gt]})
        return frame

    def pose(self):
        return self.drone.pos.copy(), (1.0, 0.0, 0.0, 0.0)

    async def sleep(self, seconds: float) -> None:
        await self.fleet.wait(seconds)

    async def think(self, fn, *args, **kwargs):  # instant: the offline clock only moves when every drone waits
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return fn(*args, **kwargs)

    def report_target(self, report: TargetReport) -> None:
        self.fleet.rec.log("reports", {"t": report.t_ns, "t_report": self.fleet.t_ns, "frame_id": report.frame_id,
                                       "drone": self.drone_id, "ned": report.ned, "confidence": report.confidence,
                                       "source": report.source, "note": report.note})

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        self.fleet.rec.log(stream, {**record, "drone": self.drone_id})

    def should_stop(self) -> bool:
        return self.elapsed_s() >= self.mission.timeout_s

    def elapsed_s(self) -> float:
        return self.fleet.t_ns / 1e9

    def sim_time_or_last(self) -> int:
        return self.fleet.t_ns


class FakeFleetContext(FleetContext):
    def __init__(self, fleet: "FakeFleet"):
        self.fleet = fleet
        self.mission = fleet.team_mission
        self.drones = fleet.contexts

    def should_stop(self) -> bool:
        return self.elapsed_s() >= self.mission.timeout_s

    def elapsed_s(self) -> float:
        return self.fleet.t_ns / 1e9

    def t_ns(self) -> int:
        return self.fleet.t_ns

    async def sleep(self, seconds: float) -> None:
        await self.fleet.wait(seconds)

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        self.fleet.rec.log(stream, record)

    def report_target(self, report: TargetReport) -> None:
        self.drones[self.fleet.frame_owner.get(report.frame_id, next(iter(self.drones)))].report_target(report)


class FakeFleet:
    def __init__(self, run: Run, rec: Recorder, team_mode: str, width=320, height=240, fov=80.0, box_px=6,
                 targets=None):
        self.run, self.rec = run, rec
        self.W, self.H, self.fov, self.box = width, height, fov, box_px
        self.targets = targets if targets is not None else run.resolved_targets()
        self.t_ns = 0
        self.frame_id = 0
        self.frame_owner: Dict[int, str] = {}
        self._waiters: list = []
        self._seq = itertools.count()
        self.fleet = run.resolved_fleet()
        self.ids = [d["id"] for d in self.fleet]
        centre = run.resolved_start()
        camera = {"width": width, "height": height, "fov_deg": fov}
        info = [{"id": d["id"], "start_ned": list(d["ned"])} for d in self.fleet]
        self.team_mission = mission_spec(run, centre, None, camera, datum_ned=centre, team=info, team_mode=team_mode)
        self.contexts = {d["id"]: FakeDroneContext(self, d["id"], d["ned"], mission_spec(
            run, d["ned"], None, camera, datum_ned=centre, drone_id=d["id"], team=info, team_mode=team_mode))
            for d in self.fleet}
        self._sample()

    async def wait(self, dt_s: float) -> None:
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self.t_ns + int(round(max(dt_s, 0.0) * 1e9)), next(self._seq), fut))
        await fut

    def _sample(self) -> None:
        for i, c in self.contexts.items():
            self.rec.log("trajectory", {"t": self.t_ns, "pos": c.drone.pos.copy(), "q": [1, 0, 0, 0], "drone": i})

    def _advance_to(self, t_ns: int) -> None:
        while self.t_ns < t_ns:
            step = min(int(SAMPLE_S * 1e9), t_ns - self.t_ns)
            for c in self.contexts.values():
                c.drone.fly(step / 1e9)
            self.t_ns += step
            self._sample()
        while self._waiters and self._waiters[0][0] <= self.t_ns:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)

    async def drive(self, coros, hard_s: float) -> bool:
        """Run the agents to the end; True if stopped at ``hard_s`` of mission time."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        try:
            while True:
                for _ in range(SETTLE_TURNS):
                    await asyncio.sleep(0)
                if all(t.done() for t in tasks):
                    for t in tasks:
                        t.result()
                    return False
                if not self._waiters:
                    raise RuntimeError("the agents wait on something other than the clock")
                if self._waiters[0][0] / 1e9 > hard_s:
                    return True
                self._advance_to(self._waiters[0][0])
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except BaseException:
                    pass


def fly_team(tmp_path: Path, run: Run, agent, targets=None) -> Dict[str, Any]:
    """Fly a FleetAgent or DistributedTeam in a FakeFleet; returns the scored result (and leaves the run dir)."""
    run_dir = Path(tmp_path) / agent.name / run.run_id
    rec = Recorder(run_dir)
    fleet = FakeFleet(run, rec, agent.team_mode, targets=targets)
    if isinstance(agent, FleetAgent):
        coros = [agent.run(FakeFleetContext(fleet))]
    else:
        assert isinstance(agent, DistributedTeam)
        radio = Radio(RadioProfile.from_spec(run.spec.get("comms")), fleet.ids, clock=lambda: fleet.t_ns / 1e9,
                      t_ns=lambda: fleet.t_ns, position=lambda i: fleet.contexts[i].drone.pos, log=rec.log,
                      seed=run.seed)
        for i, c in fleet.contexts.items():
            c.radio = radio.port(i, c.sleep)
        coros = [a.run(fleet.contexts[i]) for i, a in zip(fleet.ids, agent.agents)]
    timeout = run.spec["limits"]["timeout_s"]
    hard = asyncio.run(fleet.drive(coros, timeout * 2 + 180.0))
    end = "hard_timeout" if hard else ("timeout" if fleet.t_ns / 1e9 >= timeout else "completed")
    rec.write_json("meta.json", {
        "run_id": run.run_id, "scenario_id": run.scenario_id, "seed": run.seed, "params": run.params,
        "mission": run.spec["mission"], "agent": agent.describe(), "targets": fleet.targets,
        "search_area": run.spec["search_area"], "limits": run.spec["limits"], "success": run.spec.get("success", {}),
        "start_ned": fleet.fleet[0]["ned"], "t0_ns": 0, "end_ns": fleet.t_ns, "end_reason": end,
        "fleet": [{"id": d["id"], "start_ned": d["ned"], "yaw_deg": d["yaw_deg"]} for d in fleet.fleet],
        "team": {"mode": agent.team_mode, "size": len(fleet.ids)},
        "mission_spec": fleet.team_mission.record(),
    })
    rec.close()
    return score_run_dir(run_dir)
