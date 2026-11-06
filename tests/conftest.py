"""Shared fixtures, including a sim-free mission context for offline agent tests.

FakeWorld flies the commanded path kinematically and renders "frames" by
projecting target positions into a nadir camera, so the whole
agent -> recorder -> metrics -> scoring chain runs without Unreal.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from vivid2.geo import focal_px, quat_to_rot, rpy_to_quat, world_to_pixel
from vivid2.recorder import Recorder
from vivid2.scenario import Run, Scenario
from vivid2.stack.base import Controller, Frame, GTAnnotation, MissionContext, TargetReport
from vivid2.runner import mission_spec

ROOT = Path(__file__).resolve().parents[1]
NADIR_Q = rpy_to_quat(0.0, -math.pi / 2, 0.0)


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def blocks_run() -> Run:
    return Scenario.load(ROOT / "scenarios/forest_sar/blocks_hiker_01.jsonc").expand(seeds=[0])[0]


class FakeController(Controller):
    def __init__(self, world: "FakeWorld"):
        self.w = world

    async def takeoff(self, altitude_down_m: float) -> None:
        self.w.log("events", {"t": self.w.t_ns, "kind": "takeoff"})
        self.w.pos[2] = altitude_down_m
        self.w.tick()

    async def fly_path(self, waypoints, speed_mps) -> None:
        self.w.log("events", {"t": self.w.t_ns, "kind": "path_start", "waypoints": waypoints,
                                    "speed": speed_mps, "from": self.w.pos.tolist()})
        self.w.path = [np.asarray(p, float) for p in waypoints]
        self.w.speed = speed_mps

    def path_done(self) -> bool:
        return not self.w.path

    async def goto(self, ned, speed_mps) -> None:
        self.w.path = []  # like the real controller: a goto cancels the path in progress
        self.w.log("events", {"t": self.w.t_ns, "kind": "goto", "target": list(map(float, ned))})
        self.w.t_ns += int(np.linalg.norm(np.asarray(ned, float) - self.w.pos) / max(speed_mps, 0.1) * 1e9)
        self.w.pos = np.asarray(ned, float)
        self.w.tick()

    async def hover(self) -> None:
        self.w.path = []

    async def return_and_land(self, home_ned, speed_mps) -> None:
        self.w.log("events", {"t": self.w.t_ns, "kind": "rtl"})
        self.w.path = [np.array([home_ned[0], home_ned[1], self.w.pos[2]])]
        while self.w.path:
            self.w.advance(1.0)
        self.w.pos[2] = home_ned[2]
        self.w.tick()


class FakeWorld(MissionContext):
    """``surface(n, e)`` (optional): height of trees etc. above the flat ground, m; nadir depth
    images then show it (rays are cast into the height field)."""

    def __init__(self, run: Run, recorder: Recorder, width=960, height=720, fov=80.0, box_px=18, targets=None,
                 surface=None, surface_max_m=30.0):
        self.run = run
        self.surface = surface
        self.surface_max_m = surface_max_m
        self.recorder = recorder
        self.mission = mission_spec(run, camera={"width": width, "height": height, "fov_deg": fov})
        self.targets = targets if targets is not None else run.resolved_targets()
        self.pos = np.asarray(run.spec["dut_start"]["ned"], float)
        self.path: List[np.ndarray] = []
        self.speed = 5.0
        self.t_ns = 0
        self.W, self.H, self.fov, self.box = width, height, fov, box_px
        self.frame_id = 0
        self.reports: List[TargetReport] = []
        self.controller = FakeController(self)
        self.tick()

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        self.recorder.log(stream, record)

    def tick(self) -> None:
        self.recorder.log("trajectory", {"t": self.t_ns, "pos": self.pos.copy(), "q": [1, 0, 0, 0]})

    def advance(self, dt: float) -> None:
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
        self.t_ns += int(dt * 1e9)
        self.tick()

    async def get_frame(self) -> Optional[Frame]:
        self.advance(1.0)
        self.frame_id += 1
        gt = []
        for t in self.targets:
            proj = world_to_pixel(np.asarray(t["ned"]) + [0, 0, -0.85], self.pos, NADIR_Q, self.W, self.H, self.fov)
            if proj and 0 <= proj[0] < self.W and 0 <= proj[1] < self.H:
                u, v, _ = proj
                gt.append(GTAnnotation(t["id"], (u - self.box, v - self.box, u + self.box, v + self.box)))
        depth = self.render_depth()
        frame = Frame(self.t_ns, self.frame_id, np.zeros((self.H, self.W, 3), np.uint8), self.pos.copy(), NADIR_Q,
                      self.fov, depth=depth, gt=gt)
        self.recorder.log("frames", {"t": self.t_ns, "frame_id": self.frame_id, "cam_pos": self.pos.copy(),
                                     "cam_q": list(NADIR_Q), "width": self.W, "height": self.H, "fov_deg": self.fov,
                                     "gt": [{"object_id": g.object_id, "bbox": list(g.bbox)} for g in gt]})
        return frame

    def render_depth(self) -> np.ndarray:
        ground = self.mission.ground_down_m
        if self.surface is None:
            return np.full((self.H, self.W), ground - self.pos[2], np.float32)
        f = focal_px(self.W, self.fov)
        uu, vv = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        rays = quat_to_rot(NADIR_Q) @ np.stack([np.ones(uu.size), (uu.ravel() - self.W / 2) / f, (vv.ravel() - self.H / 2) / f])
        h = np.zeros(uu.size)
        todo = np.ones(uu.size, bool)
        for level in np.arange(self.surface_max_m, 0.0, -1.0):  # march down; the first surface reached is seen
            s = (ground - level - self.pos[2]) / rays[2, todo]
            hit = self.surface(self.pos[0] + s * rays[0, todo], self.pos[1] + s * rays[1, todo]) >= level
            idx = np.flatnonzero(todo)[hit]
            h[idx] = level
            todo[idx] = False
        return ((ground - h - self.pos[2]) / rays[2]).reshape(self.H, self.W).astype(np.float32)

    def pose(self):
        return self.pos.copy(), (1.0, 0.0, 0.0, 0.0)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)  # mission time; the drone flies on meanwhile

    def report_target(self, report: TargetReport) -> None:
        self.reports.append(report)
        self.recorder.log("reports", {"t": report.t_ns, "t_report": self.t_ns, "frame_id": report.frame_id, "ned": report.ned,
                                      "confidence": report.confidence, "source": report.source, "note": report.note})

    def should_stop(self) -> bool:
        return self.elapsed_s() >= self.mission.timeout_s

    def elapsed_s(self) -> float:
        return self.t_ns / 1e9


def write_meta(recorder: Recorder, run: Run, agent_desc: Dict[str, Any], targets, end_ns: int, end_reason="completed"):
    recorder.write_json("meta.json", {
        "run_id": run.run_id, "scenario_id": run.scenario_id, "seed": run.seed, "params": run.params,
        "mission": run.spec["mission"], "agent": agent_desc, "targets": targets,
        "search_area": run.spec["search_area"], "limits": run.spec["limits"], "success": run.spec.get("success", {}),
        "start_ned": run.spec["dut_start"]["ned"], "t0_ns": 0, "end_ns": end_ns, "end_reason": end_reason,
    })
