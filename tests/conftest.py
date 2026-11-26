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

from vivid2.cameras import GimbalState, clamp_gimbal, slew_time_s, wrap_deg
from vivid2.geo import focal_px, quat_to_rot, rpy_to_quat, world_to_pixel
from vivid2.recorder import Recorder
from vivid2.scenario import Run, Scenario
from vivid2.stack.base import Controller, Frame, GTAnnotation, MissionContext, TargetReport, damage_record
from vivid2.control import ControlStation
from vivid2.runner import mission_spec

ROOT = Path(__file__).resolve().parents[1]
NADIR_Q = rpy_to_quat(0.0, -math.pi / 2, 0.0)


@pytest.fixture
def airsim_types(monkeypatch):
    """Project AirSim's ImageType and Pose types, or stand-ins where the client is not installed (CI)."""
    try:
        import projectairsim.types  # noqa: F401
    except ImportError:
        import enum
        import sys
        import types

        mod = types.ModuleType("projectairsim.types")
        mod.ImageType = enum.IntEnum("ImageType", {"SCENE": 0, "DEPTH_PLANAR": 1, "DEPTH_PERSPECTIVE": 2,
                                                   "SEGMENTATION": 3})
        mod.Pose = mod.Quaternion = mod.Vector3 = dict
        drone = types.ModuleType("projectairsim.drone")
        drone.YawControlMode = enum.IntEnum("YawControlMode", {"MaxDegreeOfFreedom": 0, "ForwardOnly": 1})
        pkg = types.ModuleType("projectairsim")
        pkg.types, pkg.drone = mod, drone
        monkeypatch.setitem(sys.modules, "projectairsim", pkg)
        monkeypatch.setitem(sys.modules, "projectairsim.types", mod)
        monkeypatch.setitem(sys.modules, "projectairsim.drone", drone)


@pytest.fixture
def root() -> Path:
    return ROOT


def lfs_file(path: Path) -> Path:
    """``path``, or skip the test if it is a Git LFS pointer (CI checks out without LFS)."""
    if not path.exists() or path.read_bytes()[:40].startswith(b"version https://git-lfs"):
        pytest.skip(f"{path.name} is stored in Git LFS and not checked out")
    return path


@pytest.fixture(scope="session")
def damage_sim_config(tmp_path_factory) -> Path:
    """A sim_config folder with the test map's (TestYard's) damage ground truth, built from its layout
    (the committed .npz files are in Git LFS, which CI does not check out)."""
    import json as _json
    import shutil

    from vivid2.damage.layout import build_damage, write_build

    out = tmp_path_factory.mktemp("sim_config")
    for f in (ROOT / "sim_config").glob("*.jsonc"):
        shutil.copy2(f, out / f.name)
    layout = ROOT / "unreal/VIVID2Env/SourceData/TestYard/layout.json"
    write_build(build_damage(_json.loads(layout.read_text(encoding="utf-8"))), layout, out / "maps" / "TestYard", {},
                source_dir=tmp_path_factory.mktemp("source"))
    return out


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

    async def fly_path(self, waypoints, speed_mps, heading="hold") -> None:
        self.w.log("events", {"t": self.w.t_ns, "kind": "path_start", "waypoints": waypoints,
                                    "speed": speed_mps, "from": self.w.pos.tolist(),
                                    **({"heading": heading} if heading != "hold" else {})})
        self.w.path = [np.asarray(p, float) for p in waypoints]
        self.w.speed = speed_mps
        self.w.heading = heading
        if isinstance(heading, (int, float)):
            self.w.yaw_deg = wrap_deg(heading)

    def path_done(self) -> bool:
        return not self.w.path

    async def goto(self, ned, speed_mps, yaw_deg=None) -> None:
        self.w.path = []  # like the real controller: a goto cancels the path in progress
        self.w.log("events", {"t": self.w.t_ns, "kind": "goto", "target": list(map(float, ned)),
                              **({"yaw_deg": wrap_deg(yaw_deg)} if yaw_deg is not None else {})})
        if yaw_deg is not None:
            self.w.yaw_deg = wrap_deg(yaw_deg)
        self.w.t_ns += int(np.linalg.norm(np.asarray(ned, float) - self.w.pos) / max(speed_mps, 0.1) * 1e9)
        self.w.pos = np.asarray(ned, float)
        self.w.tick()

    async def turn_to(self, yaw_deg, rate_dps=None) -> None:
        self.w.path = []
        self.w.log("events", {"t": self.w.t_ns, "kind": "turn", "yaw_deg": wrap_deg(yaw_deg)})
        turn = abs(wrap_deg(yaw_deg - self.w.yaw_deg))
        self.w.t_ns += int(turn / (rate_dps or 90.0) * 1e9)
        self.w.yaw_deg = wrap_deg(yaw_deg)
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
    """``surface(n, e)`` (optional): height of trees etc. above the flat ground, m; depth images
    then show it (rays are cast into the height field).

    One camera, "Payload", on a gimbal (``point_camera``): straight down by default. The drone
    never turns, so the camera's direction is its tilt and pan.

    ``link``: a link to the control team (vivid2.control). Off by default: the offline tests and
    the goldens fly as before the link, home at the end.
    """

    def __init__(self, run: Run, recorder: Recorder, width=960, height=720, fov=80.0, box_px=18, targets=None,
                 surface=None, surface_max_m=30.0, damage=None, link=False):
        self.run = run
        self.surface = surface
        self.surface_max_m = surface_max_m
        self.recorder = recorder
        self.damage = damage  # a DamageTask (vivid2.damage.run): frames then carry its ground truth, like gt
        self.mission = mission_spec(run, camera={"width": width, "height": height, "fov_deg": fov},
                                    damage=damage.mission_view() if damage is not None else None)
        self.mission.control_link = bool(link)
        self.targets = targets if targets is not None else run.resolved_targets()
        self._control = ControlStation.for_run({**run.spec, "comms": {"control_link": True}}, self.targets,
                                              damage is not None, recorder.log) if link else None
        self.pos = np.asarray(run.spec["dut_start"]["ned"], float)
        self.path: List[np.ndarray] = []
        self.speed = 5.0
        self.t_ns = 0
        self.W, self.H, self.fov, self.box = width, height, fov, box_px
        self.frame_id = 0
        self.yaw_deg = 0.0  # the drone's heading: 0 north
        self.heading = "hold"  # how the current path is flown (see Controller.fly_path)
        self.gimbal = GimbalState(fov_deg=fov, base_fov_deg=fov)
        self.reports: List[TargetReport] = []
        self.controller = FakeController(self)
        self.tick()

    def cam_q(self):
        return rpy_to_quat(0.0, math.radians(self.gimbal.tilt_deg), math.radians(self.gimbal.pan_deg + self.yaw_deg))

    def body_q(self):
        return (1.0, 0.0, 0.0, 0.0) if self.yaw_deg == 0 else rpy_to_quat(0.0, 0.0, math.radians(self.yaw_deg))

    async def point_camera(self, tilt_deg=None, pan_deg=None, fov_deg=None):
        new = clamp_gimbal(self.gimbal, tilt_deg, pan_deg, fov_deg)
        took = slew_time_s(self.gimbal, new)
        self.gimbal = new
        if took:
            self.log("events", {"t": self.t_ns, "kind": "gimbal", **new.record(), "move_s": round(took, 3)})
            self.advance(took)
        return new.record()

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        self.recorder.log(stream, record)

    def tick(self) -> None:
        self.recorder.log("trajectory", {"t": self.t_ns, "pos": self.pos.copy(),
                                         "q": [1, 0, 0, 0] if self.yaw_deg == 0 else list(self.body_q())})

    def advance(self, dt: float) -> None:
        remaining = self.speed * dt
        while self.path and remaining > 0:
            d = self.path[0] - self.pos
            n = float(np.linalg.norm(d))
            if self.heading == "travel" and math.hypot(d[0], d[1]) > 1e-6:
                self.yaw_deg = math.degrees(math.atan2(d[1], d[0]))  # facing the direction of travel
            if n <= remaining:
                self.pos = self.path.pop(0).copy()
                remaining -= n
            else:
                self.pos = self.pos + d / n * remaining
                remaining = 0
        self.t_ns += int(dt * 1e9)
        self.tick()

    async def get_frame(self, camera: Optional[str] = None) -> Optional[Frame]:
        if camera not in (None, "Payload"):
            raise ValueError(f"FakeWorld has no camera {camera!r}")
        self.advance(1.0)
        self.frame_id += 1
        q, fov = self.cam_q(), self.gimbal.fov_deg
        gt = []
        for t in self.targets:
            proj = world_to_pixel(np.asarray(t["ned"]) + [0, 0, -0.85], self.pos, q, self.W, self.H, fov)
            if proj and 0 <= proj[0] < self.W and 0 <= proj[1] < self.H:
                u, v, _ = proj
                gt.append(GTAnnotation(t["id"], (u - self.box, v - self.box, u + self.box, v + self.box)))
        depth = self.render_depth(q, fov)
        gimbal = None if self.gimbal.is_default() else self.gimbal.record()
        frame = Frame(self.t_ns, self.frame_id, np.zeros((self.H, self.W, 3), np.uint8), self.pos.copy(), q,
                      fov, depth=depth, gt=gt, gimbal=gimbal)
        clues = {}
        if self.damage is not None:
            for c in self.damage.clues:  # a clue in the image shows (500 px)
                proj = world_to_pixel(c["ned"], self.pos, q, self.W, self.H, fov)
                if proj and 0 <= proj[0] < self.W and 0 <= proj[1] < self.H:
                    clues[c["id"]] = {"visible_px": 500, "contrast": 20.0}
            frame.damage_gt = self.damage.observe(frame, clues)
        self.recorder.log("frames", {"t": self.t_ns, "frame_id": self.frame_id, "cam_pos": self.pos.copy(),
                                     "cam_q": list(q), "width": self.W, "height": self.H, "fov_deg": fov,
                                     **({"gimbal": gimbal} if gimbal else {}),
                                     "gt": [{"object_id": g.object_id, "bbox": list(g.bbox)} for g in gt],
                                     **({"clues_visible": clues} if clues else {})})
        return frame

    def render_depth(self, cam_q=NADIR_Q, fov=None) -> np.ndarray:
        """Planar depth seen from the drone through ``cam_q``; 0 where a ray never meets the ground (sky)."""
        fov = self.fov if fov is None else fov
        ground = self.mission.ground_down_m
        if self.surface is None and tuple(cam_q) == tuple(NADIR_Q):
            return np.full((self.H, self.W), ground - self.pos[2], np.float32)
        f = focal_px(self.W, fov)
        uu, vv = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        rays = quat_to_rot(cam_q) @ np.stack([np.ones(uu.size), (uu.ravel() - self.W / 2) / f, (vv.ravel() - self.H / 2) / f])
        down = rays[2] > 1e-6
        h = np.zeros(uu.size)
        if self.surface is not None:
            todo = down.copy()
            for level in np.arange(self.surface_max_m, 0.0, -1.0):  # march down; the first surface reached is seen
                s = (ground - level - self.pos[2]) / rays[2, todo]
                hit = self.surface(self.pos[0] + s * rays[0, todo], self.pos[1] + s * rays[1, todo]) >= level
                idx = np.flatnonzero(todo)[hit]
                h[idx] = level
                todo[idx] = False
        depth = np.where(down, (ground - h - self.pos[2]) / np.where(down, rays[2], 1.0), 0.0)
        depth[depth > 1000.0] = 0.0  # beyond the camera's range
        return depth.reshape(self.H, self.W).astype(np.float32)

    def pose(self):
        return self.pos.copy(), self.body_q()

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)  # mission time; the drone flies on meanwhile

    def report_target(self, report: TargetReport) -> None:
        self.reports.append(report)
        record = {"t": report.t_ns, "t_report": self.t_ns, "frame_id": report.frame_id, "ned": report.ned,
                  "confidence": report.confidence, "source": report.source, "note": report.note}
        self.recorder.log("reports", record)
        if self._control is not None:
            self._control.report(record, cited=0 < report.frame_id <= self.frame_id)

    def report_damage(self, report) -> None:
        self.recorder.log("damage", {"t": report.t_ns, "t_report": self.t_ns, **damage_record(report)})

    def should_stop(self) -> bool:
        return self.elapsed_s() >= self.mission.timeout_s or (self._control is not None and self._control.mission_over())

    def elapsed_s(self) -> float:
        return self.t_ns / 1e9


def write_meta(recorder: Recorder, run: Run, agent_desc: Dict[str, Any], targets, end_ns: int, end_reason="completed",
               extra: Optional[Dict[str, Any]] = None, control_link: Optional[bool] = None):
    """``control_link``: recorded as the runner records it (None: left out, a run from before the link)."""
    recorder.write_json("meta.json", {**(extra or {}),
        **({"control_link": control_link} if control_link is not None else {}),
        "run_id": run.run_id, "scenario_id": run.scenario_id, "seed": run.seed, "params": run.params,
        "mission": run.spec["mission"], "agent": agent_desc, "targets": targets,
        "search_area": run.spec["search_area"], "limits": run.spec["limits"], "success": run.spec.get("success", {}),
        "start_ned": run.spec["dut_start"]["ned"], "t0_ns": 0, "end_ns": end_ns, "end_reason": end_reason,
    })
