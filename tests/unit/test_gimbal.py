"""The search camera's gimbal and the drone's camera set (vivid2.cameras), offline and on the runner's context."""
import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import NADIR_Q, FakeWorld
from vivid2.cameras import (GIMBAL_FOV_MIN, GIMBAL_SLEW_DPS, NAV, SEARCH, GimbalState, camera_infos, clamp_gimbal,
                            detect_depth_scale_rays, is_nadir, mount_pose, slew_time_s, tilt_from_nadir_deg, wrap_deg)
from vivid2.geo import rpy_to_quat
from vivid2.recorder import Recorder, RunLog
from vivid2.scenario import load_jsonc


def test_the_robot_has_one_search_camera_and_a_navigation_camera(root):
    robot = load_jsonc(root / "sim_config" / "robot_vivid_quad.jsonc")
    cams = camera_infos(robot, "Payload")
    assert list(cams) == ["Payload", "Front"]  # chase and preview cameras are for the viewer only
    pay, front = cams["Payload"], cams["Front"]
    assert (pay.role, pay.width, pay.height, pay.fov_deg, pay.types) == (SEARCH, 1920, 1440, 80.0, [0, 1, 3])
    assert (front.role, front.width, front.height, front.fov_deg, front.types) == (NAV, 640, 480, 90.0, [0, 1])
    assert pay.capture_step_ns == 250_000_000 and front.capture_step_ns == 150_000_000
    assert front.mount_rpy_deg == (0.0, -10.0, 0.0)
    spec = pay.spec()
    assert spec["gimbal"]["tilt_range_deg"] == [-90.0, 10.0] and spec["gimbal"]["fov_range_deg"] == [GIMBAL_FOV_MIN, 80.0]
    assert "gimbal" not in front.spec()


def test_gimbal_commands_are_clamped_and_cost_mission_time():
    s = GimbalState(fov_deg=80.0, base_fov_deg=80.0)
    assert s.is_default()
    t = clamp_gimbal(s, tilt_deg=40.0, pan_deg=270.0, fov_deg=2.0)
    assert (t.tilt_deg, t.pan_deg, t.fov_deg) == (10.0, -90.0, GIMBAL_FOV_MIN)
    assert clamp_gimbal(s, None, None, 500.0).fov_deg == 80.0  # no wider than the lens
    assert slew_time_s(s, s) == 0.0
    tilt = clamp_gimbal(s, -30.0, None, None)
    assert slew_time_s(s, tilt) == pytest.approx(60.0 / GIMBAL_SLEW_DPS + 0.2)
    zoom = clamp_gimbal(s, None, None, 40.0)
    assert slew_time_s(s, zoom) == pytest.approx(math.log(2.0) * 0.5 + 0.2)
    assert wrap_deg(190.0) == -170.0 and wrap_deg(-180.0) == 180.0


def test_the_default_gimbal_pose_is_the_configured_nadir_mount(root):
    from vivid2.runner import lifted_pose
    from vivid2.sim.scene_builder import camera_config

    robot = load_jsonc(root / "sim_config" / "robot_vivid_quad.jsonc")
    mount = lifted_pose(camera_config(robot, "Payload"), 0.0)
    pose = mount_pose(camera_infos(robot)["Payload"].mount_xyz, -90.0, 0.0)
    assert pose["translation"] == mount["translation"]
    assert all(pose["rotation"][k] == pytest.approx(mount["rotation"][k]) for k in "wxyz")
    assert tilt_from_nadir_deg(NADIR_Q) == pytest.approx(0.0, abs=1e-6) and is_nadir(NADIR_Q)
    assert tilt_from_nadir_deg(rpy_to_quat(0, math.radians(-30), 0)) == pytest.approx(60.0)


@pytest.mark.parametrize("metres_per_unit", [1.0, 0.001])
def test_depth_units_are_detected_from_an_oblique_photo(metres_per_unit):
    q = rpy_to_quat(0.0, math.radians(-20.0), 0.0)  # 20 degrees below the horizon, 15 m up
    w, h, fov, cam = 160, 120, 90.0, np.array([0.0, 0.0, -15.0])
    from vivid2.geo import pixel_ray_to_ground, quat_to_rot

    axis = quat_to_rot(q)[:, 0]
    raw = np.zeros((h, w), np.uint16)
    for v in range(h):
        for u in range(w):
            hit = pixel_ray_to_ground(u + 0.5, v + 0.5, cam, q, w, h, fov, 0.0)
            if hit is not None:
                raw[v, u] = min(65000, int(round(float(np.dot(hit - cam, axis)) / metres_per_unit)))
    assert detect_depth_scale_rays(raw, cam, q, fov, 0.0) == metres_per_unit
    assert detect_depth_scale_rays(np.zeros_like(raw), cam, rpy_to_quat(0, 0.3, 0), fov, 0.0) is None  # sky


def test_a_tilted_fake_camera_sees_ahead_and_reads_sky_as_no_depth(blocks_run, tmp_path):
    rec = Recorder(tmp_path / "w")
    t = blocks_run.resolved_targets()[0]
    world = FakeWorld(blocks_run, rec, width=320, height=240, targets=[t])
    world.pos = np.array([t["ned"][0] - 40.0, t["ned"][1], world.mission.ground_down_m - 15.0])
    assert not asyncio.run(world.get_frame()).gt  # straight down: 40 m behind it, not in view
    g = asyncio.run(world.point_camera(tilt_deg=-20.0))
    f = asyncio.run(world.get_frame())
    rec.close()
    assert g["tilt_deg"] == -20.0 and f.gimbal == g and [a.object_id for a in f.gt] == [t["id"]]
    assert (f.depth[0] == 0).all() and (f.depth[-1] > 0).all()  # the top row is sky
    frames = RunLog.load(tmp_path / "w")["frames"]
    assert "gimbal" not in frames[0] and frames[1]["gimbal"]["tilt_deg"] == -20.0


def test_the_runner_context_moves_the_gimbal_while_the_clock_runs(blocks_run, tmp_path, airsim_types, monkeypatch):
    import vivid2.runner as runner
    from vivid2.cameras import CameraInfo
    from vivid2.runner import SimMissionContext

    monkeypatch.setattr(runner, "palette", lambda: {})  # the segmentation palette comes from the client

    calls = []
    world = SimpleNamespace(get_sim_time=lambda: 5_000_000_000, client=SimpleNamespace())
    drone = SimpleNamespace(
        sensors_topic="/Sim/s/robots/Drone1/sensors",
        set_camera_pose=lambda cam, pose, wait: calls.append(("pose", cam, wait)),
        set_field_of_view=lambda cam, t, fov: calls.append(("fov", cam, t, fov)))
    rec = Recorder(tmp_path / "ctx")
    cams = {"Payload": CameraInfo("Payload", SEARCH, 1920, 1440, 80.0, 250_000_000, [0, 1, 3], (0.0, 0.0, 0.15))}
    ctx = SimMissionContext(blocks_run, world, drone, "Payload", rec, controller=None, fov_deg=80.0,
                            seg_ids={"Hiker1": 1}, cameras=cams)
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    ctx.sleep = fake_sleep
    out = asyncio.run(ctx.point_camera(tilt_deg=-60.0, pan_deg=45.0, fov_deg=20.0))
    assert out == {"tilt_deg": -60.0, "pan_deg": 45.0, "fov_deg": 20.0}
    assert calls[0] == ("pose", "Payload", True)
    assert [c[2] for c in calls[1:]] == [0, 1, 3] and all(c[3] == 20.0 for c in calls[1:])  # every image type
    assert slept == [pytest.approx(45.0 / 60.0 + math.log(4.0) * 0.5 + 0.2)]
    assert asyncio.run(ctx.point_camera(tilt_deg=-60.0)) == out and len(calls) == 4  # nothing moves: no command
    rec.close()
    ev = RunLog.load(tmp_path / "ctx")["events"]
    assert [e["kind"] for e in ev] == ["gimbal"] and ev[0]["pan_deg"] == 45.0
