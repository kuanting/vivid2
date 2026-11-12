"""Live check of the camera features on the packaged build: lockstep cost per camera, the gimbal and zoom,
ground truth in tilted and zoomed photos, exposure after re-pointing, and turning the drone.

Runs through the real runner context (``SimMissionContext``), so it exercises the code a stack uses.
The simulator must already be running on ForestSAR:

    python tools/probe_cameras.py --out runs/_probe_cameras
    python tools/probe_cameras.py --out runs/_probe_cameras_team --team-only   # 4 drones x 2 cameras per step

Writes summary.json (and the recorded run, frames banked) to --out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def _hiker(frame, tid):
    return next((g for g in frame.gt if g.object_id == tid), None) if frame is not None else None


def _iou(a, b):
    if a is None or b is None:
        return None
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return round(inter / area, 3) if area > 0 else None


def _yaw(drone) -> float:
    from vivid2.geo import yaw_deg_of

    q = drone.get_ground_truth_pose()["rotation"]
    return yaw_deg_of((q["w"], q["x"], q["y"], q["z"]))


async def single(args, out: Path) -> Dict[str, Any]:
    from projectairsim import Drone, ProjectAirSimClient

    from vivid2.cameras import camera_infos
    from vivid2.recorder import Recorder, RunLog
    from vivid2.runner import (SimMissionContext, allow_long_async_requests, apply_environment, load_world,
                               lifted_pose, serialize_requests, settle_exposure)
    from vivid2.scenario import Scenario
    from vivid2.sim.scene_builder import build_scene, camera_config, camera_settings, robot_config
    from vivid2.sim.segmentation import assign_ids
    from vivid2.stack.reference.controllers import AirSimController
    from vivid2.terrain import load_for_run

    sim_cfg = str(ROOT / "sim_config")
    run = Scenario.load(ROOT / args.scenario).expand(seeds=[0])[0]
    terrain, _ = load_for_run(sim_cfg, run.spec)
    targets, distractors = run.resolved_targets(terrain), run.resolved_distractors(terrain)
    fires, start = run.resolved_fires(terrain), run.resolved_start(terrain)
    scene = build_scene(run, sim_cfg, targets, distractors, start, fires=fires)
    client = ProjectAirSimClient(address=args.address)
    client.connect()
    serialize_requests(client)
    allow_long_async_requests(client)
    s: Dict[str, Any] = {"scenario": run.scenario_id}
    try:
        world = load_world(client, scene, sim_cfg)
        seg_ids = assign_ids(world, targets, distractors, fires)
        drone = Drone(client, world, "Drone1")
        apply_environment(world, run.spec.get("environment", {}))
        infos = camera_infos(robot_config(scene, "Drone1"), "Payload")
        lift = float(run.spec["search_area"]["altitude_m"])
        views = {c: {"pose": lifted_pose(camera_config(scene, c), lift), "lift_m": lift} for c in infos}
        s["exposure"] = await settle_exposure(drone, list(infos), views)
        rec = Recorder(out / "run")
        ctrl = AirSimController(drone, world.get_sim_time, rec.log)
        cam = camera_settings(scene, "Payload")
        ctx = SimMissionContext(run, world, drone, "Payload", rec, ctrl, cam["fov-degrees"], "rgb", seg_ids=seg_ids,
                                target_ids=[t["id"] for t in targets], start_ned=start, terrain=terrain,
                                camera={"width": int(cam["width"]), "height": int(cam["height"]),
                                        "fov_deg": float(cam["fov-degrees"])}, cameras=infos,
                                ground_truth_to_stack=True)
        ctx.capture_step_ns = infos["Payload"].capture_step_ns
        ground = ctx.mission.ground_down_m
        await ctrl.takeoff(ground - 60.0)
        for _ in range(60):  # make sure the climb has finished
            z = drone.get_ground_truth_pose()["translation"]["z"]
            if abs(z - (ground - 60.0)) < 2.0:
                break
            await asyncio.sleep(1.0)
        s["after_takeoff_z"] = round(drone.get_ground_truth_pose()["translation"]["z"], 1)
        print("after takeoff z", s["after_takeoff_z"], "target", round(ground - 60.0, 1), flush=True)

        # a) lockstep cost per camera set
        cost: Dict[str, Any] = {}
        for combo in (["Payload"], ["Front"], ["Payload", "Front"]):
            walls = []
            for _ in range(args.batches):
                w0 = time.perf_counter()
                got = await ctx.get_frames(combo)
                walls.append(time.perf_counter() - w0)
                assert all(got[c] is not None for c in combo), combo
                await asyncio.sleep(0.3)
            cost["+".join(combo)] = {"wall_s_mean": round(statistics.mean(walls), 3), "wall_s_max": round(max(walls), 3)}
            print("lockstep", combo, cost["+".join(combo)], flush=True)
        front = await ctx.get_frame("Front")
        cost["front_sky_fraction"] = round(float((front.depth == 0).mean()), 3) if front.depth is not None else None
        cost["front_depth_median_m"] = round(float(np.median(front.depth[front.depth > 0])), 1) if front.depth is not None else None
        s["lockstep"] = cost

        # c) gimbal sweep, looking at the hiker from 40 m south and 30 m up
        hik = targets[0]
        hn, he = float(hik["ned"][0]), float(hik["ned"][1])
        hg = float(terrain.down_at(hn, he))
        obs = [hn - 40.0, he, hg - 30.0]
        await asyncio.wait_for(ctrl.goto(obs, 6.0), 120)
        await asyncio.sleep(2.0)
        here, _ = ctx.pose()
        yaw = _yaw(drone)
        bearing = math.degrees(math.atan2(he - here[1], hn - here[0]))  # to the hiker
        s["observer"] = {"pos": [round(v, 1) for v in here], "yaw": round(yaw, 1), "bearing_to_hiker": round(bearing, 1)}
        print("observer", s["observer"], flush=True)
        ahead = bearing - yaw  # gimbal pan that points at the hiker
        sweep = []
        for tilt, pan, fov in ((-90, 0, None), (-60, ahead, None), (-37, ahead, None), (-10, ahead, None),
                               (-37, ahead + 90, None), (-37, ahead, 40), (-37, ahead, 20), (-37, ahead, 10),
                               (-37, ahead, 80)):
            t0 = ctx.elapsed_s()
            g = await ctx.point_camera(tilt_deg=tilt, pan_deg=pan, fov_deg=fov)
            moved = ctx.elapsed_s() - t0
            f = await ctx.get_frame()
            h = _hiker(f, hik["id"])
            row = {"gimbal": g, "move_mission_s": round(moved, 2), "frame_id": f.frame_id,
                   "sky_fraction": round(float((f.depth == 0).mean()), 3) if f.depth is not None else None,
                   "grey": round(float(f.rgb.mean()), 1),
                   "hiker": None if h is None else {"visible_px": h.visible_px, "contrast": h.contrast,
                                                    "bbox": [round(v) for v in h.bbox],
                                                    "bbox_vis": None if h.bbox_vis is None else [round(v) for v in h.bbox_vis],
                                                    "iou_annotation_vs_seg": _iou(h.bbox, h.bbox_vis)}}
            sweep.append(row)
            print("gimbal", row, flush=True)
        s["gimbal"] = sweep

        # e) exposure after re-pointing from straight down to the horizon
        await ctx.point_camera(tilt_deg=-90, pan_deg=0, fov_deg=80)
        await ctx.get_frame()
        await ctx.point_camera(tilt_deg=-3)
        greys = []
        for _ in range(8):
            f = await ctx.get_frame()
            greys.append(round(float(f.rgb.mean()), 1))
            await asyncio.sleep(0.3)
        s["exposure_after_tilt_to_horizon"] = greys
        print("exposure", greys, flush=True)
        await ctx.point_camera(tilt_deg=-90)

        # g) heading
        heading: Dict[str, Any] = {"turns": []}
        for target in (90.0, 180.0, -90.0, 0.0):
            t0 = ctx.elapsed_s()
            print("turn to", target, flush=True)
            try:
                await asyncio.wait_for(ctrl.turn_to(target), 60)
            except asyncio.TimeoutError:
                heading["turns"].append({"target": target, "timeout": True})
                continue
            heading["turns"].append({"target": target, "reached": round(_yaw(drone), 1),
                                     "mission_s": round(ctx.elapsed_s() - t0, 1)})
        print("turns", heading["turns"], flush=True)
        p, _ = ctx.pose()
        await asyncio.wait_for(ctrl.goto([p[0], p[1] + 20.0, p[2]], 4.0, yaw_deg=45.0), 60)
        heading["goto_yaw45"] = round(_yaw(drone), 1)
        print("goto yaw 45 ->", heading["goto_yaw45"], flush=True)
        # a square, facing the direction of travel; then a climb first (must not spin)
        p, _ = ctx.pose()
        square = [[p[0] + 30, p[1], p[2]], [p[0] + 30, p[1] + 30, p[2]], [p[0], p[1] + 30, p[2]], [p[0], p[1], p[2]]]
        bearings = [0.0, 90.0, 180.0, -90.0]
        await ctrl.fly_path(square, 4.0, heading="travel")
        errs = []
        w_end = time.monotonic() + 120
        while not ctrl.path_done() and time.monotonic() < w_end:
            pos, _ = ctx.pose()
            await asyncio.sleep(0.5)
            errs.append((round(_yaw(drone), 1), [round(v, 1) for v in pos[:2]]))
        heading["square_samples"] = errs[::4]
        heading["square_done"] = ctrl.path_done()
        print("square", heading["square_done"], errs[::6], flush=True)
        p, _ = ctx.pose()
        y0 = _yaw(drone)
        await ctrl.fly_path([[p[0], p[1], p[2] - 10.0], [p[0] + 20.0, p[1], p[2] - 10.0]], 4.0, heading="travel")
        ys = []
        w_end = time.monotonic() + 60
        while not ctrl.path_done() and time.monotonic() < w_end:
            await asyncio.sleep(0.3)
            ys.append(round(_yaw(drone), 1))
        heading["climb_then_north"] = {"start_yaw": round(y0, 1), "yaws": ys}
        s["heading"] = heading
        print("heading", json.dumps(heading)[:600], flush=True)

        await ctrl.return_and_land(start, 6.0)
        rec.close()
        frames = RunLog.load(out / "run")["frames"]
        steps: Dict[str, List[float]] = {}
        for f in frames:
            steps.setdefault(f.get("camera", "Payload"), []).append(f["capture_s"])
        s["capture_s_by_camera"] = {k: sorted(set(v)) for k, v in steps.items()}
        s["front_records_have_gt"] = any("gt" in f for f in frames if f.get("camera") == "Front")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
    return s


async def team(args, out: Path) -> Dict[str, Any]:
    from projectairsim import Drone, ProjectAirSimClient

    from probe_fleet import fleet_scene
    from vivid2.runner import allow_long_async_requests, apply_environment, load_world, serialize_requests
    from vivid2.scenario import Scenario
    from vivid2.stack.reference.controllers import AirSimController
    from vivid2.team_runtime import lockstep_fetch
    from vivid2.terrain import load_for_run

    sim_cfg = str(ROOT / "sim_config")
    run = Scenario.load(ROOT / args.scenario).expand(seeds=[0])[0]
    terrain, _ = load_for_run(sim_cfg, run.spec)
    scene, pads = fleet_scene(run, sim_cfg, terrain, 4, "drone1")
    client = ProjectAirSimClient(address=args.address)
    client.connect()
    serialize_requests(client)
    allow_long_async_requests(client)
    try:
        world = load_world(client, scene, sim_cfg)
        drones = [Drone(client, world, f"Drone{i + 1}") for i in range(4)]
        apply_environment(world, run.spec.get("environment", {}))
        ctrls = [AirSimController(d, world.get_sim_time, lambda s_, r: None) for d in drones]
        ground = float(terrain.down_at(*run.spec["dut_start"]["ned"][:2]))
        await asyncio.gather(*(c.takeoff(ground - 60.0) for c in ctrls))
        await asyncio.sleep(2.0)
        reqs = [(f"{d.sensors_topic}/{cam}", types) for d in drones
                for cam, types in (("Payload", [0, 1, 3]), ("Front", [0, 1]))]
        walls, stamps_ok, errors = [], 0, []
        for _ in range(args.batches):
            w0 = time.perf_counter()
            await asyncio.to_thread(world.pause)
            t_req = int(await asyncio.to_thread(world.get_sim_time))
            try:
                t_fetched, imgs = await lockstep_fetch(world, reqs, int(0.25e9), 0.02, 15.0 + 5.0 * 7)
                stamps_ok += all(t_req <= int(i[0]["time_stamp"]) <= t_fetched for i in imgs)
            except Exception as e:
                errors.append(repr(e))
            finally:
                await asyncio.to_thread(world.resume)
            walls.append(time.perf_counter() - w0)
            await asyncio.sleep(0.75)
        out_ = {"photos_per_batch": len(reqs), "batches": args.batches, "stamps_inside_step": stamps_ok,
                "wall_s_mean": round(statistics.mean(walls), 3), "wall_s_max": round(max(walls), 3), "errors": errors[:3]}
        print("team", out_, flush=True)
        await asyncio.gather(*(c.return_and_land(p, 5.0) for c, p in zip(ctrls, pads)))
        return out_
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


async def main(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"team": await team(args, out)} if args.team_only else {"single": await single(args, out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("wrote", out / "summary.json")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default="scenarios/forest_sar/forest_hiker_01.jsonc")
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--team-only", action="store_true", help="only the 4 drones x 2 cameras batch")
    p.add_argument("--out", default="runs/_probe_cameras")
    sys.exit(asyncio.run(main(p.parse_args())))
