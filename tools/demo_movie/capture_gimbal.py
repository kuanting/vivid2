"""Live capture for the film's search-camera shot: what the gimballed search camera can do, seen through
it. Drone1 of the demo scenario hovers 40 m from a hiker and the camera goes from straight down to
the hiker (tilt), sweeps (pan) and zooms in (80 -> 8 deg field of view). Every photo is a real
lockstep search-camera photo with its gimbal state and ground truth, as a stack would get them.

    python tools/demo_movie/capture_gimbal.py                 (simulator running, demo map open)

Writes <out>/NNN.jpg and <out>/photos.json ([{file, gimbal, gt, t}]).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def program(bearing_rel: float, depression: float):
    """(tilt, pan, fov) steps: nadir, tilt to the hiker, pan away and back, zoom in."""
    steps = [(-90.0, 0.0, 80.0)] * 3
    for k in range(1, 7):
        a = k / 6
        steps.append((-90.0 + (90.0 - depression) * a, bearing_rel * a, 80.0))
    for off in (40.0, 70.0, 40.0, 0.0):
        steps.append((-depression, bearing_rel + off, 80.0))
    for fov in (64.0, 50.0, 40.0, 32.0, 25.0, 19.0, 15.0, 12.0, 10.0, 8.0, 8.0):
        steps.append((-depression, bearing_rel, fov))
    return steps


async def capture(args) -> None:
    import cv2
    from projectairsim import Drone, ProjectAirSimClient

    from vivid2.cameras import camera_infos
    from vivid2.geo import yaw_deg_of
    from vivid2.recorder import Recorder
    from vivid2.runner import (SimMissionContext, allow_long_async_requests, apply_environment, lifted_pose,
                               load_world, serialize_requests, settle_exposure)
    from vivid2.scenario import Scenario
    from vivid2.sim.scene_builder import build_scene, camera_config, camera_settings, robot_config
    from vivid2.sim.segmentation import assign_ids
    from vivid2.stack.reference.controllers import AirSimController
    from vivid2.terrain import load_for_run

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim_cfg = str(ROOT / "sim_config")
    run = Scenario.load(ROOT / args.scenario).expand(seeds=[0])[0]
    run.spec.pop("fleet", None)
    terrain, _ = load_for_run(sim_cfg, run.spec)
    targets, distractors = run.resolved_targets(terrain), run.resolved_distractors(terrain)
    fires, start = run.resolved_fires(terrain), run.resolved_start(terrain)
    scene = build_scene(run, sim_cfg, targets, distractors, start, fires=fires)
    hik = next(t for t in targets if t["id"] == args.target)
    hn, he = float(hik["ned"][0]), float(hik["ned"][1])
    client = ProjectAirSimClient(address=args.address)
    client.connect()
    serialize_requests(client)
    allow_long_async_requests(client)
    try:
        world = load_world(client, scene, sim_cfg)
        seg_ids = assign_ids(world, targets, distractors, fires)
        drone = Drone(client, world, "Drone1")
        apply_environment(world, run.spec.get("environment", {}))
        infos = camera_infos(robot_config(scene, "Drone1"), "Payload")
        lift = float(run.spec["search_area"]["altitude_m"])
        views = {c: {"pose": lifted_pose(camera_config(scene, c), lift), "lift_m": lift} for c in infos}
        await settle_exposure(drone, list(infos), views)
        rec = Recorder(out / "log")
        ctrl = AirSimController(drone, world.get_sim_time, rec.log)
        cam = camera_settings(scene, "Payload")
        ctx = SimMissionContext(run, world, drone, "Payload", rec, ctrl, cam["fov-degrees"], "rgb", seg_ids=seg_ids,
                                target_ids=[t["id"] for t in targets], start_ned=start, terrain=terrain,
                                camera={"width": int(cam["width"]), "height": int(cam["height"]),
                                        "fov_deg": float(cam["fov-degrees"])}, cameras=infos,
                                ground_truth_to_stack=True)
        ctx.capture_step_ns = infos["Payload"].capture_step_ns
        ground0 = float(terrain.down_at(start[0], start[1]))
        await ctrl.takeoff(ground0 - 55.0)
        on, oe = hn - args.distance, he
        gd = float(terrain.down_at(on, oe))
        await asyncio.wait_for(ctrl.goto([on, oe, gd - 55.0], 6.0), 300)
        await asyncio.wait_for(ctrl.goto([on, oe, gd - args.height], 3.0), 120)
        await asyncio.sleep(3.0)
        here, _ = ctx.pose()
        q = drone.get_ground_truth_pose()["rotation"]
        yaw = yaw_deg_of((q["w"], q["x"], q["y"], q["z"]))
        bearing = math.degrees(math.atan2(he - here[1], hn - here[0]))
        hg = float(terrain.down_at(hn, he))
        depression = math.degrees(math.atan2(hg - 1.0 - here[2], math.hypot(hn - here[0], he - here[1])))
        rel = (bearing - yaw + 180.0) % 360.0 - 180.0
        photos = []
        for k, (tilt, pan, fov) in enumerate(program(rel, depression)):
            g = await ctx.point_camera(tilt_deg=tilt, pan_deg=pan, fov_deg=fov)
            f = await ctx.get_frame()
            name = f"{k:03d}.jpg"
            cv2.imwrite(str(out / name), f.rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
            gts = [{"object_id": a.object_id, "bbox": list(a.bbox), "visible_px": a.visible_px} for a in f.gt]
            photos.append({"file": name, "gimbal": {"tilt_deg": tilt, "pan_deg": pan, "fov_deg": fov},
                           "gimbal_reported": g, "gt": gts, "t": ctx.elapsed_s()})
            px = max([a["visible_px"] for a in gts if a["object_id"] == args.target] or [0])
            print(f"  {k:2d} tilt {tilt:6.1f} pan {pan:6.1f} fov {fov:5.1f}  {args.target} {px} px", flush=True)
        (out / "photos.json").write_text(json.dumps({"target": args.target, "observer": list(map(float, here)),
                                                     "yaw_deg": yaw, "photos": photos}, indent=1))
        await ctx.point_camera(tilt_deg=-90.0, pan_deg=0.0, fov_deg=80.0)
    finally:
        client.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="scenarios/demo/golden_hour_01.jsonc")
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--target", default="Hiker2")
    p.add_argument("--distance", type=float, default=40.0, help="metres south of the hiker")
    p.add_argument("--height", type=float, default=30.0, help="metres above ground")
    p.add_argument("--out", default=str(ROOT / "runs/_film/golden_hour/gimbal"))
    asyncio.run(capture(p.parse_args()))


if __name__ == "__main__":
    main()
