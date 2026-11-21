"""Live capture for the film's ground-truth shot (S09): one search-camera view as RGB, depth,
segmentation and ground-truth boxes, all from the same instant. Run records don't keep segmentation
images, so this flies Drone1 of the demo scenario to a viewpoint and asks the simulator for them.

    python tools/demo_movie/capture_quad.py --out runs/_film/golden_hour/quad   (simulator running)

Writes rgb.png, depth_mm.png (16-bit), seg.png (the simulator's colour palette) and gt.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


async def capture(args) -> None:
    import cv2
    from projectairsim import Drone, ProjectAirSimClient
    from projectairsim.types import ImageType
    from projectairsim.utils import unpack_image

    from vivid2.cameras import camera_infos
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
    run.spec.pop("fleet", None)  # one drone is enough
    terrain, _ = load_for_run(sim_cfg, run.spec)
    targets, distractors = run.resolved_targets(terrain), run.resolved_distractors(terrain)
    fires, start = run.resolved_fires(terrain), run.resolved_start(terrain)
    scene = build_scene(run, sim_cfg, targets, distractors, start, fires=fires)
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
        n, e = args.at
        gd = float(terrain.down_at(n, e))
        await ctrl.takeoff(float(terrain.down_at(start[0], start[1])) - args.height)
        await asyncio.wait_for(ctrl.goto([n, e, gd - args.height], 6.0), 300)
        await asyncio.sleep(3.0)
        frame = await ctx.get_frame()
        imgs = drone.get_images("Payload", [ImageType.SCENE, ImageType.DEPTH_PLANAR, ImageType.SEGMENTATION])
        rgb = unpack_image(imgs[ImageType.SCENE])
        seg = unpack_image(imgs[ImageType.SEGMENTATION])
        cv2.imwrite(str(out / "rgb.png"), frame.rgb if frame.rgb is not None else rgb)
        cv2.imwrite(str(out / "seg.png"), seg)
        if frame.depth is not None:
            cv2.imwrite(str(out / "depth_mm.png"), np.clip(frame.depth * 1000.0, 0, 65535).astype(np.uint16))
        gts = [{"object_id": g.object_id, "bbox": list(g.bbox), "visible_px": g.visible_px} for g in frame.gt]
        (out / "gt.json").write_text(json.dumps({"gt": gts, "cam_pos": list(map(float, frame.cam_pos)),
                                                "cam_q": list(frame.cam_q), "fov_deg": frame.fov_deg}, indent=1))
        print(f"{out}: rgb/depth/seg + {len(gts)} ground-truth boxes {[(g['object_id'], g['visible_px']) for g in gts]}")
        await ctrl.rtl() if hasattr(ctrl, "rtl") else None
    finally:
        client.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="scenarios/demo/golden_hour_01.jsonc")
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--at", type=float, nargs=2, default=[80.0, 176.0], help="viewpoint N E (m): Lin and a smouldering fire")
    p.add_argument("--height", type=float, default=45.0, help="metres above ground")
    p.add_argument("--out", default=str(ROOT / "runs/_film/golden_hour/quad"))
    asyncio.run(capture(p.parse_args()))


if __name__ == "__main__":
    main()
