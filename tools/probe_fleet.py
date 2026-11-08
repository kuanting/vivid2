"""Live probe for team runs: can several payload cameras all capture in one lockstep step?

A team run takes each drone's photo the way a single-drone run does (pause the simulator, ask
for the images, let it run one capture step, receive them paused), but for several drones in
the same step. This probe measures whether that works with N full payload cameras (1920x1440
RGB + depth + segmentation each), what it costs in wall time, and what the simulator reports
when two drones collide.

The simulator must already be running on ForestSAR (packaged build or editor):

    python tools/probe_fleet.py --out runs/_probe_fleet/chase1 --collide
    python tools/probe_fleet.py --out runs/_probe_fleet/chaseall --chase all --sizes 4 --batches 25 --land

Writes summary.json, collisions.json and mosaic.jpg (the last batch's photos) to --out.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAD_OFFSETS = [(-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)]  # 2x2 grid, 10 m apart, around the start


def fleet_scene(run, sim_cfg: str, terrain, n: int, chase: str):
    """The run's scene with Drone1 cloned into Drone1..DroneN on pads around the start."""
    from vivid2.sim.scene_builder import _fmt3, build_scene
    from vivid2.terrain import snap_to_ground

    scene = build_scene(run, sim_cfg, run.resolved_targets(terrain), run.resolved_distractors(terrain),
                        run.resolved_start(terrain))
    base = next(a for a in scene["actors"] if a.get("name") == "Drone1")
    others = [a for a in scene["actors"] if a is not base]
    start = run.spec["dut_start"]["ned"]
    robots, pads = [], []
    for i in range(n):
        dn, de = PAD_OFFSETS[i]
        pad = snap_to_ground([start[0] + dn, start[1] + de, start[2]], terrain, run.authored_ground_down_m)
        a = copy.deepcopy(base)
        a["name"] = f"Drone{i + 1}"
        a["origin"]["xyz"] = _fmt3(pad)
        if i > 0 and chase != "all":
            for s in a["robot-config"]["sensors"]:
                if s["id"] == "Chase":
                    for c in s.get("capture-settings", []):
                        c["streaming-enabled"] = False
        robots.append(a)
        pads.append([float(v) for v in pad])
    scene["actors"] = robots + others
    return scene, pads


async def lockstep_batch(client, world, drones, camera: str, types: List[int], step_ns: int,
                         timeout_s: float) -> Dict[str, Any]:
    """One pause, one GetImages per drone, one capture step, all replies received paused."""
    w0 = time.perf_counter()
    await asyncio.to_thread(world.pause)
    t_req = int(await asyncio.to_thread(world.get_sim_time))
    socks, futs = [], []
    arrivals: List[float] = [float("nan")] * len(drones)
    out: Dict[str, Any] = {"n": len(drones), "ok": [], "errors": []}
    try:
        for i, d in enumerate(drones):
            req = {"method": f"{d.sensors_topic}/{camera}/GetImages", "params": {"image_type_ids": types},
                   "version": 1.0}
            sock = client.socket_services.new_context()
            socks.append(sock)
            await sock.asend(client.preprocess_request(req))
            f = asyncio.ensure_future(sock.arecv_msg())
            f.add_done_callback(lambda _f, i=i: arrivals.__setitem__(i, time.perf_counter()))
            futs.append(f)
        await asyncio.sleep(0.02)
        w_step = time.perf_counter()
        t_fetched = int(await asyncio.to_thread(world.continue_for_sim_time, step_ns, True))
        w_stepped = time.perf_counter()
        await asyncio.wait(futs, timeout=timeout_s)
        images = []
        for i, f in enumerate(futs):
            if not f.done():
                out["ok"].append(False)
                out["errors"].append(f"Drone{i + 1}: no reply in {timeout_s} s")
                images.append(None)
                continue
            try:
                data = {int(k): v for k, v in client.postprocess_response(f.result()).items()}
                rgb = data.get(0)
                stamp = int(rgb["time_stamp"]) if rgb else None
                ok = bool(rgb) and stamp is not None and t_req <= stamp <= t_fetched
                out["ok"].append(ok)
                if not ok:
                    out["errors"].append(f"Drone{i + 1}: stamp {stamp} outside [{t_req}, {t_fetched}]")
                out.setdefault("stamp_after_request_s", []).append(None if stamp is None else (stamp - t_req) / 1e9)
                out.setdefault("types", []).append(sorted(data))
                images.append(rgb)
            except Exception as e:  # the server's own GetImages timeout or error
                out["ok"].append(False)
                out["errors"].append(f"Drone{i + 1}: {e!r}")
                images.append(None)
        out["sim_step_s"] = (t_fetched - t_req) / 1e9
        out["wall_step_s"] = w_stepped - w_step
        out["arrival_after_step_s"] = [a - w_stepped for a in arrivals]
        out["images"] = images
    finally:
        for f in futs:
            if not f.done():
                f.cancel()
        for s in socks:
            s.close()
        await asyncio.to_thread(world.resume)
    out["wall_s"] = time.perf_counter() - w0
    return out


def summarise(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    walls = sorted(b["wall_s"] for b in batches)
    n = batches[0]["n"]
    arrivals = np.array([b.get("arrival_after_step_s", [np.nan] * n) for b in batches], float)
    stamps = [s for b in batches for s in b.get("stamp_after_request_s", []) if s is not None]
    return {
        "batches": len(batches),
        "all_ok": sum(all(b["ok"]) and len(b["ok"]) == n for b in batches),
        "photos_ok": int(sum(sum(b["ok"]) for b in batches)),
        "errors": [e for b in batches for e in b["errors"]][:10],
        "wall_s_mean": round(statistics.mean(walls), 3),
        "wall_s_p95": round(walls[min(len(walls) - 1, int(0.95 * len(walls)))], 3),
        "wall_s_max": round(walls[-1], 3),
        "sim_step_s_mean": round(statistics.mean(b.get("sim_step_s", float("nan")) for b in batches), 4),
        "wall_step_s_mean": round(statistics.mean(b.get("wall_step_s", float("nan")) for b in batches), 3),
        "arrival_after_step_s_mean_per_drone": [round(float(v), 3) for v in np.nanmean(arrivals, axis=0)],
        "stamp_after_request_s": [round(min(stamps), 3), round(max(stamps), 3)] if stamps else None,
    }


def mosaic(images, path: Path) -> None:
    import cv2

    from vivid2.runner import _decode

    tiles = []
    for i, msg in enumerate(images):
        img = np.zeros((360, 480, 3), np.uint8) if not msg else cv2.resize(_decode(msg, np.uint8, 3), (480, 360))
        cv2.putText(img, f"Drone{i + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(path), np.vstack(rows))


async def main(args) -> int:
    from projectairsim import Drone, ProjectAirSimClient

    from vivid2.runner import allow_long_async_requests, apply_environment, load_world, serialize_requests
    from vivid2.scenario import Scenario
    from vivid2.stack.reference.controllers import AirSimController
    from vivid2.terrain import load_for_run

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = str(ROOT / "sim_config")
    run = Scenario.load(ROOT / args.scenario).expand(seeds=[0])[0]
    terrain, _ = load_for_run(sim_cfg, run.spec)
    n_max = max(args.sizes)
    scene, pads = fleet_scene(run, sim_cfg, terrain, n_max, args.chase)

    client = ProjectAirSimClient(address=args.address)
    client.connect()
    serialize_requests(client)
    allow_long_async_requests(client)
    summary: Dict[str, Any] = {"scenario": run.scenario_id, "chase": args.chase, "pads": pads, "results": {}}
    collisions: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    try:
        w = time.perf_counter()
        world = load_world(client, scene, sim_cfg)
        summary["load_scene_wall_s"] = round(time.perf_counter() - w, 1)
        drones = [Drone(client, world, f"Drone{i + 1}") for i in range(n_max)]
        for d in drones:
            client.subscribe(d.robot_info["collision_info"],
                             lambda _, msg, name=d.name: collisions.append(
                                 {"drone": name, "t": msg.get("time_stamp"), "object_name": msg.get("object_name"),
                                  "impact_point": msg.get("impact_point")}))
        apply_environment(world, run.spec.get("environment", {}))
        log = lambda stream, rec: events.append(rec)  # noqa: E731
        ctrls = [AirSimController(d, world.get_sim_time, log) for d in drones]
        ground = float(terrain.down_at(run.spec["dut_start"]["ned"][0], run.spec["dut_start"]["ned"][1]))
        survey_down = ground - float(run.spec["search_area"]["altitude_m"])

        w = time.perf_counter()
        await asyncio.gather(*(c.takeoff(survey_down) for c in ctrls))
        summary["takeoff_wall_s"] = round(time.perf_counter() - w, 1)
        poses = [d.get_ground_truth_pose()["translation"] for d in drones]
        summary["hover"] = [[round(p["x"], 2), round(p["y"], 2), round(p["z"], 2)] for p in poses]
        summary["collisions_by_takeoff"] = list(collisions)
        await asyncio.sleep(2.0)

        types = [0, 1, 3]
        last_images = None
        for step in args.steps:
            for n in sorted(args.sizes):
                batches = []
                for _ in range(args.batches):
                    b = await lockstep_batch(client, world, drones[:n], "Payload", types, int(round(step * 1e9)),
                                             args.reply_timeout)
                    last_images = b.pop("images")
                    batches.append(b)
                    await asyncio.sleep(args.gap)  # the simulator runs between photos, as in a mission
                s = summarise(batches)
                summary["results"][f"n{n}_step{step}"] = s
                print(f"N={n} step={step}s: {s['all_ok']}/{s['batches']} batches complete, wall "
                      f"{s['wall_s_mean']} s mean / {s['wall_s_p95']} s p95, arrivals {s['arrival_after_step_s_mean_per_drone']}, "
                      f"errors {s['errors'][:2]}", flush=True)
            if all(summary["results"][f"n{n}_step{step}"]["all_ok"] == args.batches for n in args.sizes):
                break  # this step works for every size; longer ones are not needed
        if last_images:
            mosaic(last_images, out_dir / "mosaic.jpg")

        if args.collide and n_max >= 2:
            # Drone2 flies slowly into Drone1 at the same height.
            t0 = world.get_sim_time()
            p1 = drones[0].get_ground_truth_pose()["translation"]
            await drones[1].move_to_position_async(p1["x"], p1["y"], p1["z"], velocity=1.0)
            dmin = float("inf")
            for _ in range(int(args.collide_s * 10)):
                a = drones[0].get_ground_truth_pose()["translation"]
                b = drones[1].get_ground_truth_pose()["translation"]
                dmin = min(dmin, float(np.linalg.norm([a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"]])))
                await asyncio.sleep(0.1)
            summary["collide"] = {"min_distance_m": round(dmin, 2),
                                  "records": [c for c in collisions if c["t"] is not None and c["t"] >= t0]}
            print("collision test:", json.dumps(summary["collide"])[:600], flush=True)

        if args.land:
            t0 = world.get_sim_time()
            w = time.perf_counter()
            await asyncio.gather(*(c.return_and_land(pad, 5.0) for c, pad in zip(ctrls, pads)))
            ends = [d.get_ground_truth_pose()["translation"] for d in drones]
            summary["land"] = {
                "wall_s": round(time.perf_counter() - w, 1),
                "end_distance_from_pad_m": [round(float(np.hypot(e["x"] - p[0], e["y"] - p[1])), 2)
                                            for e, p in zip(ends, pads)],
                "records": [c for c in collisions if c["t"] is not None and c["t"] >= t0]}
            print("landing:", json.dumps({k: v for k, v in summary["land"].items() if k != "records"}),
                  f"{len(summary['land']['records'])} contact records", flush=True)
    finally:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        (out_dir / "collisions.json").write_text(json.dumps(collisions, indent=1, default=str), encoding="utf-8")
        try:
            client.disconnect()
        except Exception:
            pass
    print("wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="scenarios/forest_sar/forest_hiker_01.jsonc")
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4], help="team sizes to batch (max 4)")
    p.add_argument("--steps", type=float, nargs="+", default=[0.25, 0.35, 0.45],
                   help="capture steps to try, in order, until one works for every size")
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--gap", type=float, default=0.75, help="wall seconds the simulator runs between batches")
    p.add_argument("--reply-timeout", type=float, default=30.0)
    p.add_argument("--chase", choices=["drone1", "all"], default="drone1",
                   help="which drones keep the Chase camera in the game window's view cycle")
    p.add_argument("--collide", action="store_true", help="fly Drone2 into Drone1 at the end")
    p.add_argument("--collide-s", type=float, default=20.0)
    p.add_argument("--land", action="store_true", help="land every drone on its own pad at the end")
    p.add_argument("--out", default="runs/_probe_fleet")
    raise SystemExit(asyncio.run(main(p.parse_args())))
