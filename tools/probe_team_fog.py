"""Live check: does a drone's photo look the same whether it flies alone or with teammates?

Project AirSim gives every drone its own weather actor, whose particle fog follows the drone. Before
the fix in the plugin's WeatherLib (each drone's cameras now hide the other drones' weather
actors), every camera saw every drone's fog, so a team of four photographed through four
times the fog. This probe loads the same scene with 1 and with N drones, hovers Drone1 at survey
height over its pad (teammates hover 10 m away, the worst case), and compares Drone1's payload
photos: mean grey, contrast (standard deviation) and fine detail (Laplacian).

The simulator must already be running on ForestSAR:

    python tools/probe_team_fog.py --out runs/_probe_fog

Writes summary.json and compare.jpg to --out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from probe_fleet import fleet_scene, lockstep_batch  # noqa: E402


def _stats(img: np.ndarray) -> Dict[str, float]:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"grey": round(float(grey.mean()), 2), "std": round(float(grey.std()), 2),
            "laplacian_var": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 1)}


async def photos(args, n: int) -> Dict[str, Any]:
    from projectairsim import Drone, ProjectAirSimClient

    from vivid2.runner import allow_long_async_requests, apply_environment, load_world, serialize_requests
    from vivid2.scenario import Scenario
    from vivid2.stack.reference.controllers import AirSimController
    from vivid2.terrain import load_for_run

    sim_cfg = str(ROOT / "sim_config")
    run = Scenario.load(ROOT / args.scenario).expand(seeds=[0])[0]
    terrain, _ = load_for_run(sim_cfg, run.spec)
    scene, pads = fleet_scene(run, sim_cfg, terrain, n, "drone1")
    client = ProjectAirSimClient(address=args.address)
    client.connect()
    serialize_requests(client)
    allow_long_async_requests(client)
    try:
        world = load_world(client, scene, sim_cfg)
        drones = [Drone(client, world, f"Drone{i + 1}") for i in range(n)]
        apply_environment(world, run.spec.get("environment", {}))
        ctrls = [AirSimController(d, world.get_sim_time, lambda s, r: None) for d in drones]
        ground = float(terrain.down_at(*run.spec["dut_start"]["ned"][:2]))
        survey_down = ground - float(run.spec["search_area"]["altitude_m"])
        await asyncio.gather(*(c.takeoff(survey_down) for c in ctrls))
        await asyncio.sleep(2.0)
        shots = []
        for _ in range(args.captures):  # the first ones settle the auto exposure
            shots.append(await lockstep_batch(client, world, drones[:1], "Payload", [0], int(0.25e9), 30.0))
            await asyncio.sleep(0.5)
        return {"pads": pads, "last": shots[-1]["images"], "batches": [{k: v for k, v in s.items() if k != "images"}
                                                                       for s in shots]}
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


def _decode(images) -> np.ndarray:
    from vivid2.runner import _decode as decode

    return decode(images[0], np.uint8, 3)  # Drone1's colour image


async def main(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"scenario": args.scenario}
    tiles = []
    for n in args.sizes:
        r = await photos(args, n)
        img = _decode(r["last"])
        s = _stats(img)
        summary[f"n{n}"] = s
        print(f"N={n}: Drone1 photo {s}", flush=True)
        small = cv2.resize(img, (640, 480))
        cv2.putText(small, f"N={n} std={s['std']} lap={s['laplacian_var']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        tiles.append(small)
    cv2.imwrite(str(out / "compare.jpg"), np.hstack(tiles))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", out / "summary.json")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default="scenarios/forest_team/forest_team_01.jsonc")
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--captures", type=int, default=8)
    p.add_argument("--out", default="runs/_probe_fog")
    sys.exit(asyncio.run(main(p.parse_args())))
