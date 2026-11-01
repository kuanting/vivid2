"""Turn a concrete Run into a Project AirSim scene-config dict.

The base scene and robot configs are ordinary Project AirSim JSONC files in
the campaign's sim_config directory. The builder:
  * points the DUT actor at the scenario's robot config and start pose,
  * adds target actors (hikers, debris...) as ``spawn-objects``,
  * turns on bbox annotations for those targets on the payload camera.
The result is passed to ``World.load_scene(dict)``, so nothing is written to disk.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

from vivid2.scenario import Run, load_jsonc


def _fmt3(v) -> str:
    return " ".join(f"{float(x):.3f}" for x in v)


def build_scene(run: Run, sim_config_dir: str, targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    spec = run.spec
    sc = spec["scene"]
    scene = load_jsonc(os.path.join(sim_config_dir, sc["base"]))
    robot = load_jsonc(os.path.join(sim_config_dir, sc["robot_config"]))

    drone_name = sc.get("drone_name", "Drone1")
    actor = next((a for a in scene["actors"] if a["type"] == "robot" and a["name"] == drone_name), None)
    if actor is None:
        raise ValueError(f"base scene {sc['base']} has no robot actor named {drone_name}")
    actor["origin"] = {"xyz": _fmt3(spec["dut_start"]["ned"]), "rpy-deg": f"0 0 {spec['dut_start'].get('yaw_deg', 0.0)}"}
    actor["robot-config"] = _annotate(robot, sc.get("camera", "Payload"), [t["id"] for t in targets])

    spawn = scene.setdefault("spawn-objects", {}).setdefault("sim-packaged", [])
    for t in targets:
        spawn.append({
            "name": t["id"],
            "asset-path": t["asset"],
            "origin": {"xyz": _fmt3(t["ned"]), "rpy-deg": f"0 0 {t.get('yaw_deg', 0.0)}"},
            "scale": _fmt3(t.get("scale", [1, 1, 1])),
            "physics-enabled": False,
        })
    scene["id"] = scene.get("id", "VIVID2Scene")
    return scene


def _annotate(robot: Dict[str, Any], camera_id: str, object_ids: List[str]) -> Dict[str, Any]:
    robot = copy.deepcopy(robot)
    cam = next((s for s in robot.get("sensors", []) if s["id"] == camera_id), None)
    if cam is None:
        raise ValueError(f"robot config has no camera sensor '{camera_id}'")
    ann = cam.setdefault("annotation-settings", {})
    ann["enabled"] = True
    ann["object-ids"] = sorted(set(ann.get("object-ids", [])) | set(object_ids))
    ann.setdefault("bbox2D-settings", {"alignment": "axis"})
    ann.setdefault("bbox3D-settings", {"alignment": "axis"})
    return robot


def camera_settings(robot_or_scene: Dict[str, Any], camera_id: str, drone_name: str = "Drone1") -> Dict[str, Any]:
    """Return the RGB capture settings of ``camera_id`` from a loaded scene or robot dict."""
    robot = robot_or_scene
    if "actors" in robot_or_scene:
        robot = next(a for a in robot_or_scene["actors"] if a.get("name") == drone_name)["robot-config"]
    cam = next(s for s in robot["sensors"] if s["id"] == camera_id)
    return next(c for c in cam["capture-settings"] if c["image-type"] == 0)
