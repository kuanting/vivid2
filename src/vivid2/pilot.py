"""Manual flight: fly a VIVID 2 map by hand with a radio, gamepad or keyboard (``vivid2 fly``).

Python only builds and loads the scene. The drone is flown inside Unreal by the pilot plugin
(``FpvPilot``, linked with ``vivid2 unreal setup --pilot-plugin``; not part of VIVID 2): its
flight controller runs in Project AirSim's physics loop through the external-controller-api
hook (``patches/projectairsim-pilot-hooks.patch``) and reads the sticks itself. See
docs/manual_flight.md.

Manual flights are not benchmark runs: nothing is recorded or scored, and the benchmark's own
robot config and scenes are untouched (the pilot drone is built from them here).

Drones:
    fpv5   a 5-inch freestyle quad (acro / horizon / angle), sim_config/pilot/robot_fpv5.jsonc
    quad   VIVID 2's benchmark quad (robot_vivid_quad.jsonc) with the pilot's flight controller
           instead of simple_flight, angle mode only; its search camera tilts (F / X, PgUp / PgDn)
"""
from __future__ import annotations

import copy
import dataclasses
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vivid2.scenario import Run, Scenario, load_jsonc

PILOT_FACTORY = "pilot-fc"
# The sim clock for flying by hand: the flight controller runs every 2 ms step. The scheduler's
# period is below the step so it never sleeps (Windows sleeps round up to 15.6 ms); the pilot
# plugin turns on pas.SimRealTimePacing, which keeps sim time on the wall clock.
PILOT_STEP_NS = 2_000_000
PILOT_REAL_TIME_UPDATE_NS = 500_000
# Tab switches the window to each streaming camera's render size: keep them all the same.
STREAM_SIZE = (1280, 720)

DRONES: Dict[str, Dict[str, Any]] = {
    "fpv5": {
        "robot": "pilot/robot_fpv5.jsonc",
        "name": "5-inch FPV",
        # start-landed: the robot rests where it spawns until it lifts off, so spawn it just
        # above the ground: the 5-inch frame reaches 6 cm below its origin.
        "spawn_height_m": 0.10,
    },
    "quad": {
        "robot": "robot_vivid_quad.jsonc",
        "name": "VIVID2 quad",
        "fc_profile": "vivid_quad",
        # /Drone/Quadrotor1 reaches 18 cm below its origin.
        "spawn_height_m": 0.22,
        # The pilot's search camera: Payload (scored in the benchmark) and its preview share a mount.
        "gimbal_cameras": ["Payload", "PayloadPreview"],
        "streaming": ["Chase", "PayloadPreview", "Front"],
        "crash_speed_mps": 3.0,
    },
}

# Every VIVID 2 forest map uses the same base scene (sim_config/scene_forest_sar.jsonc).
MAP_BASES = {
    "ForestSAR": "scene_forest_sar.jsonc",
    "ForestSAR_Open": "scene_forest_sar.jsonc",
    "ForestSAR_Dense": "scene_forest_sar.jsonc",
    "ForestFire": "scene_forest_sar.jsonc",
}

SERVICES_PORT = 8990


# -- robot ---------------------------------------------------------------------------------------

def air_density_ratio(altitude_m: float) -> float:
    """ISA troposphere density over sea level's at a geometric altitude (m)."""
    return float((1.0 - 2.25577e-5 * max(0.0, min(altitude_m, 11000.0))) ** 4.2559)


def quad_pilot_robot(robot: Dict[str, Any], density_ratio: float = 1.0) -> Dict[str, Any]:
    """VIVID 2's benchmark quad with the pilot's flight controller instead of simple_flight.

    Only the controller block and the cameras' streaming flags change: links, actuators, cameras'
    mounts and lenses stay those of the benchmark airframe, so a manual flight sees what an agent's
    drone sees. The motor order is the benchmark controller's actuator-order (PX4 quad-X).
    """
    spec = DRONES["quad"]
    out = copy.deepcopy(robot)
    ctrl = out["controller"]
    settings = next((v for k, v in ctrl.items() if k.endswith("-settings") and isinstance(v, dict)), None)
    if not settings or "actuator-order" not in settings:
        raise ValueError("robot config's controller has no actuator-order")
    out["controller"] = {
        "id": "PilotFC",
        "airframe-setup": ctrl.get("airframe-setup", "quadrotor-x"),
        "type": "external-controller-api",
        "external-controller-api-settings": {
            "factory": PILOT_FACTORY,
            "fc-profile": spec["fc_profile"],
            # Project AirSim's rotor is linear in its command; the profile assumes no ESC curve.
            "thrust-curve": 0.0,
            "crash-speed": spec["crash_speed_mps"],
            "name": spec["name"],
            "modes": ["angle"],
            "gimbal-cameras": list(spec["gimbal_cameras"]),
            "air-density-ratio": round(density_ratio, 4),
            "actuator-order": copy.deepcopy(settings["actuator-order"]),
        },
    }
    w, h = STREAM_SIZE
    for sensor in out.get("sensors", []):
        if sensor.get("type") != "camera":
            continue
        streaming = sensor["id"] in spec["streaming"]
        for i, cap in enumerate(sensor.get("capture-settings", [])):
            on = streaming and i == 0 and cap.get("image-type", 0) == 0
            cap["streaming-enabled"] = on
            if on:
                cap["width"], cap["height"] = w, h
    return out


def pilot_robot(drone: str, sim_config_dir: str, density_ratio: float = 1.0) -> Dict[str, Any]:
    if drone not in DRONES:
        raise ValueError(f"unknown drone {drone!r}; choose from {sorted(DRONES)}")
    robot = load_jsonc(os.path.join(sim_config_dir, DRONES[drone]["robot"]))
    if drone == "quad":
        return quad_pilot_robot(robot, density_ratio)
    settings = robot["controller"]["external-controller-api-settings"]
    settings["air-density-ratio"] = round(density_ratio, 4)
    return robot


# -- scene ---------------------------------------------------------------------------------------

def map_run(map_name: str) -> Run:
    """A run with no targets at the map's launch clearing (the base scene's robot origin)."""
    if map_name not in MAP_BASES:
        raise ValueError(f"unknown map {map_name!r}; choose from {sorted(MAP_BASES)} (or use --scenario)")
    spec = {
        "id": f"fly_{map_name}",
        "scene": {"base": MAP_BASES[map_name], "robot_config": "robot_vivid_quad.jsonc", "camera": "Payload",
                  "map": map_name},
        "dut_start": {"ned": [0.0, 0.0, -1.0], "yaw_deg": 0.0},
        "search_area": {"ground_down_m": 0.0},
        "targets": [],
    }
    return Run(run_id=spec["id"], scenario_id=spec["id"], seed=0, params={}, spec=spec)


def scenario_run(path: str, seed: int = 0, map_name: Optional[str] = None) -> Run:
    """The scenario's first sweep combination with ``seed`` (target jitter), optionally on another map."""
    extra = {"scene.map": [map_name]} if map_name else None
    runs = Scenario.load(path).expand(seeds=[seed], extra_sweep=extra)
    return runs[0]


def pilot_scene(run: Run, sim_config_dir: str, drone: str = "fpv5",
                with_targets: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The run's scene with the pilot drone landed at its start, and a summary for the console."""
    from vivid2.sim.scene_builder import build_scene
    from vivid2.terrain import load_for_run

    if run.fleet_size > 1:
        raise ValueError(f"{run.scenario_id} flies a team; manual flight flies one drone")
    terrain, _ = load_for_run(sim_config_dir, run.spec)
    targets = run.resolved_targets(terrain) if with_targets else []
    distractors = run.resolved_distractors(terrain) if with_targets else []
    fires = run.resolved_fires(terrain)
    n, e, _ = run.resolved_start(terrain)
    start = [n, e, float(terrain.down_at(n, e)) - DRONES[drone]["spawn_height_m"]]

    base = load_jsonc(os.path.join(sim_config_dir, run.spec["scene"]["base"]))
    altitude = float(base.get("home-geo-point", {}).get("altitude", 0.0)) - start[2]
    density = air_density_ratio(altitude)
    robot = pilot_robot(drone, sim_config_dir, density)

    # The scene builder annotates the scenario's scored camera with the targets; the 5-inch has no
    # Payload camera, so it annotates the drone's first camera instead and that is undone below.
    cameras = [x["id"] for x in robot.get("sensors", []) if x.get("type") == "camera"]
    unannotate = None
    if run.spec["scene"].get("camera", "Payload") not in cameras:
        spec = copy.deepcopy(run.spec)
        spec["scene"]["camera"] = unannotate = cameras[0]
        run = dataclasses.replace(run, spec=spec)
    scene = build_scene(run, sim_config_dir, targets, distractors, start, fires=fires, robot=robot)
    name = run.spec["scene"].get("drone_name", "Drone1")
    actor = next(a for a in scene["actors"] if a.get("type") == "robot" and a.get("name") == name)
    actor["start-landed"] = True
    if unannotate:
        for x in actor["robot-config"]["sensors"]:
            if x["id"] == unannotate:
                x.pop("annotation-settings", None)
    scene["clock"]["step-ns"] = PILOT_STEP_NS
    scene["clock"]["real-time-update-rate"] = PILOT_REAL_TIME_UPDATE_NS
    scene["id"] = "VIVID2Pilot"
    info = {
        "drone": DRONES[drone]["name"],
        "map": run.spec["scene"].get("map"),
        "scenario": run.scenario_id,
        "params": run.params,
        "start_ned": [round(v, 2) for v in start],
        "altitude_m": round(altitude, 1),
        "air_density_ratio": round(density, 3),
        "targets": [t["id"] for t in targets],
        "distractors": [d["id"] for d in distractors],
        "fires": [f["id"] for f in fires],
    }
    return scene, info


# -- sim -----------------------------------------------------------------------------------------

def load_into_sim(scene: Dict[str, Any], map_name: Optional[str], environment: Dict[str, Any],
                  sim_config_dir: str, address: str = "127.0.0.1") -> None:
    """Load the scene into the running sim (editor Play or a game window), check the map, set the weather."""
    from projectairsim import ProjectAirSimClient

    from vivid2.preflight import check_map
    from vivid2.runner import apply_environment, load_world

    client = ProjectAirSimClient(address)
    client.connect()
    try:
        world = load_world(client, scene, sim_config_dir)
        check_map(world, {"map": map_name} if map_name else {})
        apply_environment(world, environment or {})
    finally:
        client.disconnect()


def unreal_editor(state: Dict[str, Any]) -> Path:
    engine_dir = state.get("engine_dir")
    if not engine_dir:
        raise RuntimeError("no engine recorded in .vivid2-setup.json; run vivid2 unreal setup")
    return Path(engine_dir) / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe"


def launch_command(state: Dict[str, Any], map_name: str, extra: Sequence[str] = ()) -> List[str]:
    """The editor binary in game mode on the map: the pilot plugin is an editor-build plugin."""
    if not state.get("pilot_plugin"):
        raise RuntimeError("the pilot plugin isn't linked: vivid2 unreal setup ... --pilot-plugin <dir>")
    # The editor binary in -game mode compiles textures on first use; its asynchronous texture
    # compiler crashed (TextureCompiler.cpp:355, a texture registered during another's post-compile)
    # loading a robot on ForestSAR. Compile them synchronously instead.
    return [str(unreal_editor(state)), str(state["uproject"]), f"/Game/VIVID2/Maps/{map_name}", "-game",
            "-windowed", "-ResX=1600", "-ResY=900", "-nosplash", "-DPCVars=Editor.AsyncTextureCompilation=0",
            *extra]


def wait_for_sim(address: str, timeout_s: float, proc: Optional[subprocess.Popen] = None,
                 log: Callable[[str], None] = print) -> None:
    """Until the sim's services port accepts connections (the map has loaded and play started)."""
    deadline = time.monotonic() + timeout_s
    last = 0.0
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Unreal exited (code {proc.returncode}) before the sim started")
        try:
            with socket.create_connection((address, SERVICES_PORT), timeout=1.0):
                return
        except OSError:
            pass
        if time.monotonic() - last > 30:
            log("  waiting for the sim (first start compiles shaders)...")
            last = time.monotonic()
        time.sleep(1.0)
    raise TimeoutError(f"no sim at {address}:{SERVICES_PORT} after {timeout_s:.0f} s")


KEYS = """In the Unreal window:
  F2 / Start      setup menu (controller, rates, modes, PID tuning; K calibrates)
  Space / RB      arm (throttle fully down first)      M / LB  flight mode
  R / Y           back to the launch pad               H       help, Tab  next camera"""
KEYS_GIMBAL = "  F / X           search camera: straight down, 45 deg, horizon   PgUp / PgDn  tilt"


__all__ = ["DRONES", "MAP_BASES", "air_density_ratio", "quad_pilot_robot", "pilot_robot", "map_run",
           "scenario_run", "pilot_scene", "load_into_sim", "launch_command", "wait_for_sim", "KEYS", "KEYS_GIMBAL"]
