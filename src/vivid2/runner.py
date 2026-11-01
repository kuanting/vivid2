"""Execute runs against a live Project AirSim server and score them."""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from vivid2 import metrics as vmetrics
from vivid2.recorder import Recorder, RunLog
from vivid2.scenario import Campaign, Run
from vivid2.scoring import layer_scores, overall_score
from vivid2.sim.scene_builder import build_scene, camera_settings
from vivid2.stack.base import Agent, Frame, GTAnnotation, MissionContext, MissionSpec, TargetReport
from vivid2.stack.reference.controllers import AirSimController
from vivid2.unreal_setup import status as unreal_status

TRAJECTORY_HZ = 10.0
RTL_GRACE_S = 180.0


def _xyz(d: Dict[str, float]) -> List[float]:
    return [d["x"], d["y"], d["z"]]


def _wxyz(d: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (d["w"], d["x"], d["y"], d["z"])


def mission_spec(run: Run) -> MissionSpec:
    s = run.spec
    return MissionSpec(
        mission=s["mission"],
        brief=s["brief"],
        search_polygon_ne=s["search_area"]["polygon_ne"],
        altitude_m=s["search_area"]["altitude_m"],
        ground_down_m=s["search_area"].get("ground_down_m", 0.0),
        start_ned=s["dut_start"]["ned"],
        max_speed_mps=s["limits"].get("max_speed_mps", 8.0),
        timeout_s=s["limits"]["timeout_s"],
        expected_targets=len(s["targets"]),
    )


def run_config_hash(run: Run, agent_desc: Dict[str, Any]) -> str:
    """Fingerprint of everything that determines a run's outcome.

    The run id stays (scenario, swept params, seed) so different agents can be
    paired on the same run; this hash also covers the resolved scenario spec and
    the agent's configuration, so editing either forces a re-run on resume.
    """
    blob = json.dumps({"spec": run.spec, "seed": run.seed, "agent": agent_desc}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def apply_environment(world, env: Dict[str, Any]) -> None:
    """Mirror of projectairsim.test.EnvProfile.configure, driven by the scenario dict."""
    from projectairsim.types import WeatherParameter

    if env.get("time_of_day"):
        world.set_sun_position_from_date_time(datetime.strptime(env["time_of_day"], "%Y-%m-%d %H:%M:%S"))
    if env.get("cloud_shadow_strength") is not None:
        world.set_cloud_shadow_strength(env["cloud_shadow_strength"])
    if env.get("sunlight_intensity") is not None:
        world.set_sunlight_intensity(env["sunlight_intensity"])
    if env.get("wind_ned"):
        world.set_wind_velocity(*env["wind_ned"])
    weather = {k: v for k, v in (env.get("weather") or {}).items() if v}
    if weather:
        world.enable_weather_visual_effects()
        for name, value in weather.items():
            world.set_weather_visual_effects_param(WeatherParameter[name.upper()], float(value))


def projectairsim_commit(airsim_root: Optional[str]) -> Optional[str]:
    if not airsim_root:
        return None
    try:
        return subprocess.check_output(["git", "-C", airsim_root, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


class SimMissionContext(MissionContext):
    def __init__(self, run: Run, world, drone, camera_id: str, recorder: Recorder, controller: AirSimController, fov_deg: float):
        self.run = run
        self.world = world
        self.drone = drone
        self.camera_id = camera_id
        self.recorder = recorder
        self.controller = controller
        self.fov_deg = fov_deg
        self.mission = mission_spec(run)
        self.t0_ns = world.get_sim_time()
        self._wall0 = time.monotonic()
        self._frame_id = 0
        self.reports: List[TargetReport] = []
        self._stop = False

    # -- MissionContext API -------------------------------------------------
    async def get_frame(self) -> Optional[Frame]:
        from projectairsim.types import ImageType

        imgs = await asyncio.to_thread(self.drone.get_images, self.camera_id, [ImageType.SCENE, ImageType.DEPTH_PLANAR])
        rgb_msg = imgs.get(ImageType.SCENE)
        if not rgb_msg:
            return None
        rgb = _decode(rgb_msg, np.uint8, 3)
        depth = None
        dmsg = imgs.get(ImageType.DEPTH_PLANAR)
        if dmsg:
            depth = _decode(dmsg, np.uint16, 1).astype(np.float32) / 1000.0  # 16UC1 millimetres -> metres
        gt = []
        for a in rgb_msg.get("annotations") or []:
            c, s = a["bbox2d"]["center"], a["bbox2d"]["size"]
            gt.append(GTAnnotation(a["object_id"], (c["x"] - s["x"] / 2, c["y"] - s["y"] / 2, c["x"] + s["x"] / 2, c["y"] + s["y"] / 2)))
        self._frame_id += 1
        frame = Frame(
            t_ns=int(rgb_msg["time_stamp"]), frame_id=self._frame_id, rgb=rgb, depth=depth,
            cam_pos=np.array([rgb_msg["pos_x"], rgb_msg["pos_y"], rgb_msg["pos_z"]]),
            cam_q=(rgb_msg["rot_w"], rgb_msg["rot_x"], rgb_msg["rot_y"], rgb_msg["rot_z"]),
            fov_deg=self.fov_deg, gt=gt,
        )
        self.recorder.log("frames", {
            "t": frame.t_ns, "frame_id": frame.frame_id, "cam_pos": frame.cam_pos, "cam_q": list(frame.cam_q),
            "width": frame.width, "height": frame.height, "fov_deg": frame.fov_deg,
            "gt": [{"object_id": g.object_id, "bbox": list(g.bbox)} for g in gt],
        })
        return frame

    def pose(self):
        p = self.drone.get_ground_truth_pose()
        return np.asarray(_xyz(p["translation"])), _wxyz(p["rotation"])

    def report_target(self, report: TargetReport) -> None:
        self.reports.append(report)
        self.recorder.log("reports", {"t": report.t_ns, "frame_id": report.frame_id, "ned": report.ned,
                                      "confidence": report.confidence, "source": report.source, "note": report.note})

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        self.recorder.log(stream, record)

    def should_stop(self) -> bool:
        return self._stop or self.elapsed_s() >= self.mission.timeout_s

    def elapsed_s(self) -> float:
        return (self.world.get_sim_time() - self.t0_ns) / 1e9


def _decode(msg: Dict[str, Any], dtype, channels: int) -> np.ndarray:
    data = msg["data"]
    buf = np.array(data, dtype=np.uint8) if isinstance(data, list) else np.frombuffer(data, dtype=np.uint8)
    arr = buf.view(dtype)
    shape = (msg["height"], msg["width"]) if channels == 1 else (msg["height"], msg["width"], channels)
    return arr.reshape(shape)


async def _poll_trajectory(ctx: SimMissionContext, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            p = await asyncio.to_thread(ctx.drone.get_ground_truth_pose)
            t = await asyncio.to_thread(ctx.world.get_sim_time)
            ctx.recorder.log("trajectory", {"t": t, "pos": _xyz(p["translation"]), "q": list(_wxyz(p["rotation"]))})
        except Exception:
            pass
        await asyncio.sleep(1.0 / TRAJECTORY_HZ)


def load_world(client, scene: Dict[str, Any], sim_config_dir: str):
    """Create a projectairsim World for an in-memory scene dict.

    ``World(client)`` without a scene file still issues a request against the
    default scene topic, which fails if another scene is loaded; so build the
    object bare, load our scene, then do the rest of World's init against it.
    """
    from projectairsim import World

    world = World.__new__(World)
    world.client = client
    world.sim_config_path = sim_config_dir
    world.sim_instance_idx = -1
    world.sim_config = None
    world.home_geo_point = None
    world.load_scene(scene, delay_after_load_sec=2)
    world.import_ned_trajectory("null_trajectory", [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0])
    return world


def score_run_dir(run_dir: Path, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """(Re)compute metrics and scores for a recorded run; writes result.json.

    Without explicit ``weights``, the campaign weights stored in meta.json are used.
    """
    log = RunLog.load(run_dir)
    if weights is None:
        weights = log.meta.get("weights")
    m = vmetrics.compute_all(log)
    scores = layer_scores(m)
    scores["overall"] = overall_score({k: v for k, v in scores.items()}, m["safety"]["collisions"], weights)
    meta = log.meta
    result = {
        "run_id": meta["run_id"], "scenario_id": meta["scenario_id"], "agent": meta["agent"]["agent"],
        "agent_config": meta["agent"], "seed": meta["seed"], "params": meta["params"],
        "metrics": m, "scores": scores, "reasoning_summary": m.get("reasoning"),
    }
    with open(Path(run_dir) / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return result


async def execute_run(
    client,
    run: Run,
    agent_factory: Callable[[Run], Agent],
    sim_config_dir: str,
    out_dir: Path,
    weights: Optional[Dict[str, float]] = None,
    airsim_root: Optional[str] = None,
    agent: Optional[Agent] = None,
) -> Dict[str, Any]:
    from projectairsim import Drone

    agent = agent or agent_factory(run)
    run_dir = Path(out_dir) / agent.name / run.run_id
    rec = Recorder(run_dir)
    targets = run.resolved_targets()
    sc = run.spec["scene"]
    scene = build_scene(run, sim_config_dir, targets)
    cam = camera_settings(scene, sc.get("camera", "Payload"), sc.get("drone_name", "Drone1"))

    world = load_world(client, scene, sim_config_dir)
    drone = Drone(client, world, sc.get("drone_name", "Drone1"))
    client.subscribe(
        drone.robot_info["collision_info"],
        lambda _, msg: rec.log("collisions", {"t": msg.get("time_stamp"), "object_name": msg.get("object_name"),
                                              "impact_point": msg.get("impact_point")}),
    )
    apply_environment(world, run.spec.get("environment", {}))

    controller = AirSimController(drone, world.get_sim_time, rec.log)
    ctx = SimMissionContext(run, world, drone, sc.get("camera", "Payload"), rec, controller, cam["fov-degrees"])
    meta = {
        "run_id": run.run_id, "scenario_id": run.scenario_id, "seed": run.seed, "params": run.params,
        "mission": run.spec["mission"], "agent": agent.describe(), "targets": targets,
        "search_area": run.spec["search_area"], "limits": run.spec["limits"], "success": run.spec.get("success", {}),
        "start_ned": run.spec["dut_start"]["ned"], "t0_ns": ctx.t0_ns, "wall_start": datetime.now().isoformat(),
        "projectairsim_commit": projectairsim_commit(airsim_root), "scenario_source": run.source,
        "unreal": unreal_status(), "weights": weights or {},
        "config_hash": run_config_hash(run, agent.describe()),
    }
    rec.write_json("meta.json", meta)

    stop = asyncio.Event()
    poller = asyncio.create_task(_poll_trajectory(ctx, stop))
    end_reason = "completed"
    try:
        await asyncio.wait_for(agent.run(ctx), timeout=run.spec["limits"]["timeout_s"] * 2 + RTL_GRACE_S)
        if ctx.elapsed_s() >= ctx.mission.timeout_s:
            end_reason = "timeout"
    except asyncio.TimeoutError:
        end_reason = "hard_timeout"
    except Exception as e:
        end_reason = "error"
        rec.log("events", {"t": world.get_sim_time(), "kind": "error", "error": repr(e), "trace": traceback.format_exc()})
    finally:
        stop.set()
        await poller
        client.unsubscribe_all()  # before closing the recorder, so no late callback writes to it
        meta.update({"end_ns": world.get_sim_time(), "end_reason": end_reason, "wall_end": datetime.now().isoformat()})
        rec.write_json("meta.json", meta)
        rec.close()
    return score_run_dir(run_dir, weights)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


async def execute_campaign(
    campaign: Campaign,
    agent_factory: Callable[[Run], Agent],
    out_dir: Path,
    address: str = "127.0.0.1",
    resume: bool = True,
    airsim_root: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    from projectairsim import ProjectAirSimClient

    out = Path(out_dir) / campaign.name
    sim_cfg = campaign.sim_config_dir
    if not sim_cfg:
        raise ValueError("campaign must set sim_config_dir")
    client = ProjectAirSimClient(address=address)
    client.connect()
    results = []
    try:
        runs = campaign.runs()
        for i, run in enumerate(runs, 1):
            agent = agent_factory(run)
            name = agent.name
            run_dir = out / name / run.run_id
            done = run_dir / "result.json"
            if resume and done.exists():
                prev = _read_json(run_dir / "meta.json")
                if prev.get("config_hash") == run_config_hash(run, agent.describe()):
                    log(f"[{i}/{len(runs)}] {run.run_id} ({name}) already done, skipping")
                    results.append(json.loads(done.read_text(encoding="utf-8")))
                    continue
                log(f"[{i}/{len(runs)}] {run.run_id} ({name}) scenario or agent config changed, re-running")
            log(f"[{i}/{len(runs)}] {run.run_id} ({name}) params={run.params} seed={run.seed}")
            r = await execute_run(client, run, agent_factory, sim_cfg, out, campaign.weights, airsim_root, agent=agent)
            log(f"    -> success={r['metrics']['mission']['success']} overall={r['scores']['overall']:.1f}")
            results.append(r)
    finally:
        client.disconnect()
    return results
