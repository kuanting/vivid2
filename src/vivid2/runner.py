"""Execute runs against a live Project AirSim server and score them."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2 import metrics as vmetrics
from vivid2.preflight import PreflightError, run_preflight, scenario_map
from vivid2.radio import Radio, RadioProfile
from vivid2.recorder import Recorder, RunLog
from vivid2.scenario import Campaign, Run
from vivid2.scoring import layer_scores, overall_score, reasoning_score
from vivid2.sim.scene_builder import build_scene, camera_config, camera_settings, captures
from vivid2.sim.segmentation import SegmentationError, assign_ids, palette, split_seen, visibility
from vivid2.stack.base import (Agent, DistributedTeam, FleetAgent, Frame, GTAnnotation, MissionContext, MissionSpec,
                               TargetReport)
from vivid2.stack.reference.controllers import AirSimController
from vivid2.team_runtime import FLEET_VERSION, ClockState, SimFleetContext, TeamAgentError, TeamRuntime, run_tasks
from vivid2.viewer import make_viewer
from vivid2.terrain import TerrainModel, load_for_run
from vivid2.unreal_setup import status as unreal_status
from vivid2.vlm.base import FRAME_JPEG_MAX_SIDE, JPEG_QUALITY, encode_jpeg_bytes, sha256_hex

TRAJECTORY_HZ = 10.0
DUT_DEM_CELL_M = 20.0  # cell size of the terrain model the DUT is given
RTL_GRACE_S = 180.0
HARD_TIMEOUT_WALL_FACTOR = 3.0  # wall-clock safety net, in case the simulator stops answering
CAPTURE_MARGIN_S = 0.05  # a lockstep frame lets the simulator run the camera's capture-interval plus this
LOCKSTEP_REGISTER_S = 0.02  # wall time for the server to take the image request before the clock moves
LOCKSTEP_TIMEOUT_S = 15.0  # wall time to receive a lockstep frame (Project AirSim gives up after 5 s)
TEAM_REPLY_EXTRA_S = 5.0  # more wall time per extra drone in a team's photo batch


def _xyz(d: Dict[str, float]) -> List[float]:
    return [d["x"], d["y"], d["z"]]


def _wxyz(d: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (d["w"], d["x"], d["y"], d["z"])


def mission_spec(run: Run, start_ned=None, terrain=None, camera: Optional[Dict[str, Any]] = None,
                 datum_ned=None, drone_id: Optional[str] = None, team: Optional[List[Dict[str, Any]]] = None,
                 team_mode: Optional[str] = None) -> MissionSpec:
    """What the DUT is told. ``ground_down_m`` is the real ground under the start point.

    Scenarios author heights against flat ``search_area.ground_down_m``; on a map
    with relief that plane can be hundreds of metres off, so the DUT gets the
    terrain height at its start instead (flat maps: unchanged). ``altitude_m``
    is then height above the start's ground.

    In a team, ``start_ned`` is the drone's own pad and ``datum_ned`` the centre of the pads
    (``dut_start``): every drone measures heights from the ground there.
    """
    s = run.spec
    lim = s["limits"]
    ground = float(s["search_area"].get("ground_down_m", 0.0))
    datum = datum_ned if datum_ned is not None else start_ned
    if terrain is not None and datum is not None:
        ground = float(terrain.down_at(datum[0], datum[1]))
    # The DUT gets a coarse DEM, never the 1 m model the scoring uses.
    dut_terrain = terrain.downsample(DUT_DEM_CELL_M) if terrain is not None else TerrainModel.flat(ground)
    return MissionSpec(
        mission=s["mission"],
        brief=s["brief"],
        search_polygon_ne=s["search_area"]["polygon_ne"],
        altitude_m=s["search_area"]["altitude_m"],
        ground_down_m=ground,
        start_ned=list(start_ned) if start_ned is not None else s["dut_start"]["ned"],
        max_speed_mps=lim.get("max_speed_mps", 8.0),
        timeout_s=lim["timeout_s"],
        expected_targets=len(s["targets"]),
        min_agl_m=lim.get("min_agl_m", 0.5),
        geofence_margin_m=lim.get("geofence_margin_m", 20.0),
        energy_budget_m=lim.get("energy_budget_m"),
        camera=dict(camera) if camera else None,
        terrain=dut_terrain,
        drone_id=drone_id,
        team=team,
        team_mode=team_mode,
        min_separation_m=float(lim.get("min_separation_m", 5.0)) if team else None,
        no_fly_zones=[dict(z) for z in lim["no_fly"]] if lim.get("no_fly") else None,
    )


def serialize_requests(client) -> None:
    """Make the client's synchronous ``request()`` safe to call from several threads.

    Project AirSim's ``request()`` sends and receives on one shared nng REQ
    socket, and a second request from another thread cancels the one in flight
    (``pynng.exceptions.Canceled``). The runner polls the drone pose from a
    worker thread while the agent requests camera images, so the synchronous
    calls take turns. ``request_async`` already uses its own socket context.

    The lock is exposed as ``client.request_lock`` (reentrant), so a caller can
    make several requests back to back, e.g. a pose and its timestamp.
    """
    lock = threading.RLock()
    raw = client.request

    def request(request_data):
        with lock:
            return raw(request_data)

    client.request = request
    client.request_lock = lock


ASYNC_RECV_TIMEOUT_S = 6 * 3600.0


def allow_long_async_requests(client) -> None:
    """Give each async request (a move: takeoff, a whole survey path, landing) ASYNC_RECV_TIMEOUT_S.

    The client's socket gives up on a reply after 300 s of wall time, and an async request's context
    inherits that. A survey path is one request that lasts until the drone arrives, and with the
    simulator paused while frames are sent and recorded, its wall time is far longer than its
    mission time. Synchronous requests keep 300 s; ``run_agent`` stops a run that hangs.
    """
    sock = getattr(client, "socket_services", None)
    if sock is None:
        return
    from pynng import lib

    raw_new_context = sock.new_context

    def new_context():
        ctx = raw_new_context()
        lib.nng_ctx_set_ms(ctx._context[0], b"recv-timeout", int(ASYNC_RECV_TIMEOUT_S * 1000))
        return ctx

    sock.new_context = new_context


# Camera the live view (--view) shows until the first scored frame; robot configs without it get no preview.
PREVIEW_CAMERA = "PayloadPreview"

DEPTH_NO_DATA = 65504  # float16 max: the plugin's value where nothing was rendered (0 is also no data)


def detect_depth_scale(raw: np.ndarray, cam_height_m: float) -> Optional[float]:
    """Metres per raw unit of Project AirSim's 16UC1 planar depth, or None if undecidable.

    The encoding is documented as millimetres, but the UE 5.8 plugin build packs
    whole metres. Decide from a frame taken well above the ground: a mm-encoded
    image has valid values in the thousands, a metre-encoded one in the tens.
    """
    if cam_height_m < 5.0:
        return None
    valid = raw[(raw > 0) & (raw < DEPTH_NO_DATA)]
    if valid.size < raw.size // 2:
        return None
    return 0.001 if float(np.median(valid)) > 1000.0 else 1.0


# What the runner and the environment's settings do to a run (part of every run's config hash, so changing
# it re-runs on resume). 2: segmentation visible pixels; 3: exposure settled before the mission, contrast
# (fog) in visibility; 4: environment capped at 30 fps and Lumen's lighting range moved down for dusk;
# 5: the mission spec carries the camera, so agents no longer take a frame on the ground before takeoff
# (frame 1 is the first survey frame), and the payload camera's exposure settles at survey height;
# 6: payload frames in lockstep (the simulator runs one capture step per frame and is paused while the
# images are sent and recorded), agents pace frames by mission (sim) time, so the frames a run takes no
# longer depend on the machine's speed; a find is timed when it is reported.
SIM_VERSION = 6


def run_config_hash(run: Run, agent_desc: Dict[str, Any]) -> str:
    """Fingerprint of everything that determines a run's outcome.

    The run id stays (scenario, swept params, seed) so different agents can be
    paired on the same run; this hash also covers the resolved scenario spec, the
    agent's configuration and ``SIM_VERSION``, so changing any of them forces a
    re-run on resume.
    """
    fp: Dict[str, Any] = {"spec": run.spec, "seed": run.seed, "agent": agent_desc, "sim": SIM_VERSION}
    if run.fleet_size > 1:
        fp["fleet_rt"] = FLEET_VERSION  # team runs only: single-drone hashes stay as they were
    blob = json.dumps(fp, sort_keys=True, default=str)
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


EXPOSURE_MAX_CAPTURES = 20
EXPOSURE_PERIOD_S = 0.5
EXPOSURE_SETTLED_GREY = 1.5  # spread of mean grey level (0-255) over the last 3 captures that counts as settled


def capture_step_ns(camera: Dict[str, Any]) -> int:
    """Mission time a lockstep frame lets the simulator run: the camera's capture-interval plus a margin.

    The camera captures at the first rendered pose more than capture-interval after its last capture,
    so a step this long always yields a frame, however close together the agent asks for them.
    """
    return int(round((float(camera.get("capture-interval", 0.2)) + CAPTURE_MARGIN_S) * 1e9))


def lifted_pose(camera: Dict[str, Any], lift_m: float) -> Dict[str, Dict[str, float]]:
    """A camera's configured mount (relative to the drone, NED), raised ``lift_m`` straight up.

    Rotation as configured (``rpy-deg``, applied yaw-pitch-roll), computed here because the
    client's ``rpy_to_quaternion`` clips pitch at 89.9 degrees and a payload camera looks
    straight down.
    """
    origin = camera.get("origin", {})
    x, y, z = (float(v) for v in origin.get("xyz", "0 0 0").split())
    r, p, yw = (np.radians(float(v)) / 2.0 for v in origin.get("rpy-deg", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yw), np.sin(yw)
    q = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    return {"translation": {"x": x, "y": y, "z": z - float(lift_m)},
            "rotation": {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}}


async def settle_exposure(drone, cameras: Sequence[str],
                          views: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Capture from each camera until its auto exposure has settled to this run's light and fog.

    A camera's exposure state survives from the previous run in the same Unreal session, and it
    only adapts when the camera captures. Without this, the first frames of a run are too bright
    or too dark depending on what ran before (the very first one is sometimes black). Called
    before the mission clock starts; the frames are thrown away.

    ``views`` moves a camera for the warm-up (camera id -> ``{"pose": lifted_pose(...), "lift_m": m}``)
    and puts it back on its mount afterwards. The payload camera warms up at survey height: on
    the ground it sits centimetres above the terrain, renders flat grey, and would settle to that,
    leaving the first survey frames too dark at dusk.
    """
    from projectairsim.types import ImageType, Pose, Quaternion, Vector3

    out: Dict[str, Dict[str, Any]] = {}
    for cam in cameras:
        view = (views or {}).get(cam)
        if view:
            pose = Pose({"translation": Vector3(view["pose"]["translation"]),
                         "rotation": Quaternion(view["pose"]["rotation"])})
            await asyncio.to_thread(drone.set_camera_pose, cam, pose, True)
        greys: List[float] = []
        settled = False
        try:
            for _ in range(EXPOSURE_MAX_CAPTURES):
                imgs = await asyncio.to_thread(drone.get_images, cam, [ImageType.SCENE])
                msg = imgs.get(ImageType.SCENE)
                if msg:
                    greys.append(float(_decode(msg, np.uint8, 3).mean()))
                    if len(greys) >= 3 and greys[-1] > 1.0 and max(greys[-3:]) - min(greys[-3:]) < EXPOSURE_SETTLED_GREY:
                        settled = True
                        break
                await asyncio.sleep(EXPOSURE_PERIOD_S)
        finally:
            if view:
                await asyncio.to_thread(drone.reset_camera_pose, cam, True)
        out[cam] = {"captures": len(greys), "settled": settled, "mean_grey": [round(g, 1) for g in greys],
                    **({"lifted_m": view["lift_m"]} if view else {})}
    return out


@functools.lru_cache(maxsize=1)
def host_info() -> Dict[str, Any]:
    """The machine VIVID 2 and the agent ran on (the simulator may run on another: ``sim_address``).

    Agents think in mission time, so a model's speed on this hardware (or its API's) is part of a result.
    """
    info: Dict[str, Any] = {"node": platform.node(), "os": platform.platform(), "python": platform.python_version(),
                            "cpu": platform.processor(), "cpus": os.cpu_count()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            info["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def projectairsim_commit(airsim_root: Optional[str]) -> Optional[str]:
    if not airsim_root:
        return None
    try:
        return subprocess.check_output(["git", "-C", airsim_root, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


FRAME_BANK_MODES = ("off", "rgb", "rgbd")


def write_bank_files(run_dir: Path, frame_id: int, jpeg: bytes, depth_m: Optional[np.ndarray]) -> Dict[str, Any]:
    """Save one frame to the run's frame bank; returns the ``frames`` record fields that point at it.

    RGB is the exact JPEG the VLM is sent. Depth (``rgbd``) is stored at half
    resolution as 16-bit millimetres (0 = no data).
    """
    from PIL import Image

    bank = Path(run_dir) / "frames"
    bank.mkdir(exist_ok=True)
    rel = f"frames/{frame_id:06d}.jpg"
    (Path(run_dir) / rel).write_bytes(jpeg)
    out: Dict[str, Any] = {"rgb": rel, "rgb_sha256": sha256_hex(jpeg)}
    if depth_m is not None:
        mm = np.clip(np.nan_to_num(depth_m[::2, ::2]) * 1000.0, 0, 65535).astype(np.uint16)
        drel = f"frames/{frame_id:06d}_d.png"
        Image.fromarray(mm).save(Path(run_dir) / drel)
        out["depth"] = drel
    return out


def _shared(name: str):
    """A context attribute that lives in the run's clock state (shared by a team's drones)."""
    return property(lambda self: getattr(self.clock, name), lambda self, v: setattr(self.clock, name, v))


class SimMissionContext(MissionContext):
    """One drone's view of a live run. ``team`` (a TeamRuntime) when it flies in a team."""

    t0_ns = _shared("t0_ns")
    last_t_ns = _shared("last_t_ns")
    depth_scale = _shared("depth_scale")
    lockstep = _shared("lockstep")
    pause_to_record = _shared("pause_to_record")
    capture_step_ns = _shared("capture_step_ns")

    def __init__(self, run: Run, world, drone, camera_id: str, recorder: Recorder, controller: AirSimController,
                 fov_deg: float, frame_bank: str = "off", seg_ids: Optional[Dict[str, int]] = None,
                 target_ids: Optional[List[str]] = None, start_ned=None, terrain=None, viewer=None,
                 camera: Optional[Dict[str, Any]] = None, drone_id: Optional[str] = None,
                 team: Optional[TeamRuntime] = None, datum_ned=None, team_info=None, team_mode: Optional[str] = None,
                 fire_ids: Optional[List[str]] = None):
        self.viewer = viewer  # optional live camera windows (vivid2.viewer.RunViewer)
        self.preview_topic: Optional[str] = None  # the viewer's live preview, unsubscribed at the first frame
        if frame_bank not in FRAME_BANK_MODES:
            raise ValueError(f"frame_bank must be one of {FRAME_BANK_MODES}")
        self.frame_bank = frame_bank
        self.seg_ids = seg_ids or {}  # segmentation ID per target/distractor; empty = no visibility ground truth
        self.target_ids = set(target_ids or [])
        self.fire_ids = set(fire_ids or [])  # fires: recorded as fires_visible, never as distractors
        self._seg_colours = palette() if self.seg_ids else None
        self.run = run
        self.world = world
        self.drone = drone
        self.camera_id = camera_id
        self.recorder = recorder
        self.controller = controller
        self.fov_deg = fov_deg
        self.mission = mission_spec(run, start_ned, terrain, camera, datum_ned=datum_ned, drone_id=drone_id,
                                    team=team_info, team_mode=team_mode)
        self.team = team
        self.drone_id = drone_id or "Drone1"
        # The clock (t0, frame ids, lockstep...): the team's, or this drone's own.
        self.clock = team if team is not None else ClockState(world.get_sim_time())
        self._nav_fov: Dict[str, float] = {}  # navigation camera -> FOV, filled by the runner
        self.reports: List[TargetReport] = []
        self._stop = False

    # -- MissionContext API -------------------------------------------------
    async def get_frame(self) -> Optional[Frame]:
        """The next payload frame; it costs one capture step of mission time on any machine.

        Lockstep: pause the simulator, ask for the images, let it run exactly ``capture_step_ns``
        (the camera captures in that time), then receive and record the images while it stays
        paused. Unreal takes a second or more to read back and send three 1920x1440 images, and
        how long depends on the machine; paused, that costs no mission time. Nor does VIVID 2's own
        work (ground truth, encoding, saving). If a lockstep frame fails, frames are fetched with the
        simulator running from then on (machine-dependent, logged once) and only that work is paused.
        """
        from projectairsim.types import ImageType

        if self.team is not None:
            return await self.team.get_frame(self)  # in the team's next batch
        types = self.image_types()
        imgs = None
        paused = False
        if self.lockstep:
            try:
                t_request, t_fetched, imgs = await self._fetch_lockstep(types)
                paused = True
            except Exception as e:
                self.lockstep = False
                self.recorder.log("events", {"t": self.last_t_ns, "kind": "lockstep_unavailable", "error": repr(e)})
                with contextlib.suppress(Exception):
                    await self._resume()
        if imgs is None:
            t_request = self.sim_time_or_last()
            imgs = await asyncio.to_thread(self.drone.get_images, self.camera_id, types)
            paused = await self._pause()
            t_fetched = self.sim_time_or_last()
        wall0 = time.monotonic()
        try:
            rgb_msg = imgs.get(ImageType.SCENE)
            if not rgb_msg:
                return None
            return await self._record_frame(imgs, rgb_msg, t_request, t_fetched, wall0)
        finally:
            if paused:
                await self._resume()

    def image_types(self) -> list:
        from projectairsim.types import ImageType

        return [ImageType.SCENE, ImageType.DEPTH_PLANAR] + ([ImageType.SEGMENTATION] if self.seg_ids else [])

    async def _running_frame(self) -> Optional[Frame]:
        """A photo fetched with the simulator running, then recorded paused (a team after lockstep failed)."""
        from projectairsim.types import ImageType

        t_request = self.sim_time_or_last()
        imgs = await asyncio.to_thread(self.drone.get_images, self.camera_id, self.image_types())
        paused = await self._pause()
        t_fetched = self.sim_time_or_last()
        wall0 = time.monotonic()
        try:
            rgb_msg = imgs.get(ImageType.SCENE)
            if not rgb_msg:
                return None
            if self.team is not None:
                self.team.frame_id += 1
                self.team.frame_owner[self.team.frame_id] = self.drone_id
                return await self._record_frame(imgs, rgb_msg, t_request, t_fetched, wall0, frame_id=self.team.frame_id)
            return await self._record_frame(imgs, rgb_msg, t_request, t_fetched, wall0)
        finally:
            if paused:
                await self._resume()

    async def _fetch_lockstep(self, types) -> Tuple[int, int, Dict[int, Any]]:
        """Pause, request the images, run one capture step, receive them paused.

        Returns (sim time at the request, sim time after the step, images by type). The simulator
        is left paused; the caller resumes it.
        """
        w = self.world
        client = w.client
        await asyncio.to_thread(w.pause)
        t_request = int(await asyncio.to_thread(w.get_sim_time))
        req = {"method": f"{self.drone.sensors_topic}/{self.camera_id}/GetImages",
               "params": {"image_type_ids": [int(t) for t in types]}, "version": 1.0}
        # Its own socket context, as request_async uses, but without request_async's done-callback,
        # which disconnects the client when a request fails.
        sock = client.socket_services.new_context()
        reply = None
        try:
            await sock.asend(client.preprocess_request(req))
            reply = asyncio.ensure_future(sock.arecv_msg())
            await asyncio.sleep(LOCKSTEP_REGISTER_S)
            t_fetched = int(await asyncio.to_thread(w.continue_for_sim_time, self.capture_step_ns, True))
            msg = await asyncio.wait_for(asyncio.shield(reply), timeout=LOCKSTEP_TIMEOUT_S)
        finally:
            if reply is not None and not reply.done():
                reply.cancel()
            sock.close()
        data = client.postprocess_response(msg)
        return t_request, t_fetched, {int(k): v for k, v in data.items()}

    async def _pause(self) -> bool:
        if not self.pause_to_record:
            return False
        try:
            await asyncio.to_thread(self.world.pause)
            if self.team is not None:
                self.team.ledger.paused()
            return True
        except Exception as e:  # e.g. a real-time clock, which cannot pause
            self.pause_to_record = False
            self.recorder.log("events", {"t": self.last_t_ns, "kind": "pause_unavailable", "error": repr(e)})
            return False

    async def _resume(self) -> None:
        await asyncio.to_thread(self.world.resume)
        if self.team is not None:
            self.team.ledger.resumed()

    async def _record_frame(self, imgs, rgb_msg, t_request: int, t_fetched: int, wall0: float,
                            frame_id: Optional[int] = None, deferred: Optional[list] = None,
                            batch: Optional[int] = None) -> Frame:
        """Decode, add ground truth, encode and log one photo (the simulator is paused).

        A team batch passes the ``frame_id`` it assigned and a ``deferred`` list, which collects
        the ``frames`` record so the batch can write its records in frame order.
        """
        from projectairsim.types import ImageType

        rgb = _decode(rgb_msg, np.uint8, 3)
        depth = None
        dmsg = imgs.get(ImageType.DEPTH_PLANAR)
        if dmsg:
            raw = _decode(dmsg, np.uint16, 1)
            if self.depth_scale is None:
                self.depth_scale = detect_depth_scale(raw, self.mission.ground_down_m - rgb_msg["pos_z"])
                if self.depth_scale is not None:
                    self.recorder.log("events", {"t": int(rgb_msg["time_stamp"]), "kind": "depth_units",
                                                 "metres_per_unit": self.depth_scale})
            if self.depth_scale is not None:  # until then (on the ground) agents fall back to flat-ground rays
                depth = raw.astype(np.float32) * self.depth_scale
                depth[(raw == 0) | (raw >= DEPTH_NO_DATA)] = 0.0  # no data
        vis: Dict[str, Dict[str, Any]] = {}
        smsg = imgs.get(ImageType.SEGMENTATION) if self.seg_ids else None
        seg = None
        if smsg:
            seg = _decode(smsg, np.uint8, 3)
            # the colour image too: fog does not show in segmentation, only in the object's contrast
            vis = await asyncio.to_thread(visibility, seg, self.seg_ids, self._seg_colours,
                                          rgb if rgb.shape[:2] == seg.shape[:2] else None)
        gt = []
        for a in rgb_msg.get("annotations") or []:
            c, s = a["bbox2d"]["center"], a["bbox2d"]["size"]
            v = vis.get(a["object_id"], {})
            gt.append(GTAnnotation(a["object_id"], (c["x"] - s["x"] / 2, c["y"] - s["y"] / 2, c["x"] + s["x"] / 2, c["y"] + s["y"] / 2),
                                   v.get("visible_px"), tuple(v["bbox_vis"]) if v.get("bbox_vis") else None,
                                   v.get("contrast")))
        annotated = {g.object_id for g in gt}
        for oid, v in vis.items():  # seen in segmentation but outside the annotation's frustum test
            if oid in self.target_ids and oid not in annotated and v["visible_px"]:
                gt.append(GTAnnotation(oid, tuple(v["bbox_vis"]), v["visible_px"], tuple(v["bbox_vis"]), v.get("contrast")))
        distractors_seen, fires_seen = split_seen(vis, self.target_ids, self.fire_ids)
        if frame_id is None:
            self.clock.frame_id += 1
            frame_id = self.clock.frame_id
        # encoded once at camera resolution; what a VLM is sent is derived from these bytes (vlm_views)
        jpeg = await asyncio.to_thread(encode_jpeg_bytes, rgb, FRAME_JPEG_MAX_SIDE)
        frame = Frame(
            t_ns=int(rgb_msg["time_stamp"]), frame_id=frame_id, rgb=rgb, depth=depth,
            cam_pos=np.array([rgb_msg["pos_x"], rgb_msg["pos_y"], rgb_msg["pos_z"]]),
            cam_q=(rgb_msg["rot_w"], rgb_msg["rot_x"], rgb_msg["rot_y"], rgb_msg["rot_z"]),
            fov_deg=self.fov_deg, gt=gt, jpeg=jpeg,
        )
        bank: Dict[str, Any] = {}
        if self.frame_bank != "off":
            bank = await asyncio.to_thread(write_bank_files, self.recorder.run_dir, frame.frame_id, jpeg,
                                           depth if self.frame_bank == "rgbd" else None)
        record = {
            "t": frame.t_ns, "frame_id": frame.frame_id,
            **({"drone": self.drone_id, "batch": batch} if self.team is not None else {}),
            "cam_pos": frame.cam_pos, "cam_q": list(frame.cam_q),
            "width": frame.width, "height": frame.height, "fov_deg": frame.fov_deg,
            "gt": [_gt_record(g) for g in gt], **bank,
            **({"distractors_visible": distractors_seen} if distractors_seen else {}),
            **({"fires_visible": fires_seen} if fires_seen else {}),
            # mission time the frame took (one capture step in lockstep), and VIVID 2's work on it (paused)
            "capture_s": round(max(0, t_fetched - t_request) / 1e9, 3),
            "record_wall_s": round(time.monotonic() - wall0, 3),
        }
        if deferred is not None:
            deferred.append((frame.frame_id, record))
        else:
            self.recorder.log("frames", record)
        if self.viewer is not None:
            extra = f"t {self.elapsed_s():.0f} s  height {self.mission.ground_down_m - frame.cam_pos[2]:.0f} m"
            if self.team is not None:
                self.viewer.show_frame(frame, seg=seg, extra=f"{self.drone_id}  {extra}", drone=self.drone_id)
            else:
                self.viewer.show_frame(frame, seg=seg, extra=extra)
        if self.preview_topic is not None:
            # The test has started: stop the live preview, so Unreal stops rendering its camera.
            topic, self.preview_topic = self.preview_topic, None
            try:
                await asyncio.to_thread(self.world.client.unsubscribe, topic)
            except Exception:
                pass
        return frame

    async def get_nav_frame(self, camera: str = "Front") -> Optional[Frame]:
        from projectairsim.types import ImageType

        if camera == self.camera_id:
            raise ValueError("use get_frame() for the payload camera; navigation frames are not scored")
        if self.team is not None:
            async with self.team.nav_gate():
                imgs = await asyncio.to_thread(self.drone.get_images, camera, [ImageType.SCENE, ImageType.DEPTH_PLANAR])
        else:
            imgs = await asyncio.to_thread(self.drone.get_images, camera, [ImageType.SCENE, ImageType.DEPTH_PLANAR])
        rgb_msg = imgs.get(ImageType.SCENE)
        if not rgb_msg:
            return None
        depth = None
        dmsg = imgs.get(ImageType.DEPTH_PLANAR)
        if dmsg:
            raw = _decode(dmsg, np.uint16, 1)
            scale = self.depth_scale if self.depth_scale is not None else 1.0  # this plugin build packs whole metres
            depth = raw.astype(np.float32) * scale
            depth[(raw == 0) | (raw >= DEPTH_NO_DATA)] = 0.0
        self.clock.nav_frames += 1
        return Frame(t_ns=int(rgb_msg["time_stamp"]), frame_id=-self.clock.nav_frames, rgb=_decode(rgb_msg, np.uint8, 3),
                     cam_pos=np.array([rgb_msg["pos_x"], rgb_msg["pos_y"], rgb_msg["pos_z"]]),
                     cam_q=(rgb_msg["rot_w"], rgb_msg["rot_x"], rgb_msg["rot_y"], rgb_msg["rot_z"]),
                     fov_deg=float(self._nav_fov.get(camera, 90.0)), depth=depth)

    def pose(self):
        p = self.drone.get_ground_truth_pose()
        return np.asarray(_xyz(p["translation"])), _wxyz(p["rotation"])

    async def sleep(self, seconds: float) -> None:
        if self.team is not None:  # a batch of photos starting just before may wake it early
            await self.team.sleep_until(self.elapsed_s() + seconds, self.should_stop)
            return
        # The simulator runs at about real time between frames, so a wall sleep gets close; then top up.
        target = self.elapsed_s() + seconds
        while not self.should_stop():
            left = target - self.elapsed_s()
            if left <= 0.001:
                return
            await asyncio.sleep(min(left, 1.0))

    async def think(self, fn, *args, **kwargs):
        if self.team is None:
            return await super().think(fn, *args, **kwargs)
        w0 = time.monotonic()
        result = await super().think(fn, *args, **kwargs)
        w1 = time.monotonic()
        # Wall time the clock stood still (other drones' photos) while this drone was thinking.
        paused = self.team.ledger.overlap(w0, w1)
        st = self.team.think_stats.setdefault(self.drone_id, {"calls": 0, "wall_s": 0.0, "paused_s": 0.0})
        st["calls"] += 1
        st["wall_s"] += w1 - w0
        st["paused_s"] += paused
        if paused > 0:
            await self.sleep(paused)
        return result

    def report_target(self, report: TargetReport) -> None:
        self.reports.append(report)
        # "t" is the frame the find was made on; "t_report" when the agent reported it (after thinking)
        self.recorder.log("reports", {"t": report.t_ns, "t_report": self.sim_time_or_last(), "frame_id": report.frame_id,
                                      **({"drone": self.drone_id} if self.team is not None else {}),
                                      "ned": report.ned, "confidence": report.confidence, "source": report.source,
                                      "note": report.note})
        if self.viewer is not None:
            self.viewer.note("reports", {"frame_id": report.frame_id})

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        if self.team is not None:
            record = {**record, "drone": self.drone_id}
        self.recorder.log(stream, record)
        if self.viewer is not None and stream in ("detections", "decisions"):
            self.viewer.note(stream, record)

    def should_stop(self) -> bool:
        return self._stop or self.clock.stop or self.elapsed_s() >= self.mission.timeout_s

    def elapsed_s(self) -> float:
        return (self.world.get_sim_time() - self.t0_ns) / 1e9

    def sim_time_or_last(self) -> int:
        """Sim time, or the last one seen if the sim no longer answers."""
        try:
            self.last_t_ns = self.world.get_sim_time()
        except Exception:
            pass
        return self.last_t_ns


def _gt_record(g: GTAnnotation) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"object_id": g.object_id, "bbox": list(g.bbox)}
    if g.visible_px is not None:
        rec["visible_px"] = g.visible_px
        rec["bbox_vis"] = list(g.bbox_vis) if g.bbox_vis else None
    if g.contrast is not None:
        rec["contrast"] = round(g.contrast, 2)
    return rec


def _decode(msg: Dict[str, Any], dtype, channels: int) -> np.ndarray:
    data = msg["data"]
    buf = np.array(data, dtype=np.uint8) if isinstance(data, list) else np.frombuffer(data, dtype=np.uint8)
    arr = buf.view(dtype)
    shape = (msg["height"], msg["width"]) if channels == 1 else (msg["height"], msg["width"], channels)
    return arr.reshape(shape)


async def run_agent(agent: Agent, ctx: MissionContext, hard_s: float) -> bool:
    """Run the agent; stop it if it overruns ``hard_s`` of mission time. True if it was stopped.

    Mission time, because the simulator is paused while frames are recorded. A wall-clock limit
    (HARD_TIMEOUT_WALL_FACTOR x) still catches a simulator that stops answering. The agent's own
    errors are raised.
    """
    task = asyncio.create_task(agent.run(ctx))
    wall_end = time.monotonic() + HARD_TIMEOUT_WALL_FACTOR * hard_s
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if done:
                task.result()
                return False
            if ctx.elapsed_s() > hard_s or time.monotonic() > wall_end:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                return True
    finally:
        if not task.done():
            task.cancel()


async def _poll_trajectory(ctx: SimMissionContext, stop: asyncio.Event) -> None:
    lock = getattr(ctx.world.client, "request_lock", None) or contextlib.nullcontext()

    def pose_and_time():
        # Back to back under the request lock: a camera fetch slipping in between
        # would skew the timestamp by ~0.5 s and fake a speed spike.
        with lock:
            return ctx.drone.get_ground_truth_pose(), ctx.world.get_sim_time()

    last = None
    while not stop.is_set():
        try:
            p, t = await asyncio.to_thread(pose_and_time)
            ctx.last_t_ns = t
            if last is not None and t <= last:  # paused while a frame is recorded
                await asyncio.sleep(1.0 / TRAJECTORY_HZ)
                continue
            last = t
            ctx.recorder.log("trajectory", {"t": t, "pos": _xyz(p["translation"]), "q": list(_wxyz(p["rotation"]))})
            if ctx.viewer is not None and ctx.viewer.live:
                ctx.viewer.set_progress(ctx.mission.ground_down_m - p["translation"]["z"], (t - ctx.t0_ns) / 1e9)
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
    if log.meta.get("mode") == "open_loop":
        return _score_open_loop(run_dir, log)
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
        **({"team": {"mode": meta["team"].get("mode"), "size": meta["team"].get("size")}} if meta.get("team") else {}),
    }
    with open(Path(run_dir) / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return result


def _score_open_loop(run_dir: Path, log: RunLog) -> Dict[str, Any]:
    """Open-loop runs (saved frames replayed through a reasoner) score the reasoning layer only."""
    r = vmetrics.reasoning_metrics(log)
    score = reasoning_score(r)
    meta = log.meta
    result = {
        "mode": "open_loop", "run_id": meta["run_id"], "scenario_id": meta["scenario_id"],
        "agent": meta["agent"]["agent"], "agent_config": meta["agent"], "seed": meta["seed"], "params": meta["params"],
        "source_run": meta.get("source_run"), "metrics": {"reasoning": r},
        "scores": {"reasoning": score, "overall": score}, "reasoning_summary": r,
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
    frame_bank: str = "off",
    viewer=None,
) -> Dict[str, Any]:
    from projectairsim import Drone

    agent = agent or agent_factory(run)
    if run.fleet_size > 1 or not isinstance(agent, Agent):
        if isinstance(agent, Agent):
            raise ValueError(f"{run.run_id} flies a team of {run.fleet_size}: it needs a team agent "
                             "(vivid2 run --team central|dist)")
        return await execute_team_run(client, run, agent, sim_config_dir, out_dir, weights, airsim_root,
                                      frame_bank, viewer)
    if viewer is not None:
        viewer.start_run(f"{run.run_id}  {agent.name}")
    run_dir = Path(out_dir) / agent.name / run.run_id
    terrain, terrain_meta = load_for_run(sim_config_dir, run.spec)
    targets = run.resolved_targets(terrain)
    distractors = run.resolved_distractors(terrain)
    fires = run.resolved_fires(terrain)
    start_ned = run.resolved_start(terrain)
    sc = run.spec["scene"]
    scene = build_scene(run, sim_config_dir, targets, distractors, start_ned, fires=fires)
    cam = camera_settings(scene, sc.get("camera", "Payload"), sc.get("drone_name", "Drone1"))

    try:
        world = load_world(client, scene, sim_config_dir)
    except Exception as e:
        raise PreflightError(
            f"loading the scene failed ({e!r}). Check that map {sc.get('map', '?')!r} is open and that it has "
            f"the assets {sorted({t['asset'] for t in targets + fires})}."
        ) from e
    preflight = run_preflight(world, run.spec, targets + distractors + fires)  # raises PreflightError before anything is recorded
    seg_ids: Dict[str, int] = {}
    if captures(scene, sc.get("camera", "Payload"), 3, sc.get("drone_name", "Drone1")):
        try:
            seg_ids = assign_ids(world, targets, distractors, fires)
        except SegmentationError as e:
            raise PreflightError(str(e)) from e

    rec = Recorder(run_dir)
    drone = Drone(client, world, sc.get("drone_name", "Drone1"))
    client.subscribe(
        drone.robot_info["collision_info"],
        lambda _, msg: rec.log("collisions", {"t": msg.get("time_stamp"), "object_name": msg.get("object_name"),
                                              "impact_point": msg.get("impact_point")}),
    )
    apply_environment(world, run.spec.get("environment", {}))
    preview = None
    if viewer is not None:
        sensors = getattr(drone, "sensors", {})
        chase = sensors.get("Chase", {}).get("scene_camera")
        if chase:
            client.subscribe(chase, viewer.chase_callback())
        # Live preview of the payload view until the first scored frame (see vivid2.viewer).
        preview = sensors.get(PREVIEW_CAMERA, {}).get("scene_camera")
        if preview:
            client.subscribe(preview, viewer.preview_callback())
        else:
            viewer.no_preview()
        viewer.set_phase("on the ground, settling camera exposure (this camera sees nothing until takeoff)")
    cameras = [sc.get("camera", "Payload")]
    for nav in ("Front",):
        try:
            camera_settings(scene, nav, sc.get("drone_name", "Drone1"))
            cameras.append(nav)
        except StopIteration:
            pass  # robot config without that camera
    # Before the mission clock starts. The payload camera warms up lifted to the survey height, looking
    # at the forest the survey will see; the Front camera warms up where it is.
    lift = float(run.spec["search_area"]["altitude_m"])
    payload_cam = sc.get("camera", "Payload")
    views = {payload_cam: {"pose": lifted_pose(camera_config(scene, payload_cam, sc.get("drone_name", "Drone1")), lift),
                           "lift_m": lift}}
    exposure = await settle_exposure(drone, cameras, views)

    controller = AirSimController(drone, world.get_sim_time, rec.log)
    ctx = SimMissionContext(run, world, drone, sc.get("camera", "Payload"), rec, controller, cam["fov-degrees"], frame_bank,
                            seg_ids=seg_ids, target_ids=[t["id"] for t in targets], start_ned=start_ned,
                            terrain=terrain, viewer=viewer, fire_ids=[f["id"] for f in fires],
                            camera={"width": int(cam["width"]), "height": int(cam["height"]),
                                    "fov_deg": float(cam["fov-degrees"])})
    ctx.preview_topic = preview
    ctx.capture_step_ns = capture_step_ns(camera_config(scene, sc.get("camera", "Payload"), sc.get("drone_name", "Drone1")))
    if viewer is not None:
        viewer.set_phase("mission clock started: taking off and climbing")
    for nav in ("Front",):
        try:
            ctx._nav_fov[nav] = float(camera_settings(scene, nav, sc.get("drone_name", "Drone1"))["fov-degrees"])
        except StopIteration:
            pass  # robot config without that camera
    meta = {
        "run_id": run.run_id, "scenario_id": run.scenario_id, "seed": run.seed, "params": run.params,
        "mission": run.spec["mission"], "agent": agent.describe(), "targets": targets,
        "search_area": run.spec["search_area"], "limits": run.spec["limits"], "success": run.spec.get("success", {}),
        "start_ned": start_ned, "terrain": terrain_meta, "t0_ns": ctx.t0_ns, "wall_start": datetime.now().isoformat(),
        "projectairsim_commit": projectairsim_commit(airsim_root), "scenario_source": run.source,
        "unreal": unreal_status(), "host": host_info(), "sim_address": getattr(client, "address", None),
        "weights": weights or {},
        "config_hash": run_config_hash(run, agent.describe()),
        "map": sc.get("map"), "preflight": preflight,
        "mission_spec": ctx.mission.record(),  # what the DUT was told; open-loop evaluation reuses it
        "distractors": distractors, **({"fires": fires} if fires else {}), "segmentation_ids": seg_ids,
        # 2: ground truth carries per-target visible pixels; 3: and contrast in the colour image (fog)
        "gt_version": 3 if seg_ids else 1,
        "sim_version": SIM_VERSION, "exposure": exposure,
        "frame_bank": {"mode": frame_bank, "jpeg_quality": JPEG_QUALITY, "jpeg_max_side": FRAME_JPEG_MAX_SIDE,
                       "depth": "uint16 mm, half resolution, 0 = no data"},
    }
    rec.write_json("meta.json", meta)

    stop = asyncio.Event()
    poller = asyncio.create_task(_poll_trajectory(ctx, stop))
    end_reason = "completed"
    try:
        if await run_agent(agent, ctx, run.spec["limits"]["timeout_s"] * 2 + RTL_GRACE_S):
            end_reason = "hard_timeout"
        elif ctx.elapsed_s() >= ctx.mission.timeout_s:
            end_reason = "timeout"
    except Exception as e:
        end_reason = "error"
        rec.log("events", {"t": ctx.sim_time_or_last(), "kind": "error", "error": repr(e), "trace": traceback.format_exc()})
    finally:
        stop.set()
        await poller
        client.unsubscribe_all()  # before closing the recorder, so no late callback writes to it
        meta.update({"end_ns": ctx.sim_time_or_last(), "end_reason": end_reason, "wall_end": datetime.now().isoformat(),
                     "depth_metres_per_unit": ctx.depth_scale,
                     # lockstep stays True unless a frame failed (see the lockstep_unavailable event)
                     "frame_timing": {"lockstep": ctx.lockstep, "capture_step_s": ctx.capture_step_ns / 1e9,
                                      "pause_to_record": ctx.pause_to_record}})
        rec.write_json("meta.json", meta)
        rec.close()
    return score_run_dir(run_dir, weights)


async def execute_team_run(client, run: Run, agent, sim_config_dir: str, out_dir: Path,
                           weights: Optional[Dict[str, float]] = None, airsim_root: Optional[str] = None,
                           frame_bank: str = "off", viewer=None) -> Dict[str, Any]:
    """A run with a team of drones: a FleetAgent (central) or a DistributedTeam (one agent per drone).

    Like ``execute_run``, plus: one drone per launch pad, photos taken in shared lockstep batches
    (``vivid2.team_runtime``), and every record tagged with its ``drone``.
    """
    from projectairsim import Drone

    if not isinstance(agent, (FleetAgent, DistributedTeam)):
        raise TypeError(f"a team run needs a FleetAgent or a DistributedTeam, not {type(agent).__name__}")
    if viewer is not None:
        viewer.start_run(f"{run.run_id}  {agent.name}")
    run_dir = Path(out_dir) / agent.name / run.run_id
    terrain, terrain_meta = load_for_run(sim_config_dir, run.spec)
    targets = run.resolved_targets(terrain)
    distractors = run.resolved_distractors(terrain)
    fires = run.resolved_fires(terrain)
    centre = run.resolved_start(terrain)  # the pads sit around it; the team's height datum
    fleet = run.resolved_fleet(terrain)
    ids = [d["id"] for d in fleet]
    if isinstance(agent, DistributedTeam) and len(agent.agents) != len(fleet):
        raise ValueError(f"{agent.name}: {len(agent.agents)} agents for a team of {len(fleet)}")
    sc = run.spec["scene"]
    camera_id = sc.get("camera", "Payload")
    scene = build_scene(run, sim_config_dir, targets, distractors, centre, fleet=fleet, fires=fires)
    cam = camera_settings(scene, camera_id, ids[0])

    try:
        world = load_world(client, scene, sim_config_dir)
    except Exception as e:
        raise PreflightError(
            f"loading the scene failed ({e!r}). Check that map {sc.get('map', '?')!r} is open and that it has "
            f"the assets {sorted({t['asset'] for t in targets + fires})}."
        ) from e
    preflight = run_preflight(world, run.spec, targets + distractors + fires)
    seg_ids: Dict[str, int] = {}
    if captures(scene, camera_id, 3, ids[0]):
        try:
            seg_ids = assign_ids(world, targets, distractors, fires)
        except SegmentationError as e:
            raise PreflightError(str(e)) from e

    rec = Recorder(run_dir)
    drones = {i: Drone(client, world, i) for i in ids}
    for name, drone in drones.items():
        def on_collision(_, msg, name=name):  # the name bound now: each drone's own callback
            other = msg.get("object_name")
            rec.log("collisions", {"t": msg.get("time_stamp"), "object_name": other,
                                   "impact_point": msg.get("impact_point"), "drone": name,
                                   **({"other_drone": other} if other in drones else {})})
        client.subscribe(drone.robot_info["collision_info"], on_collision)
    apply_environment(world, run.spec.get("environment", {}))
    lead = drones[ids[0]]
    preview = None
    if viewer is not None:  # the live view follows the first drone
        sensors = getattr(lead, "sensors", {})
        chase = sensors.get("Chase", {}).get("scene_camera")
        if chase:
            client.subscribe(chase, viewer.chase_callback())
        preview = sensors.get(PREVIEW_CAMERA, {}).get("scene_camera")
        if preview:
            client.subscribe(preview, viewer.preview_callback())
        else:
            viewer.no_preview()
        viewer.set_phase(f"team of {len(ids)} on the ground, settling camera exposure")
    cameras = [camera_id]
    for nav in ("Front",):
        try:
            camera_settings(scene, nav, ids[0])
            cameras.append(nav)
        except StopIteration:
            pass
    lift = float(run.spec["search_area"]["altitude_m"])
    views = {camera_id: {"pose": lifted_pose(camera_config(scene, camera_id, ids[0]), lift), "lift_m": lift}}
    exposure = {}
    for i in ids:  # one drone at a time: the client's synchronous requests take turns anyway
        exposure[i] = await settle_exposure(drones[i], cameras, views)

    n = len(ids)
    team = TeamRuntime(world, rec, ids, register_s=LOCKSTEP_REGISTER_S,
                       reply_timeout_s=LOCKSTEP_TIMEOUT_S + TEAM_REPLY_EXTRA_S * (n - 1))
    team.capture_step_ns = capture_step_ns(camera_config(scene, camera_id, ids[0]))
    team_info = [{"id": d["id"], "start_ned": list(d["ned"])} for d in fleet]
    mode = agent.team_mode
    camera = {"width": int(cam["width"]), "height": int(cam["height"]), "fov_deg": float(cam["fov-degrees"])}
    ctxs: Dict[str, SimMissionContext] = {}
    for k, d in enumerate(fleet):
        name = d["id"]
        tagged = (lambda stream, record, name=name: rec.log(stream, {**record, "drone": name}))
        controller = AirSimController(drones[name], world.get_sim_time, tagged)
        ctx = SimMissionContext(run, world, drones[name], camera_id, rec, controller, cam["fov-degrees"], frame_bank,
                                seg_ids=seg_ids, target_ids=[t["id"] for t in targets], start_ned=d["ned"],
                                terrain=terrain, viewer=viewer, camera=camera, drone_id=name,
                                team=team, datum_ned=centre, team_info=team_info, team_mode=mode,
                                fire_ids=[f["id"] for f in fires])
        for nav in cameras[1:]:
            ctx._nav_fov[nav] = float(camera_settings(scene, nav, name)["fov-degrees"])
        ctxs[name] = ctx
    ctxs[ids[0]].preview_topic = preview
    team_mission = mission_spec(run, centre, terrain, camera, datum_ned=centre, team=team_info, team_mode=mode)
    if viewer is not None:
        viewer.set_phase("mission clock started: taking off and climbing")
    meta = {
        "run_id": run.run_id, "scenario_id": run.scenario_id, "seed": run.seed, "params": run.params,
        "mission": run.spec["mission"], "agent": agent.describe(), "targets": targets,
        "search_area": run.spec["search_area"], "limits": run.spec["limits"], "success": run.spec.get("success", {}),
        "start_ned": list(fleet[0]["ned"]),  # the first drone's pad, for tools that read one start
        "fleet": [{"id": d["id"], "start_ned": list(d["ned"]), "yaw_deg": d["yaw_deg"]} for d in fleet],
        "team": {"mode": mode, "size": n, "centre_ned": list(centre)},
        "comms": run.spec.get("comms"),
        "terrain": terrain_meta, "t0_ns": team.t0_ns, "wall_start": datetime.now().isoformat(),
        "projectairsim_commit": projectairsim_commit(airsim_root), "scenario_source": run.source,
        "unreal": unreal_status(), "host": host_info(), "sim_address": getattr(client, "address", None),
        "weights": weights or {},
        "config_hash": run_config_hash(run, agent.describe()),
        "map": sc.get("map"), "preflight": preflight,
        "mission_spec": team_mission.record(),
        "distractors": distractors, **({"fires": fires} if fires else {}), "segmentation_ids": seg_ids,
        "gt_version": 3 if seg_ids else 1,
        "sim_version": SIM_VERSION, "fleet_version": FLEET_VERSION, "exposure": exposure,
        "frame_bank": {"mode": frame_bank, "jpeg_quality": JPEG_QUALITY, "jpeg_max_side": FRAME_JPEG_MAX_SIDE,
                       "depth": "uint16 mm, half resolution, 0 = no data"},
    }
    rec.write_json("meta.json", meta)

    lead_ctx = ctxs[ids[0]]
    if isinstance(agent, FleetAgent):
        named = {"team": agent.run(SimFleetContext(team_mission, ctxs, team))}
    else:
        # The drones share only what they send over the scenario's radio.
        profile = RadioProfile.from_spec(run.spec.get("comms"))

        def position(i):
            return team.last_pos.get(i) if i in team.last_pos else ctxs[i].pose()[0]

        radio = Radio(profile, ids, clock=lead_ctx.elapsed_s, t_ns=lead_ctx.sim_time_or_last, position=position,
                      log=rec.log, seed=run.seed)
        for i, c in ctxs.items():
            c.radio = radio.port(i, c.sleep)
        meta["comms"] = profile.record()
        rec.write_json("meta.json", meta)
        named = {i: a.run(ctxs[i]) for i, a in zip(ids, agent.agents)}
    stop = asyncio.Event()
    poller = asyncio.create_task(_poll_team_trajectory(team, ctxs, stop))
    end_reason = "completed"
    try:
        if await run_tasks(named, lead_ctx.elapsed_s, run.spec["limits"]["timeout_s"] * 2 + RTL_GRACE_S,
                           HARD_TIMEOUT_WALL_FACTOR):
            end_reason = "hard_timeout"
        elif lead_ctx.elapsed_s() >= lead_ctx.mission.timeout_s:
            end_reason = "timeout"
    except Exception as e:
        end_reason = "error"
        who = e.who if isinstance(e, TeamAgentError) else None
        err = e.error if isinstance(e, TeamAgentError) else e
        rec.log("events", {"t": lead_ctx.sim_time_or_last(), "kind": "error", "error": repr(err),
                           **({"drone": who} if who and who in ctxs else {}), "trace": traceback.format_exc()})
    finally:
        team.stop = True
        stop.set()
        await poller
        client.unsubscribe_all()
        meta.update({"end_ns": lead_ctx.sim_time_or_last(), "end_reason": end_reason,
                     "wall_end": datetime.now().isoformat(), "depth_metres_per_unit": team.depth_scale,
                     "frame_timing": team.timing(),
                     "think": {i: {k: round(v, 3) for k, v in st.items()} for i, st in team.think_stats.items()}})
        rec.write_json("meta.json", meta)
        rec.close()
    return score_run_dir(run_dir, weights)


async def _poll_team_trajectory(team: TeamRuntime, ctxs: Dict[str, "SimMissionContext"], stop: asyncio.Event) -> None:
    """Every drone's pose at one sim time, 10 Hz (one ``trajectory`` record per drone)."""
    world = team.world
    lock = getattr(world.client, "request_lock", None) or contextlib.nullcontext()
    lead = next(iter(ctxs.values()))

    def poses_and_time():
        with lock:
            return [(i, c.drone.get_ground_truth_pose()) for i, c in ctxs.items()], world.get_sim_time()

    last = None
    while not stop.is_set():
        try:
            poses, t = await asyncio.to_thread(poses_and_time)
            team.last_t_ns = t
            if last is not None and t <= last:  # paused while photos are recorded
                await asyncio.sleep(1.0 / TRAJECTORY_HZ)
                continue
            last = t
            for i, p in poses:
                team.recorder.log("trajectory", {"t": t, "pos": _xyz(p["translation"]), "q": list(_wxyz(p["rotation"])),
                                                 "drone": i})
                team.last_pos[i] = _xyz(p["translation"])
            if lead.viewer is not None:
                lead.viewer.set_team_progress({i: lead.mission.ground_down_m - p["translation"]["z"] for i, p in poses},
                                              (t - team.t0_ns) / 1e9)
        except Exception:
            pass
        await asyncio.sleep(1.0 / TRAJECTORY_HZ)


def order_runs(runs: List[Run]) -> List[Run]:
    """Group runs by Unreal map (then scenario), so a campaign switches maps as rarely as possible."""
    return sorted(runs, key=lambda r: (scenario_map(r.spec) or "", r.scenario_id))


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
    frame_bank: str = "off",
    view: bool = False,
) -> List[Dict[str, Any]]:
    from projectairsim import ProjectAirSimClient

    out = Path(out_dir) / campaign.name
    sim_cfg = campaign.sim_config_dir
    if not sim_cfg:
        raise ValueError("campaign must set sim_config_dir")
    results = []
    runs = order_runs(campaign.runs())
    blocked_maps: Dict[Optional[str], str] = {}  # map -> why its runs can't start
    viewer = make_viewer(view, log)
    try:
        return await _run_all(runs, campaign, agent_factory, out, sim_cfg, address, resume, airsim_root, log,
                              frame_bank, viewer, results, blocked_maps)
    finally:
        if viewer is not None:
            viewer.close()


async def _run_all(runs, campaign, agent_factory, out, sim_cfg, address, resume, airsim_root, log, frame_bank,
                   viewer, results, blocked_maps) -> List[Dict[str, Any]]:
    from projectairsim import ProjectAirSimClient

    for i, run in enumerate(runs, 1):
        map_name = scenario_map(run.spec)
        if map_name in blocked_maps:
            log(f"[{i}/{len(runs)}] {run.run_id} skipped: {blocked_maps[map_name]}")
            continue
        agent = agent_factory(run)
        name = agent.name
        run_dir = out / name / run.run_id
        done = run_dir / "result.json"
        if resume and done.exists():
            prev = _read_json(run_dir / "meta.json")
            if prev.get("config_hash") != run_config_hash(run, agent.describe()):
                log(f"[{i}/{len(runs)}] {run.run_id} ({name}) scenario or agent config changed, re-running")
            elif prev.get("end_reason") == "error":
                log(f"[{i}/{len(runs)}] {run.run_id} ({name}) previous attempt errored, re-running")
            else:
                log(f"[{i}/{len(runs)}] {run.run_id} ({name}) already done, skipping")
                results.append(json.loads(done.read_text(encoding="utf-8")))
                continue
        log(f"[{i}/{len(runs)}] {run.run_id} ({name}) params={run.params} seed={run.seed}")
        # A fresh connection per run: Project AirSim disconnects the client when an
        # async task fails, and one bad run must not take the rest of the campaign down.
        client = ProjectAirSimClient(address=address)
        try:
            client.connect()
        except Exception as e:
            log(f"    -> cannot connect to Project AirSim at {address} ({type(e).__name__}): is the simulator "
                "running? In the editor, open the map and press Play (or start the packaged build). Stopping.")
            break
        serialize_requests(client)
        allow_long_async_requests(client)
        try:
            r = await execute_run(client, run, agent_factory, sim_cfg, out, campaign.weights, airsim_root, agent=agent,
                                  frame_bank=frame_bank, viewer=viewer)
        except PreflightError as e:
            log(f"    -> preflight failed: {e}")
            blocked_maps[map_name] = f"map {map_name!r} failed preflight"
            continue
        except Exception as e:
            log(f"    -> run failed: {e!r}")
            log(traceback.format_exc())
            continue
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
        log(f"    -> success={r['metrics']['mission']['success']} end={r['metrics']['mission']['end_reason']} "
            f"overall={r['scores']['overall']:.1f}")
        results.append(r)
    return results
