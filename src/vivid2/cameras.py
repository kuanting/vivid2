"""A drone's cameras as a stack sees them, and the gimbal of its search camera.

Every drone has one **search camera** (the robot config's primary camera, "Payload", the only one
with ground truth: it captures segmentation) on a gimbal, and **navigation cameras** ("Front")
for obstacle avoidance. Every photo from either is taken in lockstep, costs a fixed capture step of
mission time and is recorded; only search-camera photos have ground truth and are scored, and a
report must cite one.

The gimbal tilts the search camera from straight down (-90 degrees, the default) to just above
the horizon, pans it around the drone's yaw axis and zooms it. Moving it takes mission time,
the same on every machine. It is not stabilised: apart from the commanded angles the camera
stays rigid to the drone, as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2.geo import pixel_ray_to_ground, quat_to_rot

SEARCH, NAV = "search", "nav"
NAV_CAMERAS = ("Front",)  # robot cameras a stack may use to look where it flies (if the robot has them)
CAPTURE_MARGIN_S = 0.05  # a lockstep step lets the simulator run the camera's capture-interval plus this

# The search camera's gimbal
GIMBAL_TILT_RANGE = (-90.0, 10.0)  # degrees: -90 straight down, 0 the horizon
GIMBAL_PAN_RANGE = (-180.0, 180.0)  # degrees, relative to the drone's heading, positive to the right
GIMBAL_FOV_MIN = 8.0  # degrees: about 10x zoom on an 80 degree camera
GIMBAL_SLEW_DPS = 60.0
GIMBAL_SETTLE_S = 0.2  # after each move
ZOOM_S_PER_E = 0.5  # seconds per e-fold change of field of view
NADIR_TOLERANCE_DEG = 10.0  # a photo this close to straight down counts as nadir (coverage, canopy map)


@dataclass
class CameraInfo:
    """One camera of a drone, from its robot config."""

    id: str
    role: str  # SEARCH or NAV
    width: int
    height: int
    fov_deg: float
    capture_step_ns: int
    types: List[int]  # image types each photo asks for (0 colour, 1 planar depth, 3 segmentation)
    mount_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def spec(self) -> Dict[str, Any]:
        """What a stack is told about it (``MissionSpec.cameras``)."""
        out: Dict[str, Any] = {"role": self.role, "width": self.width, "height": self.height,
                               "fov_deg": self.fov_deg, "capture_s": round(self.capture_step_ns / 1e9, 3),
                               "mount_xyz": list(self.mount_xyz), "mount_rpy_deg": list(self.mount_rpy_deg)}
        if self.role == SEARCH:
            out["gimbal"] = {"tilt_range_deg": list(GIMBAL_TILT_RANGE), "pan_range_deg": list(GIMBAL_PAN_RANGE),
                             "fov_range_deg": [GIMBAL_FOV_MIN, self.fov_deg], "slew_dps": GIMBAL_SLEW_DPS}
        return out


def capture_step_ns(camera: Dict[str, Any]) -> int:
    """Mission time a lockstep photo lets the simulator run: the camera's capture-interval plus a margin.

    The camera captures at the first rendered pose more than capture-interval after its last capture,
    so a step this long always yields a photo, however close together the stack asks for them.
    """
    return int(round((float(camera.get("capture-interval", 0.2)) + CAPTURE_MARGIN_S) * 1e9))


def _captured_types(sensor: Dict[str, Any]) -> List[int]:
    return sorted({int(c["image-type"]) for c in sensor.get("capture-settings", []) if c.get("capture-enabled")})


def camera_info(sensor: Dict[str, Any], role: str) -> CameraInfo:
    rgb = next(c for c in sensor["capture-settings"] if c["image-type"] == 0)
    origin = sensor.get("origin", {})
    xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
    rpy = tuple(float(v) for v in origin.get("rpy-deg", "0 0 0").split())
    types = [t for t in _captured_types(sensor) if t in (0, 1, 3)]
    if role != SEARCH:
        types = [t for t in types if t != 3]  # no ground truth on navigation cameras
    return CameraInfo(sensor["id"], role, int(rgb["width"]), int(rgb["height"]), float(rgb["fov-degrees"]),
                      capture_step_ns(sensor), types, xyz, rpy)


def camera_infos(robot: Dict[str, Any], primary: str = "Payload") -> Dict[str, CameraInfo]:
    """The drone's search camera (``primary``) and navigation cameras, search camera first.

    ``robot`` is a robot config (its ``sensors``). Viewer-only cameras (chase, preview) are left out.
    """
    sensors = {s["id"]: s for s in robot.get("sensors", []) if s.get("type") == "camera" and s.get("enabled", True)}
    if primary not in sensors:
        raise ValueError(f"the robot has no camera {primary!r}")
    out = {primary: camera_info(sensors[primary], SEARCH)}
    for cid in NAV_CAMERAS:
        if cid in sensors and cid != primary:
            out[cid] = camera_info(sensors[cid], NAV)
    return out


# -- gimbal ------------------------------------------------------------------------------------
@dataclass
class GimbalState:
    tilt_deg: float = -90.0
    pan_deg: float = 0.0
    fov_deg: float = 80.0
    base_fov_deg: float = field(default=80.0, repr=False)

    def is_default(self) -> bool:
        return self.tilt_deg == -90.0 and self.pan_deg == 0.0 and self.fov_deg == self.base_fov_deg

    def record(self) -> Dict[str, float]:
        return {"tilt_deg": round(self.tilt_deg, 2), "pan_deg": round(self.pan_deg, 2), "fov_deg": round(self.fov_deg, 3)}


def wrap_deg(a: float) -> float:
    """An angle in degrees, wrapped to (-180, 180]."""
    a = math.fmod(float(a), 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def clamp_gimbal(state: GimbalState, tilt_deg: Optional[float], pan_deg: Optional[float],
                 fov_deg: Optional[float]) -> GimbalState:
    """The gimbal state a command asks for, within the limits (None keeps the current value)."""
    tilt = state.tilt_deg if tilt_deg is None else float(np.clip(tilt_deg, *GIMBAL_TILT_RANGE))
    pan = state.pan_deg if pan_deg is None else wrap_deg(pan_deg)
    pan = float(np.clip(pan, *GIMBAL_PAN_RANGE))
    fov = state.fov_deg if fov_deg is None else float(np.clip(fov_deg, GIMBAL_FOV_MIN, state.base_fov_deg))
    return GimbalState(tilt, pan, fov, state.base_fov_deg)


def slew_time_s(old: GimbalState, new: GimbalState) -> float:
    """Mission time a gimbal move takes: pointing at GIMBAL_SLEW_DPS, then zooming; 0 if nothing moves."""
    turn = max(abs(new.tilt_deg - old.tilt_deg), abs(wrap_deg(new.pan_deg - old.pan_deg)))
    zoom = abs(math.log(new.fov_deg / old.fov_deg)) if old.fov_deg > 0 else 0.0
    if turn < 1e-6 and zoom < 1e-9:
        return 0.0
    return turn / GIMBAL_SLEW_DPS + zoom * ZOOM_S_PER_E + GIMBAL_SETTLE_S


def mount_pose(mount_xyz: Sequence[float], tilt_deg: float, pan_deg: float) -> Dict[str, Dict[str, float]]:
    """The camera pose relative to the drone for a gimbal angle (Project AirSim ``set_camera_pose``).

    Rotation roll 0, pitch ``tilt``, yaw ``pan`` (applied yaw-pitch-roll, as ``rpy-deg`` in the
    robot config): tilt -90 pan 0 is the configured straight-down mount.
    """
    r, p, y = 0.0, math.radians(tilt_deg) / 2.0, math.radians(pan_deg) / 2.0
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    q = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    x, yy, z = (float(v) for v in mount_xyz)
    return {"translation": {"x": x, "y": yy, "z": z},
            "rotation": {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}}


def tilt_from_nadir_deg(cam_q: Sequence[float]) -> float:
    """Angle between a camera's optical axis and straight down."""
    axis = quat_to_rot(cam_q)[:, 0]
    return float(math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0)))))


def is_nadir(cam_q: Sequence[float]) -> bool:
    return tilt_from_nadir_deg(cam_q) <= NADIR_TOLERANCE_DEG


def detect_depth_scale_rays(raw: np.ndarray, cam_pos: Sequence[float], cam_q: Sequence[float], fov_deg: float,
                            ground_down_m: float) -> Optional[float]:
    """Metres per raw depth unit from a photo in any direction: 0.001 (millimetres) or 1.0 (metres).

    Samples pixels whose ray meets flat ground 5-60 m away and compares the raw values with the
    planar depth expected there: about 1000 times larger means millimetres. None when too few
    pixels qualify (sky, or the drone still on the ground).
    """
    h, w = raw.shape[:2]
    axis = quat_to_rot(cam_q)[:, 0]
    ratios = []
    for v in np.linspace(0.5, h - 0.5, 30):
        for u in np.linspace(0.5, w - 0.5, 40):
            hit = pixel_ray_to_ground(u, v, cam_pos, cam_q, w, h, fov_deg, ground_down_m)
            if hit is None:
                continue
            planar = float(np.dot(hit - np.asarray(cam_pos, float), axis))
            if not 5.0 <= planar <= 60.0:
                continue
            r = float(raw[int(v), int(u)])
            if 0 < r < 65504:
                ratios.append(r / planar)
    if len(ratios) < 50:
        return None
    return 0.001 if float(np.median(ratios)) > 31.6 else 1.0


__all__ = ["SEARCH", "NAV", "NAV_CAMERAS", "CameraInfo", "GimbalState", "camera_infos", "camera_info",
           "capture_step_ns", "clamp_gimbal", "slew_time_s", "mount_pose", "tilt_from_nadir_deg", "is_nadir",
           "detect_depth_scale_rays", "wrap_deg"]
