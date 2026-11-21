"""Project AirSim NED <-> Unreal coordinates, for placing film actors and cameras where a run had them.

Project AirSim puts the NED origin at the Unreal world origin and converts with
``TransformUtils::NedToNeuLinear(ToCentimeters(ned))`` (``UnrealTransforms.h``): north -> +X,
east -> +Y, down -> -Z, metres -> centimetres, no offset. Attitudes go through
``UnrealHelpers::ToFRotator(ToRPY(q))``: the aerospace ZYX Euler angles of the NED quaternion become
``FRotator(Pitch=pitch, Yaw=yaw, Roll=roll)`` (same as ``NedToNeuQuat`` = ``FQuat(-x, -y, z, w)``).
Unreal cameras look along +X like Project AirSim's camera frame, so a recorded camera pose maps
straight onto a CineCamera.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

FILMBACK_16x9_MM = (36.0, 20.25)  # sensor width x height of the film cameras (16:9, like the output)


def ned_to_ue_cm(ned: Sequence[float]) -> Tuple[float, float, float]:
    n, e, d = (float(v) for v in ned)
    return (100.0 * n, 100.0 * e, -100.0 * d)


def ue_cm_to_ned(ue: Sequence[float]) -> Tuple[float, float, float]:
    x, y, z = (float(v) for v in ue)
    return (x / 100.0, y / 100.0, -z / 100.0)


def quat_to_rpy_deg(q_wxyz: Sequence[float]) -> Tuple[float, float, float]:
    """Aerospace ZYX Euler angles (roll, pitch, yaw), degrees, of an NED attitude quaternion."""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def quat_ned_to_rotator(q_wxyz: Sequence[float]) -> Dict[str, float]:
    """Unreal rotator {roll, pitch, yaw} (degrees) of an NED attitude quaternion."""
    roll, pitch, yaw = quat_to_rpy_deg(q_wxyz)
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def look_at_rotator(eye_ue: Sequence[float], target_ue: Sequence[float], roll: float = 0.0) -> Dict[str, float]:
    """Rotator that points +X from ``eye`` at ``target`` (both Unreal cm)."""
    dx, dy, dz = (float(t) - float(e) for t, e in zip(target_ue, eye_ue))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def unwrap_deg(angles: Iterable[float]) -> List[float]:
    """Remove +-360 jumps so keyframe interpolation turns the short way."""
    out: List[float] = []
    for a in angles:
        a = float(a)
        if out:
            prev = out[-1]
            a = prev + ((a - prev + 180.0) % 360.0 - 180.0)
        out.append(a)
    return out


def hfov_to_focal_mm(hfov_deg: float, sensor_width_mm: float = FILMBACK_16x9_MM[0]) -> float:
    return (sensor_width_mm / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def focal_mm_to_hfov(focal_mm: float, sensor_width_mm: float = FILMBACK_16x9_MM[0]) -> float:
    return math.degrees(2.0 * math.atan((sensor_width_mm / 2.0) / focal_mm))


def camera_from_frame(rec: Dict) -> Dict:
    """A CineCamera that sees what a recorded frame saw: same pose and horizontal field of view.

    The film is 16:9 and the payload camera 4:3, so the film frame is the payload frame's central
    16:9 band (``payload_crop_16x9``)."""
    return {"location": ned_to_ue_cm(rec["cam_pos"]), "rotation": quat_ned_to_rotator(rec["cam_q"]),
            "focal_mm": hfov_to_focal_mm(float(rec["fov_deg"]))}


def payload_crop_16x9(width: int, height: int) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the central 16:9 band of a frame with the same width."""
    h = int(round(width * 9 / 16))
    h = min(h, height)
    y0 = (height - h) // 2
    return (0, y0, width, y0 + h)
