"""Camera geometry: pixel <-> world (NED) and ground-footprint helpers.

Project AirSim cameras use an x-forward, y-right, z-down optical frame, and
image messages carry the camera's world pose (pos_*, rot_*). "Planar" depth is
the distance along the camera x axis.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def quat_to_rot(q_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = q_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def focal_px(width: int, fov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def pixel_to_world(
    u: float,
    v: float,
    depth_planar: float,
    cam_pos: Sequence[float],
    cam_q: Sequence[float],
    width: int,
    height: int,
    fov_deg: float,
) -> np.ndarray:
    """Back-project a pixel with planar depth to a world NED point."""
    f = focal_px(width, fov_deg)
    p_cam = depth_planar * np.array([1.0, (u - width / 2.0) / f, (v - height / 2.0) / f])
    return np.asarray(cam_pos, dtype=float) + quat_to_rot(cam_q) @ p_cam


def pixel_ray_to_ground(
    u: float,
    v: float,
    cam_pos: Sequence[float],
    cam_q: Sequence[float],
    width: int,
    height: int,
    fov_deg: float,
    ground_down_m: float = 0.0,
) -> Optional[np.ndarray]:
    """Intersect a pixel ray with a flat ground plane (fallback when no depth)."""
    f = focal_px(width, fov_deg)
    ray = quat_to_rot(cam_q) @ np.array([1.0, (u - width / 2.0) / f, (v - height / 2.0) / f])
    if ray[2] <= 1e-6:
        return None
    s = (ground_down_m - cam_pos[2]) / ray[2]
    return None if s <= 0 else np.asarray(cam_pos, dtype=float) + s * ray


def world_to_pixel(
    p_world: Sequence[float],
    cam_pos: Sequence[float],
    cam_q: Sequence[float],
    width: int,
    height: int,
    fov_deg: float,
) -> Optional[Tuple[float, float, float]]:
    """Project a world point; returns (u, v, planar_depth) or None if behind."""
    p_cam = quat_to_rot(cam_q).T @ (np.asarray(p_world, float) - np.asarray(cam_pos, float))
    if p_cam[0] <= 1e-6:
        return None
    f = focal_px(width, fov_deg)
    return (width / 2.0 + f * p_cam[1] / p_cam[0], height / 2.0 + f * p_cam[2] / p_cam[0], p_cam[0])


def nadir_footprint(altitude_m: float, width: int, height: int, fov_deg: float) -> Tuple[float, float]:
    """Ground footprint (across-image-x, across-image-y) of a nadir camera, in metres."""
    f = focal_px(width, fov_deg)
    return (altitude_m * width / f, altitude_m * height / f)
