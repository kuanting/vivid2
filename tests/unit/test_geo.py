import math

import numpy as np

from vivid2.geo import nadir_footprint, pixel_ray_to_ground, pixel_to_world, rpy_to_quat, world_to_pixel

NADIR = rpy_to_quat(0.0, -math.pi / 2, 0.0)
W, H, FOV = 960, 720, 80.0


def test_nadir_centre_pixel_is_directly_below():
    cam = np.array([10.0, -5.0, -30.0])
    p = pixel_to_world(W / 2, H / 2, 30.0, cam, NADIR, W, H, FOV)
    np.testing.assert_allclose(p, [10.0, -5.0, 0.0], atol=1e-9)


def test_image_axes_orientation():
    cam = np.array([0.0, 0.0, -30.0])
    # Image right = east, image up = north for a nadir camera at yaw 0.
    right = pixel_to_world(W - 1, H / 2, 30.0, cam, NADIR, W, H, FOV)
    up = pixel_to_world(W / 2, 0, 30.0, cam, NADIR, W, H, FOV)
    assert right[1] > 1 and abs(right[0]) < 1e-6
    assert up[0] > 1 and abs(up[1]) < 1e-6


def test_projection_roundtrip():
    cam, q = np.array([3.0, 4.0, -40.0]), rpy_to_quat(0.1, -1.2, 0.7)
    p = np.array([12.0, 9.0, 0.0])
    u, v, d = world_to_pixel(p, cam, q, W, H, FOV)
    np.testing.assert_allclose(pixel_to_world(u, v, d, cam, q, W, H, FOV), p, atol=1e-6)
    np.testing.assert_allclose(pixel_ray_to_ground(u, v, cam, q, W, H, FOV, 0.0), p, atol=1e-6)


def test_ray_above_horizon_returns_none():
    level = rpy_to_quat(0, 0, 0)
    assert pixel_ray_to_ground(W / 2, 0, [0, 0, -10], level, W, H, FOV) is None


def test_footprint():
    across, along = nadir_footprint(30.0, W, H, FOV)
    assert math.isclose(across, 2 * 30 * math.tan(math.radians(40)), rel_tol=1e-9)
    assert math.isclose(along / across, H / W, rel_tol=1e-9)
