"""Region fields: named areas of a map with noisy, soft edges (a landslide scar, a flood zone...).

One field per region, 0 (outside) to 1 (inside), computed once on the map's mask grid and read
by everything that needs the region, so the pictures and the ground truth agree:

* Unreal: an 8-bit greyscale mask per region (``to_png``: 128 + 127 b, never 0, because PCG's
  Sample Texture drops points where a texture reads 0); the landscape material and the PCG
  graphs threshold it at ``REGION_THRESHOLD``.
* Ground truth: the same field sampled at the terrain grid's nodes (``sample``), the way the
  material samples the texture.

The maths is ``tools/burn_mask.py``'s (the ForestFire burn mask), generalised to polylines
(roads, corridors) and several regions; ``region_field`` reproduces that mask exactly.

Mask grid: ``size_px`` pixels over ``extent_m`` metres centred on the origin; pixel
[row, col] = [east, north]. Unreal samples it at U = (X + extent/2) / extent (north),
V = (Y + extent/2) / extent (east).
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _px(points_ne: Sequence[Sequence[float]], size: int, extent: float) -> np.ndarray:
    """(n, e) -> pixel (x = col = north, y = row = east), as burn_mask.py rounds them."""
    px_m = extent / size
    return np.round((np.asarray(points_ne, float)[:, [0, 1]] + extent / 2.0) / px_m).astype(np.int32)


def inside_mask(region: Dict[str, Any], size: int, extent: float) -> np.ndarray:
    """1 inside the region's shape (no edge noise), 0 outside: a polygon, or polylines ``width_m`` wide."""
    import cv2

    m = np.zeros((size, size), np.uint8)
    if "polygon_ne" in region:
        cv2.fillPoly(m, [_px(region["polygon_ne"], size, extent).reshape(-1, 1, 2)], 1)
    for line in region.get("polylines_ne", []):
        thick = max(1, int(round(float(region["width_m"]) / (extent / size))))
        cv2.polylines(m, [_px(line, size, extent).reshape(-1, 1, 2)], False, 1, thickness=thick)
    if "polygon_ne" not in region and not region.get("polylines_ne"):
        raise ValueError("a region needs polygon_ne or polylines_ne (+ width_m)")
    return m


def region_sdf(region: Dict[str, Any], size: int, extent: float) -> np.ndarray:
    """Signed distance (m, + inside) to the region's shape on the mask grid, plus its edge noise
    (smooth seeded noise ``edge.noise_m`` strong, correlated over ``edge.noise_scale_m``, ``edge.seed``)."""
    import cv2

    px_m = extent / size
    inside = inside_mask(region, size, extent)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5) * px_m
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5) * px_m
    sd = d_in - d_out
    e = region.get("edge") or {}
    if float(e.get("noise_m", 0.0)) > 0:
        rng = np.random.default_rng(int(e["seed"]))
        sigma = float(e["noise_scale_m"]) / px_m
        noise = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), sigma)
        noise *= float(e["noise_m"]) / max(float(noise.std()), 1e-6)
        sd = sd + noise
    return sd


def region_field(region: Dict[str, Any], size: int, extent: float) -> np.ndarray:
    """The region's field b in [0, 1] on the mask grid (float32, [row = east, col = north]).

    The signed distance (``region_sdf``) through a smoothstep ``edge.soft_m`` wide. No ``edge``:
    a hard edge.
    """
    px_m = extent / size
    sd = region_sdf(region, size, extent)
    e = region.get("edge")
    if not e:
        return (sd > 0).astype(np.float32)
    soft = max(float(e.get("soft_m", px_m)), 1e-6)
    mask = np.clip(0.5 + sd / soft, 0.0, 1.0)
    return mask * mask * (3 - 2 * mask)  # smoothstep


def to_png(field: np.ndarray) -> np.ndarray:
    """8-bit mask for Unreal: 128 + 127 b (never 0)."""
    return (128.0 + np.clip(field, 0.0, 1.0) * 127.0 + 0.5).astype(np.uint8)


def from_png(img: np.ndarray) -> np.ndarray:
    return np.clip((img.astype(np.float32) - 128.0) / 127.0, 0.0, 1.0)


def sample(field: np.ndarray, n: np.ndarray, e: np.ndarray, extent: float) -> np.ndarray:
    """The field at NED points, bilinear, as Unreal samples the texture (pixel centres at
    (k + 0.5) extent / size - extent / 2); clamped at the edge."""
    size = field.shape[0]
    col = (np.asarray(n, float) + extent / 2.0) / extent * size - 0.5
    row = (np.asarray(e, float) + extent / 2.0) / extent * size - 0.5
    col = np.clip(col, 0, size - 1)
    row = np.clip(row, 0, size - 1)
    c0, r0 = np.minimum(col.astype(int), size - 2), np.minimum(row.astype(int), size - 2)
    dc, dr = col - c0, row - r0
    f = field
    return ((1 - dr) * (1 - dc) * f[r0, c0] + dr * (1 - dc) * f[r0 + 1, c0]
            + (1 - dr) * dc * f[r0, c0 + 1] + dr * dc * f[r0 + 1, c0 + 1])


def fields(layout: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Every region of a layout: {name: field}."""
    m = layout["mask"]
    size, extent = int(m["size_px"]), float(m["extent_m"])
    return {name: region_field(r, size, extent) for name, r in (layout.get("regions") or {}).items()}
