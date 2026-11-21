"""Geometry for the damage task: grids, polygons, road corridors, and what a photo saw.

All positions are NED metres (north, east[, down]). ``view_mask`` is the single definition of
"this photo observed this ground cell", shared by the metrics (credit only where the drone
looked), the rules (a report must show its item in the cited photo) and the reference oracle.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2.damage import CORRIDOR_STEP_M, GSD_MAX_M, VIEW_MAX_RANGE_M
from vivid2.geo import focal_px, quat_to_rot
from vivid2.geo import points_in_polygon

OCCLUSION_STEPS = 32
OCCLUSION_TOLERANCE_M = 0.5  # terrain this far above the line of sight hides the cell


def grid_cells(polygon_ne: Sequence[Sequence[float]], cell_m: float) -> np.ndarray:
    """Centres (N, 2) of the ``cell_m`` grid cells inside a polygon (the grid starts at its bounding box)."""
    poly = np.asarray(polygon_ne, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    ns = np.arange(lo[0] + cell_m / 2, hi[0], cell_m)
    es = np.arange(lo[1] + cell_m / 2, hi[1], cell_m)
    grid = np.stack(np.meshgrid(ns, es, indexing="ij"), -1).reshape(-1, 2)
    return grid[points_in_polygon(grid, poly)] if len(grid) else grid


def in_any_polygon(points: np.ndarray, polygons: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    points = np.asarray(points, float).reshape(-1, 2)
    out = np.zeros(len(points), bool)
    for poly in polygons:
        if len(poly) >= 3:
            out |= points_in_polygon(points, poly)
    return out


def distance_to_polygon(points: np.ndarray, polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """Distance (m) from each point to a polygon: 0 inside, else to the nearest edge."""
    p = np.asarray(points, float).reshape(-1, 2)
    poly = np.asarray(polygon, float)
    best = np.full(len(p), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return np.where(points_in_polygon(p, poly), 0.0, best)


def rect_polygon(centre_ne: Sequence[float], length_m: float, width_m: float, yaw_deg: float = 0.0) -> List[List[float]]:
    """Corners of a rectangle: ``length_m`` along the heading ``yaw_deg`` (0 north), ``width_m`` across."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    hl, hw = length_m / 2.0, width_m / 2.0
    out = []
    for a, b in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)):
        out.append([float(centre_ne[0] + a * c - b * s), float(centre_ne[1] + a * s + b * c)])
    return out


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    p = np.asarray(polygon, float)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def polygon_centroid(polygon: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(polygon, float).mean(axis=0)


# -- polylines and road corridors ---------------------------------------------------------------

def polyline_length(polyline: Sequence[Sequence[float]]) -> float:
    p = np.asarray(polyline, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def point_at(polyline: Sequence[Sequence[float]], s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Point and unit tangent at arc length ``s`` along a polyline (clamped to its ends)."""
    p = np.asarray(polyline, float)
    seg = np.diff(p, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    acc = 0.0
    for i, L in enumerate(lens):
        if s <= acc + L or i == len(lens) - 1:
            t = 0.0 if L <= 0 else float(np.clip((s - acc) / L, 0.0, 1.0))
            return p[i] + t * seg[i], seg[i] / max(L, 1e-12)
        acc += L
    return p[-1], np.array([1.0, 0.0])


def split_polyline(polyline: Sequence[Sequence[float]], max_len_m: float) -> List[List[List[float]]]:
    """Cut a polyline into the fewest pieces of equal length no longer than ``max_len_m``."""
    total = polyline_length(polyline)
    n = max(1, int(math.ceil(total / max_len_m - 1e-9)))
    cuts = [total * k / n for k in range(n + 1)]
    p = np.asarray(polyline, float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        pts = [point_at(polyline, a)[0]]
        pts += [p[i] for i in range(len(p)) if a + 1e-9 < cum[i] < b - 1e-9]
        pts.append(point_at(polyline, b)[0])
        pieces.append([[round(float(x), 3) for x in q] for q in pts])
    return pieces


def corridor(polyline: Sequence[Sequence[float]], width_m: float,
             step_m: float = CORRIDOR_STEP_M) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points over a road: ``(along_m (M,), points_ne (M, 2))``.

    Every ``step_m`` along the centreline, points every ``step_m`` across the full width.
    """
    total = polyline_length(polyline)
    along = np.arange(0.0, total + 1e-9, step_m)
    k = max(1, int(round(width_m / step_m)))
    offsets = (np.arange(k + 1) / k - 0.5) * width_m
    s_out, p_out = [], []
    for s in along:
        c, t = point_at(polyline, float(s))
        normal = np.array([-t[1], t[0]])
        for o in offsets:
            s_out.append(s)
            p_out.append(c + o * normal)
    return np.asarray(s_out, float), np.asarray(p_out, float).reshape(-1, 2)


def corridor_cells(polyline: Sequence[Sequence[float]], width_m: float, cells: np.ndarray, cell_m: float) -> np.ndarray:
    """Indices of ``cells`` (grid centres ``cell_m`` apart) that a road's corridor touches."""
    _, pts = corridor(polyline, width_m, min(CORRIDOR_STEP_M * 2, cell_m / 2))
    if not len(cells):
        return np.zeros(0, int)
    lo = cells.min(axis=0) - cell_m / 2
    key = {(int(i), int(j)) for i, j in np.floor((pts - lo) / cell_m)}
    idx = np.floor((cells - lo) / cell_m).astype(int)
    return np.flatnonzero([(int(i), int(j)) in key for i, j in idx])


# -- what a photo saw ---------------------------------------------------------------------------

def _camera(frame: Any) -> Tuple[np.ndarray, Sequence[float], int, int, float]:
    if isinstance(frame, dict):
        return (np.asarray(frame["cam_pos"], float), frame["cam_q"], int(frame["width"]), int(frame["height"]),
                float(frame["fov_deg"]))
    return np.asarray(frame.cam_pos, float), frame.cam_q, frame.width, frame.height, float(frame.fov_deg)


def view_mask(frame: Any, cells_ne: np.ndarray, cells_down: np.ndarray, terrain=None,
              gsd_max_m: float = GSD_MAX_M, max_range_m: float = VIEW_MAX_RANGE_M) -> np.ndarray:
    """Which ground cells a photo observed: in the image, within range, sharp enough, not hidden by terrain.

    ``frame``: a ``frames`` record or a Frame. ``cells_down``: each cell's ground (NED down).
    Ground sampling distance of a patch at range r, incidence cos_i and planar depth x is
    sqrt(x^3 / (f^2 r cos_i)) (pinhole camera, f in pixels; flat-ground incidence). With a
    ``terrain`` model, a cell is hidden when the terrain rises above the line of sight to it.
    Trees and buildings never hide a cell (documented in docs/damage.md).
    """
    cells_ne = np.asarray(cells_ne, float).reshape(-1, 2)
    if not len(cells_ne):
        return np.zeros(0, bool)
    cam, q, w, h, fov = _camera(frame)
    pts = np.column_stack([cells_ne, np.asarray(cells_down, float).reshape(-1)])
    rel = pts - cam
    pc = rel @ quat_to_rot(q)  # camera frame: x along the optical axis, y right, z down
    x = pc[:, 0]
    front = x > 1e-6
    xs = np.where(front, x, 1.0)
    f = focal_px(w, fov)
    u, v = w / 2.0 + f * pc[:, 1] / xs, h / 2.0 + f * pc[:, 2] / xs
    rng = np.linalg.norm(rel, axis=1)
    cos_i = np.abs(rel[:, 2]) / np.maximum(rng, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        gsd = np.sqrt(xs ** 3 / (f * f * np.maximum(rng, 1e-9) * np.maximum(cos_i, 1e-6)))
    seen = front & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (rng <= max_range_m) & (gsd <= gsd_max_m)
    if terrain is not None and not getattr(terrain, "is_flat", True) and seen.any():
        idx = np.flatnonzero(seen)
        t = (np.arange(1, OCCLUSION_STEPS) / OCCLUSION_STEPS)[:, None]
        ray = cam[None, None, :] + t[..., None] * rel[idx][None, :, :]  # (steps, k, 3)
        ground = terrain.down_at(ray[..., 0], ray[..., 1])
        hidden = (ground < ray[..., 2] - OCCLUSION_TOLERANCE_M).any(axis=0)
        seen[idx[hidden]] = False
    return seen


def observed_union(frames: Sequence[Any], cells_ne: np.ndarray, cells_down: np.ndarray, terrain=None) -> np.ndarray:
    out = np.zeros(len(cells_ne), bool)
    for f in frames:
        out |= view_mask(f, cells_ne, cells_down, terrain)
    return out


def point_in_view(frame: Any, ned: Sequence[float], terrain=None) -> bool:
    """Whether one ground point (n, e, down) is observed by a photo (``view_mask`` for one cell)."""
    p = np.asarray(ned, float)
    return bool(view_mask(frame, p[None, :2], p[None, 2:3], terrain)[0])


def raster_polygons(polygons: Sequence[Dict[str, Any]], cells_ne: np.ndarray) -> np.ndarray:
    """Apply area reports in order to the cells: ``op`` "add" sets the cells inside, "remove" clears them."""
    out = np.zeros(len(cells_ne), bool)
    for p in polygons:
        inside = points_in_polygon(cells_ne, p["polygon_ne"]) if len(p["polygon_ne"]) >= 3 else np.zeros(len(cells_ne), bool)
        if p.get("op", "add") == "remove":
            out &= ~inside
        else:
            out |= inside
    return out


def mask_polygons(mask: np.ndarray, n0: float, e0: float, cell_m: float) -> List[Dict[str, Any]]:
    """Polygons outlining the True cells of a 2D grid mask (``mask[i, j]`` centred at n0 + i cell, e0 + j cell).

    Returns area-report-like dicts in drawing order ("add" for outlines, "remove" for holes),
    largest first, so a hole comes after the outline around it and an island in a hole after the
    hole. Rasterising them back at the cell centres (``raster_polygons``) gives the mask again:
    the mask is doubled before tracing, so outlines pass a quarter cell outside the boundary
    cells' centres (and holes' outlines a quarter cell outside the cells around them).
    """
    import cv2

    m = np.ascontiguousarray(mask.astype(np.uint8))
    if not m.any():
        return []
    up = np.pad(np.kron(m, np.ones((2, 2), np.uint8)), 1)  # pad so outlines on the edge close
    contours, hier = cv2.findContours(up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c, h in zip(contours, hier[0]):
        c = cv2.approxPolyDP(c, 0.4, True)  # in half cells: keeps every cell centre on its side
        pts = c.reshape(-1, 2).astype(float) - 1.0  # (x = j, y = i) in doubled cells
        if len(pts) < 3:
            continue
        poly = [[round(float(n0 + (i / 2.0 - 0.25) * cell_m), 3), round(float(e0 + (j / 2.0 - 0.25) * cell_m), 3)]
                for j, i in pts]
        out.append((abs(float(cv2.contourArea(c))), {"polygon_ne": poly, "op": "remove" if h[3] >= 0 else "add"}))
    return [p for _, p in sorted(out, key=lambda x: -x[0])]
