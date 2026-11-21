"""Terrain edits for disaster maps: what an event (or a road) did to the ground, in metres.

``vivid2 terrain build --layout layout.json --stage pre|post`` applies a layout's ``terrain``
operations to the DEM's elevations before they are quantised into the heightmap:

    terrain {
      like: "<map>"                pin the quantisation to this map's (vivid2.dem: --like)
      pre:  [op, ...]              the map before the event (e.g. a road bench)
      post: [op, ...]              what the event did (applied after ``pre``)
    }

Operations (regions are the layout's ``regions``, vivid2.damage.regions):

    {"op": "road_bench", "road": "<roads id>" | "polyline_ne": [...], "width_m", "max_grade" (0.10),
     "batter" (1.0: cut and fill faces rise 1 m per m), "smooth_m" (20)}
        a flat road along the polyline, its height a smoothed, grade-limited profile of the ground;
        beside it the ground may differ from the road's height by at most batter x distance
    {"op": "lower" | "raise", "region": name, "amount_m", "taper_m" (20), "noise_m" (0),
     "noise_scale_m" (15), "seed" (0)}
        ground lowered (a slide scar) or raised (a deposit) by amount_m, tapering to 0 over
        taper_m inside the region's (noisy) edge, with optional lumpy noise
    {"op": "flatten", "region": name | "polygon_ne": [...], "level": "median" | "min" | "max" | metres,
     "blend_m" (10)}
        ground set to one level inside, blending over blend_m outside (a town's plain)

Grids are the terrain build's: ``N``, ``E`` (vertices, ``[i = north, j = east]``) and ``z``
(elevation in metres, any datum).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from vivid2.damage import regions as rg


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bilinear(grid: np.ndarray, N: np.ndarray, E: np.ndarray, n, e) -> np.ndarray:
    n0, e0 = float(N[0, 0]), float(E[0, 0])
    cell = float(N[1, 0] - N[0, 0])
    fi = np.clip((np.asarray(n, float) - n0) / cell, 0, grid.shape[0] - 1.000001)
    fj = np.clip((np.asarray(e, float) - e0) / cell, 0, grid.shape[1] - 1.000001)
    i0, j0 = fi.astype(int), fj.astype(int)
    di, dj = fi - i0, fj - j0
    return ((1 - di) * (1 - dj) * grid[i0, j0] + di * (1 - dj) * grid[i0 + 1, j0]
            + (1 - di) * dj * grid[i0, j0 + 1] + di * dj * grid[i0 + 1, j0 + 1])


def _resample(polyline, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` m along a polyline and their arc length."""
    p = np.asarray(polyline, float)[:, :2]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1] + 1e-9, step)
    if s[-1] < cum[-1] - 1e-9:
        s = np.append(s, cum[-1])
    return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])]), s


def grade_limit(profile: np.ndarray, ds: float, max_grade: float) -> np.ndarray:
    """A profile clamped so no step climbs more than ``max_grade``: a forward pass, then a backward
    pass (which leaves every step within the limit)."""
    p = profile.astype(float).copy()
    lim = max_grade * ds
    for i in range(1, len(p)):
        p[i] = np.clip(p[i], p[i - 1] - lim, p[i - 1] + lim)
    for i in range(len(p) - 2, -1, -1):
        p[i] = np.clip(p[i], p[i + 1] - lim, p[i + 1] + lim)
    return p


def road_bench(N: np.ndarray, E: np.ndarray, z: np.ndarray, polyline, width_m: float, max_grade: float = 0.10,
               batter: float = 1.0, smooth_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Cut and fill a flat road into the ground; returns (new z, the road's height profile every 1 m)."""
    import cv2

    pts, s = _resample(polyline, 1.0)
    ground = _bilinear(z, N, E, pts[:, 0], pts[:, 1])
    k = max(1, int(round(smooth_m)))
    kernel = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    padded = np.pad(ground, 3 * k, mode="edge")
    prof = np.convolve(padded, kernel / kernel.sum(), mode="valid")
    prof = grade_limit(prof, 1.0, max_grade)
    # nearest road sample for every vertex: label each vertex with its closest sample
    cell = float(N[1, 0] - N[0, 0])
    n0, e0 = float(N[0, 0]), float(E[0, 0])
    src = np.ones(z.shape, np.uint8)
    ii = np.clip(np.rint((pts[:, 0] - n0) / cell).astype(int), 0, z.shape[0] - 1)
    jj = np.clip(np.rint((pts[:, 1] - e0) / cell).astype(int), 0, z.shape[1] - 1)
    src[ii, jj] = 0
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zero_i, zero_j = np.nonzero(src == 0)  # label k (1-based) = k-th zero pixel in row-major order
    sample_of_pixel = {}
    for k_s, (a, b) in enumerate(zip(ii, jj)):
        sample_of_pixel.setdefault((a, b), k_s)
    lut = np.array([sample_of_pixel[(a, b)] for a, b in zip(zero_i, zero_j)])
    nearest = lut[labels - 1]
    # exact distance to the two road segments around the nearest sample
    P = np.stack([N, E], -1)
    best_d = np.full(z.shape, np.inf)
    best_h = np.zeros(z.shape)
    for off in range(-4, 4):  # the labels come from an approximate distance: check the segments around them
        a_idx = np.clip(nearest + off, 0, len(pts) - 2)
        a, b = pts[a_idx], pts[a_idx + 1]
        ab = b - a
        t = np.clip(((P - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
        d = np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)
        h = prof[a_idx] + t * (prof[a_idx + 1] - prof[a_idx])
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_h = np.where(closer, h, best_h)
    half = width_m / 2.0
    out = np.where(best_d <= half, best_h,
                   best_h + np.clip(z - best_h, -batter * (best_d - half), batter * (best_d - half)))
    return out, prof


def _sdf_at(layout: Dict[str, Any], shape: Dict[str, Any], N, E) -> np.ndarray:
    """Signed distance (m, + inside) of a region (or ``{"polygon_ne": ...}``) at the vertices, on the
    layout's mask grid."""
    m = layout.get("mask") or {"size_px": 1024, "extent_m": 2.0 * float(max(np.abs(N).max(), np.abs(E).max())) + 2.0}
    extent = float(m["extent_m"])
    return rg.sample(rg.region_sdf(shape, int(m["size_px"]), extent), N, E, extent)


def _region_shape(layout: Dict[str, Any], name: str, taper_m: float, N, E) -> np.ndarray:
    """0..1 at the vertices: 1 deep inside the region, tapering to 0 at its (noisy) edge."""
    return _smoothstep(_sdf_at(layout, layout["regions"][name], N, E) / max(taper_m, 1e-6))


def _noise(shape, scale_cells: float, seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(int(seed))
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), max(scale_cells, 0.5))
    return n / max(float(n.std()), 1e-6)


def apply_edits(layout: Dict[str, Any], stage: str, N: np.ndarray, E: np.ndarray,
                z: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """The layout's terrain operations for ``stage`` ("pre", or "post" = pre then post) applied to ``z``."""
    t = layout.get("terrain") or {}
    if stage not in ("pre", "post"):
        raise ValueError("stage is pre or post")
    ops = list(t.get("pre", [])) + (list(t.get("post", [])) if stage == "post" else [])
    roads = {r["id"]: r for r in layout.get("roads", [])}
    cell = float(N[1, 0] - N[0, 0])
    report = []
    z = z.astype(float).copy()
    for op in ops:
        before = z.copy()
        kind = op["op"]
        if kind == "road_bench":
            line = op.get("polyline_ne") or roads[op["road"]]["polyline_ne"]
            width = float(op.get("width_m", roads.get(op.get("road"), {}).get("width_m", 6.0)))
            z, _ = road_bench(N, E, z, line, width, float(op.get("max_grade", 0.10)), float(op.get("batter", 1.0)),
                              float(op.get("smooth_m", 20.0)))
        elif kind in ("lower", "raise"):
            shape = _region_shape(layout, op["region"], float(op.get("taper_m", 20.0)), N, E)
            amount = float(op["amount_m"]) * shape
            if float(op.get("noise_m", 0.0)) > 0:
                amount = amount + float(op["noise_m"]) * shape * _noise(z.shape, float(op.get("noise_scale_m", 15.0)) / cell,
                                                                        int(op.get("seed", 0)))
            z = z - amount if kind == "lower" else z + amount
        elif kind == "flatten":
            shape = layout["regions"][op["region"]] if "region" in op else {"polygon_ne": op["polygon_ne"]}
            sd = _sdf_at(layout, shape, N, E)
            level = op.get("level", "median")
            if isinstance(level, str):
                lv = {"median": np.median, "min": np.min, "max": np.max}[level](z[sd >= 0])
            else:
                lv = float(level)
            w = _smoothstep(1.0 + sd / max(float(op.get("blend_m", 10.0)), 1e-6))
            z = z + w * (lv - z)
        else:
            raise ValueError(f"unknown terrain operation {kind!r}")
        dz = z - before
        changed = np.abs(dz) > 0.005
        rec = {"op": kind, "changed_vertices": int(changed.sum()),
               "cut_m3": round(float(-dz[dz < 0].sum() * cell * cell), 1),
               "fill_m3": round(float(dz[dz > 0].sum() * cell * cell), 1)}
        if changed.any():
            rec["bbox_ne"] = [[round(float(N[changed].min()), 1), round(float(E[changed].min()), 1)],
                              [round(float(N[changed].max()), 1), round(float(E[changed].max()), 1)]]
        report.append(rec)
    return z, report
