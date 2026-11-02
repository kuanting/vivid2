"""Planning metrics: how much of the search area the camera actually covered."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from vivid2.geo import pixel_ray_to_ground
from vivid2.recorder import RunLog


def points_in_polygon(pts: np.ndarray, poly: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test. pts: (N, 2)."""
    poly = np.asarray(poly, float)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def path_length(traj: List[Dict[str, Any]]) -> float:
    if len(traj) < 2:
        return 0.0
    p = np.asarray([r["pos"] for r in traj], float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def coverage(
    frames: List[Dict[str, Any]], polygon_ne: Sequence[Sequence[float]], ground_down_m: float, cell_m: float = 2.0
) -> Dict[str, float]:
    poly = np.asarray(polygon_ne, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    ns = np.arange(lo[0] + cell_m / 2, hi[0], cell_m)
    es = np.arange(lo[1] + cell_m / 2, hi[1], cell_m)
    grid = np.stack(np.meshgrid(ns, es, indexing="ij"), -1).reshape(-1, 2)
    in_area = points_in_polygon(grid, poly)
    cells = grid[in_area]
    hits = np.zeros(len(cells), int)
    for f in frames:
        w, h = f["width"], f["height"]
        corners = [
            pixel_ray_to_ground(u, v, f["cam_pos"], f["cam_q"], w, h, f["fov_deg"], ground_down_m)
            for u, v in ((0, 0), (w, 0), (w, h), (0, h))
        ]
        if any(c is None for c in corners):
            continue  # horizon in view; footprint unbounded, skip conservatively
        quad = [c[:2] for c in corners]
        hits += points_in_polygon(cells, quad)
    seen = hits > 0
    return {
        "coverage_pct": float(100.0 * seen.mean()) if len(cells) else 0.0,
        "area_m2": float(len(cells) * cell_m * cell_m),
        "covered_m2": float(seen.sum() * cell_m * cell_m),
        "revisit_ratio": float(hits[seen].mean()) if seen.any() else 0.0,
    }


def planning_metrics(log: RunLog) -> Dict[str, Any]:
    meta = log.meta
    # the ground plane the DUT was given (terrain under the start; older logs: the authored plane)
    ground = (meta.get("mission_spec") or {}).get("ground_down_m", meta["search_area"].get("ground_down_m", 0.0))
    cov = coverage(log["frames"], meta["search_area"]["polygon_ne"], ground)
    length = path_length(log["trajectory"])
    return {
        **cov,
        "path_length_m": length,
        "coverage_per_km_m2": cov["covered_m2"] / (length / 1000.0) if length > 0 else 0.0,
        **verification_stats(log["events"]),
    }


def verification_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Closer looks the agent took before reporting (``verify_start`` / ``verify_end`` events)."""
    starts = [e for e in events if e.get("kind") == "verify_start"]
    ends = [e for e in events if e.get("kind") == "verify_end"]
    time_s = sum(max(0, int(b["t"]) - int(a["t"])) for a, b in zip(starts, ends)) / 1e9
    return {"verifications": len(starts), "verifications_confirmed": sum(bool(e.get("confirmed")) for e in ends),
            "verify_time_s": time_s}
