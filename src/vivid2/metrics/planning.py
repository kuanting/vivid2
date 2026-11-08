"""Planning metrics: how much of the search area the camera actually covered."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from vivid2.geo import pixel_ray_to_ground
from vivid2.metrics.team import by_drone, fleet_of, is_team
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


def _cells(polygon_ne: Sequence[Sequence[float]], cell_m: float) -> np.ndarray:
    """Centres of the ``cell_m`` grid cells inside the search polygon."""
    poly = np.asarray(polygon_ne, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    ns = np.arange(lo[0] + cell_m / 2, hi[0], cell_m)
    es = np.arange(lo[1] + cell_m / 2, hi[1], cell_m)
    grid = np.stack(np.meshgrid(ns, es, indexing="ij"), -1).reshape(-1, 2)
    return grid[points_in_polygon(grid, poly)]


def footprint_hits(frame: Dict[str, Any], cells: np.ndarray, ground_down_m: float):
    """Which cells a frame's ground footprint covers (None if the horizon is in view)."""
    w, h = frame["width"], frame["height"]
    corners = [
        pixel_ray_to_ground(u, v, frame["cam_pos"], frame["cam_q"], w, h, frame["fov_deg"], ground_down_m)
        for u, v in ((0, 0), (w, 0), (w, h), (0, h))
    ]
    if any(c is None for c in corners):
        return None  # horizon in view; footprint unbounded, skip conservatively
    return points_in_polygon(cells, [c[:2] for c in corners])


def coverage(
    frames: List[Dict[str, Any]], polygon_ne: Sequence[Sequence[float]], ground_down_m: float, cell_m: float = 2.0
) -> Dict[str, float]:
    cells = _cells(polygon_ne, cell_m)
    hits = np.zeros(len(cells), int)
    for f in frames:
        hit = footprint_hits(f, cells, ground_down_m)
        if hit is not None:
            hits += hit
    return _coverage_summary(cells, hits, cell_m)


def _coverage_summary(cells: np.ndarray, hits: np.ndarray, cell_m: float) -> Dict[str, float]:
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
    if is_team(meta):
        return _team_planning(log, ground)
    cov = coverage(log["frames"], meta["search_area"]["polygon_ne"], ground)
    length = path_length(log["trajectory"])
    return {
        **cov,
        "path_length_m": length,
        "coverage_per_km_m2": cov["covered_m2"] / (length / 1000.0) if length > 0 else 0.0,
        **verification_stats(log["events"]),
    }


COVERAGE_MILESTONE = 0.9  # team runs: mission time until the team has seen this share of the area


def _team_planning(log: RunLog, ground: float, cell_m: float = 2.0) -> Dict[str, Any]:
    """Coverage of the whole team (the union of every drone's photos), plus how it was shared."""
    meta = log.meta
    ids = [d["id"] for d in fleet_of(meta)]
    cells = _cells(meta["search_area"]["polygon_ne"], cell_m)
    hits = np.zeros(len(cells), int)
    own = {i: np.zeros(len(cells), bool) for i in ids}
    milestone_s = None
    for f in sorted(log["frames"], key=lambda f: f["t"]):
        hit = footprint_hits(f, cells, ground)
        if hit is None:
            continue
        hits += hit
        own.setdefault(f.get("drone", ids[0]), np.zeros(len(cells), bool))[:] |= hit
        if milestone_s is None and len(cells) and (hits > 0).mean() >= COVERAGE_MILESTONE:
            milestone_s = (f["t"] - meta["t0_ns"]) / 1e9
    cov = _coverage_summary(cells, hits, cell_m)
    trajs = by_drone(log["trajectory"], ids)
    events = by_drone(log["events"], ids)
    per = {}
    for i in own:
        per[i] = {"path_length_m": path_length(trajs.get(i, [])), "covered_m2": float(own[i].sum() * cell_m * cell_m),
                  **verification_stats(events.get(i, []))}
    length = sum(p["path_length_m"] for p in per.values())
    seen_by = np.sum([own[i] for i in own], axis=0) if own else np.zeros(len(cells))
    covered = seen_by > 0
    areas = [p["covered_m2"] for p in per.values()]
    return {
        **cov,
        "path_length_m": length,
        "coverage_per_km_m2": cov["covered_m2"] / (length / 1000.0) if length > 0 else 0.0,
        **{k: sum(p[k] for p in per.values()) for k in ("verifications", "verifications_confirmed", "verify_time_s")},
        # share of the covered area that more than one drone photographed (duplicated effort)
        "coverage_overlap_pct": float(100.0 * (seen_by > 1).sum() / covered.sum()) if covered.any() else 0.0,
        # area the least-loaded drone covered, relative to the most-loaded one (1 = even split)
        "workload_balance": float(min(areas) / max(areas)) if areas and max(areas) > 0 else None,
        "time_to_90pct_coverage_s": milestone_s,
        "per_drone": per,
    }


def verification_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Closer looks the agent took before reporting (``verify_start`` / ``verify_end`` events)."""
    starts = [e for e in events if e.get("kind") == "verify_start"]
    ends = [e for e in events if e.get("kind") == "verify_end"]
    time_s = sum(max(0, int(b["t"]) - int(a["t"])) for a, b in zip(starts, ends)) / 1e9
    return {"verifications": len(starts), "verifications_confirmed": sum(bool(e.get("confirmed")) for e in ends),
            "verify_time_s": time_s}
