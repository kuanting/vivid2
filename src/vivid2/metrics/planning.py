"""Planning metrics: how much of the search area the camera actually covered."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vivid2.geo import focal_px, pixel_ray_to_ground, points_in_polygon, quat_to_rot  # noqa: F401 (re-exported)
from vivid2.metrics.cameras import search_frames
from vivid2.metrics.team import by_drone, fleet_of, is_team
from vivid2.recorder import RunLog
from vivid2.sim.segmentation import min_visible_px

PERSON_AREA_M2 = 0.85  # a person lying down (about 1.7 m x 0.5 m), the patch a photo must resolve


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


def footprint_hits(frame: Dict[str, Any], cells: np.ndarray, ground_down_m):
    """Which cells a photo covers (None if it covers nothing it can be credited for).

    A photo with the gimbal at rest (straight down, full field of view): the cells inside its four
    ground corners; if the horizon is in view (the drone banking hard), none. A photo the stack
    tilted or zoomed: the cells in view where a person would show at least the visibility
    threshold of pixels, so looking at the horizon does not count the far ground.

    ``ground_down_m`` is one ground plane, or each cell's own ground (a terrain-following survey:
    a nadir photo then covers the cells whose ground is in the image).
    """
    if frame.get("gimbal"):
        return detectable_hits(frame, cells, ground_down_m)
    if np.ndim(ground_down_m):
        return _nadir_hits_on_terrain(frame, cells, np.asarray(ground_down_m, float))
    w, h = frame["width"], frame["height"]
    corners = [
        pixel_ray_to_ground(u, v, frame["cam_pos"], frame["cam_q"], w, h, frame["fov_deg"], ground_down_m)
        for u, v in ((0, 0), (w, 0), (w, h), (0, h))
    ]
    if any(c is None for c in corners):
        return None  # horizon in view; footprint unbounded, skip conservatively
    return points_in_polygon(cells, [c[:2] for c in corners])


def _nadir_hits_on_terrain(frame: Dict[str, Any], cells: np.ndarray, ground: np.ndarray) -> Optional[np.ndarray]:
    """A nadir photo over terrain: the cells whose ground point is in the image (None if the horizon is)."""
    w, h, fov = frame["width"], frame["height"], frame["fov_deg"]
    low = float(ground.max()) if len(ground) else 0.0  # the lowest ground: below the camera
    if any(pixel_ray_to_ground(u, v, frame["cam_pos"], frame["cam_q"], w, h, fov, low) is None
           for u, v in ((0, 0), (w, 0), (w, h), (0, h))):
        return None
    cam = np.asarray(frame["cam_pos"], float)
    pc = (np.column_stack([cells, ground]) - cam) @ quat_to_rot(frame["cam_q"])
    x = pc[:, 0]
    front = x > 1e-6
    xs = np.where(front, x, 1.0)
    f = focal_px(w, fov)
    u, v = w / 2.0 + f * pc[:, 1] / xs, h / 2.0 + f * pc[:, 2] / xs
    return front & (u >= 0) & (u < w) & (v >= 0) & (v < h)


def detectable_hits(frame: Dict[str, Any], cells: np.ndarray, ground_down_m) -> np.ndarray:
    """Cells in a photo's view where a lying person (PERSON_AREA_M2) covers at least the visibility threshold.

    A ground patch of area A seen at range r, incidence cos_i and planar depth x covers
    A cos_i f^2 r / x^3 pixels (pinhole camera, focal length f in pixels). ``ground_down_m``: one
    plane or each cell's ground.
    """
    w, h, fov = frame["width"], frame["height"], frame["fov_deg"]
    cam = np.asarray(frame["cam_pos"], float)
    rel = np.column_stack([cells, np.full(len(cells), ground_down_m)]) - cam
    pc = rel @ quat_to_rot(frame["cam_q"])  # camera frame: x along the optical axis, y right, z down
    x = pc[:, 0]
    front = x > 1e-6
    xs = np.where(front, x, 1.0)
    f = focal_px(w, fov)
    u, v = w / 2.0 + f * pc[:, 1] / xs, h / 2.0 + f * pc[:, 2] / xs
    rng = np.linalg.norm(rel, axis=1)
    cos_i = np.abs(rel[:, 2]) / np.maximum(rng, 1e-9)
    px = PERSON_AREA_M2 * cos_i * f * f * rng / xs ** 3
    return front & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (px >= min_visible_px(w, h))


def coverage(
    frames: List[Dict[str, Any]], polygon_ne: Sequence[Sequence[float]], ground_down_m: float, cell_m: float = 2.0,
    terrain=None,
) -> Dict[str, float]:
    """``terrain``: measure on this ground model instead of the plane at ``ground_down_m``."""
    cells = _cells(polygon_ne, cell_m)
    ground = ground_down_m if terrain is None else np.asarray(terrain.down_at(cells[:, 0], cells[:, 1]), float)
    hits = np.zeros(len(cells), int)
    for f in frames:
        hit = footprint_hits(f, cells, ground)
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
    cov = coverage(search_frames(log["frames"], meta), meta["search_area"]["polygon_ne"], ground,
                   terrain=survey_terrain(meta))
    length = path_length(log["trajectory"])
    return {
        **cov,
        "path_length_m": length,
        "coverage_per_km_m2": cov["covered_m2"] / (length / 1000.0) if length > 0 else 0.0,
        **verification_stats(log["events"]),
    }


def survey_terrain(meta: Dict[str, Any]):
    """The run's terrain model if its survey followed the terrain (``search_area.altitude_ref`` "terrain"):
    coverage is then measured on the ground itself, not on the plane through the start. Otherwise None."""
    if (meta.get("search_area") or {}).get("altitude_ref") != "terrain":
        return None
    from vivid2.terrain import from_meta

    t = from_meta(meta)
    return None if t.is_flat else t


COVERAGE_MILESTONE = 0.9  # team runs: mission time until the team has seen this share of the area


def _team_planning(log: RunLog, ground: float, cell_m: float = 2.0) -> Dict[str, Any]:
    """Coverage of the whole team (the union of every drone's photos), plus how it was shared."""
    meta = log.meta
    ids = [d["id"] for d in fleet_of(meta)]
    cells = _cells(meta["search_area"]["polygon_ne"], cell_m)
    hits = np.zeros(len(cells), int)
    own = {i: np.zeros(len(cells), bool) for i in ids}
    milestone_s = None
    for f in sorted(search_frames(log["frames"], meta), key=lambda f: f["t"]):
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
