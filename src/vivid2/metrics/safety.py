"""Safety metrics: collisions, terrain clearance, geofence, speed, return home."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from vivid2.metrics.control import segment_speeds
from vivid2.metrics.planning import path_length, points_in_polygon
from vivid2.recorder import RunLog


def distance_outside_polygon(p: np.ndarray, poly: np.ndarray) -> float:
    """0 inside, otherwise distance to the nearest edge."""
    if points_in_polygon(p[None, :], poly)[0]:
        return 0.0
    best = float("inf")
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        s = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + s * ab))))
    return best


def safety_metrics(log: RunLog) -> Dict[str, Any]:
    meta = log.meta
    limits = meta["limits"]
    traj = log["trajectory"]
    poly = np.asarray(meta["search_area"]["polygon_ne"], float)
    ground = meta["search_area"].get("ground_down_m", 0.0)
    start = np.asarray(meta["start_ned"], float)
    margin = limits.get("geofence_margin_m", 20.0)
    min_agl = limits.get("min_agl_m", 5.0)

    collisions = log["collisions"]
    out: Dict[str, Any] = {
        "collisions": len(collisions),
        "collision_objects": sorted({c.get("object_name", "") for c in collisions}),
    }
    if not traj:
        return out
    t = np.asarray([r["t"] for r in traj], float) / 1e9
    pos = np.asarray([r["pos"] for r in traj], float)
    dt = np.diff(t, prepend=t[0])

    # Geofence: search polygon + margin, but the corridor from home is allowed.
    outside = np.array([distance_outside_polygon(p[:2], poly) for p in pos])
    near_home = np.linalg.norm(pos[:, :2] - start[:2], axis=1) <= margin
    breach = (outside > margin) & ~near_home

    # Terrain clearance, only once airborne and away from home (takeoff/landing excluded).
    agl = ground - pos[:, 2]
    airborne = ~near_home
    low = airborne & (agl < min_agl)

    energy_budget = limits.get("energy_budget_m")
    flown = path_length(traj)
    end_dist_home = float(np.linalg.norm(pos[-1, :2] - start[:2]))
    out.update({
        "geofence_breach_s": float(dt[breach].sum()),
        "max_geofence_excursion_m": float(max(0.0, (outside - margin).max())),
        "min_agl_m": float(agl[airborne].min()) if airborne.any() else None,
        "low_altitude_s": float(dt[low].sum()),
        "speed_limit_exceeded": bool(segment_speeds(t, pos).max() > 1.25 * limits.get("max_speed_mps", 8.0)),
        "energy_used_m": flown,
        "energy_budget_exceeded": bool(energy_budget is not None and flown > energy_budget),
        "returned_home": end_dist_home <= 5.0,
        "end_distance_from_home_m": end_dist_home,
        "timed_out": meta.get("end_reason") in ("timeout", "hard_timeout"),
    })
    return out
