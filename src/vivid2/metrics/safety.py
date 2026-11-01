"""Safety metrics: collisions, terrain clearance, geofence, speed, return home."""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

from vivid2.metrics.control import windowed_speeds
from vivid2.metrics.planning import path_length, points_in_polygon
from vivid2.recorder import RunLog
from vivid2.terrain import TerrainModel, from_meta


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


HOME_PAD_RADIUS_M = 5.0  # same radius as "returned home"
PAD_CONTACT_AGL_M = 2.0


def split_ground_contacts(collisions, t: np.ndarray, pos: np.ndarray, start: np.ndarray,
                          ground: Union[float, TerrainModel]):
    """Separate takeoff/landing contacts from real collisions.

    The sim reports the drone resting on the ground at spawn and touching down
    on landing as collisions. A contact counts as a pad contact when the drone
    was within the home pad radius and below 2 m AGL at that moment; everything
    else (and every contact when no trajectory was recorded) is a collision.
    """
    terrain = ground if isinstance(ground, TerrainModel) else TerrainModel.flat(ground)
    real, pad = [], []
    for c in collisions:
        if len(t) == 0 or c.get("t") is None:
            real.append(c)
            continue
        tc = float(c["t"]) / 1e9
        i = int(np.clip(np.searchsorted(t, tc), 1, max(len(t) - 1, 1)))
        p = pos[i - 1] if len(t) == 1 or abs(t[i - 1] - tc) <= abs(t[i] - tc) else pos[i]  # nearest sample
        on_pad = np.linalg.norm(p[:2] - start[:2]) <= HOME_PAD_RADIUS_M and float(terrain.agl(p)) < PAD_CONTACT_AGL_M
        (pad if on_pad else real).append(c)
    return real, pad


def safety_metrics(log: RunLog) -> Dict[str, Any]:
    meta = log.meta
    limits = meta["limits"]
    traj = log["trajectory"]
    poly = np.asarray(meta["search_area"]["polygon_ne"], float)
    terrain = from_meta(meta)  # the run's DTM, or flat ground at search_area.ground_down_m
    start = np.asarray(meta["start_ned"], float)
    margin = limits.get("geofence_margin_m", 20.0)
    min_agl = limits.get("min_agl_m", 5.0)

    t = np.asarray([r["t"] for r in traj], float) / 1e9
    pos = np.asarray([r["pos"] for r in traj], float).reshape(-1, 3)
    collisions, contacts = split_ground_contacts(log["collisions"], t, pos, start, terrain)
    out: Dict[str, Any] = {
        "collisions": len(collisions),
        "collision_objects": sorted({c.get("object_name", "") for c in collisions}),
        "ground_contacts": len(contacts),
    }
    if not traj:
        return out
    dt = np.diff(t, prepend=t[0])

    # Geofence: search polygon + margin, but the corridor from home is allowed.
    outside = np.array([distance_outside_polygon(p[:2], poly) for p in pos])
    near_home = np.linalg.norm(pos[:, :2] - start[:2], axis=1) <= margin
    breach = (outside > margin) & ~near_home

    # Terrain clearance, only once airborne and away from home (takeoff/landing excluded).
    agl = terrain.agl(pos)
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
        "speed_limit_exceeded": bool(windowed_speeds(t, pos).max() > 1.25 * limits.get("max_speed_mps", 8.0)),
        "energy_used_m": flown,
        "energy_budget_exceeded": bool(energy_budget is not None and flown > energy_budget),
        "returned_home": end_dist_home <= 5.0,
        "end_distance_from_home_m": end_dist_home,
        "timed_out": meta.get("end_reason") in ("timeout", "hard_timeout"),
    })
    return out
