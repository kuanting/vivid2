"""Safety metrics: collisions, terrain clearance, geofence, speed, return home."""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

from vivid2.metrics.control import windowed_speeds
from vivid2.metrics.planning import path_length, points_in_polygon
from vivid2.metrics.team import by_drone, contact_events, fleet_of, is_team, separation
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
    pads = np.atleast_2d(np.asarray(start, float))  # a team: contacts on any drone's pad
    real, pad = [], []
    for c in collisions:
        if len(t) == 0 or c.get("t") is None:
            real.append(c)
            continue
        tc = float(c["t"]) / 1e9
        i = int(np.clip(np.searchsorted(t, tc), 1, max(len(t) - 1, 1)))
        p = pos[i - 1] if len(t) == 1 or abs(t[i - 1] - tc) <= abs(t[i] - tc) else pos[i]  # nearest sample
        on_pad = (np.linalg.norm(pads[:, :2] - p[:2], axis=1).min() <= HOME_PAD_RADIUS_M
                  and float(terrain.agl(p)) < PAD_CONTACT_AGL_M)
        (pad if on_pad else real).append(c)
    return real, pad


def safety_metrics(log: RunLog) -> Dict[str, Any]:
    if is_team(log.meta):
        return _team_safety(log)
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
        # 2 s windows: under heavy load (HD captures, VLM calls) pose timestamps arrive unevenly and 1 s
        # averages still spike ~30 % for a single sample; sustained speeding still shows.
        "speed_limit_exceeded": bool(windowed_speeds(t, pos, 2.0).max() > 1.25 * limits.get("max_speed_mps", 8.0)),
        "energy_used_m": flown,
        "energy_budget_exceeded": bool(energy_budget is not None and flown > energy_budget),
        "returned_home": end_dist_home <= 5.0,
        "end_distance_from_home_m": end_dist_home,
        "timed_out": meta.get("end_reason") in ("timeout", "hard_timeout"),
    })
    return out


def _drone_safety(traj, collisions, start: np.ndarray, pads: np.ndarray, poly: np.ndarray, terrain: TerrainModel,
                  limits: Dict[str, Any]) -> Dict[str, Any]:
    """One team drone: its own trajectory, measured against its own pad."""
    margin = limits.get("geofence_margin_m", 20.0)
    min_agl = limits.get("min_agl_m", 5.0)
    t = np.asarray([r["t"] for r in traj], float) / 1e9
    pos = np.asarray([r["pos"] for r in traj], float).reshape(-1, 3)
    real, contacts = split_ground_contacts(collisions, t, pos, pads, terrain)
    out: Dict[str, Any] = {"real": real, "ground_contacts": len(contacts)}
    if not traj:
        return out
    dt = np.diff(t, prepend=t[0])
    outside = np.array([distance_outside_polygon(p[:2], poly) for p in pos])
    near_home = np.linalg.norm(pos[:, :2] - start[:2], axis=1) <= margin
    breach = (outside > margin) & ~near_home
    agl = terrain.agl(pos)
    airborne = ~near_home
    flown = path_length(traj)
    budget = limits.get("energy_budget_m")
    end_dist = float(np.linalg.norm(pos[-1, :2] - start[:2]))
    out.update({
        "geofence_breach_s": float(dt[breach].sum()),
        "max_geofence_excursion_m": float(max(0.0, (outside - margin).max())),
        "min_agl_m": float(agl[airborne].min()) if airborne.any() else None,
        "low_altitude_s": float(dt[airborne & (agl < min_agl)].sum()),
        "speed_limit_exceeded": bool(windowed_speeds(t, pos, 2.0).max() > 1.25 * limits.get("max_speed_mps", 8.0)),
        "energy_used_m": flown,
        "energy_budget_exceeded": bool(budget is not None and flown > budget),
        "returned_home": end_dist <= 5.0,
        "end_distance_from_home_m": end_dist,
    })
    return out


def _team_safety(log: RunLog) -> Dict[str, Any]:
    """Every drone's safety, combined: the battery budget is per drone, every drone must come home
    to its own pad, and drones flying closer than ``limits.min_separation_m`` are penalised."""
    meta = log.meta
    limits = meta["limits"]
    poly = np.asarray(meta["search_area"]["polygon_ne"], float)
    terrain = from_meta(meta)
    fleet = fleet_of(meta)
    ids = [d["id"] for d in fleet]
    pads = np.asarray([d["start_ned"] for d in fleet], float)
    trajs, colls = by_drone(log["trajectory"], ids), by_drone(log["collisions"], ids)
    per = {d["id"]: _drone_safety(trajs.get(d["id"], []), colls.get(d["id"], []), np.asarray(d["start_ned"], float),
                                  pads, poly, terrain, limits) for d in fleet}
    real = contact_events([c for p in per.values() for c in p.pop("real")])
    flown = {i: p["energy_used_m"] for i, p in per.items() if "energy_used_m" in p}
    out: Dict[str, Any] = {
        "collisions": len(real),
        "collision_objects": sorted({c.get("object_name", "") for c in real}),
        "drone_collisions": sum(1 for c in real if c.get("other_drone")),
        "ground_contacts": sum(p["ground_contacts"] for p in per.values()),
    }
    if not flown:
        return out
    flying = [p for p in per.values() if "energy_used_m" in p]
    agls = [p["min_agl_m"] for p in flying if p["min_agl_m"] is not None]
    out.update({
        "geofence_breach_s": sum(p["geofence_breach_s"] for p in flying),
        "max_geofence_excursion_m": max(p["max_geofence_excursion_m"] for p in flying),
        "min_agl_m": min(agls) if agls else None,
        "low_altitude_s": sum(p["low_altitude_s"] for p in flying),
        "speed_limit_exceeded": any(p["speed_limit_exceeded"] for p in flying),
        "energy_used_m": max(flown.values()),  # each drone has its own battery
        "energy_used_total_m": sum(flown.values()),
        "energy_budget_exceeded": any(p["energy_budget_exceeded"] for p in flying),
        "returned_home": all(p["returned_home"] for p in flying),
        "end_distance_from_home_m": max(p["end_distance_from_home_m"] for p in flying),
        "timed_out": meta.get("end_reason") in ("timeout", "hard_timeout"),
        **separation(log["trajectory"], ids, float(limits.get("min_separation_m", 5.0))),
        "per_drone": per,
    })
    return out
