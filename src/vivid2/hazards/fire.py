"""Static wildfire: fire objects in a scenario and the no-fly areas around them.

A scenario lists its fires under ``fires`` (``{id, asset, ned, yaw_deg?, description?}``). They
are blueprints in ``/Game/VIVID2/Fire``, spawned by name like the decoys, and they burn in place
for the whole run. Each blueprint has a ground-truth proxy mesh the size of its flames, so the
segmentation camera records how much of each fire a frame shows (``fires_visible`` in
``frames.jsonl``) for a later fire-spotting task. A fire is not a target: the SAR mission still
scores hikers only, and a report where only a fire is in view is a false alarm (``fp_on_fire``).

A fire's smoke drifts along the blueprint's +X axis, so a fire without ``yaw_deg`` is turned to
face downwind (``environment.wind_ned``).

``limits.no_fly`` lists the coarse fire restriction areas a crew would be given: polygons in which
the drone must stay above ``ceiling_agl_m``. The stack is told them (``MissionSpec.no_fly_zones``);
time below a ceiling inside one costs safety points (``no_fly_s``). They are coarse on purpose:
per-fire circles would give the fire-spotting answer away.

Flames and smoke move with Unreal's game clock, which follows the wall clock, so the colour image
of a fire differs a little between runs and machines (segmentation does not: the proxy is static).
``check_fires`` therefore keeps targets clear of the flames and of the downwind smoke, so that
noise never decides whether a hiker is visible.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# A target (with its jitter) must be at least this far from every fire, horizontally.
MIN_TARGET_FIRE_DISTANCE_M = 15.0
# Ground footprint of a plume drifting downwind, seen from survey height: a wedge this long,
# this wide at the fire, widening by SMOKE_PATH_SPREAD per metre downwind.
SMOKE_PATH_LENGTH_M = 60.0
SMOKE_PATH_HALF_WIDTH_M = 8.0
SMOKE_PATH_SPREAD = 0.3
CALM_WIND_MPS = 0.5  # below this the smoke rises straight up and the fires keep their authored yaw


def downwind_yaw_deg(wind_ned: Optional[Sequence[float]]) -> Optional[float]:
    """Heading (degrees, 0 = north, 90 = east) the wind blows towards; None in calm air."""
    if not wind_ned:
        return None
    n, e = float(wind_ned[0]), float(wind_ned[1])
    if math.hypot(n, e) < CALM_WIND_MPS:
        return None
    return math.degrees(math.atan2(e, n))


def fire_yaw_deg(fire: Dict[str, Any], wind_ned: Optional[Sequence[float]]) -> float:
    """The authored yaw, else downwind, else 0."""
    if "yaw_deg" in fire:
        return float(fire["yaw_deg"])
    yaw = downwind_yaw_deg(wind_ned)
    return 0.0 if yaw is None else round(yaw, 3)


def _contains(a: str, b: str) -> bool:
    return a != b and (a in b or b in a)


def check_fires(spec: Dict[str, Any]) -> List[str]:
    """Problems with a scenario's fires and no-fly areas (empty when it has neither).

    * Fire ids are unique among all spawned objects, and no id contains another: Project AirSim
      renames a spawned actor when an existing actor's name contains its id.
    * Every target, with its jitter, stays MIN_TARGET_FIRE_DISTANCE_M from each fire and outside
      each fire's downwind smoke path.
    * Every no-fly ceiling is below the survey height, and the start is outside every area.
    """
    fires = spec.get("fires") or []
    zones = (spec.get("limits") or {}).get("no_fly") or []
    if not fires and not zones:
        return []
    problems: List[str] = []
    others = [o["id"] for o in list(spec.get("targets", [])) + list(spec.get("distractors", []))]
    seen: List[str] = []
    for f in fires:
        fid = f["id"]
        if fid in seen or fid in others:
            problems.append(f"fire id {fid!r} is used twice")
        clash = [o for o in others + seen if _contains(fid, o)]
        if clash:
            problems.append(f"fire id {fid!r} and {clash} contain one another; use zero-padded ids such as Fire01")
        seen.append(fid)

    jitter = float(spec.get("target_jitter_m", 0.0))
    wind = (spec.get("environment") or {}).get("wind_ned")
    w = None
    if downwind_yaw_deg(wind) is not None:
        w = np.asarray(wind[:2], float) / math.hypot(float(wind[0]), float(wind[1]))
    for t in spec.get("targets", []):
        p = np.asarray(t["ned"][:2], float)
        for f in fires:
            rel = p - np.asarray(f["ned"][:2], float)
            if float(np.linalg.norm(rel)) - jitter < MIN_TARGET_FIRE_DISTANCE_M:
                problems.append(f"target {t['id']!r} is within {MIN_TARGET_FIRE_DISTANCE_M:g} m of fire {f['id']!r} "
                                f"(target_jitter_m included)")
                continue
            if w is not None:
                along = float(rel @ w)
                cross = abs(float(rel[0] * w[1] - rel[1] * w[0]))
                if (-jitter < along <= SMOKE_PATH_LENGTH_M + jitter
                        and cross <= SMOKE_PATH_HALF_WIDTH_M + SMOKE_PATH_SPREAD * max(along, 0.0) + jitter):
                    problems.append(f"target {t['id']!r} is in the downwind smoke of fire {f['id']!r} "
                                    f"({along:.0f} m downwind, {cross:.0f} m to the side)")

    altitude = float(spec["search_area"]["altitude_m"])
    start = np.asarray(spec["dut_start"]["ned"][:2], float)
    for i, z in enumerate(zones):
        if float(z["ceiling_agl_m"]) >= altitude:
            problems.append(f"no-fly area {i} has ceiling {z['ceiling_agl_m']} m, not below the survey height {altitude:g} m")
        if in_polygon(start, z["polygon_ne"]):
            problems.append(f"the start {list(start)} is inside no-fly area {i}")
    return problems


def in_polygon(p_ne: Sequence[float], polygon_ne: Sequence[Sequence[float]]) -> bool:
    from vivid2.metrics.planning import points_in_polygon

    return bool(points_in_polygon(np.asarray([p_ne[:2]], float), polygon_ne)[0])


def in_no_fly(zones: Optional[Iterable[Dict[str, Any]]], ne: np.ndarray, agl: np.ndarray) -> np.ndarray:
    """Per point: inside some no-fly area and below its ceiling. ``ne`` (N, 2), ``agl`` (N,)."""
    from vivid2.metrics.planning import points_in_polygon

    ne = np.asarray(ne, float).reshape(-1, 2)
    agl = np.asarray(agl, float).reshape(-1)
    out = np.zeros(len(ne), bool)
    for z in zones or []:
        out |= points_in_polygon(ne, z["polygon_ne"]) & (agl < float(z["ceiling_agl_m"]))
    return out


def no_fly_ceiling_agl(zones: Optional[Iterable[Dict[str, Any]]], n: float, e: float) -> Optional[float]:
    """The highest ceiling (m above ground) of the no-fly areas over (n, e); None outside them all."""
    ceilings = [float(z["ceiling_agl_m"]) for z in zones or [] if in_polygon((n, e), z["polygon_ne"])]
    return max(ceilings) if ceilings else None
