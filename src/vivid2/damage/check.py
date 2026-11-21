"""Placement rules for scenarios with clues or a damage task (see docs/damage.md).

``check_damage_spec`` runs when a scenario loads (no ground truth needed); ``check_damage_gt``
runs before each run (and in a unit test over every scenario in the repo), against the map's
ground truth and terrain. Both return a list of problems; empty means fine. Scenarios without
clues or a damage block are not checked here at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from vivid2.damage import COMPONENTS, DAMAGE_MISSIONS, GRADE_LEVEL
from vivid2.damage.geometry import distance_to_polygon
from vivid2.geo import points_in_polygon

SUNK_M = 0.2               # a target authored this far below the ground is partly buried
SUNK_DEBRIS_RADIUS_M = 1.0  # ... and must lie this close to debris, flood or rubble
BURIED_AREA_RADIUS_M = 2.0  # a buried victim lies this close to a damage area cell ...
ROOF_FLAT_M = 0.3          # a surface-snapped target's jitter disc must be this flat ...
ROOF_ABOVE_GROUND_M = 0.5  # ... and this far above the ground (a roof, or water)


def _contains(a: str, b: str) -> bool:
    return a != b and (a in b or b in a)


def check_damage_spec(spec: Dict[str, Any]) -> List[str]:
    clues = spec.get("clues") or []
    d = spec.get("damage")
    if not clues and not d:
        return []
    problems: List[str] = []
    if d and spec.get("mission") not in DAMAGE_MISSIONS:
        problems.append(f"a damage task needs a disaster mission ({', '.join(DAMAGE_MISSIONS)}), "
                        f"not {spec.get('mission')!r}")
    products = (d or {}).get("products", [])
    if clues and "clues" not in products:
        problems.append("the scenario has clues but its damage task does not ask for them (damage.products)")
    if "clues" in products and not clues:
        problems.append("damage.products asks for clues but the scenario has none")
    for k in (d or {}).get("weights") or {}:
        if k not in COMPONENTS:
            problems.append(f"damage.weights: unknown component {k!r} (one of {', '.join(COMPONENTS)})")
    if "areas" not in products and (d or {}).get("area_classes"):
        problems.append("damage.area_classes without the areas product")
    # ids: unique across everything spawned or scored, none containing another (Project AirSim
    # renames actors whose name contains another's)
    ids = [o["id"] for key in ("targets", "distractors", "fires", "clues") for o in spec.get(key) or []]
    ids += [b["id"] for b in (d or {}).get("buried") or []]
    seen = set()
    for i in ids:
        if i in seen:
            problems.append(f"id {i!r} is used twice")
        seen.add(i)
    for a in ids:
        for b in ids:
            if a < b and _contains(a, b):
                problems.append(f"id {a!r} and {b!r}: one contains the other")
    aoi = (d or {}).get("aoi_ne")
    if aoi:
        outside = ~points_in_polygon(np.asarray(aoi, float), spec["search_area"]["polygon_ne"])
        # vertices on the search polygon's edge count as inside
        far = [p for p, o in zip(aoi, outside) if o and distance_to_polygon(np.asarray([p], float),
                                                                            spec["search_area"]["polygon_ne"])[0] > 1e-6]
        if far:
            problems.append(f"damage.aoi_ne leaves the search area at {far[:3]}")
    return problems


def check_damage_gt(spec: Dict[str, Any], task, targets: List[Dict[str, Any]], authored_ground_down_m: float,
                    surface=None, terrain=None, buried: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """Rules against the ground truth. ``targets``: the run's resolved targets (on the terrain);
    ``task``: its DamageTask (None if the scenario has no damage block)."""
    problems: List[str] = []
    surf_targets = [t for t in targets if t.get("snap") == "surface"]
    if surf_targets:
        if surface is None:
            problems.append("targets snap to the surface but the map has no surface model (vivid2 damage build)")
        else:
            jitter = float(spec.get("target_jitter_m", 0.0))
            for t in surf_targets:
                r = float(t.get("jitter_m", jitter))
                n, e = t["ned"][0], t["ned"][1]
                a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ring_n = np.concatenate([[n], n + r * np.cos(a)])
                ring_e = np.concatenate([[e], e + r * np.sin(a)])
                s = np.asarray(surface.down_at(ring_n, ring_e), float)
                g = np.asarray((terrain.down_at(ring_n, ring_e) if terrain is not None else s), float)
                if s.max() - s.min() > ROOF_FLAT_M:
                    problems.append(f"{t['id']}: the surface under it (with its jitter) is not flat "
                                    f"({s.max() - s.min():.2f} m)")
                if (g - s).min() < ROOF_ABOVE_GROUND_M:
                    problems.append(f"{t['id']}: snaps to the surface but is not on a roof or water")
    if task is None:
        return problems
    gt = task.gt
    for p in task.products:
        if p == "areas" and not task.area_classes:
            problems.append("damage.products asks for areas but the ground truth has no area classes")
        if p == "roads" and not task.roads:
            problems.append("damage.products asks for roads but no road segment lies in the area of interest")
        if p == "buildings" and not task.buildings:
            problems.append("damage.products asks for buildings but no building lies in the area of interest")
    for k in task.area_classes:
        if k not in gt.area_classes:
            problems.append(f"damage.area_classes: {k!r} is not in the map's ground truth ({sorted(gt.area_classes)})")
    collapsed = [b for b in gt.buildings if b["grade"] == "collapsed"]
    damaged = {b["id"]: b for b in gt.buildings if GRADE_LEVEL[b["grade"]] >= 1}

    def near_damage(n: float, e: float, radius: float) -> bool:
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pn = np.concatenate([[n], n + radius * np.cos(a)])
        pe = np.concatenate([[e], e + radius * np.sin(a)])
        if any(gt.bits_at(pn, pe)) or (gt.water_at(pn, pe) > 0).any():
            return True
        p = np.asarray([[n, e]], float)
        for b in collapsed:
            spill = float(b.get("spill_m", 0.0)) + radius
            if distance_to_polygon(p, b["footprint_ne"])[0] <= spill:
                return True
        return False

    for t in targets:
        if t.get("snap") == "surface":
            continue
        ground = float(terrain.down_at(t["ned"][0], t["ned"][1])) if terrain is not None else authored_ground_down_m
        sunk = float(t["ned"][2]) - ground  # NED down: below the ground is positive
        if sunk > SUNK_M and not near_damage(t["ned"][0], t["ned"][1], SUNK_DEBRIS_RADIUS_M):
            problems.append(f"{t['id']}: partly buried ({sunk:.1f} m) but not on debris, flood or rubble")
    for b in buried or task.buried:
        n, e = b["ned"][0], b["ned"][1]
        if b.get("building"):
            if b["building"] not in damaged:
                problems.append(f"buried {b['id']}: building {b['building']!r} is not damaged in the ground truth")
            elif not points_in_polygon(np.asarray([[n, e]], float), damaged[b["building"]]["footprint_ne"])[0] and \
                    distance_to_polygon(np.asarray([[n, e]], float), damaged[b["building"]]["footprint_ne"])[0] > \
                    float(damaged[b["building"]].get("spill_m", 0.0)):
                problems.append(f"buried {b['id']}: not in or under the rubble of building {b['building']!r}")
        elif not near_damage(n, e, BURIED_AREA_RADIUS_M):
            problems.append(f"buried {b['id']}: not within {BURIED_AREA_RADIUS_M:g} m of a damage area")
    for c in task.clues:
        if not points_in_polygon(np.asarray([c["ned"][:2]], float), task.aoi)[0]:
            problems.append(f"clue {c['id']}: outside the area of interest")
    return problems
