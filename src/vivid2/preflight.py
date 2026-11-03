"""Checks run after a scenario's scene is loaded and before the drone flies.

Project AirSim can't load an Unreal level or report which one is open, and it
renames a spawned actor whose name an existing actor already contains. Either
mistake would otherwise show up only as a run where nothing is ever found, so
the runner stops early with a message that says what to fix.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class PreflightError(RuntimeError):
    """The sim is not in a state where this run can produce a valid result."""


MARKER_PREFIX = "BP_VIVID2Map_"
# Project AirSim matches actor names (not editor labels). A placed blueprint is
# named <Class>_C_<n>, or <Class>_C_UAID_<hex>_<n>; nothing else may follow the
# map name, so ForestSAR's marker never matches ForestSAR_Dense's.
MARKER_SUFFIX = r"(_C)?(_UAID_[0-9A-Fa-f]+)?(_\d+)?"
MIN_TARGET_SIZE_M = 0.05
MAX_TARGET_SIZE_M = 50.0


def scenario_map(spec: Dict[str, Any]) -> Optional[str]:
    return spec.get("scene", {}).get("map")


def map_check_patterns(scene: Dict[str, Any]) -> List[str]:
    """Actor-name regexes (full match) that must exist in the open level.

    VIVID 2 maps carry a marker actor ``BP_VIVID2Map_<map>``; other maps (such
    as the Blocks map) list a few of their own actors under ``map_check``.
    """
    if "map_check" in scene:
        return list(scene["map_check"]["actors"])
    if scene.get("map"):
        return [MARKER_PREFIX + re.escape(scene["map"]) + MARKER_SUFFIX]
    return []


def check_map(world, scene: Dict[str, Any]) -> List[str]:
    patterns = map_check_patterns(scene)
    missing = [p for p in patterns if not world.list_objects(p)]
    if missing:
        markers = world.list_objects(MARKER_PREFIX + ".*") or []
        found = f"; the open level has {sorted(markers)}" if markers else ""
        raise PreflightError(
            f"the open Unreal level is not map {scene.get('map', '?')!r}: no actor matches {missing}{found}. "
            "Open that map and press Play (editor), or restart the packaged build with "
            f"/Game/VIVID2/Maps/{scene.get('map', '<map>')}, then run again (finished runs are skipped)."
        )
    return patterns


def check_spawned(world, targets: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Every target exists under its exact name with a plausible size."""
    try:
        from projectairsim.types import BoxAlignment

        align = BoxAlignment.WORLD_AXIS
    except ImportError:  # unit tests without the sim client
        align = 0
    sizes: Dict[str, List[float]] = {}
    for t in targets:
        if not world.list_objects(re.escape(t["id"])):
            similar = world.list_objects(".*" + re.escape(t["id"]) + ".*") or []
            raise PreflightError(
                f"target {t['id']!r} was not spawned under its own name (found {sorted(similar)}). "
                "Project AirSim renames a spawned actor when an existing actor's name contains it; "
                "give the target a more specific id."
            )
        size = world.get_3d_bounding_box(t["id"], align)["size"]
        dims = [float(size["x"]), float(size["y"]), float(size["z"])]
        # The largest side tells a spawned object from an empty (0 m) or runaway box; flat things
        # such as a tarp are legitimately only centimetres thick.
        if not (MIN_TARGET_SIZE_M <= max(dims) <= MAX_TARGET_SIZE_M and min(dims) >= 0.0):
            raise PreflightError(f"target {t['id']!r} ({t['asset']}) has an implausible size {dims} m")
        sizes[t["id"]] = dims
    return sizes


def run_preflight(world, spec: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all checks; returns what was verified, for meta.json."""
    scene = spec.get("scene", {})
    return {
        "map": scene.get("map"),
        "map_actors_checked": check_map(world, scene),
        "target_sizes_m": check_spawned(world, targets),
    }
