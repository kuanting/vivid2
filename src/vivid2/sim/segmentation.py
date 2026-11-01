"""Per-target visibility from Project AirSim's segmentation camera (ground truth v2).

The sim's 2D annotation boxes ignore occlusion: a hiker under tree canopy still
gets a box. Scoring "not visible" answers as misses would punish correct ones,
so the runner counts each target's pixels in the segmentation image instead.

Project AirSim assigns segmentation IDs when a scene loads, before VIVID 2's
targets are spawned, and its default IDs come from a hash of the name that
ignores digits and capitals (so "Hiker1" and "Hiker2" collide). After spawning,
the runner therefore sets every object to background, then gives each target
and distractor its own ID. Only objects present at that moment are drawn as
occluders, so maps must not stream geometry in later (see docs/map_authoring.md).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

BACKGROUND_ID = 0
TARGET_ID_BASE = 1
DISTRACTOR_ID_BASE = 101
MAX_ID = 255
MIN_VISIBLE_PX = 20  # below this a target is "partly visible": neither a hit nor a miss
PALETTE_ORDER = "rgb"  # palette entries are RGB; decoded images are BGR (verified on Blocks: Hiker1 = (6, 108, 153))


class SegmentationError(RuntimeError):
    pass


def palette() -> np.ndarray:
    """256 x 3 uint8 colours, indexed by segmentation ID, in the decoded image's channel order."""
    from projectairsim.image_utils import SEGMENTATION_PALLETE

    arr = np.asarray(SEGMENTATION_PALLETE, np.uint8)
    return arr if PALETTE_ORDER == "bgr" else arr[:, ::-1]


def assign_ids(world, targets: List[Dict[str, Any]], distractors: Iterable[Dict[str, Any]] = ()) -> Dict[str, int]:
    """Give every target and distractor a unique segmentation ID; everything else becomes background."""
    distractors = list(distractors)
    if len(targets) > DISTRACTOR_ID_BASE - TARGET_ID_BASE or len(distractors) > MAX_ID - DISTRACTOR_ID_BASE + 1:
        raise SegmentationError("too many targets or distractors for 8-bit segmentation IDs")
    world.set_segmentation_id_by_name(".*", BACKGROUND_ID, True, True)
    ids: Dict[str, int] = {}
    for base, objs in ((TARGET_ID_BASE, targets), (DISTRACTOR_ID_BASE, distractors)):
        for i, obj in enumerate(objs):
            if not world.set_segmentation_id_by_name(obj["id"], base + i, False, True):
                raise SegmentationError(f"could not set a segmentation ID on {obj['id']!r} (not spawned?)")
            ids[obj["id"]] = base + i
    return ids


def visibility(seg: np.ndarray, ids: Dict[str, int], colours: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Visible pixel count and tight visible box (x1, y1, x2, y2) per object in one segmentation image."""
    out: Dict[str, Dict[str, Any]] = {}
    for oid, sid in ids.items():
        mask = np.all(seg == colours[sid], axis=2)
        n = int(mask.sum())
        box: Optional[List[float]] = None
        if n:
            ys, xs = np.nonzero(mask)
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        out[oid] = {"visible_px": n, "bbox_vis": box}
    return out


def visibility_state(gt: Dict[str, Any], min_px: int = MIN_VISIBLE_PX) -> str:
    """'visible', 'partial' (don't care) or 'hidden' for one ground-truth entry.

    Entries without ``visible_px`` (ground truth v1) count as visible.
    """
    px = gt.get("visible_px")
    if px is None:
        return "visible"
    return "visible" if px >= min_px else ("partial" if px > 0 else "hidden")
