"""Per-target visibility from Project AirSim's segmentation camera (ground truth v2).

The sim's 2D annotation boxes ignore occlusion: a hiker under tree canopy still
gets a box. Scoring "not visible" answers as misses would punish correct ones,
so the runner counts each target's pixels in the segmentation image instead.

The segmentation image is geometry only: fog does not touch it. A hiker that
fog has washed into the ground still has all its pixels there. So the runner
also measures each object's contrast in the rendered colour image: the CIELAB
colour difference (Delta E 1976) between the object's own pixels and a thin
ring of ground around them. Below ``MIN_CONTRAST`` (about where a person stops
showing in dusk fog from 55 m) the target counts as "partial", like one too
small to judge: neither a hit nor a miss. Camouflage in clear air (a yellow
jacket among yellow birch crowns) stays well above it.

Project AirSim assigns segmentation IDs when a scene loads, before VIVID 2's
targets are spawned, and its default IDs come from a hash of the name that
ignores digits and capitals (so "Hiker1" and "Hiker2" collide). After spawning,
the runner therefore sets every object to background, then gives each target,
distractor and fire its own ID. Only objects present at that moment are drawn as
occluders, so maps must not stream geometry in later (see docs/map_authoring.md).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

BACKGROUND_ID = 0
TARGET_ID_BASE = 1
DISTRACTOR_ID_BASE = 101
FIRE_ID_BASE = 201  # fires (vivid2.hazards.fire): recorded per frame, but neither targets nor distractors
MAX_ID = 255
MIN_VISIBLE_PX = 20  # at 960x720; below this a target is "partly visible": neither a hit nor a miss
REFERENCE_PIXELS = 960 * 720  # the threshold scales with image area, so "visible" means the same patch of ground
MIN_CONTRAST = 6.0  # CIELAB Delta E against the surrounding ground; below this fog has washed the target out
PALETTE_ORDER = "rgb"  # palette entries are RGB; decoded images are BGR (verified on Blocks: Hiker1 = (6, 108, 153))


class SegmentationError(RuntimeError):
    pass


def palette() -> np.ndarray:
    """256 x 3 uint8 colours, indexed by segmentation ID, in the decoded image's channel order."""
    from projectairsim.image_utils import SEGMENTATION_PALLETE

    arr = np.asarray(SEGMENTATION_PALLETE, np.uint8)
    return arr if PALETTE_ORDER == "bgr" else arr[:, ::-1]


def assign_ids(world, targets: List[Dict[str, Any]], distractors: Iterable[Dict[str, Any]] = (),
               fires: Iterable[Dict[str, Any]] = ()) -> Dict[str, int]:
    """Give every target, distractor and fire a unique segmentation ID; everything else becomes background."""
    distractors, fires = list(distractors), list(fires)
    if (len(targets) > DISTRACTOR_ID_BASE - TARGET_ID_BASE or len(distractors) > FIRE_ID_BASE - DISTRACTOR_ID_BASE
            or len(fires) > MAX_ID - FIRE_ID_BASE + 1):
        raise SegmentationError("too many targets, distractors or fires for 8-bit segmentation IDs")
    world.set_segmentation_id_by_name(".*", BACKGROUND_ID, True, True)
    ids: Dict[str, int] = {}
    for base, objs in ((TARGET_ID_BASE, targets), (DISTRACTOR_ID_BASE, distractors), (FIRE_ID_BASE, fires)):
        for i, obj in enumerate(objs):
            if not world.set_segmentation_id_by_name(obj["id"], base + i, False, True):
                raise SegmentationError(f"could not set a segmentation ID on {obj['id']!r} (not spawned?)")
            ids[obj["id"]] = base + i
    return ids


def visibility(seg: np.ndarray, ids: Dict[str, int], colours: np.ndarray,
               rgb: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
    """Visible pixel count and tight visible box (x1, y1, x2, y2) per object in one segmentation image.

    With the matching colour image ``rgb`` (BGR, same size), also the object's ``contrast``
    against its surroundings (see ``contrast``); None when it has no visible pixels.
    """
    out: Dict[str, Dict[str, Any]] = {}
    labelled = None
    if rgb is not None:
        labelled = ~np.all(seg == colours[BACKGROUND_ID], axis=2)
    for oid, sid in ids.items():
        mask = np.all(seg == colours[sid], axis=2)
        n = int(mask.sum())
        box: Optional[List[float]] = None
        if n:
            ys, xs = np.nonzero(mask)
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        out[oid] = {"visible_px": n, "bbox_vis": box}
        if rgb is not None:
            out[oid]["contrast"] = contrast(rgb, mask, box, labelled) if n else None
    return out


def split_seen(vis: Dict[str, Dict[str, Any]], target_ids: Iterable[str],
               fire_ids: Iterable[str] = ()) -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    """Split one frame's visibility into what the frame record keeps besides the targets.

    Returns ``(distractors_visible, fires_visible)``: visible pixels per distractor, and visible
    pixels and contrast per fire, each only for objects with visible pixels.
    """
    targets, fires = set(target_ids), set(fire_ids)
    distractors = {oid: v["visible_px"] for oid, v in vis.items()
                   if oid not in targets and oid not in fires and v["visible_px"]}
    burning = {oid: {"visible_px": v["visible_px"], "contrast": v.get("contrast")} for oid, v in vis.items()
               if oid in fires and v["visible_px"]}
    return distractors, burning


RING_GAP_PX = 2    # at 1920 px wide: skip the anti-aliased edge between object and ground
RING_WIDTH_PX = 8  # at 1920 px wide: the ground the object is seen against


def contrast(rgb: np.ndarray, mask: np.ndarray, box: List[float], labelled: Optional[np.ndarray] = None) -> Optional[float]:
    """CIELAB Delta E (1976) between the mean colour of the object's pixels and of a ring around them.

    The ring is ``RING_GAP_PX`` to ``RING_GAP_PX + RING_WIDTH_PX`` pixels out (scaled to the image
    width) and leaves out other labelled objects (``labelled``). Fog pulls both towards the fog
    colour, so the difference shrinks as the fog thickens. None if the ring is empty.
    """
    import cv2

    h, w = mask.shape
    scale = w / 1920.0
    gap = max(1, int(round(RING_GAP_PX * scale)))
    outer = gap + max(2, int(round(RING_WIDTH_PX * scale)))
    x1, y1 = max(int(box[0]) - outer, 0), max(int(box[1]) - outer, 0)
    x2, y2 = min(int(box[2]) + outer, w), min(int(box[3]) + outer, h)
    m = mask[y1:y2, x1:x2].astype(np.uint8)
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)))
    far = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1)))
    ring = far.astype(bool) & ~near.astype(bool)
    if labelled is not None:
        ring &= ~labelled[y1:y2, x1:x2]
    if not ring.any():
        return None
    lab = cv2.cvtColor(rgb[y1:y2, x1:x2].astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return float(np.linalg.norm(lab[m.astype(bool)].mean(axis=0) - lab[ring].mean(axis=0)))


def min_visible_px(width: Optional[int] = None, height: Optional[int] = None) -> int:
    """Visible-pixel threshold for an image size: MIN_VISIBLE_PX at 960x720, scaled by area
    (80 px at 1920x1440), never below MIN_VISIBLE_PX."""
    if not width or not height:
        return MIN_VISIBLE_PX
    return max(MIN_VISIBLE_PX, int(round(MIN_VISIBLE_PX * width * height / REFERENCE_PIXELS)))


def visibility_state(gt: Dict[str, Any], min_px: Optional[int] = None, frame_wh=None) -> str:
    """'visible', 'partial' (don't care) or 'hidden' for one ground-truth entry.

    The pixel threshold is ``min_px``, else scaled to the frame size ``frame_wh``. A target
    with enough pixels but contrast below ``MIN_CONTRAST`` (washed out by fog) is 'partial'.
    Entries without ``visible_px`` (ground truth v1) count as visible; entries without
    ``contrast`` (ground truth v2) are judged by pixels alone.
    """
    px = gt.get("visible_px")
    if px is None:
        return "visible"
    if min_px is None:
        min_px = min_visible_px(*(frame_wh or (None, None)))
    if px < min_px:
        return "partial" if px > 0 else "hidden"
    c = gt.get("contrast")
    return "partial" if c is not None and c < MIN_CONTRAST else "visible"
