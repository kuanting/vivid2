"""Count payload frames where the landscape failed to render.

A missing landscape tile shows as a bright, colourless void (the sky) under
floating trees. A nadir camera at survey height never sees the sky, so any
sizeable sky-coloured area in a near-nadir frame is a hole. Frames tilted more
than ``--max-tilt`` degrees from nadir (hard turns) can see the horizon and are
skipped. When the run saved depth (``--frame-bank rgbd``), only void pixels with
no depth count, so pale boulders seen up close are not mistaken for holes.

usage: python tools/count_holes.py RUN_DIR [RUN_DIR ...] [--min-frac 0.0012] [--max-tilt 25] [--list]

RUN_DIR is a run directory with ``frames.jsonl`` and ``frames/`` (recorded with
``--frame-bank``), or any directory above such run directories.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vivid2.geo import quat_to_rot  # noqa: E402


def tilt_deg(cam_q) -> float:
    """Angle between the camera's optical axis and straight down."""
    axis = quat_to_rot(cam_q)[:, 0]
    return float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0))))


def void_fraction(path: Path, depth_path: Path | None = None) -> float:
    """Share of the image covered by the largest solid bright, colourless blob (sky showing through).

    An opening removes thin bright features (birch trunks, logs); taking the
    largest blob keeps scattered pale rocks from adding up. With a depth image,
    only pixels with no depth count: a hole has nothing behind it, a pale
    boulder does.
    """
    im = cv2.imread(str(path))
    if im is None:
        return float("nan")
    small = cv2.resize(im, (480, 360), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 200) & (hsv[..., 1] < 40)).astype(np.uint8)
    if depth_path is not None:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is not None:  # frame bank depth: 0 = no data
            mask &= cv2.resize((depth == 0).astype(np.uint8), (480, 360), interpolation=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return largest / float(mask.size)


def run_dirs(roots):
    for root in roots:
        root = Path(root)
        hits = [root] if (root / "frames.jsonl").exists() else sorted(p.parent for p in root.rglob("frames.jsonl"))
        yield from hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-frac", type=float, default=0.0012, help="largest-void share that makes a frame bad")
    ap.add_argument("--max-tilt", type=float, default=25.0, help="skip frames tilted more than this from nadir")
    ap.add_argument("--list", action="store_true", help="print each bad frame")
    args = ap.parse_args(argv)

    total = checked = bad = tilted = 0
    for rd in run_dirs(args.dirs):
        n_bad = n = 0
        for line in (rd / "frames.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("rgb"):
                continue
            total += 1
            if tilt_deg(rec["cam_q"]) > args.max_tilt:
                tilted += 1
                continue
            frac = void_fraction(rd / rec["rgb"], rd / rec["depth"] if rec.get("depth") else None)
            n += 1
            if frac >= args.min_frac:
                n_bad += 1
                if args.list:
                    print(f"  {rd.name}/{rec['rgb']}  void {frac:.3f}")
        checked += n
        bad += n_bad
        print(f"{rd}: {n_bad}/{n} frames with holes")
    print(f"total: {bad}/{checked} frames with holes ({tilted} tilted frames skipped, {total} recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
