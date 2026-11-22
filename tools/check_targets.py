"""Authoring check: can every scored person (and clue) actually be seen, and does each sit on its ground?

    python tools/check_targets.py runs/<campaign>/coverage_gt/<run_id> [--min-factor 2]

Reads a run flown at survey height (the oracle or the survey agent; with ``--frame-bank rgbd`` for
the ground check) and reports, per target:

* the most visible pixels it reached in a straight-down search photo taken near survey height
  (above the start's ground, or above the ground under the drone on a terrain-following survey,
  ``search_area.altitude_ref``): each scored person must reach ``--min-factor`` x the visibility
  threshold (80 px at 1920 x 1440), each clue 1 x. Only photos where it also stands out from the
  ground around it count (contrast at least ``MIN_CONTRAST``, as the scoring's visibility rule):
  fog or dusk can wash out a target with plenty of pixels. A person buried so deep that the survey can't show it is not a fair
  target: move it, or make it a buried victim (damage.buried);
* its ground against its own feet. A target more than 0.3 m above its ground floats; more than
  0.3 m below it is sunk, which only partly buried people may be (authored below the ground on
  debris: on a run with a damage task, whose preflight checked that, it is a note, not a failure). The ground is the median height 1.5-3 m around it in the depth images when they are
  fine enough (``depth``), else the run's terrain model under it (``dtm``; for a person on a roof,
  ``"snap": "surface"``, the map's surface model, ``dsm``). Project AirSim's
  planar depth currently comes in whole metres, so today it is the terrain model, which doesn't
  hold meshes (rubble piles, boulders): check targets on meshes by eye.
  A target's position is taken as its feet, as for the VIVID 2 hikers; ``--centre-pivot`` takes
  it as the centre of a 1 m cube scaled by ``scale`` (the Blocks proxies).

Exits 1 if any target fails. See docs/damage.md ("Authoring a disaster scenario").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vivid2.geo import pixel_to_world, world_to_pixel
from vivid2.metrics.cameras import search_frames
from vivid2.recorder import RunLog
from vivid2.sim.segmentation import MIN_CONTRAST, min_visible_px
from vivid2.terrain import TerrainModel, from_meta

SURVEY_HEIGHT_SHARE = 0.8   # photos at least this share of the survey height count
RING_M = (1.5, 3.0)
FLOAT_M = 0.3


def _depth_mm(run_dir: Path, frame: Dict[str, Any]):
    from PIL import Image

    return np.asarray(Image.open(run_dir / frame["depth"]), np.float64) if frame.get("depth") else None


def depth_step_m(run_dir: Path, frame: Dict[str, Any]) -> float:
    """The depth image's resolution: the smallest step between its distinct values (inf without depth)."""
    mm = _depth_mm(run_dir, frame)
    if mm is None:
        return float("inf")
    u = np.unique(mm[mm > 0])
    return float(np.diff(u).min()) / 1000.0 if len(u) > 1 else float("inf")


def ground_ring(run_dir: Path, frame: Dict[str, Any], ned) -> float:
    """Median NED down of the depth image's ground RING_M around a point; NaN without depth."""
    mm = _depth_mm(run_dir, frame)
    if mm is None:
        return float("nan")
    depth = mm / 1000.0  # half resolution, mm
    w, h, fov = frame["width"], frame["height"], frame["fov_deg"]
    px = world_to_pixel(ned, frame["cam_pos"], frame["cam_q"], w, h, fov)
    if px is None:
        return float("nan")
    rng = px[2]
    f = (w / 2.0) / np.tan(np.radians(fov) / 2.0)
    r0, r1 = RING_M[0] * f / rng, RING_M[1] * f / rng
    vv, uu = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    uu, vv = uu * 2 + 1.0, vv * 2 + 1.0  # full-resolution pixel of each depth sample
    d = np.hypot(uu - px[0], vv - px[1])
    sel = (d >= r0) & (d <= r1) & (depth > 0.1)
    if not sel.any():
        return float("nan")
    downs = [pixel_to_world(u, v, z, frame["cam_pos"], frame["cam_q"], w, h, fov)[2]
             for u, v, z in zip(uu[sel][::7], vv[sel][::7], depth[sel][::7])]
    return float(np.median(downs))


def check(run_dir: Path, min_factor: float = 2.0, centre_pivot: bool = False) -> List[Dict[str, Any]]:
    log = RunLog.load(run_dir)
    meta = log.meta
    terrain = from_meta(meta)
    surface = TerrainModel.load(meta["surface"]["file"]) if meta.get("surface") else None
    alt = float(meta["search_area"]["altitude_m"])
    ground = float((meta.get("mission_spec") or {}).get("ground_down_m", meta["search_area"].get("ground_down_m", 0.0)))

    def height(f) -> float:  # the photo's height: above the start's ground, or the ground under it
        if meta["search_area"].get("altitude_ref") == "terrain":
            return float(terrain.down_at(f["cam_pos"][0], f["cam_pos"][1])) - f["cam_pos"][2]
        return ground - f["cam_pos"][2]

    frames = [f for f in search_frames(log["frames"], meta)
              if not f.get("gimbal") and height(f) >= SURVEY_HEIGHT_SHARE * alt]
    rows = []
    items = [(t, "person", min_factor) for t in meta["targets"]] + [(c, "clue", 1.0) for c in meta.get("clues") or []]
    for obj, kind, factor in items:
        best, best_frame, washed = 0, None, 0
        for f in frames:
            if kind == "person":
                seen = [g for g in f.get("gt", []) if g["object_id"] == obj["id"]]
            else:
                seen = [(f.get("clues_visible") or {}).get(obj["id"]) or {}]
            for g in seen:
                px = g.get("visible_px") or 0
                if g.get("contrast") is not None and g["contrast"] < MIN_CONTRAST:
                    washed = max(washed, px)  # enough pixels perhaps, but fog or dusk washed it out
                elif px > best:
                    best, best_frame = px, f
        need = factor * (min_visible_px(best_frame["width"], best_frame["height"]) if best_frame else min_visible_px())
        n, e, z = (float(v) for v in obj["ned"])
        feet = z + (float((obj.get("scale") or [1, 1, 1])[2]) / 2.0 if centre_pivot else 0.0)
        if best_frame is not None and depth_step_m(run_dir, best_frame) <= FLOAT_M / 3:
            under, source = ground_ring(run_dir, best_frame, [n, e, feet]), "depth"
        elif obj.get("snap") == "surface" and surface is not None:
            under, source = float(surface.down_at(n, e)), "dsm"
        else:
            under, source = float(terrain.down_at(n, e)), "dtm"
        lift = under - feet  # NED down: feet above the ground is positive
        problems, notes = [], []
        if best < need:
            low = f"; {washed} px with contrast below {MIN_CONTRAST:g}" if washed > best else ""
            problems.append(f"at most {best} px from survey height (needs {need:.0f}){low}")
        if np.isfinite(lift) and lift > FLOAT_M:
            problems.append(f"floats {lift:.2f} m above the ground around it")
        if np.isfinite(lift) and lift < -FLOAT_M and kind == "person":
            if meta.get("damage"):  # vivid2.damage.check put it on debris, flood or rubble before the run
                notes.append(f"partly buried {-lift:.2f} m")
            else:
                problems.append(f"sits {-lift:.2f} m below the ground around it (fine only if partly buried on purpose)")
        rows.append({"id": obj["id"], "kind": kind, "max_px": int(best), "need_px": int(round(need)),
                     "frame_id": best_frame["frame_id"] if best_frame else None,
                     "lift_m": None if not np.isfinite(lift) else round(lift, 2), "ground": source,
                     "problems": problems, "notes": notes})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--min-factor", type=float, default=2.0, help="persons must reach this x the visibility threshold")
    ap.add_argument("--centre-pivot", action="store_true",
                    help="positions are the centres of scaled 1 m cubes (the Blocks proxies), not feet")
    args = ap.parse_args(argv)
    rows = check(Path(args.run_dir), args.min_factor, args.centre_pivot)
    bad = 0
    for r in rows:
        lift = "n/a" if r["lift_m"] is None else f"{r['lift_m']:+.2f} m"
        status = ("OK" + "".join(f" ({n})" for n in r["notes"])) if not r["problems"] else "FAIL: " + "; ".join(r["problems"])
        print(f"{r['kind']:6s} {r['id']:14s} max {r['max_px']:6d} px (need {r['need_px']}) frame {r['frame_id']}  "
              f"feet vs {r['ground']} {lift}  {status}")
        bad += bool(r["problems"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
