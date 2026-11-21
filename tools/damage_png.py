"""Draw a run's damage task over an orthomosaic of its photos: ground truth, reports, observed ground.

    python tools/damage_png.py runs/<campaign>/<agent>/<run_id> [--out damage.png] [--m-per-px 0.25]

Needs a run recorded with ``--frame-bank rgb`` (or rgbd). Straight-down search photos are warped
onto the ground (flat at the terrain under each photo's centre) and pasted, nearest photo on top.
Over it:

    ground truth   area classes (tinted), road segments (red = blocked, green = open),
                   buildings (outline: green none, orange partial, red collapsed), clues (+),
                   buried victims (x)
    reports        reported areas (white outline), roads (thick dashes), buildings (inner
                   outline in the reported grade's colour), clues (o)
    observed       the ground the run's photos observed (dim outside it)

North is up. For authoring a map (does the ground truth match what the photos show?) and for
looking at what a stack reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vivid2.damage.geometry import raster_polygons
from vivid2.damage.run import DamageTask
from vivid2.geo import pixel_ray_to_ground
from vivid2.metrics.cameras import search_frames
from vivid2.metrics.damage import check_reports
from vivid2.recorder import RunLog
from vivid2.terrain import from_meta

GRADE_BGR = {"none": (80, 200, 80), "partial": (0, 160, 255), "collapsed": (40, 40, 230)}
CLASS_BGR = [(0, 140, 255), (255, 120, 0), (200, 0, 200), (0, 200, 200)]


def render(run_dir: Path, m_per_px: float = 0.25) -> np.ndarray:
    log = RunLog.load(run_dir)
    meta = log.meta
    terrain = from_meta(meta)
    task = DamageTask.from_meta(meta, terrain)
    aoi = np.asarray(task.aoi, float)
    lo, hi = aoi.min(axis=0) - 10.0, aoi.max(axis=0) + 10.0
    W = int(np.ceil((hi[1] - lo[1]) / m_per_px))  # columns: east
    H = int(np.ceil((hi[0] - lo[0]) / m_per_px))  # rows: north, top = hi

    def to_px(ne):
        ne = np.asarray(ne, float).reshape(-1, 2)
        return np.column_stack([(ne[:, 1] - lo[1]) / m_per_px, (hi[0] - ne[:, 0]) / m_per_px]).astype(np.float32)

    canvas = np.full((H, W, 3), 40, np.uint8)
    best = np.full((H, W), np.inf, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for f in search_frames(log["frames"], meta):
        if f.get("gimbal") or not f.get("rgb") or not (run_dir / f["rgb"]).exists():
            continue
        img = cv2.imread(str(run_dir / f["rgb"]))
        w, h = f["width"], f["height"]
        down = float(terrain.down_at(f["cam_pos"][0], f["cam_pos"][1]))
        corners = [pixel_ray_to_ground(u, v, f["cam_pos"], f["cam_q"], w, h, f["fov_deg"], down)
                   for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
        if any(c is None for c in corners):
            continue
        sx, sy = img.shape[1] / w, img.shape[0] / h  # the bank's JPEG may be smaller than the camera image
        src = np.float32([[0, 0], [w * sx, 0], [w * sx, h * sy], [0, h * sy]])
        Hm = cv2.getPerspectiveTransform(src, to_px([c[:2] for c in corners]))
        warped = cv2.warpPerspective(img, Hm, (W, H))
        cover = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), Hm, (W, H)) > 0
        c = to_px([f["cam_pos"][:2]])[0]
        dist = np.hypot(xx - c[0], yy - c[1]).astype(np.float32)
        take = cover & (dist < best)
        canvas[take] = warped[take]
        best[take] = dist[take]

    # observed ground: dim the rest
    frames = search_frames(log["frames"], meta)
    observed = np.zeros(len(task.cells), bool)
    for f in frames:
        observed |= task.seen(f)
    obs_img = np.zeros((H, W), np.uint8)
    half = task.cell_m / 2 / m_per_px
    for p in to_px(task.cells[observed]):
        cv2.rectangle(obs_img, (int(p[0] - half), int(p[1] - half)), (int(p[0] + half), int(p[1] + half)), 1, -1)
    canvas[obs_img == 0] = (canvas[obs_img == 0] * 0.4).astype(np.uint8)

    overlay = canvas.copy()
    for k, name in enumerate(task.area_classes):
        for p in to_px(task.cells[task.gt_cells[name]]):
            cv2.rectangle(overlay, (int(p[0] - half), int(p[1] - half)), (int(p[0] + half), int(p[1] + half)),
                          CLASS_BGR[k % len(CLASS_BGR)], -1)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    for r in task.roads:
        cv2.polylines(canvas, [to_px(r["polyline_ne"]).astype(np.int32)], False,
                      (40, 40, 230) if r["blocked"] else (80, 200, 80), max(1, int(round(r["width_m"] / m_per_px / 3))))
    for b in task.buildings:
        cv2.polylines(canvas, [to_px(b["footprint_ne"]).astype(np.int32)], True, GRADE_BGR[b["grade"]], 3)
    for c in task.clues:
        p = to_px([c["ned"][:2]])[0]
        cv2.drawMarker(canvas, (int(p[0]), int(p[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    for b in task.buried:
        p = to_px([b["ned"][:2]])[0]
        cv2.drawMarker(canvas, (int(p[0]), int(p[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

    valid, _, _ = check_reports(log, task)
    for name in task.area_classes:
        rep = raster_polygons([r for r in valid if r["kind"] == "area" and r["area_class"] == name], task.cells) & observed
        mask = np.zeros((H, W), np.uint8)
        for p in to_px(task.cells[rep]):
            cv2.rectangle(mask, (int(p[0] - half), int(p[1] - half)), (int(p[0] + half), int(p[1] + half)), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)
    final_roads, final_grades = {}, {}
    for r in valid:
        if r["kind"] == "road":
            final_roads[r["segment"]] = r["blocked"]
        elif r["kind"] == "building":
            final_grades[r["building"]] = r["grade"]
        elif r["kind"] == "clue":
            p = to_px([r["ned"][:2]])[0]
            cv2.circle(canvas, (int(p[0]), int(p[1])), 10, (255, 255, 255), 2)
    for r in task.roads:
        if r["id"] in final_roads:
            pts = to_px(r["polyline_ne"]).astype(np.int32)
            col = (40, 40, 230) if final_roads[r["id"]] else (80, 200, 80)
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(2, int(np.hypot(*(b - a)) / 12))
                for k in range(0, n, 2):
                    p0 = a + (b - a) * k / n
                    p1 = a + (b - a) * (k + 1) / n
                    cv2.line(canvas, tuple(int(v) for v in p0), tuple(int(v) for v in p1), col, 7)
    for b in task.buildings:
        if b["id"] in final_grades:
            fp = np.asarray(b["footprint_ne"], float)
            inner = fp.mean(axis=0) + 0.7 * (fp - fp.mean(axis=0))
            cv2.polylines(canvas, [to_px(inner).astype(np.int32)], True, GRADE_BGR[final_grades[b["id"]]], 2)
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="default: damage.png in the run folder")
    ap.add_argument("--m-per-px", type=float, default=0.25)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("damage"):
        print("this run has no damage task")
        return 1
    img = render(run_dir, args.m_per_px)
    out = Path(args.out) if args.out else run_dir / "damage.png"
    cv2.imwrite(str(out), img)
    print(f"{out}: {img.shape[1]} x {img.shape[0]} px, {args.m_per_px} m per px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
