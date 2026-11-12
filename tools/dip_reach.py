"""Where can the dip baseline (``--dip on``) see? Use it to place targets in reach of a dip, or out of it.

Rebuilds the canopy map from a run recorded with ``--frame-bank rgbd`` the way the agent does (nadir
depth, the coarse 20 m DEM the stack is given), then prints:

- the open ground (nothing higher than 3 m) and the spots wide enough to descend into,
- the share of the search area whose ground a dip could see (open cells within the 12 m footprint
  of a usable spot),
- each target's distance to the nearest usable spot and the canopy height over it,
- the widest clearings, as candidate placements.

usage: python tools/dip_reach.py RUN_DIR [--min-radius 6] [--png reach.png]

The PNG is north up: red = usable spots, yellow = open ground a dip would see, brown = other open
ground, green = canopy, white circles = targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from vivid2.runner import DUT_DEM_CELL_M
from vivid2.stack.reference.lowflight import CanopyMap, DipSettings
from vivid2.terrain import from_meta

# 12 m footprint of the 80 deg, 4:3 payload camera with the heading north (image width = east)
HALF_N_M, HALF_E_M = 7.5, 10.0


def build_map(run: Path, meta: dict) -> CanopyMap:
    ms = meta["mission_spec"]
    cmap = CanopyMap(ms["search_polygon_ne"], from_meta(meta).downsample(DUT_DEM_CELL_M))
    for line in (run / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if not r.get("depth") or r.get("camera", "Payload") != "Payload" or r.get("gimbal"):
            continue  # the canopy map is built from straight-down payload photos only
        # frame bank depth: uint16 mm at half resolution, 0 = no data (the field of view is unchanged)
        d = cv2.imread(str(run / r["depth"]), cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
        cmap.add(SimpleNamespace(depth=d, fov_deg=r["fov_deg"], cam_q=r["cam_q"], cam_pos=r["cam_pos"]))
    return cmap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--min-radius", type=float, default=DipSettings.min_radius_m)
    ap.add_argument("--png", type=Path)
    a = ap.parse_args()
    meta = json.loads((a.run / "meta.json").read_text(encoding="utf-8"))
    if (meta.get("frame_bank") or {}).get("mode") != "rgbd":
        raise SystemExit(f"{a.run}: needs a run recorded with --frame-bank rgbd")
    cmap = build_map(a.run, meta)
    inside = cmap.inside
    area = int(inside.sum())
    opened = cmap.open_mask() & inside
    spots = (cmap._distance() >= a.min_radius) & inside
    kn, ke = int(round(HALF_N_M / cmap.cell)), int(round(HALF_E_M / cmap.cell))
    reach = cv2.dilate(spots.astype(np.uint8), np.ones((2 * kn + 1, 2 * ke + 1), np.uint8)).astype(bool) & opened
    seen = np.isfinite(cmap.top) & inside
    print(f"{a.run.name}: search area {area * cmap.cell ** 2 / 1e4:.1f} ha, {seen.sum() / area:.0%} mapped")
    print(f"  open ground (nothing above {cmap.open_height_m:g} m): {opened.sum() / area:.1%} of the area")
    print(f"  ground a dip could see (spots of radius >= {a.min_radius:g} m): {reach.sum() / area:.1%} of the area")
    ij = np.argwhere(spots)
    spot_ne = np.stack([cmap.n0 + ij[:, 0] * cmap.cell, cmap.e0 + ij[:, 1] * cmap.cell], 1)
    for t in meta["targets"]:
        p = np.asarray(t["ned"][:2], float)
        near = float(np.hypot(*(spot_ne - p).T).min()) if len(spot_ne) else float("inf")
        i, j = cmap._cell(*p)
        print(f"  {t['id']} at ({p[0]:.0f}, {p[1]:.0f}): nearest usable spot {near:.0f} m, "
              f"canopy over it {cmap.ground[i, j] - cmap.top[i, j]:.1f} m")
    clearings = cmap.clearings(a.min_radius, spacing_m=30.0, exclude=[tuple(meta["mission_spec"]["start_ned"][:2])])
    print(f"  {len(clearings)} clearings at least 30 m apart (launch excluded), widest first:")
    for c in clearings[:10]:
        print(f"    ({c.n:5.0f}, {c.e:5.0f})  radius {c.radius_m:4.1f} m")
    if a.png:
        img = np.zeros(spots.shape + (3,), np.uint8)
        img[seen] = (40, 90, 40)
        img[opened] = (90, 140, 170)
        img[reach] = (60, 200, 255)
        img[spots] = (0, 0, 255)
        img[~inside] //= 3
        for t in meta["targets"]:
            i, j = cmap._cell(*t["ned"][:2])
            cv2.circle(img, (j, i), 6, (255, 255, 255), 1)
        img = cv2.flip(img, 0)  # north up
        cv2.imwrite(str(a.png), cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
