"""Low flight for the reference agents: "dip into clearings" (``--dip on``).

Fog hides people from a drone surveying above the canopy, but not from one that
comes down. With dips on, a reference agent:

1. flies its usual high survey, while a ``CanopyMap`` records the highest
   surface seen in every 2 m cell from the nadir depth image (depth is not
   affected by fog), as a height above the coarse terrain model the stack is
   given;
2. after the survey, flies to the clearings it found, nearest first, while time
   and energy allow, and descends vertically into each to about ``dip_agl_m``
   above the ground. It descends in steps and re-checks the clearing with each
   closer depth image, stopping if anything comes near;
3. looks at the bottom with its own perception and reasoner, then climbs back.

A guard checks the map ahead of the drone during the survey and climbs when the
canopy or the ground ahead comes within ``guard_clearance_m`` of the flight
level. (The reference controller keeps the heading fixed, so the forward camera
only looks along the track on northbound legs; the nadir map covers every
direction.)

It deliberately never flies under the canopy or between the trunks: that is the
headroom the benchmark leaves for capable stacks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vivid2.geo import focal_px, quat_to_rot
from vivid2.stack.base import Frame
from vivid2.terrain import TerrainModel

DEPTH_MAX_M = 1000.0  # ignore returns beyond this (no data / sky)


@dataclass
class DipSettings:
    """Tunable settings of the dip behaviour (all recorded in the run's config hash)."""

    dip_agl_m: float = 12.0          # look height above the ground (never below min_agl_m + 5)
    min_radius_m: float = 6.0        # clearance from the drone to the nearest tree, to descend
    max_dips: int = 30               # in practice the time budget decides
    dip_frames: int = 3              # looks at the bottom; a majority must agree to report
    open_height_m: float = 3.0       # a cell is open if its surface is this close to the ground
    cell_m: float = 2.0
    time_share: float = 0.9          # dip only while the mission (with the flight home) fits this share
    guard_clearance_m: float = 10.0  # keep the canopy/ground ahead at least this far below
    guard_lookahead_m: float = 30.0
    guard_half_width_m: float = 5.0

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Clearing:
    n: float
    e: float
    radius_m: float  # distance from the centre to the nearest non-open cell


class CanopyMap:
    """Highest surface seen per grid cell over the search area, from nadir depth images."""

    def __init__(self, polygon_ne: Sequence[Sequence[float]], terrain: TerrainModel, cell_m: float = 2.0,
                 open_height_m: float = 3.0, margin_m: float = 30.0):
        poly = np.asarray(polygon_ne, float)
        self.cell = float(cell_m)
        self.open_height_m = float(open_height_m)
        self.n0 = float(poly[:, 0].min() - margin_m)
        self.e0 = float(poly[:, 1].min() - margin_m)
        nn = int(np.ceil((poly[:, 0].max() + margin_m - self.n0) / self.cell)) + 1
        ne = int(np.ceil((poly[:, 1].max() + margin_m - self.e0) / self.cell)) + 1
        self.top = np.full((nn, ne), np.inf)  # NED down of the highest surface seen (inf = unseen)
        cn = self.n0 + np.arange(nn) * self.cell
        ce = self.e0 + np.arange(ne) * self.cell
        self.ground = np.asarray(terrain.down_at(*np.meshgrid(cn, ce, indexing="ij")), float)
        inside = np.zeros((nn, ne), np.uint8)
        pts = np.round(np.stack([(poly[:, 1] - self.e0) / self.cell, (poly[:, 0] - self.n0) / self.cell], 1) * 16)
        cv2.fillPoly(inside, [pts.astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        self.inside = inside.astype(bool)
        self._dist: Optional[np.ndarray] = None

    # --- updating -------------------------------------------------------------------------

    def add(self, frame: Frame, stride_px: int = 4) -> int:
        """Fold a depth image into the map; returns the number of surface points used."""
        if frame.depth is None:
            return 0
        h, w = frame.depth.shape[:2]
        d = frame.depth[stride_px // 2::stride_px, stride_px // 2::stride_px].astype(float)
        vs = np.arange(stride_px // 2, h, stride_px)[:d.shape[0]] + 0.5
        us = np.arange(stride_px // 2, w, stride_px)[:d.shape[1]] + 0.5
        valid = (d > 0.1) & (d < DEPTH_MAX_M)
        if not valid.any():
            return 0
        f = focal_px(w, frame.fov_deg)
        uu, vv = np.meshgrid(us, vs)
        dv = d[valid]
        cam = np.stack([dv, dv * (uu[valid] - w / 2.0) / f, dv * (vv[valid] - h / 2.0) / f])
        p = quat_to_rot(frame.cam_q) @ cam + np.asarray(frame.cam_pos, float)[:, None]
        i = np.round((p[0] - self.n0) / self.cell).astype(int)
        j = np.round((p[1] - self.e0) / self.cell).astype(int)
        ok = (i >= 0) & (i < self.top.shape[0]) & (j >= 0) & (j < self.top.shape[1])
        np.minimum.at(self.top, (i[ok], j[ok]), p[2][ok])
        self._dist = None
        return int(ok.sum())

    # --- queries ----------------------------------------------------------------------------

    def _cell(self, n: float, e: float) -> Tuple[int, int]:
        return int(round((n - self.n0) / self.cell)), int(round((e - self.e0) / self.cell))

    def height_above_ground(self) -> np.ndarray:
        """Canopy height model, m (-inf where unseen)."""
        return self.ground - self.top

    def open_mask(self) -> np.ndarray:
        return np.isfinite(self.top) & (self.ground - self.top < self.open_height_m)

    def _distance(self) -> np.ndarray:
        """Distance (m) from each open cell centre to the nearest non-open cell's edge."""
        if self._dist is None:
            raw = cv2.distanceTransform(self.open_mask().astype(np.uint8), cv2.DIST_L2, 5) * self.cell
            self._dist = np.where(raw > 0, raw - self.cell / 2.0, 0.0)
        return self._dist

    def radius_at(self, n: float, e: float) -> float:
        i, j = self._cell(n, e)
        if not (0 <= i < self.top.shape[0] and 0 <= j < self.top.shape[1]):
            return 0.0
        return float(self._distance()[i, j])

    def highest_in_corridor(self, p0: Sequence[float], p1: Sequence[float], half_width_m: float) -> Optional[float]:
        """NED down of the highest surface seen within ``half_width_m`` of the segment p0-p1 (None: nothing seen)."""
        vals = self._corridor(p0, p1, half_width_m)
        return float(vals.min()) if vals.size else None

    def highest_within(self, n: float, e: float, radius_m: float) -> Optional[float]:
        return self.highest_in_corridor((n, e), (n, e), radius_m)

    def lowest_within(self, n: float, e: float, radius_m: float) -> Optional[float]:
        """NED down of the lowest surface seen within ``radius_m`` (the floor of a clearing)."""
        vals = self._corridor((n, e), (n, e), radius_m)
        return float(vals.max()) if vals.size else None

    def _corridor(self, p0: Sequence[float], p1: Sequence[float], half_width_m: float) -> np.ndarray:
        """Surfaces seen in the cells within ``half_width_m`` of the segment p0-p1."""
        a, b = np.asarray(p0[:2], float), np.asarray(p1[:2], float)
        lo, hi = np.minimum(a, b) - half_width_m, np.maximum(a, b) + half_width_m
        i0, j0 = self._cell(lo[0], lo[1])
        i1, j1 = self._cell(hi[0], hi[1])
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, self.top.shape[0] - 1), min(j1, self.top.shape[1] - 1)
        if i1 < i0 or j1 < j0:
            return np.empty(0)
        cn = self.n0 + np.arange(i0, i1 + 1) * self.cell
        ce = self.e0 + np.arange(j0, j1 + 1) * self.cell
        pn, pe = np.meshgrid(cn, ce, indexing="ij")
        ab = b - a
        L2 = float(ab @ ab)
        t = np.zeros_like(pn) if L2 == 0 else np.clip(((pn - a[0]) * ab[0] + (pe - a[1]) * ab[1]) / L2, 0, 1)
        dist = np.hypot(pn - (a[0] + t * ab[0]), pe - (a[1] + t * ab[1]))
        vals = self.top[i0:i1 + 1, j0:j1 + 1][dist <= half_width_m]
        return vals[np.isfinite(vals)]

    def clearings(self, min_radius_m: float, spacing_m: float,
                  exclude: Sequence[Tuple[float, float]] = ()) -> List[Clearing]:
        """Open spots inside the search area where the drone can descend vertically, widest first,
        at least ``spacing_m`` apart and from the ``exclude`` points."""
        dist = self._distance()
        cand = np.argwhere((dist >= min_radius_m) & self.inside)
        if not len(cand):
            return []
        order = np.argsort(-dist[cand[:, 0], cand[:, 1]], kind="stable")
        picked: List[Clearing] = []
        taken = [np.asarray(p, float) for p in exclude]
        for k in order:
            i, j = cand[k]
            p = np.array([self.n0 + i * self.cell, self.e0 + j * self.cell])
            if all(np.hypot(*(p - q)) >= spacing_m for q in taken):
                picked.append(Clearing(float(p[0]), float(p[1]), float(dist[i, j])))
                taken.append(p)
        return picked
