"""Reference coverage planner: boustrophedon (lawnmower) sweep of a polygon."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from vivid2.geo import nadir_footprint
from vivid2.stack.base import MissionSpec, Planner

# A terrain-following survey (search_area.altitude_ref "terrain"): a waypoint every FOLLOW_STEP_M,
# each altitude_m above the highest coarse ground within FOLLOW_WINDOW_M of it, so the drone starts
# climbing before the ground rises and keeps its height above the trees on a slope.
FOLLOW_STEP_M = 10.0
FOLLOW_WINDOW_M = 25.0


def follows_terrain(mission: MissionSpec) -> bool:
    """Whether the mission's survey height is above the ground under the drone (not above the start)."""
    t = mission.terrain
    return getattr(mission, "altitude_ref", None) == "terrain" and t is not None and not t.is_flat


def follow_down_at(terrain, n: float, e: float, altitude_m: float, window_m: float = FOLLOW_WINDOW_M) -> float:
    """NED down of a terrain-following survey over (n, e): ``altitude_m`` above the highest ground
    within ``window_m`` (a 5 x 5 sample square)."""
    off = np.linspace(-window_m, window_m, 5)
    nn, ee = np.meshgrid(n + off, e + off, indexing="ij")
    return float(np.min(terrain.down_at(nn, ee))) - altitude_m


def follow_terrain(points_ne: Sequence[Sequence[float]], terrain, altitude_m: float, step_m: float = FOLLOW_STEP_M,
                   window_m: float = FOLLOW_WINDOW_M) -> List[List[float]]:
    """The path through ``points_ne`` (n, e) at ``altitude_m`` over the terrain.

    A waypoint every ``step_m`` or less (the given points included), each ``altitude_m`` above the
    highest ground within ``window_m`` along the path.
    """
    pts = np.asarray(points_ne, float)[:, :2]
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / step_m)))
        out.extend(a + (b - a) * (i / k) for i in range(1, k + 1))
    p = np.asarray(out)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    down = np.asarray(terrain.down_at(p[:, 0], p[:, 1]), float)
    z = np.array([down[np.abs(s - si) <= window_m].min() for si in s]) - altitude_m
    return [[float(n), float(e), float(d)] for (n, e), d in zip(p, z)]


def _polygon_span_at_east(poly: np.ndarray, e: float) -> Optional[Tuple[float, float]]:
    """North extent of the polygon along the line east == e."""
    ns = []
    for (n1, e1), (n2, e2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (e1 <= e < e2) or (e2 <= e < e1):
            ns.append(n1 + (e - e1) * (n2 - n1) / (e2 - e1))
    if len(ns) < 2:
        return None
    return min(ns), max(ns)


class LawnmowerPlanner(Planner):
    name = "lawnmower"

    def __init__(self, sidelap: float = 0.2, edge_inset_m: float = 2.0):
        self.sidelap = sidelap
        self.inset = edge_inset_m

    def plan(self, mission: MissionSpec, camera_fov_deg: float, image_wh: Tuple[int, int]) -> List[List[float]]:
        poly = np.asarray(mission.search_polygon_ne, float)
        down = mission.ground_down_m - mission.altitude_m
        # Lanes run north-south. For a nadir camera at yaw 0, image x (camera y
        # axis, body-right) maps to world east, so the image width sets spacing.
        across, _ =nadir_footprint(mission.altitude_m, image_wh[0], image_wh[1], camera_fov_deg)
        spacing = max(1.0, across * (1.0 - self.sidelap))
        lanes = self._lanes(poly[:, 1].min(), poly[:, 1].max(), spacing)

        waypoints: List[List[float]] = []
        flip = False
        for e in lanes:
            span = _polygon_span_at_east(poly, float(e))
            if span is None:
                continue
            n_lo, n_hi = span[0] + self.inset, span[1] - self.inset
            if n_hi <= n_lo:
                continue
            seg = [[n_lo, float(e), down], [n_hi, float(e), down]]
            waypoints.extend(seg[::-1] if flip else seg)
            flip = not flip
        if waypoints and follows_terrain(mission):
            # Same lanes, flown altitude_m over the ground; the leg from the start is part of the path.
            pts = [list(mission.start_ned[:2])] + [w[:2] for w in waypoints]
            return follow_terrain(pts, mission.terrain, float(mission.altitude_m))[1:]
        return waypoints

    def _lanes(self, e_min: float, e_max: float, spacing: float) -> np.ndarray:
        """East positions of the north-south lanes between e_min and e_max."""
        lanes = np.arange(e_min + spacing / 2.0, e_max, spacing)
        if len(lanes) == 0:
            lanes = np.array([(e_min + e_max) / 2.0])
        return lanes


class StripPlanner(LawnmowerPlanner):
    """A lawnmower over one of ``parts`` equal-width east-west strips of the search polygon.

    The reference team gives each drone its own strip. Lanes are planned within the strip, so the
    strips' lanes together cover the polygon like one drone's would, each strip from its west edge.
    """

    name = "lawnmower_strips"

    def __init__(self, part: int, parts: int, sidelap: float = 0.2, edge_inset_m: float = 2.0):
        super().__init__(sidelap, edge_inset_m)
        if not 0 <= part < parts:
            raise ValueError(f"strip {part} of {parts}")
        self.part = part
        self.parts = parts

    def _lanes(self, e_min: float, e_max: float, spacing: float) -> np.ndarray:
        width = (e_max - e_min) / self.parts
        lo = e_min + self.part * width
        return super()._lanes(lo, lo + width, spacing)
