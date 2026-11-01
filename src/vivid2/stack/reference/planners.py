"""Reference coverage planner: boustrophedon (lawnmower) sweep of a polygon."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from vivid2.geo import nadir_footprint
from vivid2.stack.base import MissionSpec, Planner


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
        e_min, e_max = poly[:, 1].min(), poly[:, 1].max()
        lanes = np.arange(e_min + spacing / 2.0, e_max, spacing)
        if len(lanes) == 0:
            lanes = np.array([(e_min + e_max) / 2.0])

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
        return waypoints
