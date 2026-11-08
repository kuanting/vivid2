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
