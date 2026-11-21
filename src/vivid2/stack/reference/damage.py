"""Reference damage mappers (the damage layer of the reference search agents; docs/damage.md).

    GroundTruthDamageMapper  the oracle: reports the damage ground truth its own photos show
                             (``Frame.damage_gt``). An upper bound for the run's flight; never ranked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from vivid2.damage.geometry import mask_polygons
from vivid2.stack.base import (AreaReport, BuildingReport, ClueReport, DamageMapper, Frame, MissionSpec, RoadReport,
                               UsageStats)


class GroundTruthDamageMapper(DamageMapper):
    """Reports what each photo's damage ground truth shows: roads, buildings and clues at once,
    area outlines at the end (``finish``), each citing a photo that showed it."""

    name = "gt_damage"
    uses_ground_truth = True

    def __init__(self):
        self._cells: Dict[str, Set[Tuple[float, float]]] = {}
        self._cell_m: Optional[float] = None
        self._last: Dict[str, Tuple[int, int]] = {}  # area class -> (frame_id, t_ns) of a photo that showed it
        self._roads: Dict[str, bool] = {}
        self._buildings: Dict[str, str] = {}
        self._clues: Set[str] = set()

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "areas": "observed_ground_truth_cells", "roads": "observed", "buildings": "observed"}

    def wants(self, frame: Frame, mission: MissionSpec) -> bool:
        return frame.damage_gt is not None

    def observe(self, frame: Frame, mission: MissionSpec):
        g = frame.damage_gt or {}
        self._cell_m = g.get("cell_m", self._cell_m)
        out: List[Any] = []
        for k, cells in (g.get("cells") or {}).items():
            if len(cells):
                self._cells.setdefault(k, set()).update((round(float(n), 3), round(float(e), 3)) for n, e in cells)
                self._last[k] = (frame.frame_id, frame.t_ns)
        for seg, blocked in (g.get("roads") or {}).items():
            if self._roads.get(seg) != blocked:
                self._roads[seg] = blocked
                out.append(RoadReport(seg, blocked, frame.frame_id, frame.t_ns, note="ground truth"))
        for bid, grade in (g.get("buildings") or {}).items():
            if self._buildings.get(bid) != grade:
                self._buildings[bid] = grade
                out.append(BuildingReport(bid, grade, frame.frame_id, frame.t_ns, note="ground truth"))
        for c in g.get("clues") or []:
            if c["id"] not in self._clues:
                self._clues.add(c["id"])
                out.append(ClueReport(list(c["ned"]), c["kind"], frame.frame_id, frame.t_ns, note="ground truth"))
        return out, UsageStats(), {}

    def finish(self, mission: MissionSpec) -> List[Any]:
        out: List[Any] = []
        cell = self._cell_m
        for k, cells in self._cells.items():
            if not cells or cell is None:
                continue
            pts = np.array(sorted(cells), float)
            n0, e0 = pts[:, 0].min(), pts[:, 1].min()
            ij = np.rint((pts - [n0, e0]) / cell).astype(int)
            mask = np.zeros(ij.max(axis=0) + 1, bool)
            mask[ij[:, 0], ij[:, 1]] = True
            fid, t = self._last[k]
            for p in mask_polygons(mask, n0, e0, cell):
                out.append(AreaReport(k, p["polygon_ne"], fid, t, op=p["op"], note="ground truth"))
        return out
