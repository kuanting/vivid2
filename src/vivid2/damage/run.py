"""One run's damage task: the products asked for, over the area of interest, against the map's ground truth."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vivid2.damage import (CLUE_KINDS, CLUE_TOLERANCE_M, DAMAGE_CELL_M, DEFAULT_COMPONENT_WEIGHTS, GRADES,
                           IN_PHOTO_BUILDING_SHARE)
from vivid2.damage.geometry import corridor_cells, grid_cells, point_at, polyline_length, view_mask
from vivid2.damage.gt import DamageGT, gt_paths, gt_record, load_gt
from vivid2.geo import points_in_polygon
from vivid2.terrain import TerrainModel


class DamageTask:
    """The damage task of one run (``spec`` is the scenario's ``damage`` block)."""

    def __init__(self, spec: Dict[str, Any], gt: DamageGT, terrain: TerrainModel,
                 search_polygon_ne: Sequence[Sequence[float]], clues: Optional[List[Dict[str, Any]]] = None,
                 buried: Optional[List[Dict[str, Any]]] = None, gt_meta: Optional[Dict[str, Any]] = None):
        self.spec = spec
        self.gt = gt
        self.gt_meta = gt_meta or {}
        self.terrain = terrain
        self.products: List[str] = list(spec["products"])
        self.aoi: List[List[float]] = [list(map(float, p)) for p in (spec.get("aoi_ne") or search_polygon_ne)]
        self.cell_m = DAMAGE_CELL_M
        self.cells = grid_cells(self.aoi, self.cell_m)
        self.cells_down = np.asarray(terrain.down_at(self.cells[:, 0], self.cells[:, 1]), float) if len(self.cells) \
            else np.zeros(0)
        self.area_classes: List[str] = (list(spec.get("area_classes") or gt.area_classes)
                                        if "areas" in self.products else [])
        self.gt_cells = {k: gt.class_at(k, self.cells[:, 0], self.cells[:, 1]) for k in self.area_classes}
        aoi = np.asarray(self.aoi, float)

        def in_aoi(p) -> bool:
            return bool(points_in_polygon(np.asarray([p], float), aoi)[0])

        self.roads = ([r for r in gt.roads if in_aoi(point_at(r["polyline_ne"], polyline_length(r["polyline_ne"]) / 2)[0])]
                      if "roads" in self.products else [])
        self.buildings = ([b for b in gt.buildings if in_aoi(np.mean(b["footprint_ne"], axis=0))]
                          if "buildings" in self.products else [])
        self.road_cells = {r["id"]: corridor_cells(r["polyline_ne"], r["width_m"], self.cells, self.cell_m)
                           for r in self.roads}
        self.building_cells = {b["id"]: self._footprint_cells(b["footprint_ne"]) for b in self.buildings}
        self.clues = list(clues or []) if "clues" in self.products else []
        self.buried = list(buried or [])

    def _footprint_cells(self, fp) -> np.ndarray:
        if not len(self.cells):
            return np.zeros(0, int)
        idx = np.flatnonzero(points_in_polygon(self.cells, fp))
        if not len(idx):  # smaller than a cell: the cell under its centre
            c = np.mean(fp, axis=0)
            idx = np.array([int(np.argmin(np.linalg.norm(self.cells - c, axis=1)))])
        return idx

    # -- construction from a run or a record ----------------------------------------------------
    @classmethod
    def for_run(cls, spec: Dict[str, Any], sim_config_dir, terrain: TerrainModel, clues=None, buried=None):
        """The damage task of a scenario spec (None without a ``damage`` block)."""
        d = spec.get("damage")
        if not d:
            return None
        map_name = d.get("map") or (spec.get("scene") or {}).get("map")
        if not map_name:
            raise ValueError("a damage task needs scene.map (or damage.map): its ground truth is per map")
        path = gt_paths(sim_config_dir, map_name)["json"]
        if not path.exists():
            raise FileNotFoundError(f"no damage ground truth for map {map_name!r} ({path}): run vivid2 damage build")
        return cls(d, load_gt(path), terrain, spec["search_area"]["polygon_ne"], clues, buried,
                   {**gt_record(path), "map": map_name})

    @classmethod
    def from_meta(cls, meta: Dict[str, Any], terrain: TerrainModel) -> "DamageTask":
        rec = meta["damage"]
        gt = load_gt(rec["gt"]["file"], rec["gt"].get("sha256"))
        spec = {k: rec[k] for k in ("products", "aoi_ne", "area_classes", "weights", "clue_tolerance_m") if k in rec}
        return cls(spec, gt, terrain, meta["search_area"]["polygon_ne"], meta.get("clues"), rec.get("buried"),
                   rec["gt"])

    # -- records --------------------------------------------------------------------------------
    def weights(self) -> Dict[str, float]:
        return {**DEFAULT_COMPONENT_WEIGHTS, **(self.spec.get("weights") or {})}

    @property
    def clue_tolerance_m(self) -> float:
        return float(self.spec.get("clue_tolerance_m", CLUE_TOLERANCE_M))

    def meta(self) -> Dict[str, Any]:
        """meta.json's ``damage`` record: the task as asked, and where its ground truth is."""
        return {"products": self.products, "aoi_ne": self.aoi, "area_classes": self.area_classes,
                **({"weights": self.spec["weights"]} if self.spec.get("weights") else {}),
                "clue_tolerance_m": self.clue_tolerance_m, "buried": self.buried,
                "gt": {k: v for k, v in self.gt_meta.items() if k in ("file", "sha256", "map")}}

    def mission_view(self, pre_event_terrain: bool = False) -> Dict[str, Any]:
        """What the stack is told (MissionSpec.damage): the task and the pre-event map, never the damage."""
        view: Dict[str, Any] = {"products": self.products, "aoi_ne": self.aoi}
        if "areas" in self.products:
            view["area_classes"] = {k: self.gt.area_classes[k].get("about", "") for k in self.area_classes}
        if "roads" in self.products:
            ids = {r["id"] for r in self.roads}
            view["roads"] = [r for r in self.gt.public_roads() if r["id"] in ids]
        if "buildings" in self.products:
            view["building_grades"] = list(GRADES)
            ids = {b["id"] for b in self.buildings}
            view["buildings"] = [b for b in self.gt.public_buildings() if b["id"] in ids]
        if "clues" in self.products:
            view["clue_kinds"] = list(CLUE_KINDS)
        view["terrain"] = "pre_event" if pre_event_terrain else "current"
        return view

    # -- what a photo saw -------------------------------------------------------------------------
    def seen(self, frame) -> np.ndarray:
        return view_mask(frame, self.cells, self.cells_down, self.terrain)

    def road_in_view(self, seen: np.ndarray, segment_id: str) -> bool:
        idx = self.road_cells.get(segment_id)
        return idx is not None and bool(seen[idx].any())

    def building_in_view(self, seen: np.ndarray, building_id: str) -> bool:
        idx = self.building_cells.get(building_id)
        return idx is not None and len(idx) > 0 and float(seen[idx].mean()) >= IN_PHOTO_BUILDING_SHARE

    def observe(self, frame, clues_visible: Optional[Dict[str, Any]] = None, min_px: int = 0) -> Dict[str, Any]:
        """The ground truth a photo shows (the reference oracle's view; never given to a stack under test).

        ``clues_visible``: visible pixels per clue id in this photo (segmentation).
        """
        seen = self.seen(frame)
        out: Dict[str, Any] = {"cell_m": self.cell_m, "cells": {}, "roads": {}, "buildings": {}, "clues": []}
        for k, g in self.gt_cells.items():
            out["cells"][k] = self.cells[seen & g]
        for r in self.roads:
            if self.road_in_view(seen, r["id"]):
                out["roads"][r["id"]] = bool(r["blocked"])
        for b in self.buildings:
            if self.building_in_view(seen, b["id"]):
                out["buildings"][b["id"]] = b["grade"]
        for c in self.clues:
            px = (clues_visible or {}).get(c["id"])
            px = px.get("visible_px", 0) if isinstance(px, dict) else (px or 0)
            if px and px >= min_px:
                out["clues"].append({"id": c["id"], "ned": list(c["ned"]), "kind": c.get("kind", "other")})
        return out
