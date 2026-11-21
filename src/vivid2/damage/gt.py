"""A map's damage ground truth (``sim_config/maps/<map>/damage_gt.json`` + ``damage_gt.npz``).

Written by ``vivid2 damage build`` from the map's layout (``vivid2.damage.layout``) and never
overwritten once runs exist (rescoring checks the checksums, like the terrain model).

``damage_gt.json``::

    version, map, layout {file, sha256}, terrain {post_sha256, pre_sha256?}
    raster {file, sha256}          the npz below
    area_classes {name: {bit, about, blocks_roads}}
    roads [{id, road, polyline_ne, width_m, kind, blocked, cause[], blocked_at_ne[]}]   (segments)
    buildings [{id, footprint_ne, height_m, base_down_m, roof, use, ems98, grade, cause,
                debris_ne?, spill_m}]
    blockers [{polygon_ne, cause}]
    water {level_down_m, polygon_ne} | null
    surface {file, sha256} | null  ground, roofs and water as one height model (snap "surface")

``damage_gt.npz`` (on the terrain model's grid, ``[i = north, j = east]``)::

    classes  uint8  bit field, one bit per area class
    debris_m float16  ground raised (+) or lowered (-) by the event, m
    water_m  float16  water depth, m (0 where dry)
    n0, e0, cell_m
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from vivid2.damage import GRADE_LEVEL

PathLike = Union[str, Path]
GT_FILE = "damage_gt.json"
RASTER_FILE = "damage_gt.npz"
SURFACE_FILE = "surface_dsm.npz"
PRE_EVENT_FILE = "pre_event_dtm.npz"


def sha256_file(path: PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class DamageGT:
    map: str
    area_classes: Dict[str, Dict[str, Any]]
    roads: List[Dict[str, Any]]
    buildings: List[Dict[str, Any]]
    classes: np.ndarray              # (nN, nE) uint8 bits
    n0: float
    e0: float
    cell_m: float
    debris_m: Optional[np.ndarray] = None
    water_m: Optional[np.ndarray] = None
    blockers: List[Dict[str, Any]] = field(default_factory=list)
    water: Optional[Dict[str, Any]] = None
    doc: Dict[str, Any] = field(default_factory=dict)  # the json as loaded

    # -- lookups ------------------------------------------------------------------------------
    def _node(self, n, e):
        nn, ne = self.classes.shape
        i = np.clip(np.rint((np.asarray(n, float) - self.n0) / self.cell_m), 0, nn - 1).astype(int)
        j = np.clip(np.rint((np.asarray(e, float) - self.e0) / self.cell_m), 0, ne - 1).astype(int)
        return i, j

    def bits_at(self, n, e) -> np.ndarray:
        """Area-class bits at NED points (nearest grid node)."""
        i, j = self._node(n, e)
        return self.classes[i, j]

    def class_at(self, name: str, n, e) -> np.ndarray:
        bit = int(self.area_classes[name]["bit"])
        return (self.bits_at(n, e) >> bit & 1).astype(bool)

    def water_at(self, n, e) -> np.ndarray:
        if self.water_m is None:
            return np.zeros(np.broadcast(np.asarray(n), np.asarray(e)).shape)
        i, j = self._node(n, e)
        return self.water_m[i, j].astype(float)

    def road(self, segment_id: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.roads if r["id"] == segment_id), None)

    def building(self, building_id: str) -> Optional[Dict[str, Any]]:
        return next((b for b in self.buildings if b["id"] == building_id), None)

    def grade_level(self, building_id: str) -> int:
        b = self.building(building_id)
        return GRADE_LEVEL[b["grade"]] if b else 0

    # -- what the stack may know (pre-event) --------------------------------------------------
    def public_roads(self) -> List[Dict[str, Any]]:
        return [{k: r[k] for k in ("id", "road", "polyline_ne", "width_m", "kind") if k in r} for r in self.roads]

    def public_buildings(self) -> List[Dict[str, Any]]:
        return [{k: b[k] for k in ("id", "footprint_ne", "height_m", "roof", "use") if k in b} for b in self.buildings]


def gt_paths(sim_config_dir: PathLike, map_name: str) -> Dict[str, Path]:
    d = Path(sim_config_dir) / "maps" / map_name
    return {"dir": d, "json": d / GT_FILE, "raster": d / RASTER_FILE, "surface": d / SURFACE_FILE,
            "pre_event": d / PRE_EVENT_FILE}


def load_gt(path: PathLike, expect_sha256: Optional[str] = None) -> DamageGT:
    """Load ``damage_gt.json`` and its raster, checking the raster's checksum (and the json's, if given)."""
    path = Path(path)
    if expect_sha256 and sha256_file(path) != expect_sha256:
        raise ValueError(f"damage ground truth {path} changed since this run was recorded")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rpath = path.parent / doc["raster"]["file"]
    if sha256_file(rpath) != doc["raster"]["sha256"]:
        raise ValueError(f"{rpath} does not match {path.name} (rebuild with vivid2 damage build)")
    z = np.load(rpath, allow_pickle=False)
    return DamageGT(
        map=doc.get("map", ""), area_classes=doc.get("area_classes", {}), roads=doc.get("roads", []),
        buildings=doc.get("buildings", []), classes=z["classes"], n0=float(z["n0"]), e0=float(z["e0"]),
        cell_m=float(z["cell_m"]), debris_m=z["debris_m"] if "debris_m" in z.files else None,
        water_m=z["water_m"] if "water_m" in z.files else None, blockers=doc.get("blockers", []),
        water=doc.get("water"), doc=doc)


def gt_record(path: PathLike) -> Dict[str, Any]:
    """meta.json's pointer to a ground truth file (rescoring finds and checks it again)."""
    path = Path(path)
    return {"file": str(path.resolve()), "sha256": sha256_file(path)}


def surface_model(sim_config_dir: PathLike, map_name: str):
    """The map's surface model (ground, roofs, water) and its meta record, or (None, None)."""
    from vivid2.terrain import TerrainModel

    p = gt_paths(sim_config_dir, map_name)["surface"]
    if not p.exists():
        return None, None
    return TerrainModel.load(p), {"file": str(p.resolve()), "sha256": sha256_file(p)}


def pre_event_model(sim_config_dir: PathLike, map_name: Optional[str]):
    """The map's pre-event terrain (what a crew's DEM shows), and its meta record, or (None, None)."""
    from vivid2.terrain import TerrainModel

    if not map_name:
        return None, None
    p = gt_paths(sim_config_dir, map_name)["pre_event"]
    if not p.exists():
        return None, None
    return TerrainModel.load(p), {"file": str(p.resolve()), "sha256": sha256_file(p)}


def write_raster(path: PathLike, classes: np.ndarray, n0: float, e0: float, cell_m: float,
                 debris_m: Optional[np.ndarray] = None, water_m: Optional[np.ndarray] = None) -> None:
    arrays = {"classes": classes.astype(np.uint8), "n0": float(n0), "e0": float(e0), "cell_m": float(cell_m)}
    if debris_m is not None:
        arrays["debris_m"] = debris_m.astype(np.float16)
    if water_m is not None:
        arrays["water_m"] = water_m.astype(np.float16)
    np.savez_compressed(path, **arrays)


def ids_of(items: Sequence[Dict[str, Any]]) -> List[str]:
    return [i["id"] for i in items]
