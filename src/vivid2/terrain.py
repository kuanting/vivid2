"""Terrain height for scoring and spawning.

Flat maps (Blocks) use the scenario's ``search_area.ground_down_m``. Maps with
relief ship a bare-ground terrain model (DTM) built from the heightmap the
landscape was made from, as ``sim_config/maps/<map>/terrain_dtm.npz``:

    down    (nN, nE) float32  NED down of the ground at each grid node, m (NaN = no data)
    n0, e0  float             NED north/east of node [0, 0], m
    cell_m  float             grid spacing, m
    source  str               where the heights came from

The runner snaps targets and the drone start to this ground, and safety
metrics measure height above it. Each run's meta.json records the file and its
SHA-256, so ``vivid2 rescore`` uses the same terrain without the simulator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass
class TerrainModel:
    down: np.ndarray  # 0-d for flat ground, else (nN, nE)
    n0: float = 0.0
    e0: float = 0.0
    cell_m: float = 1.0
    source: str = "flat"

    @classmethod
    def flat(cls, ground_down_m: float = 0.0) -> "TerrainModel":
        return cls(np.asarray(float(ground_down_m)), source="flat")

    @property
    def is_flat(self) -> bool:
        return self.down.ndim == 0

    def down_at(self, n: ArrayLike, e: ArrayLike) -> np.ndarray:
        """Ground NED down at (n, e), bilinear between nodes, clamped at the grid edge."""
        n, e = np.asarray(n, float), np.asarray(e, float)
        if self.is_flat:
            return np.broadcast_to(self.down, np.broadcast(n, e).shape).astype(float)
        nn, ne = self.down.shape
        fi = np.clip((n - self.n0) / self.cell_m, 0, nn - 1)
        fj = np.clip((e - self.e0) / self.cell_m, 0, ne - 1)
        i0, j0 = np.minimum(fi.astype(int), nn - 2).clip(0), np.minimum(fj.astype(int), ne - 2).clip(0)
        di, dj = fi - i0, fj - j0
        d = self.down
        i1, j1 = np.minimum(i0 + 1, nn - 1), np.minimum(j0 + 1, ne - 1)
        return ((1 - di) * (1 - dj) * d[i0, j0] + di * (1 - dj) * d[i1, j0]
                + (1 - di) * dj * d[i0, j1] + di * dj * d[i1, j1])

    def agl(self, pos: np.ndarray) -> np.ndarray:
        """Height above ground for NED positions (..., 3)."""
        pos = np.asarray(pos, float)
        return self.down_at(pos[..., 0], pos[..., 1]) - pos[..., 2]

    def save(self, path: Union[str, Path]) -> None:
        np.savez_compressed(path, down=self.down.astype(np.float32), n0=self.n0, e0=self.e0,
                            cell_m=self.cell_m, source=self.source)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TerrainModel":
        z = np.load(path, allow_pickle=False)
        return cls(z["down"].astype(float), float(z["n0"]), float(z["e0"]), float(z["cell_m"]), str(z["source"]))


def terrain_file(sim_config_dir: Union[str, Path], map_name: Optional[str]) -> Optional[Path]:
    """The map's terrain model. A variant map (e.g. ForestSAR_Dense, same landscape, other
    vegetation) can instead hold ``terrain.json`` with ``{"terrain_of": "ForestSAR"}``."""
    if not map_name:
        return None
    d = Path(sim_config_dir) / "maps" / map_name
    p = d / "terrain_dtm.npz"
    if p.exists():
        return p
    info = d / "terrain.json"
    if info.exists():
        import json

        base = json.loads(info.read_text(encoding="utf-8")).get("terrain_of")
        if base and base != map_name:
            return terrain_file(sim_config_dir, base)
    return None


def load_for_run(sim_config_dir: Union[str, Path], spec: Dict[str, Any]) -> "tuple[TerrainModel, Dict[str, Any]]":
    """The run's terrain plus the meta.json record that lets rescoring find it again."""
    path = terrain_file(sim_config_dir, spec.get("scene", {}).get("map"))
    if path is None:
        ground = spec["search_area"].get("ground_down_m", 0.0)
        return TerrainModel.flat(ground), {"kind": "flat", "ground_down_m": ground}
    model = TerrainModel.load(path)
    return model, {"kind": "dtm", "file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "source": model.source, "cell_m": model.cell_m}


def from_meta(meta: Dict[str, Any]) -> TerrainModel:
    """Terrain for a recorded run (falls back to the flat search-area ground)."""
    rec = meta.get("terrain") or {}
    if rec.get("kind") == "dtm":
        path = Path(rec["file"])
        if not path.exists():
            raise FileNotFoundError(f"terrain file {path} for this run is missing")
        if rec.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
            raise ValueError(f"terrain file {path} changed since this run was recorded")
        return TerrainModel.load(path)
    return TerrainModel.flat(meta.get("search_area", {}).get("ground_down_m", 0.0))


def snap_to_ground(ned, terrain: TerrainModel, authored_ground_down_m: float) -> list:
    """Keep an object's authored height above ground, but on the real terrain under it.

    Scenarios author z relative to ``search_area.ground_down_m`` (e.g. -0.85 for a
    centre-pivot 1.7 m box on flat ground); on relief that offset is applied to
    the terrain height at the object's (n, e).
    """
    n, e, z = (float(v) for v in ned)
    return [n, e, float(terrain.down_at(n, e)) + (z - authored_ground_down_m)]
