"""Build a map's terrain from a real elevation model (DEM).

One command produces both halves of a map's terrain from the same numbers, so
they can't disagree:

* an Unreal Landscape heightmap (16-bit PNG) plus the settings for Landscape
  mode's "Import from File", and
* the VIVID 2 terrain model (``sim_config/maps/<map>/terrain_dtm.npz``) that the
  runner and the metrics use for ground height.

The terrain model is computed from the rounded 16-bit heightmap, i.e. the
heights Unreal actually renders.

Coordinates. The map is centred on the home geo point, which is both the Unreal
world origin and the NED origin. Unreal X points north, Y east, Z up. Unreal
Z = 0 sits ``margin_m`` below the lowest ground, so the whole play area is inside
the band that ground-height queries cover (Z = 0..1000 m). The real elevation of
Z = 0 becomes the scene's ``home-geo-point.altitude``.

Landscape conventions (UE): world Z [cm] = location.z + (h - 32768) * scale.z / 128,
heightmap column = landscape X (north), heightmap row = landscape Y (east).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from vivid2.terrain import TerrainModel

PathLike = Union[str, Path]

# GeoTIFF tags and geokeys
TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_GEOKEYS = 34735
TAG_GDAL_NODATA = 42113
GEOKEY_MODEL_TYPE = 1024    # 2 = geographic lat/lon
GEOKEY_RASTER_TYPE = 1025   # 1 = PixelIsArea, 2 = PixelIsPoint

# WGS84
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

# Landscape layout: 63x63-quad sections, 2x2 sections per component
SECTION_QUADS = 63
SECTIONS_PER_COMPONENT = 2
COMPONENT_QUADS = SECTION_QUADS * SECTIONS_PER_COMPONENT
PLAY_AREA_TOP_M = 1000.0  # GetZAtPoint traces down from here

COPERNICUS_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
COPERNICUS_ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space "
    "GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.")


@dataclass
class GeoDEM:
    """A DEM on a regular lat/lon grid. Pixel (r, c) is centred at (lat_r0 - r*dlat, lon_c0 + c*dlon)."""

    z: np.ndarray       # (rows, cols) metres, NaN = no data
    lat_r0: float
    lon_c0: float
    dlat: float
    dlon: float
    source: str = ""

    def pixel_of(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (self.lat_r0 - np.asarray(lat)) / self.dlat, (np.asarray(lon) - self.lon_c0) / self.dlon


def read_geotiff(path: PathLike, source: Optional[str] = None) -> GeoDEM:
    """Read a single-band GeoTIFF in geographic coordinates (e.g. EPSG:4326)."""
    from PIL import Image

    with Image.open(path) as im:
        tags = im.tag_v2
        z = np.array(im, dtype=np.float64)
        scale, tie, gk = tags.get(TAG_PIXEL_SCALE), tags.get(TAG_TIEPOINT), tags.get(TAG_GEOKEYS)
        nodata = tags.get(TAG_GDAL_NODATA)
    if scale is None or tie is None or gk is None:
        raise ValueError(f"{path}: not a GeoTIFF (missing georeferencing tags)")
    keys = {gk[i]: gk[i + 3] for i in range(4, 4 + 4 * gk[3], 4) if gk[i + 1] == 0}
    if keys.get(GEOKEY_MODEL_TYPE) != 2:
        raise ValueError(f"{path}: DEM must be in geographic lat/lon (EPSG:4326); "
                         "reproject it first, e.g. gdalwarp -t_srs EPSG:4326")
    if nodata is not None:
        try:
            z[z == float(str(nodata).strip("\x00 "))] = np.nan
        except ValueError:
            pass
    i, j, _, x, y, _ = tie[:6]
    offset = 0.5 if keys.get(GEOKEY_RASTER_TYPE, 1) == 1 else 0.0  # PixelIsArea: tiepoint is a corner
    return GeoDEM(z=z, lat_r0=y - (offset - j) * scale[1], lon_c0=x + (offset - i) * scale[0],
                  dlat=scale[1], dlon=scale[0], source=source or Path(path).name)


def copernicus_tile_name(lat: float, lon: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    return (f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")


def fetch_copernicus(lat: float, lon: float, cache_dir: PathLike) -> Path:
    """Download (once) the Copernicus GLO-30 tile containing (lat, lon)."""
    name = copernicus_tile_name(lat, lon)
    path = Path(cache_dir) / f"{name}.tif"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(COPERNICUS_URL.format(name=name), tmp)
        tmp.replace(path)
    return path


def enu_to_geo(n: np.ndarray, e: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Local north/east metres to lat/lon (tangent plane; accurate to cm over a few km)."""
    s = math.sin(math.radians(lat0))
    m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s * s) ** 1.5   # meridian radius
    nr = WGS84_A / math.sqrt(1 - WGS84_E2 * s * s)                 # prime-vertical radius
    return (lat0 + np.degrees(np.asarray(n) / m),
            lon0 + np.degrees(np.asarray(e) / (nr * math.cos(math.radians(lat0)))))


def _cubic_weights(t: np.ndarray):
    t2, t3 = t * t, t * t * t
    return ((-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2)


def bicubic(z: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Catmull-Rom interpolation of z at fractional pixel (r, c)."""
    r, c = np.asarray(r, float), np.asarray(c, float)
    if r.min() < 1 or c.min() < 1 or r.max() > z.shape[0] - 3 or c.max() > z.shape[1] - 3:
        raise ValueError("the requested area runs off the edge of the DEM")
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    wr, wc = _cubic_weights(r - r0), _cubic_weights(c - c0)
    out = np.zeros(r.shape)
    for a in range(4):
        for b in range(4):
            out += wr[a] * wc[b] * z[r0 - 1 + a, c0 - 1 + b]
    return out


@dataclass
class TerrainBuild:
    terrain: TerrainModel
    heightmap: np.ndarray            # uint16 (rows = east, cols = north), as Landscape import expects
    settings: Dict[str, Any] = field(default_factory=dict)


def build_terrain(dem: GeoDEM, lat0: float, lon0: float, vertices: int = 1009, cell_m: float = 1.0,
                  margin_m: float = 10.0) -> TerrainBuild:
    """Sample the DEM on a landscape grid centred on (lat0, lon0)."""
    if (vertices - 1) % COMPONENT_QUADS:
        raise ValueError(f"vertices must be k*{COMPONENT_QUADS}+1 (e.g. 505, 1009, 2017), got {vertices}")
    half = (vertices - 1) / 2 * cell_m
    axis = np.arange(vertices) * cell_m - half
    n, e = np.meshgrid(axis, axis, indexing="ij")
    lat, lon = enu_to_geo(n, e, lat0, lon0)
    elev = bicubic(dem.z, *dem.pixel_of(lat, lon))
    if not np.isfinite(elev).all():
        raise ValueError("the DEM has no data inside the requested area")

    home_alt = float(math.floor(elev.min() - margin_m))   # real elevation of Unreal Z = 0
    z = elev - home_alt
    zmin, zmax = float(z.min()), float(z.max())
    if zmax > PLAY_AREA_TOP_M - 10:
        raise ValueError(f"relief {zmax - zmin:.0f} m does not fit under Z = {PLAY_AREA_TOP_M:.0f} m")
    loc_z_cm = int(round((zmin + zmax) / 2 * 100))
    scale_z = max(1, math.ceil(((zmax - zmin) / 2 + 1.0) * 100 / 256))   # +-256 m * scale_z/100 covers it
    h = np.rint(32768 + (z * 100 - loc_z_cm) * 128 / scale_z).clip(0, 65535).astype(np.uint16)
    zq = (loc_z_cm + (h.astype(np.float64) - 32768) * scale_z / 128) / 100

    source = f"{dem.source} @ {lat0:.6f},{lon0:.6f} {vertices}x{vertices} x {cell_m:g} m"
    terrain = TerrainModel(down=(-zq).astype(np.float32).astype(float), n0=-half, e0=-half,
                           cell_m=cell_m, source=source)
    comps = (vertices - 1) // COMPONENT_QUADS
    settings = {
        "resolution": [vertices, vertices],
        "section_size_quads": SECTION_QUADS,
        "sections_per_component": SECTIONS_PER_COMPONENT,
        "components": [comps, comps],
        "scale": [cell_m * 100, cell_m * 100, scale_z],
        "landscape_centre_cm": [0.0, 0.0, loc_z_cm],
        "landscape_corner_cm": [-half * 100, -half * 100, loc_z_cm],
        "home_geo_point": {"latitude": lat0, "longitude": lon0, "altitude": home_alt},
        "ground_ue_z_m": {"min": round(float(zq.min()), 3), "max": round(float(zq.max()), 3),
                          "origin": round(float(terrain.down_at(0.0, 0.0)) * -1, 3)},
        "height_step_cm": scale_z / 128,
    }
    return TerrainBuild(terrain=terrain, heightmap=np.ascontiguousarray(h.T), settings=settings)


def write_heightmap_png(path: PathLike, heightmap: np.ndarray) -> None:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(heightmap, dtype=np.uint16)).save(path)


def read_heightmap_png(path: PathLike) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.array(im, dtype=np.uint16)


def write_outputs(build: TerrainBuild, map_name: str, sim_config_dir: PathLike, unreal_dir: PathLike,
                  dem_info: Dict[str, Any]) -> Dict[str, Path]:
    """Write the heightmap, the terrain model and a terrain.json describing both."""
    sim_dir = Path(sim_config_dir) / "maps" / map_name
    sim_dir.mkdir(parents=True, exist_ok=True)
    png = Path(unreal_dir) / map_name / "heightmap.png"
    write_heightmap_png(png, build.heightmap)
    npz = sim_dir / "terrain_dtm.npz"
    build.terrain.save(npz)
    info = {
        "map": map_name,
        "heightmap": os.path.relpath(png, sim_dir).replace(os.sep, "/"),
        "heightmap_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "terrain_dtm_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "landscape": build.settings,
        "dem": dem_info,
    }
    meta = sim_dir / "terrain.json"
    meta.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {"heightmap": png, "terrain": npz, "info": meta}
