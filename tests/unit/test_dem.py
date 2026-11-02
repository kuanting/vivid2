"""DEM -> Landscape heightmap + terrain model (vivid2.dem)."""
import json
import math

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin, TiffTags

from vivid2 import dem as dm
from vivid2.terrain import TerrainModel

LAT0, LON0 = 24.2436, 121.2467
SLOPE_N, SLOPE_E = 0.05, -0.02   # metres per metre


def _radii(lat0):
    s = math.sin(math.radians(lat0))
    m = dm.WGS84_A * (1 - dm.WGS84_E2) / (1 - dm.WGS84_E2 * s * s) ** 1.5
    n = dm.WGS84_A / math.sqrt(1 - dm.WGS84_E2 * s * s)
    return m, n * math.cos(math.radians(lat0))


def _write_geotiff(path, raster_type=2, size=40, step=1 / 3600):
    """A plane: elevation = 2000 + SLOPE_N * north_m + SLOPE_E * east_m around (LAT0, LON0)."""
    m, ncos = _radii(LAT0)
    lat_r0 = LAT0 + size / 2 * step
    lon_c0 = LON0 - size / 2 * step
    rows, cols = np.mgrid[0:size, 0:size]
    lat, lon = lat_r0 - rows * step, lon_c0 + cols * step
    z = 2000 + SLOPE_N * np.radians(lat - LAT0) * m + SLOPE_E * np.radians(lon - LON0) * ncos
    off = 0.5 if raster_type == 1 else 0.0   # PixelIsArea tiepoints are pixel corners
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[dm.TAG_PIXEL_SCALE] = (step, step, 0.0)
    ifd.tagtype[dm.TAG_PIXEL_SCALE] = TiffTags.DOUBLE
    ifd[dm.TAG_TIEPOINT] = (0.0, 0.0, 0.0, lon_c0 - off * step, lat_r0 + off * step, 0.0)
    ifd.tagtype[dm.TAG_TIEPOINT] = TiffTags.DOUBLE
    ifd[dm.TAG_GEOKEYS] = (1, 1, 0, 2, dm.GEOKEY_MODEL_TYPE, 0, 1, 2, dm.GEOKEY_RASTER_TYPE, 0, 1, raster_type)
    ifd.tagtype[dm.TAG_GEOKEYS] = TiffTags.SHORT
    Image.fromarray(z.astype(np.float32)).save(path, tiffinfo=ifd)
    return z


@pytest.mark.parametrize("raster_type", [1, 2])
def test_read_geotiff_pixel_centres(tmp_path, raster_type):
    p = tmp_path / "dem.tif"
    z = _write_geotiff(p, raster_type)
    d = dm.read_geotiff(p)
    assert np.allclose(d.z, z.astype(np.float32))
    r, c = d.pixel_of(LAT0, LON0)
    assert r == pytest.approx(20.0, abs=1e-6) and c == pytest.approx(20.0, abs=1e-6)


def test_build_matches_dem_and_landscape_formula(tmp_path):
    p = tmp_path / "dem.tif"
    _write_geotiff(p)
    b = dm.build_terrain(dm.read_geotiff(p), LAT0, LON0, vertices=127, cell_m=1.0, margin_m=10.0)
    s = b.settings
    home = s["home_geo_point"]["altitude"]
    lowest = 2000 - 63 * (abs(SLOPE_N) + abs(SLOPE_E))
    assert home == math.floor(lowest - 10)
    step_m = s["height_step_cm"] / 100

    # terrain model = real elevation minus the elevation of Unreal Z = 0 (NED down is negative up)
    for n, e in [(0, 0), (40, -25), (-63, 63), (12.5, 7.25)]:
        up = 2000 + SLOPE_N * n + SLOPE_E * e - home
        assert -b.terrain.down_at(n, e) == pytest.approx(up, abs=step_m + 2e-3)
    assert s["ground_ue_z_m"]["min"] >= 10 - step_m

    # heightmap: column = north index, row = east index, decoded with the Landscape formula
    loc_z, scale_z = s["landscape_centre_cm"][2], s["scale"][2]
    hm = b.heightmap.astype(float)
    assert hm.shape == (127, 127)
    for i, j in [(0, 0), (126, 0), (0, 126), (90, 30)]:      # i = north index, j = east index
        z_cm = loc_z + (hm[j, i] - 32768) * scale_z / 128
        assert z_cm / 100 == pytest.approx(-b.terrain.down[i, j], abs=1e-4)
    assert hm[0, 126] > hm[0, 0]   # north is uphill (SLOPE_N > 0) along heightmap columns


def test_build_rejects_bad_layout_and_offgrid(tmp_path):
    p = tmp_path / "dem.tif"
    _write_geotiff(p)
    d = dm.read_geotiff(p)
    with pytest.raises(ValueError, match="k\\*126\\+1"):
        dm.build_terrain(d, LAT0, LON0, vertices=1000)
    with pytest.raises(ValueError, match="edge of the DEM"):
        dm.build_terrain(d, LAT0, LON0, vertices=1009)   # 1 km does not fit in a 40-pixel DEM


def test_write_outputs_round_trip(tmp_path):
    p = tmp_path / "dem.tif"
    _write_geotiff(p)
    b = dm.build_terrain(dm.read_geotiff(p), LAT0, LON0, vertices=127)
    out = dm.write_outputs(b, "TestMap", tmp_path / "sim_config", tmp_path / "unreal", {"name": "plane"})
    assert np.array_equal(dm.read_heightmap_png(out["heightmap"]), b.heightmap)
    t = TerrainModel.load(out["terrain"])
    assert np.allclose(t.down, b.terrain.down) and t.n0 == -63 and t.cell_m == 1.0
    info = json.loads(out["info"].read_text())
    assert (out["info"].parent / info["heightmap"]).resolve() == out["heightmap"].resolve()
    assert info["landscape"]["home_geo_point"]["latitude"] == LAT0


def test_copernicus_tile_name():
    assert dm.copernicus_tile_name(24.2436, 121.2467) == "Copernicus_DSM_COG_10_N24_00_E121_00_DEM"
    assert dm.copernicus_tile_name(-0.5, -70.2) == "Copernicus_DSM_COG_10_S01_00_W071_00_DEM"
