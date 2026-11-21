"""Terrain edits (vivid2.terrain_edit) and the DEM builder's new options: mosaics, sea fill, pinning."""
import json
import math

import numpy as np
import pytest

from test_dem import LAT0, LON0, _write_geotiff
from vivid2 import dem as dm
from vivid2.terrain_edit import apply_edits, grade_limit, road_bench


def _grid(size=201, cell=1.0):
    ax = (np.arange(size) - size // 2) * cell
    return np.meshgrid(ax, ax, indexing="ij")


def test_grade_limit_keeps_every_step_within_the_grade():
    p = grade_limit(np.array([0, 0, 5, 5, 5, 0, 0], float), 1.0, 0.5)
    assert np.all(np.abs(np.diff(p)) <= 0.5 + 1e-9)


def test_a_road_bench_is_flat_across_and_leaves_far_ground_alone():
    N, E = _grid()
    z = 0.3 * E  # a 30 % side slope rising to the east
    out, prof = road_bench(N, E, z, [[-80, 0], [80, 0]], 6.0, max_grade=0.1, batter=1.0)
    row = out[100 + 40]  # north 40 m, a cross-section
    assert row[100 - 3:100 + 4].max() - row[100 - 3:100 + 4].min() < 1e-6  # flat across the road
    assert np.allclose(out[:, :60], z[:, :60]) and np.allclose(out[:, 141:], z[:, 141:])  # 40 m away: untouched
    assert np.all(np.abs(out - z) <= np.abs(z - row[100]) + 1e-9)


def test_lower_and_raise_follow_the_region_and_report_volumes():
    N, E = _grid()
    layout = {"mask": {"size_px": 256, "extent_m": 256.0},
              "regions": {"scar": {"polygon_ne": [[20, -20], [60, -20], [60, 20], [20, 20]]},
                          "lobe": {"polygon_ne": [[-60, -30], [-10, -30], [-10, 30], [-60, 30]],
                                   "edge": {"noise_m": 2, "noise_scale_m": 10, "soft_m": 3, "seed": 1}}},
              "terrain": {"pre": [], "post": [{"op": "lower", "region": "scar", "amount_m": 5, "taper_m": 10},
                                              {"op": "raise", "region": "lobe", "amount_m": 4, "taper_m": 10,
                                               "noise_m": 0.5, "seed": 2}]}}
    z = np.zeros(N.shape)
    pre, rep_pre = apply_edits(layout, "pre", N, E, z)
    post, rep = apply_edits(layout, "post", N, E, z)
    assert np.array_equal(pre, z) and rep_pre == []
    assert post[100 + 40, 100] == pytest.approx(-5.0) and post[100 - 35, 100] > 3.0
    assert post[100 + 90, 100 + 90] == 0.0  # outside both regions
    assert rep[0]["cut_m3"] > 0 and rep[1]["fill_m3"] > 0 and rep[0]["bbox_ne"][0][0] >= 19


def test_flatten_levels_a_polygon_and_blends_outside():
    N, E = _grid()
    z = 0.1 * N + 0.05 * E
    out, _ = apply_edits({"terrain": {"pre": [{"op": "flatten", "polygon_ne": [[-20, -20], [20, -20], [20, 20], [-20, 20]],
                                               "level": 1.0, "blend_m": 10}]}}, "pre", N, E, z)
    assert np.allclose(out[100 - 15:100 + 16, 100 - 15:100 + 16], 1.0)
    assert np.allclose(out[:60, :60], z[:60, :60])


def test_pinned_build_matches_the_unpinned_one_and_keeps_unedited_heights(tmp_path):
    p = tmp_path / "dem.tif"
    _write_geotiff(p, size=200)
    d = dm.read_geotiff(p)
    base = dm.build_terrain(d, LAT0, LON0, vertices=253)
    pin = dm.pin_of({"landscape": base.settings})
    again = dm.build_terrain(d, LAT0, LON0, vertices=253, pin=pin)
    assert np.array_equal(again.heightmap, base.heightmap) and again.settings["headroom_m"]["below"] >= 0
    layout = {"mask": {"size_px": 256, "extent_m": 256.0},
              "regions": {"dip": {"polygon_ne": [[0, 0], [40, 0], [40, 40], [0, 40]]}},
              "terrain": {"post": [{"op": "lower", "region": "dip", "amount_m": 2, "taper_m": 5}]}}
    edited = dm.build_terrain(d, LAT0, LON0, vertices=253, pin=pin,
                              edit=lambda n, e, z: apply_edits(layout, "post", n, e, z))
    changed = edited.heightmap != base.heightmap
    rows, cols = np.nonzero(changed)  # heightmap rows are east, columns north
    assert changed.any() and (cols - 126).min() >= -1 and (rows - 126).min() >= -1
    with pytest.raises(ValueError, match="pinned heightmap range"):
        dm.build_terrain(d, LAT0, LON0, vertices=253, pin=pin, edit=lambda n, e, z: (z + 100.0, []))


def _tile(la, lo, value, step=0.01):
    n = int(round(1 / step))
    return ((la, lo), dm.GeoDEM(z=np.full((n, n), float(value)), lat_r0=la + 1 - step / 2, lon_c0=lo + step / 2,
                                dlat=step, dlon=step, source=f"{la},{lo}"))


def test_mosaic_places_tiles_and_leaves_missing_ones_empty():
    m = dm.mosaic([_tile(24, 121, 10), _tile(24, 122, 20), ((25, 121), None), _tile(25, 122, 40)])
    assert m.z.shape == (200, 200)
    r, c = m.pixel_of(24.5, 121.5)
    assert m.z[int(r), int(c)] == 10
    r, c = m.pixel_of(25.5, 121.5)
    assert np.isnan(m.z[int(r), int(c)])
    r, c = m.pixel_of(25.5, 122.5)
    assert m.z[int(r), int(c)] == 40


def test_fill_sea_gives_a_shelf_and_a_sea_mask():
    z = np.full((100, 100), 20.0)
    z[:, 60:] = 0.0      # flat sea surface reaching the edge
    z[:, 80:] = np.nan   # and no data beyond
    z[40:45, 20:25] = 0.0  # an inland pond: not sea
    d = dm.GeoDEM(z=z, lat_r0=24.0, lon_c0=121.0, dlat=1 / 3600, dlon=1 / 3600)
    filled, sea = dm.fill_sea(d)
    assert sea[:, 60:].all() and not sea[:, :60].any()
    assert np.isfinite(filled.z).all() and filled.z[50, 99] == pytest.approx(-30.0)
    assert -5.0 < filled.z[50, 61] < 0 and filled.z[42, 22] == 0.0


def test_a_winding_road_bench_matches_a_brute_force_one():
    from vivid2.terrain_edit import _resample

    ax = np.arange(-60, 61, 1.0)
    N, E = np.meshgrid(ax, ax, indexing="ij")
    z = 0.2 * E + 0.05 * N + np.sin(N / 7) * 2
    line = [[-50, -30], [-10, -10], [10, 20], [50, 25]]
    out, prof = road_bench(N, E, z, line, 6.0)
    pts, _ = _resample(line, 1.0)
    P = np.stack([N.ravel(), E.ravel()], 1)
    best, h = np.full(len(P), np.inf), np.zeros(len(P))
    for k in range(len(pts) - 1):
        a, ab = pts[k], pts[k + 1] - pts[k]
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        closer = d < best
        best[closer], h[closer] = d[closer], (prof[k] + t * (prof[k + 1] - prof[k]))[closer]
    best, h = best.reshape(N.shape), h.reshape(N.shape)
    ref = np.where(best <= 3, h, h + np.clip(z - h, -(best - 3), best - 3))
    assert np.abs(ref - out).max() < 1e-9
