"""Damage ground truth: region fields, layouts -> ground truth, geometry helpers."""
import json
import sys

import cv2
import numpy as np
import pytest

from conftest import lfs_file
from vivid2.damage import regions as rg
from vivid2.damage.geometry import (corridor, distance_to_polygon, grid_cells, mask_polygons, raster_polygons,
                                    split_polyline, view_mask)
from vivid2.damage.gt import load_gt
from vivid2.damage.layout import build_damage, write_build
from vivid2.geo import rpy_to_quat
from vivid2.terrain import TerrainModel

NADIR = list(rpy_to_quat(0.0, -np.pi / 2, 0.0))


def _flat_layout(**kw):
    base = {"map": "T", "grid": {"n0": -50, "e0": -50, "size": [101, 101], "cell_m": 1.0, "ground_down_m": 0.0},
            "mask": {"size_px": 128, "extent_m": 128.0}, "regions": {}, "area_classes": {}}
    base.update(kw)
    return base


def test_region_field_reproduces_the_forest_fire_burn_mask(root):
    lay = json.loads((root / "unreal/VIVID2Env/SourceData/ForestFire/burn_layout.json").read_text(encoding="utf-8"))
    png = cv2.imread(str(lfs_file(root / "unreal/VIVID2Env/SourceData/ForestFire/burn_mask.png")), cv2.IMREAD_UNCHANGED)
    field = rg.region_field({"polygon_ne": lay["burn_polygon_ne"], "edge": lay["edge"]}, lay["mask"]["size_px"],
                            lay["mask"]["extent_m"])
    assert np.array_equal(rg.to_png(field), png)
    sys.path.insert(0, str(root / "tools"))
    from burn_mask import burn_mask  # the tool that made it
    assert np.array_equal(burn_mask(lay), png)


def test_region_png_is_never_zero_and_samples_back():
    f = rg.region_field({"polygon_ne": [[-10, -10], [10, -10], [10, 10], [-10, 10]]}, 64, 64.0)
    png = rg.to_png(f)
    assert png.min() >= 128 and png.max() == 255
    assert rg.sample(rg.from_png(png), np.array([0.0, 30.0]), np.array([0.0, 30.0]), 64.0).tolist() == [1.0, 0.0]


def test_mask_outlines_rasterise_back_to_the_mask():
    rng = np.random.default_rng(3)
    for _ in range(10):
        m = cv2.GaussianBlur(rng.standard_normal((30, 40)), (0, 0), 2.5) > 0.2
        polys = mask_polygons(m, -15.0, 5.0, 2.0)
        ii, jj = np.meshgrid(np.arange(30), np.arange(40), indexing="ij")
        cells = np.column_stack([-15.0 + ii.ravel() * 2.0, 5.0 + jj.ravel() * 2.0])
        assert np.array_equal(raster_polygons(polys, cells).reshape(m.shape), m)


def test_polyline_helpers():
    pieces = split_polyline([[0, 0], [0, 250]], 100)
    assert len(pieces) == 3 and pieces[0][0] == [0.0, 0.0] and pieces[-1][-1] == [0.0, 250.0]
    along, pts = corridor([[0, 0], [10, 0]], 4.0, 0.5)
    assert along.max() == 10.0 and np.abs(pts[:, 1]).max() == 2.0
    d = distance_to_polygon(np.array([[0, 0], [3, 0]]), [[-1, -1], [1, -1], [1, 1], [-1, 1]])
    assert d.tolist() == [0.0, 2.0]


def test_a_road_is_blocked_by_full_width_debris_not_by_one_lane():
    road = {"id": "R", "polyline_ne": [[-40, 0], [40, 0]], "width_m": 6, "segment_m": 100}
    one_lane = _flat_layout(regions={"d": {"polygon_ne": [[-20, 1.5], [20, 1.5], [20, 10], [-20, 10]]}},
                            area_classes={"debris": {"regions": ["d"]}}, roads=[road])
    assert not build_damage(one_lane).doc["roads"][0]["blocked"]
    across = _flat_layout(regions={"d": {"polygon_ne": [[-2, -10], [2, -10], [2, 10], [-2, 10]]}},
                          area_classes={"debris": {"regions": ["d"]}}, roads=[road])
    seg = build_damage(across).doc["roads"][0]
    assert seg["blocked"] and seg["cause"] == ["debris"] and abs(seg["blocked_at_ne"][0][0]) < 3


def test_water_blocks_where_deep_and_floods_land():
    lay = _flat_layout(water={"level_down_m": -0.5, "polygon_ne": [[-10, -50], [10, -50], [10, 50], [-10, 50]]},
                       area_classes={"flood": {"water": True}},
                       roads=[{"id": "R", "polyline_ne": [[0, -40], [0, 40]], "width_m": 6}])
    b = build_damage(lay)
    assert b.doc["roads"][0]["blocked"] and "water" in b.doc["roads"][0]["cause"]
    assert b.classes.sum() > 0 and float(b.water_m.max()) == pytest.approx(0.5)
    assert float(b.surface.down_at(0, 0)) == pytest.approx(-0.5)  # the water surface


def test_building_grades_debris_and_roofs():
    lay = _flat_layout(
        regions={"d": {"polygon_ne": [[0, -30], [30, -30], [30, 30], [0, 30]]}},
        area_classes={"debris": {"regions": ["d"], "blocks_roads": False}},
        grade_from="debris",
        buildings=[{"id": "A", "footprint_ne": [[5, -5], [15, -5], [15, 5], [5, 5]], "height_m": 8},       # under debris
                   {"id": "B", "footprint_ne": [[-6, -5], [2, -5], [2, 5], [-6, 5]], "height_m": 8},       # a third
                   {"id": "C", "footprint_ne": [[-30, -5], [-20, -5], [-20, 5], [-30, 5]], "height_m": 8},  # clear
                   {"id": "D", "footprint_ne": [[-45, 20], [-35, 20], [-35, 30], [-45, 30]], "height_m": 6, "ems98": 4}],
        roads=[{"id": "R", "polyline_ne": [[-48, 31], [-30, 31]], "width_m": 4}],
        expect={"grades": {"A": "collapsed", "B": "partial", "C": "none", "D": "collapsed"}})
    b = build_damage(lay)
    assert not b.problems, b.problems
    road = b.doc["roads"][0]
    assert road["blocked"] and road["cause"] == ["building:D"]  # its rubble spills 1.8 m across most of the road
    assert float(b.surface.down_at(-25, 0)) == pytest.approx(-8.0)   # an intact roof
    assert float(b.surface.down_at(-40, 25)) == pytest.approx(-2.0)  # rubble
    bad = build_damage({**lay, "expect": {"grades": {"A": "none"}}})
    assert bad.problems


def test_written_ground_truth_loads_and_guards_its_raster(tmp_path):
    lay = _flat_layout(regions={"d": {"polygon_ne": [[0, 0], [10, 0], [10, 10], [0, 10]]}},
                       area_classes={"debris": {"regions": ["d"]}},
                       roads=[{"id": "R", "polyline_ne": [[5, -20], [5, 20]], "width_m": 6}])
    lp = tmp_path / "src" / "layout.json"
    lp.parent.mkdir()
    lp.write_text(json.dumps(lay), encoding="utf-8")
    out = write_build(build_damage(lay), lp, tmp_path / "maps" / "T", {}, source_dir=tmp_path / "src")
    gt = load_gt(out["gt"])
    assert gt.class_at("debris", 5, 5) and not gt.class_at("debris", -5, -5)
    assert gt.public_roads()[0] == {"id": "R-01", "road": "R", "polyline_ne": [[5.0, -20.0], [5.0, 20.0]],
                                    "width_m": 6.0, "kind": "road"}  # no "blocked"
    with pytest.raises(ValueError):
        load_gt(out["gt"], expect_sha256="0" * 64)
    out["raster"].write_bytes(out["raster"].read_bytes() + b"x")
    with pytest.raises(ValueError):
        load_gt(out["gt"])


def test_committed_test_map_ground_truth_matches_its_layout(root):
    lay = json.loads((root / "unreal/VIVID2Env/SourceData/TestYard/layout.json").read_text(encoding="utf-8"))
    lfs_file(root / "sim_config/maps/TestYard/damage_gt.npz")
    built = build_damage(lay)
    gt = load_gt(root / "sim_config/maps/TestYard/damage_gt.json")
    assert np.array_equal(built.classes, gt.classes)
    assert built.doc["roads"] == gt.roads and built.doc["buildings"] == gt.buildings


def test_view_mask_needs_the_cell_in_the_photo_sharp_and_unhidden():
    cells = grid_cells([[-60, -60], [60, -60], [60, 60], [-60, 60]], 2.0)
    frame = {"cam_pos": [0, 0, -35], "cam_q": NADIR, "width": 960, "height": 720, "fov_deg": 80.0}
    seen = view_mask(frame, cells, np.zeros(len(cells)))
    assert 0 < seen.mean() < 1 and seen[np.argmin(np.linalg.norm(cells, axis=1))]
    high = {**frame, "cam_pos": [0, 0, -400]}  # 0.7 m per pixel: too coarse
    assert not view_mask(high, cells, np.zeros(len(cells))).any()
    # a ridge between a low tilted camera and the far cells hides them
    ridge = TerrainModel(np.where(np.arange(121)[:, None] == 70, -30.0, 0.0) * np.ones((121, 121)), -60, -60, 1.0)
    tilted = {**frame, "cam_pos": [0, 0, -5], "cam_q": list(rpy_to_quat(0.0, -0.3, 0.0))}
    open_ = view_mask(tilted, cells, np.zeros(len(cells)))
    hidden = view_mask(tilted, cells, np.zeros(len(cells)), ridge)
    beyond = cells[:, 0] > 12
    assert open_[beyond].any() and not hidden[beyond].any()


def test_a_building_placed_by_centre_and_size_takes_height_m():
    lay = _flat_layout(buildings=[{"id": "B1", "ne": [0, 0], "size_m": [12, 8], "yaw_deg": 90, "height_m": 7, "ems98": 0}])
    b = build_damage(lay).doc["buildings"][0]
    fp = np.asarray(b["footprint_ne"])
    assert b["height_m"] == 7 and np.ptp(fp[:, 0]) == pytest.approx(8) and np.ptp(fp[:, 1]) == pytest.approx(12)


def test_the_landslide_layout_builds_its_expected_damage(root):
    """Landslide's layout on ForestSAR's terrain: the expect block (blocked segments, grades) holds."""
    lay = json.loads((root / "unreal/VIVID2Env/SourceData/Landslide/layout.json").read_text(encoding="utf-8"))
    post = TerrainModel.load(lfs_file(root / "sim_config/maps/Landslide/terrain_dtm.npz"))
    built = build_damage(lay, post)
    assert sorted(r["id"] for r in built.doc["roads"] if r["blocked"]) == lay["expect"]["blocked"]
    assert set(built.doc["area_classes"]) == {"scar", "debris"} and not built.doc["area_classes"]["scar"]["blocks_roads"]
