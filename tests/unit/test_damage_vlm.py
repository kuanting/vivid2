"""The VLM damage mapper (vlm and hybrid stacks): what it sends, how answers become reports, and a
flight on the test yard with every question answered from the ground truth."""
import asyncio
import json
import sys

import numpy as np
import pytest

from conftest import FakeWorld, write_meta
from vivid2.damage import DAMAGE_MISSIONS
from vivid2.damage.geometry import grid_cells, raster_polygons
from vivid2.damage.run import DamageTask
from vivid2.geo import rpy_to_quat
from vivid2.recorder import Recorder
from vivid2.registry import AgentFactory
from vivid2.runner import score_run_dir
from vivid2.scenario import Scenario
from vivid2.stack.base import Frame, MissionSpec, UsageStats
from vivid2.stack.reference.damage_vlm import (GRID, CellGrid, MapperPhoto, PhotoGround, VLMDamageMapper,
                                               cell_label, damage_schema, load_damage_prompt, parse_cell)
from vivid2.terrain import TerrainModel
from vivid2.vlm.base import VLMError, VLMProvider

NADIR = tuple(rpy_to_quat(0.0, -np.pi / 2, 0.0))  # straight down, the top of the photo north


class Scripted(VLMProvider):
    """Answers each call with the next scripted reply (an exception is raised)."""

    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def query(self, images, prompt, schema, system=None):
        self.calls.append((images, prompt, schema))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, UsageStats(latency_s=1.0, input_tokens=100, output_tokens=10, cost_usd=0.01, model=self.model)


@pytest.fixture
def damage_run(root):
    return Scenario.load(root / "scenarios/landslide/testyard_damage_01.jsonc").expand()[0]


@pytest.fixture
def truth_mapper(root):
    sys.path.insert(0, str(root / "tools"))
    from eval_damage_mapper import TruthAnsweringMapper

    return TruthAnsweringMapper


def _mission(roads=(), buildings=(), products=("areas", "roads", "buildings", "clues"), aoi=None):
    aoi = aoi or [[-40, -40], [40, -40], [40, 40], [-40, 40]]
    damage = {"products": list(products), "aoi_ne": aoi,
              "area_classes": {"debris": "soil and rocks"}, "roads": list(roads), "buildings": list(buildings),
              "building_grades": ["none", "partial", "collapsed"],
              "clue_kinds": ["backpack", "clothing", "other"], "terrain": "current"}
    return MissionSpec(mission="landslide_sar", brief="b", search_polygon_ne=aoi, altitude_m=35.0, ground_down_m=0.0,
                       start_ned=[0, 0, -1], max_speed_mps=6, timeout_s=600, expected_targets=0,
                       terrain=TerrainModel.flat(0.0), damage=damage)


def _frame(n=0.0, e=0.0, height=35.0, depth=None, w=640, h=480, fid=1):
    if depth is None:
        depth = np.full((h, w), height, np.float32)
    return Frame(t_ns=fid * 10**9, frame_id=fid, rgb=np.zeros((h, w, 3), np.uint8), cam_pos=np.array([n, e, -height]),
                 cam_q=NADIR, fov_deg=80.0, depth=depth)


ROAD = {"id": "R1-01", "road": "R1", "polyline_ne": [[-30, 0], [30, 0]], "width_m": 6.0, "kind": "road"}
HOUSE = {"id": "B1", "footprint_ne": [[5, 5], [5, 15], [15, 15], [15, 5]], "height_m": 6, "roof": "flat", "use": "house"}


def test_grid_cell_labels():
    assert parse_cell("C4") == (2, 3) and parse_cell(" c 4") == (2, 3) and cell_label(11, 8) == "L9"
    assert parse_cell("M1") is None and parse_cell("A10") is None and parse_cell("4C") is None and parse_cell(7) is None
    assert parse_cell("M1", (13, 9)) == (12, 0)


def test_the_cell_grid_is_the_scoring_grid():
    aoi = [[-31, -17], [40, -17], [40, 33], [-31, 33]]
    grid = CellGrid(aoi)
    assert np.array_equal(grid.cells, grid_cells(aoi, 2.0))
    assert np.array_equal(grid.lookup(grid.cells), np.arange(len(grid.cells)))
    assert list(grid.lookup(np.array([[100.0, 0.0], [np.nan, 0.0]]))) == [-1, -1]
    sel = np.hypot(grid.cells[:, 0] - 5, grid.cells[:, 1] - 5) < 12
    sel[np.hypot(grid.cells[:, 0] - 5, grid.cells[:, 1] - 5) < 4] = False  # a hole
    assert np.array_equal(raster_polygons(grid.polygons(sel), grid.cells), sel)


def test_photo_ground_uses_depth_and_falls_back_to_the_terrain():
    m = _mission()
    grid = CellGrid(m.damage["aoi_ne"])
    by_depth = PhotoGround(_frame(), m.terrain, grid)
    saturated = PhotoGround(_frame(depth=np.full((480, 640), 65.535, np.float32)), m.terrain, grid)
    no_depth = PhotoGround(_frame(depth=np.zeros((480, 640), np.float32)), m.terrain, grid)
    for g in (by_depth, saturated, no_depth):
        assert np.allclose(g.down, 0.0, atol=1e-6)
    assert np.allclose(by_depth.ne, saturated.ne) and np.allclose(by_depth.ne, no_depth.ne)
    # the centre of the photo is under the drone, its top is north (the drone faces north)
    c = np.flatnonzero((by_depth.u == 324) & (by_depth.v == 244))[0]
    assert np.allclose(by_depth.ne[c], [0, 0], atol=0.5)
    top = np.flatnonzero((by_depth.u == 324) & (by_depth.v == 4))[0]
    assert by_depth.ne[top, 0] > 20 and abs(by_depth.ne[top, 1]) < 0.5
    # a grid box holds the lattice points of its part of the photo (row * columns + column)
    assert by_depth.box[c] == 4 * GRID[0] + 6 and by_depth.box[top] == 6
    p, from_depth = by_depth.locate(320, 240)
    assert from_depth and np.allclose(p, [0, 0, 0], atol=0.5)
    p, from_depth = saturated.locate(320, 240)
    assert not from_depth and np.allclose(p, [0, 0, 0], atol=0.5)


def test_the_schema_and_prompt_follow_the_task():
    assert list(damage_schema(["roads"], [], [])["properties"]) == ["roads", "summary"]
    full = damage_schema(["areas", "roads", "buildings", "clues"], ["debris"], ["backpack"])
    assert full["required"] == ["areas", "roads", "buildings", "clues", "summary"]
    assert full["properties"]["areas"]["items"]["properties"]["area_class"]["enum"] == ["debris"]
    mapper = VLMDamageMapper(Scripted([]))
    m = _mission(roads=[ROAD], buildings=[HOUSE])
    photo = mapper.prepare(_frame(), m)
    assert photo.roads == ["R1-01"] and photo.buildings == ["B1"] and len(photo.images) == 2
    assert "R1-01" in photo.prompt and "B1" in photo.prompt and "debris: soil and rocks" in photo.prompt
    assert "{" not in photo.prompt and "35 m above the ground" in photo.prompt
    far = mapper.prepare(_frame(n=200.0), m)  # neither in the photo
    assert far.roads == [] and "No known road segment is in this photo" in far.prompt


def test_every_damage_mission_with_scenarios_has_a_damage_prompt(root):
    missions = {s.spec["mission"] for s in (Scenario.load(p) for p in (root / "scenarios").rglob("*.jsonc"))
                if s.spec.get("damage")}
    assert missions and missions <= set(DAMAGE_MISSIONS)
    for mission in sorted(missions):
        intro, sections = load_damage_prompt(mission)
        assert intro and {"areas", "roads", "roads_none", "buildings", "buildings_none", "clues", "end"} <= set(sections)


def test_answers_become_votes_and_reports():
    vlm = Scripted([
        {"areas": [{"area_class": "debris", "cells": ["F5", "G5", "Z9", "F5"]}, {"area_class": "lava", "cells": ["A1"]}],
         "roads": [{"segment": "R1-01", "status": "blocked"}, {"segment": "R9-99", "status": "blocked"}],
         "buildings": [{"building": "B1", "grade": "collapsed"}],
         "clues": [{"kind": "backpack", "point": [500, 500], "description": "red pack"},
                   {"kind": "backpack", "point": [1500, 20], "description": "off the photo"}],
         "summary": "debris across the road"},
        {"areas": [], "roads": [{"segment": "R1-01", "status": "open"}],
         "buildings": [{"building": "B1", "grade": "none"}],
         "clues": [{"kind": "backpack", "point": [505, 498], "description": "same pack"}], "summary": "s"},
        VLMError("rate limited"),
        "not an object",
    ])
    mapper = VLMDamageMapper(vlm)
    m = _mission(roads=[ROAD], buildings=[HOUSE])
    reports, usage, rec = mapper.observe(_frame(fid=1), m)
    kinds = sorted(type(r).__name__ for r in reports)
    assert kinds == ["BuildingReport", "ClueReport", "RoadReport"] and usage.cost_usd == 0.01
    assert rec["areas"] == {"debris": ["F5", "G5"]} and rec["roads"] == {"R1-01": "blocked"}
    assert len(rec["ignored"]) == 4 and rec["labelled"] == {"roads": ["R1-01"], "buildings": ["B1"]}
    clue = next(r for r in reports if type(r).__name__ == "ClueReport")
    assert np.allclose(clue.ned[:2], [0, 0], atol=0.5) and clue.frame_id == 1
    # a second look: the road open in the part this photo shows (three quarters of the segment) does
    # not outweigh a blockage seen once; a tie between grades keeps the worse one
    reports, _, rec = mapper.observe(_frame(fid=2), m)
    assert reports == [] and rec["clues"][0]["repeat"]
    assert mapper.road_state == {"R1-01": True} and mapper.building_state == {"B1": "collapsed"}
    # failures are abstentions with the error recorded, never reports
    reports, usage, rec = mapper.observe(_frame(fid=3), m)
    assert reports == [] and "rate limited" in rec["error"]
    reports, _, rec = mapper.observe(_frame(fid=4), m)
    assert reports == [] and rec["error"].startswith("invalid reply")
    areas = mapper.finish(m)
    assert {a.area_class for a in areas} == {"debris"} and all(a.frame_id == 1 for a in areas)
    sel = raster_polygons([{"polygon_ne": a.polygon_ne, "op": a.op} for a in areas], mapper.cells.cells)
    # F5 and G5 are the two boxes at the photo's centre row, left and right of the middle; four looks
    # (two of them errors that add no look), the first put them in debris, the second did not: half the
    # looks, which is enough
    assert sel.any() and np.all(np.abs(mapper.cells.cells[sel][:, 0]) < 8)


def test_a_blockage_seen_once_outweighs_partial_views_of_open_road():
    blocked = {"areas": [], "roads": [{"segment": "R1-01", "status": "blocked"}], "buildings": [], "clues": [],
               "summary": ""}
    opened = {**blocked, "roads": [{"segment": "R1-01", "status": "open"}]}
    long_road = {**ROAD, "polyline_ne": [[0, -120], [0, 120]]}  # 240 m east-west
    m = _mission(roads=[long_road], aoi=[[-40, -130], [40, -130], [40, 130], [-40, 130]])
    mapper = VLMDamageMapper(Scripted([opened, opened, blocked, opened, opened, opened, opened]))
    shares = []
    for i, e in enumerate((-90, -45, 0, 45, 90)):  # each photo shows a quarter of the segment; one the blockage
        _, _, rec = mapper.observe(_frame(e=float(e), fid=i + 1), m)
        assert rec["labelled"]["roads"] == ["R1-01"]
    assert mapper.road_state["R1-01"] is True
    assert mapper.road_votes["R1-01"]["blocked"] == 1 and mapper.road_votes["R1-01"]["open"] < 1.2
    for i in range(2):  # two looks at the whole segment, both open, do outweigh it
        mapper.observe(_frame(height=150.0, w=1920, h=1440, fid=10 + i), m)
    assert mapper.road_state["R1-01"] is False


def test_a_clue_placed_without_depth_waits_for_a_better_look():
    seen = {"areas": [], "roads": [], "buildings": [], "summary": "",
            "clues": [{"kind": "clothing", "point": [500, 500], "description": "jacket"}]}
    m = _mission()
    mapper = VLMDamageMapper(Scripted([seen, seen, seen]))
    no_depth = np.zeros((480, 640), np.float32)
    reports, _, rec = mapper.observe(_frame(depth=no_depth, fid=1), m)
    assert reports == [] and rec["clues"][0]["held"]
    reports, _, _ = mapper.observe(_frame(fid=2), m)  # placed by depth: reported at once
    assert [type(r).__name__ for r in reports] == ["ClueReport"] and reports[0].frame_id == 2
    reports, _, rec = mapper.observe(_frame(depth=no_depth, fid=3), m)
    assert reports == [] and rec["clues"][0]["repeat"]
    assert [r for r in mapper.finish(m) if type(r).__name__ == "ClueReport"] == []
    only_dem = VLMDamageMapper(Scripted([seen]))
    only_dem.observe(_frame(depth=no_depth, fid=7), m)
    late = [r for r in only_dem.finish(m) if type(r).__name__ == "ClueReport"]
    assert len(late) == 1 and late[0].frame_id == 7


def test_which_photos_it_maps():
    m = _mission()
    mapper = VLMDamageMapper(Scripted([]), max_calls=3)
    assert mapper.wants(_frame(fid=1), m)
    assert not mapper.wants(_frame(n=1.0, fid=2), m)            # nothing new
    assert not mapper.wants(_frame(n=40.0, height=10.0, fid=3), m)  # far below survey height
    tilted = _frame(n=60.0, fid=4)
    tilted.gimbal = {"tilt_deg": -45.0, "pan_deg": 0.0, "fov_deg": 80.0}
    assert not mapper.wants(tilted, m)
    assert mapper.wants(_frame(n=40.0, fid=5), m) and mapper.new_share[5] >= 0.5
    assert not mapper.wants(_frame(n=200.0, fid=6), m)          # outside the area of interest
    assert mapper.wants(_frame(n=-40.0, fid=7), m)
    assert not mapper.wants(_frame(e=40.0, fid=8), m)           # the budget is spent
    assert mapper.calls == 3


def test_the_vlm_stacks_map_damage_on_damage_runs_only(monkeypatch, damage_run, blocks_run):
    import vivid2.registry as registry

    monkeypatch.setattr(registry, "make_vlm", lambda opts, closed_loop=False: Scripted([]))
    for kind in ("vlm", "hybrid"):
        factory = AgentFactory(kind)
        mapper = factory._damage_mapper(damage_run)
        assert isinstance(mapper, VLMDamageMapper) and mapper.provider is factory._vlm
        assert factory._damage_mapper(blocks_run) is None
    agent = AgentFactory("vlm")(damage_run)
    assert agent.describe()["damage_mapper"]["name"] == "vlm_damage"
    assert "damage_mapper" not in AgentFactory("vlm")(blocks_run).describe()
    assert AgentFactory("coverage_yolo")._damage_mapper(damage_run) is None


def test_a_flight_with_true_answers_maps_the_test_yard(tmp_path, damage_run, damage_sim_config, truth_mapper):
    terrain = TerrainModel.flat(0.0)
    task = DamageTask.for_run(damage_run.spec, damage_sim_config, terrain, damage_run.resolved_clues(terrain),
                              damage_run.resolved_buried(terrain))
    run_dir = tmp_path / "vlm" / damage_run.run_id
    rec = Recorder(run_dir)
    world = FakeWorld(damage_run, rec, damage=task)
    agent = AgentFactory("coverage_gt")(damage_run)
    agent.mapper = truth_mapper(task.gt, task.clues)  # the VLM mapper, answered from the ground truth
    asyncio.run(agent.run(world))
    write_meta(rec, damage_run, agent.describe(), world.targets, world.t_ns,
               extra={"damage": task.meta(), "clues": task.clues, "mission_spec": world.mission.record()})
    rec.close()
    res = score_run_dir(run_dir)
    d = res["metrics"]["damage"]
    assert d["violations"] == [] and d["mapper"]["calls"] == agent.mapper.calls > 0 and d["mapper"]["errors"] == 0
    assert d["components"]["roads"] == 1.0 and d["components"]["buildings"] == 1.0
    assert d["components"]["clues"] == 1.0 and d["components"]["buried"] == 1.0
    assert d["components"]["areas"] > 0.75 and d["score"] > 90
    # every background look finished before the area outlines went out, at the end
    damage = [json.loads(line) for line in (run_dir / "damage.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [r["kind"] for r in damage]
    assert "area" in kinds and kinds.index("area") == len(kinds) - kinds.count("area")
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(x.get("layer") == "damage_mapper" for x in decisions) == agent.mapper.calls
