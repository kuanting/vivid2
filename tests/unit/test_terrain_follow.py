"""Terrain-following surveys (search_area.altitude_ref "terrain"): the plan, the reference agents'
survey heights, coverage on the ground itself, and that runs without it are unchanged."""
import copy
import math

import numpy as np
import pytest

from vivid2.geo import rpy_to_quat
from vivid2.metrics.planning import _cells, coverage
from vivid2.runner import mission_spec
from vivid2.scenario import validate_scenario
from vivid2.stack.base import MissionSpec
from vivid2.stack.reference.agents import LowFlightMixin
from vivid2.stack.reference.planners import FOLLOW_STEP_M, LawnmowerPlanner, follow_down_at, follows_terrain
from vivid2.terrain import TerrainModel

NADIR = list(rpy_to_quat(0.0, -math.pi / 2, 0.0))


def slope(drop_per_m: float = 0.5) -> TerrainModel:
    """Ground falling to the south: down = -100 + drop_per_m * (200 - n) ... i.e. 100 m up at n=200."""
    n = np.arange(-300, 301, 20.0)
    down = np.repeat((-100.0 - drop_per_m * (n - 200.0))[:, None], 31, axis=1)
    return TerrainModel(down.astype(np.float32), n0=-300.0, e0=-300.0, cell_m=20.0, source="test")


def mission(terrain, ref="terrain", start=(150.0, 0.0)) -> MissionSpec:
    ground = float(terrain.down_at(*start))
    return MissionSpec(mission="landslide_sar", brief="", search_polygon_ne=[[-200, -100], [200, -100], [200, 100], [-200, 100]],
                       altitude_m=60.0, ground_down_m=ground, start_ned=[start[0], start[1], ground], max_speed_mps=6.0,
                       timeout_s=900, expected_targets=1, terrain=terrain, altitude_ref=ref)


def test_flat_plan_unchanged_and_follow_plan_keeps_its_height_over_the_ground():
    t = slope()
    flat = LawnmowerPlanner().plan(mission(t, ref=None), 80.0, (1920, 1440))
    assert len({w[2] for w in flat}) == 1  # one height, as always
    follow = LawnmowerPlanner().plan(mission(t), 80.0, (1920, 1440))
    pts = np.asarray(follow)
    # the same lanes: every corner of the fixed-height plan is a waypoint of the following one
    for w in flat:
        assert np.min(np.linalg.norm(pts[:, :2] - np.asarray(w[:2]), axis=1)) < 1e-6
    steps = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    assert steps.max() <= FOLLOW_STEP_M + 1e-6
    agl = t.down_at(pts[:, 0], pts[:, 1]) - pts[:, 2]
    assert agl.min() >= 60.0 - 1e-6 and agl.max() <= 60.0 + 0.5 * 25.0 + 1e-6  # the window looks 25 m ahead
    # the first waypoint is next to the start: the leg from the start follows the ground too
    assert np.linalg.norm(pts[0, :2] - np.array([150.0, 0.0])) <= FOLLOW_STEP_M + 1e-6


def test_follow_needs_relief_and_the_option():
    assert follows_terrain(mission(slope()))
    assert not follows_terrain(mission(slope(), ref=None))
    assert not follows_terrain(mission(TerrainModel.flat(-5.0)))


class _Agent(LowFlightMixin):
    low = None


def test_agent_survey_height_follows_the_ground_and_guard_climbs():
    t = slope()
    a = _Agent()
    m = mission(t)
    a._survey_down = m.ground_down_m - m.altitude_m
    a._lf_start(m, (1920, 1440), 80.0)
    assert a._survey_down_at(0.0, 0.0) == pytest.approx(follow_down_at(t, 0.0, 0.0, 60.0))
    assert a._survey_down_at(0.0, 0.0) > a._survey_down  # 75 m lower than the start's ground: lower down
    # the way between two points is flown at the higher of their heights
    assert a._survey_down_between((0.0, 0.0), (150.0, 0.0)) == pytest.approx(a._survey_down_at(150.0, 0.0))
    a._climbed_m = 5.0
    assert a._survey_down_at(0.0, 0.0) == pytest.approx(follow_down_at(t, 0.0, 0.0, 60.0) - 5.0)
    fixed = _Agent()
    m2 = mission(t, ref=None)
    fixed._survey_down = m2.ground_down_m - m2.altitude_m
    fixed._lf_start(m2, (1920, 1440), 80.0)
    assert fixed._survey_down_at(0.0, 0.0) == fixed._survey_down == fixed._survey_down_between((0, 0), (150, 0))


def test_coverage_on_the_ground_itself():
    t = slope()
    poly = [[-60, -60], [60, -60], [60, 60], [-60, 60]]
    ground0 = float(t.down_at(0.0, 0.0))
    frame = {"t": 1, "frame_id": 1, "camera": "Payload", "cam_pos": [0.0, 0.0, ground0 - 60.0], "cam_q": NADIR,
             "width": 1920, "height": 1440, "fov_deg": 80.0}
    # The plane through a start 75 m higher up is above this camera: nothing counts on it.
    start_plane = float(t.down_at(150.0, 0.0))
    assert coverage([frame], poly, start_plane)["covered_m2"] == 0.0
    on_ground = coverage([frame], poly, start_plane, terrain=t)
    assert on_ground["covered_m2"] > 0.0
    # straight down over the slope: the footprint is about 100 m wide (80 degrees at 60 m), up to the polygon's 120 m
    cells = _cells(poly, 2.0)
    assert 0.45 * len(cells) * 4.0 < on_ground["covered_m2"] < len(cells) * 4.0
    # flat ground: the per-cell path gives what the plane gives
    flat = TerrainModel.flat(ground0)
    assert coverage([frame], poly, ground0, terrain=flat)["covered_m2"] == coverage([frame], poly, ground0)["covered_m2"]


def test_record_has_altitude_ref_only_when_set(blocks_run):
    assert "altitude_ref" not in mission_spec(blocks_run).record()
    spec = copy.deepcopy(blocks_run.spec)
    spec["search_area"]["altitude_ref"] = "terrain"
    run = copy.copy(blocks_run)
    run.spec = spec
    assert mission_spec(run).record()["altitude_ref"] == "terrain"
    spec["search_area"]["altitude_ref"] = "start"
    assert "altitude_ref" not in mission_spec(run).record()


def test_terrain_following_is_for_one_drone(blocks_run):
    spec = copy.deepcopy(blocks_run.spec)
    spec["search_area"]["altitude_ref"] = "terrain"
    validate_scenario(spec)
    spec["fleet"] = {"size": 2}
    with pytest.raises(ValueError, match="one drone only"):
        validate_scenario(spec)
