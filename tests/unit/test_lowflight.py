"""Low flight ("dip into clearings"): canopy map, clearings, safe descents, budget, guard, wiring."""
import asyncio
import json
import math

import numpy as np
import pytest

from conftest import NADIR_Q, FakeWorld, write_meta
from vivid2.recorder import Recorder
from vivid2.registry import AgentFactory
from vivid2.runner import score_run_dir
from vivid2.stack.base import Frame
from vivid2.stack.reference.agents import SearchAgent, SurveyAgent
from vivid2.stack.reference.lowflight import CanopyMap, Clearing, DipSettings
from vivid2.stack.reference.perception import GroundTruthPerception
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner
from vivid2.terrain import TerrainModel

TREE_M = 20.0
CLEARING_R = 14.0


def forest_with_clearing(cn, ce, r=CLEARING_R, tree_m=TREE_M):
    """Trees everywhere except a round clearing."""
    def surface(n, e):
        return np.where(np.hypot(np.asarray(n) - cn, np.asarray(e) - ce) <= r, 0.0, tree_m)
    return surface


class FoggyPerception(GroundTruthPerception):
    """Sees a person only from below ``see_below_m`` above the ground (fog hides them from higher)."""

    def __init__(self, targets, see_below_m=20.0):
        super().__init__(targets)
        self.see_below_m = see_below_m

    def detect(self, frame):
        if 0.0 - frame.cam_pos[2] > self.see_below_m:  # flat test ground at down 0
            return []
        return super().detect(frame)


def _events(rec, kind=None):
    rows = [json.loads(l) for l in (rec.run_dir / "events.jsonl").read_text().splitlines()]
    return [r for r in rows if kind is None or r.get("kind") == kind]


def _trajectory(rec):
    return np.array([json.loads(l)["pos"] for l in (rec.run_dir / "trajectory.jsonl").read_text().splitlines()])


def _fly(tmp_path, run, agent, surface, name="dip"):
    rec = Recorder(tmp_path / name / run.run_id)
    world = FakeWorld(run, rec, width=320, height=240, surface=surface)
    asyncio.run(agent.run(world))
    write_meta(rec, run, agent.describe(), world.targets, world.t_ns,
               "timeout" if world.elapsed_s() >= world.mission.timeout_s else "completed")
    rec.close()
    return world, rec


def _hidden_hiker_run(blocks_run):
    t = blocks_run.resolved_targets()[0]["ned"]
    return blocks_run, forest_with_clearing(t[0], t[1]), t


# --- the canopy map ------------------------------------------------------------------------------------------

def _nadir_frame(world):
    return Frame(0, 1, np.zeros((world.H, world.W, 3), np.uint8), world.pos.copy(), NADIR_Q, world.fov,
                 depth=world.render_depth())


def test_canopy_map_finds_the_clearing_and_nothing_else(tmp_path, blocks_run):
    world = FakeWorld(blocks_run, Recorder(tmp_path / "r"), width=320, height=240,
                      surface=forest_with_clearing(20.0, -10.0))
    m = CanopyMap(world.mission.search_polygon_ne, TerrainModel.flat(0.0))
    for n in (-40.0, 0.0, 20.0, 40.0):  # high frames along a lane over the clearing
        world.pos = np.array([n, -10.0, -35.0])
        assert m.add(_nadir_frame(world)) > 0
    spots = m.clearings(min_radius_m=6.0, spacing_m=15.0)
    assert len(spots) == 1
    s = spots[0]
    assert math.hypot(s.n - 20.0, s.e + 10.0) < 3.0
    assert CLEARING_R - 4.0 < s.radius_m <= CLEARING_R
    # tree tops sit 20 m above the ground; the clearing floor at the ground
    assert m.highest_within(20.0, -10.0, 3.0) == pytest.approx(0.0, abs=1.0)
    assert m.highest_in_corridor((-40, -10), (-30, -10), 3.0) == pytest.approx(-TREE_M, abs=1.0)
    # nothing is assumed about places never seen
    assert m.radius_at(500.0, 500.0) == 0.0


def test_a_clearing_outside_the_search_area_is_not_a_dip_spot(tmp_path, blocks_run):
    world = FakeWorld(blocks_run, Recorder(tmp_path / "r"), width=320, height=240,
                      surface=forest_with_clearing(75.0, 0.0))  # polygon ends at north 60
    m = CanopyMap(world.mission.search_polygon_ne, TerrainModel.flat(0.0))
    world.pos = np.array([60.0, 0.0, -40.0])
    m.add(_nadir_frame(world))
    assert m.clearings(6.0, 15.0) == []


# --- the dip agent ---------------------------------------------------------------------------------------------

def test_a_hiker_hidden_by_fog_is_found_by_dipping_into_the_clearing(tmp_path, blocks_run):
    run, surface, t = _hidden_hiker_run(blocks_run)
    agent = SearchAgent("gt-dip", LawnmowerPlanner(), DetectionThresholdReasoner(0.5),
                        FoggyPerception(run.resolved_targets()), frame_period_s=0.0, dip=DipSettings())
    world, rec = _fly(tmp_path, run, agent, surface)
    assert _events(rec, "dip_start"), "no dip"
    assert world.reports and math.hypot(world.reports[0].ned[0] - t[0], world.reports[0].ned[1] - t[1]) < 5.0
    look = _events(rec, "dip_look")[-1]
    assert look["confirmed"] and look["votes"] >= 2
    traj = _trajectory(rec)
    low = traj[(-traj[:, 2] < TREE_M + 5.0) & (-traj[:, 2] > 1.5)]  # below the treetops (+5 m), airborne
    assert len(low), "never came below the canopy top"
    # every low point is inside the clearing, with room to spare
    assert np.all(np.hypot(low[:, 0] - t[0], low[:, 1] - t[1]) <= CLEARING_R - 4.0)
    # it went down to the look height: max(dip_agl_m 12, min_agl_m 10 + 5) above the ground
    agl = -traj[:, 2]
    assert agl[agl > 1.5].min() == pytest.approx(15.0, abs=1.0)
    s = score_run_dir(rec.run_dir)
    assert s["metrics"]["mission"]["success"]


def test_without_dips_the_fog_hides_the_hiker(tmp_path, blocks_run):
    run, surface, _ = _hidden_hiker_run(blocks_run)
    agent = SearchAgent("gt", LawnmowerPlanner(), DetectionThresholdReasoner(0.5),
                        FoggyPerception(run.resolved_targets()), frame_period_s=0.0)
    world, rec = _fly(tmp_path, run, agent, surface, name="nodip")
    assert not world.reports and not _events(rec, "dip_start")


def test_the_descent_stops_above_an_obstacle(tmp_path, blocks_run):
    run, surface, t = _hidden_hiker_run(blocks_run)
    rec = Recorder(tmp_path / "obstacle")
    world = FakeWorld(run, rec, width=320, height=240, surface=surface)
    agent = SearchAgent("gt-dip", LawnmowerPlanner(), DetectionThresholdReasoner(0.5),
                        FoggyPerception(run.resolved_targets()), frame_period_s=0.0, dip=DipSettings())
    agent._survey_down = -35.0
    agent._reported = []
    agent._lf_start(world.mission, (640, 480), 90.0)
    world.pos = np.array([t[0], t[1], -35.0])
    agent._lf_observe(_nadir_frame(world))
    i, j = agent._canopy._cell(t[0], t[1])
    agent._canopy.top[i, j] = -30.0  # a branch the high survey missed, 30 m up, right under the drone

    looked = []

    async def look(ctx, spot):
        looked.append(ctx.pose()[0][2])

    asyncio.run(agent._dip(world, Clearing(t[0], t[1], 10.0), look))
    rec.close()
    bottom = _events(rec, "dip_bottom")[-1]
    assert bottom["stopped"] == "obstacle"
    assert looked and looked[0] <= -33.0  # never descended toward the branch


def test_dips_stop_when_time_runs_short(tmp_path, blocks_run):
    world = FakeWorld(blocks_run, Recorder(tmp_path / "budget"), width=320, height=240)
    agent = SearchAgent("gt-dip", LawnmowerPlanner(), DetectionThresholdReasoner(0.5), None,
                        frame_period_s=1.0, dip=DipSettings())
    agent._survey_down = -35.0
    agent._lf_start(world.mission, (640, 480), 90.0)
    spot = Clearing(40.0, 40.0, 10.0)
    here = np.array([0.0, 0.0, -35.0])
    assert agent._dip_affordable(world, here, spot)
    world.t_ns = int(0.85 * world.mission.timeout_s * 1e9)  # 85 % of the time gone
    assert not agent._dip_affordable(world, here, spot)
    world.t_ns = 0
    agent._odo_m = world.mission.energy_budget_m  # battery spent
    assert not agent._dip_affordable(world, here, spot)


def test_the_guard_climbs_over_trees_taller_than_the_survey_height_allows(tmp_path, blocks_run):
    """A stand of 30 m trees in the north half: from 35 m only 5 m of clearance, under the 10 m guard."""
    def surface(n, e):
        return np.where(np.asarray(n) > 20.0, 30.0, 0.0)

    run = blocks_run
    agent = SurveyAgent("survey-dip", LawnmowerPlanner(), frame_period_s=0.0, dip=DipSettings(max_dips=0))
    world, rec = _fly(tmp_path, run, agent, surface, name="guard")
    assert _events(rec, "guard_climb")
    traj = _trajectory(rec)
    over = traj[(traj[:, 0] > 25.0) & (-traj[:, 2] > 1.5)]
    assert len(over) and (-over[:, 2]).min() >= 30.0 + DipSettings().guard_clearance_m - 1.0


def test_survey_with_dips_records_low_frames(tmp_path, blocks_run):
    run, surface, t = _hidden_hiker_run(blocks_run)
    agent = SurveyAgent("survey-dip", LawnmowerPlanner(), frame_period_s=0.0, dip=DipSettings())
    world, rec = _fly(tmp_path, run, agent, surface, name="survey")
    frames = [json.loads(l) for l in (rec.run_dir / "frames.jsonl").read_text().splitlines()]
    low = [f for f in frames if -f["cam_pos"][2] < 20.0 and -f["cam_pos"][2] > 1.5]
    assert len(low) >= DipSettings().dip_frames
    assert any(f["gt"] for f in low)  # the hidden hiker is in the low frames


# --- wiring ------------------------------------------------------------------------------------------------------

def test_dip_flag_names_agents_and_changes_the_config_hash(blocks_run):
    plain = AgentFactory("coverage_gt", {})
    dip = AgentFactory("coverage_gt", {"dip": "on"})
    assert plain.name == "coverage_gt" and dip.name == "coverage_gt-dip"
    d_plain, d_dip = plain(blocks_run).describe(), dip(blocks_run).describe()
    assert "dip" not in d_plain  # runs recorded before dips existed keep their hash
    assert d_dip["dip"]["dip_agl_m"] == 12.0
    s = AgentFactory("survey", {"dip": "on"})
    assert s.name == "survey-dip" and s(blocks_run).describe()["dip"]
    assert "dip" not in AgentFactory("survey", {})(blocks_run).describe()


def test_a_find_under_a_crown_edge_is_located_on_the_ground_not_on_the_crown():
    """Depth around the pixel caught leaves 20 m up: geolocate falls back to the ray meeting the terrain."""
    from vivid2.geo import pixel_to_world
    from vivid2.stack.reference.agents import geolocate

    cam = np.array([0.0, 0.0, -50.0])
    depth = np.full((240, 320), 30.0, np.float32)  # a crown everywhere, 20 m above the ground at down 0
    f = Frame(0, 1, np.zeros((240, 320, 3), np.uint8), cam, NADIR_Q, 80.0, depth=depth)
    u, v = 260.0, 60.0
    crown = pixel_to_world(u, v, 30.0, cam, NADIR_Q, 320, 240, 80.0)
    p = geolocate(f, u, v, 0.0, TerrainModel.flat(0.0))
    assert p[2] == pytest.approx(0.0, abs=0.1)            # on the ground
    assert np.hypot(*(p[:2] - crown[:2])) > 5.0            # not under the crown point
    # on the ground already: depth is used as before
    f.depth[:] = 50.0
    q = geolocate(f, u, v, 0.0, TerrainModel.flat(0.0))
    assert q[2] == pytest.approx(0.0, abs=0.1) and np.allclose(q[:2], p[:2], atol=0.2)
    # a sloping coarse DEM is followed
    slope = TerrainModel(np.add.outer(np.arange(11) * -2.0, np.zeros(11)), n0=-50, e0=-50, cell_m=10.0)
    r = geolocate(Frame(0, 1, f.rgb, cam, NADIR_Q, 80.0), u, v, 0.0, slope)
    assert r[2] == pytest.approx(float(slope.down_at(r[0], r[1])), abs=0.2)
