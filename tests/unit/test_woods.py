"""Low flight in the woods: sensors for obstacle avoidance, scenarios that allow (and need) it."""
import asyncio

import commentjson
import pytest

from conftest import FakeWorld
from vivid2.recorder import Recorder
from vivid2.scenario import Campaign, Scenario


@pytest.mark.parametrize("robot", ["robot_vivid_quad.jsonc", "robot_vivid_quad_px4_sitl.jsonc"])
def test_robots_have_a_forward_depth_camera(root, robot):
    cfg = commentjson.load(open(root / "sim_config" / robot, encoding="utf-8"))
    front = next(s for s in cfg["sensors"] if s["id"] == "Front")
    types = {c["image-type"] for c in front["capture-settings"] if c["capture-enabled"]}
    assert {0, 1} <= types  # RGB + planar depth
    pitch = float(front["origin"]["rpy-deg"].split()[1])
    assert -30 <= pitch <= 0  # looks ahead, slightly down
    payload = next(s for s in cfg["sensors"] if s["id"] == "Payload")
    assert next(c for c in payload["capture-settings"] if c["image-type"] == 0)["width"] == 1920


def test_contexts_without_navigation_cameras_return_none(tmp_path, blocks_run):
    world = FakeWorld(blocks_run, Recorder(tmp_path / "r"))
    assert asyncio.run(world.get_nav_frame()) is None


def test_forest_scenarios_allow_low_flight_and_include_dense_fog(root):
    c = Campaign.load(root / "campaigns" / "forest_sar_v1.jsonc")
    forest = [r for r in c.runs() if r.spec["scene"]["map"].startswith("ForestSAR")]
    assert forest and all(r.spec["limits"]["min_agl_m"] <= 3 for r in forest)
    fogs = {r.spec["environment"]["weather"].get("fog", 0.0) for r in forest}
    assert max(fogs) >= 0.4 and 0.0 in fogs
    fog01 = Scenario.load(root / "scenarios" / "forest_sar" / "forest_fog_01.jsonc")
    assert fog01.spec["search_area"]["altitude_m"] >= 50  # the suggested survey height is the hard way
